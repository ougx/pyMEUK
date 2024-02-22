# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:19:06 2022

@author: mou
"""
import sys
import numpy as np
import pandas as pd
# from scipy.linalg import solve_triangular, pinv
from scipy.spatial.distance import squareform

from .common import cdist, pdist, euclidean_dist, weight_correcting, get_nearest, auto_cov, cross_cov, solve_linear
# from .variogram import vgm
# from .drift import drift
#%%

def calc_drift(drifts, coords):
    for d in drifts:
        if d['type'] == 'linear':
            pass


def calc_kriging_factors(
    obs_coords,
    obs_vals,
    sim_coords,
    vgm,
    drifts=1,
    search_radius=None,
    max_points=None,
    min_points=None,
):
    pass

#%%
# class cokriging(krige):
#     def __ini__(self,)
def ck5(
    obsloc: np.array,
    obsval: np.array,
    newloc: np.array,
    cov: callable|list,                 # convariance function or list of convariance function
    obsdrift: np.array=None,
    newdrift: np.array=None,
    obstime: np.array=None,
    newtime: np.array=None,
    obsloc1: np.array|list=None,        # list of secondary variable location
    obsval1: np.array|list=None,        # list of secondary variable v
    obstime1:np.array|list=None,        # list of secondary variable time
    unbias: bool=True,
    nmax :int|list=None,
    nmin :int=None,
    maxdist: float=None,
    distfunc: callable=None,
    tlag2slag: callable=None,
    solver:str='solve',
    weight_correction=False,
    write_mats: bool=False,
    standardized=True,
    validation=False,
    return_factor: bool=False,
    return_variance: bool=False,
    verbose: bool=False,
):

    # ck5: based on uk5,


    #############################################################################
    #######################      preprocessing data       #######################
    #############################################################################

    if isinstance(cov, list) or isinstance(cov, tuple):
        nvar = len(cov)
    else:
        cov = [[cov]]
        nvar = 1



    hastime  = not (obstime is None or newtime is None)
    hasdrift = not (obsdrift is None or newdrift is None)

    cov0 = cov[0][0](0, 0) if hastime else cov[0][0](0)
    usevgm = cov0 < (cov[0][0](1e5, 0) if hastime else cov[0][0](1e5))

    obslocs = [obsloc]
    obsvals = [obsval]
    obstimes = [obstime] if hastime else [None]*nvar

    if nvar>1:
        if isinstance(obsloc1, np.array):
            obslocs.append(obsloc1)
        elif isinstance(obsloc1, list) or isinstance(obsloc1, tuple):
            obslocs.extend(list(obsloc1))

        if isinstance(obsval1, np.array):
            obsvals.append(obsval1)
        elif isinstance(obsval1, list) or isinstance(obsval1, tuple):
            obsvals.extend(list(obsval1))

        if isinstance(obstime1, np.array):
            obstimes.append(obstime1)
        elif isinstance(obstime1, list) or isinstance(obstime1, tuple):
            obstimes.extend(list(obstime1))


    nnew = len(newloc)
    nobs  = [len(obsv) for obsv in obsvals]
    nobs_full = sum(nobs)


    if nmax is None:
        nmax = nobs
    elif isinstance(nmax, int):
        nmax = [nmax, ]
    nmax = np.minimum(nmax, nobs)

    if nmin is None:
        nmin = np.minimum(np.maximum(3, nmax // 3),20)
    elif isinstance(nmin, int):
        nmin = [nmin, ]
    nmin = np.array(nmin)

    assert all(nmin < nmax), f'All nmin({nmin}) must be smaller than nmax({nmax}).'

    solv_all = np.zeros(nnew, )
    if return_variance:
        variance = np.zeros(nnew, )

    ndrift = 0
    if hasdrift:
        obsdrift = np.array(obsdrift).reshape([nobs_full, -1])
        newdrift = np.array(newdrift).reshape([nnew, -1])
        ndrift = obsdrift.shape[1]

    #############################################################################
    ####################### search for local data points  #######################
    #############################################################################
    slags, tlags, totallags, nearests, ifactor = ([],)*5
    search = False
    for ivar in range(nvar):
        localsearch, slag, tlag, totallag, iuseful, nearest = get_nearest(
            obslocs[ivar],
            newloc,
            distfunc,
            nmax[ivar],
            maxdist,
            tlag2slag,
            obstime,
            newtime)
        nuseful = len(iuseful)
        ifactor.append(nearest) # the index in the original full observation data
        # check if number of obs is smaller than nmin
        if any((nfactor:=np.sum(nearest>=0, axis=0))<nmin[ivar]):
            toolittle = np.nonzero(nfactor<nmin[ivar])[0][:10]
            raise ValueError(f'Not enough observations near these locations (index) for variable {ivar+1}:\n  ' + ','.join(toolittle.astype(str)))

        # if localsearch and nuseful < nobs[ivar]:
        #     # remove the unused observations
        #     obslocs[ivar] = obslocs[ivar][iuseful]
        #     obsvals[ivar] = obsvals[ivar][iuseful]
        #     slag = slag[iuseful]
        #     totallag = totallag[iuseful]
        #     if hastime:
        #         obstimes[ivar] = obstimes[ivar][iuseful]
        #         tlag = tlag[iuseful]
        #     if ivar == 0 and hasdrift:
        #         obsdrift = obsdrift[iuseful]
        #     # update the nearest index for the smaller group of observations
        #     findex = np.full(nobs[ivar], -1, dtype=int)
        #     findex[iuseful] = np.arange(nuseful)
        #     nearest = np.where(nearest>=0, findex[nearest]+sum(nobs[:ivar]), -1)
        #     nobs[ivar] = nuseful

        search = search or localsearch
        slags.append(slag)
        tlags.append(tlag)
        totallags.append(totallag)
        nearests.append(nearest)
        nobs_full = nobs.sum()

    #############################################################################
    #######################  calculate covariance matrix  #######################
    #############################################################################
    if verbose:
        print('Calculating covariance matrix...')

    rows = []
    rhss = []
    for ivar in range(nvar):
        columns = []
        for jvar in range(nvar):
            covfunc = cov[ivar][jvar]
            if jvar == ivar:
                columns.append(
                    auto_cov(obslocs[ivar], covfunc, distfunc, obstimes[ivar])
                )
            elif jvar > ivar:
                columns.append(
                    cross_cov(obslocs[ivar], obslocs[jvar], covfunc, distfunc, obstimes[ivar], obstimes[jvar])
                )
            else:
                columns.append(rows[jvar][ivar].T)
        rows.append(columns)
        rhss.append(
            cross_cov(covfunc=covfunc,
                      distfunc=distfunc,
                      slag=slags[ivar],
                      tlag=tlags[ivar])
        )

    #############################################################################
    #######################        assemble matrix        #######################
    #############################################################################
    # left hand side
    matsize = nobs_full + ndrift + unbias * (1 if standardized else nvar)
    matLHS = np.zeros([matsize, matsize])
    matLHS[:nobs_full, :nobs_full] = np.vstack([np.hstack(columns) for columns in rows]) # covariance (obs ~ obs)

    # right hand side covariance (obs ~ grid)
    matRHS = np.zeros([matsize, nnew])
    matRHS[:matsize] = np.vstack(rhss)

    # unbias
    if unbias:
        if standardized:
            matLHS[:nobs_full, nobs_full] = 1.0
            matLHS[nobs_full, :nobs_full] = 1.0
            matRHS[nobs_full] = 1.0
            avg0 = obsvals[0].mean()
            for i in range(1, nvar):
                avg1 = obsvals[i].mean()
                obsvals[i] = obsvals[i] + avg0 - avg1

        else:
            i1 = 0
            for ivar in range(nvar):
                n1 = nobs[i]
                matA[i1:i1+n1, nobs_full+i] = 1
                matA[nobs_full+i, i1:i1+n1] = 1
                i1 += n1

    # drifts
    if ndrift > 0:
        npp = nobs[0]
        matLHS[:npp, -ndrift:] = obsdrift
        matLHS[-ndrift:, :npp] = obsdrift.T
        matRHS[-ndrift:] = newdrift.T
    obsvals = np.concatenate(obsvals, axis=0)

    #############################################################################
    #######################        solving equation       #######################
    #############################################################################
    if verbose:
        print('Solving linear system...')

    weight_correction = weight_correction if (ndrift==0 and nvar==1) else False
    if search:
        nearests = np.vstack(nearests + [np.array([[r]*nnew for r in range(nobs_full, len(matLHS))]),])
        predict = np.zeros_like(np.array([obsval[0]] * nnew))
        variance = np.zeros(nnew)
        weights = np.full([nobs_full, nnew], np.nan)
        predict_obs = np.zeros(nobs_full)

        ilocs_ = pd.Series(np.arange(nnew), index=[tuple(iobs) for iobs in nearests.T])
        for iobs, inew in ilocs_.groupby(level=0):
            kobs = iobs[iobs>=0]
            predict[inew], variance[inew], weights[np.ix_(kobs,inew)], predict_obs[kobs] = solve_linear(
                matLHS[kobs][:, kobs],
                matRHS[kobs][:, inew],
                obsvals[kobs],
                cov0,
                verbose,
                usevgm,
                solver,
                validation,
                weight_correction
            )
    else:
        predict, variance, weights, predict_obs = solve_linear(
            matLHS, matRHS, obsvals, cov0, verbose, usevgm,
                 solver, validation, weight_correction)

    if write_mats:
        np.savetxt('predict.dat', predict.reshape([nnew, -1]), fmt='%.9g')
        np.savetxt('weights.dat', weights.T, fmt='%.9g')

    if return_factor:
        # make the factor stored by new locations
        solv_all = factor_all
    if return_variance:
        solv_all = (solv_all, variance)
    return solv_all
#%%
if __name__ == '__main__':


    matA = np.random.randn(1000, 1000)
    X  = np.random.randn(1000)

    rhs = matA.dot(X)

    def xsolve():
        xsolve = np.linalg.solve(matA, rhs)
        print(np.sum((X-xsolve)**2))
        # 9.28 ms ± 375 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        # 1.5327547379529084e-20

    def lstsq():
        xsolve = np.linalg.lstsq(matA, rhs)
        print(np.sum((X-xsolve[0])**2))
        # 196 ms ± 7.78 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
        # 1.0046094894026148e-23

    def pinv():
        xsolve = np.linalg.pinv(matA, ).dot(rhs)
        print(np.sum((X-xsolve)**2))
        # 392 ms ± 60.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # 2.0459067804488128e-23
