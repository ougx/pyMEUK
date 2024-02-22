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

from .common import cdist, pdist, euclidean_dist, weight_correcting
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


def uk4(obsloc, obsval, obsdrift, newloc, newdrift, vgm, nmax=None, nmin=10, maxdist=np.inf, vgmascov=False):

    # uk4: based on ok4, add drifts
    nnew = len(newloc)
    ndrift = obsdrift.shape[1]
    nobs_full = len(obsval)
    distances = cdist(newloc, obsloc)

    if nmax is None:
        search = False
        nmax = nobs_full
        ilocs_mask = np.full(nobs_full, True)
    else:
        search = True
        ilocs = np.sort(np.argsort(distances, axis=1)[:,:nmax], axis=1)
        ilocs_mask = np.isin(np.arange(nobs_full), np.unique(ilocs))


    cov_full = np.zeros([nobs_full, nobs_full])
    cov_full[np.ix_(ilocs_mask, ilocs_mask)] = squareform(vgm(pdist(obsloc[ilocs_mask])))

    solv_all = np.full_like(np.array([obsval[0]] * len(newloc)), np.nan)
    variance = np.full_like(np.array([obsval[0]] * len(newloc)), np.nan)

    def inverse(ii, ig):

        npp = nmax # TODO exclude points outside max distance
        matA = np.zeros([npp+ndrift+1, npp+ndrift+1])
        # unbias condition
        matA[:npp, npp] = 1
        matA[npp, :npp] = 1
        # drift
        matA[:npp, npp+1:] = obsdrift[ii]
        matA[npp+1:, :npp] = obsdrift[ii].T

        matA[:npp, :npp] = cov_full[np.ix_(ii, ii)]
        rhsB = np.insert(vgm(distances[np.ix_(ig, ii)]).T, npp, 1, axis=0) # unbias
        rhsB = np.vstack(rhsB, newdrift[ig].T)                             # drift

        # matAinv = np.linalg.inv(matA)
        # X = matAinv.dot(rhsB)
        X = np.linalg.solve(matA, rhsB)

        solv_all[ig] = X[:npp,:].T.dot(obsval[ii])
        variance[ig] = (X * rhsB).sum(axis=0).reshape(solv_all.shape)
        if vgmascov:
            variance[ig] = vgm(0) - variance[ig]
    if search:
        for ii in np.unique(ilocs, axis=0):
            inverse(ii, (ilocs == ii).all(axis=1))
    else:
        inverse(ii=ilocs_mask, ig=np.full(nnew, True))

    return solv_all, variance

#%%
def uk5(obsloc: np.array,
        obsval: np.array,
        newloc: np.array,
        cov: callable,
        obsdrift: np.array=None,
        newdrift: np.array=None,
        obstime: np.array=None,
        newtime: np.array=None,
        unbias: bool=True,
        nmax :int=None,
        nmin :int=None,
        maxdist: float=None,
        usevgm: bool=False,
        verbose: bool=False,
        distfunc: callable=euclidean_dist,
        tlag2sdist: callable=lambda t: t,
        solver:str='solve',
        weight_correction=False,
        write_mats: bool=False,
        return_factor: bool=False,
        return_variance: bool=False):

    # uk5: based on ok5, add drifts
    #      2023/10/23 Update: add time dimension for spatio-temporal Kriging;
    #                         remove unused observations when local search is used
    #                         transpose distances[nobs, nnew] array so it match other arrays
    nnew = len(newloc)
    nobs_full = len(obsval)
    hastime  = not (obstime is None or newtime is None)
    hasdrift = not (obsdrift is None or newdrift is None)

    solv_all = np.zeros(nnew, )
    if return_variance:
        variance = np.zeros(nnew, )
    if return_factor:
        factor_all = np.zeros([nnew, nobs_full])

    ndrift = 0
    if hasdrift:
        obsdrift = np.array(obsdrift).reshape([nobs_full, -1])
        newdrift = np.array(newdrift).reshape([nnew, -1])
        ndrift = obsdrift.shape[1]
    if verbose:
        print('Calculating covariance matrix...')
    distances = np.zeros([nobs_full, nnew])
    for ic, cc in enumerate(obsloc):
        distances[ic] = distfunc(cc.reshape([1, -1]), newloc)

    # calculate the total distance for maxdist
    if hastime:
        tlags = np.zeros([nobs_full, nnew])
        for ic, cc in enumerate(obstime):
            tlags[ic] = np.abs(cc - newtime)
        totaldistances = np.sqrt(distances ** 2 + tlag2sdist(tlags) ** 2)
    else:
        totaldistances = distances
    # get index of the `nmax` nearest points for each new location
    if nmax is None:
        nmax = nobs_full
    if nmin is None:
        nmin = min(max(3, nmax // 3),20)
    assert nmin < nmax, f'nmin=({nmin}) must be smaller than nmax=({nmax}).'
    search = (maxdist is not None) or (nmax < nobs_full)
    if search:
        ilocs = np.sort(np.argsort(totaldistances, axis=0)[:nmax], axis=0)
        iobs_orig = np.arange(nobs_full)

        search = True
        useful_locs = np.unique(ilocs) # sorted unique index that are useful
        nuseful = len(useful_locs)
        if nuseful < nobs_full:
            # remove the unused observations
            obsloc = obsloc[useful_locs]
            obsval = obsval[useful_locs]
            distances = distances[useful_locs]
            totaldistances = totaldistances[useful_locs]
            if hastime:
                obstime = obstime[useful_locs]
                tlags = tlags[useful_locs]
            if hasdrift:
                obsdrift = obsdrift[useful_locs]
            smallindex = np.full(nobs_full, -1, dtype=int)
            smallindex[useful_locs] = np.arange(nuseful)
            ilocs = smallindex[ilocs]
            iobs_orig = np.nonzero(smallindex+1)[0]
            nobs_full = nuseful


    # calculate the variance-covariance matrix
    cov_full = np.zeros([nobs_full, nobs_full])
    np.fill_diagonal(cov_full, cov(0, 0) if hastime else cov(0))

    for ic, oc in enumerate(obsloc[:-1]): # last one is cov(o)
        slag = distfunc(oc.reshape([1, -1]), obsloc[ic+1:])
        if hastime:
            tlag = np.abs(obstime[ic] - obstime[ic+1:])
            cov_full[ic,ic+1:] = cov(slag, tlag)
        else:
            cov_full[ic,ic+1:] = cov(slag)

        cov_full[ic+1:, ic] = cov_full[ic,ic+1:]


    if verbose:
        print('Solving linear system...')


    def inverse(ii, ig, nsolve=0):
        npredict = nnew if isinstance(ig, slice) else len(ig)
        if isinstance(ii, list):
            npp = len(ii)
            nsolve += npredict
            sys.stdout.write('\r{0:50s} {1:.0%}'.format('='*int(nsolve*50/nnew), nsolve/nnew))
            sys.stdout.flush()
        else:
            npp = nmax

        matA = np.zeros([npp+ndrift+unbias, npp+ndrift+unbias])
        rhsB = np.zeros([npp+ndrift+unbias, npredict])
        # unbias term
        matA[:npp, :npp] = cov_full[ii][:, ii]
        if unbias:
            matA[:npp, npp] = 1.0
            matA[npp, :npp] = 1.0
        # drifts
        if ndrift > 0:
            matA[:npp, npp+unbias:] = obsdrift[ii]
            matA[npp+unbias:, :npp] = obsdrift[ii].T

        if hastime:
            rhsB[:npp] = cov(distances[ii][:,ig], tlags[ii][:,ig])
        else:
            rhsB[:npp] = cov(distances[ii][:,ig])

        if unbias:
            rhsB[npp] = 1.0

        if ndrift > 0:
            rhsB[npp+unbias:] = newdrift[ig].T

        if verbose:
            print('\n\n'+('='*80))
            if search:
                print('data    locations:', iobs_orig[ii])
                print('unknown locations:', ig)

            print('\nmatA (nobs + unbias + ndrift)')
            print(matA)
            print('\nrhsB (nobs + unbias + ndrift)')
            print(rhsB)
        if write_mats:
            np.savetxt('matA.dat', matA, fmt='%.9g')
            np.savetxt('rhsB.dat', rhsB, fmt='%.9g')
        if verbose:
            print(f'Solving Problem ({npp} x {npredict})')
        if solver=='lstsq':
            X = np.linalg.lstsq(matA, rhsB, rcond=None)[0]
        else:
            try:
                X = np.linalg.solve(matA, rhsB)
            except:
                print('LinAlgError: Singular matrix; using the Least-Square method.')
                X = np.linalg.lstsq(matA, rhsB, rcond=None)[0]

        if weight_correction and unbias and ndrift == 0:
            # see Deutsch (1996) CORRECTING FOR NEGATIVE WEIGHTS IN ORDINARY KRIGING
            X = weight_correcting(X, rhsB, npp, usevgm)



        solv_all[ig] = X[:npp,:].T.dot(obsval[ii])

        if verbose:
            print('\nWeights (nunknown x nobs)')
            print(X[:npp,:].T)
            print('\nResults (nunknown)')
            print(solv_all[ig])

        if write_mats:
            np.savetxt('weights.dat', X[:npp,:].T, fmt='%.9g')

        if return_factor:
            factor_all[np.ix_(ig, iobs_orig[ii])] = X[:npp,:].T


        if return_variance:
            if usevgm:
                variance[ig] = (X * rhsB).sum(axis=0)
            else:
                if hastime:
                    variance[ig] = cov(0, 0) - (X * rhsB).sum(axis=0)
                else:
                    variance[ig] = cov(0) - (X * rhsB).sum(axis=0)
        return nsolve
    if search:
        nsolve = 0
        if maxdist is None:
            ilocs_ = pd.Series(np.arange(nnew), index=[tuple(iobs) for iobs in ilocs.T])
        else:
            ilocs_ = pd.Series(np.arange(nnew), index=[tuple(iobs[totaldistances[iobs, inew]<=maxdist]) for inew,iobs in enumerate(ilocs.T)])
            mask = [len(ii)<nmin for ii in ilocs_.index]
            if any(mask):
                raise ValueError(f'Observation count is smaller than nmin=({nmin}) within maxdist=({maxdist}) for locations:\n  {",".join((np.arange(nnew)[mask]).astype(str))}')

        for ii, ig in ilocs_.groupby(level=0):
            nsolve = inverse(list(ii), ig.values, nsolve)
        print('')
    else:
        inverse(ii=slice(None), ig=slice(None))

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
