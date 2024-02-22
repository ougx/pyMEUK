# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 15:19:06 2022

@author: mou
"""
import pandas as pd
import numpy as np
from .common import euclidean_dist, weight_correcting
from sklearn.preprocessing import PowerTransformer

#%%
def sgsck(
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
    nsim :int=1,                            # number of realizations to generate; assuming constant random path among realizations
    approximate:bool=True,                  # when the matrix is singular, use an approximate estimate weighted by the covarance (point, obs)
    seed:int=None,
    verbose: bool=False,
    tranform: bool=False,               # transform the data using power transformation to normal distribution
):

    # Sequential Gaussian simulation:
    #    20240128: borrowed from: https://github.com/GatorGlaciology/GStatSim/blob/main/gstatsim.py


    _NPPMAX = 4000

    nnew = len(newloc)
    nobs = len(obsval)
    nvar = len(cov)
    nsgs = nnew + nobs
    hastime  = not (obstime is None or newtime is None)
    hasdrift = not (obsdrift is None or newdrift is None)

    if tranform:
        tf = PowerTransformer(method="box-cox" if np.sum(obsval<0)==0 else "yeo-johnson")
        obsval = tf.fit_transform(obsval)

    # construct random path and random variance
    rng = np.random.default_rng(seed)
    randompath = np.arange(nnew)
    rng.shuffle(randompath)
    randomvar = rng.standard_normal([nnew,nsim,])

    # reorder grid data
    newloc = newloc[randompath]
    if hastime:
        newtime = newtime[randompath]

    ndrift = 0
    if hasdrift:
        newdrift = newdrift[randompath]

    if hasdrift:
        obsdrift = np.array(obsdrift).reshape([nobs, -1])
        newdrift = np.array(newdrift).reshape([nnew, -1])
        ndrift = obsdrift.shape[1]


    distances = np.zeros([nsgs, nsgs])
    sgsloc = np.concatenate((obsloc,newloc, ), axis=0)

    for ic, cc in enumerate(sgsloc):
        distances[ic] = distfunc(cc.reshape([1, -1]), sgsloc)

    # calculate the total distance for maxdist
    if hastime:
        sgstime = np.concatenate((obstime,newtime,), axis=0)
        tlags = np.zeros([nsgs, nsgs])
        for ic, cc in enumerate(sgstime):
            tlags[ic] = np.abs(cc - sgstime)

    # set up parameter for search
    if nmax is None:
        nmax = min(nsgs-1, _NPPMAX)

    if nmin is None:
        nmin = min(max(3, nmax // 3),20)
    assert nmin < nmax, f'nmin=({nmin}) must be smaller than nmax=({nmax}).'

    search = (maxdist is not None) or (nmax < nsgs-1)
    if search:
        totaldistances = np.sqrt(distances ** 2 + tlag2sdist(tlags) ** 2) if hastime else distances

    # loop through the random locations
    if verbose:
        print('Solving system...')

    # inital index of obs for gaussian simulation (gs); which is the observation
    solv_all = np.zeros([nnew,nsim,])
    variance = np.zeros( nnew)
    obsval = np.array([obsval]*nsim).reshape([nsim,-1]).T

    for ix in range(nnew):

        ig = nobs+ix  # grid index of the location to be estimated
        matDist = distances[:ig,:ig]
        rhsDist = distances[:ig, ig]
        obsval_sgs = np.concatenate((obsval, solv_all[:ix], ), axis=0)

        if hastime:
            matTlag = tlags[:ig,:ig]
            rhsTlag = tlags[:ig, ig]
        if hasdrift:
            matDrift = np.concatenate((obsdrift, newdrift[:ix], ), axis=0)
            rhsDrift = newdrift[ig]

        if search:
            # search for nearest data points to krige
            dist_gs = totaldistances[:ig, ig]
            mask = np.array([True] * ig)
            if maxdist is not None:
                mask = mask & (dist_gs <= maxdist)
            if sum(mask) > nmax:
                idx = np.argpartition(dist_gs, nmax)      #
                mask[idx[nmax:]] = False
                # dist_gs[mask]

            if not all(mask):
                matDist = matDist[mask][:,mask]
                rhsDist = rhsDist[mask]
                obsval_sgs = obsval_sgs[mask]
                if hastime:
                    matTlag = matTlag[mask][:,mask]
                    rhsTlag = rhsTlag[mask]
                if hasdrift:
                    matDrift = matDrift[mask]
                    rhsDrift = rhsDrift[mask]

        # formulation
        npp = obsval_sgs.shape[0]
        matA = np.zeros([npp+ndrift+unbias, npp+ndrift+unbias])
        rhsB = np.zeros(npp+ndrift+unbias)

        # construct matrix
        matCov = np.zeros([npp, npp])
        np.fill_diagonal(matCov, cov(0, 0) if hastime else cov(0))

        for ic in range(npp): # diagonal is cov(o)
            slag = matDist[ic, ic+1:]
            if hastime:
                tlag = matTlag[ic, ic+1:]
                matCov[ic,ic+1:] = cov(slag, tlag)
            else:
                matCov[ic,ic+1:] = cov(slag)

            matCov[ic+1:, ic] = matCov[ic,ic+1:]

        matA[:npp, :npp] = matCov
        rhsB[:npp] = cov(rhsDist, rhsTlag) if hastime else cov(rhsDist)

        if unbias:
            matA[:npp, npp] = 1.0
            matA[npp, :npp] = 1.0
            rhsB[npp] = 1.0

        # drifts
        if hasdrift:
            matA[:npp, npp+unbias:] = matDrift
            matA[npp+unbias:, :npp] = matDrift.T
            rhsB[npp+unbias:] = rhsDrift


        if verbose:
            print('\n\n'+('='*80))
            print(f'Solving Problem ({npp} x 1) at {randompath[ix]}')
            print('\nmatA (nobs + unbias + ndrift)')
            print(matA)
            print('\nrhsB (nobs + unbias + ndrift)')
            print(rhsB)

        if write_mats:
            np.savetxt(f'matA_{randompath[ix]}.dat', matA, fmt='%.9g')
            np.savetxt(f'rhsB_{randompath[ix]}.dat', rhsB, fmt='%.9g')


        if solver=='lstsq':
            X = np.linalg.lstsq(matA, rhsB, rcond=None)[0]
        elif solver=='pinv':
            X = np.linalg.pinv(matA).dot(rhsB)
        else:
            try:
                X = np.linalg.solve(matA, rhsB)
            except:
                if approximate:
                    print(f'LinAlgError: Singular matrix when estimating {ix}:{randompath[ix]}; using covariance weighted mean.')
                    if usevgm:
                        if any(rhsB==0):
                            print('rhsB',    rhsB)
                            print('rhsDist', rhsDist)
                            raise ValueError('colocated data found.')
                        X = 1/rhsB/sum(1/rhsB[:npp])
                    else:
                        X = rhsB/sum(rhsB[:npp])
                else:
                    print(f'LinAlgError: Singular matrix when estimating {ix}:{randompath[ix]}; using least square.')
                    X = np.linalg.lstsq(matA, rhsB, rcond=None)[0]


        if weight_correction and unbias and ndrift == 0:
            # see Deutsch (1996) CORRECTING FOR NEGATIVE WEIGHTS IN ORDINARY KRIGING
            X = weight_correcting(X, rhsB, npp, usevgm)

        variance[ix] = X.dot(rhsB)
        if not usevgm:
            if hastime:
                variance[ix] = cov(0, 0) - variance[ix]
            else:
                variance[ix] = cov(0) - variance[ix]

        solv_all[ix] = X[:npp].dot(obsval_sgs) + randomvar[ix]*(variance[ix]**0.5)
        if verbose:
            print('\nWeights (nunknown x nobs)')
            print(X[:npp])
            print('\nResults (nunknown)')
            print(solv_all[ix])

        if write_mats:
            np.savetxt(f'weights_{randompath[ix]}.dat', X, delimiter='\n', fmt='%.9g')

    if tranform:
        solv_all = tf.inverse_transform(solv_all)

    idx = np.argsort(randompath)
    return solv_all[idx], variance[idx]
#%%
if __name__ == '__main__':


    nsize = 4000
    matA = np.random.randn(nsize, nsize)
    X  = np.random.randn(nsize)

    rhs = matA.dot(X)

    def xsolve():
        xsolve = np.linalg.solve(matA, rhs)
        print(np.sum((X-xsolve)**2))
        # 374 ms ± 14 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # 1.5327547379529084e-20

    def lstsq():
        xsolve = np.linalg.lstsq(matA, rhs)
        print(np.sum((X-xsolve[0])**2))
        # 16.9 s ± 915 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # 1.0046094894026148e-23

    def pinv():
        xsolve = np.linalg.pinv(matA, ).dot(rhs)
        print(np.sum((X-xsolve)**2))
        # 20.2 s ± 427 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        # 2.0459067804488128e-23
