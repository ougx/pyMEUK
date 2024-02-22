# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:52:33 2022

@author: michaelou
"""
import numpy as np
# from scipy.linalg import solve_triangular as solve
from scipy.spatial.distance import cdist, pdist, squareform
import scipy


#%% with nmax
def ok1(obsloc, obsval, newloc, vgm, nmax=None):

    nnew = len(newloc)
    distances = cdist(newloc, obsloc)
    search = False

    if nmax is None:
        npp = len(obsval)
    else:
        npp = nmax
        idistance = np.sort(np.argsort(distances, axis=1)[:,:npp], axis=1)
        idistance_unique = np.unique(idistance, axis=0)
        search = True


    matA = np.zeros([npp+1, npp+1])
    matA[:, -1] = 1
    matA[-1, :] = 1
    matA[npp:, npp:] = 0

    solv_all = np.full_like(np.array([obsval[0]] * len(newloc)), np.nan)

    def inverse(ii, ig):
        matA[:npp, :npp] = squareform(vgm(pdist(obsloc[ii])))
        matAinv = np.linalg.inv(matA)

        rhsB = np.insert(vgm(distances[ig][:, ii]), npp, 1, axis=1)
        solv = matAinv.dot(rhsB.T)
        solv_all[ig] = solv[:npp,:].T.dot(obsval[ii])

    if search:
        for ii in idistance_unique:
            inverse(ii, (idistance == ii).all(axis=1))
    else:
        inverse(range(npp), range(nnew))


    return solv_all


# %timeit Kh = ok()
# %timeit Kh = ok(16)
# print(Kh)

# np.allclose(ok(obsloc, obsval, newloc, vgm), ok1(obsloc, obsval, newloc, vgm))



#%%
def ok2(obsloc, obsval, newloc, vgm, nmax=None, nmin=10, maxdist=np.inf):

    nnew = len(newloc)
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

    rhs_full = np.zeros([nnew, nobs_full])
    rhs_full[:,ilocs_mask] = vgm(distances[:, ilocs_mask])

    solv_all = np.full_like(np.array([obsval[0]] * len(newloc)), np.nan)

    def inverse(ii, ig):

        npp = nmax # TODO exclude points outside max distance
        matA = np.zeros([npp+1, npp+1])
        matA[:, -1] = 1
        matA[-1, :] = 1
        matA[npp:, npp:] = 0

        matA[:npp, :npp] = cov_full[np.ix_(ii, ii)]
        matAinv = np.linalg.inv(matA)

        rhsB = np.insert(vgm(distances[np.ix_(ig, ii)]).T, npp, 1, axis=0)
        X = matAinv.dot(rhsB)

        solv_all[ig] = X[:npp,:].T.dot(obsval[ii])

    if search:
        for ii in np.unique(ilocs, axis=0):
            inverse(ii, (ilocs == ii).all(axis=1))
    else:
        inverse(ilocs_mask, np.full(nnew, True))


    return solv_all

# np.allclose(ok(obsloc, obsval, newloc, vgm), ok2(obsloc, obsval, newloc, vgm))



#%%
def ok3(obsloc, obsval, newloc, vgm, nmax=None, nmin=10, maxdist=np.inf):

    nnew = len(newloc)
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

    rhs_full = np.zeros([nnew, nobs_full])
    rhs_full[:,ilocs_mask] = vgm(distances[:, ilocs_mask])

    solv_all = np.full_like(np.array([obsval[0]] * len(newloc)), np.nan)

    def inverse(ii, ig):

        npp = nmax # TODO exclude points outside max distance
        matA = np.zeros([npp+1, npp+1])
        matA[:, -1] = 1
        matA[-1, :] = 1
        matA[npp:, npp:] = 0

        matA[:npp, :npp] = cov_full[np.ix_(ii, ii)]
        rhsB = np.insert(vgm(distances[np.ix_(ig, ii)]).T, npp, 1, axis=0)

        # matAinv = np.linalg.inv(matA)
        # X = matAinv.dot(rhsB)
        X = np.linalg.solve(matA, rhsB)

        solv_all[ig] = X[:npp,:].T.dot(obsval[ii])

    if search:
        for ii in np.unique(ilocs, axis=0):
            inverse(ii, (ilocs == ii).all(axis=1))
    else:
        inverse(ii=ilocs_mask, ig=np.full(nnew, True))

    return solv_all

# np.allclose(ok(obsloc, obsval, newloc, vgm), ok3(obsloc, obsval, newloc, vgm))


#%%
def ok4(obsloc, obsval, newloc, vgm, nmax=None, nmin=10, maxdist=np.inf, vgmascov=False):


    # ok3: use np.linalg.solve instead of np.linalg.inv (faster)
    # ok4: add covariance function option
    nnew = len(newloc)
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

    solv_all = np.full_like(np.array([obsval[0]] * nnew), np.nan)
    variance = np.zeros(nnew, )

    def inverse(ii, ig):

        npp = nmax # TODO exclude points outside max distance
        ncell = np.count_nonzero(ig)
        matA = np.zeros([npp+1, npp+1])
        matA[:, -1] = 1
        matA[-1, :] = 1
        matA[npp:, npp:] = 0

        matA[:npp, :npp] = cov_full[np.ix_(ii, ii)]
        rhsB = np.insert(vgm(distances[np.ix_(ig, ii)]).T, npp, 1, axis=0)

        # matAinv = np.linalg.inv(matA)
        # X = matAinv.dot(rhsB)
        X = np.linalg.solve(matA, rhsB)

        solv_all[ig] = X[:npp,:].T.dot(obsval[ii])
        variance[ig] = (X * rhsB).sum(axis=0)
        if vgmascov:
            variance[ig] = vgm(0) - variance[ig]
    if search:
        for ii in np.unique(ilocs, axis=0):
            inverse(ii, ig=(ilocs == ii).all(axis=1))
    else:
        inverse(ii=ilocs_mask, ig=np.full(nnew, True))

    return solv_all, variance

#%%
def ok5(obsloc, obsval, newloc, cov, obstime=None, newtime=None, nmax=None, nmin=10, maxdist=np.inf, usevgm=False, verbose=False):


    # ok3: use np.linalg.solve instead of np.linalg.inv (faster)
    # ok4: add covariance function option
    # ok5: use cov by default; add other dimension (time)
    nnew = len(newloc)
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
    cov_full[np.ix_(ilocs_mask, ilocs_mask)] = squareform(cov(pdist(obsloc[ilocs_mask])))
    np.fill_diagonal(cov_full, cov(0))

    solv_all = np.full_like(np.array([obsval[0]] * nnew), np.nan)
    variance = np.zeros(nnew, )

    def inverse(ii, ig):

        npp = nmax # TODO exclude points outside max distance
        # ncell = np.count_nonzero(ig)
        matA = np.zeros([npp+1, npp+1])
        matA[:, -1] = 1
        matA[-1, :] = 1
        matA[npp:, npp:] = 0

        matA[:npp, :npp] = cov_full[np.ix_(ii, ii)]
        rhsB = np.insert(cov(distances[np.ix_(ig, ii)]).T, npp, 1, axis=0)

        if verbose:
            print('matA')
            print(matA)
            print('rhsB')
            print(rhsB)

        try:
            # %timeit X = scipy.linalg.solve(matA, rhsB, check_finite=False, assume_a='sym')
            # %timeit X = scipy.linalg.solve_triangular(matA, rhsB, check_finite=False, )
            # %timeit X = np.linalg.solve(matA, rhsB, )
            X = np.linalg.solve(matA, rhsB)
        except:
            print('LinAlgError: Singular matrix; using the (Moore-Penrose) pseudo-inverse of a matrix.')
            matAinv = np.linalg.pinv(matA)
            X = matAinv.dot(rhsB)

        solv_all[ig] = X[:npp,:].T.dot(obsval[ii])

        if usevgm:
            variance[ig] = (X * rhsB).sum(axis=0)
        else:
            variance[ig] = cov(0) - (X * rhsB).sum(axis=0)
    if search:
        for ii in np.unique(ilocs, axis=0):
            inverse(ii, ig=(ilocs == ii).all(axis=1))
    else:
        inverse(ii=ilocs_mask, ig=np.full(nnew, True))

    return solv_all, variance

#%% deprecated
#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
