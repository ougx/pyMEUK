from scipy.spatial.distance import cdist as _cdist, pdist as _pdist
import numpy as np

_metric = 'euclidean'

def cdist(XA, XB):
    return _cdist(XA, XB, metric=_metric)

def pdist(X):
    return _pdist(X, metric=_metric)

def get_pair_index(n, n1=None, full=False):
    n1 = n1 or n
    i1, i2 = np.meshgrid(range(n), range(n1), )
    if full:
        return np.array([i1.flatten(), i2.flatten(), ]).T
    else:
        ii = i1 > i2
        return np.array([ i1[ii], i2[ii],]).T

def weight_correcting(X, rhsB, npp=None, usevgm=False):

    # Deutsch (1996) CORRECTING FOR NEGATIVE WEIGHTS IN ORDINARY KRIGING
    npp = npp or len(X)
    Xhat = X[:npp]
    XnegAvg = np.apply_along_axis(lambda x: np.mean(-x[x<0] if sum(x<0)>0 else 0), 0, Xhat)
    CnegAvg = np.apply_along_axis(lambda x: np.mean(-x[x<0] if sum(x<0)>0 else 0), 0, np.where(Xhat<0,-rhsB[:npp],0))
    Xhat[Xhat<0] = 0
    if X.ndim > 1:
        CnegAvg = CnegAvg.reshape([1,-1])
        XnegAvg = XnegAvg.reshape([1,-1])
    mask = (rhsB[:npp]>CnegAvg) if usevgm else (rhsB[:npp]<CnegAvg)
    mask = mask & (Xhat > 0) & (Xhat<XnegAvg)
    Xhat[mask] = 0
    X[:npp] = Xhat / Xhat.sum(axis=0).reshape([1,-1])
    return X


def get_pair_index2(n):
    """
    get_pair_index : 707 ms ± 66.7 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    get_pair_index2 : 24.5 s ± 88.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    """
    res = []
    for i1 in range(n):
        res += [[i1, i2] for i2 in range(i1+1, n)]
    return np.array(res)

def abs_dist(coord1, coord2):
    return np.abs(coord1 - coord2)

def euclidean_dist(coord1, coord2):
    return np.sqrt(np.sum((coord1 - coord2) ** 2, axis=1))

def great_circle_dist(coord1, coord2, earthR=6371.2): # assume the coords are long,lat
    """
    The Great Circle distance formula computes the shortest distance path of two points on the surface of the sphere.
    """
    coord1 = np.radians(coord1)
    coord2 = np.radians(coord2)
    lon1 = coord1[:,0]
    lat1 = coord1[:,1]
    lon2 = coord2[:,0]
    lat2 = coord2[:,1]
    res = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    return earthR * np.arccos(np.where(res < 0, res + np.pi, res))

def cross_dist(locs0, locs1, distfunc, times0=None, times1=None, tlag2slag=None):
    hastime  = not (times0 is None or times1 is None)
    if hastime and tlag2slag is None:
        tlag2slag = lambda t: t

    # distance matrix: rows are locs0 and columns are locs1
    distfunc = distfunc or euclidean_dist
    slag = np.array([distfunc(cc.reshape([1, -1]), locs1) for ic, cc in enumerate(locs0)])

    # calculate the total distance for maxdist
    if hastime:
        tlag = np.array([np.abs(cc - times1) for ic, cc in enumerate(times0)])
        totallag = np.sqrt(slag ** 2 + tlag2slag(tlag) ** 2)
    else:
        tlags = None
        totallag = slag
    return slag, tlag, totallag


def get_nearest(obsloc, newloc, distfunc, nmax, maxdist, tlag2slag=None, obstime=None, newtime=None):
    """calculate the nearest `nmax` `obsloc` points for each `newloc`

    Args:
        obsloc (_type_): _description_
        newloc (_type_): _description_
        distfunc (_type_): _description_
        nmax (_type_, optional): _description_. Defaults to None.
        maxdist (_type_, optional): _description_. Defaults to None.
        nmin (_type_, optional): _description_. Defaults to None.
        tlag2slag (_type_, optional): function converting the time values to spatial values. Defaults to None.
        obstime (_type_, optional): _description_. Defaults to None.
        newtime (_type_, optional): _description_. Defaults to None.
    """

    nobs = len(obsloc)

    slag, tlag, totallag = cross_dist(obsloc, newloc, distfunc, obstime, newtime, tlag2slag)

    # check if local search is needed
    localsearch = False
    if maxdist is not None:
        if any(totallag.flatten()>maxdist):
            localsearch = True

    if nmax < nobs:
        localsearch = True

    # do local search if it is needed
    nearest = None
    iuseful = np.arange(nobs)
    if localsearch:
        nearest = np.sort(np.argsort(totallag, axis=0)[:nmax], axis=0)
        if maxdist is not None:
            nearest = np.where(
                np.array([totallag[nearest[:,ic], ic] for ic in range(totallag.shape[1])]).T < maxdist,
                nearest, -1, )

        iuseful = np.unique(nearest.flatten()) # get unique index of all useful points
        iuseful = iuseful[iuseful<nobs]
    return localsearch, slag, tlag, totallag, iuseful, nearest

def remove_unused_obs(iuseful, obsloc, obsval, slag, totallag, obstime=None, tlag=None, ):
    return (
        obsloc  [iuseful],
        obsval  [iuseful],
        obstime [iuseful] if obstime is not None else None,
        slag    [iuseful],
        tlag    [iuseful] if tlag is not None else None,
        totallag[iuseful],
    )

def auto_cov(obsloc, covfunc, distfunc, obstime=None):
    nobs = len(obsloc)
    hastime = obstime is not None
    distfunc = distfunc or euclidean_dist
    matcov = np.zeros([nobs, nobs])
    np.fill_diagonal(matcov, covfunc(0, 0) if hastime else covfunc(0))
    for ic, oc in enumerate(obsloc[:-1]): # last one is cov(o)
        slag = distfunc(oc.reshape([1, -1]), obsloc[ic+1:])
        if hastime:
            tlag = np.abs(obstime[ic] - obstime[ic+1:])
            matcov[ic,ic+1:] = covfunc(slag, tlag)
        else:
            matcov[ic,ic+1:] = covfunc(slag)
        matcov[ic+1:, ic] = matcov[ic,ic+1:]
    return matcov

def cross_cov(obsloc0=None, obsloc1=None, covfunc=None, distfunc=None, obstime0=None, obstime1=None, slags=None, tlags=None):
    if slag is None:
        nobs0 = len(obsloc0)
        nobs1 = len(obsloc1)
        hastime = obstime0 is not None
        distfunc = distfunc or euclidean_dist
        matcov = np.zeros([nobs0, nobs1])
        for ic, oc in enumerate(obsloc0):
            slag = distfunc(oc.reshape([1, -1]), obsloc1)
            if hastime:
                tlag = np.abs(obstime0[ic] - obstime1)
                matcov[ic] = covfunc(slag, tlag)
            else:
                matcov[ic] = covfunc(slag)
    else:
        hastime = tlag is not None
        for ic, oc in enumerate(obsloc0):
            slag = distfunc(oc.reshape([1, -1]), obsloc1)
            if hastime:
                tlag = np.abs(obstime0[ic] - obstime1)
                matcov[ic] = covfunc(slag, tlag)
            else:
                matcov[ic] = covfunc(slag)

    return matcov

def cross_validate(krige, obsloc, obsval, *args, **kwargs):
    '''


    Parameters
    ----------
    krige : function
        DESCRIPTION.
    obsloc : TYPE
        DESCRIPTION.
    obsval : TYPE
        DESCRIPTION.
    *args : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''

    idx = np.arange(len(obsval))
    if 'newloc' in kwargs:
        kwargs.drop('newloc')
    kwargs['return_factor'] = False
    kwargs['return_variance'] = False
    newval = [krige(obsloc=obsloc[idx!=i], obsval=obsval[idx!=i], newloc=obsloc[i:i+1], *args, **kwargs)[0] for i in idx]
    return np.array(newval)

def solve_linear(lhs, rhs, obsvals, cov0=0, verbose=0, usevgm=False,
                 solver='solve', validation=False, weight_correction=False):
    nobs = len(obsvals)
    if validation:
        matAinv = np.linalg.inv(lhs)
        X = matAinv.dot(rhs)

        # cross-validation: see p215 Chiles and Delfiner Geostatistics Modeling Spatial Uncertainty 2nd ed
        matAinv = matAinv[:nobs,:nobs]
        predict_obs = []
        idx = np.arange(nobs)
        for i in range(nobs):
            mask = idx != i
            w = -matAinv[i, mask]/matAinv[i,i]
            predict_obs.append((w*obsvals[mask]).sum())
        predict_obs = np.array(predict_obs)
    else:
        predict_obs = None
        if solver=='lstsq':
            X = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        else:
            try:
                X = np.linalg.solve(lhs, rhs)
            except:
                if verbose:
                    print('LinAlgError: Singular matrix; using the Least-Square method.')
                X = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    if weight_correction:
        # see Deutsch (1996) CORRECTING FOR NEGATIVE WEIGHTS IN ORDINARY KRIGING
        X = weight_correcting(X, rhs, nobs, usevgm)

    weights = X[:nobs,:]

    variance = (X * rhs).sum(axis=0)
    if not usevgm:
        variance = cov0 - variance
    predict = weights.T.dot(obsvals)
    return predict, variance, weights, predict_obs
