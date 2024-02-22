import pyMEUK
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist
import pytest


#%% ordinary kriging; test is from https://geostat-framework.readthedocs.io/projects/pykrige/en/stable/examples/00_ordinary.html
obs = pd.DataFrame(dict(
    x=[0.3,1.9,1.1,3.3,4.7],
    y=[1.2,0.6,3.2,4.4,3.8],
    z=[0.47,0.56,0.74,1.47,1.74]
))


def get_grid(xx, yy=None):
    if yy is None:
        yy = xx
    xx, yy = np.meshgrid(xx, yy)
    return np.array([xx.flatten(), yy.flatten()], dtype=float).T


#%%  ordinary kriging with Linear variogram
def test_ordinary_kriging_lin():
    description = 'Ordinary kriging with linear variogram'
    obsloc = obs[['x','y']].values
    obsval = obs.z.values
    newloc = get_grid([1,2,3,4])
    cov = lambda h: h
    z, v =  pyMEUK.universal.uk5(obsloc, obsval, newloc, cov, usevgm=True, return_variance=True, verbose=False)

    #R Gstat Results
    correct_answer = np.array([
        [0.5152091,0.8294115],
        [0.6341512,0.6981659],
        [0.8528837,1.7881827],
        [1.0599441,2.8184175],
        [0.6120256,1.0789287],
        [0.7796277,1.4125679],
        [1.0064988,1.8628310],
        [1.2267659,2.2615966],
        [0.7104826,0.4000557],
        [0.9257884,1.2220828],
        [1.1827978,1.5615680],
        [1.4277976,1.4550653],
        [0.8775662,1.3993001],
        [1.0823395,1.3057872],
        [1.3429305,0.7962962],
        [1.5879789,0.7666918],
    ]).T

    assert np.allclose(z, correct_answer[0]) and np.allclose(v, correct_answer[1]), 'Failed: ' + description

#%% ordinary kriging with Gaussian variogram
def test_ordinary_kriging_gau():
    #    itest += 1
    description = 'Ordinary kriging with Gaussian variogram model'
    obsloc = obs[['x','y']].values
    obsval = obs.z.values
    newloc = get_grid([2,4])

    a = 2. # this is the parameter used in gstat for `Gau` function
    cov = lambda h: pyMEUK.vgm.v_gaussian (h, rng=a * 7 / 4)
    z1, v1 =  pyMEUK.universal.uk5(obsloc, obsval, newloc, cov, usevgm=True, return_variance=True)

    cov = lambda h: pyMEUK.vgm.c_gaussian (h, rng=a * 7 / 4)
    z2, v2 =  pyMEUK.universal.uk5(obsloc, obsval, newloc, cov, usevgm=False, return_variance=True)

    #R Gstat Results
    correct_answer = [0.6749654,1.2240703,1.0432411,1.6796103]
    correct_var    = [0.39560425,0.85535115,0.22686960,0.04655201]
    assert np.allclose(z1, correct_answer) and np.allclose(v1, correct_var) and np.allclose(z2, correct_answer) and np.allclose(v2, correct_var), 'Failed: ' + description

#%% ordinary kriging with max points against to R gstat
def test_ordinary_kriging_sph_nmax():
    description = 'Ordinary kriging with nmax'

    nmax=4
    verbose=False


    cov = lambda h: pyMEUK.vgm.c_spherical(h, rng=3)
    cov(4)

    obsloc = obs[['x','y']].values
    obsval = obs.z.values
    newloc = get_grid([1,2,3,4])

    z = pyMEUK.universal.uk5(obsloc, obsval, newloc, cov, nmax=nmax, verbose=verbose) #return_factor=True,

    correct_answer = np.array([
        0.5237045,
        0.5991352,
        0.8024068,
        1.0125974,
        0.5978558,
        0.6938209,
        0.9694113,
        1.1597862,
        0.7134429,
        0.8788764,
        1.1887870,
        1.4294958,
        0.8326374,
        1.0407185,
        1.3415545,
        1.5955344,
    ]).flatten()
    assert np.allclose(z, correct_answer), 'Failed: ' + description

#%% ordinary kriging with max distance
def test_ordinary_kriging_sph_maxdist():
    description = 'Ordinary kriging with max distance'

    cov = lambda h: pyMEUK.vgm.c_spherical(h, rng=3)

    obsloc = obs[['x','y']].values
    obsval = obs.z.values
    newloc = get_grid([1,2,3,4])

    z = pyMEUK.universal.uk5(obsloc, obsval, newloc, cov, maxdist=4, nmin=2) #return_factor=True,

    correct_answer = np.array([
        0.5178887,
        0.6171675,
        0.8369310,
        0.9526941,
        0.5978558,
        0.7352704,
        0.9413668,
        1.0989482,
        0.7260300,
        0.8894276,
        1.1640555,
        1.4294958,
        0.8794386,
        1.0124408,
        1.3346547,
        1.5955344,
    ]).flatten()
    assert np.allclose(z, correct_answer), 'Failed: ' + description


#%% ordinary kriging with nugget
def test_ordinary_kriging_nugget():
    description = 'Ordinary kriging with nugget'

    cov = lambda h: pyMEUK.vgm.c_spherical(h, rng=3, nugget=1)

    obsloc = obs[['x','y']].values
    obsval = obs.z.values
    newloc = get_grid([1,2,3,4])

    z, v = pyMEUK.universal.uk5(obsloc, obsval, newloc, cov, return_variance=True,) #

    correct_answer = np.array([
        [0.7324085,1.781001],
        [0.7803295,1.794066],
        [0.9045789,2.188303],
        [0.9709269,2.403917],
        [0.7801825,1.883232],
        [0.8493258,2.055560],
        [0.9646338,2.277781],
        [1.0498966,2.349489],
        [0.8468440,1.709788],
        [0.9416096,2.008485],
        [1.0946124,2.133852],
        [1.2151822,2.056502],
        [0.9335395,2.029077],
        [1.0155624,2.027375],
        [1.2001897,1.823619],
        [1.3282070,1.757926],
    ]).T
    assert np.allclose(z, correct_answer[0]) and np.allclose(v, correct_answer[1]), 'Failed: ' + description


#%% space time ordinary kriging
def test_st_ordinary_kriging():
    description = 'Spatiotemporal Ordinary Kriging'
    obsloc = obs[['x','y']].values
    obsloc = np.concatenate([obsloc]*4, axis=0)
    obstime=np.repeat(np.arange(4), 5)
    obsval=np.array([
        0.47,
        0.56,
        0.74,
        1.47,
        1.74,
        1.449745222,
        1.240477328,
        1.569039014,
        1.712168813,
        2.708559412,
        0.748497767,
        0.822926057,
        1.037391439,
        1.563942689,
        2.024361468,
        0.762813891,
        1.400204895,
        0.967438306,
        1.688028078,
        2.292060103
    ])

    # st_anistropy = 1.0
    # tlag2sdist = lambda t: st_anistropy * t

    vs = lambda h: pyMEUK.vgm.c_spherical(h, rng=3)
    vt = lambda t: pyMEUK.vgm.c_gaussian (t, rng=3.5)
    cov = lambda h, t: vs(h) * vt(t)

    newloc = np.concatenate([get_grid([2,4])]*2, axis=0)
    newtime=np.array([1.5]*4 +[2.5]*4)

    z = pyMEUK.universal.uk5(obsloc, obsval, newloc, cov,
                            obstime=obstime, newtime=newtime, verbose=False)

    correct_answer = np.array([
        1.2002896,
        1.3842994,
        1.3180261,
        2.0130670,
        0.9487037,
        1.3051629,
        1.1409658,
        1.7529133,
    ]).flatten()
    assert np.allclose(z, correct_answer), 'Failed: ' + description

