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

itest = 0

#%% universal kriging with linear drift
def test_universal_riging_xdrift_ydrift():
    #itest += 1
    description = 'Universal kriging with X and Y Drifts'

    obsloc = obs[['x','y']].values
    obsval = obs.z.values
    newloc = get_grid([1,2,3,4])

    obsdrift = obsloc
    newdrift = newloc
    cov = lambda h: h
    z, v = pyMEUK.universal.uk5(obsloc, obsval, newloc, cov, obsdrift=obsdrift, newdrift=newdrift, usevgm=True, return_variance=True)

    correct_answer = np.array([
        [0.5131416,0.8309693],
        [0.6513749,0.7047971],
        [0.8725755,2.0565615],
        [1.1087445,3.7717392],
        [0.6270960,1.0805430],
        [0.8041303,1.4331352],
        [1.0259236,1.9876822],
        [1.2653798,2.6648658],
        [0.7071362,0.4014157],
        [0.9342224,1.2308741],
        [1.1792110,1.6046311],
        [1.4293635,1.5511439],
        [0.8590728,1.6467888],
        [1.0853218,1.3481777],
        [1.3335938,0.7972078],
        [1.5882967,0.7668964],
    ]).T

    assert np.allclose(z, correct_answer[0]) and np.allclose(v, correct_answer[1]), 'Failed: ' + description
