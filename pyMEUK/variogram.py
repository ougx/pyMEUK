# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 14:48:15 2022

@author: michaelou
"""

import pandas as pd
import numpy as np
import sys
from .common import cdist, pdist, get_pair_index, great_circle_dist, euclidean_dist
#%%
class vgm:

    # @staticmethod
    # def raw_variogram1(coords, vals, times=None, cutoff=None, t_cutoff=1e30):
    #     """_summary_

    #     Args:
    #         v (dict): dictionary of vriogram `type` and `parameters`
    #     """
    #     coords, vals = np.array(coords), np.array(vals)
    #     n, d = coords.shape[:2]
    #     coords = coords.reshape([n, -1])
    #     # assert coords.ndim == 2, 'number of dimensions must be 2 for the coordinates.'
    #     assert n == len(vals), 'lengths of coords and vals are different'
    #     if cutoff is None:
    #         pp = get_pair_index(n)
    #         h = np.sum((coords[pp[:,0]] - coords[pp[:,1]]) ** 2, axis=1) ** 0.5
    #         v = (vals[pp[:,0]] - vals[pp[:,1]]) ** 2 / 2
    #         if times is not None:
    #             times = np.array(times)
    #             t = np.abs(times[pp[:,0]] - times[pp[:,1]])
    #     else:
    #         hh = []
    #         tt = []
    #         vv = []
    #         t = 0
    #         for i1 in range(n):
    #             for i2 in range(i1+1, n):
    #                 h = np.sum((coords[i1] - coords[i2]) ** 2) ** 0.5
    #                 if h > cutoff:
    #                     continue
    #                 if times is not None:
    #                     t = np.abs(times[i1] - times[i2])
    #                     if t > t_cutoff:
    #                         continue
    #                 v = (vals[i1] - vals[i2]) ** 2 / 2
    #                 hh.append(h)
    #                 tt.append(t)
    #                 vv.append(v)
    #         h = np.array(hh)
    #         v = np.array(vv)
    #         t = np.array(tt)
    #     return [h, v] if times is None else [h, v, t]

    # @staticmethod
    # def raw_variogram0(coords, vals, cutoff=np.inf, times=None, t_cutoff=np.inf, distfuc=euclidean_dist):
    #     """_summary_

    #     Args:
    #         v (dict): dictionary of vriogram `type` and `parameters`

    #     deprecated due to highly fractured lists; takes long to concatenate to form array (final step)
    #     """
    #     coords, vals = np.array(coords), np.array(vals)
    #     n, d = coords.shape[:2]
    #     coords = coords.reshape([n, -1])
    #     # assert coords.ndim == 2, 'number of dimensions must be 2 for the coordinates.'
    #     assert n == len(vals), 'lengths of coords and vals are different'

    #     hh = []
    #     tt = []
    #     vv = []

    #     hasTime = times is not None

    #     totalh = n * (n-1) / 2
    #     calc = 0

    #     for i1 in range(n):
    #         h = distfuc(coords[i1:i1+1], coords[i1+1:])
    #         mask = h<=cutoff
    #         if hasTime:
    #             t = np.abs(times[i1] - times[i1+1:])
    #             mask = mask & (t<=t_cutoff)
    #             tt.append(t[mask])

    #         hh.append(h[mask])
    #         v = (vals[i1] - vals[np.arange(i1+1,n)[mask]]) ** 2 / 2
    #         vv.append(v)
    #         calc += n - i1
    #         sys.stdout.write('\r{0:50s} {1:.0%}'.format('='*int(calc*50/totalh), calc/totalh))
    #         sys.stdout.flush()

    #     print('')
    #     if hasTime:
    #         return np.concatenate(hh), np.concatenate(vv), np.concatenate(tt)
    #     else:
    #         return np.concatenate(hh), np.concatenate(vv)


    @staticmethod
    def raw_variogram(coords, vals, cutoff=np.inf, times=None, t_cutoff=np.inf, distfuc=euclidean_dist, return_index=False):
        """_summary_

        Args:
            v (dict): dictionary of vriogram `type` and `parameters`
        """
        coords, vals = np.array(coords), np.array(vals)
        vals = vals.flatten()
        n = len(vals)
        # coords = coords.reshape([n, -1])
        # assert coords.ndim == 2, 'number of dimensions must be 2 for the coordinates.'
        assert n == len(coords), 'lengths of coords and vals are different'


        hasTime = times is not None

        totalh = int(n * (n-1) / 2)
        calc = 0

        hh = np.empty(totalh)
        tt = np.empty(totalh)
        vv = np.empty(totalh)
        if return_index:
            ij1 = np.empty(totalh, dtype=int)
            ij2 = np.empty(totalh, dtype=int)
        icurrent = 0
        
        for i1 in range(n-1):
            h = distfuc(coords[i1:i1+1], coords[i1+1:])
            mask = h<=cutoff
            nh = np.count_nonzero(mask)
            if hasTime:
                t = np.abs(times[i1] - times[i1+1:])
                mask = mask & (t<=t_cutoff)
                nh = np.count_nonzero(mask)
                tt[icurrent:icurrent+nh] = t[mask]
            hh[icurrent:icurrent+nh] = h[mask]
            v = (vals[i1] - vals[i1+1:][mask]) ** 2 / 2
            vv[icurrent:icurrent+nh] = v
            if return_index:
                ij1[icurrent:icurrent+nh] = i1
                ij2[icurrent:icurrent+nh] = np.arange(i1+1, n)[mask]
                
            icurrent += nh
            calc += n - i1
            sys.stdout.write('\r{0:50s} {1:.0%}'.format('='*int(calc*50/totalh), calc/totalh))
            sys.stdout.flush()

        # aaa = pd.DataFrame(dict(time=times[i1+1:][mask],value=vals[i1+1:][mask],dt=t[mask],dh=h[mask],v=v))
        # aaa[['x', 'y']] = coords[i1+1:][mask]
        # aaa.to_csv(r'c:\Cloud\OneDrive - University of Nebraska-Lincoln\projects\2022_MRNRD_Model\06_Model\PEST\vgm_debug.csv', )
        print('')
        hh = hh[:icurrent]
        vv = vv[:icurrent]
        if hasTime: tt = tt[:icurrent]
        results = [hh, vv, tt] if hasTime else [hh, vv]
        if return_index:
            results += [ij1, ij2]
        return results

    # @staticmethod
    # def raw_variogram2(coords, vals, cutoff=np.inf, times=None, t_cutoff=np.inf, distfuc=euclidean_dist):
    #     """_summary_

    #     Args:
    #         v (dict): dictionary of vriogram `type` and `parameters`

    #     deprecated due to highly fractured lists; takes long to form array or dataframe
    #     """
    #     coords, vals = np.array(coords), np.array(vals)
    #     n, d = coords.shape[:2]
    #     coords = coords.reshape([n, -1])
    #     # assert coords.ndim == 2, 'number of dimensions must be 2 for the coordinates.'
    #     assert n == len(vals), 'lengths of coords and vals are different'

    #     hh = []
    #     tt = []
    #     vv = []

    #     hasTime = times is not None

    #     totalh = n * (n-1) / 2
    #     calc = 0

    #     for i1 in range(n):
    #         h = distfuc(coords[i1:i1+1], coords[i1+1:])
    #         mask = h<=cutoff
    #         if hasTime:
    #             t = np.abs(times[i1] - times[i1+1:])
    #             mask = mask & (t<=t_cutoff)
    #             tt += t[mask].tolist()

    #         hh += h[mask].tolist()
    #         v = (vals[i1] - vals[np.arange(i1+1,n)[mask]]) ** 2 / 2
    #         vv += v.tolist()
    #         calc += n - i1
    #         sys.stdout.write('\r{0:50s} {1:.0%}'.format('='*int(calc*50/totalh), calc/totalh))
    #         sys.stdout.flush()

    #     print('')

    #     return [hh, vv, tt] if hasTime else [hh, vv]

    @staticmethod
    def cross_variogram(coordsA, valsA, coordsB, valsB, cutoff=np.inf, residual=True,
                        timesA=None, timesB=None, t_cutoff=np.inf, distfuc=euclidean_dist):
        """_summary_

        Args:
            v (dict): dictionary of vriogram `type` and `parameters`
        """
        coordsA, valsA = np.array(coordsA), np.array(valsA)
        coordsB, valsB = np.array(coordsB), np.array(valsB)
        # assert coordsA.ndim == 2, 'A 2-dimensional array must be passed for coordsA.'
        # assert coordsB.ndim == 2, 'A 2-dimensional array must be passed for coordsB.'
        nA, dA = coordsA.shape
        nB, dB = coordsB.shape
        coordsA = coordsA.reshape([nA, -1])
        coordsB = coordsB.reshape([nB, -1])
        assert nA == len(valsA) and nB == len(valsB), 'lengths of coords and vals are different'

        meanA = np.mean(valsA) if residual else 0.
        meanB = np.mean(valsB) if residual else 0.

        hasTime = not (timesA is None or  timesB is None)

        hh = []
        tt = []
        vv = []
        for i1 in range(nA):
            h = distfuc(coordsA[i1:i1+1], coordsB)
            mask = h<=cutoff
            if hasTime:
                t = np.abs(timesA[i1] - timesB)
                mask = mask & (t<=t_cutoff)
                tt.append(t[mask])
            hh.append(h[mask])
            v = (valsA[i1] - valsB[mask] - meanA + meanB) ** 2 / 2
            vv.append(v)
            sys.stdout.write('\r{0:50s} {1:.0%}'.format('='*int((i1+1)*50/nA), (i1+1)/nA))
            sys.stdout.flush()

        if hasTime:
            return np.concatenate(hh), np.concatenate(vv), np.concatenate(tt)
        else:
            return np.concatenate(hh), np.concatenate(vv)


    @staticmethod
    def calc_avg_variogram(
            h, v, t=None, cutoff=None, width=None, bins=None,
            t_cutoff=None, t_width=None, t_bins=None, ):
        """
        calculate the average emperical variogram values

        Parameters
        ----------
        h : TYPE
            DESCRIPTION.
        v : TYPE
            DESCRIPTION.
        width : TYPE, optional
            DESCRIPTION. The default is None.
        bins : TYPE, optional
            DESCRIPTION. The default is None.
        cutoff : TYPE, optional
            DESCRIPTION. The default is np.inf.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        hv = pd.DataFrame({'h':h,'v':v}) if t is None else pd.DataFrame({'h':h,'v':v, 't':t})
        if cutoff is None:
            cutoff = h.max()
        else:
            hv = hv[hv.h<cutoff]

        if t is not None:
            if t_cutoff is None:
                t_cutoff = t.max()
            else:
                hv = hv[hv.t<=t_cutoff]

        if width is not None:
            bins = np.arange(0, cutoff+width, width)
        if bins is None:
            bins = 15
        if type(bins) is int:
            bins = np.linspace(0, cutoff, bins+1)
        # bins[0] -= 1
        hv['iih'] = np.searchsorted(bins, hv.h)
        nlocs = ['iih']

        if t is not None:
            if t_width is not None:
                t_bins = np.arange(0, t_cutoff+t_width, t_width)
            if t_bins is None:
                t_bins = 15
            if type(t_bins) is int:
                t_bins = np.linspace(0, t_cutoff, t_bins+1)
            # t_bins[0] -= 1
            hv['iit'] = np.searchsorted(t_bins, hv.t,)
            nlocs.append('iit')

        return hv.groupby(nlocs[::-1]).mean()


    @staticmethod
    def variogram(data, coordcol, valcol, cutoff=None, width=None, boundaries=None,
                  t_col=None, t_cutoff=None, t_width=None, t_boundaries=None,
                  alpha=0, beta=0., tol_hor=None, tol_ver=None,
                  cressie=False, dX=0., distfuc='euclidean',
                  cloud=False, trend_beta=None, cross=True,
                  covariogram=False, residual=True, timedim=None):
        """
        duplicate the variogram function in R gstat

        Parameters
        ----------

        coords
        A coordinate array or a list of coordinate array(s) for one or
        variables; when more than one coordinate array, direct and cross
        (residual) variograms are calculated for all variables and variable
        pairs defined in object; in case of variogram.formula, formula defining
        the response vector and (possible) regressors, in case of absence of
        regressors, use e.g. z~1; in case of variogram.default: list with for
        each variable the vector with responses (should not be called directly)

        vals
        A value array or a list of value arrays

        cutoff
        spatial separation distance up to which point pairs are included in
        semivariance estimates; as a default, the length of the diagonal of
        the box spanning the data is divided by three.

        width
        the width of subsequent distance intervals into which data point pairs
        are grouped for semivariance estimates

        alpha
        direction in plane (x,y), in positive degrees clockwise from positive
        y (North): alpha=0 for direction North (increasing y), alpha=90 for
        direction East (increasing x); optional a vector of directions in (x,y)

        beta
        direction in z, in positive degrees up from the (x,y) plane;

        optional a vector of directions

        tol.hor
        horizontal tolerance angle in degrees

        tol.ver
        vertical tolerance angle in degrees

        cressie
        logical; if TRUE, use Cressie”s robust variogram estimate; if FALSE use
        the classical method of moments variogram estimate

        dX
        include a pair of data points $y(s_1),y(s_2)$ taken at locations $s_1$
        and $s_2$ for sample variogram calculation only when
        $||x(s_1)-x(s_2)|| < dX$ with and $x(s_i)$ the vector with regressors
        at location $s_i$, and $||.||$ the 2-norm. This allows pooled
        estimation of within-strata variograms (use a factor variable as
        regressor, and dX=0.5), or variograms of (near-)replicates in a linear
        model (addressing point pairs having similar values for regressors
        variables)

        boundaries
        numerical vector with distance interval upper boundaries; values should
        be strictly increasing

        cloud
        logical; if TRUE, calculate the semivariogram cloud

        trend.beta
        vector with trend coefficients, in case they are known. By default,
        trend coefficients are estimated from the data.

        debug.level
        integer; set gstat internal debug level

        cross
        logical or character; if FALSE, no cross variograms are computed when
        object is of class gstat and has more than one variable; if TRUE, all
        direct and cross variograms are computed; if equal to "ST", direct and
        cross variograms are computed for all pairs involving the first
        (non-time lagged) variable; if equal to "ONLY", only cross variograms
        are computed (no direct variograms).

        formula
        formula, specifying the dependent variable and possible covariates

        covariogram
        logical; compute covariogram instead of variogram?

        PR
        logical; compute pairwise relative variogram (does NOT check whether
        variable is strictly positive)

        """
        if not (isinstance(data, list) or isinstance(data, tuple)):
            data = [data]


        if isinstance(valcol, str):
            valcol = [valcol]

        if distfuc == 'euclidean':
            distfuc = euclidean_dist
        elif distfuc == 'greatcircle':
            distfuc = great_circle_dist
        else:
            raise ValueError(f'Unknow distance calculation method: {distfuc}. `distfuc` must be `euclidean` or `greatcircle`.')

        coords = [d[coordcol].values for d in data]
        vals   = [d[v].values        for d,v in zip(data, valcol)]

        if cutoff is None:
            allcoords = np.concatenate(coords, axis=0)
            maxcoord = (np.max(allcoords, axis=0) - np.min(allcoords, axis=0)).reshape([1, -1])
            zerocoord = np.zeros(maxcoord.shape)
            cutoff = distfuc(maxcoord, zerocoord) / 3


        nval = len(data)
        hasTime = False
        times = [None]*nval
        if t_col is not None:
            hasTime = True
            if isinstance(t_col, str):
                t_col = [t_col]
            times = [d[t].values     for d,t in zip(data, t_col)]

            if t_cutoff is None:
                alltimes = np.concatenate(times)
                t_cutoff = alltimes.max() - alltimes.min()

        # TODO: diretional variogram
        if tol_hor is None:
            tol_hor = np.size(alpha)
        # TODO: diretional variogram
        if tol_ver is None:
            tol_hor = np.size(beta)
        # TODO: covariogram
        if covariogram:
            pass

        columns = 'h v'
        if hasTime:
            columns += ' t'
        columns = columns.split()
        vgms = {}
        for v1 in range(nval):
            for v2 in range(v1, nval):
                print(f'Calculating variogram {v1+1}, {v2+1} ...\n')
                if v1 == v2:
                    vgms[(v1+1, v2+1)] = vgm.raw_variogram  (coords[v1], vals[v1], cutoff, times[v1], t_cutoff, distfuc)

                elif cross:
                    vgms[(v1+1, v2+1)] = vgm.cross_variogram(coords[v1], vals[v1], coords[v2], vals[v2], cutoff, True, times[v1], times[v2], t_cutoff, distfuc)

                if (v1+1, v2+1) in vgms:
                    if cloud:
                        vgms[(v1+1, v2+1)] = pd.DataFrame(vgms[(v1+1, v2+1)], index=columns).T
                    else:
                        vgms[(v1+1, v2+1)] = vgm.calc_avg_variogram(*vgms[(v1+1, v2+1)],
                                cutoff=cutoff, width=width, bins=boundaries,
                                t_cutoff=t_cutoff, t_width=t_width, t_bins=t_boundaries, )
        if len(vgms) == 1:
            return vgms[(1, 1)]
        else:
            return vgms

    @staticmethod
    def get_vgm(v, cov=False):
        """_summary_

        Args:
            v (dict): dictionary of vriogram `type` and `parameters`
        """

        if cov:
            t = "c_"
        else:
            t = "v_"

        t += v.pop('type')
        return lambda d: getattr(vgm, t)(d, **v)

    @staticmethod
    def linear(d):
        return d

    @staticmethod # TODO
    def calc_vgm(d, psill=1., rng=1.0, nugget=0., as_vgm=False):
        # sys._getframe().f_code.co_name
        return d

    @staticmethod
    def v_linear(d, psill=1., rng=1.0, nugget=0., ):
        """Linear model"""
        return psill * d / rng + np.where(d>0, nugget, 0)


    @staticmethod
    def v_power(d, exponent=1., scale=1., rng=np.inf, nugget=0.):
        """Power model"""
        return scale * d**exponent + np.where(d>0, nugget, 0)


    @staticmethod
    def v_gaussian(d, psill=1., rng=np.inf, nugget=0.):
        """Gaussian model"""
        return psill * (1.0 - np.exp(-(d**2.0) / (rng * 4.0 / 7.0) ** 2.0)) + np.where(d>0, nugget, 0)

    @staticmethod
    def c_gaussian(d, psill=1., rng=np.inf, nugget=0.):
        """Gaussian model"""
        return psill * (np.exp(-(d**2.0) / (rng * 4.0 / 7.0) ** 2.0)) + np.where(d==0, nugget, 0)


    @staticmethod
    def v_exponential(d, psill=1., rng=np.inf, nugget=0.):
        """Exponential model"""
        return psill * (1.0 - np.exp(-d / (rng / 3.0))) + np.where(d>0, nugget, 0)


    @staticmethod
    def c_exponential(d, psill=1., rng=np.inf, nugget=0.):
        """Exponential model"""
        return psill * (np.exp(-d / (rng / 3.0))) + np.where(d==0, nugget, 0)

    @staticmethod
    def v_spherical(d, psill=1., rng=np.inf, nugget=0.):
        """Spherical model"""

        hr = d/rng
        return psill * np.where(d < rng, hr*(1.5-.5*hr*hr), 1.) + np.where(d>0, nugget, 0)

    @staticmethod
    def c_spherical(d, psill=1., rng=np.inf, nugget=0.):
        """Spherical model
        vgm(d) = cov(0) - cov(d)
        where cov(0) = nugget + sill
        """
        hr = d/rng
        return psill * np.where(d < rng, (1 - 1.5 * hr + 0.5 * hr ** 3), 0)  + np.where(d==0, nugget, 0)

    @staticmethod
    def v_hole_effect(d, psill=1., rng=np.inf, nugget=0.):
        """Hole Effect model"""
        return psill * (1.0 - (1.0 - d / (rng / 3.0)) * np.exp(-d / (rng / 3.0))) + np.where(d>0, nugget, 0)

    @staticmethod
    def c_hole_effect(d, psill=1., rng=np.inf, nugget=0.):
        """Hole Effect model"""
        return psill * ((1.0 - d / (rng / 3.0)) * np.exp(-d / (rng / 3.0))) + np.where(d==0, nugget, 0)
