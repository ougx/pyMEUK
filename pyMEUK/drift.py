# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 19:39:57 2022

@author: michaelou
"""

import numpy as np
from .common import cdist, pdist
from .constant import VERYSMALL



#%%
class drift:

    @staticmethod
    def funcDrift(coords, function):
        # check if any x,y,z is in the function text
        if 'x' in function or 'y' in function or 'z' in function:
            return eval(function.replace('x', 'coords[:, 0]').replace('y', 'coords[:, 1]').replace('z', 'coords[:, 2]'))

    @staticmethod
    def wellDrift(coords, feat_coords, discharge=1., ):
        """_summary_

        Args:
            coords (_type_): 2d array (each row represent one location) of locations to evaulate the drift
            feat_coords (_type_): 2d array of locations of the well
            discharge (float or 1d array, optional): represents the well strength. Defaults to 1.

        Returns:
            1d array of well drift evaluated at the coords
        """
        coords = np.array(coords)
        feat_coords = np.array(feat_coords)
        nfeature = len(feat_coords)
        if type(discharge) is int or type(discharge) is float:
            discharge = np.repeat(discharge, nfeature)

        discharge = np.array(discharge).reshape([-1, 1])

        r = cdist(feat_coords, coords) # two dimensional array of the distance to the wells; row by well, column by target locations
        valid = r>VERYSMALL
        r = np.where(valid, r, VERYSMALL)
        res = np.where(valid, discharge * np.log(r), 0)
        return np.sum(res, axis=0)

    @staticmethod
    def circPond(coords, feat_coords, radius=1., discharge=1.):
        """_summary_

        Args:
            coords (_type_): 2d array (each row represent one location) of locations to evaulate the drift
            feat_coords (_type_): 2d array of locations of the pond centers
            radius   (float or 1d array, optional): represents the pond radius. Defaults to 1.
            discharge (float or 1d array, optional): represents the pond strength. Defaults to 1.

        Returns:
            1d array of well drift evaluated at the coords
        """
        coords = np.array(coords)
        feat_coords = np.array(feat_coords)
        nfeature = len(feat_coords)
        if type(discharge) is int or type(discharge) is float:
            discharge = np.repeat(discharge, nfeature)
        if type(radius) is int or type(radius) is float:
            radius = np.repeat(radius, nfeature)

        discharge = np.array(discharge).reshape([-1, 1])
        radius    = np.array(radius).reshape([-1, 1])

        r = cdist(feat_coords, coords) # two dimensional array of the distance to the ponds; row by pond, column by target locations
        res = np.zeros_like(r)
        inside  = np.nonzero(r < radius)
        outside = np.nonzero(r >=radius)

        res[inside[0],  inside[1]]  = -(radius[inside[0], 0 ]**2 + r[inside[0], inside[1]]**2) / 4.
        res[outside[0], outside[1]] = - radius[outside[0],0]**2 * np.log(r[outside[0], outside[1]]**2/radius[outside[0],0]**2) / 4.
        return np.sum(res * discharge, axis=0)

    @staticmethod
    def irrPondDrift(coords):
        pass

    @staticmethod
    def polyDrift(coords, maxorder=1):
        """


        Parameters
        ----------
        coords : TYPE
            DESCRIPTION.
        maxorder : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            two dimensional array; row by locations, column by different orders .

        """
        coords = np.array(coords)
        ndim = len(coords[0])
        orders = np.zeros([(maxorder+1)**ndim, 3], dtype=int)
        orders = np.array([np.tile(np.repeat(np.arange(maxorder+1), (maxorder+1)**(i)), (maxorder+1)**(ndim-i-1))
                            for i in range(ndim)]).T
        orders = orders[orders.sum(axis=1)<=maxorder]
        return np.array([(coords**order).prod(axis=1) for order in orders])

    @staticmethod
    def linearDrift(coords):
        return drift.polyDrift(coords, 1)


    @staticmethod
    def lineDrift1(coords, line_vertex, discharge=1):
        """
        Thie function estimated the line drift based on the distance to the line. It can be applied to 3d

        Parameters
        ----------
        coords : TYPE
            DESCRIPTION.
        maxorder : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            one dimensional array; line drift evaluated at coords

        """
        coords = np.array(coords)
        line_vertex = np.array(line_vertex)

        ll = 0
        vert0 = line_vertex[0]
        ndim = len(vert0)
        origin = np.zeros_like(vert0)
        res = []
        for vert1 in line_vertex[1:]:
            vertc = (vert0+vert1)/2
            l = pdist([vert0, vert1])[0]
            ll += l
            n = (vert1 - vert0) / l         # a unit vector in the direction of the line
            r1 = cdist((vert0 - coords) - np.repeat((vert0 - coords).dot(n),2).reshape([-1, ndim]) * n, [origin])[:,0]
            r2 = (coords - vertc).dot(n)
            r2 = np.where(np.abs(r2) > l/2, np.abs(r2) - l/2, 0.)
            r = np.sqrt(r1 ** 2 + r2 ** 2)
            zz = np.where(r>0, np.log(r), VERYSMALL)
            # zz = np.where(np.abs(zz-1)>VERYSMALL, zz, zz+VERYSMALL)
            # carg = (zz + 1) * np.log(zz + 1) - (zz - 1) * np.log(np.abs(zz - 1)) + 2.*np.log(l/2.)-2.

            # res.append(l / 4. / np.pi * carg)
            # res.append(carg)
            res.append(zz)
            vert0 = vert1
        return np.sum(res, axis=0) * discharge


    @staticmethod
    def lineDrift(coords, line_vertex, discharge=1):
        """
        Thie function estimated the line drift. It can be applied to 2d

        Parameters
        ----------
        coords : TYPE
            DESCRIPTION.
        maxorder : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            one dimensional array; line drift evaluated at coords

        """
        coords = np.array(coords)
        line_vertex = np.array(line_vertex)
        assert line_vertex.shape[-1] == 2 and coords.shape[-1] == 2, "lineDrift only works for two dimensional data"
        assert np.unique(line_vertex, axis=0).shape[0] == line_vertex.shape[0], "found dupplicated vertex"
        z1 = complex(*line_vertex[0])
        z  = coords[:,0] + 1j*coords[:,1]
        res = []
        for vert1 in line_vertex[1:]:
            z2 = complex(*vert1)
            l = np.abs(z2 - z1)
            zz = (2.*z-(z1+z2))/(z2-z1)
            zz = np.where(np.abs(zz-1)<VERYSMALL, zz+VERYSMALL, zz)
            zz = np.where(np.abs(zz+1)<VERYSMALL, zz+VERYSMALL, zz)
            carg  = (zz+1.)*np.log(zz+1.)-(zz-1.)*np.log(zz-1.) + 2.*np.log((z2-z1)/2.)-2.
            res.append(l * np.real(carg))
            z1 = z2
        return np.sum(res, axis=0) * discharge / 4. / np.pi


    @staticmethod
    def barrierDrift(coords, line_vertex, discharge=1, drop=5):
        """
        Thie function estimated the barrier drift. It can only be applied to 2d

        Parameters
        ----------
        coords : TYPE
            DESCRIPTION.
        line_vertex : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            one dimensional array; barrier drift evaluated at coords

        """
        assert line_vertex.shape[-1] == 2 and coords.shape[-1] == 2, "barrierDrift only works for two dimensional data"
        res1 = drift.lineDrift2d(coords, line_vertex, discharge)
        res2 = drift.lineDrift2d(coords, line_vertex+np.array([[drop, drop]]), -discharge)
        return res1 + res2

    @staticmethod
    def externalDrift(coords):
        pass

#%%

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.spatial.distance import cdist, pdist



    #%% well
    xx, yy = np.meshgrid(range(101), range(101))
    coords = (np.array([xx.flatten(), yy.flatten()]).T)

    well_coords = np.array([[25, 50], [75, 50]])

    flow = drift.wellDrift(coords, well_coords, discharge=[-3, -3], )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.imshow(flow.reshape([101, 101]), cmap='terrain', )





    #%% circPond

    xx, yy = np.meshgrid(range(101), range(101))
    coords = (np.array([xx.flatten(), yy.flatten()]).T)

    center_coords = np.array([[25, 50], [75, 50]])

    flow = drift.circPond(coords, well_coords, radius=[3,2], discharge=[-3, -3], )

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.imshow(flow.reshape([101, 101]), cmap='terrain', )




    #%% barrier
    VERYSMALL = 1e-10



    xx, yy = np.meshgrid(range(41), range(64))
    coords = (np.array([xx.flatten(), yy.flatten()]).T) * 1000 + np.array([[6461468, 1713666]])
    line_vertex = np.array([[6549802,1739166], [6479968,1739166], [6455473,1755950], [6454596,1805243]])
    res1 = drift.lineDrift1(coords, line_vertex, 1)

    line_vertex = line_vertex+np.array([[1, 1]])
    res2 = drift.lineDrift1(coords, line_vertex, -1)
    res = res1 + res2


    fig, axs = plt.subplots(1, 3, figsize=(15, 8))

    ax = axs[0]
    ax.imshow(res1.reshape([64, 41]), cmap='terrain', extent=(6461468-500, 6502468+500, 1777666+500, 1713666-500))
    ax.contour(xx * 1000 + 6461468, yy * 1000 + 1713666, res1.reshape([64, 41]))
    ax.plot(line_vertex[:,0], line_vertex[:,1], color='r')
    ax.set(xlim=(6461468, 6502468), ylim=(1713666, 1777666))

    ax = axs[1]
    ax.imshow(res2.reshape([64, 41]), cmap='terrain', extent=(6461468-500, 6502468+500, 1777666+500, 1713666-500))
    ax.contour(xx * 1000 + 6461468, yy * 1000 + 1713666, res2.reshape([64, 41]))
    ax.plot(line_vertex[:,0], line_vertex[:,1], color='r')
    ax.set(xlim=(6461468, 6502468), ylim=(1713666, 1777666))

    ax = axs[2]
    ax.imshow(res.reshape([64, 41]), cmap='terrain', extent=(6461468-500, 6502468+500, 1777666+500, 1713666-500), vmin=-2, vmax=2)
    ax.contour(xx * 1000 + 6461468, yy * 1000 + 1713666, res.reshape([64, 41]))
    ax.plot(line_vertex[:,0], line_vertex[:,1], color='r')
    ax.set(xlim=(6461468, 6502468), ylim=(1713666, 1777666))

    #%% barrier

    xx, yy = np.meshgrid(range(41), range(64))
    coords = (np.array([xx.flatten(), yy.flatten()]).T) * 1000 + np.array([[6461468, 1713666]])
    line_vertex = np.array([[6549802,1739166], [6479968,1739166], [6455473,1755950], [6454596,1805243]])
    res1 = drift.lineDrift2d(coords, line_vertex, 1)

    line_vertex = line_vertex+np.array([[1, 1]])
    res2 = drift.lineDrift2d(coords, line_vertex, -1)
    res = res1 + res2


    fig, axs = plt.subplots(1, 3, figsize=(15, 8))

    ax = axs[0]
    ax.imshow(res1.reshape([64, 41]), cmap='terrain', extent=(6461468-500, 6502468+500, 1777666+500, 1713666-500))
    ax.contour(xx * 1000 + 6461468, yy * 1000 + 1713666, res1.reshape([64, 41]))
    ax.plot(line_vertex[:,0], line_vertex[:,1], color='r')
    ax.set(xlim=(6461468, 6502468), ylim=(1713666, 1777666))

    ax = axs[1]
    ax.imshow(res2.reshape([64, 41]), cmap='terrain', extent=(6461468-500, 6502468+500, 1777666+500, 1713666-500))
    ax.contour(xx * 1000 + 6461468, yy * 1000 + 1713666, res2.reshape([64, 41]))
    ax.plot(line_vertex[:,0], line_vertex[:,1], color='r')
    ax.set(xlim=(6461468, 6502468), ylim=(1713666, 1777666))

    ax = axs[2]
    ax.imshow(res.reshape([64, 41]), cmap='terrain', extent=(6461468-500, 6502468+500, 1777666+500, 1713666-500), vmin=-2, vmax=2)
    ax.contour(xx * 1000 + 6461468, yy * 1000 + 1713666, res.reshape([64, 41]))
    ax.plot(line_vertex[:,0], line_vertex[:,1], color='r')
    ax.set(xlim=(6461468, 6502468), ylim=(1713666, 1777666))


    #%% barrier
    xx, yy = np.meshgrid(range(101), range(101))
    coords = (np.array([xx.flatten(), yy.flatten()]).T)
    line_vertex = np.array([[-100, 50], [200, 50]])

    flow = drift.lineDrift2d(coords, line_vertex, discharge=100)
    baa =  drift.barrierDrift(coords, line_vertex, discharge=100, drop=0.001)

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    axs[0].imshow(flow.reshape([101, 101]), cmap='terrain', )
    axs[1].imshow(baa.reshape([101, 101]), cmap='terrain', )


    #%% lineDrift + linearDrift
    xx, yy = np.meshgrid(range(101), range(101))
    coords = (np.array([xx.flatten(), yy.flatten()]).T)
    line_vertex = np.array([[-100, 50], [200, 50]])
    flow = drift.lineDrift(coords, line_vertex, -1)
    flow = flow - flow.min()

    flow = flow + np.tile(range(101), 101)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(flow.reshape([101, 101]), cmap='terrain', )
