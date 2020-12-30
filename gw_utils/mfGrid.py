import os, sys
import numpy as np
import flopy


def get_3d_grid(mf):
    """
    Parameters
    ----------
    mf: flopy object with dis
    Returns
    -------
    """

    nlay = mf.nlay
    nrow = mf.nrow
    ncol = mf.ncol

    grid3d = np.zeros(shape=(nlay+1, nrow, ncol))
    botm = mf.dis.botm.array.copy()
    grid3d[0,:,:] = mf.dis.top.array.copy()

    for k in range(nlay):
        grid3d[k+1, :, :] = botm[k,:,:]

    return grid3d

def get_top_active_layer(mf):
    """

    Parameters
    ----------
    mf: flopy object that has only dis and bas6

    Returns: 2d array with the index of the top active 0 indexed
    -------

    """   
    ibound3d = mf.bas6.ibound.array.copy()
    top_active_layer = np.zeros(shape=(mf.nrow, mf.ncol))/0.0
    for k in range(mf.nlay):
        mask = np.logical_and(np.isnan(top_active_layer), ibound3d[k,:,:] != 0)
        top_active_layer[mask] = k

    return top_active_layer
