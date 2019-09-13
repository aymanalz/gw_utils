import os, sys
import pandas as pd
import numpy as np
import geopandas
from general_util import *
import flopy
from epsg_ident import EpsgIdent
import shapefile
from gis_utils import *

def set_spatial_ref_from_hru_shpfile(mf, hrushp):
    """

    Parameters
    ----------
    mf : flopy object
    hrushp: hru_shapefile

    Returns
    -------

    """


    epsg = get_epsg(hrushp)

    sf = shapefile.Reader(hrushp)
    xoff = sf.bbox[0]
    yoff = sf.bbox[1]
    grid = mf.modelgrid
    grid.set_coord_info(xoff=xoff, yoff=yoff, epsg = epsg)
    return grid


def get_prj_info_from_hru_shpfile(hrushp):
    pass

def hds_to_shapefile(mfname, stress_periods = [0], gridshape = None,
                     row_col_fields = ['HRU_ROW', 'HRU_COL'], outshpfile = 'heads.shp'):
    """

    Parameters
    ----------
    mfname: MODFLOW name file
    stress_periods: a list of stress periods to add to the shapefile
    row_col_fields: a list of two strings represent the row and column fields


    Returns
    -------
    """

    pass

def hds_to_raster(hds):

    pass



