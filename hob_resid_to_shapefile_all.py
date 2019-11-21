import os
import sys
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import flopy
from flopy.utils.geometry import Polygon, LineString, Point
from flopy.export.shapefile_utils import recarray2shp, shp2recarray
import hob_util
import general_util


def plot_hob_resid(mf):
    pass

def hob_resid_to_shapefile_all(mf, stress_period = [0,-1], shpname = 'hob_shapefile.shp'):

    get_vertices = mf.modelgrid.get_cell_vertices

    # get all files
    mfname = os.path.join(mf.model_ws, mf.namefile)
    mf_files = general_util.get_mf_files(mfname)

    # read mf and get spatial reference
    hobdf = hob_util.in_hob_to_df(mfname=mfname, return_model = False)

    # read_hob_out
    hobout_df = None
    for file in mf_files.keys():
        fn = mf_files[file][1]
        basename = os.path.basename(fn)
        if ".hob.out" in basename:
            hobout_df = pd.read_csv(fn, delim_whitespace=True)




    # loop over obs and compute residual error

    obs_names = hobdf['Basename'].unique()
    geoms = []
    all_rec = []
    for obs_ in obs_names:
        curr_hob = hobdf[hobdf['Basename'] == obs_]

        # trim data based on sptress period
        start = stress_period[0]
        endd = stress_period[1]
        if endd < 0:
            endd =hobdf['stress_period'].max()
        curr_hob = curr_hob[(curr_hob['stress_period']>=start) & (curr_hob['stress_period']<= endd)]
        curr_hob_out = hobout_df[hobout_df['OBSERVATION NAME'].isin(curr_hob['name'].values)]
        err = curr_hob_out['OBSERVED VALUE'] - curr_hob_out['SIMULATED EQUIVALENT']
        curr_hob['OBSERVED VALUE'] = curr_hob_out['OBSERVED VALUE'].values
        curr_hob['SIMULATED EQUIVALENT'] = curr_hob_out['SIMULATED EQUIVALENT'].values
        curr_hob['err'] = err
        # n, mean, mse, mae
        #rec = [obs_, len(err), err.mean(), (err**2.0).mean()**0.5, (err.abs()).mean()]
        rrow = curr_hob['row'].values[0]-1
        coll = curr_hob['col'].values[0]-1
        xy = get_vertices(rrow, coll)
        for ierr in err:
            geoms.append(Point(xy[0][0], xy[0][1], 0))
        all_rec.append(curr_hob.copy())
    all_rec = pd.concat(all_rec)
    # here we generate csv file
    all_rec = all_rec.to_records()
    epsg = mf.modelgrid.epsg
    recarray2shp(all_rec, geoms, shpname, epsg=epsg)
    xxx = 1


