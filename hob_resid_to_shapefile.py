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
from . import hob_util
from . import general_util


def plot_hob_resid(mf):
    pass

def hob_resid_to_shapefile(mf, stress_period = [0,-1], shpname = 'hob_shapefile.shp', obs_name_file = 'Obs_well_names.csv'):

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
    obs_true_names = pd.read_csv(obs_name_file)
    obs_names = hobdf['Basename'].unique()
    geoms = []
    all_rec = []
    nwis_index = 0
    for obs_ in obs_names:
        curr_hob = hobdf[hobdf['Basename'] == obs_]
        well_id = int(obs_.split("_")[1])
        well_nm = obs_true_names.loc[obs_true_names['ID'] == well_id, 'Name']
        # trim data based on sptress period
        start = stress_period[0]
        endd = stress_period[1]
        if endd < 0:
            endd =hobdf['stress_period'].max()
        curr_hob = curr_hob[(curr_hob['stress_period']>=start) & (curr_hob['stress_period']<= endd)]
        curr_hob_out = hobout_df[hobout_df['OBSERVATION NAME'].isin(curr_hob['name'].values)]
        err = curr_hob_out['OBSERVED VALUE'] - curr_hob_out['SIMULATED EQUIVALENT']
        # n, mean, mse, mae
        rec = [obs_, len(err), err.mean(), (err**2.0).mean()**0.5, (err.abs()).mean()]
        rrow = curr_hob['row'].values[0]-1
        coll = curr_hob['col'].values[0]-1
        xy = get_vertices(rrow, coll)

        coff = curr_hob['coff'].values[0]
        roff = curr_hob['roff'].values[0]

        xx0 = (xy[0][0] + xy[1][0]) * 0.5
        yy0 = (xy[0][1] + xy[2][1]) * 0.5
        xSize = abs(xy[1][0] - xy[0][0])
        ySize = abs(xy[1][1] - xy[2][1])
        x_ = xx0 +  coff * xSize
        y_ = yy0 - roff * ySize
        geoms.append(Point(x_, y_, 0))

        try:
            if well_nm.values[0][0] == '3':
                nwis_index = nwis_index + 1
                new_nm = 'NW-{}'.format(nwis_index)
            else:
                new_nm = well_nm.values[0]

            rec.append(well_nm.values[0])
            rec.append(new_nm)
        except:
            nwis_index = nwis_index + 1
            new_nm = 'NW-{}'.format(nwis_index)
            rec.append('')
            rec.append(new_nm)
        all_rec.append(rec)
    all_rec = pd.DataFrame(all_rec, columns= ['obsnme', 'nobs', 'merr', 'mse', 'mae', 'wellname', 'shortNam'])
    all_rec = all_rec.to_records()
    epsg = mf.modelgrid.epsg
    recarray2shp(all_rec, geoms, shpname, epsg=epsg)
    xxx = 1


