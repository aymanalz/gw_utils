import os
import sys
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import flopy
import hob_util
import general_util


def plot_hob_resid(mf):
    pass

def hob_resid_to_shapefile(mfname, epsg = None):
    # get all files
    mf_files = general_util.get_mf_files(mfname)

    # read mf and get spatial reference
    mf, hobdf = hob_util.in_hob_to_df(mfname=None, return_model = True)

    # read_hob_out
    hobout_df = None
    for file in mf_files.keys():
        fn = mf_files[file][1]
        basename = os.path.basename(fn)
        if ".hob.out" in basename:
            hobout_df = pd.read_csv(fn, delim_whitespace=True)

    # compute the coordinates of each point and generate the geom

    # compute error statistics
    pass


