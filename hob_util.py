import os, sys
import numpy as np
import pandas as pd
import flopy
import collections
from general_util import *

def get_hob_csv_flopy(mfname):

    mf = flopy.modflow.Modflow.load(mfname, load_only=['DIS', 'BAS6', 'HOB'])
    xx = 1

def in_hob_to_df(mfname = None):
    """
    Load an HOB file and convert inf to csv file.

    Parameters
    ----------
    f : filename or file handle
        File to load.
    model : model object
        The model object (of type :class:`flopy.modflow.mf.Modflow`) to
        which this package will be added.
    ext_unit_dict : dictionary, optional
        If the arrays in the file are specified using EXTERNAL,
        or older style array control records, then `f` should be a file
        handle.  In this case ext_unit_dict is required, which can be
        constructed using the function
        :class:`flopy.utils.mfreadnam.parsenamefile`.
    check : boolean
        Check package data for common errors. (default True)

    Returns
    -------
    hob : ModflowHob package object
        ModflowHob package object.

    Examples
    --------



    """

    model = flopy.modflow.Modflow.load(mfname, load_only=['DIS', 'BAS6'])



    files = get_mf_files(mfname)
    f = files['HOB'][1]
    if not hasattr(f, 'read'):
        filename = f
        f = open(filename, 'r')
    # dataset 0 -- header
    while True:
        line = f.readline()
        if line[0] != '#':
            break

    # read dataset 1
    t = line.strip().split()
    nh = int(t[0])
    iuhobsv = int(t[3])
    hobdry = float(t[4])

    # read dataset 2
    line = f.readline()
    t = line.strip().split()
    tomulth = float(t[0])

    perlen = model.dis.perlen.array

    # read observation data
    obs_data = []
    all_records = []

    # read datasets 3-6
    nobs = 0
    while True:
        # read dataset 3
        line = f.readline()
        t = line.strip().split()
        obsnam = t[0]
        print(obsnam)
        layer = int(t[1])
        row = int(t[2]) - 1
        col = int(t[3]) - 1
        irefsp0 = int(t[4])
        toffset = float(t[5])
        roff = float(t[6])
        coff = float(t[7])
        hob = float(t[8])

        # read dataset 4 if multilayer obs
        if layer > 0:
            layer -= 1
            mlay = {layer: 1.}
        else:
            line = f.readline()
            t = line.strip().split()
            mlay = collections.OrderedDict()
            for j in range(0, abs(layer) * 2, 2):
                k = int(t[j]) - 1
                # catch case where the same layer is specified more than
                # once. In this case add previous value to the current value
                keys = list(mlay.keys())
                v = 0.
                if k in keys:
                    v = mlay[k]
                mlay[k] = float(t[j + 1]) + v
            # reset layer
            layer = -len(list(mlay.keys()))

        # read datasets 5 & 6. Index loop variable
        if irefsp0 > 0:
            itt = 1
            irefsp0 -= 1
            totim = sum(perlen[0:irefsp]) + toffset * tomulth
            names = [obsnam]
            tsd = [totim, hob]
            nobs += 1
            all_records.append([names[0], layer + 1, row + 1, col + 1, roff, coff, 1, irefsp0 + 1, totim, hob, mlay])
        else:
            names = []
            tsd = []
            # read data set 5
            line = f.readline()
            t = line.strip().split()
            itt = int(t[0])
            # dataset 6
            for j in range(abs(irefsp0)):
                line = f.readline()
                t = line.strip().split()
                names.append(t[0])
                name = t[0]
                irefsp = int(t[1]) - 1
                toffset = float(t[2])
                totim = sum(perlen[0:irefsp]) +  toffset * tomulth
                hob = float(t[3])
                tsd.append([totim, hob])
                nobs += 1

                all_records.append([name, layer+1, row+1, col+1,  roff, coff, j+1,  irefsp+1, totim, hob, mlay])


        if nobs == nh:
            break

    # close the file
    label1 = ['name', 'layer', 'row', 'col', 'roff', 'coff', 'tim_id', 'stress_period', 'totim', 'head', 'mlay']
    df = pd.DataFrame(all_records, columns=label1)
    return df

