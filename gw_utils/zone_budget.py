import os, sys
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

def generate_zone_file(fname = r'gw_zones.zon' , arr3d = None):
    """ Generate a zone file uzing numpy array"""
    nspaces = 5
    sp = " "*nspaces
    if arr3d is None:
        raise ValueError("A 3D array for zones is required to run this util")

    nlay, nrow, ncol = arr3d.shape()
    fidw = open(fname, 'w')
    fidw.write("{} {} {}\n".format(nlay, nrow, ncol))

    for layer in range(nlay):
        arr_2d = arr3d[layer]
        fidw.write("INTERNAL      ({}I{})\n".format(ncol, nspaces))
        for row in range(nrow):
            curr_row = arr_2d[row,:].tolist()
            curr_row = sp.join(curr_row[0,:].astype(int).astype(str))
            curr_row = curr_row + "\n"
            fidw.write(curr_row)

    fidw.close()

def run_zone_bud(fname = r'gw_zones.zon' , arr3d = None):
    pass

def read_csv_budget(bud_csv_file):
    df = pd.read_csv(bud_csv_file)
    return df
