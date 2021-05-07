import os, sys
import shutil
import pandas as pd
import numpy as np
import datetime
from .general_util import get_mf_files
import matplotlib.pyplot as plt

def generate_zone_file(fname = r'gw_zones.zon' , arr3d = None):
    """ Generate a zone file uzing numpy array"""
    nspaces = 5
    sp = " "*nspaces
    if arr3d is None:
        raise ValueError("A 3D array for zones is required to run this util")

    nlay, nrow, ncol = arr3d.shape
    fidw = open(fname, 'w')
    fidw.write("{} {} {}\n".format(nlay, nrow, ncol))

    for layer in range(nlay):
        arr_2d = arr3d[layer]
        fidw.write("INTERNAL      ({}I{})\n".format(ncol, nspaces))
        for row in range(nrow):
            curr_row = arr_2d[row,:]
            curr_row = sp.join(curr_row.astype(int).astype(str))
            curr_row = curr_row + "\n"
            fidw.write(curr_row)

    fidw.close()

def run_zone_bud(mf_name, zone_arr, work_space):

    # check if zonbud.exe in work_space
    if not os.path.isfile(os.path.join(work_space, "zonbud.exe")):
        raise ValueError("The executable zonbud.exe must exist in {}".format(work_space))


    zone_analysis_files = {}

    files = get_mf_files(mf_name)
    cbc_file = files['cbc'][1]
    basename = os.path.splitext(os.path.basename(files['cbc'][1]))[0]
    date = datetime.datetime.now().strftime("D_%b_%d_%Y_T_%H_%M_%S")

    cmd_fn = os.path.join(work_space, "{}_cmd_{}.txt".format(basename, date))
    out_csv = os.path.join(work_space, "{}_bud_{}.csv".format(basename, date))
    zone_arr_file =  os.path.join(work_space, "{}_zone_{}.zone".format(basename, date))
    bat_file = os.path.join(work_space, "{}_zonbud_{}.bat".format(basename, date))

    # this files to be transfered from the work folder to model output folder
    # after runing the results will copied to the work directory
    zone_analysis_files['exe'] = os.path.join(work_space, "zonbud.exe")
    zone_analysis_files['cmd_list_file'] = cmd_fn
    #zone_analysis_files['out_csv_file'] = out_csv
    zone_analysis_files['zone_file'] = zone_arr_file
    zone_analysis_files['bat_file'] = bat_file

    # Write cmd list file
    fidw = open(cmd_fn, 'w')
    fidw.write("{} csv2 \n".format(os.path.basename(out_csv)))
    fidw.write("{}\n".format(os.path.basename(cbc_file)))
    fidw.write("{}\n".format(basename))
    fidw.write("{}\n".format(os.path.basename(zone_arr_file)))
    fidw.write("A\n")
    fidw.close()

    #write zone file
    generate_zone_file(fname=zone_arr_file, arr3d=zone_arr)

    # write batch file
    fidw=open(bat_file, 'w')
    cmd = "zonbud.exe < {}".format(os.path.basename(cmd_fn))
    fidw.write(cmd)
    fidw.close()

    # transfer files to model directory
    model_ws = os.path.abspath(os.path.dirname(mf_name))
    for file in zone_analysis_files.keys():
        shutil.copy(src=zone_analysis_files[file],
                    dst = os.path.join(model_ws, os.path.basename(zone_analysis_files[file])))

    # run
    current_folder = os.getcwd()
    os.chdir(model_ws)
    os.system(bat_file)
    os.chdir(current_folder)

    # delete files transfered and clean
    for file in zone_analysis_files.keys():
        os.remove(os.path.join(model_ws, os.path.basename(zone_analysis_files[file])))

    # move csv file to workspace
    src = os.path.join(model_ws, os.path.basename(out_csv)+'.2.csv')
    os.remove(os.path.join(model_ws, os.path.basename(out_csv)+'.log'))
    dst = os.path.join(work_space,os.path.basename(out_csv) )
    shutil.move(src = src, dst = dst)

    df = pd.read_csv(dst)
    return df

def read_csv_budget(bud_csv_file):
    df = pd.read_csv(bud_csv_file)
    return df
