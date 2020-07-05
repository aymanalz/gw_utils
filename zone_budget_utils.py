import os, sys
import pandas as pd
import flopy
import numpy as np

def generate_zone_file(zone_csv_file, mffn, zone_output_file):
    """

    Parameters
    ----------
    zone_csv_file: this is the attribute table of a shapefile that has rows, cols, zone_name

    Returns
    -------

    """
    mf = flopy.modflow.Modflow.load(mffn, load_only=['DIS', 'BAS6'])

    df = pd.read_csv(zone_csv_file)
    df = df.sort_values(by=['HRU_ID'])

    # get zone_names
    zone_names = df['subbasin'].unique()
    df['ZoneID'] = 0

    for izm, zm in enumerate(zone_names):
        df.loc[df['subbasin']==zm, 'ZoneID'] = izm

    zone_2d = np.zeros(shape=(mf.nrow, mf.ncol))
    #df.drop_duplicates(subset="HRU_ID", keep='last', inplace=True)
    fidw0 = open('zone_names_id.txt', 'w')
    for izm, zm in enumerate(zone_names):
        rows = df.loc[df['subbasin'] == zm, 'HRU_ROW'].values-1
        cols = df.loc[df['subbasin'] == zm, 'HRU_COL'].values - 1
        zone_2d[rows, cols] = izm
        fidw0.write("{} * {}\n".format(zm, izm))
    fidw0.close()


    fidw = open(zone_output_file, 'w')

    # first line
    line = "{}\t{}\t{}\n".format(int(mf.nlay), int(mf.nrow), int(mf.ncol))
    fidw.write(line)
    for k in range(mf.nlay):
        line = "INTERNAL      ({}I5)\n".format(int(mf.ncol))
        fidw.write(line)
        for i in range(mf.nrow):
            for j in range(mf.ncol):
                line = "    {}".format(int(zone_2d[i, j]))
                fidw.write(line)
            fidw.write("\n")

    fidw.close()


    xx = 1

if __name__ == "__main__":
    shp = r"D:\Yucaipa_work\new_gw_zones_corrected.csv"
    mffn = r"C:\work\Slave\use_average_HFB6\m13\yucaipa.nam"
    generate_zone_file(zone_csv_file=shp, mffn= mffn, zone_output_file = 'zones_bud.zon')
