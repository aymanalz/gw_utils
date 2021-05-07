import os, sys
import numpy as np
import pandas as pd
import flopy
from . import  mfGrid

def well_to_df(mf):
    pass

def get_ts_well(mf, zone_arr):
    """

    Parameters
    ----------
    mf: flopy object
    zone_arr:  array with flags that represent the zone ids

    Returns
    -------
    well_ts: pandas dataframe, columns=['zone_id', 'sp', 'sum_rate', 'mean_rate']

    """

    if not('WEL' in mf.get_package_list()):
        raise ValueError("WELL package is not loaded...")

    zone_ids = np.unique(zone_arr)
    zone_rr_col = {}
    for zone_id in zone_ids:
        rr, cc = np.where(zone_arr == zone_id )
        zone_rr_col[zone_id] = list(zip(rr, cc))

    stress_periods = np.sort(list(mf.wel.stress_period_data.data.keys()))
    well_ts = []
    for sp in stress_periods:
        curr_ = mf.wel.stress_period_data.data[sp]
        curr_ = pd.DataFrame(curr_)
        rr = curr_['i'].values
        cc = curr_['j'].values
        curr_['rr_cc'] = list(zip(rr, cc))
        for zone_id in zone_ids:
            curr_zone_rc = zone_rr_col[zone_id]
            mask = curr_['rr_cc'].isin(curr_zone_rc)
            val_sum = curr_.loc[mask, 'flux'].sum()
            val_mean = curr_.loc[mask, 'flux'].mean()
            well_ts.append([zone_id, sp, val_sum, val_mean])

    well_ts = pd.DataFrame(well_ts, columns=['zone_id', 'sp', 'sum_rate', 'mean_rate'])
    return well_ts



def wel2arr(mf, sp ):
    """

    :param curr_wel: numpy rec array from flopy well package
    :param nrow: number of models rows
    :param ncol: number of model cols
    :return:
    """
    curr_wel = mf.wel.stress_period_data[sp]
    nrow = mf.nrow
    ncol = mf.ncol
    nlay = mf.nlay

    curr_wel = pd.DataFrame(curr_wel)
    curr_wel['ll_rr_cc'] = list(zip(curr_wel['k'], curr_wel['i'], curr_wel['j']))
    curr_wel = curr_wel.groupby(by='ll_rr_cc').sum() # sometimes more than well exists in once cell
    ll_rr_cc = [[kij[0], kij[1], kij[2]] for kij in curr_wel.index.values]
    ll_rr_cc = np.array(ll_rr_cc)
    arr = np.zeros(shape=(nlay, nrow, ncol))
    arr[ll_rr_cc[:,0], ll_rr_cc[:,1], ll_rr_cc[:,2]] = curr_wel['flux']
    arr = arr.sum(axis = 0)
    return arr

def arr2wel(mf, arr):

    well_mask = arr != 0
    rrwel, ccwel = np.where(well_mask)
    columns = ['k', 'i', 'j', 'flux']
    top_layer = mfGrid.get_top_active_layer(mf)
    lays = top_layer[rrwel, ccwel]
    flux = arr[rrwel, ccwel]

    df = pd.DataFrame(columns=columns)
    df['k'] = lays
    df['i'] = rrwel
    df['j'] = ccwel
    df['flux'] = flux

    df = df.dropna(axis=0)
    df = df[df['flux'] != 0]
    df['i'] = df['i'].astype( dtype="Int32")
    df['j'] = df['j'].astype(dtype="Int32")
    df['k'] = df['k'].astype(dtype="Int32")
    df['flux'] = df['flux'].astype(dtype="float32")

    df = df.to_records(index=False)
    return df

def well_mean_array(mf):
    pass
