import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, r'D:\Workspace\Codes\flopy-develop\flopy-develop\flopy-develop')
import flopy
import pandas as pd

global stress_start_end

def main_ss_to_tarn(mf_tran, ss_folder, ss_name, stress_start_end, obs_dict):

    mf_ss = flopy.modflow.Modflow(ss_name, model_ws=ss_folder, version=mf_tran.version)
    packages = mf_tran.get_package_list()

    for pk in packages:
        if pk == 'DIS':
            dis_ = dis_ss(mf_tran, mf_ss)


        elif pk == 'BAS6':
            bas6_ss(mf_tran, mf_ss)

        elif pk == 'SFR':
            sfr_ss(mf_tran, mf_ss)
        elif pk == 'MNW2':
            mnw2_ss(mf_tran, mf_ss, stress_start_end)

        elif pk == 'WEL':
            wel_ss(mf_tran, mf_ss, stress_start_end)

            pass

        elif pk == 'OC':
            oc_ss(mf_tran, mf_ss)

        elif pk == 'UZF':
            uzf_ss(mf_tran, mf_ss)

            pass


        elif pk == 'CHD':
            chd_ss(mf_tran, mf_ss)

        else:
            print (pk + "  Package is not changes ......")

            mpk = np.copy(mf_tran.get_package(pk))
            mpk = mpk.all()
            mpk.file_name =[ss_name + '.' + mpk.extension[0]]
            mpk.parent = mf_ss
            mpk.fn_path = os.path.join(mf_ss.model_ws, mpk.file_name[0])
            mf_ss.add_package(mpk)

    hob_ss(mf_tran, mf_ss, obs_dict, stress_start_end)
    return mf_ss

def sfr_ss(mf, mf_ss):
    sfr = mf.sfr
    dataset_5 = {0: [170, 0, 0],
                 1: [-170, 0, 0]}


    sfr = flopy.modflow.ModflowSfr2(mf_ss, nstrm=sfr.nstrm, nss=sfr.nss, const=sfr.const, nsfrpar=sfr.nsfrpar, nparseg=sfr.nparseg,
                                    dleak=sfr.dleak, ipakcb=sfr.ipakcb, nstrail=sfr.nstrail, isuzn=sfr.isuzn, nsfrsets=sfr.nsfrsets,
                                    istcb2=sfr.istcb2, reachinput=True, isfropt=sfr.isfropt, irtflg= sfr.irtflg,
                                    reach_data=sfr.reach_data, numtim=sfr.numtim, weight=sfr.weight,
                                    segment_data=sfr.segment_data,
                                    channel_geometry_data=sfr.channel_geometry_data,
                                    channel_flow_data=sfr.channel_flow_data,
                                    dataset_5=dataset_5)
    pass

def dis_ss(mf, mf_ss) : # time should consist of only two stress period 1 : ss and 2: tr
    dis = mf.dis

    dis = flopy.modflow.ModflowDis(mf_ss, nlay=dis.nlay, nrow=dis.nrow, ncol=dis.ncol,
                                   delr=dis.delr, delc=dis.delc,
                                   top=dis.top, botm=dis.botm,
                                   nper= 1, perlen=[1], nstp=[1], steady=[True],
                                   itmuni=4, lenuni=1)  # (4) days,
    return dis



def bas6_ss(mf_tran, mf_ss):
    bas = flopy.modflow.ModflowBas(mf_ss, ibound=mf_tran.bas6.ibound.array, ichflg=True, strt=mf_tran.bas6.strt.array)
def mnw2_ss(mf, mf_s, st_end):
    old_mnw2 = mf.mnw2
    pump_data = []
    sps = old_mnw2.stress_period_data.data.keys()
    for sp in sps:
        curr_ = old_mnw2.stress_period_data.data[sp]
        curr_ = pd.DataFrame(curr_)
        curr_['SP'] = sp
        pump_data.append(curr_)
    pump_data = pd.concat(pump_data)
    mask = (pump_data['SP'] >= st_end[0]) & (pump_data['SP'] <= st_end[1])
    pump_data = pump_data.loc[mask, :]
    pump_ = pump_data.groupby('wellid').mean()
    del(pump_['SP'])
    itmp = len(pump_)
    pump_ = pump_.to_records()
    mnw2 = flopy.modflow.ModflowMnw2(model=mf_s, mnwmax=old_mnw2.mnwmax,
                                     node_data=old_mnw2.node_data,
                                     stress_period_data={0:pump_},
                                     itmp=[itmp])
    xxx = 1

def wel_ss(mf, mf_s, st_end):
    i = 0
    counter = 0
    while True:
        try:
            wel_array = mf.wel.stress_period_data.to_array(i)['flux']

        except:
            break
        if i== st_end[0]:
            if i >= st_end[0] and i <= st_end[1]:
                well_sum = wel_array
                counter = counter + 1
        elif i > st_end[0]:
            if i >= st_end[0] and i <= st_end[1]:
                well_sum = well_sum + wel_array
                counter = counter + 1

        i = i + 1
        if i > max(mf.wel.stress_period_data.data.keys()):
            break

    well_averages = well_sum/ (counter)
    inac_loc = mf.bas6.ibound.array == 0
    well_averages[inac_loc] = 0

    #well_averages1 = np.mean(mf.wel.stress_period_data.array['flux'], axis=0)

    loc = np.where(np.logical_not(well_averages==0))
    flow = well_averages[loc]
    well_dict = np.vstack((loc[0], loc[1], loc[2], flow))
    well_dict = well_dict.transpose()
    ts_wells = {0:well_dict, 1:well_dict}
    wel = flopy.modflow.ModflowWel(mf_s, stress_period_data= ts_wells, ipakcb=True)
    pass

def oc_ss(mf, mf_ss):
    # Add OC package to the MODFLOW model
    options = ['PRINT HEAD', 'PRINT DRAWDOWN', 'PRINT BUDGET',
               'SAVE HEAD', 'SAVE DRAWDOWN', 'SAVE BUDGET',
               'SAVE IBOUND', 'DDREFERENCE']
    idx = 0
    spd = dict()
    for sp in mf_ss.dis.nstp:
        stress_period = idx
        step = sp - 1
        ke = (stress_period, step)
        idx = idx + 1
        spd[ke] = [options[3], options[2], options[5]]
    oc = flopy.modflow.ModflowOc(mf_ss, stress_period_data=spd, cboufm='(20i5)')
    pass

def uzf_ss(mf, mf_ss):
    uzf = mf.uzf
    annual_rain = np.load('annual_rain.npy')
    annual_rain = annual_rain # convert to ft/day
    annual_rain = annual_rain * 0.40
    ones = np.ones_like(annual_rain)
    finf = [annual_rain, annual_rain]
    pett = [0.01 * ones, 0.01 * ones]
    extdp = [6.0 * ones, 6.0 * ones]
    extwc = [(0.18 * ones), (0.18 * ones)]

    uzf = flopy.modflow.ModflowUzf1(mf_ss, nuztop=uzf.nuztop, iuzfopt=uzf.iuzfopt, irunflg=uzf.irunflg, ietflg=uzf.ietflg,
                                    ipakcb=uzf.ipakcb, iuzfcb2=uzf.iuzfcb2, ntrail2=uzf.ntrail2, nsets=uzf.nsets, nuzgag=uzf.nuzgag,
                                    surfdep=uzf.surfdep, iuzfbnd=uzf.iuzfbnd, irunbnd=uzf.irunbnd, vks=uzf.vks.array,
                                    eps=uzf.eps.array, thts= uzf.thts.array,
                                    thti=0.25, specifythtr=0, specifythti=0, nosurfleak=0,
                                    finf=finf, pet= uzf.pet[0].array,
                                    extdp= uzf.extdp[0].array, extwc=uzf.extdp[0].array)
    pass

def hob_ss(mf_tran, mf_ss, obs_dict, stress_start_end):
    tim = mf_tran.dis.nstp.array
    tim = np.cumsum(tim)
    tim_st = tim[stress_start_end[0]-1]
    tim_end = tim[stress_start_end[1]-1]
    hoblist = []
    obs_data = obs_dict
    for obs_i in range(len(obs_data)):
        obs_obj = obs_data[obs_i]
        obs = obs_obj.time_series_data['hobs']
        if len(obs) > 1:
            weight = obs.std()
        elif len(obs)==1:
            weight = 5.0 # 9 ft
        else:
            print( "Empty observation....")
        print (np.max(obs_obj.time_series_data['totim']))
        loc = np.logical_and(obs_obj.time_series_data['totim']>= tim_st, obs_obj.time_series_data['totim']<= tim_end)
        if np.any(loc):
            hmean = obs_obj.time_series_data['hobs'][loc].mean()
            tim_ser_data = np.array([[0,hmean]])

            obs1 = flopy.modflow.HeadObservation(mf_ss, obsname= obs_obj.obsname, layer=obs_obj.layer, row=obs_obj.row,
                                                 column=obs_obj.column, roff=obs_obj.roff, coff=obs_obj.coff, itt=1,
                                                 time_series_data=tim_ser_data, mlay= obs_obj.mlay)
            hoblist.append(obs1)
    hob = flopy.modflow.ModflowHob(mf_ss, iuhobsv=53, hobdry=-9999., obs_data=hoblist)

def chd_ss(mf, mf_ss):
    ipakcb = True
    Boundary_Conditions = mf.chd.stress_period_data[0]
    hds = flopy.modflow.mfchd.ModflowChd(mf_ss, stress_period_data=Boundary_Conditions, ipakcb=ipakcb)