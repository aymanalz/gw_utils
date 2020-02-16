import os, sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import hob_util
sys.path.insert(0,"C:\work\Russian_River\py_pkgs")
import flopy
import geopandas

"""
This packages can be used to tranforming an existing modflow dataset to a new grid
"""


class Grid_transformer(object):
    def __init__(self, mf=None):
        self.mf = mf

    def change_grid_layers(self, mfname, new_grid, new_ibound, new_workspace, new_name):
        """

        Parameters
        ----------
        new_grid
        new_workspace
        new_name

        Returns
        -------

        """

        new_mf = flopy.modflow.Modflow(new_name, model_ws=new_workspace, version='mfnwt')
        self.new_mf = new_mf
        self.new_grid = new_grid
        self.new_ibound = new_ibound

        # lood old model
        load_only_list = ['DIS', 'BAS6', 'GHB', 'UPW', 'SFR', 'GAGE', 'mnw2',
                          'UZF', 'HFB6', 'NWT', 'OC']

        self.old_mf = flopy.modflow.Modflow.load(os.path.basename(mfname),
                                                 model_ws= os.path.dirname(mfname),
                                                 load_only=load_only_list,  forgive = False)
        self.old_grid = np.zeros(shape=(self.old_mf.dis.nlay + 1, self.old_mf.dis.nrow, self.old_mf.dis.ncol))
        self.old_grid[0, :, :] = self.old_mf.dis.top.array.copy()
        self.old_grid[1:, :, :] = self.old_mf.dis.botm.array.copy()

        # intersection fractions
        self.all_weights = self.calc_layer_weights()

        # loop over packages and change them
        self._update_dis()
        self._update_bas6()
        self._update_hfb()
        self._update_ghb()
        self._update_upw()
        self._update_sfr()
        self._update_gage()
        self._update_mnw2()
        self._update_oc(monthly = True)
        self._update_uzf()

        self.new_mf.change_model_ws(self.new_mf.model_ws)

        self.new_mf.write_input()

        #self._update_hob()

    def _update_hfb(self):
        #
        fn = r"D:\Yucaipa_work\gis\MasterModelFaults.shp"
        hfb_att = geopandas.read_file(fn)
        nphfb = 0  # Number of horizontal-flow barrier parameters
        mxfb = 0  # Maximum number of horizontal-flow barrier barriers that will be defined using parameters

        hfb_data = []
        hydchr = 10
        curr_hfb = []
        for i in np.arange(len(hfb_att)):

            for layer in np.arange(0, self.new_mf.nlay):
                curr_hfb.append(layer)
                curr_hfb.append(hfb_att.loc[i]['ROWVAL_1']-1)
                curr_hfb.append(hfb_att.loc[i]['COLVAL_1']-1)
                curr_hfb.append(hfb_att.loc[i]['ROWVAL_2']-1)
                curr_hfb.append(hfb_att.loc[i]['COLVAL_2']-1)
                curr_hfb.append(hydchr)
                hfb_data.append(curr_hfb)
                curr_hfb = []

        nhfbnp = len(hfb_data)
        nacthfb = len(hfb_data)

        ##
        old_hfb = self.old_mf.hfb6
        old_hfd_data = pd.DataFrame(old_hfb.hfb_data)
        new_hfd_data = pd.DataFrame(hfb_data, columns=old_hfd_data.columns)
        old_rr_col = list(zip(old_hfd_data['irow1'].values, old_hfd_data['icol1'].values,
                              old_hfd_data['irow2'].values, old_hfd_data['icol2'].values))
        old_hfd_data['rr_cc'] = old_rr_col
        new_rr_col = list(zip(new_hfd_data['irow1'].values, new_hfd_data['icol1'].values,
                              new_hfd_data['irow2'].values, new_hfd_data['icol2'].values))
        new_hfd_data['rr_cc'] = new_rr_col

        for ihfb in new_rr_col:
            mask = old_hfd_data['rr_cc'] == ihfb
            if np.any(mask.values):
                cond = old_hfd_data.loc[mask, 'hydchr'].mean()
                pass
            else:
                cond = 1
            if cond == 0:
                cond = 10e-7
            masknew = new_hfd_data['rr_cc'] == ihfb
            new_hfd_data.loc[masknew, 'hydchr'] = cond

        del(new_hfd_data['rr_cc'])
        new_hfd_data = new_hfd_data.values.tolist()
        hfb = flopy.modflow.ModflowHfb(self.new_mf, nphfb=nphfb, mxfb=mxfb, nhfbnp=nhfbnp, hfb_data=new_hfd_data,
                                       nacthfb=nacthfb,  no_print=False, options=None)
        xx = 1

    def _update_hob(self):
        namfile = os.path.join(self.old_mf.model_ws, self.old_mf.namefile)
        hob_util.in_hob_to_df(mfname =namfile )


    def _update_uzf(self):
        old_uzf = self.old_mf.uzf
        hds_file = os.path.splitext(self.old_mf.uzf.fn_path)[0] + ".hds"
        hds = flopy.utils.HeadFile(hds_file)
        times = hds.get_times()
        headfield = 0
        for tt in times:
            headfield =  headfield + hds.get_data(totim = tt)
        headfield = headfield / len(times)
        ibound = 1.0 * self.old_mf.bas6.ibound.array.copy()
        ibound[ibound==0] = np.NAN
        wt = np.nanmean(ibound * headfield, axis = 0)
        iuzfbd_new = np.zeros_like(old_uzf.iuzfbnd.array)
        for kk in range(self.new_mf.nlay):
            topp = self.new_grid[kk,:,:]
            bott = self.new_grid[kk+1,:,:]
            mask_top = wt<=topp
            mask_bott = wt>=bott
            mask = np.logical_and(mask_top, mask_bott)
            iuzfbd_new[mask] = kk+1

        ibound2D_new = self.new_mf.bas6.ibound.array.sum(axis = 0)
        iuzfbd_new[np.logical_and(iuzfbd_new == 0, ibound2D_new > 0)] = 1
        old_uzf.iuzfbnd = iuzfbd_new
        self.new_mf.add_package(old_uzf)
        self.new_mf.change_model_ws(self.new_mf.model_ws)

        if 0:
            uzf = flopy.modflow.ModflowUzf1(self.new_mf, nuztop=old_uzf.nuztop, iuzfopt=old_uzf.iuzfopt,
                                            irunflg=old_uzf.irunflg, ietflg=old_uzf.ietflg,
                                            ipakcb=1, iuzfcb2=old_uzf.iuzfcb2, ntrail2=old_uzf.ntrail2,
                                            nsets=old_uzf.nsets, nuzgag=old_uzf.nuzgag,  surfdep= old_uzf.surfdep,
                                            iuzfbnd= old_uzf.iuzfbnd.array, irunbnd=old_uzf.irunbnd.array,
                                            vks= old_uzf.vks.array, eps=old_uzf.eps.array, thts= old_uzf.thts.array,
                                            thtr= old_uzf.thtr.array, thti= old_uzf.thti.array,
                                            specifysurfk =old_uzf.specifysurfk,
                                            specifythtr= old_uzf.specifythtr, specifythti= old_uzf.specifythti,
                                            nosurfleak= old_uzf.nosurfleak,  finf=old_uzf.finf.array[0][0],
                                            pet= old_uzf.pet.array[0][0],   extdp=old_uzf.extdp.array[0][0],
                                            extwc= old_uzf.extwc.array[0][0])
        pass
    def _update_oc(self, monthly = True):
        if monthly:
            options = ['PRINT HEAD', 'PRINT DRAWDOWN', 'PRINT BUDGET',
                       'SAVE HEAD', 'SAVE DRAWDOWN', 'SAVE BUDGET',
                       'SAVE IBOUND', 'DDREFERENCE']
            idx = 0
            spd = dict()
            for sp in self.new_mf.dis.nstp:
                stress_period = idx
                step = sp - 1
                ke = (stress_period, step)
                idx = idx + 1
                spd[ke] = [options[3], options[2], options[5]]
        else:
            spd = self.old_mf.oc.stress_period_data
        oc = flopy.modflow.ModflowOc(self.new_mf, stress_period_data=spd, cboufm='(20i5)')


    def _update_mnw2(self):

        old_mnw2 = self.old_mf.mnw2
        mnw2 = flopy.modflow.ModflowMnw2(model=self.new_mf, mnwmax=old_mnw2.mnwmax,
                                         node_data=old_mnw2.node_data,
                                         stress_period_data=old_mnw2.stress_period_data.data,
                                         itmp=old_mnw2.itmp )
        pass
    def _update_gage(self):
        old_gages = self.old_mf.gage
        gage = flopy.modflow.ModflowGage(self.new_mf, numgage=old_gages.numgage, gage_data=old_gages.gage_data, files=old_gages.files)
        pass

    def _update_sfr(self):

        old_sfr = self.old_mf.sfr
        nstrm = old_sfr.nstrm
        nss = old_sfr.nss
        const = old_sfr.const
        nsfrpar = old_sfr.nsfrpar
        nparseg = old_sfr.nparseg
        dleak = old_sfr.dleak
        nstrail = old_sfr.nstrail
        isuzn = old_sfr.isuzn
        nsfrsets = old_sfr.nsfrsets
        istcb2 = old_sfr.istcb2
        isfropt = old_sfr.isfropt
        irtflg = old_sfr.irtflg
        reach_data = pd.DataFrame(old_sfr.reach_data)
        ibound = self.new_mf.bas6.ibound.array.copy()
        for irec, record in reach_data.iterrows():
            row = record['i']
            col = record['j']
            ib = ibound[:,int(row), int(col)]
            for k, bb in enumerate(ib):
                if bb ==1:
                    break;
            reach_data.loc[irec, 'k']=k

        reach_data = reach_data.to_records(index = False)
        segment_data = old_sfr.segment_data
        numtim = old_sfr.numtim
        weight = old_sfr.weight
        channel_geometry_data = old_sfr.channel_geometry_data
        channel_flow_data = old_sfr.channel_flow_data
        dataset_5 = old_sfr.dataset_5
        sfr = flopy.modflow.ModflowSfr2(self.new_mf, nstrm=nstrm, nss=nss, const=const, nsfrpar= nsfrpar, nparseg = nparseg,
                                        dleak=dleak, ipakcb=1, nstrail = nstrail, isuzn = isuzn,  nsfrsets= nsfrsets,
                                        istcb2=istcb2, reachinput=True, isfropt = isfropt, irtflg = irtflg,
                                        reach_data=reach_data, numtim = numtim, weight = weight,
                                        segment_data=segment_data,
                                        channel_geometry_data=channel_geometry_data,
                                        channel_flow_data=channel_flow_data,
                                        dataset_5=dataset_5)
        pass
    def _update_dis(self):
        nlays, nrows, ncols = self.new_grid.shape
        nlays = nlays - 1
        delr = np.unique(self.old_mf.dis.delr.array)[0]
        delc = np.unique(self.old_mf.dis.delc.array)[0]

        dis = flopy.modflow.ModflowDis(self.new_mf, nlay=nlays, nrow=nrows, ncol=ncols,
                                       delr=delr, delc=delc,
                                       top=self.new_grid[0, :, :], botm=self.new_grid[1:, :, :],
                                       nper=self.old_mf.dis.nper, perlen=self.old_mf.dis.perlen.array.copy(),
                                       nstp=self.old_mf.dis.nstp.array.copy(),
                                       steady=self.old_mf.dis.steady.array.copy(),
                                       itmuni=4, lenuni=1, xul=0,
                                       yul=0)  # (4) days, 1 ft
        dis.check()

    def _update_bas6(self):
        strt = self.old_mf.bas6.strt.array.copy()
        ibound = self.old_mf.bas6.ibound.array.copy()
        ibound = ibound.astype(float)
        ibound[ibound==0] = np.NAN
        strt = strt*ibound
        wt = np.nanmean(strt, axis=0)
        #new_strt = self.calc_new_property(strt)
        new_strt = np.zeros_like(strt)
        botm_elevation = self.new_mf.dis.botm.array[-1,:,:]
        for ilay in range(self.new_mf.nlay):
            curr_heads = wt.copy()
            mask = curr_heads <= botm_elevation
            curr_heads[mask] = botm_elevation[mask] + 1.0
            if np.any(mask):
                xx = 1

            new_strt[ilay, :, :] = curr_heads
        new_strt[np.isnan(new_strt)] = 0.0
        bas = flopy.modflow.ModflowBas(self.new_mf, ibound=self.new_ibound, ichflg=True, strt=new_strt)
        bas.check()

    def _update_ghb(self):
        old_ghb_data = self.old_mf.ghb.stress_period_data.data[0]
        df = pd.DataFrame(old_ghb_data)
        rows = df['i'].values
        cols = df['j'].values
        rr_cc = list(zip(rows, cols))
        df['rr_cc'] = rr_cc
        groups_by_rr_cc = df.groupby(by='rr_cc')
        new_ghb_data = [] #lay, row, col, head, cnd
        for gg_rr_cc in groups_by_rr_cc:
            curr_rr_cc = gg_rr_cc[0]
            old_conds = []
            old_heads = []
            for ii, old_ghb in gg_rr_cc[1].iterrows():
                if old_ghb['cond'] > 0:
                    old_cell_thk = self.old_mf.dis.thickness.array[old_ghb['k'], curr_rr_cc[0], curr_rr_cc[1]]
                    if old_cell_thk > 0:
                        kvalue = old_ghb['cond'] / old_cell_thk
                        old_conds.append(kvalue)
                        old_heads.append(old_ghb['bhead'])
            if len(old_conds) == 0:
                raise ValueError("GHB cond for current cell is zero ----> {}".format(curr_rr_cc))
            ghb_is_assigned = 0
            for kknew in range(self.new_mf.nlay):
                new_cell_thk = self.new_mf.dis.thickness.array[kknew, curr_rr_cc[0], curr_rr_cc[1]]
                cnd = new_cell_thk * np.mean(old_conds)
                curr_botm = self.new_mf.dis.botm.array[kknew, curr_rr_cc[0], curr_rr_cc[1] ]
                headd = np.mean(old_heads)
                if curr_botm<headd:
                    new_ghb_data.append([kknew,curr_rr_cc[0], curr_rr_cc[1], headd, cnd] )  # lay, row, col, head, cnd
                    ghb_is_assigned = 1

            if ghb_is_assigned==0:
                raise ValueError("No GHB is assigned at horizontal location {}".format(curr_rr_cc))


        ghb = flopy.modflow.ModflowGhb(self.new_mf, stress_period_data=new_ghb_data,
                                       ipakcb=1)
        ghb.check()

    def _update_upw(self):

        nlays, nrows, ncols = self.new_grid.shape
        nlays = nlays - 1
        kh = self.old_mf.upw.hk.array.copy()
        new_kh = self.calc_new_property(kh)
        vka = self.old_mf.upw.vka.array.copy()
        new_vka = self.calc_new_property(vka)
        sy = self.old_mf.upw.sy.array.copy()
        new_sy = self.calc_new_property(sy)
        ss = self.old_mf.upw.ss.array.copy()
        new_ss = self.calc_new_property(ss)
        laytyp = np.ones(nlays)
        laywet = np.zeros(nlays)

        upw = flopy.modflow.mfupw.ModflowUpw(self.new_mf, laytyp=laytyp, layavg=0, chani=1.0, layvka=0, laywet=laywet,
                                             hdry=-1e+30, iphdry=0, hk=new_kh, hani=1.0, vka=new_vka, ss=new_ss,
                                             sy=new_sy, vkcb=0.0, noparcheck=False, extension='upw', unitnumber=None,
                                             filenames=None, ipakcb=1)

        upw.check()
        pass

    def calc_new_property(self, old_3dval):
        # old grid info
        o_nlay, o_nrow, o_ncol = self.old_grid.shape
        o_nlay = o_nlay - 1
        o_thk = np.zeros(shape=(o_nlay, o_nrow, o_ncol))
        for k in range(o_nlay):
            o_thk[k, :, :] = self.old_grid[k, :, :] - self.old_grid[k + 1, :, :]

        # new grid info
        n_nlay, n_nrow, n_ncol = self.new_grid.shape
        n_nlay = n_nlay - 1
        new_3d_value = np.zeros(shape=(n_nlay, n_nrow, n_ncol))

        # old ibound
        ibound_old = self.old_mf.bas6.ibound.array.copy()
        for kk_new in range(n_nlay):
            curr_weight = self.all_weights[kk_new]
            curr_val = curr_weight * o_thk * old_3dval * ibound_old
            curr_val = curr_val.sum(axis=0)
            w = curr_weight * o_thk * ibound_old
            w = w.sum(axis=0)
            curr_val = curr_val / w
            curr_val[np.isnan(curr_val)] = 0
            new_3d_value[kk_new, :, :] = curr_val

        return new_3d_value

    def calc_over_lap(self, layer1, layer2):
        """

        Parameters
        ----------
        layer1 [top1,botm1] ---> 2*nrow*ncol
        layer2 [top2, botm2]---> 2*nrow*ncol

        Returns
        -------
        overlap ---> 1*nrow*ncol
        """

        maxtop = np.maximum(layer1[0, :, :], layer2[0, :, :])
        minbotm = np.minimum(layer1[1, :, :], layer2[1, :, :])
        thk1 = layer1[0, :, :] - layer1[1, :, :]
        thk2 = layer2[0, :, :] - layer2[1, :, :]
        overlap = thk1 + thk2 - (maxtop - minbotm)
        overlap[overlap < 0] = 0
        return overlap

    def calc_layer_weights(self):
        """
        Calculate the fraction of overlab between two grids.
        Returns
        -------
        a dictionary keyed by new grid layer 1 and values is a 3d matrix
        with weights that has size of the old grid

        """
        all_weights = {}
        new_nlay = self.new_grid.shape[0] - 1
        old_nlay = self.old_grid.shape[0] - 1
        for k in range(new_nlay):
            nlay = self.new_grid[k:k + 2, :, :]
            weights = np.zeros(shape=(self.old_grid.shape[0] - 1,
                                      self.old_grid.shape[1],
                                      self.old_grid.shape[2])
                               )
            for k2 in range(old_nlay):
                olay = self.old_grid[k2:k2 + 2, :, :]
                curr_overlap = self.calc_over_lap(olay, nlay)
                othk = olay[0, :, :] - olay[1, :, :]
                over_frac = curr_overlap / othk
                over_frac[np.isnan(over_frac)] = 0  # becuase othk can be zero
                over_frac[over_frac > 1] = 1.0
                weights[k2, :, :] = over_frac
            all_weights[k] = weights.copy()

        return all_weights


if __name__ == "__main__":
    if 0:
        mfname = r"D:\Yucaipa_work\tran_mf3dGIT\yucaipa.nam"
        mf = flopy.modflow.Modflow.load(mfname, load_only=['DIS', 'BAS6'])

        # generate an example new grid
        top = mf.dis.top.array.copy()
        botm = mf.dis.botm.array.copy()
        nrow = mf.nrow
        ncol = mf.ncol
        #   We need to split the top active layer
        #       Add Layer 0 to Layer 1
        new_thk = np.zeros(shape=(5, nrow, ncol))
        new_ibound = np.zeros_like(new_thk, dtype=int)
        thk = mf.dis.thickness.array.copy()
        lay1 = thk[0, :, :] + thk[1, :, :]
        new_thk[0, :, :] = lay1 * 0.8
        new_thk[1, :, :] = lay1 * 0.2
        new_thk[2, :, :] = thk[2, :, :] * 0.5
        new_thk[3, :, :] = thk[2, :, :] * 0.5
        new_thk[4, :, :] = thk[3, :, :]

        #       Ibound
        new_ibound[0, :, :] = mf.bas6.ibound.array[1, :, :]
        new_ibound[1, :, :] = mf.bas6.ibound.array[1, :, :]
        new_ibound[2, :, :] = mf.bas6.ibound.array[2, :, :]
        new_ibound[3, :, :] = mf.bas6.ibound.array[2, :, :]
        new_ibound[4, :, :] = mf.bas6.ibound.array[3, :, :]

        new_grid = np.zeros(shape=(6, nrow, ncol))
        for lay in range(6):
            if lay == 0:
                new_grid[lay, :, :] = top
                continue
            new_grid[lay, :, :] = new_grid[lay - 1, :, :] - new_thk[lay - 1, :, :]

        Gt = Grid_transformer(mf=mfname)
        Gt.change_grid_layers(new_grid=new_grid, new_ibound=new_ibound,
                              new_workspace=r"D:\Yucaipa_work\Yuc4",
                              new_name='yucaipa_v2')
