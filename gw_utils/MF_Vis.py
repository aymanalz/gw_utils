import os, sys
import gw_utils
import pandas

import flopy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, CheckButtons
from flopy.plot import plotutil
from gw_utils.general_util import get_mf_files
from gw_utils import plot_heads
class D3_info(object):
    def __init__(self):
        pass

    def layout(self):
        # Button
        labels = [ '3D Grid', 'T', 'LogT' ]
        pass



class Vis(object):
    def __init__(self, fn):
        mf = flopy.modflow.Modflow.load(fn, model_ws=os.path.dirname(fn), load_only=['DIS', 'BAS6', 'UPW'])
        mf_files = get_mf_files(fn)
        hds_fn = mf_files['hds'][1]
        try:
            hob_out_fn = mf_files['HOB'][1]
            self.fn_hobOut = hob_out_fn
        except:
            pass
        try:  # text file
            import flopy.utils.formattedfile as ff
            hds = ff.FormattedHeadFile(hds_fn, precision='single')
        except:  # binary
            hds = flopy.utils.HeadFile(hds_fn)
            pass

        self.mf = mf
        self.hds = hds
        self.layout()

    def layout(self):

        # show main figure
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.all_times = self.hds.get_times()

        self.curr_time_index = 0
        self.curr_layer = 0
        self.togle_TS = 0
        self.curr_label = 'Grid Elev'

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()  # only windows

        # add  buttons
        ax_Extra = plt.axes([0.5, 0.02, 0.05, 0.05])
        self.Extrabn = Button(ax_Extra, 'Extra')
        self.Extrabn.on_clicked(self.strat_extra_frame)

        ax_timeseries = plt.axes([0.55, 0.02, 0.05, 0.05])
        self.Tsbn = Button(ax_timeseries, 'Hydrograph')
        self.Tsbn.on_clicked(self.PotTS)

        axBackward = plt.axes([0.6, 0.02, 0.05, 0.05])
        self.Bkbn = Button(axBackward, '<<')
        self.Bkbn.on_clicked(self.plot_backward)

        axForward = plt.axes([0.65, 0.02, 0.05, 0.05])
        self.Forbn = Button(axForward, '>>')
        self.Forbn.on_clicked(self.plot_forward)

        axUP= plt.axes([0.7, 0.02, 0.05, 0.05])
        self.upbn = Button(axUP, 'Up')
        self.upbn.on_clicked(self.changeLayerUp)

        axDN = plt.axes([0.75, 0.02, 0.05, 0.05])
        self.Dnbn = Button(axDN, 'Down')
        self.Dnbn.on_clicked(self.chaneLayerDn)

        #axcolor = 'lightgoldenrodyellow'
        plt.text(0.01, 0.875, "MF Package", fontsize=14, transform=plt.gcf().transFigure)
        rax = plt.axes([0.01, 0.65, 0.09, 0.22])
        radio = RadioButtons(rax, ('Grid Elev', 'Heads', 'Drawdown', 'flow direction', 'Grid Thickness', 'IBOUND', 'STRT', 'HK', 'VK', 'SS', 'SY', 'GWDepth'))
        radio.on_clicked(self.data_mode)

        plt.text(0.01, 0.16, "Point Layer", fontsize=14, transform=plt.gcf().transFigure)
        cax = plt.axes([0.01, 0.001, 0.09, 0.15])
        self.CheckedPointlabels = []
        self.checkPoints = CheckButtons(cax, ['HOBS', 'WELLS', 'GAGES'], actives = (False, False, False))
        self.checkPoints.on_clicked(self.plot_points)


        self.LayerTxt = plt.text(0.8, 0.87, "Layer {}".format(self.curr_layer), fontsize=14,
                                 transform=plt.gcf().transFigure)


        self.TimeTxt = plt.text(0.2, 0.87, "Totim {}".format(self.all_times[self.curr_time_index]), fontsize=14,
                                 transform=plt.gcf().transFigure)
        self.curr_layer = -1
        self.chaneLayerDn()

        plt.show()

    def strat_extra_frame(self, event):
        self.plot_thickness_trans()


    def plot_thickness_trans(self):
        thk = self.mf.modelgrid.thick
        ib3d = self.mf.bas6.ibound.array
        ib3d[ib3d != 0] = 1
        hk = self.mf.upw.hk.array


        Total_thickness = (thk * ib3d).sum(axis = 0)
        trans = (thk * ib3d*hk).sum(axis = 0)
        ib2 = ib3d.sum(axis=0)

        ib2 = ib2 / ib2

        fig1, ax1 = plt.subplots()
        s1 = ax1.imshow(np.log10(trans*ib2), cmap = 'jet')
        ax1.set_title("Log Total Transmissivity")
        fig1.colorbar(s1, ax = ax1)
        fig1.canvas.draw()

        fig2, ax2 = plt.subplots()
        s2 = ax2.imshow(Total_thickness * ib2, cmap = 'plasma')
        ax2.set_title("Log Total Thickness")
        fig2.colorbar(s2, ax=ax2)
        fig2.canvas.draw()

        plt.show()
        xx = 1
        pass

    def plot_points(self, event):
        from gw_utils.hob_util import hob_output_to_df, in_hob_to_df # ugly
        fn = os.path.join(self.mf.model_ws, self.mf.namefile)

        self.hobin = in_hob_to_df(mfname = fn)
        self.houout = hob_output_to_df(fn, mf=self.mf)

        status = self.checkPoints.get_status()
        for i, label in enumerate(self.checkPoints.labels):
            if status[i]:
                self.CheckedPointlabels.append(label.get_text())

        for label in self.CheckedPointlabels:
            if label == 'HOBS':
                self.axHob = self.ax.plot(self.hobin['col'], self.hobin['row'], marker= '.',
                                          markeredgecolor = 'b', picker = 5, linestyle = 'None')
                self.fig.canvas.mpl_connect('pick_event', self.on_pick_hob)

        self.fig.canvas.draw()
        xx = 1

    def on_pick_hob(self, event):
        # make code to select the colosets
        try:
            self.AxSelection.remove()
        except:
            pass

        ind = event.ind
        if len(ind) > 1:
            datax, datay = event.artist.get_data()
            datax, datay = [datax[i] for i in ind], [datay[i] for i in ind]
            msx, msy = event.mouseevent.xdata, event.mouseevent.ydata
            dist = np.sqrt((np.array(datax) - msx) ** 2 + (np.array(datay) - msy) ** 2)
            close_i = np.argmin(dist)
            ind = [ind[np.argmin(dist)]]
        x= datax[close_i]
        y = datay[close_i]

        self.AxSelection = self.ax.scatter(x, y,
                                           marker='*', c='r',
                                           s=200)

        name = self.hobin.iloc[ind]['name'].values[0]
        hrow =self.hobin.iloc[ind]['row'].values[0]
        hcol = self.hobin.iloc[ind]['col'].values[0]

        curr_hob_in = self.hobin[(self.hobin['row'] == hrow) & (self.hobin['col'] == hcol)]
        names = curr_hob_in['name'].unique()

        curr_obs_df = self.houout[self.houout['OBSERVATION NAME'].isin(names)]
        curr_obs_df = curr_obs_df.drop_duplicates(subset='OBSERVATION NAME', keep='last')
        curr_hob_in = curr_hob_in.drop_duplicates(subset='name', keep='first')

        self.plot_head_ts(x, y)

        tim = curr_hob_in['totim'].values
        obs_values =curr_obs_df['OBSERVED VALUE'].values
        sim_values = curr_obs_df['SIMULATED EQUIVALENT'].values

        maskSim = 0
        sim_values[sim_values==maskSim] = np.NaN

        self.axhydro.scatter(tim, obs_values, c = 'r', label='OBS')
        self.axhydro.scatter(tim, sim_values, c= 'b', label='SIM')
        self.axhydro.legend()
        title = self.axhydro.get_title()
        if "." in names[0]:
            name = names[0].split(".")[0]
        else:
            name = names[0]
        title = title + "\n" + name
        kkeys = curr_hob_in['mlay'].values[0].keys()
        vals = curr_hob_in['mlay'].values[0].values()

        kval = zip(kkeys, vals)
        title = title + "\n"
        for kv in list(kval):
            title = title + " {} : {}, ".format(kv[0], kv[1])
        self.axhydro.set_title(title)

        event.canvas.draw()
        event.canvas.flush_events()

    def get_arr(self):
        if self.curr_label in ['Heads', 'flow direction']:
            totim = self.all_times[self.curr_time_index]
            self.maxDataLayer = self.mf.nlay
            arr = self.hds.get_data(totim=totim)

        elif self.curr_label == 'Grid Elev':
            self.maxDataLayer = self.mf.nlay
            arr = np.zeros((self.mf.nlay+1, self.mf.nrow, self.mf.ncol))
            arr[0,:,:] = self.mf.dis.top.array
            arr[1:, :, :] = self.mf.dis.botm.array

        elif self.curr_label =='Drawdown':
            totim = self.all_times[self.curr_time_index]
            totim0 = self.all_times[0]
            self.maxDataLayer = self.mf.nlay
            arr = self.hds.get_data(totim=totim)-self.hds.get_data(totim=totim0)

        elif self.curr_label =='HK':
            arr = self.mf.upw.hk.array

        elif self.curr_label =='VK':
            arr = self.mf.upw.vka.array

        elif self.curr_label =='IBOUND':
            arr = self.mf.bas6.ibound.array

        elif self.curr_label == 'STRT':
            arr = self.mf.bas6.strt.array

        elif self.curr_label == 'SS':
            arr = self.mf.upw.ss.array

        elif self.curr_label == 'SY':
            arr = self.mf.upw.sy.array

        elif self.curr_label == 'GWDepth':
            totim = self.all_times[self.curr_time_index]
            self.maxDataLayer = self.mf.nlay
            arr = self.hds.get_data(totim=totim)
            arr = arr.copy()
            ttop = self.mf.dis.top.array.copy()
            for k in range(self.mf.nlay):
                arr[k, :,:] = ttop - arr[k,:,:]

        elif self.curr_label ==  'Grid Thickness':
            arr = self.mf.modelgrid.thick


        return arr

    def data_mode(self, label):
        self.curr_label = label


    def PotTS(self, event):
        if self.togle_TS == 0:
            self.togle_TS = 1
            self.Tsbn.color = 'red'
            cid1 = self.fig.canvas.mpl_connect('button_press_event', lambda event: self.click_hydrograph(event))
            print("red")

        else:
            self.togle_TS = 0
            self.Tsbn.color = 'green'
            cid2 = self.fig.canvas.mpl_connect('close_event', lambda event: self.close_hydro(event))

    def close_hydro(self, event):
        pass

    def plot_head_ts(self, x, y):
        Ly = self.mf.dis.delc.array.sum()
        Lx = self.mf.dis.delr.array.sum()
        x = Lx * x / self.mf.ncol
        y = -Ly * y / self.mf.nrow

        modelgrid  = self.mf.modelgrid
        rr_cc1 = modelgrid.intersect(x, y)
        self.fighydro, self.axhydro = plt.subplots()

        elevs = []
        pre_elev = 0
        for k in range(self.mf.nlay):
            rr_cc = (k, rr_cc1[0], rr_cc1[1])
            ib = self.mf.bas6.ibound.array[k, rr_cc1[0], rr_cc1[1]]
            # get elevation
            if k == 0:
                elev = self.mf.dis.top.array[rr_cc1[0], rr_cc1[1]]
                pre_elev = elev
                elevs.append(elev)

            if ib != 0:
                elev = self.mf.dis.botm.array[k, rr_cc1[0], rr_cc1[1]]
                pre_elev = elev
                elevs.append(elev)
            else:
                elevs.append(pre_elev)

            ts = self.hds.get_ts(rr_cc)
            ts[:, 1][ts[:, 1] == self.mf.bas6.hnoflo] = np.NaN

            lb = "Head {}".format(k + 1)
            self.axhydro.plot(ts[:, 0], ts[:, 1], label=lb)
        # plot layers
        if 0:
            for ielev, elev in enumerate(elevs):
                ts2 = ts.copy()
                ts2[:, 1] = elev
                ax.plot(ts2[:, 0], ts2[:, 1], label="Layer {}".format(ielev))
        plt.legend()
        plt.title("Row = {}, Col= {}".format(rr_cc1[0], rr_cc1[1]))
        plt.show()

    def click_hydrograph(self,event):
            if self.togle_TS==0:
                return
            try:
                self.axHeadLocation.remove()
            except:
                pass
            self.axHeadLocation = self.ax.scatter(event.xdata, event.ydata, color='r')
            event.canvas.draw()
            event.canvas.flush_events()

            ##
            if 0:
                Ly = self.mf.dis.delc.array.sum()
                Lx = self.mf.dis.delr.array.sum()
                x = Lx * event.xdata/self.mf.ncol
                y = Ly - Ly * event.ydata / self.mf.nrow

            self.plot_head_ts(event.xdata,event.ydata)


    def chaneLayerDn(self, event=1):
        self.curr_layer = self.curr_layer + 1
        if self.curr_layer > self.mf.nlay-1 :
            self.curr_layer = self.curr_layer - 1
            return False
        self.update_fig()
        print(self.curr_layer)

    def changeLayerUp(self, event):
        self.curr_layer = self.curr_layer - 1
        if self.curr_layer < 0:
            self.curr_layer = self.curr_layer + 1
            return False
        self.update_fig()
        print(self.curr_layer)

    def update_fig(self):
        arr = self.get_arr()
        #self.fig.suptitle("Totim = {} ".format(totim))
        if self.curr_label == 'flow direction':
            self._plot_dir(arr)
        else:
            self._plot_arr(arr)

    def _plot_dir(self, arr):
        self.ax.clear()
        arr = arr[self.curr_layer, :,:]
        ibound = self.mf.bas6.ibound.array[self.curr_layer, :,:]
        arr[ibound == 0] = np.NaN
        # *******************
        # second subplot
        x = np.arange(0, self.mf.ncol, 1)
        y = np.arange(0, self.mf.nrow, 1)
        X, Y = np.meshgrid(x, y)
        dx, dy = np.gradient(arr)
        dx = dx
        dy = dy
        n = -2
        color = np.sqrt(((dx - n) / 2)**2 + ((dy - n) / 2)**2)
        f = lambda x: np.sign(x) * np.log10(1 + np.abs(x))
        im = self.ax.quiver(X, Y, f(dx), f(dy), color, scale=0.5, units="xy")
        #im =  self.ax.streamplot(X, Y, dx, dy)

        self.ax.contour(arr, colors = 'k')
        self.ax.invert_yaxis()
        self.LayerTxt.remove()
        self.LayerTxt = plt.text(0.8, 0.87, "Layer {}".format(self.curr_layer+1), fontsize=14,
                                 transform=plt.gcf().transFigure, color='r')
        self.TimeTxt.remove()
        self.TimeTxt = plt.text(0.2, 0.87, "Totim {}".format(self.all_times[self.curr_time_index]), fontsize=14,
                                 transform=plt.gcf().transFigure)
        self.fig.canvas.draw()


        try:
            self.cax.remove()
        except:
            pass
        self.cax = self.fig.colorbar(im, fraction = 0.02, pad = 0.01, ax= self.ax, orientation =	'vertical' )

    def _plot_arr(self, arr):
        self.ax.clear()
        arr = arr[self.curr_layer, :,:]
        ibound = self.mf.bas6.ibound.array[self.curr_layer, :,:]
        arr[ibound == 0] = np.NaN
        im = self.ax.imshow(arr)


        self.ax.contour(arr, colors = 'k')

        self.LayerTxt.remove()
        self.LayerTxt = plt.text(0.8, 0.87, "Layer {}".format(self.curr_layer+1), fontsize=14,
                                 transform=plt.gcf().transFigure, color='r')
        self.TimeTxt.remove()
        self.TimeTxt = plt.text(0.2, 0.87, "Totim {}".format(self.all_times[self.curr_time_index]), fontsize=14,
                                 transform=plt.gcf().transFigure)
        self.fig.canvas.draw()

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(self.ax)
        try:
            self.cax.remove()
            del(divider)
        except:
            pass
        self.cax = self.fig.colorbar(im, fraction = 0.02, pad = 0.01, ax= self.ax, orientation =	'vertical' )

    def plot_forward(self, event):
        self.curr_time_index = self.curr_time_index + 1
        if self.curr_time_index> len(self.all_times)-1:
            self.curr_time_index = self.curr_time_index  - 1
            return False
        self.update_fig()

    def plot_backward(self, event):
        self.curr_time_index = self.curr_time_index - 1
        if self.curr_time_index < 0:
            self.curr_time_index = self.curr_time_index - 1
            return False

        self.update_fig()




if __name__ == "__main__":
    #fn = r"D:\Models\San_Antonio\PEST_Runing_2\local_run\model\SACr.nam"
    fn = r"D:\Workspace\projects\RussianRiver\RR_GSFLOW_GIT\RR_GSFLOW\GSFLOW\archive\current_version\windows\rr_tr.nam"
    fn = r"D:\Workspace\projects\RussianRiver\RR_GSFLOW_GIT\RR_GSFLOW\MODFLOW\modflow_calibration\ss_calibration\slave_dir\mf_dataset\rr_ss.nam"

    Vis(fn = fn)
    pass
