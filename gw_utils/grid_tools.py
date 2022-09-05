import os, sys
import numpy as np
import matplotlib.pyplot as plt
import flopy
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import minimum_filter


def main(mf, min_thk_ratio=0.1, min_con=30.0,
                    min_thk=50, max_slope=0.1, buffer_sigma=3,
                    value_sigma=3.0, apply_min_thickness = False,
                    fix_layer_connections = False,
                    smooth_grid = False):

    ## this is an example
    gf = Grid_Fixer(mf, min_thk_ratio= min_thk_ratio, min_con=min_con,
                    min_thk=min_thk, max_slope=max_slope, buffer_sigma=buffer_sigma,
                    value_sigma=value_sigma)

    nlay = mf.dis.nlay
    nrow = mf.nrow
    ncol = mf.ncol

    grid = np.zeros(shape=(nlay + 1, nrow, ncol))

    for i in range(nlay):
        if i == 0:
            grid[0] = mf.dis.top.array.copy()
        grid[i + 1] = mf.dis.botm.array[i]

    tk = np.zeros((nlay, nrow, ncol))
    for k in range(nlay):
        tk[k, :, :] = grid[k, :, :] - grid[k + 1]
    ibound = mf.bas6.ibound.array.copy()
    ibound[tk <= 0] = 0

    gf.ibound = ibound
    gf.grid = grid

    # fix min thickness
    if apply_min_thickness:
        gf.grid = gf.force_min_thikness()

    # fix connection issue
    if fix_layer_connections:
        gf.grid = gf.connection_fix()

    # fix steep angles
    if smooth_grid:
        gf.grid = gf.smooth_grid()

    mf.dis.botm = gf.grid[1:, :, :]
    return mf.dis


class Grid_Fixer(object):
    def __init__(self, mf, min_thk_ratio=0.1, min_con=30.0,
                 min_thk=50, max_slope=0.1, buffer_sigma=3, value_sigma=3.0):
        self.grid = []
        self.cell_size = mf.modelgrid.delc[0]
        self.min_thk_ratio = min_thk_ratio
        self.min_con = min_con
        self.min_thk = min_thk
        self.max_slope = max_slope
        self.buffer_sigma = buffer_sigma
        self.value_sigma = value_sigma

    def filter(self, U, sig):

        V = U.copy()
        V[U != U] = 0
        VV = gaussian_filter(V, sigma=sig)

        W = 0 * U.copy() + 1
        W[U != U] = 0
        WW = gaussian_filter(W, sigma=sig)

        Z = VV / WW
        return Z

    def grid_wt2(self):
        datum = 2400
        ibound = self.ibound

        grid = np.copy(self.grid)
        wt = np.mean(self.wt, axis=0)
        loc_ch = grid[-1, :, :] >= datum
        loc_nan = ibound == 0

        mask = np.zeros_like(loc_ch, dtype=float)
        mask[loc_ch] = 1
        mask[101:, :] = 0
        mask[loc_nan] = np.nan

        sat_thikness = grid[0, :, :] - grid[-1, :, :]
        thickness = np.zeros_like(grid[1:, :, :])
        for lay in range(grid.shape[0] - 1):
            thickness[lay, :, :] = grid[lay, :, :] - grid[lay + 1, :, :]
        new_thikness = np.zeros_like(thickness)
        for lay in range(grid.shape[0] - 1):
            curr_th = np.copy(thickness[lay, :, :])
            curr_th2 = np.copy(thickness[lay, :, :])
            if lay in [1, 2]:
                curr_th[loc_ch] = 0.0
                curr_th[loc_nan] = np.nan
                new_thikness[lay, :, :] = curr_th
                dfff = (thickness[lay, :, :] - curr_th)
                new_thikness[0, :, :] = new_thikness[0, :, :] + dfff
            else:
                new_thikness[lay, :, :] = np.copy(thickness[lay, :, :])

                pass
        new_grid = np.copy(grid)

        # new_thikness[new_thikness < 20.0] = 0.0

        for lay in range(grid.shape[0] - 1):
            new_grid[lay + 1] = new_grid[lay] - new_thikness[lay, :, :]
        return new_grid

    def grid_wt(self):
        datum = 2400
        ibound = self.ibound
        grid = np.copy(self.grid)
        wt = np.mean(self.wt, axis=0)
        loc = wt >= datum
        loc_nan = ibound == 0
        mask = np.zeros_like(loc, dtype=float)
        mask[loc] = 1
        mask[loc_nan] = np.nan
        mask2 = self.filter(mask, 2)
        mask2[loc_nan] = np.nan
        zone_to_change = np.logical_and((mask2 > 0.1), (mask2 < 0.99))
        alpha = np.zeros_like(zone_to_change, dtype=float)
        alpha[zone_to_change] = 1
        alpha = self.filter(alpha, 2)

        # diff = wt >= datum
        sat_thikness = grid[0, :, :] - grid[-1, :, :]
        thickness = np.zeros_like(grid[1:, :, :])
        for lay in range(grid.shape[0] - 1):
            thickness[lay, :, :] = grid[lay, :, :] - grid[lay + 1, :, :]
        new_thikness = np.zeros_like(thickness)
        for lay in range(grid.shape[0] - 1):
            curr_th = np.copy(thickness[lay, :, :])
            curr_th2 = np.copy(thickness[lay, :, :])
            if lay in [1, 2]:
                curr_th[loc] = 0.0
                curr_th2[loc] = 0.0
                curr_th[loc_nan] = np.nan
                curr_th = self.filter(curr_th, 5)
                # curr_th = curr_th2 * (1-alpha) + curr_th*alpha
                curr_th[loc_nan] = np.nan
                new_thikness[lay, :, :] = curr_th
                # loc2 = np.logical_and(curr_th>0, curr_th<20.0)
                # curr_th[loc2] = 0.0
                dfff = (thickness[lay, :, :] - curr_th)
                # dfff[dfff<0] = 0
                new_thikness[0, :, :] = new_thikness[0, :, :] + dfff
            else:
                new_thikness[lay, :, :] = np.copy(thickness[lay, :, :])

                pass
        new_grid = np.copy(grid)

        new_thikness[new_thikness < 20.0] = 0.0
        lcc = np.logical_and(new_thikness[1, :, :] == 0.0, new_thikness[2, :, :] > 0.0)

        second_lay1 = np.copy(new_thikness[2, :, :])

        addi = np.zeros_like(second_lay1)
        addi[lcc] = second_lay1[lcc]

        new_thikness[0, :, :] = new_thikness[0, :, :] + addi
        second_lay1[lcc] = 0
        new_thikness[2, :, :] = second_lay1

        for lay in range(grid.shape[0] - 1):
            new_grid[lay + 1] = new_grid[lay] - new_thikness[lay, :, :]
        return new_grid

    def filter(self, U, sig):

        V = U.copy()
        V[U != U] = 0
        VV = gaussian_filter(V, sigma=sig)

        W = 0 * U.copy() + 1
        W[U != U] = 0
        WW = gaussian_filter(W, sigma=sig)

        Z = VV / WW
        return Z

    def connection_fix(self):
        grid = np.copy(self.grid)
        lay, rows, cols = self.grid.shape
        new_grid = np.zeros_like(grid)
        new_grid[0, :, :] = grid[0, :, :]
        flg_ib = hasattr(self, 'ibound')
        kij_connections = []
        for k in range(lay - 1):
            for j in range(cols):
                for i in range(rows):
                    bt_list = []

                    # x+
                    if j < cols - 1:
                        if flg_ib:
                            bflg = self.ibound[k, i, j + 1]
                        if grid[k, i, j + 1] - grid[k + 1, i, j] < self.min_con and bflg == 1:
                            bt_list.append(grid[k, i, j + 1] - self.min_con)
                    # x-
                    if j > 0:
                        if flg_ib:
                            bflg = self.ibound[k, i, j - 1]
                        if grid[k, i, j - 1] - grid[
                            k + 1, i, j] < self.min_con and bflg == 1:  # curr botm is higher than right top
                            bt_list.append(grid[k, i, j - 1] - self.min_con)

                    if i < rows - 1:
                        if flg_ib:
                            bflg = self.ibound[k, i + 1, j]
                        if grid[k, i + 1, j] - grid[
                            k + 1, i, j] < self.min_con and bflg == 1:  # curr botm is higher than right top
                            bt_list.append(grid[k, i + 1, j] - self.min_con)

                    if i > 0:
                        if flg_ib:
                            bflg = self.ibound[k, i - 1, j]

                        if grid[k, i - 1, j] - grid[
                            k + 1, i, j] < self.min_con and bflg == 1:  # curr botm is higher than right top
                            bt_list.append(grid[k, i - 1, j] - self.min_con)

                    if len(bt_list) > 0:
                        kij_connections.append([k, i, j])
                        shift = grid[k + 1, i, j] - np.min(bt_list)
                        new_grid[(k + 1):, i, j] = grid[(k + 1):, i, j] - shift
                        grid[(k + 1):, i, j] = grid[(k + 1):, i, j] - shift
                    else:
                        new_grid[k + 1, i, j] = grid[k + 1, i, j]

        return new_grid

    def smooth_grid(self):
        """
        smooth the thicknesses
        :return:
        """
        self.thickness = np.zeros_like(self.grid[:-1, :, :])
        loc00 = self.ibound == 0
        for lay in range(self.thickness.shape[0]):
            ttop = np.copy(self.grid[lay, :, :])
            bnt = np.copy(self.grid[lay + 1, :, :])
            thkk = ttop - bnt
            thkk[loc00[lay]] = np.nan
            self.thickness[lay, :, :] = thkk

        new_thk = np.zeros_like(self.thickness, dtype=float)

        for lay in range(self.thickness.shape[0]):
            gr = np.gradient(self.thickness[lay, :, :])
            gg = np.power(gr[0], 2.0) + np.power(gr[1], 2.0)
            gg = np.power(gg, 0.5)
            gg = gg / (self.cell_size)

            zone_fix = gg > self.max_slope
            mask = np.zeros_like(zone_fix, dtype=float)
            mask[zone_fix] = 1.0
            alpha = self.filter(mask, self.buffer_sigma)
            alpha = alpha / np.max(alpha)
            sgrid = self.filter(self.thickness[lay, :, :], self.value_sigma)
            new_thickness = sgrid * alpha + (1 - alpha) * self.thickness[lay, :, :]
            new_thk[lay, :, :] = new_thickness
        new_grid = np.zeros_like(self.grid)
        new_grid[0, :, :] = self.grid[0, :, :]
        topp = new_grid[0, :, :]
        new_thk[np.isnan(new_thk)] = 0
        for lay in range(self.thickness.shape[0]):
            new_grid[lay + 1] = topp - new_thk[lay, :, :]
            topp = new_grid[lay + 1]

        return new_grid

    def force_min_conection(self):
        """
        Not Ready --  do not use
        Returns
        -------

        """
        nlay = self.grid.shape[0] - 1

        new_grid = np.zeros_like(self.grid)
        for layer_i in range(nlay):
            if layer_i == 0:
                top = np.copy(self.grid[layer_i, :, :])
                botm = np.copy(self.grid[layer_i + 1, :, :])
                new_grid[layer_i, :, :] = np.copy(top)
            else:
                top = botm + down_shift
                botm = np.copy(self.grid[layer_i + 1, :, :])

                # check for small thickness
                thickness = top - botm
                thickness[thickness < self.min_thk] = self.min_thk
                botm = top - thickness
                new_grid[layer_i, :, :] = np.copy(top)

            # compute down-sheft distant
            down_shift = self.find_disconc_area(top, botm)

        botm = botm + down_shift
        thickness = top - botm
        thickness[thickness < self.min_thk] = self.min_thk
        botm = top - thickness
        new_grid[layer_i + 1, :, :] = botm
        return new_grid

    def find_disconc_area(self, top, botm):

        # left - right direction
        tp_right = top[1:, :]
        tp_left = top[:-1, :]
        bt_right = botm[1:, :]
        bt_left = botm[:-1, :]
        shift_xx = np.zeros_like(botm)
        shift_yy = np.zeros_like(botm)
        # max up
        maskup = tp_right >= tp_left
        max_up = np.zeros_like(maskup, dtype=float)
        max_up[maskup] = tp_right[maskup]
        max_up[np.logical_not(maskup)] = tp_left[np.logical_not(maskup)]

        # min dn
        maskdn = bt_right <= bt_left
        min_dn = np.zeros_like(maskdn, dtype=float)
        min_dn[maskdn] = bt_right[maskdn]
        min_dn[np.logical_not(maskdn)] = bt_left[np.logical_not(maskdn)]
        # postive values means disconnection
        con_xx = (tp_right - bt_right) + (tp_left - bt_left) - (max_up - min_dn)
        if hasattr(self, 'ibound'):
            flg = np.diff(self.ibound, axis=0)
            loc = (np.logical_or(flg == 1, flg == -1))

        locA = tp_left <= tp_right
        locB = tp_left > tp_right
        nofix_zone = con_xx >= self.min_con
        nofix_zone[loc] = True
        con_xx[nofix_zone] = 0.0

        con_xx[np.logical_not(nofix_zone)] = con_xx[np.logical_not(nofix_zone)] - self.min_con
        # change right
        rcon = np.zeros_like(con_xx)
        rcon[locA] = con_xx[locA]
        lcon = np.zeros_like(con_xx)
        lcon[locB] = con_xx[locB]

        shift_xx[1:, :] = rcon
        shift_xx[:-1, :] = lcon

        # back - front direction
        tp_right = top[:, 1:]
        tp_left = top[:, :-1]
        bt_right = botm[:, 1:]
        bt_left = botm[:, :-1]
        locA = tp_left <= tp_right
        locB = tp_left > tp_right

        # max up
        maskup = tp_right >= tp_left
        max_up = np.zeros_like(maskup, dtype=float)
        max_up[maskup] = tp_right[maskup]
        max_up[np.logical_not(maskup)] = tp_left[np.logical_not(maskup)]

        # min dn
        maskdn = bt_right <= bt_left
        min_dn = np.zeros_like(maskdn, dtype=float)
        min_dn[maskdn] = bt_right[maskdn]
        min_dn[np.logical_not(maskdn)] = bt_left[np.logical_not(maskdn)]

        # postive values means disconnection
        con_yy = (tp_right - bt_right) + (tp_left - bt_left) - (max_up - min_dn)
        if hasattr(self, 'ibound'):
            flg = np.diff(self.ibound, axis=1)
            loc = (np.logical_or(flg == 1, flg == -1))

        nofix_zone = con_yy >= self.min_con
        nofix_zone[loc] = True
        con_yy[nofix_zone] = 0
        con_yy[np.logical_not(nofix_zone)] = con_yy[np.logical_not(nofix_zone)] - self.min_con

        # shift_yy[:,1:] = con_yy
        #################3
        rcon = np.zeros_like(con_yy)
        rcon[locA] = con_yy[locA]
        lcon = np.zeros_like(con_yy)
        lcon[locB] = con_yy[locB]

        shift_yy[:, 1:] = lcon
        shift_yy[:, :-1] = rcon
        ###########333333
        loc = np.abs(shift_yy) > np.abs(shift_xx)
        shift_xx[loc] = shift_yy[loc]
        return shift_xx

    def force_min_thikness(self):
        """
        force minimum thikcness ratio
        :return:
        """
        self.sat_thk = self.grid[0, :, :] - self.grid[-1, :, :]
        self.thickness = np.zeros_like(self.grid[:-1, :, :])
        for lay in range(self.thickness.shape[0]):
            ttop = self.grid[lay, :, :]
            bnt = self.grid[lay + 1, :, :]
            self.thickness[lay, :, :] = ttop - bnt

        # compute thickness ratio
        ratio = self.thickness / self.sat_thk
        ratio[ratio < self.min_thk_ratio] = self.min_thk_ratio
        ratio = ratio * self.ibound
        r_rmin = ratio - self.min_thk_ratio
        rr = r_rmin / np.sum(r_rmin, axis=0)
        sum_ratio = np.sum(ratio, axis=0)
        e = sum_ratio - 1.0
        ratio = ratio - e * rr
        self.thickness = ratio * self.sat_thk

        # update elevations
        ttop = np.copy(self.grid[0, :, :])
        new_grid = np.zeros_like(self.grid)
        new_grid[0, :, :] = ttop
        for lay in np.arange(self.thickness.shape[0]):
            new_grid[lay + 1, :, :] = ttop - self.thickness[lay, :, :]
            ttop = np.copy(new_grid[lay + 1, :, :])

        return new_grid


if __name__ == "__main__":
    mf_file = r"D:\Workspace\projects\RussianRiver\RR_GSFLOW_GIT\RR_GSFLOW\GSFLOW\archive\20220706_01\windows\rr_tr.nam"
    mf = flopy.modflow.Modflow.load(mf_file, load_only=['DIS', 'BAS6'])



    dis = main(mf, min_thk_ratio=0.1, min_con=30.0,
                    min_thk=50, max_slope=0.1, buffer_sigma=3,
                    value_sigma=3.0, apply_min_thickness = False,
                    fix_layer_connections = False,
                    smooth_grid = False)
