import os, sys
import flopy
from selection import Zone_selector
import numpy as np
import matplotlib.pyplot as plt


def findrowcolumn(pt, xedge, yedge):

    # make sure xedge and yedge are numpy arrays
    if not isinstance(xedge, np.ndarray):
        xedge = np.array(xedge)
    if not isinstance(yedge, np.ndarray):
        yedge = np.array(yedge)

    # find column
    jcol = -100
    for jdx, xmf in enumerate(xedge):
        if xmf > pt[0]:
            jcol = jdx - 1
            break

    # find row
    irow = -100
    for jdx, ymf in enumerate(yedge):
        if ymf < pt[1]:
            irow = jdx - 1
            break
    return irow, jcol

def cell_value_points(pts, xedge, yedge, vdata):


    # make sure xedge and yedge are numpy arrays
    if not isinstance(xedge, np.ndarray):
        xedge = np.array(xedge)
    if not isinstance(yedge, np.ndarray):
        yedge = np.array(yedge)
    if not isinstance(vdata, np.ndarray):
        vdata = np.array(vdata)

    vcell = []
    for  pp in pts:
        xt, yt = pp[0]
        # find the modflow cell containing point
        irow, jcol = findrowcolumn((xt, yt), xedge, yedge)
        if irow >= 0 and jcol >= 0:
            if np.isnan(vdata[irow, jcol]):
                vcell.append(np.nan)
            else:
                v = np.asarray(vdata[irow, jcol])
                vcell.append(v)

    return np.array(vcell)


fn =  r"D:\Workspace\projects\RussianRiver\RR_GSFLOW_GIT\RR_GSFLOW\GSFLOW\archive\current_version\GSFLOW\worker_dir_ies\gsflow_model_updated\windows\rr_tr.nam"
mf = flopy.modflow.Modflow.load(fn, model_ws= os.path.dirname(fn), load_only=['DIS', 'BAS6', 'UPW'])
ib = mf.bas6.ibound.array
Zmax = np.max(mf.dis.top.array[ib[0,:,:]>0])
Zmin = np.min(mf.dis.botm.array[ib>0])
ib = mf.bas6.ibound.array.sum(axis = 0)
ib = ib/ib
thk = mf.modelgrid.thick.sum(axis = 0)
bk = mf.dis.top.array * ib
bk = thk * ib

if 0:
    ib = mf.bas6.ibound.array * 1.0
    ib[ib>0] = 1
    ib[ib==0] = np.NAN
    mf.dis.top  = mf.dis.top.array * ib[0,:,:]
    mf.dis.botm = mf.dis.botm.array * ib

mz = Zone_selector(bk)
y = -1*np.array(mz.ploygy) * 300 + 150
x = np.array(mz.polygx) * 300 + 150
rr_cc_list = []
for xy in zip(x,y):
    rr_cc = mf.modelgrid.intersect(xy[0], xy[1])
    rr_cc_list.append(rr_cc)
grid3D = np.zeros((mf.nlay+1,mf.nrow, mf.ncol ))
grid3D[0,:,:] = mf.dis.top.array
grid3D[1:,:,:] = mf.dis.botm.array
ibb = mf.bas6.ibound.array
lines = {}
for k in range(mf.nlay + 1):
    line = []

    for rr_cc in rr_cc_list:
        if k > 0:
            ggg = ibb[k-1, rr_cc[0], rr_cc[1]]
        else:
            ggg = 1
        val = grid3D[k, rr_cc[0], rr_cc[1]]
        if ggg == 1:
            line.append(val)
        else:
            line.append(np.NAN)
    #plt.plot(line)

from flopy.plot.plotutil import UnstructuredPlotUtilities
pts = list(zip(x,y))
pts = np.array(pts)
xpts = UnstructuredPlotUtilities.line_intersect_grid(pts, mf.modelgrid.xyzvertices[0],
                                         mf.modelgrid.xyzvertices[1])
xpts = list(xpts.values())
zpts = []
ibv = []
for k in range(0, mf.nlay+1):
    yvertix = mf.modelgrid.xyzvertices[1][:,0]
    xvertix = mf.modelgrid.xyzvertices[0][0,:]

    zpts.append(cell_value_points(xpts, xvertix,
                                           yvertix,
                                           grid3D[k, :, :]))
    ibv.append(cell_value_points(xpts, xvertix,
                                           yvertix,
                                           mf.bas6.ibound.array[2, :, :]))
    xx = 1


zpts = np.array(zpts)
plt.plot(zpts.T)
plt.show()


if 0:
xsect = flopy.plot.PlotCrossSection(model=mf, line= {'line':list(zip(x,y))})
plt.plot(xsect.zpts.T)
patches = xsect.plot_ibound()
linecollection = xsect.plot_grid()
plt.ylim([Zmin, Zmax])
plt.show()

plt.figure()
ib = 1.0 * mf.bas6.ibound.array
ib[ib>0] = 1
ib = ib.sum(axis = 0)

ib[ib==0] = np.NAN
plt.imshow(ib)
xx = 1