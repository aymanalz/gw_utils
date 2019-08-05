import os
import sys
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mf_output_gis import *
# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join('..', '..'))
    sys.path.append(fpth)
    import flopy

from flopy.utils.geometry import Polygon, LineString, Point
from flopy.export.shapefile_utils import recarray2shp, shp2recarray
from flopy.utils.modpathfile import PathlineFile, EndpointFile
from flopy.utils import geometry
from flopy.utils.reference import epsgRef
import geopandas
ep = epsgRef()
ep.reset()

mfname = r"D:\Workspace\projects\RussianRiver\RR_GSFLOW_MODEL\RR_GSFLOW\MODFLOW\steady_state\rr_ss.nam"
hrushp = r"D:\Workspace\projects\RussianRiver\modsim\hru_param_tzones.shp"
hru_df = geopandas.read_file(hrushp)
m = flopy.modflow.Modflow.load(mfname, load_only=['DIS', 'BAS6'])
set_spatial_ref_from_hru_shpfile(m, hrushp)
hds = flopy.utils.HeadFile(r"D:\Workspace\projects\RussianRiver\RR_GSFLOW_MODEL\RR_GSFLOW\MODFLOW\steady_state\rr_ss"
                          r".hds")
hds.mg = m.modelgrid
hds.model = m
#hds.to_shapefile('xxx1.shp', totim= 1.0)
hds = hds.get_data(totim = 1.0)
get_vertices = m.modelgrid.get_cell_vertices # function to get the referenced vertices for a model cell
geoms = []
hd1 = []
hd2 = []
hd3 = []
wl = []
wldepth = []
ibound = m.bas6.ibound.array.copy().astype(float)
ibound2d = np.sum(ibound, axis = 0)

ibound[ibound==0] = np.nan
all_wls = []
all_depths = []
for irow, row in hru_df.iterrows():
    print(100*(irow * 1.0)/float(len(hru_df)))
    rr = row['HRU_ROW']-1
    cc = row['HRU_COL']-1
    geoms.append(Polygon(get_vertices(rr, cc)))
    m.bas6.ibound.array
    hd1 = hds[0, rr, cc] * ibound[0,rr, cc]
    hd2 = hds[1, rr, cc] * ibound[1,rr, cc]
    hd3 = hds[2, rr, cc] * ibound[2,rr, cc]
    wl = np.nanmean([hd1, hd2, hd3])
    wldepth = m.dis.top.array[rr, cc] - wl
    if ibound2d[rr, cc] == 0:
        wl = -999
        wldepth = -999
    all_depths.append(wldepth)
    all_wls.append(wl)

df = pd.DataFrame()
df['wl'] = all_wls
df['depths'] = all_depths

xxx = 1