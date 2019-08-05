import os, sys
import flopy
from mf_output_gis import *
mfname = r"D:\Workspace\projects\RussianRiver\RR_GSFLOW_MODEL\RR_GSFLOW\MODFLOW\steady_state\rr_ss.nam"
hrushp = r"D:\Workspace\projects\RussianRiver\modsim\hru_param_tzones.shp"

mf = flopy.modflow.Modflow.load(mfname, load_only=['DIS', 'BAS6'])

set_spatial_ref_from_hru_shpfile(mf, hrushp)

pass
