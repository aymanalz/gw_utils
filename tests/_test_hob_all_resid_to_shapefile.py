import sys, os
sys.path.insert(0,r"D:\codes")
import flopy
from gw_utils import *



mfname = r"D:\Yucaipa_work\tran_mf3d_best10_18 - Copy24\yucaipa.nam"
mf = flopy.modflow.Modflow.load(mfname, model_ws = os.path.dirname(mfname) ,   load_only=['DIS', 'BAS6', 'MNW2'])

sr = "xul:473715; yul:3771885; rotation:0; proj4_str:+proj=utm +zone=11 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ;" \
     " units:meters; lenuni:1; length_multiplier:0.3048 ;start_datetime:1-1-1970"

xoff = 473715
yoff = 3751785
epsg = 26911

# if the model units are different than projection units, then
# we have to change delc, delr
mf.dis.delc = mf.dis.delc.array * 0.3048
mf.dis.delr = mf.dis.delr.array * 0.3048
mf.modelgrid.set_coord_info(xoff = xoff, yoff = yoff, epsg = epsg)

hob_resid_to_shapefile_all(mf)
#hob_output_to_shp(mf)
xxx = 1