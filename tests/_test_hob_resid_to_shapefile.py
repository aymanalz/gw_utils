import sys
sys.path.insert(0,r"D:\Models\Yucaipa\codes")
sys.path.insert(0,r"D:\Models\Yucaipa\codes\flopy")
import flopy
from gw_utils import *



mfname = r"D:\Models\Yucaipa\Yuc\pilot_KF_and_pest\model\model\tran_mf3d\yucaipa.nam"
mf = flopy.modflow.Modflow.load(mfname, load_only=['DIS', 'BAS6', 'MNW2'])

sr = "xul:473715; yul:3771885; rotation:0; proj4_str:+proj=utm +zone=11 +ellps=GRS80 +datum=NAD83 +units=m +no_defs ;" \
     " units:meters; lenuni:1; length_multiplier:0.3048 ;start_datetime:1-1-1970"

xoff = 473715
yoff = 3751785
epsg = 26911

mf.dis.delc = mf.dis.delc.array * 0.3048
mf.dis.delr = mf.dis.delr.array * 0.3048
mf.mnw2.export(r'D:\Models\Yucaipa\codes\gw_utils\tests\mnw2.shp')

#hob_output_to_shp(mfname, epsg = epsg)
xxx = 1