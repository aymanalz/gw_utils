
import sys
sys.path.insert(0,r"D:\Models\Yucaipa\codes")

import gw_utils
import datetime


mfname = r"D:\Models\Yucaipa\Yuc\pilot_KF_and_pest\model\model\tran_mf3d\yucaipa.nam"
start_date = datetime.datetime(1970, 1, 1)
end_date = datetime.datetime(2014, 12, 31)

gw_utils.plot_all_heads(mfname, start_date, end_date)