
import sys
sys.path.insert(0,r"D:\codes")
print(sys.path)

import gw_utils
import datetime
from gw_utils import plot_heads_only_obs


mfname = r"C:\work\Slave\use_average_HFB6\F6\yucaipa.nam"
start_date = datetime.datetime(1947, 1, 1)
end_date = datetime.datetime(2014, 12, 31)

if 0:
    gw_utils.plot_all_heads(mfname, start_date, end_date, x_limit = [1970, 2015], pdf_file = 'Final3NewSolver.pdf',
                            obs_name_file = 'Obs_well_names.csv')
if 1:
    plot_heads_only_obs.plot_all_heads(mfname, start_date, end_date, x_limit = [1970, 2015], pdf_file = 'wh6.pdf',
                            obs_name_file = 'Obs_well_names.csv')