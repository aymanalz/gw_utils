
import sys
sys.path.insert(0,r"D:\codes")
print(sys.path)

import gw_utils
import datetime
from gw_utils import plot_heads_only_obs

if 0:
    for i in range(12):
        mfname = r"C:\work\Slave\use_average_HFB6\Root\M_{}\yucaipa.nam".format(i)
        start_date = datetime.datetime(1947, 1, 1)
        end_date = datetime.datetime(2014, 12, 31)
        gw_utils.plot_all_heads(mfname, start_date, end_date, x_limit=[1947, 2015], pdf_file='mmm{}.pdf'.format(i),
                                obs_name_file='Obs_well_names.csv')




#exit()
mfname = r"C:\work\Slave\use_average_HFB6\Final_Final5\yucaipa.nam"
start_date = datetime.datetime(1947, 1, 1)
end_date = datetime.datetime(2014, 12, 31)

if 1:
    gw_utils.plot_all_heads(mfname, start_date, end_date, x_limit = [1970, 2014], pdf_file = 'Final_june5.pdf',
                            obs_name_file = 'obs_wells_naming.csv')
if 0:
    plot_heads_only_obs.plot_all_heads(mfname, start_date, end_date, x_limit = [1970, 2015], pdf_file = 'wh7.pdf',
                            obs_name_file = 'Obs_well_names.csv')