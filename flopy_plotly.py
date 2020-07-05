import sys
sys.path.insert(0,r"D:\codes")
print(sys.path)
import numpy as np
import gw_utils
import flopy
# D:\Yucaipa_work\Yuc4
#C:\work\Slave\model_1_no_run\model\tran_mf3d\
hfile = r"D:\Yucaipa_work\Yuc4\yucaipa.hds"
hds = flopy.utils.HeadFile(hfile)
hh = hds.get_data(totim = 24837)
hh[hh<0] = np.nan
import plotly.graph_objects as go
levels =np.arange(1700,3000, 100 )

fig = go.Figure(data = go.Contour(z= hh[0,:,:], contours=dict(
            start=1700,
            end=2500,
            size=50,
        )))
fig.update_yaxes(autorange="reversed")
fig.show()


xx = 1

