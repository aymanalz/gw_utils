import os, sys
sys.path.insert(0,r"D:\Models\Yucaipa\codes")
sys.path.insert(0,r"D:\Models\Yucaipa\codes\flopy")
import flopy
import pandas as pd

from gw_utils import *

columns = ['x', 'y', 'value', 'va2']
data = [[477090.000160, 3768509.999800, 1.000000, -0.003420],
[478890.000240, 3767309.999800, 1.000000, 0.007340],
[479490.000270, 3766109.999700, 1.000000, -0.009710]]



df = pd.DataFrame(data, columns = columns)
epsg = 26911
point_to_shapefile(df, xy_field = ['x', 'y'], epsg = epsg)



pass

