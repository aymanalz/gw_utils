from gis_utils import *


epsg_code = get_epsg(r"D:\Workspace\projects\RussianRiver\modsim\hru_param_tzones.prj")
print(epsg_code)
output = getWKT_PRJ(epsg_code)
print(output)

zz = 1