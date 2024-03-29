import os, sys
import numpy as np
try:
    from epsg_ident import EpsgIdent
    import pyproj
    import shapefile
except:
    print ("Error importing some GIS packages....")

import flopy
from flopy.utils.geometry import Polygon, LineString, Point
from flopy.export.shapefile_utils import recarray2shp, shp2recarray

def get_epsg(shp_file):
    """

    Parameters
    ----------
    shp_file

    Returns
    -------

    """
    file_parts = os.path.splitext(shp_file)
    prj_file =file_parts[0] + ".prj"
    ident = EpsgIdent()
    ident.read_prj_from_file(prj_file)
    return ident.get_epsg()


def getWKT_PRJ(epsg_code):
    import urllib
    with urllib.request.urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg_code)) as url:
        wkt = url.read()
        # I'm guessing this would output the html source code ?
        wkt = wkt.decode("utf-8")
        print(wkt)
    remove_spaces = wkt.replace(" ", "")
    output = remove_spaces.replace("\n", "")
    return output

def point_to_shapefile(df, shpfile = 'temp_shp.shp', xy_field = [], epsg = None):
    """
    Convert a dataframe to shapefile.
    Returns
    -------
    """
    geoms = []
    for irec, rec in df.iterrows():
        x_ = rec[xy_field[0]]
        y_ = rec[xy_field[1]]
        geoms.append(Point(x_, y_, 0))
    att_table = df.to_records()
    recarray2shp(att_table, geoms, shpfile, epsg=epsg)
    pass

def array_to_ascii_raster(mf, array, raster_file = 'rast_ascii.txt', ibound = None):
    """

    Parameters
    ----------
    mf : flopy object with grid that is georeferenced and the basic pacakge
    array

    Returns
    -------

    """
    header = []
    header.append("ncols         {}\n".format(mf.modelgrid.ncol))
    header.append("nrows         {}\n".format(mf.modelgrid.nrow))
    header.append("xllcorner     {}\n".format(mf.modelgrid.xoffset))
    header.append("yllcorner     {}\n".format(mf.modelgrid.yoffset))
    header.append("cellsize      {}\n".format(mf.modelgrid.delc[0]))
    header.append("NODATA_value  -9999\n")

    if not(ibound is None):
        array[ibound==0] = -9999

    fidw = open(raster_file, 'w')
    fidw.writelines(header)
    #np.savetxt(raster_file, array)
    for line in array:
        line2 = ' '.join(map(str, line))
        fidw.write(line2)
        fidw.write("\n")
    fidw.close()
