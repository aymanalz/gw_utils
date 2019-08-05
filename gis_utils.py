import os, sys
from epsg_ident import EpsgIdent
import pyproj
import shapefile


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
    #wkt = urllib.urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg_code))
    with urllib.request.urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg_code)) as url:
        wkt = url.read()
        # I'm guessing this would output the html source code ?
        wkt = wkt.decode("utf-8")
        print(wkt)
    remove_spaces = wkt.replace(" ", "")
    output = remove_spaces.replace("\n", "")
    return output
