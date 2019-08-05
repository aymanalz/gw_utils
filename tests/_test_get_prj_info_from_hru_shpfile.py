import shapefile



fn = r"D:\Workspace\projects\RussianRiver\modsim\hru_param_tzones.shp"
shape = shapefile.Reader(fn)
feature = shape.shapeRecords()[0]
