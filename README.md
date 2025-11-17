# LCZ2CLM

Python script to create a urban properties input file for Community Land Model
[CLM, Oleson et al. NCAR/TN-480+STR] from Local Climate Zone data
[LCZ, https://essd.copernicus.org/articles/14/3835/2022/].

Required data:

    - LCZ tif file [https://zenodo.org/records/8419340]
    - Original CLM45 surface data file. The data can be created by using the
      THESISUrbanPropertiesTool
      [https://github.com/NCAR/THESISUrbanPropertiesTool]
      and the input data at [https://zenodo.org/records/15501878]
    - World countries Generalized (shapefile and csv data)
      [https://hub.arcgis.com/datasets/esri::world-countries-generalized]

The user must first run the `make_countries.sh` script to create the remapped
raster data with country borders.

The python script requires a lot of memory to be run. It is not optimized.
