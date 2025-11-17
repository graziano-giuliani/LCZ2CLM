#!/bin/bash
#
gdal_rasterize -a FID -sql 'select FID,* from World_Countries_Generalized' \
	-of netCDF -tr 1000.0 1000.0 -ot Uint16 \
	World_Countries_Generalized.shp countries.nc
gdalwarp -r nearest -t_srs EPSG:4326 \
	-te -170.000777512594  80.0380950691605 \
	     180.000823255411 -60.0945974464225 \
	     -ts 38962 15599 countries.nc countries_remap.nc

echo Done
