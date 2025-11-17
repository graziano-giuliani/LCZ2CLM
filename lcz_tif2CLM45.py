#!/usr/bin/env python3

import numpy as np
import xarray as xr
import os
import sys
import dask
import csv
from tqdm import trange
from scipy.interpolate import griddata

wt_new = 0.9
wt_old = 1.0 - wt_new
resoultion_divider = 10
water = 17

lcz_tif_file = 'lcz_v3.tif'
clm_urban_file = 'lcz_mksrf_urban.nc'
countries_file = 'countries_remap.nc'
urbparam_file = 'URBPARM_LCZ.TBL'

state_old_region = {
   'AF0' : 24, 'AL0' : 11, 'DZ0' : 16, 'AS0' : 21, 'AD0' : 33, 'AO0' :  4,
   'AI0' :  7, 'AQ0' :  0, 'AG0' :  7, 'AR0' : 30, 'AM0' :  5, 'AW0' :  7,
   'AU0' :  2, 'AT0' : 33, 'AZ0' :  5, 'PT1' : 28, 'BS0' :  7, 'BH0' : 15,
   'BD0' : 24, 'BB0' :  7, 'BY0' : 22, 'BE0' : 33, 'BZ0' : 14, 'BJ0' : 32,
   'BM0' :  7, 'BT0' : 24, 'BO0' : 31, 'BQ0' :  7, 'BA0' : 11, 'BW0' :  9,
   'BV0' : 19, 'BR0' :  3, 'IO0' : 23, 'VG0' :  7, 'BN0' : 27, 'BG0' : 11,
   'BF0' : 32, 'BI0' :  9, 'CV0' : 16, 'KH0' : 27, 'CM0' :  4, 'CA0' :  6,
   'ES0' : 28, 'KY0' :  7, 'CF0' : 32, 'TD0' :  4, 'CL0' : 30, 'CN0' :  8,
   'CX0' :  2, 'CC0' :  2, 'CO0' : 31, 'KM0' :  9, 'CG0' :  4, 'CD0' :  4,
   'CK0' : 21, 'CR0' : 14, 'CI0' : 32, 'HR0' : 11, 'CU0' :  7, 'CW0' :  7,
   'CY0' : 15, 'CZ0' : 33, 'DK0' : 19, 'DJ0' : 15, 'DM0' :  7, 'DO0' :  7,
   'EC0' : 31, 'EG0' : 16, 'SV0' : 14, 'GQ0' :  4, 'ER0' : 16, 'EE0' : 11,
   'SZ0' : 23, 'ET0' :  9, 'FK0' : 30, 'FO0' : 19, 'FJ0' : 21, 'FI0' : 19,
   'FR0' : 33, 'GF0' : 31, 'PF0' : 21, 'TF0' : 21, 'GA0' :  4, 'GM0' : 32,
   'GE0' :  5, 'DE0' : 33, 'GH0' : 32, 'GI0' : 28, 'TF1' : 23, 'GR0' : 28,
   'GL0' : 12, 'GD0' :  7, 'GP0' :  7, 'GU0' : 21, 'GT0' : 14, 'GG0' : 19,
   'GN0' : 32, 'GW0' : 32, 'GY0' : 31, 'HT0' :  7, 'HM0' :  2, 'HN0' : 14,
   'HU0' : 11, 'IS0' : 19, 'IN0' : 13, 'ID0' : 27, 'IR0' : 15, 'IQ0' : 15,
   'IE0' : 19, 'IM0' : 19, 'IL0' : 15, 'IT0' : 28, 'JM0' :  7, 'JP0' : 10,
   'JE0' : 19, 'JO0' : 15, 'TF2' : 23, 'KZ0' :  5, 'KE0' :  9, 'KI0' : 21,
   'KW0' : 15, 'KG0' :  5, 'LA0' : 27, 'LV0' : 11, 'LB0' : 15, 'LS0' : 23,
   'LR0' : 32, 'LY0' : 16, 'LI0' : 33, 'LT0' : 11, 'LU0' : 33, 'MG0' :  9,
   'PT0' : 28, 'MW0' :  9, 'MY0' : 27, 'MV0' : 24, 'ML0' : 32, 'MT0' : 28,
   'MH0' : 21, 'MQ0' :  7, 'MR0' : 16, 'MU0' : 23, 'YT0' : 23, 'MX0' : 14,
   'FM0' : 21, 'MD0' : 11, 'MC0' : 33, 'MN0' :  5, 'ME0' : 11, 'MS0' :  7,
   'MA0' : 16, 'MZ0' :  9, 'MM0' : 27, 'NA0' : 23, 'NR0' : 21, 'NP0' : 24,
   'NL0' : 33, 'NC0' : 21, 'NZ0' : 21, 'NI0' : 14, 'NE0' : 32, 'NG0' : 32,
   'NU0' : 21, 'NF0' :  2, 'KP0' : 10, 'MK0' : 11, 'MP0' : 21, 'NO0' : 19,
   'OM0' : 15, 'PK0' : 24, 'PW0' : 21, 'PS0' : 15, 'PA0' : 14, 'PG0' : 21,
   'PY0' : 30, 'PE0' : 30, 'PH0' : 27, 'PN0' : 21, 'PL0' : 11, 'PT2' : 28,
   'PR0' : 26, 'QA0' : 15, 'RE0' : 23, 'RO0' : 11, 'RU0' : 22, 'RW0' :  9,
   'BQ1' :  7, 'BL0' : 33, 'BQ2' :  7, 'SH0' : 16, 'KN0' :  7, 'LC0' :  7,
   'MF0' :  7, 'PM0' :  6, 'VC0' :  7, 'WS0' : 21, 'SM0' : 28, 'ST0' :  4,
   'SA0' : 15, 'SN0' : 32, 'RS0' : 11, 'SC0' : 24, 'SL0' : 32, 'SG0' : 27,
   'SX0' :  7, 'SK0' : 33, 'SI0' : 28, 'SB0' : 21, 'SO0' :  9, 'ZA0' : 23,
   'GS0' : 30, 'KR0' : 10, 'SS0' : 16, 'ES1' : 28, 'LK0' : 24, 'SD0' : 16,
   'SR0' : 31, 'SJ0' : 19, 'SE0' : 19, 'CH0' : 33, 'SY0' : 15, 'TJ0' :  5,
   'TZ0' :  9, 'TH0' : 27, 'TL0' : 27, 'TG0' : 32, 'TK0' : 21, 'TO0' : 21,
   'TT0' :  7, 'TN0' : 16, 'TR0' : 15, 'TM0' :  5, 'TC0' :  7, 'TV0' : 21,
   'UG0' :  9, 'UA0' : 11, 'AE0' : 15, 'GB0' : 19, 'US0' : 18, 'UM0' : 21,
   'UY0' : 30, 'VI0' : 21, 'UZ0' :  5, 'VU0' : 21, 'VA0' : 28, 'VE0' : 31,
   'VN0' : 27, 'WF0' : 21, 'YE0' : 15, 'ZM0' :  9, 'ZW0' : 23,
   }

class_old_class = {
        'Comp High-Rise'     : 1,
        'Comp Mid-Rise'      : 2,
        'Comp Low-Rise'      : 2,
        'Op H-Rise'          : 2,
        'Op M-Rise'          : 2,
        'Op L-Rise'          : 3,
        'Lightweight L-Rise' : 3,
        'Large L-Rise'       : 3,
        'Sparsely Built'     : 3,
        'Heavy Indus'        : 3,
        'Asphalt'            : 3
        }

ncountries = len(state_old_region)
nclass = len(class_old_class)

def getval_old_new(x,y):
    yy = np.full(np.shape(y), -999.0)
    i = 0
    for stat in state_old_region:
        j = 0
        for cla in class_old_class:
            yy[Ellipsis,i,j] = (wt_new * y[Ellipsis,i,j] +
                       wt_old * x[Ellipsis,
                                  state_old_region[stat]-1,
                                  class_old_class[cla]-1])
            j = j + 1
        i = i + 1
    avg = np.mean(yy[yy > -998.0])
    yy[:] = np.where(yy < -998.0, avg, yy)
    return yy

def getval_new(x,y):
    yy = np.full(np.shape(y), -999.0)
    i = 0
    for stat in state_old_region:
        j = 0
        for cla in class_old_class:
            yy[Ellipsis,i,j] = x[Ellipsis,
                                 state_old_region[stat]-1,
                                 class_old_class[cla]-1]
            j = j + 1
        i = i + 1
    avg = np.mean(yy[yy > -998.0])
    yy[:] = np.where(yy < -998.0, avg, yy)
    return yy

def _griddata(arr, xi, method: str):
    ar1d = arr.ravel()
    valid = np.isfinite(ar1d)
    if valid.all():
        return arr
    return griddata(
        points=tuple(x[valid] for x in xi),
        values=ar1d[valid],
        xi=xi,
        method=method,
        fill_value=np.nan,
    ).reshape(arr.shape)

def interpolate_na(da, dim, method="nearest",
                   use_coordinates=True, keep_attrs=True):
    # Create points only once.
    if use_coordinates:
        coords = [da.coords[d] for d in dim]
    else:
        coords = [np.arange(da.sizes[d]) for d in dim]

    xi = tuple(x.ravel() for x in np.meshgrid(*coords, indexing="ij"))
    arr = xr.apply_ufunc(
        _griddata,
        da,
        input_core_dims=[dim],
        output_core_dims=[dim],
        output_dtypes=[da.dtype],
        dask="parallelized",
        vectorize=True,
        keep_attrs=keep_attrs,
        kwargs={"xi": xi, "method": method},
    ).transpose(*da.dims)
    return arr

def parse_URBPARM_LCZ_TBL(fname):
    keyparam = {
        'Number of urban categories' : dict( name = 'NCAT',
                                             dimension = 1,
                                             ktype = np.int8 ),
        'ZR' : dict( name = 'ROOF_LEVEL', dimension = 11,
                     ktype = np.float32 ),
        'SIGMA_ZED' : dict( name = 'ROOF_LEVEL_STD', dimension = 11,
                            units = 'm', ktype = np.float32 ),
        'ROOF_WIDTH' : dict( name = 'ROOF_WIDTH', dimension = 11,
                             units = 'm', ktype = np.float32 ),
        'ROAD_WIDTH' : dict( name = 'ROAD_WIDTH', dimension = 11,
                             units = 'm', ktype = np.float32 ),
        'AH' : dict( name = 'ANTHROPOGENIC_HEAT', dimension = 11,
                     units = 'W m-2', ktype = np.float32 ),
        'ALH' : dict( name = 'ANTHROPOGENIC_LATENT_HEAT', dimension = 11,
                      units = 'W m-2', ktype = np.float32 ),
        'AKANDA_URBAN' : dict( name = 'KANDA_COEFFICIENT', dimension = 11,
                               units = '1', ktype = np.float32 ),
        'DDZR' : dict( name = 'ROOF_THICKNESS', dimension = 4,
                       units = 'm', ktype = np.float32 ),
        'DDZB' : dict( name = 'WALL_THICKNESS', dimension = 4,
                       units = 'm', ktype = np.float32 ),
        'DDZG' : dict( name = 'GROUND_THICKNESS', dimension = 4,
                       units = 'm', ktype = np.float32 ),
        'BOUNDR' : dict( name = 'LOWER_BOUNDARY_ROOF_LAYER', dimension = 1,
                         ktype = np.int8 ),
        'BOUNDB' : dict( name = 'LOWER_BOUNDARY_WALL_LAYER', dimension = 1,
                         ktype = np.int8 ),
        'BOUNDG' : dict( name = 'LOWER_BOUNDARY_GROUND_LAYER', dimension = 1,
                         ktype = np.int8 ),
        'CH_SCHEME' : dict( name = 'WALL_AND_ROAD_SCHEME', dimension = 1,
                            ktype = np.int8 ),
        'TS_SCHEME' : dict( name = 'SURFACE_LAYER_TRANSMISSION_SCHEME',
                            dimension = 1, ktype = np.int8 ),
        'AHOPTION' : dict( name = 'ANTHROPOGENIC_HEAT_SCHEME',
                           dimension = 1, ktype = np.int8 ),
        'AHDIUPRF' : dict( name = 'ANTHROPOGENIC_HEAT_DIURNAL_PROFILE',
                           dimension = 24, ktype = np.float32 ),
        'ALHOPTION' : dict( name = 'ANTHROPOGENIC_LATENT_HEAT_SCHEME',
                            dimension = 1, ktype = np.int8 ),
        'ALHSEASON' : dict( name = 'ANTHROPOGENIC_HEAT_SEASONAL_PROFILE',
                            dimension = 4, ktype = np.float32 ),
        'ALHDIUPRF' : dict( name = 'ANTHROPOGENIC_LATENT_HEAT_DIURNAL_PROFILE',
                            dimension = 48, ktype = np.float32 ),
        'OASIS' : dict( name = 'OASIS_EFFECT_FACTOR',
                        dimension = 1, ktype = np.float32 ),
        'IMP_SCHEME' : dict( name = 'EVAPORATION_SCHEME',
                             dimension = 1, ktype = np.int8 ),
        'PORIMP' : dict( name = 'GROUND_POROSITY',
                         dimension = 3, ktype = np.float32 ),
        'DENGIMP' : dict( name = 'GROUND_WATER_HOLDING_DEPTH',
                          dimension = 3, units = 'm', ktype = np.float32 ),
        'IRI_SCHEME' : dict( name = 'IRRIGATION_SCHEME', dimension = 1,
                             ktype = np.int8 ),
        'GROPTION' : dict( name = 'GREEN_ROOF_SCHEME', dimension = 1,
                           ktype = np.int8 ),
        'FGR' : dict( name = 'GREEN_ROOF_FRACTION', dimension = 1,
                      ktype = np.float32 ),
        'DZGR' : dict( name = 'GREEN_ROOF_LAYER_THICKNESS', dimension = 4,
                       units = 'm', ktype = np.float32 ),
        'FRC_URB' : dict( name = 'NON_VEGETATED_FRACTION', dimension = 11,
                          units = '1', ktype = np.float32 ),
        'CAPR' : dict( name = 'ROOT_HEAT_CAPACITY', dimension = 11,
                       units = 'J m-3 K-1', ktype = np.float32 ),
        'CAPB' : dict( name = 'WALL_HEAT_CAPACITY', dimension = 11,
                       units = 'J m-3 K-1', ktype = np.float32 ),
        'CAPG' : dict( name = 'GROUND_HEAT_CAPACITY', dimension = 11,
                       units = 'J m-3 K-1', ktype = np.float32 ),
        'AKSR' : dict( name = 'ROOF_THERMAL_CONDUCTIVITY', dimension = 11,
                       units = 'J m-3 K-1', ktype = np.float32 ),
        'AKSB' : dict( name = 'WALL_THERMAL_CONDUCTIVITY', dimension = 11,
                       units = 'J m-3 K-1', ktype = np.float32 ),
        'AKSG' : dict( name = 'GROUND_THERMAL_CONDUCTIVITY', dimension = 11,
                       units = 'J m-3 K-1', ktype = np.float32 ),
        'ALBR' : dict( name = 'ROOF_SURFACE_ALBEDO', dimension = 11,
                       units = '1', ktype = np.float32 ),
        'ALBB' : dict( name = 'WALL_SURFACE_ALBEDO', dimension = 11,
                       units = '1', ktype = np.float32 ),
        'ALBG' : dict( name = 'GROUND_SURFACE_ALBEDO', dimension = 11,
                       units = '1', ktype = np.float32 ),
        'EPSR' : dict( name = 'ROOF_SURFACE_EMISSIVITY', dimension = 11,
                       units = '1', ktype = np.float32 ),
        'EPSB' : dict( name = 'WALL_SURFACE_EMISSIVITY', dimension = 11,
                       units = '1', ktype = np.float32 ),
        'EPSG' : dict( name = 'GROUND_SURFACE_EMISSIVITY', dimension = 11,
                       units = '1', ktype = np.float32 ),
        'ZOR' : dict( name = 'ROOF_MOMENTUM_ROUGHNESS_LENGHT', dimension = 11,
                      units = '1', ktype = np.float32 ),
        'ZOB' : dict( name = 'WALL_MOMENTUM_ROUGHNESS_LENGHT', dimension = 11,
                      units = '1', ktype = np.float32 ),
        'ZOG' : dict( name = 'GROUND_MOMENTUM_ROUGHNESS_LENGHT', dimension = 11,
                      units = '1', ktype = np.float32 ),
        'TRLEND' : dict( name = 'ROOF_TEMPERATURE_LOWER_BOUNDARY_CONDITION',
                         dimension = 11, units = 'K', ktype = np.float32 ),
        'TBLEND' : dict( name = 'WALL_TEMPERATURE_LOWER_BOUNDARY_CONDITION',
                         dimension = 11, units = 'K', ktype = np.float32 ),
        'TGLEND' : dict( name = 'GROUND_TEMPERATURE_LOWER_BOUNDARY_CONDITION',
                         dimension = 11, units = 'K', ktype = np.float32 ),
        'COP' : dict( name = 'AC_SYSTEM_PERFORMANCE_COEFFICIENT',
                      dimension = 11, units = '1', ktype = np.float32 ),
        'BLDAC_FRC' : dict( name = 'AC_SYSTEM_BUILDING_FRACTION',
                            dimension = 11, units = '1', ktype = np.float32 ),
        'COOLED_FRC' : dict( name = 'AC_SYSTEM_COOLED_FLOOR_FRACTION',
                             dimension = 11, units = '1', ktype = np.float32 ),
        'PWIN' : dict( name = 'WINDOWS_WALL_COVERAGE_FRACTION',
                       dimension = 11, units = '1', ktype = np.float32 ),
        'BETA' : dict( name = 'HEAT_EXCHANGER_THERMAL_EFFICIENCY',
                       dimension = 11, units = '1', ktype = np.float32 ),
        'SW_COND' : dict( name = 'AIR_CONDITIONING_SWITCH', dimension = 11,
                          units = '1', ktype = np.int8 ),
        'TIME_ON' : dict( name = 'AC_SYSTEM_LOCAL_TIME_START', dimension = 11,
                          units = 'h', ktype = np.float32 ),
        'TIME_OFF' : dict( name = 'AC_SYSTEM_LOCAL_TIME_END', dimension = 11,
                           units = 'h', ktype = np.float32 ),
        'TARGTEMP' : dict( name = 'AC_SYSTEM_TARGET_TEMPERATURE',
                           dimension = 11, units = 'K', ktype = np.float32 ),
        'GAPTEMP' : dict( name = 'AC_SYSTEM_COMFORT_RANGE', dimension = 11,
                          units = 'K', ktype = np.float32 ),
        'TARGHUM' : dict( name = 'AC_SYSTEM_TARGET_HUMIDITY', dimension = 11,
                          units = 'kg kg-1', ktype = np.float32 ),
        'GAPHUM' : dict( name = 'AC_SYSTEM_COMFORT_RANGE', dimension = 11,
                         units = 'kg kg-1', ktype = np.float32 ),
        'PERFLO' : dict( name = 'PEAK_PERSON_PER_UNIT_AREA', dimension = 11,
                         units = 'm-2', ktype = np.float32 ),
        'HSEQUIP' : dict( name = 'DIURNAL_HEATING_EQUIPMENT_PROFILE',
                          dimension = 24, units = 'm-2', ktype = np.float32 ),
        'HSEQUIP_SCALE_FACTOR' : dict( name = 'EQUIPMENT_SCALE_FACTOR',
                                       dimension = 11, units = 'W m-2',
                                       ktype = np.float32 ),
        'GR_FLAG' : dict( name = 'GREEN_ROOF_MODEL', dimension = 1,
                          ktype = np.int8 ),
        'GR_TYPE' : dict( name = 'GREEN_ROOF_TYPE', dimension = 1,
                          ktype = np.int8 ),
        'GR_FRAC_ROOF' : dict( name = 'GREEN_ROOF_FRACTION', dimension = 11,
                               units = 'W m-2', ktype = np.float32 ),
        'IRHO' : dict( name = 'SPRINKLER_IRRIGATION_DIURNAL_PROFILE',
                       dimension = 24, units = 'W m-2', ktype = np.float32 ),
        'PV_FRAC_ROOF' : dict( name = 'PHOTOVOLTAIC_ROOF_FRACTION',
                               dimension = 11, units = '1',
                               ktype = np.float32 ),
        'STREET PARAMETERS' : dict( name = 'STREET_PARAMETERS',
                                    dimension = [11,2,2], units = 'm',
                                    ktype = np.float32 ),
        'BUILDING HEIGHTS' : dict( name = 'BUIDING_HEIGHTS',
                                   dimension = [11,2,5], units = ['m', '%'],
                                   ktype = np.float32 ),
    }

    parsed = { }
    linecount = 0
    df = open(fname, 'r').readlines( )
    for line in df:
        line = line.partition('#')[0]
        line = line.rstrip( )
        linecount = linecount + 1
        element = line.split(':')
        if element[0]:
            if element[0] in keyparam:
                desc = keyparam[element[0]]
                key = desc['name']
                tbp = element[1].lstrip( ).rstrip( )
                if key == 'BUIDING_HEIGHTS':
                    if key not in parsed:
                        parsed[key] = { }
                    ui = int(tbp)-1
                    ilc = linecount
                    ii = 0
                    hgts = np.zeros([2,5], np.float32)
                    while 'END' not in df[ilc]:
                        xx = df[ilc].split()
                        try:
                            hgts[:,ii] = list(float(x) for x in xx)
                            ii = ii + 1
                        except:
                            pass
                        ilc = ilc + 1
                    if 'values' not in parsed[key]:
                        parsed[key]['values'] = np.zeros(
                                desc['dimension'],
                                dtype=desc['ktype'])
                        if 'units' in desc:
                            parsed[key]['units'] = desc['units']
                    parsed[key]['values'][ui,:,:] = hgts
                elif key == 'STREET_PARAMETERS':
                    parsed[key] = { }
                    parsed[key]['values'] = np.zeros(desc['dimension'],
                                dtype=desc['ktype'])
                    ilc = linecount
                    while 'END' not in df[ilc]:
                        xx = df[ilc].split()
                        try:
                            ff = list(float(x) for x in xx)
                            ii = int(ff[0])-1
                            if ff[1] > 0:
                                jj = 1
                            else:
                                jj = 0
                            parsed[key]['values'][ii,jj,0] = ff[2]
                            parsed[key]['values'][ii,jj,1] = ff[3]
                        except:
                            pass
                        ilc = ilc + 1
                else:
                    parsed[key] = { }
                    if desc['ktype'] == np.float32:
                        try:
                            vals = list(float(x) for x in tbp.split(','))
                        except:
                            vals = list(float(x) for x in tbp.split(' '))
                    elif desc['ktype'] == np.int8:
                        vals = list(int(x) for x in tbp.split(','))
                    if 'units' in desc:
                        parsed[key]['units'] = desc['units']
                    if len(vals) == desc['dimension']:
                        if len(vals) == 1:
                            parsed[key]['values'] = vals[0]
                        else:
                            parsed[key]['values'] = vals
    return(parsed)

ccodes = csv.DictReader(open('World_Countries_Generalized.csv'))

ds = xr.open_dataset(lcz_tif_file, engine="rasterio")
urban_input = ds['band_data'].isel(band=0)
ny,nx = np.shape(urban_input)
ds = ds.chunk({'x': nx, 'y': resoultion_divider})

ds2 = xr.open_dataset('mksrf_urban.nc', mode='r')
nlevurb = ds2.dims['nlevurb']

nplon = ds.x.to_numpy( )
nplat = ds.y.to_numpy( )

lostart = nplon[0]
lastart = nplat[0]
dlon = nplon[1]-lostart
dlat = nplat[1]-lastart
hlo = dlon/2.0
hla = dlat/2.0
hstep = resoultion_divider//2

xlostart = (nplon[hstep]+nplon[hstep-1])/2.0
xlastart = (nplat[hstep]+nplat[hstep-1])/2.0
xloend = (nplon[nx-hstep]+nplon[nx-hstep-1])/2.0
xlaend = (nplat[ny-hstep]+nplat[ny-hstep-1])/2.0

nnlon = nx//resoultion_divider
nnlat = ny//resoultion_divider

xlon = xr.DataArray(name = "lon",
                    dims = ["lon",],
                    data = np.linspace(xlostart,xloend,nnlon),
                    attrs = dict(standard_name = "longitude",
                                 units = "degrees_east"))
xlat = xr.DataArray(name = "lat",
                    dims = ["lat",],
                    data = np.linspace(xlastart,xlaend,nnlat),
                    attrs = dict(standard_name = "latitude",
                                 units = "degrees_north"))
xdensity = xr.DataArray(name = "density_class",
                        dims = ["density_class",],
                        data = list(class_old_class),
                        attrs = dict(standard_name = "density_class,",
                                     units = "class"))
xregion = xr.DataArray(name="country_code",
                       dims=["region"],
                       data = list(x['Two-character ISO Code for the Country']
                                   for x in ccodes),
                       attrs = dict(long_name = 'ISO code for country'))
xradfrq = xr.DataArray(name="radiation_frequency",
                       dims=["numrad"],
                       data=["visible", "near-infrared"],
                       attrs = dict(long_name = "Radiation spectral region"))
xradspc = xr.DataArray(name="radiation_transmission",
                       dims=["numsolar"],
                       data=["direct", "diffuse"],
                       attrs = dict(long_name = "Radiation transmission"))
xurblev = xr.DataArray(name="urban_levels",
                       dims=["nlevurb"],
                       data=np.linspace(1,nlevurb,nlevurb),
                       attrs = dict(long_name = "Level number"))

out_nx = len(xlon)
out_ny = len(xlat)
classmax = 11

ds1 = xr.open_dataset(countries_file,mode='r')
ds1 = ds1.chunk({'lon' : out_nx, 'lat' : 10*resoultion_divider})
country_id = ds1.Band1

pp = parse_URBPARM_LCZ_TBL(urbparam_file)

x = np.array(pp['ROOF_LEVEL']['values'])
ht_roof = np.repeat(x[np.newaxis,:], np.size(xregion), axis=0)
old_ht_roof = ds2.HT_ROOF
ht_roof = getval_old_new(old_ht_roof,ht_roof)

wind_hgt_canyon = np.where(ht_roof > 0.0,0.5 * ht_roof,-999.0)

y = np.array(pp['ROAD_WIDTH']['values'])
y = y/x
canyon_hwr = np.repeat(y[np.newaxis,:], np.size(xregion), axis=0)
old_canyon_hwr = ds2.CANYON_HWR
canyon_hwr = getval_old_new(old_canyon_hwr,canyon_hwr)

x = np.array(pp['ROAD_WIDTH']['values'])
y = np.array(pp['ROOF_WIDTH']['values'])
y = y/(x+y)
wtlunit_roof = np.repeat(y[np.newaxis,:], np.size(xregion), axis=0)
old_wtlunit_roof = ds2.WTLUNIT_ROOF
wtlunit_roof = getval_old_new(old_wtlunit_roof,wtlunit_roof)

wtroad_perv = np.repeat(x[np.newaxis,:], np.size(xregion), axis=0)
old_wtroad_perv = ds2.WTROAD_PERV
wtroad_perv = getval_new(old_wtroad_perv,wtroad_perv)

x = np.array(pp['ROOF_SURFACE_EMISSIVITY']['values'])
em_roof = np.repeat(x[np.newaxis,:], np.size(xregion), axis=0)
old_em_roof = ds2.EM_ROOF
em_roof = getval_old_new(old_em_roof,em_roof)

x = np.array(pp['WALL_SURFACE_EMISSIVITY']['values'])
em_wall = np.repeat(x[np.newaxis,:], np.size(xregion), axis=0)
old_em_wall = ds2.EM_WALL
em_wall = getval_old_new(old_em_wall,em_wall)

x = np.array(pp['GROUND_SURFACE_EMISSIVITY']['values'])
em_improad = np.repeat(x[np.newaxis,:], np.size(xregion), axis=0)
old_em_improad = ds2.EM_IMPROAD
em_improad = getval_old_new(old_em_improad,em_improad)
em_perroad = np.repeat(x[np.newaxis,:], np.size(xregion), axis=0)
old_em_perroad = ds2.EM_PERROAD
em_perroad = getval_old_new(old_em_perroad,em_perroad)

old_nlev_improad = ds2.NLEV_IMPROAD
nlev_improad = np.repeat(x[np.newaxis,:], np.size(xregion), axis=0)
nlev_improad = getval_new(old_nlev_improad,nlev_improad)

old_thick_roof = ds2.THICK_ROOF
thick_roof = np.repeat(x[np.newaxis,:], np.size(xregion), axis=0)
thick_roof = getval_new(old_thick_roof,thick_roof)

old_thick_wall = ds2.THICK_WALL
thick_wall = np.repeat(x[np.newaxis,:], np.size(xregion), axis=0)
thick_wall = getval_new(old_thick_wall,thick_wall)

old_t_building_min = ds2.T_BUILDING_MIN
t_building_min = np.repeat(x[np.newaxis,:], np.size(xregion), axis=0)
t_building_min = getval_new(old_t_building_min,t_building_min)

old_t_building_max = ds2.T_BUILDING_MAX
t_building_max = np.repeat(x[np.newaxis,:], np.size(xregion), axis=0)
t_building_max = getval_new(old_t_building_max,t_building_max)

old_tk_roof = ds2.TK_ROOF
tk_roof = np.zeros((np.size(xurblev), np.size(xregion), np.size(xdensity)))
tk_roof = getval_new(old_tk_roof,tk_roof)

old_tk_wall = ds2.TK_WALL
tk_wall = np.zeros((np.size(xurblev), np.size(xregion), np.size(xdensity)))
tk_wall = getval_new(old_tk_wall,tk_wall)

old_tk_improad = ds2.TK_IMPROAD
tk_improad = np.zeros((np.size(xurblev), np.size(xregion), np.size(xdensity)))
tk_improad = getval_new(old_tk_improad,tk_improad)

old_cv_roof = ds2.CV_ROOF
cv_roof = np.zeros((np.size(xurblev), np.size(xregion), np.size(xdensity)))
cv_roof = getval_new(old_cv_roof,cv_roof)

old_cv_wall = ds2.CV_WALL
cv_wall = np.zeros((np.size(xurblev), np.size(xregion), np.size(xdensity)))
cv_wall = getval_new(old_cv_wall,cv_wall)

old_cv_improad = ds2.CV_IMPROAD
cv_improad = np.zeros((np.size(xurblev), np.size(xregion), np.size(xdensity)))
cv_improad = getval_new(old_cv_improad,cv_improad)

old_alb_roof = ds2.ALB_ROOF
alb_roof = np.zeros((np.size(xradspc), np.size(xradfrq),
                     np.size(xregion), np.size(xdensity)))
alb_roof = getval_new(old_alb_roof,alb_roof)

old_alb_wall = ds2.ALB_WALL
alb_wall = np.zeros((np.size(xradspc), np.size(xradfrq),
                     np.size(xregion), np.size(xdensity)))
alb_wall = getval_new(old_alb_wall,alb_wall)

old_alb_improad = ds2.ALB_IMPROAD
alb_improad = np.zeros((np.size(xradspc), np.size(xradfrq),
                        np.size(xregion), np.size(xdensity)))
alb_improad = getval_new(old_alb_improad,alb_improad)

old_alb_perroad = ds2.ALB_PERROAD
alb_perroad = np.zeros((np.size(xradspc), np.size(xradfrq),
                        np.size(xregion), np.size(xdensity)))
alb_perroad = getval_new(old_alb_perroad,alb_perroad)

ntot = resoultion_divider*resoultion_divider
ntotw = 3*(ntot//4)  # if 75% water, it is water ;)

ht_roof = xr.DataArray(name='HT_ROOF', data = ht_roof,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='height of roof', units='m'))
wind_hgt_canyon = xr.DataArray(name='WIND_HGT_CANYON', data = wind_hgt_canyon,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='height of wind in canyon', units='m'))
canyon_hwr = xr.DataArray(name='CANYON_HWR', data = canyon_hwr,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='canyon height to width ratio',
                             units='unitless'))
wtlunit_roof = xr.DataArray(name='WTLUNIT_ROOF', data = wtlunit_roof,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='fraction of roof', units='unitless'))
wtroad_perv = xr.DataArray(name='WTROAD_PERV', data = wtroad_perv,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='fraction of pervious road',
                             units='unitless'))
em_roof = xr.DataArray(name='EM_ROOF', data = em_roof,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='emissivity of roof',
                             units='cunitless'))
em_wall = xr.DataArray(name='EM_WALL', data = em_wall,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='emissivity of wall',
                             units='unitless'))
em_improad = xr.DataArray(name='EM_IMPROAD', data = em_improad,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='emissivity of impervious road',
                             units='unitless'))
em_perroad = xr.DataArray(name='EM_PERROAD', data = em_perroad,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='emissivity of pervious road',
                             units='unitless'))
nlev_improad = xr.DataArray(name='NLEV_IMPROAD', data = nlev_improad,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='number of impervious road layers',
                             units='unitless'))
thick_roof = xr.DataArray(name='THICK_ROOF', data = thick_roof,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='hickness of roof', units='m'))
thick_wall = xr.DataArray(name='THICK_WALL', data = thick_wall,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='thickness of wall', units='m'))
t_building_min = xr.DataArray(name='T_BUILDING_MIN', data = t_building_min,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='minimum interior building temperature',
                             units='K'))
t_building_max = xr.DataArray(name='T_BUILDING_MAX', data = t_building_max,
                dims = ['region','density_class'],
                coords = dict(region=xregion, density_class=xdensity),
                attrs = dict(long_name='maximum interior building temperature',
                             units='K'))
tk_roof = xr.DataArray(name='TK_ROOF', data=tk_roof,
                       dims=['nlevurb','region','density_class'],
                       coords = dict(nlevurb=xurblev,
                                     region=xregion, density_class=xdensity),
                       attrs = dict(long_name='thermal conductivity of roof',
                                    units='W m-1 k-1'))
tk_wall = xr.DataArray(name='TK_WALL', data=tk_wall,
                       dims=['nlevurb','region','density_class'],
                       coords = dict(nlevurb=xurblev,
                                     region=xregion, density_class=xdensity),
                       attrs = dict(long_name='thermal conductivity of wall',
                                    units='W m-1 k-1'))
tk_improad = xr.DataArray(name='TK_IMPROAD', data=tk_improad,
                       dims=['nlevurb','region','density_class'],
                       coords = dict(nlevurb=xurblev,
                                     region=xregion, density_class=xdensity),
                       attrs = dict(long_name='thermal conductivity of impervious road',
                                    units='W m-1 k-1'))
cv_roof = xr.DataArray(name='CV_ROOF', data=cv_roof,
                       dims=['nlevurb','region','density_class'],
                       coords = dict(nlevurb=xurblev,
                                     region=xregion, density_class=xdensity),
                       attrs = dict(long_name='volumetric heat capacity of roof',
                                    units='J m-3 k-1'))
cv_wall = xr.DataArray(name='CV_WALL', data=cv_wall,
                       dims=['nlevurb','region','density_class'],
                       coords = dict(nlevurb=xurblev,
                                     region=xregion, density_class=xdensity),
                       attrs = dict(long_name='volumetric heat capacity of wall',
                                    units='J m-3 k-1'))
cv_improad = xr.DataArray(name='CV_IMPROAD', data=cv_improad,
                       dims=['nlevurb','region','density_class'],
                       coords = dict(nlevurb=xurblev,
                                     region=xregion, density_class=xdensity),
                       attrs = dict(long_name='volumetric heat capacity of impervious road',
                                    units='J m-3 k-1'))
alb_roof = xr.DataArray(name='ALB_ROOF', data=alb_roof,
                dims = ['numsolar','numrad','region','density_class'],
                coords = dict(numsolar=xradspc, numrad=xradfrq,
                              region=xregion, density_class=xdensity),
                attrs = dict(long_name='albedo of roof',
                             units='unitless'))
alb_wall = xr.DataArray(name='ALB_WALL', data=alb_wall,
                dims = ['numsolar','numrad','region','density_class'],
                coords = dict(numsolar=xradspc, numrad=xradfrq,
                              region=xregion, density_class=xdensity),
                attrs = dict(long_name='albedo of wall',
                             units='unitless'))
alb_improad = xr.DataArray(name='ALB_IMPROAD', data=alb_improad,
                dims = ['numsolar','numrad','region','density_class'],
                coords = dict(numsolar=xradspc, numrad=xradfrq,
                              region=xregion, density_class=xdensity),
                attrs = dict(long_name='albedo of impervious road',
                             units='unitless'))
alb_perroad = xr.DataArray(name='ALB_PERROAD', data=alb_perroad,
                dims = ['numsolar','numrad','region','density_class'],
                coords = dict(numsolar=xradspc, numrad=xradfrq,
                              region=xregion, density_class=xdensity),
                attrs = dict(long_name='albedo of pervious road',
                             units='unitless'))

dso = xr.merge([canyon_hwr, wtlunit_roof, wtroad_perv, em_roof, em_wall,
                em_improad, em_perroad, alb_roof, alb_wall, alb_improad,
                alb_perroad, ht_roof, wind_hgt_canyon, tk_roof, tk_wall,
                tk_improad, cv_roof, cv_wall, cv_improad, nlev_improad,
                thick_roof, thick_wall, t_building_min, t_building_max],
               compat='override')

fieldpct = np.zeros([classmax,out_ny,out_nx], dtype=np.float32)
mask = np.zeros([out_ny,out_nx], dtype=np.uint8)

if True:
    for j in trange(out_ny, desc='latitude'):
        jj1 = j*resoultion_divider
        jj2 = jj1 + resoultion_divider
        rspace = urban_input[jj1:jj2,:].fillna(0).astype(np.uint8).to_numpy( )
        for i in trange(out_nx, desc='longitude'):
            ii1 = i*resoultion_divider
            ii2 = ii1 + resoultion_divider
            space = rspace[:,ii1:ii2]
            nwat = np.sum( (space == water) | (space == 0) )
            if ( nwat >= ntotw ):
                mask[j,i] = 0
                fieldpct[:,j,i] = 0.0
            else:
                mask[j,i] = 1
                nland = ntot-nwat
                nveg = np.sum((space > 10) & (space < 16))
                nurban = nland-nveg
                if ( nurban > 0 ):
                    fieldpct[0,j,i] = np.sum(space == 1)/nland * 100.0
                    fieldpct[1,j,i] = np.sum(space == 2)/nland * 100.0
                    fieldpct[2,j,i] = np.sum(space == 3)/nland * 100.0
                    fieldpct[3,j,i] = np.sum(space == 4)/nland * 100.0
                    fieldpct[4,j,i] = np.sum(space == 5)/nland * 100.0
                    fieldpct[5,j,i] = np.sum(space == 6)/nland * 100.0
                    fieldpct[6,j,i] = np.sum(space == 7)/nland * 100.0
                    fieldpct[7,j,i] = np.sum(space == 8)/nland * 100.0
                    fieldpct[8,j,i] = np.sum(space == 9)/nland * 100.0
                    fieldpct[9,j,i] = np.sum(space == 10)/nland * 100.0
                    fieldpct[10,j,i] = 0.0 # ? also mountains here...
                else:
                    fieldpct[:,j,i] = 0.0
            del(space)
        del(rspace)

urbpct = xr.DataArray(name = "PCT_URBAN",
                      dims = ["density_class","lat","lon"],
                      data = fieldpct,
                      coords = dict(density_class=xdensity,
                                    lon = xlon, lat = xlat),
                      attrs = dict(standard_name = 'area_type',
                                   long_name = 'Urban Data Type',
                                   units = '%',),
                     )
landmask = xr.DataArray(name = "LANDMASK",
                        dims = ["lat","lon"],
                        data = mask,
                        attrs = dict(standard_name = 'land_binary_mask',
                                     long_name = "land mask",
                                     units = "unitless",),
                       )
urbpct = urbpct.chunk({'density_class' : 1,
                       'lon': 10*resoultion_divider,
                       'lat': 10*resoultion_divider,})
landmask = landmask.chunk({'lon': 10*resoultion_divider,
                           'lat': 10*resoultion_divider,})
print('Data created!')
dso = xr.merge([dso, urbpct, landmask], compat='override')

region_id = xr.DataArray(
                name='REGION_ID',
                dims = landmask.dims,
                coords = landmask.coords,
                attrs = dict(long_name='Region ID', units='unitless'),
                data = country_id.values)

region_id[...] = xr.where(landmask, region_id+1, np.nan)
print('Region id masked 1!')
region_id = interpolate_na(region_id, ['lat','lon'], method="nearest")
print('Region id interpolated!')
region_id[...] = xr.where(landmask, region_id, 0)
print('Region id masked 2!')
region_id = region_id.chunk({'lon': 10*resoultion_divider,
                             'lat': 10*resoultion_divider,})
print('Region id Created!')

dso = xr.merge([dso,region_id], compat='override')

encode = { "PCT_URBAN" : { 'zlib': True,
                           'complevel' : 4,
                     },
           "LANDMASK"  : { 'zlib': True,
                           'complevel' : 4,
                     },
           "REGION_ID" : { 'zlib': True,
                           'complevel' : 4,
                         },
         }

print('Saving to file...')
dso.to_netcdf(clm_urban_file, format = 'NETCDF4', encoding = encode)

ds.close( )
ds1.close( )
ds2.close( )
dso.close( )

print('Done')
