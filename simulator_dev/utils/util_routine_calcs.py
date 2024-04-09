import pandas as pd
import datetime as dt
import os, sys
import numpy
import warnings
warnings.filterwarnings('ignore')
import math
import matplotlib
from mpl_toolkits.basemap import Basemap

# # map rssi to distance: See Calculation : https://journals.sagepub.com/doi/full/10.1155/2014/371350

def get_RSSI_to_distance_estimate(rssi):
    MP_dBm = -21.5 
    C = 10*3.385 # env factors: 10*(2-5)
    distance_metre = 10**((MP_dBm-rssi)/C)
    return distance_metre

# Test calcs - distance from geopy
def util_convert_htwidth_GPS2Cartcoord(ref_latlon, calc_latlon, ht_width=2E2):
    basemap = Basemap(projection='lcc', resolution='i', lat_0=ref_latlon[0], lon_0=ref_latlon[1], width=ht_width, height=ht_width)
    X, Y = basemap(calc_latlon[1], calc_latlon[0])
    return X, Y, basemap

def haversine(coord1, coord2):
    R = 6371000 # Earth radius in meters
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    phi1, phi2 = math.radians(lat1), math.radians(lat2) 
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + \
        math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    y, x = 2*R*math.sqrt(a), 2*R*math.sqrt(1 - a)
    return y, x, R

def haversine_distance(coord1, coord2):
    y, x, R = haversine(coord1, coord2)
    return 2*R*math.atan2(y, x)
