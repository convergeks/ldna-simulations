import toml
import matlab.engine
import os, sys
import copy
import matplotlib.pyplot as plt
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#
sys.path.append(os.getcwd())
import utils.conversion as conv
import utils.util_get_coords as get_coords
import utils.util_routine_calcs as calc

# Files
PATH = os.getcwd()
FILE = 'env/config.toml'
ENV_FILE = os.path.join(PATH, FILE)

# config class
class Setup(object):
    def __init__(self,
                 env_path=ENV_FILE,
                 beacon_coords={}):
        self.config = toml.load(env_path)
        self.init_beacons(beacon_coords)
        self.ble_tx_settings()
        # init other coordinates
        self.ref_coord()
        self.get_basemap()

    def init_beacons(self, beacon_coords={}):
        self.beacon_structure = beacon_coords
        self.beacon_coords_lonlat = list(zip(*list(beacon_coords.values())))

    def ble_tx_settings(self):
        # power
        channel = self.config['channel']
        self.TxFreq = self.config['freqChannelHz'][channel]
        # power
        dBm = self.config['power_dBm']
        self.TxPower = conv.convert_dbm_watts(dBm)
        self.SystemLoss = self.config['systemLoss']

    def ref_coord(self):
        lons = self.beacon_coords_lonlat[1]
        lats = self.beacon_coords_lonlat[0]
        mid_lon = np.mean(lons)
        mid_lat = np.mean(lats)
        self.ref_lonlat = mid_lon, mid_lat
        self.ref_latlon = mid_lat, mid_lon

    def get_basemap(self):
        test_latlon = self.beacon_coords_lonlat[0][0], self.beacon_coords_lonlat[1][0]
        test_lonlat = test_latlon[1], test_latlon[0]
        _, _, bm = calc.util_convert_htwidth_GPS2Cartcoord(self.ref_latlon, test_latlon, ht_width=200)
        self.basemap = bm
        d = calc.haversine_distance(self.ref_lonlat, test_lonlat)
        # print('Verify distance calc (Google map d=26.75m): ', d)


if __name__ == "__main__":
    setup = Setup(beacon_coords={0: (53.30571095302461, -1.1693787574768069),
                                 1: (53.30600585638605, -1.1712026596069338),
                                 2: (53.30496727463392, -1.168498992919922)}
                  )
