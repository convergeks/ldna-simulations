import toml
import matlab.engine
import os, sys
import copy
import matplotlib.pyplot as plt
import random
import numpy as np
#
sys.path.append(os.getcwd())
import utils.conversion as conv
import utils.util_get_coords as get_coords


# Files
PATH = os.getcwd()
FILE = 'env/config.toml'
ENV_FILE = os.path.join(PATH, FILE)

# config class
class Setup(object):
    def __init__(self, env_path=ENV_FILE):
        self.config = toml.load(env_path)
        self.map_path = os.path.join(self.config['ROOT'], self.config['OSM_Map'])
        # init other params
        self.init_beacons()
        self.ble_tx_settings()
        # init other coordinates
        self.ref_coord()

    def init_beacons(self):
        N = self.config['num_beacons']
        prefix = self.config['beacon_prename']
        self.beacon_structure = {f'{prefix}:{i}': {'lonlat_coord': None} for i in range(N)}

    def ble_tx_settings(self):
        # power
        channel = self.config['channel']
        self.TxFreq = self.config['freqChannelHz'][channel]
        # power
        dBm = self.config['power_dBm']
        self.TxPower = conv.convert_dbm_watts(dBm)
        self.SystemLoss = self.config['systemLoss']

    def ref_coord(self):
        mid_lon = np.mean([self.config['LL_Lon'], self.config['UR_Lon']])
        mid_lat = np.mean([self.config['LL_Lat'], self.config['UR_Lat']])
        self.ref_latlon = mid_lon, mid_lat

    # -- getter , setters --
    def get_coords(self):
        ref_lonlat = self.ref_latlon 
        LL_lonlat = self.config['LL_Lon'], self.config['LL_Lat']
        UR_lonlat = self.config['UR_Lon'], self.config['UR_Lat']
        Alpha = self.config['YardOrientaion_deg']*np.pi/180
        out_tuple = get_coords.get_XY1_XY2_alpha(ref_coords=ref_lonlat,
                                                          coord_diag1=LL_lonlat,
                                                          coord_diag2=UR_lonlat,
                                                          ang=Alpha
                                                )
        XY1, XY2, DThAlpha, bm = out_tuple
        # basemap
        self.basemap = bm
        # xy
        x, y = get_coords.define_ellipse(XY1, XY2, DThAlpha)
        self.boundary_xy_list = list(zip(x, y))
        self.XY1 = XY1
        self.XY2 = XY2

    def set_beacon_coords(self):
        beacons = self.beacon_structure.keys()
        XY = copy.deepcopy(self.boundary_xy_list)
        random.shuffle(XY)
        for bcn in beacons:
            xy1 = XY.pop(0)
            xy2 = XY.pop(1)
            x12, y12 = tuple(zip(*[xy1, xy2]))
            x0, y0 = np.mean(x12), np.mean(y12)
            bcn_lat, bcn_lon = self.basemap(x0, y0, inverse=True)
            self.beacon_structure[bcn]['lonlat_coord'] = bcn_lon, bcn_lat
            self.beacon_structure[bcn]['xy_coord'] = x0, y0
            # put it back
            XY.append(xy1)
            XY.append(xy2)

    def get_beacon_coords(self):
        print('** Beacon Data ***')
        print(self.beacon_structure)

    def show_beacon_xycoords(self):
        XY = self.boundary_xy_list
        x, y = tuple(zip(*XY))
        X1, Y1 = self.XY1
        X2, Y2 = self.XY2
    
        fig, axes = plt.subplots()
        axes.scatter(x, y, color='k')
        axes.plot(X1, Y1, 'bx')
        axes.plot(X2, Y2, 'bx')
        for bcn, b_data in self.beacon_structure.items():
            x0, y0 = b_data['xy_coord']
            axes.scatter(x0, y0, color='r', s=60)
        plt.show()


if __name__ == "__main__":
    setup = Setup()
    setup.get_coords()
    setup.set_beacon_coords()
    setup.get_beacon_coords()
    setup.show_beacon_xycoords()
