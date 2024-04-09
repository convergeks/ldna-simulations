import toml
import matlab.engine
import os, sys
import copy
import matplotlib.pyplot as plt
import random
import numpy as np
#
sys.path.append(os.getcwd())
import sim_core.sim_p0_setup as p0
import utils.util_routine_calcs as util

# Files


# -- scripts --
class GenerateData(p0.Setup):
    def __init__(self):
        # init
        super().__init__()
        # this init
        self.init()
        
        # generate beacons
        self.gen_beacon_data()
        # self.show_beacon_xycoords()
        # generate gateways
        self.gen_gateway_data()
        # self.show_beacon_gateways_xycoords()
        self.show_beacon_gateways_lonlatcoords()

    def init(self):
        # main gateway data 
        self.gateway_list = self.config['gateways']
        # for debugging
        random.seed(26)

    def init_matlab(self):
        self.matlab_engine = matlab.engine.start_matlab()
        self.OSMFile = os.path.join(self.config['ROOT'], self.config['OSM_Map'])
    
    # -- Beacon --
    def gen_beacon_data(self):
        self.get_coords()
        self.set_beacon_coords()
        self.get_beacon_coords()

    # -- Gateway --
    def gen_gateway_data(self):
        # 
        self.gateway_locations = {gw: {} for gw in self.gateway_list}
        XY = copy.deepcopy(self.boundary_xy_list)
        random.shuffle(XY)
        used_XY = []
        for gw in self.gateway_list:
            xy1 = XY.pop(0)
            self.gateway_locations[gw]['xy_coord'] = xy1
            x, y = xy1
            gw_lat, gw_lon = self.basemap(x, y, inverse=True)
            self.gateway_locations[gw]['lonlat_coord'] = gw_lon, gw_lat
            # put it back
            used_XY.append(xy1)
        XY = XY + used_XY

    def show_beacon_gateways_xycoords(self):
        XY = self.boundary_xy_list
        x, y = tuple(zip(*XY))
        X1, Y1 = self.XY1
        X2, Y2 = self.XY2
    
        _, axes = plt.subplots(figsize=(10, 10))        
        axes.scatter(x, y, color='k', label='bounds')
        axes.plot([X1, X2], [Y1, Y2], 'bx', label='reference')
        for bcn, b_data in self.beacon_structure.items():
            x0, y0 = b_data['xy_coord']
            axes.scatter(x0, y0, color='r', s=100, label=bcn)
        clrs = list('gbrckmy')
        for gw, g_data in self.gateway_locations.items():
            x0, y0 = g_data['xy_coord']
            clr = clrs.pop(0)
            axes.scatter(x0, y0, color=clr, marker='s', 
                         s=200, label=gw)
            clrs.append(clr)
        axes.legend()
        axes.grid('on')
        plt.show()

    def show_beacon_gateways_lonlatcoords(self):
        XY = self.boundary_xy_list
        lonlat_list = [self.basemap(x, y, inverse=True) for x, y in XY]
        lats, lons = tuple(zip(*lonlat_list))
        X1, Y1 = self.XY1
        X2, Y2 = self.XY2
        lonlat_1 = self.basemap(X1, Y1, inverse=True)
        lonlat_2 = self.basemap(X2, Y2, inverse=True)
        lat1, lon1 = lonlat_1
        lat2, lon2 = lonlat_2
    
        _, axes = plt.subplots(figsize=(10, 10))        
        axes.scatter(lons, lats, color='k', label='bounds')
        axes.plot([lon1, lon2], [lat1, lat2], 'bx', label='reference')
        for bcn, b_data in self.beacon_structure.items():
            x0, y0 = b_data['lonlat_coord']
            axes.scatter(x0, y0, color='r', s=100, label=bcn)
        clrs = list('gbrckmy')
        for gw, g_data in self.gateway_locations.items():
            x0, y0 = g_data['lonlat_coord']
            clr = clrs.pop(0)
            axes.scatter(x0, y0, color=clr, marker='s', 
                         s=200, label=gw)
            clrs.append(clr)
        axes.legend()
        axes.grid('on')
        plt.show()

    # -- BLE signal generation --
    def start_matlab(self):
        self.init_matlab()
        eng = self.matlab_engine

    def get_tx(self, beacon_id):
        # iterate over the tx positions
        eng = self.matlab_engine
        beacon_lon, beacon_lat = self.beacon_structure[beacon_id]['lonlat_coord']
        beacon_TxHeight = self.config['TxHeight']
        tx = eng.txsite("Name", "Transmitting beacon on unit", 
                        "Latitude", beacon_lat, 
                        "Longitude", beacon_lon, 
                        "AntennaHeight", beacon_TxHeight,
                        "TransmitterPower", self.TxPower,
                        "TransmitterFrequency", self.TxFreq,
                        "SystemLoss", self.SystemLoss)
        self.tx = tx

    def get_rx(self, gateway_id):
        #   receiver 
        eng = self.matlab_engine
        gateway_lon, gateway_lat = self.gateway_locations[gateway_id]['lonlat_coord']
        gateway_RxHeight = self.config['RxHeight']        
        rx = eng.rxsite('Name', "Crane Receiver", 
                        "Latitude", gateway_lat,
                        "Longitude", gateway_lon,
                        "AntennaHeight", gateway_RxHeight) 
        self.rx = rx       

    def calc_txsignal(self, SHOW=True):
        eng = self.matlab_engine
        rx = self.rx
        tx = self.tx
        # propagation model
        model_name  = "longley-rice"
        var_name  = "SituationVariabilityTolerance"
        rtLongleyRice_SitVarLow= eng.propagationModel(model_name, var_name, 0.25)
        rtLongleyRice_SitVarMid= eng.propagationModel(model_name, var_name, 0.55)
        rtLongleyRice_SitVarHigh= eng.propagationModel(model_name, var_name, 0.75)
        # signal strength
        ss_LR0 = eng.sigstrength(rx, tx, rtLongleyRice_SitVarLow)
        ss_LR1 = eng.sigstrength(rx, tx, rtLongleyRice_SitVarMid)
        ss_LR2 = eng.sigstrength(rx, tx, rtLongleyRice_SitVarHigh)
        var_low = (ss_LR0 - ss_LR1)/ss_LR1
        var_hi= (ss_LR1 - ss_LR2)/ss_LR1

        ss_fs_Mid = self.calc_fs(rx, tx, rtLongleyRice_SitVarMid)
        ss_fs_low = ss_fs_Mid*(1 + var_low)
        ss_fs_hi = ss_fs_Mid*(1 - var_hi)

        d_fs_mid = util.get_RSSI_to_distance_estimate(ss_fs_Mid)
        d_fs_low = util.get_RSSI_to_distance_estimate(ss_fs_low)
        d_fs_hi = util.get_RSSI_to_distance_estimate(ss_fs_hi)

        # print("Receieved power with weather: " + ss + "dBm" )
        print(f"Received power with Longley Rice (Mid Var +/-0.2CI): {ss_fs_Mid:.1f} ({ss_fs_low:.1f},{ss_fs_hi:.1f}) dBm " +
                f"(distance={d_fs_mid:.1f} ({d_fs_low:.1f}, {d_fs_hi:.1f}) m)")
        
        # Todo - visulaiser - combine streamlit with contour plot map
        # sig_strengths = [i for i in range(-95, -15, 5)]
        # if SHOW :
        #     viewer = eng.siteviewer("Buildings", self.OSMFile,"Basemap", "topographic")
        #     eng.coverage(tx, rtLongleyRice_SitVarMid, 
        #                 "SignalStrengths", sig_strengths[0],
        #                 "MaxRange", 250,
        #                 "Resolution", 2,
        #                 "Transparency", 0.6)
            
    def calc_fs(self, rx, tx, prop_model):
        eng = self.matlab_engine
        fc = 10e9
        pm = prop_model
        wavelength = eng.physconst('LightSpeed')/fc
        pl0 = eng.fspl(eng.distance(rx, tx), wavelength)
        ss = eng.sigstrength(rx, tx, pm)
        pl = -pl0*0.9 + 0.1*ss
        return pl
    

if __name__ == "__main__":
    gendata = GenerateData()
    # start calcs
    gendata.start_matlab()
    # produce calcs
    for bcn in gendata.beacon_structure:
        gendata.get_tx(beacon_id=bcn)
        for gw in gendata.gateway_locations:
            print(gw, gendata.gateway_locations[gw]['lonlat_coord'])
            gendata.get_rx(gateway_id=gw)            
            gendata.calc_txsignal()

