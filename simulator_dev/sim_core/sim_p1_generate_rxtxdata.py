import toml
import matlab.engine
import os, sys
import copy
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import time
import datetime
#
sys.path.append(os.getcwd())
import sim_core.sim_p0_setupdevices as p0
import utils.util_routine_calcs as util

# Settings
ADV_INTERVAL = 3
GPS_FLEET_UPDATE_INTERVAL = 20
DELTA_LATLON = 1.5e-5   # radius of 1.57m
# Files
ROOT = os.getcwd()
if "sim_core" in ROOT:
    ROOT = os.getcwd().parent
DEFAULT_OUTPATH = os.path.join(ROOT, "dump")
DEFAULT_CSVPATH = os.path.join(ROOT, "dump/csvfiles")

# -- scripts --
class GenerateData(p0.Setup):
    def __init__(self, 
                 beacon_coords={},
                 gateway_coords={},
                 start_time=datetime.datetime.now()):
        # time
        self.sim_start_time = start_time
        # init
        st = time.time()
        super().__init__(beacon_coords=beacon_coords)
        e1 = time.time()
        print(f'e1={e1-st:.2f}secs')
        # generate gateways
        self.init_gateways(gateway_coords)
        e2 = time.time()
        # print(f'e2={e2-st:.2f}secs')
        self.start_matlab()
        e3 = time.time()
        print(f'e3={e3-st:.2f}secs')
        # matlab computation
        self.analyse_beacons_gateways()
        # 
        self.generate_milestone_enabling_data()
        self.generate_full_timeseries()
        e4 = time.time()
        print(f'e4={e4-st:.2f}secs')

    def init_matlab(self):
        self.matlab_engine = matlab.engine.start_matlab()
        self.OSMFile = os.path.join(self.config['ROOT'], self.config['OSM_Map'])
        
    # -- Gateway --
    def init_gateways(self, gateway_coords={}):
        self.gateway_locations = gateway_coords

    # -- Analyse for beacons and gateways --
    def analyse_beacons_gateways(self):
        rssi_gps_dict = {'gateway_marker_index':[],
                          'beacon_id': [],
                          'gateway_id': [],
                          'rssi': [],
                          'rssi_var': [],
                          'beacon_lonlat': [],
                          'gateway_lonlat': [],
                          'distance': [],
                          'distance_err': [],
                          'gps_acc': [],
                          }
        for bid, blatlon in self.beacon_structure.items():
            blonlat = blatlon[1], blatlon[0]
            tx = self.get_tx(blonlat)
            for gid, gmarkers in self.gateway_locations.items():
                for idx, glatlon in enumerate(gmarkers):
                    glonlat = glatlon[1], glatlon[0]
                    rx = self.get_rx(lonlat=glonlat)
                    res = self.calc_txsignal(rx, tx)
                    rssi, rssi_pm_err, dist, dist_pm = res
                    gps_acc = np.abs(2 + np.random.randn())
                    rssi_gps_dict['gateway_marker_index'].append(idx)
                    rssi_gps_dict['beacon_id'].append(bid)
                    rssi_gps_dict['gateway_id'].append(gid)
                    rssi_gps_dict['rssi'].append(rssi)
                    rssi_gps_dict['rssi_var'].append(rssi_pm_err)
                    rssi_gps_dict['beacon_lonlat'].append(blonlat)
                    rssi_gps_dict['gateway_lonlat'].append(glonlat)
                    rssi_gps_dict['distance'].append(dist)
                    rssi_gps_dict['distance_err'].append(dist_pm)
                    rssi_gps_dict['gps_acc'].append(gps_acc)
        self.rssi_gps_df = pd.DataFrame.from_dict(rssi_gps_dict)
        # print(self.rssi_gps_df.head())

    def generate_milestone_enabling_data(self, gateway_speed_mps=0.8):
        self.milestones_df = {}
        all_df = copy.deepcopy(self.rssi_gps_df)
        for bid, df in all_df.groupby('beacon_id'):
            df = df.sort_values(by='gateway_marker_index', ascending=True)
            df['gateway_status'] = 'Milestone'
            df['gateway_status'].iloc[0] = 'Static'
            df['gateway_status'].iloc[-1] = 'Static'
            # duration at location@ make first and last resident for 5 mins
            df['duration_secs'] = 20
            df['duration_secs'].iloc[0] = 300
            df['duration_secs'].iloc[-1] = 300
            # lat-long
            df['longitude'] = df.gateway_lonlat.apply(lambda x: np.float(x[0]))
            df['latitude'] = df.gateway_lonlat.apply(lambda x: np.float(x[1]))
            df['Delta_lat'] = df.latitude.diff()
            df['Delta_lon'] = df.longitude.diff()
            df['Delta_lat'].iloc[0] = DELTA_LATLON
            df['Delta_lon'].iloc[0] = DELTA_LATLON
            # compute move rate
            df['Delta_s'] = df.apply(lambda x: util.haversine_distance((x.Delta_lon, x.Delta_lat), (0, 0)), axis=1)
            df['Delta_t'] = df.Delta_s.apply(lambda x: x/gateway_speed_mps)
            # compute rate of change of lat/lon
            df['Delta_lat_rate'] = df.apply(lambda x: x.Delta_lat/x.Delta_t, axis=1)
            df['Delta_lon_rate'] = df.apply(lambda x: x.Delta_lon/x.Delta_t, axis=1)
            # compute rssi change
            df['rssi_delta'] = df.rssi.diff()
            df['rssi_delta'].iloc[0] = 2.5
            #
            self.milestones_df[bid] = df

    def generate_full_timeseries(self):
        ref_df = self.milestones_df
        gw_gps_data = {'time': [], 'latitude': [], 'longitude': [],
                       'accuracy': [], 'gateway_id': [], 'marker_index': []}
        bcn_rssi_data = {'time': [], 'rssi': [], 'gateway_id': [],
                         'beacon_id': [], 'marker_index': []}
        update_field_data = lambda x_map, y_data: {fld: farr.append(y_data[fld]) for fld, farr in x_map.items()} 
        #
        for bcn_id, bdf in ref_df.items():
            epoch_time = self.sim_start_time
            milestone_start_time = epoch_time
            milestone_time = datetime.timedelta(seconds=0)
            for _, row in bdf.iterrows():
                idx = row.gateway_marker_index
                milestone_start_time += milestone_time
                # milestone_time = 0
                for ii in range(1, row.duration_secs, ADV_INTERVAL):
                    # update miestone time
                    milestone_time = datetime.timedelta(seconds=ii)
                    gw_id = row.gateway_id
                    # bcn_id = row.beacon_id
                    if row.gateway_status == 'Static':
                        # sampling from the same pool
                        # beacon
                        bcn_rssi = row.rssi + np.random.random_integers(low=-row.rssi_var[0],
                                                                        high=row.rssi_var[1])
                        # gateway - only update if index meets criterion
                        if ii%GPS_FLEET_UPDATE_INTERVAL == 0:
                            gps_time = milestone_start_time + milestone_time
                            gps_dlat = DELTA_LATLON*np.random.randn()/2
                            gps_dlon = DELTA_LATLON*np.random.randn()/2
                            gps_lat = row.latitude + gps_dlat
                            gps_lon = row.longitude + gps_dlon
                            loc_err = (gps_dlat, gps_dlon)
                            gps_acc = util.haversine_distance(loc_err, (0, 0))
                            #store gateway data
                            gps_data = {'time': gps_time, 'latitude': gps_lat, 'longitude': gps_lon, 'accuracy': gps_acc,
                                        'gateway_id': gw_id, 'marker_index': idx}   
                            #gw_gps_data = update_field_data(gw_gps_data, gps_data)
                            for fld, val in gps_data.items():
                                gw_gps_data[fld].append(val)
                    else:
                        bcn_drssi = row.rssi_delta + np.random.random_integers(low=-row.rssi_var[0],
                                                                            high=row.rssi_var[1])
                        bcn_rssi = row.rssi + bcn_drssi
                        #
                        gps_dlat = row.Delta_lat_rate*ii + DELTA_LATLON*np.random.randn()/2
                        gps_dlon = row.Delta_lon_rate*ii + DELTA_LATLON*np.random.randn()/2
                        gps_lat = row.latitude + gps_dlat
                        gps_lon = row.longitude + gps_dlon
                        loc_err = (gps_dlat, gps_dlon)
                        gps_acc = util.haversine_distance(loc_err, (0, 0))
                        #store gateway data
                        gps_data = {'time': gps_time, 'latitude': gps_lat, 'longitude': gps_lon, 'accuracy': gps_acc,
                                    'gateway_id': gw_id, 'marker_index': idx}
                        # gw_gps_data = update_field_data(gw_gps_data, gps_data)
                        for fld, val in gps_data.items():
                            gw_gps_data[fld].append(val)
                    # beacon data
                    bcn_time = milestone_start_time + milestone_time
                    beacon_data = {'time': bcn_time, 'rssi': bcn_rssi, 'gateway_id': gw_id, 'beacon_id': bcn_id, 'marker_index': idx}
                    # bcn_rssi_data = update_field_data(bcn_rssi_data, beacon_data)
                    # bcn_rssi_data = {fld: bcn_rssi_data[fld].append(val) for fld, val in beacon_data.items()}
                    for fld, val in beacon_data.items():
                        bcn_rssi_data[fld].append(val)
        # create
        bcn_df = pd.DataFrame.from_dict(bcn_rssi_data)
        bcn_df = bcn_df.reset_index()
        self.beacon_df = bcn_df
        gw_df = pd.DataFrame.from_dict(gw_gps_data)
        gw_df = gw_df.reset_index()
        self.gateway_df = gw_df

    def generate_csv_file_bydevice(self, csvpath=DEFAULT_CSVPATH):
        beacon_fields = ['Gateway ID','ID','Rss I','Time','Tracker ID']
        beacon_field_map = {'gateway_id': 'Gateway ID',
                            'index': 'ID',
                            'rssi': 'Rss I',
                            'time': 'Time',
                            'beacon_id': 'Tracker ID'}        
        gateway_fields = ['Accuracy','Device ID','ID','Latitude','Longitude','Time','X','Y','Z']
        gateway_field_map = {'accuracy': 'Accuracy',
                              'gateway_id': 'Device ID',
                              'index': 'ID',
                              'latitude': 'Latitude',
                              'longitude': 'Longitude',
                              'time': 'Time'
                              }
        beacon_df = self.beacon_df[list(beacon_field_map.keys())]
        beacon_df = beacon_df.rename(columns=beacon_field_map)
        beacon_df.set_index('Gateway ID', inplace=True)
        beacon_df.to_csv(os.path.join(csvpath, 'test_beacon.csv'))
        #
        gateway_df = self.gateway_df[list(gateway_field_map.keys())]
        gateway_df = gateway_df.rename(columns=gateway_field_map)
        gateway_df['X'] = ''
        gateway_df['Y'] = ''
        gateway_df['Z'] = ''
        gateway_df.set_index('Accuracy', inplace=True)
        gateway_df.to_csv(os.path.join(csvpath, 'test_gateway.csv'))
    
    def generate_timeseries_milestones(self):
        df = copy.deepcopy(self.rssi_gps_df)
        df['time'] = df.gateway_marker_index.apply(lambda x: self.sim_start_time + datetime.timedelta(seconds=10*x))
        df['longitude'] = df.gateway_lonlat.apply(lambda x: np.float(x[0]))
        df['latitude'] = df.gateway_lonlat.apply(lambda x: np.float(x[1]))
        df = df.reset_index()
        self.timeseries_rssi_gps_df = df
        self.generate_csv_file_bymilestone()

    def generate_csv_file_bymilestone(self, csvpath=DEFAULT_CSVPATH):
        beacon_fields = ['Gateway ID','ID','Rss I','Time','Tracker ID']
        beacon_field_map = {'gateway_id': 'Gateway ID',
                            'index': 'ID',
                            'rssi': 'Rss I',
                            'time': 'Time',
                            'beacon_id': 'Tracker ID'}        
        gateway_fields = ['Accuracy','Device ID','ID','Latitude','Longitude','Time','X','Y','Z']
        gateway_field_map = {'gps_acc': 'Accuracy',
                              'gateway_id': 'Device ID',
                              'index': 'ID',
                              'latitude': 'Latitude',
                              'longitude': 'Longitude',
                              'time': 'Time'
                              }
        beacon_df = self.timeseries_rssi_gps_df[list(beacon_field_map.keys())]
        self.beacon_df = beacon_df.rename(columns=beacon_field_map)
        self.beacon_df.set_index('Gateway ID', inplace=True)
        self.beacon_df.to_csv(os.path.join(csvpath, 'test_beacon.csv'))
        #
        gateway_df = self.timeseries_rssi_gps_df[list(gateway_field_map.keys())]
        self.gateway_df = gateway_df.rename(columns=gateway_field_map)
        self.gateway_df['X'] = ''
        self.gateway_df['Y'] = ''
        self.gateway_df['Z'] = ''
        self.gateway_df.set_index('Accuracy', inplace=True)
        self.gateway_df.to_csv(os.path.join(csvpath, 'test_gateway.csv'))

    # -- BLE signal generation --
    def start_matlab(self):
        self.init_matlab()
        eng = self.matlab_engine

    def get_tx(self, lonlat):
        # iterate over the tx positions
        eng = self.matlab_engine
        beacon_TxHeight = self.config['TxHeight']
        tx = eng.txsite("Name", "Transmitting beacon on unit",
                        "Latitude", lonlat[1],
                        "Longitude", lonlat[0],
                        "AntennaHeight", beacon_TxHeight,
                        "TransmitterPower", self.TxPower,
                        "TransmitterFrequency", self.TxFreq,
                        "SystemLoss", self.SystemLoss)
        return tx

    def get_rx(self, lonlat):
        #   receiver 
        eng = self.matlab_engine
        gateway_RxHeight = self.config['RxHeight']
        rx = eng.rxsite('Name', "Crane Receiver",
                        "Latitude", lonlat[1],
                        "Longitude", lonlat[0],
                        "AntennaHeight", gateway_RxHeight)
        return rx

    def calc_txsignal(self, rx, tx):
        eng = self.matlab_engine
        # propagation model
        model_name  = "longley-rice"
        var_name  = "SituationVariabilityTolerance"
        rtLongleyRice_SitVarLow= eng.propagationModel(model_name, var_name, 0.95)
        rtLongleyRice_SitVarMid= eng.propagationModel(model_name, var_name, 0.55)
        rtLongleyRice_SitVarHigh= eng.propagationModel(model_name, var_name, 0.05)
        # signal strength
        ss_LR0 = eng.sigstrength(rx, tx, rtLongleyRice_SitVarLow)
        ss_LR1 = eng.sigstrength(rx, tx, rtLongleyRice_SitVarMid)
        ss_LR2 = eng.sigstrength(rx, tx, rtLongleyRice_SitVarHigh)
        # var_low = (ss_LR0 - ss_LR1)/ss_LR1
        # var_hi= (ss_LR1 - ss_LR2)/ss_LR1

        ss_fs_Mid = ss_LR1 #self.calc_fs(rx, tx, rtLongleyRice_SitVarMid)
        ss_fs_low = ss_LR0 #ss_fs_Mid*(1 + var_low)
        ss_fs_hi = ss_LR2 # ss_fs_Mid*(1 - var_hi)
        ss_err_rel_pm = (ss_fs_Mid-ss_fs_low, ss_fs_hi-ss_fs_Mid)
        #
        d_fs_mid = util.get_RSSI_to_distance_estimate(ss_fs_Mid)
        d_fs_low = util.get_RSSI_to_distance_estimate(ss_fs_low)
        d_fs_hi = util.get_RSSI_to_distance_estimate(ss_fs_hi)
        d_err_rel_pm = (d_fs_mid-d_fs_low, d_fs_hi-d_fs_mid)
        #
        return ss_fs_Mid, ss_err_rel_pm, d_fs_mid, d_err_rel_pm
            
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
    GW_MARKERS = [(53.304499263489, -1.1687564849853518),
                  (53.30513556505881, -1.170376539230347),
                  (53.306544370312125, -1.169400215148926),
                  (53.30463389735762, -1.1737775802612307)
                  ]
    gendata = GenerateData(beacon_coords={0: (53.30571095302461, -1.1693787574768069),
                                          1: (53.30600585638605, -1.1712026596069338),
                                          2: (53.30496727463392, -1.168498992919922)},
                            gateway_coords={'G1': GW_MARKERS}
                          )
    print(gendata.rssi_gps_df.head())