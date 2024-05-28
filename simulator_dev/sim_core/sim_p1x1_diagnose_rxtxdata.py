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
import utils.conversion as conv
import utils.util_matlab_tools as matutil

# Debug Settings
VERBOSE = False

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
DEFAULT_IMPORT_DATAPATH = os.path.join(ROOT, "dump/datasource/dump/Tinglev/datasource/raw")
# env
ENV_FILE = 'env/config.toml'
ENV_FILE = os.path.join(ROOT, ENV_FILE)
# Matlab Coverage Analysis
RSSI_COVERAGEEST_METHOD = 'RayTrace' # 'LongleyRice'

# -- scripts --
class DiagnoseData():
    def __init__(self,
                 target_beacon_id=None,
                 real_gateway_id='',
                 beacon_data={},
                 gateway_df={},
                 beacon_df={},
                 override_gateway_height_m=None,
                 start_time=datetime.datetime.now()-datetime.timedelta(seconds=3600),
                 end_time=datetime.datetime.now(),
                 osm_map = None
                 ):
        self.start_end_time = start_time, end_time
        print(self.start_end_time)
        self.OSMFile = osm_map
        self.override_gateway_height_m = override_gateway_height_m
        #
        if target_beacon_id is None:
            self.target_beacon_ids = [id for id in beacon_data]
            self.target_beacon_id = self.target_beacon_ids[0]
        else:
            self.target_beacon_id = target_beacon_id
        self.beacon_data = beacon_data[self.target_beacon_id]
        #
        beacon_df = beacon_df.rename(columns={'measure_value::double': 'rssi'})
        self.beacon_df = beacon_df
        self.gateway_df = gateway_df
        self.gateway_id = real_gateway_id
        #
        self.valid_checks = self.run_checks()
        if VERBOSE:
            print('Validity checks: ', self.valid_checks)

    # --  checks: collated ---
    def run_checks(self):
        # 0: filter on gateway_id, beacon_id
        ret_check = self.apply_mac_filters()
        if not ret_check:
            print('Failed at mac filter')
            return False
        # 1: check start, end time matches gateway_df and beacon_df
        #self.review_start_end_times()
        ret_check = self.apply_time_filters()
        if not ret_check:
            print('Failed at time filter')
            return False
        # 2: check simulated beacon data latlon exist
        ret_check = self.extract_simulated_beacon()
        if not ret_check:
            print('Failed at sim beacon extract filter')
            return False
        return True
    
    # -- checks: specify --
    def extract_simulated_beacon(self):
        latlon = self.beacon_data['latlon']
        print('Latlon: ', latlon)
        if not isinstance(latlon, tuple):
            return False
        lat, lon = latlon
        self.simulated_beacon = {'mac': self.beacon_data['mac'],
                                 'lonlat': (float(lon), float(lat))
                                 }
        return True
    
    def apply_mac_filters(self):
        self.real_beacon_id = self.beacon_data['real_mac']
        self.real_gateway_df = self.gateway_df[self.gateway_df.gateway==self.gateway_id]
        if self.real_gateway_df.shape[0]==0:
            return False
        this_beacon_df = self.beacon_df[self.beacon_df.gateway==self.gateway_id]
        if this_beacon_df.shape[0]==0:
            return False
        self.real_beacon_df  = this_beacon_df[this_beacon_df.beacon==self.real_beacon_id]
        if self.real_beacon_df.shape[0]==0:
            return False
        return True
    
    def review_start_end_times(self):
        start_time, end_time = self.start_end_time
        print(start_time, end_time)
        new_start_time = max([self.real_gateway_df.time.min(), self.real_beacon_df.time.min()])
        if new_start_time.time() > start_time:
            if VERBOSE:
                print('Generate new test start time:', new_start_time)
            new_start_time = new_start_time.time()
        #
        new_end_time = min([self.real_gateway_df.time.max(), self.real_beacon_df.time.max()])
        if new_end_time.time() < end_time:
            if VERBOSE:
                print('Generate new test end time:', new_start_time)            
            new_end_time = new_end_time.time()
        self.tight_start_end_time = new_start_time, new_end_time
    
    def apply_time_filters(self):
        start_time, end_time = self.start_end_time
        print(start_time, end_time)
        if VERBOSE:
            print(f'Analysis from {start_time} to {end_time}')
        gateway_df = self.real_gateway_df
        gateway_df['valid'] = gateway_df.time.apply(lambda x: start_time<=x.time()<=end_time)
        self.real_gateway_df = gateway_df[gateway_df['valid']]
        print('Gateway start, end times: ', self.real_gateway_df.time.min(), self.real_gateway_df.time.max())
        if self.real_gateway_df.shape[0]==0:
            return False
        #
        beacon_df = self.real_beacon_df
        beacon_df['valid'] = beacon_df.time.apply(lambda x: start_time<=x.time()<=end_time)
        self.real_beacon_df = beacon_df[beacon_df['valid']]
        print('Beacon start, end times: ', self.real_beacon_df.time.min(), self.real_beacon_df.time.max())
        if self.real_beacon_df.shape[0]==0:
            return False
        return True
    
    # --- calc from matlab ---

    # -- OSM Map Based coverage calcs --
    def calc_osm_coverage(self, 
                          coverage_method = RSSI_COVERAGEEST_METHOD):
        #
        beacon_tx = self.beacon_tx
        #
        eng = self.matlab_engine
        self.site_viewer = eng.siteviewer('Buildings', self.OSMFile,
                                            'Basemap', "topographic")
        if coverage_method == 'LongleyRice':
            model_name  = "longley-rice"
            var_name  = "SituationVariabilityTolerance"
            rtLongleyRice_SitVarMid= eng.propagationModel(model_name, var_name, 0.55)
            #
            data_out  = eng.coverage(beacon_tx, rtLongleyRice_SitVarMid,
                                    'MaxRange', 250)
            #bcn_data_field = f'{beacon_name}_mdl'
            #eng.workspace[bcn_data_field] = data_out
            eng.workspace["this_mdl"] = data_out
            #
            osm_rssi_data = {'Latitude': [], 'Longitude': [], 'Power': []}
            osm_rssi_data['Latitude'] = matutil.util_conv_matlab_arr_to_list(eng.eval("this_mdl.Data.Latitude"))
            osm_rssi_data['Longitude'] = matutil.util_conv_matlab_arr_to_list(eng.eval("this_mdl.Data.Longitude"))
            osm_rssi_data['Power'] = matutil.util_conv_matlab_arr_to_list(eng.eval("this_mdl.Data.Power"))
            osm_rssi_data_df = pd.DataFrame.from_dict(osm_rssi_data)
            self.osm_rssi_data[self.target_beacon_id] = osm_rssi_data_df
        else:
            self.propagation_model = eng.propagationModel("raytracing",
                                                          "Method", "sbr",
                                                          "AngularSeparation", "low",
                                                          "MaxNumReflections", 0,
                                                          "SurfaceMaterial", "concrete")
    #
    def calc_rssi_from_osm_coverage(self,
                                    gateway_rx,
                                    coverage_method=RSSI_COVERAGEEST_METHOD):
        #
        eng = self.matlab_engine
        viewer = self.site_viewer
        pm = self.propagation_model
        eng.workspace['rx'] = gateway_rx
        beacon_tx = self.beacon_tx
        #
        rx_lonlat = eng.eval("rx.Longitude"), eng.eval("rx.Latitude")
        if coverage_method == 'LongleyRice':
            # estimated ss
            this_df = self.osm_rssi_data[self.target_beacon_id]
            this_df['rx_dist'] = this_df.apply(lambda x: util.haversine_distance((x.Longitude, x.Latitude), rx_lonlat), axis=1)
            nearest_idx = this_df['rx_dist'].argmin()
            nearest_data = this_df.iloc[nearest_idx]
            #
            rssi = nearest_data.Power
            # placeholder
            rssi_pm = (5, 5)
        else:
            los = eng.los(beacon_tx, gateway_rx, "Map", viewer)
            rays = eng.raytrace(beacon_tx, gateway_rx, pm, "Map", viewer)
            if los:
                eng.workspace["rays"] = rays[0]
                rssi = self.TxPower-eng.eval("rays.PathLoss")
                # placeholder
                rssi_pm = (5, 5)    # in dBm
            else:
                rssi = np.nan
                # placeholder
                rssi_pm = (np.nan, np.nan)
        return rssi, rssi_pm
  
    # -- simulated beacon data --
    def calc_rxtx(self, row):
        glonlat = row.longitude, row.latitude
        rx = self.get_rx(lonlat=glonlat)
        if self.OSMFile is not None:
            res, res_var = self.calc_rssi_from_osm_coverage(rx)
        else:
            res, res_var = self.calc_txsignal(rx, self.beacon_tx)
        return res, res_var

    def create_simulated_beacon_data(self):
        st0  = time.time()
        # initialise
        self.init_rxtx_calcs()
        et1  = time.time()
        if VERBOSE:
            print(f'time: {et1-st0:.1f}secs')
        #
        beacon_lonlat = self.simulated_beacon['lonlat']
        self.beacon_tx = self.get_tx(lonlat=beacon_lonlat)
        #
        if self.OSMFile is not None:
            self.calc_osm_coverage()
        #
        et2  = time.time()
        if VERBOSE:
            print(f'time: {et2-et1:.1f}secs')
        # compute rssi
        sim_unified_df = self.real_gateway_df
        sim_unified_df['sim_id'] = self.target_beacon_id
        sim_unified_df['sim_mac'] = self.simulated_beacon['mac']
        sim_unified_df['rssi_var_tuple'] = sim_unified_df.apply(lambda x: self.calc_rxtx(x), axis=1)
        et3  = time.time()
        if VERBOSE:
            print(f'time: {et3-et2:.1f}secs')
        sim_unified_df['rssi_val'] = sim_unified_df.rssi_var_tuple.apply(lambda x: x[0] if np.isnan(x[0])
                                                                                        else x[0] + np.random.random_integers(low=-x[1][0], high=x[1][1]))
        sim_unified_df['rssi'] = sim_unified_df.rssi_val.apply(lambda x: x if x > self.config['RSSI_THRESH'] else np.nan)
        et4  = time.time()
        if VERBOSE:
            print(f'time: {et4-et3:.1f}secs')
        self.simbeacon_realgw_df = sim_unified_df
        if VERBOSE:
            print('Size of simulated dataframe', sim_unified_df.shape)
            print(sim_unified_df.head())
    
    # -- evaluate : real vs sims data
    def compare_sims_vs_real(self):
        # visualise - timeseries
        sim_df = self.simbeacon_realgw_df
        real_df = self.real_beacon_df
        #
        fig, axes = plt.subplots(nrows=2, figsize=(15,10), sharex=True, sharey=True)
        ax = axes[0]
        ax.plot(sim_df.time, sim_df.rssi, 'r', marker='o')
        ax = axes[1]
        ax.plot(real_df.time, real_df.rssi, 'k', marker='o')
        # visualise - stats
        # report analysis
        plt.show()

    # -- BLE signal generation --    
    def init_rxtx_calcs(self):
        self.setup_ble()
        self.start_matlab()

    def setup_ble(self, env_path=ENV_FILE):
        self.config = toml.load(env_path)
        self.ble_tx_settings()

    def ble_tx_settings(self):
        # power
        channel = self.config['channel']
        self.TxFreq = self.config['freqChannelHz'][channel]
        # power
        dBm = self.config['power_dBm']
        self.TxPower = conv.convert_dbm_watts(dBm)
        self.SystemLoss = self.config['systemLoss']

    def start_matlab(self):
        self.matlab_engine = matlab.engine.start_matlab()
        self.osm_rssi_data = {}
        # self.OSMFile = os.path.join(self.config['ROOT'], self.config['OSM_Map'])

    def close_matlab_engine(self):
        self.matlab_engine.quit()

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
        if self.override_gateway_height_m is None:
            gateway_RxHeight = self.config['RxHeight']
        else:
            gateway_RxHeight = self.override_gateway_height_m
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
        ss_fs_Mid = ss_LR1
        ss_fs_low = ss_LR0
        ss_fs_hi = ss_LR2
        ss_err_rel_pm = (ss_fs_Mid-ss_fs_low, ss_fs_hi-ss_fs_Mid)        
        return ss_fs_Mid, ss_err_rel_pm
            
    def calc_fs(self, rx, tx, prop_model):
        eng = self.matlab_engine
        fc = 10e9
        pm = prop_model
        wavelength = eng.physconst('LightSpeed')/fc
        pl0 = eng.fspl(eng.distance(rx, tx), wavelength)
        ss = eng.sigstrength(rx, tx, pm)
        pl = -pl0*0.9 + 0.1*ss
        return pl

# ------------------------------
def import_data(folder_path=DEFAULT_IMPORT_DATAPATH, filename=None):
    if filename is not None:
        import_file = os.path.join(folder_path, filename)
        df = pd.read_csv(import_file)
        df['time'] = pd.to_datetime(df['time'])
        df['time'] = df['time'].dt.tz_localize(datetime.timezone.utc)
        df = df.sort_values(by=['time'], ascending=True)
        return df
    else :
        return None

#----
def main(folder_path=DEFAULT_IMPORT_DATAPATH):
    #
    filenames = os.listdir(folder_path)
    #
    gateway_filenames = [f for f in filenames if 'gateway' in f]
    gateway_filename = sorted(gateway_filenames)[1]
    gateway_df = import_data(filename=gateway_filename)
    gateway_list = list(gateway_df.gateway.unique())
    sel_gateway = [g for g in gateway_list if 'c6a6' in g][0]
    #
    beacon_filenames = [f for f in filenames if 'beacon' in f]
    beacon_filename = sorted(beacon_filenames)[1]
    beacon_df = import_data(filename=beacon_filename)
    # Show            
    print('Testing ...')
    print(f'Gateway: {gateway_filename}, Beacon: {beacon_filename}, Gateway ID={sel_gateway}')
    #
    beacon_test_data = {}
    beacon_test_data['1'] = {'name': 'Test-Beacon-1',
                        'mac': 'BC7465737401',
                        'latlon': (str(54.93485), str(9.26023)),
                        'real_mac': 'BC5729059804'
                        }
    start_time = max([gateway_df.time.min(), beacon_df.time.min()])
    end_time = start_time + datetime.timedelta(seconds=3600)
    start_end_time = start_time.time(), end_time.time()
    return sel_gateway, beacon_test_data, gateway_df, beacon_df, start_end_time

if __name__ == "__main__":
    st = time.time()
    sel_gateway, beacon_test_data, gateway_df, beacon_df, start_end_time = main()
    start_time, end_time = start_end_time    
    diagnostics = DiagnoseData(real_gateway_id=sel_gateway,
                                beacon_data=beacon_test_data,
                                gateway_df=gateway_df,
                                beacon_df=beacon_df,
                                start_time=start_time,
                                end_time=end_time
                                )
    et0 = time.time()
    print(f'Time taken={et0-st:.1f}secs')
    diagnostics.create_simulated_beacon_data()
    et1 = time.time()
    print(f'Time taken={et1-et0:.1f}secs')
    diagnostics.compare_sims_vs_real()