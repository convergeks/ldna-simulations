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
DEFAULT_IMPORT_DATAPATH = os.path.join(ROOT, "dump/datasource/dump/OxHum_Interactions/datasource/raw/Interactions")

# env
ENV_FILE = 'env/config.toml'
ENV_FILE = os.path.join(ROOT, ENV_FILE)
# Matlab Coverage Analysis
RSSI_COVERAGEEST_METHOD = 'RayTrace' # 'LongleyRice'

# -- scripts ---
class InteractionAnalysis():
    def __init__(self,
                 target_beacon_id=None,
                 real_gateway_id='',
                 beacon_data={},
                 gateway_df={},
                 beacon_df={},
                 start_time=datetime.datetime.now()-datetime.timedelta(seconds=3600),
                 end_time=datetime.datetime.now(),
                 osm_map=None
                 ):
        self.start_end_datetime = start_time, end_time
        self.start_end_time = start_time.time(), end_time.time()
        self.OSMFile = osm_map
        print(self.start_end_time)
        #
        if target_beacon_id is None:
            self.target_beacon_ids = [id for id in beacon_data]
            self.target_beacon_id = self.target_beacon_ids[0]
        else:
            self.target_beacon_id = target_beacon_id
        # TODO: Placeholder
        self.beacon_data = beacon_data  #[self.target_beacon_id]
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
        #
        return True
        
    # -- checks: specify --
    def apply_mac_filters(self):
        self.real_beacon_id = self.target_beacon_id
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
        gateway_df = gateway_df.sort_values(by=['time'], ascending=True)
        self.real_gateway_df = gateway_df[gateway_df['valid']]
        print('Gateway start, end times: ', self.real_gateway_df.time.min(), self.real_gateway_df.time.max())
        if self.real_gateway_df.shape[0]==0:
            return False
        #
        beacon_df = self.real_beacon_df
        beacon_df['valid'] = beacon_df.time.apply(lambda x: start_time<=x.time()<=end_time)
        beacon_df = beacon_df.sort_values(by=['time'], ascending=True)
        self.real_beacon_df = beacon_df[beacon_df['valid']]
        print('Beacon start, end times: ', self.real_beacon_df.time.min(), self.real_beacon_df.time.max())
        if self.real_beacon_df.shape[0]==0:
            return False
        return True

# -- evaluate : review real data
# TODO: Compare vs sims data
    def review_real(self):
        # visualise - timeseries
        real_df = self.real_beacon_df
        #
        fig, axes = plt.subplots(nrows=1, figsize=(15,6))
        ax = axes
        ax.plot(real_df.time, real_df.rssi, 'k', marker='o')
        plt.show()


# -------- Testing ------------
def import_testdata():
    folder_path = DEFAULT_IMPORT_DATAPATH
    gw_filename = 'data_gateway_fromS3_20231017.csv'
    bcn_filename = 'data_beacon_fromS3_20231017.csv'
    #
    import_gateway_file = os.path.join(folder_path, gw_filename)
    import_beacon_file = os.path.join(folder_path, bcn_filename)
    #
    gateway_df = pd.read_csv(import_gateway_file)
    gateway_df['time'] = pd.to_datetime(gateway_df['time'])
    gateway_df['time'] = gateway_df['time'].dt.tz_localize(datetime.timezone.utc)
    #
    beacon_df = pd.read_csv(import_beacon_file)
    beacon_df['time'] = pd.to_datetime(beacon_df['time'])
    beacon_df['time'] = beacon_df['time'].dt.tz_localize(datetime.timezone.utc)
    data_df_map = {'gateway': gateway_df, 'beacon': beacon_df}
    return data_df_map


def run_tests():
    print('Review data...')
    data_df_map = import_testdata()
    gateway_df = data_df_map['gateway']
    beacon_df = data_df_map['beacon']
    #
    test_date = datetime.date(2023, 10, 17)
    UTC_TZ = datetime.timezone.utc
    start_time = datetime.time(8, 0, 0, 0, tzinfo=UTC_TZ)
    end_time = datetime.time(9, 0, 0, 0, tzinfo=UTC_TZ)
    start_datetime = datetime.datetime.combine(test_date, start_time)
    end_datetime = datetime.datetime.combine(test_date, end_time)
    #
    real_beacon_id = 'BC572903D750'
    sel_gateway_id = 'e00fce68b2a1e04ccbe11a63'
    osm_map = None
    #
    interactions = InteractionAnalysis(
                                        target_beacon_id=real_beacon_id,
                                        real_gateway_id=sel_gateway_id,
                                        beacon_data={}, #TODO - Extract simulated info
                                        gateway_df=gateway_df,
                                        beacon_df=beacon_df,
                                        start_time=start_datetime,
                                        end_time=end_datetime,
                                        osm_map=osm_map
                                        )
    print(interactions.valid_checks)


if __name__ == "__main__":
    run_tests()