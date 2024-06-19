import os, sys
import toml
import pickle
from pathlib import Path
import time
import ast
#
import folium
import streamlit as st
from streamlit_folium import st_folium, folium_static
import streamlit.components.v1 as st_components
#
import pandas as pd
import numpy as np
#
import datetime as dt
from datetime import datetime
import pytz 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpld3
import plotly.graph_objects as go
import plotly.express as px
# import local code
# import sim_p0_setupdevices as p0
# import sim_p1_generate_rxtxdata as p1
# import sim_p1x1_diagnose_rxtxdata as p1x1
import sim_p1x2_interactions_rxtxdata as p1x2
#
import utils.util_osm_tools as uosm
import utils.util_routine_calcs as calcs
#
SRC_PATH = '/home/kiran/source/Working/testingTools/ks-dev-tools/conDNA-tools/lift-history-tools/src'
sys.path.append(SRC_PATH)
import p1_read_output_mapped_data as p1_unit
SRC_PATH = os.path.join(SRC_PATH, 'sitesettings')
sys.path.append(SRC_PATH)
import gateways as gwdata

# -- options --
INTIMG_PATH = '/home/kiran/source/Working/logisticsDNA/ldna-simulator/simulator_dev/dump/datasource/Interactions'
INTIMG_FILE = 'interaction_statemachine.png'
SITE = 'Tinglev'#'Oxford'

# Repeated Settings- need settings config
GW_ICON_TYPES = {'start': 'play',
                 'mid': 'forward',
                 'end': 'stop'
                }
GW_ICON_CLRS = {'start': 'black',
                 'mid': 'green',
                 'end': 'red'
                }
BEACON_ICON_TYPES = ['cog']
DEV_CLR = {'Gateway': 'green', 'Beacon': 'blue'}

# -- scripts --
class sim_interaction_analysis(object):
    def __init__(self, session_state):
        self.session_state = session_state
        data_df = session_state['imported_data']
        self.imported_data = data_df
        # data
        self.gateway_df = data_df['gateway']
        self.gateway_list = sorted(self.gateway_df.gateway.unique())
        self.beacon_df = data_df['beacon']
        self.beacon_list = sorted(self.beacon_df.beacon.unique())
        self.beacon_list = [bid for bid in self.beacon_list if bid[:2] in ['DD', 'BC']]
        #units
        self.unit_df = p1_unit.read_unitmacdb()
        all_beacon_unit_map = self.unit_df.set_index('beacon').to_dict()['Unit_ID']
        self.beacon_unit_map = {mac: all_beacon_unit_map[mac] for mac in self.beacon_list if mac in all_beacon_unit_map }
        self.bcn_name_inv = {v:k for k, v in self.beacon_unit_map.items()}
        st.write(self.imported_data.keys())
        # gateways
        all_site_gateway_name_map = gwdata.extid_name_map[SITE]
        print(self.gateway_list)
        self.gateway_name_map = {mac: all_site_gateway_name_map[mac] for mac in self.gateway_list 
                                                                        if mac in all_site_gateway_name_map}
        #
        self.gw_name_inv = {v:k for k, v in self.gateway_name_map.items()}
        # init
        self.session_state['conversion_key'] = {}


    def setup_ui(self):
        # ui
        self.import_col1, self.import_col2 = st.columns(2)

    # def setup_vis(self):
    #     # vis
    #     self.draw_col1, self.draw_col2 = st.columns(2)

    # -- define key session data --
    def define_session_info(self):
         self.key_list = {'beacon': 'int_sel_bcn_mac',
                          'gateway': 'int_sel_gw_name',
                          'sim_date': 'sim_date',
                          'state': ['state_alpha_st', 'state_beta', 'state_gamma', 'state_delta', 'state_mu', 'state_nu', 'state_alpha_end'],
                          'locations': ['int_gw_start_loc', 'int_start_latlon', 'int_end_latlon'],
                          'loc_times': ['int_gw_start_time', 'int_start_time', 'int_end_time'],
                          'antenna_hts': ['ant_bcn_low', 'ant_bcn_hi', 'ant_gw_low', 'ant_gw_hi'],
                          'hook_times': ['A_start', 'A_end', 'B_start', 'B_end', 'C_start', 'C_end', 'D_start', 'D_end']
                         }
         self.key_name_map = {
                                'int_gw_start_time': 'Gateway Move to Start time',
                                'int_start_time': 'Lift Start time',
                                'int_end_time': 'Lift End time',
                                'A_start': 'A: Start time',
                                'A_end': 'A: End time',
                                'B_start': 'B: Start time',
                                'B_end': 'B: End time',
                                'C_start': 'C: Start time',
                                'C_end': 'C: End time',
                                'D_start': 'D: Start time',
                                'D_end': 'D: End time'
                             }
         
    # find times
    def iterate_for_times(self, df, target_time, st_or_end='start'):
        count = 0
        if df.shape[0] == 0:
            print('Warning: empty gateway nothing to find')
            return df
        # 
        this_df = df
        tm = target_time
        while (count == 0) or (this_df.shape[0]==0):
            if st_or_end == 'end':
                end_int = tm + dt.timedelta(seconds=(count+1)*10)
                this_df = df[df.time <= end_int]
            else:
                st_int = tm - dt.timedelta(seconds=(count+1)*10)
                this_df = df[df.time >= st_int]
            count += 1
            if count > 12:
                print('Warning: nothing found within 2mins')
                break
        return this_df
    
    # find locations at times
    def find_locations(self):
        loc_data = {
                    'locations': ['int_gw_start_loc',
                                  'int_start_latlon',
                                  'int_end_latlon'
                                  ],
                    'loc_times': ['int_gw_start_time',
                                  'int_start_time',
                                  'int_end_time'
                                  ],
                    }
        for fld_idx, fld_tm in enumerate(loc_data['loc_times']):
            #
            loc_fld = loc_data['locations'][fld_idx]
            #
            tm_str = st.session_state[fld_tm]            
            tm_val = self.util_convert_times(tm_str)
            this_date = self.test_date
            tm = datetime.combine(this_date, tm_val)
            #
            st_int = tm - dt.timedelta(seconds=10)
            end_int = tm + dt.timedelta(seconds=10)
            #
            gw0 = self.gw_df
            gw1 = self.iterate_for_times(gw0,
                                         st_int,
                                         st_or_end='start')
            if gw1.shape[0]>0:
                gw2 = self.iterate_for_times(gw1,
                                             end_int,
                                             st_or_end='end')
                gw2['delta_time'] = gw2.time.apply(lambda x: abs((x-tm).total_seconds()))
                closest_index = gw2.delta_time.argmin()
                #latlon = gw2.latitude.mean(), gw2.longitude.mean()
                latlon = gw2.latitude.iloc[closest_index], gw2.longitude.iloc[closest_index]
                st.session_state[loc_fld] = f'({latlon[0]:.6f},{latlon[1]:.6f})'
                # st.write(f'{loc_fld}: {st.session_state[loc_fld]}')
            else:
                print('Problem, check on gps at', tm_str)
                continue

    # -- specify data --
    def specify_antenna_data(self):
        # heights
        st.session_state['ant_bcn_low'] = '1'
        st.session_state['ant_bcn_hi'] = '10'
        st.session_state['ant_gw_low'] = '3'
        st.session_state['ant_gw_hi'] = '13'
        # times
        hhmmss = "08:00:00"#"hh:mm:ss"
        hhmmss_Ast, hhmmss_Aend = hhmmss, hhmmss
        hhmmss_Bst, hhmmss_Bend = hhmmss, hhmmss
        hhmmss_Cst, hhmmss_Cend = hhmmss, hhmmss
        hhmmss_Dst, hhmmss_Dend = hhmmss, hhmmss
        st.session_state['A_start'] = hhmmss
        st.session_state['A_end'] = hhmmss
        st.session_state['B_start'] = hhmmss
        st.session_state['B_end'] = hhmmss
        st.session_state['C_start'] = hhmmss
        st.session_state['C_end'] = hhmmss
        st.session_state['D_start'] = hhmmss
        st.session_state['D_end'] = hhmmss

    # -- apply conversion --
    def apply_conversions(self):
        fields = {'locations': 'loc',
                  'loc_times':'time',
                  'hook_times': 'time',
                  'antenna_hts': 'float'
                  }
        #
        for fld, fldtype in fields.items():
            for this_key in self.key_list[fld]:
                if not this_key in st.session_state:
                    continue
                ss_str = st.session_state[this_key]
                if fldtype == 'time':
                    ss_val = self.util_convert_times(ss_str)
                elif fldtype == 'loc':
                    ss_val = self.util_convert_locations(ss_str)
                elif fldtype == 'float':
                    ss_val = self.util_convert_str_to_float(ss_str)
                self.session_state['conversion_key'][this_key] = ss_val
        return True

    # -- define conversions --
    def util_convert_times(self, str_hhmmss):
        if ':' not in str_hhmmss:
            print('Warning, wrong format. Looking for hh:mm:ss')
            return False
        time_list = str.split(str_hhmmss, ':')
        hh = int(time_list[0])
        mm = int(time_list[1])
        ss = int(time_list[2])
        this_time = dt.time(hh, mm, ss, 0, tzinfo=dt.timezone.utc)
        return this_time
    
    def util_convert_locations(self, str_latlon):
        if ',' not in str_latlon:
            print('Warning, wrong format. Looking for (Lat,Lon)')
            return False
        lat, lon = ast.literal_eval(str_latlon)
        return lon, lat
    
    def util_convert_str_to_float(self, str_num):
        return np.float(str_num)
        
    def ui_gateway(self):
        # capture info
        # gw
        with self.import_col1:
            st.subheader('Lifted Beacon')
            bcn_col1, bcn_col2 = st.columns(2)
            with bcn_col1:
                this_beacon_name = st.selectbox('Select lifted beacon',
                                              sorted(list(self.beacon_unit_map.values())),
                                              key='int_sel_bcn_mac',
                                              index=0)
                self.real_beacon_id = self.bcn_name_inv[this_beacon_name]
            with bcn_col2:
                st.text('Beacon MAC')
                st.write(self.real_beacon_id)
            #                
            st.subheader('Lifting Gateway ')
            st.checkbox('Show lift details', False, key='lift_details')
            gateway_col1, gateway_col2 = st.columns(2)
            with gateway_col1:
                this_gateway_name = st.selectbox('Select lifting gateway',
                                               sorted(list(self.gateway_name_map.values())),
                                               key='int_sel_gw_name',
                                               index=0)
                this_gateway_id = self.gw_name_inv[this_gateway_name] 
                self.sel_gateway_id = this_gateway_id
                st.text('Gateway MAC')
                st.write(self.sel_gateway_id)
                #
                self.gw_df = self.gateway_df[self.gateway_df.gateway==this_gateway_id]
                self.test_date = self.gateway_df.time.mean().date()
                if st.session_state['lift_details']:
                    if ('int_sel_gw_type' in st.session_state) and (st.session_state['int_sel_gw_type']):
                        sm_file = os.path.join(INTIMG_PATH, INTIMG_FILE)
                        st.image(sm_file)
            #
            gw_types = ['tower crane', 'gantry crane', 'forklift']
            with gateway_col2:
                self.gateway_type = st.selectbox('Gateway type', gw_types,
                                                 key='int_sel_gw_type',
                                                 index=0)
                # if self.gateway_type == 'tower crane':
                if st.session_state['lift_details']:
                    st.checkbox('alpha (start)', True, 'state_alpha_st')
                    st.checkbox('beta', True, 'state_beta')
                    st.checkbox('gamma', True, 'state_gamma')
                    st.checkbox('delta', True, 'state_delta')
                    st.checkbox('mu', True, 'state_mu')
                    st.checkbox('nu', True, 'state_nu')
                    st.checkbox('alpha (end)', True, 'state_alpha_end')
        # state machine
        with self.import_col2:
            # defaults:
            hhmmss = "08:00:00"#"hh:mm:ss"
            hhmmss_gwst = hhmmss
            hhmmss_ls = "08:00:00"
            hhmmss_le = "09:00:00"
            #
            st.subheader('Location Information')
            #
            # if st.session_state['state_alpha_st']:
            gw_sttime_col, gw_stloc_col = st.columns(2)
            with gw_stloc_col:
                if ('int_gw_start_loc' not in st.session_state):
                    st.session_state['int_gw_start_loc'] = '(0,0)'
                st.text('Gateway Move to Start Loc(Lat, Lon):')
                st.write(st.session_state['int_gw_start_loc'])
            with gw_sttime_col:
                st.text_input('Gateway Move to Start time',  hhmmss_gwst, key='int_gw_start_time')
            #
            sttime_col, stloc_col  = st.columns(2)
            with stloc_col:
                if ('int_start_latlon' not in st.session_state):
                    st.session_state['int_start_latlon'] = '(0,0)'
                st.text('Beacon Lift Location (Lat, Lon)')
                st.write(st.session_state['int_start_latlon'])
            with sttime_col:
                st.text_input('Lift Start time',  hhmmss_ls, key='int_start_time')
            endtime_col, endloc_col = st.columns(2)
            with endloc_col:
                if ('int_end_latlon' not in st.session_state):
                    st.session_state['int_end_latlon'] = '(0,0)'
                st.text('Beacon Drop Location (Lat, Lon)')
                st.write(st.session_state['int_end_latlon'])
            with endtime_col:
                st.text_input('Lift End time', hhmmss_le, key='int_end_time')
            #
            self.specify_antenna_data()
            if st.checkbox('Specify antenna data'):
                st.subheader('Antenna Heights')
                lobcn_col1, hibcn_col2 = st.columns(2)
                with lobcn_col1:
                    st.text_input('Beacon min height', st.session_state['ant_bcn_low'])
                with hibcn_col2:
                    st.text_input('Beacon max height', st.session_state['ant_bcn_hi'])
                logw_col1, higw_col2 = st.columns(2)
                with logw_col1:
                    st.text_input('Gateway min height', st.session_state['ant_gw_low'])
                with higw_col2:
                    st.text_input('Gateway max height', st.session_state['ant_gw_hi'])
                # Hook times
                st.subheader('Antenna Hook Time')
                A_start_col1, A_end_col2 = st.columns(2)
                with A_start_col1:
                    st.text_input('A: Start time', st.session_state['A_start'])
                with A_end_col2:
                    st.text_input('A: End time',  st.session_state['A_end'])
                B_start_col1, B_end_col2 = st.columns(2)
                with B_start_col1:
                    st.text_input('B: Start time',  st.session_state['B_start'])
                with B_end_col2:
                    st.text_input('B: End time', st.session_state['B_end'])
                C_start_col1, C_end_col2 = st.columns(2)
                with C_start_col1:
                    st.text_input('C: Start time',  st.session_state['C_start'])
                with C_end_col2:
                    st.text_input('C: End time',  st.session_state['C_end'])
                D_start_col1, D_end_col2 = st.columns(2)
                with D_start_col1:
                    st.text_input('D: Start time',  st.session_state['D_start'])
                with D_end_col2:
                    st.text_input('D: End time',  st.session_state['D_end'])
                #
    
    def show_gps_map(self, draw_count=0):
        interaction_anls = st.session_state['interactions']
        gps_interval_df = interaction_anls.real_gateway_df
        if gps_interval_df.shape[0]==0:
            st.write('Warning: No data to show')
            return
        #
        #TODO:
        # 1. Retrieve markers at specified intervals and draw them
        #
        cols = gps_interval_df.columns
        gps_interval_df = gps_interval_df.reset_index()
        gps_interval_df = gps_interval_df[cols]
        LATLON = 51.7600, -1.2637
        #gps_interval_df.latitude.mean(), gps_interval_df.longitude.mean()
        #
        # with self.draw_col1:
        st.select_slider('Zoom Levels', options=[i for i in range(15, 20)], value=17, key='int_zoom_level_import')
        m = folium.Map(location=list(LATLON),
                       max_zoom=20,
                       zoom_start=st.session_state['int_zoom_level_import'],
                       width=700) #, zoom_control=True)
        if self.session_state['osm_map_load'] and len(self.session_state['osm_data']):
            pbffilename = self.session_state['osm_data']['osm_pbf']
            buildings = uosm.util_osm_get_buildings(pbffilename)
            folium.GeoJson(buildings).add_to(m)
        #
        N = gps_interval_df.shape[0]
        num_label = {idx: 'start' if (idx == 0) else 'end' if idx==(N-1) else 'mid' for idx in range(N)}            
        coordinates = []
        for idx, row in gps_interval_df.iterrows():
            this_latlon = row.latitude, row.longitude
            coordinates.append(list(this_latlon))
            icon_type = GW_ICON_TYPES[num_label[idx]]
            icon_clr = GW_ICON_CLRS[num_label[idx]]
            this_hhmmss = str(row.time.time())[:8]
            this_date = row.time.date().strftime('%B %d')
            ttl = f'{this_date}@{this_hhmmss}, <br> Marker{idx} <br> ID={self.sel_gateway_id}'
            clr = DEV_CLR['Gateway']
            if num_label[idx] != 'mid':
                this_icon=folium.Icon(color=icon_clr, icon=icon_type)
                folium.Marker(
                            list(this_latlon),  icon=this_icon,
                            tooltip=ttl, #this_latlon,
                            popup=this_latlon
                            ).add_to(m)
            else:
                folium.CircleMarker(list(this_latlon),
                                    raidus=10,
                                    opacity=0.5,
                                    color=clr,
                                    fill_color=clr,
                                    popup=this_latlon,
                                    tooltip=ttl
                                    ).add_to(m)
            m.add_child(folium.LatLngPopup())
        #
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            st.checkbox('Show Connector', key='show_connector')
        with sel_col2:
            st.checkbox('Show Static Markers', key='show_static_text')
        if st.session_state['show_connector']:
            if len(coordinates)>0:
                folium.PolyLine(
                        locations=coordinates,
                        color="#FF0000",
                        weight=2,
                        tooltip="Connect the dots",
                        ).add_to(m)
                
        # for native zoom                
        # folium.TileLayer(name='OpenStreetMap', max_native_zoom=30).add_to(m)
        #
        mapdata = st_folium(m, height=1400, width=1400)
        #
        st.session_state['interaction_folium_map'] = m
        st.session_state['interaction_mapdata'] = mapdata
        # show table
        self.gps_interval_df = gps_interval_df
        self.calc_gateway_move()
        st.write(self.gps_interval_df)
        #
        return (draw_count+1)
    
    def calc_gateway_move(self):
        gps_df = self.gps_interval_df
        cols = list(gps_df.columns)
        #
        gps_df['dtime'] = gps_df.time.diff().dt.total_seconds().fillna(1)
        gps_df = gps_df[gps_df.dtime > 0]
        if gps_df.shape[0] == 0:
            return
        gps_df['dlat'] = gps_df.latitude.diff()
        gps_df['dlon'] = gps_df.latitude.diff()
        gps_df = gps_df.dropna()
        if gps_df.shape[0] == 0:
            return
        gps_df['rate_coords'] = gps_df.apply(lambda x:
                                                    (x.dlon, x.dlat), axis=1)
        # gps_df['rate_coords'] = gps_df.apply(lambda x:
        #                                             (x.dlon/x.dtime, x.dlat/x.dtime), axis=1)
        gps_df['rate'] = gps_df.apply(lambda x: calcs.haversine_distance((0,0), x.rate_coords)/x.dtime, axis=1)
        #
        cols.append('rate')
        self.gps_interval_df = gps_df[cols]


    def show_rssi_plots(self):
        interaction_anls = st.session_state['interactions']
        gps_interval_df = interaction_anls.real_gateway_df
        if gps_interval_df.shape[0]==0:
            st.write('Warning: No data to show')
            return
        #
        opt_col1, opt_col2, opt_col3 = st.columns(3)
        # compare init
        with opt_col1:
            st.checkbox('Add more beacons', key='int_compare_beacons')
            if st.session_state['int_compare_beacons']:        
                st.text_input('apply filters', key='int_compare_filter')
                if ',' in st.session_state['int_compare_filter']:
                    compare_beacon_list = [mac for mac, uid in self.beacon_unit_map.items() 
                                            if uid in st.session_state['int_compare_filter']]
                else:
                    compare_beacon_list = [mac for mac, uid in self.beacon_unit_map.items()
                                        if st.session_state['int_compare_filter'] in uid]
                st.write([self.beacon_unit_map[m] for m in compare_beacon_list])
                st.session_state['int_compare_list'] = compare_beacon_list
            else:
                st.session_state['int_compare_list'] = []
        with opt_col2:
            st.checkbox('Show Simulated Median', key='int_sim_median_curve')
        with opt_col3:
            st.checkbox('Show Interactive Plot', key='int_show_interactive')
        # plot
        this_fig = self.plot_real_rssi(compare_list=st.session_state['int_compare_list'],
                                       interactive = st.session_state['int_show_interactive']
                                       )
        if st.session_state['int_show_interactive']:
            st.plotly_chart(this_fig)
        else:
            st.pyplot(this_fig)


    # -- plotting --
    def plot_real_rssi(self,
                       compare_list=[],
                       interactive=False):
        # 
        interaction_anls = st.session_state['interactions']
        real_df = interaction_anls.real_beacon_df
        real_bid = real_df.beacon.iloc[0]
        real_bname = self.beacon_unit_map[real_bid]
        if len(compare_list)==0:
            fig, axes = plt.subplots(nrows=2, figsize=(15, 10)) #, sharex=True, sharey=True)
        else:
            fig, axes = plt.subplots(nrows=3, figsize=(15, 15), sharex=True, sharey=False)
        # Primary Beacon
        this_lbl = f'Primary Element : {real_bname}'
        ax = axes[0]
        if interactive:
            ax.plot(real_df.time, real_df.rssi, 'k', marker='o', alpha=0.65, label=real_bname)
        else:
            ax.plot(real_df.time, real_df.rssi, 'k', marker='o', alpha=0.65, label=real_bname)
        ax.set_title(this_lbl, fontsize=30)
        ax.grid('on')
        ax.set_ylabel('rssi / dBm', fontsize=20)
        # Show gateways
        gps_df = self.gps_interval_df
        this_gwname = self.gateway_name_map[self.sel_gateway_id]
        ax = axes[1]
        if interactive:
            ax.plot(gps_df.time, gps_df.rate, 'k',  marker='o', alpha=0.65, label=this_gwname)
        else:
            ax.plot(gps_df.time, gps_df.rate, 'k', marker='o', alpha=0.65, label=this_gwname)
        ax.set_title(this_gwname, fontsize=30)
        ax.grid('on')
        ax.set_ylabel('rate / mps', fontsize=20)

        # Compared Beacons
        if len(compare_list)>0:
            ax = axes[2]
            clrmap = list('rgbcmy')
            color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
            listed_df = []
            if self.real_beacon_id not in compare_list:
                compare_list.append(self.real_beacon_id)
            for bid in compare_list:
                this_bname = self.beacon_unit_map[bid]
                if bid == self.real_beacon_id:
                    if interactive:
                        ax.plot(real_df.time, real_df.rssi, 'k', alpha=0.65, label=real_bname)
                    else:
                        ax.plot(real_df.time, real_df.rssi, 'k', marker='o', alpha=0.65, label=real_bname)
                    continue
                this_beacon_df = interaction_anls.beacon_df[interaction_anls.beacon_df.gateway==interaction_anls.gateway_id]
                this_df = this_beacon_df[this_beacon_df.beacon==bid]
                # filters
                start_time, end_time = interaction_anls.start_end_time
                this_df['valid'] = this_df.time.apply(lambda x: start_time<=x.time()<=end_time)
                this_df = this_df[this_df['valid']]
                if this_df.shape[0]==0:
                    continue
                # if interactive:
                #     clr = color_list.pop(0)
                #     color_list.append(clr)
                # else:
                clr = clrmap.pop(0)
                clrmap.append(clr)
                #
                this_df = this_df.sort_values(by=['time'], ascending=True)
                this_df['UnitID'] = this_bname
                this_df['colour'] = clr
                listed_df.append(this_df)
                #
                if interactive:
                    this_df.plot.line( x='time', y='rssi', c=clr,
                                       label=this_bname, ax=ax)
                else:
                    ax.plot(this_df.time, this_df.rssi, clr, marker='o', alpha=0.65, label=this_bname)

            if interactive and len(listed_df)>0:
                listed_df = pd.concat(listed_df)


            ax.grid('on')
            ax.set_ylabel('rssi / dBm', fontsize=20)
            # if not interactive:
            ax.legend();
            # ax.set_ylim([-95, -30])
        # if st.session_state['sim_median_curve']:
        #     ax.plot(sim_df.time, sim_df.rssi_av, 'r', color='b')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
        if not interactive:
            plt.setp(ax.get_xticklabels(), rotation=90,
                        fontsize = 20,
                        ha="right");
            plt.setp(ax.get_yticklabels(),
                        fontsize = 20,
                        ha="right");
            self.apply_timelabels(axes)
        return fig
    
    def apply_timelabels(self, axes):
        for ax in axes:
            min_level, max_level = ax.get_ylim()
            for this_label in self.key_list['loc_times']:
                this_time = self.session_state['conversion_key'][this_label]
                this_date = self.test_date
                this_datetime = datetime.combine(this_date, this_time)
                ax.vlines(this_datetime, min_level, max_level, color='r', linestyle=':',
                    alpha=0.995,
                    label='Lift Events')  # The vertical stems.

                xy = (this_datetime, 0.775*max_level)
                this_text = self.key_name_map[this_label]
                ax.annotate(this_text, xy=xy,
                            rotation=90,
                            fontsize=9)

    
    # -- diagnostics --
    def run_diagnostics(self):
        conv_key = self.session_state['conversion_key']
        all_times = [conv_key[key] for key in self.key_list['loc_times'] if key in conv_key]
        #
        start_time = min(all_times)
        start_datetime = datetime.combine(self.test_date, start_time) - dt.timedelta(seconds=10*60)
        #
        end_time = max(all_times)
        end_datetime = datetime.combine(self.test_date, end_time) + dt.timedelta(seconds=10*60)
        #
        if st.checkbox('Review data', key='review_interaction_data'):
            st.write(start_datetime)
            st.write(end_datetime)
        #
        if self.session_state['osm_map_load']:
            osm_map = self.session_state['osm_data']['osm']
        else:
            osm_map = None
        if self.real_beacon_id not in self.beacon_df.beacon.unique():
            return False
        if self.sel_gateway_id not in self.beacon_df.gateway.unique():
            return False
        #
        if st.session_state['review_interaction_data']:
            st.write('Review macs...')
            st.write(self.real_beacon_id, self.sel_gateway_id)
        #
        interactions = p1x2.InteractionAnalysis(
                                                target_beacon_id=self.real_beacon_id,
                                                real_gateway_id=self.sel_gateway_id,
                                                beacon_data={}, #TODO - Extract simulated info
                                                gateway_df=self.gateway_df,
                                                beacon_df=self.beacon_df,
                                                start_time=start_datetime,
                                                end_time=end_datetime,
                                                osm_map=osm_map
                                                )
        st.session_state['interactions'] = interactions
        if st.session_state['review_interaction_data']:
            st.write(interactions.valid_checks)
        return interactions.valid_checks

    # TODO:
    # Ref plan: URL https://miro.com/app/board/uXjVNJyjLp8=/?moveToWidget=3458764590544335652&cot=14
    # 1. Package up all the data and extract lat/lon, times, hts from string to values : Thurs DONE
    # 2. Send to p1x2_interaction_analysis  Thurs
    # ** In p1x2 interaction analysis **
    # 3. Add a beacon at the beacon lift location : Thurs
    # 4. Filter gateway on times : Thurs
    # 5. Simulate the rssi profile : Thurs
    # Diagnostic:
    #  -> 6. Compare against the selected beacon : Thurs
    # Predictive:
    # -> 7. Add several beacons near lift locations and drop location : Thurs
    # Both:
    #  -> 8. Confounding analysis - include all beacons with similar rssi profiles in the period: Thurs
