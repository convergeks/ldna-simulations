import os
import toml
import pickle
from pathlib import Path
#
import folium
import streamlit as st
from streamlit_folium import st_folium, folium_static
#
import pandas as pd
import numpy as np
#
import datetime as dt
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import local code
import sim_p0_setupdevices as p0
import sim_p1_generate_rxtxdata as p1

# paths
ROOT = os.getcwd()
if "sim_core" in ROOT:
    ROOT = os.getcwd().parent
DEFAULT_OUTPATH = os.path.join(ROOT, "dump")
DEFAULT_CSVPATH = os.path.join(ROOT, "dump/csvfiles")

# ----- options -----
# site & dates
SITE = {"CEMC": 0, "Oxford": 1, "Tinglev": 2, "FPMcCann": 3, "Creagh": 4}
#        , 'Oxford Hum.': 1, 'Converge Use': 2}
FILTERS = {"Whole Site": 0}
DEVICES = ['Gateway', 'Beacon']
GATEWAY_ID = 'G1'

# folium markers ref - https://darigak.medium.com/your-guide-to-folium-markers-b9324fc7d65d
# using icons from: https://getbootstrap.com/docs/3.3/components/
#  these icons -more in number, not working: https://fontawesome.com/v4/icons/
GW_ICON_TYPES = {'start': 'play',
                 'mid': 'forward',
                 'end': 'stop'
                }
BEACON_ICON_TYPES = ['cog']
DEV_CLR = {'Gateway': 'green', 'Beacon': 'blue'}
ACTIONS = ['Generate', 'View']

# --- width ---
def _max_width_(percent_width: int = 75):
    max_width_str = f"max-width: {percent_width}%;"
    st.markdown(
        f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """,
        unsafe_allow_html=True,
    )


def set_fontsizes():
    st.markdown(
        """<style>
    div[class*="stToggle"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 40px;
    }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- helper funcs
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def get_site_latlon(site, env_path=p0.ENV_FILE):
    config = toml.load(env_path)
    site_latlon = {item['id']: (item['lat'], item['lon']) for item in config['latlon']}
    latlon = site_latlon[site]
    return latlon

# -- store session state across runs---
def get_stored_sessionfields():
    session_fields = ['site', 'device', 'action', 'prior_site', 'filters', 'clicked_latlon',
                      'show_markers', 'folium_map', 'mapdata', 'results', 'zoom_level',
                      'start_date', 'start_time', 'sim_start_timestamp', 'load_file',
                      'gateway_prefix', 'beacon_prefix'
                     ]
    return session_fields

def get_state_filepath(folder_path=DEFAULT_OUTPATH):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select session file', filenames, key='load_file')
    return folder_path, selected_filename

def set_state_filepath(folder_path=DEFAULT_OUTPATH):
    flname = st.text_input('session filename', '')
    session_file = os.path.join(folder_path, flname)
    return session_file

def save_session_state(this_session_state, session_file):
    #
    session_fields = get_stored_sessionfields()
    session_dict = {key:val for key, val in this_session_state.items() if key in session_fields}
    try:
        with open(session_file, 'wb') as f:
            print("Saving session_state") #, session_dict)
            pickle.dump(session_dict, f)
    except Exception as e:
        print("Error during saving session state:", e)

def load_session_state(VERBOSE=False):
    folder_path, selected_filename = get_state_filepath()
    session_file = os.path.join(folder_path, selected_filename)
    if os.path.exists(session_file):
        try:
            with open(session_file, 'rb') as f:
                loaded_state = pickle.load(f)
                if VERBOSE:
                    print("Loaded session state:", selected_filename) #, loaded_state)
                return loaded_state
        except Exception as e:
            print("Error during loading session state:", e)
            st.write(f'Error in loading file: {session_file}')
    else:
        print(f"{session_file} not found")
    return None

# ---- main streamlit section ----
# set max width
_max_width_(percent_width=100)
# invoke session state
# if "site" not in st.session_state:
#     # # settigs
session_fields = ['site', 'device', 'action', '']
session_action_list = ['Current', 'New/Reset', 'Load Prior']
# displays a file uploaded widget
with st.sidebar:

    # Configuration/State Store and Load
    st.header('Session')
    col1, col2 = st.columns(2)
    with col1:
        st.radio("Session actions", session_action_list, index=0, key="session_action")
    with col2:
        if st.session_state["session_action"] == session_action_list[1]:
            if st.button('create'):
                st.session_state["site"] = "CEMC"
                st.session_state["device"] = DEVICES[0]
                st.session_state["action"] = ACTIONS[0]
                site_name = st.session_state["site"]
                st.session_state["prior_site"] = st.session_state["site"]
                st.session_state["filters"] = "Whole Site"
                st.session_state["clicked_latlon"] = {d: [] for d in DEVICES}
                st.session_state['show_markers'] = False
                st.session_state['folium_map'] = None
                st.session_state['mapdata'] = {}
                st.session_state['results'] = {'rssi_gps': pd.DataFrame(),
                                               'beacon': pd.DataFrame(),
                                               'gateway': pd.DataFrame()}
                st.session_state['zoom_level'] = 17
                st.session_state['load_file'] = ''
                st.session_state['gateway_prefix'] = 'GW'
                st.session_state['beacon_prefix'] = 'BC'
        if st.session_state["session_action"] == session_action_list[2]:
            loaded_state = load_session_state()
            if st.button('load'):
                if loaded_state is not None:
                    st.session_state['sel_load_file'] = st.session_state['load_file']
                    for key, val in loaded_state.items():
                        #print(key, val)
                        if key in st.session_state:
                            st.session_state.pop(key)
                        st.session_state[key] = val
                        if (key=='site') and (val in SITE):
                            site_index = SITE[val]
                        if key == 'load_file':
                            continue

    if 'sel_load_file' in st.session_state:
        st.write('Session file in Use: {0}'.format(st.session_state['sel_load_file']) )

    st.subheader('Session Record')
    session_file = set_state_filepath()
    if st.button('Store Existing Session'):
        save_session_state(this_session_state=st.session_state,
                            session_file=session_file)
        
    if "site" in st.session_state:
        st.header("Selectables")
        site_name = st.selectbox("Site: ", SITE.keys(), index=0, key="site")
        site_code = SITE[site_name]
        # filter_type = st.selectbox("Filter on..: ", FILTERS.keys(), index=0, key="filter")
        # this_filter = FILTERS[filter_type]
        # st.session_state["filters"] = this_filter
        if site_name != st.session_state["prior_site"]:
            st.session_state["prior_site"] = st.session_state["site"]
        # Actions
        st.header("Simulation Setup")
        st.radio("Actions", ACTIONS, index=0, key="action")
        if st.session_state['action'] == 'Generate':
            # Generate
            st.radio("Device: ", DEVICES, index=0, key="device")
        gw = st.text_input('Gateway Prefix', 'GWsimulated', key='gateway_prefix')
        bcn = st.text_input('Beacon Prefix', 'BCsimulated', key='beacon_prefix')
        if st.button('Clear All Devices'):
            st.session_state["clicked_latlon"] = {d: [] for d in DEVICES}
        st.header("Simulation Parameters")    
        this_date = st.date_input("Start Date", datetime.today(), key='start_date')
        this_time = st.time_input("Start Time", datetime.now(), key='start_time')
        st.session_state['sim_start_timestamp'] = datetime.combine(this_date, this_time).replace(tzinfo=dt.timezone.utc)
        st.header('Current Settings')
        ts = st.session_state['sim_start_timestamp']
        st.write(f'Sim Start Time: {ts}') 
        for dev in DEVICES:
            dev_markers = st.session_state["clicked_latlon"][dev]
            st.write(dev + ' Markers=' + str(len(dev_markers)))

with st.container():
    if 'prior_site' in st.session_state:
        # Wait till we get a session started
        st.header("Generate Markers: {0}".format( st.session_state["site"]))
        LATLON = get_site_latlon(st.session_state["site"])
        #
        st.select_slider('Zoom Levels', options=[i for i in range(10,20)], key='zoom_level')
        m = folium.Map(location=list(LATLON), zoom_start=st.session_state['zoom_level'], width=700)
        # actions
        if st.session_state['action'] == 'Generate':
            dev_marker_dict = {st.session_state["device"]: st.session_state["clicked_latlon"][st.session_state["device"]]}
        else:
            dev_marker_dict = st.session_state["clicked_latlon"]

        for dev_type, dev_markers in dev_marker_dict.items():
            N = len(dev_markers)
            num_label = {idx: 'start' if (idx == 0) else 'end' if idx==(N-1) else 'mid' for idx in range(N)}
            for idx, this_latlon in enumerate(dev_markers):
                #Setup the content of the popup
                if dev_type == 'Gateway':
                    icon_type = GW_ICON_TYPES[num_label[idx]]
                    gateway_id = st.session_state['gateway_prefix']+str(GATEWAY_ID)
                    ttl = f'Gateway ID={gateway_id}, Marker{idx}'
                else:
                    icon_type = BEACON_ICON_TYPES[0]                    
                    beacon_id = st.session_state['beacon_prefix']+str(idx)
                    ttl = f'Beacon {beacon_id}, Marker={idx}'

                clr = DEV_CLR[dev_type]
                this_icon=folium.Icon(color=clr, icon=icon_type)
                folium.Marker(
                                list(this_latlon),  icon=this_icon, tooltip=ttl, 
                                popup=this_latlon
                                ).add_to(m)
                m.add_child(folium.LatLngPopup())
        mapdata = st_folium(m, height=700, width=700)
        #
        st.session_state['folium_map'] = m
        st.session_state['mapdata'] = mapdata

        # main display
        mapdata  = st.session_state['mapdata']
        if ('last_clicked' in mapdata) and isinstance(mapdata['last_clicked'], dict):
            this_latlon = (mapdata['last_clicked']['lat'], mapdata['last_clicked']['lng'])
            st.session_state["clicked_latlon"][st.session_state["device"]].append(this_latlon)


        if st.button('Compute'):
            beacon_coord_list = st.session_state["clicked_latlon"]['Beacon']
            beacon_coord_map = {st.session_state['beacon_prefix']+str(bid): coord for bid, coord in enumerate(beacon_coord_list)}
            gateway_markers = st.session_state["clicked_latlon"]['Gateway']
            gateway_coords_dict = {st.session_state['gateway_prefix']+str(GATEWAY_ID): gateway_markers}
            gendata = p1.GenerateData(beacon_coords=beacon_coord_map,
                                        gateway_coords=gateway_coords_dict,
                                        start_time=st.session_state['sim_start_timestamp']
                                        )
            gendata.generate_csv_file_bydevice(csvpath=DEFAULT_CSVPATH)
            st.session_state['results'] = {'rssi_gps': gendata.rssi_gps_df,
                                           'beacon': gendata.beacon_df,
                                           'gateway': gendata.gateway_df
                                           }

        if st.button('Show Results'):
            all_df = st.session_state['results']['beacon']
            for bid, df in all_df.groupby('beacon_id'):
                clrs = list('cbkry')
                st.write(df)
                arr = np.random.normal(1, 1, size=100)
                fig, ax = plt.subplots()
                # ax.errorbar(df.time, df.rssi, df.rssi_error,
                #             capsize=5, marker='o',
                #             color=clr, label=f'Beacon{bid}')
                for gw_mark_id, mdf in df.groupby('marker_index'):
                    clr = clrs.pop(0)
                    clrs.append(clr)
                    ax.plot(mdf.time, mdf.rssi, marker='o',
                            color=clr, label=f'Marker {gw_mark_id}')
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
                plt.setp(ax.get_xticklabels(), rotation=90,
                            fontsize = 12,
                            ha="right");
                ax.set_title(f'Beacon : {bid}')
                ax.grid('on')
                ax.legend()
                ax.set_ylim([-95, -30])
                ax.set_ylabel('rssi / dBm')
                st.pyplot(fig)
