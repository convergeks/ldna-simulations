import os
import toml
import pickle
from pathlib import Path
import time
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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mpld3
import plotly.graph_objects as go
# import local code
import sim_p0_setupdevices as p0
import sim_p1_generate_rxtxdata as p1
import sim_p1x1_diagnose_rxtxdata as p1x1

# reg - qrcode
import segno
import io
from PIL import Image

# paths
ROOT = os.getcwd()
if "sim_core" in ROOT:
    ROOT = os.getcwd().parent
DEFAULT_OUTPATH = os.path.join(ROOT, "dump")
DEFAULT_CSVPATH = os.path.join(ROOT, "dump/csvfiles")
DEFAULT_IMPORT_DATAPATH = os.path.join(ROOT, "dump/datasource/dump/Tinglev/datasource/raw")

# ----- options -----
# site & dates
SITE = {"CEMC": 0, "Oxford": 1, "Tinglev": 2, "FPMcCann": 3, "Creagh": 4, 'KS': 5}
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
GW_ICON_CLRS = {'start': 'black',
                 'mid': 'green',
                 'end': 'red'
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

def get_gps_beacon_filepath(folder_path=DEFAULT_IMPORT_DATAPATH):
    filenames = os.listdir(folder_path)
    gateway_filenames = [f for f in filenames if 'gateway' in f]
    gateway_filenames = sorted(gateway_filenames)
    beacon_filenames = [f for f in filenames if 'beacon' in f]
    beacon_filenames = sorted(beacon_filenames)
    beacon_filename = ''
    gateway_filename = ''
    gateway_filename = st.selectbox('Select gateway file', gateway_filenames, key='import_gateway_file')
    beacon_filename = st.selectbox('Select beacon file', beacon_filenames, key='import_beacon_file')    
    st.write(gateway_filename)
    st.write(beacon_filename)
    if st.button('import_data'):
        selected_filename_map = {'gps': gateway_filename, 'rssi': beacon_filename}
        return folder_path, selected_filename_map
    else:
        return None, None

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

def import_data():
    folder_path, selected_filename = get_gps_beacon_filepath()
    if selected_filename is not None:
        import_gateway_file = os.path.join(folder_path, selected_filename['gps'])
        import_beacon_file = os.path.join(folder_path, selected_filename['rssi'])    
        gateway_df = pd.read_csv(import_gateway_file)
        gateway_df['time'] = pd.to_datetime(gateway_df['time'])
        gateway_df['time'] = gateway_df['time'].dt.tz_localize(dt.timezone.utc)
        beacon_df = pd.read_csv(import_beacon_file)
        beacon_df['time'] = pd.to_datetime(beacon_df['time'])
        beacon_df['time'] = beacon_df['time'].dt.tz_localize(dt.timezone.utc)
        data_df_map = {'gateway': gateway_df, 'beacon': beacon_df}
        return data_df_map
    else :
        return None
    
# --- beacon util ===
def get_beacon_id_str(pre, idx):
    if 10<idx<100:
        beacon_id = pre+str(idx)
    elif idx<10:
        beacon_id = pre+'0' +str(idx)
    else:
        beacon_id = pre[:-1] + str(idx)
    return beacon_id

# --- show plots
def compare_simreal_time(diagnostics):
    # raw timeseries compare
    sim_df = diagnostics.simbeacon_realgw_df
    sim_bid = sim_df.sim_mac.iloc[0]
    real_df = diagnostics.real_beacon_df
    real_bid = real_df.beacon.iloc[0]
    #
    fig, axes = plt.subplots(nrows=2, figsize=(15,10), sharex=True, sharey=True)
    ax = axes[0]
    ax.plot(sim_df.time, sim_df.rssi, 'r', marker='o')#, color='Simulated')
    ax.set_title(f'Simulated Beacon ', fontsize=30)#: {sim_bid}')
    ax.grid('on')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    plt.setp(ax.get_yticklabels(),
                fontsize = 20,
                ha="right");
    plt.setp(ax.get_xticklabels(), rotation=90,
                fontsize = 20,
                ha="right");
    # ax.set_ylim([-95, -30])
    ax.set_ylabel('rssi / dBm', fontsize=20)
    ax = axes[1]
    this_lbl = f'Real Beacon : {real_bid}'
    ax.plot(real_df.time, real_df.rssi, 'k', marker='o')#, label=this_lbl)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b %H:%M"))
    plt.setp(ax.get_xticklabels(), rotation=90,
                fontsize = 20,
                ha="right");
    plt.setp(ax.get_yticklabels(),
                fontsize = 20,
                ha="right");
    ax.set_title(this_lbl, fontsize=30)
    ax.grid('on')
    # ax.legend()
    # ax.set_ylim([-95, -30])
    ax.set_ylabel('rssi / dBm', fontsize=20)
    return fig

def compare_simreal_stats(diagnostics):
    # visualise - stats
    sim_df = diagnostics.simbeacon_realgw_df
    real_df = diagnostics.real_beacon_df
    #
    fig, axes = plt.subplots(nrows=2, figsize=(15,10), sharex=True, sharey=True)
    ax = axes[0]
    # --- do it ---

    # TODO: report analysis - what does this look like?
    return fig

# ---- main streamlit section ----
# set max width
st.set_page_config(layout="wide")
# _max_width_(percent_width=50)
# invoke session state
# if "site" not in st.session_state:
#     # # settigs
session_fields = ['site', 'device', 'action', '']
session_action_list = ['Current', 'New/Reset', 'Load Prior Session', 'Import GPS Data']
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
                st.session_state['gateway_prefix'] = 'gwsimulate'
                st.session_state['beacon_prefix'] = 'BC' + 'sims'.encode('utf-8').hex().upper()
                st.session_state["imported_data"] = {}
                st.session_state['import_beacon_prefix'] = 'BC' + 'test'.encode('utf-8').hex().upper()
                st.session_state["imported_beacon_data"] = {}
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
                    st.session_state["imported_data"] = {}
                    st.session_state["imported_beacon_data"] = {}
        if st.session_state["session_action"] == session_action_list[3]:
            st.session_state['import_beacon_prefix'] = 'BC' + 'test'.encode('utf-8').hex().upper()
            data_df_map = import_data()
            if data_df_map is not None:
                st.write(data_df_map.keys())
                st.session_state["imported_data"] = data_df_map
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
        gw = st.text_input('Gateway Prefix', 'gwsimulate', key='gateway_prefix')
        bcn = st.text_input('Beacon Prefix', 'BC'+ 'sims'.encode('utf-8').hex().upper(), key='beacon_prefix')
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

tab_build, tab_reg, tab_import = st.tabs(["Build Sims", "Register", "Import Gateway Data"])
with tab_build: 
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
                        beacon_id = get_beacon_id_str(st.session_state['beacon_prefix'], idx)                   
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
                print(beacon_coord_list)
                beacon_coord_map = {get_beacon_id_str(st.session_state['beacon_prefix'], bid): coord for bid, coord in enumerate(beacon_coord_list)}
                print(list(beacon_coord_map.keys()))
                # beacon_coord_map = {st.session_state['beacon_prefix']+str(bid): coord for bid, coord in enumerate(beacon_coord_list)}
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

with tab_reg:
    with st.container():
        if ('clicked_latlon' in st.session_state) and ('Beacon' in st.session_state["clicked_latlon"]):
            beacon_coord_list = st.session_state["clicked_latlon"]['Beacon']
            if len(beacon_coord_list)>0:
                beacon_coord_map = {get_beacon_id_str(st.session_state['beacon_prefix'], bid): coord for bid, coord in enumerate(beacon_coord_list)}
                beacon_ids = list(beacon_coord_map.keys())
                sel_beacon = st.selectbox('Select beacon', beacon_ids)
                unit_txt=st.text_input('Unit ID', '')
                beacon_txt=st.text_input('Beacon ID', f'MAC:{sel_beacon},SERIAL:{sel_beacon[-2:]};')
                if st.button('Generate QRcodes'):
                    unit_qrcode = segno.make(unit_txt)
                    beacon_qrcode = segno.make(beacon_txt)
                    # bcn_img = beacon_qrcode.to_pil(scale=3)
                    fig, axes = plt.subplots(ncols=2)
                    ax = axes[0]
                    unit_out = io.BytesIO()
                    unit_qrcode.save(unit_out, scale=3, kind='png')
                    unit_out.seek(0)  # Important to let PIL / Pillow load the image
                    unit_img = Image.open(unit_out)  # Done, do what ever you want with the PIL/Pillow image
                    ax.imshow(unit_img)
                    ax.axis('off')
                    ax.set_title(unit_txt)
                    ax = axes[1]
                    beacon_out = io.BytesIO()
                    beacon_qrcode.save(beacon_out, scale=3, kind='png')
                    beacon_out.seek(0)  # Important to let PIL / Pillow load the image
                    bcn_img = Image.open(beacon_out)  # Done, do what ever you want with the PIL/Pillow image
                    ax.imshow(bcn_img)
                    ax.axis('off')
                    ax.set_title(beacon_txt)
                    st.pyplot(fig)

with tab_import:
    with st.container():
        if 'imported_data' in st.session_state and (len(st.session_state['imported_data']) > 0):
            data_df = st.session_state['imported_data']
            gateway_df = data_df['gateway']
            gateway_list = sorted(gateway_df.gateway.unique())
            beacon_df = data_df['beacon']
            beacon_list = sorted(beacon_df.beacon.unique())
            beacon_list = [bid for bid in beacon_list if bid[:2]=='BC']
            #
            import_col1, import_col2 = st.columns(2)
            with import_col1:
                this_gateway_id = st.selectbox('Select gateway to review', gateway_list, key='review_gateway', index=0)
                gateway_df = gateway_df[gateway_df.gateway==this_gateway_id]
                #debug
                # st.write(st.session_state["review_gateway"], this_gateway_id)

                # start, end time
                import_start_time = st.time_input("Start Time", gateway_df.time.min(), key='import_start_time', step=5*60)
                import_end_time = st.time_input("End Time", gateway_df.time.max(), key='import_end_time', step=5*60 )
                # convert gateway data into milestones
                gateway_df['valid'] = gateway_df.time.apply(lambda x: import_start_time<=x.time()<=import_end_time)
                gps_df = gateway_df[gateway_df['valid']]
                # st.write(gateway_df.shape, gps_df.shape, gps_df.time.min(), gps_df.time.max())            
                gps_df['minutes_from_start'] = gps_df.time.apply(lambda x: int((x - gps_df.time.min()).total_seconds()/60))
                gps_df['Nsecs_from_start'] = gps_df.time.apply(lambda x: int((x - gps_df.time.min()).total_seconds()/6))
                gps_milestones_dict = {'gateway':[], 'time':[], 'latitude':[], 'longitude':[], 'accuracy':[]}
                for this_minute, gdf in gps_df.groupby('Nsecs_from_start'):
                    gdf = gdf.sort_values(by=['time'], ascending=True)
                    this_row = gdf.iloc[0]
                    for fld in gps_milestones_dict:
                        gps_milestones_dict[fld].append(this_row[fld])
                gps_milestones_df = pd.DataFrame.from_dict(gps_milestones_dict)
                gps_milestones_df = gps_milestones_df.sort_values(by=['time'], ascending=True)
                # rssi_df = beacon_df.time.apply(lambda x: import_start_time<=x.time.time()<=import_end_time)
                # show map
                LATLON = gps_milestones_df.latitude.mean(), gps_milestones_df.longitude.mean()
                #
            st.select_slider('Zoom Levels', options=[i for i in range(10,20)], key='zoom_level_import')
            m = folium.Map(location=list(LATLON), zoom_start=st.session_state['zoom_level_import'], width=700)
            #
            N = gps_milestones_df.shape[0]
            num_label = {idx: 'start' if (idx == 0) else 'end' if idx==(N-1) else 'mid' for idx in range(N)}            
            coordinates = []
            for idx, row in gps_milestones_df.iterrows():
                this_latlon = row.latitude, row.longitude
                coordinates.append(list(this_latlon))
                icon_type = GW_ICON_TYPES[num_label[idx]]
                icon_clr = GW_ICON_CLRS[num_label[idx]]
                this_hhmmss = str(row.time.time())[:8]
                this_date = row.time.date().strftime('%B %d')
                # this_hhmm = time.strftime('%H:%M', row.time)
                ttl = f'{this_date}@{this_hhmmss}, <br> Marker{idx} <br> ID={this_gateway_id}'
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
            if st.checkbox('Show Connector'):
                folium.PolyLine(
                                locations=coordinates,
                                color="#FF0000",
                                weight=2,
                                tooltip="Connect the dots",
                            ).add_to(m)
            # add beacons
            with import_col2:
                add_beacons_numbers = st.text_input('# of Beacons to add', '0')
                Nbeacons = int(add_beacons_numbers)
                if Nbeacons>0:
                    import_beacon_number = st.selectbox('Select beacon number', np.arange(1, Nbeacons+1), key='import_num_beacons')
                    import_beacon_handle = st.text_input('Beacon handle', f'Test-Beacon-{import_beacon_number}')
                    beacon_mac = get_beacon_id_str(st.session_state['import_beacon_prefix'], import_beacon_number)
                    import_beacon_mac = st.text_input('Beacon mac address', beacon_mac)
                    import_beacon_latitude = st.text_input('Latitude', '', key='import_lat')
                    import_beacon_longitude = st.text_input('Longitude', '', key='import_lon')
                    if len(beacon_list)>0:
                        map_to_real_beacon = st.selectbox('Simulator models real beacon:', beacon_list, index=0, key='real_beacon')
                    else:
                        map_to_real_beacon = ''

                    if len(import_beacon_latitude)>0 & len(import_beacon_longitude)>0:
                        imported_bcn_latlon = float(import_beacon_latitude), float(import_beacon_longitude)
                if 'imported_beacon_data' not in st.session_state:
                    st.session_state["imported_beacon_data"] = {} 
                imported_beacon_data = st.session_state["imported_beacon_data"]
                if st.button('Store/Show beacon data'):
                    st.write(st.session_state['import_lat'], st.session_state['import_lon'])
                    imported_beacon_data[import_beacon_number] = {'name': import_beacon_handle,
                                                                  'mac': import_beacon_mac,
                                                                  'latlon': (st.session_state['import_lat'], st.session_state['import_lon']),
                                                                  'real_mac': st.session_state['real_beacon']
                                                                  }
                    st.session_state["imported_beacon_data"] = imported_beacon_data
                #
                if st.button('Run Diagnostics'):
                    beacon_data = st.session_state["imported_beacon_data"]
                    this_beacon_data = beacon_data[import_beacon_number]
                    st_time = time.time()
                    diagnostics = p1x1.DiagnoseData(
                                                real_gateway_id=st.session_state["review_gateway"],
                                                beacon_data=st.session_state["imported_beacon_data"],
                                                gateway_df=gateway_df,
                                                beacon_df=beacon_df,
                                                start_time=st.session_state["import_start_time"],
                                                end_time=st.session_state["import_end_time"]
                                                )
                    st.write(diagnostics.valid_checks)
                    # Analyse
                    et0 = time.time()
                    st.write(f'Setup: Time taken={et0-st_time:.1f}secs')
                    diagnostics.create_simulated_beacon_data()
                    et1 = time.time()
                    st.write(f'Simulate: Time taken={et1-et0:.1f}secs')
                    #
                    st.session_state['diagnostics'] = diagnostics

                #
                for bcn_num, bcn_data in st.session_state["imported_beacon_data"].items():
                    bcn_name = bcn_data['name']
                    bcn_mac = bcn_data['mac']
                    bcn_latlon = bcn_data['latlon']
                    real_bcn_mac = bcn_data['real_mac']
                    icon_type = BEACON_ICON_TYPES[0]
                    ttl = f'Beacon Name={bcn_name}, <br> Mac={bcn_mac}, <br> Real Mac={real_bcn_mac}'
                    clr = DEV_CLR['Beacon']
                    this_icon=folium.Icon(color=clr, icon=icon_type)
                    folium.Marker(
                                    list(bcn_latlon),  icon=this_icon, tooltip=ttl,
                                    popup=bcn_latlon
                                    ).add_to(m)
                
            #
            mapdata = st_folium(m, height=1000, width=1000)

            # diagnostics.compare_sims_vs_real()
            if 'diagnostics' in st.session_state:
                diagnostics = st.session_state['diagnostics']
                this_fig = compare_simreal_time(diagnostics)
                # Plot
                # st.pyplot(this_fig)
                # fig_html = mpld3.fig_to_html(this_fig)
                # st_components.html(fig_html, height=1200)
                st.plotly_chart(this_fig)


    