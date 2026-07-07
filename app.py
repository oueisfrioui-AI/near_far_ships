"""
AIS Stationary Ship Detector
-----------------------------
Upload an AIS CSV, map your columns, and detect ships that stopped for
at least N hours - classified as near-port or far-from-port - shown on
an interactive map.

Run with:
    streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from sklearn.neighbors import BallTree

st.set_page_config(page_title="AIS Stationary Ship Detector", layout="wide")

EARTH_RADIUS_M = 6371000.0


# ============================================================
# Helper functions
# ============================================================
def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner=False)
def load_ais_csv(file_bytes, id_col, time_col, lat_col, lon_col, time_format):
    """Load only the needed columns, with memory-efficient dtypes."""
    usecols = [id_col, time_col, lat_col, lon_col]
    df = pd.read_csv(
        io.BytesIO(file_bytes),
        usecols=usecols,
        dtype={id_col: str, lat_col: "float32", lon_col: "float32"},
    )
    df.columns = df.columns.str.strip()

    if time_format == "Epoch milliseconds":
        df[time_col] = pd.to_datetime(df[time_col], unit="ms")
    elif time_format == "Epoch seconds":
        df[time_col] = pd.to_datetime(df[time_col], unit="s")
    else:  # auto-detect datetime string
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df = df.rename(columns={id_col: "vessel_id", time_col: "t", lat_col: "lat", lon_col: "lon"})
    df = df.dropna(subset=["t", "lat", "lon"])
    df = df.sort_values(["vessel_id", "t"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def detect_stationary_episodes(df, dist_threshold_m, min_gap_hours):
    g = df.groupby("vessel_id", sort=False)

    df = df.copy()
    df["prev_time"] = g["t"].shift()
    df["prev_lat"] = g["lat"].shift()
    df["prev_lon"] = g["lon"].shift()

    df["time_gap"] = df["t"] - df["prev_time"]
    df["dist_m"] = haversine_m(df["prev_lat"], df["prev_lon"], df["lat"], df["lon"])
    df = df.dropna(subset=["prev_time"])

    min_gap = pd.Timedelta(hours=min_gap_hours)
    stationary_gaps = df[
        (df["time_gap"] >= min_gap) & (df["dist_m"] <= dist_threshold_m)
    ][["vessel_id", "prev_time", "t", "time_gap", "dist_m", "prev_lat", "prev_lon", "lat", "lon"]].reset_index(
        drop=True
    )

    if stationary_gaps.empty:
        return stationary_gaps.assign(
            episode_start=None, episode_end=None, total_duration=None,
            start_lat=None, start_lon=None, end_lat=None, end_lon=None,
            net_displacement_m=None,
        )

    sg = stationary_gaps.sort_values(["vessel_id", "prev_time"]).reset_index(drop=True)
    new_episode = sg["vessel_id"].ne(sg["vessel_id"].shift()) | (sg["prev_time"] > sg["t"].shift())
    sg["episode_id"] = new_episode.cumsum()

    episodes = (
        sg.groupby(["episode_id", "vessel_id"])
        .agg(
            episode_start=("prev_time", "first"),
            episode_end=("t", "last"),
            start_lat=("prev_lat", "first"),
            start_lon=("prev_lon", "first"),
            end_lat=("lat", "last"),
            end_lon=("lon", "last"),
            max_gap_dist_m=("dist_m", "max"),
        )
        .reset_index()
        .drop(columns="episode_id")
    )

    episodes["total_duration"] = episodes["episode_end"] - episodes["episode_start"]
    episodes["net_displacement_m"] = haversine_m(
        episodes["start_lat"], episodes["start_lon"], episodes["end_lat"], episodes["end_lon"]
    )
    return episodes


@st.cache_data(show_spinner=False)
def load_ports(file_bytes):
    ports = pd.read_csv(
        io.BytesIO(file_bytes),
        usecols=["Main Port Name", "Alternate Port Name", "Country Code", "Latitude", "Longitude"],
    )
    ports["port_name"] = ports["Main Port Name"].fillna(ports["Alternate Port Name"])
    ports = ports.rename(columns={"Country Code": "country", "Latitude": "port_lat", "Longitude": "port_lon"})[
        ["port_name", "country", "port_lat", "port_lon"]
    ]
    ports = ports.dropna(subset=["port_lat", "port_lon"]).reset_index(drop=True)
    return ports


@st.cache_data(show_spinner=False)
def flag_port_stops(episodes_df, ports_df, port_dist_threshold_m):
    port_coords_rad = np.radians(ports_df[["port_lat", "port_lon"]].values)
    tree = BallTree(port_coords_rad, metric="haversine")

    coords_rad = np.radians(episodes_df[["start_lat", "start_lon"]].values)
    dist_rad, idx = tree.query(coords_rad, k=1)
    dist_m = dist_rad.flatten() * EARTH_RADIUS_M

    episodes_df = episodes_df.copy()
    episodes_df["nearest_port"] = ports_df.iloc[idx.flatten()]["port_name"].values
    episodes_df["nearest_port_country"] = ports_df.iloc[idx.flatten()]["country"].values
    episodes_df["dist_to_port_m"] = dist_m
    episodes_df["at_port"] = dist_m <= port_dist_threshold_m
    return episodes_df


def build_map(ports_df, stops_far_df, stops_near_df):
    if len(stops_far_df) or len(stops_near_df):
        all_lats = pd.concat(
            [stops_far_df["start_lat"], stops_near_df["start_lat"]], ignore_index=True
        )
        center_lat = all_lats.mean()
        all_lons = pd.concat(
            [stops_far_df["start_lon"], stops_near_df["start_lon"]], ignore_index=True
        )
        center_lon = all_lons.mean()
    else:
        center_lat, center_lon = ports_df["port_lat"].mean(), ports_df["port_lon"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles="cartodbpositron")

    port_layer = folium.FeatureGroup(name="Ports", show=True)
    port_cluster = MarkerCluster().add_to(port_layer)
    for _, row in ports_df.iterrows():
        folium.Marker(
            location=[row["port_lat"], row["port_lon"]],
            popup=folium.Popup(f"<b>{row['port_name']}</b><br>{row['country']}", max_width=250),
            tooltip=row["port_name"],
            icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
        ).add_to(port_cluster)
    port_layer.add_to(m)

    near_layer = folium.FeatureGroup(name="Stops near port", show=True)
    near_cluster = MarkerCluster().add_to(near_layer)
    for _, row in stops_near_df.iterrows():
        popup_html = (
            f"<b>Vessel:</b> {row['vessel_id']}<br>"
            f"<b>Start:</b> {row['episode_start']}<br>"
            f"<b>End:</b> {row['episode_end']}<br>"
            f"<b>Duration:</b> {row['total_duration']}<br>"
            f"<b>Nearest port:</b> {row['nearest_port']} ({row['dist_to_port_m']:.0f} m)"
        )
        folium.Marker(
            location=[row["start_lat"], row["start_lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Vessel {row['vessel_id']}",
            icon=folium.Icon(color="green", icon="ship", prefix="fa"),
        ).add_to(near_cluster)
    near_layer.add_to(m)

    far_layer = folium.FeatureGroup(name="Stops far from port", show=True)
    far_cluster = MarkerCluster().add_to(far_layer)
    for _, row in stops_far_df.iterrows():
        popup_html = (
            f"<b>Vessel:</b> {row['vessel_id']}<br>"
            f"<b>Start:</b> {row['episode_start']}<br>"
            f"<b>End:</b> {row['episode_end']}<br>"
            f"<b>Duration:</b> {row['total_duration']}<br>"
            f"<b>Nearest port:</b> {row['nearest_port']} ({row['dist_to_port_m']:.0f} m away)"
        )
        folium.Marker(
            location=[row["start_lat"], row["start_lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Vessel {row['vessel_id']}",
            icon=folium.Icon(color="red", icon="ship", prefix="fa"),
        ).add_to(far_cluster)
    far_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ============================================================
# Session state initialization
# ============================================================
# We store the computed analysis results here so that reruns triggered by
# things OTHER than the "Run analysis" button (e.g. panning/zooming/clicking
# the st_folium map, which is a bidirectional component and reruns the whole
# script) don't wipe out what's on screen. st.button() only returns True on
# the single run where the click happened, so gating the results display on
# it directly is what was causing the "reset" behaviour.
if "result" not in st.session_state:
    st.session_state.result = None          # full episodes dataframe (no port match)
if "stops_at_port" not in st.session_state:
    st.session_state.stops_at_port = None
if "stops_far_from_port" not in st.session_state:
    st.session_state.stops_far_from_port = None
if "ports" not in st.session_state:
    st.session_state.ports = None
if "n_rows" not in st.session_state:
    st.session_state.n_rows = None
if "n_vessels" not in st.session_state:
    st.session_state.n_vessels = None


# ============================================================
# Sidebar - inputs
# ============================================================
st.title("🚢 AIS Stationary Ship Detector")
st.write(
    "Upload an AIS CSV, map your columns, and find ships that stayed in place "
    "for at least N hours - split into stops near a port and stops far from any port."
)

st.sidebar.header("1. Upload AIS data")
ais_file = st.sidebar.file_uploader("AIS CSV file", type=["csv"])

st.sidebar.header("2. Upload port reference data")
st.sidebar.caption("World Port Index CSV (or similar, with Main Port Name / Latitude / Longitude columns)")
ports_file = st.sidebar.file_uploader("Ports CSV file", type=["csv"])

if ais_file is not None:
    ais_bytes = ais_file.getvalue()
    preview_cols = pd.read_csv(io.BytesIO(ais_bytes), nrows=5).columns.tolist()

    st.sidebar.header("3. Map your columns")
    id_col = st.sidebar.selectbox("Vessel ID column", preview_cols)
    time_col = st.sidebar.selectbox("Timestamp column", preview_cols)
    lat_col = st.sidebar.selectbox("Latitude column", preview_cols)
    lon_col = st.sidebar.selectbox("Longitude column", preview_cols)

    time_format = st.sidebar.selectbox(
        "Timestamp format", ["Auto-detect (date string)", "Epoch milliseconds", "Epoch seconds"]
    )

    st.sidebar.header("4. Parameters")
    dist_threshold_m = st.sidebar.slider("Same-position tolerance (m)", 10, 1000, 100, step=10)
    min_gap_hours = st.sidebar.slider("Minimum stationary duration (hours)", 1, 24, 1)
    port_dist_threshold_m = st.sidebar.slider("Near-port distance threshold (m)", 500, 50000, 5000, step=500)

    run = st.sidebar.button("Run analysis", type="primary")

    # --- Only (re)compute when the button is actually clicked. -------------
    # Everything computed here is written into st.session_state so it
    # survives later reruns that aren't caused by this button (map clicks,
    # tab switches, download-button clicks, etc.)
    if run:
        with st.spinner("Loading AIS data..."):
            df = load_ais_csv(ais_bytes, id_col, time_col, lat_col, lon_col, time_format)
        st.session_state.n_rows = len(df)
        st.session_state.n_vessels = df["vessel_id"].nunique()

        with st.spinner("Detecting stationary episodes..."):
            episodes = detect_stationary_episodes(df, dist_threshold_m, min_gap_hours)
        result = episodes[episodes["total_duration"] >= pd.Timedelta(hours=min_gap_hours)].reset_index(drop=True)

        if ports_file is not None and len(result) > 0:
            ports_bytes = ports_file.getvalue()
            with st.spinner("Loading port reference data..."):
                ports = load_ports(ports_bytes)
            with st.spinner("Matching stops to nearest ports..."):
                result = flag_port_stops(result, ports, port_dist_threshold_m)

            st.session_state.ports = ports
            st.session_state.stops_at_port = result[result["at_port"]].reset_index(drop=True)
            st.session_state.stops_far_from_port = result[~result["at_port"]].reset_index(drop=True)
            st.session_state.result = None
        else:
            st.session_state.result = result
            st.session_state.ports = None
            st.session_state.stops_at_port = None
            st.session_state.stops_far_from_port = None

    # --- Render from session_state (independent of the button value). ------
    if st.session_state.n_rows is not None:
        st.success(
            f"Loaded {st.session_state.n_rows:,} rows across "
            f"{st.session_state.n_vessels:,} vessels."
        )

    has_port_results = st.session_state.stops_at_port is not None
    has_plain_result = st.session_state.result is not None

    if has_port_results:
        stops_at_port = st.session_state.stops_at_port
        stops_far_from_port = st.session_state.stops_far_from_port
        ports = st.session_state.ports

        st.success(
            f"Found {len(stops_at_port) + len(stops_far_from_port):,} stationary "
            f"episodes of at least {min_gap_hours}h."
        )

        col1, col2 = st.columns(2)
        col1.metric("Stops near a port", len(stops_at_port))
        col2.metric("Stops far from any port", len(stops_far_from_port))

        st.subheader("🗺️ Interactive map")
        m = build_map(ports, stops_far_from_port, stops_at_port)
        st_folium(m, width=1200, height=600)

        tab1, tab2 = st.tabs(["Stops near port", "Stops far from port"])
        with tab1:
            st.dataframe(stops_at_port)
            st.download_button(
                "Download CSV",
                stops_at_port.to_csv(index=False).encode("utf-8"),
                "stops_at_port.csv",
                "text/csv",
                key="dl_at_port",
            )
        with tab2:
            st.dataframe(stops_far_from_port)
            st.download_button(
                "Download CSV",
                stops_far_from_port.to_csv(index=False).encode("utf-8"),
                "stops_far_from_port.csv",
                "text/csv",
                key="dl_far_port",
            )

    elif has_plain_result:
        result = st.session_state.result
        if len(result) == 0:
            st.warning("No stationary episodes found with the current parameters. Try loosening the thresholds.")
        else:
            st.success(f"Found {len(result):,} stationary episodes of at least {min_gap_hours}h.")
            st.info("Upload a port reference CSV in the sidebar to classify stops as near/far from port.")
            st.dataframe(result)
            st.download_button(
                "Download CSV",
                result.to_csv(index=False).encode("utf-8"),
                "stationary_episodes.csv",
                "text/csv",
                key="dl_plain",
            )
    elif not run:
        st.info("Set your parameters and click **Run analysis** in the sidebar.")

else:
    st.info("Upload an AIS CSV file in the sidebar to get started.")
