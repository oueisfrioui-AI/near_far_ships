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
from streamlit_folium import folium_static
from sklearn.neighbors import BallTree

st.set_page_config(page_title="AIS Stationary Ship Detector", layout="wide")

EARTH_RADIUS_M = 6371000.0


# ============================================================
# Helper functions
# ============================================================
def guess_time_format(sample_series):
    """Inspect a sample of raw timestamp values and guess whether they're
    epoch milliseconds, epoch seconds, or a date string, based on magnitude.
    Returns the index matching the Timestamp format selectbox options."""
    sample = sample_series.dropna().astype(str).str.strip()
    sample = sample[sample != ""]
    if sample.empty:
        return 0
    numeric = pd.to_numeric(sample, errors="coerce")
    if numeric.notna().all():
        median_val = numeric.median()
        if median_val > 1e12:
            return 1  # Epoch milliseconds (~13 digits, e.g. 1700000000000)
        elif median_val > 1e9:
            return 2  # Epoch seconds (~10 digits, e.g. 1700000000)
    return 0  # Auto-detect (date string)


def guess_column(columns, candidates):
    """Case-insensitive match of column names against a list of likely candidates.
    Returns the index of the best match in `columns`, or 0 if nothing matches."""
    lower_map = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return columns.index(lower_map[candidate])
    # fall back to substring match (e.g. "vessel_mmsi", "ship_id_no")
    for i, col in enumerate(columns):
        col_l = col.lower().strip()
        if any(candidate in col_l for candidate in candidates):
            return i
    return 0


# Common column-name variants seen across different AIS data providers
ID_CANDIDATES = ["vessel_id", "mmsi", "imo", "ship_id", "vesselid", "shipid", "vessel_name", "mmsi_no"]
TIME_CANDIDATES = ["timestamp", "time", "datetime", "basedatetime", "base_date_time", "t", "date_time"]
LAT_CANDIDATES = ["lat", "latitude"]
LON_CANDIDATES = ["lon", "lng", "long", "longitude"]


def fmt_duration(td):
    """Format a Timedelta as a short human-readable string."""
    if pd.isna(td):
        return "-"
    hours = td.total_seconds() / 3600
    if hours >= 24:
        return f"{hours / 24:.1f} days"
    return f"{hours:.1f} h"


def duration_marker_color(total_duration, base):
    """Pick a shade of the base color (green/red) based on how long the stop
    lasted: light = short, normal = medium, dark = long."""
    hours = total_duration.total_seconds() / 3600
    if hours < 6:
        return f"light{base}"
    elif hours < 24:
        return base
    else:
        return f"dark{base}"


@st.cache_data(show_spinner=False)
def compute_approach_tracks(df, stops_df, lookback_points=5):
    """For each stop, grab the vessel's last few pings before the stop began,
    so we can draw a faint line showing how it approached the stop location."""
    stops_df = stops_df.copy()
    grouped = {vid: g for vid, g in df.groupby("vessel_id", sort=False)}
    tracks = []
    for _, row in stops_df.iterrows():
        g = grouped.get(row["vessel_id"])
        if g is None:
            tracks.append([])
            continue
        prior = g[g["t"] < row["episode_start"]].tail(lookback_points)
        coords = list(zip(prior["lat"].tolist(), prior["lon"].tolist()))
        coords.append((row["start_lat"], row["start_lon"]))
        tracks.append(coords)
    stops_df["approach_track"] = tracks
    return stops_df


@st.cache_data(show_spinner=False)
def get_time_range(file_bytes, time_col, time_format):
    """Read just the timestamp column to find the overall min/max date range."""
    s = pd.read_csv(io.BytesIO(file_bytes), usecols=[time_col])[time_col]
    if time_format == "Epoch milliseconds":
        s = pd.to_datetime(s, unit="ms", errors="coerce")
    elif time_format == "Epoch seconds":
        s = pd.to_datetime(s, unit="s", errors="coerce")
    else:
        s = pd.to_datetime(s, errors="coerce")
    s = s.dropna()
    if s.empty:
        return None, None
    return s.min(), s.max()


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


def build_map(ports_df, stops_far_df, stops_near_df, fit_to_data=False):
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
            "<div style='word-wrap: break-word; overflow-wrap: break-word; white-space: normal; max-width: 260px;'>"
            f"<b>Vessel:</b> {row['vessel_id']}<br>"
            f"<b>Start:</b> {row['episode_start']}<br>"
            f"<b>End:</b> {row['episode_end']}<br>"
            f"<b>Duration:</b> {row['total_duration']}<br>"
            f"<b>Moved while stationary:</b> {row['net_displacement_m']:.0f} m<br>"
            f"<b>Nearest port:</b> {row['nearest_port']} ({row['dist_to_port_m']:.0f} m)"
            "</div>"
        )
        folium.Marker(
            location=[row["start_lat"], row["start_lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Vessel {row['vessel_id']}",
            icon=folium.Icon(color=duration_marker_color(row["total_duration"], "green"), icon="ship", prefix="fa"),
        ).add_to(near_cluster)
        track = row.get("approach_track")
        if track and len(track) >= 2:
            folium.PolyLine(
                locations=track, color="gray", weight=2, opacity=0.6, dash_array="5,5"
            ).add_to(near_layer)
    near_layer.add_to(m)

    far_layer = folium.FeatureGroup(name="Stops far from port", show=True)
    far_cluster = MarkerCluster().add_to(far_layer)
    for _, row in stops_far_df.iterrows():
        popup_html = (
            "<div style='word-wrap: break-word; overflow-wrap: break-word; white-space: normal; max-width: 260px;'>"
            f"<b>Vessel:</b> {row['vessel_id']}<br>"
            f"<b>Start:</b> {row['episode_start']}<br>"
            f"<b>End:</b> {row['episode_end']}<br>"
            f"<b>Duration:</b> {row['total_duration']}<br>"
            f"<b>Moved while stationary:</b> {row['net_displacement_m']:.0f} m<br>"
            f"<b>Nearest port:</b> {row['nearest_port']} ({row['dist_to_port_m']:.0f} m away)"
            "</div>"
        )
        folium.Marker(
            location=[row["start_lat"], row["start_lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Vessel {row['vessel_id']}",
            icon=folium.Icon(color=duration_marker_color(row["total_duration"], "red"), icon="ship", prefix="fa"),
        ).add_to(far_cluster)
        track = row.get("approach_track")
        if track and len(track) >= 2:
            folium.PolyLine(
                locations=track, color="gray", weight=2, opacity=0.6, dash_array="5,5"
            ).add_to(far_layer)
    far_layer.add_to(m)

    if fit_to_data:
        bounds = []
        for df_ in (stops_near_df, stops_far_df):
            for _, row in df_.iterrows():
                bounds.append([row["start_lat"], row["start_lon"]])
                track = row.get("approach_track")
                if track:
                    bounds.extend([[lat, lon] for lat, lon in track])
        if bounds:
            m.fit_bounds(bounds, padding=(30, 30))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


LEGEND_HTML = """
<div style="padding: 10px 14px; border: 1px solid #ccc; border-radius: 6px;
            font-size: 13px; line-height: 1.9; background-color: #fafafa;">
    <b>Legend</b><br>
    <span style="display:inline-block;width:12px;height:12px;background:#2A81CB;border-radius:50%;margin-right:6px;"></span>Port<br>
    <span style="display:inline-block;width:12px;height:12px;background:#8fce8f;border-radius:50%;margin-right:6px;"></span>Near port, stop &lt; 6h<br>
    <span style="display:inline-block;width:12px;height:12px;background:#3f9c35;border-radius:50%;margin-right:6px;"></span>Near port, stop 6&ndash;24h<br>
    <span style="display:inline-block;width:12px;height:12px;background:#1a5c14;border-radius:50%;margin-right:6px;"></span>Near port, stop &gt; 24h<br>
    <span style="display:inline-block;width:12px;height:12px;background:#f28b82;border-radius:50%;margin-right:6px;"></span>Far from port, stop &lt; 6h<br>
    <span style="display:inline-block;width:12px;height:12px;background:#d63e2a;border-radius:50%;margin-right:6px;"></span>Far from port, stop 6&ndash;24h<br>
    <span style="display:inline-block;width:12px;height:12px;background:#7a1710;border-radius:50%;margin-right:6px;"></span>Far from port, stop &gt; 24h<br>
    <span style="color:#888; font-weight:bold;">- - -</span> Approach path leading into a stop
</div>
"""


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
    preview_df = pd.read_csv(io.BytesIO(ais_bytes), nrows=20)
    preview_cols = preview_df.columns.tolist()

    st.sidebar.header("3. Map your columns")
    id_col = st.sidebar.selectbox(
        "Vessel ID column", preview_cols,
        index=guess_column(preview_cols, ID_CANDIDATES),
        help="Auto-detects common names like vessel_id, MMSI, IMO, ship_id.",
    )
    time_col = st.sidebar.selectbox(
        "Timestamp column", preview_cols,
        index=guess_column(preview_cols, TIME_CANDIDATES),
    )
    lat_col = st.sidebar.selectbox(
        "Latitude column", preview_cols,
        index=guess_column(preview_cols, LAT_CANDIDATES),
    )
    lon_col = st.sidebar.selectbox(
        "Longitude column", preview_cols,
        index=guess_column(preview_cols, LON_CANDIDATES),
    )

    time_format = st.sidebar.selectbox(
        "Timestamp format", ["Auto-detect (date string)", "Epoch milliseconds", "Epoch seconds"],
        index=guess_time_format(preview_df[time_col]),
    )

    min_dt, max_dt = get_time_range(ais_bytes, time_col, time_format)
    if min_dt is not None:
        st.sidebar.header("4. Date range filter")
        date_range = st.sidebar.date_input(
            "Restrict to date range",
            value=(min_dt.date(), max_dt.date()),
            min_value=min_dt.date(),
            max_value=max_dt.date(),
            help=f"Data spans {min_dt.date()} to {max_dt.date()}.",
        )
    else:
        date_range = None

    st.sidebar.header("5. Parameters")
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

        if date_range and len(date_range) == 2:
            start_date, end_date = date_range
            df = df[
                (df["t"] >= pd.Timestamp(start_date))
                & (df["t"] < pd.Timestamp(end_date) + pd.Timedelta(days=1))
            ].reset_index(drop=True)

        st.session_state.n_rows = len(df)
        st.session_state.n_vessels = df["vessel_id"].nunique()

        with st.spinner("Detecting stationary episodes..."):
            episodes = detect_stationary_episodes(df, dist_threshold_m, min_gap_hours)
        result = episodes[episodes["total_duration"] >= pd.Timedelta(hours=min_gap_hours)].reset_index(drop=True)

        if len(result) > 0:
            with st.spinner("Tracing approach paths..."):
                result = compute_approach_tracks(df, result)

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

        all_stops = pd.concat([stops_at_port, stops_far_from_port], ignore_index=True)
        st.subheader("📊 Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Unique vessels", f"{all_stops['vessel_id'].nunique():,}")
        s2.metric("Avg stop duration", fmt_duration(all_stops["total_duration"].mean()))
        longest_row = all_stops.loc[all_stops["total_duration"].idxmax()]
        s3.metric("Longest stop", fmt_duration(longest_row["total_duration"]))
        s4.metric("Avg drift while stopped", f"{all_stops['net_displacement_m'].mean():.0f} m")
        st.caption(
            f"Longest stop: vessel **{longest_row['vessel_id']}**, "
            f"{longest_row['episode_start']} → {longest_row['episode_end']}."
        )

        st.subheader("🗺️ Interactive map")

        all_vessels = sorted(set(stops_at_port["vessel_id"]).union(stops_far_from_port["vessel_id"]))
        vessel_choice = st.selectbox("Filter map by vessel", ["All vessels"] + all_vessels)
        if vessel_choice != "All vessels":
            map_near = stops_at_port[stops_at_port["vessel_id"] == vessel_choice]
            map_far = stops_far_from_port[stops_far_from_port["vessel_id"] == vessel_choice]
        else:
            map_near = stops_at_port
            map_far = stops_far_from_port

        map_col, legend_col = st.columns([5, 1])
        with map_col:
            m = build_map(ports, map_far, map_near, fit_to_data=(vessel_choice != "All vessels"))
            folium_static(m, width=1000, height=600)
        with legend_col:
            st.markdown(LEGEND_HTML, unsafe_allow_html=True)

        st.markdown(
            "<style>div[data-testid='stDownloadButton'] button {white-space: nowrap;}</style>",
            unsafe_allow_html=True,
        )
        col_dl1, col_dl_spacer, col_dl2 = st.columns([2, 3, 2])
        col_dl1.download_button(
            "Near port (CSV)",
            stops_at_port.drop(columns=["approach_track"], errors="ignore").to_csv(index=False).encode("utf-8"),
            "stops_at_port.csv",
            "text/csv",
            key="dl_at_port",
            use_container_width=True,
        )
        col_dl2.download_button(
            "Far from port (CSV)",
            stops_far_from_port.drop(columns=["approach_track"], errors="ignore").to_csv(index=False).encode("utf-8"),
            "stops_far_from_port.csv",
            "text/csv",
            key="dl_far_port",
            use_container_width=True,
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
