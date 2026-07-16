cat > /home/claude/ais_app/app.py << 'PYEOF'
"""
AIS Stationary Ship Detector
-----------------------------
Upload an AIS CSV, map your columns, and detect ships that stopped for
at least N hours - classified as near-port or far-from-port - shown on
an interactive map. Also detects "dark" vessels (AIS gaps of N+ hours),
accounting for regions with naturally low/no AIS coverage.

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


def shorten_id(vessel_id, keep=4):
    """Shorten a long vessel identifier for display purposes only, e.g.
    '123456789' -> '1234…6789'. Leaves short IDs untouched."""
    s = str(vessel_id)
    if len(s) <= keep * 2 + 1:
        return s
    return f"{s[:keep]}\u2026{s[-keep:]}"


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


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


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


@st.cache_data(show_spinner=False)
def detect_sts_candidates(stops_far_df, max_distance_m=150, min_overlap_hours=1.0):
    """Find pairs of DIFFERENT vessels whose far-from-port stops were both
    close together in space AND overlapping in time - a signal consistent
    with a ship-to-ship transfer (as opposed to two ships coincidentally
    sheltering in the same general area at different times)."""
    df = stops_far_df.reset_index(drop=True)
    if len(df) < 2:
        return pd.DataFrame()

    coords_rad = np.radians(df[["start_lat", "start_lon"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    radius_rad = max_distance_m / EARTH_RADIUS_M
    neighbor_lists = tree.query_radius(coords_rad, r=radius_rad)

    pairs = []
    seen = set()
    for i, neighbors in enumerate(neighbor_lists):
        for j in neighbors:
            j = int(j)
            if j <= i or (i, j) in seen:
                continue
            seen.add((i, j))
            if df.loc[i, "vessel_id"] == df.loc[j, "vessel_id"]:
                continue
            start_i, end_i = df.loc[i, "episode_start"], df.loc[i, "episode_end"]
            start_j, end_j = df.loc[j, "episode_start"], df.loc[j, "episode_end"]
            overlap_start = max(start_i, start_j)
            overlap_end = min(end_i, end_j)
            overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
            if overlap_hours < min_overlap_hours:
                continue
            dist_m = haversine_m(
                df.loc[i, "start_lat"], df.loc[i, "start_lon"], df.loc[j, "start_lat"], df.loc[j, "start_lon"]
            )
            pairs.append({
                "vessel_a": df.loc[i, "vessel_id"], "lat_a": df.loc[i, "start_lat"], "lon_a": df.loc[i, "start_lon"],
                "vessel_b": df.loc[j, "vessel_id"], "lat_b": df.loc[j, "start_lat"], "lon_b": df.loc[j, "start_lon"],
                "mid_lat": (df.loc[i, "start_lat"] + df.loc[j, "start_lat"]) / 2,
                "mid_lon": (df.loc[i, "start_lon"] + df.loc[j, "start_lon"]) / 2,
                "distance_m": dist_m,
                "overlap_start": overlap_start,
                "overlap_end": overlap_end,
                "overlap_hours": overlap_hours,
            })
    return pd.DataFrame(pairs)


@st.cache_data(show_spinner=False)
def score_sts_candidates(pairs_df, stops_far_df, max_distance_m, isolation_radius_m=1000, overlap_cap_hours=24):
    """Score each candidate pair on three factors:
    - how close together they were (closer = more suspicious)
    - how long their stops overlapped (longer = more suspicious)
    - how isolated that location is (few other vessels ever stop nearby = more
      suspicious; a spot where many different vessels stop is more likely a
      known shelter/waiting area than a private transfer point)."""
    if pairs_df.empty:
        return pairs_df

    pairs_df = pairs_df.copy()
    all_coords_rad = np.radians(stops_far_df[["start_lat", "start_lon"]].values)
    all_tree = BallTree(all_coords_rad, metric="haversine")
    isolation_radius_rad = isolation_radius_m / EARTH_RADIUS_M

    nearby_vessel_counts = []
    for _, row in pairs_df.iterrows():
        pt_rad = np.radians([[row["mid_lat"], row["mid_lon"]]])
        idx = all_tree.query_radius(pt_rad, r=isolation_radius_rad)[0]
        nearby_vessel_counts.append(stops_far_df.iloc[idx]["vessel_id"].nunique())
    pairs_df["vessels_nearby"] = nearby_vessel_counts

    pairs_df["distance_score"] = (1 - (pairs_df["distance_m"] / max_distance_m)).clip(0, 1)
    pairs_df["overlap_score"] = (pairs_df["overlap_hours"] / overlap_cap_hours).clip(0, 1)
    # 2 nearby vessels = just this pair (fully isolated, score 1); decays toward 0 by +8 more vessels
    pairs_df["isolation_score"] = (1 - ((pairs_df["vessels_nearby"] - 2) / 8)).clip(0, 1)

    pairs_df["score"] = (
        0.4 * pairs_df["distance_score"] + 0.4 * pairs_df["overlap_score"] + 0.2 * pairs_df["isolation_score"]
    )
    return pairs_df.sort_values("score", ascending=False).reset_index(drop=True)


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


# ============================================================
# Dark vessel detection
# ============================================================
@st.cache_data(show_spinner=False)
def compute_region_bounds(df, edge_margin_pct=5):
    """Bounding box of the whole uploaded dataset, plus an inward margin band.
    Used to flag when a vessel's last known position was near the edge of the
    monitored region - since a dataset that only covers a region (not the
    whole globe) cannot distinguish 'went dark' from 'sailed outside the area
    we have any visibility into' once a vessel nears that edge."""
    lat_min, lat_max = float(df["lat"].min()), float(df["lat"].max())
    lon_min, lon_max = float(df["lon"].min()), float(df["lon"].max())
    lat_margin = (lat_max - lat_min) * (edge_margin_pct / 100)
    lon_margin = (lon_max - lon_min) * (edge_margin_pct / 100)
    return {
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max,
        "lat_margin": lat_margin, "lon_margin": lon_margin,
    }


def is_near_region_edge(lat, lon, bounds):
    """Vectorized check: is this point within the inward margin band of the
    dataset's overall bounding box?"""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    return (
        (lat <= bounds["lat_min"] + bounds["lat_margin"]) |
        (lat >= bounds["lat_max"] - bounds["lat_margin"]) |
        (lon <= bounds["lon_min"] + bounds["lon_margin"]) |
        (lon >= bounds["lon_max"] - bounds["lon_margin"])
    )


@st.cache_data(show_spinner=False)
def build_coverage_grid(df, grid_size_deg):
    """Estimate typical AIS reporting density per grid cell using the WHOLE
    fleet's pings (not just one vessel) - a self-calibrating proxy for how
    well-covered an area is by AIS receivers/satellites. In cells where many
    different vessels' reports arrive close together in time, the area is
    'well covered'; in cells where reports (from anyone) are naturally sparse,
    the area has 'low coverage', making a long silence from one vessel there
    much less suspicious."""
    d = df[["t", "lat", "lon"]].copy()
    d["cell_lat"] = np.floor(d["lat"].astype("float64") / grid_size_deg) * grid_size_deg
    d["cell_lon"] = np.floor(d["lon"].astype("float64") / grid_size_deg) * grid_size_deg
    d = d.sort_values(["cell_lat", "cell_lon", "t"])
    d["gap_any_vessel_hours"] = (
        d.groupby(["cell_lat", "cell_lon"])["t"].diff().dt.total_seconds() / 3600
    )

    grid = d.groupby(["cell_lat", "cell_lon"]).agg(
        n_reports=("t", "size"),
        median_gap_hours=("gap_any_vessel_hours", "median"),
    ).reset_index()
    return grid


def lookup_cell_coverage(lat, lon, grid_size_deg, coverage_grid):
    """For arrays of lat/lon, look up each point's grid cell coverage stats."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    cell_lat = np.floor(lat / grid_size_deg) * grid_size_deg
    cell_lon = np.floor(lon / grid_size_deg) * grid_size_deg
    lookup_df = pd.DataFrame({"cell_lat": cell_lat, "cell_lon": cell_lon})
    merged = lookup_df.merge(coverage_grid, on=["cell_lat", "cell_lon"], how="left")
    return merged["n_reports"].values, merged["median_gap_hours"].values


@st.cache_data(show_spinner=False)
def detect_dark_gaps(df, dark_threshold_hours):
    """Find gaps in AIS reporting per vessel of at least dark_threshold_hours,
    regardless of whether position changed in between - candidate 'went dark'
    events. Also records how far the vessel appears to have moved between its
    last report and its reappearance, since large displacement over a long
    silence is itself informative (still moving vs. sitting still and dark)."""
    g = df.groupby("vessel_id", sort=False)
    d = df.copy()
    d["prev_time"] = g["t"].shift()
    d["prev_lat"] = g["lat"].shift()
    d["prev_lon"] = g["lon"].shift()
    d["gap_hours"] = (d["t"] - d["prev_time"]).dt.total_seconds() / 3600
    d = d.dropna(subset=["prev_time"])

    dark = d[d["gap_hours"] >= dark_threshold_hours][
        ["vessel_id", "prev_time", "prev_lat", "prev_lon", "t", "lat", "lon", "gap_hours"]
    ].rename(columns={
        "prev_time": "last_seen_time", "prev_lat": "last_seen_lat", "prev_lon": "last_seen_lon",
        "t": "reappear_time", "lat": "reappear_lat", "lon": "reappear_lon",
    }).reset_index(drop=True)

    if dark.empty:
        return dark.assign(displacement_m=[])

    dark["displacement_m"] = haversine_m(
        dark["last_seen_lat"], dark["last_seen_lon"], dark["reappear_lat"], dark["reappear_lon"]
    )
    return dark


@st.cache_data(show_spinner=False)
def classify_dark_gaps(dark_df, coverage_grid, bounds, grid_size_deg, well_covered_gap_hours, min_reports_per_cell):
    """Classify each dark-gap event into one of four buckets, checked in this
    order (first match wins):

    1. Possibly left monitored area - the vessel's last known position was near
       the edge of the dataset's overall bounding box, so we cannot tell
       whether it went dark or simply sailed outside our visibility.
    2. Insufficient coverage data - too few historical reports (from any
       vessel) in that grid cell to trust a coverage estimate either way.
    3. Likely dark (AIS off) - the cell is normally well-reported (typical
       gap <= well_covered_gap_hours) yet this vessel went silent much longer.
    4. Likely coverage gap - the cell is naturally sparse for everyone, so a
       long silence there isn't surprising on its own.
    """
    if dark_df.empty:
        return dark_df.assign(
            near_region_edge=[], cell_n_reports=[], cell_median_gap_hours=[], classification=[]
        )

    dark_df = dark_df.copy()

    near_edge = is_near_region_edge(dark_df["last_seen_lat"].values, dark_df["last_seen_lon"].values, bounds)
    dark_df["near_region_edge"] = near_edge

    n_reports, median_gap_hours = lookup_cell_coverage(
        dark_df["last_seen_lat"].values, dark_df["last_seen_lon"].values, grid_size_deg, coverage_grid
    )
    dark_df["cell_n_reports"] = n_reports
    dark_df["cell_median_gap_hours"] = median_gap_hours

    insufficient = (
        dark_df["cell_n_reports"].isna()
        | (dark_df["cell_n_reports"] < min_reports_per_cell)
        | dark_df["cell_median_gap_hours"].isna()
    )
    well_covered = dark_df["cell_median_gap_hours"] <= well_covered_gap_hours

    conditions = [
        dark_df["near_region_edge"],
        insufficient,
        well_covered,
    ]
    choices = ["Possibly left monitored area", "Insufficient coverage data", "Likely dark (AIS off)"]
    dark_df["classification"] = np.select(conditions, choices, default="Likely coverage gap")

    return dark_df.sort_values("gap_hours", ascending=False).reset_index(drop=True)


def build_map(
    ports_df, stops_far_df, stops_near_df, fit_to_data=False, tile_style="light",
    sts_pairs_df=None, dark_df=None, dark_categories_to_show=None, show_region_bounds=None,
):
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

    if tile_style == "satellite":
        m = folium.Map(location=[center_lat, center_lon], zoom_start=3, tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
            name="Satellite",
        ).add_to(m)
    else:
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

    if sts_pairs_df is not None and len(sts_pairs_df):
        sts_layer = folium.FeatureGroup(name="Potential ship-to-ship transfers", show=True)
        for _, row in sts_pairs_df.iterrows():
            folium.PolyLine(
                locations=[[row["lat_a"], row["lon_a"]], [row["lat_b"], row["lon_b"]]],
                color="purple", weight=3, opacity=0.85,
            ).add_to(sts_layer)
            popup_html = (
                "<div style='word-wrap: break-word; overflow-wrap: break-word; white-space: normal; max-width: 260px;'>"
                f"<b>Vessel A:</b> {row['vessel_a']}<br>"
                f"<b>Vessel B:</b> {row['vessel_b']}<br>"
                f"<b>Distance apart:</b> {row['distance_m']:.0f} m<br>"
                f"<b>Overlap:</b> {row['overlap_hours']:.1f} h "
                f"({row['overlap_start']} &rarr; {row['overlap_end']})<br>"
                f"<b>Suspicion score:</b> {row['score']:.2f}"
                "</div>"
            )
            folium.CircleMarker(
                location=[row["mid_lat"], row["mid_lon"]],
                radius=7, color="purple", fill=True, fill_opacity=0.9,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['vessel_a']} \u2194 {row['vessel_b']} (score {row['score']:.2f})",
            ).add_to(sts_layer)
        sts_layer.add_to(m)

    dark_colors = {
        "Likely dark (AIS off)": "black",
        "Likely coverage gap": "orange",
        "Possibly left monitored area": "gray",
        "Insufficient coverage data": "lightgray",
    }
    if dark_df is not None and len(dark_df) and dark_categories_to_show:
        for category in dark_categories_to_show:
            subset = dark_df[dark_df["classification"] == category]
            if not len(subset):
                continue
            color = dark_colors.get(category, "black")
            layer = folium.FeatureGroup(name=f"Dark vessels: {category}", show=(category == "Likely dark (AIS off)"))
            for _, row in subset.iterrows():
                popup_html = (
                    "<div style='word-wrap: break-word; overflow-wrap: break-word; white-space: normal; max-width: 260px;'>"
                    f"<b>Vessel:</b> {row['vessel_id']}<br>"
                    f"<b>Classification:</b> {row['classification']}<br>"
                    f"<b>Last seen:</b> {row['last_seen_time']}<br>"
                    f"<b>Reappeared:</b> {row['reappear_time']}<br>"
                    f"<b>Silence:</b> {row['gap_hours']:.1f} h<br>"
                    f"<b>Displacement while dark:</b> {row['displacement_m']:.0f} m<br>"
                    f"<b>Typical local reporting gap:</b> "
                    f"{'n/a' if pd.isna(row['cell_median_gap_hours']) else f'{row[chr(39)+chr(99)+chr(101)+chr(108)+chr(108)+chr(95)+chr(109)+chr(101)+chr(100)+chr(105)+chr(97)+chr(110)+chr(95)+chr(103)+chr(97)+chr(112)+chr(95)+chr(104)+chr(111)+chr(117)+chr(114)+chr(115)+chr(39)]:.1f} h'}"
                    "</div>"
                )
                folium.PolyLine(
                    locations=[[row["last_seen_lat"], row["last_seen_lon"]], [row["reappear_lat"], row["reappear_lon"]]],
                    color=color, weight=2, opacity=0.7, dash_array="2,8",
                ).add_to(layer)
                folium.CircleMarker(
                    location=[row["last_seen_lat"], row["last_seen_lon"]],
                    radius=6, color=color, fill=True, fill_opacity=0.9,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{row['vessel_id']} \u2014 {row['classification']}",
                ).add_to(layer)
            layer.add_to(m)

    if show_region_bounds:
        b = show_region_bounds
        folium.Rectangle(
            bounds=[[b["lat_min"], b["lon_min"]], [b["lat_max"], b["lon_max"]]],
            color="steelblue", weight=1.5, fill=False, dash_array="6,4",
            tooltip="Dataset bounding box",
        ).add_to(m)
        folium.Rectangle(
            bounds=[
                [b["lat_min"] + b["lat_margin"], b["lon_min"] + b["lon_margin"]],
                [b["lat_max"] - b["lat_margin"], b["lon_max"] - b["lon_margin"]],
            ],
            color="steelblue", weight=1, fill=False, dash_array="2,6", opacity=0.6,
            tooltip="Edge margin - stops here may just have left the monitored area",
        ).add_to(m)

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
    <span style="color:#888; font-weight:bold;">- - -</span> Approach path leading into a stop<br>
    <span style="color:#8e24aa; font-weight:bold;">&#9679;&#8212;&#9679;</span> Potential ship-to-ship transfer<br>
    <span style="display:inline-block;width:12px;height:12px;background:#111;border-radius:50%;margin-right:6px;"></span>Likely dark (AIS off)<br>
    <span style="display:inline-block;width:12px;height:12px;background:#e67e22;border-radius:50%;margin-right:6px;"></span>Likely coverage gap<br>
    <span style="display:inline-block;width:12px;height:12px;background:#888;border-radius:50%;margin-right:6px;"></span>Possibly left monitored area<br>
    <span style="display:inline-block;width:12px;height:12px;background:#ccc;border-radius:50%;margin-right:6px;"></span>Insufficient coverage data
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
for key, default in [
    ("result", None), ("stops_at_port", None), ("stops_far_from_port", None),
    ("ports", None), ("n_rows", None), ("n_vessels", None), ("map_cache", {}),
    ("sts_pairs", None), ("dark_gaps", None), ("region_bounds", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# Sidebar - inputs
# ============================================================
st.title("🚢 AIS Stationary Ship Detector")
st.write(
    "Upload an AIS CSV, map your columns, and find ships that stayed in place "
    "for at least N hours - split into stops near a port and stops far from any port. "
    "Also flags vessels that went quiet (\"dark\") for an extended period, "
    "accounting for areas with naturally low AIS coverage."
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

    st.sidebar.header("5. Stationary stop parameters")
    dist_threshold_m = st.sidebar.slider("Same-position tolerance (m)", 10, 1000, 100, step=10)
    min_gap_hours = st.sidebar.slider("Minimum stationary duration (hours)", 1, 24, 1)
    port_dist_threshold_m = st.sidebar.slider("Near-port distance threshold (m)", 500, 50000, 5000, step=500)

    st.sidebar.header("6. Ship-to-ship transfer detection")
    st.sidebar.caption("Flags pairs of different vessels stopped close together, far from any port, with overlapping stop times.")
    sts_max_distance_m = st.sidebar.slider("Max distance between vessels (m)", 20, 500, 150, step=10)
    sts_min_overlap_hours = st.sidebar.slider("Min overlapping stop duration (hours)", 0.5, 12.0, 1.0, step=0.5)

    st.sidebar.header("7. Dark vessel detection")
    st.sidebar.caption(
        "Flags vessels that stopped transmitting AIS for an extended period. "
        "Classification accounts for areas with naturally low AIS coverage and "
        "for vessels that may simply have left the region this dataset covers."
    )
    dark_threshold_hours = st.sidebar.slider("Dark threshold (hours)", 1, 48, 8)
    coverage_grid_size_deg = st.sidebar.select_slider(
        "Coverage grid cell size (degrees)", options=[0.1, 0.25, 0.5, 1.0, 2.0], value=0.5,
        help="Smaller cells = finer-grained coverage estimate, but need more data per cell to be reliable.",
    )
    well_covered_gap_hours = st.sidebar.slider(
        "Well-covered area: max typical gap (hours)", 0.5, 6.0, 2.0, step=0.5,
        help="If the typical reporting gap in a cell (from ANY vessel) is at or below this, the area counts as well-covered.",
    )
    min_reports_per_cell = st.sidebar.slider(
        "Min reports per cell for confidence", 5, 200, 20, step=5,
        help="Cells with fewer historical reports than this are marked 'insufficient data' rather than guessed at.",
    )
    edge_margin_pct = st.sidebar.slider(
        "Region-edge margin (% of bounding box)", 1, 20, 5,
        help="Stops whose last known position falls within this margin of the dataset's outer edge are treated as 'possibly left the monitored area' rather than dark.",
    )

    run = st.sidebar.button("Run analysis", type="primary")

    # --- Only (re)compute when the button is actually clicked. -------------
    # Everything computed here is written into st.session_state so it
    # survives later reruns that aren't caused by this button (map clicks,
    # tab switches, download-button clicks, etc.)
    if run:
        st.session_state.map_cache = {}
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

            with st.spinner("Scanning for ship-to-ship transfer candidates..."):
                sts_pairs = detect_sts_candidates(
                    st.session_state.stops_far_from_port,
                    max_distance_m=sts_max_distance_m,
                    min_overlap_hours=sts_min_overlap_hours,
                )
                sts_pairs = score_sts_candidates(
                    sts_pairs,
                    st.session_state.stops_far_from_port,
                    max_distance_m=sts_max_distance_m,
                )
            st.session_state.sts_pairs = sts_pairs
        else:
            st.session_state.result = result
            st.session_state.ports = None
            st.session_state.stops_at_port = None
            st.session_state.stops_far_from_port = None
            st.session_state.sts_pairs = None

        with st.spinner("Computing region bounds..."):
            bounds = compute_region_bounds(df, edge_margin_pct)
        with st.spinner("Building self-calibrated AIS coverage grid..."):
            coverage_grid = build_coverage_grid(df, coverage_grid_size_deg)
        with st.spinner("Detecting dark vessel gaps..."):
            dark_gaps = detect_dark_gaps(df, dark_threshold_hours)
            dark_gaps = classify_dark_gaps(
                dark_gaps, coverage_grid, bounds, coverage_grid_size_deg,
                well_covered_gap_hours, min_reports_per_cell,
            )
        st.session_state.dark_gaps = dark_gaps
        st.session_state.region_bounds = bounds

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
        sts_pairs = st.session_state.sts_pairs
        dark_gaps = st.session_state.dark_gaps
        region_bounds = st.session_state.region_bounds

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
        picker_col1, picker_col2 = st.columns([2, 1])
        vessel_choice = picker_col1.selectbox("Filter map by vessel", ["All vessels"] + all_vessels)
        map_style = picker_col2.radio("Map style", ["Light", "Satellite"], horizontal=True)

        dark_cat_col, bounds_col = st.columns([3, 1])
        all_dark_categories = [
            "Likely dark (AIS off)", "Likely coverage gap",
            "Possibly left monitored area", "Insufficient coverage data",
        ]
        dark_categories_to_show = dark_cat_col.multiselect(
            "Show on map: dark-vessel categories", all_dark_categories,
            default=["Likely dark (AIS off)"],
        )
        show_region_bounds_toggle = bounds_col.checkbox("Show region bounds", value=False)

        if vessel_choice != "All vessels":
            map_near = stops_at_port[stops_at_port["vessel_id"] == vessel_choice]
            map_far = stops_far_from_port[stops_far_from_port["vessel_id"] == vessel_choice]
            if sts_pairs is not None and len(sts_pairs):
                map_sts = sts_pairs[
                    (sts_pairs["vessel_a"] == vessel_choice) | (sts_pairs["vessel_b"] == vessel_choice)
                ]
            else:
                map_sts = sts_pairs
            if dark_gaps is not None and len(dark_gaps):
                map_dark = dark_gaps[dark_gaps["vessel_id"] == vessel_choice]
            else:
                map_dark = dark_gaps
        else:
            map_near = stops_at_port
            map_far = stops_far_from_port
            map_sts = sts_pairs
            map_dark = dark_gaps

        map_col, legend_col = st.columns([5, 1])
        with map_col:
            cache_key = (
                vessel_choice, map_style, tuple(sorted(dark_categories_to_show)), show_region_bounds_toggle,
            )
            if cache_key not in st.session_state.map_cache:
                st.session_state.map_cache[cache_key] = build_map(
                    ports, map_far, map_near,
                    fit_to_data=(vessel_choice != "All vessels"),
                    tile_style=map_style.lower(),
                    sts_pairs_df=map_sts,
                    dark_df=map_dark,
                    dark_categories_to_show=dark_categories_to_show,
                    show_region_bounds=region_bounds if show_region_bounds_toggle else None,
                )
            m = st.session_state.map_cache[cache_key]
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

        st.subheader("🔗 Potential ship-to-ship transfers")
        st.caption(
            "Pairs of different vessels stopped far from any port, within "
            f"{sts_max_distance_m} m of each other, with stop windows overlapping "
            f"at least {sts_min_overlap_hours}h. Ranked by a suspicion score "
            "combining distance, overlap duration, and how isolated the spot is."
        )
        if sts_pairs is not None and len(sts_pairs):
            metric_col, toggle_col = st.columns([3, 1])
            metric_col.metric("Candidate pairs found", len(sts_pairs))
            show_full_ids = toggle_col.checkbox("Show full vessel IDs", value=False, key="sts_full_ids")

            display_cols = [
                "vessel_a", "vessel_b", "distance_m", "overlap_hours",
                "overlap_start", "overlap_end", "vessels_nearby", "score",
            ]
            sts_display = sts_pairs[display_cols].round(
                {"distance_m": 0, "overlap_hours": 1, "score": 2}
            )
            if not show_full_ids:
                sts_display["vessel_a"] = sts_display["vessel_a"].apply(shorten_id)
                sts_display["vessel_b"] = sts_display["vessel_b"].apply(shorten_id)

            st.dataframe(
                sts_display,
                use_container_width=True,
                column_config={
                    "vessel_a": st.column_config.TextColumn("Vessel A", width="small"),
                    "vessel_b": st.column_config.TextColumn("Vessel B", width="small"),
                    "distance_m": st.column_config.NumberColumn("Distance (m)", width="small"),
                    "overlap_hours": st.column_config.NumberColumn("Overlap (h)", width="small"),
                    "vessels_nearby": st.column_config.NumberColumn("Nearby vessels", width="small"),
                    "score": st.column_config.NumberColumn("Score", width="small"),
                },
            )
            if not show_full_ids:
                st.caption("Vessel IDs shortened for readability — tick \"Show full vessel IDs\" or use the CSV download for the complete values.")
            st.download_button(
                "Ship-to-ship candidates (CSV)",
                sts_pairs.to_csv(index=False).encode("utf-8"),
                "sts_candidates.csv",
                "text/csv",
                key="dl_sts",
            )
        else:
            st.info("No ship-to-ship transfer candidates found with the current thresholds.")

        st.subheader("🌑 Dark vessel detection")
        st.caption(
            f"Vessels silent for at least {dark_threshold_hours}h, classified using a coverage grid "
            "built from this dataset's own reporting density, plus a check for whether the vessel's "
            "last known position was near the edge of the area this dataset covers."
        )
        if dark_gaps is not None and len(dark_gaps):
            counts = dark_gaps["classification"].value_counts()
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            dcol1.metric("Likely dark (AIS off)", int(counts.get("Likely dark (AIS off)", 0)))
            dcol2.metric("Likely coverage gap", int(counts.get("Likely coverage gap", 0)))
            dcol3.metric("Possibly left area", int(counts.get("Possibly left monitored area", 0)))
            dcol4.metric("Insufficient data", int(counts.get("Insufficient coverage data", 0)))

            filter_col, toggle_col2 = st.columns([3, 1])
            category_filter = filter_col.multiselect(
                "Filter table by classification", all_dark_categories, default=["Likely dark (AIS off)"],
            )
            show_full_ids_dark = toggle_col2.checkbox("Show full vessel IDs", value=False, key="dark_full_ids")

            dark_display_df = dark_gaps[dark_gaps["classification"].isin(category_filter)] if category_filter else dark_gaps
            display_cols_dark = [
                "vessel_id", "classification", "last_seen_time", "reappear_time",
                "gap_hours", "displacement_m", "cell_median_gap_hours", "cell_n_reports",
            ]
            dark_display = dark_display_df[display_cols_dark].round(
                {"gap_hours": 1, "displacement_m": 0, "cell_median_gap_hours": 2}
            )
            if not show_full_ids_dark:
                dark_display = dark_display.copy()
                dark_display["vessel_id"] = dark_display["vessel_id"].apply(shorten_id)

            st.dataframe(
                dark_display,
                use_container_width=True,
                column_config={
                    "vessel_id": st.column_config.TextColumn("Vessel", width="small"),
                    "classification": st.column_config.TextColumn("Classification", width="medium"),
                    "gap_hours": st.column_config.NumberColumn("Silence (h)", width="small"),
                    "displacement_m": st.column_config.NumberColumn("Moved while dark (m)", width="small"),
                    "cell_median_gap_hours": st.column_config.NumberColumn("Typical local gap (h)", width="small"),
                    "cell_n_reports": st.column_config.NumberColumn("Reports in cell", width="small"),
                },
            )
            if not show_full_ids_dark:
                st.caption("Vessel IDs shortened for readability — tick \"Show full vessel IDs\" or use the CSV download for the complete values.")
            st.download_button(
                "Dark vessel events (CSV)",
                dark_gaps.to_csv(index=False).encode("utf-8"),
                "dark_vessel_events.csv",
                "text/csv",
                key="dl_dark",
            )
        else:
            st.info("No dark-vessel gaps found with the current threshold.")

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

        dark_gaps = st.session_state.dark_gaps
        if dark_gaps is not None and len(dark_gaps):
            st.subheader("🌑 Dark vessel detection")
            counts = dark_gaps["classification"].value_counts()
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            dcol1.metric("Likely dark (AIS off)", int(counts.get("Likely dark (AIS off)", 0)))
            dcol2.metric("Likely coverage gap", int(counts.get("Likely coverage gap", 0)))
            dcol3.metric("Possibly left area", int(counts.get("Possibly left monitored area", 0)))
            dcol4.metric("Insufficient data", int(counts.get("Insufficient coverage data", 0)))
            st.dataframe(dark_gaps, use_container_width=True)
            st.download_button(
                "Dark vessel events (CSV)",
                dark_gaps.to_csv(index=False).encode("utf-8"),
                "dark_vessel_events.csv",
                "text/csv",
                key="dl_dark_plain",
            )
    elif not run:
        st.info("Set your parameters and click **Run analysis** in the sidebar.")

else:
    st.info("Upload an AIS CSV file in the sidebar to get started.")
