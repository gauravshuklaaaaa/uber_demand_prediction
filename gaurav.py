import datetime as dt
import os
from pathlib import Path

import dagshub
import joblib
import mlflow
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from sklearn import set_config
from sklearn.pipeline import Pipeline

# 1. Page & Config Setup
set_config(transform_output="pandas")
st.set_page_config(
    page_title="NYC Uber Demand Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stMetric label {
        color: #A0AAB0 !important;
        font-weight: 600;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8080 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255, 75, 75, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

REPO_OWNER = "gauravshuklaaaaa"
REPO_NAME = "uber_demand_prediction"
MODEL_NAME = "uber_demand_prediction_model"

# 2. MLflow & DagsHub Non-Interactive Setup
dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")

# Safely check secrets without crashing if secrets.toml is missing
if not dagshub_token:
    try:
        dagshub_token = st.secrets["DAGSHUB_USER_TOKEN"]
    except Exception:
        dagshub_token = None

if dagshub_token:
    dagshub.auth.add_app_token(dagshub_token)
    os.environ["MLFLOW_TRACKING_USERNAME"] = REPO_OWNER
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

try:
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
except Exception as e:
    st.sidebar.info("Running DagsHub in standalone mode")

mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

# 3. Path Configurations
root_path = Path(__file__).resolve().parent
plot_data_path = root_path / "data/external/plot_data.csv"
data_path = root_path / "data/processed/test.csv"
kmeans_path = root_path / "models/mb_kmeans.joblib"
scaler_path = root_path / "models/scaler.joblib"
encoder_path = root_path / "models/encoder.joblib"


# 4. Resource & Data Loaders
@st.cache_resource
def load_final_model():
    model_uri = f"models:/{MODEL_NAME}/latest"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        st.sidebar.success("⚡ Connected to MLflow Model Registry")
        return model
    except Exception as e:
        st.sidebar.warning(f"MLflow fetch fallback: {e}")
        return joblib.load(root_path / "models/model.joblib")


@st.cache_data
def load_datasets():
    df_p = pd.read_csv(plot_data_path)
    df_t = pd.read_csv(
        data_path, parse_dates=["tpep_pickup_datetime"]
    ).set_index("tpep_pickup_datetime")
    return df_p, df_t


# Initialize loaded artifacts
model = load_final_model()
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
kmeans = joblib.load(kmeans_path)
df_plot, df = load_datasets()

# 5. Hex/RGB Colors mapping for PyDeck & UI Legends
COLOR_PALETTE = [
    [255, 0, 0],
    [255, 69, 0],
    [255, 140, 0],
    [255, 215, 0],
    [173, 255, 47],
    [50, 205, 50],
    [0, 128, 0],
    [0, 255, 0],
    [0, 255, 255],
    [30, 144, 255],
    [0, 0, 255],
    [138, 43, 226],
    [255, 0, 255],
]

unique_regions = df_plot["region"].unique()
region_colors_rgb = {
    r: COLOR_PALETTE[i % len(COLOR_PALETTE)]
    for i, r in enumerate(unique_regions)
}

df_plot["color_rgb"] = df_plot["region"].map(region_colors_rgb)
df_plot["color_hex"] = df_plot["color_rgb"].apply(
    lambda rgb: f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
)

# Reliable Built-in PyDeck Map Themes (No Mapbox Token Required)
THEME_MAP = {
    "🌙 Dark Mode": pdk.map_styles.CARTO_DARK,
    "☀️ Light Mode": pdk.map_styles.CARTO_LIGHT,
    "🗺️ Road View": pdk.map_styles.ROAD,
}

# 6. Sidebar Controls
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/000000/uber.png",
        width=60,
    )
    st.title("Control Panel")
    st.markdown("---")

    map_view = st.radio(
        "🗺️ Map Perspective",
        ["Neighborhood Clusters", "Complete NYC Grid"],
        help="Select localized top-neighbor view or full spatial distribution.",
    )

    theme_label = st.selectbox(
        "🎨 Map Theme",
        list(THEME_MAP.keys()),
    )
    map_style = THEME_MAP[theme_label]

    st.markdown("---")
    st.caption("🚀 **Deployment Status:** MLflow & DagsHub Active")

# 7. Main Dashboard Header
st.title("🚕 NYC Uber Demand Intelligence Dashboard")
st.caption(
    "Real-time cluster-based demand forecasting and geospatial visualization engine."
)
st.markdown("---")

# Date & Time Selection Panel
st.subheader("⏱️ Select Target Datetime")
c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    date_val = st.date_input("Date", dt.date(2016, 3, 20))
with c2:
    time_val = st.time_input("Time", dt.time(14, 30))
with c3:
    st.write(" ")
    st.write(" ")
    predict_btn = st.button("🚀 Predict Demand", use_container_width=True)

if date_val and time_val:
    prediction_time = dt.datetime.combine(date_val, time_val) + dt.timedelta(
        minutes=15
    )
    target_ts = pd.Timestamp(prediction_time)

    if target_ts in df.index:
        matched_ts = target_ts
    else:
        nearest_idx = df.index.get_indexer([target_ts], method="nearest")[0]
        matched_ts = df.index[nearest_idx]
        st.info(
            f"ℹ️ Nearest Timestamp Matched: **{matched_ts.strftime('%Y-%m-%d %I:%M %p')}**"
        )

    # Dynamic Driver Location: Select driver location dynamically based on the target timestamp!
    # Different timestamp = Different Driver location.
    seed_value = abs(hash(str(matched_ts))) % 10000
    sample = df_plot.sample(1, random_state=seed_value).reset_index(drop=True)

    curr_lat, curr_lon, curr_reg = (
        sample["pickup_latitude"].item(),
        sample["pickup_longitude"].item(),
        sample["region"].item(),
    )

    # Inference Pipeline Setup
    pipe = Pipeline([("encoder", encoder), ("reg", model)])

    # Neighbor Filter Logic
    if map_view == "Neighborhood Clusters":
        scaled_c = scaler.transform(sample.iloc[:, 0:2])
        dists = kmeans.transform(scaled_c).values.flatten().tolist()
        neighbors = sorted(list(enumerate(dists)), key=lambda x: x[1])[:9]
        neighbor_indices = sorted([n[0] for n in neighbors])

        display_df = df_plot[
            df_plot["region"].isin(neighbor_indices)
        ].copy()
        input_data = (
            df.loc[[matched_ts], :]
            if isinstance(df.loc[matched_ts], pd.Series)
            else df.loc[matched_ts]
        )
        input_data = input_data[
            input_data["region"].isin(neighbor_indices)
        ]
    else:
        display_df = df_plot.copy()
        input_data = (
            df.loc[[matched_ts], :]
            if isinstance(df.loc[matched_ts], pd.Series)
            else df.loc[matched_ts]
        )

    # Perform Inference
    if not input_data.empty:
        raw_preds = pipe.predict(
            input_data.drop(columns=["total_pickups"], errors="ignore")
        )
        input_data["predicted_pickups"] = [
            int(max(0, p)) for p in raw_preds
        ]
        
        # Calculate Proximity (Distance to Driver) & Sort: Nearness First, Max Rides Second
        region_coords = df_plot.groupby("region")[["pickup_latitude", "pickup_longitude"]].mean().reset_index()
        input_data = input_data.merge(region_coords, on="region", how="left")
        
        input_data["dist_to_driver"] = np.sqrt(
            (input_data["pickup_latitude"] - curr_lat)**2 +
            (input_data["pickup_longitude"] - curr_lon)**2
        )
        
        input_data = input_data.sort_values(
            by=["dist_to_driver", "predicted_pickups"],
            ascending=[True, False]
        ).reset_index(drop=True)
    else:
        input_data["predicted_pickups"] = []

    # Cluster Points Layer
    cluster_layer = pdk.Layer(
        "ScatterplotLayer",
        data=display_df,
        get_position=["pickup_longitude", "pickup_latitude"],
        get_color="color_rgb",
        get_radius=80 if map_view == "Neighborhood Clusters" else 40,
        pickable=True,
        opacity=0.7,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=3,
        radius_max_pixels=15,
    )

    # Dedicated Current Location Pin Layer (Glowing Neon Marker)
    driver_df = pd.DataFrame([{
        "pickup_latitude": curr_lat,
        "pickup_longitude": curr_lon,
        "label": f"Driver Location (Region {curr_reg})"
    }])
    
    driver_pin_layer = pdk.Layer(
        "ScatterplotLayer",
        data=driver_df,
        get_position=["pickup_longitude", "pickup_latitude"],
        get_color=[255, 0, 128, 255], # Bright Pink/Magenta Pin
        get_radius=200 if map_view == "Neighborhood Clusters" else 400,
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        get_line_width=15,
        radius_min_pixels=8,
        radius_max_pixels=20,
    )

    # View State with Min/Max Zoom Constraints
    view_state = pdk.ViewState(
        latitude=curr_lat,
        longitude=curr_lon,
        zoom=11.5 if map_view == "Neighborhood Clusters" else 10.2,
        min_zoom=9.5,
        max_zoom=13.5,
        pitch=30,
    )

    col_map, col_info = st.columns([3, 1])

    with col_map:
        st.subheader("🗺️ Interactive Geospatial Map")
        r = pdk.Deck(
            layers=[cluster_layer, driver_pin_layer],
            initial_view_state=view_state,
            map_style=map_style,
            tooltip={
                "html": "<b>Region ID:</b> {region}<br/><b>Lat:</b> {pickup_latitude}<br/><b>Lon:</b> {pickup_longitude}",
                "style": {
                    "backgroundColor": "#121212",
                    "color": "white",
                    "fontSize": "12px",
                },
            },
        )
        st.pydeck_chart(r)

    with col_info:
        st.subheader("📍 Current Location")
        st.metric(label="Current Region", value=f"Region {curr_reg}")
        st.caption(f"Lat: `{curr_lat:.4f}` | Lon: `{curr_lon:.4f}`")
        st.markdown("🩷 **Magenta Pin** = Driver Location")

    # 8. Detailed Demand Predictions & Metrics Dashboard
    st.markdown("---")
    st.subheader("📊 Predicted Demand Insights")

    if not input_data.empty:
        tot_demand = input_data["predicted_pickups"].sum()
        avg_demand = int(input_data["predicted_pickups"].mean())
        
        max_idx = input_data["predicted_pickups"].argmax()
        max_region = int(input_data.iloc[max_idx]["region"])

        m1, m2, m3 = st.columns(3)
        m1.metric("Total NYC Predicted Pickups", f"{tot_demand:,} Rides")
        m2.metric("Highest Demand Region", f"Region {max_region}")
        m3.metric("Average Regional Pickup", f"~{avg_demand} Rides")

        st.write(" ")

        # Regional Metric Cards (Sorted by Nearness + Demand)
        st.markdown("##### 📍 Region-wise Breakdown (Ordered by Proximity & Demand)")
        region_cols = st.columns(3)
        for idx, row in enumerate(input_data.iterrows()):
            data = row[1]
            r_id = int(data["region"])
            p_val = int(data["predicted_pickups"])

            color_row = display_df[display_df["region"] == r_id]
            hex_c = color_row["color_hex"].iloc[0] if not color_row.empty else "#FF4B4B"

            is_driver_region = (r_id == curr_reg)
            is_high = p_val > avg_demand
            
            badge_text = "📍 Driver Region" if is_driver_region else ("↑ High Demand" if is_high else "↑ Normal Demand")
            badge_color = "#FF0080" if is_driver_region else ("#00E676" if is_high else "#A0AAB0")

            with region_cols[idx % 3]:
                st.markdown(
                    f"""
                    <div style="
                        background: rgba(255, 255, 255, 0.05);
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        border-radius: 12px;
                        padding: 15px 20px;
                        margin-bottom: 15px;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    ">
                        <div style="display: flex; align-items: center; gap: 8px; font-size: 14px; color: #A0AAB0; font-weight: 600;">
                            <span style="
                                display: inline-block;
                                width: 12px;
                                height: 12px;
                                background-color: {hex_c};
                                border-radius: 3px;
                            "></span>
                            <span>Region {r_id}</span>
                        </div>
                        <div style="font-size: 26px; font-weight: 700; color: white; margin: 6px 0;">
                            {p_val} Pickups
                        </div>
                        <div style="
                            font-size: 12px;
                            font-weight: 600;
                            color: {badge_color};
                            background: rgba(255, 255, 255, 0.05);
                            display: inline-block;
                            padding: 2px 8px;
                            border-radius: 10px;
                        ">
                            {badge_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Bar Visual Chart & Download Trigger
        st.markdown("---")
        c_chart, c_download = st.columns([3, 1])

        with c_chart:
            st.markdown("##### 📈 Regional Distribution Chart")
            st.bar_chart(
                data=input_data,
                x="region",
                y="predicted_pickups",
                color="#FF4B4B",
            )

        with c_download:
            st.markdown("##### 💾 Export Predictions")
            st.write("Download predicted demand metrics for downstream analysis.")
            csv_data = input_data[
                ["region", "predicted_pickups"]
            ].to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"uber_demand_{matched_ts.strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    else:
        st.error(
            "❌ Selected timestamp parameters standard datasets mein available nahi hain."
        )