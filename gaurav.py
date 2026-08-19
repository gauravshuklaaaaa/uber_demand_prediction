import datetime as dt
from pathlib import Path

import dagshub
import joblib
import mlflow
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

# Custom High-End Styling
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
    .legend-box {
        background: rgba(18, 18, 18, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 13px;
        max-height: 380px;
        overflow-y: auto;
    }
    .legend-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .legend-left {
        display: flex;
        align-items: center;
    }
    .color-dot {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        margin-right: 10px;
        display: inline-block;
    }
    .pickup-badge {
        background: rgba(255, 75, 75, 0.2);
        color: #FF8080;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: bold;
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

# 2. MLflow & DagsHub Setup
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
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


model = load_final_model()
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
kmeans = joblib.load(kmeans_path)
df_plot, df = load_datasets()

# 5. Hex/RGB Colors mapping
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

    map_style = st.selectbox(
        "🎨 Map Theme",
        ["mapbox://styles/mapbox/dark-v10", "mapbox://styles/mapbox/light-v10"],
    )

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

    # Driver / Sample Location Selection
    sample = df_plot.sample(1, random_state=42).reset_index(drop=True)
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
        ].sort_values("region")
    else:
        display_df = df_plot.copy()
        input_data = (
            df.loc[[matched_ts], :]
            if isinstance(df.loc[matched_ts], pd.Series)
            else df.loc[matched_ts]
        )
        input_data = input_data.sort_values("region")

    # Perform Inference
    if not input_data.empty:
        raw_preds = pipe.predict(
            input_data.drop(columns=["total_pickups"], errors="ignore")
        )
        input_data["predicted_pickups"] = [
            int(max(0, p)) for p in raw_preds
        ]
    else:
        input_data["predicted_pickups"] = []

    # Map Drivers Current Location Highlight Layer
    driver_df = pd.DataFrame([{
        "latitude": curr_lat,
        "longitude": curr_lon,
        "region": curr_reg
    }])

    driver_layer = pdk.Layer(
        "ScatterplotLayer",
        data=driver_df,
        get_position=["longitude", "latitude"],
        get_color=[255, 255, 255],  # White color dot for driver
        get_radius=150,
        pickable=True,
        stroked=True,
        get_line_color=[0, 0, 0],
        get_line_width=3,
        filled=True,
    )

    regions_layer = pdk.Layer(
        "ScatterplotLayer",
        data=display_df,
        get_position=["pickup_longitude", "pickup_latitude"],
        get_color="color_rgb",
        get_radius=80 if map_view == "Neighborhood Clusters" else 40,
        pickable=True,
        opacity=0.8,
        stroked=True,
        filled=True,
        radius_scale=1,
        radius_min_pixels=3,
        radius_max_pixels=15,
    )

    view_state = pdk.ViewState(
        latitude=curr_lat,
        longitude=curr_lon,
        zoom=12 if map_view == "Neighborhood Clusters" else 10,
        pitch=30,
    )

    # Layout Split: Left (PyDeck Map) | Right (Driver Info & Unified Legend/Predictions)
    col_map, col_info = st.columns([2.5, 1])

    with col_map:
        st.subheader("🗺️ Geospatial Map")
        r = pdk.Deck(
            layers=[regions_layer, driver_layer],
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
        # Driver's Current Location Section
        st.subheader("📍 Driver Current Status")
        st.success(f"**Current Region:** Region {curr_reg}")
        st.caption(f"Lat: `{curr_lat:.4f}` | Lon: `{curr_lon:.4f}`")

        st.markdown("---")
        st.subheader("🎨 Region Demand & Legend")

        # Unified Legend + Prediction Table Block
        if not input_data.empty:
            preds_map = dict(zip(input_data["region"], input_data["predicted_pickups"]))
            active_regions = sorted(display_df["region"].unique())
            
            legend_html = "<div class='legend-box'>"
            for reg in active_regions:
                hex_c = display_df[display_df["region"] == reg]["color_hex"].iloc[0]
                p_val = preds_map.get(reg, 0)
                is_driver = " (Driver Location)" if reg == curr_reg else ""

                legend_html += f"""
                <div class='legend-item'>
                    <div class='legend-left'>
                        <span class='color-dot' style='background-color: {hex_c};'></span>
                        <span><b>Region {reg}</b>{is_driver}</span>
                    </div>
                    <span class='pickup-badge'>{p_val} rides</span>
                </div>
                """
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)

    # 8. Detailed Insights & Metrics Dashboard
    st.markdown("---")
    st.subheader("📊 Detailed Demand Insights")

    if not input_data.empty:
        tot_demand = input_data["predicted_pickups"].sum()
        avg_demand = int(input_data["predicted_pickups"].mean())
        max_region = input_data.loc[
            input_data["predicted_pickups"].idxmax(), "region"
        ]

        m1, m2, m3 = st.columns(3)
        m1.metric("Total NYC Predicted Pickups", f"{tot_demand:,} Rides")
        m2.metric("Highest Demand Region", f"Region {max_region}")
        m3.metric("Average Regional Pickup", f"~{avg_demand} Rides")

        st.write(" ")

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