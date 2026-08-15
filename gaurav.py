import joblib
import dagshub
import mlflow
import pandas as pd
import streamlit as st
from pathlib import Path
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn import set_config
from time import sleep

# --- Basic Config ---
set_config(transform_output="pandas")
st.set_page_config(page_title="NYC Uber Demand", page_icon="🚕", layout="wide")

# --- MLflow & DagsHub Setup ---
# Credentials as per your confirmation
REPO_OWNER = 'gauravshuklaaaaa'
REPO_NAME = 'uber_demand_prediction'  # Underscore confirmed
MODEL_NAME = 'uber_new_model'
ALIAS = 'challenger'

# Initialize DagsHub
dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")

# --- Optimized Model Loader ---
@st.cache_resource 
def load_final_model():
    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    try:
        # Trying to pull from MLflow Registry via Alias
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        st.sidebar.error(f"MLflow Load Error: {e}")
        st.sidebar.info("Falling back to local model.joblib...")
        # Local Fallback
        return joblib.load(Path(__file__).parent / "models/model.joblib")

# --- Path Handling ---
root_path = Path(__file__).parent
plot_data_path = root_path / "data/external/plot_data.csv"
data_path = root_path / "data/processed/test.csv"
kmeans_path = root_path / "models/mb_kmeans.joblib"
scaler_path = root_path / "models/scaler.joblib"
encoder_path = root_path / "models/encoder.joblib"

# --- Load All Objects ---
model = load_final_model()
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
kmeans = joblib.load(kmeans_path)

# --- Data Loading ---
df_plot = pd.read_csv(plot_data_path)
df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")

# --- Streamlit UI Layout ---
st.title("NYC Uber Demand Prediction 🚕")
st.markdown("---")

# Sidebar
st.sidebar.header("Settings")
map_view = st.sidebar.selectbox("Choose Map View", ["Neighborhood Regions", "Complete NYC Map"])

# Inputs
st.subheader("📅 Select Date and Time")
c1, c2 = st.columns(2)
with c1:
    date_val = st.date_input("Date", dt.date(2016, 3, 20))
with c2:
    time_val = st.time_input("Time", dt.time(14, 30))

if date_val and time_val:
    # Logic for Prediction Window
    prediction_time = dt.datetime.combine(date_val, time_val) + dt.timedelta(minutes=15)
    target_ts = pd.Timestamp(prediction_time)
    
    st.success(f"Predicting for: **{prediction_time.strftime('%I:%M %p')}**")

    # Location Logic
    sample = df_plot.sample(1).reset_index(drop=True)
    curr_lat, curr_lon, curr_reg = sample["pickup_latitude"].item(), sample["pickup_longitude"].item(), sample["region"].item()
    
    st.info(f"📍 **Location:** {curr_lat}, {curr_lon} | **Region:** {curr_reg}")

    # Pipeline logic
    pipe = Pipeline([('encoder', encoder), ('reg', model)])
    
    # Colors
    colors_list = ["#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F", "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00", "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF", "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3", "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347", "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA"]
    region_colors = {r: colors_list[i % len(colors_list)] for i, r in enumerate(df_plot["region"].unique())}
    df_plot["color"] = df_plot["region"].map(region_colors)

    # Filtering & Mapping
    if map_view == "Neighborhood Regions":
        scaled_c = scaler.transform(sample.iloc[:, 0:2])
        dists = kmeans.transform(scaled_c).values.flatten().tolist()
        neighbors = sorted(list(enumerate(dists)), key=lambda x: x[1])[:9]
        neighbor_indices = sorted([n[0] for n in neighbors])
        
        display_df = df_plot[df_plot["region"].isin(neighbor_indices)]
        st.map(display_df, latitude="pickup_latitude", longitude="pickup_longitude", color="color", size=10)
        
        # Predictions for those regions
        input_data = df.loc[target_ts, :]
        input_data = input_data[input_data["region"].isin(neighbor_indices)].sort_values("region")
        preds = pipe.predict(input_data.drop(columns=["total_pickups"]))
        
        st.subheader("Neighborhood Demand Estimates")
        for i, idx in enumerate(neighbor_indices):
            st.write(f"🟢 **Region {idx}:** {int(max(0, preds[i]))} pickups expected")

    else:
        st.map(df_plot, latitude="pickup_latitude", longitude="pickup_longitude", color="color", size=2)
        input_data = df.loc[target_ts, :].sort_values("region")
        preds = pipe.predict(input_data.drop(columns=["total_pickups"]))
        st.write("NYC-wide predictions calculated.")