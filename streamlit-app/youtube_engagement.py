import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import isodate
import os

# ────────────── Streamlit Page Setup ──────────────
st.set_page_config(page_title="YouTube Engagement Rate Dashboard", page_icon="📊", layout="wide")
st.title("🎥 YouTube Engagement Rate Analysis Dashboard")

# ────────────── Load Dataset ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "ALL_VIDEO_DETAILS.csv")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error("🚫 File not found! Please check the path.")
    st.stop()

CHANNEL_ID = os.getenv("YOUTUBE_CHANNEL_ID")

# ────────────── Load Data ──────────────
df = pd.read_csv(DATA_PATH)
df = df[df["Channel_ID"] == CHANNEL_ID].copy()

if df.empty:
    st.error("❌ No videos found for this Channel ID.")
    st.stop()

# ────────────── Cleaning ──────────────
df["Views"] = df["Views"].replace(0, np.nan)
df.dropna(subset=["Views"], inplace=True)

# Engagement Rate
df["Engagement_Rate"] = ((df["Likes"] + df["Comments_Count"]) / df["Views"]) * 100

# ────────────── Feature Engineering ──────────────
df["Title_Length"] = df["Title"].astype(str).apply(len)
df["Description_Length"] = df["Description"].astype(str).apply(len)
df["Tag_Count"] = df["Tags"].astype(str).apply(lambda x: len(x.split("|")) if isinstance(x, str) else 0)

# Convert ISO 8601 duration to seconds
def iso_to_seconds(duration):
    try:
        return isodate.parse_duration(duration).total_seconds()
    except:
        return np.nan

df["Duration_Seconds"] = df["Duration"].apply(iso_to_seconds)

df["Has_Captions"] = df["Caption"].apply(lambda x: 1 if x == "true" else 0)
df["Is_HD"] = df["Definition"].apply(lambda x: 1 if x == "hd" else 0)

# ML Features
features = [
    "Likes", "Comments_Count", "Views",
    "Title_Length", "Description_Length", "Tag_Count",
    "Duration_Seconds", "Has_Captions", "Is_HD"
]

df_ml = df[features + ["Engagement_Rate"]].fillna(0)

X = df_ml[features]
y = df_ml["Engagement_Rate"]

# ────────────── Train/Test Split ──────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ────────────── RandomForest Model ──────────────
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# ────────────── Feature Importance Plot ──────────────
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

fig = px.bar(
    x=[features[i] for i in indices],
    y=importances[indices],
    color=importances[indices],
    color_continuous_scale=px.colors.sequential.Viridis,
    title="Feature Importance for Predicting YouTube Engagement Rate"
)
fig.update_layout(
    xaxis_title="Features",
    yaxis_title="Importance",
    xaxis_tickangle=-45
)

# ────────────── Layout: Graph (Left) | Metrics (Right) ──────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 Model Performance Metrics")
    c1, c2 = st.columns(2)
    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.3f}")

    st.markdown(f"""
    *Top Feature: {features[indices[0]]}*

    This feature has the strongest contribution to predicting engagement rate.
    """)
