import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ────────────── Streamlit Page Setup ──────────────
st.set_page_config(
    page_title="LinkedIn Engagement Rate Dashboard",
    page_icon="💼",
    layout="wide"
)
st.title("💼 LinkedIn Engagement Rate Analysis Dashboard")

# ────────────── Load Dataset ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "linkedin_new.csv")

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error("🚫 Could not load dataset. Check file path.")
    st.stop()

# ────────────── Basic Cleaning ──────────────
df["text"] = df["text"].astype(str)
df["hashtags"] = df["hashtags"].astype(str)

# ────────────── Engagement Metrics ──────────────
df["Total_Engagement"] = (
    df["likes"] +
    df["comments_count"] +
    df["shares"]
)

# LinkedIn has no impressions → raw engagement
df["Engagement_Rate"] = df["Total_Engagement"]

# ────────────── Text-Based Features ──────────────
df["Text_Length"] = df["text"].apply(len)
df["Word_Count"] = df["text"].apply(lambda x: len(x.split()))

df["Avg_Word_Length"] = df["text"].apply(
    lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0
)

df["Paragraph_Count"] = df["text"].apply(
    lambda x: len([p for p in x.split("\n") if p.strip() != ""])
)

df["Emoji_Count"] = df["text"].apply(
    lambda x: len(re.findall(r"[^\w\s,.]", x))
)

df["Mention_Count"] = df["text"].str.count("@")

# ────────────── Hashtag Features ──────────────
df["Hashtag_Count"] = df["hashtags"].apply(
    lambda x: len([h for h in x.split(",") if h.strip().startswith("#")])
)

df["Hashtag_Density"] = df["Hashtag_Count"] / df["Word_Count"].replace(0, 1)

# ────────────── Media Feature ──────────────
df["Has_Media"] = df["media_url"].notna().astype(int)

# ────────────── Time Features ──────────────
df["date"] = pd.to_datetime(df["date"], errors="coerce")

df["Hour"] = df["date"].dt.hour.fillna(0)
df["DayOfWeek"] = df["date"].dt.dayofweek.fillna(0)
df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

# ────────────── ML Features ──────────────
features = [
    "Text_Length",
    "Word_Count",
    "Avg_Word_Length",
    "Paragraph_Count",
    "Emoji_Count",
    "Mention_Count",
    "Hashtag_Count",
    "Hashtag_Density",
    "Has_Media",
    "Hour",
    "DayOfWeek",
    "Is_Weekend"
]

df_ml = df[features + ["Engagement_Rate"]].fillna(0)

X = df_ml[features]
y = df_ml["Engagement_Rate"]

# ────────────── Train/Test Split ──────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ────────────── RandomForest Model ──────────────
model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)
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
    color_continuous_scale=px.colors.sequential.Blues,
    title="Feature Importance for Predicting LinkedIn Engagement"
)

fig.update_layout(
    xaxis_title="Features",
    yaxis_title="Importance Score",
    xaxis_tickangle=-45
)

# ────────────── Layout: Graph | Metrics ──────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 Model Performance")
    c1, c2 = st.columns(2)

    c1.metric("R² Score", f"{r2:.3f}")
    c2.metric("MAE", f"{mae:.3f}")

    st.markdown(f"""
    **Top Influencing Feature:**  
    *{features[indices[0]]}*

    This feature contributes the most to predicting LinkedIn engagement.
    """)
