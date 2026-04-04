import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
import os

# ────────────── Page Setup ──────────────
st.set_page_config(
    page_title="Twitter Engagement Rate Dashboard",
    page_icon="🐦",
    layout="wide"
)
st.title("🐦 Twitter Engagement Rate Analysis Dashboard")

# ────────────── Load Dataset ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "twitter_data.csv")

try:
    df = pd.read_csv(DATA_PATH)
except Exception:
    st.error("🚫 Twitter CSV file not found. Check path in .env")
    st.stop()

df["text"] = df["text"].astype(str)

# ────────────── Engagement Metrics ──────────────
df["Total_Engagement"] = (
    df["public_metrics_like_count"] +
    df["public_metrics_retweet_count"] +
    df["public_metrics_reply_count"] +
    df["public_metrics_quote_count"] +
    df["public_metrics_bookmark_count"]
)

df["Engagement_Rate"] = (
    df["Total_Engagement"] /
    df["public_metrics_impression_count"].replace(0, np.nan)
) * 100

df["Engagement_Rate"] = df["Engagement_Rate"].fillna(0)

# ────────────── Feature Engineering ──────────────
df["Text_Length"] = df["text"].apply(len)
df["Word_Count"] = df["text"].apply(lambda x: len(x.split()))

df["Hashtag_Count"] = df["entities_hashtags"].astype(str).apply(
    lambda x: len(x.split(",")) if x != "nan" else 0
)

df["Mention_Count"] = df["entities_mentions"].astype(str).apply(
    lambda x: len(x.split(",")) if x != "nan" else 0
)

df["URL_Count"] = df["entities_urls"].astype(str).apply(
    lambda x: len(x.split(",")) if x != "nan" else 0
)

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df["Hour"] = df["created_at"].dt.hour.fillna(0)
df["DayOfWeek"] = df["created_at"].dt.dayofweek.fillna(0)

# ────────────── ML Dataset ──────────────
features = [
    "public_metrics_like_count",
    "public_metrics_retweet_count",
    "public_metrics_reply_count",
    "public_metrics_quote_count",
    "public_metrics_bookmark_count",
    "Text_Length",
    "Word_Count",
    "Hashtag_Count",
    "Mention_Count",
    "URL_Count",
    "Hour",
    "DayOfWeek"
]

df_ml = df[features + ["Engagement_Rate"]].fillna(0)

X = df_ml[features]
y = df_ml["Engagement_Rate"]

# ────────────── Train/Test Split ──────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ────────────── Model Training ──────────────
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# ────────────── Feature Importance ──────────────
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

fig = px.bar(
    x=[features[i] for i in indices],
    y=importances[indices],
    color=importances[indices],
    color_continuous_scale="Viridis",
    title="Feature Importance for Predicting Twitter Engagement Rate"
)

fig.update_layout(
    xaxis_title="Features",
    yaxis_title="Importance Score",
    xaxis_tickangle=-45
)

# ────────────── Layout ──────────────
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
    🏆 *{features[indices[0]]}*

    This feature has the strongest impact on Twitter engagement rate prediction.
    """)

# ────────────── Sample Preview ──────────────
st.subheader("📄 Sample Tweet Data")
st.dataframe(
    df[["text", "Engagement_Rate"]].sort_values(
        "Engagement_Rate", ascending=False
    ).head(10),
    use_container_width=True
)
