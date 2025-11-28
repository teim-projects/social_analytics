import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# ────────────── Streamlit page setup ──────────────
st.set_page_config(page_title="Engagement Rate Analysis", page_icon="📊", layout="wide")
st.title("📊 Instagram Engagement Rate Analysis Dashboard")

# ────────────── Load dataset ──────────────
load_dotenv()  # Load environment variables from .env file
DATA_PATH = os.getenv("DATA_PATH")
try:
    df = pd.read_excel(DATA_PATH)
except FileNotFoundError:
    st.error("🚫 File not found! Please check the path.")
    st.stop()

# ────────────── Preprocessing ──────────────
df['Reach'].replace(0, np.nan, inplace=True)
df.dropna(subset=['Reach'], inplace=True)
df['Total_Engagement'] = df['Likes'] + df['Comments'] + df['Shares'] + df['Saves']
df['Engagement_Rate_Percent'] = (df['Total_Engagement'] / df['Reach']) * 100
df['Engagement_Rate_Follower'] = (df['Total_Engagement'] / df['Follower_Count']) * 100

features = ['Hashtag_Count', 'Caption_Length', 'Likes', 'Comments', 'Shares', 'Saves',
            'Impressions', 'Follower_Count', 'Sentiment_Score_Comments']

# ────────────── ML Model ──────────────
df_ml = df[features + ['Engagement_Rate_Percent']].fillna(0)
X = df_ml[features]
y = df_ml['Engagement_Rate_Percent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("Feature Importance for Engagement Rate Prediction")

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plotly bar chart for feature importance
import plotly.express as px
fig = px.bar(
    x=[features[i] for i in indices],
    y=importances[indices],
    color=importances[indices],
    color_continuous_scale=px.colors.sequential.Viridis,
    title="Feature Importance for Engagement Rate Prediction"
)
fig.update_layout(
    xaxis_title="Features",
    yaxis_title="Importance",
    xaxis_tickangle=-45
)

# ───── Layout: Graph (left) | Metrics (right)
col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Model Performance Metrics")
    cola, colb = st.columns(2)
    cola.metric("R² Score", f"{r2:.3f}")
    colb.metric("MAE", f"{mae:.3f}")

    # Add a brief summary text (optional)
    st.markdown(f"""
    **Feature Importance Summary**  
    - Top feature: **{features[indices[0]]}**  
    - Indicates that `{features[indices[0]]}` is most influential for engagement rate prediction.  
    """)

# col1, col2 = st.columns([2,1])
# with col1:
#     # ────────────── Feature Importance ──────────────
#     st.subheader("Feature Importance")
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1]

#     fig, ax = plt.subplots(figsize=(10,5))
#     ax.bar(range(X.shape[1]), importances[indices], align='center')
#     ax.set_xticks(range(X.shape[1]))
#     ax.set_xticklabels([features[i] for i in indices], rotation=45, ha='right')
#     ax.set_title("Feature Importance for Engagement Rate Prediction")
#     st.pyplot(fig)
# with col2:
#     # ────────────── Metrics ──────────────
#     st.subheader("Model Performance Metrics")
#     cola, colb = st.columns(2)
#     cola.metric("R² Score", f"{r2:.3f}")
#     colb.metric("MAE", f"{mae:.3f}")

# ────────────── Engagement Table ──────────────
st.subheader("Top 20 Posts by Engagement Rate")
engagement_table = df[['Post_ID','Post_Type','Likes','Comments','Shares','Saves','Reach','Follower_Count',
                       'Total_Engagement','Engagement_Rate_Percent','Engagement_Rate_Follower']].sort_values(
                       by='Engagement_Rate_Percent', ascending=False)
engagement_table = engagement_table.head(20)
st.dataframe(engagement_table)

# ────────────── Download CSV ──────────────
csv = engagement_table.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Engagement Rate Report",
    data=csv,
    file_name="engagement_rate_report.csv",
    mime='text/csv'
)