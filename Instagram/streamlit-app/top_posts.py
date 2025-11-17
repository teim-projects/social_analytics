import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os

# ────────────── Streamlit page setup ──────────────
st.set_page_config(page_title="Top Posts Dashboard", page_icon="🏆", layout="wide")
st.title("🏆 Instagram Top Posts Dashboard")

# ────────────── Load dataset ──────────────
load_dotenv()  # Load environment variables from .env file
DATA_PATH = os.getenv("DATA_PATH")
try:
    df = pd.read_excel(DATA_PATH)
except FileNotFoundError:
    st.error("🚫 File not found! Please check the path.")
    st.stop()

# ────────────── Preprocessing ──────────────
df['Total_Engagement'] = df['Likes'] + df['Comments'] + df['Shares'] + df['Saves']
df['Engagement_Rate_Percent'] = (df['Total_Engagement'] / df['Reach'].replace(0, pd.NA)) * 100
df['Engagement_Rate_Percent'] = df['Engagement_Rate_Percent'].fillna(0).round(2)

# ────────────── Sidebar Filters ──────────────
st.sidebar.header("📌 Filters")
post_types = df['Post_Type'].unique().tolist()
selected_post_types = st.sidebar.multiselect("Select Post Type(s)", options=post_types, default=post_types)

# ────────────── Filtered Data ──────────────
df_filtered = df[df['Post_Type'].isin(selected_post_types)]
top_posts = df_filtered.sort_values(by='Engagement_Rate_Percent', ascending=False).head(20)

# ────────────── Display Top Posts ──────────────
st.subheader("Top 20 Posts by Engagement Rate")
st.dataframe(top_posts[['Post_ID', 'Post_Type', 'Likes', 'Comments', 'Shares', 'Saves', 'Reach', 'Engagement_Rate_Percent']])

# ────────────── Top Posts Bar Chart ──────────────
fig = px.bar(top_posts, x='Post_ID', y='Engagement_Rate_Percent', color='Post_Type',
             color_discrete_sequence=px.colors.qualitative.Bold,
             title="Engagement Rate of Top 20 Posts")
st.plotly_chart(fig, use_container_width=True)

# ────────────── Download CSV ──────────────
csv = top_posts.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Top Posts CSV",
    data=csv,
    file_name="top_posts.csv",
    mime='text/csv'
)