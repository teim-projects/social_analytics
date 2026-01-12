import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
import os

# ───────────────── Streamlit Setup ─────────────────
st.set_page_config(page_title="YouTube Top & Bottom Videos Dashboard", page_icon="🎬", layout="wide")
st.title("🎬 YouTube Top & Bottom Videos Dashboard")

# ───────────────── Load dataset ─────────────────
load_dotenv()

VIDEO_DATA_PATH = os.getenv("Youtube_video_data")
CHANNEL_DATA_PATH = os.getenv("Youtube_channel_data")
COMMENTS_DATA_PATH = os.getenv("Youtube_comment_data")

try:
    df_videos = pd.read_csv(VIDEO_DATA_PATH)
except FileNotFoundError:
    st.error("🚫 Video data file not found! Please check Youtube_video_data path in .env")
    st.stop()

# Optional datasets (not used here, but loaded safely)
df_channels = None
df_comments = None

if CHANNEL_DATA_PATH and os.path.exists(CHANNEL_DATA_PATH):
    df_channels = pd.read_csv(CHANNEL_DATA_PATH)

if COMMENTS_DATA_PATH and os.path.exists(COMMENTS_DATA_PATH):
    df_comments = pd.read_csv(COMMENTS_DATA_PATH)

# ───────────────── Preprocessing ─────────────────
# Engagement = Likes + Comments_Count
df_videos["Total_Engagement"] = df_videos["Likes"] + df_videos["Comments_Count"]

# Engagement Rate = (Engagement / Views) × 100
df_videos["Engagement_Rate_Percent"] = (
    (df_videos["Total_Engagement"] / df_videos["Views"].replace(0, pd.NA)) * 100
).fillna(0).round(2)

# ───────────────── Sidebar Filters ─────────────────
st.sidebar.header("📌 Filters")

channel_list = df_videos["Channel_Title"].unique().tolist()
selected_channels = st.sidebar.multiselect(
    "Select Channel(s)",
    options=channel_list,
    default=channel_list
)

definition_types = df_videos["Definition"].unique().tolist()
selected_defs = st.sidebar.multiselect(
    "Select Video Definition",
    options=definition_types,
    default=definition_types
)

# ───────────────── Apply Filters ─────────────────
df_filtered = df_videos[
    (df_videos["Channel_Title"].isin(selected_channels)) &
    (df_videos["Definition"].isin(selected_defs))
]

# ───────────────── TOP VIDEOS ─────────────────
top_videos = (
    df_filtered
    .sort_values(by="Engagement_Rate_Percent", ascending=False)
    .head(20)
)

st.subheader("🏆 Top 20 Videos by Engagement Rate")

st.dataframe(top_videos[[
    "Video_ID", "Title", "Channel_Title", "Views", "Likes",
    "Comments_Count", "Total_Engagement", "Engagement_Rate_Percent"
]])

csv_top = top_videos.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Top Videos CSV",
    data=csv_top,
    file_name="youtube_top_videos.csv",
    mime="text/csv"
)

fig_top = px.bar(
    top_videos,
    x="Title",
    y="Engagement_Rate_Percent",
    color="Channel_Title",
    color_discrete_sequence=px.colors.qualitative.Bold,
    title="Top 20 YouTube Videos by Engagement Rate"
)
st.plotly_chart(fig_top, use_container_width=True)

st.divider()

# ───────────────── BOTTOM (WORST) VIDEOS ─────────────────
bottom_videos = (
    df_filtered
    .sort_values(by="Engagement_Rate_Percent", ascending=True)
    .head(20)
)

st.subheader("📉 Bottom 20 Videos by Engagement Rate")

st.dataframe(bottom_videos[[
    "Video_ID", "Title", "Channel_Title", "Views", "Likes",
    "Comments_Count", "Total_Engagement", "Engagement_Rate_Percent"
]])

csv_bottom = bottom_videos.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Bottom Videos CSV",
    data=csv_bottom,
    file_name="youtube_bottom_videos.csv",
    mime="text/csv"
)

fig_bottom = px.bar(
    bottom_videos,
    x="Title",
    y="Engagement_Rate_Percent",
    color="Channel_Title",
    color_discrete_sequence=px.colors.qualitative.Bold,
    title="Bottom 20 YouTube Videos by Engagement Rate"
)
st.plotly_chart(fig_bottom, use_container_width=True) 

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from dotenv import load_dotenv
# import os

# # ───────────────── Streamlit Setup ─────────────────
# st.set_page_config(page_title="YouTube Top Videos Dashboard", page_icon="🎬", layout="wide")
# st.title("🎬 YouTube Top Videos Dashboard")

# # ───────────────── Load dataset ─────────────────
# load_dotenv()

# VIDEO_DATA_PATH = os.getenv("Youtube_video_data")
# CHANNEL_DATA_PATH = os.getenv("Youtube_channel_data")
# COMMENTS_DATA_PATH = os.getenv("Youtube_comment_data")

# try:
#     df_videos = pd.read_csv(VIDEO_DATA_PATH)
# except FileNotFoundError:
#     st.error("🚫 Video data file not found! Please check YOUTUBE_VIDEO_DATA path in .env")
#     st.stop()

# # Optional: load channels / comments if needed
# df_channels = None
# df_comments = None
# if CHANNEL_DATA_PATH and os.path.exists(CHANNEL_DATA_PATH):
#     df_channels = pd.read_csv(CHANNEL_DATA_PATH)

# if COMMENTS_DATA_PATH and os.path.exists(COMMENTS_DATA_PATH):
#     df_comments = pd.read_csv(COMMENTS_DATA_PATH)

# # ───────────────── Preprocessing ─────────────────
# # Engagement = Likes + Comments_Count
# df_videos["Total_Engagement"] = df_videos["Likes"] + df_videos["Comments_Count"]

# # Engagement Rate = Engagement / Views × 100
# df_videos["Engagement_Rate_Percent"] = (
#     (df_videos["Total_Engagement"] / df_videos["Views"].replace(0, pd.NA)) * 100
# ).fillna(0).round(2)

# # ───────────────── Sidebar Filters ─────────────────
# st.sidebar.header("📌 Filters")

# channel_list = df_videos["Channel_Title"].unique().tolist()
# selected_channels = st.sidebar.multiselect("Select Channel(s)", options=channel_list, default=channel_list)

# definition_types = df_videos["Definition"].unique().tolist()
# selected_defs = st.sidebar.multiselect("Select Video Definition", options=definition_types, default=definition_types)

# # ───────────────── Apply Filters ─────────────────
# df_filtered = df_videos[
#     (df_videos["Channel_Title"].isin(selected_channels)) &
#     (df_videos["Definition"].isin(selected_defs))
# ]

# # Select Top 20 videos by Engagement Rate
# top_videos = df_filtered.sort_values(by="Engagement_Rate_Percent", ascending=False).head(20)

# # ───────────────── Display Table ─────────────────
# st.subheader("Top 20 Videos by Engagement Rate")

# st.dataframe(top_videos[[
#     "Video_ID", "Title", "Channel_Title", "Views", "Likes",
#     "Comments_Count", "Total_Engagement", "Engagement_Rate_Percent"
# ]])

# # ───────────────── Bar Chart ─────────────────
# fig = px.bar(
#     top_videos,
#     x="Title",
#     y="Engagement_Rate_Percent",
#     color="Channel_Title",
#     color_discrete_sequence=px.colors.qualitative.Bold,
#     title="Top 20 YouTube Videos by Engagement Rate"
# )

# st.plotly_chart(fig, use_container_width=True)

# # ───────────────── Download CSV ─────────────────
# csv = top_videos.to_csv(index=False).encode("utf-8")

# st.download_button(
#     label="📥 Download Top Videos CSV",
#     data=csv,
#     file_name="youtube_top_videos.csv",
#     mime="text/csv"
# )
