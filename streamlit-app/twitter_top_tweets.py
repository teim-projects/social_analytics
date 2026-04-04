import streamlit as st
import pandas as pd
import plotly.express as px
import re
import os

# ───────────────── Streamlit Setup ─────────────────
st.set_page_config(
    page_title="Twitter Top & Bottom Tweets Dashboard",
    page_icon="🐦",
    layout="wide"
)
st.title("🐦 Twitter Top & Bottom Tweets Dashboard")

# ───────────────── Load Environment & Data ─────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
TWITTER_DATA_PATH = os.path.join(DATA_DIR, "twitter_data.csv")

try:
    df = pd.read_csv(TWITTER_DATA_PATH)
except Exception:
    st.error("🚫 Twitter data file not found! Check TWITTER_DATA_PATH in .env")
    st.stop()

# ───────────────── Preprocessing ─────────────────
df["text"] = df["text"].astype(str)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# Remove Emojis
def remove_emojis(text):
    return re.sub("[\U00010000-\U0010ffff]", "", text)

df["clean_text"] = df["text"].apply(remove_emojis)

# Short text for plots
df["short_text"] = df["clean_text"].apply(
    lambda x: x[:60] + "..." if len(x) > 60 else x
)

# Engagement Calculation
df["Total_Engagement"] = (
    df["public_metrics_like_count"] +
    df["public_metrics_retweet_count"] +
    df["public_metrics_reply_count"] +
    df["public_metrics_quote_count"] +
    df["public_metrics_bookmark_count"]
)

df["Engagement_Rate_Percent"] = (
    (df["Total_Engagement"] /
     df["public_metrics_impression_count"].replace(0, pd.NA)) * 100
).fillna(0).round(2)

# ───────────────── Sidebar Filters ─────────────────
st.sidebar.header("📌 Filters")

df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce").dt.tz_localize(None)

from_date_default = df["created_at"].min().date()
to_date_default = df["created_at"].max().date()

col1, col2 = st.sidebar.columns(2)

from_date = col1.date_input(
    "From Date",
    value=from_date_default,
    key="from_date_input"
)

to_date = col2.date_input(
    "To Date",
    value=to_date_default,
    key="to_date_input"
)

if from_date > to_date:
    st.sidebar.error("🚫 From Date cannot be after To Date")

df_filtered = df[
    (df["created_at"].dt.date >= from_date) &
    (df["created_at"].dt.date <= to_date)
]

# ───────────────── TOP TWEETS ─────────────────
top_tweets = (
    df_filtered
    .sort_values(by="Engagement_Rate_Percent", ascending=False)
    .head(20)
)

st.subheader("🔥 Top 20 Tweets by Engagement Rate")

st.dataframe(top_tweets[[
    "created_at",
    "clean_text",
    "public_metrics_impression_count",
    "Total_Engagement",
    "Engagement_Rate_Percent"
]])

csv_top = top_tweets.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Top Tweets CSV",
    data=csv_top,
    file_name="twitter_top_tweets.csv",
    mime="text/csv"
)

fig_top = px.bar(
    top_tweets,
    x="short_text",
    y="Engagement_Rate_Percent",
    title="Top 20 Tweets by Engagement Rate",
)

fig_top.update_layout(
    xaxis_title="Tweet",
    yaxis_title="Engagement Rate (%)",
    xaxis_tickangle=-45
)

st.plotly_chart(fig_top, use_container_width=True)

st.divider()

# ───────────────── BOTTOM (WORST) TWEETS ─────────────────
bottom_tweets = (
    df_filtered
    .sort_values(by="Engagement_Rate_Percent", ascending=True)
    .head(20)
)

st.subheader("📉 Bottom 20 Tweets by Engagement Rate")

st.dataframe(bottom_tweets[[
    "created_at",
    "clean_text",
    "public_metrics_impression_count",
    "Total_Engagement",
    "Engagement_Rate_Percent"
]])

csv_bottom = bottom_tweets.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Bottom Tweets CSV",
    data=csv_bottom,
    file_name="twitter_bottom_tweets.csv",
    mime="text/csv"
)

fig_bottom = px.bar(
    bottom_tweets,
    x="short_text",
    y="Engagement_Rate_Percent",
    title="Bottom 20 Tweets by Engagement Rate",
)

fig_bottom.update_layout(
    xaxis_title="Tweet",
    yaxis_title="Engagement Rate (%)",
    xaxis_tickangle=-45
)

st.plotly_chart(fig_bottom, use_container_width=True)

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import re
# import os
# from dotenv import load_dotenv

# # ───────────────── Streamlit Setup ─────────────────
# st.set_page_config(
#     page_title="Twitter Top Tweets Dashboard",
#     page_icon="🐦",
#     layout="wide"
# )
# st.title("🐦 Twitter Top Tweets Dashboard")

# # ───────────────── Load Environment & Data ─────────────────
# load_dotenv()
# TWITTER_DATA_PATH = os.getenv("TWITTER_DATA_PATH")

# try:
#     df = pd.read_csv(TWITTER_DATA_PATH)
# except Exception:
#     st.error("🚫 Twitter data file not found! Check TWITTER_TWEETS_DATA in .env")
#     st.stop()

# # ───────────────── Preprocessing ─────────────────
# df["text"] = df["text"].astype(str)
# df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

# # Remove Emojis
# def remove_emojis(text):
#     return re.sub("[\U00010000-\U0010ffff]", "", text)

# df["clean_text"] = df["text"].apply(remove_emojis)

# # Short text for plots
# df["short_text"] = df["clean_text"].apply(
#     lambda x: x[:60] + "..." if len(x) > 60 else x
# )

# # Engagement Calculation
# df["Total_Engagement"] = (
#     df["public_metrics_like_count"] +
#     df["public_metrics_retweet_count"] +
#     df["public_metrics_reply_count"] +
#     df["public_metrics_quote_count"] +
#     df["public_metrics_bookmark_count"]
# )

# df["Engagement_Rate_Percent"] = (
#     (df["Total_Engagement"] /
#      df["public_metrics_impression_count"].replace(0, pd.NA)) * 100
# ).fillna(0).round(2)

# # ───────────────── Sidebar Filters ─────────────────
# st.sidebar.header("📌 Filters")

# # Ensure datetime is clean
# df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce").dt.tz_localize(None)

# # Default dates
# from_date_default = df["created_at"].min().date()
# to_date_default = df["created_at"].max().date()

# # Two-column date inputs
# col1, col2 = st.sidebar.columns(2)

# from_date = col1.date_input(
#     "From Date",
#     value=from_date_default,
#     key="from_date_input"
# )

# to_date = col2.date_input(
#     "To Date",
#     value=to_date_default,
#     key="to_date_input"
# )

# # Safety check (optional but recommended)
# if from_date > to_date:
#     st.sidebar.error("🚫 From Date cannot be after To Date")

# # Apply filter
# df_filtered = df[
#     (df["created_at"].dt.date >= from_date) &
#     (df["created_at"].dt.date <= to_date)
# ]

# # ───────────────── Top Tweets ─────────────────
# top_tweets = df_filtered.sort_values(
#     by="Engagement_Rate_Percent",
#     ascending=False
# ).head(20)

# # ───────────────── Display Table ─────────────────
# st.subheader("🔥 Top 20 Tweets by Engagement Rate")

# st.dataframe(top_tweets[[
#     "created_at",
#     "clean_text",
#     "public_metrics_impression_count",
#     "Total_Engagement",
#     "Engagement_Rate_Percent"
# ]])

# # ───────────────── Bar Chart ─────────────────
# fig = px.bar(
#     top_tweets,
#     x="short_text",
#     y="Engagement_Rate_Percent",
#     title="Top 20 Tweets by Engagement Rate",
# )

# fig.update_layout(
#     xaxis_title="Tweet",
#     yaxis_title="Engagement Rate (%)",
#     xaxis_tickangle=-45
# )

# st.plotly_chart(fig, use_container_width=True)

# # ───────────────── Download CSV ─────────────────
# csv = top_tweets.to_csv(index=False).encode("utf-8")

# st.download_button(
#     label="📥 Download Top Tweets CSV",
#     data=csv,
#     file_name="twitter_top_tweets.csv",
#     mime="text/csv"
# )
