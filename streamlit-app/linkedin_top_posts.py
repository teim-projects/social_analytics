import streamlit as st
import pandas as pd
import plotly.express as px
import re
import os
from dotenv import load_dotenv

# ───────────────── Streamlit Setup ─────────────────
st.set_page_config(
    page_title="LinkedIn Top & Bottom Posts Dashboard",
    page_icon="💼",
    layout="wide"
)
st.title("💼 LinkedIn Top & Bottom Posts Dashboard")

# ───────────────── Load Environment & Data ─────────────────
load_dotenv()
LINKEDIN_DATA_PATH = os.getenv("LINKEDIN_POSTS_DATA")

try:
    df = pd.read_csv(LINKEDIN_DATA_PATH)
except Exception:
    st.error("🚫 LinkedIn data file not found! Check LINKEDIN_POSTS_DATA in .env")
    st.stop()

# ───────────────── Preprocessing ─────────────────
df["text"] = df["text"].astype(str)

# Parse date safely
df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)

# Remove emojis
def remove_emojis(text):
    return re.sub(r"[\U00010000-\U0010ffff]", "", text)

df["clean_text"] = df["text"].apply(remove_emojis)

# Short text for plotting
df["short_text"] = df["clean_text"].apply(
    lambda x: x[:80] + "..." if len(x) > 80 else x
)

# Engagement calculation
df["Total_Engagement"] = (
    df["likes"].fillna(0) +
    df["comments_count"].fillna(0) +
    df["shares"].fillna(0)
)

# ───────────────── TOP POSTS ─────────────────
top_posts = (
    df
    .sort_values(by="Total_Engagement", ascending=False)
    .head(20)
)

st.subheader("🔥 Top 20 LinkedIn Posts by Engagement")

st.dataframe(top_posts[[
    "date",
    "clean_text",
    "likes",
    "comments_count",
    "shares",
    "Total_Engagement"
]])

csv_top = top_posts.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Top LinkedIn Posts CSV",
    data=csv_top,
    file_name="linkedin_top_posts.csv",
    mime="text/csv"
)

fig_top = px.bar(
    top_posts,
    x="short_text",
    y="Total_Engagement",
    title="Top 20 LinkedIn Posts by Engagement"
)

fig_top.update_layout(
    xaxis_title="Post",
    yaxis_title="Total Engagement",
    xaxis_tickangle=-45
)

st.plotly_chart(fig_top, use_container_width=True)

st.divider()

# ───────────────── BOTTOM (WORST) POSTS ─────────────────
bottom_posts = (
    df
    .sort_values(by="Total_Engagement", ascending=True)
    .head(20)
)

st.subheader("📉 Bottom 20 LinkedIn Posts by Engagement")

st.dataframe(bottom_posts[[
    "date",
    "clean_text",
    "likes",
    "comments_count",
    "shares",
    "Total_Engagement"
]])

csv_bottom = bottom_posts.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Bottom LinkedIn Posts CSV",
    data=csv_bottom,
    file_name="linkedin_bottom_posts.csv",
    mime="text/csv"
)

fig_bottom = px.bar(
    bottom_posts,
    x="short_text",
    y="Total_Engagement",
    title="Bottom 20 LinkedIn Posts by Engagement"
)

fig_bottom.update_layout(
    xaxis_title="Post",
    yaxis_title="Total Engagement",
    xaxis_tickangle=-45
)

st.plotly_chart(fig_bottom, use_container_width=True)

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import re
# import os
# from dotenv import load_dotenv
# from datetime import date

# # ───────────────── Streamlit Setup ─────────────────
# st.set_page_config(
#     page_title="LinkedIn Top Posts Dashboard",
#     page_icon="💼",
#     layout="wide"
# )
# st.title("💼 LinkedIn Top Posts Dashboard")

# # ───────────────── Load Environment & Data ─────────────────
# load_dotenv()
# LINKEDIN_DATA_PATH = os.getenv("LINKEDIN_POSTS_DATA")

# try:
#     df = pd.read_csv(LINKEDIN_DATA_PATH)
# except Exception:
#     st.error("🚫 LinkedIn data file not found! Check LINKEDIN_POSTS_DATA in .env")
#     st.stop()

# # ───────────────── Preprocessing ─────────────────
# df["text"] = df["text"].astype(str)

# # Parse date safely
# df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)

# # Remove emoji
# def remove_emojis(text):
#     return re.sub(r"[\U00010000-\U0010ffff]", "", text)

# df["clean_text"] = df["text"].apply(remove_emojis)

# df["short_text"] = df["clean_text"].apply(
#     lambda x: x[:80] + "..." if len(x) > 80 else x
# )

# # Engagement calculation
# df["Total_Engagement"] = (
#     df["likes"].fillna(0) +
#     df["comments_count"].fillna(0) +
#     df["shares"].fillna(0)
# )

# # ───────────────── Top Posts ─────────────────
# top_posts = (
#     df
#     .sort_values(by="Total_Engagement", ascending=False)
#     .head(20)
# )

# # ───────────────── Display Table ─────────────────
# st.subheader("🔥 Top 20 LinkedIn Posts by Engagement")

# st.dataframe(top_posts[[
#     "date",
#     "clean_text",
#     "likes",
#     "comments_count",
#     "shares",
#     "Total_Engagement"
# ]])

# # ───────────────── Bar Chart ─────────────────
# fig = px.bar(
#     top_posts,
#     x="short_text",
#     y="Total_Engagement",
#     title="Top 20 LinkedIn Posts by Engagement"
# )

# fig.update_layout(
#     xaxis_title="Post",
#     yaxis_title="Total Engagement",
#     xaxis_tickangle=-45
# )

# st.plotly_chart(fig, use_container_width=True)

# # ───────────────── Download CSV ─────────────────
# csv = top_posts.to_csv(index=False).encode("utf-8")

# st.download_button(
#     label="📥 Download Top LinkedIn Posts CSV",
#     data=csv,
#     file_name="linkedin_top_posts.csv",
#     mime="text/csv"
# )
