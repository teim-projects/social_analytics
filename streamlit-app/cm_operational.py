import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ────────────── PAGE CONFIG ──────────────
st.set_page_config(page_title="Full Analytics Dashboard", layout="wide")
st.title("📊 Content and Monetization Operational Analytics Dashboard")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")

POSTS_PATH = os.path.join(DATA_DIR, "Jan-01-2025_Jan-01-2026_1444967610559863.csv")
MONET_PATH = os.path.join(DATA_DIR, "Jan-01-2025_Jan-01-2026_946170394605045.csv")

df_posts = pd.read_csv(POSTS_PATH)
df_monet = pd.read_csv(MONET_PATH)

# =========================================================
# COMMON HELPER
# =========================================================
def plot_with_summary(fig, summary):
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(summary)

# =========================================================
# ================= CONTENT DASHBOARD ======================
# =========================================================

df_posts["Publish time"] = pd.to_datetime(df_posts["Publish time"], errors="coerce")
df_posts = df_posts.dropna(subset=["Publish time"])

df_posts["Views"] = pd.to_numeric(df_posts["Views"], errors="coerce").fillna(0)
df_posts["Duration (sec)"] = pd.to_numeric(df_posts["Duration (sec)"], errors="coerce").fillna(0)
df_posts["Views from Organic posts"] = pd.to_numeric(df_posts["Views from Organic posts"], errors="coerce").fillna(0)
df_posts["Views from Boosted posts"] = pd.to_numeric(df_posts["Views from Boosted posts"], errors="coerce").fillna(0)
df_posts["Ad CPM (USD)"] = pd.to_numeric(df_posts["Ad CPM (USD)"], errors="coerce").fillna(0)

df_posts["Hour"] = df_posts["Publish time"].dt.hour

# FILTERS
st.sidebar.header("📌 Content Filters")

min_date = df_posts["Publish time"].min().date()
max_date = df_posts["Publish time"].max().date()

defaults_posts = {
    "from_date_p": min_date,
    "to_date_p": max_date,
    "view_range": (int(df_posts["Views"].min()), int(df_posts["Views"].max()))
}

for key, val in defaults_posts.items():
    if key not in st.session_state:
        st.session_state[key] = val

def clear_posts():
    for key, val in defaults_posts.items():
        st.session_state[key] = val

st.sidebar.button("🧹 Clear Content Filters", on_click=clear_posts)

col1, col2 = st.sidebar.columns(2)
col1.date_input("From", key="from_date_p")
col2.date_input("To", key="to_date_p")

st.sidebar.slider("Views Range",
    int(df_posts["Views"].min()),
    int(df_posts["Views"].max()),
    key="view_range"
)

df_posts_filtered = df_posts[
    (df_posts["Publish time"].dt.date >= st.session_state.from_date_p) &
    (df_posts["Publish time"].dt.date <= st.session_state.to_date_p) &
    (df_posts["Views"] >= st.session_state.view_range[0]) &
    (df_posts["Views"] <= st.session_state.view_range[1])
]

if df_posts_filtered.empty:
    st.warning("⚠️ No data for Content filters")
    st.stop()

date_col = None
for col in df_monet.columns:
    if "date" in col.lower() or "time" in col.lower():
        date_col = col
        break

df_monet[date_col] = pd.to_datetime(df_monet[date_col], errors="coerce")
df_monet = df_monet.dropna(subset=[date_col])
df_monet["Date"] = df_monet[date_col]

num_cols = df_monet.select_dtypes(include=np.number).columns
df_monet[num_cols] = df_monet[num_cols].fillna(0)

df_monet_filtered = df_monet.copy()

# 1 Duration vs Views
st.subheader("1️⃣ Duration vs Views")

fig2 = px.scatter(df_posts_filtered, x="Duration (sec)", y="Views")

corr = df_posts_filtered["Duration (sec)"].corr(df_posts_filtered["Views"])

if corr > 0.3:
    text = f"Positive correlation ({corr:.2f}) → longer videos driving more reach.\n👉 Strategy: Invest in storytelling."
elif corr < -0.3:
    text = f"Negative correlation ({corr:.2f}) → shorter videos performing better.\n👉 Strategy: Focus on reels."
else:
    text = f"Weak correlation ({corr:.2f}) → duration not primary driver.\n👉 Improve hooks."

summary2 = f"📊 **BUSINESS SUMMARY**\n\n{text}"
plot_with_summary(fig2, summary2)

# 2 Posting Hour
st.subheader("2️⃣ Posting Hour Performance")

hour_perf = df_posts_filtered.groupby("Hour")["Views"].mean()

fig4 = px.line(x=hour_perf.index, y=hour_perf.values, markers=True)

best_hour = hour_perf.idxmax()
worst_hour = hour_perf.idxmin()

summary4 = f"""
📊 **BUSINESS SUMMARY**

Best posting hour: {best_hour}  
Worst posting hour: {worst_hour}  

👉 Optimize scheduling accordingly.
"""
plot_with_summary(fig4, summary4)

# 3 Duration Band
st.subheader("3️⃣ Duration Band Performance")

bins = [0,15,30,60,120,300,1000]
labels = ["0-15","15-30","30-60","60-120","120-300","300+"]

df_posts_filtered["Duration Band"] = pd.cut(df_posts_filtered["Duration (sec)"], bins=bins, labels=labels)

dur_perf = df_posts_filtered.groupby("Duration Band")["Views"].mean()

fig5 = px.bar(x=dur_perf.index.astype(str), y=dur_perf.values)

best_band = dur_perf.idxmax()

summary5 = f"""
📊 **BUSINESS SUMMARY**

Best performing duration band: {best_band}  

👉 Replicate this format.
"""
plot_with_summary(fig5, summary5)

# 4 Day of Week
st.subheader("4️⃣ Engagement by Day")

df_monet_filtered["Day"] = df_monet_filtered["Date"].dt.day_name()

dow_perf = df_monet_filtered.groupby("Day")["Unique user engagements"].mean()

fig12 = px.bar(x=dow_perf.index, y=dow_perf.values)

summary12 = f"""
📊 **BUSINESS SUMMARY**

Best day: {dow_perf.idxmax()}
"""
plot_with_summary(fig12, summary12)

