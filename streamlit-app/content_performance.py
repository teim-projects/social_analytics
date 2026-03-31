import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ────────────── PAGE CONFIG ──────────────
st.set_page_config(page_title="Content Performance Dashboard", layout="wide")
st.title("📊 Content Performance Analytics Dashboard")

# ────────────── LOAD DATA ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Jan-01-2025_Jan-01-2026_1444967610559863.csv")
df_posts = pd.read_csv(DATA_PATH)

# ────────────── PREPROCESSING ──────────────
df_posts["Publish time"] = pd.to_datetime(df_posts["Publish time"], errors="coerce")
df_posts = df_posts.dropna(subset=["Publish time"])

df_posts["Views"] = pd.to_numeric(df_posts["Views"], errors="coerce").fillna(0)
df_posts["Duration (sec)"] = pd.to_numeric(df_posts["Duration (sec)"], errors="coerce").fillna(0)
df_posts["Views from Organic posts"] = pd.to_numeric(df_posts["Views from Organic posts"], errors="coerce").fillna(0)
df_posts["Views from Boosted posts"] = pd.to_numeric(df_posts["Views from Boosted posts"], errors="coerce").fillna(0)
df_posts["Ad CPM (USD)"] = pd.to_numeric(df_posts["Ad CPM (USD)"], errors="coerce").fillna(0)

df_posts["Hour"] = df_posts["Publish time"].dt.hour

# ────────────── SIDEBAR FILTERS ──────────────
st.sidebar.header("📌 Filters")

min_date = df_posts["Publish time"].min().date()
max_date = df_posts["Publish time"].max().date()

# DEFAULTS
defaults = {
    "from_date": min_date,
    "to_date": max_date,
    "view_range": (int(df_posts["Views"].min()), int(df_posts["Views"].max()))
}

# INIT SESSION STATE
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# CLEAR FILTERS FUNCTION
def clear_filters():
    for key, val in defaults.items():
        st.session_state[key] = val

st.sidebar.button("🧹 Clear Filters", on_click=clear_filters)

# DATE FILTERS (FIXED WITH KEY)
col1, col2 = st.sidebar.columns(2)

col1.date_input("From", key="from_date")
col2.date_input("To", key="to_date")

# VIEW RANGE (FIXED WITH KEY)
st.sidebar.slider(
    "Views Range",
    int(df_posts["Views"].min()),
    int(df_posts["Views"].max()),
    key="view_range"
)

# ────────────── APPLY FILTERS ──────────────
df_filtered = df_posts[
    (df_posts["Publish time"].dt.date >= st.session_state.from_date) &
    (df_posts["Publish time"].dt.date <= st.session_state.to_date) &
    (df_posts["Views"] >= st.session_state.view_range[0]) &
    (df_posts["Views"] <= st.session_state.view_range[1])
]

if df_filtered.empty:
    st.warning("⚠️ No data for selected filters")
    st.stop()

# ────────────── HELPER ──────────────
def plot_with_summary(fig, summary):
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(summary)

# =========================================================
# 1️⃣ MONTHLY TREND
# =========================================================
st.subheader("1️⃣ Monthly Views Trend")

monthly_views = df_filtered.groupby(
    df_filtered["Publish time"].dt.to_period("M")
)["Views"].sum()

fig1 = px.line(
    x=monthly_views.index.astype(str),
    y=monthly_views.values,
    markers=True,
    labels={"x": "Month", "y": "Total Views"}
)

growth = monthly_views.pct_change().mean() * 100
best_month = monthly_views.idxmax()
worst_month = monthly_views.idxmin()

summary1 = f"""
📊 **BUSINESS SUMMARY**

Average monthly growth rate: {growth:.2f}%  
Best performing month: {best_month}  
Lowest performing month: {worst_month}  

{'✅ Strong upward trend → scale ad spend and content volume.' if growth > 5 else
 '⚠ Declining trend → content fatigue or targeting issue.' if growth < -5 else
 '➡ Stable performance → experiment with new creatives.'}
"""

plot_with_summary(fig1, summary1)

# =========================================================
# 2️⃣ DURATION VS VIEWS
# =========================================================
st.subheader("2️⃣ Duration vs Views")

fig2 = px.scatter(
    df_filtered,
    x="Duration (sec)",
    y="Views",
    labels={
        "Duration (sec)": "Video Duration (seconds)",
        "Views": "Total Views"
    }
)

corr = df_filtered["Duration (sec)"].corr(df_filtered["Views"])

if corr > 0.3:
    text = f"Positive correlation ({corr:.2f}) → longer videos driving more reach.\n👉 Strategy: Invest in storytelling and retention optimization."
elif corr < -0.3:
    text = f"Negative correlation ({corr:.2f}) → shorter videos performing better.\n👉 Strategy: Focus on reels/short-form creatives."
else:
    text = f"Weak correlation ({corr:.2f}) → duration not primary driver.\n👉 Strategy: Improve hook, audio trend, thumbnail."

summary2 = f"📊 **BUSINESS SUMMARY**\n\n{text}"

plot_with_summary(fig2, summary2)

# =========================================================
# 3️⃣ ORGANIC EFFICIENCY
# =========================================================
st.subheader("3️⃣ Organic Efficiency")

df_filtered["Organic Efficiency"] = df_filtered["Views from Organic posts"] / (df_filtered["Views"] + 1)

fig3 = px.histogram(
    df_filtered,
    x="Organic Efficiency",
    labels={"Organic Efficiency": "Organic Efficiency Ratio"}
)

avg_org = df_filtered["Organic Efficiency"].mean()

if avg_org > 0.6:
    text = f"High organic efficiency ({avg_org:.2f}) → strong content-market fit.\n👉 Reduce paid dependency and scale viral formats."
elif avg_org < 0.3:
    text = f"Low organic efficiency ({avg_org:.2f}) → heavy reliance on boosting.\n👉 Improve creative quality and trend alignment."
else:
    text = f"Moderate organic efficiency ({avg_org:.2f}).\n👉 Balanced strategy recommended."

summary3 = f"📊 **BUSINESS SUMMARY**\n\n{text}"

plot_with_summary(fig3, summary3)

# =========================================================
# 4️⃣ POSTING HOUR
# =========================================================
st.subheader("4️⃣ Posting Hour Performance")

hour_perf = df_filtered.groupby("Hour")["Views"].mean()

fig4 = px.line(
    x=hour_perf.index,
    y=hour_perf.values,
    markers=True,
    labels={"x": "Hour of Day", "y": "Average Views"}
)

best_hour = hour_perf.idxmax()
worst_hour = hour_perf.idxmin()

spread = hour_perf.max() - hour_perf.min()

if spread > hour_perf.mean()*0.5:
    text = "👉 Posting time significantly affects reach. Scheduling optimisation recommended."
else:
    text = "👉 Posting time has limited impact. Focus more on content quality."

summary4 = f"""
📊 **BUSINESS SUMMARY**

Best posting hour: {best_hour}  
Worst posting hour: {worst_hour}  

{text}
"""

plot_with_summary(fig4, summary4)

# =========================================================
# 5️⃣ DURATION BAND
# =========================================================
st.subheader("5️⃣ Duration Band Performance")

bins = [0,15,30,60,120,300,1000]
labels = ["0-15","15-30","30-60","60-120","120-300","300+"]

df_filtered["Duration Band"] = pd.cut(df_filtered["Duration (sec)"], bins=bins, labels=labels)

dur_perf = df_filtered.groupby("Duration Band")["Views"].mean()

fig5 = px.bar(
    x=dur_perf.index.astype(str),
    y=dur_perf.values,
    labels={"x": "Duration Band (seconds)", "y": "Average Views"}
)

best_band = dur_perf.idxmax()

summary5 = f"""
📊 **BUSINESS SUMMARY**

Best performing duration band: {best_band}  

👉 Replicate content formats in this duration range.
"""

plot_with_summary(fig5, summary5)

# =========================================================
# 6️⃣ VIRAL CONTENT
# =========================================================
st.subheader("6️⃣ Viral Content Analysis")

threshold = df_filtered["Views"].quantile(0.9)
df_filtered["Viral"] = df_filtered["Views"] >= threshold

viral_ratio = df_filtered["Viral"].mean()

fig6 = px.histogram(
    df_filtered,
    x="Views",
    labels={"Views": "Total Views"}
)

summary6 = f"""
📊 **BUSINESS SUMMARY**

Viral content ratio: {viral_ratio*100:.2f}%  

{'👉 Low viral rate. Need experimentation in hooks and trends.' if viral_ratio < 0.1 else
 '👉 Healthy viral discovery pipeline.'}
"""

plot_with_summary(fig6, summary6)

# =========================================================
# 7️⃣ ORGANIC VS PAID
# =========================================================
st.subheader("7️⃣ Organic vs Paid")

total_org = df_filtered["Views from Organic posts"].sum()
total_paid = df_filtered["Views from Boosted posts"].sum()

fig7 = px.pie(
    values=[total_org, total_paid],
    names=["Organic", "Paid"]
)

org_dependency = total_org / (total_org + total_paid + 1)

if org_dependency > 0.7:
    text = "👉 Strong organic engine. Paid ads can be used only for scaling."
elif org_dependency < 0.3:
    text = "👉 Heavy reliance on paid reach. Creative improvement needed."
else:
    text = "👉 Balanced organic-paid mix."

summary7 = f"""
📊 **BUSINESS SUMMARY**

Organic dependency score: {org_dependency:.2f}  

{text}
"""

plot_with_summary(fig7, summary7)