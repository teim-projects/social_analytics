import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ────────────── PAGE CONFIG ──────────────
st.set_page_config(page_title="Monetization Dashboard", layout="wide")
st.title("💰 Monetization Analytics Dashboard")

# ────────────── LOAD DATA ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Jan-01-2025_Jan-01-2026_946170394605045.csv")

df_monet = pd.read_csv(DATA_PATH)

# ────────────── AUTO DETECT DATE COLUMN ──────────────
date_col = None
for col in df_monet.columns:
    if "date" in col.lower() or "time" in col.lower():
        date_col = col
        break

df_monet[date_col] = pd.to_datetime(df_monet[date_col], errors="coerce")
df_monet = df_monet.dropna(subset=[date_col])

# standardize column name
df_monet["Date"] = df_monet[date_col]

# ────────────── CLEAN NUMERIC ──────────────
num_cols = df_monet.select_dtypes(include=np.number).columns
df_monet[num_cols] = df_monet[num_cols].fillna(0)

# ────────────── SIDEBAR FILTERS ──────────────
st.sidebar.header("📌 Filters")

min_date = df_monet["Date"].min().date()
max_date = df_monet["Date"].max().date()

defaults = {
    "from_date_m": min_date,
    "to_date_m": max_date,
    "eng_range": (
        int(df_monet["Unique user engagements"].min()),
        int(df_monet["Unique user engagements"].max())
    )
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def clear_filters():
    for key, val in defaults.items():
        st.session_state[key] = val

st.sidebar.button("🧹 Clear Filters", on_click=clear_filters)

# DATE FILTER
col1, col2 = st.sidebar.columns(2)
col1.date_input("From", key="from_date_m")
col2.date_input("To", key="to_date_m")

# ENGAGEMENT RANGE
st.sidebar.slider(
    "Engagement Range",
    int(df_monet["Unique user engagements"].min()),
    int(df_monet["Unique user engagements"].max()),
    key="eng_range"
)

# ────────────── APPLY FILTERS ──────────────
df_filtered = df_monet[
    (df_monet["Date"].dt.date >= st.session_state.from_date_m) &
    (df_monet["Date"].dt.date <= st.session_state.to_date_m) &
    (df_monet["Unique user engagements"] >= st.session_state.eng_range[0]) &
    (df_monet["Unique user engagements"] <= st.session_state.eng_range[1])
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
# 1️⃣ MONTHLY ENGAGEMENT TREND
# =========================================================
st.subheader("1️⃣ Monthly Engagement Trend")

monthly_eng = df_filtered.groupby(
    df_filtered["Date"].dt.to_period("M")
)["Unique user engagements"].sum()

fig1 = px.line(
    x=monthly_eng.index.astype(str),
    y=monthly_eng.values,
    markers=True,
    labels={"x": "Month", "y": "Total Engagement"}
)

growth = monthly_eng.pct_change().mean()*100

if growth > 5:
    text = f"Engagement growing at {growth:.2f}% → audience expansion."
elif growth < -5:
    text = f"Engagement declining at {growth:.2f}% → content relevance risk."
else:
    text = "Stable engagement trend."

summary1 = f"""
📊 **BUSINESS SUMMARY**

{text}
"""

plot_with_summary(fig1, summary1)

# =========================================================
# 2️⃣ ENGAGEMENT VARIABILITY
# =========================================================
st.subheader("2️⃣ Engagement Variability")

fig2 = px.histogram(
    df_filtered,
    x="Unique user engagements",
    labels={"Unique user engagements": "Engagement per Post"}
)

std = monthly_eng.std()
mean = monthly_eng.mean()
cv = std/(mean+1)

if cv > 1:
    text = "👉 Highly inconsistent audience response."
elif cv < 0.3:
    text = "👉 Stable but plateaued growth."
else:
    text = "👉 Healthy performance variability."

summary2 = f"""
📊 **BUSINESS SUMMARY**

Engagement variability score: {cv:.2f}  

{text}
"""

plot_with_summary(fig2, summary2)

# =========================================================
# 3️⃣ BEST MONTH
# =========================================================
st.subheader("3️⃣ Best Performing Month")

fig3 = px.bar(
    x=monthly_eng.index.astype(str),
    y=monthly_eng.values,
    labels={"x": "Month", "y": "Total Engagement"}
)

best_month = monthly_eng.idxmax()

summary3 = f"""
📊 **BUSINESS SUMMARY**

Highest engagement month: {best_month}  

👉 Analyse content strategy used during this period.
"""

plot_with_summary(fig3, summary3)

# =========================================================
# 4️⃣ MOMENTUM ANALYSIS
# =========================================================
st.subheader("4️⃣ Engagement Momentum")

momentum = monthly_eng.diff().mean()

fig4 = px.line(
    x=monthly_eng.index.astype(str),
    y=monthly_eng.diff().fillna(0),
    markers=True,
    labels={"x": "Month", "y": "Engagement Change"}
)

if momentum > 0:
    text = "👉 Positive engagement momentum → scale posting frequency."
else:
    text = "👉 Negative momentum → need creative refresh."

summary4 = f"""
📊 **BUSINESS SUMMARY**

{text}
"""

plot_with_summary(fig4, summary4)

# =========================================================
# 5️⃣ DAY OF WEEK PERFORMANCE
# =========================================================
st.subheader("5️⃣ Engagement by Day of Week")

df_filtered["Day"] = df_filtered["Date"].dt.day_name()

dow_perf = df_filtered.groupby("Day")["Unique user engagements"].mean()

dow_perf = dow_perf.reindex([
    "Monday","Tuesday","Wednesday","Thursday",
    "Friday","Saturday","Sunday"
])

fig5 = px.bar(
    x=dow_perf.index,
    y=dow_perf.values,
    labels={"x": "Day of Week", "y": "Average Engagement"}
)

best_day = dow_perf.idxmax()

summary5 = f"""
📊 **BUSINESS SUMMARY**

Best engagement day: {best_day}  

👉 Prioritize posting high-quality ads on this day.
"""

plot_with_summary(fig5, summary5)