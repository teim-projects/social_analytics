import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ────────────── PAGE CONFIG ──────────────
st.set_page_config(page_title="Campaign Dashboard", layout="wide")
st.title("📊 Campaign Performance Analytics")

# ────────────── LOAD DATA ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "Shree-Laxmi-Stone-Depot-Campaigns-1-Jan-2025-1-Jan-2026.csv")

df = pd.read_csv(DATA_PATH)

# CLEAN COLUMNS
df.columns = df.columns.str.strip()

# ────────────── NUMERIC CLEANING ──────────────
num_cols = [
    "Amount spent (INR)", "Impressions", "Reach", "Link clicks",
    "CTR (link click-through rate)",
    "CPC (cost per link click) (INR)",
    "CPM (cost per 1,000 impressions) (INR)",
    "Results", "Cost per results", "Landing page views", "Frequency"
]

for col in num_cols:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "")
            .str.replace("₹", "")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ────────────── FILTERS ──────────────
st.sidebar.header("📌 Filters")

campaigns = df["Campaign name"].dropna().unique()

if "campaign_filter" not in st.session_state:
    st.session_state.campaign_filter = list(campaigns)

if "spend_range" not in st.session_state:
    st.session_state.spend_range = (
        int(df["Amount spent (INR)"].min()),
        int(df["Amount spent (INR)"].max())
    )

def clear_filters():
    st.session_state.campaign_filter = list(campaigns)
    st.session_state.spend_range = (
        int(df["Amount spent (INR)"].min()),
        int(df["Amount spent (INR)"].max())
    )

st.sidebar.button("🧹 Clear Filters", on_click=clear_filters)

st.sidebar.multiselect("Campaign Name", campaigns, key="campaign_filter")

st.sidebar.slider(
    "Spend Range",
    int(df["Amount spent (INR)"].min()),
    int(df["Amount spent (INR)"].max()),
    key="spend_range"
)

# APPLY FILTER
df_f = df.copy()

df_f = df_f[df_f["Campaign name"].isin(st.session_state.campaign_filter)]

df_f = df_f[
    (df_f["Amount spent (INR)"] >= st.session_state.spend_range[0]) &
    (df_f["Amount spent (INR)"] <= st.session_state.spend_range[1])
]

if df_f.empty:
    st.warning("No data for selected filters")
    st.stop()

# LAYOUT FUNCTION
def plot_with_summary(fig, summary):
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(summary)

# ─────────────────────────────────────────────
# 1️⃣ ROI RANKING
# ─────────────────────────────────────────────
st.subheader("1️⃣ Campaign ROI Ranking")

df_f["ROI"] = df_f["Results"] / (df_f["Amount spent (INR)"] + 1)
roi_sorted = df_f.sort_values("ROI")

fig1 = px.bar(
    roi_sorted,
    x="ROI",
    y="Campaign name",
    orientation="h"
)

spread = roi_sorted["ROI"].max() - roi_sorted["ROI"].min()

summary1 = f"""
📊 **BUSINESS SUMMARY**

{"👉 Huge ROI gap → reallocate budget to top campaigns." if spread > 0.05 else
 "👉 Campaign performance relatively balanced."}
"""

plot_with_summary(fig1, summary1)

# ─────────────────────────────────────────────
# 2️⃣ ROI COMPARISON (SECOND GRAPH IN COLAB)
# ─────────────────────────────────────────────
st.subheader("2️⃣ Campaign ROI Comparison")

roi_sorted_desc = df_f.sort_values("ROI", ascending=False)

fig2 = px.bar(
    roi_sorted_desc,
    x="ROI",
    y="Campaign name",
    orientation="h",
    color="ROI"
)

spread = roi_sorted_desc["ROI"].max() - roi_sorted_desc["ROI"].min()

summary2 = f"""
📊 **BUSINESS SUMMARY**

{"👉 Huge ROI gap → scale top campaign and reduce budget on weak ones." if spread > 0.05 else
 "👉 ROI similar → diversification strategy ok."}
"""

plot_with_summary(fig2, summary2)

# ─────────────────────────────────────────────
# 3️⃣ COST PER RESULT
# ─────────────────────────────────────────────
st.subheader("3️⃣ Cost per Result by Campaign")

df_f["CPR"] = df_f["Amount spent (INR)"] / (df_f["Results"] + 1)
cpr_sorted = df_f.sort_values("CPR")

fig3 = px.bar(
    cpr_sorted,
    x="CPR",
    y="Campaign name",
    orientation="h"
)

avg_cpr = df_f["CPR"].mean()

summary3 = f"""
📊 **BUSINESS SUMMARY**

{"👉 Some campaigns highly inefficient → pause or optimise." if cpr_sorted["CPR"].iloc[-1] > avg_cpr*1.5 else
 "👉 Cost efficiency fairly stable."}
"""

plot_with_summary(fig3, summary3)

# ─────────────────────────────────────────────
# 4️⃣ 🚨 TRAFFIC GENERATION EFFICIENCY (YOUR GRAPH)
# ─────────────────────────────────────────────
st.subheader("4️⃣ Traffic Generation Efficiency")

df_f["Click Efficiency"] = df_f["Link clicks"] / (df_f["Amount spent (INR)"] + 1)

click_sorted = df_f.sort_values("Click Efficiency", ascending=False)

fig4 = px.bar(
    click_sorted,
    x="Click Efficiency",
    y="Campaign name",
    orientation="h",
    title="Traffic Generation Efficiency"
)

spread = click_sorted["Click Efficiency"].max() - click_sorted["Click Efficiency"].min()

summary4 = f"""
📊 **BUSINESS SUMMARY**

{"👉 Strong variation in traffic efficiency → budget reallocation possible." if spread > 0.02 else
 "👉 Campaigns generating traffic similarly."}
"""

plot_with_summary(fig4, summary4)

# ─────────────────────────────────────────────
# 5️⃣ EXPOSURE (FINAL GRAPH)
# ─────────────────────────────────────────────
st.subheader("5️⃣ Audience Exposure by Campaign")

df_f["Exposure"] = df_f["Impressions"] / (df_f["Reach"] + 1)

exp_sorted = df_f.sort_values("Exposure", ascending=False)

fig5 = px.bar(
    exp_sorted,
    x="Exposure",
    y="Campaign name",
    orientation="h"
)

max_exp = df_f["Exposure"].max()

summary5 = f"""
📊 **BUSINESS SUMMARY**

{"👉 Some campaigns over-exposing same audience → fatigue risk." if max_exp > 2 else
 "👉 Audience expansion still possible."}
"""

plot_with_summary(fig5, summary5)