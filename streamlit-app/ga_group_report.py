# =====================================
# GOOGLE ADS AD GROUP PERFORMANCE DASHBOARD
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from dotenv import load_dotenv

# =====================================
# 1. PAGE SETUP
# =====================================
st.set_page_config(
    page_title="Google Ads Ad Group Dashboard",
    layout="wide"
)

st.title("📊 Google Ads Ad Group Performance Dashboard")

load_dotenv()
DATA_PATH = os.getenv("GA_GROUP_REPORT")  # set in .env

# =====================================
# 2. LOAD & CLEAN DATA
# =====================================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, engine="python", skiprows=2)

    # Remove summary rows
    df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")]
    df.reset_index(drop=True, inplace=True)

    numeric_cols = ["Clicks", "Impr.", "CTR", "Conv. rate", "Conversions"]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop duplicate currency column if exists
    if "Currency code.1" in df.columns:
        df = df.drop(columns=["Currency code.1"])

    # Cost per conversion
    df["Cost per Conversion"] = df["Cost"] / df["Conversions"]
    df["CTR (%)"] = df["CTR"]

    return df


df = load_data(DATA_PATH)

# =====================================
# 3. SIDEBAR FILTERS (SESSION SAFE)
# =====================================

campaigns = sorted(df["Campaign"].dropna().unique())
statuses = sorted(df["Ad group status"].dropna().unique())
adgroups = sorted(df["Ad group"].dropna().unique())

col1, col2 = st.sidebar.columns([1, 1])

with col1:
    st.markdown("### 📌 Filters")

with col2:
    if st.button("🧹 Clear All Filters", key="clear_filters"):
        st.session_state.campaign_filter = campaigns
        st.session_state.status_filter = statuses
        st.session_state.adgroup_filter = []

# --- SESSION INIT ---
if "campaign_filter" not in st.session_state:
    st.session_state.campaign_filter = campaigns

if "status_filter" not in st.session_state:
    st.session_state.status_filter = statuses

if "adgroup_filter" not in st.session_state:
    st.session_state.adgroup_filter = []

# --- FILTER WIDGETS ---
selected_campaigns = st.sidebar.multiselect(
    "Campaign",
    campaigns,
    default=st.session_state.campaign_filter
)

selected_status = st.sidebar.multiselect(
    "Ad Group Status",
    statuses,
    default=st.session_state.status_filter
)

selected_adgroups = st.sidebar.multiselect(
    "Ad Group",
    adgroups,
    default=st.session_state.adgroup_filter
)

# Persist state
st.session_state.campaign_filter = selected_campaigns
st.session_state.status_filter = selected_status
st.session_state.adgroup_filter = selected_adgroups

# =====================================
# 4. APPLY FILTERS
# =====================================
filtered_df = df[
    df["Campaign"].isin(st.session_state.campaign_filter) &
    df["Ad group status"].isin(st.session_state.status_filter)
]

if st.session_state.adgroup_filter:
    filtered_df = filtered_df[
        filtered_df["Ad group"].isin(st.session_state.adgroup_filter)
    ]

if filtered_df.empty:
    st.warning("⚠️ No data available for selected filter combination.")
    st.stop()

# =====================================
# 5. KPI SUMMARY
# =====================================
k1, k2, k3, k4 = st.columns(4)

k1.metric("Ad Groups", filtered_df["Ad group"].nunique())
k2.metric("Total Cost (₹)", int(filtered_df["Cost"].sum()))
k3.metric("Total Conversions", int(filtered_df["Conversions"].sum()))
k4.metric("Avg Cost / Conversion", round(filtered_df["Cost per Conversion"].mean(), 2))

# =====================================
# 6. GRAPHS + INSIGHTS
# =====================================
graph_no = 1

# 1️⃣ Cost per Conversion by Ad Group
ad_perf = filtered_df.sort_values("Cost per Conversion")

fig = px.bar(
    ad_perf,
    x="Ad group",
    y="Cost per Conversion",
    title=f"{graph_no}️⃣ Ad Group Cost Efficiency"
)

best = ad_perf.iloc[0]
worst = ad_perf.iloc[-1]

c1, c2 = st.columns([3, 2])
c1.plotly_chart(fig, use_container_width=True)
c2.markdown(
    f"""
**Insight**
- Best: **{best['Ad group']}** (₹{best['Cost per Conversion']:.2f})
- Worst: **{worst['Ad group']}** (₹{worst['Cost per Conversion']:.2f})
"""
)
graph_no += 1

# 2️⃣ Cost vs Conversions
fig = px.scatter(
    filtered_df,
    x="Cost",
    y="Conversions",
    title=f"{graph_no}️⃣ Spend vs Conversions by Ad Group"
)

inefficient = filtered_df[
    (filtered_df["Cost"] > filtered_df["Cost"].median()) &
    (filtered_df["Conversions"] < filtered_df["Conversions"].median())
]

c1, c2 = st.columns([3, 2])
c1.plotly_chart(fig, use_container_width=True)
c2.markdown(
    f"""
**Insight**
- {len(inefficient)} ad groups show **high spend but low conversions**
- Action: Review bids, keywords & creatives
"""
)
graph_no += 1

# 3️⃣ Cost vs CTR
fig = px.scatter(
    filtered_df,
    x="Cost",
    y="CTR (%)",
    title=f"{graph_no}️⃣ Cost vs Click-Through Rate"
)

high_quality = filtered_df[
    (filtered_df["CTR (%)"] >= filtered_df["CTR (%)"].median()) &
    (filtered_df["Cost"] <= filtered_df["Cost"].median())
]

c1, c2 = st.columns([3, 2])
c1.plotly_chart(fig, use_container_width=True)
c2.markdown(
    f"""
**Insight**
- {len(high_quality)} ad groups deliver **high CTR at lower cost**
- Use as benchmarks
"""
)
graph_no += 1

# 4️⃣ Cost vs Conversion Rate
fig = px.scatter(
    filtered_df,
    x="Cost",
    y="Conv. rate",
    title=f"{graph_no}️⃣ Cost vs Conversion Rate"
)

strong = filtered_df[
    filtered_df["Conv. rate"] >= filtered_df["Conv. rate"].median()
]

c1, c2 = st.columns([3, 2])
c1.plotly_chart(fig, use_container_width=True)
c2.markdown(
    f"""
**Insight**
- {len(strong)} ad groups show **above-average conversion rate**
- Allocate more qualified traffic
"""
)
graph_no += 1

# =====================================
# 7. RECOMMENDED ACTION TABLE
# =====================================
st.subheader("📌 Ad Group Action Recommendations")

avg_cpc = filtered_df["Cost per Conversion"].mean()

def recommend(cpc):
    if pd.isna(cpc):
        return "No conversions – review setup"
    elif cpc < avg_cpc * 0.7:
        return "Scale this ad group"
    elif cpc < avg_cpc * 1.3:
        return "Optimize & monitor"
    else:
        return "Limit or restructure"

filtered_df["Recommended Action"] = filtered_df["Cost per Conversion"].apply(recommend)

st.dataframe(
    filtered_df[
        ["Ad group", "Cost per Conversion", "Recommended Action"]
    ].sort_values("Cost per Conversion"),
    use_container_width=True
)
