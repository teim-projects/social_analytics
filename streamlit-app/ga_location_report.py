# =====================================
# GOOGLE ADS LOCATION REPORT DASHBOARD
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import os

# =====================================
# 1. STREAMLIT PAGE SETUP
# =====================================
st.set_page_config(page_title="Google Ads Location Report", layout="wide")
st.title("📍 Google Ads Location Performance Dashboard")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Location_report.csv")

# =====================================
# 2. LOAD & CLEAN DATA
# =====================================
@st.cache_data
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    meta, data = [], []
    for line in lines:
        if "Location report" in line or "All time" in line or line.startswith("Total:"):
            meta.append(line)
        else:
            data.append(line)

    df = pd.read_csv(StringIO("".join(data)))
    df.columns = [
        "location","campaign","impressions","interactions",
        "interaction_rate","currency","avg_cost","cost",
        "conversion_rate","conversions","cost_per_conversion"
    ]

    def clean(x):
        return (
            x.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .replace("nan", np.nan)
            .astype(float)
        )

    for c in [
        "impressions","interactions","interaction_rate",
        "avg_cost","cost","conversion_rate",
        "conversions","cost_per_conversion"
    ]:
        df[c] = clean(df[c])

    return df, meta


df, meta_lines = load_data(DATA_PATH)

# =====================================
# 3. SIDEBAR FILTERS (FIXED SESSION MGMT)
# =====================================

# =====================================
# SIDEBAR : FILTERS + CLEAR BUTTON
# =====================================

# =====================================
# PREPARE OPTIONS
# =====================================
all_locations = sorted(df["location"].dropna().unique().tolist())
all_campaigns = sorted(df["campaign"].dropna().unique().tolist())

with st.sidebar:
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.header("🔍 Filters")

    with col_f2:
        if st.button("❌ Clear Filters"):
            st.session_state.location_filter = []
            st.session_state.campaign_filter = all_campaigns
            st.session_state.select_all_locations_prev = False
            st.rerun()

# =====================================
# SESSION INIT
# =====================================
if "location_filter" not in st.session_state:
    st.session_state.location_filter = []

if "campaign_filter" not in st.session_state:
    st.session_state.campaign_filter = all_campaigns

if "select_all_locations_prev" not in st.session_state:
    st.session_state.select_all_locations_prev = False

# =====================================
# SELECT ALL LOCATIONS CHECKBOX
# =====================================
select_all_locations = st.sidebar.checkbox(
    "Select All Locations",
    key="select_all_locations"
)

# Checkbox turned OFF → clear
if st.session_state.select_all_locations_prev and not select_all_locations:
    st.session_state.location_filter = []

# Checkbox turned ON → select all
if select_all_locations:
    st.session_state.location_filter = all_locations

# =====================================
# LOCATION MULTISELECT (STATE-OWNED)
# =====================================
st.sidebar.multiselect(
    "Select Location(s)",
    options=all_locations,
    key="location_filter"
)

st.session_state.select_all_locations_prev = select_all_locations

# =====================================
# CAMPAIGN FILTER
# =====================================
st.sidebar.multiselect(
    "Select Campaign(s)",
    options=all_campaigns,
    key="campaign_filter"
)

# =====================================
# APPLY FILTERS
# =====================================
filtered_df = df[df["campaign"].isin(st.session_state.campaign_filter)]

if st.session_state.location_filter:
    filtered_df = filtered_df[
        filtered_df["location"].isin(st.session_state.location_filter)
    ]

# =====================================
# EMPTY STATE HANDLING
# =====================================
if not st.session_state.location_filter:
    st.warning("⚠️ Please select at least one Location to display the analysis.")
    st.stop()

if filtered_df.empty:
    st.warning("⚠️ No data available for the selected filter combination.")
    st.stop()

# =====================================
# 6. KPI SUMMARY
# =====================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Locations", filtered_df["location"].nunique())
k2.metric("Impressions", int(filtered_df["impressions"].sum()))
k3.metric("Interactions", int(filtered_df["interactions"].sum()))
k4.metric("Conversions", int(filtered_df["conversions"].sum()))

# =====================================
# 7. GRAPHS + INSIGHTS (NUMBERED)
# =====================================
graph_no = 1

# 1️⃣ Top Locations by Impressions
top = filtered_df.sort_values("impressions", ascending=False).head(10)
fig = px.bar(top, x="impressions", y="location", orientation="h",
             title=f"{graph_no}️⃣ Top Locations by Impressions")

c1, c2 = st.columns([3,2])
c1.plotly_chart(fig, use_container_width=True)
c2.markdown(
    f"**Highest reach:** {top.iloc[0]['location']}  \n"
    f"**Impressions:** {int(top.iloc[0]['impressions']):,}"
)
graph_no += 1

# 2️⃣ Interaction Rate Distribution
fig = px.histogram(filtered_df, x="interaction_rate", nbins=20,
                   title=f"{graph_no}️⃣ Interaction Rate Distribution")

min_ir = filtered_df.loc[filtered_df["interaction_rate"].idxmin()]
max_ir = filtered_df.loc[filtered_df["interaction_rate"].idxmax()]

c1, c2 = st.columns([3,2])
c1.plotly_chart(fig, use_container_width=True)
c2.markdown(
    f"**Lowest:** {min_ir['interaction_rate']:.2f}% ({min_ir['location']})  \n"
    f"**Highest:** {max_ir['interaction_rate']:.2f}% ({max_ir['location']})"
)
graph_no += 1

# 3️⃣ Cost vs Conversions
fig = px.scatter(filtered_df, x="cost", y="conversions",
                 title=f"{graph_no}️⃣ Cost vs Conversions")

best = (
    filtered_df[filtered_df["conversions"] > 0]
    .sort_values("cost_per_conversion")
    .iloc[0]
)

c1, c2 = st.columns([3,2])
c1.plotly_chart(fig, use_container_width=True)
c2.markdown(
    f"**Best ROI Location:** {best['location']}  \n"
    f"**Cost / Conversion:** ₹{best['cost_per_conversion']:.2f}"
)
graph_no += 1

# 4️⃣ Campaign Performance
camp = (
    filtered_df.groupby("campaign")
    .agg(
        impressions=("impressions","sum"),
        interactions=("interactions","sum"),
        conversions=("conversions","sum")
    )
    .reset_index()
)
camp["interaction_rate"] = camp["interactions"] / camp["impressions"] * 100

fig = px.bar(camp, x="conversions", y="campaign", orientation="h",
             title=f"{graph_no}️⃣ Conversions by Campaign")

top_c = camp.loc[camp["conversions"].idxmax()]

c1, c2 = st.columns([3,2])
c1.plotly_chart(fig, use_container_width=True)
c2.markdown(
    f"**Top Campaign:** {top_c['campaign']}  \n"
    f"**Conversions:** {int(top_c['conversions'])}  \n"
    f"**Interaction Rate:** {top_c['interaction_rate']:.2f}%"
)
graph_no += 1

# 5️⃣ Low Performance Locations
st.subheader(f"{graph_no}️⃣ Low-Performance Location Insights")

low = filtered_df[
    (filtered_df["impressions"] > 1000) &
    (filtered_df["interaction_rate"] < 2)
]

if not low.empty:
    st.info(
        "⚠️ Locations with high reach but poor engagement: "
        + ", ".join(low["location"].head(3))
    )
else:
    st.success("No locations show critical low engagement.")
