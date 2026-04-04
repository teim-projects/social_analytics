import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
import os

# ───────────────── Page Setup ─────────────────
st.set_page_config(
    page_title="Google Ads Campaign Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Google Ads Campaign Performance EDA Dashboard")

# ───────────────── Load & Clean Data ─────────────────
@st.cache_data
def load_campaign_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_lines = [
        line for line in lines
        if not (
            "Campaign report" in line
            or "All time" in line
            or line.startswith("Total:")
        )
    ]

    df = pd.read_csv(StringIO("".join(data_lines)))

    df.columns = [
        "Campaign_Status", "Campaign", "Budget", "Budget_Name",
        "Budget_Type", "Status", "Status_Reasons",
        "Optimization_Score", "Campaign_Type",
        "Impressions", "Clicks", "Currency",
        "Avg_CPC", "Phone_Calls", "Phone_Impressions",
        "Cost", "Value_per_Conv", "All_Conversions",
        "Bid_Strategy", "Conversion_Rate",
        "Conversions", "Cost_per_Conversion"
    ]

    def clean_numeric(series):
        return (
            series.astype(str)
            .str.strip()
            .replace(["--", "—", "", "nan"], np.nan)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

    numeric_cols = [
        "Budget", "Optimization_Score",
        "Impressions", "Clicks", "Avg_CPC",
        "Phone_Calls", "Phone_Impressions",
        "Cost", "Value_per_Conv",
        "All_Conversions", "Conversion_Rate",
        "Conversions", "Cost_per_Conversion"
    ]

    for col in numeric_cols:
        df[col] = clean_numeric(df[col])

    df["CTR"] = (df["Clicks"] / df["Impressions"]) * 100
    return df

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Campaign_report.csv")
df = load_campaign_data(DATA_PATH)

# ───────────────── Sidebar Filters ─────────────────

with st.sidebar:
    c1, c2 = st.sidebar.columns([1, 1])

    if "campaign_filter" not in st.session_state:
        st.session_state.campaign_filter = df["Campaign"].unique().tolist()

    if "type_filter" not in st.session_state:
        st.session_state.type_filter = df["Campaign_Type"].unique().tolist()

    with c1:
        st.header("🎯 Filters")

    with c2:
        if st.button("❌ Clear Filters"):
            st.session_state.campaign_filter = df["Campaign"].unique().tolist()
            st.session_state.type_filter = df["Campaign_Type"].unique().tolist()
            st.rerun()

    selected_campaigns = st.multiselect(
        "Campaign",
        options=df["Campaign"].unique(),
        default=st.session_state.campaign_filter
    )

    selected_types = st.multiselect(
        "Campaign Type",
        options=df["Campaign_Type"].unique(),
        default=st.session_state.type_filter
    )

    st.session_state.campaign_filter = selected_campaigns
    st.session_state.type_filter = selected_types

    df_f = df[
        (df["Campaign"].isin(selected_campaigns)) &
        (df["Campaign_Type"].isin(selected_types))
    ]

# ───────────────── 1️⃣ Top Campaigns by Impressions ─────────────────
top_impr = df_f.sort_values("Impressions", ascending=False).head(10)

left, right = st.columns([3, 2])

with left:
    st.subheader("1️⃣ Top Campaigns by Impressions")
    fig = px.bar(
        top_impr,
        x="Impressions",
        y="Campaign",
        orientation="h",
        title="Top 10 Campaigns by Reach"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    top = top_impr.iloc[0]
    st.markdown("### 📌 Summary")
    st.write(
        f"**{top['Campaign']}** achieved the highest reach with "
        f"{int(top['Impressions']):,} impressions."
    )
    st.markdown("### ✅ Recommended Action")
    st.write("Leverage high-reach campaigns for awareness and remarketing strategies.")

# ───────────────── 2️⃣ CTR Distribution ─────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("2️⃣ Click-Through Rate Distribution")
    fig = px.histogram(
        df_f,
        x="CTR",
        nbins=20,
        title="CTR Distribution Across Campaigns"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    min_ctr = df_f.loc[df_f["CTR"].idxmin()]
    max_ctr = df_f.loc[df_f["CTR"].idxmax()]

    st.markdown("### 📌 Summary")
    st.write(
        f"CTR ranges from **{min_ctr['CTR']:.2f}%** "
        f"({min_ctr['Campaign']}) to **{max_ctr['CTR']:.2f}%** "
        f"({max_ctr['Campaign']})."
    )
    st.markdown("### ✅ Recommended Action")
    st.write("Optimize creatives and keywords for low-CTR campaigns.")

# ───────────────── 3️⃣ Cost vs Conversions ─────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("3️⃣ Cost vs Conversions")
    fig = px.scatter(
        df_f,
        x="Cost",
        y="Conversions",
        color="Campaign_Type",
        size="Impressions",
        title="Spend vs Conversion Efficiency"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    best_roi = (
        df_f[df_f["Conversions"] > 0]
        .sort_values("Cost_per_Conversion")
        .iloc[0]
    )

    st.markdown("### 📌 Summary")
    st.write(
        f"**{best_roi['Campaign']}** shows the best efficiency "
        f"with cost per conversion of ₹{best_roi['Cost_per_Conversion']:.2f}."
    )
    st.markdown("### ✅ Recommended Action")
    st.write("Scale efficient campaigns while controlling high-cost outliers.")

# ───────────────── 4️⃣ Conversions by Campaign Type ─────────────────
type_summary = (
    df_f.groupby("Campaign_Type", as_index=False)
    .agg(
        Impressions=("Impressions", "sum"),
        Clicks=("Clicks", "sum"),
        Conversions=("Conversions", "sum"),
        Cost=("Cost", "sum")
    )
)

left, right = st.columns([3, 2])

with left:
    st.subheader("4️⃣ Performance by Campaign Type")
    fig = px.bar(
        type_summary,
        x="Campaign_Type",
        y="Conversions",
        title="Total Conversions by Campaign Type"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    top_type = type_summary.loc[type_summary["Conversions"].idxmax()]
    st.markdown("### 📌 Summary")
    st.write(
        f"**{top_type['Campaign_Type']}** campaigns deliver the highest "
        f"number of conversions."
    )
    st.markdown("### ✅ Recommended Action")
    st.write("Prioritize top-performing campaign types in budget allocation.")

# ───────────────── 5️⃣ High Spend, Low Conversion Campaigns ─────────────────
inefficient = df_f[
    (df_f["Cost"] > df_f["Cost"].median()) &
    (df_f["Conversions"] < df_f["Conversions"].median())
]

left, right = st.columns([3, 2])

with left:
    st.subheader("5️⃣ High Spend but Low Conversion Campaigns")
    fig = px.bar(
        inefficient,
        x="Cost",
        y="Campaign",
        orientation="h",
        title="Inefficient Campaign Spend"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### 📌 Summary")
    st.write(
        f"{inefficient.shape[0]} campaigns incur high spend with "
        f"below-average conversions."
    )
    st.markdown("### ✅ Recommended Action")
    st.write(
        "Review bidding, targeting, and creatives for these campaigns "
        "to reduce wasted spend."
    )
