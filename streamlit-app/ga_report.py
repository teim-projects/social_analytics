import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Ad_report.csv")

# ─────────────────────────────
# Page Configuration
# ─────────────────────────────
st.set_page_config(
    page_title="Google Ads Ad Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Google Ads Ad Performance Dashboard")
st.caption("EDA, creative insights & automated recommendations")

# ─────────────────────────────
# Load & Clean Data
# ─────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        engine="python",
        skiprows=2
    )

    df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")]
    df.reset_index(drop=True, inplace=True)

    df["Ad strength"] = (
        df["Ad strength"]
        .astype(str)
        .str.strip()
        .replace({"--": "Not Evaluated", "—": "Not Evaluated", "nan": "Not Evaluated"})
    )

    numeric_cols = [
        "Impr.", "Interactions", "Interaction rate",
        "Conv. rate", "Conversions", "Cost", "Cost / conv."
    ]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "")
            .str.replace("%", "")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

ad_df = load_data()

# ─────────────────────────────
# Sidebar Filters
# ─────────────────────────────

# =====================================
# SIDEBAR : FILTERS + CLEAR BUTTON
# =====================================
with st.sidebar:
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.header("🔎 Filters")

    with col_f2:
        if st.button("❌ Clear Filters"):
            st.session_state.campaign_filter = []
            st.session_state.ad_group_filter = []
            st.session_state.ad_strength_filter = []
            st.rerun()


# =====================================
# SESSION STATE INIT
# =====================================
if "campaign_filter" not in st.session_state:
    st.session_state.campaign_filter = []

if "ad_group_filter" not in st.session_state:
    st.session_state.ad_group_filter = []

if "ad_strength_filter" not in st.session_state:
    st.session_state.ad_strength_filter = []


# =====================================
# SIDEBAR MULTISELECTS (STATE-OWNED)
# =====================================
st.sidebar.multiselect(
    "Campaign",
    sorted(ad_df["Campaign"].dropna().unique()),
    key="campaign_filter"
)

st.sidebar.multiselect(
    "Ad Group",
    sorted(ad_df["Ad group"].dropna().unique()),
    key="ad_group_filter"
)

st.sidebar.multiselect(
    "Ad Strength",
    sorted(ad_df["Ad strength"].dropna().unique()),
    key="ad_strength_filter"
)


# =====================================
# APPLY FILTERS
# =====================================
filtered_df = ad_df.copy()

if st.session_state.campaign_filter:
    filtered_df = filtered_df[
        filtered_df["Campaign"].isin(st.session_state.campaign_filter)
    ]

if st.session_state.ad_group_filter:
    filtered_df = filtered_df[
        filtered_df["Ad group"].isin(st.session_state.ad_group_filter)
    ]

if st.session_state.ad_strength_filter:
    filtered_df = filtered_df[
        filtered_df["Ad strength"].isin(st.session_state.ad_strength_filter)
    ]

# ─────────────────────────────
# KPI Section
# ─────────────────────────────
st.subheader("📌 Key Performance Indicators")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Impressions", f"{int(filtered_df['Impr.'].sum()):,}")
k2.metric("Total Interactions", f"{int(filtered_df['Interactions'].sum()):,}")
k3.metric("Total Cost (₹)", f"{filtered_df['Cost'].sum():,.2f}")
k4.metric("Total Conversions", f"{filtered_df['Conversions'].sum():,.2f}")

st.divider()

# ─────────────────────────────
# Helper: Plotly Graph + Summary
# ─────────────────────────────
def graph_with_summary(fig, summary, number, title):
    st.subheader(f"{number} {title}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(summary)
    st.divider()

# ─────────────────────────────
# 1️⃣ Top Headlines by Usage Frequency
# ─────────────────────────────
headline_cols = [
    col for col in filtered_df.columns
    if col.startswith("Headline ") and "position" not in col
]

headlines = (
    filtered_df[headline_cols]
    .replace("--", pd.NA)
    .stack()
    .dropna()
)

headline_usage = headlines.value_counts().reset_index()
headline_usage.columns = ["Headline", "Usage Count"]
top_headlines = headline_usage.head(10)

fig_headlines = px.bar(
    top_headlines,
    x="Usage Count",
    y="Headline",
    orientation="h",
    title="Top Headlines by Usage Frequency"
)

graph_with_summary(
    fig_headlines,
    f"""
    **Insight:**  
    A small subset of headlines dominates creative usage.

    **Recommendation:**  
    Reuse high-performing headlines and test variations for underused creatives.
    """,
    "1️⃣",
    "Headline Usage Analysis"
)

# # ─────────────────────────────
# # 1️⃣ Cost per Conversion by Campaign
# # ─────────────────────────────
# campaign_perf = (
#     filtered_df
#     .groupby("Campaign", as_index=False)
#     .agg({
#         "Impr.": "sum",
#         "Interactions": "sum",
#         "Cost": "sum",
#         "Conversions": "sum"
#     })
# )

# campaign_perf["Cost per Conversion"] = (
#     campaign_perf["Cost"] / campaign_perf["Conversions"]
# )

# best_campaign = campaign_perf.loc[campaign_perf["Cost per Conversion"].idxmin()]
# worst_campaign = campaign_perf.loc[campaign_perf["Cost per Conversion"].idxmax()]

# fig_campaign = px.bar(
#     campaign_perf,
#     x="Campaign",
#     y="Cost per Conversion",
#     title="Cost per Conversion by Campaign",
#     labels={"Cost per Conversion": "Cost per Conversion (₹)"},
#     hover_data=["Cost", "Conversions"]
# )

# graph_with_summary(
#     fig_campaign,
#     f"""
#     **Insight:**  
#     **{best_campaign['Campaign']}** is the most cost-efficient campaign  
#     (₹{best_campaign['Cost per Conversion']:.2f} per conversion).  
#     **{worst_campaign['Campaign']}** is the most expensive.

#     **Recommendation:**  
#     Scale efficient campaigns and optimize or restructure high-cost campaigns.
#     """,
#     "1️⃣",
#     "Campaign Cost Efficiency"
# )

# # ─────────────────────────────
# # 2️⃣ Ad Strength vs Cost per Conversion
# # ─────────────────────────────
# ad_strength_perf = (
#     filtered_df
#     .groupby("Ad strength", as_index=False)
#     .agg({
#         "Impr.": "sum",
#         "Interactions": "sum",
#         "Conversions": "sum",
#         "Cost": "sum"
#     })
# )

# ad_strength_perf["Cost per Conversion"] = (
#     ad_strength_perf["Cost"] / ad_strength_perf["Conversions"]
# )

# best_strength = ad_strength_perf.loc[
#     ad_strength_perf["Cost per Conversion"].idxmin()
# ]

# fig_strength = px.bar(
#     ad_strength_perf,
#     x="Ad strength",
#     y="Cost per Conversion",
#     title="Ad Strength vs Cost Efficiency",
#     labels={"Cost per Conversion": "Cost per Conversion (₹)"},
#     hover_data=["Impr.", "Interactions"]
# )

# graph_with_summary(
#     fig_strength,
#     f"""
#     **Insight:**  
#     Ads with **{best_strength['Ad strength']}** strength deliver the lowest
#     cost per conversion.

#     **Recommendation:**  
#     Improve creative quality to achieve higher Ad Strength ratings.
#     """,
#     "2️⃣",
#     "Ad Strength Performance"
# )

# # ─────────────────────────────
# # 4️⃣ Wasteful Ads (High Cost, Zero Conversions)
# # ─────────────────────────────
# waste_ads = filtered_df[
#     (filtered_df["Cost"] > filtered_df["Cost"].median()) &
#     (filtered_df["Conversions"] == 0)
# ]

# st.subheader("4️⃣ Wasteful Ads Identification")

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.dataframe(
#         waste_ads[
#             ["Campaign", "Ad group", "Cost", "Interactions", "Conversions"]
#         ],
#         use_container_width=True
#     )

# with col2:
#     st.markdown(
#         f"""
#         **Insight:**  
#         **{len(waste_ads)} ads** consume above-median cost  
#         while generating **zero conversions**.

#         **Recommendation:**  
#         Pause, fix targeting, or replace these ads immediately.
#         """
#     )

# st.divider()


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from dotenv import load_dotenv  

# load_dotenv()

# DATA_PATH = os.getenv("GA_REPORT_PATH")

# # ─────────────────────────────
# # Page Configuration
# # ─────────────────────────────
# st.set_page_config(
#     page_title="Google Ads Ad Performance Dashboard",
#     page_icon="📊",
#     layout="wide"
# )

# st.title("📊 Google Ads Ad Performance Dashboard")
# st.caption("EDA, creative insights & automated recommendations")

# # ─────────────────────────────
# # Load & Clean Data
# # ─────────────────────────────
# @st.cache_data
# def load_data():
#     df = pd.read_csv(
#         DATA_PATH,
#         engine="python",
#         skiprows=2
#     )

#     df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")]
#     df.reset_index(drop=True, inplace=True)

#     # Clean Ad strength
#     df["Ad strength"] = (
#         df["Ad strength"]
#         .astype(str)
#         .str.strip()
#         .replace({"--": "Not Evaluated", "—": "Not Evaluated", "nan": "Not Evaluated"})
#     )

#     numeric_cols = [
#         "Impr.", "Interactions", "Interaction rate",
#         "Conv. rate", "Conversions", "Cost", "Cost / conv."
#     ]

#     for col in numeric_cols:
#         df[col] = (
#             df[col]
#             .astype(str)
#             .str.replace(",", "")
#             .str.replace("%", "")
#         )
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     return df

# ad_df = load_data()

# # ─────────────────────────────
# # Sidebar Filters (Session Managed)
# # ─────────────────────────────
# st.sidebar.header("🔎 Filters")

# campaigns = st.sidebar.multiselect(
#     "Campaign",
#     sorted(ad_df["Campaign"].dropna().unique())
# )

# ad_groups = st.sidebar.multiselect(
#     "Ad Group",
#     sorted(ad_df["Ad group"].dropna().unique())
# )

# ad_strengths = st.sidebar.multiselect(
#     "Ad Strength",
#     sorted(ad_df["Ad strength"].dropna().unique())
# )

# # Apply Filters
# filtered_df = ad_df.copy()

# if campaigns:
#     filtered_df = filtered_df[filtered_df["Campaign"].isin(campaigns)]

# if ad_groups:
#     filtered_df = filtered_df[filtered_df["Ad group"].isin(ad_groups)]

# if ad_strengths:
#     filtered_df = filtered_df[filtered_df["Ad strength"].isin(ad_strengths)]

# # ─────────────────────────────
# # KPI Section
# # ─────────────────────────────
# st.subheader("📌 Key Performance Indicators")

# k1, k2, k3, k4 = st.columns(4)
# k1.metric("Total Impressions", f"{int(filtered_df['Impr.'].sum()):,}")
# k2.metric("Total Interactions", f"{int(filtered_df['Interactions'].sum()):,}")
# k3.metric("Total Cost (₹)", f"{filtered_df['Cost'].sum():,.2f}")
# k4.metric("Total Conversions", f"{filtered_df['Conversions'].sum():,.2f}")

# st.divider()

# # ─────────────────────────────
# # Helper Function (Graph + Summary Layout)
# # ─────────────────────────────
# def graph_with_summary(plot_func, summary, number, title):
#     st.subheader(f"{number} {title}")
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         plot_func()
#     with col2:
#         st.markdown(summary)
#     st.divider()

# # ─────────────────────────────
# # 1️⃣ Cost per Conversion by Campaign
# # ─────────────────────────────
# campaign_perf = (
#     filtered_df
#     .groupby("Campaign", as_index=False)
#     .agg({
#         "Impr.": "sum",
#         "Interactions": "sum",
#         "Cost": "sum",
#         "Conversions": "sum"
#     })
# )

# campaign_perf["Cost per Conversion"] = (
#     campaign_perf["Cost"] / campaign_perf["Conversions"]
# )

# avg_cpc = campaign_perf["Cost per Conversion"].mean()
# best_campaign = campaign_perf.loc[campaign_perf["Cost per Conversion"].idxmin()]
# worst_campaign = campaign_perf.loc[campaign_perf["Cost per Conversion"].idxmax()]

# def plot_campaign_cpc():
#     plt.figure()
#     plt.bar(campaign_perf["Campaign"], campaign_perf["Cost per Conversion"])
#     plt.xticks(rotation=45, ha="right")
#     plt.ylabel("Cost per Conversion (₹)")
#     plt.title("Cost per Conversion by Campaign")
#     st.pyplot(plt)

# graph_with_summary(
#     plot_campaign_cpc,
#     f"""
#     **Insight:**  
#     '{best_campaign['Campaign']}' is the most cost-efficient campaign  
#     (₹{best_campaign['Cost per Conversion']:.2f} per conversion).  
#     '{worst_campaign['Campaign']}' is the most expensive.

#     **Recommendation:**  
#     Scale efficient campaigns and optimize or restructure high-cost campaigns.
#     """,
#     "1️⃣",
#     "Campaign Cost Efficiency"
# )

# # ─────────────────────────────
# # 2️⃣ Ad Strength vs Cost per Conversion
# # ─────────────────────────────
# ad_strength_perf = (
#     filtered_df
#     .groupby("Ad strength", as_index=False)
#     .agg({
#         "Impr.": "sum",
#         "Interactions": "sum",
#         "Conversions": "sum",
#         "Cost": "sum"
#     })
# )

# ad_strength_perf["Cost per Conversion"] = (
#     ad_strength_perf["Cost"] / ad_strength_perf["Conversions"]
# )

# best_strength = ad_strength_perf.loc[
#     ad_strength_perf["Cost per Conversion"].idxmin()
# ]

# def plot_ad_strength_cpc():
#     plt.figure()
#     plt.bar(ad_strength_perf["Ad strength"], ad_strength_perf["Cost per Conversion"])
#     plt.ylabel("Cost per Conversion (₹)")
#     plt.title("Ad Strength vs Cost Efficiency")
#     st.pyplot(plt)

# graph_with_summary(
#     plot_ad_strength_cpc,
#     f"""
#     **Insight:**  
#     Ads with **'{best_strength['Ad strength']}'** strength deliver the lowest
#     cost per conversion.

#     **Recommendation:**  
#     Improve creative quality to reach higher Ad Strength ratings.
#     """,
#     "2️⃣",
#     "Ad Strength Performance"
# )

# # ─────────────────────────────
# # 3️⃣ Top Headlines by Usage Frequency
# # ─────────────────────────────
# headline_cols = [
#     col for col in filtered_df.columns
#     if col.startswith("Headline ") and "position" not in col
# ]

# headlines = (
#     filtered_df[headline_cols]
#     .replace("--", pd.NA)
#     .stack()
#     .dropna()
# )

# headline_usage = headlines.value_counts().reset_index()
# headline_usage.columns = ["Headline", "Usage Count"]
# top_headlines = headline_usage.head(10)

# def plot_top_headlines():
#     plt.figure()
#     plt.barh(top_headlines["Headline"], top_headlines["Usage Count"])
#     plt.gca().invert_yaxis()
#     plt.xlabel("Usage Count")
#     plt.title("Top Headlines by Usage")
#     st.pyplot(plt)

# graph_with_summary(
#     plot_top_headlines,
#     f"""
#     **Insight:**  
#     A small subset of headlines dominates creative usage.

#     **Recommendation:**  
#     Reuse high-performing headlines and test variations for low-usage ones.
#     """,
#     "3️⃣",
#     "Headline Usage Analysis"
# )

# # ─────────────────────────────
# # 4️⃣ Wasteful Ads (High Cost, Zero Conversions)
# # ─────────────────────────────
# waste_ads = filtered_df[
#     (filtered_df["Cost"] > filtered_df["Cost"].median()) &
#     (filtered_df["Conversions"] == 0)
# ]

# st.subheader("4️⃣ Wasteful Ads Identification")

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.dataframe(
#         waste_ads[
#             ["Campaign", "Ad group", "Cost", "Interactions", "Conversions"]
#         ],
#         use_container_width=True
#     )

# with col2:
#     st.markdown(
#         f"""
#         **Insight:**  
#         **{len(waste_ads)} ads** consume above-median cost
#         while generating **zero conversions**.

#         **Recommendation:**  
#         Pause, fix targeting, or replace these ads immediately.
#         """
#     )

# st.divider()
