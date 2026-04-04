import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os

# ───────────────────────── PAGE SETUP ─────────────────────────
st.set_page_config(
    page_title="Google Ads Asset Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Google Ads Asset Performance Dashboard")

# ───────────────────────── ENV SETUP ─────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Ad_asset_report.csv")

if not DATA_PATH:
    st.error("Environment variable GA_ASSET_REPORT_PATH is not set.")
    st.stop()

# ───────────────────────── SESSION STATE INIT ─────────────────────────
if "filters" not in st.session_state:
    st.session_state.filters = {
        "campaign": [],
        "asset_type": [],
        "asset_status": []
    }

# ───────────────────────── LOAD & CLEAN DATA ─────────────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, engine="python", skiprows=2)
    df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")]
    df.reset_index(drop=True, inplace=True)

    numeric_cols = ["Impr.", "Interactions", "Interaction rate", "Clicks", "Cost"]
    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Cost per Interaction"] = df["Cost"] / df["Interactions"]

    df["Last updated"] = pd.to_datetime(df["Last updated"], errors="coerce")
    df["Days Since Update"] = (
        pd.Timestamp.today().normalize() - df["Last updated"]
    ).dt.days

    return df

asset_df = load_data(DATA_PATH)

# ───────────────────────── FILTERS (SESSION-DRIVEN) ─────────────────────────
with st.sidebar:
    col1, col2 = st.columns(2)

    with col1:
        st.header("Filters")

    with col2:
        if st.button("❌ Clear Filters"):
            st.session_state.filters = {
                "campaign": [],
                "asset_type": [],
                "asset_status": []
            }
            st.rerun()

    st.session_state.filters["campaign"] = st.multiselect(
        "Campaign",
        sorted(asset_df["Campaign"].dropna().unique()),
        default=st.session_state.filters["campaign"]
    )

    st.session_state.filters["asset_type"] = st.multiselect(
        "Asset Type",
        sorted(asset_df["Asset type"].dropna().unique()),
        default=st.session_state.filters["asset_type"]
    )

    st.session_state.filters["asset_status"] = st.multiselect(
        "Asset Status",
        sorted(asset_df["Asset status"].dropna().unique()),
        default=st.session_state.filters["asset_status"]
    )

# ───────────────────────── APPLY FILTERS ─────────────────────────
df = asset_df.copy()
f = st.session_state.filters

if f["campaign"]:
    df = df[df["Campaign"].isin(f["campaign"])]

if f["asset_type"]:
    df = df[df["Asset type"].isin(f["asset_type"])]

if f["asset_status"]:
    df = df[df["Asset status"].isin(f["asset_status"])]

# Empty state handling
if df.empty:
    st.warning("No data available for the selected filter combination.")
    st.stop()

# ───────────────────────── HELPER LAYOUT ─────────────────────────
def graph_section(title, fig, summary):
    left, right = st.columns([3, 2])
    with left:
        st.subheader(title)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("### 🔍 Insight")
        st.write(summary)

# ───────────────────────── 1. DISTRIBUTION OF INTERACTIONS ─────────────────────────
fig1 = px.histogram(df, x="Interactions", nbins=20)
high_engagement = (df["Interactions"] > df["Interactions"].median()).sum()

graph_section(
    "1️⃣ Distribution of Asset Interactions",
    fig1,
    f"{high_engagement} assets perform above median interactions, indicating engagement concentration."
)

# ───────────────────────── AGGREGATIONS ─────────────────────────
asset_type_perf = (
    df.groupby("Asset type", as_index=False)
    .agg({"Cost": "sum", "Interactions": "sum"})
)
asset_type_perf["Cost per Interaction"] = (
    asset_type_perf["Cost"] / asset_type_perf["Interactions"]
)

# ───────────────────────── 2. CPI BY ASSET TYPE ─────────────────────────
fig2 = px.bar(asset_type_perf, x="Asset type", y="Cost per Interaction")
best = asset_type_perf.loc[asset_type_perf["Cost per Interaction"].idxmin()]
worst = asset_type_perf.loc[asset_type_perf["Cost per Interaction"].idxmax()]

graph_section(
    "2️⃣ Asset Type Efficiency (CPI)",
    fig2,
    f"Most efficient: {best['Asset type']} (₹{best['Cost per Interaction']:.2f}). "
    f"Least efficient: {worst['Asset type']} (₹{worst['Cost per Interaction']:.2f})."
)

# ───────────────────────── 3. ENGAGEMENT BY ASSET TYPE ─────────────────────────
fig3 = px.bar(asset_type_perf, x="Asset type", y="Interactions")
top_eng = asset_type_perf.loc[asset_type_perf["Interactions"].idxmax()]

graph_section(
    "3️⃣ Engagement Contribution by Asset Type",
    fig3,
    f"{top_eng['Asset type']} drives the highest engagement "
    f"({int(top_eng['Interactions']):,} interactions)."
)

# ───────────────────────── 4. SPEND BY ASSET TYPE ─────────────────────────
fig4 = px.bar(asset_type_perf, x="Asset type", y="Cost")
highest_spend = asset_type_perf.loc[asset_type_perf["Cost"].idxmax()]

graph_section(
    "4️⃣ Spend Distribution by Asset Type",
    fig4,
    f"Highest spend is on {highest_spend['Asset type']} "
    f"(₹{highest_spend['Cost']:,.2f})."
)

# ───────────────────────── 5. COST VS INTERACTION RATE ─────────────────────────
fig5 = px.scatter(
    df,
    x="Cost",
    y="Interaction rate",
    hover_data=["Asset", "Asset type"]
)

efficient = df[
    (df["Cost"] <= df["Cost"].median()) &
    (df["Interaction rate"] >= df["Interaction rate"].median())
]

graph_section(
    "5️⃣ Cost vs Interaction Rate (Asset Level)",
    fig5,
    f"{len(efficient)} assets fall into the high-efficiency zone "
    f"(low cost, high interaction rate)."
)

# ───────────────────────── 6. TOP 10 ASSETS BY INTERACTIONS ─────────────────────────
top_assets = df.sort_values("Interactions", ascending=False).head(10)

fig6 = px.bar(
    top_assets,
    x="Interactions",
    y="Asset",
    orientation="h"
)

graph_section(
    "6️⃣ Top Performing Assets",
    fig6,
    f"'{top_assets.iloc[0]['Asset']}' is the highest engagement asset."
)

# # ───────────────────────── 7. ASSET STATUS DISTRIBUTION ─────────────────────────
# status_counts = df["Asset status"].value_counts().reset_index()
# status_counts.columns = ["Asset status", "Count"]

# fig7 = px.bar(status_counts, x="Asset status", y="Count")
# inactive = status_counts[status_counts["Asset status"] != "Enabled"]["Count"].sum()

# graph_section(
#     "7️⃣ Asset Status Distribution",
#     fig7,
#     f"{inactive} assets are inactive or non-serving and should be reviewed."
# )

# ───────────────────────── 8. FRESHNESS VS ENGAGEMENT ─────────────────────────
fig8 = px.scatter(df, x="Days Since Update", y="Interactions")

graph_section(
    "7️⃣ Asset Freshness vs Engagement",
    fig8,
    "Recently updated assets tend to show higher engagement."
)

# # ───────────────────────── 9. PARETO ANALYSIS ─────────────────────────
# pareto = df.sort_values("Interactions", ascending=False)
# pareto["Cumulative Interaction %"] = (
#     pareto["Interactions"].cumsum() / pareto["Interactions"].sum() * 100
# )

# fig9 = px.line(pareto, y="Cumulative Interaction %")
# fig9.add_hline(y=80, line_dash="dash")

# top_80 = pareto[pareto["Cumulative Interaction %"] <= 80]

# graph_section(
#     "9️⃣ Pareto Analysis (80/20 Rule)",
#     fig9,
#     f"{len(top_80)} assets generate 80% of total interactions."
# )

# ───────────────────────── 10. SPEND VS VALUE SHARE ─────────────────────────
spend_value = (
    df.groupby("Asset type", as_index=False)
    .agg({"Cost": "sum", "Interactions": "sum"})
)

spend_value["Cost Share (%)"] = spend_value["Cost"] / spend_value["Cost"].sum() * 100
spend_value["Interaction Share (%)"] = (
    spend_value["Interactions"] / spend_value["Interactions"].sum() * 100
)

fig10 = px.scatter(
    spend_value,
    x="Cost Share (%)",
    y="Interaction Share (%)",
    text="Asset type"
)

waste = spend_value[
    spend_value["Cost Share (%)"] > spend_value["Interaction Share (%)"]
]

graph_section(
    "8️⃣ Spend vs Value Contribution",
    fig10,
    f"{len(waste)} asset types consume more budget than value delivered."
)

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import numpy as np
# from dotenv import load_dotenv
# import os

# # ───────────────────────── PAGE SETUP ─────────────────────────
# st.set_page_config(
#     page_title="Google Ads Asset Performance Dashboard",
#     page_icon="📊",
#     layout="wide"
# )

# st.title("📊 Google Ads Asset Performance Dashboard")

# load_dotenv()

# DATA_PATH = os.getenv("GA_ASSET_REPORT_PATH")

# # ───────────────────────── SESSION STATE ─────────────────────────
# if "filters" not in st.session_state:
#     st.session_state.filters = {
#         "campaign": [],
#         "asset_type": [],
#         "asset_status": []
#     }

# # ───────────────────────── LOAD & CLEAN DATA ─────────────────────────
# @st.cache_data
# def load_data():
#     df = pd.read_csv(DATA_PATH, engine="python", skiprows=2)
#     df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")]
#     df.reset_index(drop=True, inplace=True)

#     numeric_cols = ["Impr.", "Interactions", "Interaction rate", "Clicks", "Cost"]
#     for col in numeric_cols:
#         df[col] = (
#             df[col]
#             .astype(str)
#             .str.replace(",", "")
#             .str.replace("%", "")
#         )
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     df["Cost per Interaction"] = df["Cost"] / df["Interactions"]
#     df["Last updated"] = pd.to_datetime(df["Last updated"], errors="coerce")
#     df["Days Since Update"] = (pd.Timestamp.today() - df["Last updated"]).dt.days

#     return df

# asset_df = load_data()

# # ───────────────────────── FILTERS ─────────────────────────
# with st.sidebar:
#     col1, col2 = st.columns(2)

#     with col1:
#         st.header("Filters")

#     with col2:
#         if st.button("❌ Clear Filters"):
#             st.session_state.filters = {"campaign": [], "asset_type": [], "asset_status": []}
#             st.rerun()
    
#     campaign_filter = st.multiselect(
#         "Campaign",
#         sorted(asset_df["Campaign"].dropna().unique()),
#         default=st.session_state.filters["campaign"]
#     )

#     asset_type_filter = st.multiselect(
#         "Asset Type",
#         sorted(asset_df["Asset type"].dropna().unique()),
#         default=st.session_state.filters["asset_type"]
#     )

#     asset_status_filter = st.multiselect(
#         "Asset Status",
#         sorted(asset_df["Asset status"].dropna().unique()),
#         default=st.session_state.filters["asset_status"]
#     )

#     st.session_state.filters["campaign"] = campaign_filter
#     st.session_state.filters["asset_type"] = asset_type_filter
#     st.session_state.filters["asset_status"] = asset_status_filter

# # Apply filters
# df = asset_df.copy()
# if campaign_filter:
#     df = df[df["Campaign"].isin(campaign_filter)]
# if asset_type_filter:
#     df = df[df["Asset type"].isin(asset_type_filter)]
# if asset_status_filter:
#     df = df[df["Asset status"].isin(asset_status_filter)]

# # ───────────────────────── HELPER FOR SECTION LAYOUT ─────────────────────────
# def graph_section(title, fig, summary):
#     left, right = st.columns([3, 2])
#     with left:
#         st.subheader(title)
#         st.plotly_chart(fig, use_container_width=True)
#     with right:
#         st.markdown("### 🔍 Insight")
#         st.write(summary)

# # ───────────────────────── 1. DISTRIBUTION OF INTERACTIONS ─────────────────────────
# fig1 = px.histogram(
#     df,
#     x="Interactions",
#     nbins=20,
#     title="Distribution of Asset Interactions"
# )

# high_engagement = (df["Interactions"] > df["Interactions"].median()).sum()

# graph_section(
#     "1️⃣ Distribution of Asset Interactions",
#     fig1,
#     f"{high_engagement} assets perform above median interactions, indicating engagement concentration."
# )

# # ───────────────────────── AGGREGATIONS ─────────────────────────
# asset_type_perf = (
#     df.groupby("Asset type", as_index=False)
#     .agg({"Cost": "sum", "Interactions": "sum"})
# )
# asset_type_perf["Cost per Interaction"] = (
#     asset_type_perf["Cost"] / asset_type_perf["Interactions"]
# )

# # ───────────────────────── 2. CPI BY ASSET TYPE ─────────────────────────
# fig2 = px.bar(
#     asset_type_perf,
#     x="Asset type",
#     y="Cost per Interaction",
#     title="Cost per Interaction by Asset Type"
# )

# best = asset_type_perf.loc[asset_type_perf["Cost per Interaction"].idxmin()]
# worst = asset_type_perf.loc[asset_type_perf["Cost per Interaction"].idxmax()]

# graph_section(
#     "2️⃣ Asset Type Efficiency (CPI)",
#     fig2,
#     f"Most efficient: {best['Asset type']} (₹{best['Cost per Interaction']:.2f}). "
#     f"Least efficient: {worst['Asset type']} (₹{worst['Cost per Interaction']:.2f})."
# )

# # ───────────────────────── 3. ENGAGEMENT BY ASSET TYPE ─────────────────────────
# fig3 = px.bar(
#     asset_type_perf,
#     x="Asset type",
#     y="Interactions",
#     title="Engagement by Asset Type"
# )

# top_eng = asset_type_perf.loc[asset_type_perf["Interactions"].idxmax()]

# graph_section(
#     "3️⃣ Engagement Contribution by Asset Type",
#     fig3,
#     f"{top_eng['Asset type']} drives the highest engagement "
#     f"({int(top_eng['Interactions']):,} interactions)."
# )

# # ───────────────────────── 4. SPEND BY ASSET TYPE ─────────────────────────
# fig4 = px.bar(
#     asset_type_perf,
#     x="Asset type",
#     y="Cost",
#     title="Spend Distribution by Asset Type"
# )

# highest_spend = asset_type_perf.loc[asset_type_perf["Cost"].idxmax()]

# graph_section(
#     "4️⃣ Spend Distribution by Asset Type",
#     fig4,
#     f"Highest spend is on {highest_spend['Asset type']} "
#     f"(₹{highest_spend['Cost']:,.2f})."
# )

# # ───────────────────────── 5. COST VS INTERACTION RATE ─────────────────────────
# fig5 = px.scatter(
#     df,
#     x="Cost",
#     y="Interaction rate",
#     title="Cost vs Interaction Rate",
#     hover_data=["Asset", "Asset type"]
# )

# efficient = df[
#     (df["Cost"] <= df["Cost"].median()) &
#     (df["Interaction rate"] >= df["Interaction rate"].median())
# ]

# graph_section(
#     "5️⃣ Cost vs Interaction Rate (Asset Level)",
#     fig5,
#     f"{len(efficient)} assets fall into the high-efficiency zone "
#     f"(low cost, high interaction rate)."
# )

# # ───────────────────────── 6. TOP 10 ASSETS BY INTERACTIONS ─────────────────────────
# top_assets = df.sort_values("Interactions", ascending=False).head(10)

# fig6 = px.bar(
#     top_assets,
#     x="Interactions",
#     y="Asset",
#     orientation="h",
#     title="Top 10 Assets by Interactions"
# )

# graph_section(
#     "6️⃣ Top Performing Assets",
#     fig6,
#     f"'{top_assets.iloc[0]['Asset']}' is the highest engagement asset."
# )

# # ───────────────────────── 7. ASSET STATUS DISTRIBUTION ─────────────────────────
# status_counts = df["Asset status"].value_counts().reset_index()
# status_counts.columns = ["Asset status", "Count"]

# fig7 = px.bar(
#     status_counts,
#     x="Asset status",
#     y="Count",
#     title="Asset Status Distribution"
# )

# inactive = status_counts[status_counts["Asset status"] != "Enabled"]["Count"].sum()

# graph_section(
#     "7️⃣ Asset Status Distribution",
#     fig7,
#     f"{inactive} assets are inactive or non-serving and should be reviewed."
# )

# # ───────────────────────── 8. FRESHNESS VS ENGAGEMENT ─────────────────────────
# fig8 = px.scatter(
#     df,
#     x="Days Since Update",
#     y="Interactions",
#     title="Asset Freshness vs Engagement"
# )

# graph_section(
#     "8️⃣ Asset Freshness vs Engagement",
#     fig8,
#     "Recently updated assets tend to show higher engagement."
# )

# # ───────────────────────── 9. PARETO ANALYSIS ─────────────────────────
# pareto = df.sort_values("Interactions", ascending=False)
# pareto["Cumulative Interaction %"] = (
#     pareto["Interactions"].cumsum() /
#     pareto["Interactions"].sum() * 100
# )

# fig9 = px.line(
#     pareto,
#     y="Cumulative Interaction %",
#     title="Pareto Analysis of Asset Engagement"
# )
# fig9.add_hline(y=80, line_dash="dash")

# top_80 = pareto[pareto["Cumulative Interaction %"] <= 80]

# graph_section(
#     "9️⃣ Pareto Analysis (80/20 Rule)",
#     fig9,
#     f"{len(top_80)} assets generate 80% of total interactions."
# )

# # ───────────────────────── 10. SPEND VS VALUE SHARE ─────────────────────────
# spend_value = (
#     df.groupby("Asset type", as_index=False)
#     .agg({"Cost": "sum", "Interactions": "sum"})
# )

# spend_value["Cost Share (%)"] = spend_value["Cost"] / spend_value["Cost"].sum() * 100
# spend_value["Interaction Share (%)"] = (
#     spend_value["Interactions"] / spend_value["Interactions"].sum() * 100
# )

# fig10 = px.scatter(
#     spend_value,
#     x="Cost Share (%)",
#     y="Interaction Share (%)",
#     text="Asset type",
#     title="Spend vs Value Contribution"
# )

# waste = spend_value[
#     spend_value["Cost Share (%)"] > spend_value["Interaction Share (%)"]
# ]

# graph_section(
#     "🔟 Spend vs Value Contribution",
#     fig10,
#     f"{len(waste)} asset types consume more budget than value delivered."
# )

# ----------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from datetime import datetime
# from dotenv import load_dotenv
# import os

# load_dotenv()

# DATA_PATH = os.getenv("GA_ASSET_REPORT_PATH")

# # ─────────────────────────────
# # Page Configuration
# # ─────────────────────────────
# st.set_page_config(
#     page_title="Google Ads Asset Performance Dashboard",
#     page_icon="📊",
#     layout="wide"
# )

# st.title("📊 Google Ads Asset Performance Dashboard")
# st.caption("Comprehensive EDA & automated insights for Google Ads asset-level performance")

# # ─────────────────────────────
# # Load & Clean Data
# # ─────────────────────────────
# @st.cache_data
# def load_data():
#     df = pd.read_csv(DATA_PATH, engine="python", skiprows=2)

#     df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")]
#     df.reset_index(drop=True, inplace=True)

#     numeric_cols = ["Impr.", "Interactions", "Interaction rate", "Clicks", "Cost"]
#     for col in numeric_cols:
#         df[col] = (
#             df[col].astype(str)
#             .str.replace(",", "")
#             .str.replace("%", "")
#         )
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     df["Cost per Interaction"] = df["Cost"] / df["Interactions"]
#     df["Last updated"] = pd.to_datetime(df["Last updated"], errors="coerce")
#     df["Days Since Update"] = (pd.Timestamp.today() - df["Last updated"]).dt.days

#     return df

# asset_df = load_data()

# # ─────────────────────────────
# # Sidebar Filters
# # ─────────────────────────────
# st.sidebar.header("🔎 Filters")

# campaigns = st.sidebar.multiselect(
#     "Campaign",
#     sorted(asset_df["Campaign"].dropna().unique())
# )

# asset_types = st.sidebar.multiselect(
#     "Asset Type",
#     sorted(asset_df["Asset type"].dropna().unique())
# )

# statuses = st.sidebar.multiselect(
#     "Asset Status",
#     sorted(asset_df["Asset status"].dropna().unique())
# )

# date_range = st.sidebar.slider(
#     "Days Since Last Update",
#     int(asset_df["Days Since Update"].min()),
#     int(asset_df["Days Since Update"].max()),
#     (
#         int(asset_df["Days Since Update"].min()),
#         int(asset_df["Days Since Update"].max())
#     )
# )

# filtered_df = asset_df.copy()

# if campaigns:
#     filtered_df = filtered_df[filtered_df["Campaign"].isin(campaigns)]
# if asset_types:
#     filtered_df = filtered_df[filtered_df["Asset type"].isin(asset_types)]
# if statuses:
#     filtered_df = filtered_df[filtered_df["Asset status"].isin(statuses)]

# filtered_df = filtered_df[
#     (filtered_df["Days Since Update"] >= date_range[0]) &
#     (filtered_df["Days Since Update"] <= date_range[1])
# ]

# # ─────────────────────────────
# # KPI Section
# # ─────────────────────────────
# st.subheader("📌 Key Performance Indicators")

# k1, k2, k3, k4 = st.columns(4)
# k1.metric("Total Cost (₹)", f"{filtered_df['Cost'].sum():,.2f}")
# k2.metric("Total Interactions", f"{int(filtered_df['Interactions'].sum()):,}")
# k3.metric("Avg Cost / Interaction (₹)", f"{filtered_df['Cost per Interaction'].mean():.2f}")
# k4.metric("Total Assets", f"{filtered_df['Asset'].nunique()}")

# st.divider()

# # ─────────────────────────────
# # Helper Layout
# # ─────────────────────────────
# def graph_with_summary(fig, summary, number, title):
#     st.subheader(f"{number} {title}")
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.plotly_chart(fig, use_container_width=True)
#     with col2:
#         st.markdown(summary)
#     st.divider()

# # ─────────────────────────────
# # 1️⃣ Distribution of Interactions
# # ─────────────────────────────
# fig_interactions = px.histogram(
#     filtered_df,
#     x="Interactions",
#     nbins=25,
#     title="Distribution of Asset Interactions"
# )

# high_engagement = (filtered_df["Interactions"] > filtered_df["Interactions"].median()).sum()

# graph_with_summary(
#     fig_interactions,
#     f"""
#     **Insight:**  
#     Interaction distribution is highly skewed.  
#     **{high_engagement} assets** perform above the median interaction level.

#     **Recommendation:**  
#     Focus optimization on high-engagement creatives.
#     """,
#     "1️⃣",
#     "Distribution of Asset Interactions"
# )

# # ─────────────────────────────
# # 2️⃣ Asset Type vs Cost per Interaction
# # ─────────────────────────────
# asset_type_perf = (
#     filtered_df.groupby("Asset type", as_index=False)
#     .agg({"Cost": "sum", "Interactions": "sum"})
# )
# asset_type_perf["Cost per Interaction"] = (
#     asset_type_perf["Cost"] / asset_type_perf["Interactions"]
# )

# best = asset_type_perf.loc[asset_type_perf["Cost per Interaction"].idxmin()]
# worst = asset_type_perf.loc[asset_type_perf["Cost per Interaction"].idxmax()]

# fig_asset_type = px.bar(
#     asset_type_perf,
#     x="Asset type",
#     y="Cost per Interaction",
#     title="Asset Type Efficiency",
#     labels={"Cost per Interaction": "Cost per Interaction (₹)"},
#     hover_data=["Cost", "Interactions"]
# )

# graph_with_summary(
#     fig_asset_type,
#     f"""
#     **Insight:**  
#     **{best['Asset type']}** is the most cost-efficient asset type  
#     (₹{best['Cost per Interaction']:.2f} per interaction).  
#     **{worst['Asset type']}** is the least efficient.

#     **Recommendation:**  
#     Scale efficient asset types and optimize underperformers.
#     """,
#     "2️⃣",
#     "Asset Type vs Cost per Interaction"
# )

# # ─────────────────────────────
# # 3️⃣ Top 10 Assets by Interactions
# # ─────────────────────────────
# top_assets = filtered_df.sort_values("Interactions", ascending=False).head(10)
# top_asset = top_assets.iloc[0]

# fig_top_assets = px.bar(
#     top_assets,
#     x="Interactions",
#     y="Asset",
#     orientation="h",
#     title="Top 10 Assets by Engagement"
# )

# graph_with_summary(
#     fig_top_assets,
#     f"""
#     **Insight:**  
#     **'{top_asset['Asset']}'** is the highest-performing asset  
#     with **{int(top_asset['Interactions']):,} interactions**.

#     **Recommendation:**  
#     Reuse and replicate similar creatives.
#     """,
#     "3️⃣",
#     "Top Performing Assets"
# )

# # ─────────────────────────────
# # 4️⃣ Asset Status Distribution
# # ─────────────────────────────
# status_counts = filtered_df["Asset status"].value_counts().reset_index()
# status_counts.columns = ["Status", "Count"]

# inactive_assets = status_counts[status_counts["Status"] != "Enabled"]["Count"].sum()

# fig_status = px.bar(
#     status_counts,
#     x="Status",
#     y="Count",
#     title="Asset Status Distribution"
# )

# graph_with_summary(
#     fig_status,
#     f"""
#     **Insight:**  
#     **{inactive_assets} assets** are inactive or non-serving.

#     **Recommendation:**  
#     Periodically clean unused assets.
#     """,
#     "4️⃣",
#     "Asset Status Distribution"
# )

# # ─────────────────────────────
# # 5️⃣ Pareto Analysis (80/20 Rule)
# # ─────────────────────────────
# pareto = filtered_df.sort_values("Interactions", ascending=False)
# pareto["Cumulative %"] = pareto["Interactions"].cumsum() / pareto["Interactions"].sum() * 100
# top_80 = pareto[pareto["Cumulative %"] <= 80]

# fig_pareto = px.line(
#     pareto,
#     y="Cumulative %",
#     title="Pareto Analysis of Asset Engagement"
# )

# fig_pareto.add_hline(y=80, line_dash="dash")

# graph_with_summary(
#     fig_pareto,
#     f"""
#     **Insight:**  
#     **{len(top_80)} out of {len(filtered_df)} assets**  
#     contribute to **80% of total interactions**.

#     **Recommendation:**  
#     Concentrate budgets on high-impact assets.
#     """,
#     "5️⃣",
#     "Pareto Analysis (80/20 Rule)"
# )

# ----------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from datetime import datetime
# from dotenv import load_dotenv
# import os

# load_dotenv()

# DATA_PATH = os.getenv("GA_ASSET_REPORT_PATH")

# # ─────────────────────────────
# # Streamlit Page Config
# # ─────────────────────────────
# st.set_page_config(
#     page_title="Google Ads Asset Performance Dashboard",
#     page_icon="📊",
#     layout="wide"
# )

# st.title("📊 Google Ads Asset Performance Dashboard")
# st.caption("Comprehensive EDA & automated insights for Google Ads asset-level performance")

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

#     numeric_cols = ["Impr.", "Interactions", "Interaction rate", "Clicks", "Cost"]
#     for col in numeric_cols:
#         df[col] = (
#             df[col]
#             .astype(str)
#             .str.replace(",", "")
#             .str.replace("%", "")
#         )
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     df["Cost per Interaction"] = df["Cost"] / df["Interactions"]
#     df["Last updated"] = pd.to_datetime(df["Last updated"], errors="coerce")
#     df["Days Since Update"] = (pd.Timestamp.today() - df["Last updated"]).dt.days

#     return df

# asset_df = load_data()

# # ─────────────────────────────
# # Sidebar Filters (Session Managed)
# # ─────────────────────────────
# st.sidebar.header("🔎 Filters")

# campaigns = st.sidebar.multiselect(
#     "Campaign",
#     sorted(asset_df["Campaign"].dropna().unique())
# )

# asset_types = st.sidebar.multiselect(
#     "Asset Type",
#     sorted(asset_df["Asset type"].dropna().unique())
# )

# statuses = st.sidebar.multiselect(
#     "Asset Status",
#     sorted(asset_df["Asset status"].dropna().unique())
# )

# date_range = st.sidebar.slider(
#     "Days Since Last Update",
#     min_value=int(asset_df["Days Since Update"].min()),
#     max_value=int(asset_df["Days Since Update"].max()),
#     value=(
#         int(asset_df["Days Since Update"].min()),
#         int(asset_df["Days Since Update"].max())
#     )
# )

# # Apply filters
# filtered_df = asset_df.copy()

# if campaigns:
#     filtered_df = filtered_df[filtered_df["Campaign"].isin(campaigns)]

# if asset_types:
#     filtered_df = filtered_df[filtered_df["Asset type"].isin(asset_types)]

# if statuses:
#     filtered_df = filtered_df[filtered_df["Asset status"].isin(statuses)]

# filtered_df = filtered_df[
#     (filtered_df["Days Since Update"] >= date_range[0]) &
#     (filtered_df["Days Since Update"] <= date_range[1])
# ]

# # ─────────────────────────────
# # KPI Section
# # ─────────────────────────────
# st.subheader("📌 Key Performance Indicators")

# k1, k2, k3, k4 = st.columns(4)
# k1.metric("Total Cost (₹)", f"{filtered_df['Cost'].sum():,.2f}")
# k2.metric("Total Interactions", f"{int(filtered_df['Interactions'].sum()):,}")
# k3.metric("Avg Cost / Interaction (₹)", f"{filtered_df['Cost per Interaction'].mean():.2f}")
# k4.metric("Total Assets", f"{filtered_df['Asset'].nunique()}")

# st.divider()

# # ─────────────────────────────
# # Helper Function (Graph + Insight)
# # ─────────────────────────────
# def graph_with_summary(fig_func, summary_text, number, title):
#     st.subheader(f"{number} {title}")
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         fig_func()
#     with col2:
#         st.markdown(summary_text)
#     st.divider()

# # ─────────────────────────────
# # 1️⃣ Distribution of Interactions
# # ─────────────────────────────
# def plot_interaction_dist():
#     plt.figure()
#     plt.hist(filtered_df["Interactions"].dropna(), bins=20)
#     plt.xlabel("Interactions")
#     plt.ylabel("Assets")
#     plt.title("Distribution of Asset Interactions")
#     st.pyplot(plt)

# high_engagement = (filtered_df["Interactions"] > filtered_df["Interactions"].median()).sum()

# graph_with_summary(
#     plot_interaction_dist,
#     f"""
#     **Insight:**  
#     Interaction distribution is highly skewed.  
#     **{high_engagement} assets** perform above the median interaction level,  
#     indicating engagement is driven by a limited number of creatives.
#     """,
#     "1️⃣",
#     "Distribution of Asset Interactions"
# )

# # ─────────────────────────────
# # 2️⃣ Asset Type vs Cost per Interaction
# # ─────────────────────────────
# asset_type_perf = (
#     filtered_df
#     .groupby("Asset type", as_index=False)
#     .agg({"Cost": "sum", "Interactions": "sum"})
# )
# asset_type_perf["Cost per Interaction"] = (
#     asset_type_perf["Cost"] / asset_type_perf["Interactions"]
# )

# best = asset_type_perf.loc[asset_type_perf["Cost per Interaction"].idxmin()]
# worst = asset_type_perf.loc[asset_type_perf["Cost per Interaction"].idxmax()]

# def plot_asset_type_cpi():
#     plt.figure()
#     plt.bar(asset_type_perf["Asset type"], asset_type_perf["Cost per Interaction"])
#     plt.xticks(rotation=45, ha="right")
#     plt.ylabel("Cost per Interaction (₹)")
#     plt.title("Asset Type Efficiency")
#     st.pyplot(plt)

# graph_with_summary(
#     plot_asset_type_cpi,
#     f"""
#     **Insight:**  
#     **{best['Asset type']}** is the most cost-efficient asset type  
#     (₹{best['Cost per Interaction']:.2f} per interaction).  
#     **{worst['Asset type']}** is the least efficient.  

#     **Recommendation:**  
#     Scale efficient asset types and optimize or reduce inefficient ones.
#     """,
#     "2️⃣",
#     "Asset Type vs Cost per Interaction"
# )

# # ─────────────────────────────
# # 3️⃣ Top 10 Assets by Interactions
# # ─────────────────────────────
# top_assets = (
#     filtered_df
#     .sort_values("Interactions", ascending=False)
#     .head(10)
# )

# def plot_top_assets():
#     plt.figure()
#     plt.barh(top_assets["Asset"], top_assets["Interactions"])
#     plt.gca().invert_yaxis()
#     plt.xlabel("Interactions")
#     plt.title("Top 10 Assets by Engagement")
#     st.pyplot(plt)

# top_asset = top_assets.iloc[0]

# graph_with_summary(
#     plot_top_assets,
#     f"""
#     **Insight:**  
#     The asset **'{top_asset['Asset']}'** delivers the highest engagement  
#     with **{int(top_asset['Interactions']):,} interactions**.

#     **Recommendation:**  
#     Replicate or reuse similar creatives to maximize performance.
#     """,
#     "3️⃣",
#     "Top Performing Assets"
# )

# # ─────────────────────────────
# # 4️⃣ Asset Status Distribution
# # ─────────────────────────────
# status_counts = filtered_df["Asset status"].value_counts()
# inactive_assets = status_counts.drop("Enabled", errors="ignore").sum()

# def plot_asset_status():
#     plt.figure()
#     plt.bar(status_counts.index, status_counts.values)
#     plt.xlabel("Asset Status")
#     plt.ylabel("Count")
#     plt.title("Asset Status Distribution")
#     st.pyplot(plt)

# graph_with_summary(
#     plot_asset_status,
#     f"""
#     **Insight:**  
#     **{inactive_assets} assets** are inactive or non-serving.

#     **Recommendation:**  
#     Regular cleanup of unused assets improves account clarity and control.
#     """,
#     "4️⃣",
#     "Asset Status Distribution"
# )

# # ─────────────────────────────
# # 5️⃣ Pareto Analysis (80/20)
# # ─────────────────────────────
# pareto = filtered_df.sort_values("Interactions", ascending=False)
# pareto["Cumulative %"] = pareto["Interactions"].cumsum() / pareto["Interactions"].sum() * 100
# top_80 = pareto[pareto["Cumulative %"] <= 80]

# def plot_pareto():
#     plt.figure()
#     plt.plot(pareto["Cumulative %"].values)
#     plt.axhline(80)
#     plt.ylabel("Cumulative Interaction %")
#     plt.xlabel("Assets Ranked")
#     plt.title("Pareto Analysis of Asset Engagement")
#     st.pyplot(plt)

# graph_with_summary(
#     plot_pareto,
#     f"""
#     **Insight:**  
#     **{len(top_80)} out of {len(filtered_df)} assets** generate  
#     **80% of total interactions**.

#     **Recommendation:**  
#     Focus budgets and creative strategy on high-impact assets.
#     """,
#     "5️⃣",
#     "Pareto Analysis (80/20 Rule)"
# )