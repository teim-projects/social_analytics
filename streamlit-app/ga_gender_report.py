import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# ───────────────── Page Setup ─────────────────
st.set_page_config(
    page_title="Google Ads Demographic Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Google Ads Demographic Performance Dashboard")

load_dotenv()
DATA_PATH = os.getenv("GA_GENDER_REPORT_PATH")

# ───────────────── Load Data ─────────────────
try:
    df = pd.read_csv(DATA_PATH, skiprows=2)
except Exception as e:
    st.error(f"❌ Could not load Google Ads dataset: {e}")
    st.stop()

# ───────────────── Remove TOTAL Rows ─────────────────
df = df[~df["Gender"].astype(str).str.contains("total", case=False, na=False)]

# ───────────────── Data Cleaning ─────────────────
numeric_cols = [
    "Impr.", "Interactions", "Avg. cost", "Cost",
    "Conversions", "Cost / conv.", "Conv. rate"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Impr."].fillna(1, inplace=True)
df["Interactions"].fillna(0, inplace=True)
df["Avg. cost"].fillna(0, inplace=True)
df["Cost"].fillna(0, inplace=True)
df["Conversions"].fillna(0, inplace=True)
df["Cost / conv."].fillna(0, inplace=True)
df["Conv. rate"].fillna(0, inplace=True)

df["Interaction rate"] = (
    df["Interactions"] / df["Impr."]
).replace([float("inf"), -float("inf")], 0)

# ───────────────── Filter Options ─────────────────
campaigns = df["Campaign"].dropna().unique().tolist()
genders = df["Gender"].dropna().unique().tolist()
statuses = df["Status"].dropna().unique().tolist()

# ───────────────── Session State Defaults ─────────────────
if "selected_campaigns" not in st.session_state:
    st.session_state.selected_campaigns = campaigns

if "selected_genders" not in st.session_state:
    st.session_state.selected_genders = genders

if "selected_status" not in st.session_state:
    st.session_state.selected_status = statuses

# ───────────────── Sidebar Filters ─────────────────
with st.sidebar:
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.header("📌 Filters")

    with col_f2:
        if st.button("❌ Clear"):
            st.session_state.selected_campaigns = campaigns
            st.session_state.selected_genders = genders
            st.session_state.selected_status = statuses
            st.rerun()

    st.multiselect(
        "Campaign",
        campaigns,
        key="selected_campaigns"
    )

    st.multiselect(
        "Gender",
        genders,
        key="selected_genders"
    )

    st.multiselect(
        "Ad Status",
        statuses,
        key="selected_status"
    )

# ───────────────── Apply Filters ─────────────────
df_filtered = df[
    (df["Campaign"].isin(st.session_state.selected_campaigns)) &
    (df["Gender"].isin(st.session_state.selected_genders)) &
    (df["Status"].isin(st.session_state.selected_status))
]

if df_filtered.empty:
    st.warning("⚠️ No data available for selected filters")
    st.stop()

# ───────────────── Helper Function ─────────────────
def plot_with_summary(fig, summary):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(summary)

# ───────────────── 1️⃣ Impressions by Gender ─────────────────
st.subheader("1️⃣ Impressions Distribution by Gender")

imp_gender = (
    df_filtered.groupby("Gender")["Impr."]
    .sum()
    .sort_values(ascending=False)
)

fig = px.bar(
    imp_gender,
    x=imp_gender.index,
    y=imp_gender.values,
    color=imp_gender.index,
    title="Total Impressions by Gender"
)

summary = f"""
**Key Insight**
{imp_gender.to_frame().to_markdown()}

**Recommendation**
- Focus creatives on **{imp_gender.idxmax()}**
"""

plot_with_summary(fig, summary)

# ───────────────── 2️⃣ Conversions by Gender ─────────────────
st.subheader("2️⃣ Conversions by Gender")

conv_gender = (
    df_filtered.groupby("Gender")["Conversions"]
    .sum()
    .sort_values(ascending=False)
)

fig = px.bar(
    conv_gender,
    x=conv_gender.index,
    y=conv_gender.values,
    color=conv_gender.index,
    title="Conversions by Gender"
)

summary = f"""
**Performance**
{conv_gender.to_frame().to_markdown()}

**Optimization Tip**
- Scale **{conv_gender.idxmax()}**
"""

plot_with_summary(fig, summary)

# ───────────────── 3️⃣ Cost vs Conversions ─────────────────
st.subheader("3️⃣ Cost vs Conversions")

fig = px.scatter(
    df_filtered,
    x="Cost",
    y="Conversions",
    color="Campaign",
    size="Impr.",
    hover_data=["Gender", "Ad group"],
    title="Cost vs Conversions"
)

plot_with_summary(
    fig,
    "Bottom-right = efficient (low cost, high conversions ✅)"
)

# ───────────────── 4️⃣ Conversion Rate by Campaign ─────────────────
st.subheader("4️⃣ Conversion Rate by Campaign")

conv_rate_campaign = (
    df_filtered.groupby("Campaign")["Conv. rate"]
    .mean()
    .sort_values(ascending=False)
)

fig = px.bar(
    conv_rate_campaign,
    x=conv_rate_campaign.index,
    y=conv_rate_campaign.values,
    title="Average Conversion Rate by Campaign"
)

plot_with_summary(
    fig,
    f"Top campaign: **{conv_rate_campaign.idxmax()}**"
)

# ───────────────── 5️⃣ Cost per Conversion ─────────────────
st.subheader("5️⃣ Cost per Conversion (CPA)")

cpa_campaign = (
    df_filtered.groupby("Campaign")["Cost / conv."]
    .mean()
    .sort_values()
)

fig = px.bar(
    cpa_campaign,
    x=cpa_campaign.index,
    y=cpa_campaign.values,
    title="Average Cost per Conversion"
)

plot_with_summary(
    fig,
    f"Best CPA: **{cpa_campaign.idxmin()}**"
)

# ───────────────── 6️⃣ Interaction Rate vs Conversion Rate ─────────────────
st.subheader("6️⃣ Interaction Rate vs Conversion Rate")

fig = px.scatter(
    df_filtered,
    x="Interaction rate",
    y="Conv. rate",
    color="Gender",
    title="Interaction Rate vs Conversion Rate"
)

plot_with_summary(fig, "Quality traffic > volume")

# ───────────────── 7️⃣ Demographic Status Performance ─────────────────
st.subheader("7️⃣ Performance by Demographic Status")

demo_perf = df_filtered.groupby("Demographic status")["Conversions"].sum()

fig = px.bar(
    demo_perf,
    x=demo_perf.index,
    y=demo_perf.values,
    title="Conversions by Demographic Status"
)

plot_with_summary(fig, "Optimize bids by intent")

# ───────────────── 8️⃣ Campaign × Gender Heatmap ─────────────────
st.subheader("8️⃣ Campaign vs Gender Conversion Heatmap")

pivot = pd.pivot_table(
    df_filtered,
    values="Conversions",
    index="Campaign",
    columns="Gender",
    aggfunc="sum"
)

fig = px.imshow(
    pivot,
    text_auto=True,
    aspect="auto",
    title="Conversions Heatmap"
)

plot_with_summary(fig, "Gender-specific bid tuning")

# ----------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import os

# # ───────────────── Page Setup ─────────────────
# st.set_page_config(
#     page_title="Google Ads Demographic Analytics",
#     page_icon="📊",
#     layout="wide"
# )

# st.title("📊 Google Ads Demographic Performance Dashboard")

# # ───────────────── Load Data ─────────────────
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(CURRENT_DIR, "data")
# DATA_PATH = os.path.join(DATA_DIR, "gender_report.csv")

# try:
#     # Skip first 2 junk header rows
#     df = pd.read_csv(DATA_PATH, skiprows=2)
# except Exception as e:
#     st.error(f"❌ Could not load Google Ads dataset: {e}")
#     st.stop()

# # ───────────────── REMOVE TOTAL ROWS ─────────────────
# # Remove rows like "Total: Genders", "Total", etc.
# df = df[
#     ~df["Gender"].astype(str).str.contains("total", case=False, na=False)
# ]

# # ───────────────── Data Cleaning ─────────────────
# numeric_cols = [
#     "Impr.", "Interactions", "Avg. cost", "Cost",
#     "Conversions", "Cost / conv."
# ]

# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# df["Interaction rate"] = (
#     df["Interactions"] / df["Impr."]
# ).replace([float("inf"), -float("inf")], 0)

# df["Conv. rate"] = pd.to_numeric(df["Conv. rate"], errors="coerce")

# # ───────────────── Sidebar Filters ─────────────────
# st.sidebar.header("📌 Filters")

# campaigns = df["Campaign"].dropna().unique().tolist()
# genders = df["Gender"].dropna().unique().tolist()
# statuses = df["Status"].dropna().unique().tolist()

# selected_campaigns = st.sidebar.multiselect(
#     "Campaign",
#     campaigns,
#     default=campaigns
# )

# selected_genders = st.sidebar.multiselect(
#     "Gender",
#     genders,
#     default=genders
# )

# selected_status = st.sidebar.multiselect(
#     "Ad Status",
#     statuses,
#     default=statuses
# )

# df_filtered = df[
#     (df["Campaign"].isin(selected_campaigns)) &
#     (df["Gender"].isin(selected_genders)) &
#     (df["Status"].isin(selected_status))
# ]

# if df_filtered.empty:
#     st.warning("⚠️ No data available for selected filters")
#     st.stop()

# # ───────────────── Helper ─────────────────
# def plot_with_summary(fig, summary):
#     c1, c2 = st.columns([2, 1])
#     with c1:
#         st.plotly_chart(fig, use_container_width=True)
#     with c2:
#         st.markdown(summary)

# # ───────────────── 1️⃣ Impressions by Gender ─────────────────
# st.subheader("1️⃣ Impressions Distribution by Gender")

# imp_gender = df_filtered.groupby("Gender")["Impr."].sum().sort_values(ascending=False)

# fig = px.bar(
#     imp_gender,
#     x=imp_gender.index,
#     y=imp_gender.values,
#     color=imp_gender.index,
#     title="Total Impressions by Gender"
# )

# summary = f"""
# **Key Insight**
# {imp_gender.to_frame().to_markdown()}

# **Recommendation**
# - Focus creatives and bids on **{imp_gender.idxmax()}**
# - Test custom messaging for underperforming gender
# """

# plot_with_summary(fig, summary)

# # ───────────────── 2️⃣ Conversions by Gender ─────────────────
# st.subheader("2️⃣ Conversions by Gender")

# conv_gender = df_filtered.groupby("Gender")["Conversions"].sum().sort_values(ascending=False)

# fig = px.bar(
#     conv_gender,
#     x=conv_gender.index,
#     y=conv_gender.values,
#     color=conv_gender.index,
#     title="Conversions by Gender"
# )

# summary = f"""
# **Performance**
# {conv_gender.to_frame().to_markdown()}

# **Optimization Tip**
# - Allocate more budget to **{conv_gender.idxmax()}**
# - Reduce spend on low-conversion segments
# """

# plot_with_summary(fig, summary)

# # ───────────────── 3️⃣ Cost vs Conversions ─────────────────
# st.subheader("3️⃣ Cost vs Conversions (Efficiency View)")

# fig = px.scatter(
#     df_filtered,
#     x="Cost",
#     y="Conversions",
#     color="Campaign",
#     size="Impr.",
#     hover_data=["Gender", "Ad group"],
#     title="Cost vs Conversions"
# )

# summary = """
# **Interpretation**
# - Top-left = inefficient (high cost, low conversions ❌)
# - Bottom-right = efficient (low cost, high conversions ✅)
# """

# plot_with_summary(fig, summary)

# # ───────────────── 4️⃣ Conversion Rate by Campaign ─────────────────
# st.subheader("4️⃣ Conversion Rate by Campaign")

# conv_rate_campaign = (
#     df_filtered.groupby("Campaign")["Conv. rate"]
#     .mean()
#     .sort_values(ascending=False)
# )

# fig = px.bar(
#     conv_rate_campaign,
#     x=conv_rate_campaign.index,
#     y=conv_rate_campaign.values,
#     title="Average Conversion Rate by Campaign"
# )

# summary = f"""
# **Top Campaign**
# - {conv_rate_campaign.idxmax()} has highest conversion efficiency
# """

# plot_with_summary(fig, summary)

# # ───────────────── 5️⃣ Cost per Conversion ─────────────────
# st.subheader("5️⃣ Cost per Conversion (CPA Analysis)")

# cpa_campaign = (
#     df_filtered.groupby("Campaign")["Cost / conv."]
#     .mean()
#     .sort_values()
# )

# fig = px.bar(
#     cpa_campaign,
#     x=cpa_campaign.index,
#     y=cpa_campaign.values,
#     title="Average Cost per Conversion by Campaign"
# )

# summary = f"""
# **Best Efficiency**
# - Lowest CPA: **{cpa_campaign.idxmin()}**

# **Worst Performer**
# - Highest CPA: **{cpa_campaign.idxmax()}**
# """

# plot_with_summary(fig, summary)

# # ───────────────── 6️⃣ Interaction Rate vs Conversion Rate ─────────────────
# st.subheader("6️⃣ Interaction Rate vs Conversion Rate")

# fig = px.scatter(
#     df_filtered,
#     x="Interaction rate",
#     y="Conv. rate",
#     color="Gender",
#     title="Interaction Rate vs Conversion Rate"
# )

# plot_with_summary(fig, "Focus on quality traffic over volume.")

# # ───────────────── 7️⃣ Demographic Status Performance ─────────────────
# st.subheader("7️⃣ Performance by Demographic Status")

# demo_perf = df_filtered.groupby("Demographic status")["Conversions"].sum()

# fig = px.bar(
#     demo_perf,
#     x=demo_perf.index,
#     y=demo_perf.values,
#     title="Conversions by Demographic Status"
# )

# plot_with_summary(fig, "Optimize bids by demographic intent.")

# # ───────────────── 8️⃣ Campaign × Gender Heatmap ─────────────────
# st.subheader("8️⃣ Campaign vs Gender Conversion Heatmap")

# pivot = pd.pivot_table(
#     df_filtered,
#     values="Conversions",
#     index="Campaign",
#     columns="Gender",
#     aggfunc="sum"
# )

# fig = px.imshow(
#     pivot,
#     text_auto=True,
#     aspect="auto",
#     title="Conversions Heatmap (Campaign × Gender)"
# )

# plot_with_summary(fig, "Great for gender-specific bid tuning.")

# # ───────────────── 9️⃣ Summary ─────────────────
# st.subheader("📌 Smart Campaign Optimization Summary")

# st.success("Dashboard cleaned, totals removed, insights ready 🚀")
