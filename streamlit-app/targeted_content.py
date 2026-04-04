import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ────────────── Page Setup ──────────────
st.set_page_config(
    page_title="Google Ads Targeted Content Dashboard",
    page_icon="📈",
    layout="wide"
)
st.title("📈 Google Ads Performance EDA Dashboard")

# ────────────── Load Data ──────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(base_dir, "data", "Targeted_content_report.csv")

try:
    df = pd.read_csv(DATA_PATH, skiprows=2)
except Exception as e:
    st.error(f"❌ Could not load dataset: {e}")
    st.stop()

# ────────────── Data Cleaning (UPDATED) ──────────────

numeric_cols = [
    "Impr.", "Interactions", "Interaction rate",
    "Avg. cost", "Cost",
    "Conv. rate", "Conversions", "Cost / conv."
]

# Convert to numeric
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace NaN with 0 (safe for ads data)
df[numeric_cols] = df[numeric_cols].fillna(0)

# 🔥 Remove rows where ALL numeric metrics are zero
df = df.loc[~(df[numeric_cols].sum(axis=1) == 0)]

# Optional: reset index for clean display
df.reset_index(drop=True, inplace=True)

# ────────────── Sidebar Filters ──────────────
st.sidebar.header("📌 Filters")

campaigns = df["Campaign"].unique().tolist()
ad_groups = df["Ad group"].unique().tolist()
types = df["Type"].unique().tolist()

selected_campaigns = st.sidebar.multiselect(
    "Campaigns", campaigns, default=campaigns
)

selected_adgroups = st.sidebar.multiselect(
    "Ad Groups", ad_groups, default=ad_groups
)

selected_types = st.sidebar.multiselect(
    "Placement Type", types, default=types
)

df_f = df[
    df["Campaign"].isin(selected_campaigns) &
    df["Ad group"].isin(selected_adgroups) &
    df["Type"].isin(selected_types)
]

if df_f.empty:
    st.warning("⚠️ No data for selected filters")
    st.stop()

# ────────────── Helper ──────────────
def plot_with_summary(fig, summary):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(summary)

# ────────────── 1️⃣ Cost Distribution ──────────────
st.subheader("1️⃣ Cost Distribution")
fig = px.histogram(
    df_f,
    x="Cost",
    nbins=30,
    title="Spend Distribution",
    marginal="box"
)
summary = f"""
**Spend Summary**
- Total Spend: ₹{df_f['Cost'].sum():,.0f}
- Median Spend: ₹{df_f['Cost'].median():,.0f}
- Insight: Few placements consume most of the budget → budget concentration risk.
"""
plot_with_summary(fig, summary)

# ────────────── 2️⃣ Conversions Distribution ──────────────
st.subheader("2️⃣ Conversions Distribution")
fig = px.histogram(
    df_f,
    x="Conversions",
    nbins=30,
    title="Conversions Distribution"
)
summary = f"""
**Conversion Summary**
- Total Conversions: {int(df_f['Conversions'].sum())}
- Median Conversions: {df_f['Conversions'].median()}
- Insight: Majority of placements deliver very low conversions.
"""
plot_with_summary(fig, summary)

# ────────────── 3️⃣ Cost per Conversion ──────────────
st.subheader("3️⃣ Cost per Conversion (CPA)")
fig = px.box(
    df_f,
    y="Cost / conv.",
    title="Cost per Conversion Spread"
)
summary = f"""
**CPA Summary**
- Median CPA: ₹{df_f['Cost / conv.'].median():,.0f}
- High variance indicates optimization opportunities.
"""
plot_with_summary(fig, summary)

# ────────────── 4️⃣ Cost vs Conversions ──────────────
st.subheader("4️⃣ Cost vs Conversions")
fig = px.scatter(
    df_f,
    x="Cost",
    y="Conversions",
    color="Campaign",
    hover_data=["Placement"],
    title="Efficiency Map: Cost vs Conversions"
)
summary = """
**Efficiency Map**
- Top-left: Inefficient spend (high cost, low conversions)
- Bottom-right: Scale candidates (low cost, high conversions)
"""
plot_with_summary(fig, summary)

# ────────────── 5️⃣ Interaction Rate vs Conversion Rate ──────────────
st.subheader("5️⃣ Interaction Rate vs Conversion Rate")
fig = px.scatter(
    df_f,
    x="Interaction rate",
    y="Conv. rate",
    color="Type",
    title="Engagement vs Conversion Quality"
)
summary = """
**Engagement vs Conversion**
- High interaction ≠ high conversion
- Focus on placements with strong Conv. Rate
"""
plot_with_summary(fig, summary)

# ────────────── 6️⃣ Campaign Performance ──────────────
st.subheader("6️⃣ Campaign Performance Overview")

camp_perf = df_f.groupby("Campaign").agg({
    "Cost": "sum",
    "Conversions": "sum",
    "Cost / conv.": "mean"
}).sort_values("Conversions", ascending=False)

fig = px.bar(
    camp_perf,
    y=camp_perf.index,
    x="Conversions",
    orientation="h",
    title="Campaigns by Conversions"
)

summary = f"""
**Top Campaign:** {camp_perf.index[0]}
- Highest conversions
- Avg CPA: ₹{camp_perf.iloc[0]['Cost / conv.']:.0f}
"""
plot_with_summary(fig, summary)

# ────────────── 7️⃣ Waste Detection ──────────────
st.subheader("7️⃣ 🚨 High Spend – Low Conversion Placements")

waste = df_f[
    (df_f["Cost"] > df_f["Cost"].median()) &
    (df_f["Conversions"] == 0)
].sort_values("Cost", ascending=False)

st.dataframe(
    waste[["Placement", "Campaign", "Cost", "Interactions"]],
    use_container_width=True
)

st.markdown("""
**Recommendation**
- Pause or exclude these placements immediately
- They consume budget with zero return
""")

# ────────────── 8️⃣ Smart Recommendations ──────────────
st.subheader("8️⃣ 🧠 Smart Optimization Summary")

best = df_f[
    (df_f["Conversions"] > df_f["Conversions"].median()) &
    (df_f["Cost / conv."] < df_f["Cost / conv."].median())
]

worst = df_f[
    (df_f["Conversions"] == 0) &
    (df_f["Cost"] > df_f["Cost"].median())
]

st.markdown(f"""
### ✅ Scale These
- {best['Placement'].nunique()} high-efficiency placements
- Low CPA + strong conversion volume

### ❌ Pause / Optimize These
- {worst['Placement'].nunique()} waste placements
- High spend, zero conversions
""")

# ---------------------------------------------------------------------------------------------------
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import os

# # ────────────── Page Setup ──────────────
# st.set_page_config(
#     page_title="Google Ads Performance EDA Dashboard",
#     page_icon="📈",
#     layout="wide"
# )
# st.title("📈 Google Ads Performance EDA Dashboard")

# # ────────────── Load Data ──────────────
# base_dir = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(base_dir, "data", "Targeted_content_report.csv")

# try:
#     df = pd.read_csv(DATA_PATH, skiprows=2)
# except Exception as e:
#     st.error(f"❌ Could not load dataset: {e}")
#     st.stop()

# # ────────────── Basic Cleaning (SAFE FOR UI) ──────────────

# numeric_cols = [
#     "Impr.", "Interactions", "Interaction rate",
#     "Avg. cost", "Cost",
#     "Conv. rate", "Conversions", "Cost / conv."
# ]

# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# df[numeric_cols] = df[numeric_cols].fillna(0)

# # Remove rows where ALL metrics are zero (safe)
# df = df.loc[~(df[numeric_cols].sum(axis=1) == 0)]

# # ────────────── Sidebar Filters (FROM FULL DATA) ──────────────
# st.sidebar.header("📌 Filters")

# campaigns = sorted(df["Campaign"].dropna().unique().tolist())
# ad_groups = sorted(df["Ad group"].dropna().unique().tolist())
# types = sorted(df["Type"].dropna().unique().tolist())

# selected_campaigns = st.sidebar.multiselect(
#     "Campaigns", campaigns, default=campaigns
# )

# selected_adgroups = st.sidebar.multiselect(
#     "Ad Groups", ad_groups, default=ad_groups
# )

# selected_types = st.sidebar.multiselect(
#     "Placement Type", types, default=types
# )

# df_f = df[
#     df["Campaign"].isin(selected_campaigns) &
#     df["Ad group"].isin(selected_adgroups) &
#     df["Type"].isin(selected_types)
# ]

# if df_f.empty:
#     st.warning("⚠️ No data for selected filters")
#     st.stop()

# # ────────────── STRICT EDA CLEANING (AFTER FILTERS) ──────────────

# df_eda = df_f[
#     (df_f["Interaction rate"] > 0) &
#     (df_f["Conv. rate"] > 0)
# ]

# if df_eda.empty:
#     st.warning("⚠️ No non-zero engagement & conversion data after cleaning")
#     st.stop()

# # ────────────── Helper ──────────────
# def plot_with_summary(fig, summary):
#     c1, c2 = st.columns([2, 1])
#     with c1:
#         st.plotly_chart(fig, use_container_width=True)
#     with c2:
#         st.markdown(summary)

# # ────────────── 1️⃣ Cost Distribution ──────────────
# st.subheader("1️⃣ Cost Distribution")
# fig = px.histogram(
#     df_eda, x="Cost", nbins=30, title="Spend Distribution", marginal="box"
# )
# summary = f"""
# **Spend Summary**
# - Total Spend: ₹{df_eda['Cost'].sum():,.0f}
# - Median Spend: ₹{df_eda['Cost'].median():,.0f}
# """
# plot_with_summary(fig, summary)

# # ────────────── 2️⃣ Conversions Distribution ──────────────
# st.subheader("2️⃣ Conversions Distribution")
# fig = px.histogram(
#     df_eda, x="Conversions", nbins=30, title="Conversions Distribution"
# )
# summary = f"""
# **Total Conversions:** {int(df_eda['Conversions'].sum())}
# """
# plot_with_summary(fig, summary)

# # ────────────── 3️⃣ Cost per Conversion ──────────────
# st.subheader("3️⃣ Cost per Conversion (CPA)")
# fig = px.box(
#     df_eda, y="Cost / conv.", title="Cost per Conversion Spread"
# )
# summary = f"""
# **Median CPA:** ₹{df_eda['Cost / conv.'].median():,.0f}
# """
# plot_with_summary(fig, summary)

# # ────────────── 4️⃣ Cost vs Conversions ──────────────
# st.subheader("4️⃣ Cost vs Conversions")
# fig = px.scatter(
#     df_eda,
#     x="Cost",
#     y="Conversions",
#     color="Campaign",
#     hover_data=["Placement"],
#     title="Efficiency Map"
# )
# summary = """
# Top-right = scale | Bottom-left = optimize
# """
# plot_with_summary(fig, summary)

# # ────────────── 5️⃣ Interaction Rate vs Conversion Rate ──────────────
# st.subheader("5️⃣ Interaction Rate vs Conversion Rate")
# fig = px.scatter(
#     df_eda,
#     x="Interaction rate",
#     y="Conv. rate",
#     color="Type",
#     title="Engagement vs Conversion Quality"
# )
# summary = """
# Zero engagement & zero conversion rows removed.
# """
# plot_with_summary(fig, summary)

# # ────────────── 6️⃣ Campaign Performance ──────────────
# st.subheader("6️⃣ Campaign Performance Overview")

# camp_perf = df_eda.groupby("Campaign").agg({
#     "Cost": "sum",
#     "Conversions": "sum",
#     "Cost / conv.": "mean"
# }).sort_values("Conversions", ascending=False)

# fig = px.bar(
#     camp_perf,
#     y=camp_perf.index,
#     x="Conversions",
#     orientation="h",
#     title="Campaigns by Conversions"
# )
# plot_with_summary(fig, f"Top Campaign: {camp_perf.index[0]}")

# # ────────────── 7️⃣ Waste Detection ──────────────
# st.subheader("7️⃣ 🚨 High Spend – Zero Conversion Placements")

# waste = df_f[
#     (df_f["Cost"] > df_f["Cost"].median()) &
#     (df_f["Conversions"] == 0)
# ]

# st.dataframe(
#     waste[["Placement", "Campaign", "Cost", "Interactions"]],
#     use_container_width=True
# )

# # ────────────── 8️⃣ Smart Optimization Summary ──────────────
# st.subheader("8️⃣ 🧠 Smart Optimization Summary")

# best = df_eda[
#     (df_eda["Conversions"] > df_eda["Conversions"].median()) &
#     (df_eda["Cost / conv."] < df_eda["Cost / conv."].median())
# ]

# st.markdown(f"""
# ### ✅ Scale
# - {best['Placement'].nunique()} high-efficiency placements
# """)

# ---------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import os

# # ────────────── Page Setup ──────────────
# st.set_page_config(page_title="Google Ads Targeted Content Dashboard", page_icon="📈", layout="wide")
# st.title("📈 Google Ads Performance EDA Dashboard")

# # ────────────── Load Data ──────────────
# base_dir = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(base_dir, "data", "Targeted_content_report.csv")

# try:
#     df = pd.read_csv(DATA_PATH, skiprows=2)
# except Exception as e:
#     st.error(f"❌ Could not load dataset: {e}")
#     st.stop()

# # ────────────── Data Cleaning ──────────────
# num_cols = [
#     "Impr.", "Interactions", "Interaction rate",
#     "Avg. cost", "Cost",
#     "Conv. rate", "Conversions", "Cost / conv."
# ]

# for col in num_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")

# df.fillna(0, inplace=True)

# # ────────────── Sidebar Filters ──────────────
# st.sidebar.header("📌 Filters")

# campaigns = df["Campaign"].unique().tolist()
# ad_groups = df["Ad group"].unique().tolist()
# types = df["Type"].unique().tolist()

# selected_campaigns = st.sidebar.multiselect(
#     "Campaigns", campaigns, default=campaigns
# )

# selected_adgroups = st.sidebar.multiselect(
#     "Ad Groups", ad_groups, default=ad_groups
# )

# selected_types = st.sidebar.multiselect(
#     "Placement Type", types, default=types
# )

# df_f = df[
#     df["Campaign"].isin(selected_campaigns) &
#     df["Ad group"].isin(selected_adgroups) &
#     df["Type"].isin(selected_types)
# ]

# if df_f.empty:
#     st.warning("⚠️ No data for selected filters")
#     st.stop()

# # ────────────── Helper ──────────────
# def plot_with_summary(fig, summary):
#     c1, c2 = st.columns([2,1])
#     with c1:
#         st.plotly_chart(fig, use_container_width=True)
#     with c2:
#         st.markdown(summary)

# # ────────────── 1️⃣ Cost Distribution ──────────────
# st.subheader("1️⃣ Cost Distribution")
# fig = px.histogram(df_f, x="Cost", nbins=30, title="Spend Distribution", marginal="box")
# summary = f"""
# **Spend Summary**
# - Total Spend: ₹{df_f['Cost'].sum():,.0f}
# - Median Spend: ₹{df_f['Cost'].median():,.0f}
# - Insight: Few placements consume most of the budget → budget concentration risk.
# """
# plot_with_summary(fig, summary)

# # ────────────── 2️⃣ Conversions Distribution ──────────────
# st.subheader("2️⃣ Conversions Distribution")
# fig = px.histogram(df_f, x="Conversions", nbins=30, title="Conversions Distribution")
# summary = f"""
# **Conversion Summary**
# - Total Conversions: {int(df_f['Conversions'].sum())}
# - Median Conversions: {df_f['Conversions'].median()}
# - Insight: Majority of placements deliver very low conversions.
# """
# plot_with_summary(fig, summary)

# # ────────────── 3️⃣ Cost per Conversion ──────────────
# st.subheader("3️⃣ Cost per Conversion (CPA)")
# fig = px.box(df_f, y="Cost / conv.", title="Cost per Conversion Spread")
# summary = f"""
# **CPA Summary**
# - Median CPA: ₹{df_f['Cost / conv.'].median():,.0f}
# - High variance indicates optimization opportunities.
# """
# plot_with_summary(fig, summary)

# # ────────────── 4️⃣ Cost vs Conversions ──────────────
# st.subheader("4️⃣ Cost vs Conversions")
# fig = px.scatter(
#     df_f,
#     x="Cost",
#     y="Conversions",
#     color="Campaign",
#     hover_data=["Placement"],
#     title="Efficiency Map: Cost vs Conversions"
# )
# summary = """
# **Efficiency Map**
# - Top-left: Inefficient spend (high cost, low conversions)
# - Bottom-right: Scale candidates (low cost, high conversions)
# """
# plot_with_summary(fig, summary)

# # ────────────── 5️⃣ Interaction Rate vs Conversion Rate ──────────────
# st.subheader("5️⃣ Interaction Rate vs Conversion Rate")
# fig = px.scatter(
#     df_f,
#     x="Interaction rate",
#     y="Conv. rate",
#     color="Type",
#     title="Engagement vs Conversion Quality"
# )
# summary = """
# **Engagement vs Conversion**
# - High interaction ≠ high conversion
# - Focus on placements with strong Conv. Rate
# """
# plot_with_summary(fig, summary)

# # ────────────── 6️⃣ Campaign Performance ──────────────
# st.subheader("6️⃣ Campaign Performance Overview")
# camp_perf = df_f.groupby("Campaign").agg({
#     "Cost":"sum",
#     "Conversions":"sum",
#     "Cost / conv.":"mean"
# }).sort_values("Conversions", ascending=False)

# fig = px.bar(
#     camp_perf,
#     y=camp_perf.index,
#     x="Conversions",
#     orientation="h",
#     title="Campaigns by Conversions"
# )
# summary = f"""
# **Top Campaign:** {camp_perf.index[0]}
# - Highest conversions
# - Avg CPA: ₹{camp_perf.iloc[0]['Cost / conv.']:.0f}
# """
# plot_with_summary(fig, summary)

# # ────────────── 7️⃣ Waste Detection ──────────────
# st.subheader("7️⃣ 🚨 High Spend – Low Conversion Placements")

# waste = df_f[
#     (df_f["Cost"] > df_f["Cost"].median()) &
#     (df_f["Conversions"] == 0)
# ].sort_values("Cost", ascending=False)

# st.dataframe(waste[[
#     "Placement", "Campaign", "Cost", "Interactions"
# ]])

# st.markdown("""
# **Recommendation**
# - Pause or exclude these placements immediately
# - They consume budget with zero return
# """)

# # ────────────── 8️⃣ Smart Recommendations ──────────────
# st.subheader("8️⃣ 🧠 Smart Optimization Summary")

# best = df_f[
#     (df_f["Conversions"] > df_f["Conversions"].median()) &
#     (df_f["Cost / conv."] < df_f["Cost / conv."].median())
# ]

# worst = df_f[
#     (df_f["Conversions"] == 0) &
#     (df_f["Cost"] > df_f["Cost"].median())
# ]

# st.markdown(f"""
# ### ✅ Scale These
# - {best['Placement'].nunique()} high-efficiency placements
# - Low CPA + strong conversion volume

# ### ❌ Pause / Optimize These
# - {worst['Placement'].nunique()} waste placements
# - High spend, zero conversions
# """)
