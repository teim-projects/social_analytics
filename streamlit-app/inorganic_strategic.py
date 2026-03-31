import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ────────────── PAGE CONFIG ──────────────
st.set_page_config(page_title="Ads Performance Dashboard", layout="wide")
st.title("📊 Ads Performance Analytics Dashboard")

# ────────────── LOAD ADS DATA ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")

DATA_PATH = os.path.join(DATA_DIR, "Shree-Laxmi-Stone-Depot-Ad-sets-1-Jan-2025-1-Jan-2026.csv")
df_ads = pd.read_csv(DATA_PATH)

# ────────────── CLEANING ──────────────
df_ads.columns = df_ads.columns.str.strip().str.replace("\n", " ")

for col in df_ads.select_dtypes(include="object").columns:
    df_ads[col] = df_ads[col].astype(str).str.strip()

# ────────────── HELPER FOR SAFE COLUMN ACCESS ──────────────
def get_col(name):
    for col in df_ads.columns:
        if col.lower() == name.lower():
            return col
    return None

# ────────────── NUMERIC CLEANING ──────────────
num_cols = [
    "Amount spent (INR)", "Reach", "Impressions", "Frequency",
    "Link clicks", "CTR (link click-through rate)",
    "CPC (cost per link click) (INR)",
    "CPM (cost per 1,000 impressions) (INR)",
    "Landing page views", "Results",
    "Cost per results", "Ad set budget"
]

for col in num_cols:
    real = get_col(col)
    if real:
        df_ads[real] = (
            df_ads[real].astype(str)
            .str.replace(",", "")
            .str.replace("₹", "")
        )
        df_ads[real] = pd.to_numeric(df_ads[real], errors="coerce").fillna(0)

col_adset = get_col("Ad set name")
col_delivery = get_col("Ad delivery")

# ────────────── LOAD CREATIVE DATA ──────────────
DATA_PATH = os.path.join(DATA_DIR, "Shree-Laxmi-Stone-Depot-Ads-1-Jan-2025-1-Jan-2026.csv")
df_creative = pd.read_csv(DATA_PATH)

# CLEAN COLUMN NAMES
df_creative.columns = df_creative.columns.str.strip()

# ────────────── NUMERIC CLEANING ──────────────
num_cols = [
    "Amount spent (INR)", "Impressions", "Reach", "Link clicks",
    "CTR (link click-through rate)", "CPC (cost per link click) (INR)",
    "CPM (cost per 1,000 impressions) (INR)", "Results",
    "Cost per results", "Landing page views", "Frequency"
]

for col in num_cols:
    if col in df_creative.columns:
        df_creative[col] = (
            df_creative[col].astype(str)
            .str.replace(",", "")
            .str.replace("₹", "")
        )
        df_creative[col] = pd.to_numeric(df_creative[col], errors="coerce").fillna(0)

# ────────────── LOAD CAMPAIGN DATA ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "Shree-Laxmi-Stone-Depot-Campaigns-1-Jan-2025-1-Jan-2026.csv")

df_campaign = pd.read_csv(DATA_PATH)

# CLEAN COLUMNS
df_campaign.columns = df_campaign.columns.str.strip()

# ────────────── NUMERIC CLEANING ──────────────
num_cols = [
    "Amount spent (INR)", "Impressions", "Reach", "Link clicks",
    "CTR (link click-through rate)",
    "CPC (cost per link click) (INR)",
    "CPM (cost per 1,000 impressions) (INR)",
    "Results", "Cost per results", "Landing page views", "Frequency"
]

for col in num_cols:
    if col in df_campaign.columns:
        df_campaign[col] = (
            df_campaign[col].astype(str)
            .str.replace(",", "")
            .str.replace("₹", "")
        )
        df_campaign[col] = pd.to_numeric(df_campaign[col], errors="coerce").fillna(0)

# LAYOUT FUNCTION
def plot_with_summary(fig, summary):
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(summary)

spend = get_col("Amount spent (INR)")
reach = get_col("Reach")
clicks = get_col("Link clicks")
ctr = get_col("CTR (link click-through rate)")
ctr_perf = df_ads.groupby(col_adset)[ctr].mean().sort_values()
freq = get_col("Frequency")

# =========================================================
# 1 DIMINISHING RETURNS
# =========================================================
st.subheader("1️⃣ Diminishing Returns Curve")

fig7 = px.scatter(df_ads, x=spend, y=clicks,
                  labels={spend:"Amount Spent", clicks:"Link Clicks"})

corr = df_ads[spend].corr(df_ads[clicks])

summary7 = """
📊 BUSINESS SUMMARY
"""
if corr > 0.75:
    summary7 += "\nBudget scaling still efficient."
elif corr < 0.4:
    summary7 += "\nStrong diminishing returns → optimise creatives before increasing spend."
else:
    summary7 += "\nModerate scaling potential."

plot_with_summary(fig7, summary7)

# =========================================================
# 2 AUDIENCE QUALITY
# =========================================================
st.subheader("2️⃣ Audience Quality Quadrant")

cpm = get_col("CPM (cost per 1,000 impressions) (INR)")

fig8 = px.scatter(df_ads, x=cpm, y=ctr,
                  labels={cpm:"CPM", ctr:"CTR"})

summary8 = """
📊 BUSINESS SUMMARY

Top-right quadrant → premium high intent audience.
Bottom-right → expensive low intent audience (avoid).
"""

plot_with_summary(fig8, summary8)

# =========================================================
# 3 BUDGET DISTRIBUTION
# =========================================================
st.subheader("3️⃣ Budget Distribution")

spend_share = df_ads[spend] / df_ads[spend].sum()

fig10 = px.histogram(spend_share, nbins=20,
                     labels={"value":"Spend Share"})

max_share = spend_share.max()

summary10 = """
📊 BUSINESS SUMMARY
"""
summary10 += "\nBudget heavily concentrated in few ad sets → risk of performance shock." if max_share > 0.4 \
    else "\nHealthy budget diversification."

plot_with_summary(fig10, summary10)

# 4 SPEND VS RESULTS
st.subheader("4️⃣ Spend vs Results")

fig14 = px.scatter(df_ads, x="Amount spent (INR)", y="Results")

spend = df_ads["Amount spent (INR)"]
results = df_ads["Results"]

corr = spend.corr(results)

cost_per_result_overall = spend.sum() / (results.sum() + 1)
low_roi_ratio = (results == 0).mean()

high_spend_threshold = spend.quantile(0.75)
high_spend_eff = results[spend >= high_spend_threshold].mean()
low_spend_eff = results[spend < high_spend_threshold].mean()

summary14 = f"""
📊 **BUSINESS SUMMARY**

Correlation between spend and results: {corr:.2f}  
Overall cost per result: ₹{cost_per_result_overall:.2f}

{"✅ Increasing budget strongly improves outcomes → safe to scale campaigns." if corr > 0.7 else
 "⚠ Budget increase not improving results → targeting / creative inefficiency." if corr < 0.3 else
 "➡ Moderate budget effectiveness."}

Ad sets generating zero results: {low_roi_ratio*100:.1f}%

{"👉 Large portion of spend may be wasted → pause or optimise weak ad sets." if low_roi_ratio > 0.3 else ""}

{"👉 Diminishing returns detected → optimise before scaling." if high_spend_eff < low_spend_eff else
 "👉 High spend ad sets still productive."}
"""

plot_with_summary(fig14, summary14)

# 5 BUDGET VS ACTUAL SPEND
st.subheader("5️⃣ Budget vs Actual Spend")

fig13 = px.scatter(df_ads, x="Ad set budget", y="Amount spent (INR)")

budget = df_ads["Ad set budget"]
spend = df_ads["Amount spent (INR)"]

mask = budget > 0
budget = budget[mask]
spend = spend[mask]

corr = budget.corr(spend)

under_spend_ratio = (spend < 0.7 * budget).mean()
over_spend_ratio = (spend > 1.1 * budget).mean()

summary13 = f"""
📊 **BUSINESS SUMMARY**

Correlation between planned budget and actual spend: {corr:.2f}

{"✅ Spend closely follows planned budget → stable delivery system." if corr > 0.7 else
 "⚠ Weak relation → budget planning not translating into delivery." if corr < 0.3 else
 "➡ Moderate alignment between budget and spend."}

Ad sets under-spending (<70% budget): {under_spend_ratio*100:.1f}%  
Ad sets over-spending (>110% budget): {over_spend_ratio*100:.1f}%

{"👉 Many ad sets unable to utilise budget → audience size too small / bid too low." if under_spend_ratio > 0.4 else ""}
{"👉 Some ad sets scaling beyond plan → high performing segments detected." if over_spend_ratio > 0.3 else ""}
"""

plot_with_summary(fig13, summary13)

# =========================================================
# 6 SPEND SHARE
# =========================================================
st.subheader("6️⃣ Budget Share by Creative")

spend_share = df_creative.groupby("Ad name")["Amount spent (INR)"].sum()
spend_share = spend_share / spend_share.sum()

fig5 = px.bar(x=spend_share.index, y=spend_share.values)

max_share = spend_share.max()

summary5 = f"""
📊 **BUSINESS SUMMARY**

{"👉 Budget heavily dependent on few creatives → performance risk." if max_share > 0.4 else
 "👉 Budget diversified across creatives."}
"""

plot_with_summary(fig5, summary5)

# =========================================================
# 7 RESULT SHARE
# =========================================================
st.subheader("7️⃣ Result Contribution")

result_share = df_creative.groupby("Ad name")["Results"].sum()
result_share = result_share / result_share.sum()

fig6 = px.bar(x=result_share.index, y=result_share.values)

top_result = result_share.max()

summary6 = f"""
📊 **BUSINESS SUMMARY**

{"👉 One creative driving majority outcomes → scale aggressively." if top_result > 0.5 else
 "👉 Multiple creatives contributing → stable portfolio."}
"""

plot_with_summary(fig6, summary6)

# ─────────────────────────────────────────────
# 8 ROI RANKING
# ─────────────────────────────────────────────
st.subheader("8️⃣ Campaign ROI Ranking")

df_campaign["ROI"] = df_campaign["Results"] / (df_campaign["Amount spent (INR)"] + 1)
roi_sorted = df_campaign.sort_values("ROI")

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
# 9 ROI COMPARISON (SECOND GRAPH IN COLAB)
# ─────────────────────────────────────────────
st.subheader("9️⃣ Campaign ROI Comparison")

roi_sorted_desc = df_campaign.sort_values("ROI", ascending=False)

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

