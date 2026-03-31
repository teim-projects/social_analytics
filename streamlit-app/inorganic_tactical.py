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
# 1️⃣ CTR PERFORMANCE
# =========================================================
st.subheader("1️⃣ CTR Performance by Ad Set")

ctr = get_col("CTR (link click-through rate)")
ctr_perf = df_ads.groupby(col_adset)[ctr].mean().sort_values()

fig1 = px.bar(x=ctr_perf.values, y=ctr_perf.index, orientation="h",
              labels={"x":"CTR","y":"Ad Set Name"})

best_set = ctr_perf.idxmax()
worst_set = ctr_perf.idxmin()

summary1 = f"""
📊 BUSINESS SUMMARY

Best performing ad set: {best_set}
Lowest performing ad set: {worst_set}
"""

plot_with_summary(fig1, summary1)

# =========================================================
# 2️⃣ CPC vs CTR
# =========================================================
st.subheader("2️⃣ CPC vs CTR Efficiency")

cpc = get_col("CPC (cost per link click) (INR)")

fig2 = px.scatter(df_ads, x=cpc, y=ctr,
                  labels={cpc:"CPC (INR)", ctr:"CTR"})

corr = df_ads[cpc].corr(df_ads[ctr])

summary2 = """
📊 BUSINESS SUMMARY
"""
summary2 += "\nHigher CTR reducing CPC → strong creative optimisation opportunity." if corr < -0.3 \
    else "\nWeak efficiency relationship → audience mismatch possible."

plot_with_summary(fig2, summary2)

# =========================================================
# 3️⃣ FREQUENCY vs CTR
# =========================================================
st.subheader("3️⃣ Frequency vs CTR")

freq = get_col("Frequency")

fig3 = px.scatter(df_ads, x=freq, y=ctr,
                  labels={freq:"Frequency", ctr:"CTR"})

corr = df_ads[freq].corr(df_ads[ctr])

summary3 = """
📊 BUSINESS SUMMARY
"""
summary3 += "\nIncreasing frequency lowering CTR → ad fatigue risk." if corr < -0.3 \
    else "\nNo strong fatigue signal yet."

plot_with_summary(fig3, summary3)

# =========================================================
# 4 FREQUENCY BAND
# =========================================================
st.subheader("4️⃣ CTR by Frequency Band")

bins = [0,1,2,3,4,10]
labels_band = ["0-1","1-2","2-3","3-4","4+"]

df_ads["Freq Band"] = pd.cut(df_ads[freq], bins=bins, labels=labels_band)

freq_perf = df_ads.groupby("Freq Band")[ctr].mean()

fig9 = px.bar(x=freq_perf.index.astype(str), y=freq_perf.values,
              labels={"x":"Frequency Band","y":"CTR"})

best_band = freq_perf.idxmax()

summary9 = f"""
📊 BUSINESS SUMMARY

Optimal frequency band: {best_band}
"""

plot_with_summary(fig9, summary9)

# =========================================================
# 5 CTR SPREAD
# =========================================================
st.subheader("5️⃣ CTR Performance Spread")

ctr_sorted = df_creative.sort_values("CTR (link click-through rate)", ascending=False)

top_ctr = ctr_sorted.iloc[0]["CTR (link click-through rate)"]
low_ctr = ctr_sorted.iloc[-1]["CTR (link click-through rate)"]

fig1 = px.bar(ctr_sorted.head(10), x="Ad name", y="CTR (link click-through rate)")

spread = top_ctr - low_ctr

summary1 = f"""
📊 **BUSINESS SUMMARY**

Highest CTR: {top_ctr:.2f}  
Lowest CTR: {low_ctr:.2f}

{"👉 Huge creative performance gap → strong opportunity to scale top creatives and pause weak ones." if spread > 2 else
 "👉 Creatives performing similarly → targeting may be main driver." if spread < 0.5 else
 "👉 Moderate variation → selective optimisation recommended."}
"""

plot_with_summary(fig1, summary1)

# =========================================================
# 6 CPR ANALYSIS
# =========================================================
st.subheader("6️⃣ Cost per Result Analysis")

df_creative["CPR Overall"] = df_creative["Amount spent (INR)"] / (df_creative["Results"] + 1)

fig2 = px.histogram(df_creative, x="CPR Overall")

avg_cpr = df_creative["CPR Overall"].mean()
high_waste_ratio = (df_creative["CPR Overall"] > avg_cpr).mean()

summary2 = f"""
📊 **BUSINESS SUMMARY**

Average Cost per Result: ₹{avg_cpr:.2f}  
High cost creatives ratio: {high_waste_ratio*100:.1f}%

{"👉 Significant budget inefficiency → immediate reallocation needed." if high_waste_ratio > 0.4 else
 "👉 Budget utilisation relatively controlled."}
"""

plot_with_summary(fig2, summary2)

# =========================================================
# 7 CAMPAIGN AGE vs CTR
# =========================================================
st.subheader("7️⃣ Campaign Age vs CTR")

df_ads["Starts"] = pd.to_datetime(df_ads["Starts"], errors="coerce")
df_ads["Reporting starts"] = pd.to_datetime(df_ads["Reporting starts"], errors="coerce")

df_ads["Campaign Age Days"] = (
    df_ads["Reporting starts"] - df_ads["Starts"]
).dt.days

fig6 = px.scatter(df_ads, x="Campaign Age Days", y=ctr,
                  labels={"Campaign Age Days":"Campaign Age (Days)", ctr:"CTR"})

corr = df_ads["Campaign Age Days"].corr(df_ads[ctr])

summary6 = """
📊 BUSINESS SUMMARY
"""
if corr < -0.3:
    summary6 += "\nOlder campaigns losing effectiveness → creative refresh needed."
elif corr > 0.3:
    summary6 += "\nCampaign learning improving performance over time."
else:
    summary6 += "\nNo strong lifecycle effect."

plot_with_summary(fig6, summary6)

# 8 CAMPAIGN AGE vs CLICKS
st.subheader("8️⃣ Campaign Age vs Clicks")

if "Starts" in df_ads.columns:
    df_ads["Starts"] = pd.to_datetime(df_ads["Starts"], errors="coerce")
    df_ads["Campaign Age"] = (
        df_ads["Reporting starts"] - df_ads["Starts"]
    ).dt.days

    fig11 = px.scatter(df_ads, x="Campaign Age", y="Link clicks")

    corr = df_ads["Campaign Age"].corr(df_ads["Link clicks"])

    summary11 = f"""
📊 **BUSINESS SUMMARY**

{"Performance improving with time → learning phase successful." if corr > 0.3 else
 "Older campaigns declining → creative fatigue." if corr < -0.3 else
 "Lifecycle effect weak."}
"""
    plot_with_summary(fig11, summary11)

# 9 COST PER RESULT SPREAD
st.subheader("9️⃣ Cost per Result Spread")

fig15 = px.box(df_ads, x="Cost per results")

cpr = df_ads["Cost per results"]

median_cpr = cpr.median()
mean_cpr = cpr.mean()
std_cpr = cpr.std()

q1 = cpr.quantile(0.25)
q3 = cpr.quantile(0.75)
iqr = q3 - q1

high_outlier_ratio = (cpr > q3 + 1.5 * iqr).mean()
zero_ratio = (cpr == 0).mean()

summary15 = f"""
📊 **BUSINESS SUMMARY**

Average cost per result: ₹{mean_cpr:.2f}  
Median cost per result: ₹{median_cpr:.2f}  
Cost variability (std): ₹{std_cpr:.2f}

{"⚠ Very high variability → inconsistent campaign efficiency." if std_cpr > mean_cpr else
 "➡ Cost efficiency relatively stable."}

High-cost outlier ad sets: {high_outlier_ratio*100:.1f}%

{"👉 Significant budget leakage risk → optimise or pause expensive ad sets." if high_outlier_ratio > 0.25 else ""}

Ad sets with zero cost per result: {zero_ratio*100:.1f}%

{"👉 Many ad sets generating no measurable results → tracking / objective issue possible." if zero_ratio > 0.3 else ""}
"""

plot_with_summary(fig15, summary15)

# ─────────────────────────────────────────────
# 10 COST PER RESULT
# ─────────────────────────────────────────────
st.subheader("1️⃣0️⃣ Cost per Result by Campaign")

df_campaign["CPR"] = df_campaign["Amount spent (INR)"] / (df_campaign["Results"] + 1)
cpr_sorted = df_campaign.sort_values("CPR")

fig3 = px.bar(
    cpr_sorted,
    x="CPR",
    y="Campaign name",
    orientation="h"
)

avg_cpr = df_campaign["CPR"].mean()

summary3 = f"""
📊 **BUSINESS SUMMARY**

{"👉 Some campaigns highly inefficient → pause or optimise." if cpr_sorted["CPR"].iloc[-1] > avg_cpr*1.5 else
 "👉 Cost efficiency fairly stable."}
"""

plot_with_summary(fig3, summary3)

# ─────────────────────────────────────────────
# 11 🚨 TRAFFIC GENERATION EFFICIENCY (YOUR GRAPH)
# ─────────────────────────────────────────────
st.subheader("1️⃣1️⃣ Traffic Generation Efficiency")

df_campaign["Click Efficiency"] = df_campaign["Link clicks"] / (df_campaign["Amount spent (INR)"] + 1)

click_sorted = df_campaign.sort_values("Click Efficiency", ascending=False)

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

