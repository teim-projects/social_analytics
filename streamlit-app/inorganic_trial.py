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
# 4️⃣ SPEND vs REACH
# =========================================================
st.subheader("4️⃣ Spend vs Reach")

spend = get_col("Amount spent (INR)")
reach = get_col("Reach")

if spend is None or reach is None:
    st.error(f"Column missing → spend: {spend}, reach: {reach}")
    st.stop()

fig4 = px.scatter(df_ads, x=spend, y=reach,
                  labels={spend:"Amount Spent (INR)", reach:"Reach"})

corr = df_ads[spend].corr(df_ads[reach])

summary4 = """
📊 BUSINESS SUMMARY
"""
summary4 += "\nBudget directly expanding audience reach → scaling possible." if corr > 0.7 \
    else "\nSpend saturation observed → targeting refinement needed."

plot_with_summary(fig4, summary4)

# =========================================================
# 5️⃣ LP EFFICIENCY
# =========================================================
st.subheader("5️⃣ Landing Page Efficiency")

lp = get_col("Landing page views")
clicks = get_col("Link clicks")

df_ads["LP Efficiency"] = df_ads[lp] / (df_ads[clicks] + 1)

fig5 = px.histogram(df_ads, x="LP Efficiency",
                    labels={"LP Efficiency":"Landing Page Efficiency"})

avg_eff = df_ads["LP Efficiency"].mean()

summary5 = """
📊 BUSINESS SUMMARY
"""
summary5 += "\nMany users not reaching landing page → slow load / poor UX." if avg_eff < 0.5 \
    else "\nGood click-to-landing conversion."

plot_with_summary(fig5, summary5)

# =========================================================
# 6️⃣ CAMPAIGN AGE vs CTR
# =========================================================
st.subheader("6️⃣ Campaign Age vs CTR")

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

# =========================================================
# 7️⃣ DIMINISHING RETURNS
# =========================================================
st.subheader("7️⃣ Diminishing Returns Curve")

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
# 8️⃣ AUDIENCE QUALITY
# =========================================================
st.subheader("8️⃣ Audience Quality Quadrant")

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
# 9️⃣ FREQUENCY BAND
# =========================================================
st.subheader("9️⃣ CTR by Frequency Band")

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
# 🔟 BUDGET DISTRIBUTION
# =========================================================
st.subheader("🔟 Budget Distribution")

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

st.subheader("11️⃣ Campaign Age vs Clicks")

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

st.subheader("12️⃣ Reach vs Impressions")

fig12 = px.scatter(df_ads, x="Reach", y="Impressions")

corr = df_ads["Reach"].corr(df_ads["Impressions"])

summary12 = f"""
📊 **BUSINESS SUMMARY**

{"Impressions mainly expanding new audience." if corr > 0.9 else
 "High repeat exposure → frequency saturation risk."}
"""

plot_with_summary(fig12, summary12)

st.subheader("13️⃣ Budget vs Actual Spend")

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

st.subheader("14️⃣ Spend vs Results")

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

st.subheader("15️⃣ Cost per Result Spread")

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

# =========================================================
# 1️⃣ CTR SPREAD
# =========================================================
st.subheader("1️⃣ CTR Performance Spread")

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
# 2️⃣ CPR ANALYSIS
# =========================================================
st.subheader("2️⃣ Cost per Result Analysis")

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
# 3️⃣ LANDING PAGE RATE
# =========================================================
st.subheader("3️⃣ Landing Page Conversion")

df_creative["LP Rate"] = df_creative["Landing page views"] / (df_creative["Link clicks"] + 1)

fig3 = px.histogram(df_creative, x="LP Rate")

avg_lp = df_creative["LP Rate"].mean()

summary3 = f"""
📊 **BUSINESS SUMMARY**

Average Landing Conversion Rate: {avg_lp:.2f}

{"👉 Many users drop before landing → landing page UX or intent mismatch." if avg_lp < 0.5 else
 "👉 Click traffic quality acceptable."}
"""

plot_with_summary(fig3, summary3)

# =========================================================
# 4️⃣ EXPOSURE
# =========================================================
st.subheader("4️⃣ Exposure Analysis")

df_creative["Exposure"] = df_creative["Impressions"] / (df_creative["Reach"] + 1)

fig4 = px.histogram(df_creative, x="Exposure")

max_exp = df_creative["Exposure"].max()

summary4 = f"""
📊 **BUSINESS SUMMARY**

Maximum exposure ratio: {max_exp:.2f}

{"👉 Same audience seeing ad repeatedly → fatigue risk." if max_exp > 2 else
 "👉 Audience expansion still happening."}
"""

plot_with_summary(fig4, summary4)

# =========================================================
# 5️⃣ SPEND SHARE
# =========================================================
st.subheader("5️⃣ Budget Share by Creative")

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
# 6️⃣ RESULT SHARE
# =========================================================
st.subheader("6️⃣ Result Contribution")

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

# =========================================================
# 7️⃣ CLICK EFFICIENCY
# =========================================================
st.subheader("7️⃣ Click Efficiency")

df_creative["Click Efficiency"] = df_creative["Link clicks"] / (df_creative["Amount spent (INR)"] + 1)

fig7 = px.histogram(df_creative, x="Click Efficiency")

avg_eff = df_creative["Click Efficiency"].mean()

summary7 = f"""
📊 **BUSINESS SUMMARY**

{"Overall click generation efficiency low → creative optimisation needed." if avg_eff < 0.02 else
 "Some creatives generating strong traffic → scaling candidates exist."}
"""

plot_with_summary(fig7, summary7)

# =========================================================
# 8️⃣ CLICK RATE
# =========================================================
st.subheader("8️⃣ Click Rate Funnel")

df_creative["Click Rate"] = df_creative["Link clicks"] / (df_creative["Impressions"] + 1)

fig8 = px.histogram(df_creative, x="Click Rate")

spread = df_creative["Click Rate"].max() - df_creative["Click Rate"].min()

summary8 = f"""
📊 **BUSINESS SUMMARY**

{"Huge difference in hook effectiveness across creatives." if spread > 0.01 else
 "Creatives performing similarly at impression stage."}
"""

plot_with_summary(fig8, summary8)

# =========================================================
# 9️⃣ SPEND RISK
# =========================================================
st.subheader("9️⃣ Spend Risk")

df_creative["Spend Share"] = df_creative["Amount spent (INR)"] / df_creative["Amount spent (INR)"].sum()

fig9 = px.histogram(df_creative, x="Spend Share")

risk_ratio = (df_creative["Spend Share"] > 0.3).mean()

summary9 = f"""
📊 **BUSINESS SUMMARY**

{"Budget concentrated in few creatives → high dependency risk." if risk_ratio > 0.3 else
 "Budget allocation diversified."}
"""

plot_with_summary(fig9, summary9)

# =========================================================
# 🔟 RESULT EFFICIENCY
# =========================================================
st.subheader("🔟 Result Efficiency")

df_creative["Result Efficiency"] = df_creative["Results"] / (df_creative["Amount spent (INR)"] + 1)

fig10 = px.histogram(df_creative, x="Result Efficiency")

top = df_creative["Result Efficiency"].max()
bottom = df_creative["Result Efficiency"].min()

summary10 = f"""
📊 **BUSINESS SUMMARY**

{"Major ROI gap → reallocate spend immediately." if top > 2 * bottom else
 "ROI fairly balanced across creatives."}
"""

plot_with_summary(fig10, summary10)

# ─────────────────────────────────────────────
# 1️⃣ ROI RANKING
# ─────────────────────────────────────────────
st.subheader("1️⃣ Campaign ROI Ranking")

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
# 2️⃣ ROI COMPARISON (SECOND GRAPH IN COLAB)
# ─────────────────────────────────────────────
st.subheader("2️⃣ Campaign ROI Comparison")

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

# ─────────────────────────────────────────────
# 3️⃣ COST PER RESULT
# ─────────────────────────────────────────────
st.subheader("3️⃣ Cost per Result by Campaign")

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
# 4️⃣ 🚨 TRAFFIC GENERATION EFFICIENCY (YOUR GRAPH)
# ─────────────────────────────────────────────
st.subheader("4️⃣ Traffic Generation Efficiency")

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

# ─────────────────────────────────────────────
# 5️⃣ EXPOSURE (FINAL GRAPH)
# ─────────────────────────────────────────────
st.subheader("5️⃣ Audience Exposure by Campaign")

df_campaign["Exposure"] = df_campaign["Impressions"] / (df_campaign["Reach"] + 1)

exp_sorted = df_campaign.sort_values("Exposure", ascending=False)

fig5 = px.bar(
    exp_sorted,
    x="Exposure",
    y="Campaign name",
    orientation="h"
)

max_exp = df_campaign["Exposure"].max()

summary5 = f"""
📊 **BUSINESS SUMMARY**

{"👉 Some campaigns over-exposing same audience → fatigue risk." if max_exp > 2 else
 "👉 Audience expansion still possible."}
"""

plot_with_summary(fig5, summary5)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import os

# # ────────────── PAGE CONFIG ──────────────
# st.set_page_config(page_title="Full Marketing Dashboard", layout="wide")
# st.title("📊 Full Marketing Analytics Dashboard")

# def plot_with_summary(fig, summary):
#     col1, col2 = st.columns([2,1])
#     with col1:
#         st.plotly_chart(fig, use_container_width=True)
#     with col2:
#         st.markdown(summary)

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(CURRENT_DIR, "data")

# # ────────────── HELPER FOR SAFE COLUMN ACCESS ──────────────
# def get_col(name):
#     for col in df_ads.columns:
#         if col.lower() == name.lower():
#             return col
#     return None

# # =========================================================
# # ================= ADS DASHBOARD (1–15) ===================
# # =========================================================

# st.header("📢 Ads Performance")

# df_ads = pd.read_csv(os.path.join(DATA_DIR,"Shree-Laxmi-Stone-Depot-Ad-sets-1-Jan-2025-1-Jan-2026.csv"))

# df_ads.columns = df_ads.columns.str.strip().str.replace("\n"," ")

# for col in df_ads.columns:
#     df_ads[col] = df_ads[col].astype(str).str.replace(",","").str.replace("₹","")

# df_ads = df_ads.apply(pd.to_numeric, errors="ignore")
# df_filtered = df_ads.copy()

# # define cols
# ctr = "CTR (link click-through rate)"
# cpc = "CPC (cost per link click) (INR)"
# freq = "Frequency"
# spend = "Amount spent (INR)"
# reach = "Reach"
# clicks = "Link clicks"
# cpm = "CPM (cost per 1,000 impressions) (INR)"

# # 1
# st.subheader("1️⃣ CTR Performance")
# ctr_perf = df_ads.groupby("Ad set name")[ctr].mean()
# fig1 = px.bar(x=ctr_perf.values, y=ctr_perf.index, orientation="h")
# best_set = ctr_perf.idxmax()
# worst_set = ctr_perf.idxmin()
# summary = f"""
# 📊 BUSINESS SUMMARY

# Best performing ad set: {best_set}
# Lowest performing ad set: {worst_set}
# """
# plot_with_summary(fig1, summary)

# # 2
# st.subheader("2️⃣ CPC vs CTR")
# fig2 = px.scatter(df_ads, x=cpc, y=ctr)
# corr = df_filtered[cpc].corr(df_filtered[ctr])
# summary = """
# 📊 BUSINESS SUMMARY
# """
# summary += "\nHigher CTR reducing CPC → strong creative optimisation opportunity." if corr < -0.3 \
#     else "\nWeak efficiency relationship → audience mismatch possible."
# plot_with_summary(fig2, summary)

# # 3
# st.subheader("3️⃣ Frequency vs CTR")
# fig3 = px.scatter(df_ads, x=freq, y=ctr)
# corr = df_filtered[freq].corr(df_filtered[ctr])

# summary = """
# 📊 BUSINESS SUMMARY
# """
# summary += "\nIncreasing frequency lowering CTR → ad fatigue risk." if corr < -0.3 \
#     else "\nNo strong fatigue signal yet."
# plot_with_summary(fig3, summary)

# # 4
# st.subheader("4️⃣ Spend vs Reach")
# fig4 = px.scatter(df_ads, x=spend, y=reach)
# spend = get_col("Amount spent (INR)")
# reach = get_col("Reach")
# corr = df_filtered[spend].corr(df_filtered[reach])

# summary4 = """
# 📊 BUSINESS SUMMARY
# """
# summary4 += "\nBudget directly expanding audience reach → scaling possible." if corr > 0.7 \
#     else "\nSpend saturation observed → targeting refinement needed."

# plot_with_summary(fig4, summary)

# # 5
# st.subheader("5️⃣ Landing Page Efficiency")
# df_ads["LP Efficiency"] = df_ads["Landing page views"]/(df_ads[clicks]+1)
# fig5 = px.histogram(df_ads, x="LP Efficiency")
# lp = get_col("Landing page views")
# clicks = get_col("Link clicks")

# df_filtered["LP Efficiency"] = df_filtered[lp] / (df_filtered[clicks] + 1)

# avg_eff = df_filtered["LP Efficiency"].mean()

# summary5 = """
# 📊 BUSINESS SUMMARY
# """
# summary5 += "\nMany users not reaching landing page → slow load / poor UX." if avg_eff < 0.5 \
#     else "\nGood click-to-landing conversion."


# plot_with_summary(fig5, summary)

# # 6
# st.subheader("6️⃣ Campaign Age vs CTR")
# df_ads["Starts"] = pd.to_datetime(df_ads["Starts"], errors="coerce")
# df_ads["Reporting starts"] = pd.to_datetime(df_ads["Reporting starts"], errors="coerce")
# df_ads["Age"] = (df_ads["Reporting starts"] - df_ads["Starts"]).dt.days
# fig6 = px.scatter(df_ads, x="Age", y=ctr)

# plot_with_summary(fig6, summary)

# # 7
# st.subheader("7️⃣ Diminishing Returns")
# fig7 = px.scatter(df_ads, x=spend, y=clicks)
# plot_with_summary(fig7, summary)

# # 8
# st.subheader("8️⃣ Audience Quality")
# fig8 = px.scatter(df_ads, x=cpm, y=ctr)
# plot_with_summary(fig8, summary)

# # 9
# st.subheader("9️⃣ Frequency Band")
# df_ads["Freq Band"] = pd.cut(df_ads[freq],[0,1,2,3,4,10])
# fig9 = px.bar(df_ads.groupby("Freq Band")[ctr].mean())
# plot_with_summary(fig9, summary)

# # 10
# st.subheader("🔟 Budget Distribution")
# fig10 = px.histogram(df_ads[spend]/df_ads[spend].sum())
# plot_with_summary(fig10, summary)

# # 11
# st.subheader("11️⃣ Age vs Clicks")
# fig11 = px.scatter(df_ads, x="Age", y=clicks)
# plot_with_summary(fig11, summary)

# # 12
# st.subheader("12️⃣ Reach vs Impressions")
# fig12 = px.scatter(df_ads, x="Reach", y="Impressions")
# plot_with_summary(fig12, summary)

# # 13
# st.subheader("13️⃣ Budget vs Spend")
# fig13 = px.scatter(df_ads, x="Ad set budget", y=spend)
# plot_with_summary(fig13, summary)

# # 14
# st.subheader("14️⃣ Spend vs Results")
# fig14 = px.scatter(df_ads, x=spend, y="Results")
# plot_with_summary(fig14, summary)

# # 15
# st.subheader("15️⃣ Cost per Result Spread")
# fig15 = px.box(df_ads, x="Cost per results")
# plot_with_summary(fig15, summary)

# # =========================================================
# # ================= CREATIVE (16–25) =======================
# # =========================================================

# st.header("🎨 Creative Performance")

# df_creative = pd.read_csv(os.path.join(DATA_DIR,"Shree-Laxmi-Stone-Depot-Ads-1-Jan-2025-1-Jan-2026.csv"))

# for col in df_creative.columns:
#     df_creative[col] = df_creative[col].astype(str).str.replace(",","").str.replace("₹","")

# df_creative = df_creative.apply(pd.to_numeric, errors="ignore")

# # 16
# st.subheader("16️⃣ CTR Spread")
# fig16 = px.bar(df_creative.sort_values(ctr,ascending=False).head(10),
#                x="Ad name", y=ctr)
# plot_with_summary(fig16, summary)

# # 17
# st.subheader("17️⃣ CPR")
# df_creative["CPR"] = df_creative[spend]/(df_creative["Results"]+1)
# fig17 = px.histogram(df_creative, x="CPR")
# plot_with_summary(fig17, summary)

# # 18
# st.subheader("18️⃣ Landing")
# df_creative["LP Rate"] = df_creative["Landing page views"]/(df_creative[clicks]+1)
# fig18 = px.histogram(df_creative, x="LP Rate")
# plot_with_summary(fig18, summary)

# # 19
# st.subheader("19️⃣ Exposure")
# df_creative["Exposure"] = df_creative["Impressions"]/(df_creative["Reach"]+1)
# fig19 = px.histogram(df_creative, x="Exposure")
# plot_with_summary(fig19, summary)

# # 20
# st.subheader("20️⃣ Spend Share")
# fig20 = px.bar(df_creative.groupby("Ad name")[spend].sum())
# plot_with_summary(fig20, summary)

# # 21
# st.subheader("21️⃣ Result Share")
# fig21 = px.bar(df_creative.groupby("Ad name")["Results"].sum())
# plot_with_summary(fig21, summary)

# # 22
# st.subheader("22️⃣ Click Efficiency")
# df_creative["Click Eff"] = df_creative[clicks]/(df_creative[spend]+1)
# fig22 = px.histogram(df_creative, x="Click Eff")
# plot_with_summary(fig22, summary)

# # 23
# st.subheader("23️⃣ Click Rate")
# df_creative["Click Rate"] = df_creative[clicks]/(df_creative["Impressions"]+1)
# fig23 = px.histogram(df_creative, x="Click Rate")
# plot_with_summary(fig23, summary)

# # 24
# st.subheader("24️⃣ Spend Risk")
# df_creative["Spend Share"] = df_creative[spend]/df_creative[spend].sum()
# fig24 = px.histogram(df_creative, x="Spend Share")
# plot_with_summary(fig24, summary)

# # 25
# st.subheader("25️⃣ Result Efficiency")
# df_creative["Result Eff"] = df_creative["Results"]/(df_creative[spend]+1)
# fig25 = px.histogram(df_creative, x="Result Eff")
# plot_with_summary(fig25, summary)

# # =========================================================
# # ================= CAMPAIGN (26–30) =======================
# # =========================================================

# st.header("📊 Campaign Performance")

# df_campaign = pd.read_csv(os.path.join(DATA_DIR,"Shree-Laxmi-Stone-Depot-Campaigns-1-Jan-2025-1-Jan-2026.csv"))

# for col in df_campaign.columns:
#     df_campaign[col] = df_campaign[col].astype(str).str.replace(",","").str.replace("₹","")

# df_campaign = df_campaign.apply(pd.to_numeric, errors="ignore")

# # 26
# st.subheader("26️⃣ ROI")
# df_campaign["ROI"] = df_campaign["Results"]/(df_campaign[spend]+1)
# fig26 = px.bar(df_campaign, x="ROI", y="Campaign name", orientation="h")
# plot_with_summary(fig26, summary)

# # 27
# st.subheader("27️⃣ ROI Comparison")
# fig27 = px.bar(df_campaign.sort_values("ROI",ascending=False),
#                x="ROI", y="Campaign name", orientation="h")
# plot_with_summary(fig27, summary)

# # 28
# st.subheader("28️⃣ CPR")
# df_campaign["CPR"] = df_campaign[spend]/(df_campaign["Results"]+1)
# fig28 = px.bar(df_campaign, x="CPR", y="Campaign name", orientation="h")
# plot_with_summary(fig28, summary)

# # 29
# st.subheader("29️⃣ Click Efficiency")
# df_campaign["Click Eff"] = df_campaign[clicks]/(df_campaign[spend]+1)
# fig29 = px.bar(df_campaign, x="Click Eff", y="Campaign name", orientation="h")
# plot_with_summary(fig29, summary)

# # 30
# st.subheader("30️⃣ Exposure")
# df_campaign["Exposure"] = df_campaign["Impressions"]/(df_campaign["Reach"]+1)
# fig30 = px.bar(df_campaign, x="Exposure", y="Campaign name", orientation="h")
# plot_with_summary(fig30, summary)