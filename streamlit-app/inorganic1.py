import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ────────────── PAGE CONFIG ──────────────
st.set_page_config(page_title="Ads Performance Dashboard", layout="wide")
st.title("📊 Ads Performance Analytics Dashboard")

# ────────────── LOAD DATA ──────────────
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

# ────────────── SIDEBAR FILTERS ──────────────
st.sidebar.header("📌 Filters")

col_adset = get_col("Ad set name")
col_delivery = get_col("Ad delivery")

# defaults
defaults = {
    "adset": df_ads[col_adset].unique().tolist() if col_adset else [],
    "delivery": df_ads[col_delivery].unique().tolist() if col_delivery else []
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def clear_filters():
    for k, v in defaults.items():
        st.session_state[k] = v

st.sidebar.button("🧹 Clear Filters", on_click=clear_filters)

if col_adset:
    st.session_state.adset = st.sidebar.multiselect(
        "Ad Set", df_ads[col_adset].unique(), default=st.session_state.adset
    )

if col_delivery:
    st.session_state.delivery = st.sidebar.multiselect(
        "Ad Delivery", df_ads[col_delivery].unique(), default=st.session_state.delivery
    )

# ────────────── APPLY FILTERS ──────────────
df_filtered = df_ads.copy()

if col_adset and len(st.session_state.adset) > 0:
    df_filtered = df_filtered[df_filtered[col_adset].isin(st.session_state.adset)]

if col_delivery and len(st.session_state.delivery) > 0:
    df_filtered = df_filtered[df_filtered[col_delivery].isin(st.session_state.delivery)]

if df_filtered.empty:
    st.warning("⚠️ No data for selected filters → showing full dataset")
    df_filtered = df_ads.copy()

# ────────────── HELPER ──────────────
def plot_with_summary(fig, summary):
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(summary)

# =========================================================
# 1️⃣ CTR PERFORMANCE
# =========================================================
st.subheader("1️⃣ CTR Performance by Ad Set")

ctr = get_col("CTR (link click-through rate)")
ctr_perf = df_filtered.groupby(col_adset)[ctr].mean().sort_values()

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

fig2 = px.scatter(df_filtered, x=cpc, y=ctr,
                  labels={cpc:"CPC (INR)", ctr:"CTR"})

corr = df_filtered[cpc].corr(df_filtered[ctr])

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

fig3 = px.scatter(df_filtered, x=freq, y=ctr,
                  labels={freq:"Frequency", ctr:"CTR"})

corr = df_filtered[freq].corr(df_filtered[ctr])

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

fig4 = px.scatter(df_filtered, x=spend, y=reach,
                  labels={spend:"Amount Spent (INR)", reach:"Reach"})

corr = df_filtered[spend].corr(df_filtered[reach])

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

df_filtered["LP Efficiency"] = df_filtered[lp] / (df_filtered[clicks] + 1)

fig5 = px.histogram(df_filtered, x="LP Efficiency",
                    labels={"LP Efficiency":"Landing Page Efficiency"})

avg_eff = df_filtered["LP Efficiency"].mean()

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

df_filtered["Starts"] = pd.to_datetime(df_filtered["Starts"], errors="coerce")
df_filtered["Reporting starts"] = pd.to_datetime(df_filtered["Reporting starts"], errors="coerce")

df_filtered["Campaign Age Days"] = (
    df_filtered["Reporting starts"] - df_filtered["Starts"]
).dt.days

fig6 = px.scatter(df_filtered, x="Campaign Age Days", y=ctr,
                  labels={"Campaign Age Days":"Campaign Age (Days)", ctr:"CTR"})

corr = df_filtered["Campaign Age Days"].corr(df_filtered[ctr])

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

fig7 = px.scatter(df_filtered, x=spend, y=clicks,
                  labels={spend:"Amount Spent", clicks:"Link Clicks"})

corr = df_filtered[spend].corr(df_filtered[clicks])

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

fig8 = px.scatter(df_filtered, x=cpm, y=ctr,
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

df_filtered["Freq Band"] = pd.cut(df_filtered[freq], bins=bins, labels=labels_band)

freq_perf = df_filtered.groupby("Freq Band")[ctr].mean()

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

spend_share = df_filtered[spend] / df_filtered[spend].sum()

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

if "Starts" in df_filtered.columns:
    df_filtered["Starts"] = pd.to_datetime(df_filtered["Starts"], errors="coerce")
    df_filtered["Campaign Age"] = (
        df_filtered["Reporting starts"] - df_filtered["Starts"]
    ).dt.days

    fig11 = px.scatter(df_filtered, x="Campaign Age", y="Link clicks")

    corr = df_filtered["Campaign Age"].corr(df_filtered["Link clicks"])

    summary11 = f"""
📊 **BUSINESS SUMMARY**

{"Performance improving with time → learning phase successful." if corr > 0.3 else
 "Older campaigns declining → creative fatigue." if corr < -0.3 else
 "Lifecycle effect weak."}
"""
    plot_with_summary(fig11, summary11)

st.subheader("12️⃣ Reach vs Impressions")

fig12 = px.scatter(df_filtered, x="Reach", y="Impressions")

corr = df_filtered["Reach"].corr(df_filtered["Impressions"])

summary12 = f"""
📊 **BUSINESS SUMMARY**

{"Impressions mainly expanding new audience." if corr > 0.9 else
 "High repeat exposure → frequency saturation risk."}
"""

plot_with_summary(fig12, summary12)

st.subheader("13️⃣ Budget vs Actual Spend")

fig13 = px.scatter(df_filtered, x="Ad set budget", y="Amount spent (INR)")

budget = df_filtered["Ad set budget"]
spend = df_filtered["Amount spent (INR)"]

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

fig14 = px.scatter(df_filtered, x="Amount spent (INR)", y="Results")

spend = df_filtered["Amount spent (INR)"]
results = df_filtered["Results"]

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

fig15 = px.box(df_filtered, x="Cost per results")

cpr = df_filtered["Cost per results"]

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

