import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ────────────── PAGE CONFIG ──────────────
st.set_page_config(page_title="Creative Performance Dashboard", layout="wide")
st.title("📊 Creative Performance Analytics")

# ────────────── LOAD DATA ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
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

# ────────────── SIDEBAR FILTERS ──────────────
st.sidebar.header("📌 Filters")

ad_col = "Ad name" if "Ad name" in df_creative.columns else None

defaults = {
    "ad_filter": list(df_creative[ad_col].dropna().unique()) if ad_col else [],
    "spend_range": (
        int(df_creative["Amount spent (INR)"].min()),
        int(df_creative["Amount spent (INR)"].max())
    )
}

# SESSION INIT
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# CLEAR BUTTON
def clear_filters():
    for key in defaults:
        st.session_state[key] = defaults[key]

st.sidebar.button("🧹 Clear Filters", on_click=clear_filters)

# FILTERS
if ad_col:
    st.sidebar.multiselect(
        "Ad Name",
        options=list(df_creative[ad_col].dropna().unique()),
        key="ad_filter"
    )

st.sidebar.slider(
    "Amount Spent Range",
    int(df_creative["Amount spent (INR)"].min()),
    int(df_creative["Amount spent (INR)"].max()),
    key="spend_range"
)

# APPLY FILTERS
df_filtered = df_creative.copy()

if ad_col and st.session_state.ad_filter:
    df_filtered = df_filtered[df_filtered[ad_col].isin(st.session_state.ad_filter)]

df_filtered = df_filtered[
    (df_filtered["Amount spent (INR)"] >= st.session_state.spend_range[0]) &
    (df_filtered["Amount spent (INR)"] <= st.session_state.spend_range[1])
]

if df_filtered.empty:
    st.warning("⚠️ No data for selected filters")
    st.stop()

# LAYOUT HELPER
def plot_with_summary(fig, summary):
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(summary)

# =========================================================
# 1️⃣ CTR SPREAD
# =========================================================
st.subheader("1️⃣ CTR Performance Spread")

ctr_sorted = df_filtered.sort_values("CTR (link click-through rate)", ascending=False)

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

df_filtered["CPR Overall"] = df_filtered["Amount spent (INR)"] / (df_filtered["Results"] + 1)

fig2 = px.histogram(df_filtered, x="CPR Overall")

avg_cpr = df_filtered["CPR Overall"].mean()
high_waste_ratio = (df_filtered["CPR Overall"] > avg_cpr).mean()

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

df_filtered["LP Rate"] = df_filtered["Landing page views"] / (df_filtered["Link clicks"] + 1)

fig3 = px.histogram(df_filtered, x="LP Rate")

avg_lp = df_filtered["LP Rate"].mean()

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

df_filtered["Exposure"] = df_filtered["Impressions"] / (df_filtered["Reach"] + 1)

fig4 = px.histogram(df_filtered, x="Exposure")

max_exp = df_filtered["Exposure"].max()

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

spend_share = df_filtered.groupby("Ad name")["Amount spent (INR)"].sum()
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

result_share = df_filtered.groupby("Ad name")["Results"].sum()
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

df_filtered["Click Efficiency"] = df_filtered["Link clicks"] / (df_filtered["Amount spent (INR)"] + 1)

fig7 = px.histogram(df_filtered, x="Click Efficiency")

avg_eff = df_filtered["Click Efficiency"].mean()

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

df_filtered["Click Rate"] = df_filtered["Link clicks"] / (df_filtered["Impressions"] + 1)

fig8 = px.histogram(df_filtered, x="Click Rate")

spread = df_filtered["Click Rate"].max() - df_filtered["Click Rate"].min()

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

df_filtered["Spend Share"] = df_filtered["Amount spent (INR)"] / df_filtered["Amount spent (INR)"].sum()

fig9 = px.histogram(df_filtered, x="Spend Share")

risk_ratio = (df_filtered["Spend Share"] > 0.3).mean()

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

df_filtered["Result Efficiency"] = df_filtered["Results"] / (df_filtered["Amount spent (INR)"] + 1)

fig10 = px.histogram(df_filtered, x="Result Efficiency")

top = df_filtered["Result Efficiency"].max()
bottom = df_filtered["Result Efficiency"].min()

summary10 = f"""
📊 **BUSINESS SUMMARY**

{"Major ROI gap → reallocate spend immediately." if top > 2 * bottom else
 "ROI fairly balanced across creatives."}
"""

plot_with_summary(fig10, summary10)