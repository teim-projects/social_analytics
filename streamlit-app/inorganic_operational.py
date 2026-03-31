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
# 1 SPEND vs REACH
# =========================================================
st.subheader("1️⃣ Spend vs Reach")

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

# 2 REACH vs IMPRESSIONS
st.subheader("2️⃣ Reach vs Impressions")

fig12 = px.scatter(df_ads, x="Reach", y="Impressions")

corr = df_ads["Reach"].corr(df_ads["Impressions"])

summary12 = f"""
📊 **BUSINESS SUMMARY**

{"Impressions mainly expanding new audience." if corr > 0.9 else
 "High repeat exposure → frequency saturation risk."}
"""

plot_with_summary(fig12, summary12)

# ─────────────────────────────────────────────
# 3 EXPOSURE (FINAL GRAPH)
# ─────────────────────────────────────────────
st.subheader("3️⃣ Audience Exposure by Campaign")

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

# =========================================================
# 4 LP EFFICIENCY
# =========================================================
st.subheader("4️⃣ Landing Page Efficiency")

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
# 5 LANDING PAGE RATE
# =========================================================
st.subheader("5️⃣ Landing Page Conversion")

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
# 6 EXPOSURE
# =========================================================
st.subheader("6️⃣ Exposure Analysis")

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
