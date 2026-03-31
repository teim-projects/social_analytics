import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ────────────── PAGE CONFIG ──────────────
st.set_page_config(page_title="Full Analytics Dashboard", layout="wide")
st.title("📊 Content and Monetization Strategic Analytics Dashboard")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")

POSTS_PATH = os.path.join(DATA_DIR, "Jan-01-2025_Jan-01-2026_1444967610559863.csv")
MONET_PATH = os.path.join(DATA_DIR, "Jan-01-2025_Jan-01-2026_946170394605045.csv")

df_posts = pd.read_csv(POSTS_PATH)
df_monet = pd.read_csv(MONET_PATH)

# =========================================================
# COMMON HELPER
# =========================================================
def plot_with_summary(fig, summary):
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(summary)

# =========================================================
# ================= CONTENT DASHBOARD ======================
# =========================================================

df_posts["Publish time"] = pd.to_datetime(df_posts["Publish time"], errors="coerce")
df_posts = df_posts.dropna(subset=["Publish time"])

df_posts["Views"] = pd.to_numeric(df_posts["Views"], errors="coerce").fillna(0)
df_posts["Duration (sec)"] = pd.to_numeric(df_posts["Duration (sec)"], errors="coerce").fillna(0)
df_posts["Views from Organic posts"] = pd.to_numeric(df_posts["Views from Organic posts"], errors="coerce").fillna(0)
df_posts["Views from Boosted posts"] = pd.to_numeric(df_posts["Views from Boosted posts"], errors="coerce").fillna(0)
df_posts["Ad CPM (USD)"] = pd.to_numeric(df_posts["Ad CPM (USD)"], errors="coerce").fillna(0)

df_posts["Hour"] = df_posts["Publish time"].dt.hour

# FILTERS
st.sidebar.header("📌 Content Filters")

min_date = df_posts["Publish time"].min().date()
max_date = df_posts["Publish time"].max().date()

defaults_posts = {
    "from_date_p": min_date,
    "to_date_p": max_date,
    "view_range": (int(df_posts["Views"].min()), int(df_posts["Views"].max()))
}

for key, val in defaults_posts.items():
    if key not in st.session_state:
        st.session_state[key] = val

def clear_posts():
    for key, val in defaults_posts.items():
        st.session_state[key] = val

st.sidebar.button("🧹 Clear Content Filters", on_click=clear_posts)

col1, col2 = st.sidebar.columns(2)
col1.date_input("From", key="from_date_p")
col2.date_input("To", key="to_date_p")

st.sidebar.slider("Views Range",
    int(df_posts["Views"].min()),
    int(df_posts["Views"].max()),
    key="view_range"
)

df_posts_filtered = df_posts[
    (df_posts["Publish time"].dt.date >= st.session_state.from_date_p) &
    (df_posts["Publish time"].dt.date <= st.session_state.to_date_p) &
    (df_posts["Views"] >= st.session_state.view_range[0]) &
    (df_posts["Views"] <= st.session_state.view_range[1])
]

if df_posts_filtered.empty:
    st.warning("⚠️ No data for Content filters")
    st.stop()

date_col = None
for col in df_monet.columns:
    if "date" in col.lower() or "time" in col.lower():
        date_col = col
        break

df_monet[date_col] = pd.to_datetime(df_monet[date_col], errors="coerce")
df_monet = df_monet.dropna(subset=[date_col])
df_monet["Date"] = df_monet[date_col]

num_cols = df_monet.select_dtypes(include=np.number).columns
df_monet[num_cols] = df_monet[num_cols].fillna(0)

df_monet_filtered = df_monet.copy()

# 1️⃣ Monthly Trend
st.subheader("1️⃣ Monthly Views Trend")

monthly_views = df_posts_filtered.groupby(
    df_posts_filtered["Publish time"].dt.to_period("M")
)["Views"].sum()

fig1 = px.line(x=monthly_views.index.astype(str), y=monthly_views.values, markers=True)

growth = monthly_views.pct_change().mean() * 100
best_month = monthly_views.idxmax()
worst_month = monthly_views.idxmin()

summary1 = f"""
📊 **BUSINESS SUMMARY**

Average monthly growth rate: {growth:.2f}%  
Best performing month: {best_month}  
Lowest performing month: {worst_month}  

{'✅ Strong upward trend → scale ad spend and content volume.' if growth > 5 else
 '⚠ Declining trend → content fatigue or targeting issue.' if growth < -5 else
 '➡ Stable performance → experiment with new creatives.'}
"""
plot_with_summary(fig1, summary1)

# 2 Viral
st.subheader("2️⃣ Viral Content Analysis")

threshold = df_posts_filtered["Views"].quantile(0.9)
df_posts_filtered["Viral"] = df_posts_filtered["Views"] >= threshold

viral_ratio = df_posts_filtered["Viral"].mean()

fig6 = px.histogram(df_posts_filtered, x="Views")

summary6 = f"""
📊 **BUSINESS SUMMARY**

Viral content ratio: {viral_ratio*100:.2f}%  

{'👉 Low viral rate. Need experimentation.' if viral_ratio < 0.1 else
 '👉 Healthy viral pipeline.'}
"""
plot_with_summary(fig6, summary6)

# 3 Monthly Engagement
st.subheader("3️⃣ Monthly Engagement Trend")

monthly_eng = df_monet_filtered.groupby(df_monet_filtered["Date"].dt.to_period("M"))["Unique user engagements"].sum()

fig8 = px.line(x=monthly_eng.index.astype(str), y=monthly_eng.values, markers=True)

growth = monthly_eng.pct_change().mean()*100

summary8 = f"""
📊 **BUSINESS SUMMARY**

Growth rate: {growth:.2f}%
"""
plot_with_summary(fig8, summary8)

# 4 Engagement Variability
st.subheader("4️⃣ Engagement Variability")

fig9 = px.histogram(df_monet_filtered, x="Unique user engagements")

cv = monthly_eng.std()/(monthly_eng.mean()+1)

summary9 = f"""
📊 **BUSINESS SUMMARY**

Variability score: {cv:.2f}
"""
plot_with_summary(fig9, summary9)

# 5 Best Month
st.subheader("5️⃣ Best Performing Month")

fig10 = px.bar(x=monthly_eng.index.astype(str), y=monthly_eng.values)

summary10 = f"""
📊 **BUSINESS SUMMARY**

Best month: {monthly_eng.idxmax()}
"""
plot_with_summary(fig10, summary10)

# 6 Engagement Momentum
st.subheader("6️⃣ Engagement Momentum")

fig11 = px.line(x=monthly_eng.index.astype(str), y=monthly_eng.diff().fillna(0), markers=True)

summary11 = f"""
📊 **BUSINESS SUMMARY**

Momentum trend analysed.
"""
plot_with_summary(fig11, summary11)

