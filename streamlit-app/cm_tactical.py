import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ────────────── PAGE CONFIG ──────────────
st.set_page_config(page_title="Full Analytics Dashboard", layout="wide")
st.title("📊 Content and Monetization Tactical Analytics Dashboard")

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

# 1 Organic Efficiency
st.subheader("1️⃣ Organic Efficiency")

df_posts_filtered["Organic Efficiency"] = df_posts_filtered["Views from Organic posts"]/(df_posts_filtered["Views"]+1)

fig3 = px.histogram(df_posts_filtered, x="Organic Efficiency")

avg_org = df_posts_filtered["Organic Efficiency"].mean()

if avg_org > 0.6:
    text = f"High organic efficiency ({avg_org:.2f}) → strong content-market fit."
elif avg_org < 0.3:
    text = f"Low organic efficiency ({avg_org:.2f}) → heavy reliance on boosting."
else:
    text = f"Moderate organic efficiency ({avg_org:.2f})."

summary3 = f"📊 **BUSINESS SUMMARY**\n\n{text}"
plot_with_summary(fig3, summary3)

# 2 Organic vs Paid
st.subheader("2️⃣ Organic vs Paid")

total_org = df_posts_filtered["Views from Organic posts"].sum()
total_paid = df_posts_filtered["Views from Boosted posts"].sum()

fig7 = px.pie(values=[total_org, total_paid], names=["Organic","Paid"])

org_dependency = total_org/(total_org+total_paid+1)

summary7 = f"""
📊 **BUSINESS SUMMARY**

Organic dependency score: {org_dependency:.2f}  

{'👉 Strong organic engine.' if org_dependency > 0.7 else
 '👉 Heavy paid reliance.' if org_dependency < 0.3 else
 '👉 Balanced strategy.'}
"""
plot_with_summary(fig7, summary7)