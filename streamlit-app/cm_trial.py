import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ────────────── PAGE CONFIG ──────────────
st.set_page_config(page_title="Full Analytics Dashboard", layout="wide")
st.title("📊 Full Analytics Dashboard")

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
st.header("📊 Content Performance Analytics")

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

# 2️⃣ Duration vs Views
st.subheader("2️⃣ Duration vs Views")

fig2 = px.scatter(df_posts_filtered, x="Duration (sec)", y="Views")

corr = df_posts_filtered["Duration (sec)"].corr(df_posts_filtered["Views"])

if corr > 0.3:
    text = f"Positive correlation ({corr:.2f}) → longer videos driving more reach.\n👉 Strategy: Invest in storytelling."
elif corr < -0.3:
    text = f"Negative correlation ({corr:.2f}) → shorter videos performing better.\n👉 Strategy: Focus on reels."
else:
    text = f"Weak correlation ({corr:.2f}) → duration not primary driver.\n👉 Improve hooks."

summary2 = f"📊 **BUSINESS SUMMARY**\n\n{text}"
plot_with_summary(fig2, summary2)

# 3️⃣ Organic Efficiency
st.subheader("3️⃣ Organic Efficiency")

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

# 4️⃣ Posting Hour
st.subheader("4️⃣ Posting Hour Performance")

hour_perf = df_posts_filtered.groupby("Hour")["Views"].mean()

fig4 = px.line(x=hour_perf.index, y=hour_perf.values, markers=True)

best_hour = hour_perf.idxmax()
worst_hour = hour_perf.idxmin()

summary4 = f"""
📊 **BUSINESS SUMMARY**

Best posting hour: {best_hour}  
Worst posting hour: {worst_hour}  

👉 Optimize scheduling accordingly.
"""
plot_with_summary(fig4, summary4)

# 5️⃣ Duration Band
st.subheader("5️⃣ Duration Band Performance")

bins = [0,15,30,60,120,300,1000]
labels = ["0-15","15-30","30-60","60-120","120-300","300+"]

df_posts_filtered["Duration Band"] = pd.cut(df_posts_filtered["Duration (sec)"], bins=bins, labels=labels)

dur_perf = df_posts_filtered.groupby("Duration Band")["Views"].mean()

fig5 = px.bar(x=dur_perf.index.astype(str), y=dur_perf.values)

best_band = dur_perf.idxmax()

summary5 = f"""
📊 **BUSINESS SUMMARY**

Best performing duration band: {best_band}  

👉 Replicate this format.
"""
plot_with_summary(fig5, summary5)

# 6️⃣ Viral
st.subheader("6️⃣ Viral Content Analysis")

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

# 7️⃣ Organic vs Paid
st.subheader("7️⃣ Organic vs Paid")

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

# =========================================================
# ================= MONETIZATION DASHBOARD =================
# =========================================================
st.header("💰 Monetization Analytics")

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

# 8️⃣ Monthly Engagement
st.subheader("8️⃣ Monthly Engagement Trend")

monthly_eng = df_monet_filtered.groupby(df_monet_filtered["Date"].dt.to_period("M"))["Unique user engagements"].sum()

fig8 = px.line(x=monthly_eng.index.astype(str), y=monthly_eng.values, markers=True)

growth = monthly_eng.pct_change().mean()*100

summary8 = f"""
📊 **BUSINESS SUMMARY**

Growth rate: {growth:.2f}%
"""
plot_with_summary(fig8, summary8)

# 9️⃣ Variability
st.subheader("9️⃣ Engagement Variability")

fig9 = px.histogram(df_monet_filtered, x="Unique user engagements")

cv = monthly_eng.std()/(monthly_eng.mean()+1)

summary9 = f"""
📊 **BUSINESS SUMMARY**

Variability score: {cv:.2f}
"""
plot_with_summary(fig9, summary9)

# 🔟 Best Month
st.subheader("🔟 Best Performing Month")

fig10 = px.bar(x=monthly_eng.index.astype(str), y=monthly_eng.values)

summary10 = f"""
📊 **BUSINESS SUMMARY**

Best month: {monthly_eng.idxmax()}
"""
plot_with_summary(fig10, summary10)

# 1️⃣1️⃣ Momentum
st.subheader("1️⃣1️⃣ Engagement Momentum")

fig11 = px.line(x=monthly_eng.index.astype(str), y=monthly_eng.diff().fillna(0), markers=True)

summary11 = f"""
📊 **BUSINESS SUMMARY**

Momentum trend analysed.
"""
plot_with_summary(fig11, summary11)

# 1️⃣2️⃣ Day of Week
st.subheader("1️⃣2️⃣ Engagement by Day")

df_monet_filtered["Day"] = df_monet_filtered["Date"].dt.day_name()

dow_perf = df_monet_filtered.groupby("Day")["Unique user engagements"].mean()

fig12 = px.bar(x=dow_perf.index, y=dow_perf.values)

summary12 = f"""
📊 **BUSINESS SUMMARY**

Best day: {dow_perf.idxmax()}
"""
plot_with_summary(fig12, summary12)