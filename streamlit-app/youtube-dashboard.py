# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from dotenv import load_dotenv
import os

st.set_page_config(page_title="YouTube Dashboard", page_icon="🎥", layout="wide")
st.title("🎥 YouTube Dashboard")

# ------------------ Load dataset ------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "ALL_VIDEO_DETAILS.csv")

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"❌ Could not load YouTube video dataset: {e}")
    st.stop()


if df is None:
    st.error(
        "No dataset found. Set DATA_PATH in your .env to your Excel/CSV, "
        "or replace the fallback_path in the script with your file path."
    )
    st.stop()

# st.success(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")

# ------------------ Basic cleaning & expected columns ------------------
# Normalize column names (strip whitespace)
df.columns = [c.strip() for c in df.columns]

# Ensure core columns exist
for col in ["Views", "Likes", "Comments_Count", "Published_At", "Duration", "Title", "Tags"]:
    if col not in df.columns:
        # create empty columns if missing to avoid code break; some graphs will then be skipped
        df[col] = np.nan

# Convert datatypes
df["Published_At"] = pd.to_datetime(df["Published_At"], errors="coerce")
df["Views"] = pd.to_numeric(df["Views"], errors="coerce")
df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce")
df["Comments_Count"] = pd.to_numeric(df["Comments_Count"], errors="coerce")

# Parse ISO-8601 Duration (e.g., PT1H2M3S)
def parse_duration_iso(duration):
    if pd.isna(duration): return np.nan
    s = str(duration)
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    m = pattern.match(s)
    if not m:
        # try if it's already seconds or minutes
        try:
            return float(s)
        except Exception:
            return np.nan
    h, mm, ss = m.groups()
    return int(h or 0) * 3600 + int(mm or 0) * 60 + int(ss or 0)

df["Duration_sec"] = df["Duration"].apply(parse_duration_iso)
df["Duration_min"] = df["Duration_sec"] / 60.0

# Title length and description length
df["Title"] = df["Title"].astype(str)
if "Description" in df.columns:
    df["Description"] = df["Description"].astype(str)
    df["Desc_Length"] = df["Description"].str.len()
else:
    df["Desc_Length"] = np.nan
df["Title_Length"] = df["Title"].str.len()

# Tag count (assume comma-separated)
def count_tags(tag_str):
    if pd.isna(tag_str): return 0
    s = str(tag_str).strip()
    if s in ["", "[]"]: return 0
    # if tags appear in a list-like string e.g. "['a','b']"
    s2 = s.strip("[]")
    if s2 == "": return 0
    # split by comma but handle quoted commas
    return max(1, s2.count(",") + 1) if s2 else 0

df["Tag_Count"] = df["Tags"].apply(count_tags)

# Like & Comment rates
df["Like_Rate"] = df["Likes"] / df["Views"]
df["Comment_Rate"] = df["Comments_Count"] / df["Views"]

# Filter out videos without views (avoid log issues)
df = df[df["Views"].notna() & (df["Views"] > 0)].copy()

# ------------------ Session state defaults + sidebar ------------------
all_graphs = [
    "Performance Quadrant (Views vs Like Rate)",
    "Views Distribution (log10)",
    "Views vs Likes (log–log)",
    "Views vs Comments (log–log)",
    "Monthly Views (Growth Over Time)",
    "Duration Distribution",
    "Duration vs Views (color = Like Rate)",
    "Duration Sweet Spot (median views by bucket)",
    "Upload Time Heatmap (Avg Views by day/hour)",
    "Average Views by Upload Hour",
    "Title Length vs Views",
    "Engagement Funnel (Like & Comment Rate)",
    "Hidden Gems (High Like Rate, Low Views)",
    "Tag Count vs Views"
]

# ────────────── Sidebar Filters with Clear Button ──────────────
col1, col2 = st.sidebar.columns([1, 1])  # Adjust width ratio as needed
col1.header("📌 Filters")
def clear_all_filters():
    for k, v in defaults.items():
        st.session_state[k] = v
    if "graph_selector" in st.session_state:
        del st.session_state["graph_selector"]
    if "graph_search" in st.session_state:
        del st.session_state["graph_search"]
    if "last_search_text" in st.session_state:
        del st.session_state["last_search_text"]

col2.button("🧹 Clear All Filters", on_click=clear_all_filters)

# extract date limits
min_date = df["Published_At"].min().date() if df["Published_At"].notna().any() else pd.Timestamp.today().date()
max_date = df["Published_At"].max().date() if df["Published_At"].notna().any() else pd.Timestamp.today().date()

defaults = {
    "from_date": min_date,
    "to_date": max_date,
    "selected_graphs": all_graphs.copy(),
    "search_text": ""
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# date inputs
colA, colB = st.sidebar.columns(2)
from_date = colA.date_input("From Date", value=st.session_state["from_date"], key="from_date_input")
to_date = colB.date_input("To Date", value=st.session_state["to_date"], key="to_date_input")
if from_date > to_date:
    st.sidebar.error("'From Date' cannot be after 'To Date'")
    st.stop()
st.session_state["from_date"] = from_date
st.session_state["to_date"] = to_date

# Tag filter (unique tags list might be big, show top tags)
top_tags = []
if "Tags" in df.columns:
    # flatten tags strings and take top N unique tokens
    tags_series = df["Tags"].dropna().astype(str)
    tokens = []
    for t in tags_series:
        # naive split by comma
        parts = [p.strip().strip("'\"") for p in re.split(r',|\|', t) if p.strip()]
        tokens.extend(parts)
    token_counts = pd.Series(tokens).value_counts()
    top_tags = token_counts.head(200).index.tolist()  # limit choices
selected_tags = st.sidebar.multiselect("Filter by Tag (optional)", options=top_tags, default=[])

# Graph search & multiselect
# st.sidebar.subheader("Graph Selection")
search_text = st.sidebar.text_input("Search graphs", value=st.session_state.get("search_text",""), key="graph_search")
filtered_graphs = [g for g in all_graphs if search_text.lower() in g.lower()] if search_text else all_graphs

if 'last_search_text' not in st.session_state or st.session_state.last_search_text != search_text:
    if 'graph_selector' in st.session_state:
        del st.session_state['graph_selector']
    st.session_state.selected_graphs = filtered_graphs.copy()

st.session_state.last_search_text = search_text

# selected_graphs = st.sidebar.multiselect(
#     "Select graphs to show",
#     options=filtered_graphs,
#     default=[g for g in st.session_state.get("selected_graphs", all_graphs) if g in filtered_graphs],
#     key="graph_selector"
# )
# st.session_state.selected_graphs = selected_graphs

# ------------------ Apply filters ------------------
df_filtered = df.copy()
df_filtered = df_filtered[
    (df_filtered["Published_At"].dt.date >= st.session_state["from_date"]) &
    (df_filtered["Published_At"].dt.date <= st.session_state["to_date"])
]

if selected_tags:
    # keep rows where any of the selected tags appear in the Tags string
    mask = df_filtered["Tags"].fillna("").apply(lambda s: any(t.lower() in s.lower() for t in selected_tags))
    df_filtered = df_filtered[mask]

if df_filtered.empty:
    st.warning("No data after applying filters.")
    st.stop()

# helper to render plot + summary
def plot_with_summary(fig, summary_md):
    c1, c2 = st.columns([2,1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(summary_md)

# ------------------ Graphs (14 from the two scripts) ------------------

# 1. Performance Quadrant (Views vs Like Rate)
if "Performance Quadrant (Views vs Like Rate)" in st.session_state.selected_graphs:
    st.subheader("1️⃣ Performance Quadrant — Views vs Like Rate")
    df_filtered["log_Views"] = np.log10(df_filtered["Views"])
    df_filtered["Like_Rate"] = df_filtered["Like_Rate"].fillna(0)
    views_med = df_filtered["log_Views"].median()
    like_med = df_filtered["Like_Rate"].median()
    def classify_q(r):
        if r["log_Views"] >= views_med and r["Like_Rate"] >= like_med:
            return "High Views & High Engagement"
        elif r["log_Views"] >= views_med:
            return "High Views & Low Engagement"
        elif r["Like_Rate"] >= like_med:
            return "Low Views & High Engagement"
        else:
            return "Low Views & Low Engagement"
    df_filtered["Perf_Quadrant"] = df_filtered.apply(classify_q, axis=1)
    fig = px.scatter(df_filtered, x="log_Views", y="Like_Rate", color="Perf_Quadrant",
                     hover_data=["Title","Views","Likes"], color_discrete_sequence=px.colors.qualitative.Bold)
    fig.add_vline(x=views_med)
    fig.add_hline(y=like_med)
    summary = f"Median(log10 Views)={views_med:.2f}, Median(Like Rate)={like_med:.4f}"
    plot_with_summary(fig, summary)

# 2. Views Distribution (log10)
if "Views Distribution (log10)" in st.session_state.selected_graphs:
    st.subheader("2️⃣ Views Distribution (log10)")
    fig = px.histogram(df_filtered, x=np.log10(df_filtered["Views"]), nbins=40, labels={"x":"log10(Views)"}, color_discrete_sequence=px.colors.sequential.Viridis)
    summary = f"Median views: {int(df_filtered['Views'].median())}"
    plot_with_summary(fig, summary)

# 3. Views vs Likes (log–log)
if "Views vs Likes (log–log)" in st.session_state.selected_graphs:
    st.subheader("3️⃣ Views vs Likes (log–log)")
    mask = (df_filtered["Likes"] > 0) & (df_filtered["Views"] > 0)
    if mask.any():
        fig = px.scatter(df_filtered[mask], x=np.log10(df_filtered[mask]["Likes"]), y=np.log10(df_filtered[mask]["Views"]),
                         hover_data=["Title","Views","Likes"], opacity=0.6, color_discrete_sequence=px.colors.qualitative.Bold)
        plot_with_summary(fig, "- Relationship between Likes & Views (log–log)")
    else:
        st.info("Not enough data (Likes>0 & Views>0) for this graph.")

# 4. Views vs Comments (log–log)
if "Views vs Comments (log–log)" in st.session_state.selected_graphs:
    st.subheader("4️⃣ Views vs Comments (log–log)")
    mask2 = (df_filtered["Comments_Count"] > 0) & (df_filtered["Views"] > 0)
    if mask2.any():
        fig = px.scatter(df_filtered[mask2], x=np.log10(df_filtered[mask2]["Comments_Count"]), y=np.log10(df_filtered[mask2]["Views"]),
                         hover_data=["Title","Views","Comments_Count"], opacity=0.6, color_discrete_sequence=px.colors.qualitative.Bold)
        plot_with_summary(fig, "- Relationship between Comments & Views (log–log)")
    else:
        st.info("Not enough data (Comments_Count>0 & Views>0) for this graph.")

# 5. Monthly Views (Growth Over Time)
if "Monthly Views (Growth Over Time)" in st.session_state.selected_graphs:
    st.subheader("5️⃣ Monthly Views (Growth Over Time)")
    if df_filtered["Published_At"].notna().any():
        monthly = df_filtered.set_index("Published_At").resample("M")["Views"].sum().reset_index()
        fig = px.line(monthly, x="Published_At", y="Views", title="Total Views per Month", color_discrete_sequence=px.colors.sequential.Viridis)
        plot_with_summary(fig, "- Sum of views each month")
    else:
        st.info("No Published_At timestamps to compute monthly trend.")

# 6. Duration Distribution
if "Duration Distribution" in st.session_state.selected_graphs:
    st.subheader("6️⃣ Duration Distribution (seconds)")
    if df_filtered["Duration_sec"].notna().any():
        fig = px.histogram(df_filtered, x="Duration_sec", nbins=40, title="Video Duration Distribution (seconds)", color_discrete_sequence=px.colors.sequential.Viridis)
        plot_with_summary(fig, "- How long your videos typically are")
    else:
        st.info("No Duration data available.")

# 7. Duration vs Views (color = Like Rate)
if "Duration vs Views (color = Like Rate)" in st.session_state.selected_graphs:
    st.subheader("7️⃣ Duration vs Views (color = Like Rate)")
    if df_filtered["Duration_sec"].notna().any():
        fig = px.scatter(df_filtered, x="Duration_sec", y="Views", color="Like_Rate", hover_data=["Title"], color_continuous_scale="Viridis")
        plot_with_summary(fig, "- Do longer videos get more views / better engagement?")
    else:
        st.info("No Duration data available.")

# 8. Duration Sweet Spot (median views by bucket)
if "Duration Sweet Spot (median views by bucket)" in st.session_state.selected_graphs:
    st.subheader("8️⃣ Duration Sweet Spot (median views by duration bucket)")
    if df_filtered["Duration_min"].notna().any():
        bins = [0, 5, 10, 15, 30, 60, np.inf]
        labels = ["0–5","5–10","10–15","15–30","30–60","60+"]
        df_filtered["Duration_Bucket"] = pd.cut(df_filtered["Duration_min"], bins=bins, labels=labels, right=False)
        duration_group = df_filtered.groupby("Duration_Bucket")["Views"].median().reset_index().dropna()
        fig = px.bar(duration_group, x="Duration_Bucket", y="Views", title="Median Views by Duration Bucket", color_discrete_sequence=px.colors.sequential.Viridis)
        plot_with_summary(fig, "- Shows median views by video length bucket")
    else:
        st.info("No Duration data available.")

# 9. Upload Time Heatmap (Avg Views by day/hour)
if "Upload Time Heatmap (Avg Views by day/hour)" in st.session_state.selected_graphs:
    st.subheader("9️⃣ Upload Time Heatmap — Average Views by Day & Hour")
    if df_filtered["Published_At"].notna().any():
        df_filtered["DayOfWeek"] = df_filtered["Published_At"].dt.dayofweek
        df_filtered["Hour"] = df_filtered["Published_At"].dt.hour
        dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
        df_filtered["DayName"] = df_filtered["DayOfWeek"].map(dow_map)
        heat = df_filtered.groupby(["DayName","Hour"])["Views"].mean().reset_index()
        fig = px.density_heatmap(heat, x="Hour", y="DayName", z="Views", nbinsx=24, title="Avg Views by Upload Time", color_continuous_scale="Viridis")
        plot_with_summary(fig, "- Best hour/day to upload (based on avg views)")
    else:
        st.info("No Published_At timestamps to compute heatmap.")

# 10. Average Views by Upload Hour
if "Average Views by Upload Hour" in st.session_state.selected_graphs:
    st.subheader("🔟 Average Views by Upload Hour")
    if df_filtered["Published_At"].notna().any():
        hourly = df_filtered.groupby(df_filtered["Published_At"].dt.hour)["Views"].mean().reset_index()
        hourly.columns = ["Hour","Views"]
        fig = px.bar(hourly, x="Hour", y="Views", title="Average Views by Upload Hour", color_discrete_sequence=px.colors.sequential.Viridis)
        plot_with_summary(fig, "- Which upload hour performs better on average")
    else:
        st.info("No Published_At timestamps to compute hourly summary.")

# 11. Title Length vs Views (with trendline)
if "Title Length vs Views" in st.session_state.selected_graphs:
    st.subheader("1️⃣1️⃣ Title Length vs Views (with trendline)")
    if df_filtered["Title_Length"].notna().any():
        fig = px.scatter(df_filtered, x="Title_Length", y="Views", hover_data=["Title"], trendline="ols", title="Title Length vs Views", color_discrete_sequence=px.colors.qualitative.Bold)
        plot_with_summary(fig, "- Is title length correlated with views?")
    else:
        st.info("No Title information available.")

# 12. Engagement Funnel (Like & Comment Rate)
if "Engagement Funnel (Like & Comment Rate)" in st.session_state.selected_graphs:
    st.subheader("1️⃣2️⃣ Engagement Funnel — Avg Like & Comment Rate")
    rates = pd.DataFrame({
        "Metric":["Like Rate","Comment Rate"],
        "Value":[df_filtered["Like_Rate"].mean(), df_filtered["Comment_Rate"].mean()]
    })
    fig = px.bar(rates, x="Metric", y="Value", title="Average Engagement Funnel", color_discrete_sequence=px.colors.sequential.Viridis)
    plot_with_summary(fig, "- Snapshot of average engagement rates")

# 13. Hidden Gems (High Like Rate, Low Views)
if "Hidden Gems (High Like Rate, Low Views)" in st.session_state.selected_graphs:
    st.subheader("1️⃣3️⃣ Hidden Gems (High Like Rate, Low Views)")
    views_25 = df_filtered["Views"].quantile(0.25)
    like_75 = df_filtered["Like_Rate"].quantile(0.75)
    hidden = df_filtered[(df_filtered["Views"] <= views_25) & (df_filtered["Like_Rate"] >= like_75)].sort_values("Like_Rate", ascending=False)
    if not hidden.empty:
        st.write(hidden[["Title","Views","Likes","Like_Rate"]].head(15))
        fig = px.scatter(hidden, x="Views", y="Like_Rate", hover_data=["Title"], title="Hidden Gems", color_discrete_sequence=px.colors.sequential.Reds)
        plot_with_summary(fig, "- Videos with great engagement but low reach")
    else:
        st.info("No hidden gem videos detected for selected filters.")

# 14. Tag Count vs Views
if "Tag Count vs Views" in st.session_state.selected_graphs:
    st.subheader("1️⃣4️⃣ Tag Count vs Views")
    if "Tag_Count" in df_filtered.columns and df_filtered["Tag_Count"].notna().any():
        fig = px.box(df_filtered, x="Tag_Count", y="Views", title="Views by Number of Tags", color_discrete_sequence=px.colors.sequential.Reds)
        plot_with_summary(fig, "- Do more tags help reach?")
    else:
        st.info("No Tags information available.")
