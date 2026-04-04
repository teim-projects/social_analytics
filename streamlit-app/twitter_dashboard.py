import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import ast
import os

# ───────────────────────── Page Setup ─────────────────────────
st.set_page_config(
    page_title="Twitter Analytics Dashboard",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Twitter Engagement Analytics Dashboard")

# ───────────────────────── Load Data ─────────────────────────
@st.cache_data(show_spinner=False)
def load_data(path):
    df = pd.read_csv(path)

    # Normalize columns
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Datetime handling
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["date"] = df["created_at"].dt.date
    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.day_name()

    # Rename metrics
    df.rename(columns={
        'public_metrics_like_count': 'likes',
        'public_metrics_retweet_count': 'retweets',
        'public_metrics_reply_count': 'replies',
        'public_metrics_quote_count': 'quotes',
        'public_metrics_bookmark_count': 'bookmarks',
        'public_metrics_impression_count': 'impressions'
    }, inplace=True)

    # KPIs
    df["total_engagement"] = (
        df["likes"] + df["retweets"] +
        df["replies"] + df["quotes"] +
        df["bookmarks"]
    )

    df["engagement_rate"] = (
        df["total_engagement"] / df["impressions"]
    ) * 100

    return df

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CURRENT_DIR, "data", "twitter_data.csv")

if "df" not in st.session_state:
    st.session_state.df = load_data(DATA_PATH)

df = st.session_state.df

# ───────────────────────── Defaults & Session State ─────────────────────────
metrics = [
    "likes", "retweets", "replies",
    "quotes", "bookmarks",
    "impressions", "total_engagement", "engagement_rate"
]

weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

defaults = {
    "from_date": df["date"].min(),
    "to_date": df["date"].max(),
    "weekday_filter": weekdays,
    "hour_range": (0, 23),
    "min_engagement": 0,
    "metric_selected": "total_engagement"
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ───────────────────────── Clear Filters ─────────────────────────
def clear_all_filters():
    for k, v in defaults.items():
        st.session_state[k] = v

# ───────────────────────── Sidebar Filters ─────────────────────────
col1, col2 = st.sidebar.columns([3,1])
col1.header("📌 Filters")
col2.button("🧹 Clear", on_click=clear_all_filters)

# Date filter (Instagram-style)
d1, d2 = st.sidebar.columns(2)
from_date = d1.date_input("From", st.session_state.from_date)
to_date = d2.date_input("To", st.session_state.to_date)

if from_date > to_date:
    st.sidebar.error("❌ Invalid date range")
    st.stop()

st.session_state.from_date = from_date
st.session_state.to_date = to_date

# Weekday filter
weekday_filter = st.sidebar.multiselect(
    "Weekdays",
    weekdays,
    default=st.session_state.weekday_filter
)
st.session_state.weekday_filter = weekday_filter

# Hour range filter
hour_range = st.sidebar.slider(
    "Posting Hour Range",
    0, 23,
    st.session_state.hour_range
)
st.session_state.hour_range = hour_range

# Engagement threshold
min_eng = st.sidebar.slider(
    "Minimum Total Engagement",
    0,
    int(df["total_engagement"].max()),
    st.session_state.min_engagement
)
st.session_state.min_engagement = min_eng

# ───────────────────────── Apply Filters (SAFE LAYERING) ─────────────────────────
base_df = df[
    (df["date"] >= st.session_state.from_date) &
    (df["date"] <= st.session_state.to_date)
]

time_df = base_df[
    (base_df["weekday"].isin(st.session_state.weekday_filter)) &
    (base_df["hour"] >= st.session_state.hour_range[0]) &
    (base_df["hour"] <= st.session_state.hour_range[1])
]

analysis_df = time_df[
    time_df["total_engagement"] >= st.session_state.min_engagement
]

if analysis_df.empty:
    st.warning("⚠️ No data after filters")
    st.stop()

# ───────────────────────── Helper: Chart + Summary ─────────────────────────
def chart_with_summary(fig, summary):
    c1, c2 = st.columns([3,1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### 🔍 Insight")
        st.markdown(summary)

# ───────────────────────── KPIs ─────────────────────────
st.subheader("📊 Key Metrics")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Tweets", len(analysis_df))
k2.metric("Avg Engagement", round(analysis_df["total_engagement"].mean(), 2))
k3.metric("Avg Engagement Rate (%)", round(analysis_df["engagement_rate"].mean(), 2))
k4.metric(
    "Best Posting Hour",
    analysis_df.groupby("hour")["total_engagement"].mean().idxmax()
)

st.divider()

# ───────────────────────── Metric Distribution ─────────────────────────
metric_selected = st.selectbox(
    "Select Metric",
    metrics,
    index=metrics.index(st.session_state.metric_selected)
)
st.session_state.metric_selected = metric_selected

fig = px.histogram(
    analysis_df,
    x=metric_selected,
    nbins=30,
    title=f"Distribution of {metric_selected}"
)

chart_with_summary(
    fig,
    "Shows spread, skewness and abnormal spikes in the selected metric."
)

# ───────────────────────── Boxplot ─────────────────────────
fig = px.box(
    analysis_df[metrics],
    title="Outlier Detection Across Metrics"
)

chart_with_summary(
    fig,
    "Highlights extreme values and variability across engagement metrics."
)

# ───────────────────────── Correlation Heatmap ─────────────────────────
corr = analysis_df[metrics].corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    title="Correlation Heatmap"
)

chart_with_summary(
    fig,
    "Retweets and likes show strongest correlation with total engagement."
)

# ───────────────────────── Scatter Relationships ─────────────────────────
fig = px.scatter(
    analysis_df,
    x="likes",
    y="retweets",
    title="Likes vs Retweets"
)

chart_with_summary(
    fig,
    "Strong relationship indicates content resonance."
)

fig = px.scatter(
    analysis_df,
    x="impressions",
    y="total_engagement",
    title="Impressions vs Engagement"
)

chart_with_summary(
    fig,
    "High impressions help, but engagement quality still matters."
)

# ───────────────────────── Time Series (DATE ONLY) ─────────────────────────
daily_eng = base_df.groupby("date")["total_engagement"].mean().reset_index()

fig = px.line(
    daily_eng,
    x="date",
    y="total_engagement",
    title="Average Daily Engagement"
)

chart_with_summary(
    fig,
    "Reveals campaign-level engagement trends over time."
)

# ───────────────────────── Hour & Weekday Analysis ─────────────────────────
hourly = analysis_df.groupby("hour")["total_engagement"].mean().reset_index()

fig = px.line(
    hourly,
    x="hour",
    y="total_engagement",
    title="Engagement by Posting Hour"
)

chart_with_summary(
    fig,
    "Identifies best posting hours."
)

weekday_eng = (
    analysis_df.groupby("weekday")["total_engagement"]
    .mean()
    .reindex(weekdays)
    .reset_index()
)

fig = px.bar(
    weekday_eng,
    x="weekday",
    y="total_engagement",
    title="Engagement by Weekday"
)

chart_with_summary(
    fig,
    "Shows which days perform best for posting."
)

# ───────────────────────── Top Tweets ─────────────────────────
st.subheader("🔥 Top 10 Tweets by Engagement")

st.dataframe(
    analysis_df.sort_values("total_engagement", ascending=False)
    [["text","likes","retweets","replies","impressions"]]
    .head(10),
    use_container_width=True
)

# ───────────────────────── Hashtag Analysis ─────────────────────────
def extract_hashtags(val):
    if pd.isna(val) or val in ["", "[]"]:
        return []
    try:
        return [x["tag"].lower() for x in ast.literal_eval(val)]
    except:
        return []

analysis_df["hashtag_list"] = analysis_df["entities_hashtags"].apply(extract_hashtags)
hashtags_df = analysis_df.explode("hashtag_list")
hashtags_df = hashtags_df[hashtags_df["hashtag_list"] != ""]

hashtag_metrics = (
    hashtags_df.groupby("hashtag_list")
    .agg(avg_engagement=("total_engagement","mean"))
    .reset_index()
    .sort_values("avg_engagement", ascending=False)
    .head(15)
)

fig = px.bar(
    hashtag_metrics,
    x="avg_engagement",
    y="hashtag_list",
    title="Top Hashtags by Engagement"
)

chart_with_summary(
    fig,
    "Hashtags that consistently boost engagement."
)

# ───────────────────────── Virality Score ─────────────────────────
analysis_df["virality_score"] = (
    analysis_df["retweets"] / analysis_df["total_engagement"]
)

fig = px.histogram(
    analysis_df,
    x="virality_score",
    nbins=30,
    title="Virality Score Distribution"
)

chart_with_summary(
    fig,
    "Higher values indicate content spreading beyond followers."
)

# ───────────────────────── Engagement Drivers ─────────────────────────
corr_impact = (
    analysis_df[
        ["likes","retweets","replies","quotes","bookmarks","impressions","total_engagement"]
    ]
    .corr()["total_engagement"]
    .drop("total_engagement")
    .sort_values()
    .reset_index()
)

fig = px.bar(
    corr_impact,
    x="total_engagement",
    y="index",
    orientation="h",
    title="What Drives Engagement the Most"
)

chart_with_summary(
    fig,
    "Retweets and likes are the strongest engagement drivers."
)

# ---------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import ast

# # ───────────────────────── Page Setup ─────────────────────────
# st.set_page_config(
#     page_title="Twitter EDA Dashboard",
#     page_icon="🐦",
#     layout="wide"
# )

# st.title("🐦 Twitter Engagement Analytics Dashboard")

# # ───────────────────────── Load Data ─────────────────────────
# @st.cache_data(show_spinner=False)
# def load_data(path):
#     df = pd.read_csv(path)

#     df.columns = (
#         df.columns.str.strip()
#         .str.lower()
#         .str.replace(" ", "_")
#     )

#     df["created_at"] = pd.to_datetime(df["created_at"])
#     df["date"] = df["created_at"].dt.date
#     df["hour"] = df["created_at"].dt.hour
#     df["weekday"] = df["created_at"].dt.day_name()

#     df.rename(columns={
#         'public_metrics_like_count': 'likes',
#         'public_metrics_retweet_count': 'retweets',
#         'public_metrics_reply_count': 'replies',
#         'public_metrics_quote_count': 'quotes',
#         'public_metrics_bookmark_count': 'bookmarks',
#         'public_metrics_impression_count': 'impressions'
#     }, inplace=True)

#     df["total_engagement"] = (
#         df["likes"] + df["retweets"] +
#         df["replies"] + df["quotes"] +
#         df["bookmarks"]
#     )

#     df["engagement_rate"] = (df["total_engagement"] / df["impressions"]) * 100

#     return df


# if "df" not in st.session_state:
#     st.session_state.df = load_data("C:\\TEIM Project\\social_analytics\\streamlit-app\\data\\twitter_data.csv")

# df = st.session_state.df

# # ───────────────────────── Sidebar Filters ─────────────────────────
# st.sidebar.header("🎯 Filters")

# min_date, max_date = min(df["date"]), max(df["date"])

# from_date, to_date = st.sidebar.date_input(
#     "Date Range",
#     [min_date, max_date]
# )

# filtered_df = df[
#     (df["date"] >= from_date) &
#     (df["date"] <= to_date)
# ]

# st.session_state.filtered_df = filtered_df

# metrics = [
#     "likes", "retweets", "replies",
#     "quotes", "bookmarks",
#     "impressions", "total_engagement", "engagement_rate"
# ]

# # ───────────────────────── KPI Section ─────────────────────────
# st.subheader("📌 Key Performance Indicators")

# k1, k2, k3, k4 = st.columns(4)

# k1.metric("Total Tweets", len(filtered_df))
# k2.metric("Avg Engagement", round(filtered_df["total_engagement"].mean(), 2))
# k3.metric("Avg Engagement Rate (%)", round(filtered_df["engagement_rate"].mean(), 2))
# k4.metric(
#     "Best Posting Hour",
#     filtered_df.groupby("hour")["total_engagement"].mean().idxmax()
# )

# st.divider()

# # ───────────────────────── Helper: Chart + Insight Layout ─────────────────────────
# def chart_with_summary(chart, title, summary):
#     c1, c2 = st.columns([3, 1])
#     with c1:
#         st.plotly_chart(chart, use_container_width=True)
#     with c2:
#         st.markdown(f"### 🔍 Insight\n{summary}")

# # ───────────────────────── Metric Distributions ─────────────────────────
# st.subheader("📊 Metric Distributions")

# metric_selected = st.selectbox("Select Metric", metrics)

# fig = px.histogram(
#     filtered_df,
#     x=metric_selected,
#     nbins=30,
#     title=f"Distribution of {metric_selected}"
# )

# chart_with_summary(
#     fig,
#     "",
#     f"Shows how **{metric_selected}** values are distributed. "
#     "Helps detect skewness and abnormal spikes."
# )

# # ───────────────────────── Box Plot ─────────────────────────
# fig = px.box(
#     filtered_df[metrics],
#     title="Outlier Detection Across Metrics"
# )

# chart_with_summary(
#     fig,
#     "",
#     "Highlights **outliers** and variance across engagement metrics."
# )

# # ───────────────────────── Correlation Heatmap ─────────────────────────
# corr = filtered_df[metrics].corr()

# fig = px.imshow(
#     corr,
#     text_auto=".2f",
#     title="Correlation Heatmap"
# )

# chart_with_summary(
#     fig,
#     "",
#     "Shows which metrics move together. "
#     "**Retweets & likes** usually dominate engagement."
# )

# # ───────────────────────── Scatter Relationships ─────────────────────────
# fig = px.scatter(
#     filtered_df,
#     x="likes",
#     y="retweets",
#     title="Likes vs Retweets"
# )

# chart_with_summary(
#     fig,
#     "",
#     "Strong correlation indicates **content resonance**."
# )

# fig = px.scatter(
#     filtered_df,
#     x="impressions",
#     y="total_engagement",
#     title="Impressions vs Engagement"
# )

# chart_with_summary(
#     fig,
#     "",
#     "Higher impressions generally yield higher engagement, "
#     "but content quality still matters."
# )

# # ───────────────────────── Time Series ─────────────────────────
# daily_eng = filtered_df.groupby("date")["total_engagement"].mean().reset_index()

# fig = px.line(
#     daily_eng,
#     x="date",
#     y="total_engagement",
#     title="Average Daily Engagement"
# )

# chart_with_summary(
#     fig,
#     "",
#     "Reveals engagement growth, drops, or campaign impact over time."
# )

# # ───────────────────────── Hourly & Weekday ─────────────────────────
# hourly_eng = filtered_df.groupby("hour")["total_engagement"].mean().reset_index()

# fig = px.line(
#     hourly_eng,
#     x="hour",
#     y="total_engagement",
#     title="Engagement by Posting Hour"
# )

# chart_with_summary(
#     fig,
#     "",
#     "Helps identify **best posting hours**."
# )

# weekday_order = [
#     "Monday","Tuesday","Wednesday",
#     "Thursday","Friday","Saturday","Sunday"
# ]

# weekday_eng = (
#     filtered_df.groupby("weekday")["total_engagement"]
#     .mean()
#     .reindex(weekday_order)
#     .reset_index()
# )

# fig = px.bar(
#     weekday_eng,
#     x="weekday",
#     y="total_engagement",
#     title="Engagement by Weekday"
# )

# chart_with_summary(
#     fig,
#     "",
#     "Shows which **days perform best** for posting."
# )

# # ───────────────────────── Top Tweets ─────────────────────────
# st.subheader("🔥 Top 10 Tweets by Engagement")

# st.dataframe(
#     filtered_df.sort_values(
#         "total_engagement", ascending=False
#     )[
#         ["text", "likes", "retweets", "replies", "impressions"]
#     ].head(10),
#     use_container_width=True
# )

# # ───────────────────────── Hashtag Analysis ─────────────────────────
# def extract_hashtags(val):
#     if pd.isna(val) or val == "" or val == "[]":
#         return []
#     try:
#         return [x["tag"].lower() for x in ast.literal_eval(val)]
#     except:
#         return []

# filtered_df["hashtag_list"] = filtered_df["entities_hashtags"].apply(extract_hashtags)

# hashtags_df = filtered_df.explode("hashtag_list")
# hashtags_df = hashtags_df[hashtags_df["hashtag_list"] != ""]

# hashtag_metrics = (
#     hashtags_df.groupby("hashtag_list")
#     .agg(avg_engagement=("total_engagement", "mean"))
#     .reset_index()
#     .sort_values("avg_engagement", ascending=False)
#     .head(15)
# )

# fig = px.bar(
#     hashtag_metrics,
#     x="avg_engagement",
#     y="hashtag_list",
#     title="Top Hashtags by Engagement"
# )

# chart_with_summary(
#     fig,
#     "",
#     "Identifies **hashtags that amplify reach**."
# )

# # ───────────────────────── Virality Score ─────────────────────────
# filtered_df["virality_score"] = (
#     filtered_df["retweets"] / filtered_df["total_engagement"]
# )

# fig = px.histogram(
#     filtered_df,
#     x="virality_score",
#     nbins=30,
#     title="Virality Score Distribution"
# )

# chart_with_summary(
#     fig,
#     "",
#     "Higher values indicate content spreading **beyond followers**."
# )

# # ───────────────────────── Engagement Drivers ─────────────────────────
# corr_impact = (
#     filtered_df[
#         ["likes","retweets","replies","quotes","bookmarks","impressions","total_engagement"]
#     ]
#     .corr()["total_engagement"]
#     .drop("total_engagement")
#     .sort_values()
#     .reset_index()
# )

# fig = px.bar(
#     corr_impact,
#     x="total_engagement",
#     y="index",
#     orientation="h",
#     title="What Drives Engagement the Most"
# )

# chart_with_summary(
#     fig,
#     "",
#     "**Retweets & likes** have the strongest impact on engagement."
# )
