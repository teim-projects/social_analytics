# --------------------------------------------------------------------------------------------------
# -------------------------------------INITIAL LINKEDIN DASHBOARD-----------------------------------
# --------------------------------------------------------------------------------------------------

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from datetime import datetime, timedelta

# ───────────────── Page Setup ─────────────────
st.set_page_config(
    page_title="LinkedIn Engagement EDA",
    page_icon="💼",
    layout="wide"
)

st.title("💼 LinkedIn Engagement EDA Dashboard")
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "linkedin_new.csv")

# ───────────────── Load Data ─────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df_raw = load_data()

# ───────────────── LinkedIn Relative Date Conversion ─────────────────
def linkedin_date_to_datetime(date_str):
    if pd.isna(date_str):
        return np.nan

    date_str = str(date_str).lower()
    now = datetime.now()

    match = re.search(r"(\d+)\s*(d|w|mo|yr)", date_str)
    if not match:
        return np.nan

    value = int(match.group(1))
    unit = match.group(2)

    if unit == "d":
        return now - timedelta(days=value)
    elif unit == "w":
        return now - timedelta(weeks=value)
    elif unit == "mo":
        return now - timedelta(days=value * 30)
    elif unit == "yr":
        return now - timedelta(days=value * 365)

    return np.nan

df_raw["date_parsed"] = df_raw["date"].apply(linkedin_date_to_datetime)
df = df_raw[df_raw["date_parsed"].notna()].copy()

# ───────────────── Numeric Cleaning ─────────────────
num_cols = [
    "likes", "comments_count", "shares",
    "likes_safe", "comments_safe",
    "shares_safe", "impressions_safe"
]

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["likes_safe"] = df.get("likes_safe", df.get("likes", 0))
df["comments_safe"] = df.get("comments_safe", df.get("comments_count", 0))
df["shares_safe"] = df.get("shares_safe", df.get("shares", 0))
df["impressions_safe"] = df.get("impressions_safe", 0)

df["total_engagement"] = (
    df["likes_safe"] +
    df["comments_safe"] +
    df["shares_safe"]
)

# ───────────────── Time Features ─────────────────
df["date_only"] = df["date_parsed"].dt.date
df["hour"] = df["date_parsed"].dt.hour
df["weekday"] = df["date_parsed"].dt.day_name()

weekday_order = [
    "Monday","Tuesday","Wednesday",
    "Thursday","Friday","Saturday","Sunday"
]

# ───────────────── Session State Init ─────────────────
def init_session():
    defaults = {
        "from_date": df["date_only"].min(),
        "to_date": df["date_only"].max(),
        "weekday": weekday_order,
        "hour_range": (0, 23),
        "likes_range": (int(df["likes_safe"].min()), int(df["likes_safe"].max())),
        "comments_range": (int(df["comments_safe"].min()), int(df["comments_safe"].max())),
        "shares_range": (int(df["shares_safe"].min()), int(df["shares_safe"].max())),
        "eng_range": (int(df["total_engagement"].min()), int(df["total_engagement"].max())),
        "selected_graphs": [
            "Distribution of Likes",
            "Distribution of Comments",
            "Distribution of Shares",
            "Distribution of Impressions",
            "Distribution of Total Engagement",
            "Outlier Detection",
            "Engagement Drivers",
            "Engagement Over Time",
            "7-Day Engagement Trend",
            "Engagement by Hour",
            "Engagement by Weekday",
            "Viral Posts Analysis"
        ]
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ───────────────── Sidebar Filters ─────────────────
st.sidebar.header("🔍 Filters")

c1, c2 = st.sidebar.columns(2)
with c1:
    st.session_state.from_date = st.date_input(
        "From", st.session_state.from_date
    )
with c2:
    st.session_state.to_date = st.date_input(
        "To", st.session_state.to_date
    )

st.session_state.weekday = st.sidebar.multiselect(
    "Weekday", weekday_order, default=st.session_state.weekday
)

st.session_state.hour_range = st.sidebar.slider(
    "Posting Hour", 0, 23, st.session_state.hour_range
)

st.session_state.likes_range = st.sidebar.slider(
    "Likes",
    int(df["likes_safe"].min()),
    int(df["likes_safe"].max()),
    st.session_state.likes_range
)

st.session_state.comments_range = st.sidebar.slider(
    "Comments",
    int(df["comments_safe"].min()),
    int(df["comments_safe"].max()),
    st.session_state.comments_range
)

st.session_state.shares_range = st.sidebar.slider(
    "Shares",
    int(df["shares_safe"].min()),
    int(df["shares_safe"].max()),
    st.session_state.shares_range
)

st.session_state.eng_range = st.sidebar.slider(
    "Total Engagement",
    int(df["total_engagement"].min()),
    int(df["total_engagement"].max()),
    st.session_state.eng_range
)

st.session_state.selected_graphs = st.sidebar.multiselect(
    "Select Graphs",
    st.session_state.selected_graphs,
    default=st.session_state.selected_graphs
)

# ───────────────── Apply Filters ─────────────────
df_f = df[
    (df["date_only"].between(st.session_state.from_date, st.session_state.to_date)) &
    (df["weekday"].isin(st.session_state.weekday)) &
    (df["hour"].between(*st.session_state.hour_range)) &
    (df["likes_safe"].between(*st.session_state.likes_range)) &
    (df["comments_safe"].between(*st.session_state.comments_range)) &
    (df["shares_safe"].between(*st.session_state.shares_range)) &
    (df["total_engagement"].between(*st.session_state.eng_range))
]

if df_f.empty:
    st.warning("⚠ No data after applying filters")
    st.stop()

# ───────────────── Helper Layout ─────────────────
def plot_with_summary(fig, summary):
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(summary)

# =========================================================
# ====================== GRAPHS ===========================
# =========================================================

# 1️⃣ Distribution of Likes
if "Distribution of Likes" in st.session_state.selected_graphs:
    stats = df_f["likes_safe"].describe()
    fig = px.histogram(df_f, x="likes_safe", nbins=30, marginal="box",
                       title="1️⃣ Distribution of Likes")
    summary = f"""
**Likes Distribution Summary**  
{stats.to_frame().to_markdown()}

**Interpretation**  
Most posts receive around {df_f["likes_safe"].median():.0f} likes, while a small number
achieve extremely high values.

**Conclusion**  
Engagement on LinkedIn is concentrated on a limited set of high-performing posts.
"""
    plot_with_summary(fig, summary)

# 2️⃣ Distribution of Comments
if "Distribution of Comments" in st.session_state.selected_graphs:
    stats = df_f["comments_safe"].describe()
    fig = px.histogram(df_f, x="comments_safe", nbins=30, marginal="box",
                       title="2️⃣ Distribution of Comments")
    summary = f"""
**Comments Distribution Summary**  
{stats.to_frame().to_markdown()}

**Conclusion**  
Commenting behavior is selective and driven by discussion-oriented content.
"""
    plot_with_summary(fig, summary)

# 3️⃣ Distribution of Shares
if "Distribution of Shares" in st.session_state.selected_graphs:
    stats = df_f["shares_safe"].describe()
    fig = px.histogram(df_f, x="shares_safe", nbins=30, marginal="box",
                       title="3️⃣ Distribution of Shares")
    summary = f"""
**Shares Distribution Summary**  
{stats.to_frame().to_markdown()}

**Conclusion**  
Shares represent strong content resonance and advocacy rather than passive interaction.
"""
    plot_with_summary(fig, summary)

# 4️⃣ Distribution of Impressions
if "Distribution of Impressions" in st.session_state.selected_graphs:
    stats = df_f["impressions_safe"].describe()
    fig = px.histogram(df_f, x="impressions_safe", nbins=30, marginal="box",
                       title="4️⃣ Distribution of Impressions")
    summary = f"""
**Impressions Summary**  
{stats.to_frame().to_markdown()}

**Conclusion**  
LinkedIn algorithm selectively amplifies a subset of posts.
"""
    plot_with_summary(fig, summary)

# 5️⃣ Distribution of Total Engagement
if "Distribution of Total Engagement" in st.session_state.selected_graphs:
    stats = df_f["total_engagement"].describe()
    fig = px.histogram(df_f, x="total_engagement", nbins=30, marginal="box",
                       title="5️⃣ Distribution of Total Engagement")
    summary = f"""
**Total Engagement Summary**  
{stats.to_frame().to_markdown()}

**Conclusion**  
Overall engagement is dominated by a small number of viral posts.
"""
    plot_with_summary(fig, summary)

# 6️⃣ Outlier Detection
if "Outlier Detection" in st.session_state.selected_graphs:
    fig = px.box(df_f,
                 y=["likes_safe","comments_safe","shares_safe","total_engagement"],
                 title="6️⃣ Outlier Detection")
    summary = """
**Outlier Analysis**

Viral posts appear as extreme values and represent high-impact content rather than noise.
"""
    plot_with_summary(fig, summary)

# 7️⃣ Engagement Drivers
if "Engagement Drivers" in st.session_state.selected_graphs:
    corr = (
        df_f[["likes_safe","comments_safe","shares_safe","total_engagement"]]
        .corr()["total_engagement"]
        .drop("total_engagement")
    )
    fig = px.bar(corr, orientation="h", title="7️⃣ Engagement Drivers")
    summary = """
**Driver Analysis**

Likes are the strongest contributors to total engagement.
"""
    plot_with_summary(fig, summary)

# 8️⃣ Engagement Over Time
if "Engagement Over Time" in st.session_state.selected_graphs:
    daily = df_f.groupby("date_only")["total_engagement"].mean().reset_index()
    fig = px.line(daily, x="date_only", y="total_engagement",
                  title="8️⃣ Engagement Over Time")
    summary = """
**Temporal Analysis**

Engagement fluctuates over time, reflecting audience activity cycles.
"""
    plot_with_summary(fig, summary)

# 9️⃣ 7-Day Trend
if "7-Day Engagement Trend" in st.session_state.selected_graphs:
    daily["rolling_7"] = daily["total_engagement"].rolling(7).mean()
    fig = px.line(daily, x="date_only", y="rolling_7",
                  title="9️⃣ 7-Day Engagement Trend")
    summary = """
**Trend Analysis**

Rolling averages reveal long-term engagement direction.
"""
    plot_with_summary(fig, summary)

# 🔟 Engagement by Hour
if "Engagement by Hour" in st.session_state.selected_graphs:
    hourly = df_f.groupby("hour")["total_engagement"].mean().reset_index()
    best_hour = hourly.loc[hourly["total_engagement"].idxmax(), "hour"]
    fig = px.line(hourly, x="hour", y="total_engagement", markers=True,
                  title="🔟 Engagement by Hour")
    summary = f"""
**Hourly Pattern**

Peak engagement occurs around **{best_hour}:00**.
"""
    plot_with_summary(fig, summary)

# 1️⃣1️⃣ Engagement by Weekday
if "Engagement by Weekday" in st.session_state.selected_graphs:
    weekday_avg = (
        df_f.groupby("weekday")["total_engagement"]
        .mean().reindex(weekday_order).reset_index()
    )
    best_day = weekday_avg.loc[weekday_avg["total_engagement"].idxmax(), "weekday"]
    fig = px.bar(weekday_avg, x="weekday", y="total_engagement",
                 title="1️⃣1️⃣ Engagement by Weekday")
    summary = f"""
**Weekday Pattern**

Highest engagement is observed on **{best_day}**.
"""
    plot_with_summary(fig, summary)

# 1️⃣2️⃣ Viral Posts
if "Viral Posts Analysis" in st.session_state.selected_graphs:
    threshold = df_f["total_engagement"].quantile(0.95)
    viral = df_f[df_f["total_engagement"] >= threshold]
    fig = px.scatter(viral, x="hour", y="total_engagement", color="weekday",
                     title="1️⃣2️⃣ Viral Posts Analysis")
    summary = """
**Viral Content Insight**

Viral posts cluster around specific hours and weekdays, confirming timing effects.
"""
    plot_with_summary(fig, summary)

# ---------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import re
# from datetime import datetime, timedelta

# # ───────────────── Page Setup ─────────────────
# st.set_page_config(
#     page_title="LinkedIn Engagement EDA",
#     page_icon="💼",
#     layout="wide"
# )

# st.title("💼 LinkedIn Engagement EDA Dashboard")

# # ───────────────── Load Data ─────────────────
# @st.cache_data
# def load_data():
#     return pd.read_csv(r"C:\TEIM Project\social_analytics\streamlit-app\data\linkedin_new.csv")

# df_raw = load_data()

# # ───────────────── LinkedIn Relative Date Conversion ─────────────────
# def linkedin_date_to_datetime(date_str):
#     if pd.isna(date_str):
#         return np.nan

#     date_str = str(date_str).lower()
#     now = datetime.now()

#     match = re.search(r"(\d+)\s*(d|w|mo|yr)", date_str)
#     if not match:
#         return np.nan

#     value, unit = int(match.group(1)), match.group(2)

#     if unit == "d":
#         return now - timedelta(days=value)
#     if unit == "w":
#         return now - timedelta(weeks=value)
#     if unit == "mo":
#         return now - timedelta(days=value * 30)
#     if unit == "yr":
#         return now - timedelta(days=value * 365)

#     return np.nan

# df_raw["date_parsed"] = df_raw["date"].apply(linkedin_date_to_datetime)
# df = df_raw[df_raw["date_parsed"].notna()].copy()

# # ───────────────── Numeric Cleaning ─────────────────
# for col in ["likes","comments_count","shares","impressions"]:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# df["total_engagement"] = df["likes"] + df["comments_count"] + df["shares"]

# # ───────────────── Time Features ─────────────────
# df["date_only"] = df["date_parsed"].dt.date
# df["hour"] = df["date_parsed"].dt.hour
# df["weekday"] = df["date_parsed"].dt.day_name()

# # ───────────────── Sidebar Filters ─────────────────
# st.sidebar.header("🔍 Filters")

# min_date, max_date = df["date_only"].min(), df["date_only"].max()
# c1, c2 = st.sidebar.columns(2)

# with c1:
#     from_date = st.date_input("From", min_date)
# with c2:
#     to_date = st.date_input("To", max_date)

# likes_range = st.sidebar.slider("Likes", int(df.likes.min()), int(df.likes.max()),
#                                 (int(df.likes.min()), int(df.likes.max())))
# comments_range = st.sidebar.slider("Comments", int(df.comments_count.min()), int(df.comments_count.max()),
#                                    (int(df.comments_count.min()), int(df.comments_count.max())))
# shares_range = st.sidebar.slider("Shares", int(df.shares.min()), int(df.shares.max()),
#                                  (int(df.shares.min()), int(df.shares.max())))
# eng_range = st.sidebar.slider("Total Engagement", int(df.total_engagement.min()),
#                               int(df.total_engagement.max()),
#                               (int(df.total_engagement.min()), int(df.total_engagement.max())))

# # ───────────────── Apply Filters ─────────────────
# df_filtered = df[
#     (df.date_only.between(from_date, to_date)) &
#     (df.likes.between(*likes_range)) &
#     (df.comments_count.between(*comments_range)) &
#     (df.shares.between(*shares_range)) &
#     (df.total_engagement.between(*eng_range))
# ]

# if df_filtered.empty:
#     st.warning("⚠ No data after applying filters")
#     st.stop()

# # ───────────────── Session Management ─────────────────
# if "selected_graphs" not in st.session_state:
#     st.session_state.selected_graphs = [
#         "Distribution of Likes","Distribution of Comments","Distribution of Shares",
#         "Distribution of Impressions","Total Engagement Distribution",
#         "Outlier Detection","Engagement Drivers","Engagement Over Time",
#         "7-Day Smoothed Trend","Engagement by Hour","Engagement by Weekday","Viral Posts Analysis"
#     ]

# # ───────────────── Helper Layout ─────────────────
# def plot_with_summary(fig, summary):
#     c1, c2 = st.columns([3, 1])
#     with c1:
#         st.plotly_chart(fig, use_container_width=True)
#     with c2:
#         st.markdown(summary)

# # ───────────────── 1️⃣ Distribution of Likes ─────────────────
# st.subheader("1️⃣ Distribution of Likes")
# likes_stats = df_filtered["likes"].describe()
# fig = px.histogram(df_filtered, x="likes", nbins=30, marginal="box")
# plot_with_summary(fig, f"""
# ### 📌 Graph Summary
# {likes_stats.to_frame().to_markdown()}

# ### 🔍 Key Insight
# Likes show a heavy right-skew, indicating a few highly popular posts.

# ### ✅ Conclusion
# Engagement on LinkedIn is driven by a limited number of high-performing posts.
# """)

# # ───────────────── 2️⃣ Distribution of Comments ─────────────────
# st.subheader("2️⃣ Distribution of Comments")
# comments_stats = df_filtered["comments_count"].describe()
# fig = px.histogram(df_filtered, x="comments_count", nbins=30, marginal="box")
# plot_with_summary(fig, f"""
# ### 📌 Graph Summary
# {comments_stats.to_frame().to_markdown()}

# ### 🔍 Key Insight
# Most posts receive minimal comments, reflecting selective user participation.

# ### ✅ Conclusion
# Comments indicate deeper engagement and are concentrated on discussion-driven posts.
# """)

# # ───────────────── 3️⃣ Distribution of Shares ─────────────────
# st.subheader("3️⃣ Distribution of Shares")
# shares_stats = df_filtered["shares"].describe()
# fig = px.histogram(df_filtered, x="shares", nbins=30, marginal="box")
# plot_with_summary(fig, f"""
# ### 📌 Graph Summary
# {shares_stats.to_frame().to_markdown()}

# ### 🔍 Key Insight
# Sharing behavior is rare and limited to high-value content.

# ### ✅ Conclusion
# Shares represent strong content relevance rather than casual interaction.
# """)

# # ───────────────── 4️⃣ Distribution of Impressions ─────────────────
# st.subheader("4️⃣ Distribution of Impressions")
# fig = px.histogram(df_filtered, x="impressions", nbins=30, marginal="box")
# plot_with_summary(fig, """
# ### 📌 Graph Summary
# Impressions distribution across posts.

# ### 🔍 Key Insight
# Visibility varies widely due to algorithmic amplification.

# ### ✅ Conclusion
# Not all posts receive equal exposure on LinkedIn feeds.
# """)

# # ───────────────── 5️⃣ Total Engagement Distribution ─────────────────
# st.subheader("5️⃣ Total Engagement Distribution")
# eng_stats = df_filtered["total_engagement"].describe()
# fig = px.histogram(df_filtered, x="total_engagement", nbins=30, marginal="box")
# plot_with_summary(fig, f"""
# ### 📌 Graph Summary
# {eng_stats.to_frame().to_markdown()}

# ### 🔍 Key Insight
# Total engagement is dominated by a few viral posts.

# ### ✅ Conclusion
# Overall engagement distribution is highly skewed.
# """)

# # ───────────────── 6️⃣ Outlier Detection ─────────────────
# st.subheader("6️⃣ Outlier Detection")
# fig = px.box(df_filtered, y=["likes","comments_count","shares"])
# plot_with_summary(fig, """
# ### 📌 Graph Summary
# Boxplots highlight extreme engagement values.

# ### 🔍 Key Insight
# Outliers correspond to viral or high-impact posts.

# ### ✅ Conclusion
# Outlier analysis helps identify top-performing content.
# """)

# # ───────────────── 7️⃣ Engagement Drivers ─────────────────
# st.subheader("7️⃣ Engagement Drivers")
# corr = df_filtered[["likes","comments_count","shares","total_engagement"]].corr()["total_engagement"].drop("total_engagement")
# fig = px.bar(corr, orientation="h")
# plot_with_summary(fig, """
# ### 📌 Graph Summary
# Correlation between engagement metrics.

# ### 🔍 Key Insight
# Likes contribute most to total engagement.

# ### ✅ Conclusion
# Optimizing for likes yields the highest engagement returns.
# """)

# # ───────────────── 8️⃣ Engagement Over Time ─────────────────
# st.subheader("8️⃣ Engagement Over Time")
# daily = df_filtered.groupby("date_only")["total_engagement"].mean().reset_index()
# fig = px.line(daily, x="date_only", y="total_engagement")
# plot_with_summary(fig, """
# ### 📌 Graph Summary
# Average engagement trend over time.

# ### 🔍 Key Insight
# Engagement fluctuates with content strategy and audience behavior.

# ### ✅ Conclusion
# Temporal trends assist in planning posting schedules.
# """)

# # ───────────────── 9️⃣ 7-Day Smoothed Trend ─────────────────
# st.subheader("9️⃣ 7-Day Smoothed Engagement Trend")
# daily["rolling"] = daily["total_engagement"].rolling(7).mean()
# fig = px.line(daily, x="date_only", y="rolling")
# plot_with_summary(fig, """
# ### 📌 Graph Summary
# Smoothed engagement trend using rolling average.

# ### 🔍 Key Insight
# Short-term volatility is reduced.

# ### ✅ Conclusion
# Smoothed trends reveal long-term engagement patterns.
# """)

# # ───────────────── 🔟 Engagement by Hour ─────────────────
# st.subheader("🔟 Engagement by Hour")
# hourly = df_filtered.groupby("hour")["total_engagement"].mean().reset_index()
# best_hour = hourly.loc[hourly.total_engagement.idxmax(), "hour"]
# fig = px.line(hourly, x="hour", y="total_engagement", markers=True)
# plot_with_summary(fig, f"""
# ### 📌 Graph Summary
# Engagement across posting hours.

# ### 🔍 Key Insight
# Peak engagement occurs around {best_hour}:00.

# ### ✅ Conclusion
# Posting time significantly influences performance.
# """)

# # ───────────────── 1️⃣1️⃣ Engagement by Weekday ─────────────────
# st.subheader("1️⃣1️⃣ Engagement by Weekday")
# weekday = df_filtered.groupby("weekday")["total_engagement"].mean().reset_index()
# best_day = weekday.loc[weekday.total_engagement.idxmax(), "weekday"]
# fig = px.bar(weekday, x="weekday", y="total_engagement")
# plot_with_summary(fig, f"""
# ### 📌 Graph Summary
# Engagement variation across weekdays.

# ### 🔍 Key Insight
# Highest engagement is observed on {best_day}.

# ### ✅ Conclusion
# Weekday choice impacts post visibility and interaction.
# """)

# # ───────────────── 1️⃣2️⃣ Viral Posts Analysis ─────────────────
# st.subheader("1️⃣2️⃣ Viral Posts Analysis")
# threshold = df_filtered.total_engagement.quantile(0.95)
# viral = df_filtered[df_filtered.total_engagement >= threshold]
# fig = px.scatter(viral, x="hour", y="total_engagement", color="weekday")
# plot_with_summary(fig, """
# ### 📌 Graph Summary
# Posts above 95th percentile engagement.

# ### 🔍 Key Insight
# Viral posts cluster around specific times and days.

# ### ✅ Conclusion
# Strategic timing improves chances of virality.
# """)

# --------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import re
# from datetime import datetime, timedelta

# # ───────────────── Page Setup ─────────────────
# st.set_page_config(
#     page_title="LinkedIn Engagement EDA",
#     page_icon="💼",
#     layout="wide"
# )

# st.title("💼 LinkedIn Engagement EDA Dashboard")

# # ───────────────── Load Data ─────────────────
# @st.cache_data
# def load_data():
#     return pd.read_csv(
#         r"C:\TEIM Project\social_analytics\streamlit-app\data\linkedin_new.csv"
#     )

# df_raw = load_data()

# # ───────────────── LinkedIn Relative Date Conversion ─────────────────
# def linkedin_date_to_datetime(date_str):
#     if pd.isna(date_str):
#         return np.nan

#     date_str = str(date_str).lower()
#     now = datetime.now()

#     match = re.search(r"(\d+)\s*(d|w|mo|yr)", date_str)
#     if not match:
#         return np.nan

#     value = int(match.group(1))
#     unit = match.group(2)

#     if unit == "d":
#         return now - timedelta(days=value)
#     elif unit == "w":
#         return now - timedelta(weeks=value)
#     elif unit == "mo":
#         return now - timedelta(days=value * 30)
#     elif unit == "yr":
#         return now - timedelta(days=value * 365)

#     return np.nan

# df_raw["date_parsed"] = df_raw["date"].apply(linkedin_date_to_datetime)
# df = df_raw[df_raw["date_parsed"].notna()].copy()

# # ───────────────── Numeric Cleaning ─────────────────
# num_cols = [
#     "likes", "comments_count", "shares",
#     "likes_safe", "comments_safe",
#     "shares_safe", "impressions_safe"
# ]

# for col in num_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# df["likes_safe"] = df.get("likes_safe", df.get("likes", 0))
# df["comments_safe"] = df.get("comments_safe", df.get("comments_count", 0))
# df["shares_safe"] = df.get("shares_safe", df.get("shares", 0))
# df["impressions_safe"] = df.get("impressions_safe", 0)

# df["total_engagement"] = (
#     df["likes_safe"] +
#     df["comments_safe"] +
#     df["shares_safe"]
# )

# # ───────────────── Time Features ─────────────────
# df["date_only"] = df["date_parsed"].dt.date
# df["hour"] = df["date_parsed"].dt.hour
# df["weekday"] = df["date_parsed"].dt.day_name()

# weekday_order = [
#     "Monday","Tuesday","Wednesday",
#     "Thursday","Friday","Saturday","Sunday"
# ]

# # ───────────────── Session State Init ─────────────────
# def init_session_state():
#     defaults = {
#         "from_date": df["date_only"].min(),
#         "to_date": df["date_only"].max(),
#         "weekday": weekday_order,
#         "hour_range": (0, 23),
#         "likes_range": (int(df["likes_safe"].min()), int(df["likes_safe"].max())),
#         "comments_range": (int(df["comments_safe"].min()), int(df["comments_safe"].max())),
#         "shares_range": (int(df["shares_safe"].min()), int(df["shares_safe"].max())),
#         "eng_range": (int(df["total_engagement"].min()), int(df["total_engagement"].max()))
#     }

#     for k, v in defaults.items():
#         if k not in st.session_state:
#             st.session_state[k] = v

# init_session_state()

# # ───────────────── Sidebar Filters ─────────────────
# st.sidebar.header("🔍 Filters")

# c1, c2 = st.sidebar.columns(2)
# with c1:
#     st.session_state.from_date = st.date_input(
#         "From",
#         st.session_state.from_date,
#         min_value=df["date_only"].min(),
#         max_value=df["date_only"].max()
#     )

# with c2:
#     st.session_state.to_date = st.date_input(
#         "To",
#         st.session_state.to_date,
#         min_value=df["date_only"].min(),
#         max_value=df["date_only"].max()
#     )

# st.session_state.weekday = st.sidebar.multiselect(
#     "Weekday",
#     weekday_order,
#     default=st.session_state.weekday
# )

# st.session_state.hour_range = st.sidebar.slider(
#     "Posting Hour",
#     0, 23,
#     st.session_state.hour_range
# )

# st.session_state.likes_range = st.sidebar.slider(
#     "Likes",
#     int(df["likes_safe"].min()),
#     int(df["likes_safe"].max()),
#     st.session_state.likes_range
# )

# st.session_state.comments_range = st.sidebar.slider(
#     "Comments",
#     int(df["comments_safe"].min()),
#     int(df["comments_safe"].max()),
#     st.session_state.comments_range
# )

# st.session_state.shares_range = st.sidebar.slider(
#     "Shares",
#     int(df["shares_safe"].min()),
#     int(df["shares_safe"].max()),
#     st.session_state.shares_range
# )

# st.session_state.eng_range = st.sidebar.slider(
#     "Total Engagement",
#     int(df["total_engagement"].min()),
#     int(df["total_engagement"].max()),
#     st.session_state.eng_range
# )

# # ───────────────── Apply Filters ─────────────────
# df_f = df[
#     (df["date_only"].between(st.session_state.from_date, st.session_state.to_date)) &
#     (df["weekday"].isin(st.session_state.weekday)) &
#     (df["hour"].between(*st.session_state.hour_range)) &
#     (df["likes_safe"].between(*st.session_state.likes_range)) &
#     (df["comments_safe"].between(*st.session_state.comments_range)) &
#     (df["shares_safe"].between(*st.session_state.shares_range)) &
#     (df["total_engagement"].between(*st.session_state.eng_range))
# ]

# if df_f.empty:
#     st.warning("⚠ No data after applying filters")
#     st.stop()

# # ───────────────── Helper Layout ─────────────────
# def plot_with_summary(fig, text):
#     c1, c2 = st.columns([3, 1])
#     with c1:
#         st.plotly_chart(fig, use_container_width=True)
#     with c2:
#         st.markdown(text)

# # ───────────────── 1–5 Metric Distributions ─────────────────
# metrics = [
#     "likes_safe", "comments_safe",
#     "shares_safe", "impressions_safe",
#     "total_engagement"
# ]

# for i, metric in enumerate(metrics, start=1):
#     fig = px.histogram(
#         df_f,
#         x=metric,
#         nbins=30,
#         marginal="box",
#         title=f"{i}️⃣ Distribution of {metric.replace('_',' ').title()}"
#     )
#     plot_with_summary(
#         fig,
#         "Highly right-skewed distribution, indicating a small number of high-performing posts dominate engagement."
#     )

# # ───────────────── 6️⃣ Outlier Detection ─────────────────
# fig = px.box(
#     df_f,
#     y=metrics,
#     title="6️⃣ Outlier Detection Across Engagement Metrics"
# )

# plot_with_summary(
#     fig,
#     "Extreme outliers correspond to viral posts and represent valuable strategic signals."
# )

# # ───────────────── 7️⃣ Engagement Drivers ─────────────────
# corr = (
#     df_f[["likes_safe","comments_safe","shares_safe","total_engagement"]]
#     .corr()["total_engagement"]
#     .drop("total_engagement")
#     .sort_values()
# )

# fig = px.bar(
#     corr,
#     orientation="h",
#     title="7️⃣ Correlation with Total Engagement"
# )

# plot_with_summary(
#     fig,
#     "Likes exhibit the strongest correlation with total engagement, followed by comments and shares."
# )

# # ───────────────── 8️⃣ Engagement Over Time ─────────────────
# daily = df_f.groupby("date_only")["total_engagement"].mean().reset_index()

# fig = px.line(
#     daily,
#     x="date_only",
#     y="total_engagement",
#     title="8️⃣ Average Engagement Over Time"
# )

# plot_with_summary(
#     fig,
#     "Engagement varies temporally, reflecting content performance cycles and audience responsiveness."
# )

# # ───────────────── 9️⃣ Rolling Trend ─────────────────
# daily["rolling_7"] = daily["total_engagement"].rolling(7).mean()

# fig = px.line(
#     daily,
#     x="date_only",
#     y="rolling_7",
#     title="9️⃣ 7-Day Smoothed Engagement Trend"
# )

# plot_with_summary(
#     fig,
#     "Rolling averages reduce noise and reveal long-term engagement direction."
# )

# # ───────────────── 🔟 Hourly Engagement ─────────────────
# hourly = df_f.groupby("hour")["total_engagement"].mean().reset_index()
# best_hour = hourly.loc[hourly["total_engagement"].idxmax(), "hour"]

# fig = px.line(
#     hourly,
#     x="hour",
#     y="total_engagement",
#     markers=True,
#     title="🔟 Average Engagement by Posting Hour"
# )

# plot_with_summary(
#     fig,
#     f"Engagement peaks around **{best_hour}:00**, suggesting optimal posting windows."
# )

# # ───────────────── 1️⃣1️⃣ Weekday Engagement ─────────────────
# weekday_avg = (
#     df_f.groupby("weekday")["total_engagement"]
#     .mean()
#     .reindex(weekday_order)
#     .reset_index()
# )

# best_day = weekday_avg.loc[
#     weekday_avg["total_engagement"].idxmax(), "weekday"
# ]

# fig = px.bar(
#     weekday_avg,
#     x="weekday",
#     y="total_engagement",
#     title="1️⃣1️⃣ Average Engagement by Day of Week"
# )

# plot_with_summary(
#     fig,
#     f"Posts published on **{best_day}** achieve the highest average engagement."
# )

# # ───────────────── 1️⃣2️⃣ Viral Posts ─────────────────
# viral_threshold = df_f["total_engagement"].quantile(0.95)
# viral = df_f[df_f["total_engagement"] >= viral_threshold]

# fig = px.scatter(
#     viral,
#     x="hour",
#     y="total_engagement",
#     color="weekday",
#     title="1️⃣2️⃣ Viral Post Timing Analysis (Top 5%)"
# )

# plot_with_summary(
#     fig,
#     "Viral posts cluster around specific hours and weekdays, highlighting the importance of posting time."
# )

# --------------------------------------------------------------------------------------------------
# -------------------------------------BEFORE TIME SERIES ANALYSIS----------------------------------
# --------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# import re
# from datetime import datetime, timedelta

# # ───────────────── Page Setup ─────────────────
# st.set_page_config(
#     page_title="LinkedIn Engagement EDA",
#     page_icon="💼",
#     layout="wide"
# )

# st.title("💼 LinkedIn Engagement EDA Dashboard")

# # ───────────────── Load Data ─────────────────
# @st.cache_data
# def load_data():
#     return pd.read_csv("C:\TEIM Project\social_analytics\streamlit-app\data\linkedin_updated.csv")

# df_raw = load_data()

# # ───────────────── LinkedIn Relative Date Conversion ─────────────────
# def linkedin_date_to_datetime(date_str):
#     if pd.isna(date_str):
#         return np.nan

#     date_str = str(date_str).lower()
#     now = datetime.now()

#     match = re.search(r"(\d+)\s*(d|w|mo|yr)", date_str)
#     if not match:
#         return np.nan

#     value = int(match.group(1))
#     unit = match.group(2)

#     if unit == "d":
#         return now - timedelta(days=value)
#     elif unit == "w":
#         return now - timedelta(weeks=value)
#     elif unit == "mo":
#         return now - timedelta(days=value * 30)
#     elif unit == "yr":
#         return now - timedelta(days=value * 365)

#     return np.nan

# df_raw["date_parsed"] = df_raw["date"].apply(linkedin_date_to_datetime)
# df = df_raw[df_raw["date_parsed"].notna()].copy()

# # ───────────────── Numeric Cleaning ─────────────────
# num_cols = [
#     "likes", "comments_count", "shares",
#     "likes_safe", "comments_safe",
#     "shares_safe", "impressions_safe"
# ]

# for col in num_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# df["likes_safe"] = df.get("likes_safe", df.get("likes", 0))
# df["comments_safe"] = df.get("comments_safe", df.get("comments_count", 0))
# df["shares_safe"] = df.get("shares_safe", df.get("shares", 0))
# df["impressions_safe"] = df.get("impressions_safe", 0)

# df["total_engagement"] = (
#     df["likes_safe"] +
#     df["comments_safe"] +
#     df["shares_safe"]
# )

# # ───────────────── Time Features ─────────────────
# df["date_only"] = df["date_parsed"].dt.date
# df["hour"] = df["date_parsed"].dt.hour
# df["weekday"] = df["date_parsed"].dt.day_name()

# # ───────────────── Sidebar Filters ─────────────────
# st.sidebar.header("🔍 Filters")

# # ── Date Filter
# min_date = df["date_only"].min()
# max_date = df["date_only"].max()

# col1, col2 = st.sidebar.columns(2)

# with col1:
#     from_date = st.date_input(
#         "From",
#         min_date,
#         min_value=min_date,
#         max_value=max_date
#     )

# with col2:
#     to_date = st.date_input(
#         "To",
#         max_date,
#         min_value=min_date,
#         max_value=max_date
#     )

# # ── Engagement Sliders

# likes_range = st.sidebar.slider(
#     "Likes",
#     int(df["likes_safe"].min()),
#     int(df["likes_safe"].max()),
#     (int(df["likes_safe"].min()), int(df["likes_safe"].max()))
# )

# comments_range = st.sidebar.slider(
#     "Comments",
#     int(df["comments_safe"].min()),
#     int(df["comments_safe"].max()),
#     (int(df["comments_safe"].min()), int(df["comments_safe"].max()))
# )

# shares_range = st.sidebar.slider(
#     "Shares",
#     int(df["shares_safe"].min()),
#     int(df["shares_safe"].max()),
#     (int(df["shares_safe"].min()), int(df["shares_safe"].max()))
# )

# eng_range = st.sidebar.slider(
#     "Total Engagement",
#     int(df["total_engagement"].min()),
#     int(df["total_engagement"].max()),
#     (int(df["total_engagement"].min()), int(df["total_engagement"].max()))
# )

# # ───────────────── Apply Filters ─────────────────
# df_f = df[
#     (df["date_only"] >= from_date) &
#     (df["date_only"] <= to_date) &
#     (df["likes_safe"].between(*likes_range)) &
#     (df["comments_safe"].between(*comments_range)) &
#     (df["shares_safe"].between(*shares_range)) &
#     (df["total_engagement"].between(*eng_range))
# ]

# if df_f.empty:
#     st.warning("⚠ No data after applying filters")
#     st.stop()

# # ───────────────── Helper Layout ─────────────────
# def plot_with_summary(fig, text):
#     c1, c2 = st.columns([3, 1])
#     with c1:
#         st.plotly_chart(fig, use_container_width=True)
#     with c2:
#         st.markdown(text)

# # ───────────────── 1️⃣ Metric Distributions ─────────────────
# metrics = [
#     "likes_safe",
#     "comments_safe",
#     "shares_safe",
#     "impressions_safe",
#     "total_engagement"
# ]

# for metric in metrics:
#     fig = px.histogram(
#         df_f,
#         x=metric,
#         nbins=30,
#         marginal="box",
#         title=f"Distribution of {metric.replace('_',' ').title()}"
#     )

#     plot_with_summary(
#         fig,
#         "Right-skewed distribution.\n\nFew viral posts dominate engagement."
#     )

# # ───────────────── 2️⃣ Outlier Detection ─────────────────
# fig = px.box(
#     df_f,
#     y=metrics,
#     title="Outlier Detection – LinkedIn Engagement"
# )

# plot_with_summary(
#     fig,
#     "Outliers = viral posts.\n\nThese are valuable signals."
# )

# # ───────────────── 3️⃣ Engagement Drivers ─────────────────
# corr = (
#     df_f[["likes_safe","comments_safe","shares_safe","total_engagement"]]
#     .corr()["total_engagement"]
#     .drop("total_engagement")
#     .sort_values()
# )

# fig = px.bar(
#     corr,
#     orientation="h",
#     title="What Drives Engagement"
# )

# plot_with_summary(
#     fig,
#     "Likes are the strongest driver of engagement."
# )

# # ───────────────── 4️⃣ Engagement Over Time ─────────────────
# daily = df_f.groupby("date_only")["total_engagement"].mean().reset_index()

# fig = px.line(
#     daily,
#     x="date_only",
#     y="total_engagement",
#     title="Average Engagement Over Time"
# )

# best_hour = df_f.groupby("hour")["total_engagement"].mean().idxmax()

# plot_with_summary(
#     fig,
#     f"Best posting hour: **{best_hour}:00**"
# )