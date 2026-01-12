# --------------------------------VIDEO WISE------------------------------------------

import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ────────────── Page setup ──────────────
st.set_page_config(
    page_title="YouTube Sentiment Dashboard",
    page_icon="🎬",
    layout="wide"
)
st.title("🎬 YouTube Sentiment Dashboard")

# ────────────── Load data ──────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(base_dir, "models", "transformer_annotated_comments.pkl")

    if not os.path.exists(pkl_path):
        st.error(f"❌ Pickle file not found at:\n{pkl_path}")
        st.stop()

    df = pd.read_pickle(pkl_path)

    required_cols = [
        "Comment",
        "sentiment_label",
        "sentiment_score",
        "Published_At",
        "Likes",
        "Video_Title"
    ]

    for c in required_cols:
        if c not in df.columns:
            st.error(f"Required column missing in pickle: {c}")
            st.stop()

    df = df.copy()
    df["Comment"] = df["Comment"].astype(str)
    df["Video_Title"] = df["Video_Title"].astype(str)
    df["Published_At"] = pd.to_datetime(df["Published_At"], errors="coerce")
    df["Likes"] = pd.to_numeric(df["Likes"], errors="coerce").fillna(0).astype(int)
    df["sentiment_label"] = df["sentiment_label"].astype(str)

    return df


df = load_data()

# ────────────── Initialize session state (ONCE) ──────────────
if "date_from" not in st.session_state:
    st.session_state["date_from"] = df["Published_At"].min().date()

if "date_to" not in st.session_state:
    st.session_state["date_to"] = df["Published_At"].max().date()

if "selected_sentiments" not in st.session_state:
    st.session_state["selected_sentiments"] = ["Positive", "Neutral", "Negative"]

if "selected_videos" not in st.session_state:
    st.session_state["selected_videos"] = []

if "likes_range" not in st.session_state:
    st.session_state["likes_range"] = (0, int(df["Likes"].max()))

if "search_text" not in st.session_state:
    st.session_state["search_text"] = ""

if "view_all_videos" not in st.session_state:
    st.session_state["view_all_videos"] = True

# Defaults (for Clear All)
DEFAULTS = {
    "date_from": df["Published_At"].min().date(),
    "date_to": df["Published_At"].max().date(),
    "selected_sentiments": ["Positive", "Neutral", "Negative"],
    "selected_videos": [],
    "view_all_videos": True,
    "likes_range": (0, int(df["Likes"].max())),
    "search_text": ""
}

def clear_all_filters():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# ────────────── Sidebar filters ──────────────
st.sidebar.header("📌 Filters")
st.sidebar.button("🧹 Clear All Filters", on_click=clear_all_filters)

# DATE FILTERS (FIXED)
c1, c2 = st.sidebar.columns(2)
c1.date_input("From", key="date_from")
c2.date_input("To", key="date_to")

if st.session_state["date_from"] > st.session_state["date_to"]:
    st.sidebar.error("❌ 'From' date cannot be after 'To' date")
    st.stop()

# SENTIMENT FILTER
sentiment_options = ["Positive", "Neutral", "Negative"]
st.sidebar.multiselect(
    "Select sentiment(s)",
    sentiment_options,
    key="selected_sentiments"
)

# VIDEO FILTER (user must select)
st.sidebar.multiselect(
    "Select video(s)",
    sorted(df["Video_Title"].unique()),
    key="selected_videos",
    disabled=st.session_state["view_all_videos"]
)

st.sidebar.checkbox(
    "📺 View all videos",
    key="view_all_videos"
)

# LIKES FILTER
st.sidebar.slider(
    "Likes Range",
    0,
    int(df["Likes"].max()),
    key="likes_range"
)

# SEARCH
st.sidebar.text_input(
    "Search comments by keyword",
    key="search_text"
)

# ────────────── Apply composite filters ──────────────
mask = pd.Series(True, index=df.index)

mask &= df["Published_At"].dt.date.between(
    st.session_state["date_from"],
    st.session_state["date_to"]
)

mask &= df["sentiment_label"].isin(st.session_state["selected_sentiments"])

# VIDEO FILTER LOGIC (FIXED)
if not st.session_state["view_all_videos"]:
    if st.session_state["selected_videos"]:
        mask &= df["Video_Title"].isin(st.session_state["selected_videos"])
    else:
        mask &= False  # no video selected & not viewing all

likes_min, likes_max = st.session_state["likes_range"]
mask &= df["Likes"].between(likes_min, likes_max)

if st.session_state["search_text"].strip():
    mask &= df["Comment"].str.contains(
        st.session_state["search_text"],
        case=False,
        na=False
    )

df_filtered = df[mask].reset_index(drop=True)

if (
    not st.session_state["view_all_videos"]
    and not st.session_state["selected_videos"]
):
    st.info("👈 Select one or more videos or enable **View all videos** to begin analysis")
    st.stop()

if df_filtered.empty:
    st.warning("⚠️ No data after applying filters.")
    st.stop()

# ────────────── KPIs ──────────────
st.markdown("### Summary")

c1, c2, c3, c4 = st.columns(4)
total = len(df_filtered)

c1.metric("Comments shown", f"{total:,}")
c2.metric("Positive %",
          f"{(df_filtered.sentiment_label == 'Positive').mean() * 100:.1f}%")
c3.metric("Neutral %",
          f"{(df_filtered.sentiment_label == 'Neutral').mean() * 100:.1f}%")
c4.metric("Negative %",
          f"{(df_filtered.sentiment_label == 'Negative').mean() * 100:.1f}%")

# ────────────── 1️⃣ Sentiment Distribution ──────────────
st.markdown("---")
st.subheader("1️⃣ Sentiment Distribution")

g1, g2 = st.columns([3, 1])

counts = (
    df_filtered["sentiment_label"]
    .value_counts()
    .reindex(sentiment_options)
    .fillna(0)
)

with g1:
    fig_bar = px.bar(
        x=counts.index,
        y=counts.values,
        labels={"x": "Sentiment", "y": "Count"},
        color=counts.index,
        color_discrete_map={
            "Positive": "#2ca02c",
            "Neutral": "#7f7f7f",
            "Negative": "#d62728"
        }
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with g2:
    st.markdown("#### Summary")
    for s in sentiment_options:
        st.write(f"- {s}: **{int(counts[s])}**")

# ────────────── 2️⃣ Sentiment Share ──────────────
st.markdown("---")
st.subheader("2️⃣ Sentiment Share")

g1, g2 = st.columns([3, 1])

with g1:
    fig_pie = px.pie(
        names=counts.index,
        values=counts.values,
        color=counts.index,
        color_discrete_map={
            "Positive": "#2ca02c",
            "Neutral": "#7f7f7f",
            "Negative": "#d62728"
        }
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with g2:
    st.markdown("#### Summary")
    st.write(f"- Largest segment: **{counts.idxmax()}**")

# ────────────── 3️⃣ Monthly Trend ──────────────
st.markdown("---")
st.subheader("3️⃣ Monthly Sentiment Trend")

g1, g2 = st.columns([3, 1])

df_month = df_filtered.copy()
df_month["month"] = df_month["Published_At"].dt.to_period("M").dt.to_timestamp()

monthly = (
    df_month
    .groupby(["month", "sentiment_label"])
    .size()
    .unstack(fill_value=0)
)

with g1:
    if not monthly.empty:
        fig_trend = px.line(
            monthly.reset_index(),
            x="month",
            y=monthly.columns
        )
        st.plotly_chart(fig_trend, use_container_width=True)

with g2:
    st.markdown("#### Summary")
    if not monthly.empty:
        latest = monthly.iloc[-1]
        for s, v in latest.items():
            st.write(f"- {s}: {v}")

# ────────────── 4️⃣ Likes vs Sentiment ──────────────
st.markdown("---")
st.subheader("4️⃣ Likes vs Sentiment")

g1, g2 = st.columns([3, 1])

with g1:
    fig_box = px.box(
        df_filtered,
        x="sentiment_label",
        y="Likes",
        category_orders={"sentiment_label": sentiment_options}
    )
    st.plotly_chart(fig_box, use_container_width=True)

with g2:
    st.markdown("#### Summary")
    medians = df_filtered.groupby("sentiment_label")["Likes"].median()
    for s, v in medians.items():
        st.write(f"- Median Likes ({s}): {int(v)}")

# ────────────── Tables ──────────────
st.markdown("---")
st.subheader("Top Positive Comments")
st.dataframe(
    df_filtered[df_filtered.sentiment_label == "Positive"]
    .sort_values(["sentiment_score", "Likes"], ascending=False)
    .head(10),
    use_container_width=True
)

st.subheader("Top Negative Comments")
st.dataframe(
    df_filtered[df_filtered.sentiment_label == "Negative"]
    .sort_values(["sentiment_score", "Likes"], ascending=False)
    .head(10),
    use_container_width=True
)

# --------------------------------------------------------------------------------------------
# ---------------------------------BEFORE VIDEO WISE------------------------------------------
# --------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import datetime
# import os

# # ────────────── Streamlit page setup ──────────────
# st.set_page_config(page_title="YouTube Sentiment Dashboard", page_icon="🎬", layout="wide")
# st.title("🎬 YouTube Sentiment Dashboard")

# # ────────────── Helpers & Load ──────────────
# @st.cache_data(show_spinner=False)
# def load_data() -> pd.DataFrame:

#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     pkl_path = os.path.join(base_dir, "models", "youtube_comments_sentiment.pkl")

#     if not os.path.exists(pkl_path):
#         st.error(f"❌ Pickle file not found at:\n{pkl_path}")
#         st.stop()

#     df = pd.read_pickle(pkl_path)

#     for c in ["Comment", "sentiment_label", "sentiment_score", "Published_At"]:
#         if c not in df.columns:
#             st.error(f"Required column missing in pickle: {c}")
#             st.stop()

#     df = df.copy()
#     df["Comment"] = df["Comment"].astype(str)
#     df["Published_At"] = pd.to_datetime(df["Published_At"], errors="coerce")
#     df["Likes"] = pd.to_numeric(df.get("Likes", 0), errors="coerce").fillna(0).astype(int)
#     df["sentiment_label"] = df["sentiment_label"].astype(str)

#     return df

# df = load_data()

# # ------------------ Defaults storage ------------------
# defaults = {}
# original_defaults = {}   # <-- will store ORIGINAL values permanently

# def init_defaults():
#     defaults["date_from"] = df["Published_At"].min().date()
#     defaults["date_to"] = df["Published_At"].max().date()
#     defaults["selected_sentiments"] = ["Positive", "Neutral", "Negative"]
#     defaults["likes_min"] = 0
#     defaults["likes_max"] = int(df["Likes"].max())
#     defaults["search_text"] = ""

# init_defaults()

# # --------- Initialize missing session keys safely ---------
# if "date_from" not in st.session_state:
#     st.session_state["date_from"] = defaults["date_from"]

# if "date_to" not in st.session_state:
#     st.session_state["date_to"] = defaults["date_to"]

# if "selected_sentiments" not in st.session_state:
#     st.session_state["selected_sentiments"] = defaults["selected_sentiments"]

# if "likes_min" not in st.session_state:
#     st.session_state["likes_min"] = defaults["likes_min"]

# if "likes_max" not in st.session_state:
#     st.session_state["likes_max"] = defaults["likes_max"]

# if "search_text" not in st.session_state:
#     st.session_state["search_text"] = defaults["search_text"]

# if "orig" not in st.session_state:
#     st.session_state.orig = {
#         "date_from_input": defaults["date_from"],
#         "date_to_input": defaults["date_to"],
#         "sentiment_select": defaults["selected_sentiments"],
#         "likes_slider": (defaults["likes_min"], defaults["likes_max"]),
#         "search_input": defaults["search_text"]
#     }

# # Save ORIGINAL defaults permanently
# if "original_defaults_saved" not in st.session_state:
#     st.session_state.original_defaults = defaults.copy()
#     st.session_state.original_defaults_saved = True

# def clear_all_filters():
#     # Reset widget keys
#     st.session_state["date_from_input"] = st.session_state.orig["date_from_input"]
#     st.session_state["date_to_input"] = st.session_state.orig["date_to_input"]
#     st.session_state["sentiment_select"] = st.session_state.orig["sentiment_select"]
#     st.session_state["likes_slider"] = st.session_state.orig["likes_slider"]
#     st.session_state["search_input"] = st.session_state.orig["search_input"]

#     # Reset the actual filter keys
#     st.session_state["date_from"] = st.session_state.orig["date_from_input"]
#     st.session_state["date_to"] = st.session_state.orig["date_to_input"]
#     st.session_state["selected_sentiments"] = st.session_state.orig["sentiment_select"]
#     st.session_state["likes_min"], st.session_state["likes_max"] = st.session_state.orig["likes_slider"]
#     st.session_state["search_text"] = st.session_state.orig["search_input"]

# # ------------------ Sidebar Filters ------------------
# col1, col2 = st.sidebar.columns([1, 1])
# with col1:
#     st.header("📌 Filters")
# with col2:
#     st.button("🧹 Clear All Filters", on_click=clear_all_filters)

# # DATE FILTERS
# col1, col2 = st.sidebar.columns(2)

# new_from_date = col1.date_input(
#     "From",
#     value=st.session_state["date_from"],
#     key="date_from_input"
# )
# new_to_date = col2.date_input(
#     "To",
#     value=st.session_state["date_to"],
#     key="date_to_input"
# )

# if new_from_date > new_to_date:
#     st.sidebar.error("❌ 'From' date cannot be after 'To' date")
#     st.stop()

# st.session_state["date_from"] = new_from_date
# st.session_state["date_to"] = new_to_date

# # SENTIMENT FILTER
# sentiment_options = ["Positive", "Neutral", "Negative"]
# selected_sentiments = st.sidebar.multiselect(
#     "Select sentiment(s)",
#     options=sentiment_options,
#     default=st.session_state["selected_sentiments"],
#     key="sentiment_select"
# )

# st.session_state["selected_sentiments"] = selected_sentiments

# # ------------------ LIKES SLIDER (FULLY FIXED) ------------------

# # Ensure likes_slider key exists BEFORE reading it
# if "likes_slider" not in st.session_state:
#     st.session_state["likes_slider"] = (defaults["likes_min"], defaults["likes_max"])

# likes_min_default, likes_max_default = st.session_state["likes_slider"]

# likes_min, likes_max = st.sidebar.slider(
#     "Likes Range",
#     min_value=0,
#     max_value=int(df["Likes"].max()),
#     value=(likes_min_default, likes_max_default),
#     step=1,
#     key="likes_slider"
# )

# # Update session values from slider
# st.session_state["likes_min"] = likes_min
# st.session_state["likes_max"] = likes_max

# # SEARCH
# search_text = st.sidebar.text_input("Search comments by keyword", value=st.session_state["search_text"])
# st.session_state["search_text"] = search_text

# # ------------------ APPLY FILTERS ------------------
# mask = pd.Series(True, index=df.index)

# mask &= df["Published_At"].dt.date.between(
#     st.session_state["date_from"], st.session_state["date_to"]
# )

# mask &= df["sentiment_label"].isin(st.session_state["selected_sentiments"])

# mask &= df["Likes"].between(
#     st.session_state["likes_min"], st.session_state["likes_max"]
# )

# if st.session_state["search_text"].strip():
#     kw = st.session_state["search_text"].lower()
#     mask &= df["Comment"].str.lower().str.contains(kw, na=False)

# df_filtered = df[mask].reset_index(drop=True)

# if df_filtered.empty:
#     st.warning("⚠️ No data after applying filters. Try expanding the date range or clearing filters.")

# # ────────────── Layout: KPIs + Stacked Graphs (with right-side summaries) + Tables ──────────────
# st.markdown("### Summary")
# col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
# col_kpi1.metric("Comments shown", f"{len(df_filtered):,}")
# # avoid divide-by-zero in percent metrics
# total = len(df_filtered) if len(df_filtered)>0 else 1
# col_kpi2.metric("Positive %", f"{(df_filtered['sentiment_label']=='Positive').mean()*100:.1f}%")
# col_kpi3.metric("Neutral %", f"{(df_filtered['sentiment_label']=='Neutral').mean()*100:.1f}%")
# col_kpi4.metric("Negative %", f"{(df_filtered['sentiment_label']=='Negative').mean()*100:.1f}%")

# # ----- 1) Sentiment Distribution (bar) + summary on right -----
# st.markdown("---")
# st.subheader("1️⃣ Sentiment Distribution")
# gcol_left, gcol_right = st.columns([3,1])

# with gcol_left:
#     counts = df_filtered['sentiment_label'].value_counts().reindex(sentiment_options).fillna(0)
#     fig_bar = px.bar(
#         x=counts.index,
#         y=counts.values,
#         labels={"x":"Sentiment","y":"Count"},
#         title="Sentiment Counts",
#         color=counts.index,
#         color_discrete_map={"Positive":"#2ca02c","Neutral":"#7f7f7f","Negative":"#d62728"}
#     )
#     st.plotly_chart(fig_bar, use_container_width=True)

# with gcol_right:
#     st.markdown("#### Summary")
#     pos_cnt = int(counts.get("Positive", 0))
#     neu_cnt = int(counts.get("Neutral", 0))
#     neg_cnt = int(counts.get("Negative", 0))
#     st.write(f"- Total comments shown: **{len(df_filtered):,}**")
#     st.write(f"- Positive: **{pos_cnt}** ({(pos_cnt/total*100):.1f}% )")
#     st.write(f"- Neutral: **{neu_cnt}** ({(neu_cnt/total*100):.1f}% )")
#     st.write(f"- Negative: **{neg_cnt}** ({(neg_cnt/total*100):.1f}% )")
#     # quick top-line insight
#     if pos_cnt > neg_cnt:
#         st.info("Overall sentiment is more Positive than Negative for the selected filters.")
#     elif neg_cnt > pos_cnt:
#         st.info("Overall sentiment is more Negative than Positive for the selected filters.")
#     else:
#         st.info("Positive and Negative counts are similar.")

# # ----- 2) Sentiment Share (pie) — stacked underneath, with summary -----
# st.markdown("---")
# st.subheader("2️⃣ Sentiment Share")
# gcol_left, gcol_right = st.columns([3,1])
# with gcol_left:
#     fig_pie = px.pie(names=counts.index, values=counts.values, title="Sentiment Share",
#                      color=counts.index, color_discrete_map={"Positive":"#2ca02c","Neutral":"#7f7f7f","Negative":"#d62728"})
#     st.plotly_chart(fig_pie, use_container_width=True)
# with gcol_right:
#     st.markdown("#### Summary")
#     st.write("Pie shows proportion of sentiment classes for current filters.")
#     st.write("- Useful to see distribution at-a-glance.")
#     top = counts.idxmax() if not counts.empty else "N/A"
#     st.write(f"- Largest segment: **{top}**")

# # ----- 3) Monthly Sentiment Trend (line) + summary -----
# st.markdown("---")
# st.subheader("3️⃣ Monthly Sentiment Trend")
# gcol_left, gcol_right = st.columns([3,1])
# with gcol_left:
#     if 'Published_At' in df_filtered.columns and not df_filtered['Published_At'].isna().all():
#         df_month = df_filtered.copy()
#         df_month['month'] = df_month['Published_At'].dt.to_period('M').dt.to_timestamp()
#         monthly = df_month.groupby(['month','sentiment_label']).size().unstack(fill_value=0)
#         if not monthly.empty:
#             fig_trend = px.line(monthly.reset_index(), x='month', y=monthly.columns, title='Monthly Comment Count by Sentiment')
#             st.plotly_chart(fig_trend, use_container_width=True)
#         else:
#             st.info("No monthly data to plot.")
#     else:
#         st.info("Published_At column missing or invalid.")

# with gcol_right:
#     st.markdown("#### Summary")

#     # Check if 'monthly' exists and has data
#     if 'monthly' in locals() and isinstance(monthly, pd.DataFrame) and not monthly.empty:
#         recent_month = monthly.index.max()
#         recent_counts = monthly.loc[recent_month].to_dict()

#         st.write(f"- Latest month: **{recent_month.date()}**")
#         for s, v in recent_counts.items():
#             st.write(f"  - {s}: {v}")

#         # Trend comparison
#         if len(monthly.index) >= 2:
#             last = monthly.sum(axis=1).iloc[-1]
#             prev = monthly.sum(axis=1).iloc[-2]

#             if last > prev:
#                 st.success("Comments increased in the latest month compared to previous month.")
#             elif last < prev:
#                 st.warning("Comments decreased in the latest month compared to previous month.")
#             else:
#                 st.info("Comment volume stable month-over-month.")
#     else:
#         st.write("- No monthly summary available.")

# # ----- 4) Likes vs Sentiment (boxplot) + summary -----
# st.markdown("---")
# st.subheader("4️⃣ Likes vs Sentiment")
# gcol_left, gcol_right = st.columns([3,1])
# with gcol_left:
#     if 'Likes' in df_filtered.columns and not df_filtered['Likes'].isna().all():
#         fig_box = px.box(df_filtered, x='sentiment_label', y='Likes', title='Likes Distribution by Sentiment',
#                          category_orders={"sentiment_label": sentiment_options})
#         st.plotly_chart(fig_box, use_container_width=True)
#     else:
#         st.info("Likes column missing or all zero.")
# with gcol_right:
#     st.markdown("#### Summary")
#     if 'Likes' in df_filtered.columns and not df_filtered['Likes'].isna().all():
#         medians = df_filtered.groupby('sentiment_label')['Likes'].median().reindex(sentiment_options).fillna(0).to_dict()
#         for s, m in medians.items():
#             st.write(f"- Median Likes ({s}): **{int(m)}**")
#         st.write("- Boxplot helps spot outliers and typical engagement per sentiment.")
#     else:
#         st.write("- No likes summary available.")

# # ────────────── Top Positive and Top Negative (stacked vertically) ──────────────
# st.markdown("---")
# st.subheader("Top Positive Comments (filtered)")
# pos_table = df_filtered[df_filtered['sentiment_label']=='Positive'].sort_values(by=['sentiment_score','Likes'], ascending=[False, False]).head(10)
# if not pos_table.empty:
#     display_cols = ['Published_At','Author','Comment','sentiment_score','Likes'] if 'Author' in pos_table.columns else ['Published_At','Comment','sentiment_score','Likes']
#     st.dataframe(pos_table[display_cols], use_container_width=True)
#     csv = pos_table.to_csv(index=False)
#     st.download_button("Download Top Positive CSV", data=csv, file_name="top_positive_comments.csv")
# else:
#     st.info("No positive comments for current filters.")

# st.markdown("---")
# st.subheader("Top Negative Comments (filtered)")
# neg_table = df_filtered[df_filtered['sentiment_label']=='Negative'].sort_values(by=['sentiment_score','Likes'], ascending=[False, False]).head(10)
# if not neg_table.empty:
#     display_cols = ['Published_At','Author','Comment','sentiment_score','Likes'] if 'Author' in neg_table.columns else ['Published_At','Comment','sentiment_score','Likes']
#     st.dataframe(neg_table[display_cols], use_container_width=True)
#     csvn = neg_table.to_csv(index=False)
#     st.download_button("Download Top Negative CSV", data=csvn, file_name="top_negative_comments.csv")
# else:
#     st.info("No negative comments for current filters.")

# -----------------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import datetime
# import os

# # ────────────── Streamlit page setup ──────────────
# st.set_page_config(page_title="YouTube Sentiment Dashboard", page_icon="🎬", layout="wide")
# st.title("🎬 YouTube Sentiment Dashboard")

# # ────────────── Helpers & Load ──────────────
# @st.cache_data(show_spinner=False)
# def load_data() -> pd.DataFrame:

#     # Determine correct absolute path
#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     pkl_path = os.path.join(base_dir, "models", "youtube_comments_sentiment.pkl")

#     # st.write("Loading pickle from:", pkl_path)  # Optional debug

#     if not os.path.exists(pkl_path):
#         st.error(f"❌ Pickle file not found at:\n{pkl_path}")
#         st.stop()

#     df = pd.read_pickle(pkl_path)

#     # ensure required columns
#     for c in ["Comment", "sentiment_label", "sentiment_score", "Published_At"]:
#         if c not in df.columns:
#             st.error(f"Required column missing in pickle: {c}")
#             st.stop()

#     # normalize types
#     df = df.copy()
#     df["Comment"] = df["Comment"].astype(str)
#     df["Published_At"] = pd.to_datetime(df["Published_At"], errors="coerce")
#     df["Likes"] = pd.to_numeric(df.get("Likes", 0), errors="coerce").fillna(0).astype(int)
#     df["sentiment_label"] = df["sentiment_label"].astype(str)

#     return df

# # Load
# df = load_data()

# # ────────────── Sidebar: Filters & Session State ──────────────
# st.sidebar.header("📌 Filters")

# # Clear all filters function
# defaults = {}

# def init_defaults():
#     defaults["date_from"] = df["Published_At"].min().date() if not df["Published_At"].isna().all() else datetime.date.today()
#     defaults["date_to"] = df["Published_At"].max().date() if not df["Published_At"].isna().all() else datetime.date.today()
#     defaults["selected_sentiments"] = ["Positive", "Neutral", "Negative"]
#     defaults["likes_min"] = 0
#     defaults["likes_max"] = int(df["Likes"].max() if "Likes" in df.columns else 0)
#     defaults["search_text"] = ""

# init_defaults()

# # initialize session state with defaults if missing
# for k, v in defaults.items():
#     if k not in st.session_state:
#         st.session_state[k] = v

# # Clear button
# col1, col2 = st.sidebar.columns([3,1])
# col1.write("")
# if col2.button("🧹 Clear All Filters"):
#     for k, v in defaults.items():
#         st.session_state[k] = v

# # Date range inputs
# st.sidebar.subheader("Date range")
# from_date = st.sidebar.date_input("From", value=st.session_state["date_from"], key="date_from_input")
# to_date = st.sidebar.date_input("To", value=st.session_state["date_to"], key="date_to_input")

# if from_date > to_date:
#     st.sidebar.error("'From' date cannot be after 'To' date")

# st.session_state["date_from"] = from_date
# st.session_state["date_to"] = to_date

# # Sentiment multiselect
# st.sidebar.subheader("Sentiment")
# sentiment_options = ["Positive", "Neutral", "Negative"]
# selected_sentiments = st.sidebar.multiselect(
#     "Select sentiment(s)",
#     options=sentiment_options,
#     default=st.session_state.get("selected_sentiments", sentiment_options),
#     key="sentiment_select"
# )
# if not selected_sentiments:
#     st.sidebar.warning("Select at least one sentiment to view charts")

# st.session_state["selected_sentiments"] = selected_sentiments

# # Likes slider
# st.sidebar.subheader("Likes range")
# likes_min_default = st.session_state.get("likes_min", 0)
# likes_max_default = st.session_state.get("likes_max", int(df["Likes"].max() if "Likes" in df.columns else 0))
# likes_min, likes_max = st.sidebar.slider(
#     "Likes",
#     min_value=0,
#     max_value=likes_max_default if likes_max_default>0 else 100,
#     value=(likes_min_default, likes_max_default if likes_max_default>0 else 100),
#     step=1,
#     key="likes_slider"
# )
# st.session_state["likes_min"] = likes_min
# st.session_state["likes_max"] = likes_max

# # Search box
# st.sidebar.subheader("Search comments")
# search_text = st.sidebar.text_input("Keyword or phrase", value=st.session_state.get("search_text", ""), key="search_input")
# st.session_state["search_text"] = search_text

# # ────────────── Apply Filters ──────────────
# mask = pd.Series(True, index=df.index)

# # Date filter
# if not df["Published_At"].isna().all():
#     mask &= df["Published_At"].dt.date.between(st.session_state["date_from"], st.session_state["date_to"])

# # Sentiment filter
# if selected_sentiments:
#     mask &= df["sentiment_label"].isin(selected_sentiments)

# # Likes filter
# mask &= df["Likes"].between(st.session_state["likes_min"], st.session_state["likes_max"])

# # Search filter
# if st.session_state["search_text"].strip():
#     kw = st.session_state["search_text"].strip().lower()
#     mask &= df["Comment"].str.lower().str.contains(kw, na=False)

# df_filtered = df[mask].reset_index(drop=True)

# if df_filtered.empty:
#     st.warning("⚠️ No data after applying filters. Try expanding the date range or clearing filters.")

# # ────────────── Layout: KPIs + Charts + Tables ──────────────
# st.markdown("### Summary")
# col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
# col_kpi1.metric("Comments shown", f"{len(df_filtered):,}")
# col_kpi2.metric("Positive %", f"{(df_filtered['sentiment_label']=='Positive').mean()*100:.1f}%")
# col_kpi3.metric("Neutral %", f"{(df_filtered['sentiment_label']=='Neutral').mean()*100:.1f}%")
# col_kpi4.metric("Negative %", f"{(df_filtered['sentiment_label']=='Negative').mean()*100:.1f}%")

# # Sentiment distribution bar
# st.markdown("---")
# left_col, right_col = st.columns([2,1])
# with left_col:
#     st.subheader("Sentiment Distribution")
#     counts = df_filtered['sentiment_label'].value_counts().reindex(sentiment_options).fillna(0)
#     fig_bar = px.bar(
#         x=counts.index,
#         y=counts.values,
#         labels={"x":"Sentiment","y":"Count"},
#         title="Sentiment Counts",
#         color=counts.index,
#         color_discrete_map={"Positive":"#2ca02c","Neutral":"#7f7f7f","Negative":"#d62728"}
#     )
#     st.plotly_chart(fig_bar, use_container_width=True)

# with right_col:
#     st.subheader("Sentiment Share")
#     fig_pie = px.pie(names=counts.index, values=counts.values, title="Sentiment Share",
#                      color=counts.index, color_discrete_map={"Positive":"#2ca02c","Neutral":"#7f7f7f","Negative":"#d62728"})
#     st.plotly_chart(fig_pie, use_container_width=True)

# # Monthly trend
# st.subheader("Monthly Sentiment Trend")
# if 'Published_At' in df_filtered.columns and not df_filtered['Published_At'].isna().all():
#     df_filtered['month'] = df_filtered['Published_At'].dt.to_period('M').dt.to_timestamp()
#     monthly = df_filtered.groupby(['month','sentiment_label']).size().unstack(fill_value=0)
#     if not monthly.empty:
#         fig_trend = px.line(monthly.reset_index(), x='month', y=monthly.columns, title='Monthly Comment Count by Sentiment')
#         st.plotly_chart(fig_trend, use_container_width=True)
#     else:
#         st.info("No monthly data to plot.")
# else:
#     st.info("Published_At column missing or invalid.")

# # Likes vs Sentiment boxplot
# st.subheader("Likes vs Sentiment")
# if 'Likes' in df_filtered.columns and not df_filtered['Likes'].isna().all():
#     fig_box = px.box(df_filtered, x='sentiment_label', y='Likes', title='Likes Distribution by Sentiment',
#                      category_orders={"sentiment_label": sentiment_options})
#     st.plotly_chart(fig_box, use_container_width=True)
# else:
#     st.info("Likes column missing or all zero.")

# # Top Positive and Negative tables
# st.markdown("---")
# st.subheader("Top Positive & Negative Comments (filtered)")

# # Prepare tables
# pos_table = df_filtered[df_filtered['sentiment_label']=='Positive'].sort_values(by=['sentiment_score','Likes'], ascending=[False, False]).head(10)
# neg_table = df_filtered[df_filtered['sentiment_label']=='Negative'].sort_values(by=['sentiment_score','Likes'], ascending=[False, False]).head(10)

# colp, coln = st.columns(2)
# with colp:
#     st.markdown("**Top Positive Comments**")
#     if not pos_table.empty:
#         st.dataframe(pos_table[['Published_At','Author'] + ['Comment','sentiment_score','Likes'] if 'Author' in pos_table.columns else ['Published_At','Comment','sentiment_score','Likes']], use_container_width=True)
#         csv = pos_table.to_csv(index=False)
#         st.download_button("Download Top Positive CSV", data=csv, file_name="top_positive_comments.csv")
#     else:
#         st.info("No positive comments in filtered data.")

# with coln:
#     st.markdown("**Top Negative Comments**")
#     if not neg_table.empty:
#         st.dataframe(neg_table[['Published_At','Author'] + ['Comment','sentiment_score','Likes'] if 'Author' in neg_table.columns else ['Published_At','Comment','sentiment_score','Likes']], use_container_width=True)
#         csvn = neg_table.to_csv(index=False)
#         st.download_button("Download Top Negative CSV", data=csvn, file_name="top_negative_comments.csv")
#     else:
#         st.info("No negative comments in filtered data.")

# # Raw data viewer with search and pagination
# st.markdown("---")
# st.subheader("Raw Comments (preview)")
# if not df_filtered.empty:
#     st.dataframe(df_filtered[['Published_At','sentiment_label','sentiment_score','Likes','Comment']].rename(columns={
#         'Published_At':'Date','sentiment_label':'Sentiment','sentiment_score':'Score'
#     }), height=400)

# # Footer / tips
# st.markdown("---")
# st.write("Tips: Use the filters on the left to narrow results. Click 'Clear All Filters' to reset.")

# # End
