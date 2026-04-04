import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ────────────── Page Setup ──────────────
st.set_page_config(
    page_title="Twitter Sentiment Dashboard",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Twitter Sentiment Dashboard")
st.caption("RoBERTa-based sentiment analysis (cardiffnlp)")

# ────────────── Load Data ──────────────
@st.cache_data(show_spinner=False)
def load_data():
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(CURRENT_DIR, "data")
    path = os.path.join(DATA_DIR, "twitter_sentiment.pkl")

    if not os.path.exists(path):
        st.error(f"❌ Pickle file not found: {path}")
        st.stop()

    df = pd.read_pickle(path)
    df["text"] = df["text"].astype(str)

    return df

df = load_data()

# ────────────── Sidebar Filters ──────────────
st.sidebar.header("📌 Filters")

sentiment_options = ["Positive", "Neutral", "Negative"]
selected_sentiments = st.sidebar.multiselect(
    "Select sentiment(s)",
    sentiment_options,
    default=sentiment_options
)

confidence_threshold = st.sidebar.slider(
    "Minimum confidence score",
    0.0, 1.0, 0.0, 0.05
)

keyword = st.sidebar.text_input("Search tweet text")

# ────────────── Apply Filters ──────────────
df_filtered = df[df["sentiment_label"].isin(selected_sentiments)]

df_filtered = df_filtered[
    df_filtered["sentiment_score"] >= confidence_threshold
]

if keyword.strip():
    df_filtered = df_filtered[
        df_filtered["text"].str.lower().str.contains(keyword.lower())
    ]

if df_filtered.empty:
    st.warning("⚠️ No tweets after applying filters.")

# ────────────── KPIs ──────────────
st.markdown("### 📊 Summary")
col1, col2, col3, col4 = st.columns(4)

total = len(df_filtered) if len(df_filtered) > 0 else 1

col1.metric("Tweets Shown", f"{len(df_filtered):,}")
col2.metric("Positive %", f"{(df_filtered['sentiment_label']=='Positive').mean()*100:.1f}%")
col3.metric("Neutral %", f"{(df_filtered['sentiment_label']=='Neutral').mean()*100:.1f}%")
col4.metric("Negative %", f"{(df_filtered['sentiment_label']=='Negative').mean()*100:.1f}%")

# ────────────── 1️⃣ Sentiment Distribution ──────────────
st.markdown("---")
st.subheader("1️⃣ Sentiment Distribution")

counts = df_filtered["sentiment_label"].value_counts().reindex(sentiment_options).fillna(0)

fig_bar = px.bar(
    x=counts.index,
    y=counts.values,
    labels={"x": "Sentiment", "y": "Count"},
    title="Sentiment Distribution",
    color=counts.index,
    color_discrete_map={
        "Positive": "#2ca02c",
        "Neutral": "#7f7f7f",
        "Negative": "#d62728"
    }
)
st.plotly_chart(fig_bar, use_container_width=True)

# ────────────── 2️⃣ Positive Confidence Histogram ──────────────
st.markdown("---")
st.subheader("2️⃣ Sentiment Confidence Histogram")

fig_hist = px.histogram(
    df_filtered,
    x="sentiment_score",
    nbins=30,
    title="Sentiment Confidence Distribution"
)
st.plotly_chart(fig_hist, use_container_width=True)

# ────────────── 3️⃣ Negative vs Positive Scatter ──────────────
if {"neg_score", "pos_score"}.issubset(df.columns):

    st.markdown("---")
    st.subheader("3️⃣ Negative vs Positive Confidence")

    fig_scatter = px.scatter(
        df_filtered,
        x="neg_score",
        y="pos_score",
        color="sentiment_label",
        title="Negative vs Positive Confidence",
        hover_data=["text"]
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ────────────── 4️⃣ Confidence Spread Boxplot ──────────────
if {"neg_score", "neu_score", "pos_score"}.issubset(df.columns):

    st.markdown("---")
    st.subheader("4️⃣ Confidence Spread by Sentiment")

    confidence_df = df_filtered.melt(
        id_vars="sentiment_label",
        value_vars=["neg_score", "neu_score", "pos_score"],
        var_name="Score Type",
        value_name="Score"
    )

    fig_box = px.box(
        confidence_df,
        x="sentiment_label",
        y="Score",
        color="Score Type",
        title="Sentiment Confidence Distribution"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ────────────── Top Tweets ──────────────
st.markdown("---")
st.subheader("🔥 Top Positive Tweets")

top_positive = (
    df_filtered[df_filtered["sentiment_label"] == "Positive"]
    .sort_values("sentiment_score", ascending=False)
    .head(10)
)

if not top_positive.empty:
    st.dataframe(
        top_positive[["text", "sentiment_score"]],
        use_container_width=True
    )
    st.download_button(
        "Download Top Positive Tweets",
        data=top_positive.to_csv(index=False),
        file_name="top_positive_tweets.csv"
    )
else:
    st.info("No strong positive tweets.")

st.markdown("---")
st.subheader("⚠️ Top Negative Tweets")

top_negative = (
    df_filtered[df_filtered["sentiment_label"] == "Negative"]
    .sort_values("sentiment_score", ascending=False)
    .head(10)
)

if not top_negative.empty:
    st.dataframe(
        top_negative[["text", "sentiment_score"]],
        use_container_width=True
    )
    st.download_button(
        "Download Top Negative Tweets",
        data=top_negative.to_csv(index=False),
        file_name="top_negative_tweets.csv"
    )
else:
    st.info("No strong negative tweets.")
