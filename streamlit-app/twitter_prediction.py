import streamlit as st
import pandas as pd
import numpy as np
import pickle
from textblob import TextBlob
from scipy.sparse import hstack
from datetime import datetime
import os
import datetime as dt
from dotenv import load_dotenv

# ───────────────────────────────
# Page Config
# ───────────────────────────────
st.set_page_config(
    page_title="Twitter Engagement Predictor",
    page_icon="🐦",
    layout="centered"
)

st.title("🐦 Twitter Engagement Predictor")
st.write(
    "Predict **Retweets, Replies, Likes, Quotes, Bookmarks & Impressions** "
    "based on tweet content and posting time."
)

# ───────────────────────────────
# Load Models
# ───────────────────────────────
load_dotenv()

MODEL_PATH = os.getenv("TWITTER_MODEL_PATH")
TFIDF_PATH = os.getenv("TWITTER_TFIDF_PATH")

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    model = load_pickle(MODEL_PATH)
    tfidf = load_pickle(TFIDF_PATH)
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.stop()

# ───────────────────────────────
# Sidebar Inputs
# ───────────────────────────────
st.sidebar.header("✏️ Tweet Inputs")

tweet_text = st.sidebar.text_area(
    "Tweet Text",
    value="Amazing trailer! What do you think? 🇮🇳🔥",
    height=120
)

hashtags = st.sidebar.text_input(
    "Hashtags (comma-separated)",
    "#Trailer,#IndependenceDay"
)

mentions = st.sidebar.text_input(
    "Mentions (comma-separated)",
    "@actor1,@actor2"
)

tweet_date = st.sidebar.date_input(
    "Tweet Date",
    value=dt.date.today()
)

tweet_time = st.sidebar.time_input(
    "Tweet Time",
    value=dt.datetime.now().time()
)

created_at = dt.datetime.combine(tweet_date, tweet_time)

# ───────────────────────────────
# Feature Engineering
# ───────────────────────────────
def build_features(text, created_at, mentions, hashtags):
    text_length = len(text)
    word_count = len(text.split())
    exclamation_count = text.count("!")
    question_count = text.count("?")
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

    sentiment = TextBlob(text).sentiment.polarity

    hashtag_count = len(hashtags.split(",")) if hashtags else 0
    mention_count = len(mentions.split(",")) if mentions else 0

    hour = created_at.hour
    day_of_week = created_at.weekday()

    numeric_features = np.array([
        text_length,
        word_count,
        exclamation_count,
        question_count,
        uppercase_ratio,
        sentiment,
        hashtag_count,
        mention_count,
        hour,
        day_of_week
    ]).reshape(1, -1)

    text_vector = tfidf.transform([text])
    final_input = hstack([text_vector, numeric_features])

    return final_input

# ───────────────────────────────
# Prediction
# ───────────────────────────────
if st.button("🚀 Predict Engagement"):
    X_input = build_features(
        tweet_text,
        created_at,
        mentions,
        hashtags
    )

    preds = model.predict(X_input)[0]

    st.subheader("📊 Predicted Engagement")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    col1.metric("🔁 Retweets", f"{int(preds[0]):,}")
    col2.metric("💬 Replies", f"{int(preds[1]):,}")
    col3.metric("❤️ Likes", f"{int(preds[2]):,}")

    col4.metric("🔄 Quotes", f"{int(preds[3]):,}")
    col5.metric("🔖 Bookmarks", f"{int(preds[4]):,}")
    col6.metric("👁️ Impressions", f"{int(preds[5]):,}")
