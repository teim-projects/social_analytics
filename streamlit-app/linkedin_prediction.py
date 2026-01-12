import streamlit as st
import pandas as pd
import pickle
import os
from dotenv import load_dotenv

# ───────────────────────────────
# Page Config
# ───────────────────────────────
st.set_page_config(
    page_title="LinkedIn Engagement Predictor",
    page_icon="💼",
    layout="centered"
)

st.title("💼 LinkedIn Engagement Predictor")
st.write(
    "Predict **Likes, Comments & Shares** for a LinkedIn post "
    "based on content, hashtags, and posting context."
)

# ───────────────────────────────
# Load Model
# ───────────────────────────────
load_dotenv()

PIPELINE_PATH = os.getenv("LINKEDIN_MODEL_PATH")

@st.cache_resource
def load_pipeline(path):
    with open(path, "rb") as f:
        return pickle.load(f)

try:
    pipeline = load_pipeline(PIPELINE_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ───────────────────────────────
# Sidebar Inputs
# ───────────────────────────────
st.sidebar.header("✍️ Post Details")

post_text = st.sidebar.text_area(
    "Post Text",
    height=180,
    value=(
        "✨ A Full-Circle Moment — Code-IT 8th Edition ✨\n\n"
        "From being a participant in 2023 to leading Code-IT this year, "
        "this journey has been about growth, leadership, and community."
    )
)

hashtags = st.sidebar.text_input(
    "Hashtags (space or comma separated)",
    "#WomenInTech #Leadership #Community"
)

has_media = st.sidebar.selectbox(
    "Includes Media?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

days_since_post = st.sidebar.number_input(
    "Days Since Posting",
    min_value=0,
    max_value=365,
    value=1
)

# ───────────────────────────────
# Prediction
# ───────────────────────────────
if st.button("🚀 Predict Engagement"):
    input_df = pd.DataFrame([{
        "text": post_text,
        "hashtags": hashtags,
        "text_length": len(post_text),
        "word_count": len(post_text.split()),
        "has_media": has_media,
        "days_since_post": days_since_post
    }])

    preds = pipeline.predict(input_df)[0]

    st.subheader("📊 Predicted Engagement")

    col1, col2, col3 = st.columns(3)

    col1.metric("👍 Likes", f"{int(preds[0]):,}")
    col2.metric("💬 Comments", f"{int(preds[1]):,}")
    col3.metric("🔁 Shares", f"{int(preds[2]):,}")
