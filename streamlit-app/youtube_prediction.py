# import streamlit as st
# import pandas as pd
# import pickle
# import os
# from dotenv import load_dotenv

# # ───────────────────────────────
# # Streamlit UI
# # ───────────────────────────────
# st.set_page_config(page_title="YouTube Video Engagement Predictor", page_icon="📊", layout="centered")
# st.title("📊 YouTube Video Engagement Predictor")

# st.write("Enter video details on the left panel to predict Views, Likes, and Comments using your trained ML models.")

# # ───────────────────────────────
# # Load Models
# # ───────────────────────────────
# @st.cache_resource
# def load_model(path):
#     try:
#         with open(path, "rb") as file:
#             return pickle.load(file)
#     except:
#         st.error(f"❌ Could not load model: {path}")
#         return None

# load_dotenv()
# model_views_path = os.getenv("YOUTUBE_MODEL_VIEWS")
# model_likes_path = os.getenv("YOUTUBE_MODEL_LIKES") 
# model_comments_path = os.getenv("YOUTUBE_MODEL_COMMENTS")

# model_views = load_model(model_views_path)
# model_likes = load_model(model_likes_path)
# model_comments = load_model(model_comments_path)

# if not (model_views and model_likes and model_comments):
#     st.stop()

# # ───────────────────────────────
# # Sidebar User Inputs
# # ───────────────────────────────
# st.sidebar.header("📌 Enter Video Features")

# title_len = st.sidebar.number_input("Title Length", min_value=0, value=40)
# desc_len = st.sidebar.number_input("Description Length", min_value=0, value=120)
# tags_len = st.sidebar.number_input("Number of Tags", min_value=0, value=10)

# year = st.sidebar.number_input("Year", min_value=2005, max_value=2100, value=2025)
# month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=2)
# day = st.sidebar.number_input("Day of the Month", min_value=1, max_value=31, value=6)
# weekday = st.sidebar.selectbox("Weekday (0=Mon, 6=Sun)", list(range(7)), index=4)
# hour = st.sidebar.number_input("Hour (0-23)", min_value=0, max_value=23, value=18)

# # Features for prediction
# feature_dict = {
#     "title_len": title_len,
#     "desc_len": desc_len,
#     "tags_len": tags_len,
#     "year": year,
#     "month": month,
#     "day": day,
#     "weekday": weekday,
#     "hour": hour
# }

# # Convert to DataFrame
# input_df = pd.DataFrame([feature_dict], columns=model_views.feature_names_in_)

# # ───────────────────────────────
# # Prediction Section
# # ───────────────────────────────
# if st.button("Predict Engagement 🚀"):
#     pred_views = int(model_views.predict(input_df)[0])
#     pred_likes = int(model_likes.predict(input_df)[0])
#     pred_comments = int(model_comments.predict(input_df)[0])

#     st.subheader("📈 Predicted Engagement")
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         st.metric("Views", f"{pred_views:,}")

#     with col2:
#         st.metric("Likes", f"{pred_likes:,}")

#     with col3:
#         st.metric("Comments", f"{pred_comments:,}")

import streamlit as st
import pandas as pd
import pickle
import os
from dotenv import load_dotenv

# ───────────────────────────────
# PAGE CONFIG
# ───────────────────────────────
st.set_page_config(
    page_title="YouTube Video Engagement Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 YouTube Video Engagement Predictor")
st.write(
    "Enter video metadata to predict **Views, Likes, and Comments** "
    "using a trained machine learning model."
)

# ───────────────────────────────
# LOAD MODEL & ENCODERS
# ───────────────────────────────
load_dotenv()

MODEL_PATH = os.getenv("YOUTUBE_MODEL")
ENCODERS_PATH = os.getenv("YOUTUBE_ENCODERS")
MLB_PATH = os.getenv("YOUTUBE_MLB")

@st.cache_resource
def load_pickle(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except:
        st.error(f"❌ Failed to load: {path}")
        return None

model = load_pickle(MODEL_PATH)
encoders = load_pickle(ENCODERS_PATH)
mlb = load_pickle(MLB_PATH)

if not (model and encoders and mlb):
    st.stop()

# ───────────────────────────────
# SIDEBAR INPUTS (SESSION STYLE)
# ───────────────────────────────
st.sidebar.header("📌 Video Details")

channel_title = st.sidebar.selectbox(
    "Channel Title",
    encoders["Channel_Title"].classes_
)

publish_year = st.sidebar.number_input(
    "Publish Year", min_value=2005, max_value=2100, value=2025
)

publish_month = st.sidebar.selectbox(
    "Publish Month", list(range(1, 13)), index=2
)

publish_day = st.sidebar.number_input(
    "Publish Day", min_value=1, max_value=31, value=6
)

publish_weekday = st.sidebar.selectbox(
    "Publish Weekday (0 = Monday)",
    list(range(7)),
    index=3
)

publish_hour = st.sidebar.number_input(
    "Publish Hour (0–23)", min_value=0, max_value=23, value=18
)

duration_sec = st.sidebar.number_input(
    "Video Duration (seconds)", min_value=1, value=1200
)

tags_input = st.sidebar.text_input(
    "Tags (comma-separated)",
    value="challenge, funny, family friendly"
)

definition = st.sidebar.selectbox(
    "Video Definition",
    encoders["Definition"].classes_
)

privacy_status = st.sidebar.selectbox(
    "Privacy Status",
    encoders["Privacy_Status"].classes_
)

caption = st.sidebar.checkbox("Has Captions", value=True)
embeddable = st.sidebar.checkbox("Embeddable", value=True)
made_for_kids = st.sidebar.checkbox("Made for Kids", value=False)

# Convert tags to list
tags = [t.strip().lower() for t in tags_input.split(",") if t.strip()]

# ───────────────────────────────
# PREDICTION FUNCTION
# ───────────────────────────────
def predict_engagement():
    base_data = {
        "Channel_Title": encoders["Channel_Title"].transform([channel_title])[0],
        "publish_year": publish_year,
        "publish_month": publish_month,
        "publish_day": publish_day,
        "publish_weekday": publish_weekday,
        "publish_hour": publish_hour,
        "duration_sec": duration_sec,
        "Definition": encoders["Definition"].transform([definition])[0],
        "Caption": int(caption),
        "Privacy_Status": encoders["Privacy_Status"].transform([privacy_status])[0],
        "Embeddable": int(embeddable),
        "Made_For_Kids": int(made_for_kids),
    }

    X_base = pd.DataFrame([base_data])

    tag_vector = mlb.transform([tags])
    tag_df = pd.DataFrame(
        tag_vector,
        columns=[f"tag_{t}" for t in mlb.classes_]
    )

    X_final = pd.concat([X_base, tag_df], axis=1)

    preds = model.predict(X_final)[0]

    return {
        "Views": int(preds[0]),
        "Likes": int(preds[1]),
        "Comments": int(preds[2]),
    }

# ───────────────────────────────
# PREDICT BUTTON
# ───────────────────────────────
if st.button("🚀 Predict Engagement"):
    result = predict_engagement()

    st.subheader("📈 Predicted Engagement")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Views", f"{result['Views']:,}")

    with col2:
        st.metric("Likes", f"{result['Likes']:,}")

    with col3:
        st.metric("Comments", f"{result['Comments']:,}")
