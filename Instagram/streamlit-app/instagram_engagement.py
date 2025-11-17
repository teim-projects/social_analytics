import streamlit as st
import pandas as pd
import pickle
from dotenv import load_dotenv
import os

# ────────────── Streamlit page setup ──────────────
st.set_page_config(page_title="Predict Instagram Post Engagement", page_icon="🤖", layout="centered")
st.title("🤖 Predict Instagram Post Engagement")

# ────────────── Load the trained Random Forest model ──────────────
load_dotenv()  # Load environment variables from .env file
MODEL_PATH = os.getenv("MODEL_PATH")

try:
    with open(MODEL_PATH, "rb") as file:
        rf_model = pickle.load(file)
    # st.success(f"✅ Model loaded from: {MODEL_PATH}")
except FileNotFoundError:
    st.error("🚫 Model file not found! Please check the path.")
    st.stop()

# ────────────── User Input ──────────────
st.sidebar.header("📌 Enter Post Details")

# Assuming these are the columns/features used in your model
feature_inputs = {
    "Media_Type": st.sidebar.selectbox("Media Type (1=Image, 2=Video, etc.)", [1, 2, 3], index=0),
    "Followers": st.sidebar.number_input("Number of Followers", min_value=0, value=5000, step=100),
    "Post_Year": st.sidebar.number_input("Post Year", min_value=2000, max_value=2100, value=2025),
    "Post_Month": st.sidebar.number_input("Post Month", min_value=1, max_value=12, value=10),
    "Post_Day": st.sidebar.number_input("Post Day", min_value=1, max_value=31, value=3),
    "Post_Weekday": st.sidebar.selectbox("Day of the Week (0=Mon, 6=Sun)", list(range(7)), index=5),
    "Post_Hour": st.sidebar.number_input("Post Hour (0-23)", min_value=0, max_value=23, value=18)
}

# Create a DataFrame for prediction
new_post = pd.DataFrame([feature_inputs], columns=rf_model.feature_names_in_)

# ────────────── Make Prediction ──────────────
if st.button("Predict Engagement [Likes, Comments, Shares]"):
    predicted_values = rf_model.predict(new_post)
    predicted_values = predicted_values[0]  # Get array from prediction
   
    st.subheader("📈 Predicted Engagement Metrics")
    st.write(f"Likes: {int(predicted_values[0])}")
    st.write(f"Comments: {int(predicted_values[1])}")
    st.write(f"Shares: {int(predicted_values[2])}")