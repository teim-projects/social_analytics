import streamlit as st
import pandas as pd
import pickle
import os

# ────────────── Streamlit page setup ──────────────
st.set_page_config(page_title="Predict Instagram Engagement", page_icon="🤖")

st.title("🤖 Predict Instagram Post Engagement")

# ────────────── Load RF Model ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(CURRENT_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        rf_model = pickle.load(f)
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

model_features = list(rf_model.feature_names_in_)  # model-required columns

# ────────────── User Inputs ──────────────
st.sidebar.header("📌 Enter Post Details")

user_inputs = {
    "Media_Type": st.sidebar.selectbox("Media Type", [1, 2], index=0),
    "Followers": st.sidebar.number_input("Followers", 0, 10_000_000, 5000),
    "Post_Year": st.sidebar.number_input("Year", 2000, 2100, 2025),
    "Post_Month": st.sidebar.number_input("Month", 1, 12, 10),
    "Post_Day": st.sidebar.number_input("Day", 1, 31, 3),
    "Post_Weekday": st.sidebar.selectbox("Weekday (0=Mon)", list(range(7)), index=5),
    "Post_Hour": st.sidebar.number_input("Hour", 0, 23, 18)
}

# Convert to DataFrame with ALL required model columns
df_input = pd.DataFrame(columns=model_features)
df_input.loc[0] = 0  # fill with zeros
for col in user_inputs:
    if col in df_input.columns:
        df_input.at[0, col] = user_inputs[col]

# ────────────── Predict ──────────────
if st.button("Predict Engagement"):
    try:
        pred = rf_model.predict(df_input)[0]
        st.subheader("📈 Predicted Engagement")
        st.write(f"Likes: {int(pred[0])}")
        st.write(f"Comments: {int(pred[1])}")
        st.write(f"Shares: {int(pred[2])}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
