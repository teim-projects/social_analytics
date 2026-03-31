import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide")

# ============================
# SESSION STATE
# ============================
if "filters" not in st.session_state:
    st.session_state.filters = {}

# ============================
# LOAD DATA
# ============================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
# DATA_PATH = os.path.join(DATA_DIR, "Jan-01-2025_Jan-01-2026_1444967610559863.csv")
# df_posts = pd.read_csv(DATA_PATH)

@st.cache_data
def load_data():
    follows = pd.read_csv(os.path.join(DATA_DIR, "Follows.csv"), skiprows=2, encoding="utf-16")
    interactions = pd.read_csv(os.path.join(DATA_DIR, "Interactions (1).csv"), skiprows=2, encoding="utf-16")
    visits = pd.read_csv(os.path.join(DATA_DIR, "Visits.csv"), skiprows=2, encoding="utf-16")
    views = pd.read_csv(os.path.join(DATA_DIR, "Views (2).csv"), skiprows=2, encoding="utf-16")
    clicks = pd.read_csv(os.path.join(DATA_DIR, "Link clicks.csv"), skiprows=2, encoding="utf-16")

    follows["Date"] = pd.to_datetime(follows["Date"])
    interactions["Date"] = pd.to_datetime(interactions["Date"])
    visits["Date"] = pd.to_datetime(visits["Date"])
    views["Date"] = pd.to_datetime(views["Date"])
    clicks["Date"] = pd.to_datetime(clicks["Date"])

    follows.rename(columns={"Primary": "Follows"}, inplace=True)
    interactions.rename(columns={"Primary": "Interactions"}, inplace=True)
    visits.rename(columns={"Primary": "Visits"}, inplace=True)
    views.rename(columns={"Primary": "Views"}, inplace=True)
    clicks.rename(columns={"Primary": "Link_Clicks"}, inplace=True)

    df = follows.merge(interactions, on="Date") \
        .merge(visits, on="Date") \
        .merge(views, on="Date") \
        .merge(clicks, on="Date")

    df["Engagement_Rate"] = df["Interactions"] / df["Views"]
    df["Click_Through_Rate"] = df["Link_Clicks"] / df["Views"]

    return df


df = load_data()

# ============================
# FILTERS
# ============================
st.sidebar.title("Filters")

date_range = st.sidebar.date_input("Select Date Range", [df["Date"].min(), df["Date"].max()])

filtered_df = df[(df["Date"] >= pd.to_datetime(date_range[0])) &
                 (df["Date"] <= pd.to_datetime(date_range[1]))]

# ============================
# TITLE
# ============================
st.title("📊 Facebook Tactical Analytics Dashboard")

# ============================
# CORRELATION HEATMAP
# ============================
col1, col2 = st.columns([2, 1])

with col1:
    corr = filtered_df.drop(columns="Date").corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    ### 📌 Insights
    - Shows relationships between metrics.
    - Strong correlation = metrics move together.
    - Helps identify key drivers like Views influencing others.
    """)

# ============================
# ENGAGEMENT RATE GRAPH
# ============================
col1, col2 = st.columns([2, 1])

with col1:
    fig = px.line(filtered_df, x="Date", y="Engagement_Rate", title="Engagement Rate Over Time")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    best_day = filtered_df.loc[filtered_df["Engagement_Rate"].idxmax()]

    st.markdown(f"""
    ### 📌 Insights
    - Highest engagement on **{best_day['Date'].date()}**
    - Rate: **{best_day['Engagement_Rate']:.4f}**
    - Indicates most engaging content day.
    """)

