import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ────────────── Streamlit page setup ──────────────
st.set_page_config(page_title="Top & Bottom Posts Dashboard", page_icon="📊", layout="wide")
st.title("📊 Instagram Top & Bottom Posts Dashboard")

# ────────────── Load dataset ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "instagram_analytics_data.xlsx")

try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    st.error(f"🚫 Error loading file: {e}")
    st.write("Looking for file:", DATA_PATH)
    st.stop()

# ────────────── Preprocessing ──────────────
df['Total_Engagement'] = df['Likes'] + df['Comments'] + df['Shares'] + df['Saves']
df['Engagement_Rate_Percent'] = (
    df['Total_Engagement'] / df['Reach'].replace(0, pd.NA)
) * 100
df['Engagement_Rate_Percent'] = df['Engagement_Rate_Percent'].fillna(0).round(2)

# ────────────── Sidebar Filters ──────────────
st.sidebar.header("📌 Filters")
post_types = df['Post_Type'].unique().tolist()
selected_post_types = st.sidebar.multiselect(
    "Select Post Type(s)",
    options=post_types,
    default=post_types
)

# ────────────── Filtered Data ──────────────
df_filtered = df[df['Post_Type'].isin(selected_post_types)]

# ────────────── TOP POSTS ──────────────
top_posts = (
    df_filtered
    .sort_values(by='Engagement_Rate_Percent', ascending=False)
    .head(20)
)

st.subheader("🏆 Top 20 Posts by Engagement Rate")
st.dataframe(
    top_posts[
        ['Post_ID', 'Post_Type', 'Likes', 'Comments', 'Shares',
         'Saves', 'Reach', 'Engagement_Rate_Percent']
    ]
)

csv_top = top_posts.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Top Posts CSV",
    data=csv_top,
    file_name="top_posts.csv",
    mime='text/csv'
)

fig_top = px.bar(
    top_posts,
    x='Post_ID',
    y='Engagement_Rate_Percent',
    color='Post_Type',
    color_discrete_sequence=px.colors.qualitative.Bold,
    title="Engagement Rate of Top 20 Posts"
)
st.plotly_chart(fig_top, use_container_width=True)

st.divider()  # clean visual separation

# ────────────── BOTTOM POSTS ──────────────
worst_posts = (
    df_filtered
    .sort_values(by='Engagement_Rate_Percent', ascending=True)
    .head(20)
)

st.subheader("📉 Bottom 20 Posts by Engagement Rate")
st.dataframe(
    worst_posts[
        ['Post_ID', 'Post_Type', 'Likes', 'Comments', 'Shares',
         'Saves', 'Reach', 'Engagement_Rate_Percent']
    ]
)

csv_worst = worst_posts.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Worst Posts CSV",
    data=csv_worst,
    file_name="worst_posts.csv",
    mime='text/csv'
)

fig_worst = px.bar(
    worst_posts,
    x='Post_ID',
    y='Engagement_Rate_Percent',
    color='Post_Type',
    color_discrete_sequence=px.colors.qualitative.Bold,
    title="Engagement Rate of Worst 20 Posts"
)
st.plotly_chart(fig_worst, use_container_width=True)

# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from dotenv import load_dotenv
# import os

# # ────────────── Streamlit page setup ──────────────
# st.set_page_config(page_title="Top Posts Dashboard", page_icon="🏆", layout="wide")
# st.title("🏆 Instagram Top Posts Dashboard")

# # ────────────── Load dataset ──────────────
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# # data/ folder inside streamlit-app
# DATA_DIR = os.path.join(CURRENT_DIR, "data")

# # THE EXCEL FILE NAME YOU WANT TO LOAD
# DATA_PATH = os.path.join(DATA_DIR, "instagram_analytics_data.xlsx")   # <-- put correct file name here

# # Load dataset safely
# try:
#     df = pd.read_excel(DATA_PATH)
# except Exception as e:
#     st.error(f"🚫 Error loading file: {e}")
#     st.write("Looking for file:", DATA_PATH)
#     st.stop()


# # ────────────── Preprocessing ──────────────
# df['Total_Engagement'] = df['Likes'] + df['Comments'] + df['Shares'] + df['Saves']
# df['Engagement_Rate_Percent'] = (df['Total_Engagement'] / df['Reach'].replace(0, pd.NA)) * 100
# df['Engagement_Rate_Percent'] = df['Engagement_Rate_Percent'].fillna(0).round(2)

# # ────────────── Sidebar Filters ──────────────
# st.sidebar.header("📌 Filters")
# post_types = df['Post_Type'].unique().tolist()
# selected_post_types = st.sidebar.multiselect("Select Post Type(s)", options=post_types, default=post_types)

# # ────────────── Filtered Data ──────────────
# df_filtered = df[df['Post_Type'].isin(selected_post_types)]
# top_posts = df_filtered.sort_values(by='Engagement_Rate_Percent', ascending=False).head(20)

# # ────────────── Display Top Posts ──────────────
# st.subheader("Top 20 Posts by Engagement Rate")
# st.dataframe(top_posts[['Post_ID', 'Post_Type', 'Likes', 'Comments', 'Shares', 'Saves', 'Reach', 'Engagement_Rate_Percent']])

# # ────────────── Top Posts Bar Chart ──────────────
# fig = px.bar(top_posts, x='Post_ID', y='Engagement_Rate_Percent', color='Post_Type',
#              color_discrete_sequence=px.colors.qualitative.Bold,
#              title="Engagement Rate of Top 20 Posts")
# st.plotly_chart(fig, use_container_width=True)

# # ────────────── Download CSV ──────────────
# csv = top_posts.to_csv(index=False).encode('utf-8')
# st.download_button(
#     label="📥 Download Top Posts CSV",
#     data=csv,
#     file_name="top_posts.csv",
#     mime='text/csv'
# )