import streamlit as st
import pandas as pd
import plotly.express as px
import datetime
from dotenv import load_dotenv
import os

# ────────────── Streamlit page setup ──────────────
st.set_page_config(page_title="Instagram Analytics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Instagram Data Visualization Dashboard")

# ────────────── Load dataset ──────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "instagram_analytics_data.xlsx")

try:
    df = pd.read_excel(DATA_PATH)
except Exception as e:
    st.error(f"❌ Could not load Instagram dataset: {e}")
    st.stop()


df['Post_Date'] = pd.to_datetime(df['Post_Date'], errors='coerce')

all_graphs = [
    "Distribution of Likes",
    "Distribution of Comments",
    "Likes vs Shares",
    "Average Reach by Post Type",
    "Impressions Over Time",
    "Correlation Matrix",
    "Posts by Day of Week",
    "Posts by Post Type",
    "Average Engagement by Post Type",
    "Engagement Distribution by Day",
    "Story Completion Rate by Day of Week",
    "Story Completion Rate by Post Type",
    "Average Ad Spend (USD) by Post Type",
    "Average Ad Revenue (USD) by Post Type",
    "Average ROAS by Post Type",
    "Average CPA (USD) by Post Type",
    "Top 10 Hashtag 1",
    "Top 10 Hashtag 2",
    "Count of Posts by Time of Day"
]

# ────────────── CLEAR ALL FILTERS CALLBACK ──────────────
def clear_all_filters():
    # Reset defaults
    for key, val in defaults.items():
        st.session_state[key] = val

    # Remove multiselect key to force reset
    if 'graph_selector' in st.session_state:
        del st.session_state['graph_selector']

    # Remove search box key to clear it visually
    if 'graph_search' in st.session_state:
        del st.session_state['graph_search']

    # Remove last search tracking key
    if 'last_search_text' in st.session_state:
        del st.session_state['last_search_text']

    # Trigger rerun
    # st.rerun()  # If your Streamlit version <1.22, otherwise use st.rerun()

# # ────────────── Sidebar Filters ──────────────
# st.sidebar.header("📌 Filters")

# ────────────── Sidebar Filters with Clear Button ──────────────
col1, col2 = st.sidebar.columns([1, 1])  # Adjust width ratio as needed
col1.header("📌 Filters")
col2.button("🧹 Clear All Filters", on_click=clear_all_filters)


# Extract valid filter limits
post_types = df['Post_Type'].dropna().unique().tolist()
min_date = df['Post_Date'].min().date()
max_date = df['Post_Date'].max().date()

# ────────────── Initialize Session State ──────────────
defaults = {
    "selected_post_types": post_types,
    "from_date": min_date,
    "to_date": max_date,
    "selected_graphs": all_graphs.copy(),
    "search_text": ""
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ────────────── POST TYPE FILTER ──────────────
selected_post_types = st.sidebar.multiselect(
    "Select Post Type(s)",
    options=post_types,
    default=st.session_state.selected_post_types,
    key="post_type_filter"
)
st.session_state.selected_post_types = selected_post_types

# ────────────── DATE FILTERS (UI shows all months, internal filtering) ──────────────
col1, col2 = st.sidebar.columns(2)

# Ensure Post_Date column is proper datetime
df['Post_Date'] = pd.to_datetime(df['Post_Date'], errors='coerce').dt.tz_localize(None)

# Defaults from session state or dataset
from_date_default = st.session_state.get("from_date", df['Post_Date'].min().date())
to_date_default   = st.session_state.get("to_date", df['Post_Date'].max().date())

# Draw date inputs WITHOUT min_value / max_value
new_from_date = col1.date_input(
    "From Date",
    value=from_date_default,
    key="from_date_input"
)

new_to_date = col2.date_input(
    "To Date",
    value=to_date_default,
    key="to_date_input"
)

# Validation
if new_from_date > new_to_date:
    st.sidebar.error("❌ 'From Date' cannot be after 'To Date'")
    st.stop()

# Persist new values in session state
st.session_state.from_date = new_from_date
st.session_state.to_date   = new_to_date

# ────────────── GRAPH SEARCH ──────────────
st.sidebar.subheader("Graph Selection")

search_text = st.sidebar.text_input(
    "🔍 Search or type to filter graphs",
    value=st.session_state.get("graph_search", ""),  # defaults to empty string
    key="graph_search"
)

# Filter graphs
filtered_graphs = [g for g in all_graphs if search_text.lower() in g.lower()] if search_text else all_graphs

# If search changed, reset multiselect
if 'last_search_text' not in st.session_state or st.session_state.last_search_text != search_text:
    # Clear the stored selection for the multiselect key
    if 'graph_selector' in st.session_state:
        del st.session_state['graph_selector']
    # Update selected_graphs in session state
    st.session_state.selected_graphs = filtered_graphs.copy()

st.session_state.last_search_text = search_text

# Ensure default values exist in options to avoid errors
valid_defaults = [g for g in st.session_state.selected_graphs if g in filtered_graphs]

# # Render multiselect
# selected_graphs = st.sidebar.multiselect(
#     "Select which graphs to display",
#     options=filtered_graphs,
#     default=valid_defaults,
#     key="graph_selector"
# )

# Persist selection
# st.session_state.selected_graphs = selected_graphs

# ────────────── APPLY FILTERS ──────────────
df_filtered = df[
    df['Post_Type'].isin(st.session_state.selected_post_types)
    & (df['Post_Date'].dt.date >= st.session_state.from_date)
    & (df['Post_Date'].dt.date <= st.session_state.to_date)
]

if df_filtered.empty:
    st.warning("⚠️ No data available for the selected filters.")
    st.stop()


# ────────────── Helper function ──────────────
def plot_with_summary(fig, summary_text):
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(summary_text)

# ────────────── 1️⃣ Distribution of Likes ──────────────
if "Distribution of Likes" in st.session_state.selected_graphs:
    st.subheader("1️⃣ Distribution of Likes")
    fig = px.histogram(df_filtered, x='Likes', nbins=30, color_discrete_sequence=px.colors.sequential.Viridis,
                    marginal="box", title="Distribution of Likes")
    likes_stats = df_filtered['Likes'].describe()
    summary_likes = f"""
**Likes Summary**  
{likes_stats.to_frame().to_markdown()}  
- Conclusion: Most posts receive around {df_filtered['Likes'].median():.0f} likes; a few viral posts indicate high engagement.
"""
    plot_with_summary(fig, summary_likes)

# ────────────── 2️⃣ Distribution of Comments ──────────────
if "Distribution of Comments" in st.session_state.selected_graphs:
    st.subheader("2️⃣ Distribution of Comments")
    fig = px.box(df_filtered, y='Comments', color_discrete_sequence=px.colors.sequential.Magma, title="Distribution of Comments")
    comments_stats = df_filtered['Comments'].describe()
    summary_comments = f"""
**Comments Summary**  
{comments_stats.to_frame().to_markdown()}  
- Conclusion: Most posts receive fewer comments than likes, showing less interactive engagement; some posts are highly engaging.
"""
    plot_with_summary(fig, summary_comments)

# ────────────── 3️⃣ Likes vs Shares ──────────────
if "Likes vs Shares" in st.session_state.selected_graphs:
    st.subheader("3️⃣ Likes vs Shares")
    fig = px.scatter(df_filtered, x='Likes', y='Shares', color='Post_Type', hover_data=['Post_Date', 'Post_Type'],
                    color_discrete_sequence=px.colors.qualitative.Bold, title="Likes vs Shares")
    corr_ls = df_filtered['Likes'].corr(df_filtered['Shares'])
    summary_likes_shares = f"""
**Likes vs Shares Summary**  
- Correlation: {corr_ls:.2f}  
- Conclusion: {'Posts with more likes tend to get more shares.' if corr_ls>0.5 else 'No strong relationship between likes and shares; engagement may vary by content type.'}
"""
    plot_with_summary(fig, summary_likes_shares)

# ────────────── 4️⃣ Average Reach by Post Type ──────────────
if "Average Reach by Post Type" in st.session_state.selected_graphs:
    st.subheader("4️⃣ Average Reach by Post Type")
    avg_reach = df_filtered.groupby('Post_Type')['Reach'].mean().sort_values(ascending=False)
    fig = px.bar(avg_reach, x=avg_reach.index, y=avg_reach.values, color=avg_reach.index,
                color_discrete_sequence=px.colors.sequential.Viridis, title="Average Reach by Post Type")
    summary_reach = f"""
**Average Reach by Post Type**  
{avg_reach.to_frame().to_markdown()}  
- Conclusion: {avg_reach.idxmax()} posts reach the largest audience; prioritize these post types.
"""
    plot_with_summary(fig, summary_reach)

# ────────────── 5️⃣ Impressions Over Time ──────────────
if "Impressions Over Time" in st.session_state.selected_graphs:
    st.subheader("5️⃣ Impressions Over Time")
    df_sorted = df_filtered.sort_values('Post_Date')
    fig = px.line(df_sorted, x='Post_Date', y='Impressions', color='Post_Type', hover_data=['Likes','Comments'],
                color_discrete_sequence=px.colors.qualitative.Bold, title="Impressions Over Time")
    trend = df_sorted['Impressions'].diff().mean()
    summary_impressions = f"""
**Impressions Over Time**  
- Trend: {'Increasing 📈' if trend>0 else 'Decreasing 📉'}  
- Conclusion: Overall impressions are {'growing' if trend>0 else 'declining'}, indicating audience expansion or contraction over time.
"""
    plot_with_summary(fig, summary_impressions)

# ────────────── 6️⃣ Correlation Matrix ──────────────
if "Correlation Matrix" in st.session_state.selected_graphs:
    st.subheader("6️⃣ Correlation Matrix")
    numerical_df = df_filtered.select_dtypes(include=['float64','int64'])
    if not numerical_df.empty:
        fig = px.imshow(numerical_df.corr(), color_continuous_scale='RdBu', text_auto=True, title="Correlation Matrix")
        summary_corr = "**Correlation Matrix** shows relationships between numerical metrics. Values near 1 or -1 indicate strong relationships."
        plot_with_summary(fig, summary_corr)

# ────────────── 7️⃣ Posts by Day of Week ──────────────
if "Posts by Day of Week" in st.session_state.selected_graphs:
    if 'Day_of_Week' in df_filtered.columns:
        st.subheader("7️⃣ Count of Posts by Day of Week")
        counts_day = df_filtered['Day_of_Week'].value_counts()
        fig = px.bar(counts_day, x=counts_day.index, y=counts_day.values, color=counts_day.index,
                    color_discrete_sequence=px.colors.sequential.Plasma, title="Posts by Day of Week")
        summary_day = f"""
**Posts by Day of Week**  
{counts_day.to_frame().to_markdown()}  
- Conclusion: Highest posts occur on {counts_day.idxmax()}; schedule important content accordingly.
"""
        plot_with_summary(fig, summary_day)

# ────────────── 8️⃣ Posts by Post Type ──────────────
if "Posts by Post Type" in st.session_state.selected_graphs:
    st.subheader("8️⃣ Count of Posts by Post Type")
    counts_type = df_filtered['Post_Type'].value_counts()
    fig = px.bar(counts_type, x=counts_type.index, y=counts_type.values, color=counts_type.index,
                color_discrete_sequence=px.colors.sequential.Viridis, title="Posts by Post Type")
    summary_type = f"""
**Posts by Post Type**  
{counts_type.to_frame().to_markdown()}  
- Conclusion: Most posts are {counts_type.idxmax()}; reflects current content strategy.
"""
    plot_with_summary(fig, summary_type)

# ────────────── 9️⃣ Average Engagement by Post Type ──────────────
if "Average Engagement by Post Type" in st.session_state.selected_graphs:
    if 'Engagement_Total' in df_filtered.columns:
        st.subheader("9️⃣ Average Engagement by Post Type")
        avg_eng = df_filtered.groupby('Post_Type')['Engagement_Total'].mean().sort_values(ascending=False)
        fig = px.bar(avg_eng, x=avg_eng.index, y=avg_eng.values, color=avg_eng.index,
                    color_discrete_sequence=px.colors.sequential.Magma, title="Average Engagement by Post Type")
        summary_eng = f"""
**Average Engagement by Post Type**  
{avg_eng.to_frame().to_markdown()}  
- Conclusion: {avg_eng.idxmax()} posts have the highest engagement; replicate similar content.
"""
        plot_with_summary(fig, summary_eng)

# ────────────── 10️⃣ Engagement Distribution by Day ──────────────
if "Engagement Distribution by Day" in st.session_state.selected_graphs:
    if 'Day_of_Week' in df_filtered.columns and 'Engagement_Total' in df_filtered.columns:
        st.subheader("10️⃣ Engagement Distribution by Day of Week")
        fig = px.box(df_filtered, x='Day_of_Week', y='Engagement_Total', color='Day_of_Week',
                    color_discrete_sequence=px.colors.sequential.Plasma, title="Engagement Distribution by Day")
        summary_eng_day = f"""
**Engagement Distribution by Day**  
{df_filtered.groupby('Day_of_Week')['Engagement_Total'].describe().to_markdown()}  
- Conclusion: Engagement varies by day; schedule key posts on high-median days.
"""
        plot_with_summary(fig, summary_eng_day)

# --------------------------------------------

# ────────────── 11️⃣ Story Completion Rate by Day of Week ──────────────
if "Story Completion Rate by Day of Week" in st.session_state.selected_graphs:
    if 'Story_Completion_Rate' in df_filtered.columns and 'Day_of_Week' in df_filtered.columns:
        st.subheader("11️⃣ Story Completion Rate by Day of Week")
        avg_story = df_filtered.groupby('Day_of_Week')['Story_Completion_Rate'].mean().sort_values(ascending=False)
        fig = px.bar(
            avg_story,
            x=avg_story.index,
            y=avg_story.values,
            color=avg_story.index,
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Story Completion Rate by Day"
        )
        summary_story_day = f"""
**Story Completion Rate by Day**  
{avg_story.to_frame().to_markdown()}  
- Conclusion: Highest completion on {avg_story.idxmax()}; post stories on these days for better retention.
"""
        plot_with_summary(fig, summary_story_day)

# ────────────── 12️⃣ Story Completion Rate by Post Type ──────────────
if "Story Completion Rate by Post Type" in st.session_state.selected_graphs:
    if 'Story_Completion_Rate' in df_filtered.columns and 'Post_Type' in df_filtered.columns:
        st.subheader("12️⃣ Story Completion Rate by Post Type")
        fig = px.box(
            df_filtered,
            x='Post_Type',
            y='Story_Completion_Rate',
            color='Post_Type',
            color_discrete_sequence=px.colors.sequential.Magma,
            title="Story Completion Rate by Post Type"
        )
        summary_story_type = f"""
**Story Completion Rate by Post Type**  
{df_filtered.groupby('Post_Type')['Story_Completion_Rate'].describe().to_markdown()}  
- Conclusion: Certain post types retain viewers longer; focus on these for stories.
"""
        plot_with_summary(fig, summary_story_type)

# ────────────── 13️⃣ Average Ad Spend (USD) by Post Type ──────────────
if "Average Ad Spend (USD) by Post Type" in st.session_state.selected_graphs:
    if 'Ad_Spend_USD' in df_filtered.columns:
        st.subheader("13️⃣ Average Ad Spend (USD) by Post Type")
        avg_metric = df_filtered.groupby('Post_Type')['Ad_Spend_USD'].mean().sort_values(ascending=False)
        fig = px.bar(
            avg_metric,
            x=avg_metric.index,
            y=avg_metric.values,
            color=avg_metric.index,
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Average Ad Spend (USD) by Post Type"
        )
        summary_metric = f"""
**Average Ad Spend (USD) by Post Type**  
{avg_metric.to_frame().to_markdown()}  
- Conclusion: {avg_metric.idxmax()} posts have the highest ad spend.
"""
        plot_with_summary(fig, summary_metric)

# ────────────── 14️⃣ Average Ad Revenue (USD) by Post Type ──────────────
if "Average Ad Revenue (USD) by Post Type" in st.session_state.selected_graphs:
    if 'Ad_Revenue_USD' in df_filtered.columns:
        st.subheader("14️⃣ Average Ad Revenue (USD) by Post Type")
        avg_metric = df_filtered.groupby('Post_Type')['Ad_Revenue_USD'].mean().sort_values(ascending=False)
        fig = px.bar(
            avg_metric,
            x=avg_metric.index,
            y=avg_metric.values,
            color=avg_metric.index,
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Average Ad Revenue (USD) by Post Type"
        )
        summary_metric = f"""
**Average Ad Revenue (USD) by Post Type**  
{avg_metric.to_frame().to_markdown()}  
- Conclusion: {avg_metric.idxmax()} posts generate the highest revenue.
"""
        plot_with_summary(fig, summary_metric)

# ────────────── 15️⃣ Average ROAS by Post Type ──────────────
if "Average ROAS by Post Type" in st.session_state.selected_graphs:
    if 'ROAS' in df_filtered.columns:
        st.subheader("15️⃣ Average ROAS by Post Type")
        avg_metric = df_filtered.groupby('Post_Type')['ROAS'].mean().sort_values(ascending=False)
        fig = px.bar(
            avg_metric,
            x=avg_metric.index,
            y=avg_metric.values,
            color=avg_metric.index,
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Average ROAS by Post Type"
        )
        summary_metric = f"""
**Average ROAS by Post Type**  
{avg_metric.to_frame().to_markdown()}  
- Conclusion: {avg_metric.idxmax()} posts deliver the best return on ad spend.
"""
        plot_with_summary(fig, summary_metric)

# ────────────── 16️⃣ Average CPA (USD) by Post Type ──────────────
if "Average CPA (USD) by Post Type" in st.session_state.selected_graphs:
    if 'CPA_USD' in df_filtered.columns:
        st.subheader("16️⃣ Average CPA (USD) by Post Type")
        avg_metric = df_filtered.groupby('Post_Type')['CPA_USD'].mean().sort_values(ascending=False)
        fig = px.bar(
            avg_metric,
            x=avg_metric.index,
            y=avg_metric.values,
            color=avg_metric.index,
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Average CPA (USD) by Post Type"
        )
        summary_metric = f"""
**Average CPA (USD) by Post Type**  
{avg_metric.to_frame().to_markdown()}  
- Conclusion: {avg_metric.idxmin()} posts have the lowest acquisition cost — the most efficient.
"""
        plot_with_summary(fig, summary_metric)

# ────────────── 17️⃣ Top 10 Hashtag 1 ──────────────
if "Top 10 Hashtag 1" in st.session_state.selected_graphs:
    if 'Top_Hashtag_1' in df_filtered.columns:
        st.subheader("17️⃣ Top 10 Hashtag 1")
        top_10 = df_filtered['Top_Hashtag_1'].value_counts().nlargest(10)
        fig = px.bar(
            top_10,
            x=top_10.index,
            y=top_10.values,
            color=top_10.index,
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Top 10 Hashtag 1"
        )
        summary_hashtag = f"""
**Top Hashtag 1**  
{top_10.to_frame().to_markdown()}  
- Conclusion: These hashtags are trending; using them increases reach and engagement.
"""
        plot_with_summary(fig, summary_hashtag)

# ────────────── 18️⃣ Top 10 Hashtag 2 ──────────────
if "Top 10 Hashtag 2" in st.session_state.selected_graphs:
    if 'Top_Hashtag_2' in df_filtered.columns:
        st.subheader("18️⃣ Top 10 Hashtag 2")
        top_10 = df_filtered['Top_Hashtag_2'].value_counts().nlargest(10)
        fig = px.bar(
            top_10,
            x=top_10.index,
            y=top_10.values,
            color=top_10.index,
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Top 10 Hashtag 2"
        )
        summary_hashtag = f"""
**Top Hashtag 2**  
{top_10.to_frame().to_markdown()}  
- Conclusion: These hashtags are trending; using them increases reach and engagement.
"""
        plot_with_summary(fig, summary_hashtag)

# ────────────── 19️⃣ Count of Posts by Time of Day ──────────────
if "Count of Posts by Time of Day" in st.session_state.selected_graphs:
    if 'Time_of_Day' in df_filtered.columns:
        st.subheader("19️⃣ Count of Posts by Time of Day")
        df_filtered['Hour_of_Day'] = pd.to_datetime(df_filtered['Time_of_Day']).dt.hour

        def time_category(hour):
            if 5 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 17:
                return 'Afternoon'
            elif 17 <= hour < 21:
                return 'Evening'
            else:
                return 'Night'

        df_filtered['Time_Category'] = df_filtered['Hour_of_Day'].apply(time_category)
        time_counts = df_filtered['Time_Category'].value_counts().reindex(['Morning', 'Afternoon', 'Evening', 'Night'])
        fig = px.bar(
            time_counts,
            x=time_counts.index,
            y=time_counts.values,
            color=time_counts.index,
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Count of Posts by Time of Day"
        )
        summary_time_of_day = f"""
**Posts by Time of Day**  
{time_counts.to_frame().to_markdown()}  
- Conclusion: Most posts are published during {time_counts.idxmax()}; schedule posts accordingly for higher engagement.
"""
        plot_with_summary(fig, summary_time_of_day)


# ----------------------------------------------------------------------------------------------------------
# /////////////////////////////////// BEFORE SEARCH ////////////////////////////////////////////////////////
# ----------------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import plotly.express as px

# # ────────────── Streamlit page setup ──────────────
# st.set_page_config(page_title="Instagram Analytics Dashboard", page_icon="📊", layout="wide")
# st.title("📊 Instagram Data Visualization Dashboard")

# # ────────────── Load dataset ──────────────
# default_file_path = r"C:\Users\Dell\Downloads\Social_media_monitoring_platform\Social_media_monitoring_platform\Instagram\streamlit-app\data\instagram_analytics_data.xlsx"
# try:
#     df = pd.read_excel(default_file_path)
# except FileNotFoundError:
#     st.error("🚫 Default file not found! Please check the file path.")
#     st.stop()

# df['Post_Date'] = pd.to_datetime(df['Post_Date'], errors='coerce')

# # ────────────── Sidebar filters ──────────────
# st.sidebar.header("📌 Filters")

# # Post Type filter
# post_types = df['Post_Type'].unique().tolist()
# selected_post_types = st.sidebar.multiselect("Select Post Type(s)", options=post_types, default=post_types)

# # ────────────── Separate From/To Date Pickers ──────────────
# min_date = df['Post_Date'].min()
# max_date = df['Post_Date'].max()

# col1, col2 = st.sidebar.columns(2)
# from_date = col1.date_input("From Date", min_date)
# to_date = col2.date_input("To Date", max_date)

# # Ensure from_date <= to_date
# if from_date > to_date:
#     st.sidebar.error("❌ 'From Date' cannot be after 'To Date'")
#     st.stop()

# # Filter dataframe dynamically
# df_filtered = df[df['Post_Type'].isin(selected_post_types)]
# df_filtered = df_filtered[(df_filtered['Post_Date'] >= pd.Timestamp(from_date)) & 
#                           (df_filtered['Post_Date'] <= pd.Timestamp(to_date))]

# # ────────────── Helper function ──────────────
# def plot_with_summary(fig, summary_text):
#     col1, col2 = st.columns([2,1])
#     with col1:
#         st.plotly_chart(fig, use_container_width=True)
#     with col2:
#         st.markdown(summary_text)

# # ────────────── 1️⃣ Distribution of Likes ──────────────
# st.subheader("1️⃣ Distribution of Likes")
# fig = px.histogram(df_filtered, x='Likes', nbins=30, color_discrete_sequence=px.colors.sequential.Viridis,
#                    marginal="box", title="Distribution of Likes")
# likes_stats = df_filtered['Likes'].describe()
# summary_likes = f"""
# *Likes Summary*  
# {likes_stats.to_frame().to_markdown()}  
# - Conclusion: Most posts receive around {df_filtered['Likes'].median():.0f} likes; a few viral posts indicate high engagement.
# """
# plot_with_summary(fig, summary_likes)

# # ────────────── 2️⃣ Distribution of Comments ──────────────
# st.subheader("2️⃣ Distribution of Comments")
# fig = px.box(df_filtered, y='Comments', color_discrete_sequence=px.colors.sequential.Magma, title="Distribution of Comments")
# comments_stats = df_filtered['Comments'].describe()
# summary_comments = f"""
# *Comments Summary*  
# {comments_stats.to_frame().to_markdown()}  
# - Conclusion: Most posts receive fewer comments than likes, showing less interactive engagement; some posts are highly engaging.
# """
# plot_with_summary(fig, summary_comments)

# # ────────────── 3️⃣ Likes vs Shares ──────────────
# st.subheader("3️⃣ Likes vs Shares")
# fig = px.scatter(df_filtered, x='Likes', y='Shares', color='Post_Type', hover_data=['Post_Date', 'Post_Type'],
#                  color_discrete_sequence=px.colors.qualitative.Bold, title="Likes vs Shares")
# corr_ls = df_filtered['Likes'].corr(df_filtered['Shares'])
# summary_likes_shares = f"""
# *Likes vs Shares Summary*  
# - Correlation: {corr_ls:.2f}  
# - Conclusion: {'Posts with more likes tend to get more shares.' if corr_ls>0.5 else 'No strong relationship between likes and shares; engagement may vary by content type.'}
# """
# plot_with_summary(fig, summary_likes_shares)

# # ────────────── 4️⃣ Average Reach by Post Type ──────────────
# st.subheader("4️⃣ Average Reach by Post Type")
# avg_reach = df_filtered.groupby('Post_Type')['Reach'].mean().sort_values(ascending=False)
# fig = px.bar(avg_reach, x=avg_reach.index, y=avg_reach.values, color=avg_reach.index,
#              color_discrete_sequence=px.colors.sequential.Viridis, title="Average Reach by Post Type")
# summary_reach = f"""
# *Average Reach by Post Type*  
# {avg_reach.to_frame().to_markdown()}  
# - Conclusion: {avg_reach.idxmax()} posts reach the largest audience; prioritize these post types.
# """
# plot_with_summary(fig, summary_reach)

# # ────────────── 5️⃣ Impressions Over Time ──────────────
# st.subheader("5️⃣ Impressions Over Time")
# df_sorted = df_filtered.sort_values('Post_Date')
# fig = px.line(df_sorted, x='Post_Date', y='Impressions', color='Post_Type', hover_data=['Likes','Comments'],
#               color_discrete_sequence=px.colors.qualitative.Bold, title="Impressions Over Time")
# trend = df_sorted['Impressions'].diff().mean()
# summary_impressions = f"""
# *Impressions Over Time*  
# - Trend: {'Increasing 📈' if trend>0 else 'Decreasing 📉'}  
# - Conclusion: Overall impressions are {'growing' if trend>0 else 'declining'}, indicating audience expansion or contraction over time.
# """
# plot_with_summary(fig, summary_impressions)

# # ────────────── 6️⃣ Correlation Matrix ──────────────
# st.subheader("6️⃣ Correlation Matrix")
# numerical_df = df_filtered.select_dtypes(include=['float64','int64'])
# if not numerical_df.empty:
#     fig = px.imshow(numerical_df.corr(), color_continuous_scale='RdBu', text_auto=True, title="Correlation Matrix")
#     summary_corr = "*Correlation Matrix* shows relationships between numerical metrics. Values near 1 or -1 indicate strong relationships."
#     plot_with_summary(fig, summary_corr)

# # ────────────── 7️⃣ Posts by Day of Week ──────────────
# if 'Day_of_Week' in df_filtered.columns:
#     st.subheader("7️⃣ Count of Posts by Day of Week")
#     counts_day = df_filtered['Day_of_Week'].value_counts()
#     fig = px.bar(counts_day, x=counts_day.index, y=counts_day.values, color=counts_day.index,
#                  color_discrete_sequence=px.colors.sequential.Plasma, title="Posts by Day of Week")
#     summary_day = f"""
# *Posts by Day of Week*  
# {counts_day.to_frame().to_markdown()}  
# - Conclusion: Highest posts occur on {counts_day.idxmax()}; schedule important content accordingly.
# """
#     plot_with_summary(fig, summary_day)

# # ────────────── 8️⃣ Posts by Post Type ──────────────
# st.subheader("8️⃣ Count of Posts by Post Type")
# counts_type = df_filtered['Post_Type'].value_counts()
# fig = px.bar(counts_type, x=counts_type.index, y=counts_type.values, color=counts_type.index,
#              color_discrete_sequence=px.colors.sequential.Viridis, title="Posts by Post Type")
# summary_type = f"""
# *Posts by Post Type*  
# {counts_type.to_frame().to_markdown()}  
# - Conclusion: Most posts are {counts_type.idxmax()}; reflects current content strategy.
# """
# plot_with_summary(fig, summary_type)

# # ────────────── 9️⃣ Average Engagement by Post Type ──────────────
# if 'Engagement_Total' in df_filtered.columns:
#     st.subheader("9️⃣ Average Engagement by Post Type")
#     avg_eng = df_filtered.groupby('Post_Type')['Engagement_Total'].mean().sort_values(ascending=False)
#     fig = px.bar(avg_eng, x=avg_eng.index, y=avg_eng.values, color=avg_eng.index,
#                  color_discrete_sequence=px.colors.sequential.Magma, title="Average Engagement by Post Type")
#     summary_eng = f"""
# *Average Engagement by Post Type*  
# {avg_eng.to_frame().to_markdown()}  
# - Conclusion: {avg_eng.idxmax()} posts have the highest engagement; replicate similar content.
# """
#     plot_with_summary(fig, summary_eng)

# # ────────────── 10️⃣ Engagement Distribution by Day ──────────────
# if 'Day_of_Week' in df_filtered.columns and 'Engagement_Total' in df_filtered.columns:
#     st.subheader("10️⃣ Engagement Distribution by Day of Week")
#     fig = px.box(df_filtered, x='Day_of_Week', y='Engagement_Total', color='Day_of_Week',
#                  color_discrete_sequence=px.colors.sequential.Plasma, title="Engagement Distribution by Day")
#     summary_eng_day = f"""
# *Engagement Distribution by Day*  
# {df_filtered.groupby('Day_of_Week')['Engagement_Total'].describe().to_markdown()}  
# - Conclusion: Engagement varies by day; schedule key posts on high-median days.
# """
#     plot_with_summary(fig, summary_eng_day)

# # ────────────── 11️⃣ Story Completion Rate ──────────────
# if 'Story_Completion_Rate' in df_filtered.columns and 'Day_of_Week' in df_filtered.columns:
#     st.subheader("11️⃣ Story Completion Rate by Day of Week")
#     avg_story = df_filtered.groupby('Day_of_Week')['Story_Completion_Rate'].mean().sort_values(ascending=False)
#     fig = px.bar(avg_story, x=avg_story.index, y=avg_story.values, color=avg_story.index,
#                  color_discrete_sequence=px.colors.sequential.Viridis, title="Story Completion Rate by Day")
#     summary_story_day = f"""
# *Story Completion Rate by Day*  
# {avg_story.to_frame().to_markdown()}  
# - Conclusion: Highest completion on {avg_story.idxmax()}; post stories on these days for better retention.
# """
#     plot_with_summary(fig, summary_story_day)

#     st.subheader("Story Completion Rate by Post Type")
#     fig = px.box(df_filtered, x='Post_Type', y='Story_Completion_Rate', color='Post_Type',
#                  color_discrete_sequence=px.colors.sequential.Magma, title="Story Completion Rate by Post Type")
#     summary_story_type = f"""
# *Story Completion Rate by Post Type*  
# {df_filtered.groupby('Post_Type')['Story_Completion_Rate'].describe().to_markdown()}  
# - Conclusion: Certain post types retain viewers longer; focus on these for stories.
# """
#     plot_with_summary(fig, summary_story_type)

# # ────────────── 12️⃣ - 17️⃣ Ad Metrics, Hashtags, Time of Day ──────────────
# metrics_cols = [
#     ('Ad_Spend_USD','Average Ad Spend (USD) by Post Type'),
#     ('Ad_Revenue_USD','Average Ad Revenue (USD) by Post Type'),
#     ('ROAS','Average ROAS by Post Type'),
#     ('CPA_USD','Average CPA (USD) by Post Type')
# ]

# for col, title in metrics_cols:
#     if col in df_filtered.columns:
#         st.subheader(f"{title}")
#         avg_metric = df_filtered.groupby('Post_Type')[col].mean().sort_values(ascending=False)
#         fig = px.bar(avg_metric, x=avg_metric.index, y=avg_metric.values, color=avg_metric.index,
#                      color_discrete_sequence=px.colors.sequential.Viridis, title=title)
#         summary_metric = f"""
# *{title}*  
# {avg_metric.to_frame().to_markdown()}  
# - Conclusion: {avg_metric.idxmax()} posts have the highest {col}; optimize strategy accordingly.
# """
#         plot_with_summary(fig, summary_metric)

#         # Distribution by Day
#         if 'Day_of_Week' in df_filtered.columns:
#             st.subheader(f"{title} Distribution by Day of Week")
#             fig = px.box(df_filtered, x='Day_of_Week', y=col, color='Day_of_Week',
#                          color_discrete_sequence=px.colors.sequential.Plasma, title=f"{title} Distribution by Day")
#             summary_day_dist = f"Boxplot showing {title} distribution by day of the week."
#             plot_with_summary(fig, summary_day_dist)

# # Top Hashtags
# for i, hashtag_col in enumerate(['Top_Hashtag_1','Top_Hashtag_2'], start=16):
#     if hashtag_col in df_filtered.columns:
#         st.subheader(f"{i}️⃣ Top 10 {hashtag_col}")
#         top_10 = df_filtered[hashtag_col].value_counts().nlargest(10)
#         fig = px.bar(top_10, x=top_10.index, y=top_10.values, color=top_10.index,
#                      color_discrete_sequence=px.colors.sequential.Viridis, title=f"Top 10 Hashtags ({hashtag_col})")
#         summary_hashtag = f"""
# *Top Hashtags ({hashtag_col})*  
# {top_10.to_frame().to_markdown()}  
# - Conclusion: These hashtags are trending; using them increases reach and engagement.
# """
#         plot_with_summary(fig, summary_hashtag)

# # ────────────── 17️⃣ Count of Posts by Time of Day ──────────────
# if 'Time_of_Day' in df_filtered.columns:
#     st.subheader("17️⃣ Count of Posts by Time of Day")
#     df_filtered['Hour_of_Day'] = pd.to_datetime(df_filtered['Time_of_Day']).dt.hour
#     def time_category(hour):
#         if 5 <= hour < 12:
#             return 'Morning'
#         elif 12 <= hour < 17:
#             return 'Afternoon'
#         elif 17 <= hour < 21:
#             return 'Evening'
#         else:
#             return 'Night'
#     df_filtered['Time_Category'] = df_filtered['Hour_of_Day'].apply(time_category)
#     time_counts = df_filtered['Time_Category'].value_counts().reindex(['Morning','Afternoon','Evening','Night'])
#     fig = px.bar(time_counts, x=time_counts.index, y=time_counts.values, color=time_counts.index,
#                  color_discrete_sequence=px.colors.sequential.Viridis, title="Count of Posts by Time of Day")
#     summary_time_of_day = f"""
# *Posts by Time of Day*  
# {time_counts.to_frame().to_markdown()}  
# - Conclusion: Most posts are published during {time_counts.idxmax()}; schedule posts accordingly for higher engagement.
# """
#     plot_with_summary(fig, summary_time_of_day)

# ----------------------------------------------------------------------------------------------------------
# //////////////////////////////////////////////////////////////////////////////////////////////////////////
# ----------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------
# ////////////////////// GRAPHS WITHOUT PLOTLY ... ONLY MATPLOTLIB/SEABORN /////////////////////////////////
# ----------------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # ────────────── Streamlit page setup ──────────────
# st.set_page_config(
#     page_title="Instagram Analytics Dashboard",
#     page_icon="📊",
#     layout="wide"
# )

# st.title("📊 Instagram Data Visualization Dashboard")

# # ────────────── Load dataset ──────────────
# default_file_path = r"C:\Users\Dell\Downloads\Social_media_monitoring_platform\Social_media_monitoring_platform\Instagram\streamlit-app\data\instagram_analytics_data.xlsx"
# try:
#     df = pd.read_excel(default_file_path)
# except FileNotFoundError:
#     st.error("🚫 Default file not found! Please check the file path.")
#     st.stop()

# # Display data preview
# st.subheader("📄 Data Preview")
# st.dataframe(df.head())

# # ────────────── Expected columns ──────────────
# required_cols = ['Likes', 'Comments', 'Shares', 'Reach', 'Post_Type', 'Post_Date', 'Impressions']
# if not all(col in df.columns for col in required_cols):
#     st.error("⚠️ Your dataset must include columns: " + ", ".join(required_cols))
#     st.stop()

# sns.set_style('whitegrid')
# df['Post_Date'] = pd.to_datetime(df['Post_Date'], errors='coerce')

# # ────────────── Helper function ──────────────
# def plot_with_summary(fig, summary_text):
#     col1, col2 = st.columns([2,1])
#     with col1:
#         st.pyplot(fig)
#     with col2:
#         st.markdown(summary_text)

# # ────────────── 1️⃣ Distribution of Likes ──────────────
# st.subheader("1️⃣ Distribution of Likes")
# fig, ax = plt.subplots(figsize=(8,5))
# sns.histplot(df['Likes'], kde=True, bins=30, ax=ax)
# ax.set_title('Distribution of Likes')
# likes_stats = df['Likes'].describe()
# summary_likes = f"""
# **Likes Summary**  
# {likes_stats.to_frame().to_markdown()}  
# - Conclusion: Most posts receive around {df['Likes'].median():.0f} likes; a few viral posts indicate high engagement content.
# """
# plot_with_summary(fig, summary_likes)

# # ────────────── 2️⃣ Distribution of Comments ──────────────
# st.subheader("2️⃣ Distribution of Comments")
# fig, ax = plt.subplots(figsize=(8,5))
# sns.boxplot(y=df['Comments'], ax=ax)
# ax.set_title('Distribution of Comments')
# comments_stats = df['Comments'].describe()
# summary_comments = f"""
# **Comments Summary**  
# {comments_stats.to_frame().to_markdown()}  
# - Conclusion: Most posts receive fewer comments than likes, indicating less interactive engagement; outliers show highly engaging posts.
# """
# plot_with_summary(fig, summary_comments)

# # ────────────── 3️⃣ Likes vs Shares ──────────────
# st.subheader("3️⃣ Likes vs Shares")
# fig, ax = plt.subplots(figsize=(8,5))
# sns.scatterplot(x=df['Likes'], y=df['Shares'], alpha=0.6, ax=ax)
# ax.set_title('Likes vs Shares')
# corr_ls = df['Likes'].corr(df['Shares'])
# summary_likes_shares = f"""
# **Likes vs Shares Summary**  
# - Correlation: {corr_ls:.2f}  
# - Conclusion: {'Posts with more likes tend to get more shares.' if corr_ls>0.5 else 'No strong relationship between likes and shares; engagement may vary by content type.'}
# """
# plot_with_summary(fig, summary_likes_shares)

# # ────────────── 4️⃣ Average Reach by Post Type ──────────────
# st.subheader("4️⃣ Average Reach by Post Type")
# avg_reach_by_post_type = df.groupby('Post_Type')['Reach'].mean().sort_values(ascending=False)
# fig, ax = plt.subplots(figsize=(8,5))
# sns.barplot(x=avg_reach_by_post_type.index, y=avg_reach_by_post_type.values, palette='viridis', ax=ax)
# plt.xticks(rotation=45)
# ax.set_title('Average Reach by Post Type')
# summary_reach = f"""
# **Average Reach by Post Type**  
# {avg_reach_by_post_type.to_frame().to_markdown()}  
# - Conclusion: {avg_reach_by_post_type.idxmax()} posts reach the largest audience; prioritize these post types for maximum reach.
# """
# plot_with_summary(fig, summary_reach)

# # ────────────── 5️⃣ Impressions Over Time ──────────────
# st.subheader("5️⃣ Impressions Over Time")
# df_sorted = df.sort_values('Post_Date')
# fig, ax = plt.subplots(figsize=(10,5))
# ax.plot(df_sorted['Post_Date'], df_sorted['Impressions'])
# ax.set_title('Impressions Over Time')
# plt.xticks(rotation=45)
# trend = df_sorted['Impressions'].diff().mean()
# summary_impressions = f"""
# **Impressions Over Time**  
# - Trend: {'Increasing 📈' if trend>0 else 'Decreasing 📉'}  
# - Conclusion: Overall impressions are {'growing' if trend>0 else 'declining'}, indicating audience expansion or contraction over time.
# """
# plot_with_summary(fig, summary_impressions)

# # ────────────── 6️⃣ Correlation Matrix ──────────────
# st.subheader("6️⃣ Correlation Matrix Heatmap")
# numerical_df = df.select_dtypes(include=['float64','int64'])
# if not numerical_df.empty:
#     fig, ax = plt.subplots(figsize=(8,6))
#     sns.heatmap(numerical_df.corr(), annot=False, cmap='coolwarm', ax=ax)
#     ax.set_title('Correlation Matrix')
#     summary_corr = "**Correlation Matrix** shows relationships between numerical metrics. Values close to 1 or -1 indicate strong positive or negative relationships."
#     plot_with_summary(fig, summary_corr)

# # ────────────── 7️⃣ Posts by Day of Week ──────────────
# if 'Day_of_Week' in df.columns:
#     st.subheader("7️⃣ Count of Posts by Day of Week")
#     fig, ax = plt.subplots(figsize=(8,5))
#     sns.countplot(data=df, x='Day_of_Week', order=df['Day_of_Week'].value_counts().index, palette='viridis', ax=ax)
#     ax.set_title('Posts by Day of Week')
#     plt.xticks(rotation=45)
#     counts_day = df['Day_of_Week'].value_counts()
#     summary_day = f"""
# **Posts by Day of Week**  
# {counts_day.to_frame().to_markdown()}  
# - Conclusion: Highest posts occur on {counts_day.idxmax()}; plan important content accordingly.
# """
#     plot_with_summary(fig, summary_day)

# # ────────────── 8️⃣ Posts by Post Type ──────────────
# st.subheader("8️⃣ Count of Posts by Post Type")
# fig, ax = plt.subplots(figsize=(8,5))
# sns.countplot(data=df, x='Post_Type', order=df['Post_Type'].value_counts().index, palette='viridis', ax=ax)
# ax.set_title('Posts by Post Type')
# plt.xticks(rotation=45)
# counts_type = df['Post_Type'].value_counts()
# summary_type = f"""
# **Posts by Post Type**  
# {counts_type.to_frame().to_markdown()}  
# - Conclusion: Most posts are {counts_type.idxmax()}; reflects current content strategy.
# """
# plot_with_summary(fig, summary_type)

# # ────────────── 9️⃣ Average Engagement by Post Type ──────────────
# if 'Engagement_Total' in df.columns:
#     st.subheader("🔟 Engagement Rate Distribution by Day of Week")
#     avg_eng_by_type = df.groupby('Post_Type')['Engagement_Total'].mean().sort_values(ascending=False)
#     fig, ax = plt.subplots(figsize=(8,5))
#     sns.barplot(x=avg_eng_by_type.index, y=avg_eng_by_type.values, palette='viridis', ax=ax)
#     plt.xticks(rotation=45)
#     ax.set_title('Average Engagement Rate by Post Type')
#     summary_engagement = f"""
# **Average Engagement Rate by Post Type**  
# {avg_eng_by_type.to_frame().to_markdown()}  
# - Conclusion: {avg_eng_by_type.idxmax()} posts get highest engagement; replicate similar content for higher interaction.
# """
#     plot_with_summary(fig, summary_engagement)

# # ────────────── 10️⃣ Engagement Distribution by Day ──────────────
# st.subheader("🔟 Engagement Rate Distribution by Day of Week")
# if 'Day_of_Week' in df.columns and 'Engagement_Total' in df.columns:
#     fig, ax = plt.subplots(figsize=(8,5))
#     sns.boxplot(data=df, x='Day_of_Week', y='Engagement_Total', palette='viridis', ax=ax)
#     ax.set_title('Engagement Distribution by Day')
#     plt.xticks(rotation=45)
#     summary_eng_day = f"""
# **Engagement Distribution by Day**  
# {df.groupby('Day_of_Week')['Engagement_Total'].describe().to_markdown()}  
# - Conclusion: Engagement varies across days; schedule high-priority posts on days with higher median engagement.
# """
#     plot_with_summary(fig, summary_eng_day)

# # ────────────── 11️⃣ Story Completion Rate ──────────────
# if 'Story_Completion_Rate' in df.columns:
#     st.subheader("11️⃣ Average Story Completion Rate by Day of Week")
#     avg_story_by_day = df.groupby('Day_of_Week')['Story_Completion_Rate'].mean().sort_values(ascending=False)
#     fig, ax = plt.subplots(figsize=(8,5))
#     sns.barplot(x=avg_story_by_day.index, y=avg_story_by_day.values, palette='viridis', ax=ax)
#     plt.xticks(rotation=45)
#     ax.set_title('Average Story Completion Rate by Day')
#     summary_story_day = f"""
# **Story Completion Rate by Day**  
# {avg_story_by_day.to_frame().to_markdown()}  
# - Conclusion: Highest completion on {avg_story_by_day.idxmax()}; schedule stories on these days for better retention.
# """
#     plot_with_summary(fig, summary_story_day)

#     # Story completion by post type
#     fig, ax = plt.subplots(figsize=(8,5))
#     sns.boxplot(data=df, x='Post_Type', y='Story_Completion_Rate', palette='viridis', ax=ax)
#     plt.xticks(rotation=45)
#     ax.set_title('Story Completion Rate by Post Type')
#     summary_story_type = f"""
# **Story Completion Rate by Post Type**  
# {df.groupby('Post_Type')['Story_Completion_Rate'].describe().to_markdown()}  
# - Conclusion: Certain post types keep viewers engaged longer; focus on these types for story content.
# """
#     plot_with_summary(fig, summary_story_type)

# # ----------------------------------------------------------------------------------------------------------

# # ────────────── 12️⃣ Average Ad Spend by Post Type ──────────────
# if 'Ad_Spend_USD' in df.columns:
#     st.subheader("12️⃣ Average Ad Spend (USD) by Post Type")
#     avg_ad_spend_by_post_type = df.groupby('Post_Type')['Ad_Spend_USD'].mean().sort_values(ascending=False)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.barplot(x=avg_ad_spend_by_post_type.index, y=avg_ad_spend_by_post_type.values, palette='viridis', ax=ax)
#     plt.xticks(rotation=45)
#     ax.set_title("Average Ad Spend by Post Type")
#     summary_ad_spend = f"""
# **Average Ad Spend by Post Type**  
# {avg_ad_spend_by_post_type.to_frame().to_markdown()}  
# - Conclusion: {avg_ad_spend_by_post_type.idxmax()} posts have the highest ad spend; optimize budget accordingly.
# """
#     plot_with_summary(fig, summary_ad_spend)

# # ────────────── 13️⃣ Average Ad Revenue by Post Type ──────────────
# if 'Ad_Revenue_USD' in df.columns:
#     st.subheader("13️⃣ Average Ad Revenue (USD) by Post Type")
#     avg_ad_revenue_by_post_type = df.groupby('Post_Type')['Ad_Revenue_USD'].mean().sort_values(ascending=False)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.barplot(x=avg_ad_revenue_by_post_type.index, y=avg_ad_revenue_by_post_type.values, palette='viridis', ax=ax)
#     plt.xticks(rotation=45)
#     ax.set_title("Average Ad Revenue by Post Type")
#     summary_ad_revenue = f"""
# **Average Ad Revenue by Post Type**  
# {avg_ad_revenue_by_post_type.to_frame().to_markdown()}  
# - Conclusion: {avg_ad_revenue_by_post_type.idxmax()} posts generate the most revenue; focus on similar content.
# """
#     plot_with_summary(fig, summary_ad_revenue)

# # ────────────── 14️⃣ Average ROAS by Post Type ──────────────
# if 'ROAS' in df.columns:
#     st.subheader("14️⃣ Average ROAS by Post Type")
#     avg_roas_by_post_type = df.groupby('Post_Type')['ROAS'].mean().sort_values(ascending=False)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.barplot(x=avg_roas_by_post_type.index, y=avg_roas_by_post_type.values, palette='viridis', ax=ax)
#     plt.xticks(rotation=45)
#     ax.set_title("Average ROAS by Post Type")
#     summary_roas = f"""
# **Average ROAS by Post Type**  
# {avg_roas_by_post_type.to_frame().to_markdown()}  
# - Conclusion: {avg_roas_by_post_type.idxmax()} posts have the highest ROAS; maximize return on ad spend by posting more of these.
# """
#     plot_with_summary(fig, summary_roas)

#     st.subheader("ROAS Distribution by Day of Week")
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.boxplot(data=df, x='Day_of_Week', y='ROAS', palette='viridis', ax=ax)
#     plt.xticks(rotation=45)
#     ax.set_title("ROAS Distribution by Day")
#     summary_roas_dist = "Boxplot showing ROAS distribution by day of the week. Helps plan posting days for optimal ROI."
#     plot_with_summary(fig, summary_roas_dist)

# # ────────────── 15️⃣ Average CPA by Post Type ──────────────
# if 'CPA_USD' in df.columns:
#     st.subheader("15️⃣ Average CPA (USD) by Post Type")
#     avg_cpa_by_post_type = df.groupby('Post_Type')['CPA_USD'].mean().sort_values(ascending=False)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.barplot(x=avg_cpa_by_post_type.index, y=avg_cpa_by_post_type.values, palette='viridis', ax=ax)
#     plt.xticks(rotation=45)
#     ax.set_title("Average CPA by Post Type")
#     summary_cpa = f"""
# **Average CPA by Post Type**  
# {avg_cpa_by_post_type.to_frame().to_markdown()}  
# - Conclusion: {avg_cpa_by_post_type.idxmin()} posts have the lowest CPA; efficient for cost optimization.
# """
#     plot_with_summary(fig, summary_cpa)

#     st.subheader("CPA (USD) Distribution by Day of Week")
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.boxplot(data=df, x='Day_of_Week', y='CPA_USD', palette='viridis', ax=ax)
#     plt.xticks(rotation=45)
#     ax.set_title("CPA Distribution by Day")
#     summary_cpa_dist = "Boxplot showing CPA distribution by day of the week. Helps identify cost-effective posting days."
#     plot_with_summary(fig, summary_cpa_dist)

# # ────────────── 16️⃣ Top 10 Hashtags ──────────────
# if 'Top_Hashtag_1' in df.columns:
#     st.subheader("16️⃣ Top 10 #Top_Hashtag_1")
#     top_10_hashtag_1 = df['Top_Hashtag_1'].value_counts().nlargest(10)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.barplot(x=top_10_hashtag_1.index, y=top_10_hashtag_1.values, palette='viridis', ax=ax)
#     plt.xticks(rotation=45, ha='right')
#     ax.set_title("Top 10 Hashtags (#Top_Hashtag_1)")
#     summary_hashtag1 = f"""
# **Top Hashtags (#Top_Hashtag_1)**  
# {top_10_hashtag_1.to_frame().to_markdown()}  
# - Conclusion: These hashtags are trending; using them can increase reach and engagement.
# """
#     plot_with_summary(fig, summary_hashtag1)

# if 'Top_Hashtag_2' in df.columns:
#     st.subheader("17️⃣ Top 10 #Top_Hashtag_2")
#     top_10_hashtag_2 = df['Top_Hashtag_2'].value_counts().nlargest(10)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.barplot(x=top_10_hashtag_2.index, y=top_10_hashtag_2.values, palette='viridis', ax=ax)
#     plt.xticks(rotation=45, ha='right')
#     ax.set_title("Top 10 Hashtags (#Top_Hashtag_2)")
#     summary_hashtag2 = f"""
# **Top Hashtags (#Top_Hashtag_2)**  
# {top_10_hashtag_2.to_frame().to_markdown()}  
# - Conclusion: These hashtags are trending; using them strategically can boost post performance.
# """
#     plot_with_summary(fig, summary_hashtag2)

# # ────────────── 17️⃣ Count of Posts by Time of Day ──────────────
# if 'Time_of_Day' in df.columns:
#     st.subheader("17️⃣ Count of Posts by Time of Day")
#     df['Hour_of_Day'] = pd.to_datetime(df['Time_of_Day']).dt.hour

#     def time_category(hour):
#         if 5 <= hour < 12:
#             return 'Morning'
#         elif 12 <= hour < 17:
#             return 'Afternoon'
#         elif 17 <= hour < 21:
#             return 'Evening'
#         else:
#             return 'Night'

#     df['Time_Category'] = df['Hour_of_Day'].apply(time_category)
#     time_counts = df['Time_Category'].value_counts().reindex(['Morning','Afternoon','Evening','Night'])

#     fig, ax = plt.subplots(figsize=(8, 5))
#     sns.countplot(data=df, x='Time_Category', order=['Morning','Afternoon','Evening','Night'], palette='viridis', ax=ax)
#     ax.set_title("Count of Posts by Time of Day")
#     summary_time_of_day = f"""
# **Posts by Time of Day**  
# {time_counts.to_frame().to_markdown()}  
# - Conclusion: Most posts are published during {time_counts.idxmax()}; consider this when scheduling for higher engagement.
# """
#     plot_with_summary(fig, summary_time_of_day)

