# =====================================
# GOOGLE ADS LANDING PAGE REPORT DASHBOARD
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# =====================================
# 1. STREAMLIT PAGE SETUP
# =====================================
st.set_page_config(
    page_title="Landing Page Report Dashboard",
    layout="wide"
)
st.title("🌐 Google Ads Landing Page Report Dashboard")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Landing_page_report.csv")

# =====================================
# 2. LOAD DATA
# =====================================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, skiprows=2, sep=",", engine="python")

    # Clean column names
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
        .str.replace("/", "_", regex=False)
    )

    # Clean numeric-like columns
    numeric_like_cols = [
        "impr","interactions","interaction_rate","avg_cost",
        "cost","conversions","conv_rate","clicks","avg_cpc",
        "ctr","mobile_speed_score"
    ]

    for col in numeric_like_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("₹", "", regex=False)
                .str.replace("inr", "", case=False, regex=False)
                .str.replace("--", "", regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

df = load_data(DATA_PATH)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# =====================================
# 3. SIDEBAR FILTERS (PROPER SESSION MGMT)
# =====================================

# =====================================
# SIDEBAR : FILTERS + CLEAR BUTTON
# =====================================
with st.sidebar:
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.header("🔍 Filters")

    with col_f2:
        if st.button("❌ Clear Filters"):
            # Clear landing page filters
            st.session_state.select_all_landing = False
            st.session_state.filter_landing_page = []

            # Clear other categorical filters
            for col in categorical_cols:
                if col != "landing_page":
                    st.session_state[f"filter_{col}"] = df[col].dropna().unique().tolist()

            st.rerun()


# =====================================
# LANDING PAGE FILTER
# =====================================
landing_pages = df["landing_page"].dropna().unique().tolist()

if "select_all_landing" not in st.session_state:
    st.session_state.select_all_landing = False

if "filter_landing_page" not in st.session_state:
    st.session_state.filter_landing_page = []

# Select All checkbox
select_all = st.sidebar.checkbox(
    "Select All Landing Pages",
    key="select_all_landing"
)

if select_all:
    st.session_state.filter_landing_page = landing_pages
else:
    # Clear only if user unchecks manually
    if st.session_state.filter_landing_page == landing_pages:
        st.session_state.filter_landing_page = []

# Multiselect (widget OWNS the state)
st.sidebar.multiselect(
    "Select Landing Page(s)",
    options=landing_pages,
    key="filter_landing_page"
)


# =====================================
# OTHER CATEGORICAL FILTERS
# =====================================
for col in categorical_cols:
    if col == "landing_page":
        continue

    if f"filter_{col}" not in st.session_state:
        st.session_state[f"filter_{col}"] = df[col].dropna().unique().tolist()

    st.sidebar.multiselect(
        f"Select {col}",
        options=df[col].dropna().unique().tolist(),
        key=f"filter_{col}"
    )


# =====================================
# APPLY FILTERS
# =====================================
filtered_df = df.copy()

for col in categorical_cols:
    filtered_df = filtered_df[
        filtered_df[col].isin(st.session_state.get(f"filter_{col}", []))
    ]


# =====================================
# EMPTY STATE HANDLING
# =====================================
if not st.session_state.filter_landing_page or filtered_df.empty:
    st.warning("⚠️ Please select at least one Landing Page to display the graphs.")
    st.stop()

# =====================================
# 6. HELPER FUNCTIONS
# =====================================
def numerical_plot(df, col, n):
    data = df[col].dropna()
    if data.empty:
        return None, None

    fig = px.histogram(
        data,
        nbins=30,
        title=f"{n}️⃣ Distribution of {col}"
    )

    stats = {
        "mean": data.mean(),
        "median": data.median(),
        "skew": data.skew(),
        "outlier_pct": (
            ((data < data.quantile(0.25) - 1.5*(data.quantile(0.75)-data.quantile(0.25))) |
             (data > data.quantile(0.75) + 1.5*(data.quantile(0.75)-data.quantile(0.25))))
            .mean() * 100
        )
    }
    return fig, stats

def scatter_plot(df, x, y, n, logx=False, logy=False):
    d = df[[x, y]].dropna()
    if d.empty:
        return None
    return px.scatter(
        d, x=x, y=y,
        log_x=logx, log_y=logy,
        title=f"{n}️⃣ {x.replace('_',' ').title()} vs {y.replace('_',' ').title()}"
    )

# =====================================
# 7. NUMERICAL ANALYSIS
# =====================================
graph_count = 1
st.header("📊 Numerical Feature Analysis")

for col in numeric_cols:
    fig, stats = numerical_plot(filtered_df, col, graph_count)
    if fig:
        c1, c2 = st.columns([3,2])
        c1.plotly_chart(fig, use_container_width=True)
        c2.markdown(
            f"""
**Summary**
- Mean: {stats['mean']:.2f}
- Median: {stats['median']:.2f}
- Skewness: {stats['skew']:.2f}
- Outliers: {stats['outlier_pct']:.2f}%
"""
        )
        graph_count += 1

# =====================================
# 8. TOP LANDING PAGES
# =====================================
st.header("📊 Landing Page Insights")

top_lp = filtered_df.sort_values("clicks", ascending=False).head(10)

fig = px.bar(
    top_lp,
    x="clicks",
    y="landing_page",
    orientation="h",
    title=f"{graph_count}️⃣ Top Landing Pages by Clicks"
)

c1, c2 = st.columns([3,2])
c1.plotly_chart(fig, use_container_width=True)
c2.markdown(
    f"""
**Insight**
- These landing pages drive the highest traffic
- Total clicks: {int(top_lp['clicks'].sum())}
"""
)
graph_count += 1

# =====================================
# 9. SCATTER INSIGHTS
# =====================================
fig = scatter_plot(filtered_df, "cost", "clicks", graph_count, True, True)
if fig:
    c1, c2 = st.columns([3,2])
    c1.plotly_chart(fig, use_container_width=True)
    c2.markdown("Higher cost does not always guarantee higher clicks.")
    graph_count += 1

fig = scatter_plot(filtered_df, "mobile_speed_score", "ctr", graph_count, True, True)
if fig:
    c1, c2 = st.columns([3,2])
    c1.plotly_chart(fig, use_container_width=True)
    c2.markdown("Faster mobile pages generally improve CTR.")
    graph_count += 1

# =====================================
# 10. KPIs
# =====================================
st.header("📊 Summary KPIs")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric("Landing Pages", filtered_df["landing_page"].nunique())
k2.metric("Impressions", int(filtered_df["impr"].sum()))
k3.metric("Clicks", int(filtered_df["clicks"].sum()))
k4.metric("Avg CPC", round(filtered_df["avg_cpc"].mean(), 2))
k5.metric("Total Cost", round(filtered_df["cost"].sum(), 2))

# # =====================================
# # GOOGLE ADS LANDING PAGE REPORT DASHBOARD
# # =====================================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from dotenv import load_dotenv
# import os

# # =====================================
# # 1. STREAMLIT PAGE SETUP
# # =====================================
# st.set_page_config(
#     page_title="Landing Page Report Dashboard",
#     layout="wide"
# )
# st.title("🌐 Google Ads Landing Page Report Dashboard")

# load_dotenv()
# DATA_PATH = os.getenv("GA_LANDING_PAGE")  # Env variable pointing to CSV

# # =====================================
# # 2. LOAD DATA
# # =====================================
# @st.cache_data
# def load_data(path):
#     df = pd.read_csv(path, skiprows=2, sep=",", engine="python")
#     # Clean columns
#     df.columns = (
#         df.columns
#         .astype(str)
#         .str.strip()
#         .str.lower()
#         .str.replace(" ", "_")
#         .str.replace(".", "", regex=False)
#         .str.replace("/", "_", regex=False)
#     )
#     # Clean numeric-like columns
#     numeric_like_cols = [
#         'impr','interactions','interaction_rate','avg_cost',
#         'cost','conversions','conv_rate','clicks','avg_cpc',
#         'ctr','mobile_speed_score'
#     ]
#     for col in numeric_like_cols:
#         if col in df.columns:
#             df[col] = (
#                 df[col].astype(str)
#                      .str.replace(",", "", regex=False)
#                      .str.replace("%", "", regex=False)
#                      .str.replace("₹", "", regex=False)
#                      .str.replace("inr", "", case=False, regex=False)
#                      .str.replace("--", "", regex=False)
#                      .str.strip()
#             )
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#     return df

# df = load_data(DATA_PATH)

# # Identify numeric and categorical columns
# numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# # =====================================
# # 3. SIDEBAR FILTERS WITH SESSION MANAGEMENT
# # =====================================
# st.sidebar.header("🔍 Filters")

# # Select All checkbox
# if 'select_all_landing' not in st.session_state:
#     st.session_state['select_all_landing'] = False

# # Landing Page filter
# if 'filter_landing_page' not in st.session_state:
#     st.session_state['filter_landing_page'] = []

# select_all = st.sidebar.checkbox(
#     "Select All Landing Pages",
#     value=st.session_state['select_all_landing']
# )

# if select_all:
#     st.session_state['filter_landing_page'] = df['landing_page'].dropna().unique().tolist()
#     st.session_state['select_all_landing'] = True
# else:
#     selected_lp = st.sidebar.multiselect(
#         "Select Landing Page(s)",
#         options=df['landing_page'].dropna().unique().tolist(),
#         default=st.session_state['filter_landing_page']
#     )
#     st.session_state['filter_landing_page'] = selected_lp
#     st.session_state['select_all_landing'] = False

# # Initialize session_state for other categorical filters
# for col in categorical_cols:
#     if col != 'landing_page':
#         if f"filter_{col}" not in st.session_state:
#             st.session_state[f"filter_{col}"] = df[col].dropna().unique().tolist()

#         selected = st.sidebar.multiselect(
#             f"Select {col}",
#             options=df[col].dropna().unique().tolist(),
#             default=st.session_state[f"filter_{col}"]
#         )
#         if not selected:
#             selected = df[col].dropna().unique().tolist()
#         st.session_state[f"filter_{col}"] = selected

# # Apply all filters to dataframe
# filtered_df = df.copy()
# for col in categorical_cols:
#     filtered_df = filtered_df[filtered_df[col].isin(st.session_state[f"filter_{col}"])]

# # ⚠️ SHOW WARNING IF NO LANDING PAGE IS SELECTED
# if not st.session_state['filter_landing_page'] or filtered_df.empty:
#     st.warning("⚠️ Please select at least one Landing Page to display the graphs.")
# else:
#     # =====================================
#     # 4. NUMERICAL EDA FUNCTION
#     # =====================================
#     def numerical_eda_plotly(df, col, graph_num):
#         data = df[col].dropna()
#         if data.empty:
#             return None, None
#         mean = data.mean()
#         median = data.median()
#         skew = data.skew()
#         q1, q3 = data.quantile([0.25, 0.75])
#         iqr = q3 - q1
#         outlier_pct = ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).mean()*100
#         fig = px.histogram(
#             data,
#             nbins=30,
#             title=f"{graph_num}️⃣ Distribution of {col}",
#             labels={col: col, "count": "Frequency"}
#         )
#         return fig, {"mean": mean, "median": median, "skew": skew, "outlier_pct": outlier_pct}

#     # =====================================
#     # 5. CATEGORICAL EDA FUNCTION
#     # =====================================
#     def categorical_eda_plotly(df, col, graph_num):
#         data = df[col].dropna()
#         if data.empty:
#             return None, None
#         vc = data.value_counts().head(10)
#         fig = px.bar(
#             x=vc.index,
#             y=vc.values,
#             labels={"x": col, "y": "Count"},
#             title=f"{graph_num}️⃣ Category Distribution: {col}"
#         )
#         return fig, vc

#     # =====================================
#     # 6. SCATTER / SPECIAL PLOTS FUNCTION
#     # =====================================
#     def scatter_plot(df, x_col, y_col, graph_num, log_x=False, log_y=False):
#         data = df[[x_col, y_col]].dropna()
#         if data.empty:
#             return None
#         fig = px.scatter(
#             data,
#             x=x_col,
#             y=y_col,
#             title=f"{graph_num}️⃣ {x_col.replace('_',' ').title()} vs {y_col.replace('_',' ').title()}",
#             labels={x_col: x_col, y_col: y_col},
#             log_x=log_x,
#             log_y=log_y
#         )
#         return fig

#     # =====================================
#     # 7. DISPLAY NUMERICAL GRAPHS
#     # =====================================
#     graph_count = 1
#     st.header("📊 Numerical Feature Analysis")

#     for col in numeric_cols:
#         fig, stats = numerical_eda_plotly(filtered_df, col, graph_count)
#         if fig:
#             col1, col2 = st.columns([3,2])
#             col1.plotly_chart(fig, use_container_width=True)
#             col2.markdown(
#                 f"""
# **Summary for {col}**
# - Average Value      : {stats['mean']:.2f}  
# - Typical Value      : {stats['median']:.2f}  
# - Distribution Skew  : {stats['skew']:.2f}  
# - Extreme Values (%) : {stats['outlier_pct']:.2f}%  
# """
#             )
#             graph_count += 1

#     # =====================================
#     # 8. DISPLAY CATEGORICAL GRAPHS
#     # =====================================
#     st.header("📊 Categorical Feature Analysis")
#     for col in categorical_cols:
#         fig, vc = categorical_eda_plotly(filtered_df, col, graph_count)
#         if fig:
#             col1, col2 = st.columns([3,2])
#             col1.plotly_chart(fig, use_container_width=True)
#             summary_text = "\n".join([f"{idx}: {round(val,0)}" for idx,val in zip(vc.index, vc.values)])
#             col2.markdown(
#                 f"""
# **Top Categories for {col}**
# {summary_text}
# """
#             )
#             graph_count += 1

#     # =====================================
#     # 9. SPECIAL PLOTS FROM ORIGINAL CODE
#     # =====================================
#     st.header("📊 Landing Page Insights")

#     # Top Landing Pages by Clicks
#     if 'landing_page' in filtered_df.columns and 'clicks' in filtered_df.columns:
#         top_lp = filtered_df.sort_values('clicks', ascending=False).head(10)
#         fig = px.bar(
#             top_lp,
#             x='clicks',
#             y='landing_page',
#             orientation='h',
#             title=f"{graph_count}️⃣ Top Landing Pages by Clicks"
#         )
#         col1, col2 = st.columns([3,2])
#         col1.plotly_chart(fig, use_container_width=True)
#         col2.markdown(
#             f"Top 10 landing pages driving clicks.\nTotal clicks: {top_lp['clicks'].sum()}"
#         )
#         graph_count += 1

#     # Cost vs Clicks (log scale)
#     if 'cost' in filtered_df.columns and 'clicks' in filtered_df.columns:
#         fig = scatter_plot(filtered_df, 'cost', 'clicks', graph_count, log_x=True, log_y=True)
#         if fig:
#             col1, col2 = st.columns([3,2])
#             col1.plotly_chart(fig, use_container_width=True)
#             col2.markdown("Relationship between cost and clicks (log scale).")
#             graph_count += 1

#     # Mobile Speed Score vs CTR
#     if 'mobile_speed_score' in filtered_df.columns and 'ctr' in filtered_df.columns:
#         fig = scatter_plot(filtered_df, 'mobile_speed_score', 'ctr', graph_count, log_x=True, log_y=True)
#         if fig:
#             col1, col2 = st.columns([3,2])
#             col1.plotly_chart(fig, use_container_width=True)
#             col2.markdown("Relationship between mobile page speed and CTR (log scale).")
#             graph_count += 1

#     # =====================================
#     # 10. FINAL KPIs
#     # =====================================
#     st.header("📊 Summary KPIs")
#     kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

#     kpi1.metric("Landing Pages", filtered_df['landing_page'].nunique() if 'landing_page' in filtered_df.columns else 0)
#     kpi2.metric("Total Impressions", int(filtered_df['impr'].sum()) if 'impr' in filtered_df.columns else 0)
#     kpi3.metric("Total Clicks", int(filtered_df['clicks'].sum()) if 'clicks' in filtered_df.columns else 0)
#     kpi4.metric("Average CPC", round(filtered_df['avg_cpc'].mean(),2) if 'avg_cpc' in filtered_df.columns else 0)
#     kpi5.metric("Total Cost", round(filtered_df['cost'].sum(),2) if 'cost' in filtered_df.columns else 0)

# ----------------------------------------------------------------------------------------------------

# # =====================================
# # GOOGLE ADS LANDING PAGE REPORT DASHBOARD
# # =====================================

# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from dotenv import load_dotenv
# import os

# # =====================================
# # 1. STREAMLIT PAGE SETUP
# # =====================================
# st.set_page_config(
#     page_title="Landing Page Report Dashboard",
#     layout="wide"
# )
# st.title("🌐 Google Ads Landing Page Report Dashboard")

# load_dotenv()
# DATA_PATH = os.getenv("GA_LANDING_PAGE")  # Make sure this env variable points to your CSV

# # =====================================
# # 2. LOAD DATA
# # =====================================
# @st.cache_data
# def load_data(path):
#     df = pd.read_csv(path, skiprows=2, sep=",", engine="python")
#     # Clean columns
#     df.columns = (
#         df.columns
#         .astype(str)
#         .str.strip()
#         .str.lower()
#         .str.replace(" ", "_")
#         .str.replace(".", "", regex=False)
#         .str.replace("/", "_", regex=False)
#     )
#     # Clean numeric-like columns
#     numeric_like_cols = [
#         'impr','interactions','interaction_rate','avg_cost',
#         'cost','conversions','conv_rate','clicks','avg_cpc',
#         'ctr','mobile_speed_score'
#     ]
#     for col in numeric_like_cols:
#         if col in df.columns:
#             df[col] = (
#                 df[col].astype(str)
#                      .str.replace(",", "", regex=False)
#                      .str.replace("%", "", regex=False)
#                      .str.replace("₹", "", regex=False)
#                      .str.replace("inr", "", case=False, regex=False)
#                      .str.replace("--", "", regex=False)
#                      .str.strip()
#             )
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#     return df

# df = load_data(DATA_PATH)

# # Identify numeric and categorical columns
# numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# # =====================================
# # 3. SIDEBAR FILTERS WITH SESSION MANAGEMENT
# # =====================================

# st.sidebar.header("🔍 Filters")

# # Select All checkbox
# if 'select_all_landing' not in st.session_state:
#     st.session_state['select_all_landing'] = False

# # Landing Page filter
# if 'filter_landing_page' not in st.session_state:
#     st.session_state['filter_landing_page'] = []

# select_all = st.sidebar.checkbox(
#     "Select All Landing Pages",
#     value=st.session_state['select_all_landing']
# )

# if select_all:
#     st.session_state['filter_landing_page'] = df['landing_page'].dropna().unique().tolist()
#     st.session_state['select_all_landing'] = True
# else:
#     # Multiselect with NONE selected by default
#     selected_lp = st.sidebar.multiselect(
#         "Select Landing Page(s)",
#         options=df['landing_page'].dropna().unique().tolist(),
#         default=st.session_state['filter_landing_page']
#     )
#     st.session_state['filter_landing_page'] = selected_lp
#     st.session_state['select_all_landing'] = False

# # Initialize session_state for other categorical filters
# for col in categorical_cols:
#     if col != 'landing_page':
#         if f"filter_{col}" not in st.session_state:
#             st.session_state[f"filter_{col}"] = df[col].dropna().unique().tolist()

#         selected = st.sidebar.multiselect(
#             f"Select {col}",
#             options=df[col].dropna().unique().tolist(),
#             default=st.session_state[f"filter_{col}"]
#         )
#         if not selected:
#             selected = df[col].dropna().unique().tolist()
#         st.session_state[f"filter_{col}"] = selected

# # Apply all filters to dataframe
# filtered_df = df.copy()
# for col in categorical_cols:
#     filtered_df = filtered_df[filtered_df[col].isin(st.session_state[f"filter_{col}"])]

# # =====================================
# # 4. NUMERICAL EDA FUNCTION
# # =====================================
# def numerical_eda_plotly(df, col, graph_num):
#     data = df[col].dropna()
#     if data.empty:
#         return None, None
#     mean = data.mean()
#     median = data.median()
#     skew = data.skew()
#     q1, q3 = data.quantile([0.25, 0.75])
#     iqr = q3 - q1
#     outlier_pct = ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).mean()*100
#     fig = px.histogram(
#         data,
#         nbins=30,
#         title=f"{graph_num}️⃣ Distribution of {col}",
#         labels={col: col, "count": "Frequency"}
#     )
#     return fig, {"mean": mean, "median": median, "skew": skew, "outlier_pct": outlier_pct}

# # =====================================
# # 5. CATEGORICAL EDA FUNCTION
# # =====================================
# def categorical_eda_plotly(df, col, graph_num):
#     data = df[col].dropna()
#     if data.empty:
#         return None, None
#     vc = data.value_counts().head(10)
#     fig = px.bar(
#         x=vc.index,
#         y=vc.values,
#         labels={"x": col, "y": "Count"},
#         title=f"{graph_num}️⃣ Category Distribution: {col}"
#     )
#     return fig, vc

# # =====================================
# # 6. SCATTER / SPECIAL PLOTS FUNCTION
# # =====================================
# def scatter_plot(df, x_col, y_col, graph_num, log_x=False, log_y=False):
#     data = df[[x_col, y_col]].dropna()
#     if data.empty:
#         return None
#     fig = px.scatter(
#         data,
#         x=x_col,
#         y=y_col,
#         title=f"{graph_num}️⃣ {x_col.replace('_',' ').title()} vs {y_col.replace('_',' ').title()}",
#         labels={x_col: x_col, y_col: y_col},
#         log_x=log_x,
#         log_y=log_y
#     )
#     return fig

# # =====================================
# # 7. DISPLAY NUMERICAL GRAPHS
# # =====================================
# graph_count = 1
# st.header("📊 Numerical Feature Analysis")

# for col in numeric_cols:
#     fig, stats = numerical_eda_plotly(filtered_df, col, graph_count)
#     if fig:
#         col1, col2 = st.columns([3,2])
#         col1.plotly_chart(fig, use_container_width=True)
#         col2.markdown(
#             f"""
# **Summary for {col}**
# - Average Value      : {stats['mean']:.2f}  
# - Typical Value      : {stats['median']:.2f}  
# - Distribution Skew  : {stats['skew']:.2f}  
# - Extreme Values (%) : {stats['outlier_pct']:.2f}%  
# """
#         )
#         graph_count += 1

# # =====================================
# # 8. DISPLAY CATEGORICAL GRAPHS
# # =====================================
# st.header("📊 Categorical Feature Analysis")
# for col in categorical_cols:
#     fig, vc = categorical_eda_plotly(filtered_df, col, graph_count)
#     if fig:
#         col1, col2 = st.columns([3,2])
#         col1.plotly_chart(fig, use_container_width=True)
#         summary_text = "\n".join([f"{idx}: {round(val,0)}" for idx,val in zip(vc.index, vc.values)])
#         col2.markdown(
#             f"""
# **Top Categories for {col}**
# {summary_text}
# """
#         )
#         graph_count += 1

# # =====================================
# # 9. SPECIAL PLOTS FROM ORIGINAL CODE
# # =====================================
# st.header("📊 Landing Page Insights")

# # Top Landing Pages by Clicks
# if 'landing_page' in filtered_df.columns and 'clicks' in filtered_df.columns:
#     top_lp = filtered_df.sort_values('clicks', ascending=False).head(10)
#     fig = px.bar(
#         top_lp,
#         x='clicks',
#         y='landing_page',
#         orientation='h',
#         title=f"{graph_count}️⃣ Top Landing Pages by Clicks"
#     )
#     col1, col2 = st.columns([3,2])
#     col1.plotly_chart(fig, use_container_width=True)
#     col2.markdown(
#         f"Top 10 landing pages driving clicks.\nTotal clicks: {top_lp['clicks'].sum()}"
#     )
#     graph_count += 1

# # Cost vs Clicks (log scale)
# if 'cost' in filtered_df.columns and 'clicks' in filtered_df.columns:
#     fig = scatter_plot(filtered_df, 'cost', 'clicks', graph_count, log_x=True, log_y=True)
#     if fig:
#         col1, col2 = st.columns([3,2])
#         col1.plotly_chart(fig, use_container_width=True)
#         col2.markdown(
#             f"Relationship between cost and clicks (log scale)."
#         )
#         graph_count += 1

# # Mobile Speed Score vs CTR
# if 'mobile_speed_score' in filtered_df.columns and 'ctr' in filtered_df.columns:
#     fig = scatter_plot(filtered_df, 'mobile_speed_score', 'ctr', graph_count, log_x=True, log_y=True)
#     if fig:
#         col1, col2 = st.columns([3,2])
#         col1.plotly_chart(fig, use_container_width=True)
#         col2.markdown(
#             f"Relationship between mobile page speed and CTR (log scale)."
#         )
#         graph_count += 1

# # =====================================
# # 10. FINAL KPIs
# # =====================================
# st.header("📊 Summary KPIs")
# kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

# kpi1.metric("Landing Pages", filtered_df['landing_page'].nunique() if 'landing_page' in filtered_df.columns else 0)
# kpi2.metric("Total Impressions", int(filtered_df['impr'].sum()) if 'impr' in filtered_df.columns else 0)
# kpi3.metric("Total Clicks", int(filtered_df['clicks'].sum()) if 'clicks' in filtered_df.columns else 0)
# kpi4.metric("Average CPC", round(filtered_df['avg_cpc'].mean(),2) if 'avg_cpc' in filtered_df.columns else 0)
# kpi5.metric("Total Cost", round(filtered_df['cost'].sum(),2) if 'cost' in filtered_df.columns else 0)
