# =====================================
# GOOGLE ADS HOUSEHOLD INCOME REPORT DASHBOARD
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
import os

# =====================================
# 1. STREAMLIT PAGE SETUP
# =====================================
st.set_page_config(
    page_title="Household Income Report Dashboard",
    layout="wide"
)
st.title("💰 Google Ads Household Income Report Dashboard")

load_dotenv()
DATA_PATH = os.getenv("GA_HOUSEHOLD_INCOME")

# =====================================
# 2. LOAD DATA SAFELY
# =====================================
@st.cache_data
def load_data(path):
    # Try reading normally
    try:
        df = pd.read_csv(path, engine="python")
    except pd.errors.ParserError:
        # Skip first 2 rows if parsing fails (common in GA exports)
        df = pd.read_csv(path, engine="python", skiprows=2)

    # Remove summary rows like 'Total: Account'
    if df.shape[1] > 1:
        df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")].reset_index(drop=True)

    return df

df = load_data(DATA_PATH)

# =====================================
# 3. CLEAN NUMERIC-LIKE COLUMNS
# =====================================
def clean_numeric_column(series):
    return (
        series.astype(str)
              .str.strip()
              .replace(["--", "-", "NA", "N/A", ""], np.nan)
              .str.replace("%", "", regex=False)
              .str.replace(",", "", regex=False)
              .astype(float)
    )

numeric_like_cols = [
    "Impr.",
    "Interactions",
    "Interaction rate",
    "Conv. rate",
    "Conversions"
]

for col in numeric_like_cols:
    if col in df.columns:
        df[col] = clean_numeric_column(df[col])

# Identify numeric and categorical columns
num_cols = df.select_dtypes(include=np.number).columns.tolist()
cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# =====================================
# 4. SIDEBAR FILTERS (SESSION-MANAGED)
# =====================================

# ------------------ Sidebar: Title + Clear Button ------------------
with st.sidebar:
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.header("🔍 Filters")

    with col_f2:
        if st.button("❌ Clear"):
            for col in cat_cols:
                st.session_state[f"filter_{col}"] = df[col].dropna().unique().tolist()
            st.rerun()

# ------------------ Initialize Session State ------------------
for col in cat_cols:
    if f"filter_{col}" not in st.session_state:
        st.session_state[f"filter_{col}"] = df[col].dropna().unique().tolist()

# ------------------ Sidebar Multiselects (CORRECT) ------------------
for col in cat_cols:
    st.sidebar.multiselect(
        f"Select {col}",
        options=df[col].dropna().unique().tolist(),
        key=f"filter_{col}"
    )

# ------------------ Apply Filters ------------------
filtered_df = df.copy()

for col in cat_cols:
    filtered_df = filtered_df[
        filtered_df[col].isin(st.session_state[f"filter_{col}"])
    ]

# ------------------ Optional Empty Check ------------------
if filtered_df.empty:
    st.warning("⚠️ No data available for selected filters")
    st.stop()

# =====================================
# 5. NUMERICAL EDA FUNCTION
# =====================================
def numerical_eda_plotly(df, col, graph_num):
    data = df[col].dropna()
    if data.empty:
        return None, None

    mean = data.mean()
    median = data.median()
    skew = data.skew()
    q1, q3 = data.quantile([0.25, 0.75])
    iqr = q3 - q1
    outlier_pct = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).mean() * 100

    fig = px.histogram(
        data,
        nbins=30,
        title=f"{graph_num}️⃣ Distribution of {col}",
        labels={col: col, "count": "Frequency"}
    )

    return fig, {"mean": mean, "median": median, "skew": skew, "outlier_pct": outlier_pct}

# =====================================
# 6. CATEGORICAL EDA FUNCTION
# =====================================
def categorical_eda_plotly(df, col, graph_num):
    data = df[col].dropna()
    if data.empty:
        return None, None

    vc = data.value_counts().head(10)
    fig = px.bar(
        x=vc.index,
        y=vc.values,
        labels={"x": col, "y": "Count"},
        title=f"{graph_num}️⃣ Category Distribution: {col}"
    )
    return fig, vc

# =====================================
# 7. DISPLAY NUMERICAL GRAPHS
# =====================================
graph_count = 1
st.header("📊 Numerical Feature Analysis")

for col in num_cols:
    fig, stats = numerical_eda_plotly(filtered_df, col, graph_count)
    if fig:
        col1, col2 = st.columns([3, 2])
        col1.plotly_chart(fig, use_container_width=True)
        col2.markdown(
            f"""
**Summary for {col}**
- Average Value      : {stats['mean']:.2f}  
- Typical Value      : {stats['median']:.2f}  
- Distribution Skew  : {stats['skew']:.2f}  
- Extreme Values (%) : {stats['outlier_pct']:.2f}%  

**Recommended Action (Client Insight)**
"""
        )
        # Column-specific insights
        if col.lower() in ["impr.", "impressions"]:
            col2.markdown(
                "• Focus budget on campaigns/ad groups generating consistent reach  \n"
                "• Reduce spend where reach is high but interactions are low"
            )
        elif "interaction rate" in col.lower():
            col2.markdown(
                "• Scale creatives and audiences with above-median interaction rate  \n"
                "• Refresh creatives in low-performing segments"
            )
        elif "interactions" in col.lower():
            col2.markdown(
                "• Identify top interaction drivers and replicate messaging  \n"
                "• Pause ads with high impressions but low interactions"
            )
        elif "conv" in col.lower():
            col2.markdown(
                "• Allocate more budget to high-conversion segments  \n"
                "• Investigate drop-offs in low-converting campaigns"
            )
        elif "cost" in col.lower():
            col2.markdown(
                "• Shift spend from high-cost, low-return segments  \n"
                "• Optimize bidding strategies for cost-effective reach"
            )
        else:
            col2.markdown(
                "• Prioritize segments above median performance  \n"
                "• Deprioritize consistently underperforming values"
            )
        graph_count += 1

# =====================================
# 8. DISPLAY CATEGORICAL GRAPHS
# =====================================
st.header("📊 Categorical Feature Analysis")

for col in cat_cols:
    fig, vc = categorical_eda_plotly(filtered_df, col, graph_count)
    if fig:
        col1, col2 = st.columns([3, 2])
        col1.plotly_chart(fig, use_container_width=True)
        summary_text = "\n".join([f"{idx}: {round(val,0)}" for idx, val in zip(vc.index, vc.values)])
        col2.markdown(
            f"""
**Top Categories for {col}**
{summary_text}

**Recommended Action (Client Insight)**
"""
        )
        if vc.iloc[0] > 60:
            col2.markdown(
                f"• Majority traffic comes from '{vc.index[0]}'  \n"
                "• Optimize messaging and offers for dominant segment"
            )
        elif len(vc) > 15:
            col2.markdown(
                "• Performance spread across many categories  \n"
                "• Group similar segments to simplify targeting"
            )
        else:
            col2.markdown(
                "• Distribution is well balanced  \n"
                "• Compare engagement/conversion across categories"
            )
        graph_count += 1

# =====================================
# 9. CORRELATION MATRIX
# =====================================
if len(num_cols) > 1:
    st.header("📊 Correlation Analysis")
    corr = filtered_df[num_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title=f"{graph_count}️⃣ Correlation Matrix (Numerical Features)"
    )
    col1, col2 = st.columns([3, 2])
    col1.plotly_chart(fig, use_container_width=True)
    col2.markdown(
        """
**Correlation Summary**
- Shows relationship between cost, impressions & engagement metrics

**Recommended Action (Client Insight)**
- Identify metrics most strongly linked to conversions
- Prioritize optimization of these metrics (e.g., interaction rate)
- Avoid increasing spend on metrics that do not improve conversions
"""
    )
    graph_count += 1


# # =====================================
# # GOOGLE ADS HOUSEHOLD INCOME REPORT DASHBOARD
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
#     page_title="Household Income Report Dashboard",
#     layout="wide"
# )
# st.title("💰 Google Ads Household Income Report Dashboard")

# load_dotenv()

# DATA_PATH = os.getenv("GA_HOUSEHOLD_INCOME")

# # =====================================
# # 2. LOAD DATA SAFELY
# # =====================================

# @st.cache_data
# def load_data(path):
#     # Try reading normally
#     try:
#         df = pd.read_csv(path, engine="python")
#     except pd.errors.ParserError:
#         # If parsing fails, skip first 2 rows (common in GA exports)
#         df = pd.read_csv(path, engine="python", skiprows=2)
    
#     # Remove summary rows like 'Total: Account'
#     if df.shape[1] > 1:
#         df = df[~df.iloc[:,0].astype(str).str.startswith("Total")].reset_index(drop=True)
    
#     return df

# df = load_data(DATA_PATH)

# # =====================================
# # 3. CLEAN NUMERIC-LIKE COLUMNS
# # =====================================
# def clean_numeric_column(series):
#     return (
#         series.astype(str)
#               .str.strip()
#               .replace(["--", "-", "NA", "N/A", ""], np.nan)
#               .str.replace("%", "", regex=False)
#               .str.replace(",", "", regex=False)
#               .astype(float)
#     )

# numeric_like_cols = [
#     "Impr.",
#     "Interactions",
#     "Interaction rate",
#     "Conv. rate",
#     "Conversions"
# ]

# for col in numeric_like_cols:
#     if col in df.columns:
#         df[col] = clean_numeric_column(df[col])

# # Identify numeric and categorical columns
# num_cols = df.select_dtypes(include=np.number).columns.tolist()
# cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

# # =====================================
# # 4. SIDEBAR FILTERS (SESSION-MANAGED)
# # =====================================
# st.sidebar.header("🔍 Filters")

# # Initialize session_state for each categorical filter
# for col in cat_cols:
#     if f"filter_{col}" not in st.session_state:
#         st.session_state[f"filter_{col}"] = df[col].dropna().unique().tolist()

# # Create multiselects and update session_state
# for col in cat_cols:
#     selected = st.sidebar.multiselect(
#         f"Select {col}",
#         options=df[col].dropna().unique().tolist(),
#         default=st.session_state[f"filter_{col}"]
#     )

#     # Handle empty selection
#     if not selected:
#         st.warning(f"No {col} selected — showing all by default.")
#         selected = df[col].dropna().unique().tolist()

#     st.session_state[f"filter_{col}"] = selected

# # Apply all filters to dataframe
# filtered_df = df.copy()
# for col in cat_cols:
#     filtered_df = filtered_df[filtered_df[col].isin(st.session_state[f"filter_{col}"])]

# # =====================================
# # 5. NUMERICAL EDA FUNCTION
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
#     outlier_pct = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).mean() * 100

#     fig = px.histogram(
#         data,
#         nbins=30,
#         title=f"{graph_num}️⃣ Distribution of {col}",
#         labels={col: col, "count": "Frequency"}
#     )

#     return fig, {"mean": mean, "median": median, "skew": skew, "outlier_pct": outlier_pct}

# # =====================================
# # 6. CATEGORICAL EDA FUNCTION
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
# # 7. DISPLAY NUMERICAL GRAPHS
# # =====================================
# graph_count = 1
# st.header("📊 Numerical Feature Analysis")

# for col in num_cols:
#     fig, stats = numerical_eda_plotly(filtered_df, col, graph_count)
#     if fig:
#         col1, col2 = st.columns([3, 2])
#         col1.plotly_chart(fig, use_container_width=True)
#         col2.markdown(
#             f"""
# **Summary for {col}**
# - Average Value      : {stats['mean']:.2f}  
# - Typical Value      : {stats['median']:.2f}  
# - Distribution Skew  : {stats['skew']:.2f}  
# - Extreme Values (%) : {stats['outlier_pct']:.2f}%  

# **Recommended Action (Client Insight)**
# """
#         )
#         # Column-specific insights
#         if col.lower() in ["impr.", "impressions"]:
#             col2.markdown(
#                 "• Focus budget on campaigns/ad groups generating consistent reach  \n"
#                 "• Reduce spend where reach is high but interactions are low"
#             )
#         elif "interaction rate" in col.lower():
#             col2.markdown(
#                 "• Scale creatives and audiences with above-median interaction rate  \n"
#                 "• Refresh creatives in low-performing segments"
#             )
#         elif "interactions" in col.lower():
#             col2.markdown(
#                 "• Identify top interaction drivers and replicate messaging  \n"
#                 "• Pause ads with high impressions but low interactions"
#             )
#         elif "conv" in col.lower():
#             col2.markdown(
#                 "• Allocate more budget to high-conversion segments  \n"
#                 "• Investigate drop-offs in low-converting campaigns"
#             )
#         elif "cost" in col.lower():
#             col2.markdown(
#                 "• Shift spend from high-cost, low-return segments  \n"
#                 "• Optimize bidding strategies for cost-effective reach"
#             )
#         else:
#             col2.markdown(
#                 "• Prioritize segments above median performance  \n"
#                 "• Deprioritize consistently underperforming values"
#             )
#         graph_count += 1

# # =====================================
# # 8. DISPLAY CATEGORICAL GRAPHS
# # =====================================
# st.header("📊 Categorical Feature Analysis")

# for col in cat_cols:
#     fig, vc = categorical_eda_plotly(filtered_df, col, graph_count)
#     if fig:
#         col1, col2 = st.columns([3, 2])
#         col1.plotly_chart(fig, use_container_width=True)
#         summary_text = "\n".join([f"{idx}: {round(val,0)}" for idx, val in zip(vc.index, vc.values)])
#         col2.markdown(
#             f"""
# **Top Categories for {col}**
# {summary_text}

# **Recommended Action (Client Insight)**
# """
#         )
#         if vc.iloc[0] > 60:
#             col2.markdown(
#                 f"• Majority traffic comes from '{vc.index[0]}'  \n"
#                 "• Optimize messaging and offers for dominant segment"
#             )
#         elif len(vc) > 15:
#             col2.markdown(
#                 "• Performance spread across many categories  \n"
#                 "• Group similar segments to simplify targeting"
#             )
#         else:
#             col2.markdown(
#                 "• Distribution is well balanced  \n"
#                 "• Compare engagement/conversion across categories"
#             )
#         graph_count += 1

# # =====================================
# # 9. CORRELATION MATRIX
# # =====================================
# if len(num_cols) > 1:
#     st.header("📊 Correlation Analysis")
#     corr = filtered_df[num_cols].corr()
#     fig = px.imshow(
#         corr,
#         text_auto=True,
#         color_continuous_scale="RdBu_r",
#         title=f"{graph_count}️⃣ Correlation Matrix (Numerical Features)"
#     )
#     col1, col2 = st.columns([3, 2])
#     col1.plotly_chart(fig, use_container_width=True)
#     col2.markdown(
#         """
# **Correlation Summary**
# - Shows relationship between cost, impressions & engagement metrics

# **Recommended Action (Client Insight)**
# - Identify metrics most strongly linked to conversions
# - Prioritize optimization of these metrics (e.g., interaction rate)
# - Avoid increasing spend on metrics that do not improve conversions
# """
#     )
#     graph_count += 1
