import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from dotenv import load_dotenv
import os

# ───────────────── Page Config (MUST BE first) ─────────────────
st.set_page_config(
    page_title="Google Ads Search Terms Analysis",
    page_icon="🔎",
    layout="wide"
)

# ───────────────── Load ENV & Data Path ─────────────────
load_dotenv()
DATA_PATH = os.getenv("GA_SEARCH_TERMS")

st.title("🔎 Google Ads Search Terms Performance Dashboard")

# ───────────────── Load & Clean Data ─────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, engine="python", skiprows=2)

    # Remove summary rows
    df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")]
    df.reset_index(drop=True, inplace=True)

    numeric_cols = [
        "Impr.", "Clicks", "Interactions", "Interaction rate",
        "CTR", "Cost", "Conversions", "Conv. rate"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("%", "", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Cost / conv." in df.columns:
        df["Cost per Conversion"] = df["Cost / conv."]
    else:
        df["Cost per Conversion"] = df["Cost"] / df["Conversions"].replace(0, pd.NA)

    df["Query Length"] = df["Search term"].astype(str).apply(lambda x: len(x.split()))
    df["Query Bucket"] = df["Query Length"].apply(
        lambda x: "Short (1–2)" if x <= 2 else
                  "Medium (3–4)" if x <= 4 else
                  "Long (5+)"
    )

    return df

df = load_data(DATA_PATH)

# ───────────────── SIDEBAR FILTERS ─────────────────
with st.sidebar:
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.header("🎯 Filters")

    with col_f2:
        if st.button("❌ Clear Filters"):
            st.session_state.select_all_terms = True
            st.session_state.search_terms = []
            st.session_state.bucket_filter = list(df["Query Bucket"].unique())
            st.session_state.cost_range = (float(df["Cost"].min()), float(df["Cost"].max()))
            st.session_state.conv_range = (int(df["Conversions"].min()), int(df["Conversions"].max()))
            st.rerun()

    # Initialize session state if missing
    if "select_all_terms" not in st.session_state:
        st.session_state.select_all_terms = True
    if "search_terms" not in st.session_state:
        st.session_state.search_terms = []
    if "bucket_filter" not in st.session_state:
        st.session_state.bucket_filter = list(df["Query Bucket"].unique())
    if "cost_range" not in st.session_state:
        st.session_state.cost_range = (float(df["Cost"].min()), float(df["Cost"].max()))
    if "conv_range" not in st.session_state:
        st.session_state.conv_range = (int(df["Conversions"].min()), int(df["Conversions"].max()))

    # Select All toggle
    st.session_state.select_all_terms = st.checkbox(
        "Select all search terms",
        value=st.session_state.select_all_terms
    )

    # All terms list
    all_terms = sorted(df["Search term"].dropna().unique())

    # If select all → all terms are selected, else use session state
    if st.session_state.select_all_terms:
        default_terms = all_terms
    else:
        default_terms = st.session_state.search_terms

    # Multiselect (always visible)
    st.session_state.search_terms = st.multiselect(
        "Search Terms",
        options=all_terms,
        default=default_terms
    )

    # Query Bucket multiselect
    st.session_state.bucket_filter = st.multiselect(
        "Query Length Bucket",
        options=list(df["Query Bucket"].unique()),
        default=st.session_state.bucket_filter
    )

    # Cost Range slider
    st.session_state.cost_range = st.slider(
        "Cost Range (₹)",
        float(df["Cost"].min()),
        float(df["Cost"].max()),
        value=st.session_state.cost_range
    )

    # Conversions Range slider
    st.session_state.conv_range = st.slider(
        "Conversions Range",
        int(df["Conversions"].min()),
        int(df["Conversions"].max()),
        value=st.session_state.conv_range
    )

# ───────────────── APPLY FILTERS ─────────────────
filtered_df = df[
    (df["Search term"].isin(st.session_state.search_terms)) &
    (df["Query Bucket"].isin(st.session_state.bucket_filter)) &
    (df["Cost"].between(st.session_state.cost_range[0], st.session_state.cost_range[1])) &
    (df["Conversions"].between(st.session_state.conv_range[0], st.session_state.conv_range[1]))
]

if filtered_df.empty:
    st.warning("⚠ No data after applying filters.")
    st.stop()

# ───────────────── AGGREGATIONS ─────────────────
term_perf = (
    filtered_df
    .groupby("Search term", as_index=False)
    .agg(
        Impressions=("Impr.", "sum"),
        Cost=("Cost", "sum"),
        Conversions=("Conversions", "sum"),
        Cost_per_Conversion=("Cost per Conversion", "mean")
    )
)

# ───────────────── 1️⃣ Spend vs Conversions ─────────────────
st.subheader("1️⃣ Search Term Spend vs Conversions")
col1, col2 = st.columns([3, 2])
with col1:
    fig1 = px.scatter(
        term_perf,
        x="Cost",
        y="Conversions",
        size="Impressions",
        hover_name="Search term",
        color="Cost_per_Conversion",
        color_continuous_scale="Plasma",
        title="Spend vs Conversions"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    waste_terms = term_perf[
        (term_perf["Cost"] > term_perf["Cost"].median()) &
        (term_perf["Conversions"] == 0)
    ]
    st.markdown(
        f"""
**Summary**  
• **{len(waste_terms)}** terms spent money with zero conversions  

**Recommended Action**  
Add these as **negative keywords**
"""
    )

# ───────────────── 2️⃣ Query Length vs CPA ─────────────────
st.subheader("2️⃣ Query Length vs Conversion Efficiency")
length_perf = (
    filtered_df
    .groupby("Query Length", as_index=False)
    .agg(
        Avg_CPC=("Cost per Conversion", "mean"),
        Conversions=("Conversions", "sum")
    )
)
col1, col2 = st.columns([3, 2])
with col1:
    fig2 = px.line(
        length_perf,
        x="Query Length",
        y="Avg_CPC",
        markers=True,
        title="Query Length vs Avg CPA"
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    best_len = length_perf.loc[length_perf["Avg_CPC"].idxmin()]
    st.markdown(
        f"""
**Summary**  
• Best efficiency at **{int(best_len['Query Length'])} words**

**Recommended Action**  
Focus on longer, high-intent queries
"""
    )

# ───────────────── 3️⃣ Cost Share vs Conversion Share ─────────────────
st.subheader("3️⃣ Value Contribution Analysis")
share_df = term_perf.copy()
share_df["Cost Share (%)"] = share_df["Cost"] / share_df["Cost"].sum() * 100
share_df["Conversion Share (%)"] = share_df["Conversions"] / share_df["Conversions"].sum() * 100
col1, col2 = st.columns([3, 2])
with col1:
    fig3 = px.scatter(
        share_df,
        x="Cost Share (%)",
        y="Conversion Share (%)",
        hover_name="Search term",
        title="Cost Share vs Conversion Share"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    efficient = share_df[share_df["Conversion Share (%)"] > share_df["Cost Share (%)"]]
    st.markdown(
        f"""
**Summary**  
• **{len(efficient)}** terms outperform their spend

**Recommended Action**  
Increase bids on these terms
"""
    )

# ───────────────── 4️⃣ Waste Word Frequency ─────────────────
st.subheader("4️⃣ Wasteful Search Term Word Patterns")
waste_words = filtered_df[
    (filtered_df["Cost"] > filtered_df["Cost"].median()) &
    (filtered_df["Conversions"] == 0)
]["Search term"]
tokens = [t.lower() for t in waste_words.dropna()]
all_tokens = []
for t in tokens:
    all_tokens.extend(t.split())
word_df = pd.DataFrame(Counter(all_tokens).most_common(15), columns=["Word", "Frequency"])
col1, col2 = st.columns([3, 2])
with col1:
    fig4 = px.bar(word_df, x="Word", y="Frequency", title="Common Waste Words")
    st.plotly_chart(fig4, use_container_width=True)
with col2:
    st.markdown(
        """
**Summary**  
Repeated words appear in non-converting queries  

**Recommended Action**  
Add as phrase/broad negatives
"""
    )

# ───────────────── 5️⃣ Query Bucket Performance ─────────────────
st.subheader("5️⃣ Query Length Bucket Performance")
bucket_perf = (
    filtered_df
    .groupby("Query Bucket", as_index=False)
    .agg(
        Cost=("Cost", "sum"),
        Conversions=("Conversions", "sum"),
        Avg_CPC=("Cost per Conversion", "mean")
    )
)
col1, col2 = st.columns([3, 2])
with col1:
    fig5 = px.bar(
        bucket_perf,
        x="Query Bucket",
        y="Avg_CPC",
        color="Avg_CPC",
        color_continuous_scale="Turbo",
        title="Avg CPA by Query Bucket"
    )
    st.plotly_chart(fig5, use_container_width=True)

with col2:
    best_bucket = bucket_perf.loc[bucket_perf["Avg_CPC"].idxmin()]
    st.markdown(
        f"""
**Summary**  
• Best: **{best_bucket['Query Bucket']}**

**Recommended Action**  
Expand long-tail coverage
"""
    )


# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from collections import Counter
# from dotenv import load_dotenv
# import os

# # ───────────────── Page Config (MUST be first) ─────────────────
# st.set_page_config(
#     page_title="Google Ads Search Terms Analysis",
#     page_icon="🔎",
#     layout="wide"
# )

# # ───────────────── Load ENV & Data Path ─────────────────
# load_dotenv()
# DATA_PATH = os.getenv("GA_SEARCH_TERMS")

# st.title("🔎 Google Ads Search Terms Performance Dashboard")
# # st.caption(f"Data source: {DATA_PATH}")

# # ───────────────── Load & Clean Data ─────────────────
# @st.cache_data
# def load_data(path):
#     df = pd.read_csv(path, engine="python", skiprows=2)

#     # Remove summary rows
#     df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")]
#     df.reset_index(drop=True, inplace=True)

#     numeric_cols = [
#         "Impr.", "Clicks", "Interactions", "Interaction rate",
#         "CTR", "Cost", "Conversions", "Conv. rate"
#     ]

#     for col in numeric_cols:
#         if col in df.columns:
#             df[col] = (
#                 df[col]
#                 .astype(str)
#                 .str.replace(",", "", regex=False)
#                 .str.replace("%", "", regex=False)
#             )
#             df[col] = pd.to_numeric(df[col], errors="coerce")

#     if "Cost / conv." in df.columns:
#         df["Cost per Conversion"] = df["Cost / conv."]
#     else:
#         df["Cost per Conversion"] = df["Cost"] / df["Conversions"].replace(0, pd.NA)

#     df["Query Length"] = df["Search term"].astype(str).apply(lambda x: len(x.split()))

#     df["Query Bucket"] = df["Query Length"].apply(
#         lambda x: "Short (1–2)" if x <= 2 else
#                   "Medium (3–4)" if x <= 4 else
#                   "Long (5+)"
#     )

#     return df


# df = load_data(DATA_PATH)

# # ───────────────── Sidebar Filters (SAFE & FAST) ─────────────────

# # ────────────── SIDEBAR FILTERS ──────────────
# with st.sidebar:
#     col_f1, col_f2 = st.columns([1, 1])

#     with col_f1:
#         st.header("🎯 Filters")

#     with col_f2:
#         if st.button("❌ Clear Filters"):
#             # Reset all filters in session state
#             st.session_state.select_all_terms = True
#             st.session_state.search_terms = []
#             st.session_state.bucket_filter = list(df["Query Bucket"].unique())
#             st.session_state.cost_range = (
#                 float(df["Cost"].min()), float(df["Cost"].max())
#             )
#             st.session_state.conv_range = (
#                 int(df["Conversions"].min()), int(df["Conversions"].max())
#             )
#             st.rerun()

#     # ------------------ Select All Toggle ------------------
#     if "select_all_terms" not in st.session_state:
#         st.session_state.select_all_terms = True

#     select_all_terms = st.checkbox(
#         "Select all search terms",
#         value=st.session_state.select_all_terms,
#         key="select_all_terms"
#     )

#     # ------------------ Search Terms ------------------
#     all_terms = sorted(df["Search term"].dropna().unique())
#     if select_all_terms:
#         search_terms = all_terms
#         st.session_state.search_terms = all_terms
#     else:
#         if "search_terms" not in st.session_state:
#             st.session_state.search_terms = []
#         search_terms = st.multiselect(
#             "Search Terms",
#             options=all_terms,
#             default=st.session_state.search_terms,
#             key="search_terms"
#         )

#     # ------------------ Query Bucket ------------------
#     if "bucket_filter" not in st.session_state:
#         st.session_state.bucket_filter = list(df["Query Bucket"].unique())

#     bucket_filter = st.multiselect(
#         "Query Length Bucket",
#         options=df["Query Bucket"].unique(),
#         default=st.session_state.bucket_filter,
#         key="bucket_filter"
#     )

#     # ------------------ Cost Range ------------------
#     if "cost_range" not in st.session_state:
#         st.session_state.cost_range = (
#             float(df["Cost"].min()), float(df["Cost"].max())
#         )

#     cost_range = st.slider(
#         "Cost Range (₹)",
#         float(df["Cost"].min()),
#         float(df["Cost"].max()),
#         value=st.session_state.cost_range,
#         key="cost_range"
#     )

#     # ------------------ Conversions Range ------------------
#     if "conv_range" not in st.session_state:
#         st.session_state.conv_range = (
#             int(df["Conversions"].min()), int(df["Conversions"].max())
#         )

#     conv_range = st.slider(
#         "Conversions Range",
#         int(df["Conversions"].min()),
#         int(df["Conversions"].max()),
#         value=st.session_state.conv_range,
#         key="conv_range"
#     )

# # ────────────── APPLY FILTERS ──────────────
# filtered_df = df[
#     (df["Search term"].isin(search_terms)) &
#     (df["Query Bucket"].isin(bucket_filter)) &
#     (df["Cost"].between(cost_range[0], cost_range[1])) &
#     (df["Conversions"].between(conv_range[0], conv_range[1]))
# ]

# if filtered_df.empty:
#     st.warning("⚠ No data after applying filters.")
#     st.stop()

# # ────────────── AGGREGATIONS ──────────────
# term_perf = (
#     filtered_df
#     .groupby("Search term", as_index=False)
#     .agg(
#         Impressions=("Impr.", "sum"),
#         Cost=("Cost", "sum"),
#         Conversions=("Conversions", "sum"),
#         Cost_per_Conversion=("Cost per Conversion", "mean")
#     )
# )

# # ───────────────── 1️⃣ Spend vs Conversions ─────────────────
# st.subheader("1️⃣ Search Term Spend vs Conversions")

# col1, col2 = st.columns([3, 2])

# with col1:
#     fig1 = px.scatter(
#         term_perf,
#         x="Cost",
#         y="Conversions",
#         size="Impressions",
#         hover_name="Search term",
#         color="Cost_per_Conversion",
#         color_continuous_scale="Plasma",
#         title="Spend vs Conversions"
#     )
#     st.plotly_chart(fig1, use_container_width=True)

# with col2:
#     waste_terms = term_perf[
#         (term_perf["Cost"] > term_perf["Cost"].median()) &
#         (term_perf["Conversions"] == 0)
#     ]

#     st.markdown(
#         f"""
# **Summary**  
# • **{len(waste_terms)}** terms spent money with zero conversions  

# **Recommended Action**  
# Add these as **negative keywords**
# """
#     )

# # ───────────────── 2️⃣ Query Length vs CPA ─────────────────
# st.subheader("2️⃣ Query Length vs Conversion Efficiency")

# length_perf = (
#     filtered_df
#     .groupby("Query Length", as_index=False)
#     .agg(
#         Avg_CPC=("Cost per Conversion", "mean"),
#         Conversions=("Conversions", "sum")
#     )
# )

# col1, col2 = st.columns([3, 2])

# with col1:
#     fig3 = px.line(
#         length_perf,
#         x="Query Length",
#         y="Avg_CPC",
#         markers=True,
#         title="Query Length vs Avg CPA"
#     )
#     st.plotly_chart(fig3, use_container_width=True)

# with col2:
#     best_len = length_perf.loc[length_perf["Avg_CPC"].idxmin()]

#     st.markdown(
#         f"""
# **Summary**  
# • Best efficiency at **{int(best_len['Query Length'])} words**

# **Recommended Action**  
# Focus on longer, high-intent queries
# """
#     )

# # ───────────────── 3️⃣ Cost Share vs Conversion Share ─────────────────
# st.subheader("3️⃣ Value Contribution Analysis")

# share_df = term_perf.copy()
# share_df["Cost Share (%)"] = share_df["Cost"] / share_df["Cost"].sum() * 100
# share_df["Conversion Share (%)"] = (
#     share_df["Conversions"] / share_df["Conversions"].sum() * 100
# )

# col1, col2 = st.columns([3, 2])

# with col1:
#     fig4 = px.scatter(
#         share_df,
#         x="Cost Share (%)",
#         y="Conversion Share (%)",
#         hover_name="Search term",
#         title="Cost Share vs Conversion Share"
#     )
#     st.plotly_chart(fig4, use_container_width=True)

# with col2:
#     efficient = share_df[
#         share_df["Conversion Share (%)"] > share_df["Cost Share (%)"]
#     ]

#     st.markdown(
#         f"""
# **Summary**  
# • **{len(efficient)}** terms outperform their spend

# **Recommended Action**  
# Increase bids on these terms
# """
#     )

# # ───────────────── 4️⃣ Waste Word Frequency ─────────────────
# st.subheader("4️⃣ Wasteful Search Term Word Patterns")

# waste_words = filtered_df[
#     (filtered_df["Cost"] > filtered_df["Cost"].median()) &
#     (filtered_df["Conversions"] == 0)
# ]["Search term"]

# tokens = []
# for t in waste_words.dropna():
#     tokens.extend(t.lower().split())

# word_df = pd.DataFrame(
#     Counter(tokens).most_common(15),
#     columns=["Word", "Frequency"]
# )

# col1, col2 = st.columns([3, 2])

# with col1:
#     fig5 = px.bar(
#         word_df,
#         x="Word",
#         y="Frequency",
#         title="Common Waste Words"
#     )
#     st.plotly_chart(fig5, use_container_width=True)

# with col2:
#     st.markdown(
#         """
# **Summary**  
# Repeated words appear in non-converting queries  

# **Recommended Action**  
# Add as phrase/broad negatives
# """
#     )

# # ───────────────── 5️⃣ Query Bucket Performance ─────────────────
# st.subheader("5️⃣ Query Length Bucket Performance")

# bucket_perf = (
#     filtered_df
#     .groupby("Query Bucket", as_index=False)
#     .agg(
#         Cost=("Cost", "sum"),
#         Conversions=("Conversions", "sum"),
#         Avg_CPC=("Cost per Conversion", "mean")
#     )
# )

# col1, col2 = st.columns([3, 2])

# with col1:
#     fig6 = px.bar(
#         bucket_perf,
#         x="Query Bucket",
#         y="Avg_CPC",
#         color="Avg_CPC",
#         color_continuous_scale="Turbo",
#         title="Avg CPA by Query Bucket"
#     )
#     st.plotly_chart(fig6, use_container_width=True)

# with col2:
#     best_bucket = bucket_perf.loc[bucket_perf["Avg_CPC"].idxmin()]

#     st.markdown(
#         f"""
# **Summary**  
# • Best: **{best_bucket['Query Bucket']}**

# **Recommended Action**  
# Expand long-tail coverage
# """
#     )