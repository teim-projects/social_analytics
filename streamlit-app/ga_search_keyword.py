# ===================== PAGE CONFIG (MUST BE FIRST) =====================
import streamlit as st
st.set_page_config(
    page_title="Google Ads Search Keyword EDA",
    page_icon="🔍",
    layout="wide"
)

# ===================== IMPORTS =====================
import pandas as pd
import plotly.express as px
import os

# ===================== DATA LOADING =====================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Search_keyword_report.csv")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path, engine="python", skiprows=2)
    df = df.iloc[:-11].reset_index(drop=True)
    return df

df = load_data(DATA_PATH)

# ===================== DATA CLEANING =====================
numeric_cols = [
    "Impr.", "Clicks", "Interactions", "Interaction rate",
    "CTR", "Cost", "Conversions", "Conv. rate"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "")
            .str.replace("%", "")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Cost per Conversion
if "Cost / conv." in df.columns:
    df["Cost per Conversion"] = pd.to_numeric(
        df["Cost / conv."].astype(str).str.replace(",", ""),
        errors="coerce"
    )
else:
    df["Cost per Conversion"] = df["Cost"] / df["Conversions"]

# Keyword length
df["Keyword Length"] = df["Keyword"].astype(str).apply(lambda x: len(x.split()))

# Phone calls
if "Phone calls" in df.columns:
    df["Phone calls"] = pd.to_numeric(
        df["Phone calls"].astype(str).str.replace(",", ""),
        errors="coerce"
    )

# ===================== CONSTANTS =====================
ALL_KEYWORDS = sorted(df["Keyword"].dropna().unique())
ALL_MATCH_TYPES = (
    sorted(df["Match type"].dropna().unique())
    if "Match type" in df.columns else []
)

# ===================== SESSION STATE INIT =====================
if "keywords" not in st.session_state:
    st.session_state.keywords = []

if "select_all_keywords" not in st.session_state:
    st.session_state.select_all_keywords = False

if "match_types" not in st.session_state:
    st.session_state.match_types = []

if "cost_range" not in st.session_state:
    st.session_state.cost_range = (
        float(df["Cost"].min()),
        float(df["Cost"].max())
    )

# ===================== CALLBACKS (THE FIX) =====================
def on_select_all_keywords():
    if st.session_state.select_all_keywords:
        st.session_state.keywords = ALL_KEYWORDS.copy()
    else:
        st.session_state.keywords = []

def on_keywords_change():
    st.session_state.select_all_keywords = (
        set(st.session_state.keywords) == set(ALL_KEYWORDS)
    )

def clear_filters():
    st.session_state.keywords = []
    st.session_state.select_all_keywords = False
    st.session_state.match_types = []
    st.session_state.cost_range = (
        float(df["Cost"].min()),
        float(df["Cost"].max())
    )

# ===================== SIDEBAR =====================
with st.sidebar:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("🎯 Filters")

    with col2:
        st.button("❌ Clear", on_click=clear_filters)

    # ✅ Select all toggle
    st.checkbox(
        "Select all keywords",
        key="select_all_keywords",
        on_change=on_select_all_keywords
    )

    # ✅ Keyword multiselect (NO DOUBLE CLICK)
    st.multiselect(
        "Select Keywords",
        options=ALL_KEYWORDS,
        key="keywords",
        on_change=on_keywords_change
    )

    # Match type filter
    if ALL_MATCH_TYPES:
        st.multiselect(
            "Select Match Types",
            options=ALL_MATCH_TYPES,
            key="match_types"
        )

    # Cost filter
    st.slider(
        "Cost Range (₹)",
        float(df["Cost"].min()),
        float(df["Cost"].max()),
        key="cost_range"
    )

# ===================== TITLE (ALWAYS SHOWN) =====================
st.title("🔍 Google Ads – Search Keyword Performance Dashboard")

# ===================== APPLY FILTERS =====================
filtered_df = df.copy()

if st.session_state.keywords:
    filtered_df = filtered_df[
        filtered_df["Keyword"].isin(st.session_state.keywords)
    ]

if st.session_state.match_types and "Match type" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["Match type"].isin(st.session_state.match_types)
    ]

filtered_df = filtered_df[
    filtered_df["Cost"].between(*st.session_state.cost_range)
]

# ===================== EMPTY STATE =====================
if not st.session_state.keywords:
    st.info("👈 Please select one or more keywords to view analysis.")
    st.stop()

# ===================== 1️⃣ KEYWORD COST EFFICIENCY =====================
kw_perf = (
    filtered_df
    .groupby("Keyword", as_index=False)
    .agg(
        Impressions=("Impr.", "sum"),
        Cost=("Cost", "sum"),
        Conversions=("Conversions", "sum"),
        Cost_per_Conversion=("Cost per Conversion", "mean")
    )
    .sort_values("Cost_per_Conversion")
)

col1, col2 = st.columns([3, 2])

with col1:
    fig = px.bar(
        kw_perf.head(20),
        x="Cost_per_Conversion",
        y="Keyword",
        orientation="h",
        title="1️⃣ Top Keywords by Cost Efficiency",
        color="Cost_per_Conversion",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    best_kw = kw_perf.iloc[0]
    worst_kw = kw_perf.iloc[-1]

    st.subheader("📌 Insight")
    st.write(
        f"**Most Efficient:** `{best_kw['Keyword']}` "
        f"(₹{best_kw['Cost_per_Conversion']:.2f})"
    )
    st.write(f"**Least Efficient:** `{worst_kw['Keyword']}`")
    st.success("Increase bids on efficient keywords and reduce waste.")

# ===================== 2️⃣ WASTE KEYWORDS =====================
waste_kw = kw_perf[
    (kw_perf["Cost"] > kw_perf["Cost"].median()) &
    (kw_perf["Conversions"] == 0)
]

fig = px.bar(
    waste_kw.sort_values("Cost", ascending=False),
    x="Cost",
    y="Keyword",
    orientation="h",
    title="2️⃣ Waste Keywords (High Spend, Zero Conversions)",
    color="Cost",
    color_continuous_scale="Reds"
)
st.plotly_chart(fig, use_container_width=True)

# ===================== 3️⃣ KEYWORD LENGTH VS CPC =====================
kw_length_perf = (
    filtered_df
    .groupby("Keyword Length", as_index=False)
    .agg(Avg_CPC=("Cost per Conversion", "mean"))
)

fig = px.line(
    kw_length_perf,
    x="Keyword Length",
    y="Avg_CPC",
    markers=True,
    title="3️⃣ Keyword Length vs Cost per Conversion"
)
st.plotly_chart(fig, use_container_width=True)

# ===================== 4️⃣ PARETO ANALYSIS =====================
pareto_kw = kw_perf.sort_values("Conversions", ascending=False)
pareto_kw["Cumulative Conversions (%)"] = (
    pareto_kw["Conversions"].cumsum() /
    pareto_kw["Conversions"].sum() * 100
)

fig = px.line(
    pareto_kw,
    x="Keyword",
    y="Cumulative Conversions (%)",
    title="4️⃣ Pareto Analysis – Keywords Driving 80% Conversions"
)
st.plotly_chart(fig, use_container_width=True)

# ===================== 5️⃣ PHONE CALL KEYWORDS =====================
if "Phone calls" in filtered_df.columns:
    call_kw = filtered_df[filtered_df["Phone calls"] > 0]

    call_perf = (
        call_kw
        .groupby("Keyword", as_index=False)
        .agg(
            Phone_Calls=("Phone calls", "sum"),
            Cost=("Cost", "sum")
        )
        .sort_values("Phone_Calls", ascending=False)
    )

    fig = px.bar(
        call_perf.head(15),
        x="Phone_Calls",
        y="Keyword",
        orientation="h",
        title="5️⃣ Keywords Driving Phone Calls",
        color="Phone_Calls",
        color_continuous_scale="Greens"
    )
    st.plotly_chart(fig, use_container_width=True)

# ===================== 6️⃣ FINAL URL COVERAGE =====================
url_coverage = filtered_df["Final URL"].notna().value_counts().reset_index()
url_coverage.columns = ["Has Final URL", "Count"]

fig = px.pie(
    url_coverage,
    names="Has Final URL",
    values="Count",
    title="6️⃣ Final URL Coverage"
)
st.plotly_chart(fig, use_container_width=True)


# # ===================== PAGE CONFIG (MUST BE FIRST) =====================
# import streamlit as st
# st.set_page_config(
#     page_title="Google Ads Search Keyword EDA",
#     page_icon="🔍",
#     layout="wide"
# )

# # ===================== IMPORTS =====================
# import pandas as pd
# import plotly.express as px
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # ===================== DATA LOADING =====================
# DATA_PATH = os.getenv("GA_SEARCH_KEYWORD_PATH")

# @st.cache_data
# def load_data(path):
#     df = pd.read_csv(path, engine="python", skiprows=2)
#     df = df.iloc[:-11].reset_index(drop=True)
#     return df

# df = load_data(DATA_PATH)

# # ===================== DATA CLEANING =====================
# numeric_cols = [
#     "Impr.", "Clicks", "Interactions", "Interaction rate",
#     "CTR", "Cost", "Conversions", "Conv. rate"
# ]

# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = (
#             df[col].astype(str)
#             .str.replace(",", "")
#             .str.replace("%", "")
#         )
#         df[col] = pd.to_numeric(df[col], errors="coerce")

# # Cost per Conversion
# if "Cost / conv." in df.columns:
#     df["Cost per Conversion"] = pd.to_numeric(
#         df["Cost / conv."].astype(str).str.replace(",", ""),
#         errors="coerce"
#     )
# else:
#     df["Cost per Conversion"] = df["Cost"] / df["Conversions"]

# # Keyword length
# df["Keyword Length"] = df["Keyword"].astype(str).apply(lambda x: len(x.split()))

# # Phone calls
# if "Phone calls" in df.columns:
#     df["Phone calls"] = pd.to_numeric(
#         df["Phone calls"].astype(str).str.replace(",", ""),
#         errors="coerce"
#     )

# # ===================== SESSION STATE INIT =====================
# ALL_KEYWORDS = sorted(df["Keyword"].dropna().unique())

# if "keywords" not in st.session_state:
#     st.session_state.keywords = []

# if "select_all_keywords" not in st.session_state:
#     st.session_state.select_all_keywords = False

# if "match_types" not in st.session_state:
#     st.session_state.match_types = []

# if "cost_range" not in st.session_state:
#     st.session_state.cost_range = (
#         float(df["Cost"].min()),
#         float(df["Cost"].max())
#     )

# # ===================== SIDEBAR FILTERS =====================
# with st.sidebar:
#     col1, col2 = st.columns([1, 1])

#     with col1:
#         st.header("🎯 Filters")

#     with col2:
#         if st.button("❌ Clear"):
#             st.session_state.select_all_keywords = False
#             st.session_state.keywords = []
#             st.session_state.match_types = []
#             st.session_state.cost_range = (
#                 float(df["Cost"].min()),
#                 float(df["Cost"].max())
#             )
#             st.rerun()

#     # ---- Select all toggle ----
#     select_all = st.checkbox(
#         "Select all keywords",
#         value=st.session_state.select_all_keywords
#     )

#     # ---- Keyword selector ----
#     selected_keywords = st.multiselect(
#         "Select Keywords",
#         options=ALL_KEYWORDS,
#         default=ALL_KEYWORDS if select_all else st.session_state.keywords
#     )

#     # ---- Sync checkbox state safely ----
#     if select_all:
#         st.session_state.keywords = ALL_KEYWORDS
#         st.session_state.select_all_keywords = True
#     else:
#         st.session_state.keywords = selected_keywords
#         st.session_state.select_all_keywords = False

#     # ---- Match type ----
#     if "Match type" in df.columns:
#         st.session_state.match_types = st.multiselect(
#             "Select Match Types",
#             options=sorted(df["Match type"].dropna().unique()),
#             default=st.session_state.match_types
#         )

#     # ---- Cost ----
#     st.session_state.cost_range = st.slider(
#         "Cost Range (₹)",
#         float(df["Cost"].min()),
#         float(df["Cost"].max()),
#         st.session_state.cost_range
#     )

# # ===================== APPLY FILTERS =====================
# filtered_df = df.copy()

# if st.session_state.keywords:
#     filtered_df = filtered_df[
#         filtered_df["Keyword"].isin(st.session_state.keywords)
#     ]

# if st.session_state.match_types and "Match type" in filtered_df.columns:
#     filtered_df = filtered_df[
#         filtered_df["Match type"].isin(st.session_state.match_types)
#     ]

# filtered_df = filtered_df[
#     filtered_df["Cost"].between(*st.session_state.cost_range)
# ]

# # ===================== TITLE =====================
# st.title("🔍 Google Ads – Search Keyword Performance Dashboard")

# if not st.session_state.keywords:
#     st.info("👈 Select one or more keywords from the sidebar to see insights.")

# # ===================== 1️⃣ KEYWORD COST EFFICIENCY =====================
# if not filtered_df.empty:
#     kw_perf = (
#         filtered_df
#         .groupby("Keyword", as_index=False)
#         .agg(
#             Impressions=("Impr.", "sum"),
#             Cost=("Cost", "sum"),
#             Conversions=("Conversions", "sum"),
#             Cost_per_Conversion=("Cost per Conversion", "mean")
#         )
#         .sort_values("Cost_per_Conversion")
#     )

#     col1, col2 = st.columns([3, 2])

#     with col1:
#         fig = px.bar(
#             kw_perf.head(20),
#             x="Cost_per_Conversion",
#             y="Keyword",
#             orientation="h",
#             title="1️⃣ Top Keywords by Cost Efficiency",
#             color="Cost_per_Conversion",
#             color_continuous_scale="Viridis"
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     with col2:
#         best_kw = kw_perf.iloc[0]
#         worst_kw = kw_perf.iloc[-1]

#         st.subheader("📌 Insight")
#         st.write(
#             f"**Most Efficient:** `{best_kw['Keyword']}` "
#             f"(₹{best_kw['Cost_per_Conversion']:.2f})"
#         )
#         st.write(f"**Least Efficient:** `{worst_kw['Keyword']}`")
#         st.success("Increase bids on efficient keywords and reduce waste.")

# # ===================== 2️⃣ WASTE KEYWORDS =====================
#     waste_kw = kw_perf[
#         (kw_perf["Cost"] > kw_perf["Cost"].median()) &
#         (kw_perf["Conversions"] == 0)
#     ]

#     fig = px.bar(
#         waste_kw.sort_values("Cost", ascending=False),
#         x="Cost",
#         y="Keyword",
#         orientation="h",
#         title="2️⃣ Waste Keywords (High Spend, Zero Conversions)",
#         color="Cost",
#         color_continuous_scale="Reds"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # ===================== 3️⃣ KEYWORD LENGTH VS CPC =====================
#     kw_length_perf = (
#         filtered_df
#         .groupby("Keyword Length", as_index=False)
#         .agg(
#             Avg_CPC=("Cost per Conversion", "mean")
#         )
#     )

#     fig = px.line(
#         kw_length_perf,
#         x="Keyword Length",
#         y="Avg_CPC",
#         markers=True,
#         title="3️⃣ Keyword Length vs Cost per Conversion"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # ===================== 4️⃣ PARETO ANALYSIS =====================
#     pareto_kw = kw_perf.sort_values("Conversions", ascending=False)
#     pareto_kw["Cumulative Conversions (%)"] = (
#         pareto_kw["Conversions"].cumsum() /
#         pareto_kw["Conversions"].sum() * 100
#     )

#     fig = px.line(
#         pareto_kw,
#         x="Keyword",
#         y="Cumulative Conversions (%)",
#         title="4️⃣ Pareto Analysis – Keywords Driving 80% Conversions"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# # ===================== 5️⃣ PHONE CALL KEYWORDS =====================
#     if "Phone calls" in filtered_df.columns:
#         call_kw = filtered_df[filtered_df["Phone calls"] > 0]

#         call_perf = (
#             call_kw
#             .groupby("Keyword", as_index=False)
#             .agg(
#                 Phone_Calls=("Phone calls", "sum"),
#                 Cost=("Cost", "sum")
#             )
#             .sort_values("Phone_Calls", ascending=False)
#         )

#         fig = px.bar(
#             call_perf.head(15),
#             x="Phone_Calls",
#             y="Keyword",
#             orientation="h",
#             title="5️⃣ Keywords Driving Phone Calls",
#             color="Phone_Calls",
#             color_continuous_scale="Greens"
#         )
#         st.plotly_chart(fig, use_container_width=True)

# # ===================== 6️⃣ FINAL URL COVERAGE =====================
#     url_coverage = filtered_df["Final URL"].notna().value_counts().reset_index()
#     url_coverage.columns = ["Has Final URL", "Count"]

#     fig = px.pie(
#         url_coverage,
#         names="Has Final URL",
#         values="Count",
#         title="6️⃣ Final URL Coverage"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------------------------------

# # ===================== PAGE CONFIG (MUST BE FIRST) =====================
# import streamlit as st
# st.set_page_config(
#     page_title="Google Ads Search Keyword EDA",
#     page_icon="🔍",
#     layout="wide"
# )

# # ===================== IMPORTS =====================
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # ===================== DATA LOADING =====================
# DATA_PATH = os.getenv("GA_SEARCH_KEYWORD_PATH")

# @st.cache_data
# def load_data(path):
#     df = pd.read_csv(path, engine="python", skiprows=2)
#     df = df.iloc[:-11].reset_index(drop=True)
#     return df

# df = load_data(DATA_PATH)

# # ===================== DATA CLEANING =====================
# numeric_cols = [
#     "Impr.", "Clicks", "Interactions", "Interaction rate",
#     "CTR", "Cost", "Conversions", "Conv. rate"
# ]

# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = (
#             df[col].astype(str)
#             .str.replace(",", "")
#             .str.replace("%", "")
#         )
#         df[col] = pd.to_numeric(df[col], errors="coerce")

# # Cost per Conversion
# if "Cost / conv." in df.columns:
#     df["Cost per Conversion"] = pd.to_numeric(
#         df["Cost / conv."].astype(str).str.replace(",", ""),
#         errors="coerce"
#     )
# else:
#     df["Cost per Conversion"] = df["Cost"] / df["Conversions"]

# # Keyword length
# df["Keyword Length"] = df["Keyword"].astype(str).apply(lambda x: len(x.split()))

# # Phone calls
# if "Phone calls" in df.columns:
#     df["Phone calls"] = pd.to_numeric(
#         df["Phone calls"].astype(str).str.replace(",", ""),
#         errors="coerce"
#     )

# # ===================== SESSION STATE =====================
# if "filtered_df" not in st.session_state:
#     st.session_state.filtered_df = df.copy()

# # ===================== SIDEBAR FILTERS =====================

# # ───────────────── Sidebar Filters ─────────────────
# with st.sidebar:
#     col_f1, col_f2 = st.columns([1, 1])

#     with col_f1:
#         st.header("🎯 Filters")

#     with col_f2:
#         if st.button("❌ Clear Filters"):
#             st.session_state.keywords = df["Keyword"].dropna().unique().tolist()

#             if "Match type" in df.columns:
#                 st.session_state.match_types = df["Match type"].dropna().unique().tolist()

#             st.session_state.cost_range = (
#                 float(df["Cost"].min()),
#                 float(df["Cost"].max())
#             )
#             st.rerun()

# # ───────────────── Initialize Session State ─────────────────
# if "keywords" not in st.session_state:
#     st.session_state.keywords = df["Keyword"].dropna().unique().tolist()

# if "match_types" not in st.session_state:
#     if "Match type" in df.columns:
#         st.session_state.match_types = df["Match type"].dropna().unique().tolist()
#     else:
#         st.session_state.match_types = []

# if "cost_range" not in st.session_state:
#     st.session_state.cost_range = (
#         float(df["Cost"].min()),
#         float(df["Cost"].max())
#     )

# # ───────────────── Filter Widgets (KEY-BASED) ─────────────────
# st.sidebar.multiselect(
#     "Select Keywords",
#     options=sorted(df["Keyword"].dropna().unique()),
#     key="keywords"
# )

# if "Match type" in df.columns:
#     st.sidebar.multiselect(
#         "Select Match Types",
#         options=sorted(df["Match type"].dropna().unique()),
#         key="match_types"
#     )

# st.sidebar.slider(
#     "Cost Range (₹)",
#     float(df["Cost"].min()),
#     float(df["Cost"].max()),
#     key="cost_range"
# )

# # ───────────────── Apply Filters ─────────────────
# filtered_df = df.copy()

# if st.session_state.keywords:
#     filtered_df = filtered_df[
#         filtered_df["Keyword"].isin(st.session_state.keywords)
#     ]

# if st.session_state.match_types and "Match type" in filtered_df.columns:
#     filtered_df = filtered_df[
#         filtered_df["Match type"].isin(st.session_state.match_types)
#     ]

# filtered_df = filtered_df[
#     filtered_df["Cost"].between(*st.session_state.cost_range)
# ]

# st.session_state.filtered_df = filtered_df

# # ===================== TITLE =====================
# st.title("🔍 Google Ads – Search Keyword Performance Dashboard")

# # ===================== 1️⃣ KEYWORD COST EFFICIENCY =====================
# kw_perf = (
#     filtered_df
#     .groupby("Keyword", as_index=False)
#     .agg(
#         Impressions=("Impr.", "sum"),
#         Cost=("Cost", "sum"),
#         Conversions=("Conversions", "sum"),
#         Cost_per_Conversion=("Cost per Conversion", "mean")
#     )
#     .sort_values("Cost_per_Conversion")
# )

# best_kw = kw_perf.iloc[0]
# worst_kw = kw_perf.iloc[-1]

# col1, col2 = st.columns([3, 2])

# with col1:
#     fig = px.bar(
#         kw_perf.head(20),
#         x="Cost_per_Conversion",
#         y="Keyword",
#         orientation="h",
#         title="1️⃣ Top Keywords by Cost Efficiency",
#         color="Cost_per_Conversion",
#         color_continuous_scale="Viridis"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with col2:
#     st.subheader("📌 Insight")
#     st.write(
#         f"**Most Efficient Keyword:** `{best_kw['Keyword']}` "
#         f"(₹{best_kw['Cost_per_Conversion']:.2f} per conversion)"
#     )
#     st.write(
#         f"**Least Efficient Keyword:** `{worst_kw['Keyword']}`"
#     )
#     st.success("Increase bids on efficient keywords and optimize or pause inefficient ones.")

# # ===================== 2️⃣ WASTE KEYWORDS =====================
# waste_kw = kw_perf[
#     (kw_perf["Cost"] > kw_perf["Cost"].median()) &
#     (kw_perf["Conversions"] == 0)
# ]

# col1, col2 = st.columns([3, 2])

# with col1:
#     fig = px.bar(
#         waste_kw.sort_values("Cost", ascending=False),
#         x="Cost",
#         y="Keyword",
#         orientation="h",
#         title="2️⃣ Waste Keywords (High Spend, Zero Conversions)",
#         color="Cost",
#         color_continuous_scale="Reds"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with col2:
#     st.subheader("📌 Insight")
#     st.write(f"{len(waste_kw)} keywords spent heavily without conversions.")
#     st.warning("Pause these keywords or review match types & intent alignment.")

# # ===================== 3️⃣ KEYWORD LENGTH VS CPC =====================
# kw_length_perf = (
#     filtered_df
#     .groupby("Keyword Length", as_index=False)
#     .agg(
#         Cost=("Cost", "sum"),
#         Conversions=("Conversions", "sum"),
#         Avg_CPC=("Cost per Conversion", "mean")
#     )
# )

# best_len = kw_length_perf.loc[kw_length_perf["Avg_CPC"].idxmin()]

# col1, col2 = st.columns([3, 2])

# with col1:
#     fig = px.line(
#         kw_length_perf,
#         x="Keyword Length",
#         y="Avg_CPC",
#         markers=True,
#         title="3️⃣ Keyword Length vs Cost per Conversion"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with col2:
#     st.subheader("📌 Insight")
#     st.write(
#         f"Best efficiency observed for **{int(best_len['Keyword Length'])}-word keywords**."
#     )
#     st.success("Expand long-tail keywords and reduce overly generic terms.")

# # ===================== 4️⃣ PARETO (80/20) ANALYSIS =====================
# pareto_kw = kw_perf.sort_values("Conversions", ascending=False)
# pareto_kw["Cumulative Conversions (%)"] = (
#     pareto_kw["Conversions"].cumsum() /
#     pareto_kw["Conversions"].sum() * 100
# )

# top_kw = pareto_kw[pareto_kw["Cumulative Conversions (%)"] <= 80]

# col1, col2 = st.columns([3, 2])

# with col1:
#     fig = px.line(
#         pareto_kw,
#         x="Keyword",
#         y="Cumulative Conversions (%)",
#         title="4️⃣ Pareto Analysis – Keywords Driving 80% Conversions"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with col2:
#     st.subheader("📌 Insight")
#     st.write(f"{len(top_kw)} keywords drive ~80% of conversions.")
#     st.success("Aggressively protect and optimize these high-impact keywords.")

# # ===================== 5️⃣ PHONE CALL KEYWORDS =====================
# if "Phone calls" in filtered_df.columns:
#     call_kw = filtered_df[filtered_df["Phone calls"] > 0]

#     call_perf = (
#         call_kw
#         .groupby("Keyword", as_index=False)
#         .agg(
#             Phone_Calls=("Phone calls", "sum"),
#             Cost=("Cost", "sum")
#         )
#         .sort_values("Phone_Calls", ascending=False)
#     )

#     col1, col2 = st.columns([3, 2])

#     with col1:
#         fig = px.bar(
#             call_perf.head(15),
#             x="Phone_Calls",
#             y="Keyword",
#             orientation="h",
#             title="5️⃣ Keywords Driving Phone Calls",
#             color="Phone_Calls",
#             color_continuous_scale="Greens"
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     with col2:
#         st.subheader("📌 Insight")
#         st.write("Some keywords primarily drive phone calls.")
#         st.success("Ensure call tracking & call extensions are optimized.")

# # ===================== 6️⃣ FINAL URL COVERAGE =====================
# url_coverage = filtered_df["Final URL"].notna().value_counts().reset_index()
# url_coverage.columns = ["Has Final URL", "Count"]

# col1, col2 = st.columns([3, 2])

# with col1:
#     fig = px.pie(
#         url_coverage,
#         names="Has Final URL",
#         values="Count",
#         title="6️⃣ Final URL Coverage"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with col2:
#     st.subheader("📌 Insight")
#     st.warning(
#         "Many keywords lack a dedicated landing page."
#     )
#     st.success(
#         "Assign relevant landing pages to high-intent keywords."
#     )
