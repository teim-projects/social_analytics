import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ───────────────── Page Setup ─────────────────
st.set_page_config(
    page_title="Google Ads Age Performance Dashboard",
    page_icon="📊",
    layout="wide"
)
st.title("📊 Google Ads Age Performance EDA Dashboard")

# ───────────────── Load Data ─────────────────
@st.cache_data
def load_data(path):
    df = pd.read_csv(path, engine="python", skiprows=2)
    df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")].reset_index(drop=True)
    return df

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Age_report.csv")
age_df = load_data(DATA_PATH)

# ───────────────── Data Cleaning ─────────────────
numeric_cols = [
    "Impr.", "Clicks", "Interactions",
    "Interaction rate", "CTR",
    "Cost", "Conversions", "Conv. rate"
]

for col in numeric_cols:
    if col in age_df.columns:
        age_df[col] = (
            age_df[col].astype(str)
            .str.replace(",", "")
            .str.replace("%", "")
        )
        age_df[col] = pd.to_numeric(age_df[col], errors="coerce")

if "Cost / conv." in age_df.columns:
    age_df["Cost per Conversion"] = age_df["Cost / conv."]
else:
    age_df["Cost per Conversion"] = age_df["Cost"] / age_df["Conversions"]

# ───────────────── Sidebar Filters ─────────────────
with st.sidebar:
    col_f1, col_f2 = st.sidebar.columns([1, 1])

    with col_f1:
        st.header("🎯 Filters")

    with col_f2:
        if st.button("❌ Clear Filters"):
            st.session_state.age_filter = age_df["Age"].unique().tolist()
            st.rerun()

    if "age_filter" not in st.session_state:
        st.session_state.age_filter = age_df["Age"].unique().tolist()

    selected_ages = st.multiselect(
        "Age Groups",
        options=age_df["Age"].unique(),
        default=st.session_state.age_filter
    )

    st.session_state.age_filter = selected_ages
    df_filtered = age_df[age_df["Age"].isin(selected_ages)]

# ───────────────── Aggregations ─────────────────
age_perf = (
    df_filtered.groupby("Age", as_index=False)
    .agg(
        Impressions=("Impr.", "sum"),
        Cost=("Cost", "sum"),
        Conversions=("Conversions", "sum"),
        CPC=("Cost per Conversion", "mean")
    )
    .sort_values("CPC")
)

# ───────────────── 1️⃣ Cost Efficiency by Age ─────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("1️⃣ Cost Efficiency by Age Group")
    fig = px.bar(
        age_perf,
        x="Age",
        y="CPC",
        title="Cost per Conversion by Age Group"
    )
    fig.update_layout(xaxis_title="Age Group", yaxis_title="Cost per Conversion (₹)")
    st.plotly_chart(fig, use_container_width=True)

with right:
    best = age_perf.iloc[0]
    worst = age_perf.iloc[-1]

    st.markdown("### 📌 Summary")
    st.write(
        f"**{best['Age']}** is the most cost-efficient "
        f"(₹{best['CPC']:.2f}), while **{worst['Age']}** is the least efficient."
    )
    st.markdown("### ✅ Recommended Action")
    st.write("Increase budget for efficient age groups and reduce spend on poor performers.")

# ───────────────── 2️⃣ Conversion Share ─────────────────
age_share = age_perf.copy()
age_share["Conversion Share (%)"] = (
    age_share["Conversions"] / age_share["Conversions"].sum() * 100
)

left, right = st.columns([3, 2])

with left:
    st.subheader("2️⃣ Conversion Contribution by Age")
    fig = px.bar(
        age_share,
        x="Age",
        y="Conversion Share (%)",
        title="Conversion Share by Age Group"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    top_age = age_share.loc[age_share["Conversion Share (%)"].idxmax()]
    st.markdown("### 📌 Summary")
    st.write(f"**{top_age['Age']}** contributes the highest share of conversions.")
    st.markdown("### ✅ Recommended Action")
    st.write("Ensure sufficient budget allocation for high-contributing age groups.")

# ───────────────── 3️⃣ Wasted Spend Analysis ─────────────────
waste_ages = age_perf[
    (age_perf["Cost"] > age_perf["Cost"].median()) &
    (age_perf["Conversions"] < age_perf["Conversions"].median())
]

left, right = st.columns([3, 2])

with left:
    st.subheader("3️⃣ High Spend but Low Conversion Age Groups")
    fig = px.scatter(
        age_perf,
        x="Cost",
        y="Conversions",
        size="Impressions",
        color="Age",
        title="Spend vs Conversions"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### 📌 Summary")
    st.write(f"{len(waste_ages)} age group(s) incur high spend with low returns.")
    st.markdown("### ✅ Recommended Action")
    st.write("Exclude or reduce bids for these age groups.")

# ───────────────── 4️⃣ Stability Analysis ─────────────────
age_consistency = (
    df_filtered.groupby("Age", as_index=False)
    .agg(
        Avg_CPC=("Cost per Conversion", "mean"),
        Std_CPC=("Cost per Conversion", "std")
    )
)

stable = age_consistency.loc[age_consistency["Std_CPC"].idxmin()]
risky = age_consistency.loc[age_consistency["Std_CPC"].idxmax()]

left, right = st.columns([3, 2])

with left:
    st.subheader("4️⃣ Cost Stability by Age")
    fig = px.bar(
        age_consistency,
        x="Age",
        y="Std_CPC",
        title="Volatility in Cost per Conversion"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### 📌 Summary")
    st.write(
        f"**{stable['Age']}** is the most stable, while "
        f"**{risky['Age']}** is the most volatile."
    )
    st.markdown("### ✅ Recommended Action")
    st.write("Allocate stable budgets to consistent age groups and test volatile ones carefully.")

# ───────────────── 5️⃣ Optimization Impact ─────────────────
threshold = age_perf["CPC"].quantile(0.30)
optimized = age_perf[age_perf["CPC"] <= threshold]

left, right = st.columns([3, 2])

with left:
    st.subheader("5️⃣ Optimization Opportunity")
    fig = px.bar(
        optimized,
        x="Age",
        y="CPC",
        title="Top 30% Most Efficient Age Groups"
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("### 📌 Summary")
    st.write(
        f"Focusing on efficient age groups reduces CPC from "
        f"₹{age_perf['CPC'].mean():.2f} to ₹{optimized['CPC'].mean():.2f}."
    )
    st.markdown("### ✅ Recommended Action")
    st.write("Concentrate spend on high-efficiency age segments to maximize ROI.")


# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import os
# from dotenv import load_dotenv

# # ───────────────── Page Setup ─────────────────
# st.set_page_config(
#     page_title="Google Ads Age Performance Dashboard",
#     page_icon="📊",
#     layout="wide"
# )
# st.title("📊 Google Ads Age Performance EDA Dashboard")

# load_dotenv()

# DATA_PATH = os.getenv("GA_AGE_REPORT_PATH")

# # ───────────────── Load Data ─────────────────
# @st.cache_data
# def load_data(path):
#     df = pd.read_csv(path, engine="python", skiprows=2)
#     df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")].reset_index(drop=True)
#     return df

# age_df = load_data(DATA_PATH)

# # ───────────────── Data Cleaning ─────────────────
# numeric_cols = [
#     "Impr.", "Clicks", "Interactions",
#     "Interaction rate", "CTR",
#     "Cost", "Conversions", "Conv. rate"
# ]

# for col in numeric_cols:
#     if col in age_df.columns:
#         age_df[col] = (
#             age_df[col].astype(str)
#             .str.replace(",", "")
#             .str.replace("%", "")
#         )
#         age_df[col] = pd.to_numeric(age_df[col], errors="coerce")

# if "Cost / conv." in age_df.columns:
#     age_df["Cost per Conversion"] = age_df["Cost / conv."]
# else:
#     age_df["Cost per Conversion"] = age_df["Cost"] / age_df["Conversions"]

# # ───────────────── Session State ─────────────────
# if "age_filter" not in st.session_state:
#     st.session_state.age_filter = age_df["Age"].unique().tolist()

# # ───────────────── Filters ─────────────────
# col1, col2 = st.columns([4, 1])

# with col1:
#     st.subheader("🎯 Filters")
#     selected_ages = st.multiselect(
#         "Select Age Groups",
#         options=age_df["Age"].unique(),
#         default=st.session_state.age_filter
#     )

# with col2:
#     st.write("")
#     st.write("")
#     if st.button("❌ Clear Filters"):
#         st.session_state.age_filter = age_df["Age"].unique().tolist()
#         st.rerun()

# st.session_state.age_filter = selected_ages
# df_filtered = age_df[age_df["Age"].isin(selected_ages)]

# # ───────────────── Aggregations ─────────────────
# age_perf = (
#     df_filtered.groupby("Age", as_index=False)
#     .agg(
#         Impressions=("Impr.", "sum"),
#         Cost=("Cost", "sum"),
#         Conversions=("Conversions", "sum"),
#         CPC=("Cost per Conversion", "mean")
#     )
#     .sort_values("CPC")
# )

# # ───────────────── 1️⃣ Cost Efficiency by Age ─────────────────
# left, right = st.columns([3, 2])

# with left:
#     st.subheader("1️⃣ Cost Efficiency by Age Group")
#     fig = px.bar(
#         age_perf,
#         x="Age",
#         y="CPC",
#         title="Cost per Conversion by Age Group"
#     )
#     fig.update_layout(xaxis_title="Age Group", yaxis_title="Cost per Conversion (₹)")
#     st.plotly_chart(fig, use_container_width=True)

# with right:
#     best = age_perf.iloc[0]
#     worst = age_perf.iloc[-1]

#     st.markdown("### 📌 Summary")
#     st.write(
#         f"**{best['Age']}** is the most cost-efficient age group "
#         f"(₹{best['CPC']:.2f} per conversion), while "
#         f"**{worst['Age']}** performs the worst."
#     )
#     st.markdown("### ✅ Recommended Action")
#     st.write("Increase budget for efficient age groups and reduce spend on low performers.")

# # ───────────────── 2️⃣ Conversion Share ─────────────────
# age_share = age_perf.copy()
# age_share["Conversion Share (%)"] = (
#     age_share["Conversions"] / age_share["Conversions"].sum() * 100
# )

# left, right = st.columns([3, 2])

# with left:
#     st.subheader("2️⃣ Conversion Contribution by Age")
#     fig = px.bar(
#         age_share,
#         x="Age",
#         y="Conversion Share (%)",
#         title="Conversion Share by Age Group"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with right:
#     top_age = age_share.loc[age_share["Conversion Share (%)"].idxmax()]
#     st.markdown("### 📌 Summary")
#     st.write(
#         f"**{top_age['Age']}** contributes the highest share of conversions."
#     )
#     st.markdown("### ✅ Recommended Action")
#     st.write("Ensure sufficient budget allocation for high-contributing age groups.")

# # ───────────────── 3️⃣ Wasted Spend Analysis ─────────────────
# waste_ages = age_perf[
#     (age_perf["Cost"] > age_perf["Cost"].median()) &
#     (age_perf["Conversions"] < age_perf["Conversions"].median())
# ]

# left, right = st.columns([3, 2])

# with left:
#     st.subheader("3️⃣ High Spend but Low Conversion Age Groups")
#     fig = px.scatter(
#         age_perf,
#         x="Cost",
#         y="Conversions",
#         size="Impressions",
#         color="Age",
#         title="Spend vs Conversions"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with right:
#     st.markdown("### 📌 Summary")
#     st.write(f"{len(waste_ages)} age group(s) incur high spend with low returns.")
#     st.markdown("### ✅ Recommended Action")
#     st.write("Exclude or reduce bids for these age groups.")

# # ───────────────── 4️⃣ Stability Analysis ─────────────────
# age_consistency = (
#     df_filtered.groupby("Age", as_index=False)
#     .agg(
#         Avg_CPC=("Cost per Conversion", "mean"),
#         Std_CPC=("Cost per Conversion", "std")
#     )
# )

# stable = age_consistency.loc[age_consistency["Std_CPC"].idxmin()]
# risky = age_consistency.loc[age_consistency["Std_CPC"].idxmax()]

# left, right = st.columns([3, 2])

# with left:
#     st.subheader("4️⃣ Cost Stability by Age")
#     fig = px.bar(
#         age_consistency,
#         x="Age",
#         y="Std_CPC",
#         title="Volatility in Cost per Conversion"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with right:
#     st.markdown("### 📌 Summary")
#     st.write(
#         f"**{stable['Age']}** is the most stable, while "
#         f"**{risky['Age']}** is the most volatile."
#     )
#     st.markdown("### ✅ Recommended Action")
#     st.write("Maintain stable budgets for consistent age groups and test volatile ones carefully.")

# # ───────────────── 5️⃣ Optimization Impact ─────────────────
# threshold = age_perf["CPC"].quantile(0.30)
# optimized = age_perf[age_perf["CPC"] <= threshold]

# left, right = st.columns([3, 2])

# with left:
#     st.subheader("5️⃣ Optimization Opportunity")
#     fig = px.bar(
#         optimized,
#         x="Age",
#         y="CPC",
#         title="Top 30% Most Efficient Age Groups"
#     )
#     st.plotly_chart(fig, use_container_width=True)

# with right:
#     st.markdown("### 📌 Summary")
#     st.write(
#         f"Optimizing targeting reduces average CPC from "
#         f"₹{age_perf['CPC'].mean():.2f} to ₹{optimized['CPC'].mean():.2f}."
#     )
#     st.markdown("### ✅ Recommended Action")
#     st.write("Focus spend on top-performing age segments to improve ROI.")
