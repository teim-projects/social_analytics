import streamlit as st
import pandas as pd
import plotly.express as px
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Ad_schedule_report.csv")

# ───────────────── Streamlit Page Setup ─────────────────
st.set_page_config(
    page_title="Google Ads Schedule Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Google Ads Schedule Performance Dashboard")

# ───────────────── Load & Clean Data ─────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        engine="python",
        skiprows=2
    )

    # Remove summary rows
    df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")]
    df.reset_index(drop=True, inplace=True)

    numeric_cols = [
        "Impr.",
        "Interactions",
        "Interaction rate",
        "Conv. rate",
        "Conversions",
        "Cost",
        "Cost / conv."
    ]

    for col in numeric_cols:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "")
            .str.replace("%", "")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Cost per Conversion"] = df["Cost / conv."]

    return df


df = load_data()

# ───────────────── Sidebar Filters ─────────────────
with st.sidebar:
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.header("🎯 Filters")

    with col_f2:
        if st.button("❌ Clear Filters"):
            st.session_state.selected_slots = df["Day & time"].unique().tolist()
            st.session_state.conv_range = (
                int(df["Conversions"].min()),
                int(df["Conversions"].max())
            )
            st.session_state.cost_range = (
                float(df["Cost"].min()),
                float(df["Cost"].max())
            )
            st.rerun()

# ───────────────── Initialize Session State ─────────────────
if "selected_slots" not in st.session_state:
    st.session_state.selected_slots = df["Day & time"].unique().tolist()

if "conv_range" not in st.session_state:
    st.session_state.conv_range = (
        int(df["Conversions"].min()),
        int(df["Conversions"].max())
    )

if "cost_range" not in st.session_state:
    st.session_state.cost_range = (
        float(df["Cost"].min()),
        float(df["Cost"].max())
    )

# ───────────────── Filter Widgets (KEY-BASED) ─────────────────
st.sidebar.multiselect(
    "Select Day & Time Blocks",
    options=sorted(df["Day & time"].unique()),
    key="selected_slots"
)

st.sidebar.slider(
    "Conversions Range",
    int(df["Conversions"].min()),
    int(df["Conversions"].max()),
    key="conv_range"
)

st.sidebar.slider(
    "Cost Range (₹)",
    float(df["Cost"].min()),
    float(df["Cost"].max()),
    key="cost_range"
)

# ───────────────── Apply Filters ─────────────────
filtered_df = df[
    (df["Day & time"].isin(st.session_state.selected_slots)) &
    (df["Conversions"].between(*st.session_state.conv_range)) &
    (df["Cost"].between(*st.session_state.cost_range))
]

# ───────────────── Graph 1: Cost per Conversion ─────────────────
st.subheader("1️⃣ Cost per Conversion by Scheduled Time Block")

col1, col2 = st.columns([3, 2])

with col1:
    fig1 = px.bar(
        filtered_df.sort_values("Cost per Conversion"),
        x="Cost per Conversion",
        y="Day & time",
        orientation="h",
        color="Cost per Conversion",
        color_continuous_scale="Viridis",
        title="Efficiency by Time Block"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    best = filtered_df.loc[filtered_df["Cost per Conversion"].idxmin()]
    worst = filtered_df.loc[filtered_df["Cost per Conversion"].idxmax()]

    st.markdown(
        f"""
**Summary:**  
- Most efficient slot: **{best['Day & time']}**  
- Cost per conversion: **₹{best['Cost per Conversion']:.2f}**  

- Least efficient slot: **{worst['Day & time']}**  
- Cost per conversion: **₹{worst['Cost per Conversion']:.2f}**

**Recommended Action:**  
Increase bids for efficient slots and reduce bids for inefficient ones.
"""
    )

# ───────────────── Graph 2: Spend vs Conversions ─────────────────
st.subheader("2️⃣ Spend vs Conversions Analysis")

col1, col2 = st.columns([3, 2])

with col1:
    fig2 = px.scatter(
        filtered_df,
        x="Cost",
        y="Conversions",
        size="Impr.",
        color="Cost per Conversion",
        hover_name="Day & time",
        color_continuous_scale="Plasma",
        title="Spend vs Conversions by Time Block"
    )
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    waste_slots = filtered_df[
        (filtered_df["Cost"] > filtered_df["Cost"].median()) &
        (filtered_df["Conversions"] < filtered_df["Conversions"].median())
    ]

    st.markdown(
        f"""
**Summary:**  
- **{len(waste_slots)}** time blocks show high spend but low conversions.

**Recommended Action:**  
Reduce bids or pause ads during these time blocks.
"""
    )

# ───────────────── Graph 3: Conversion Share ─────────────────
st.subheader("3️⃣ Conversion Share by Time Block")

conversion_share = filtered_df.copy()
conversion_share["Conversion Share (%)"] = (
    conversion_share["Conversions"] /
    conversion_share["Conversions"].sum() * 100
)

col1, col2 = st.columns([3, 2])

with col1:
    fig3 = px.bar(
        conversion_share.sort_values("Conversion Share (%)", ascending=False),
        x="Day & time",
        y="Conversion Share (%)",
        color="Conversion Share (%)",
        color_continuous_scale="Turbo",
        title="Distribution of Conversions Across Time Blocks"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    top_slot = conversion_share.loc[
        conversion_share["Conversion Share (%)"].idxmax()
    ]

    st.markdown(
        f"""
**Summary:**  
- **{top_slot['Day & time']}** contributes the highest share of conversions.

**Recommended Action:**  
Ensure ads remain active during high-conversion slots.
"""
    )

# ───────────────── Graph 4: Stability vs Volatility ─────────────────
st.subheader("4️⃣ Stability of Cost per Conversion")

consistency = (
    filtered_df
    .groupby("Day & time", as_index=False)
    .agg(
        Avg_CPC=("Cost per Conversion", "mean"),
        Std_CPC=("Cost per Conversion", "std")
    )
)

col1, col2 = st.columns([3, 2])

with col1:
    fig4 = px.scatter(
        consistency,
        x="Avg_CPC",
        y="Std_CPC",
        text="Day & time",
        title="Stability vs Volatility of CPC",
        labels={"Std_CPC": "CPC Volatility"}
    )
    fig4.update_traces(textposition="top center")
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    stable = consistency.loc[consistency["Std_CPC"].idxmin()]
    risky = consistency.loc[consistency["Std_CPC"].idxmax()]

    st.markdown(
        f"""
**Summary:**  
- Most stable slot: **{stable['Day & time']}**  
- Most volatile slot: **{risky['Day & time']}**

**Recommended Action:**  
Allocate steady budget to stable slots and apply cautious limits to volatile ones.
"""
    )

# -------------------------------------------------------------------------------------------------------

# # ───────────────── Graph 5: Optimized Schedule Impact ─────────────────
# st.subheader("5️⃣ Optimized Scheduling Impact on CPA")

# threshold = filtered_df["Cost per Conversion"].quantile(0.30)
# optimized = filtered_df[filtered_df["Cost per Conversion"] <= threshold]

# original_avg = filtered_df["Cost per Conversion"].mean()
# optimized_avg = optimized["Cost per Conversion"].mean()

# col1, col2 = st.columns([3, 2])

# with col1:
#     fig5 = px.box(
#         pd.DataFrame({
#             "Scenario": ["Original"] * len(filtered_df) + ["Optimized"] * len(optimized),
#             "Cost per Conversion": pd.concat([
#                 filtered_df["Cost per Conversion"],
#                 optimized["Cost per Conversion"]
#             ])
#         }),
#         x="Scenario",
#         y="Cost per Conversion",
#         title="Impact of Optimized Scheduling"
#     )
#     st.plotly_chart(fig5, use_container_width=True)

# with col2:
#     st.markdown(
#         f"""
# **Summary:**  
# - Average CPA reduced from **₹{original_avg:.2f}** to **₹{optimized_avg:.2f}**

# **Recommended Action:**  
# Tighten schedules to high-efficiency time blocks to improve ROI.
# """
#     )
