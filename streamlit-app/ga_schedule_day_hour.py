import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dotenv import load_dotenv
import os

load_dotenv()

DATA_PATH = os.getenv("GA_SCHEDULE_DAY_HOUR_PATH")

# ─────────────────────────────
# Page Config
# ─────────────────────────────
st.set_page_config(
    page_title="Google Ads Schedule (Day & Hour) Performance Dashboard",
    page_icon="⏰",
    layout="wide"
)

st.title("⏰ Google Ads Schedule Performance Dashboard")
st.caption("Interactive day & hour-wise efficiency analysis")

# ─────────────────────────────
# Load & Clean Data
# ─────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        engine="python",
        skiprows=2
    )

    df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")]
    df = df.dropna(subset=["Day of the week", "Hour of the day"])
    df.reset_index(drop=True, inplace=True)

    numeric_cols = [
        "Impr.", "Interactions", "Interaction rate",
        "Conv. rate", "Conversions", "Cost", "Cost / conv."
    ]

    for col in numeric_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "")
            .str.replace("%", "")
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Cost per Conversion"] = df["Cost / conv."]

    return df

schedule_df = load_data()

# ─────────────────────────────
# Sidebar Filters (Session Managed)
# ─────────────────────────────

# =====================================
# PREPARE DEFAULT VALUES
# =====================================
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

min_hour = int(schedule_df["Hour of the day"].min())
max_hour = int(schedule_df["Hour of the day"].max())


# =====================================
# SESSION STATE INIT
# =====================================
if "day_filter" not in st.session_state:
    st.session_state.day_filter = day_order

if "hour_filter" not in st.session_state:
    st.session_state.hour_filter = (min_hour, max_hour)


# =====================================
# SIDEBAR : FILTERS + CLEAR BUTTON
# =====================================
with st.sidebar:
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.header("🔎 Filters")

    with col_f2:
        if st.button("❌ Clear Filters"):
            st.session_state.day_filter = day_order
            st.session_state.hour_filter = (min_hour, max_hour)
            st.rerun()

# =====================================
# SIDEBAR FILTERS (STATE-OWNED)
# =====================================
st.sidebar.multiselect(
    "Day of the Week",
    options=day_order,
    key="day_filter"
)

st.sidebar.slider(
    "Hour of the Day",
    min_value=min_hour,
    max_value=max_hour,
    key="hour_filter"
)

# =====================================
# APPLY FILTERS
# =====================================
filtered_df = schedule_df.copy()

filtered_df = filtered_df[
    (filtered_df["Day of the week"].isin(st.session_state.day_filter)) &
    (filtered_df["Hour of the day"] >= st.session_state.hour_filter[0]) &
    (filtered_df["Hour of the day"] <= st.session_state.hour_filter[1])
]

# ─────────────────────────────
# KPI Section
# ─────────────────────────────
st.subheader("📌 Schedule KPIs")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Cost (₹)", f"{filtered_df['Cost'].sum():,.2f}")
k2.metric("Total Conversions", f"{filtered_df['Conversions'].sum():,.2f}")
k3.metric("Avg Cost / Conversion (₹)", f"{filtered_df['Cost per Conversion'].mean():.2f}")
k4.metric("Active Time Slots", len(filtered_df))

st.divider()

# ─────────────────────────────
# Helper: Graph + Summary Layout
# ─────────────────────────────
def graph_with_summary(fig, summary, number, title):
    st.subheader(f"{number} {title}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(summary)
    st.divider()

# ─────────────────────────────
# 1️⃣ Day-wise Cost per Conversion
# ─────────────────────────────
day_perf = (
    filtered_df
    .groupby("Day of the week", as_index=False)
    .agg({
        "Cost": "sum",
        "Conversions": "sum",
        "Cost per Conversion": "mean"
    })
)

day_perf["Day of the week"] = pd.Categorical(
    day_perf["Day of the week"],
    categories=day_order,
    ordered=True
)

day_perf = day_perf.sort_values("Day of the week")

best_day = day_perf.loc[day_perf["Cost per Conversion"].idxmin()]
worst_day = day_perf.loc[day_perf["Cost per Conversion"].idxmax()]

fig_day = px.bar(
    day_perf,
    x="Day of the week",
    y="Cost per Conversion",
    title="Day-wise Conversion Efficiency",
    labels={"Cost per Conversion": "Cost per Conversion (₹)"}
)

graph_with_summary(
    fig_day,
    f"""
    **Insight:**  
    **{best_day['Day of the week']}** delivers the lowest average cost per conversion  
    (₹{best_day['Cost per Conversion']:.2f}), while  
    **{worst_day['Day of the week']}** is the least efficient.

    **Recommendation:**  
    Increase bids or budgets on high-performing days and reduce exposure on inefficient days.
    """,
    "1️⃣",
    "Day-wise Conversion Efficiency"
)

# ─────────────────────────────
# 2️⃣ Hourly Cost per Conversion
# ─────────────────────────────
hour_perf = (
    filtered_df
    .groupby("Hour of the day", as_index=False)
    .agg({
        "Cost": "sum",
        "Conversions": "sum",
        "Cost per Conversion": "mean"
    })
)

best_hour = hour_perf.loc[hour_perf["Cost per Conversion"].idxmin()]
worst_hour = hour_perf.loc[hour_perf["Cost per Conversion"].idxmax()]

fig_hour = px.line(
    hour_perf,
    x="Hour of the day",
    y="Cost per Conversion",
    markers=True,
    title="Hourly Conversion Efficiency",
    labels={"Cost per Conversion": "Cost per Conversion (₹)"}
)

graph_with_summary(
    fig_hour,
    f"""
    **Insight:**  
    Ads are most efficient around **hour {int(best_hour['Hour of the day'])}**,  
    while **hour {int(worst_hour['Hour of the day'])}** shows the highest cost per conversion.

    **Recommendation:**  
    Prioritize ad delivery during efficient hours and reduce bids during poor-performing hours.
    """,
    "2️⃣",
    "Hourly Conversion Efficiency"
)

# ─────────────────────────────
# 3️⃣ Spend vs Conversions (Day–Hour Slots)
# ─────────────────────────────
day_hour_perf = (
    filtered_df
    .groupby(["Day of the week", "Hour of the day"], as_index=False)
    .agg({
        "Cost": "sum",
        "Conversions": "sum",
        "Cost per Conversion": "mean"
    })
)

best_slot = day_hour_perf.loc[day_hour_perf["Cost per Conversion"].idxmin()]
worst_slot = day_hour_perf.loc[day_hour_perf["Cost per Conversion"].idxmax()]

fig_scatter = px.scatter(
    filtered_df,
    x="Cost",
    y="Conversions",
    color="Day of the week",
    hover_data=["Hour of the day"],
    title="Spend vs Conversions by Time Slot"
)

graph_with_summary(
    fig_scatter,
    f"""
    **Insight:**  
    Best slot: **{best_slot['Day of the week']} at hour {int(best_slot['Hour of the day'])}**  
    Worst slot: **{worst_slot['Day of the week']} at hour {int(worst_slot['Hour of the day'])}**

    **Recommendation:**  
    Aggressively schedule ads during high-performing slots and restrict spend during weak slots.
    """,
    "3️⃣",
    "Day–Hour Slot Efficiency"
)

# ─────────────────────────────
# 4️⃣ High Spend – Low Conversion Slots
# ─────────────────────────────
waste_slots = filtered_df[
    (filtered_df["Cost"] > filtered_df["Cost"].quantile(0.75)) &
    (filtered_df["Conversions"] < filtered_df["Conversions"].quantile(0.25))
]

st.subheader("4️⃣ Inefficient Time Slots")

col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(
        waste_slots[
            ["Day of the week", "Hour of the day", "Cost", "Conversions"]
        ],
        use_container_width=True
    )

with col2:
    st.markdown(
        f"""
        **Insight:**  
        **{len(waste_slots)} time slots** show high spend with low conversion output.

        **Recommendation:**  
        Reduce bids or exclude these slots from scheduling to minimize wasted spend.
        """
    )

st.divider()

# ─────────────────────────────
# 5️⃣ Conversion Share by Hour
# ─────────────────────────────
hour_share = (
    filtered_df
    .groupby("Hour of the day", as_index=False)
    .agg(Conversions=("Conversions", "sum"))
)

hour_share["Conversion Share (%)"] = (
    hour_share["Conversions"] /
    hour_share["Conversions"].sum() * 100
)

peak_hour = hour_share.loc[
    hour_share["Conversion Share (%)"].idxmax()
]

fig_share = px.line(
    hour_share,
    x="Hour of the day",
    y="Conversion Share (%)",
    markers=True,
    title="Distribution of Conversions Across Hours"
)

graph_with_summary(
    fig_share,
    f"""
    **Insight:**  
    **Hour {int(peak_hour['Hour of the day'])}** contributes the highest share of total conversions.

    **Recommendation:**  
    Ensure sufficient budget availability during peak conversion hours.
    """,
    "5️⃣",
    "Hourly Conversion Share"
)


# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from dotenv import load_dotenv
# import os

# load_dotenv()

# DATA_PATH = os.getenv("GA_SCHEDULE_DAY_HOUR_PATH")

# # ─────────────────────────────
# # Page Config
# # ─────────────────────────────
# st.set_page_config(
#     page_title="Google Ads Schedule Performance Dashboard",
#     page_icon="⏰",
#     layout="wide"
# )

# st.title("⏰ Google Ads Schedule Performance Dashboard")
# st.caption("Day & hour-wise efficiency analysis with actionable insights")

# # ─────────────────────────────
# # Load & Clean Data
# # ─────────────────────────────
# @st.cache_data
# def load_data():
#     df = pd.read_csv(
#         DATA_PATH,
#         engine="python",
#         skiprows=2
#     )

#     df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")]
#     df = df.dropna(subset=["Day of the week", "Hour of the day"])
#     df.reset_index(drop=True, inplace=True)

#     numeric_cols = [
#         "Impr.", "Interactions", "Interaction rate",
#         "Conv. rate", "Conversions", "Cost", "Cost / conv."
#     ]

#     for col in numeric_cols:
#         df[col] = (
#             df[col]
#             .astype(str)
#             .str.replace(",", "")
#             .str.replace("%", "")
#         )
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     df["Cost per Conversion"] = df["Cost / conv."]

#     return df

# schedule_df = load_data()

# # ─────────────────────────────
# # Sidebar Filters (Session Managed)
# # ─────────────────────────────
# st.sidebar.header("🔎 Filters")

# days = st.sidebar.multiselect(
#     "Day of the Week",
#     ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
# )

# hour_range = st.sidebar.slider(
#     "Hour of the Day",
#     min_value=int(schedule_df["Hour of the day"].min()),
#     max_value=int(schedule_df["Hour of the day"].max()),
#     value=(
#         int(schedule_df["Hour of the day"].min()),
#         int(schedule_df["Hour of the day"].max())
#     )
# )

# filtered_df = schedule_df.copy()

# if days:
#     filtered_df = filtered_df[filtered_df["Day of the week"].isin(days)]

# filtered_df = filtered_df[
#     (filtered_df["Hour of the day"] >= hour_range[0]) &
#     (filtered_df["Hour of the day"] <= hour_range[1])
# ]

# # ─────────────────────────────
# # KPI Section
# # ─────────────────────────────
# st.subheader("📌 Key Schedule KPIs")

# k1, k2, k3, k4 = st.columns(4)
# k1.metric("Total Cost (₹)", f"{filtered_df['Cost'].sum():,.2f}")
# k2.metric("Total Conversions", f"{filtered_df['Conversions'].sum():,.2f}")
# k3.metric("Avg Cost / Conversion (₹)", f"{filtered_df['Cost per Conversion'].mean():.2f}")
# k4.metric("Active Time Slots", len(filtered_df))

# st.divider()

# # ─────────────────────────────
# # Helper: Graph + Summary Layout
# # ─────────────────────────────
# def graph_with_summary(plot_func, summary, number, title):
#     st.subheader(f"{number} {title}")
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         plot_func()
#     with col2:
#         st.markdown(summary)
#     st.divider()

# # ─────────────────────────────
# # 1️⃣ Day-wise Cost per Conversion
# # ─────────────────────────────
# day_perf = (
#     filtered_df
#     .groupby("Day of the week", as_index=False)
#     .agg({
#         "Cost": "sum",
#         "Conversions": "sum",
#         "Cost per Conversion": "mean"
#     })
# )

# best_day = day_perf.loc[day_perf["Cost per Conversion"].idxmin()]
# worst_day = day_perf.loc[day_perf["Cost per Conversion"].idxmax()]

# def plot_day_efficiency():
#     plt.figure()
#     plt.bar(day_perf["Day of the week"], day_perf["Cost per Conversion"])
#     plt.xlabel("Day of the Week")
#     plt.ylabel("Cost per Conversion (₹)")
#     plt.title("Day-wise Conversion Efficiency")
#     st.pyplot(plt)

# graph_with_summary(
#     plot_day_efficiency,
#     f"""
#     **Insight:**  
#     **{best_day['Day of the week']}** delivers the lowest average cost per conversion  
#     (₹{best_day['Cost per Conversion']:.2f}), while **{worst_day['Day of the week']}**
#     is the least efficient.

#     **Recommendation:**  
#     Increase bids or budgets on high-performing days and reduce exposure on inefficient days.
#     """,
#     "1️⃣",
#     "Day-wise Conversion Efficiency"
# )

# # ─────────────────────────────
# # 2️⃣ Hourly Cost per Conversion
# # ─────────────────────────────
# hour_perf = (
#     filtered_df
#     .groupby("Hour of the day", as_index=False)
#     .agg({
#         "Cost": "sum",
#         "Conversions": "sum",
#         "Cost per Conversion": "mean"
#     })
# )

# best_hour = hour_perf.loc[hour_perf["Cost per Conversion"].idxmin()]
# worst_hour = hour_perf.loc[hour_perf["Cost per Conversion"].idxmax()]

# def plot_hour_efficiency():
#     plt.figure()
#     plt.plot(hour_perf["Hour of the day"], hour_perf["Cost per Conversion"])
#     plt.xlabel("Hour of the Day")
#     plt.ylabel("Cost per Conversion (₹)")
#     plt.title("Hourly Conversion Efficiency")
#     st.pyplot(plt)

# graph_with_summary(
#     plot_hour_efficiency,
#     f"""
#     **Insight:**  
#     Ads are most efficient around **hour {int(best_hour['Hour of the day'])}**,  
#     while **hour {int(worst_hour['Hour of the day'])}** shows the highest cost per conversion.

#     **Recommendation:**  
#     Prioritize ad delivery during efficient hours and reduce bids during poor-performing hours.
#     """,
#     "2️⃣",
#     "Hourly Conversion Efficiency"
# )

# # ─────────────────────────────
# # 3️⃣ Best & Worst Day–Hour Slots
# # ─────────────────────────────
# day_hour_perf = (
#     filtered_df
#     .groupby(["Day of the week", "Hour of the day"], as_index=False)
#     .agg({
#         "Cost": "sum",
#         "Conversions": "sum",
#         "Cost per Conversion": "mean"
#     })
# )

# best_slot = day_hour_perf.loc[day_hour_perf["Cost per Conversion"].idxmin()]
# worst_slot = day_hour_perf.loc[day_hour_perf["Cost per Conversion"].idxmax()]

# def plot_day_hour_scatter():
#     plt.figure()
#     plt.scatter(filtered_df["Cost"], filtered_df["Conversions"])
#     plt.xlabel("Cost (₹)")
#     plt.ylabel("Conversions")
#     plt.title("Spend vs Conversions by Time Slot")
#     st.pyplot(plt)

# graph_with_summary(
#     plot_day_hour_scatter,
#     f"""
#     **Insight:**  
#     The most efficient slot is **{best_slot['Day of the week']} at hour {int(best_slot['Hour of the day'])}**.  
#     The least efficient slot is **{worst_slot['Day of the week']} at hour {int(worst_slot['Hour of the day'])}**.

#     **Recommendation:**  
#     Aggressively schedule ads during high-performing slots and restrict spend during weak slots.
#     """,
#     "3️⃣",
#     "Day–Hour Slot Efficiency"
# )

# # ─────────────────────────────
# # 4️⃣ High Spend – Low Conversion Slots
# # ─────────────────────────────
# waste_slots = filtered_df[
#     (filtered_df["Cost"] > filtered_df["Cost"].quantile(0.75)) &
#     (filtered_df["Conversions"] < filtered_df["Conversions"].quantile(0.25))
# ]

# st.subheader("4️⃣ Inefficient Time Slots")

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.dataframe(
#         waste_slots[
#             ["Day of the week", "Hour of the day", "Cost", "Conversions"]
#         ],
#         use_container_width=True
#     )

# with col2:
#     st.markdown(
#         f"""
#         **Insight:**  
#         **{len(waste_slots)} time slots** fall into a high-spend, low-conversion category.

#         **Recommendation:**  
#         Reduce bids or exclude these slots from ad scheduling to minimize wasted spend.
#         """
#     )

# st.divider()

# # ─────────────────────────────
# # 5️⃣ Conversion Share by Hour
# # ─────────────────────────────
# hour_share = (
#     filtered_df
#     .groupby("Hour of the day", as_index=False)
#     .agg(Conversions=("Conversions", "sum"))
# )

# hour_share["Conversion Share (%)"] = (
#     hour_share["Conversions"] /
#     hour_share["Conversions"].sum() * 100
# )

# peak_hour = hour_share.loc[
#     hour_share["Conversion Share (%)"].idxmax()
# ]

# def plot_conversion_share():
#     plt.figure()
#     plt.plot(
#         hour_share["Hour of the day"],
#         hour_share["Conversion Share (%)"]
#     )
#     plt.xlabel("Hour of the Day")
#     plt.ylabel("Conversion Share (%)")
#     plt.title("Distribution of Conversions Across Hours")
#     st.pyplot(plt)

# graph_with_summary(
#     plot_conversion_share,
#     f"""
#     **Insight:**  
#     **Hour {int(peak_hour['Hour of the day'])}** contributes the highest share of total conversions.

#     **Recommendation:**  
#     Ensure sufficient budget availability during peak conversion hours to avoid missed opportunities.
#     """,
#     "5️⃣",
#     "Hourly Conversion Share"
# )
