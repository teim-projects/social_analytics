import streamlit as st
import pandas as pd
import plotly.express as px
import os

# ───────────────── Page Config ─────────────────
st.set_page_config(
    page_title="Google Ads Device Performance Dashboard",
    layout="wide"
)

st.title("📱 Google Ads Device Performance Dashboard")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "Device_report.csv")

# ───────────────── Load Data ─────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(
        DATA_PATH,
        engine="python",
        skiprows=2
    )

    # Remove summary rows like "Total: Account"
    df = df[~df.iloc[:, 0].astype(str).str.startswith("Total")].reset_index(drop=True)

    numeric_cols = [
        "Impr.", "Clicks", "Interactions",
        "Interaction rate", "CTR", "Cost",
        "Conversions", "Conv. rate", "Cost / conv."
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "")
                .str.replace("%", "")
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Compute Cost per Conversion
    if "Cost / conv." in df.columns:
        df["Cost per Conversion"] = df["Cost / conv."]
    else:
        df["Cost per Conversion"] = df["Cost"] / df["Conversions"]

    return df

device_df = load_data()

# ───────────────── Initialize Session State ─────────────────
if "device_filter" not in st.session_state:
    st.session_state.device_filter = device_df["Device"].dropna().unique().tolist()

# ───────────────── Sidebar Filters ─────────────────
with st.sidebar:
    col_f1, col_f2 = st.columns([1, 1])

    with col_f1:
        st.header("🔍 Filters")

    with col_f2:
        if st.button("❌ Clear Filters"):
            st.session_state.device_filter = (
                device_df["Device"].dropna().unique().tolist()
            )
            st.rerun()

    all_devices = device_df["Device"].dropna().unique().tolist()

    # 🔥 IMPORTANT FIX: use key, NOT default
    selected_devices = st.multiselect(
        "Select Device(s)",
        options=all_devices,
        key="device_filter"   # <- this controls the widget
    )

    # Safety fallback
    if not selected_devices:
        st.warning("No devices selected — displaying all devices by default.")
        st.session_state.device_filter = all_devices.copy()

    # 5️⃣ Filter dataframe
    filtered_df = device_df[device_df["Device"].isin(selected_devices)]

# ───────────────── Aggregations ─────────────────
device_perf = (
    filtered_df
    .groupby("Device", as_index=False)
    .agg(
        Impressions=("Impr.", "sum"),
        Cost=("Cost", "sum"),
        Conversions=("Conversions", "sum"),
        Cost_per_Conversion=("Cost per Conversion", "mean")
    )
)

avg_cpc = device_perf["Cost_per_Conversion"].mean()

# ───────────────── Graph 1 ─────────────────
st.subheader("1️⃣ Cost Efficiency by Device")
col1, col2 = st.columns([3, 2])

fig1 = px.bar(
    device_perf,
    x="Device",
    y="Cost_per_Conversion",
    title="Cost per Conversion by Device"
)
col1.plotly_chart(fig1, use_container_width=True)

best = device_perf.loc[device_perf["Cost_per_Conversion"].idxmin()]
worst = device_perf.loc[device_perf["Cost_per_Conversion"].idxmax()]

col2.markdown(
    f"""
**Summary**
- Most efficient: **{best['Device']}**
- Avg CPC: ₹{best['Cost_per_Conversion']:.2f}
- Least efficient: **{worst['Device']}**

**Recommended Action**
- Increase bids on efficient devices
- Reduce spend on high CPC devices
"""
)

# ───────────────── Graph 4 ─────────────────
st.subheader("2️⃣ Cost Share vs Conversion Share")
share = device_perf.copy()
share["Cost Share (%)"] = share["Cost"] / share["Cost"].sum() * 100
share["Conversion Share (%)"] = share["Conversions"] / share["Conversions"].sum() * 100

col1, col2 = st.columns([3, 2])

fig4 = px.bar(
    share,
    x="Device",
    y=["Cost Share (%)", "Conversion Share (%)"],
    barmode="group",
    title="Spend vs Conversion Contribution"
)
col1.plotly_chart(fig4, use_container_width=True)

efficient = share[share["Conversion Share (%)"] > share["Cost Share (%)"]]

col2.markdown(
    f"""
**Summary**
- {len(efficient)} device(s) generate more value than spend

**Recommended Action**
- Shift budget toward high-value devices
"""
)

# ───────────────── Graph 5 ─────────────────
st.subheader("3️⃣ Performance Stability by Device")
stability = (
    filtered_df
    .groupby("Device", as_index=False)
    .agg(
        Avg_CPC=("Cost per Conversion", "mean"),
        Std_CPC=("Cost per Conversion", "std")
    )
)

# FIX: Handle NaN std
stability["Std_CPC"] = stability["Std_CPC"].fillna(0)
stability["Bubble_Size"] = stability["Std_CPC"] + 1

col1, col2 = st.columns([3, 2])

fig5 = px.scatter(
    stability,
    x="Avg_CPC",
    y="Std_CPC",
    size="Bubble_Size",
    text="Device",
    title="CPC Stability Analysis (Lower = More Stable)"
)
col1.plotly_chart(fig5, use_container_width=True)

stable = stability.loc[stability["Std_CPC"].idxmin()]
risky = stability.loc[stability["Std_CPC"].idxmax()]

col2.markdown(
    f"""
**Summary**
- Most stable device: **{stable['Device']}**
- Most volatile device: **{risky['Device']}**

**Recommended Action**
- Scale stable devices
- Optimize volatile devices carefully
"""
)

# ───────────────── Graph 6 ─────────────────
st.subheader("4️⃣ Funnel Leakage Analysis")
funnel = (
    filtered_df
    .groupby("Device", as_index=False)
    .agg(
        Interactions=("Interactions", "sum"),
        Conversions=("Conversions", "sum")
    )
)
funnel["Interaction-to-Conversion Ratio"] = funnel["Interactions"] / funnel["Conversions"].replace(0, pd.NA)

col1, col2 = st.columns([3, 2])

fig6 = px.bar(
    funnel,
    x="Device",
    y="Interaction-to-Conversion Ratio",
    title="Engagement vs Conversion Efficiency"
)
col1.plotly_chart(fig6, use_container_width=True)

leak = funnel[funnel["Interaction-to-Conversion Ratio"] > funnel["Interaction-to-Conversion Ratio"].median()]

col2.markdown(
    f"""
**Summary**
- {len(leak)} device(s) show funnel leakage

**Recommended Action**
- Improve UX and conversion flow
"""
)

# ───────────────── Device Actions Table ─────────────────
st.subheader("📌 Device-Level Optimization Actions")

def device_action(cpc):
    if pd.isna(cpc):
        return "No conversions"
    elif cpc < avg_cpc * 0.7:
        return "Scale aggressively"
    elif cpc < avg_cpc * 1.3:
        return "Optimize & monitor"
    else:
        return "Reduce bids"

device_perf["Recommended Action"] = device_perf["Cost_per_Conversion"].apply(device_action)

st.dataframe(
    device_perf[["Device", "Cost_per_Conversion", "Conversions", "Recommended Action"]],
    use_container_width=True
)
