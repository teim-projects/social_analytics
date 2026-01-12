import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import StringIO
from dotenv import load_dotenv
import os

# ───────────────── Page Setup ─────────────────
st.set_page_config(
    page_title="Google Ads Targeted Content Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Google Ads Targeted Content Performance Dashboard")

# ───────────────── Load & Clean Data ─────────────────
@st.cache_data
def load_targeted_content_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data_lines = [
        line for line in lines
        if not (
            "Targeted content report" in line
            or "All time" in line
            or line.startswith("Total:")
        )
    ]

    df = pd.read_csv(StringIO("".join(data_lines)))

    df.columns = [
        "Placement", "Placement_URL", "Type", "Campaign",
        "Ad_Group", "Impressions", "Interactions",
        "Interaction_Rate", "Currency", "Avg_Cost",
        "Cost", "Conversion_Rate", "Conversions",
        "Cost_per_Conversion"
    ]

    def clean_numeric(series):
        cleaned = (
            series.astype(str)
            .str.strip()
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .replace(["--", "—", "", "nan"], np.nan)
        )
        return pd.to_numeric(cleaned, errors="coerce")

    numeric_cols = [
        "Impressions", "Interactions", "Interaction_Rate",
        "Avg_Cost", "Cost", "Conversion_Rate",
        "Conversions", "Cost_per_Conversion"
    ]

    for col in numeric_cols:
        df[col] = clean_numeric(df[col])

    # Remove non-informative rows
    df = df[
        ~(
            (df["Impressions"] > 0) &
            (df["Interactions"] == 0) &
            (df["Cost"] == 0)
        )
    ].reset_index(drop=True)

    return df

# ───────────────── Load Data ─────────────────
load_dotenv()
DATA_PATH = os.getenv("GA_TARGETED_CONTENT_REPORT")
if not DATA_PATH:
    st.error("❌ GA_TARGETED_CONTENT_REPORT not set")
    st.stop()

df = load_targeted_content_data(DATA_PATH)
all_placements = sorted(df["Placement"].unique())
all_types = sorted(df["Type"].unique())

# ───────────────── Session State Init ─────────────────
st.session_state.setdefault("placements", [])
st.session_state.setdefault("types", all_types)
st.session_state.setdefault("select_all", False)
st.session_state.setdefault("placement_key", 0)

# ───────────────── Sidebar Filters ─────────────────
with st.sidebar:
    c1, c2 = st.sidebar.columns([1, 1])

    with c1:
        st.header("🎯 Filters")

    with c2:
        if st.button("❌ Clear Filters"):
            st.session_state.placements = []
            st.session_state.select_all = False
            st.session_state.types = all_types
            st.session_state.placement_key += 1  # force multiselect rerender
            st.rerun()

    # ───────────────── Select All Placements ─────────────────
    select_all_clicked = st.sidebar.checkbox(
        "Select All Placements",
        value=st.session_state.select_all
    )

    # Update placements immediately when checkbox is toggled
    if select_all_clicked != st.session_state.select_all:
        if select_all_clicked:
            st.session_state.placements = all_placements.copy()
        else:
            st.session_state.placements = []
        st.session_state.select_all = select_all_clicked
        st.session_state.placement_key += 1  # force multiselect rerender
        st.rerun()  # immediately refresh to apply selection in one go

    # ───────────────── Placement Multiselect ─────────────────
    placements = st.sidebar.multiselect(
        "Placement",
        options=all_placements,
        default=st.session_state.placements,
        key=f"placement_multiselect_{st.session_state.placement_key}"
    )

    # Manual selection overrides checkbox
    if set(placements) != set(st.session_state.placements):
        st.session_state.placements = placements
        st.session_state.select_all = len(placements) == len(all_placements)
        st.session_state.placement_key += 1
        st.rerun()

    # ───────────────── Placement Type Filter ─────────────────
    types = st.sidebar.multiselect(
        "Placement Type",
        options=all_types,
        default=st.session_state.types
    )
    st.session_state.types = types

# ───────────────── Empty State ─────────────────
if len(st.session_state.placements) == 0:
    st.warning("👈 Please select one or more placements to view analytics.")
    st.stop()

# ───────────────── Apply Filters ─────────────────
df_f = df[
    (df["Placement"].isin(st.session_state.placements)) &
    (df["Type"].isin(st.session_state.types))
]

# ───────────────── KPI SUMMARY ─────────────────
st.subheader("📌 Overall Performance Summary")
k1, k2, k3 = st.columns(3)
k1.metric("Placements", df_f.shape[0])
k2.metric("Total Impressions", f"{int(df_f['Impressions'].sum()):,}")
k3.metric("Total Spend (INR)", f"{df_f['Cost'].sum():,.2f}")

# ───────────────── EDA GRAPHS ─────────────────
def show_graphs(df_f):
    # 1️⃣ Top Placements by Impressions
    st.markdown("## 1️⃣ Top Placements by Impressions")
    c1, c2 = st.columns([2, 1])
    top_impr = df_f.sort_values("Impressions", ascending=False).head(10)
    with c1:
        st.plotly_chart(px.bar(top_impr, x="Impressions", y="Placement", orientation="h"), use_container_width=True)
    with c2:
        t = top_impr.iloc[0]
        st.info(f"**{t['Placement']}** generated the highest reach with **{int(t['Impressions']):,} impressions**.")

    # 2️⃣ Interaction Rate Distribution
    st.markdown("## 2️⃣ Interaction Rate Distribution")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(px.histogram(df_f, x="Interaction_Rate", nbins=20), use_container_width=True)
    with c2:
        st.info(f"Rates range from **{df_f['Interaction_Rate'].min():.2f}%** to **{df_f['Interaction_Rate'].max():.2f}%**.")

    # 3️⃣ Cost vs Interactions
    st.markdown("## 3️⃣ Cost vs Interactions")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(px.scatter(df_f, x="Cost", y="Interactions", hover_data=["Placement","Campaign","Type"]), use_container_width=True)
    with c2:
        best = df_f[df_f["Interactions"]>0].sort_values("Avg_Cost").iloc[0]
        st.success(f"**{best['Placement']}** is most efficient at **INR {best['Avg_Cost']:.2f} per interaction**.")

    # 4️⃣ Performance by Placement Type
    st.markdown("## 4️⃣ Performance by Placement Type")
    summary = df_f.groupby("Type").agg(Impressions=("Impressions","sum"), Interactions=("Interactions","sum"), Cost=("Cost","sum")).reset_index()
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(px.bar(summary, x="Type", y="Interactions"), use_container_width=True)
    with c2:
        top = summary.loc[summary["Interactions"].idxmax()]
        st.info(f"**{top['Type']}** placements generate the most engagement (**{int(top['Interactions'])} interactions**).")

    # 5️⃣ High Impressions but Low Engagement
    st.markdown("## 5️⃣ High Impressions but Low Engagement")
    low_eng = df_f[(df_f["Impressions"] > df_f["Impressions"].median()) & (df_f["Interaction_Rate"] < 1)]
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(low_eng[["Placement","Impressions","Interaction_Rate","Cost"]], use_container_width=True)
    with c2:
        st.warning(f"**{low_eng.shape[0]} placements** have high visibility but very low engagement (<1%).")

show_graphs(df_f)