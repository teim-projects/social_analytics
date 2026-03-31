# ============================
# IMPORT LIBRARIES
# ============================

import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
from collections import Counter
from wordcloud import WordCloud
import pdfplumber
from pdfminer.high_level import extract_text
from sklearn.cluster import KMeans

st.set_page_config(layout="wide")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "data")
# DATA_PATH = os.path.join(DATA_DIR, "Shree-Laxmi-Stone-Depot-Ad-sets-1-Jan-2025-1-Jan-2026.csv")
# df_ads = pd.read_csv(DATA_PATH)

# ============================
# SESSION STATE
# ============================

if "filters" not in st.session_state:
    st.session_state.filters = {}

# ============================
# LOAD TEXT DATA
# ============================

file_path = os.path.join(DATA_DIR, "sevafacility_content_1772178354541.xlsx - All posts.pdf")
# file_path = "/content/sevafacility_content_1772178354541.xls - All posts.pdf"
text = extract_text(file_path)

# CLEANING
clean_text = re.sub(r'\n',' ',text)
clean_text = re.sub(r'[^\w\s#]','',clean_text)
clean_text = clean_text.lower()
clean_text = re.sub(r'\s+',' ',clean_text)

# POSTS
posts = re.split(r'#sevafacility',clean_text)
df = pd.DataFrame(posts,columns=["post"])
df = df[df.post.str.len()>50]
df.reset_index(drop=True,inplace=True)

df['length']=df['post'].apply(len)

# ============================
# HASHTAGS
# ============================

hashtags = re.findall(r'#\w+',clean_text)
hashtag_freq = Counter(hashtags)

top_hashtags = pd.DataFrame(
    hashtag_freq.most_common(30),
    columns=["hashtag","count"]
)

# ============================
# WORD FREQUENCY
# ============================

words = clean_text.split()
stopwords = ['the','and','for','with','your','our','you','are','today','call','now','from','all','safe','keep']
filtered = [w for w in words if w not in stopwords]

word_freq = Counter(filtered)

top_words = pd.DataFrame(
    word_freq.most_common(30),
    columns=["word","count"]
)

# ============================
# CATEGORY
# ============================

def categorize(post):
    if 'happy' in post or 'wishing' in post:
        return "Festival"
    elif 'call' in post or 'contact' in post:
        return "Promotional"
    elif 'protect' in post:
        return "Awareness"
    elif 'solution' in post:
        return "Product"
    else:
        return "General"

df['category'] = df['post'].apply(categorize)

# ============================
# THEMES
# ============================

themes = {
"mosquito": clean_text.count("mosquito"),
"rodent": clean_text.count("rodent"),
"cockroach": clean_text.count("cockroach"),
"fumigation": clean_text.count("fumigation"),
"hygiene": clean_text.count("hygiene")
}

theme_df = pd.DataFrame(themes.items(), columns=["Theme","Count"])

# ============================
# CTA
# ============================

cta = {
"Call": clean_text.count("call"),
"Contact": clean_text.count("contact"),
"Visit": clean_text.count("visit"),
"Book": clean_text.count("book")
}

cta_df = pd.DataFrame(cta.items(), columns=["CTA","Count"])

# ============================
# PDF EXTRACTION FUNCTION
# ============================

def extract_text_data(path):
    text=""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t=page.extract_text()
            if t:
                text+=t+"\n"

    lines=text.split("\n")
    data=[]

    for line in lines:
        parts=line.rsplit(" ",1)
        if len(parts)==2:
            name=parts[0].strip()
            value=parts[1].strip()
            if value.isdigit():
                data.append([name,int(value)])

    df=pd.DataFrame(data,columns=["category","value"])
    return df

# ============================
# LOAD FOLLOWERS & VISITORS
# ============================

followers_company=extract_text_data(os.path.join(DATA_DIR, "sevafacility_followers_1772178400931.xlsx - Company size.pdf"))
followers_industry=extract_text_data(os.path.join(DATA_DIR, "sevafacility_followers_1772178400931.xlsx - Industry.pdf"))
followers_seniority=extract_text_data(os.path.join(DATA_DIR, "sevafacility_followers_1772178400931.xlsx - Seniority.pdf"))
followers_job=extract_text_data(os.path.join(DATA_DIR, "sevafacility_followers_1772178400931.xlsx - Job function.pdf"))
followers_location=extract_text_data(os.path.join(DATA_DIR, "sevafacility_followers_1772178400931.xlsx - Location.pdf"))

visitors_company=extract_text_data(os.path.join(DATA_DIR, "sevafacility_visitors_1772178386457.xlsx - Company size.pdf"))
visitors_industry=extract_text_data(os.path.join(DATA_DIR, "sevafacility_visitors_1772178386457.xlsx - Industry.pdf"))
visitors_seniority=extract_text_data(os.path.join(DATA_DIR, "sevafacility_visitors_1772178386457.xlsx - Seniority.pdf"))
visitors_job=extract_text_data(os.path.join(DATA_DIR, "sevafacility_visitors_1772178386457.xlsx - Job function.pdf"))
visitors_location=extract_text_data(os.path.join(DATA_DIR, "sevafacility_visitors_1772178386457.xlsx - Location.pdf"))

followers_company.columns=["category","followers"]
followers_industry.columns=["category","followers"]
followers_seniority.columns=["category","followers"]
followers_job.columns=["category","followers"]
followers_location.columns=["category","followers"]

visitors_company.columns=["category","views"]
visitors_industry.columns=["category","views"]
visitors_seniority.columns=["category","views"]
visitors_job.columns=["category","views"]
visitors_location.columns=["category","views"]

# ============================
# TIME SERIES
# ============================

def extract_time_series(path):
    text=""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t=page.extract_text()
            if t:
                text+=t+"\n"
    return text.split("\n")

lines=extract_time_series(os.path.join(DATA_DIR, "sevafacility_visitors_1772178386457.xlsx - Visitor metrics.pdf"))

data=[]
for line in lines:
    match=re.match(r'(\d{2}/\d{2}/\d{4})\s+(\d+)\s+(\d+)\s+(\d+)', line)
    if match:
        data.append([
            match.group(1),
            int(match.group(2)),
            int(match.group(3)),
            int(match.group(4))
        ])

df_time=pd.DataFrame(data,columns=["date","desktop_views","mobile_views","total_views"])
df_time['date']=pd.to_datetime(df_time['date'])
df_time=df_time.sort_values("date")

# ============================
# UI
# ============================

st.title("📊 Seva Facility Analytics Dashboard")

# ============================
# GRAPH FUNCTION (2-COLUMN)
# ============================

def show_graph(fig, summary):
    col1, col2 = st.columns([3,1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.info(summary)

# ============================
# GRAPHS
# ============================

# 1 CUMULATIVE GROWTH
total = df_time['total_views'].sum()

summary = f"""
Total cumulative traffic reached **{total} views**.
Shows overall growth of platform visibility.
"""
df_time['cumulative']=df_time['total_views'].cumsum()

fig = px.line(df_time, x="date", y="cumulative", title="1️⃣ Cumulative Growth")
show_graph(fig, summary)

# ============================
# CLUSTERING
# ============================

# 2 INDUSTRY SEGMENTS
industry=pd.merge(followers_industry, visitors_industry, on="category", how="outer").fillna(0)

X=industry[['followers','views']]
kmeans=KMeans(n_clusters=3)
industry['segment']=kmeans.fit_predict(X)
seg_counts = industry['segment'].value_counts().to_dict()

summary = f"""
Data is segmented into clusters: {seg_counts}.
Each cluster represents different performance groups for targeting strategy.
"""

fig = px.scatter(industry, x="followers", y="views", color="segment", title="2️⃣ Industry Segments")
show_graph(fig, summary)

# 3 COMPANY SIZE SEGMENTS
industry = pd.merge(
    followers_industry,
    visitors_industry,
    on="category",
    how="outer"
).fillna(0)

fig = px.bar(
    industry.sort_values("followers", ascending=False).head(15),
    x="category",
    y=["followers","views"],
    barmode="group",
    title="3️⃣ Industry Market Comparison"
)

top_ind = industry.sort_values("followers", ascending=False).iloc[0]

summary = f"""
**{top_ind['category']}** leads in followers with {top_ind['followers']} and {top_ind['views']} views.
Gap between views and followers indicates conversion opportunity in high-traffic industries.
"""

show_graph(fig, summary)

# 4 ENGAGEMENT VARIABILITY
location = pd.merge(
    followers_location,
    visitors_location,
    on="category",
    how="outer"
).fillna(0)

location["gap"] = location["views"] - location["followers"]

fig = px.bar(
    location.sort_values("followers", ascending=False).head(10),
    x="category",
    y=["followers","views"],
    barmode="group",
    title="4️⃣ Location Market Strength"
)

top_loc = location.sort_values("followers", ascending=False).iloc[0]

summary = f"""
**{top_loc['category']}** shows strongest market with {top_loc['followers']} followers.
Gap between views and followers highlights regions with untapped conversion potential.
"""

show_graph(fig, summary)

# 5 DECISION MAKER SENIORITY
seniority = pd.merge(
    followers_seniority,
    visitors_seniority,
    on="category",
    how="outer"
).fillna(0)

fig = px.bar(
    seniority,
    x="category",
    y=["followers","views"],
    barmode="group",
    title="5️⃣ Decision Maker Engagement"
)

top_sen = seniority.sort_values("followers", ascending=False).iloc[0]

summary = f"""
**{top_sen['category']}** level shows highest engagement.
Higher views vs followers in some levels indicate awareness without conversion.
"""

show_graph(fig, summary)

# 6 ENGAGEMENT BY CONTENT THEME
fig = px.bar(theme_df, x="Theme", y="Count", title="6️⃣ Service Focus Area")

top_theme = theme_df.sort_values("Count", ascending=False).iloc[0]

summary = f"""
Primary service focus is **{top_theme['Theme']}** with {top_theme['Count']} mentions.
Indicates core service positioning in content strategy.
"""

show_graph(fig, summary)