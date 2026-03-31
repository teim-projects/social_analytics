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

# 1 HASHTAGS
top_tag = top_hashtags.iloc[0]

summary = f"""
Top hashtag is **{top_tag['hashtag']}** with {top_tag['count']} occurrences.
This indicates strong reliance on specific tags; diversifying hashtags can improve discoverability.
"""
fig = px.bar(top_hashtags, x="count", y="hashtag", orientation='h', title="1️⃣ Top Hashtags")
show_graph(fig, summary)

# 2 CATEGORY
cat_counts = df['category'].value_counts()
top_cat = cat_counts.idxmax()

summary = f"""
**{top_cat}** is the dominant content category with {cat_counts.max()} posts.
This suggests a strong focus on this content type, potentially limiting variety.
"""
fig = px.bar(df['category'].value_counts().reset_index(),
             x='category', y='count', title="2️⃣ Category Distribution")
show_graph(fig, summary)

# 3 LENGTH HIST
avg_len = df['length'].mean()

summary = f"""
Average post length is **{avg_len:.0f} characters**, indicating moderate content size.
Optimizing length based on engagement metrics could improve performance.
"""
fig = px.histogram(df, x="length", title="3️⃣ Post Length Distribution")
show_graph(fig, summary)

# 4 CTA
top_cta = cta_df.sort_values("Count", ascending=False).iloc[0]

summary = f"""
Primary CTA used is **{top_cta['CTA']}** appearing {top_cta['Count']} times.
Indicates strong push toward direct action and conversions.
"""
fig = px.bar(cta_df, x="CTA", y="Count", title="4️⃣ CTA Usage")
show_graph(fig, summary)

# 5 COMPANY SIZE
top = followers_company.sort_values("followers", ascending=False).iloc[0]

summary = f"""
Highest follower segment is **{top['category']}** with {top['followers']} users.
Indicates strongest penetration in this company size segment.
"""
fig = px.bar(followers_company, x="followers", y="category", title="5️⃣ Followers Company Size")
show_graph(fig, summary)

# 6 INDUSTRY
top = followers_industry.sort_values("followers", ascending=False).iloc[0]

summary = f"""
Top industry is **{top['category']}** with {top['followers']} followers.
This industry shows highest alignment with the service offering.
"""
fig = px.bar(followers_industry.sort_values("followers",ascending=False).head(15),
             x="followers", y="category", title="6️⃣ Top Industries")
show_graph(fig, summary)

# 7 SENIORITY
top = followers_seniority.sort_values("followers", ascending=False).iloc[0]

summary = f"""
Most audience belongs to **{top['category']}** level with {top['followers']} users.
Indicates engagement is strongest at this decision level.
"""
fig = px.bar(followers_seniority, x="category", y="followers", title="7️⃣ Seniority")
show_graph(fig, summary)

# 8 JOB
top = followers_job.sort_values("followers", ascending=False).iloc[0]

summary = f"""
Top job function is **{top['category']}** with {top['followers']} followers.
Indicates highest relevance among these roles.
"""
fig = px.bar(followers_job.head(10), x="category", y="followers", title="8️⃣ Job Roles")
show_graph(fig, summary)

# 9 LOCATION
top = followers_location.sort_values("followers", ascending=False).iloc[0]

summary = f"""
Top location is **{top['category']}** with {top['followers']} followers.
Indicates strongest geographic market presence.
"""
fig = px.bar(followers_location.head(10), x="category", y="followers", title="9️⃣ Locations")
show_graph(fig, summary)

# ============================
# VISITORS
# ============================

#  10 COMPANY SIZE
top = visitors_company.sort_values("views", ascending=False).iloc[0]

summary = f"""
Highest visitor segment is **{top['category']}** with {top['views']} views.
Indicates strongest traffic source among company sizes.
"""
fig = px.bar(visitors_company, x="views", y="category", title="1️⃣0️⃣ Visitor Company Size")
show_graph(fig, summary)

# 11 INDUSTRY
top = visitors_industry.sort_values("views", ascending=False).iloc[0]

summary = f"""
Top visitor industry is **{top['category']}** with {top['views']} views.
Shows where maximum awareness is generated.
"""
fig = px.bar(visitors_industry.head(15), x="category", y="views", title="1️⃣1️⃣ Visitor Industry")
show_graph(fig, summary)

# 12 SENIORITY
top = visitors_seniority.sort_values("views", ascending=False).iloc[0]

summary = f"""
Most visitors belong to **{top['category']}** level with {top['views']} views.
Indicates strong engagement at this level.
"""
fig = px.bar(visitors_seniority, x="category", y="views", title="1️⃣2️⃣ Visitor Seniority")
show_graph(fig, summary)

# 13 JOB
top = visitors_job.sort_values("views", ascending=False).iloc[0]

summary = f"""
Top visitor job role is **{top['category']}** with {top['views']} views.
Reflects highest interest from this functional group.
"""
fig = px.bar(visitors_job, x="category", y="views", title="1️⃣3️⃣ Visitor Job")
show_graph(fig, summary)

# 14 LOCATION
top = visitors_location.sort_values("views", ascending=False).iloc[0]

summary = f"""
Top visitor location is **{top['category']}** with {top['views']} views.
Indicates highest traffic generation from this region.
"""
fig = px.bar(visitors_location.head(10), x="category", y="views", title="1️⃣4️⃣ Visitor Location")
show_graph(fig, summary)

# 15 TRAFFIC DISTRIBUTION
median = df_time['total_views'].median()

summary = f"""
Median daily traffic is **{median:.0f} views**, indicating typical performance level.
Most days cluster around this value.
"""
fig = px.histogram(df_time, x="total_views", title="1️⃣5️⃣ Traffic Distribution")
show_graph(fig, summary)

