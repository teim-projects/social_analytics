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

# 1 WORDS
top_word = top_words.iloc[0]

summary = f"""
Most frequent keyword is **{top_word['word']}**, appearing {top_word['count']} times.
Content heavily emphasizes this theme, indicating a focused messaging strategy.
"""
fig = px.bar(top_words, x="count", y="word", orientation='h', title="1️⃣ Top Words")
show_graph(fig, summary)

# 2 THEMES
top_theme = theme_df.sort_values("Count", ascending=False).iloc[0]

summary = f"""
Most discussed theme is **{top_theme['Theme']}** with {top_theme['Count']} mentions.
This reflects the core service focus area in content strategy.
"""
fig = px.bar(theme_df, x="Theme", y="Count", title="2️⃣ Themes")
show_graph(fig, summary)

# ============================
# FOLLOWERS GRAPHS
# ============================

# ============================
# TIME SERIES
# ============================

# 3 DAILY VIEWS
max_day = df_time.loc[df_time['total_views'].idxmax()]

summary = f"""
Peak traffic was **{max_day['total_views']} views** on {max_day['date'].date()}.
Indicates impact of specific campaigns or events.
"""
fig = px.line(df_time, x="date", y="total_views", title="3️⃣ Daily Views")
show_graph(fig, summary)

# 4 DEVICE TRAFFIC
mobile_share = df_time['mobile_views'].sum() / df_time['total_views'].sum()

summary = f"""
Mobile contributes **{mobile_share:.0%}** of total traffic,
indicating strong mobile-first audience behavior.
"""
fig = px.line(df_time, x="date", y=["desktop_views","mobile_views"], title="4️⃣ Device Traffic")
show_graph(fig, summary)

# 5 TREND + 7DAY AVG
avg = df_time['total_views'].mean()

summary = f"""
Average daily traffic is **{avg:.0f} views**.
Trend shows fluctuations around this baseline.
"""
df_time['7day_avg']=df_time['total_views'].rolling(7).mean()

fig = px.line(df_time, x="date", y=["total_views","7day_avg"], title="5️⃣ Trend")
show_graph(fig, summary)

# 6 WEEKDAY PATTERN
monthly = df_time.copy()
monthly['month'] = monthly['date'].dt.to_period('M').astype(str)

# GROUP DATA
monthly = monthly.groupby('month')['total_views'].sum().reset_index()

# NOW FIND TOP MONTH
top_month = monthly.sort_values("total_views", ascending=False).iloc[0]

summary = f"""
Highest traffic month is **{top_month['month']}** with {top_month['total_views']} views.
Indicates peak performance period.
"""
fig = px.bar(monthly, x="month", y="total_views", title="6️⃣ Monthly Traffic")
show_graph(fig, summary)

# 7 GROWTH %
df_time['growth_%']=df_time['total_views'].pct_change()*100
max_growth = df_time['growth_%'].max()

summary = f"""
Maximum growth observed was **{max_growth:.2f}%**.
Shows strong spikes likely driven by campaigns.
"""

fig = px.line(df_time, x="date", y="growth_%", title="7️⃣ Growth %")
show_graph(fig, summary)

# 8 WEEKDAY TRAFFIC
df_time['weekday']=df_time['date'].dt.day_name()
best_day = df_time.groupby('weekday')['total_views'].mean().idxmax()

summary = f"""
**{best_day}** has the highest average traffic.
This is the most effective day for posting.
"""

fig = px.bar(df_time.groupby('weekday')['total_views'].mean().reset_index(),
             x="weekday", y="total_views", title="8️⃣ Weekday Traffic")
show_graph(fig, summary)

# ============================
# FINAL
# ============================

# st.success("✅ All 30 graphs included with insights")