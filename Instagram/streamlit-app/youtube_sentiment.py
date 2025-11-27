import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from dotenv import load_dotenv
import os

# 1. Load Data
load_dotenv()
default_file_path = os.getenv("Youtube_comment_data")
df = pd.read_csv(default_file_path)
df["Comment"] = df["Comment"].astype(str)

# 2. SUPER FAST SENTIMENT MODEL
model_name = "distilbert-base-uncased-finetuned-sst-2-english" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

labels = ["Negative", "Positive"]  # This model has ONLY 2 classes

device = "cpu"
model.to(device)
model.eval()

# 3. Sentiment Function
def get_sentiment_batch(text_list):
    tokens = tokenizer(
        text_list,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128     
    ).to(device)

    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.softmax(outputs.logits, dim=1)

    sentiments = []
    scores = []

    for p in probs:
        p = p.cpu().tolist()
        label = labels[p.index(max(p))]
        sentiments.append(label)
        scores.append(max(p))

    return sentiments, scores

# 4. Apply Sentiment in Batches
comments = df["Comment"].tolist()
batch_size = 64     # <<< works fast even on CPU

all_labels = []
all_scores = []

for i in tqdm(range(0, len(comments), batch_size)):
    batch = comments[i:i+batch_size]
    l, s = get_sentiment_batch(batch)
    all_labels.extend(l)
    all_scores.extend(s)

df["sentiment_label"] = all_labels
df["sentiment_score"] = all_scores

# 5. Save Final CSV
df.to_csv("fast_sentiment_output.csv", index=False)

print("\nFAST SENTIMENT ANALYSIS COMPLETE!")
print(f"Total comments processed: {len(df)}")
print("Output saved as fast_sentiment_output.csv")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import re
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from dotenv import load_dotenv
# import os

# load_dotenv()
# st.set_page_config(page_title="YouTube Sentiment Dashboard", layout="wide")

# st.title("📊 YouTube Comments Transformer Sentiment Dashboard")


# # ==========================================================
# # 1. LOAD DATA
# # ==========================================================
# # @st.cache_data

# default_file_path = os.getenv("Youtube_comment_data")
# def load_data():
#     df = pd.read_csv(default_file_path)
#     df["Comment"] = df["Comment"].astype(str)
#     return df

# df = load_data()
# st.success(f"Loaded {len(df)} comments.")


# # ==========================================================
# # 2. LOAD TRANSFORMER MODEL
# # ==========================================================
# @st.cache_resource
# def load_model():
#     model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#     model.half()
#     return tokenizer, model, device

# tokenizer, model, device = load_model()
# labels = ["Negative", "Neutral", "Positive"]


# # ==========================================================
# # 3. RELIGIOUS SPAM FILTER
# # ==========================================================
# religion_keywords = [
#     "god", "jesus", "christ", "allah", "pray", "prayer", "bible", "lord",
#     "christian", "islam", "muslim", "church", "miracle", "holy",
#     "scripture", "bless", "hallelujah", "savior"
# ]

# def is_religious_spam(text):
#     t = text.lower()
#     k = sum(1 for w in religion_keywords if w in t)
#     if k >= 2: return True
#     if k >= 1 and len(t.split()) > 25: return True
#     if "god is real" in t or "jesus loves you" in t: return True
#     if "quick proof" in t and "god" in t: return True
#     if any(x in t for x in [
#         "if there is a building", "if you see the creation",
#         "christ died for your sins", "salvation", "repent", "gospel"
#     ]): return True
#     return False


# # ==========================================================
# # 4. BEGGING SPAM FILTER
# # ==========================================================
# def is_begging_spam(text):
#     t = text.lower()
#     if "help me" in t: return True
#     if "plz" in t or "elp" in t: return True
#     if ("help" in t and any(e in t for e in ["😞", "😢", "😭"])): return True
#     if t.count("help") >= 2: return True
#     return False


# # ==========================================================
# # 5. CONTENT RELEVANT FILTER
# # ==========================================================
# content_keywords = [
#     "video", "tutorial", "content", "lesson",
#     "ken", "ken jee",
#     "you explained", "explained well",
#     "data", "data science", "analytics",
#     "python", "machine learning", "ml",
#     "project", "portfolio", "resume", "interview",
#     "career", "job", "advice", "tip"
# ]

# def is_relevant_comment(text):
#     t = text.lower()
#     if any(k in t for k in content_keywords): return True
#     if "ken" in t or "ken jee" in t: return True
#     if any(p in t for p in ["thanks", "thank you", "this helped", "very helpful"]):
#         return True
#     return False


# # ==========================================================
# # APPLY FILTERS
# # ==========================================================
# def apply_filters(df):
#     df = df[df["Comment"].apply(is_religious_spam) == False]
#     df = df[df["Comment"].apply(is_begging_spam) == False]
#     df = df[df["Comment"].apply(is_relevant_comment) == True]
#     return df.reset_index(drop=True)

# df_filtered = apply_filters(df)
# st.info(f"After filtering: {len(df_filtered)} comments remain.")


# # ==========================================================
# # 6. TRANSFORMER SENTIMENT (BATCHED)
# # ==========================================================
# @st.cache_data
# def run_sentiment(df):
#     comments = df["Comment"].tolist()
#     batch_size = 32

#     scores = []
#     labels_output = []

#     for i in range(0, len(comments), batch_size):
#         batch = comments[i:i+batch_size]

#         tokens = tokenizer(
#             batch,
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#             max_length=512
#         ).to(device)

#         with torch.no_grad():
#             outputs = model(
#                 input_ids=tokens["input_ids"],
#                 attention_mask=tokens["attention_mask"]
#             )

#         probs = torch.softmax(outputs.logits, dim=1)

#         for p in probs:
#             p = p.cpu().numpy().tolist()
#             label = labels[np.argmax(p)]
#             score = max(p)
#             labels_output.append(label)
#             scores.append(score)

#     df["sentiment_label"] = labels_output
#     df["sentiment_score"] = scores
#     return df

# st.write("⏳ Running transformer sentiment …")
# df_final = run_sentiment(df_filtered)
# st.success("Sentiment analysis completed!")


# # ==========================================================
# # 7. DASHBOARD VISUALS
# # ==========================================================
# st.header("📈 Sentiment Distribution")

# counts = df_final["sentiment_label"].value_counts()

# fig, ax = plt.subplots(figsize=(6,4))
# counts.plot(kind="bar", ax=ax, color=["red","grey","green"])
# ax.set_title("Sentiment Distribution")
# st.pyplot(fig)


# # PIE
# fig2, ax2 = plt.subplots(figsize=(6,6))
# ax2.pie(
#     counts.values,
#     labels=counts.index,
#     autopct="%1.1f%%"
# )
# ax2.set_title("Sentiment Share")
# st.pyplot(fig2)


# # ==========================================================
# # MONTHLY TREND
# # ==========================================================
# st.header("📅 Monthly Sentiment Trend")

# df_final["Published_At"] = pd.to_datetime(df_final["Published_At"], errors="coerce")
# df_final["month"] = df_final["Published_At"].dt.to_period("M")

# monthly = df_final.groupby(["month", "sentiment_label"]).size().unstack().fillna(0)

# if len(monthly) > 0:
#     fig3, ax3 = plt.subplots(figsize=(10,4))
#     for col in monthly.columns:
#         ax3.plot(monthly.index.to_timestamp(), monthly[col], label=col)
#     ax3.legend()
#     ax3.set_title("Monthly Sentiment Trend")
#     st.pyplot(fig3)
# else:
#     st.warning("Not enough date data available")


# # ==========================================================
# # LIKES vs SENTIMENT
# # ==========================================================
# st.header("👍 Likes vs Sentiment")

# if "Likes" in df_final.columns:
#     fig4, ax4 = plt.subplots(figsize=(6,4))
#     df_final.boxplot(column="Likes", by="sentiment_label", ax=ax4)
#     ax4.set_title("Likes vs Sentiment")
#     st.pyplot(fig4)
# else:
#     st.warning("Likes column not found")


# # ==========================================================
# # TOP POSITIVE / NEGATIVE
# # ==========================================================
# st.header("🏆 Top Positive Comments")
# st.dataframe(
#     df_final[df_final["sentiment_label"]=="Positive"]
#     .sort_values(["sentiment_score","Likes"], ascending=[False,False])
#     .head(10)
# )

# st.header("💔 Top Negative Comments")
# st.dataframe(
#     df_final[df_final["sentiment_label"]=="Negative"]
#     .sort_values(["sentiment_score","Likes"], ascending=[False,True])
#     .head(10)
# )


# # ==========================================================
# # DOWNLOAD FINAL CSV
# # ==========================================================
# st.download_button(
#     "⬇️ Download Final Annotated CSV",
#     df_final.to_csv(index=False),
#     "transformer_annotated_comments.csv"
# )
