"""
One-time script: classify all 8,760 negative reviews with comments.
Run once:  python classify_all.py
Saves:     data/sample_classified_full.csv
After this, the notebook and app don't need OpenAI for classification.
"""

import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
MODEL = "gpt-6-luna"

OUT_PATH = "data/sample_classified_full.csv"

if os.path.exists(OUT_PATH):
    print(f"{OUT_PATH} already exists. Delete it to re-classify.")
    exit()

# Load + merge + filter
dashboard = pd.read_csv("olist_dashboard.csv")
reviews = pd.read_csv("data/olist_order_reviews_dataset.csv")

merged = dashboard.merge(
    reviews[["order_id", "review_comment_title", "review_comment_message"]],
    on="order_id", how="left",
)
negative = merged[(merged["is_negative"] == 1) & merged["review_comment_message"].notna()].copy()
print(f"Classifying {len(negative)} reviews...")


class ReviewCategory(BaseModel):
    category: Literal[
        "late_delivery", "never_arrived", "wrong_item", "broken_damaged",
        "poor_quality", "customer_service_issue", "misleading_desc", "other",
    ]


def classify_review(text):
    r = client.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Classify the Portuguese customer review into one category: late_delivery (arrived late), never_arrived (not delivered), wrong_item (different from order), broken_damaged (physically broken), poor_quality (works but bad), customer_service_issue (unreachable support, ignored cancellation, poor communication), misleading_desc (not as described), other."},
            {"role": "user", "content": text},
        ],
        response_format=ReviewCategory,
        reasoning_effort="none",
    )
    return r.choices[0].message.parsed.category


def classify_safe(text):
    try:
        return classify_review(text)
    except Exception:
        return "error"


texts = negative["review_comment_message"].tolist()
results = [None] * len(texts)

def classify_at(i):
    results[i] = classify_safe(texts[i])

with ThreadPoolExecutor(max_workers=10) as ex:
    list(tqdm(ex.map(classify_at, range(len(texts))), total=len(texts)))

negative["category"] = results
negative.to_csv(OUT_PATH, index=False)
print(f"\nSaved to {OUT_PATH}")
print(negative["category"].value_counts())