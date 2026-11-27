# Olist Review Insight — Why customers leave bad reviews

**End-to-end analysis of 94,563 orders and 8,760 Portuguese review comments from the Brazilian e-commerce marketplace Olist. SQL + Python for the data work, Power BI and Tableau for the dashboards, a Streamlit app with an LLM-based agent and local semantic search for the demo.**

---

## 🔗 Links

| | |
|---|---|
| 🧠 **Live app** (Streamlit) | [olist-review-risk.streamlit.app](https://olist-review-risk.streamlit.app/) |
| 📊 **Power BI dashboard** (PDF preview) | [dashboard/dashboard.pdf](dashboard/dashboard.pdf) |
| 📈 **Tableau dashboard** (Part 1) | [Olist – Delivery or Product?](https://public.tableau.com/app/profile/pariya.lakani/viz/OlistDeliveryorProduct/OlistReviewDrivers) |
| 🎤 **Presentation slides** (Part 2) | [Google Slides](https://docs.google.com/presentation/d/1CsBHF1f0oCt6ZM0mj0v31Xrn1liXnsXX/edit?usp=sharing&ouid=109599469251149380786&rtpof=true&sd=true) |
| 🎤 **Presentation slides** (Part 1) | [Google Slides](https://docs.google.com/presentation/d/1EruG-10wZLD4qMhrh9Fn2lYXXDfpRdol/edit?usp=sharing&ouid=109599469251149380786&rtpof=true&sd=true) |
| 🗂️ **Source data** | [Brazilian E-Commerce Public Dataset — Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |

---

## TL;DR

> **62.4%** of late orders get a negative review.
> But because 93% of orders arrive on time, **66% of all negative reviews still come from on-time orders.**
> So "fix late delivery" doesn't solve the problem.
> Reading the **8,760 Portuguese review comments** reveals what those on-time complaints are really about — and surfaces one issue that was invisible in the old metrics.

**The surprise finding:** 782 orders were marked delivered on time, but customers wrote that the product never arrived — 752 of those reviews were written **after** the recorded delivery date. Likely a tracking-system or last-mile issue worth auditing.

---

## The business problem

Olist is a marketplace: sellers list, Olist handles logistics, customers rate the order from 1 to 5. The marketplace sees the negative-review rate but not the reason, and support capacity is finite — they need to know whether to invest in **logistics**, **sellers**, or **product moderation** first.

This project answers that in two passes:

- **Part 1 — Timing and rating.** Hypothesis tests, EDA and a predictive model show late delivery is the strongest single driver of bad reviews. But two thirds of bad reviews come from on-time orders, so delivery alone cannot be the whole story.
- **Part 2 — Content analysis.** An LLM classifies the 8,760 Portuguese review comments into 8 complaint types; a local semantic index lets managers search by theme; a Streamlit app ties it together with a scoring tool and a chat agent.

---

## Key findings

**From the data at a glance**

| Metric | Value |
|---|---:|
| Orders analysed | 94,563 |
| Negative review rate (score 1-2) | 12.3% |
| Late delivery rate | 6.7% |
| Negative rate on on-time orders | 8.7% |
| Negative rate on late orders | **62.4%** |
| Share of negative reviews that come from on-time orders | **66%** |
| Negative reviews with Portuguese comments | 8,760 |
| **Phantom-delivery** cases (on-time & "never arrived") | **782** |

**What the reviews are actually about**

| Rank | Category | Count | % | On-time share |
|---|---|---:|---:|---:|
| 1 | Never arrived | 2,284 | 26.1% | 34% |
| 2 | Wrong item | 1,720 | 19.6% | 96% |
| 3 | Other | 1,000 | 11.4% | 75% |
| 4 | Late delivery | 983 | 11.2% | 32% |
| 5 | Misleading description | 791 | 9.0% | 97% |
| 6 | Broken or damaged | 737 | 8.4% | 95% |
| 7 | Poor quality | 724 | 8.3% | 94% |
| 8 | Customer service issue | 521 | 5.9% | 64% |

Three insights jump out:

1. **Late delivery is the strongest single driver** — a late order is about 7× more likely to get a negative review, and the jump is sharpest after 3 days late (8.7% → 32% → 68%).
2. **On-time negative reviews are mostly product issues.** Wrong item (96% on-time), misleading description (97%), poor quality (94%), broken (95%). Combined, product-related complaints are **~64% of all on-time negatives** — a seller-moderation and listing-quality problem, not a logistics one.
3. **Phantom delivery.** 782 "never arrived" complaints are on orders the system marked delivered on time — and 752 of those reviews were written *after* the recorded delivery date. Not impatient customers; the delivery records appear unreliable for this subset.

Three independent methods agreed on finding 1 — EDA, hypothesis tests (Cramér's V 0.41 for delivery vs 0.07 for product category), and a Gradient Boosting model that performs almost as well with delivery features alone as with all features. Findings 2 and 3 come from Part 2's content analysis.

---

## Recommendations

**For Olist operations**

1. **Reduce late deliveries**, especially delays past 3 days — the main lever.
2. **Proactively inform customers** when an order will be late, and send the review request only after the order arrives.
3. **Audit the 782 phantom-delivery orders** — the tracking-system integrity is in doubt for this subset.
4. **Invest in seller and listing moderation** — on-time product issues (wrong item, misleading description, broken, poor quality) are now the bigger problem by volume.
5. **Deploy the review-insight agent** as a daily-use tool for the ops / customer-service team, so they can explore complaints without a data analyst.

Recommendations are **hypotheses to confirm with operations** — the analysis shows associations, not causality.

---

## Part 2 — The new work

### 1. Classification of 8,760 Portuguese comments

- 8 categories designed from direct inspection of the data (not auto-discovered).
- `classify_all.py` uses an OpenAI model with **Pydantic structured output** so the model can only pick one of the 8 labels — it cannot invent a 9th.
- Classification is run **once, offline**; results are cached to `data/sample_classified_full.csv`. The app and notebooks read the CSV, so no classification API call happens at demo time.
- **Validation:** 18 random reviews manually translated and checked — 18 / 18 matched. Small by production standards; meaningful as a sanity check.

### 2. Phantom delivery — the headline finding

Among the 782 "on-time but never arrived" complaints, the review-vs-delivery-date comparison shows:
- **752** reviews written *after* the recorded delivery date (real complaints)
- **30** reviews written *before* (possibly impatient customers)

A representative customer comment: *"The tracking shows it was delivered. I have not received it."*

This pattern is invisible to the standard `is_late` metric. For Olist it is an operational audit item — possibly a tracking-system, carrier-reporting, or last-mile issue.

### 3. RAG-powered chat agent

A manager can ask natural-language questions in English. The agent uses four tools:

| Tool | Purpose |
|---|---|
| `get_category_counts` | counts per complaint category, filterable by delivery status |
| `get_sample_reviews` | random reviews from one category |
| `search_reviews` | **semantic search** across all 8,760 reviews, by meaning |
| `translate_text` | Portuguese → English for every example before display |

The classification labels answer the questions I thought of when I designed them. **Semantic search answers the questions I did not** — "packaging", "refund", "customer service" — because it matches on meaning, not label.

**All embeddings run locally.** The model (`paraphrase-multilingual-MiniLM-L12-v2`, ~400 MB) encodes each review into a 384-dimensional vector, stored in a local **Chroma** index. Review text never leaves the laptop for retrieval. The LLM call that writes the final answer runs on **Groq** (`openai/gpt-oss-120b`) for speed and cost; the agent system prompt constrains it to answer only from tool results.

### 4. Streamlit app

Four tabs:
- **Predict** — single-order risk score (Part 1 model)
- **Batch upload** — score a CSV of orders at once
- **Dashboard** — interactive charts with delivery-status filter
- **Ask the agent** — chat UI, with starter questions and a download-conversation export

---

## Part 1 — Original analysis (summary)

The first pass established that **delivery timing** is the dominant signal for negative reviews:

- A late order is ~7× more likely to get a negative review.
- Hypothesis tests (chi-squared, Cramér's V): delivery 0.41 (strong), product category 0.07 (weak).
- A tuned **Gradient Boosting** model with only delivery features performs nearly as well as the full model — confirming delivery's dominance.
- Model metrics on negative reviews: precision 0.38, recall 0.46, F1 0.42. It finds about half of the unhappy customers; the ceiling is set by the on-time negatives, whose reasons are not in the structured data. **Part 2 is designed to recover that missing signal from the review text.**

Full notebooks under `notebooks/01…03`.

---

## Workflow and tools

| Step | Tool | File |
|---|---|---|
| Data load into MySQL | Python (pandas), SQLAlchemy | [`notebooks/01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb) |
| Build analysis table | SQL (MySQL) | [`sql/01_build_analysis_table.sql`](sql/01_build_analysis_table.sql) |
| Cleaning, EDA, hypothesis tests | Python (pandas, matplotlib, seaborn, scipy) | [`notebooks/02_cleaning_eda.ipynb`](notebooks/02_cleaning_eda.ipynb) |
| Analysis queries | SQL | [`sql/02_analysis_queries.sql`](sql/02_analysis_queries.sql) |
| ML model | scikit-learn (Pipeline, Gradient Boosting) | [`notebooks/03_ml_model.ipynb`](notebooks/03_ml_model.ipynb) |
| Dashboards | Power BI, Tableau | [`dashboard/`](dashboard/) |
| **Part 2 — LLM classification** | OpenAI, Pydantic | [`classify_all.py`](classify_all.py) |
| **Part 2 — Agent notebook** | Groq (OpenAI SDK compatible) | [`notebooks/04_review_agent_organized.ipynb`](notebooks/04_review_agent_organized.ipynb) |
| **Part 2 — RAG / semantic search** | Sentence Transformers + Chroma | [`notebooks/05_rag_search.ipynb`](notebooks/05_rag_search.ipynb) |
| **Part 2 — App** | Streamlit + Groq + RAG | [`app_final.py`](app_final.py) |

---

## Repository structure

```
├── notebooks/
│   ├── 01_data_understanding.ipynb        Part 1 — load + inspect raw data
│   ├── 02_cleaning_eda.ipynb              Part 1 — EDA, hypothesis tests
│   ├── 03_ml_model.ipynb                  Part 1 — Gradient Boosting model
│   ├── 04_review_agent_organized.ipynb    Part 2 — classification + agent
│   └── 05_rag_search.ipynb                Part 2 — Chroma index + RAG agent
├── sql/                                   Part 1 — build + analysis queries
├── dashboard/
│   ├── dashboard.pbix                     Power BI (Part 2 updated)
│   ├── dashboard.pdf                      PDF export
│   └── Olist – Delivery or Product.twbx   Tableau (Part 1)
├── data/
│   └── sample_classified_full.csv         Cached LLM classifications (8,760)
├── app_final.py                           Streamlit app with all 4 tabs
├── classify_all.py                        One-time offline classification script
├── build_dashboard_with_category.py       Adds `category` to olist_dashboard.csv
├── model.pkl                              Trained Part 1 model
├── olist_dashboard.csv                    Clean analysis table (one row per order)
├── sample_orders.csv                      200 test orders for the batch upload
├── requirements.txt
└── README.md
```

---

## How to run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the Olist dataset from Kaggle into ./data/
#    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

# 3. Part 1 — build the analysis table
#    (load into local MySQL via notebooks/01, then run sql/01, then notebooks/02-03)

# 4. Part 2 — one-time classification (needs an OpenAI API key in .env)
python classify_all.py

# 5. Part 2 — build the semantic-search index
#    Open notebooks/05_rag_search.ipynb and Run All (first run takes ~5 min)

# 6. Run the app (needs GROQ_API_KEY in .env)
streamlit run app_final.py
```

`.env` keys used:
- `OPENAI_API_KEY` — only for the one-time `classify_all.py` run
- `GROQ_API_KEY` — for the agent at demo time

---

## Limitations

- **Data is from 2016–2018.** Current Olist operations may differ.
- **Classification is validated on 18 manual samples** — a reasonable sanity check for a portfolio project, not production-scale.
- **The analysis shows associations, not causality.** The "what if" numbers (e.g. ~3,400 preventable negative reviews) are estimates under the association.
- **Only orders with one seller** (98.7% of the dataset) — multi-seller orders cannot be attributed to one party.
- **No structured product-quality signal** — product attributes are only proxies (category, seller, price).
- **The agent's answers depend on the LLM.** The system prompt constrains it to tool results, but LLM outputs should be treated as drafts for an analyst to verify.

---

## About

**Pariya Lakani** · Data analyst (QA / Trust & Safety background)
Berlin, Germany · [LinkedIn](https://www.linkedin.com/in/pariya-lakani/)

Project originally built as the final project of the Ironhack Data Analytics bootcamp; Part 2 is a self-directed extension.
