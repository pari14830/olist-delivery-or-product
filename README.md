# Olist: Are bad reviews caused by delivery or product?

Final project of the Ironhack Data Analytics bootcamp.

**Main question:** why do customers leave bad reviews on the Olist marketplace: because of the delivery, or because of the product and the seller? And where should the marketplace act first?

**Links**
- 🔮 **Live app:** [olist-review-risk.streamlit.app](https://olist-review-risk.streamlit.app/)
- 📊 **Dashboard:** [dashboard/dashboard.pdf](dashboard/dashboard.pdf) (Power BI file: [dashboard/dashboard.pbix](dashboard/dashboard.pbix))
- 🗂️ **Data:** [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

## Key findings

| | |
|---|---|
| Orders analysed | 94,563 |
| Negative review rate (score 1-2) | 12.3% |
| Late delivery rate | 6.7% |
| Negative rate, on-time orders | 8.7% |
| Negative rate, late orders | **62.4%** |

1. **Late delivery is the strongest driver.** A late order is about 7 times more likely to get a negative review. The biggest jump is after 3 days late (1-3 days: 32%, 4-7 days: 68%).
2. **Fixing delivery has the biggest impact.** Fixing late delivery could prevent about 3,400 negative reviews, fixing the 10 worst sellers about 200.
3. **Some sellers are much worse than others**, even when delivery is on time (up to 54% negative). But they cause only about 4% of the negative reviews.
4. **Product category has only a small effect.** The worst category is about 2 times the average.
5. **Many customers review before the order arrives.** 5% of reviews are written before delivery. 93.5% of these orders are late, and 71% of these reviews are negative.

Three different methods give the same answer: the EDA, the hypothesis tests (Cramér's V: delivery 0.41 strong, category 0.07 weak) and the machine learning model (with only delivery features, it is almost as good as with all features).

## Recommendations

1. **Reduce late deliveries**, especially delays of more than 3 days (main lever).
2. **Proactively inform customers** when an order will be late, and send the review request only after the order arrives.
3. **Work with the worst sellers** (quick win).
4. **Next step:** analyse the review comments to understand the negative reviews on on-time orders (66% of all negative reviews).

---

## Workflow and tools

| Step | Tool | File |
|---|---|---|
| 1. Data understanding, load into MySQL | Python (pandas), SQLAlchemy | [notebooks/01_data_understanding.ipynb](notebooks/01_data_understanding.ipynb) |
| 2. Join the tables into one analysis table | SQL (MySQL) | [sql/01_build_analysis_table.sql](sql/01_build_analysis_table.sql) |
| 3. Cleaning, EDA, hypothesis tests | Python (pandas, matplotlib, seaborn, scipy) | [notebooks/02_cleaning_eda.ipynb](notebooks/02_cleaning_eda.ipynb) |
| 4. Analysis queries | SQL (MySQL) | [sql/02_analysis_queries.sql](sql/02_analysis_queries.sql) |
| 5. Dashboard | Power BI | [dashboard/](dashboard/) |
| 6. Machine learning | scikit-learn (Pipeline, Gradient Boosting) | [notebooks/03_ml_model.ipynb](notebooks/03_ml_model.ipynb) |
| 7. App | Streamlit | [app.py](app.py) |

## Machine learning

**Goal:** predict if an order will get a negative review, so Olist can act early (for example, an apology email or a coupon).

- Models compared: Logistic Regression, Random Forest, Gradient Boosting
- Imbalanced data (12% negative): `class_weight='balanced'`, models compared with precision, recall and F1
- Random Forest overfitted (train ROC-AUC 1.0 vs test 0.72), fixed with smaller trees
- **Final model: Gradient Boosting (tuned)**: precision 0.38, recall 0.46, F1 0.42 for negative reviews

The model finds almost half of the unhappy customers. It cannot go much higher, because most negative reviews are on-time orders and their reasons (product quality, wrong size...) are not in the data.

## Limitations

- Only orders with one seller (98.7% of orders).
- The data has no real product quality information, only proxies (category, seller, price).
- The results show associations, not proof of cause.
- The "what if" numbers (for example 3,400 fewer negative reviews) are estimates.

## Data cleaning decisions

- Only delivered orders with a delivery date.
- One row per order (the review is for the whole order). Payment columns were dropped, because they multiplied the rows.
- Orders with more than one seller (1.3%) were removed, because the review cannot be linked to one seller.
- Negative review = score 1 or 2. Late order = delivered after the estimated date.

---

## How to run

1. Download the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and put the CSV files in a folder called `data/`.
2. Run `notebooks/01_data_understanding.ipynb` (loads the tables into a local MySQL database `olist`).
3. Run `sql/01_build_analysis_table.sql` in MySQL.
4. Run `notebooks/02_cleaning_eda.ipynb` (creates `olist_dashboard.csv`) and `notebooks/03_ml_model.ipynb` (creates `model.pkl`).
5. Run the app: `pip install -r requirements.txt`, then `streamlit run app.py`.

## Repository structure

```
├── notebooks/            # data understanding, EDA, machine learning
├── sql/                  # build the analysis table, analysis queries
├── dashboard/            # Power BI dashboard (.pbix and .pdf)
├── .streamlit/           # app theme
├── app.py                # Streamlit app
├── model.pkl             # trained model (pipeline)
├── olist_dashboard.zip   # clean table (one row per order), used by the app
├── sample_orders.csv     # 200 test orders to try the batch upload in the app
└── requirements.txt
```

---

**Author:** Pariya Lakani
