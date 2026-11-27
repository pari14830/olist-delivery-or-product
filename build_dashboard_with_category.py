"""Build a new CSV identical to olist_dashboard.csv
but with one extra 'category' column (negative orders only)."""
import pandas as pd

dashboard = pd.read_csv("olist_dashboard.csv")
classified = pd.read_csv("data/sample_classified_full.csv")

classified["category"] = classified["category"].replace(
    {"seller_issue": "customer_service_issue"}
)

# Merge category onto ALL dashboard rows (left join keeps every row)
dashboard = dashboard.merge(
    classified[["order_id", "category"]].drop_duplicates(subset=["order_id"]),
    on="order_id",
    how="left",
)

# Non-negative orders or negatives without comment -> label clearly
dashboard["category"] = dashboard["category"].fillna("no_complaint")

print(f"Rows: {len(dashboard)}  (should match original olist_dashboard.csv)")
print(f"Columns: {len(dashboard.columns)}  (one more than original)")
print("\nCategory counts:")
print(dashboard["category"].value_counts())

# Save as NEW file — olist_dashboard.csv dast nakhord
dashboard.to_csv("olist_dashboard.csv", index=False)
print("\nSaved: data/olist_dashboard_with_category.csv")