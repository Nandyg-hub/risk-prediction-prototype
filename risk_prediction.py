"""
risk_prediction.py
----------------------------------
Core prototype pipeline for "Risk Prediction from Complaint Trends".

Steps:
1. Load synthetic complaint data (data/complaints.csv)
2. Aggregate into daily per-product complaint features (rolling counts,
   rolling severity, rate-of-change of both)
3. Derive a "future risk event" label by looking 30 days AHEAD for a
   spike in high-severity complaints (this simulates "did a safety issue
   actually emerge shortly after this point in time?")
4. Train a classifier to predict that label using ONLY features available
   at the time (no lookahead leakage into features, only into the label)
5. Evaluate performance and plot risk score vs. time for a product with
   an embedded escalation, to visually demonstrate early warning.

This is a PROTOTYPE for demonstration purposes only. It is not validated,
not trained on real data, and not intended for actual clinical/regulatory
decision-making.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "data/complaints.csv"
OUT_DIR = "outputs"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# ---------------------------------------------------------------
# 2. Build a full daily calendar per product (fill days with 0 complaints)
# ---------------------------------------------------------------
all_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
products = df["product_id"].unique()

daily = (
    df.groupby(["product_id", "date"])
      .agg(complaint_count=("complaint_id", "count"),
           avg_severity=("severity", "mean"),
           max_severity=("severity", "max"),
           high_severity_count=("severity", lambda s: (s >= 4).sum()))
      .reset_index()
)

full_index = pd.MultiIndex.from_product([products, all_dates], names=["product_id", "date"])
daily_full = (
    daily.set_index(["product_id", "date"])
         .reindex(full_index, fill_value=0)
         .reset_index()
)
daily_full["avg_severity"] = daily_full["avg_severity"].fillna(0)

# ---------------------------------------------------------------
# 3. Feature engineering: rolling windows (per product, using only PAST data)
# ---------------------------------------------------------------
daily_full = daily_full.sort_values(["product_id", "date"])

feature_frames = []
for pid, grp in daily_full.groupby("product_id"):
    grp = grp.sort_values("date").copy()
    grp["roll7_count"] = grp["complaint_count"].rolling(7, min_periods=1).sum()
    grp["roll30_count"] = grp["complaint_count"].rolling(30, min_periods=1).sum()
    grp["roll30_high_sev"] = grp["high_severity_count"].rolling(30, min_periods=1).sum()
    grp["roll30_avg_severity"] = grp["avg_severity"].rolling(30, min_periods=1).mean()
    grp["trend_7_vs_30"] = (grp["roll7_count"] * (30/7)) - grp["roll30_count"]  # positive = accelerating
    feature_frames.append(grp)

feat = pd.concat(feature_frames, ignore_index=True)

# ---------------------------------------------------------------
# 4. Label: does a "risk event" occur in the NEXT 30 days?
# A risk event day = day with >=3 high-severity (4-5) complaints in a 7-day window
# ---------------------------------------------------------------
event_frames = []
for pid, grp in feat.groupby("product_id"):
    grp = grp.sort_values("date").copy()
    grp["high_sev_7d"] = grp["high_severity_count"].rolling(10, min_periods=1).sum()
    grp["is_event_day"] = (grp["high_sev_7d"] >= 2).astype(int)

    # label = will an event day occur within the NEXT 30 days (excluding today)?
    future_event = grp["is_event_day"].shift(-1).rolling(30, min_periods=1).max().shift(-29)
    # simpler/robust approach: reverse rolling max over next 30 days
    reversed_event = grp["is_event_day"][::-1]
    future_window = reversed_event.rolling(30, min_periods=1).max()[::-1]
    grp["label_future_risk"] = future_window.shift(-1).fillna(0).astype(int)
    event_frames.append(grp)

labeled = pd.concat(event_frames, ignore_index=True)

# ---------------------------------------------------------------
# 5. Train / test split BY TIME (train on first ~65% of days, test on the
#    remaining ~35%). This matches the real use case: train on historical
#    data, predict risk going forward, across ALL products.
# ---------------------------------------------------------------
feature_cols = [
    "complaint_count", "avg_severity", "max_severity", "high_severity_count",
    "roll7_count", "roll30_count", "roll30_high_sev", "roll30_avg_severity",
    "trend_7_vs_30"
]

split_date = labeled["date"].quantile(0.65)
train_df = labeled[labeled["date"] <= split_date]
test_df = labeled[labeled["date"] > split_date]

X_train, y_train = train_df[feature_cols], train_df["label_future_risk"]
X_test, y_test = test_df[feature_cols], test_df["label_future_risk"]

print(f"Train period: up to {split_date.date()}  ({len(train_df)} rows, "
      f"{y_train.sum()} positive labels)")
print(f"Test period: after {split_date.date()}  ({len(test_df)} rows, "
      f"{y_test.sum()} positive labels)\n")

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

pred_proba = model.predict_proba(X_test)[:, 1]
pred_label = model.predict(X_test)

print("=== Held-out (future time period) evaluation ===")
print(classification_report(y_test, pred_label, digits=3))
try:
    auc = roc_auc_score(y_test, pred_proba)
    print(f"ROC-AUC: {auc:.3f}")
except ValueError:
    print("ROC-AUC could not be computed (only one class present in test set).")

# feature importances
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop feature importances:")
print(importances)

# ---------------------------------------------------------------
# 6. Score ALL products over time (for visualization) and save results
# ---------------------------------------------------------------
labeled["risk_score"] = model.predict_proba(labeled[feature_cols])[:, 1]
labeled.to_csv(f"{OUT_DIR}/risk_scores_full.csv", index=False)

# ---------------------------------------------------------------
# 7. Plot: risk score over time for the embedded escalation products
# ---------------------------------------------------------------
risky_products_known = ["DEV-002", "DEV-006", "DEV-009"]  # matches generator
fig, axes = plt.subplots(len(risky_products_known), 1, figsize=(11, 9), sharex=True)

for ax, pid in zip(axes, risky_products_known):
    sub = labeled[labeled["product_id"] == pid].sort_values("date")
    ax.plot(sub["date"], sub["risk_score"], color="crimson", label="Predicted risk score")
    ax2 = ax.twinx()
    ax2.bar(sub["date"], sub["complaint_count"], color="steelblue", alpha=0.3, width=1, label="Daily complaints")
    ax.set_title(f"{pid}: Predicted Risk Score vs. Complaint Volume Over Time")
    ax.set_ylabel("Risk score (0-1)")
    ax2.set_ylabel("Daily complaints")
    ax.set_ylim(0, 1)

axes[-1].set_xlabel("Date")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/risk_score_timeline.png", dpi=150)
print(f"\nSaved timeline chart to {OUT_DIR}/risk_score_timeline.png")

# ---------------------------------------------------------------
# 8. Plot: feature importance bar chart
# ---------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax, color="teal")
ax.invert_yaxis()
ax.set_title("Feature Importance: What Drives the Risk Prediction")
ax.set_xlabel("Importance")
fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
print(f"Saved feature importance chart to {OUT_DIR}/feature_importance.png")

print("\nDone. This is a PROTOTYPE using synthetic data only — see README.md for limitations.")
