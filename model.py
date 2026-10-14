import json
import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------- Configuration ----------
# Default: C:\Users\<your-user>\Downloads\AssessmentDataScientist.xlsx
# Or set environment variable ASSESSMENT_XLSX to your workbook path.
XLSX_PATH = Path(os.environ.get(
    "ASSESSMENT_XLSX",
    str(Path.home() / "Downloads" / "AssessmentDataScientist.xlsx")
))
OUT = Path(__file__).resolve().parent
N_SPLITS = 5
THRESHOLD = 0.50

if not XLSX_PATH.exists():
    raise FileNotFoundError(
        f"Workbook not found: {XLSX_PATH}\n"
        "Edit XLSX_PATH or set the ASSESSMENT_XLSX environment variable."
    )

# ---------- Load data ----------
apps = pd.read_excel(XLSX_PATH, sheet_name="Application Data")
loans = pd.read_excel(XLSX_PATH, sheet_name="Loan Performance")

for col in ("customer_id",):
    if col not in apps.columns:
        raise ValueError(f"Application Data is missing required column: {col}")
for col in ("idLoan", "flgGood"):
    if col not in loans.columns:
        raise ValueError(f"Loan Performance is missing required column: {col}")

apps["customer_id"] = apps["customer_id"].astype("string").str.strip()
loans["idLoan"] = loans["idLoan"].astype("string").str.strip()

# Assumption based on the observed ID pattern: loan IDs end in -01, -02, etc.
# Verify this rule against the official data dictionary before relying on results.
loans["cid"] = loans["idLoan"].str.replace(r"-\d{2}$", "", regex=True).str.casefold()
apps["_join_cid"] = apps["customer_id"].str.casefold()
loans = loans[loans["idLoan"].str.match(r".+-\d{2}$", na=False)].copy()

# Avoid accidental many-to-many expansion where possible; preserve all matching
# rows, but report counts so the join assumption can be checked.
d = apps.merge(
    loans, left_on="_join_cid", right_on="cid", how="inner",
    suffixes=("_app", "_loan")
)

# Target mapping: Bad=1, Good=0. Confirm this matches the workbook's label definition.
target = d["flgGood"]
if pd.api.types.is_numeric_dtype(target):
    vals = set(pd.to_numeric(target, errors="coerce").dropna().unique())
    if vals.issubset({0, 1}) and vals:
        # Original script treats flgGood==1 as Good, so Bad is 0.
        d["bad"] = (pd.to_numeric(target, errors="coerce") == 0).astype(int)
    else:
        raise ValueError(f"Unexpected numeric flgGood values: {sorted(vals)}")
else:
    labels = target.astype("string").str.strip().str.casefold()
    unknown = set(labels.dropna().unique()) - {"good", "bad"}
    if unknown:
        raise ValueError(f"Unexpected flgGood labels: {sorted(unknown)}")
    d["bad"] = labels.map({"good": 0, "bad": 1})

d = d.dropna(subset=["bad"]).copy()
d["bad"] = d["bad"].astype(int)
if d["bad"].nunique() != 2:
    raise ValueError("Both Good and Bad outcomes are needed for classification metrics.")

# ---------- Feature engineering ----------
def num(col):
    if col in d.columns:
        return pd.to_numeric(d[col], errors="coerce")
    return pd.Series(np.nan, index=d.index, dtype=float)

if {"application_when", "birth_date"}.issubset(d.columns):
    d["application_when"] = pd.to_datetime(d["application_when"], errors="coerce")
    d["birth_date"] = pd.to_datetime(d["birth_date"], errors="coerce")
    d["age"] = (d["application_when"] - d["birth_date"]).dt.days / 365.25
else:
    d["age"] = np.nan

d["income"] = num("monthly_income_amount")
d["amount_requested"] = num("amount_requested")
d["monthly_rent_amount"] = num("monthly_rent_amount")
d["payment_amount"] = num("payment_amount")
d["loan_duration"] = num("loan_duration")
d["num_payments"] = num("num_payments")

income_positive = d["income"].where(d["income"] > 0)
d["rent_ratio"] = d["monthly_rent_amount"] / income_positive
d["pay_ratio"] = d["payment_amount"] / income_positive
d["req_ratio"] = d["amount_requested"] / income_positive

def text_series(col):
    if col in d.columns:
        return d[col].astype("string").str.strip().str.casefold()
    return pd.Series("", index=d.index, dtype="string")

d["own"] = text_series("residence_rent_or_own").isin(
    {"own", "owned", "homeowner"}
).astype(int)
d["dd"] = text_series("bank_account_direct_deposit").isin(
    {"1", "yes", "true", "y"}
).astype(int)
d["ach"] = text_series("payment_ach").isin(
    {"1", "yes", "true", "y"}
).astype(int)

email_map = {"3 months or less": 0, "4-12 months": 1, "1 year or more": 2}
res_map = {"6 months or less": 0, "7-12 months": 1, "1-2 years": 2, "3+ years": 3}
d["email_dur"] = d["email_duration"].map(email_map) if "email_duration" in d else np.nan
d["res_dur"] = d["residence_duration"].map(res_map) if "residence_duration" in d else np.nan
d["bank_dur"] = d["bank_account_duration"].map(res_map) if "bank_account_duration" in d else np.nan
d["freq_weekly"] = text_series("payment_frequency").eq("weekly").astype(int)
d["freq_biweekly"] = text_series("payment_frequency").eq("bi-weekly").astype(int)
d["has_other_phone"] = d["other_phone_type"].notna().astype(int) if "other_phone_type" in d else 0
d["mobile_home"] = text_series("home_phone_type").eq("mobile").astype(int)

if "application_when" in d.columns:
    d["dow"] = d["application_when"].dt.dayofweek
    d["hour"] = d["application_when"].dt.hour
else:
    d["dow"] = np.nan
    d["hour"] = np.nan

score_candidates = [
    "raw_l2c_score", "raw_FICO_telecom", "raw_FICO_retail",
    "raw_FICO_bank_card", "raw_FICO_money"
]
score_cols = [c for c in score_candidates if c in d.columns]
base_candidates = [
    "amount_requested", "age", "income", "monthly_rent_amount",
    "rent_ratio", "pay_ratio", "req_ratio", "own", "dd", "ach",
    "loan_duration", "num_payments", "payment_amount", "email_dur",
    "res_dur", "bank_dur", "freq_weekly", "freq_biweekly",
    "has_other_phone", "mobile_home", "dow", "hour"
]
base_cols = [c for c in base_candidates if c in d.columns]
feature_sets = {
    "Scores only": score_cols,
    "Application only": base_cols,
    "All features": base_cols + score_cols
}
feature_sets = {name: cols for name, cols in feature_sets.items() if cols}
if not feature_sets:
    raise ValueError("No usable predictor columns were found.")

# Note: approved/offer outcome fields are intentionally excluded to reduce leakage risk.
# Review all remaining fields for whether they are available at prediction time.

models = {
    "Logistic": make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000)
    ),
    "GBM": GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.05,
        subsample=0.7, random_state=1
    )
}

# ---------- Cross-validated evaluation ----------
y = d["bad"].to_numpy()
min_class_count = int(pd.Series(y).value_counts().min())
fold_count = min(N_SPLITS, min_class_count)
if fold_count < 2:
    raise ValueError("Too few examples in one class for stratified cross-validation.")

cv = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=42)
results = {}
predictions = {}

for set_name, cols in feature_sets.items():
    X = d[cols].apply(pd.to_numeric, errors="coerce")
    for model_name, estimator in models.items():
        oof_probability = np.full(len(d), np.nan, dtype=float)

        # Median imputation is learned from each training fold only.
        for train_idx, valid_idx in cv.split(X, y):
            X_train = X.iloc[train_idx].copy()
            X_valid = X.iloc[valid_idx].copy()
            medians = X_train.median(numeric_only=True).fillna(0)
            X_train = X_train.fillna(medians).fillna(0)
            X_valid = X_valid.fillna(medians).fillna(0)

            fitted = clone(estimator)
            fitted.fit(X_train, y[train_idx])
            oof_probability[valid_idx] = fitted.predict_proba(X_valid)[:, 1]

        if np.isnan(oof_probability).any():
            raise RuntimeError(f"Missing out-of-fold predictions: {set_name}/{model_name}")

        # Threshold-based metrics; positive class is Bad (1).
        y_pred = (oof_probability >= THRESHOLD).astype(int)
        key = f"{set_name} / {model_name}"
        results[key] = {
            "roc_auc": float(roc_auc_score(y, oof_probability)),
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision_bad_class": float(precision_score(y, y_pred, pos_label=1, zero_division=0)),
            "recall_bad_class": float(recall_score(y, y_pred, pos_label=1, zero_division=0)),
            "threshold": THRESHOLD,
            "positive_class": "Bad",
            "n": int(len(y))
        }
        predictions[key] = oof_probability
        r = results[key]
        print(
            f"{key}: AUC={r['roc_auc']:.3f}, Accuracy={r['accuracy']:.3f}, "
            f"Precision(Bad)={r['precision_bad_class']:.3f}, "
            f"Recall(Bad)={r['recall_bad_class']:.3f}"
        )

primary_key = "All features / Logistic" if "All features / Logistic" in predictions else next(iter(predictions))
d["p"] = predictions[primary_key]
d["dec"] = pd.qcut(d["p"].rank(method="first"), q=5, labels=False, duplicates="drop")

# ---------- Save dashboard artifacts ----------
with open(OUT / "res.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
d.to_pickle(OUT / "d.pkl")

all_cols = feature_sets.get("All features", feature_sets.get("Application only", next(iter(feature_sets.values()))))
X_all = d[all_cols].apply(pd.to_numeric, errors="coerce")
medians_all = X_all.median(numeric_only=True).fillna(0)
X_all = X_all.fillna(medians_all).fillna(0)
coef_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.1, max_iter=2000)
).fit(X_all, y)
co = pd.Series(coef_model[-1].coef_[0], index=all_cols).sort_values()
co.to_pickle(OUT / "co.pkl")

# ROC chart from the same out-of-fold predictions used for metrics.
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(7, 4.2))
fpr, tpr, _ = roc_curve(y, d["p"])
auc = roc_auc_score(y, d["p"])
ax.plot(fpr, tpr, color="#1f4e79", linewidth=2.5, label=f"OOF model (AUC {auc:.2f})")
ax.plot([0, 1], [0, 1], "--", color="grey", label="Random baseline")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC Curve (out-of-fold predictions)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT / "roc.png", dpi=160)
plt.close()

# Observed bad rate by predicted-risk quintile.
fig, ax = plt.subplots(figsize=(7, 4.2))
groups = d.groupby("dec", observed=False)["bad"].agg(["mean", "count"]).dropna()
labels = [f"Q{i+1}" for i in range(len(groups))]
rates = groups["mean"].to_numpy() * 100
colors = ["#2e7d32", "#7cb342", "#fbc02d", "#f57c00", "#c62828"][:len(labels)]
ax.bar(labels, rates, color=colors)
ax.axhline(d["bad"].mean() * 100, linestyle="--", color="grey", label="Overall bad rate")
ax.set_ylabel("Observed bad-loan rate (%)")
ax.set_xlabel("Predicted-risk quintile (Q1 = lowest predicted risk)")
ax.set_title("Observed Bad Rate by Predicted-Risk Quintile")
for i, value in enumerate(rates):
    ax.text(i, value + 0.8, f"{value:.0f}%", ha="center")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT / "quint.png", dpi=160)
plt.close()

# Coefficient chart uses standardized predictors.
fig, ax = plt.subplots(figsize=(7, 4.2))
top = co.reindex(co.abs().sort_values().tail(10).index)
bar_colors = ["#c62828" if value > 0 else "#2e7d32" for value in top.values]
ax.barh(top.index, top.values, color=bar_colors)
ax.set_xlabel("Standardized logistic coefficient (+ = higher odds of Bad)")
ax.set_title("Largest Absolute Logistic Regression Coefficients")
plt.tight_layout()
plt.savefig(OUT / "coef.png", dpi=160)
plt.close()

pd.DataFrame(results).T.to_csv(OUT / "model_metrics.csv")
print(f"\nPrimary dashboard model: {primary_key}")
print(f"Matched rows used for modelling: {len(d)}")
print(f"Bad outcome rate in matched modelling rows: {d['bad'].mean():.3f}")
print(f"Saved outputs to: {OUT}")
print("These are cross-validated exploratory metrics, not proof of production readiness.")
