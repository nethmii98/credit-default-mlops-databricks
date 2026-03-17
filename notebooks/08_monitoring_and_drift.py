# Databricks notebook source
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, recall_score

TRAIN_TABLE = "workspace.default.credit_default_train"
BATCH_PRED_TABLE = "credit_default_predictions_batch"
AB_LOG_TABLE = "credit_default_ab_test_logs"

MANUAL_INFERENCE_LOG_TABLE = "credit_default_inference_logs"

MONITORING_SUMMARY_TABLE = "credit_default_monitoring_summary"
FEATURE_DRIFT_TABLE = "credit_default_feature_drift"
PERFORMANCE_TABLE = "credit_default_performance_monitoring"

HIGH_RISK_THRESHOLD = 0.50

FEATURE_COLS = [
    "avg_delay_6m",
    "avg_utilization_6m",
    "credit_limit",
    "age",
]


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """
    Population Stability Index (PSI)
    """
    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)

    if expected.empty or actual.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.unique(np.quantile(expected, quantiles))

    if len(breakpoints) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = np.where(
        expected_counts == 0, 1e-6, expected_counts / expected_counts.sum()
    )
    actual_pct = np.where(actual_counts == 0, 1e-6, actual_counts / actual_counts.sum())

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def summarize_shift(train_s: pd.Series, recent_s: pd.Series) -> dict:
    train_mean = float(train_s.mean())
    recent_mean = float(recent_s.mean())

    train_std = float(train_s.std(ddof=0))
    recent_std = float(recent_s.std(ddof=0))

    return {
        "train_mean": train_mean,
        "recent_mean": recent_mean,
        "mean_shift": recent_mean - train_mean,
        "train_std": train_std,
        "recent_std": recent_std,
        "std_shift": recent_std - train_std,
        "psi": compute_psi(train_s, recent_s),
    }


train_pdf = spark.table(TRAIN_TABLE).toPandas()

if train_pdf.empty:
    raise ValueError(f"No rows found in {TRAIN_TABLE}")

print(train_pdf.shape)

existing_tables = {row.tableName for row in spark.sql("SHOW TABLES").collect()}

if AB_LOG_TABLE in existing_tables:
    inference_source = AB_LOG_TABLE
elif BATCH_PRED_TABLE in existing_tables:
    inference_source = BATCH_PRED_TABLE
elif MANUAL_INFERENCE_LOG_TABLE in existing_tables:
    inference_source = MANUAL_INFERENCE_LOG_TABLE
else:
    raise ValueError("No inference or prediction log table found.")

print(f"Using inference source: {inference_source}")

inference_pdf = spark.table(inference_source).toPandas()

if inference_pdf.empty:
    raise ValueError(f"No rows found in {inference_source}")

print(inference_pdf.shape)

pred_prob_col = (
    "default_probability" if "default_probability" in inference_pdf.columns else None
)
if pred_prob_col is None:
    raise ValueError(
        "Expected column 'default_probability' not found in inference data."
    )

model_version_col = None
for candidate in ["model_version", "routed_model"]:
    if candidate in inference_pdf.columns:
        model_version_col = candidate
        break

if model_version_col is None:
    model_version_col = "model_group_fallback"
    inference_pdf[model_version_col] = "unknown"

prediction_summary = {
    "monitoring_timestamp": datetime.now(timezone.utc),
    "inference_source": inference_source,
    "num_predictions": int(len(inference_pdf)),
    "avg_predicted_default_probability": float(inference_pdf[pred_prob_col].mean()),
    "high_risk_proportion": float(
        (inference_pdf[pred_prob_col] >= HIGH_RISK_THRESHOLD).mean()
    ),
    "num_model_groups": int(inference_pdf[model_version_col].nunique()),
}

prediction_summary_pdf = pd.DataFrame([prediction_summary])
prediction_summary_pdf

# counts by model version / model name
counts_by_model_pdf = (
    inference_pdf.groupby(model_version_col)
    .size()
    .reset_index(name="request_count")
    .sort_values("request_count", ascending=False)
)

counts_by_model_pdf

missing_feature_cols = [c for c in FEATURE_COLS if c not in inference_pdf.columns]
if missing_feature_cols:
    print(
        f"Warning: missing feature columns in inference source: {missing_feature_cols}"
    )
    print("Skipping drift for those columns.")

drift_rows = []
for col in FEATURE_COLS:
    if col not in train_pdf.columns or col not in inference_pdf.columns:
        continue

    train_s = train_pdf[col]
    recent_s = inference_pdf[col]

    drift_stats = summarize_shift(train_s, recent_s)
    drift_rows.append(
        {
            "feature_name": col,
            "monitoring_timestamp": datetime.now(timezone.utc),
            **drift_stats,
        }
    )

feature_drift_pdf = pd.DataFrame(drift_rows)
feature_drift_pdf

performance_rows = []

label_col = None
for candidate in ["actual_label", "default_next_month"]:
    if candidate in inference_pdf.columns:
        label_col = candidate
        break

if label_col is not None:
    for model_key, group_df in inference_pdf.groupby(model_version_col):
        if group_df[label_col].nunique() < 2:
            continue

        auc = roc_auc_score(group_df[label_col], group_df[pred_prob_col])

        pred_binary = (group_df[pred_prob_col] >= HIGH_RISK_THRESHOLD).astype(int)
        recall = recall_score(group_df[label_col], pred_binary, zero_division=0)
        default_rate = float(group_df[label_col].mean())

        performance_rows.append(
            {
                "monitoring_timestamp": datetime.now(timezone.utc),
                "model_group": str(model_key),
                "rolling_auc": float(auc),
                "rolling_recall": float(recall),
                "rolling_default_rate": float(default_rate),
                "window_size": int(len(group_df)),
            }
        )

performance_pdf = pd.DataFrame(performance_rows)
performance_pdf

spark.createDataFrame(prediction_summary_pdf).write.format("delta").mode(
    "append"
).saveAsTable(MONITORING_SUMMARY_TABLE)

if not counts_by_model_pdf.empty:
    spark.createDataFrame(counts_by_model_pdf).write.format("delta").mode(
        "overwrite"
    ).saveAsTable("credit_default_counts_by_model")

if not feature_drift_pdf.empty:
    spark.createDataFrame(feature_drift_pdf).write.format("delta").mode(
        "append"
    ).saveAsTable(FEATURE_DRIFT_TABLE)

if not performance_pdf.empty:
    spark.createDataFrame(performance_pdf).write.format("delta").mode(
        "append"
    ).saveAsTable(PERFORMANCE_TABLE)

print("Monitoring outputs saved.")
