# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "1"
# dependencies = [
#   "mlflow",
# ]
# ///
import time
import uuid
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EXPERIMENT_NAME = "/Shared/credit-default-mlops"
INPUT_TABLE = "workspace.default.credit_default_test"
OUTPUT_TABLE = "credit_default_ab_test_logs"

MODEL_A_NAME = "random_forest"
MODEL_B_NAME = "gradient_boosting"

RANDOM_STATE = 42
ROUTING_SPLIT = 0.50

mlflow.set_experiment(EXPERIMENT_NAME)

test_pdf = spark.table(INPUT_TABLE).toPandas()

if test_pdf.empty:
    raise ValueError(f"No rows found in input table: {INPUT_TABLE}")

print(test_pdf.shape)
test_pdf.head()

id_col = "client_id"
target_col = "default_next_month"

if id_col not in test_pdf.columns:
    raise ValueError(f"Missing required id column: {id_col}")

if target_col not in test_pdf.columns:
    raise ValueError(f"Missing required target column: {target_col}")

runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT_NAME],
    output_format="pandas",
)

if runs.empty:
    raise ValueError("No MLflow runs found.")

required_cols = ["run_id", "params.model_name", "metrics.val_auc"]
missing_cols = [c for c in required_cols if c not in runs.columns]
if missing_cols:
    raise ValueError(f"Missing required MLflow columns: {missing_cols}")

runs = runs.dropna(subset=["params.model_name", "metrics.val_auc"]).copy()

def get_best_run_id_for_model(model_name: str) -> str:
    model_runs = runs[runs["params.model_name"] == model_name].copy()
    if model_runs.empty:
        raise ValueError(f"No runs found for model_name={model_name}")
    best_row = model_runs.sort_values("metrics.val_auc", ascending=False).iloc[0]
    return best_row["run_id"]

run_id_a = get_best_run_id_for_model(MODEL_A_NAME)
run_id_b = get_best_run_id_for_model(MODEL_B_NAME)

print(f"Model A ({MODEL_A_NAME}) run_id: {run_id_a}")
print(f"Model B ({MODEL_B_NAME}) run_id: {run_id_b}")

model_a_uri = f"runs:/{run_id_a}/model"
model_b_uri = f"runs:/{run_id_b}/model"

model_a = mlflow.sklearn.load_model(model_a_uri)
model_b = mlflow.sklearn.load_model(model_b_uri)

feature_pdf = test_pdf.drop(columns=[target_col], errors="ignore")
customer_ids = feature_pdf[id_col].copy()
X = feature_pdf.drop(columns=[id_col], errors="ignore").copy()
y_true = test_pdf[target_col].astype(int).copy()

print(X.shape)
X.head()

rng = np.random.default_rng(RANDOM_STATE)

routing = np.where(
    rng.random(len(X)) < ROUTING_SPLIT,
    "A",
    "B",
)

routing_model_name = np.where(
    routing == "A",
    MODEL_A_NAME,
    MODEL_B_NAME,
)

records = []

for idx in range(len(X)):
    request_id = str(uuid.uuid4())
    customer_id = customer_ids.iloc[idx]
    x_row = X.iloc[[idx]]
    y_actual = int(y_true.iloc[idx])

    routed_group = routing[idx]
    routed_model = routing_model_name[idx]

    model = model_a if routed_group == "A" else model_b

    start = time.perf_counter()
    pred = int(model.predict(x_row)[0])
    prob = float(model.predict_proba(x_row)[0, 1])
    latency_ms = (time.perf_counter() - start) * 1000.0

    records.append(
        {
            "request_id": request_id,
            "client_id": customer_id,
            "routed_group": routed_group,
            "routed_model": routed_model,
            "prediction": pred,
            "default_probability": prob,
            "actual_label": y_actual,
            "latency_ms": float(latency_ms),
            "approval_decision": int(prob < 0.50),  
            "scored_timestamp": datetime.now(timezone.utc),
        }
    )

ab_pdf = pd.DataFrame(records)
ab_pdf.head()

ab_sdf = spark.createDataFrame(ab_pdf)
ab_sdf.write.format("delta").mode("overwrite").saveAsTable(OUTPUT_TABLE)

print(f"Saved A/B logs to {OUTPUT_TABLE}")

summary_rows = []

for model_name, group_df in ab_pdf.groupby("routed_model"):
    auc = roc_auc_score(group_df["actual_label"], group_df["default_probability"])
    avg_predicted_risk = group_df["default_probability"].mean()
    approval_rate = group_df["approval_decision"].mean()
    avg_latency_ms = group_df["latency_ms"].mean()

    summary_rows.append(
        {
            "routed_model": model_name,
            "num_requests": int(len(group_df)),
            "auc": float(auc),
            "approval_rate": float(approval_rate),
            "avg_predicted_risk": float(avg_predicted_risk),
            "avg_latency_ms": float(avg_latency_ms),
        }
    )

summary_pdf = pd.DataFrame(summary_rows).sort_values("auc", ascending=False)
summary_pdf
