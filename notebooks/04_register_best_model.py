# Databricks notebook source
import mlflow
from mlflow import MlflowClient

EXPERIMENT_NAME = "/Shared/credit-default-mlops"

REGISTERED_MODEL_NAME = "credit_default_model"

PROJECT_TAG = "credit-default-mlops"
DATASET_TAG = "default-of-credit-card-clients"
STAGE_TAG = "candidate"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient()

exp = client.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    raise ValueError(f"Experiment not found: {EXPERIMENT_NAME}")

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    output_format="pandas",
)

if runs.empty:
    raise ValueError("No MLflow runs found.")

required_cols = [
    "run_id",
    "metrics.val_auc",
    "metrics.val_recall",
    "params.model_name",
]
missing_required = [c for c in required_cols if c not in runs.columns]
if missing_required:
    raise ValueError(f"Missing required columns in MLflow runs: {missing_required}")

candidate_runs = runs.dropna(subset=["metrics.val_auc", "metrics.val_recall"]).copy()

if candidate_runs.empty:
    raise ValueError("No candidate runs with val_auc and val_recall found.")

best_auc = candidate_runs["metrics.val_auc"].max()

close_auc = candidate_runs[
    candidate_runs["metrics.val_auc"] >= (best_auc - 0.01)
].copy()

winner = close_auc.sort_values(
    by=["metrics.val_recall", "metrics.val_auc"],
    ascending=[False, False],
).iloc[0]

best_run_id = winner["run_id"]
best_model_name = winner.get("params.model_name", "unknown_model")
best_val_auc = float(winner["metrics.val_auc"])
best_val_recall = float(winner["metrics.val_recall"])

print("Selected run:")
print(f"run_id={best_run_id}")
print(f"model={best_model_name}")
print(f"val_auc={best_val_auc:.4f}")
print(f"val_recall={best_val_recall:.4f}")

model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=REGISTERED_MODEL_NAME,
)

print(f"Registered model version: {registered_model.version}")

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered_model.version,
    key="project",
    value=PROJECT_TAG,
)

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered_model.version,
    key="dataset",
    value=DATASET_TAG,
)

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered_model.version,
    key="stage",
    value=STAGE_TAG,
)

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered_model.version,
    key="selected_model_name",
    value=str(best_model_name),
)

client.set_model_version_tag(
    name=REGISTERED_MODEL_NAME,
    version=registered_model.version,
    key="selection_rule",
    value="highest_val_auc_then_higher_val_recall_if_within_0.01_auc",
)

print("Model version tags added successfully.")
