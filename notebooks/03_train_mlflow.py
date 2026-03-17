# Databricks notebook source
import mlflow
import mlflow.sklearn
from pathlib import Path
import tempfile
import sys

bundle_root = Path.cwd().parent
if str(bundle_root) not in sys.path:
    sys.path.insert(0, str(bundle_root))

from src.train import (
    split_dataset,
    prepare_xy,
    get_models,
    fit_pipeline,
    evaluate_model,
    save_confusion_matrix_artifact,
    save_feature_importance_artifact,
)

from src.evaluate import find_best_threshold_by_recall_constraint, apply_threshold
from sklearn.metrics import precision_score, recall_score, f1_score
from mlflow.models import infer_signature

FEATURE_TABLE = "credit_default_features"
TRAIN_TABLE = "credit_default_train"
VAL_TABLE = "credit_default_val"
TEST_TABLE = "credit_default_test"

EXPERIMENT_NAME = "/Shared/credit-default-mlops"

mlflow.set_experiment(EXPERIMENT_NAME)

feature_sdf = spark.table(FEATURE_TABLE)
feature_pdf = feature_sdf.toPandas()

train_df, val_df, test_df = split_dataset(feature_pdf, target_col="default_next_month")

spark.createDataFrame(train_df).write.format("delta").mode("overwrite").saveAsTable(
    TRAIN_TABLE
)
spark.createDataFrame(val_df).write.format("delta").mode("overwrite").saveAsTable(
    VAL_TABLE
)
spark.createDataFrame(test_df).write.format("delta").mode("overwrite").saveAsTable(
    TEST_TABLE
)

X_train, y_train = prepare_xy(train_df)
X_val, y_val = prepare_xy(val_df)
X_test, y_test = prepare_xy(test_df)

models = get_models()

# best_run_name = None
# best_val_auc = -1.0

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        pipeline = fit_pipeline(model, X_train, y_train)

        val_metrics = evaluate_model(pipeline, X_val, y_val)
        test_metrics = evaluate_model(pipeline, X_test, y_test)

        val_prob = pipeline.predict_proba(X_val)[:, 1]
        threshold_result = find_best_threshold_by_recall_constraint(
            y_true=y_val,
            y_prob=val_prob,
            min_recall=0.70,
        )
        chosen_threshold = float(threshold_result["threshold"])

        test_prob = pipeline.predict_proba(X_test)[:, 1]
        test_pred = apply_threshold(test_prob, chosen_threshold)

        test_precision = precision_score(y_test, test_pred, zero_division=0)
        test_recall = recall_score(y_test, test_pred, zero_division=0)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("selection_rule", "max_val_auc_then_val_recall_then_threshold")
        mlflow.log_param("target_col", "default_next_month")
        mlflow.log_param("threshold_selection_min_recall", 0.70)
        mlflow.log_param("decision_threshold", chosen_threshold)

        if hasattr(model, "get_params"):
            mlflow.log_params({k: str(v) for k, v in model.get_params().items()})

        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metric(
            "val_threshold_precision", float(threshold_result["precision"])
        )
        mlflow.log_metric("val_threshold_recall", float(threshold_result["recall"]))
        mlflow.log_metric("val_threshold_f1", float(threshold_result["f1"]))

        mlflow.log_metric("test_precision_at_threshold", float(test_precision))
        mlflow.log_metric("test_recall_at_threshold", float(test_recall))
        mlflow.log_metric("test_f1_at_threshold", float(test_f1))

        # mlflow.log_param("train_rows", len(train_df))
        # mlflow.log_param("val_rows", len(val_df))
        # mlflow.log_param("test_rows", len(test_df))

        # if hasattr(model, "get_params"):
        #     params = model.get_params()
        #     safe_params = {k: str(v) for k, v in params.items()}
        #     mlflow.log_params(safe_params)

        # mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        # mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            cm_path = save_confusion_matrix_artifact(pipeline, X_test, y_test, tmp_path)
            fi_path = save_feature_importance_artifact(pipeline, X_train, tmp_path)

            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")
            mlflow.log_artifact(str(fi_path), artifact_path="evaluation")

        input_example = X_train.head(5)
        pred_example = pipeline.predict(input_example)

        signature = infer_signature(input_example, pred_example)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
