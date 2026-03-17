# Databricks notebook source
import mlflow
import mlflow.pyfunc
import pandas as pd
from pyspark.sql import functions as F

REGISTERED_MODEL_NAME = "workspace.default.credit_default_model"
INPUT_TABLE = "workspace.default.credit_default_test"
OUTPUT_TABLE = "workspace.default.credit_default_predictions_batch"

mlflow.set_registry_uri("databricks-uc")

client = mlflow.MlflowClient()

latest_versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")

if not latest_versions:
    raise ValueError(f"No registered versions found for model: {REGISTERED_MODEL_NAME}")

latest_model_version = max(latest_versions, key=lambda mv: int(mv.version))
model_version = latest_model_version.version

model_uri = f"models:/{REGISTERED_MODEL_NAME}/{model_version}"
print(f"Using model URI: {model_uri}")

model = mlflow.pyfunc.load_model(model_uri)

test_sdf = spark.table(INPUT_TABLE)
test_pdf = test_sdf.toPandas()

print(test_pdf.shape)
test_pdf.head()

id_col = "client_id"
target_col = "default_next_month"

feature_pdf = test_pdf.drop(columns=[target_col], errors="ignore")

ids = feature_pdf[id_col].copy()

feature_input_pdf = feature_pdf.drop(columns=[id_col], errors="ignore")

predictions = model.predict(feature_input_pdf)

sk_model = mlflow.sklearn.load_model(model_uri)
default_probabilities = sk_model.predict_proba(feature_input_pdf)[:, 1]

results_pdf = pd.DataFrame(
    {
        "client_id": ids,
        "prediction": predictions.astype(int),
        "default_probability": default_probabilities.astype(float),
        "model_version": str(model_version),
    }
)

results_sdf = spark.createDataFrame(results_pdf).withColumn(
    "scored_timestamp", F.current_timestamp()
)

display(results_sdf.limit(10))

results_sdf.write.format("delta").mode("overwrite").saveAsTable(OUTPUT_TABLE)

print(f"Saved batch predictions to {OUTPUT_TABLE}")
