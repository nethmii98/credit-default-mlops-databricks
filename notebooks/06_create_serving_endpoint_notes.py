# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "1"
# dependencies = [
#   "mlflow",
# ]
# ///
import json
import mlflow
from mlflow import MlflowClient

REGISTERED_MODEL_NAME = "workspace.default.credit_default_model"
ENDPOINT_NAME = "credit-default-endpoint"

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

model_versions = client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")

if not model_versions:
    raise ValueError(f"No registered model versions found for {REGISTERED_MODEL_NAME}")

latest_version = max(model_versions, key=lambda mv: int(mv.version))
MODEL_VERSION = latest_version.version

print(f"Registered model: {REGISTERED_MODEL_NAME}")
print(f"Selected model version: {MODEL_VERSION}")

serving_notes = {
    "endpoint_name": ENDPOINT_NAME,
    "registered_model_name": REGISTERED_MODEL_NAME,
    "model_version": str(MODEL_VERSION),
    "intended_usage": "Real-time credit default inference",
    "input_pattern": "Full feature payload",
    "output_pattern": "Prediction and/or probability",
    "recommended_status": "Create late in project, test, take screenshots, then stop/delete if not needed",
}

print(json.dumps(serving_notes, indent=2))

sample_payload = {
    "dataframe_records": [
        {
            "credit_limit": 200000,
            "gender": "2",
            "education_level": "2",
            "marital_status": "1",
            "age": 29,
            "repayment_status_sep": 1,
            "repayment_status_aug": 0,
            "repayment_status_jul": 0,
            "repayment_status_jun": 0,
            "repayment_status_may": 0,
            "repayment_status_apr": 0,
            "bill_amount_sep": 12000,
            "bill_amount_aug": 10000,
            "bill_amount_jul": 9000,
            "bill_amount_jun": 8000,
            "bill_amount_may": 7500,
            "bill_amount_apr": 7000,
            "payment_amount_sep": 3000,
            "payment_amount_aug": 2500,
            "payment_amount_jul": 2200,
            "payment_amount_jun": 2000,
            "payment_amount_may": 1800,
            "payment_amount_apr": 1700,
            "avg_delay_6m": 0.1667,
            "max_delay_6m": 1,
            "late_payment_count_6m": 1,
            "recent_delay_trend": 1.0,
            "avg_bill_amt_6m": 8916.67,
            "max_bill_amt_6m": 12000,
            "bill_volatility_6m": 1816.59,
            "bill_growth_rate": 1.56,
            "avg_pay_amt_6m": 2200.0,
            "payment_volatility_6m": 474.34,
            "total_paid_6m": 13200.0,
            "utilization_latest": 0.06,
            "avg_utilization_6m": 0.0446,
            "pay_to_bill_ratio_1": 0.25,
            "avg_pay_to_bill_ratio_6m": 0.258,
            "age_bucket": "26_35",
            "limit_bal_x_late_payment_count": 200000
        }
    ]
}

print("Sample request payload:")
print(json.dumps(sample_payload, indent=2))

sample_response_shape = {
    "predictions": [0]
}

print("Example response shape:")
print(json.dumps(sample_response_shape, indent=2))

manual_steps = [
    "1. Open Databricks > Serving.",
    "2. Click 'Create serving endpoint'.",
    "3. Set endpoint name to: credit-default-endpoint",
    f"4. Select registered model: {REGISTERED_MODEL_NAME}",
    f"5. Select model version: {MODEL_VERSION}",
    "6. Choose a small serving configuration to reduce cost.",
    "7. Wait for the endpoint status to become Ready.",
    "8. Open the Query tab and paste the sample payload from this notebook.",
    "9. Save screenshots for the project README.",
    "10. Stop or delete the endpoint when finished if no longer needed.",
]

for step in manual_steps:
    print(step)


curl_template = f"""
curl -X POST \\
  https://<workspace-url>/serving-endpoints/{ENDPOINT_NAME}/invocations \\
  -H "Authorization: Bearer <databricks-token>" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(sample_payload)}'
"""

print("curl template:")
print(curl_template)

python_requests_template = f'''
import requests
import json

url = "https://<workspace-url>/serving-endpoints/{ENDPOINT_NAME}/invocations"
token = "<databricks-token>"

payload = {json.dumps(sample_payload, indent=4)}

headers = {{
    "Authorization": f"Bearer {{token}}",
    "Content-Type": "application/json",
}}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print(response.status_code)
print(response.text)
'''

print("Python requests template:")
print(python_requests_template)

readme_notes = f"""
Serving endpoint summary
- Endpoint name: {ENDPOINT_NAME}
- Registered model: {REGISTERED_MODEL_NAME}
- Model version: {MODEL_VERSION}
- Input type: full engineered feature payload
- Output type: binary default prediction
- Deployment target: Databricks Model Serving
- Status: create near the end of the project to control trial/workspace cost
"""

print(readme_notes)

cleanup_notes = [
    "After testing the endpoint:",
    "- save screenshots",
    "- save one sample request and response",
    "- stop or delete the endpoint if it is no longer needed",
]

for note in cleanup_notes:
    print(note)
