import pandas as pd
from src.features import build_features # Import the feature engineering pipeline from the project module

# Define source (silver) and destination (feature) table names
SILVER_TABLE = "credit_default_silver"
FEATURE_TABLE = "credit_default_features"

# Read the cleaned dataset
silver_spark_df = spark.table(SILVER_TABLE)

# Convert Spark DataFrame to Pandas for feature engineering
silver_pdf = silver_spark_df.toPandas()

# Generate new engineered features using the feature pipeline
feature_pdf = build_features(silver_pdf, encode=False)  

# Define the final feature set that will be used for model training
final_cols = [
    "client_id",
    "default_next_month",
    "credit_limit",
    "age",
    "avg_delay_6m",
    "max_delay_6m",
    "late_payment_count_6m",
    "recent_delay_trend",
    "avg_bill_amt_6m",
    "max_bill_amt_6m",
    "bill_volatility_6m",
    "bill_growth_rate",
    "avg_pay_amt_6m",
    "payment_volatility_6m",
    "total_paid_6m",
    "utilization_latest",
    "avg_utilization_6m",
    "pay_to_bill_ratio_sep",
    "avg_pay_to_bill_ratio_6m",
    "age_bucket",
    "limit_bal_x_late_payment_count",
]

feature_pdf = feature_pdf[final_cols].copy()

# Convert the Pandas DataFrame back to a Spark DataFrame
feature_spark_df = spark.createDataFrame(feature_pdf)

# Write the feature dataset to a Delta table
feature_spark_df.write.format("delta").mode("overwrite").saveAsTable(FEATURE_TABLE)

print(f"Saved feature table: {FEATURE_TABLE}")

# Ensure client_id cannot be NULL in the feature table
spark.sql(f"""
ALTER TABLE {FEATURE_TABLE}
ALTER COLUMN client_id SET NOT NULL
""")

print("Feature table created successfully.")