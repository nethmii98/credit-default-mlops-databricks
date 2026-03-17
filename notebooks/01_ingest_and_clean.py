# Databricks notebook source
# Import required libraries
import pandas as pd
from pyspark.sql import functions as F

# Build the full path to the dataset in the "data" folder
data_path = "/Volumes/workspace/default/credit_card_data/UCI_Credit_Card.csv"


# Load the dataset into a pandas DataFrame
df = pd.read_csv(data_path)


# Save the raw data in a delta table
df_raw = spark.createDataFrame(df).withColumn("ingestion_ts", F.current_timestamp())
df_raw.write.format("delta").mode("append").saveAsTable("credit_default_bronze")

df_silver = spark.table("credit_default_bronze").drop("ingestion_ts")

# Rename columns
column_mapping = {
    "ID": "client_id",
    "LIMIT_BAL": "credit_limit",
    "SEX": "gender",
    "EDUCATION": "education_level",
    "MARRIAGE": "marital_status",
    "AGE": "age",
    "PAY_0": "repayment_status_sep",
    "PAY_2": "repayment_status_aug",
    "PAY_3": "repayment_status_jul",
    "PAY_4": "repayment_status_jun",
    "PAY_5": "repayment_status_may",
    "PAY_6": "repayment_status_apr",
    "BILL_AMT1": "bill_amount_sep",
    "BILL_AMT2": "bill_amount_aug",
    "BILL_AMT3": "bill_amount_jul",
    "BILL_AMT4": "bill_amount_jun",
    "BILL_AMT5": "bill_amount_may",
    "BILL_AMT6": "bill_amount_apr",
    "PAY_AMT1": "payment_amount_sep",
    "PAY_AMT2": "payment_amount_aug",
    "PAY_AMT3": "payment_amount_jul",
    "PAY_AMT4": "payment_amount_jun",
    "PAY_AMT5": "payment_amount_may",
    "PAY_AMT6": "payment_amount_apr",
    "default.payment.next.month": "default_next_month",
}

for old_col, new_col in column_mapping.items():
    df_silver = df_silver.withColumnRenamed(old_col, new_col)

# SCHEMA VALIDATION
# Check that all expected columns are present and in the correct order.
expected_columns = [
    "client_id",
    "credit_limit",
    "gender",
    "education_level",
    "marital_status",
    "age",
    "repayment_status_sep",
    "repayment_status_aug",
    "repayment_status_jul",
    "repayment_status_jun",
    "repayment_status_may",
    "repayment_status_apr",
    "bill_amount_sep",
    "bill_amount_aug",
    "bill_amount_jul",
    "bill_amount_jun",
    "bill_amount_may",
    "bill_amount_apr",
    "payment_amount_sep",
    "payment_amount_aug",
    "payment_amount_jul",
    "payment_amount_jun",
    "payment_amount_may",
    "payment_amount_apr",
    "default_next_month",
]

if list(df_silver.columns) != expected_columns:
    missing_cols = set(expected_columns) - set(df_silver.columns)
    extra_cols = set(df_silver.columns) - set(expected_columns)
    raise ValueError(
        f"Schema validation failed. Missing columns: {missing_cols}. Extra columns: {extra_cols}."
    )

# Check expected data types at a high level
expected_dtypes = {
    "client_id": {"int", "bigint"},
    "credit_limit": {"int", "bigint", "float", "double", "decimal"},
    "gender": {"int", "bigint"},
    "education_level": {"int", "bigint"},
    "marital_status": {"int", "bigint"},
    "age": {"int", "bigint"},
    "repayment_status_sep": {"int", "bigint"},
    "repayment_status_aug": {"int", "bigint"},
    "repayment_status_jul": {"int", "bigint"},
    "repayment_status_jun": {"int", "bigint"},
    "repayment_status_may": {"int", "bigint"},
    "repayment_status_apr": {"int", "bigint"},
    "bill_amount_sep": {"int", "bigint", "float", "double", "decimal"},
    "bill_amount_aug": {"int", "bigint", "float", "double", "decimal"},
    "bill_amount_jul": {"int", "bigint", "float", "double", "decimal"},
    "bill_amount_jun": {"int", "bigint", "float", "double", "decimal"},
    "bill_amount_may": {"int", "bigint", "float", "double", "decimal"},
    "bill_amount_apr": {"int", "bigint", "float", "double", "decimal"},
    "payment_amount_sep": {"int", "bigint", "float", "double", "decimal"},
    "payment_amount_aug": {"int", "bigint", "float", "double", "decimal"},
    "payment_amount_jul": {"int", "bigint", "float", "double", "decimal"},
    "payment_amount_jun": {"int", "bigint", "float", "double", "decimal"},
    "payment_amount_may": {"int", "bigint", "float", "double", "decimal"},
    "payment_amount_apr": {"int", "bigint", "float", "double", "decimal"},
    "default_next_month": {"int", "bigint"},
}

actual_dtypes = dict(df_silver.dtypes)

for col, expected_types in expected_dtypes.items():
    actual_type = actual_dtypes.get(col)

    if actual_type is None:
        raise ValueError(f"Column '{col}' is missing from the DataFrame.")

    # Handle decimal types like decimal(10,0)
    normalized_type = "decimal" if actual_type.startswith("decimal") else actual_type

    if normalized_type not in expected_types:
        raise TypeError(
            f"Column '{col}' has dtype '{actual_type}', expected one of {sorted(expected_types)}."
        )

# MISSING VALUE CHECK
missing_counts_df = df_silver.select(
    [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df_silver.columns
    ]
)

missing_counts = missing_counts_df.collect()[0].asDict()

missing_cols = {
    col_name: count for col_name, count in missing_counts.items() if count > 0
}

if missing_cols:
    raise ValueError(f"Missing values found: {missing_cols}")

# DUPLICATE CHECK
duplicate_client_ids = (
    df_silver.groupBy("client_id").count().filter(F.col("count") > 1).count()
)

if duplicate_client_ids > 0:
    raise ValueError(f"Found {duplicate_client_ids} duplicate client_id values.")

# INCORRECT CATEGORY CHECK
# Define allowed raw category values before cleaning
allowed_gender = {1, 2}
allowed_education = {0, 1, 2, 3, 4, 5, 6}
allowed_marital_status = {0, 1, 2, 3}
allowed_default = {0, 1}

# Repayment status values observed/documented in this dataset
allowed_repayment_status = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

category_rules = {
    "gender": allowed_gender,
    "education_level": allowed_education,
    "marital_status": allowed_marital_status,
    "default_next_month": allowed_default,
    "repayment_status_sep": allowed_repayment_status,
    "repayment_status_aug": allowed_repayment_status,
    "repayment_status_jul": allowed_repayment_status,
    "repayment_status_jun": allowed_repayment_status,
    "repayment_status_may": allowed_repayment_status,
    "repayment_status_apr": allowed_repayment_status,
}

for column_name, allowed_values in category_rules.items():
    actual_values = {
        row[column_name]
        for row in df_silver.select(column_name).distinct().collect()
        if row[column_name] is not None
    }

    invalid_values = actual_values - allowed_values

    if invalid_values:
        raise ValueError(
            f"Column '{column_name}' contains invalid values: {sorted(invalid_values)}"
        )

# CATEGORY CLEANING
# Clean the education_level column
# 0 = undocumented
# 5, 6 = unknown
# These are merged into category 4 ("others")
df_silver = df_silver.withColumn(
    "education_level",
    F.when(F.col("education_level").isin(0, 5, 6), 4).otherwise(
        F.col("education_level")
    ),
)

# clean marital_status undocumented value
df_silver = df_silver.withColumn(
    "marital_status",
    F.when(F.col("marital_status") == 0, 3).otherwise(F.col("marital_status")),
)

# SET CATEGORICAL DTYPES
categorical_cols = ["gender", "education_level", "marital_status"]

for c in categorical_cols:
    df_silver = df_silver.withColumn(c, F.col(c).cast("string"))


# Save the silver table
df_silver.write.format("delta").mode("overwrite").saveAsTable("credit_default_silver")

spark.sql("""
ALTER TABLE credit_default_silver
SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
