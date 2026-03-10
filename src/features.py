from __future__ import annotations

import numpy as np
import pandas as pd

PAY_COLS = ["repayment_status_sep", "repayment_status_aug", "repayment_status_jul", "repayment_status_jun", "repayment_status_may", "repayment_status_apr"]
BILL_COLS = ["bill_amount_sep", "bill_amount_aug", "bill_amount_jul", "bill_amount_jun", "bill_amount_may", "bill_amount_apr"]
PAY_AMT_COLS = ["payment_amount_sep", "payment_amount_aug", "payment_amount_jul", "payment_amount_jun", "payment_amount_may", "payment_amount_apr"]

def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Divide safely and return 0 when denominator is 0 or missing.
    """
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def add_repayment_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    out["avg_delay_6m"] = out[PAY_COLS].mean(axis=1)
    out["max_delay_6m"] = out[PAY_COLS].max(axis=1)
    out["late_payment_count_6m"] = (out[PAY_COLS] > 0).sum(axis=1)

    older_delay_avg = out[["repayment_status_jul", "repayment_status_jun", "repayment_status_may", "repayment_status_apr"]].mean(axis=1)
    out["recent_delay_trend"] = out["repayment_status_sep"] - older_delay_avg

    return out

def add_bill_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["avg_bill_amt_6m"] = out[BILL_COLS].mean(axis=1)
    out["max_bill_amt_6m"] = out[BILL_COLS].max(axis=1)
    out["bill_volatility_6m"] = out[BILL_COLS].std(axis=1).fillna(0.0)

    older_bill_avg = out[["bill_amount_jun", "bill_amount_may", "bill_amount_apr"]].mean(axis=1)
    out["bill_growth_rate"] = _safe_divide(out["bill_amount_sep"], older_bill_avg)

    return out

def add_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["avg_pay_amt_6m"] = out[PAY_AMT_COLS].mean(axis=1)
    out["payment_volatility_6m"] = out[PAY_AMT_COLS].std(axis=1).fillna(0.0)
    out["total_paid_6m"] = out[PAY_AMT_COLS].sum(axis=1)

    return out

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    out["utilization_latest"] = _safe_divide(out["bill_amount_sep"], out["credit_limit"])
    
    utilization_cols = []
    months = ["sep", "aug", "jul", "jun", "may", "apr"]
    for month in months:
        col_name = f"utilization_{month}"
        out[col_name] = _safe_divide(out[f"bill_amount_{month}"], out["credit_limit"])
        utilization_cols.append(col_name)

    out["avg_utilization_6m"] = out[utilization_cols].mean(axis=1)

    out["pay_to_bill_ratio_sep"] = _safe_divide(out["payment_amount_sep"], out["bill_amount_sep"])

    ratio_cols = []
    for month in months:
        col_name = f"pay_to_bill_ratio_{month}"
        out[col_name] = _safe_divide(out[f"payment_amount_{month}"], out[f"bill_amount_{month}"])
        ratio_cols.append(col_name)

    out["avg_pay_to_bill_ratio_6m"] = out[ratio_cols].mean(axis=1)
    
    return out

def add_customer_profile_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["age_bucket"] = pd.cut(
        out["age"],
        bins=[0, 25, 35, 50, 120],
        labels=["18_25", "26_35", "36_50", "51_plus"],
        right=True,
    ).astype(str)

    out["limit_bal_x_late_payment_count"] = out["credit_limit"] * out["late_payment_count_6m"]

    return out

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    categorical_cols = ["gender", "education_level", "marital_status", "age_bucket"]
    out = pd.get_dummies(out, columns=categorical_cols, drop_first=False)

    return out

def build_features(df: pd.DataFrame, encode: bool = False) -> pd.DataFrame:
    out = df.copy()
    
    out = add_repayment_features(out)
    out = add_bill_features(out)
    out = add_payment_features(out)
    out = add_ratio_features(out)
    out = add_customer_profile_features(out)

    if encode:
        out = encode_categoricals(out)

    return out