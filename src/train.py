from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

import matplotlib.pyplot as plt

TARGET_COL = "default_next_month"
ID_COL = "client_id"


def split_dataset(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=(test_size + val_size),
        stratify=df[target_col],
        random_state=random_state,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_size / (test_size + val_size)),
        stratify=temp_df[target_col],
        random_state=random_state,
    )

    return train_df, val_df, test_df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def get_models() -> Dict[str, object]:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
    else:
        models["gradient_boosting"] = GradientBoostingClassifier(random_state=42)

    return models


def prepare_xy(df: pd.DataFrame, target_col: str = TARGET_COL):
    drop_cols = [target_col]
    if ID_COL in df.columns:
        drop_cols.append(ID_COL)

    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[target_col].astype(int)
    return X, y


def fit_pipeline(model, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", clone(model)),
        ]
    )

    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = None

    metrics = {
        "f1": float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        metrics["auc"] = float(roc_auc_score(y, y_prob))

    return metrics


def save_confusion_matrix_artifact(
    model: Pipeline, X: pd.DataFrame, y: pd.Series, output_dir: Path
) -> Path:
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    fig.tight_layout()

    output_path = output_dir / "confusion_matrix.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_feature_importance_artifact(
    model: Pipeline, X: pd.DataFrame, output_dir: Path
) -> Path:
    model_step = model.named_steps["model"]
    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model_step, "feature_importances_"):
        importances = model_step.feature_importances_
    elif hasattr(model_step, "coef_"):
        importances = np.abs(model_step.coef_[0])
    else:
        data = pd.DataFrame(
            {"feature": feature_names, "importance": np.zeros(len(feature_names))}
        )
        output_path = output_dir / "feature_importance.csv"
        data.to_csv(output_path, index=False)
        return output_path

    imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    ax.set_title("Top 20 Feature Importances")
    fig.tight_layout()

    output_path = output_dir / "feature_importance.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
