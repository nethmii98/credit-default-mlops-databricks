from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def find_best_threshold_by_recall_constraint(
    y_true,
    y_prob,
    min_recall: float = 0.70,
) -> dict:
    """
    Pick the threshold with recall >= min_recall and highest precision.
    If none satisfy the constraint, return the threshold with highest recall.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )

    threshold_df = pd.DataFrame(rows)

    valid = threshold_df[threshold_df["recall"] >= min_recall].copy()

    if not valid.empty:
        best = valid.sort_values(
            ["precision", "f1", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
    else:
        best = threshold_df.sort_values(
            ["recall", "precision", "f1"],
            ascending=[False, False, False],
        ).iloc[0]

    return best.to_dict()


def apply_threshold(y_prob, threshold: float):
    return (y_prob >= threshold).astype(int)
