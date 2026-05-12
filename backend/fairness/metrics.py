"""
Fairness metrics for binary classification.

All metrics assume:
  - y_true:  ground-truth labels  {0, 1}
  - y_pred:  predicted labels     {0, 1}
  - groups:  protected attribute values (any categorical)
  - privileged: the "reference" group label
"""

import numpy as np
from sklearn.metrics import accuracy_score


# ── helpers ───────────────────────────────────────────────────────────────────

def _positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """P(ŷ = 1)"""
    if len(y_pred) == 0:
        return 0.0
    return float(np.mean(y_pred))


def _tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True positive rate = P(ŷ=1 | y=1)"""
    mask = y_true == 1
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(y_pred[mask]))


# ── metrics ───────────────────────────────────────────────────────────────────

def disparate_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    privileged: object,
) -> float:
    """
    DI = P(ŷ=1 | unprivileged) / P(ŷ=1 | privileged)

    DI < 0.8 is the "80% rule" threshold for adverse impact.
    Perfect fairness → DI = 1.0
    """
    priv_mask = groups == privileged
    unpriv_mask = ~priv_mask

    pr_priv = _positive_rate(y_true[priv_mask], y_pred[priv_mask])
    pr_unpriv = _positive_rate(y_true[unpriv_mask], y_pred[unpriv_mask])

    if pr_priv == 0:
        return 1.0 if pr_unpriv == 0 else 0.0
    return pr_unpriv / pr_priv


def statistical_parity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    privileged: object,
) -> float:
    """
    SPD = P(ŷ=1 | unprivileged) − P(ŷ=1 | privileged)

    Perfect fairness → SPD = 0.0
    """
    priv_mask = groups == privileged
    unpriv_mask = ~priv_mask

    pr_priv = _positive_rate(y_true[priv_mask], y_pred[priv_mask])
    pr_unpriv = _positive_rate(y_true[unpriv_mask], y_pred[unpriv_mask])

    return pr_unpriv - pr_priv


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    privileged: object,
) -> float:
    """
    EOD = TPR(unprivileged) − TPR(privileged)

    Perfect fairness → EOD = 0.0
    """
    priv_mask = groups == privileged
    unpriv_mask = ~priv_mask

    tpr_priv = _tpr(y_true[priv_mask], y_pred[priv_mask])
    tpr_unpriv = _tpr(y_true[unpriv_mask], y_pred[unpriv_mask])

    return tpr_unpriv - tpr_priv


def accuracy_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """Per-group accuracy."""
    result: dict[str, float] = {}
    for g in np.unique(groups):
        mask = groups == g
        result[str(g)] = float(accuracy_score(y_true[mask], y_pred[mask]))
    return result


# ── aggregated audit ──────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    privileged: object,
) -> dict:
    return {
        "disparate_impact": disparate_impact(y_true, y_pred, groups, privileged),
        "statistical_parity_difference": statistical_parity_difference(y_true, y_pred, groups, privileged),
        "equal_opportunity_difference": equal_opportunity_difference(y_true, y_pred, groups, privileged),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_by_group": accuracy_by_group(y_true, y_pred, groups),
    }
