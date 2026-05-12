"""
Bias mitigation — reweighting strategy.

Reweighting assigns sample weights so that each (group, label) cell
has the weight it would have under independence between group and label.

References: Kamiran & Calders (2012) "Data preprocessing techniques for
classification without discrimination".
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from fairness.metrics import compute_all_metrics


def compute_sample_weights(
    groups: np.ndarray,
    y_true: np.ndarray,
) -> np.ndarray:
    """
    Return per-sample weights so P(group) * P(label) == P(group, label)
    under the reweighted distribution.
    """
    n = len(y_true)
    weights = np.ones(n)

    for g in np.unique(groups):
        for label in np.unique(y_true):
            mask = (groups == g) & (y_true == label)
            expected = (np.mean(groups == g) * np.mean(y_true == label))
            observed = np.mean(mask)
            if observed > 0:
                weights[mask] = expected / observed

    # Normalize so mean weight == 1
    weights /= weights.mean()
    return weights


def apply_reweighting(
    model,                  # sklearn-compatible estimator (fitted)
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_test: np.ndarray,
    privileged: object,
    feature_names: list[str],
) -> dict:
    """
    1. Compute sample weights for X_train.
    2. Retrain a clone of `model` using those weights.
    3. Evaluate fairness metrics on X_test.

    Returns a dict with the post-mitigation metrics.
    """
    weights = compute_sample_weights(groups_train, y_train)

    mitigated_model = clone(model)
    mitigated_model.fit(X_train, y_train, sample_weight=weights)

    y_pred_mitigated = mitigated_model.predict(X_test)

    post_metrics = compute_all_metrics(y_test, y_pred_mitigated, groups_test, privileged)

    return {
        "strategy": "reweighting",
        "disparate_impact_after": post_metrics["disparate_impact"],
        "statistical_parity_difference_after": post_metrics["statistical_parity_difference"],
        "equal_opportunity_difference_after": post_metrics["equal_opportunity_difference"],
        "accuracy_after": post_metrics["accuracy"],
        "mitigated_model": mitigated_model,
    }
