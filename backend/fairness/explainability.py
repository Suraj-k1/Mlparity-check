"""
Explainability tools:
  1. Feature importance  — from tree-based models or permutation importance
  2. Partial Dependence Plots (PDP) — marginal effect of each feature
"""

import numpy as np
from sklearn.inspection import partial_dependence, permutation_importance


def get_feature_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    method: str = "auto",
) -> dict[str, float]:
    """
    Returns a dict {feature_name: importance_score}.

    Method priority:
      "auto"        → native coef_/feature_importances_ if available, else permutation
      "permutation" → always uses permutation importance (model-agnostic)
    """
    importances: np.ndarray | None = None

    if method == "auto":
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_).flatten()

    if importances is None:
        # Fallback: permutation importance (slower but universal)
        result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
        importances = result.importances_mean

    # Clip negatives (can appear in permutation importance)
    importances = np.clip(importances, 0, None)

    total = importances.sum()
    if total > 0:
        importances = importances / total  # normalise to [0, 1]

    return {name: float(score) for name, score in zip(feature_names, importances)}


def get_pdp_data(
    model,
    X: np.ndarray,
    feature_names: list[str],
    top_n: int = 5,
    grid_resolution: int = 20,
) -> dict[str, dict]:
    """
    Compute PDP for the top_n most important features.

    Returns {feature_name: {x_values: [...], y_values: [...]}}.
    """
    importance_scores = get_feature_importance(model, X, None, feature_names, method="auto")
    # pick top_n by importance; fall back if fewer features exist
    sorted_feats = sorted(importance_scores, key=importance_scores.get, reverse=True)
    selected = sorted_feats[:min(top_n, len(feature_names))]

    pdp_results: dict[str, dict] = {}

    for feat in selected:
        feat_idx = feature_names.index(feat)
        try:
            pdp = partial_dependence(
                model,
                X,
                features=[feat_idx],
                grid_resolution=grid_resolution,
                kind="average",
            )
            pdp_results[feat] = {
                "x_values": pdp["grid_values"][0].tolist(),
                "y_values": pdp["average"][0].tolist(),
            }
        except Exception:
            # Some estimators don't support partial_dependence
            pass

    return pdp_results


def run_explainability(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict:
    importance = get_feature_importance(model, X, y, feature_names)
    pdp = get_pdp_data(model, X, feature_names)
    return {
        "feature_importance": importance,
        "pdp_data": pdp,
    }
