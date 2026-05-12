"""
Celery tasks for asynchronous fairness auditing.

Task flow:
  run_audit_task
    → load_data_and_model()
    → compute_all_metrics()
    → (optional) apply_reweighting()
    → (optional) run_explainability()
    → persist AuditReport to DB
    → update AuditJob status
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from core.celery_app import celery_app
from core.database import SessionLocal
from models.db_models import AuditJob, AuditReport, JobStatus
from fairness.metrics import compute_all_metrics
from fairness.mitigation import apply_reweighting
from fairness.explainability import run_explainability


# ── toy dataset / model registry ─────────────────────────────────────────────
# In production replace with S3/GCS fetches or a model registry (MLflow, etc.)

SYNTHETIC_PRIVILEGED_GROUP = "A"
ADULT_PRIVILEGED_GROUP = "Male"

def _get_dataset(dataset_name: str, sensitive_feature: str, target_column: str):
    """
    Returns (X_train, X_test, y_train, y_test, groups_train, groups_test, feature_names).
    Uses sklearn's Adult Income dataset as a built-in demo.
    """
    from sklearn.datasets import fetch_openml

    if dataset_name == "adult":
        data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
        df: pd.DataFrame = data.frame
        df.dropna(inplace=True)

        if target_column not in df.columns:
            raise ValueError(f"Unknown target column '{target_column}' for dataset '{dataset_name}'")
        if sensitive_feature not in df.columns:
            raise ValueError(f"Unknown sensitive feature '{sensitive_feature}' for dataset '{dataset_name}'")

        # encode categoricals
        cat_cols = df.select_dtypes(include="category").columns.tolist()
        for col in cat_cols:
            if col != target_column and col != sensitive_feature:
                df[col] = df[col].cat.codes

        # encode target
        df[target_column] = (df[target_column].astype(str).str.strip() == ">50K").astype(int)

        # sensitive feature
        groups = df[sensitive_feature].astype(str).values

        feature_cols = [c for c in df.columns if c != target_column and c != sensitive_feature]
        X = df[feature_cols].select_dtypes(include="number").values
        feature_names = df[feature_cols].select_dtypes(include="number").columns.tolist()
        y = df[target_column].values
        privileged = ADULT_PRIVILEGED_GROUP

    else:
        # Generic fallback: generate synthetic data
        rng = np.random.default_rng(42)
        n = 1000
        X = rng.standard_normal((n, 5))
        groups = rng.choice(["A", "B"], size=n, p=[0.6, 0.4])
        # Introduce bias: group B gets lower positive probability
        y = (X[:, 0] + (groups == "A").astype(float) * 0.8 > 0).astype(int)
        feature_names = [f"feature_{i}" for i in range(5)]
        privileged = SYNTHETIC_PRIVILEGED_GROUP

    X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, g_train, g_test, feature_names, privileged


def _get_model(model_name: str):
    """Toy model registry. Replace with joblib.load from S3 in production."""
    registry = {
        "random_forest": RandomForestClassifier(n_estimators=50, random_state=42),
        "logistic_regression": LogisticRegression(max_iter=500, random_state=42),
    }
    return registry.get(model_name, RandomForestClassifier(n_estimators=50, random_state=42))


# ── main celery task ──────────────────────────────────────────────────────────

@celery_app.task(bind=True, max_retries=3)
def run_audit_task(
    self,
    job_id: str,
    model_name: str,
    dataset_name: str,
    sensitive_feature: str,
    target_column: str,
    apply_mitigation: bool,
    run_explain: bool,
):
    db = SessionLocal()
    try:
        # Mark job as running
        job = db.query(AuditJob).filter(AuditJob.id == job_id).first()
        if not job:
            return
        job.status = JobStatus.RUNNING
        db.commit()

        # 1. Load data + model
        (X_train, X_test, y_train, y_test,
         g_train, g_test, feature_names, privileged) = _get_dataset(
            dataset_name, sensitive_feature, target_column
        )

        model = _get_model(model_name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # 2. Compute baseline fairness metrics
        metrics = compute_all_metrics(y_test, y_pred, g_test, privileged)

        # 3. Optional: reweighting mitigation
        mitigation_result = None
        if apply_mitigation:
            raw = apply_reweighting(
                model, X_train, y_train, g_train,
                X_test, y_test, g_test, privileged, feature_names,
            )
            mitigation_result = {
                "strategy": raw["strategy"],
                "disparate_impact_after": raw["disparate_impact_after"],
                "statistical_parity_difference_after": raw["statistical_parity_difference_after"],
                "equal_opportunity_difference_after": raw["equal_opportunity_difference_after"],
                "accuracy_after": raw["accuracy_after"],
            }

        # 4. Optional: explainability
        explainability_result = None
        if run_explain:
            explainability_result = run_explainability(model, X_test, y_test, feature_names)

        # 5. Persist report
        report = AuditReport(
            job_id=job_id,
            metrics=metrics,
            mitigation=mitigation_result,
            explainability=explainability_result,
        )
        db.add(report)

        job.status = JobStatus.COMPLETED
        db.commit()

    except Exception as exc:
        db.rollback()
        job = db.query(AuditJob).filter(AuditJob.id == job_id).first()
        should_retry = self.request.retries < self.max_retries
        if job:
            job.status = JobStatus.RUNNING if should_retry else JobStatus.FAILED
            job.error_message = f"Retrying after error: {exc}" if should_retry else str(exc)
            db.commit()
        if should_retry:
            raise self.retry(exc=exc, countdown=60)
        raise
    finally:
        db.close()
