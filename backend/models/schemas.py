from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime
from typing import Any
from models.db_models import JobStatus


class AppBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


# ── Audit submission ──────────────────────────────────────────────────────────

class AuditRequest(AppBaseModel):
    model_name: str = Field(..., description="Identifier for the ML model to audit")
    dataset_name: str = Field(..., description="Identifier for the dataset (must be registered)")
    sensitive_feature: str = Field(..., description="Column name of the protected attribute (e.g. 'gender')")
    target_column: str = Field(..., description="Target/label column name")
    apply_mitigation: bool = Field(False, description="Run reweighting mitigation after audit")
    run_explainability: bool = Field(True, description="Compute feature importance and PDP")


class AuditJobResponse(AppBaseModel):
    job_id: str
    status: JobStatus
    model_name: str
    dataset_name: str
    sensitive_feature: str
    target_column: str
    celery_task_id: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


# ── Fairness metrics ──────────────────────────────────────────────────────────

class FairnessMetrics(AppBaseModel):
    disparate_impact: float
    statistical_parity_difference: float
    equal_opportunity_difference: float
    accuracy: float
    accuracy_by_group: dict[str, float]


class MitigationResult(AppBaseModel):
    strategy: str
    disparate_impact_after: float
    statistical_parity_difference_after: float
    equal_opportunity_difference_after: float
    accuracy_after: float


class ExplainabilityResult(AppBaseModel):
    feature_importance: dict[str, float]   # feature → importance score
    pdp_data: dict[str, Any]               # feature → {x_values, y_values}


class AuditReportResponse(AppBaseModel):
    report_id: str
    job_id: str
    metrics: FairnessMetrics
    mitigation: MitigationResult | None = None
    explainability: ExplainabilityResult | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


# ── Job status ────────────────────────────────────────────────────────────────

class JobStatusResponse(AppBaseModel):
    job_id: str
    status: JobStatus
    model_name: str
    dataset_name: str
    sensitive_feature: str
    target_column: str
    celery_task_id: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
