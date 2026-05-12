from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from core.database import get_db
from models.db_models import AuditJob, JobStatus
from models.schemas import AuditRequest, AuditJobResponse, JobStatusResponse
from tasks.audit_tasks import run_audit_task

router = APIRouter(prefix="/audits", tags=["audits"])


@router.post("/", response_model=AuditJobResponse, status_code=status.HTTP_202_ACCEPTED)
def submit_audit(payload: AuditRequest, db: Session = Depends(get_db)):
    """
    Submit a new fairness audit. Returns a job_id you can poll for status.
    The heavy lifting runs asynchronously in a Celery worker.
    """
    job = AuditJob(
        model_name=payload.model_name,
        dataset_name=payload.dataset_name,
        sensitive_feature=payload.sensitive_feature,
        target_column=payload.target_column,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    task = run_audit_task.apply_async(
        kwargs={
            "job_id": job.id,
            "model_name": payload.model_name,
            "dataset_name": payload.dataset_name,
            "sensitive_feature": payload.sensitive_feature,
            "target_column": payload.target_column,
            "apply_mitigation": payload.apply_mitigation,
            "run_explain": payload.run_explainability,
        }
    )
    job.celery_task_id = task.id
    db.commit()
    db.refresh(job)

    return job


@router.get("/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(AuditJob).filter(AuditJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/", response_model=list[JobStatusResponse])
def list_jobs(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    return db.query(AuditJob).order_by(AuditJob.created_at.desc()).offset(skip).limit(limit).all()
