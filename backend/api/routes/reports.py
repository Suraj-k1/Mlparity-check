from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from core.database import get_db
from models.db_models import AuditReport
from models.schemas import AuditReportResponse, FairnessMetrics, MitigationResult, ExplainabilityResult

router = APIRouter(prefix="/reports", tags=["reports"])


def _parse_report(report: AuditReport) -> AuditReportResponse:
    mitigation = MitigationResult(**report.mitigation) if report.mitigation else None
    explainability = ExplainabilityResult(**report.explainability) if report.explainability else None
    return AuditReportResponse(
        report_id=report.id,
        job_id=report.job_id,
        metrics=FairnessMetrics(**report.metrics),
        mitigation=mitigation,
        explainability=explainability,
        created_at=report.created_at,
    )


@router.get("/{job_id}", response_model=AuditReportResponse)
def get_report_by_job(job_id: str, db: Session = Depends(get_db)):
    """Fetch the audit report for a completed job."""
    report = db.query(AuditReport).filter(AuditReport.job_id == job_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found. Job may still be running.")
    return _parse_report(report)


@router.get("/", response_model=list[AuditReportResponse])
def list_reports(skip: int = 0, limit: int = 50, db: Session = Depends(get_db)):
    reports = db.query(AuditReport).order_by(AuditReport.created_at.desc()).offset(skip).limit(limit).all()
    return [_parse_report(r) for r in reports]
