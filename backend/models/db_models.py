import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, JSON, Enum as SAEnum, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from core.database import Base
import enum


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AuditJob(Base):
    __tablename__ = "audit_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[JobStatus] = mapped_column(SAEnum(JobStatus), default=JobStatus.PENDING)
    model_name: Mapped[str] = mapped_column(String(255))
    dataset_name: Mapped[str] = mapped_column(String(255))
    sensitive_feature: Mapped[str] = mapped_column(String(100))
    target_column: Mapped[str] = mapped_column(String(100))
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def job_id(self) -> str:
        return self.id


class AuditReport(Base):
    __tablename__ = "audit_reports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String, ForeignKey("audit_jobs.id"), index=True, unique=True)
    metrics: Mapped[dict] = mapped_column(JSON)            # fairness metrics
    mitigation: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # post-mitigation metrics
    explainability: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    @property
    def report_id(self) -> str:
        return self.id
