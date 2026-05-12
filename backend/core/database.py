from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import StaticPool
from core.config import settings


def _engine_kwargs() -> dict:
    if settings.DATABASE_URL.startswith("sqlite"):
        kwargs = {"connect_args": {"check_same_thread": False}}
        if settings.DATABASE_URL in {"sqlite://", "sqlite:///:memory:"}:
            kwargs["poolclass"] = StaticPool
        return kwargs
    return {"pool_pre_ping": True}


engine = create_engine(settings.DATABASE_URL, **_engine_kwargs())
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency — yields a DB session, closes when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables. Called at startup."""
    from models.db_models import AuditJob, AuditReport  # noqa: F401 — register models
    Base.metadata.create_all(bind=engine)
