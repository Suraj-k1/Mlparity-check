from celery import Celery
from core.config import settings

celery_app = Celery(
    "fairness_troops",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["tasks.audit_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_always_eager=settings.CELERY_TASK_ALWAYS_EAGER,
    task_store_eager_result=settings.CELERY_TASK_STORE_EAGER_RESULT,
    # Retry failed tasks once after 60s
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,
    task_max_retries=3,
    broker_connection_retry_on_startup=True,
)
