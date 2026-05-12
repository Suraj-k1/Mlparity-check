from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/fairness_troops"
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_TASK_ALWAYS_EAGER: bool = False
    CELERY_TASK_STORE_EAGER_RESULT: bool = True
    SECRET_KEY: str = "dev-secret-key"
    ENVIRONMENT: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()
