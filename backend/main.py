from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.database import init_db
from api.routes.audit import router as audit_router
from api.routes.reports import router as reports_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables
    init_db()
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="Fairness Troops API",
    description="Bias & fairness auditing toolkit for ML models",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(audit_router)
app.include_router(reports_router)


@app.get("/health")
def health():
    return {"status": "ok"}
