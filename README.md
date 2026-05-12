# Fairness Troops

A Dockerized fairness auditing demo for binary classification models.

## What Runs

- Streamlit frontend: `http://localhost:8501`
- FastAPI backend: `http://localhost:8000`
- Celery worker: asynchronous audit execution
- Redis: Celery broker/result backend
- PostgreSQL: audit jobs and reports
- Flower: `http://localhost:5555`

## Requirements

- Docker Desktop with Docker Compose
- PowerShell, for the optional smoke test on Windows

You do not need to install Redis, PostgreSQL, Celery, or Python packages on your host machine.

## Start Everything

```powershell
cd D:\fairness-troops
Copy-Item .env.example .env
docker compose up --build
```

Open:

- UI: http://localhost:8501
- API docs: http://localhost:8000/docs
- Celery monitor: http://localhost:5555

The UI defaults to the `synthetic` dataset, which runs fully offline. The `adult` dataset option downloads data from OpenML and needs internet access inside the backend/worker containers.

## Smoke Test

In a second PowerShell window, run:

```powershell
cd D:\fairness-troops
.\scripts\smoke-test.ps1
```

The script checks `/health`, submits a synthetic audit, polls the job status, fetches the report, and prints a couple of metric values.

## Useful Commands

```powershell
# Rebuild and start
docker compose up --build

# Stop containers
docker compose down

# Stop containers and clear the Postgres volume
docker compose down -v

# Show backend logs
docker compose logs -f backend

# Show worker logs
docker compose logs -f worker
```

If you previously ran an older version of this project, use `docker compose down -v` once before starting again so PostgreSQL recreates the current tables.

## API Example

```powershell
$body = @{
  model_name = "random_forest"
  dataset_name = "synthetic"
  sensitive_feature = "group"
  target_column = "label"
  apply_mitigation = $true
  run_explainability = $true
} | ConvertTo-Json

$job = Invoke-RestMethod -Method Post -Uri "http://localhost:8000/audits/" -ContentType "application/json" -Body $body
$job.job_id
```

Then poll:

```powershell
Invoke-RestMethod "http://localhost:8000/audits/$($job.job_id)/status"
Invoke-RestMethod "http://localhost:8000/reports/$($job.job_id)"
```

## Project Structure

```text
fairness-troops/
  docker-compose.yml
  .env.example
  scripts/
    smoke-test.ps1
  backend/
    main.py
    core/
      config.py
      database.py
      celery_app.py
    api/routes/
      audit.py
      reports.py
    fairness/
      metrics.py
      mitigation.py
      explainability.py
    tasks/
      audit_tasks.py
    models/
      db_models.py
      schemas.py
  frontend/
    app.py
```
