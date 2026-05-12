$ErrorActionPreference = "Stop"

$baseUrl = $env:API_URL
if (-not $baseUrl) {
    $baseUrl = "http://localhost:8000"
}

Write-Host "Checking API health at $baseUrl/health ..."
Invoke-RestMethod -Method Get -Uri "$baseUrl/health" | Out-Null

$payload = @{
    model_name = "random_forest"
    dataset_name = "synthetic"
    sensitive_feature = "group"
    target_column = "label"
    apply_mitigation = $true
    run_explainability = $true
} | ConvertTo-Json

Write-Host "Submitting synthetic audit ..."
$job = Invoke-RestMethod -Method Post -Uri "$baseUrl/audits/" -ContentType "application/json" -Body $payload
$jobId = $job.job_id
Write-Host "Job ID: $jobId"

$status = $null
for ($i = 1; $i -le 60; $i++) {
    Start-Sleep -Seconds 2
    $status = Invoke-RestMethod -Method Get -Uri "$baseUrl/audits/$jobId/status"
    Write-Host "[$i] Status: $($status.status)"

    if ($status.status -eq "completed") {
        $report = Invoke-RestMethod -Method Get -Uri "$baseUrl/reports/$jobId"
        Write-Host "Smoke test passed."
        Write-Host "Accuracy: $($report.metrics.accuracy)"
        Write-Host "Disparate impact: $($report.metrics.disparate_impact)"
        exit 0
    }

    if ($status.status -eq "failed") {
        throw "Audit failed: $($status.error_message)"
    }
}

throw "Timed out waiting for audit $jobId to complete. Last status: $($status.status)"
