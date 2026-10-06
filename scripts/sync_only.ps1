param()

Write-Host "[Sync Mode] Initiating ChimeraAI Cloud Sync on RTX Node..."
$RepoDir = Join-Path $PSScriptRoot ".."
Set-Location $RepoDir

Write-Host "1. Pulling latest architecture updates from GitHub..."
git pull origin main

Write-Host "2. Verifying Kaggle dataset structure..."
$DataDir = Join-Path $RepoDir "data"
$CheckpointsDir = Join-Path $RepoDir "checkpoints"

if (-not (Test-Path $DataDir)) {
    Write-Host "[Warning] data/ folder missing. Pulling language/code/math datasets from Kaggle..."
    New-Item -ItemType Directory -Path $DataDir | Out-Null
    # kaggle datasets download ... (Assuming user manually downloaded or uses API)
    Write-Host "[Action Required] Ensure Kaggle datasets are extracted into data/"
} else {
    Write-Host "[OK] data/ folder found."
}

if (-not (Test-Path $CheckpointsDir)) {
    Write-Host "[Warning] checkpoints/ folder missing. Creating directory..."
    New-Item -ItemType Directory -Path $CheckpointsDir | Out-Null
} else {
    Write-Host "[OK] checkpoints/ folder found."
}

Write-Host "========================================="
Write-Host "SYNC COMPLETE. SYSTEM STANDBY."
Write-Host "Training is deliberately paused as requested."
Write-Host "========================================="
