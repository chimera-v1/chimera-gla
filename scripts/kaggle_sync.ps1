param(
    [string]$Action
)

$CheckpointsDir = Join-Path $PSScriptRoot "..\checkpoints"
$DataDir = Join-Path $PSScriptRoot "..\data"

if ($Action -eq "push") {
    Write-Host "[Kaggle Sync] Pushing checkpoints to Kaggle private dataset..."
    # Ensure dataset metadata exists
    if (-not (Test-Path (Join-Path $CheckpointsDir "dataset-metadata.json"))) {
        Write-Host "Initializing Kaggle dataset metadata in checkpoints/..."
        kaggle datasets init -p $CheckpointsDir
        
        # Modify the json file to set the dataset name to chimera-checkpoints
        $jsonPath = Join-Path $CheckpointsDir "dataset-metadata.json"
        $json = Get-Content $jsonPath | ConvertFrom-Json
        $json.id = "your-kaggle-username/chimera-checkpoints"
        $json.title = "Chimera Checkpoints"
        $json | ConvertTo-Json | Set-Content $jsonPath
        
        Write-Host "Creating new private dataset..."
        kaggle datasets create -p $CheckpointsDir --private
    } else {
        Write-Host "Creating new version for existing dataset..."
        kaggle datasets version -p $CheckpointsDir -m "Automated checkpoint backup"
    }
    Write-Host "[Kaggle Sync] Push complete."
}
elseif ($Action -eq "pull") {
    Write-Host "[Kaggle Sync] Pulling new dataset shards from Kaggle to local data/ folder..."
    # E.g. pulling a dataset named chimera-v1/curriculum-data
    kaggle datasets download -d chimera-v1/curriculum-data -p $DataDir --unzip
    Write-Host "[Kaggle Sync] Pull complete. Offline mode ready."
}
else {
    Write-Host "Usage: .\kaggle_sync.ps1 -Action <push|pull>"
}
