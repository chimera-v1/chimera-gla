param(
    [string]$RemoteHost = "192.168.68.116",
    [string]$RemoteUser = "Admin",
    [string]$DurationHours = "1"
)

Write-Host "[Agentic Execution Loop] Establishing unattended SSH to $RemoteUser@$RemoteHost..."

# Using StrictHostKeyChecking=no and BatchMode=yes for unattended SSH
$sshCommand = @"
cd C:\Users\Admin\OneDrive\Desktop\ChimeraAI\
Write-Host 'Connected. Pulling latest architecture...'
git pull origin main

Write-Host 'Checking .gitignore and data integrity...'
if (-not (Test-Path .gitignore)) { throw '.gitignore missing!' }
if (-not (Test-Path data)) { Write-Host 'data/ missing. Running Kaggle pull...'; .\scripts\kaggle_sync.ps1 -Action pull }
if (-not (Test-Path checkpoints)) { New-Item -ItemType Directory -Path checkpoints }

Write-Host 'Activating virtual environment and starting training loop...'
if (Test-Path venv\Scripts\Activate.ps1) {
    . venv\Scripts\Activate.ps1
}

python src\train.py --duration_hours $DurationHours
"@

# We use Invoke-Command over SSH if PowerShell remoting is enabled, or raw SSH.
# Using raw SSH via standard client:
$tempScript = "temp_remote_script.ps1"
Set-Content -Path $tempScript -Value $sshCommand

Write-Host "Executing payload on remote RTX machine..."
# Run the script remotely
ssh -o StrictHostKeyChecking=no -o BatchMode=yes $RemoteUser@$RemoteHost "powershell -c `"`$script = Get-Content -Raw $tempScript; Invoke-Expression `$script`""

# Cleanup
Remove-Item $tempScript

Write-Host "[Agentic Execution Loop] Remote execution complete."
