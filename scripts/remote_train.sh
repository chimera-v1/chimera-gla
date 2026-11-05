#!/usr/bin/env bash
# ==============================================================================
# ChimeraAI - Remote Training Trigger (macOS / Linux Dev -> Windows RTX Rig)
# ==============================================================================
# Usage:
#   ./scripts/remote_train.sh [REMOTE_HOST] [REMOTE_USER] [DURATION_HOURS]
# Defaults:
#   Host: 192.168.68.122 (or $RTX_HOST)
#   User: vinay          (or $RTX_USER)
#   Hours: 1
# ==============================================================================

set -e

REMOTE_HOST="${1:-${RTX_HOST:-192.168.68.122}}"
REMOTE_USER="${2:-${RTX_USER:-vinay}}"
DURATION_HOURS="${3:-1}"

echo "========================================================"
echo " [Agentic Execution Loop] Triggering RTX 5060 Training"
echo " Target: ${REMOTE_USER}@${REMOTE_HOST}"
echo " Duration: ${DURATION_HOURS} hour(s)"
echo "========================================================"

# Remote command executed inside PowerShell on the Windows RTX Machine
REMOTE_CMD="cd C:\Users\Admin\OneDrive\Desktop\ChimeraAI\; \
Write-Host 'Connected. Pulling latest architecture...'; \
git pull origin main; \
if (-not (Test-Path .gitignore)) { throw '.gitignore missing!' }; \
if (-not (Test-Path checkpoints)) { New-Item -ItemType Directory -Path checkpoints }; \
if (Test-Path venv\Scripts\Activate.ps1) { . venv\Scripts\Activate.ps1 }; \
python src\train.py --duration_hours ${DURATION_HOURS}"

ssh -o StrictHostKeyChecking=no "${REMOTE_USER}@${REMOTE_HOST}" "powershell -NoProfile -Command \"${REMOTE_CMD}\""

echo "========================================================"
echo " [Agentic Execution Loop] Remote training complete."
echo "========================================================"

