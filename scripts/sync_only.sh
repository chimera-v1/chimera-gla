#!/usr/bin/env bash
# ==============================================================================
# ChimeraAI - Standby Code Sync (macOS / Linux Dev -> Windows RTX Rig)
# ==============================================================================
# Usage:
#   ./scripts/sync_only.sh [REMOTE_HOST] [REMOTE_USER]
# ==============================================================================

set -e

REMOTE_HOST="${1:-${RTX_HOST:-192.168.68.122}}"
REMOTE_USER="${2:-${RTX_USER:-vinay}}"

echo "========================================================"
echo " [Sync Mode] Initiating ChimeraAI Cloud Sync on RTX Node"
echo " Target: ${REMOTE_USER}@${REMOTE_HOST}"
echo "========================================================"

REMOTE_CMD="cd C:\Users\Admin\OneDrive\Desktop\ChimeraAI\; \
Write-Host '1. Pulling latest architecture updates from GitHub...'; \
git pull origin main; \
Write-Host '========================================='; \
Write-Host 'SYNC COMPLETE. SYSTEM STANDBY.'; \
Write-Host 'Training is deliberately paused.'; \
Write-Host '========================================='"

ssh -o StrictHostKeyChecking=no "${REMOTE_USER}@${REMOTE_HOST}" "powershell -NoProfile -Command \"${REMOTE_CMD}\""

