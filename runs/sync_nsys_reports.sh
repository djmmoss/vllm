#!/bin/bash
# Sync .nsys-rep files from remote server to local runs directory
#
# This script uses rsync to download all .nsys-rep files from the remote
# server's runs directory to the local runs directory, showing progress.

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Remote server configuration (matching settings.json)
REMOTE_USER="dmoss-mfa"
REMOTE_HOST="lyris"
REMOTE_BASE="/home/dmoss/scratch"
REMOTE_RUNS_DIR="${REMOTE_BASE}/meta-dsr1-gb200/runs"

# Local destination
LOCAL_RUNS_DIR="$SCRIPT_DIR"

echo "Syncing .nsys-rep files from remote server..."
echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RUNS_DIR}/"
echo "Local:  ${LOCAL_RUNS_DIR}/"
echo ""

# Create local runs directory if it doesn't exist
mkdir -p "$LOCAL_RUNS_DIR"

# Sync only .nsys-rep files with progress (including subdirectories)
# The -a flag preserves directory structure, timestamps, and permissions
rsync -avzP \
    --include='*/' \
    --include='*.nsys-rep' \
    --include='*.nsys-rep.*' \
    --exclude='*' \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_RUNS_DIR}/" \
    "$LOCAL_RUNS_DIR/"

echo ""
echo "Sync complete!"

