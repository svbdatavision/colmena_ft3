#!/bin/bash
# Wrapper that executes the daily pipeline using a single Python process.
set -euo pipefail

cd /app
python /app/run_daily_pipeline.py "$@"
