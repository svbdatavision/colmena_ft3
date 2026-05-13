#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"
MAX_RETRIES="${MAX_RETRIES:-5}"
INITIAL_BACKOFF_SECONDS="${INITIAL_BACKOFF_SECONDS:-2}"

echo "[cloud-env] Repo root: $ROOT_DIR"
echo "[cloud-env] Python bin: $PYTHON_BIN"
echo "[cloud-env] Max retries: $MAX_RETRIES"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[cloud-env] ERROR: Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "[cloud-env] ERROR: requirements file not found: $REQUIREMENTS_FILE" >&2
  exit 1
fi

retry_cmd() {
  local description="$1"
  shift

  local attempt=1
  local sleep_seconds="$INITIAL_BACKOFF_SECONDS"
  while true; do
    echo "[cloud-env] $description (attempt $attempt/$MAX_RETRIES)"
    if "$@"; then
      return 0
    fi

    if [ "$attempt" -ge "$MAX_RETRIES" ]; then
      echo "[cloud-env] ERROR: $description failed after $MAX_RETRIES attempts." >&2
      return 1
    fi

    echo "[cloud-env] WARNING: $description failed; retrying in ${sleep_seconds}s..."
    sleep "$sleep_seconds"
    sleep_seconds=$((sleep_seconds * 2))
    attempt=$((attempt + 1))
  done
}

TMP_REQUIREMENTS="$(mktemp /tmp/ft3-requirements.XXXXXX.txt)"
cleanup() {
  rm -f "$TMP_REQUIREMENTS"
}
trap cleanup EXIT

copy_requirements() {
  cp "$REQUIREMENTS_FILE" "$TMP_REQUIREMENTS"
}

retry_cmd "Copy requirements to local tmp" copy_requirements

echo "[cloud-env] Upgrading pip/setuptools/wheel..."
retry_cmd "Upgrade pip tooling" "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

echo "[cloud-env] Installing dependencies from requirements.txt..."
retry_cmd "Install dependencies" "$PYTHON_BIN" -m pip install -r "$TMP_REQUIREMENTS"

echo "[cloud-env] Verifying critical imports..."
"$PYTHON_BIN" - <<'PY'
import importlib
import sys

modules = [
    "pandas",
    "numpy",
    "sklearn",
    "lightgbm",
    "optuna",
    "shap",
]

failures = []
for module_name in modules:
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        failures.append((module_name, str(exc)))

if failures:
    print("[cloud-env] ERROR: import verification failed:", file=sys.stderr)
    for module_name, error in failures:
        print(f"  - {module_name}: {error}", file=sys.stderr)
    raise SystemExit(1)

print("[cloud-env] Import verification OK.")
PY

echo "[cloud-env] Setup complete."
