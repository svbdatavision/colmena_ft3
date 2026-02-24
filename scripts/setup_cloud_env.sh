#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
REQUIREMENTS_FILE="$ROOT_DIR/requirements.txt"

echo "[cloud-env] Repo root: $ROOT_DIR"
echo "[cloud-env] Python bin: $PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[cloud-env] ERROR: Python interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "[cloud-env] ERROR: requirements file not found: $REQUIREMENTS_FILE" >&2
  exit 1
fi

echo "[cloud-env] Upgrading pip/setuptools/wheel..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

echo "[cloud-env] Installing dependencies from requirements.txt..."
"$PYTHON_BIN" -m pip install -r "$REQUIREMENTS_FILE"

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
    "databricks.sql",
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
