#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
MATURIN_BIN="${MATURIN_BIN:-$ROOT_DIR/.venv/bin/maturin}"
TWINE_BIN="${TWINE_BIN:-$ROOT_DIR/.venv/bin/twine}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/dist/pypi}"
PACKAGE_GLOB="${PACKAGE_GLOB:-$OUT_DIR/*}"
REPOSITORY_URL="${REPOSITORY_URL:-https://upload.pypi.org/legacy/}"
UPLOAD="${UPLOAD:-0}"

require_executable() {
  local path="$1"
  local name="$2"
  if [[ ! -x "$path" ]]; then
    echo "missing $name at $path" >&2
    exit 1
  fi
}

require_executable "$PYTHON_BIN" "python"
require_executable "$MATURIN_BIN" "maturin"
require_executable "$TWINE_BIN" "twine"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

rm -rf "$ROOT_DIR/bindings/python/vectlite/__pycache__"
rm -rf "$ROOT_DIR/bindings/python/tests/__pycache__"

"$MATURIN_BIN" build \
  -m "$ROOT_DIR/bindings/python/Cargo.toml" \
  --release \
  --interpreter "$PYTHON_BIN" \
  --compatibility pypi \
  --out "$OUT_DIR"

"$MATURIN_BIN" sdist \
  -m "$ROOT_DIR/bindings/python/Cargo.toml" \
  --out "$OUT_DIR"

"$TWINE_BIN" check $PACKAGE_GLOB

if [[ "$UPLOAD" != "1" ]]; then
  echo "built distributions in $OUT_DIR"
  echo "set UPLOAD=1 and PYPI_API_TOKEN to upload to PyPI"
  exit 0
fi

if [[ -z "${PYPI_API_TOKEN:-}" ]]; then
  echo "PYPI_API_TOKEN is required when UPLOAD=1" >&2
  exit 1
fi

"$TWINE_BIN" upload \
  --repository-url "$REPOSITORY_URL" \
  --non-interactive \
  --skip-existing \
  --username "__token__" \
  --password "$PYPI_API_TOKEN" \
  $PACKAGE_GLOB
