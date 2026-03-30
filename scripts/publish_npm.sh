#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="$ROOT_DIR/bindings/node"
CACHE_DIR="${NPM_CONFIG_CACHE:-$ROOT_DIR/dist/npm-cache}"

mkdir -p "$CACHE_DIR"

pushd "$PACKAGE_DIR" >/dev/null

function has_npm_auth() {
  if [[ -n "${NPM_TOKEN:-}" ]]; then
    return 0
  fi

  npm_config_cache="$CACHE_DIR" npm whoami >/dev/null 2>&1
}

echo "==> npm pack --dry-run"
npm_config_cache="$CACHE_DIR" npm pack --dry-run

if has_npm_auth; then
  echo "==> npm publish --dry-run"
  if [[ -n "${NPM_TOKEN:-}" ]]; then
    USERCONFIG="$(mktemp)"
    trap 'rm -f "$USERCONFIG"' EXIT
    printf '%s\n' "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > "$USERCONFIG"
    NPM_CONFIG_USERCONFIG="$USERCONFIG" npm_config_cache="$CACHE_DIR" npm publish --dry-run
  else
    npm_config_cache="$CACHE_DIR" npm publish --dry-run
  fi
else
  echo "Skipping npm publish --dry-run: no npm auth is configured in this shell."
fi

if [[ "${UPLOAD:-0}" != "1" ]]; then
  echo "Dry-run complete. Set UPLOAD=1 to publish."
  popd >/dev/null
  exit 0
fi

if [[ -n "${NPM_TOKEN:-}" ]]; then
  USERCONFIG="$(mktemp)"
  trap 'rm -f "$USERCONFIG"' EXIT
  printf '%s\n' "//registry.npmjs.org/:_authToken=${NPM_TOKEN}" > "$USERCONFIG"
  echo "==> npm publish"
  NPM_CONFIG_USERCONFIG="$USERCONFIG" npm_config_cache="$CACHE_DIR" npm publish
else
  if ! npm_config_cache="$CACHE_DIR" npm whoami >/dev/null 2>&1; then
    echo "UPLOAD=1 requires either NPM_TOKEN or an existing npm login on this machine." >&2
    popd >/dev/null
    exit 1
  fi
  echo "==> npm publish"
  npm_config_cache="$CACHE_DIR" npm publish
fi

popd >/dev/null
