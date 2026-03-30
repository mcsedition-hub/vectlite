#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="mcsedition-hub/vectlite"

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/create_github_release.sh <tag>" >&2
  echo "example: bash scripts/create_github_release.sh node-v0.1.4" >&2
  exit 1
fi

TAG="$1"
DOCS_URL="https://vectlite.mcsedition.org/"
CHANGELOG_URL="https://github.com/mcsedition-hub/vectlite/blob/main/CHANGELOG.md"

case "$TAG" in
  py-v*)
    VERSION="${TAG#py-v}"
    TITLE="Python ${VERSION}"
    INSTALL_LINE='pip install vectlite'
    PACKAGE_URL="https://pypi.org/project/vectlite/${VERSION}/"
    ;;
  node-v*)
    VERSION="${TAG#node-v}"
    TITLE="Node.js ${VERSION}"
    INSTALL_LINE='npm install vectlite'
    PACKAGE_URL="https://www.npmjs.com/package/vectlite"
    ;;
  *)
    echo "unsupported tag '$TAG'" >&2
    echo "expected a tag like py-vX.Y.Z or node-vX.Y.Z" >&2
    exit 1
    ;;
esac

NOTES_FILE="$(mktemp)"
trap 'rm -f "$NOTES_FILE"' EXIT

cat >"$NOTES_FILE" <<EOF
Official docs: ${DOCS_URL}
Package: ${PACKAGE_URL}
Install: \`${INSTALL_LINE}\`
Changelog: ${CHANGELOG_URL}

EOF

pushd "$ROOT_DIR" >/dev/null
gh release create "$TAG" \
  --repo "$REPO" \
  --verify-tag \
  --title "$TITLE" \
  --generate-notes \
  --notes-file "$NOTES_FILE"
popd >/dev/null
