# PyPI Release

`vectlite` can be published to the real PyPI registry either locally with an API token or from GitHub Actions.

This is a maintainer-only workflow. Contributors should use the build-only path to validate packaging, not upload releases.

## Prerequisites

- a PyPI account: https://pypi.org/account/register/
- a PyPI API token with project upload access
- local tools installed in `.venv`: `maturin` and `twine`

## Build Only

```bash
# Local packaging validation only. No upload happens here.
./scripts/publish_pypi.sh
```

This builds:

- a wheel in `dist/pypi/`
- an sdist in `dist/pypi/`
- and runs `twine check` on both artifacts

## Upload To PyPI

Maintainer-only:

```bash
export PYPI_API_TOKEN="pypi-..."
UPLOAD=1 ./scripts/publish_pypi.sh
```

The upload target defaults to `https://upload.pypi.org/legacy/`.
The script uploads with `--skip-existing`, so rerunning the same release will not fail if the exact files were already accepted.

## GitHub Actions Release

Maintainer-only.

The release workflow in `.github/workflows/wheels.yml` supports two publication modes:

- `PYPI_API_TOKEN` repository secret present: publish with token authentication
- no `PYPI_API_TOKEN` secret: fall back to trusted publishing

To configure the token for GitHub Actions:

```bash
gh secret set PYPI_API_TOKEN --repo mcsedition-hub/vectlite
```

Then push a version tag:

```bash
git tag py-vX.Y.Z
git push origin py-vX.Y.Z
bash scripts/create_github_release.sh py-vX.Y.Z
```

## Notes

- Update `CHANGELOG.md` before tagging a new version so the release notes and package metadata stay aligned.
- Python releases use `py-vX.Y.Z` tags so they do not collide with Node package releases.
- `scripts/create_github_release.sh` creates a GitHub Release with a standard docs/package preamble and auto-generated notes.
- PyPI versions are immutable once uploaded.
- The release script removes local `__pycache__` directories before building so the wheel stays clean.
- A token exported in your interactive shell is not automatically visible to the Codex shell session.
