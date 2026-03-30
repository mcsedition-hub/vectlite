# TestPyPI Release

`vectlite` can be staged on TestPyPI before a real PyPI push.

This is a maintainer-only workflow. Contributors should use the build-only path to validate packaging, not upload staged releases.

## Prerequisites

- a separate TestPyPI account: https://test.pypi.org/account/register/
- a TestPyPI API token with project upload access
- local tools installed in `.venv`: `maturin` and `twine`

## Build Only

```bash
# Local packaging validation only. No upload happens here.
./scripts/publish_testpypi.sh
```

This builds:

- a wheel in `dist/testpypi/`
- an sdist in `dist/testpypi/`
- and runs `twine check` on both artifacts

## Upload To TestPyPI

Maintainer-only:

```bash
export TEST_PYPI_API_TOKEN="pypi-..."
UPLOAD=1 ./scripts/publish_testpypi.sh
```

The upload target defaults to `https://test.pypi.org/legacy/`.
The script uploads with `--skip-existing`, so rerunning the same release does not fail if an artifact is already present.

## GitHub Actions Release

Maintainer-only.

The repository also has a dedicated workflow:

```bash
gh workflow run "Publish to TestPyPI" --repo mcsedition-hub/vectlite
```

That workflow builds the sdist and the cross-platform wheel matrix, then publishes to TestPyPI with `secrets.TEST_PYPI_API_TOKEN`.

## Install From TestPyPI

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  vectlite
```

## Notes

- Update `CHANGELOG.md` before cutting a new test release so the staged package matches the documented changes.
- TestPyPI is separate from PyPI. Accounts and tokens are not shared.
- Just like PyPI, you should treat versions as immutable once uploaded. `--skip-existing` only makes reruns idempotent; it does not replace files.
- The release script removes local `__pycache__` directories before building so the wheel stays clean.
