# Contributing to vectlite

Thanks for contributing to `vectlite`.

The project is intentionally small at the API and storage boundary. Contributions are welcome, but changes that touch the public API, file format, or retrieval behavior need to preserve coherence across the Rust core and the language bindings.

## Before You Start

- Open an issue before writing a large PR.
- Open an issue first for changes to the `.vdb` / `.wal` format, ANN persistence, filter semantics, or public Python API.
- Keep PRs focused. Small, reviewable changes are much easier to merge than broad refactors.

## Good First Contribution Areas

- Bug fixes with tests
- Documentation improvements
- Python ergonomics that do not widen the storage contract
- Benchmarks, examples, and smoke tests
- Node binding groundwork in `bindings/node`

## Changes That Need Design Discussion First

- On-disk format changes
- WAL or recovery model changes
- ANN algorithm swaps or major search semantics changes
- Breaking API changes in Python or Rust
- Large dependency additions, especially model/runtime dependencies

## Local Setup

From the repository root:

```bash
# Rust
cargo test --workspace

# Python
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip pytest maturin
python -m pip install -e bindings/python
python -m pytest bindings/python/tests -q
```

If you are working on packaging locally, you can validate that the package still builds:

```bash
bash scripts/publish_testpypi.sh
bash scripts/publish_pypi.sh
```

These commands only build artifacts locally and run `twine check`.

Do not publish from a contributor PR. Real uploads to TestPyPI or PyPI are maintainer-only release steps and require project credentials or GitHub release permissions.

## Pull Request Expectations

Every PR should:

- explain the user-visible change and why it is needed
- include tests for behavior changes
- update docs when the public API or developer workflow changes
- avoid unrelated refactors in the same branch

If your PR changes visible behavior, update `CHANGELOG.md`.

If your PR changes search behavior, include enough context to review regressions:

- before/after examples, or
- tests that pin the new behavior, or
- benchmark notes if performance is the goal

## Compatibility Rules

Treat these as high-sensitivity areas:

- `.vdb` snapshot format
- `.wal` replay semantics
- ANN sidecar persistence
- metadata filter semantics
- Python package public API

Do not assume a breaking change is acceptable just because the project is early. If the change can surprise existing users, call it out explicitly in the PR.

## Review Bar

The maintainers will usually look for:

- correctness
- tests
- API consistency
- documentation updates
- storage compatibility
- blast radius of the change

PRs may be asked to split if they mix product, refactor, and release-engineering concerns together.

## Release Notes

When a PR changes anything user-visible, add an entry to `CHANGELOG.md` under `Unreleased`.

## Code Style

- Follow existing patterns in the touched area instead of introducing a new local style.
- Keep comments sparse and useful.
- Avoid adding dependencies unless they materially improve the project.
- Do not reformat unrelated files.

## Security

Do not commit secrets, tokens, or private model credentials. If you discover a security issue, contact the maintainers privately instead of opening a public issue with exploit details.
