# Changelog

All notable changes to `vectlite` will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning while the public API stabilizes.

## [Unreleased]

No unreleased changes yet.

## [0.1.1] - 2026-03-30

### Added

- First public PyPI release of `vectlite` as an embedded Python package.
- Cross-platform GitHub Actions workflows for CI, wheel builds, TestPyPI publishing, and PyPI publishing.
- Crash-safe persistence with a snapshot, write-ahead log, and persisted ANN sidecars in the Rust core.
- Dense ANN, sparse BM25-style retrieval, hybrid dense+sparse fusion, MMR diversification, and RRF fusion.
- Python transactions, namespaces, named vectors, rerank hooks, built-in rerankers, and search diagnostics.
- Richer metadata value support across the Rust core and Python binding, including `None`, lists, and dictionaries.

### Changed

- README and package docs now advertise `pip install vectlite` as the default install path.
- Release scripts now support idempotent reruns with `--skip-existing`.
- The release workflows now use a supported macOS Intel runner for private-repo wheel builds.

### Fixed

- Duplicate inserts now raise a dedicated error instead of silently behaving like upserts.
- `open()` in the Python binding now raises `VectLiteError` for vectlite-specific failures.
- The Python package now exposes `__version__`.
- Package metadata now includes project URLs for the repository, issues, and changelog.
