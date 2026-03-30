# Changelog

All notable changes to `vectlite` will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning while the public API stabilizes.

## [Unreleased]

### Added

- Added a contribution guide, project code of conduct, pull request template, issue templates, and maintainer notes for reviewing community PRs.

### Changed

- The repository README now points contributors to the contribution and conduct docs before opening pull requests.

## [0.1.3] - 2026-03-30

### Changed

- GitHub Actions workflows now use Node 24 native action versions for checkout, Python setup, and artifact upload/download, instead of forcing Node 24 through a workflow environment flag.
- The GitHub repository README now leads with a fuller product overview, install guidance, quick start, and feature map.
- The Python package README now reflects the broader surface area of the published package, including collections, snapshots, analyzers, rerankers, and diagnostics.

## [0.1.2] - 2026-03-30

### Added

- GitHub Actions CI for Rust formatting and tests plus Python install, test, and packaging validation across Linux, macOS, and Windows.
- Dedicated GitHub Actions release flows for TestPyPI staging and PyPI publishing with repository secrets.
- Project changelog with versioned release notes in the repository root.

### Changed

- Repository and Python package documentation now point directly to the changelog and the published PyPI install path.
- Release documentation now treats changelog updates as part of the standard cut process.
- Release examples now use version placeholders instead of hardcoded historical tags.

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
