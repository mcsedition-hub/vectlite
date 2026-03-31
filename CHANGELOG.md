# Changelog

All notable changes to `vectlite` will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning while the public API stabilizes.

## [Unreleased]

## [0.1.10] - 2026-03-31

### Added

- The repository README and Python package README now document `bulk_ingest()`, batch record formats, and a fuller database methods reference including maintenance and diagnostics APIs.

### Changed

- Python sparse-query parameters now raise a clearer `TypeError` when callers pass a string instead of the `dict[str, float]` returned by `vectlite.sparse_terms()`.
- Dimension mismatch errors now explain how to recover after changing embedding models by deleting the existing `.vdb` file or creating a new database path.
- `insert_many()`, `upsert_many()`, and transaction commits now defer index rebuilds until the end of the batch, removing the rebuild-per-operation cost from bulk writes.
- Internal WAL batch application now skips sparse index rebuilds when an operation does not touch sparse terms.
- The PyPI release workflow now reads the workspace version from `[workspace.package]` before validating `py-v*` tags.
- The npm release workflow now falls back to the repository `NPM_TOKEN` secret when present, while still keeping trusted publishing as the default path when no token is configured.

### Fixed

- Upserts that replace a previously sparse record with a record that has no sparse terms now rebuild sparse search state correctly instead of leaving stale sparse candidates behind.
- Sparse-only searches no longer fall back to returning zero-score full-scan results when no sparse candidates match.

## [0.1.8] - 2026-03-30

### Fixed

- Node `0.1.8` keeps the staged Windows prebuilt in place during the prebuilt-loader smoke test, avoiding an `EPERM` cleanup failure on GitHub Actions and allowing npm publication to complete.

## [0.1.7] - 2026-03-30

### Fixed

- Node `0.1.7` is the clean npm release that ships both the async text-embedder support and the Windows prebuilt-loader cleanup fix from the correct tagged commit.

## [0.1.6] - 2026-03-30

### Fixed

- The Node prebuilt-loader smoke test now cleans up safely on Windows, so the cross-platform npm publish workflow can complete instead of failing on `EPERM` during test cleanup.

## [0.1.5] - 2026-03-30

### Fixed

- Node `upsertText()`, `searchText()`, and `searchTextWithStats()` now support async embedding functions that return a `Promise`, matching the documented usage.

## [0.1.4] - 2026-03-30

### Added

- Added a contribution guide, project code of conduct, pull request template, issue templates, and maintainer notes for reviewing community PRs.
- Added an initial Node binding in `bindings/node` with a native `napi-rs` addon, JavaScript wrapper, TypeScript declarations, and smoke tests for CRUD, collections, and text helpers.
- Added a GitHub Actions workflow for npm trusted publishing, with tag-to-package-version validation and package tarball checks before publish.
- Added Node prebuilt-binary support for macOS x64/arm64, Linux x64 (glibc), and Windows x64, with a source-build fallback on unsupported targets.

### Changed

- The repository README now points contributors to the contribution and conduct docs before opening pull requests.
- Contribution and release docs now distinguish local packaging validation from maintainer-only publishing steps, so public contributors are not told to upload releases.
- Local packaging commands in the docs are now explicitly labeled as no-upload validation steps.
- The main CI workflow now runs Node smoke tests on Linux, macOS, and Windows in addition to the Rust and Python checks.
- The repository README now shows the Node binding as available from source instead of just planned.
- The Node package is now structured as a self-contained source-build npm package with prepack/install scripts and a maintainer npm release flow.
- The repository and Node package docs now advertise `npm install vectlite` as the default Node install path.
- Python and Node package releases now use separate tag namespaces (`py-v*` and `node-v*`) so Node-only releases do not trigger PyPI publication.
- Public package metadata and README links now point to the official docs site at `https://vectlite.mcsedition.org/`.
- GitHub Releases can now be created through `scripts/create_github_release.sh`, which prepends links to the official docs, package page, install command, and changelog before auto-generated notes.

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
