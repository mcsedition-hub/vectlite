# Maintainer Notes

This file is the operational guide for reviewing and merging community contributions into `vectlite`.

## Branch Policy

Recommended branch protection for `main`:

- require pull requests before merging
- require status checks to pass before merging
- require branches to be up to date before merging
- restrict force pushes
- restrict branch deletion
- prefer squash merge

Suggested required checks:

- `Rust / fmt + test`
- `Python / ubuntu-latest / py3.9`
- `Python / ubuntu-latest / py3.12`
- `Python / macos-14 / py3.12`
- `Python / windows-latest / py3.12`
- `Python packaging`

## Review Priorities

When reviewing a PR, prioritize in this order:

1. correctness and regressions
2. on-disk compatibility and recovery semantics
3. API consistency across Python and Rust
4. tests and docs
5. release impact

## Changes That Need Extra Scrutiny

- anything touching `.vdb`, `.wal`, or ANN sidecars
- anything changing filter semantics
- concurrency, file locking, or crash recovery behavior
- changes to public Python signatures or return payloads
- dependency additions that affect distribution size or portability

## Merge Rules

- do not merge red CI
- do not merge undocumented public behavior changes
- do not merge format changes without an issue or design discussion
- prefer follow-up PRs over large last-minute scope creep inside review

## Release Hygiene

Before cutting a release:

1. make sure `CHANGELOG.md` is updated
2. make sure README and package docs match the shipped surface
3. make sure `cargo test --workspace` passes
4. make sure `python -m pytest bindings/python/tests -q` passes
5. stage on TestPyPI when the package description or packaging changed
6. create the GitHub Release with `bash scripts/create_github_release.sh <tag>` so the release notes always link to the official docs

## Community Workflow

- direct contributors with design-heavy proposals to an issue first
- accept bug-fix PRs quickly when they are narrow and tested
- keep roadmap ownership centralized so the project does not drift into incompatible APIs
