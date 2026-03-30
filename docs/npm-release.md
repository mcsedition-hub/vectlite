# npm Release

`vectlite` for Node is released from `bindings/node`.

Current published package:

- npm: https://www.npmjs.com/package/vectlite

## Local Validation

From the repository root:

```bash
bash scripts/publish_npm.sh
```

That does two local validations:

- `npm pack --dry-run`
- `npm publish --dry-run`

No upload happens unless `UPLOAD=1` is set.

Contributors can use that command to validate the package locally. It is not the preferred production release path.

To validate the prebuilt path locally for the current machine:

```bash
cd bindings/node
npm test
```

## Preferred Publish Path

Use GitHub Actions trusted publishing for real npm releases when the package has a trusted publisher configured on npmjs.com.

On npmjs.com, open the `vectlite` package settings and add a trusted publisher with these exact values:

- Organization or user: `mcsedition-hub`
- Repository: `vectlite`
- Workflow filename: `publish-npm.yml`
- Environment name: leave blank

Once that is configured, release from GitHub by pushing a version tag that matches `bindings/node/package.json`:

```bash
git tag node-vX.Y.Z
git push origin node-vX.Y.Z
bash scripts/create_github_release.sh node-vX.Y.Z
```

You can also trigger the workflow manually from the Actions tab for the current version.

The workflow builds and embeds prebuilt binaries for:

- `darwin-x64`
- `darwin-arm64`
- `linux-x64-gnu`
- `win32-x64-msvc`

## Fallback Token Publish

If you intentionally need a local token-based fallback:

```bash
export NPM_TOKEN="npm_..."
UPLOAD=1 bash scripts/publish_npm.sh
```

If this machine is already logged in with npm, you can also publish without `NPM_TOKEN`:

```bash
UPLOAD=1 bash scripts/publish_npm.sh
```

## Notes

- Preferred release path: GitHub Actions trusted publishing with OIDC.
- The current npm package prefers prebuilt binaries and only falls back to source-build on unsupported targets.
- End users need Rust/Cargo only when no matching prebuilt is available.
- Node releases use `node-vX.Y.Z` tags so they do not collide with PyPI releases.
- `scripts/create_github_release.sh` creates a GitHub Release with a standard docs/package preamble and auto-generated notes.
- Additional prebuilt targets are a future step.
- After trusted publishing is working, restrict or revoke old publish tokens.

Official docs:

- npm trusted publishers: https://docs.npmjs.com/trusted-publishers/
- GitHub Actions publishing Node packages: https://docs.github.com/actions/publishing-packages/publishing-nodejs-packages
