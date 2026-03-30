# npm Release

`vectlite` for Node is released from `bindings/node`.

## Maintainer Flow

From the repository root:

```bash
bash scripts/publish_npm.sh
```

That does two local validations:

- `npm pack --dry-run`
- `npm publish --dry-run`

No upload happens unless `UPLOAD=1` is set.

## Publish

If you have an npm token:

```bash
export NPM_TOKEN="npm_..."
UPLOAD=1 bash scripts/publish_npm.sh
```

If this machine is already logged in with npm, you can also publish without `NPM_TOKEN`:

```bash
UPLOAD=1 bash scripts/publish_npm.sh
```

## Notes

- The current npm package is a source-build package.
- End users need Rust/Cargo installed when they run `npm install vectlite`.
- Prebuilt binaries are a future step.
