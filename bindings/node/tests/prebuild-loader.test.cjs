const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

function linuxLibc() {
  if (process.platform !== 'linux') {
    return null
  }

  const report = process.report?.getReport?.()
  return report?.header?.glibcVersionRuntime ? 'gnu' : 'musl'
}

function runtimePrebuildTag() {
  switch (process.platform) {
    case 'darwin':
      if (process.arch === 'x64') return 'darwin-x64'
      if (process.arch === 'arm64') return 'darwin-arm64'
      return null
    case 'linux': {
      const libc = linuxLibc()
      if (process.arch === 'x64' && libc === 'gnu') return 'linux-x64-gnu'
      if (process.arch === 'arm64' && libc === 'gnu') return 'linux-arm64-gnu'
      return null
    }
    case 'win32':
      if (process.arch === 'x64') return 'win32-x64-msvc'
      if (process.arch === 'arm64') return 'win32-arm64-msvc'
      return null
    default:
      return null
  }
}

function tempPath(name) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vectlite-node-'))
  return path.join(dir, name)
}

test('prebuilt loader works when the root addon is absent', () => {
  const packageRoot = path.resolve(__dirname, '..')
  const rootAddon = path.join(packageRoot, 'vectlite.node')
  const prebuildTag = runtimePrebuildTag()
  assert.ok(prebuildTag, 'current runtime should map to a prebuild tag in CI')

  const prebuildDir = path.join(packageRoot, 'prebuilds', prebuildTag)
  const prebuiltAddon = path.join(prebuildDir, 'vectlite.node')
  const backupAddon = path.join(packageRoot, 'vectlite.node.backup')

  fs.mkdirSync(prebuildDir, { recursive: true })
  fs.copyFileSync(rootAddon, prebuiltAddon)
  fs.renameSync(rootAddon, backupAddon)

  try {
    const modulePath = path.join(packageRoot, 'index.js')
    delete require.cache[modulePath]
    const vectlite = require(modulePath)

    const db = vectlite.open(tempPath('prebuilt.vdb'), { dimension: 2 })
    db.upsert('doc1', [1, 0], { source: 'prebuilt' })
    assert.equal(db.count(), 1)
  } finally {
    if (fs.existsSync(backupAddon)) {
      fs.renameSync(backupAddon, rootAddon)
    }
    fs.rmSync(path.join(packageRoot, 'prebuilds'), { recursive: true, force: true })
  }
})
