import { existsSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const __dirname = dirname(fileURLToPath(import.meta.url))
const packageRoot = resolve(__dirname, '..')

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

const prebuildTag = runtimePrebuildTag()
const prebuiltPath =
  prebuildTag == null ? null : join(packageRoot, 'prebuilds', prebuildTag, 'vectlite.node')

if (prebuiltPath != null && existsSync(prebuiltPath)) {
  console.log(`Using prebuilt addon: ${prebuiltPath}`)
  process.exit(0)
}

const result = spawnSync(process.execPath, [join(__dirname, 'build-addon.mjs')], {
  stdio: 'inherit',
})

process.exit(result.status ?? 1)
