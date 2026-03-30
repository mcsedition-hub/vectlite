import { cpSync, existsSync, mkdirSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

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

const prebuildTag = process.env.VECTLITE_PREBUILD_TAG ?? runtimePrebuildTag()
if (prebuildTag == null) {
  console.error('Unable to determine a prebuild tag for this platform.')
  process.exit(1)
}

const source = join(packageRoot, 'vectlite.node')
if (!existsSync(source)) {
  console.error(`Missing built addon at ${source}. Run the build first.`)
  process.exit(1)
}

const destDir = join(packageRoot, 'prebuilds', prebuildTag)
const dest = join(destDir, 'vectlite.node')

mkdirSync(destDir, { recursive: true })
cpSync(source, dest)
console.log(`Staged prebuilt ${prebuildTag}: ${dest}`)
