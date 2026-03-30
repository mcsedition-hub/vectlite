import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const packageRoot = resolve(__dirname, '..')
const repoRoot = resolve(packageRoot, '..', '..')
const nativeRoot = join(packageRoot, 'native')

const packageJson = JSON.parse(readFileSync(join(packageRoot, 'package.json'), 'utf8'))

function concretizeManifest(manifest, version, edition, license, corePath = null) {
  let output = manifest
    .replace(/^version\.workspace = true$/m, `version = "${version}"`)
    .replace(/^edition\.workspace = true$/m, `edition = "${edition}"`)
    .replace(/^license\.workspace = true$/m, `license = "${license}"`)

  if (corePath !== null) {
    output = output.replace(
      'vectlite = { package = "vectlite-core", path = "../../crates/vectlite-core" }',
      `vectlite = { package = "vectlite-core", path = "${corePath}" }`,
    )
  }

  return output
}

rmSync(nativeRoot, { recursive: true, force: true })
mkdirSync(join(nativeRoot, 'vectlite-core'), { recursive: true })

const nodeManifest = readFileSync(join(packageRoot, 'Cargo.toml'), 'utf8')
const coreManifest = readFileSync(join(repoRoot, 'crates', 'vectlite-core', 'Cargo.toml'), 'utf8')
const nativeBuild = readFileSync(join(packageRoot, 'build.rs'), 'utf8')

writeFileSync(
  join(nativeRoot, 'Cargo.toml'),
  concretizeManifest(nodeManifest, packageJson.version, '2024', packageJson.license, './vectlite-core'),
)
writeFileSync(
  join(nativeRoot, 'vectlite-core', 'Cargo.toml'),
  concretizeManifest(coreManifest, packageJson.version, '2024', packageJson.license),
)
writeFileSync(join(nativeRoot, 'build.rs'), nativeBuild)

cpSync(join(packageRoot, 'src'), join(nativeRoot, 'src'), { recursive: true })
cpSync(join(repoRoot, 'crates', 'vectlite-core', 'src'), join(nativeRoot, 'vectlite-core', 'src'), {
  recursive: true,
})

if (existsSync(join(repoRoot, 'LICENSE'))) {
  cpSync(join(repoRoot, 'LICENSE'), join(packageRoot, 'LICENSE'))
}
