import { copyFileSync, existsSync, mkdirSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const __dirname = dirname(fileURLToPath(import.meta.url))
const packageRoot = resolve(__dirname, '..')
const repoRoot = resolve(packageRoot, '..', '..')

const result = spawnSync('cargo', ['build', '-p', 'vectlite-node', '--release'], {
  cwd: repoRoot,
  stdio: 'inherit',
})

if (result.status !== 0) {
  process.exit(result.status ?? 1)
}

const artifactName = (() => {
  switch (process.platform) {
    case 'darwin':
      return 'libvectlite_node.dylib'
    case 'win32':
      return 'vectlite_node.dll'
    default:
      return 'libvectlite_node.so'
  }
})()

const source = join(repoRoot, 'target', 'release', artifactName)
const output = join(packageRoot, 'vectlite.node')

if (!existsSync(source)) {
  console.error(`Missing built addon artifact: ${source}`)
  process.exit(1)
}

mkdirSync(dirname(output), { recursive: true })
copyFileSync(source, output)
console.log(`Copied ${source} -> ${output}`)
