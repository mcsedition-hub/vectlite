import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const packageRoot = resolve(__dirname, '..')
const sourceRoot = resolve(process.argv[2] ?? join(packageRoot, '..', '..', 'dist', 'node-prebuilds'))
const destRoot = join(packageRoot, 'prebuilds')

for (const entry of readdirSync(sourceRoot, { withFileTypes: true })) {
  if (!entry.isDirectory() || !entry.name.startsWith('prebuild-')) {
    continue
  }

  const prebuildTag = entry.name.slice('prebuild-'.length)
  const source = join(sourceRoot, entry.name, 'vectlite.node')
  if (!existsSync(source)) {
    console.error(`Missing prebuilt artifact for ${prebuildTag}: ${source}`)
    process.exit(1)
  }

  const destDir = join(destRoot, prebuildTag)
  mkdirSync(destDir, { recursive: true })
  cpSync(source, join(destDir, 'vectlite.node'))
  console.log(`Collected ${prebuildTag}`)
}
