import { existsSync, rmSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const packageRoot = resolve(__dirname, '..')
const nativeRoot = join(packageRoot, 'native')
const licensePath = join(packageRoot, 'LICENSE')

rmSync(nativeRoot, { recursive: true, force: true })

if (existsSync(licensePath)) {
  rmSync(licensePath, { force: true })
}
