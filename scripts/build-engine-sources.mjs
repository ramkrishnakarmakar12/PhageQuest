/**
 * Inline the Python engine source tree into a TypeScript module the Pyodide
 * worker can mount.
 *
 * The alternative -- building a wheel -- introduces a packaging step that can
 * silently go stale, and the single claim this platform cannot afford to have
 * drift is that the browser runs the same engine as the server. Generating from
 * the checked-out source makes divergence impossible rather than unlikely.
 *
 * Run automatically by `npm run dev` and `npm run build`.
 */
import { readFileSync, writeFileSync, mkdirSync, readdirSync, statSync } from 'node:fs'
import { join, relative, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const engineRoot = join(here, '..', 'engine', 'src')
const outFile = join(here, '..', 'apps', 'web', 'src', 'lib', 'engineSources.ts')

function walk(dir) {
  const out = []
  for (const entry of readdirSync(dir)) {
    if (entry === '__pycache__' || entry.endsWith('.pyc')) continue
    const full = join(dir, entry)
    if (statSync(full).isDirectory()) out.push(...walk(full))
    else if (entry.endsWith('.py')) out.push(full)
  }
  return out
}

const files = walk(engineRoot).sort()
const entries = files.map((f) => {
  const rel = relative(engineRoot, f).split('\\').join('/')
  return `  ${JSON.stringify(rel)}: ${JSON.stringify(readFileSync(f, 'utf8'))},`
})
const bytes = files.reduce((n, f) => n + statSync(f).size, 0)

mkdirSync(dirname(outFile), { recursive: true })
writeFileSync(
  outFile,
  `/**
 * GENERATED FILE -- do not edit.
 *
 * Produced by scripts/build-engine-sources.mjs from engine/src. This is the
 * same Python source the API server imports; the Pyodide worker writes it into
 * its virtual filesystem at boot so Tier 0 and Tier 1 cannot diverge.
 *
 * ${files.length} modules, ${(bytes / 1024).toFixed(0)} KB of source.
 */
export const ENGINE_SOURCES: Record<string, string> = {
${entries.join('\n')}
}

export const ENGINE_MODULE_COUNT = ${files.length}
export const ENGINE_SOURCE_BYTES = ${bytes}
`,
  'utf8',
)

console.log(`engineSources.ts: ${files.length} modules, ${(bytes / 1024).toFixed(0)} KB`)
