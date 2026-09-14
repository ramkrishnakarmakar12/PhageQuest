/**
 * Tier 0: the statistics engine, running in the student's browser.
 *
 * Specification section 07:
 *
 *   "about 90% of the Grades 5-12 workload can run in the student's browser,
 *    at zero marginal compute cost and with no sandbox-escape surface on our
 *    infrastructure... student data never reaches our servers -- which is a
 *    DPDP argument as much as a cost one."
 *
 * This worker loads Pyodide, mounts the *same* `phagequest_engine` package the
 * server runs, and executes tool calls locally. The transcript it produces is
 * byte-identical in shape to a server-side one, which is the whole point: a
 * teacher reviewing work cannot tell, and does not need to care, which tier ran
 * it.
 *
 * Why the engine source is shipped as a virtual filesystem rather than a wheel:
 * a wheel needs a packaging step that can silently fall out of date with the
 * checked-out source, and the one thing that must never drift is the claim that
 * the browser runs the same code as the server. The build inlines the source
 * tree, so they cannot diverge.
 *
 * Everything runs off the main thread, so a 20-second geometry computation
 * never freezes the tab on a shared lab PC.
 */

/// <reference lib="webworker" />

import { ENGINE_SOURCES } from '../lib/engineSources'

const PYODIDE_VERSION = '0.28.3'
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`

type PyodideAPI = {
  loadPackage: (names: string[], opts?: unknown) => Promise<void>
  runPython: (code: string) => unknown
  runPythonAsync: (code: string) => Promise<unknown>
  FS: {
    mkdirTree: (path: string) => void
    writeFile: (path: string, data: string, opts?: { encoding: string }) => void
  }
  globals: { set: (k: string, v: unknown) => void; get: (k: string) => unknown }
}

type PyodideModule = {
  loadPyodide: (opts: { indexURL: string }) => Promise<PyodideAPI>
}

let pyodide: PyodideAPI | null = null
let booting: Promise<PyodideAPI> | null = null

function post(msg: Record<string, unknown>) {
  ;(self as unknown as Worker).postMessage(msg)
}

async function boot(): Promise<PyodideAPI> {
  if (pyodide) return pyodide
  if (booting) return booting

  booting = (async () => {
    post({ type: 'boot', stage: 'runtime', message: 'Loading Python…' })
    // This is a MODULE worker (vite `worker.format: 'es'`), where
    // `importScripts` does not exist -- so Pyodide is brought in with a dynamic
    // import of its ESM build. `@vite-ignore` keeps the bundler from trying to
    // resolve a CDN URL at build time.
    const mod: PyodideModule = await import(/* @vite-ignore */ `${PYODIDE_URL}pyodide.mjs`)
    const py = await mod.loadPyodide({ indexURL: PYODIDE_URL })

    post({ type: 'boot', stage: 'packages', message: 'Loading numpy and scipy…' })
    // numpy + scipy only. The engine has no other dependency, by design:
    // every GPL/AGPL package the specification flags stays out of Tier 0.
    await py.loadPackage(['numpy', 'scipy'])

    post({ type: 'boot', stage: 'engine', message: 'Mounting the engine…' })
    const root = '/lib/phagequest'
    py.FS.mkdirTree(root)
    for (const [path, source] of Object.entries(ENGINE_SOURCES)) {
      const full = `${root}/${path}`
      const dir = full.slice(0, full.lastIndexOf('/'))
      py.FS.mkdirTree(dir)
      py.FS.writeFile(full, source, { encoding: 'utf8' })
    }

    await py.runPythonAsync(`
import sys, json
sys.path.insert(0, ${JSON.stringify(root)})
import phagequest_engine as pq
from phagequest_engine import registry

_store = pq.TranscriptStore()
pq.set_store(_store)

def _run(tool, arguments, grade):
    """Execute one tool and return the result plus its full transcript."""
    result = registry.call_tool(tool, arguments, grade=grade)
    tid = result.get("transcript_id")
    t = _store.get(tid) if tid else None
    return json.dumps({
        "result": result,
        "transcript": t.to_dict() if t else None,
    }, default=str)

def _ledger(data_id):
    return json.dumps(pq.ledger.ledger_for(data_id, store=_store).to_dict(), default=str)

def _catalogue(grade):
    return json.dumps({
        "tools": registry.tool_schemas(grade=grade),
        "engine": pq.transcript.engine_versions(),
        "simulation_parameters": pq.simulation.PARAM_META,
        "simulation_models": {k: {"label": v.label, "params": v.params,
                                  "teaches": v.teaches, "grades": v.grades}
                              for k, v in pq.simulation.MODELS.items()},
        "curve_models": {k: {"label": v["label"], "params": v["params"],
                             "teaches": v["teaches"]}
                         for k, v in pq.curves.MODELS.items()},
        "templates": {k: v.to_dict() for k, v in pq.validation.TEMPLATES.items()},
    }, default=str)

def _scan(text):
    return json.dumps(pq.guards.numeric_scan(text, store=_store).to_dict(), default=str)

def _reset():
    _store.clear()
    return "ok"
`)
    pyodide = py
    post({
      type: 'ready',
      versions: JSON.parse(
        py.runPython('import json; json.dumps(pq.transcript.engine_versions())') as string,
      ),
    })
    return py
  })()

  return booting
}

self.onmessage = async (event: MessageEvent) => {
  const { id, type, payload } = event.data ?? {}
  try {
    const py = await boot()
    let out: unknown

    switch (type) {
      case 'boot':
        out = { ready: true }
        break
      case 'catalogue':
        py.globals.set('_grade', payload.grade)
        out = JSON.parse(py.runPython('_catalogue(_grade)') as string)
        break
      case 'run': {
        py.globals.set('_tool', payload.tool)
        py.globals.set('_args', JSON.stringify(payload.arguments ?? {}))
        py.globals.set('_grade', payload.grade ?? null)
        const started = performance.now()
        const raw = (await py.runPythonAsync(
          '_run(_tool, json.loads(_args), _grade)',
        )) as string
        const parsed = JSON.parse(raw)
        out = { ...parsed, wallMs: Math.round(performance.now() - started) }
        break
      }
      case 'ledger':
        py.globals.set('_did', payload.dataId)
        out = JSON.parse(py.runPython('_ledger(_did)') as string)
        break
      case 'scan':
        py.globals.set('_text', payload.text)
        out = JSON.parse(py.runPython('_scan(_text)') as string)
        break
      case 'reset':
        out = { reset: py.runPython('_reset()') }
        break
      default:
        throw new Error(`Unknown worker message: ${type}`)
    }

    post({ id, type: 'result', payload: out })
  } catch (error) {
    post({
      id,
      type: 'error',
      payload: {
        message: friendly(error),
        // A failure here is not a silent fallback to the server. The
        // specification's cost and DPDP arguments both rest on Tier 0 actually
        // running, so a broken Tier 0 must be visible, not papered over.
        tier: 0,
      },
    })
  }
}

/** Turn a runtime failure into something a teacher can act on. */
function friendly(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error)
  if (/fetch|network|Failed to load|ERR_|import\(\)|dynamically imported/i.test(raw)) {
    return (
      `The Python runtime could not be downloaded (${raw}). It is fetched once from a CDN and ` +
      `then cached, so this is almost always a blocked or offline network on first use. Ask ` +
      `whoever manages the network to allow ${PYODIDE_URL}, or use a deployment that serves ` +
      `Pyodide from the same origin as the platform.`
    )
  }
  return raw
}

export {}
