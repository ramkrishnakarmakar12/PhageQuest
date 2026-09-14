/**
 * The client's view of the engine.
 *
 * One rule governs this file, and it is the platform invariant:
 *
 *   a number reaches the screen only as part of a `BoundResult`, which carries
 *   the transcript that produced it.
 *
 * There is deliberately no helper here that returns a bare number. Anything
 * that wants to render a p-value has to hold the transcript too, which means
 * the transcript panel can always be opened next to it.
 */

import EngineWorker from '../worker/engine.worker?worker'

export type Transcript = {
  transcript_id: string
  tool: string
  call: string
  params: Record<string, unknown>
  data_id: string | null
  engine: Record<string, string>
  started_at: number
  duration_ms: number
  result: Record<string, unknown>
  notes: string[]
  error: string | null
}

export type ToolResult = Record<string, unknown> & {
  transcript_id?: string
  engine?: string
  error?: string
  message?: string
  refused?: boolean
  reason?: string
  plain_language?: string
  p_value?: number | null
}

export type BoundResult = {
  result: ToolResult
  transcript: Transcript | null
  wallMs: number
}

export type ToolSchema = {
  name: string
  description: string
  input_schema: { properties: Record<string, any>; required: string[] }
  grades: string
  curriculum: string
  category: string
  inferential: boolean
}

export type Catalogue = {
  tools: ToolSchema[]
  engine: Record<string, string>
  simulation_parameters: Record<string, any>
  simulation_models: Record<string, { label: string; params: string[]; teaches: string; grades: string }>
  curve_models: Record<string, { label: string; params: string[]; teaches: string }>
  templates: Record<string, any>
}

export type BootStage = { stage: string; message: string }

type Pending = { resolve: (v: any) => void; reject: (e: any) => void }

class Engine {
  private worker: Worker | null = null
  private pending = new Map<number, Pending>()
  private seq = 0
  private readyPromise: Promise<Record<string, string>> | null = null
  private bootListeners = new Set<(s: BootStage) => void>()

  onBoot(fn: (s: BootStage) => void): () => void {
    this.bootListeners.add(fn)
    return () => this.bootListeners.delete(fn)
  }

  /** Start Pyodide. Called once, early, so the 40-minute period is not spent waiting. */
  start(): Promise<Record<string, string>> {
    if (this.readyPromise) return this.readyPromise

    this.readyPromise = new Promise((resolve, reject) => {
      const worker = new EngineWorker()
      this.worker = worker

      worker.onmessage = (event: MessageEvent) => {
        const { id, type, payload, stage, message, versions } = event.data ?? {}
        if (type === 'boot') {
          this.bootListeners.forEach((fn) => fn({ stage, message }))
          return
        }
        if (type === 'ready') {
          resolve(versions)
          return
        }
        const entry = this.pending.get(id)
        if (!entry) return
        this.pending.delete(id)
        if (type === 'error') entry.reject(new Error(payload?.message ?? 'engine error'))
        else entry.resolve(payload)
      }

      worker.onerror = (e) => reject(new Error(e.message))
      this.send('boot', {}).catch(reject)
    })

    return this.readyPromise
  }

  private send<T>(type: string, payload: unknown): Promise<T> {
    if (!this.worker) this.start()
    const id = ++this.seq
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.worker!.postMessage({ id, type, payload })
    })
  }

  /** What this grade is allowed to run, plus the metadata the UI needs. */
  catalogue(grade: number): Promise<Catalogue> {
    return this.send<Catalogue>('catalogue', { grade })
  }

  /**
   * Run one tool. The ONLY way a number is produced client-side.
   *
   * `grade` is passed through and enforced inside the Python registry, so the
   * ladder holds even if the UI is tampered with in a console.
   */
  run(tool: string, args: Record<string, unknown>, grade: number): Promise<BoundResult> {
    return this.send<BoundResult>('run', { tool, arguments: args, grade })
  }

  /** The multiple-comparison ledger for one dataset, from local transcripts. */
  ledger(dataId: string): Promise<{
    tests_run_so_far: number
    correction_applied: string | null
    entries: Array<{ transcript_id: string; tool: string; p_raw: number; p_adjusted: number; still_significant: boolean }>
    message: string
    severity: string
  }> {
    return this.send('ledger', { dataId })
  }

  /** Check arbitrary prose against locally computed numbers. */
  scan(text: string): Promise<{ passed: boolean; message: string; unmatched: Array<{ token: string }> }> {
    return this.send('scan', { text })
  }

  reset(): Promise<unknown> {
    return this.send('reset', {})
  }
}

export const engine = new Engine()

/** Format a number the way the platform does everywhere: never more precision
 *  than the computation justifies, and never a bare 0 for a small p-value. */
export function fmt(value: unknown, digits = 3): string {
  if (value === null || value === undefined) return '—'
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  if (value !== 0 && Math.abs(value) < 1e-4) return value.toExponential(2)
  if (Number.isInteger(value) && Math.abs(value) < 1e6) return value.toLocaleString()
  if (Math.abs(value) >= 1e6) return value.toExponential(2)
  return value.toFixed(digits)
}

export function fmtP(p: unknown): string {
  if (typeof p !== 'number' || !Number.isFinite(p)) return '—'
  if (p < 0.0001) return 'p < 0.0001'
  return `p = ${p < 0.001 ? p.toExponential(2) : p.toFixed(4)}`
}
