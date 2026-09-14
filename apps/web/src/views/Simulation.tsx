/**
 * Host-phage interaction simulation (specification section 03.3).
 *
 * "Runs the experiment they cannot run -- infects a virtual culture, changes
 *  the multiplicity of infection, watches the population crash and rebound,
 *  and discovers resistance."
 *
 * The sliders carry units and an explanation, and a value outside the range
 * real phage have been measured in produces a visible warning rather than a
 * silent answer. A simulation a student cannot tell apart from reality is a
 * worse teaching object than no simulation.
 *
 * All of this runs in the browser. The specification's line -- "the simulation
 * block costs us nothing per student" -- is only true if that is actually where
 * it runs, so it is.
 */

import { useEffect, useMemo, useState } from 'react'
import { Chart, seriesSpec, scatterFitSpec } from '../components/Chart'
import { ResultCard } from '../components/Result'
import { engine, fmt } from '../lib/engine'
import { useStore } from '../lib/store'

export function Simulation() {
  const { grade, catalogue, results, pushResult } = useStore()
  const models = catalogue?.simulation_models ?? {}
  const meta = catalogue?.simulation_parameters ?? {}

  const [model, setModel] = useState('lytic')
  const [params, setParams] = useState<Record<string, number>>({})
  const [s0, setS0] = useState(1e6)
  const [p0, setP0] = useState(1e4)
  const [hours, setHours] = useState(36)
  const [logY, setLogY] = useState(true)
  const [busy, setBusy] = useState(false)
  const [sweepParam, setSweepParam] = useState('adsorption')

  const spec = models[model]

  useEffect(() => {
    if (!spec) return
    const next: Record<string, number> = {}
    for (const p of spec.params) next[p] = meta[p]?.default ?? params[p] ?? 1
    setParams(next)
    setSweepParam(spec.params.find((p) => p !== 'n_stages') ?? spec.params[0])
  }, [model, catalogue])

  const mine = results.filter((r) => r.label.startsWith('simulation'))
  const moi = p0 / Math.max(s0, 1)

  async function runSim() {
    setBusy(true)
    try {
      const r = await engine.run('simulate', { model, params, S0: s0, P0: p0, hours }, grade)
      pushResult(`simulation · ${spec?.label ?? model}`, r)
    } finally {
      setBusy(false)
    }
  }

  async function runSweep() {
    setBusy(true)
    try {
      const r = await engine.run('parameter_sweep',
        { model, parameter: sweepParam, n_values: 9, base_params: params }, grade)
      pushResult(`simulation · sweep of ${sweepParam}`, r)
    } finally {
      setBusy(false)
    }
  }

  async function runOneStep() {
    setBusy(true)
    try {
      const r = await engine.run('one_step_growth', { latent: params.latent ?? 0.5, burst: params.burst ?? 100, n_stages: 20 }, grade)
      pushResult('simulation · one-step growth', r)
    } finally {
      setBusy(false)
    }
  }

  async function runPlaque() {
    setBusy(true)
    try {
      const r = await engine.run('plaque_growth', { size: 121, steps: 100 }, grade)
      pushResult('simulation · plaque on a lawn', r)
    } finally {
      setBusy(false)
    }
  }

  if (!catalogue) return <p className="busy">Loading the models…</p>

  return (
    <div className="lab">
      <section className="panel">
        <h2>The experiment you cannot run</h2>
        <div className="row wrap">
          {Object.entries(models).map(([key, m]) => (
            <button
              key={key}
              className={key === model ? 'chip on' : 'chip'}
              onClick={() => setModel(key)}
              title={m.teaches}
            >
              {m.label} <span className="muted small">G{m.grades}</span>
            </button>
          ))}
        </div>
        {spec ? <p className="teaches">{spec.teaches}</p> : null}
      </section>

      <section className="panel">
        <h2>Set it up</h2>
        <div className="row wrap">
          <LogField label="Bacteria at the start" unit="cells/mL" value={s0} onChange={setS0} />
          <LogField label="Phage at the start" unit="phage/mL" value={p0} onChange={setP0} />
          <label>
            hours
            <input type="number" min={1} max={200} value={hours}
                   onChange={(e) => setHours(Number(e.target.value))} />
          </label>
        </div>
        <p className={`banner ${moi > 100 || moi < 1e-6 ? 'warn' : 'info'}`}>
          <strong>MOI {fmt(moi)}</strong> — {moi >= 1
            ? 'more phage than bacteria. Almost every cell is hit at once, so you see one round of lysis and then nothing.'
            : 'fewer phage than bacteria. The phage must go through several rounds to clear the culture, which is the more interesting case.'}
        </p>

        <div className="sliders">
          {(spec?.params ?? []).map((p) => (
            <Slider
              key={p}
              name={p}
              meta={meta[p]}
              value={params[p] ?? meta[p]?.default ?? 0}
              onChange={(v) => setParams((s) => ({ ...s, [p]: v }))}
            />
          ))}
        </div>

        <div className="row wrap">
          <button className="primary" onClick={runSim} disabled={busy}>Run it</button>
          <label>
            sweep
            <select value={sweepParam} onChange={(e) => setSweepParam(e.target.value)}>
              {(spec?.params ?? []).map((p) => (
                <option key={p} value={p}>{meta[p]?.label ?? p}</option>
              ))}
            </select>
          </label>
          <button onClick={runSweep} disabled={busy}>Sweep it (G8-L12 style)</button>
          <button onClick={runOneStep} disabled={busy}>One-step growth curve</button>
          <button onClick={runPlaque} disabled={busy}>Grow a plaque</button>
          <label className="inline">
            <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} />
            log scale
          </label>
        </div>
      </section>

      {busy ? <p className="busy">Solving…</p> : null}

      <section className="results">
        {mine.map((entry) => (
          <ResultCard key={entry.id} entry={entry}>
            {entry.result.series ? (
              <Chart
                spec={seriesSpec(entry.result.time as number[], entry.result.series as any,
                  { logY, yTitle: 'per mL' })}
                height={280}
              />
            ) : null}
            {entry.result.response ? (
              <Chart
                spec={scatterFitSpec(
                  (entry.result.response as any).x,
                  (entry.result.response as any).y,
                  undefined,
                  { x: String(entry.result.parameter_label ?? ''), y: 'log10 amplification' },
                )}
              />
            ) : null}
            {entry.result.runs ? <OneStep runs={entry.result.runs as any} /> : null}
            {entry.result.frames ? <PlaqueCanvas result={entry.result as any} /> : null}
            {entry.result.summary ? <SimSummary summary={entry.result.summary as any} /> : null}
          </ResultCard>
        ))}
      </section>
    </div>
  )
}

function Slider({ name, meta, value, onChange }: {
  name: string; meta: any; value: number; onChange: (v: number) => void
}) {
  if (!meta) return null
  const isLog = !!meta.log
  const toSlider = (v: number) =>
    isLog ? (Math.log10(Math.max(v, meta.min || 1e-12)) - Math.log10(meta.min || 1e-12)) /
            (Math.log10(meta.max) - Math.log10(meta.min || 1e-12)) * 100
          : ((v - meta.min) / (meta.max - meta.min)) * 100
  const fromSlider = (s: number) =>
    isLog ? Math.pow(10, Math.log10(meta.min || 1e-12) +
            (s / 100) * (Math.log10(meta.max) - Math.log10(meta.min || 1e-12)))
          : meta.min + (s / 100) * (meta.max - meta.min)

  const [lo, hi] = meta.typical ?? [meta.min, meta.max]
  const outside = value < lo || value > hi

  return (
    <div className={`slider${outside ? ' outside' : ''}`}>
      <div className="shead">
        <label htmlFor={name}>{meta.label}</label>
        <output>
          {isLog ? value.toExponential(2) : fmt(value, meta.integer ? 0 : 3)}{' '}
          <span className="muted">{meta.unit}</span>
        </output>
      </div>
      <input
        id={name} type="range" min={0} max={100} step={0.5}
        value={toSlider(value)}
        onChange={(e) => {
          const v = fromSlider(Number(e.target.value))
          onChange(meta.integer ? Math.round(v) : v)
        }}
      />
      <p className="small muted">{meta.explain}</p>
      {outside ? (
        <p className="small warnText">
          Outside the range measured for real phage ({isLog ? lo.toExponential(0) : lo}–
          {isLog ? hi.toExponential(0) : hi}). The simulation will still run — that is what a
          simulation is for — but do not report the result as what a real phage does.
        </p>
      ) : null}
    </div>
  )
}

function LogField({ label, unit, value, onChange }: {
  label: string; unit: string; value: number; onChange: (v: number) => void
}) {
  return (
    <label>
      {label} <span className="muted small">{unit}</span>
      <input
        type="range" min={2} max={11} step={0.1}
        value={Math.log10(Math.max(value, 100))}
        onChange={(e) => onChange(Math.pow(10, Number(e.target.value)))}
      />
      <output className="small">{value.toExponential(2)}</output>
    </label>
  )
}

function SimSummary({ summary }: { summary: any }) {
  const rows: Array<[string, string]> = [
    ['lowest the bacteria got', `${fmt(summary.nadir_host)} /mL at ${fmt(summary.nadir_time, 1)} h`],
    ['phage multiplied by', `10^${fmt(summary.log10_amplification, 1)}`],
    ['peak phage', `${fmt(summary.peak_phage)} /mL at ${fmt(summary.peak_phage_time, 1)} h`],
    ['bacteria at the end', fmt(summary.final_host)],
  ]
  if (summary.final_resistant_fraction != null) {
    rows.push(['of those, resistant', `${Math.round(summary.final_resistant_fraction * 100)}%`])
  }
  return (
    <div className="tscroll">
      <table>
        <tbody>{rows.map(([k, v]) => <tr key={k}><td>{k}</td><td className="num">{v}</td></tr>)}</tbody>
      </table>
    </div>
  )
}

function OneStep({ runs }: { runs: Record<string, any> }) {
  const keys = Object.keys(runs)
  const time = runs[keys[0]].time as number[]
  const series: Record<string, number[]> = {}
  for (const k of keys) series[`${runs[k].n_stages} stage${runs[k].n_stages === 1 ? '' : 's'}`] = runs[k].free_phage
  return <Chart spec={seriesSpec(time, series, { logY: true, yTitle: 'free phage' })} height={260} />
}

/** The plaque lattice, drawn frame by frame. */
function PlaqueCanvas({ result }: { result: any }) {
  const frames: number[][][] = result.frames
  const [i, setI] = useState(frames.length - 1)
  const size = result.size as number
  const px = Math.max(1, Math.floor(340 / size))

  const dataUrl = useMemo(() => {
    const colours = ['#DCEBE8', '#A8481F', '#16211F', '#8A6A18'] // lawn, infected, lysed, resistant
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')!
    const img = ctx.createImageData(size, size)
    const grid = frames[Math.min(i, frames.length - 1)]
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const hex = colours[grid[r][c]] ?? '#000'
        const n = (r * size + c) * 4
        img.data[n] = parseInt(hex.slice(1, 3), 16)
        img.data[n + 1] = parseInt(hex.slice(3, 5), 16)
        img.data[n + 2] = parseInt(hex.slice(5, 7), 16)
        img.data[n + 3] = 255
      }
    }
    ctx.putImageData(img, 0, 0)
    return canvas.toDataURL()
  }, [i, frames, size])

  return (
    <div className="plaque">
      <img src={dataUrl} width={size * px} height={size * px}
           style={{ imageRendering: 'pixelated' }} alt="simulated bacterial lawn" />
      <input type="range" min={0} max={frames.length - 1} value={i}
             onChange={(e) => setI(Number(e.target.value))} />
      <p className="small muted">
        step {i * (result.frame_every ?? 1)} of {result.steps_run} ·{' '}
        <span style={{ color: '#DCEBE8', background: '#16211F', padding: '0 4px' }}>lawn</span>{' '}
        <span style={{ color: '#A8481F' }}>infected</span>{' '}
        <span style={{ color: '#16211F' }}>lysed</span>
      </p>
    </div>
  )
}
