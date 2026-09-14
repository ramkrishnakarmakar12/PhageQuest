/**
 * Data exploration (specification section 03.1) and the statistical core (04).
 *
 * The flow is fixed and the order is the point:
 *
 *     paste your table  ->  it is validated, row by row
 *     ->  look at it     (no p-values anywhere)
 *     ->  write down what you expect
 *     ->  only then, test it
 *
 * The "test it" button is disabled until a prediction exists. That is the
 * specification's point 3 made structural rather than advisory: "G7-L9 and
 * G8-L9 both make students write a prediction before observing... The platform
 * simply records it and holds them to it."
 */

import { useMemo, useState } from 'react'
import { Chart, groupsSpec, nullDistributionSpec, scatterFitSpec } from '../components/Chart'
import { ResultCard } from '../components/Result'
import { engine, fmt } from '../lib/engine'
import { useStore, type Dataset } from '../lib/store'

const SAMPLE = `level,replicate,measurement
pH 5,1,11.2
pH 5,2,9.8
pH 5,3,12.1
pH 5,4,10.4
pH 7,1,18.3
pH 7,2,20.1
pH 7,3,19.4
pH 7,4,21.0
pH 9,1,12.9
pH 9,2,11.4
pH 9,3,13.8
pH 9,4,10.9`

export function DataLab() {
  const { grade, datasets, activeDatasetId, setActiveDataset, addDataset,
          results, pushResult, addPrereg, preregsFor } = useStore()
  const [raw, setRaw] = useState(SAMPLE)
  const [busy, setBusy] = useState<string | null>(null)
  const [groupCol, setGroupCol] = useState('')
  const [valueCol, setValueCol] = useState('')
  const [prediction, setPrediction] = useState('')
  const [direction, setDirection] = useState<'greater' | 'less' | 'two-sided'>('two-sided')
  const [justification, setJustification] = useState('')

  const dataset = datasets.find((d) => d.id === activeDatasetId) ?? null
  const dataId = useMemo(
    () => results.find((r) => r.label.includes(dataset?.name ?? '###'))?.transcript?.data_id ?? null,
    [results, dataset],
  )
  const preregs = dataId ? preregsFor(dataId) : []
  const canTest = preregs.length > 0

  const myResults = results.filter((r) => r.label.startsWith(dataset?.name ?? '###'))
  const ledgerCount = myResults.filter((r) => r.result.p_value != null).length

  async function load() {
    setBusy('Checking your table…')
    try {
      const { rows, columns } = parseTable(raw)
      const { result } = await engine.run('validate_table', { rows }, grade)
      const ds: Dataset = {
        id: `d${Date.now()}`, name: `dataset ${datasets.length + 1}`,
        template: (result.template as string) ?? null,
        rows, columns, validation: result,
      }
      addDataset(ds)
      const numeric = columns.filter((c) => rows.every((r) => typeof r[c] === 'number'))
      const categorical = columns.filter((c) => !numeric.includes(c))
      setGroupCol(categorical[0] ?? columns[0] ?? '')
      setValueCol(numeric[numeric.length - 1] ?? columns[1] ?? '')
    } finally {
      setBusy(null)
    }
  }

  function groupsFromDataset(): Record<string, number[]> {
    if (!dataset) return {}
    const out: Record<string, number[]> = {}
    for (const row of dataset.rows) {
      const g = String(row[groupCol] ?? 'all')
      const v = Number(row[valueCol])
      if (Number.isFinite(v)) (out[g] ??= []).push(v)
    }
    return out
  }

  async function run(tool: string, args: Record<string, unknown>, label: string) {
    setBusy(`Running ${tool}…`)
    try {
      const r = await engine.run(tool, args, grade)
      pushResult(`${dataset?.name ?? 'data'} · ${label}`, r)
    } finally {
      setBusy(null)
    }
  }

  const groups = groupsFromDataset()
  const groupNames = Object.keys(groups)

  return (
    <div className="lab">
      <section className="panel">
        <h2>1 · Your table</h2>
        <p className="muted">
          Paste what your class measured — CSV, or copied straight out of a spreadsheet.
          Every row is checked before any statistics run.
        </p>
        <textarea value={raw} onChange={(e) => setRaw(e.target.value)} rows={10} spellCheck={false} />
        <div className="row">
          <button className="primary" onClick={load} disabled={!!busy}>
            Check and load
          </button>
          {datasets.length > 1 ? (
            <select value={activeDatasetId ?? ''} onChange={(e) => setActiveDataset(e.target.value)}>
              {datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          ) : null}
        </div>

        {dataset?.validation ? <Validation report={dataset.validation} /> : null}
      </section>

      {dataset ? (
        <>
          <section className="panel">
            <h2>2 · Look at it first</h2>
            <p className="muted">
              No test yet. What do the numbers actually look like, and how much do repeats
              of the same condition disagree with each other?
            </p>
            <div className="row">
              <label>
                condition
                <select value={groupCol} onChange={(e) => setGroupCol(e.target.value)}>
                  {dataset.columns.map((c) => <option key={c}>{c}</option>)}
                </select>
              </label>
              <label>
                measurement
                <select value={valueCol} onChange={(e) => setValueCol(e.target.value)}>
                  {dataset.columns.map((c) => <option key={c}>{c}</option>)}
                </select>
              </label>
              <button onClick={() => run('describe', { groups }, 'summary')} disabled={!!busy}>
                Summarise
              </button>
            </div>
            <p className="small muted">
              {groupNames.length} condition{groupNames.length === 1 ? '' : 's'}:{' '}
              {groupNames.map((g) => `${g} (n=${groups[g].length})`).join(', ') || '—'}
            </p>
          </section>

          <section className="panel">
            <h2>3 · Say what you expect</h2>
            {preregs.length === 0 ? (
              <p className="muted">
                Before running a test, write down what you think will happen. Not as a formality:
                a prediction made after seeing the answer is not a prediction, and the difference
                between those two is most of what makes something science.
              </p>
            ) : (
              <ol className="preregs">
                {preregs.map((p) => (
                  <li key={p.version}>
                    <span className="pill">{p.version === 1 ? 'prediction' : `amendment ${p.version}`}</span>{' '}
                    {p.prediction} <span className="muted small">({p.direction})</span>
                  </li>
                ))}
              </ol>
            )}
            <div className="row">
              <input
                placeholder="I think pH 7 will grow the most, because…"
                value={prediction}
                onChange={(e) => setPrediction(e.target.value)}
              />
              <select value={direction} onChange={(e) => setDirection(e.target.value as any)}>
                <option value="two-sided">just different</option>
                <option value="greater">higher</option>
                <option value="less">lower</option>
              </select>
              <button
                onClick={() => {
                  if (!prediction.trim()) return
                  addPrereg({
                    dataId: dataId ?? dataset.id,
                    question: `${valueCol} across ${groupCol}`,
                    prediction, direction,
                  })
                  setPrediction('')
                }}
                disabled={!prediction.trim()}
              >
                Lock it in
              </button>
            </div>
            {preregs.length > 1 ? (
              <p className="banner warn">
                <strong>Amended after the fact.</strong> Both versions stay in the record and your
                teacher sees them. Your result is judged against the first one.
              </p>
            ) : null}
          </section>

          <section className="panel">
            <h2>4 · Now test it</h2>
            {!canTest ? (
              <p className="banner warn">Write a prediction above first.</p>
            ) : null}
            <label className="full">
              Why this test, on this data? (logged, and shown to your teacher)
              <input
                placeholder="Four independent conditions, four replicates each, measuring growth."
                value={justification}
                onChange={(e) => setJustification(e.target.value)}
              />
            </label>
            <div className="row wrap">
              <button
                className="primary"
                disabled={!canTest || !justification.trim() || groupNames.length < 2 || !!busy}
                onClick={() => run('compare_groups', { groups, justification }, 'comparison')}
              >
                Are these really different?
              </button>
              {groupNames.length === 2 ? (
                <button
                  disabled={!canTest || !justification.trim() || !!busy}
                  onClick={() =>
                    run('shuffle_test',
                      { group_a: groups[groupNames[0]], group_b: groups[groupNames[1]], justification },
                      'shuffle test')
                  }
                >
                  Show me the shuffling
                </button>
              ) : null}
              <button
                disabled={!!busy}
                onClick={() => run('n_for_anova',
                  { effect_size_f: 0.4, k_groups: Math.max(2, groupNames.length) },
                  'sample size needed')}
              >
                How many did we need?
              </button>
            </div>
            {ledgerCount >= 2 ? (
              <p className="small muted">
                {ledgerCount} tests run on this dataset so far. From three, corrections apply
                automatically and the count is shown to your teacher.
              </p>
            ) : null}
          </section>
        </>
      ) : null}

      {busy ? <p className="busy">{busy}</p> : null}

      <section className="results">
        {myResults.map((entry) => (
          <ResultCard key={entry.id} entry={entry}>
            {entry.result.groups ? (
              <Chart spec={groupsSpec(entry.result.groups as any)} />
            ) : null}
            {entry.result.descriptives && entry.result.p_value != null ? (
              <Chart spec={groupsSpec(entry.result.descriptives as any)} />
            ) : null}
            {Array.isArray(entry.result.null_distribution) ? (
              <Chart
                spec={nullDistributionSpec(
                  entry.result.null_distribution as number[],
                  entry.result.observed as number,
                )}
              />
            ) : null}
            {entry.result.curve ? (
              <Chart spec={scatterFitSpec([], [], entry.result.curve as any)} />
            ) : null}
          </ResultCard>
        ))}
      </section>
    </div>
  )
}

function Validation({ report }: { report: any }) {
  const issues = (report.issues ?? []) as any[]
  return (
    <div className={`banner ${report.valid ? 'info' : 'block'}`}>
      <strong>{report.plain_language}</strong>
      {issues.length ? (
        <ul className="issues">
          {issues.slice(0, 8).map((i, n) => (
            <li key={n} className={i.severity}>
              {i.row ? <code>row {i.row}</code> : null} {i.message}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  )
}

/** Tolerant CSV/TSV parsing. A student pastes from a spreadsheet, not from a file format. */
function parseTable(text: string): { rows: Record<string, unknown>[]; columns: string[] } {
  const lines = text.trim().split(/\r?\n/).filter((l) => l.trim())
  if (!lines.length) return { rows: [], columns: [] }
  const delim = lines[0].includes('\t') ? '\t' : lines[0].includes(';') ? ';' : ','
  const columns = lines[0].split(delim).map((c) => c.trim())
  const rows = lines.slice(1).map((line) => {
    const cells = line.split(delim)
    const row: Record<string, unknown> = {}
    columns.forEach((c, i) => {
      const cell = (cells[i] ?? '').trim()
      const num = Number(cell)
      row[c] = cell !== '' && Number.isFinite(num) ? num : cell
    })
    return row
  })
  return { rows, columns }
}

export { fmt }
