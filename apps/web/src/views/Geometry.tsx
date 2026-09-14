/**
 * The geometry capstone (specification section 11) — Fingerprint, Islands, Bridges.
 *
 * The Islands module is a slider, and that is the whole design:
 *
 *   "The lesson is the plateau: a result that survives seventeen parameter
 *    values is a result; a result that appears at one value is an artefact of
 *    the setting. This single interactive slider teaches parameter sensitivity
 *    more effectively than any lecture."
 *
 * So the student does not get a k-NN graph at k=20 with an answer under it.
 * They get every k at once, and the answer is the flat part.
 */

import { useState } from 'react'
import { Chart, sweepSpec } from '../components/Chart'
import { ResultCard } from '../components/Result'
import { engine, fmt } from '../lib/engine'
import { useStore } from '../lib/store'

const DEMO_FASTA = `>Misomonster
${randomish('ACGT', 0, 6000)}
>TinyPebbles
${randomish('ACGT', 1, 6000)}`

export function Geometry() {
  const { grade, results, pushResult } = useStore()
  const [fasta, setFasta] = useState(DEMO_FASTA)
  const [matrix, setMatrix] = useState<{ names: string[]; D: number[][] } | null>(null)
  const [busy, setBusy] = useState<string | null>(null)

  const mine = results.filter((r) => r.label.startsWith('geometry'))

  async function fingerprint() {
    setBusy('Counting 4-letter words…')
    try {
      const parsed = parseFasta(fasta)
      if (!parsed.length) return
      const r = await engine.run('tetranucleotide_vector',
        { sequence: parsed[0].sequence, name: parsed[0].id }, grade)
      pushResult('geometry · fingerprint', r)
    } finally { setBusy(null) }
  }

  async function buildAndSweep() {
    setBusy('Building the distance matrix…')
    try {
      const parsed = parseFasta(fasta)
      if (parsed.length < 6) {
        setBusy(null)
        alert('The k sweep needs at least six genomes to mean anything. Load a larger set.')
        return
      }
      const vectors: number[][] = []
      const names: string[] = []
      for (const rec of parsed) {
        const { result } = await engine.run('tetranucleotide_vector',
          { sequence: rec.sequence, name: rec.id }, grade)
        if (result.frequencies) {
          vectors.push(result.frequencies as number[])
          names.push(rec.id)
        }
      }
      const dist = await engine.run('cosine_distance_matrix', { vectors, names }, grade)
      const D = dist.result.matrix as number[][]
      setMatrix({ names, D })

      setBusy('Sweeping k…')
      const sweep = await engine.run('component_sweep',
        { distance_matrix: D, names, k_min: 3, k_max: Math.min(40, names.length - 1) }, grade)
      pushResult('geometry · islands (k sweep)', sweep)
    } finally { setBusy(null) }
  }

  async function bridges() {
    if (!matrix) return
    setBusy('Computing curvature on every edge — this is the slow one…')
    try {
      const r = await engine.run('bridge_analysis',
        { distance_matrix: matrix.D, names: matrix.names,
          k_max: Math.min(40, matrix.names.length - 1) }, grade)
      pushResult('geometry · bridges (Ollivier-Ricci)', r)
    } finally { setBusy(null) }
  }

  return (
    <div className="lab">
      <section className="panel">
        <h2>Three modules, one pipeline</h2>
        <ol className="modules">
          <li>
            <strong>Fingerprint</strong> <span className="pill">G7–9</span>
            <p>
              Count every 4-letter word in a genome. 256 numbers. This is the barcode-matching
              lesson from Grade 5, grown up — and there is no package for it, because it is a
              counter over a sliding window.
            </p>
          </li>
          <li>
            <strong>Islands</strong> <span className="pill">G9–11</span>
            <p>
              Join each genome to its k closest relatives and sweep k. Watch the islands merge.
              The answer is not the picture at any one k — it is the range of k where the picture
              stops changing.
            </p>
          </li>
          <li>
            <strong>Bridges</strong> <span className="pill">G11–12</span>
            <p>
              Find the edges holding two worlds together, using Ollivier-Ricci curvature. Then go
              and look those genomes up. Geometry says where to look; biology has to say why.
              That question is genuinely open.
            </p>
          </li>
        </ol>
      </section>

      <section className="panel">
        <h2>Genomes</h2>
        <p className="muted">
          Paste FASTA. Two genomes is enough for a fingerprint; the k sweep needs at least six,
          and the published analysis this reproduces used 128.
        </p>
        <textarea value={fasta} onChange={(e) => setFasta(e.target.value)} rows={6} spellCheck={false} />
        <div className="row wrap">
          <button onClick={fingerprint} disabled={!!busy}>Fingerprint the first one</button>
          <button className="primary" onClick={buildAndSweep} disabled={!!busy}>
            Build the graph and sweep k
          </button>
          <button onClick={bridges} disabled={!!busy || !matrix}>Find the bridges</button>
        </div>
        {matrix ? (
          <p className="small muted">
            {matrix.names.length} genomes loaded; distance matrix is {matrix.names.length}×
            {matrix.names.length}.
          </p>
        ) : null}
      </section>

      {busy ? <p className="busy">{busy}</p> : null}

      <section className="results">
        {mine.map((entry) => (
          <ResultCard key={entry.id} entry={entry}>
            {entry.result.component_counts ? (
              <>
                <Chart
                  spec={sweepSpec(
                    entry.result.k_values as number[],
                    entry.result.component_counts as number[],
                    entry.result.plateau as any,
                  )}
                />
                {entry.result.plateau ? <Plateau plateau={entry.result.plateau as any} /> : null}
              </>
            ) : null}
            {entry.result.top_kmers ? <TopKmers result={entry.result as any} /> : null}
            {entry.result.unweighted ? <Bridges result={entry.result as any} /> : null}
          </ResultCard>
        ))}
      </section>
    </div>
  )
}

function Plateau({ plateau }: { plateau: any }) {
  return (
    <div className="banner info">
      <strong>
        The split into {plateau.n_components} groups of {plateau.sizes.join(' and ')} holds from
        k = {plateau.k_from} to k = {plateau.k_to}
      </strong>
      <p>
        That is {plateau.length} consecutive values of the one parameter you chose arbitrarily.
        A structure that survives {plateau.length} different settings is a property of the
        genomes; one that appears at a single setting is a property of the setting.
      </p>
      {plateau.members_named ? (
        <div className="clades">
          {plateau.members_named.map((members: string[], i: number) => (
            <details key={i}>
              <summary>group {i + 1} — {members.length} genomes</summary>
              <p className="small mono">{members.join(', ')}</p>
            </details>
          ))}
        </div>
      ) : null}
    </div>
  )
}

function TopKmers({ result }: { result: any }) {
  return (
    <div className="tscroll">
      <table>
        <thead><tr><th>4-mer</th><th className="num">frequency</th><th className="num">× chance</th></tr></thead>
        <tbody>
          {result.top_kmers.slice(0, 6).map((k: any) => (
            <tr key={k.kmer}>
              <td className="mono">{k.kmer}</td>
              <td className="num">{(k.frequency * 100).toFixed(2)}%</td>
              <td className="num">{fmt(k.frequency * 256, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Bridges({ result }: { result: any }) {
  const u = result.unweighted
  return (
    <>
      <div className="stats">
        <div className="stat primary">
          <div className="v">{u.n_bridge_edges}</div>
          <div className="l">bridge edges</div>
          <div className="n">first connection at k = {result.bridge_k}</div>
        </div>
        <div className="stat">
          <div className="v">{fmt(u.mean_curvature_bridge, 3)}</div>
          <div className="l">mean κ on the bridge</div>
          <div className="n">against {fmt(u.mean_curvature_interior, 3)} inside the clades</div>
        </div>
        {u.test ? (
          <div className="stat">
            <div className="v">{fmt(u.test.p_value)}</div>
            <div className="l">Mann-Whitney, cross &lt; within</div>
            <div className="n">{u.test.n_cross} bridge vs {u.test.n_within} interior edges</div>
          </div>
        ) : null}
      </div>

      {u.hubs?.length ? (
        <div className="banner info">
          <strong>The genomes to go and look up</strong>
          <ul>
            {u.hubs.slice(0, 4).map((h: any) => (
              <li key={h.index}>
                <span className="mono">{h.name ?? `#${h.index}`}</span> carries {h.bridge_edges} of
                the {u.n_bridge_edges} bridge edges
              </li>
            ))}
          </ul>
          <p>{result.teaches}</p>
        </div>
      ) : null}

      <div className="banner warn">
        <strong>Before you quote any of this</strong>
        <p>{result.weighting_caveat}</p>
      </div>
    </>
  )
}

function parseFasta(text: string): Array<{ id: string; sequence: string }> {
  const out: Array<{ id: string; sequence: string }> = []
  let id: string | null = null
  let parts: string[] = []
  const flush = () => {
    if (id !== null) out.push({ id, sequence: parts.join('') })
    id = null
    parts = []
  }
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim()
    if (!t) continue
    if (t.startsWith('>')) { flush(); id = t.slice(1).split(/\s+/)[0] || `seq${out.length + 1}` }
    else if (id !== null) parts.push(t.toUpperCase().replace(/[^ACGTU]/g, ''))
  }
  flush()
  return out
}

/** Deterministic filler so the demo box is never empty; not real data, and labelled as such. */
function randomish(alphabet: string, seed: number, n: number): string {
  let s = seed * 9301 + 49297
  let out = ''
  for (let i = 0; i < n; i++) {
    s = (s * 9301 + 49297) % 233280
    const bias = seed === 0 ? (s % 100 < 40 ? 0 : s % 4) : (s % 100 < 40 ? 2 : s % 4)
    out += alphabet[bias]
  }
  return out
}
