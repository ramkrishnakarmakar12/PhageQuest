/**
 * Charts.
 *
 * Vega-Lite, per the specification's reasoning (section 03.1):
 *
 *   "The chart is a JSON spec -- inspectable, diffable, and mappable
 *    one-to-one onto what the student said they wanted. That auditability is
 *    worth more here than Plotly's polish."
 *
 * So the spec is shown next to the chart, on request. A picture that cannot be
 * checked is the same problem as a number that cannot be checked.
 */
import { useEffect, useRef, useState } from 'react'
import type { TopLevelSpec as VisualizationSpec } from 'vega-lite'

/**
 * Vega is loaded on demand, not in the initial bundle.
 *
 * Specification section 02, constraint 2: "A shared lab PC, often on a metered
 * line." Vega plus Vega-Lite is most of a megabyte, and a Grade 5 entering
 * strip colours does not need it until the first chart is drawn. Splitting it
 * out takes the first paint from ~1.1 MB to ~180 KB, and the chart library then
 * arrives once and is cached.
 */
let embedPromise: Promise<typeof import('vega-embed')['default']> | null = null
function loadEmbed() {
  embedPromise ??= import('vega-embed').then((m) => m.default)
  return embedPromise
}

export function Chart({ spec, height = 240 }: { spec: VisualizationSpec; height?: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const [showSpec, setShowSpec] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!ref.current) return
    let view: { finalize: () => void } | null = null
    let cancelled = false
    loadEmbed()
      .then((embed) => {
        if (cancelled || !ref.current) return null
        return embed(ref.current, spec, { actions: false, renderer: 'canvas' })
      })
      .then((r) => {
        if (r) view = r.view
        setError(null)
      })
      .catch((e) => setError(String(e)))
    return () => {
      cancelled = true
      view?.finalize()
    }
  }, [spec])

  return (
    <div className="chart">
      {error ? <p className="muted small">Chart could not be drawn: {error}</p> : null}
      <div ref={ref} style={{ minHeight: height }} />
      <button className="linkish small" onClick={() => setShowSpec((v) => !v)}>
        {showSpec ? 'hide' : 'show'} the chart specification
      </button>
      {showSpec ? <pre className="spec">{JSON.stringify(spec, null, 2)}</pre> : null}
    </div>
  )
}

const FONT = 'ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif'

/** Shared look, so every chart in the platform reads as one system. */
export const base = {
  $schema: 'https://vega.github.io/schema/vega-lite/v5.json',
  config: {
    font: FONT,
    axis: { labelFontSize: 11, titleFontSize: 11, titleFontWeight: 600, grid: true,
            gridColor: 'rgba(128,144,140,0.18)', domainColor: 'rgba(128,144,140,0.4)',
            tickColor: 'rgba(128,144,140,0.4)', labelColor: '#5E706B', titleColor: '#33433F' },
    legend: { labelFontSize: 11, titleFontSize: 11, titleColor: '#33433F', labelColor: '#5E706B' },
    view: { stroke: null },
    range: { category: ['#136A66', '#A8481F', '#7A3563', '#8A6A18', '#4E7C72', '#2F6A61'] },
    background: 'transparent',
  },
} as const

export function groupsSpec(groups: Record<string, { mean: number; sd: number; n: number; mean_ci: number[] }>) {
  const values = Object.entries(groups).map(([name, d]) => ({
    group: name, mean: d.mean, lo: d.mean_ci?.[0] ?? d.mean, hi: d.mean_ci?.[1] ?? d.mean, n: d.n,
  }))
  return {
    ...base,
    data: { values },
    height: 200,
    layer: [
      { mark: { type: 'bar', cornerRadiusEnd: 2, opacity: 0.85 },
        encoding: {
          x: { field: 'group', type: 'nominal', title: null, axis: { labelAngle: 0 } },
          y: { field: 'mean', type: 'quantitative', title: 'mean' },
          color: { field: 'group', type: 'nominal', legend: null },
          tooltip: [{ field: 'group' }, { field: 'mean', format: '.4g' }, { field: 'n' }],
        } },
      { mark: { type: 'errorbar', ticks: true, color: '#16211F' },
        encoding: {
          x: { field: 'group', type: 'nominal' },
          y: { field: 'lo', type: 'quantitative', title: 'mean' },
          y2: { field: 'hi' },
        } },
    ],
  } as unknown as VisualizationSpec
}

export function nullDistributionSpec(nulls: number[], observed: number) {
  return {
    ...base,
    height: 190,
    layer: [
      { data: { values: nulls.map((v) => ({ v })) },
        mark: { type: 'bar', opacity: 0.75, color: '#4E7C72' },
        encoding: {
          x: { field: 'v', type: 'quantitative', bin: { maxbins: 40 },
               title: 'difference produced by chance alone' },
          y: { aggregate: 'count', type: 'quantitative', title: 'how many shuffles' },
        } },
      { data: { values: [{ v: observed }] },
        mark: { type: 'rule', color: '#A8481F', size: 2.5 },
        encoding: { x: { field: 'v', type: 'quantitative' } } },
      { data: { values: [{ v: observed, label: 'your result' }] },
        mark: { type: 'text', dy: -8, align: 'left', dx: 5, color: '#A8481F', fontWeight: 600 },
        encoding: { x: { field: 'v', type: 'quantitative' }, text: { field: 'label' },
                    y: { value: 10 } } },
    ],
  } as unknown as VisualizationSpec
}

export function seriesSpec(
  time: number[],
  series: Record<string, number[]>,
  opts: { logY?: boolean; xTitle?: string; yTitle?: string } = {},
) {
  const values = time.flatMap((t, i) =>
    Object.entries(series).map(([name, ys]) => ({ t, name, y: ys[i] })),
  )
  return {
    ...base,
    data: { values: values.filter((d) => Number.isFinite(d.y)) },
    height: 260,
    mark: { type: 'line', strokeWidth: 2, interpolate: 'monotone' },
    encoding: {
      x: { field: 't', type: 'quantitative', title: opts.xTitle ?? 'hours' },
      y: {
        field: 'y', type: 'quantitative', title: opts.yTitle ?? 'per mL',
        scale: opts.logY ? { type: 'symlog', constant: 1 } : {},
      },
      color: { field: 'name', type: 'nominal', title: null },
      tooltip: [{ field: 'name' }, { field: 't', format: '.2f' }, { field: 'y', format: '.3g' }],
    },
  } as unknown as VisualizationSpec
}

export function scatterFitSpec(
  x: number[], y: number[],
  curve?: { x: number[]; y: number[]; band_low: number[]; band_high: number[] },
  labels: { x?: string; y?: string } = {},
) {
  const layers: any[] = []
  if (curve?.band_low?.length) {
    layers.push({
      data: { values: curve.x.map((cx, i) => ({ x: cx, lo: curve.band_low[i], hi: curve.band_high[i] })) },
      mark: { type: 'area', opacity: 0.18, color: '#136A66' },
      encoding: {
        x: { field: 'x', type: 'quantitative' },
        y: { field: 'lo', type: 'quantitative' }, y2: { field: 'hi' },
      },
    })
  }
  if (curve?.x?.length) {
    layers.push({
      data: { values: curve.x.map((cx, i) => ({ x: cx, y: curve.y[i] })) },
      mark: { type: 'line', color: '#136A66', strokeWidth: 2 },
      encoding: { x: { field: 'x', type: 'quantitative' }, y: { field: 'y', type: 'quantitative' } },
    })
  }
  layers.push({
    data: { values: x.map((xv, i) => ({ x: xv, y: y[i] })) },
    mark: { type: 'point', filled: true, size: 70, color: '#A8481F' },
    encoding: {
      x: { field: 'x', type: 'quantitative', title: labels.x ?? 'x' },
      y: { field: 'y', type: 'quantitative', title: labels.y ?? 'y' },
      tooltip: [{ field: 'x', format: '.4g' }, { field: 'y', format: '.4g' }],
    },
  })
  return { ...base, height: 260, layer: layers } as unknown as VisualizationSpec
}

export function sweepSpec(ks: number[], counts: number[], plateau?: { k_from: number; k_to: number } | null) {
  const layers: any[] = []
  if (plateau) {
    layers.push({
      data: { values: [{ a: plateau.k_from, b: plateau.k_to }] },
      mark: { type: 'rect', opacity: 0.16, color: '#7A3563' },
      encoding: { x: { field: 'a', type: 'quantitative' }, x2: { field: 'b' } },
    })
  }
  layers.push({
    data: { values: ks.map((k, i) => ({ k, n: counts[i] })) },
    mark: { type: 'line', interpolate: 'step-after', strokeWidth: 2.5, color: '#7A3563',
            point: { filled: true, size: 40 } },
    encoding: {
      x: { field: 'k', type: 'quantitative', title: 'k (nearest neighbours)' },
      y: { field: 'n', type: 'quantitative', title: 'separate groups', scale: { domainMin: 0 } },
      tooltip: [{ field: 'k' }, { field: 'n', title: 'components' }],
    },
  })
  return { ...base, height: 220, layer: layers } as unknown as VisualizationSpec
}
