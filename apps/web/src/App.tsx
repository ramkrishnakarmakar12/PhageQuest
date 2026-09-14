/**
 * Shell.
 *
 * The grade selector is not a cosmetic preference. It changes which tools the
 * Python registry will execute at all (specification section 05: "the same
 * engine, exposed at six different depths"), and the registry enforces it, so
 * switching it here genuinely changes what the platform can do rather than what
 * it shows.
 */
import { useEffect, useState } from 'react'
import { engine } from './lib/engine'
import { useStore, type Grade } from './lib/store'
import { DataLab } from './views/DataLab'
import { Simulation } from './views/Simulation'
import { Geometry } from './views/Geometry'
import { Assistant } from './views/Assistant'

const TABS = [
  { id: 'data', label: 'Data', min: 5, blurb: 'Your table, summarised and tested' },
  { id: 'simulate', label: 'Simulate', min: 6, blurb: 'The experiment you cannot run' },
  { id: 'geometry', label: 'Geometry', min: 9, blurb: 'Fingerprints, islands and bridges' },
  { id: 'assistant', label: 'Ask', min: 5, blurb: 'A chatbot that cannot invent a number' },
] as const

export function App() {
  const { grade, setGrade, booted, bootMessage, versions, setBoot, setCatalogue, results } = useStore()
  const [tab, setTab] = useState<string>('data')

  useEffect(() => {
    const off = engine.onBoot((s) => setBoot(false, s.message))
    engine
      .start()
      .then((v) => setBoot(true, 'ready', v))
      .catch((e) => setBoot(false, `Could not start the engine: ${e.message}`))
    return off
  }, [])

  useEffect(() => {
    if (!booted) return
    engine.catalogue(grade).then(setCatalogue)
  }, [booted, grade])

  const visible = TABS.filter((t) => grade >= t.min)
  const active = visible.find((t) => t.id === tab) ?? visible[0]

  return (
    <div className="app">
      <header className="mast">
        <div className="strata"><i /><i /><i /><i /></div>
        <div className="mastrow">
          <div>
            <p className="eyebrow">PhageQuest · Bioinformatics &amp; Simulation Discovery Platform</p>
            <h1>Every number is bound to the computation that produced it.</h1>
          </div>
          <label className="gradepick">
            Grade
            <select value={grade} onChange={(e) => setGrade(Number(e.target.value) as Grade)}>
              {[5, 6, 7, 8, 9, 10, 11, 12].map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          </label>
        </div>

        <nav className="tabs">
          {visible.map((t) => (
            <button key={t.id} className={t.id === active?.id ? 'tab on' : 'tab'}
                    onClick={() => setTab(t.id)}>
              <span>{t.label}</span>
              <em>{t.blurb}</em>
            </button>
          ))}
        </nav>
      </header>

      <main>
        {!booted ? (
          <div className="boot">
            <p className="busy">{bootMessage}</p>
            <p className="muted small">
              Python, numpy and scipy are loading into this tab. They are cached after the first
              visit, and once they are here everything except the assistant works with no internet
              at all — which is also why none of your data leaves this device.
            </p>
          </div>
        ) : active?.id === 'data' ? <DataLab />
          : active?.id === 'simulate' ? <Simulation />
          : active?.id === 'geometry' ? <Geometry />
          : <Assistant />}
      </main>

      <footer>
        <div className="strata"><i /><i /><i /><i /></div>
        <p>
          {booted ? (
            <>
              Running <strong>in this browser</strong>: scipy {versions.scipy}, numpy {versions.numpy},
              Python {versions.python} ({versions.runtime}). {results.length} execution record
              {results.length === 1 ? '' : 's'} this session.
            </>
          ) : 'Starting the engine…'}
        </p>
        <p className="small muted">
          Student data stays on this device unless you share a workspace. Under the DPDP Act 2023
          every user under 18 is a child, and that is this entire platform's audience.
        </p>
      </footer>
    </div>
  )
}
