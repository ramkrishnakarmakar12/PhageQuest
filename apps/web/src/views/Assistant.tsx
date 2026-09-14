/**
 * The assistant (specification section 06).
 *
 * Two things about this screen are unusual and both are deliberate.
 *
 * First, the guard report is shown, not hidden. When the numeric scan blocks a
 * reply the student sees that it was blocked and why. A safety mechanism the
 * user never sees is a safety mechanism the user never learns to expect.
 *
 * Second, the transcript panel sits beside every answer. The specification's
 * claim is that "the transcript is the product"; a UI that tucks it behind a
 * developer menu does not make that claim, it just holds the opinion.
 */

import { useRef, useState } from 'react'
import { TranscriptToggle } from '../components/Result'
import { useStore } from '../lib/store'

type Message = {
  role: 'student' | 'assistant'
  text: string
  blocked?: boolean
  reasons?: string[]
  transcripts?: any[]
  guards?: Record<string, any>
  redactions?: string[]
}

const STARTERS = [
  'control: 12, 14, 11, 13, 12, 15 and treatment: 18, 21, 19, 20, 22, 19 — is the treatment bigger?',
  'We plated 7 plaques from a 1:1000000 dilution. What is the titre?',
  'Why did one of our plates have no plaques at all?',
  'How many replicates do we need to see a difference this size?',
]

export function Assistant() {
  const { grade } = useStore()
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const boxRef = useRef<HTMLDivElement>(null)

  async function send(text: string) {
    if (!text.trim() || busy) return
    setMessages((m) => [...m, { role: 'student', text }])
    setInput('')
    setBusy(true)
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ...authHeader() },
        body: JSON.stringify({ message: text }),
      })
      if (!res.ok) throw new Error(`${res.status}`)
      const body = await res.json()
      setMessages((m) => [...m, {
        role: 'assistant', text: body.reply, blocked: body.blocked,
        reasons: body.block_reasons, transcripts: body.transcripts,
        guards: body.guards, redactions: body.redactions,
      }])
    } catch (e) {
      setMessages((m) => [...m, {
        role: 'assistant',
        text: 'I could not reach the assistant service. Everything else on this platform — the '
            + 'statistics, the simulations, the geometry — runs in your browser and still works. '
            + 'The assistant is the one part that needs the server.',
        blocked: true, reasons: ['offline'],
      }])
    } finally {
      setBusy(false)
      requestAnimationFrame(() => boxRef.current?.scrollTo({ top: 1e9, behavior: 'smooth' }))
    }
  }

  return (
    <div className="chat">
      <section className="panel intro">
        <h2>Ask about your data</h2>
        <p className="muted">
          This assistant chooses which test to run and explains the result. It does not compute
          anything itself, and it cannot: every number it writes is checked against the actual
          calculation before you see it, and a number that does not match is stripped. If you see
          a blocked reply below, that mechanism working is what you are looking at.
        </p>
        <p className="small muted">
          It will also ask you what you expect before it runs a test, and it will not help you try
          analyses until one of them works. Both of those are on purpose.
        </p>
      </section>

      <div className="messages" ref={boxRef}>
        {messages.length === 0 ? (
          <div className="starters">
            {STARTERS.map((s) => (
              <button key={s} className="starter" onClick={() => send(s)}>{s}</button>
            ))}
          </div>
        ) : null}

        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}${m.blocked ? ' blocked' : ''}`}>
            {m.role === 'assistant' && m.blocked ? (
              <div className="blockhead">
                <span className="pill warn">held back</span>
                {(m.reasons ?? []).map((r) => (
                  <span key={r} className="pill">{r.replace(/_/g, ' ')}</span>
                ))}
              </div>
            ) : null}

            <p>{m.text}</p>

            {m.redactions?.length ? (
              <p className="small muted">
                Removed before this left your device: {m.redactions.join(', ')}. Under the DPDP
                Act every user here is legally a child, so identifying details are stripped from
                anything sent to a language model.
              </p>
            ) : null}

            {m.transcripts?.length ? (
              <div className="attached">
                {m.transcripts.map((t) => (
                  <TranscriptToggle key={t.transcript_id} transcript={t as any} />
                ))}
              </div>
            ) : null}

            {m.guards ? <Guards guards={m.guards} /> : null}
          </div>
        ))}

        {busy ? <div className="msg assistant"><p className="muted">Working…</p></div> : null}
      </div>

      <form
        className="composer"
        onSubmit={(e) => { e.preventDefault(); send(input) }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={`Grade ${grade} — paste your numbers, or ask a question`}
          disabled={busy}
        />
        <button className="primary" disabled={busy || !input.trim()}>Ask</button>
      </form>
    </div>
  )
}

function Guards({ guards }: { guards: Record<string, any> }) {
  const scan = guards.numeric_scan
  const claims = guards.claims
  const behaviour = guards.specification_search_behaviour
  const interesting = (scan && !scan.passed) || (claims && !claims.passed) || behaviour?.detected
  if (!interesting && !scan) return null

  return (
    <details className="guards">
      <summary className={interesting ? 'warnText' : 'muted'}>
        {interesting ? 'what the platform stopped, and why' : `checks passed (${scan?.n_checked ?? 0} numbers verified)`}
      </summary>
      <ul>
        {scan ? (
          <li className={scan.passed ? '' : 'warnText'}>
            <strong>numbers</strong>: {scan.message}
          </li>
        ) : null}
        {claims && !claims.passed
          ? claims.violations.map((v: any) => (
              <li key={v.id} className="warnText"><strong>{v.id.replace(/_/g, ' ')}</strong>: {v.message}</li>
            ))
          : null}
        {behaviour?.detected ? (
          <li className="warnText"><strong>repeated testing</strong>: {behaviour.response}</li>
        ) : null}
      </ul>
    </details>
  )
}

function authHeader(): Record<string, string> {
  const token = localStorage.getItem('phagequest.token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}
