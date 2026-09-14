/**
 * Client state.
 *
 * `results` is an append-only log, never a "latest result" slot. That is a
 * deliberate constraint rather than a convenience: the multiple-comparison
 * ledger only means something if earlier analyses are still there, and a UI
 * that overwrites the previous answer quietly teaches the opposite of what the
 * platform is for.
 */

import { create } from 'zustand'
import type { BoundResult, Catalogue } from './engine'

export type Grade = 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12

export type LogEntry = BoundResult & {
  id: string
  at: number
  label: string
}

export type PreRegistration = {
  dataId: string
  question: string
  prediction: string
  direction: 'greater' | 'less' | 'two-sided'
  at: number
  version: number
}

export type Dataset = {
  id: string
  name: string
  template: string | null
  rows: Record<string, unknown>[]
  columns: string[]
  validation: any | null
}

type State = {
  grade: Grade
  booted: boolean
  bootMessage: string
  versions: Record<string, string>
  catalogue: Catalogue | null

  datasets: Dataset[]
  activeDatasetId: string | null

  results: LogEntry[]
  openTranscriptId: string | null

  preregs: PreRegistration[]

  setGrade: (g: Grade) => void
  setBoot: (booted: boolean, message?: string, versions?: Record<string, string>) => void
  setCatalogue: (c: Catalogue) => void

  addDataset: (d: Dataset) => void
  setActiveDataset: (id: string | null) => void

  pushResult: (label: string, r: BoundResult) => LogEntry
  openTranscript: (id: string | null) => void

  addPrereg: (p: Omit<PreRegistration, 'at' | 'version'>) => PreRegistration
  preregsFor: (dataId: string) => PreRegistration[]

  /** Every inferential result this session, on this dataset. Drives the ledger badge. */
  testsOn: (dataId: string | null) => LogEntry[]
}

let counter = 0

export const useStore = create<State>((set, get) => ({
  grade: 8,
  booted: false,
  bootMessage: 'Starting…',
  versions: {},
  catalogue: null,

  datasets: [],
  activeDatasetId: null,

  results: [],
  openTranscriptId: null,

  preregs: [],

  setGrade: (grade) => set({ grade, catalogue: null }),
  setBoot: (booted, bootMessage, versions) =>
    set((s) => ({
      booted,
      bootMessage: bootMessage ?? s.bootMessage,
      versions: versions ?? s.versions,
    })),
  setCatalogue: (catalogue) => set({ catalogue }),

  addDataset: (d) => set((s) => ({ datasets: [...s.datasets, d], activeDatasetId: d.id })),
  setActiveDataset: (activeDatasetId) => set({ activeDatasetId }),

  pushResult: (label, r) => {
    const entry: LogEntry = { ...r, id: `r${++counter}`, at: Date.now(), label }
    set((s) => ({ results: [entry, ...s.results] }))
    return entry
  },
  openTranscript: (openTranscriptId) => set({ openTranscriptId }),

  addPrereg: (p) => {
    const existing = get().preregs.filter((x) => x.dataId === p.dataId)
    const entry: PreRegistration = { ...p, at: Date.now(), version: existing.length + 1 }
    set((s) => ({ preregs: [...s.preregs, entry] }))
    return entry
  },
  preregsFor: (dataId) =>
    get()
      .preregs.filter((p) => p.dataId === dataId)
      .sort((a, b) => a.version - b.version),

  testsOn: (dataId) =>
    dataId
      ? get().results.filter(
          (r) => r.transcript?.data_id === dataId && r.result.p_value != null,
        )
      : [],
}))
