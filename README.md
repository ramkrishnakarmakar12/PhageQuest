# PhageQuest Discovery Platform

Authentic phage genomics and an honest, grade-appropriate statistical reasoning
layer, in the same place. Built from the Block 2 technical specification for
Cheenta / Paninieight.

The design rule the specification sets, and that this repository follows:

> Build the statistics layer first and the bioinformatics on top of it — not the
> other way round. Every competitor did it the other way round, which is exactly
> why they all have the same hole.

---

## The invariant

> No numeric statistical claim may be rendered to a user unless it is bound to a
> specific execution record — code, engine version, raw output, timestamp —
> produced by a real statistics engine.

This is enforced in four places, not documented in one:

| Where | What it does |
|---|---|
| `engine/src/phagequest_engine/transcript.py` | Every engine function is wrapped by `@bound`, which records the call, the data fingerprint, the library versions and the raw output, and returns a `transcript_id`. A result cannot exist without one. |
| `engine/src/phagequest_engine/guards.py` | `numeric_scan` parses model prose for `p =`, `t(…)`, effect sizes, intervals and percentages, and cross-checks every one against the turn's execution records. An unmatched number blocks or is redacted. |
| `apps/api/app/main.py` | Transcripts posted from the browser are **re-executed server-side and compared**. A mismatch is stored flagged, never silently corrected. |
| `apps/web/src/components/Result.tsx` | A result with no transcript renders a warning instead of a number. |

The scan's tolerance is derived from how a number is *written*: `p = .023` claims
three decimals, so it matches a stored value only within 0.0005. An earlier
version used a generous tolerance at every decimal place, which made `0.023`
"match" a stored `0.0` and let a fabricated p-value through — the exact failure
the module exists to prevent. `test_numeric_scan_rejects_a_plausible_near_miss`
is the regression test.

---

## Layout

```
engine/          the statistics, simulation and geometry core (Python)
  src/phagequest_engine/
    transcript.py      execution records — the binding
    guards.py          numeric scan, pre-registration, spec-search, claim guard
    inference.py       the statistical core, organised by the question a student asks
    assumptions.py     checks that CHOOSE the test rather than decorate it
    effects.py         effect sizes, BSD reimplementations of pingouin's (GPL-3)
    power.py           "how many samples do we need?" — matches G*Power
    stability.py       the automatic perturbation check
    ledger.py          the visible multiple-comparison ledger
    resampling.py      shuffling, bootstrap, jackknife
    curves.py          dose-response and growth curves with real intervals
    diversity.py       alpha/beta diversity, CLR, PERMANOVA
    geometry.py        TNF → k-NN → plateau → Ollivier-Ricci (section 11)
    sequence.py        FASTA, translation, ORFs, codon usage
    validation.py      upload checking with errors a Grade 6 can act on
    simulation/        host–phage ODEs, Gillespie, plaque lattice, lab arithmetic
    registry.py        the typed tool boundary
    assistant.py       the turn loop with every guard wired in
  tests/           123 tests

apps/api/        FastAPI: identity, workspaces, review queue, cache, job queue
apps/web/        React + Vite; the engine runs in a Pyodide web worker
scripts/         build-engine-sources.mjs — inlines engine/src for the browser
```

---

## Running it

```bash
# Python engine
pip install -e engine[dev]
pytest engine/tests -q                 # 123 tests

# API
pip install -e apps/api
uvicorn app.main:app --reload --app-dir apps/api

# Web
npm install
npm run dev                            # http://localhost:5173
npm run build
```

`npm test` runs the engine tests, the API tests and the TypeScript typecheck.

---

## The three execution tiers

Roughly 90% of the Grades 5–12 workload runs in the student's browser. That is
the highest-leverage decision in the whole design: it costs nothing per student,
works offline after the first load, and means student data never reaches a
server — which is a DPDP argument as much as a cost one.

| Tier | Where | What |
|---|---|---|
| **0 — default** | Pyodide in a web worker | The entire statistical core, the simulations, the geometry, sequence handling. numpy + scipy only. |
| **1 — sandboxed** | FastAPI `/run` | Heavier fitting and anything a device cannot run in WebAssembly. Same package, identical output. |
| **2 — batch** | Queued jobs | Pharokka, taxmyphage, BACPHLIP, skani. Minutes, not seconds — scheduled between lessons, with a hard per-school ceiling. |

**Tier 0 and Tier 1 cannot diverge.** `scripts/build-engine-sources.mjs` inlines
`engine/src` into a module the worker mounts into Pyodide's filesystem, so the
browser runs the same source the server imports. A wheel would have introduced a
packaging step that could silently go stale, and that is the one claim this
platform cannot afford to have drift.

Pyodide is fetched once from a CDN and cached. If it cannot be reached the app
says so plainly and names the URL to allow, rather than failing quietly or
pretending to compute.

---

## What the platform refuses to do

Refusals are features here, and each one is tested:

- **Fewer than 3 observations per group** — no significance test. It describes the data instead and says why.
- **Machine learning below n=100**, or more features than samples — refused, not caveated. A caveat reads as permission.
- **A t-test on compositional data** — routed to CLR + PERMANOVA, with an explanation of why percentages cannot be compared that way.
- **A test before a prediction is written** — the button is disabled until one exists.
- **"Drop that outlier and try again"** — detected by phrasing *and*, more importantly, by behaviour. The specification's central finding is that frontier-model guardrails proved sensitive to framing rather than intent; repeated tests on one dataset with shrinking n is a signal that does not care how politely it is asked.
- **"These clusters are far apart on the UMAP, so they're different"** — blocked at the model layer. UMAP distances are not distances.
- **A curve fit with more parameters than the data can support.**
- **A batch job that would exceed the school's monthly ceiling** — refused, not billed. (Terra and DNAnexus are disqualified in the specification for exactly this: "A student's infinite loop produces a real invoice.")

---

## Licence posture

The engine depends on **numpy and scipy only**. No GPL (`pingouin`, `igraph`,
`leidenalg`), no AGPL (`giotto-tda`). Two tests enforce this on every run:
`test_the_engines_own_dependencies_are_clean` and
`test_no_engine_module_imports_a_copyleft_package`.

Consequences of that choice, all of which turned out to be improvements:

- **Effect sizes** are BSD reimplementations rather than `pingouin` (GPL-3).
- **Power analysis** uses scipy's noncentral t and F directly instead of `statsmodels`, so the browser path is not the weaker one. Values match G*Power: d=0.8 → 26/group, d=0.5 → 64/group, f=0.25 with 3 groups → 53/group.
- **Ollivier-Ricci curvature** is exact optimal transport via `scipy.optimize.linprog` rather than `GraphRicciCurvature`, which pins `scipy<=1.13.1` and pulls in a `networkit` C++ build. This removes the dependency, the version pin and the build — and puts the definition in front of a Grade 12 student as forty readable lines.
- **Tetranucleotide frequency** is a `Counter` over a sliding window. There is no package for it; the specification lists "a tetranucleotide frequency package" as a classic hallucination site.

`guards.NAME_COLLISIONS` encodes the package-name traps (`pip install drc`
installs a Django comment module; `jellyfish` is a string-distance library;
`basico` is an unrelated 2019 package) and `check_package_names` runs over this
repository's own dependency files in CI.

---

## Children's data (DPDP Act 2023)

Every user under 18 is legally a child in India, which is this platform's entire
Grades 5–12 audience. What that changes here:

- **No identities are stored.** A student is a school-issued `pseudonym`. No name, no email, no date of birth. The school holds the mapping: the school is the Data Fiduciary, the platform is the Data Processor.
- **Students cannot be enrolled** until the school confirms it holds verifiable parental consent (Rule 10). The endpoint returns 412 until it does.
- **Site coordinates are rounded to ~100 m** before storage. Five decimal places locates a child to within a metre; three is a habitat.
- **Anything sent to a language model is redacted first** — emails, phone numbers, ID-shaped numbers, precise coordinates and the student's own pseudonym.
- **Tier 0 means most data never leaves the device at all**, which is the strongest version of compliance available: data you never processed.
- **Audit events** record what happened and to which pseudonym, never what was measured.

---

## What is not built

Stated plainly, because a specification implemented at 80% and described as 100%
is the same category of error as a fabricated p-value:

- **Tier 2 is a queue, not a runner.** The job model, quota enforcement, cost ceiling and API are complete and tested; the containers that actually run Pharokka, taxmyphage, BACPHLIP and skani are not in this repository. `GET /jobs/kinds` declares each tool's real database footprint (≈1.5 GB shipped total).
- **The PECAAN-shaped annotation review UI** is not built. It is step 4 of the specification's own build order; steps 1, 2, 3 and 5 are.
- **The LLM provider is the deterministic mock.** The interface is `LLMProvider`; a real provider is one class. Every guard is tested against the mock's `misbehave` mode, which fabricates statistics on purpose.
- **Pyodide boot is unverified in CI here** — the build container's egress blocks the CDN. The worker's Python bootstrap is executed and tested under CPython, and the engine has no CPython-only dependency, but the first thing to do on a real network is load the page and watch the footer report `runtime: pyodide`.
- **No Quarto report export**, no marimo/JupyterLite notebook surface. Both are open questions in the specification rather than settled decisions.

---

## The geometry module (section 11)

Reproduces the pipeline: genome → 256-D tetranucleotide vector → cosine distance
→ k-NN graph → components swept over k → Ollivier-Ricci curvature → the
negatively curved bridge edges.

Exposed as the three teaching modules the specification names — **Fingerprint**
(7–9), **Islands** (9–11), **Bridges** (11–12).

`bridge_analysis` runs the curvature **weighted and unweighted, always**, and
writes its caveat from the numbers it actually computed rather than from a
stored sentence. On a graph where the two agree it says so and explains why
(when the transport metric and the denominator share units the quantity is
scale-invariant); where they diverge it says that instead. An earlier version
asserted a dramatic divergence that the code did not produce — which would have
been the platform committing the exact sin it exists to prevent.

---

## Build order

Following the specification's own ordering:

1. ✅ The statistics engine and its transcript binding
2. ✅ Data exploration on curriculum-shaped data
3. ✅ The chatbot, with the numeric-scan guard from day one
4. ⬜ Pharokka annotation in a PECAAN-shaped review UI
5. ✅ Simulation
6. ✅ The curvature capstone
