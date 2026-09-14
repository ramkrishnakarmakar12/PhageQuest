"""The visible multiple-comparison ledger (spec section 06, point 4).

    "Run a fifth test on the same data and the platform says so and applies
    correction, in the interface, where the student and teacher can see it."

The ledger is keyed on `data_id` (the fingerprint of the student's table), so it
survives page reloads, follows the dataset rather than the session, and cannot
be evaded by reopening the notebook. It is also the detector for the
specification-search loop the spec asks the assistant to refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np

from .transcript import Transcript, TranscriptStore, get_store

__all__ = ["LedgerEntry", "LedgerReport", "ledger_for", "holm_bonferroni", "benjamini_hochberg"]


@dataclass
class LedgerEntry:
    transcript_id: str
    tool: str
    call: str
    p_raw: float
    p_adjusted: float
    still_significant: bool

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LedgerReport:
    data_id: Optional[str]
    tests_run_so_far: int
    correction_applied: Optional[str]
    alpha: float
    entries: List[Dict]
    message: str
    severity: str = "info"

    def to_dict(self) -> Dict:
        return asdict(self)


def holm_bonferroni(pvals: List[float], alpha: float = 0.05) -> List[float]:
    """Holm-Bonferroni step-down. Controls familywise error, uniformly more
    powerful than plain Bonferroni, and simple enough to explain to a Grade 8:
    sort the p-values, multiply the smallest by how many tests you ran."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return [float(x) for x in adj]


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """FDR control -- the right correction for an exploratory sweep (e.g. the
    Grade 8 four-parameter enzyme sweep) where the cost of a miss exceeds the
    cost of a false alarm."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    running = 1.0
    for rank, idx in enumerate(order):
        val = p[idx] * m / (m - rank)
        running = min(running, val)
        adj[idx] = min(1.0, running)
    return [float(x) for x in adj]


def ledger_for(
    data_id: Optional[str],
    alpha: float = 0.05,
    method: str = "holm",
    store: Optional[TranscriptStore] = None,
    threshold: int = 3,
) -> LedgerReport:
    """Build the ledger for every inferential test run so far on this dataset.

    Correction kicks in at `threshold` tests. Below that we still show the count,
    because the point is that the student watches the number climb -- the
    counter is the teaching object, not the correction.
    """
    st = store or get_store()
    tests: List[Transcript] = st.inferential_on(data_id)
    n = len(tests)

    if n == 0:
        return LedgerReport(data_id, 0, None, alpha, [],
                            "No inferential tests have been run on this dataset yet.", "info")

    praw = [float(t.result.get("p_value")) for t in tests]
    if n >= threshold:
        padj = holm_bonferroni(praw, alpha) if method == "holm" else benjamini_hochberg(praw)
        applied = "holm-bonferroni" if method == "holm" else "benjamini-hochberg"
    else:
        padj = list(praw)
        applied = None

    entries = [
        LedgerEntry(t.transcript_id, t.tool, t.call, pr, pa, bool(pa < alpha)).to_dict()
        for t, pr, pa in zip(tests, praw, padj)
    ]

    lost = [e for e in entries if e["p_raw"] < alpha <= e["p_adjusted"]]
    if applied is None:
        msg = (f"{n} inferential test{'s' if n != 1 else ''} run on this dataset. "
               f"At {threshold} the platform starts correcting for multiple comparisons.")
        sev = "info"
    elif lost:
        msg = (f"{n} tests have now been run on this dataset. After {applied} correction, "
               f"{len(lost)} result{'s' if len(lost) != 1 else ''} that looked significant no longer "
               f"{'are' if len(lost) != 1 else 'is'}. That is what running many tests does: "
               f"with {n} tests at alpha {alpha}, you expect about {n * alpha:.1f} false alarms by chance alone.")
        sev = "warn"
    else:
        msg = (f"{n} tests run on this dataset; {applied} correction applied and every previously "
               f"significant result survives it.")
        sev = "info"

    return LedgerReport(data_id, n, applied, alpha, entries, msg, sev)
