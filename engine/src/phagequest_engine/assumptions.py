"""Assumption checks that choose the test, rather than decorating it.

Galaxy and BV-BRC have no statistical grammar (spec section 01); jamovi has the
checks but you must know to ask. Here the checks *drive the routing*: the engine
runs them first, picks the test from the result, and reports both the choice and
the reason in `GroupComparison.test_used`, so a teacher who is not a statistician
can see why Kruskal-Wallis appeared instead of ANOVA.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

__all__ = ["AssumptionResult", "check_normality", "check_variance", "check_cell_counts",
           "check_sample_size", "assumption_panel"]


@dataclass
class AssumptionResult:
    name: str
    passed: bool
    statistic: float | None
    p_value: float | None
    detail: str
    severity: str = "info"      # info | warn | block

    def to_dict(self) -> Dict:
        return asdict(self)


def _clean(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def check_normality(groups: Sequence[Sequence[float]], alpha: float = 0.05) -> AssumptionResult:
    """Shapiro-Wilk per group.

    Note the honest framing in `detail`: at n=6 this test cannot detect
    non-normality, so "passed" here means "no evidence against", not "normal".
    The platform says so rather than letting a green tick imply more than it can.
    """
    gs = [_clean(g) for g in groups]
    usable = [g for g in gs if len(g) >= 3]
    if not usable:
        return AssumptionResult("normality", False, None, None,
                                "Too few observations to assess normality at all.", "warn")
    ps, stat = [], None
    for g in usable:
        if len(g) > 5000 or np.allclose(g, g[0]):
            continue
        r = stats.shapiro(g)
        ps.append(float(r.pvalue))
        stat = float(r.statistic)
    if not ps:
        return AssumptionResult("normality", True, None, None,
                                "Constant or very large groups; Shapiro-Wilk not applied.", "info")
    worst = min(ps)
    smallest_n = min(len(g) for g in usable)
    passed = worst >= alpha
    if smallest_n < 10:
        detail = (f"Smallest p across groups = {worst:.3f}. With n={smallest_n} this test has very "
                  f"little power, so this is not evidence that the data IS normal -- it is only the "
                  f"absence of evidence that it is not. The engine will prefer a rank-based or "
                  f"resampling test regardless.")
        sev = "warn"
    else:
        detail = f"Smallest Shapiro-Wilk p across groups = {worst:.3f} (alpha {alpha})."
        sev = "info" if passed else "warn"
    return AssumptionResult("normality", passed, stat, worst, detail, sev)


def check_variance(groups: Sequence[Sequence[float]], alpha: float = 0.05) -> AssumptionResult:
    """Levene's test (median-centred / Brown-Forsythe), robust to non-normality."""
    gs = [g for g in (_clean(x) for x in groups) if len(g) >= 2]
    if len(gs) < 2:
        return AssumptionResult("equal_variance", False, None, None,
                                "Fewer than two usable groups.", "block")
    if any(np.allclose(g, g[0]) for g in gs):
        return AssumptionResult("equal_variance", True, None, None,
                                "At least one group has zero spread; Levene not applied.", "warn")
    r = stats.levene(*gs, center="median")
    passed = float(r.pvalue) >= alpha
    ratio = max(g.var(ddof=1) for g in gs) / max(1e-300, min(g.var(ddof=1) for g in gs))
    return AssumptionResult(
        "equal_variance", passed, float(r.statistic), float(r.pvalue),
        f"Levene p = {float(r.pvalue):.3f}; largest/smallest variance ratio = {ratio:.1f}."
        + ("" if passed else " Welch / Alexander-Govern will be used instead of classical ANOVA."),
        "info" if passed else "warn",
    )


def check_cell_counts(table: np.ndarray, minimum: int = 5) -> AssumptionResult:
    """Expected-count rule for chi-square. Below it, Fisher's exact is used."""
    t = np.asarray(table, dtype=float)
    if t.size == 0 or t.sum() == 0:
        return AssumptionResult("expected_counts", False, None, None, "Empty table.", "block")
    expected = np.outer(t.sum(axis=1), t.sum(axis=0)) / t.sum()
    worst = float(expected.min())
    frac_low = float((expected < minimum).mean())
    passed = worst >= minimum
    return AssumptionResult(
        "expected_counts", passed, worst, None,
        f"Smallest expected count = {worst:.2f}; {frac_low:.0%} of cells below {minimum}."
        + ("" if passed else " Chi-square is unreliable here; Fisher's exact test will be used."),
        "info" if passed else "warn",
    )


def check_sample_size(groups: Sequence[Sequence[float]], floor: int = 3) -> AssumptionResult:
    """The check that actually matters most for this user base.

    A class dataset is n=4 to n=30. Below `floor` per group, the engine refuses
    to run an inferential test at all and returns a descriptive answer -- a
    refusal is a better lesson than a p-value computed on three points.
    """
    ns = [len(_clean(g)) for g in groups]
    smallest = min(ns) if ns else 0
    passed = smallest >= floor
    sev = "info" if smallest >= 5 else ("warn" if passed else "block")
    return AssumptionResult(
        "sample_size", passed, float(smallest), None,
        f"Group sizes: {ns}."
        + ("" if smallest >= 5 else
           f" With only {smallest} observations in the smallest group, any conclusion is fragile; "
           f"the stability check below is the number to read, not the p-value."),
        sev,
    )


def assumption_panel(groups: Sequence[Sequence[float]], alpha: float = 0.05) -> List[AssumptionResult]:
    return [check_sample_size(groups), check_normality(groups, alpha), check_variance(groups, alpha)]
