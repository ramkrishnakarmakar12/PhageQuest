"""Effect sizes and their intervals.

Spec section 04, row 2: *"How big is the difference?"* The curriculum's own
language -- "which showed the most growth?" -- asks for an effect size, not a
significance test. So every comparison in this engine returns one, always,
with an interval, and the frontend shows the effect size first and the p-value
second.

pingouin has the nicest API for this and is GPL-3 (spec section 09). These are
BSD/MIT-only reimplementations over numpy/scipy so the platform core carries no
copyleft obligation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

__all__ = [
    "cohens_d",
    "hedges_g",
    "glass_delta",
    "cles",
    "rank_biserial",
    "eta_squared",
    "epsilon_squared",
    "cramers_v",
    "effsize_ci",
    "interpret",
]


def _clean(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    return a[np.isfinite(a)]


def cohens_d(a: Sequence[float], b: Sequence[float], paired: bool = False) -> float:
    """Standardised mean difference, pooled SD (independent) or SD of differences (paired)."""
    x, y = _clean(a), _clean(b)
    if paired:
        n = min(len(x), len(y))
        d = x[:n] - y[:n]
        sd = d.std(ddof=1)
        return float(d.mean() / sd) if sd > 0 else 0.0
    n1, n2 = len(x), len(y)
    s_pool = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    return float((x.mean() - y.mean()) / s_pool) if s_pool > 0 else 0.0


def hedges_g(a: Sequence[float], b: Sequence[float], paired: bool = False) -> float:
    """Cohen's d with the small-sample bias correction.

    Class datasets are n=4 to n=30 (spec section 04, row 4). At n=10 per group
    the correction is about 4%, which is not nothing when the whole point is
    honesty about small samples. Hedges' g is the default the platform reports.
    """
    d = cohens_d(a, b, paired=paired)
    n1, n2 = len(_clean(a)), len(_clean(b))
    df = (n1 - 1) if paired else (n1 + n2 - 2)
    if df <= 1:
        return d
    J = 1.0 - (3.0 / (4.0 * df - 1.0))
    return float(d * J)


def glass_delta(treatment: Sequence[float], control: Sequence[float]) -> float:
    """Standardised by the control SD only -- right when the treatment changes spread."""
    t, c = _clean(treatment), _clean(control)
    sd = c.std(ddof=1)
    return float((t.mean() - c.mean()) / sd) if sd > 0 else 0.0


def cles(a: Sequence[float], b: Sequence[float]) -> float:
    """Common-language effect size: P(a random value from A > a random value from B).

    This is the effect size a Grade 7 student can actually read: "68 times out
    of 100, a tube from the salty group beat a tube from the fresh group."
    """
    x, y = _clean(a), _clean(b)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    diff = x[:, None] - y[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / diff.size)


def rank_biserial(a: Sequence[float], b: Sequence[float]) -> float:
    """Effect size companion to Mann-Whitney U. Ranges -1..1, 0 = no difference."""
    return float(2.0 * cles(a, b) - 1.0)


def eta_squared(groups: Sequence[Sequence[float]]) -> float:
    """Proportion of total variance explained by group membership (one-way)."""
    gs = [_clean(g) for g in groups]
    allv = np.concatenate(gs)
    if allv.size == 0:
        return float("nan")
    grand = allv.mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in gs if len(g))
    ss_total = float(((allv - grand) ** 2).sum())
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def epsilon_squared(groups: Sequence[Sequence[float]]) -> float:
    """Rank-based analogue of eta-squared, the companion to Kruskal-Wallis."""
    gs = [_clean(g) for g in groups]
    n = sum(len(g) for g in gs)
    k = len(gs)
    if n <= k:
        return float("nan")
    H = stats.kruskal(*gs).statistic if all(len(g) > 0 for g in gs) else float("nan")
    return float((H - k + 1) / (n - k))


def cramers_v(table: np.ndarray) -> float:
    """Effect size for a contingency table, with the Bergsma bias correction."""
    t = np.asarray(table, dtype=float)
    chi2 = stats.chi2_contingency(t, correction=False).statistic
    n = t.sum()
    if n <= 0:
        return float("nan")
    phi2 = chi2 / n
    r, c = t.shape
    phi2c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    cc = c - (c - 1) ** 2 / (n - 1)
    denom = min(rc - 1, cc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def effsize_ci(
    a: Sequence[float],
    b: Sequence[float],
    kind: str = "hedges_g",
    confidence: float = 0.95,
    n_boot: int = 4999,
    seed: int = 20260901,
) -> Tuple[float, float]:
    """Bootstrap percentile interval around an effect size.

    Analytic intervals for d rely on a noncentral t that is wrong for the
    non-normal, tiny samples this platform actually sees. Bootstrapping is both
    more defensible here and more teachable (spec section 04, row 4).
    """
    x, y = _clean(a), _clean(b)
    if len(x) < 2 or len(y) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    fn = {
        "hedges_g": lambda p, q: hedges_g(p, q),
        "cohens_d": lambda p, q: cohens_d(p, q),
        "cles": cles,
        "rank_biserial": rank_biserial,
    }[kind]
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = fn(rng.choice(x, len(x), replace=True), rng.choice(y, len(y), replace=True))
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return (float("nan"), float("nan"))
    lo = (1 - confidence) / 2 * 100
    return (float(np.percentile(boots, lo)), float(np.percentile(boots, 100 - lo)))


# Deliberately conservative wording. The platform never says "large effect" to a
# student without also showing the interval, because a huge point estimate with
# an interval spanning zero is the single most common way a class dataset lies.
_BANDS = {
    "hedges_g": [(0.2, "negligible"), (0.5, "small"), (0.8, "moderate"), (np.inf, "large")],
    "cohens_d": [(0.2, "negligible"), (0.5, "small"), (0.8, "moderate"), (np.inf, "large")],
    "eta_squared": [(0.01, "negligible"), (0.06, "small"), (0.14, "moderate"), (np.inf, "large")],
    "epsilon_squared": [(0.01, "negligible"), (0.08, "small"), (0.26, "moderate"), (np.inf, "large")],
    "cramers_v": [(0.1, "negligible"), (0.3, "small"), (0.5, "moderate"), (np.inf, "large")],
    "rank_biserial": [(0.1, "negligible"), (0.3, "small"), (0.5, "moderate"), (np.inf, "large")],
}


def interpret(value: float, kind: str = "hedges_g", ci: Optional[Tuple[float, float]] = None) -> str:
    if not np.isfinite(value):
        return "not estimable"
    bands = _BANDS.get(kind, _BANDS["hedges_g"])
    label = "large"
    for cut, name in bands:
        if abs(value) < cut:
            label = name
            break
    if ci is not None and np.isfinite(ci[0]) and np.isfinite(ci[1]) and ci[0] * ci[1] <= 0:
        return f"{label} in this sample, but the interval includes zero -- the direction is not settled"
    return label
