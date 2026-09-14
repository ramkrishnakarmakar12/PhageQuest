"""Power and sample size (spec section 04, row 3).

    "The most-skipped and most valuable module in the whole platform. Making
    students compute required n before running G7-L7's control-vs-test
    decomposition -- and showing achieved power afterwards -- does more for
    scientific literacy than any amount of machine learning."

statsmodels has these and is Tier-1 only. scipy's noncentral t and F
distributions give exact answers in Tier-0, so the browser path is not the
weaker one. Implemented over `scipy.stats.nct` / `ncf` with a bisection solve.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats

from .transcript import bound

__all__ = ["power_ttest", "n_for_ttest", "power_anova", "n_for_anova", "achieved_power"]


def _safe(value: float, nc: float) -> float:
    """scipy's noncentral t/F return NaN for large noncentrality.

    That region is numerically awkward and pedagogically uninteresting: a
    noncentrality that large means the design is overwhelmingly powered. Rather
    than propagate a NaN into a student's answer, saturate it -- and only in the
    direction the mathematics guarantees.
    """
    if np.isfinite(value):
        return float(value)
    return 1.0 if nc > 10 else float("nan")


def _power_ttest(n_per_group: float, d: float, alpha: float = 0.05,
                 alternative: str = "two-sided") -> float:
    if n_per_group < 2 or d == 0:
        return alpha if d == 0 else 0.0
    df = 2 * n_per_group - 2
    nc = abs(d) * np.sqrt(n_per_group / 2.0)
    with np.errstate(all="ignore"):
        if alternative == "two-sided":
            crit = stats.t.ppf(1 - alpha / 2, df)
            v = stats.nct.sf(crit, df, nc) + stats.nct.cdf(-crit, df, nc)
        else:
            crit = stats.t.ppf(1 - alpha, df)
            v = stats.nct.sf(crit, df, nc)
    return _safe(v, nc)


def _solve_n(power_fn, target: float, lo: float = 2.001, hi: float = 1e6) -> float:
    """NaN-safe bisection for the smallest n reaching `target` power.

    `brentq` walks straight into the NaN region above n~1000 for large effects.
    Bisection with a monotonicity assumption (power increases with n, which is
    true for every design here) is both safer and easier to reason about.
    """
    if power_fn(lo) >= target:
        return lo
    if not (power_fn(hi) >= target):
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = power_fn(mid)
        if not np.isfinite(p):
            hi = mid          # NaN only occurs where power is effectively 1
            continue
        if p >= target:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-4:
            break
    return hi


def _power_anova(n_per_group: float, f: float, k_groups: int, alpha: float = 0.05) -> float:
    if n_per_group < 2 or f == 0:
        return alpha if f == 0 else 0.0
    df1 = k_groups - 1
    df2 = k_groups * (n_per_group - 1)
    if df2 < 1:
        return 0.0
    nc = f ** 2 * n_per_group * k_groups
    with np.errstate(all="ignore"):
        crit = stats.f.ppf(1 - alpha, df1, df2)
        v = stats.ncf.sf(crit, df1, df2, nc)
    return _safe(v, nc)


@bound("power_ttest")
def power_ttest(n_per_group: int, effect_size: float, alpha: float = 0.05,
                alternative: str = "two-sided") -> Dict[str, Any]:
    """Power of a two-group comparison at a given n and effect size."""
    p = _power_ttest(float(n_per_group), float(effect_size), alpha, alternative)
    return {
        "power": p, "n_per_group": int(n_per_group), "effect_size": float(effect_size),
        "effect_size_type": "cohens_d", "alpha": alpha, "alternative": alternative,
        "plain_language": (
            f"With {n_per_group} per group, if the real difference is d = {effect_size:g}, this "
            f"experiment finds it about {p * 100:.0f} times out of 100. "
            + ("That is a well-powered experiment." if p >= 0.8 else
               f"That means it MISSES a real difference about {(1 - p) * 100:.0f} times out of 100 -- "
               f"so a non-significant result here would tell you almost nothing.")),
        "adequate": bool(p >= 0.8),
    }


@bound("n_for_ttest")
def n_for_ttest(effect_size: float, power: float = 0.8, alpha: float = 0.05,
                alternative: str = "two-sided") -> Dict[str, Any]:
    """*"How many samples do we need?"* -- asked BEFORE the experiment.

    This is the function the platform puts in front of a student at the design
    stage of G7-L7 and G7-L9, before a single tube is filled.
    """
    d = abs(float(effect_size))
    if d <= 0:
        return {"refused": True, "reason": "An effect size of zero needs infinitely many samples. "
                                           "Decide how big a difference would actually matter first."}
    n = _solve_n(lambda x: _power_ttest(x, d, alpha, alternative), power)
    n_ceil = int(np.ceil(n))
    return {
        "n_per_group": n_ceil, "n_total": 2 * n_ceil,
        "effect_size": d, "effect_size_type": "cohens_d",
        "power": power, "alpha": alpha, "alternative": alternative,
        "feasible": bool(n_ceil <= 30),
        "plain_language": (
            f"To have a {power * 100:.0f}% chance of detecting a difference of d = {d:g}, you need "
            f"{n_ceil} in EACH group ({2 * n_ceil} total). "
            + (f"That is achievable in a class experiment." if n_ceil <= 30 else
               f"That is more than a class can realistically produce. Either pool data across "
               f"classes, or accept that this experiment can only detect a larger difference -- "
               f"and say so in your report rather than pretending otherwise.")),
    }


@bound("power_anova")
def power_anova(n_per_group: int, effect_size_f: float, k_groups: int,
                alpha: float = 0.05) -> Dict[str, Any]:
    """Power for a one-way design. `effect_size_f` is Cohen's f (f^2 = eta^2/(1-eta^2))."""
    p = _power_anova(float(n_per_group), float(effect_size_f), int(k_groups), alpha)
    eta2 = effect_size_f ** 2 / (1 + effect_size_f ** 2)
    return {
        "power": p, "n_per_group": int(n_per_group), "k_groups": int(k_groups),
        "effect_size": float(effect_size_f), "effect_size_type": "cohens_f",
        "equivalent_eta_squared": float(eta2), "alpha": alpha,
        "adequate": bool(p >= 0.8),
        "plain_language": (
            f"{k_groups} groups of {n_per_group}: this design detects an effect of f = "
            f"{effect_size_f:g} (eta-squared {eta2:.2f}) about {p * 100:.0f} times out of 100."),
    }


@bound("n_for_anova")
def n_for_anova(effect_size_f: float, k_groups: int, power: float = 0.8,
                alpha: float = 0.05) -> Dict[str, Any]:
    """Required n per group for a one-way design.

    G7-L9 uses four salt levels, G8-L9 four pH levels, G6-L9 three temperatures.
    This is the number that tells a class whether three tubes per level is a
    real experiment or a decorative one.
    """
    f = abs(float(effect_size_f))
    if f <= 0:
        return {"refused": True, "reason": "Effect size must be greater than zero."}
    n = _solve_n(lambda x: _power_anova(x, f, int(k_groups), alpha), power)
    n_ceil = int(np.ceil(n))
    return {
        "n_per_group": n_ceil, "n_total": n_ceil * int(k_groups),
        "k_groups": int(k_groups), "effect_size": f, "effect_size_type": "cohens_f",
        "power": power, "alpha": alpha, "feasible": bool(n_ceil <= 30),
        "plain_language": (
            f"With {k_groups} levels and a target of {power * 100:.0f}% power, you need {n_ceil} "
            f"replicates at EACH level -- {n_ceil * k_groups} tubes in total."),
    }


@bound("achieved_power", data_args=("observed_effect",))
def achieved_power(observed_effect: float, n_per_group: int, k_groups: int = 2,
                   alpha: float = 0.05, effect_size_type: str = "hedges_g") -> Dict[str, Any]:
    """Power the experiment actually had, reported after the fact.

    Deliberately framed as "what this experiment COULD have detected", not as
    "post-hoc power", because post-hoc power computed from the observed effect
    is a well-known statistical fallacy -- it is a deterministic function of the
    p-value and adds no information. The honest use is the *detectable* effect:
    what is the smallest difference this design had a fair chance of finding?
    """
    d = abs(float(observed_effect))
    if effect_size_type in ("eta_squared", "epsilon_squared"):
        f = np.sqrt(max(0.0, d) / max(1e-12, 1 - d))
        p = _power_anova(float(n_per_group), f, k_groups, alpha)
        mde = _solve_n(lambda x: _power_anova(float(n_per_group), x, k_groups, alpha), 0.8,
                       lo=1e-4, hi=50.0)
        mde_type = "cohens_f"
    else:
        p = _power_ttest(float(n_per_group), d, alpha)
        mde = _solve_n(lambda x: _power_ttest(float(n_per_group), x, alpha), 0.8,
                       lo=1e-4, hi=50.0)
        mde_type = "cohens_d"
    return {
        "power_at_observed_effect": p,
        "minimum_detectable_effect": float(mde),
        "minimum_detectable_effect_type": mde_type,
        "n_per_group": int(n_per_group), "alpha": alpha,
        "caution": ("Power computed from the effect you observed is not evidence about your result -- "
                    "it is just the p-value in a different costume. Use the minimum detectable effect "
                    "instead: it describes the experiment, not the outcome."),
        "plain_language": (
            f"With {n_per_group} per group, the smallest effect this design had a fair (80%) chance "
            f"of detecting was about {mde_type.replace('_', ' ')} = {mde:.2f}. "
            f"Your observed effect was {d:.2f}. "
            + ("So the experiment was big enough to see an effect this size."
               if d >= mde else
               "So an effect this size was below what this experiment could reliably detect -- "
               "which is why the result is uncertain, and it is a design problem, not a data problem.")),
    }
