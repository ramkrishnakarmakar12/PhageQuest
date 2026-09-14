"""Resampling: the platform's default route for small, weird data.

Spec section 04, row 4:

    "Class datasets are n=4 to n=30 and routinely non-normal. Resampling is
    *also* more teachable than a t-distribution: shuffle the labels ten thousand
    times and see how often chance beats you. A Grade 8 student can understand
    that. They cannot understand a t-table."

So these are not the advanced menu. `shuffle_test` is what a Grade 7 sees when
they press "test it", and it returns the null distribution itself so the
frontend can draw the histogram with the observed value marked on it. The
picture is the explanation.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from .transcript import bound

__all__ = ["bootstrap_ci", "shuffle_test", "monte_carlo_p", "jackknife"]

_STATS: Dict[str, Callable[[np.ndarray], float]] = {
    "mean": np.mean,
    "median": np.median,
    "sd": lambda a: float(np.std(a, ddof=1)),
    "max": np.max,
    "min": np.min,
    "iqr": lambda a: float(np.percentile(a, 75) - np.percentile(a, 25)),
}


@bound("bootstrap_ci", data_args=("values",))
def bootstrap_ci(values: Sequence[float], statistic: str = "mean", confidence: float = 0.95,
                 n_resamples: int = 9999, method: str = "BCa",
                 seed: int = 20260901) -> Dict[str, Any]:
    """Bias-corrected accelerated bootstrap interval on any simple statistic.

    BCa rather than percentile: at n=8 the percentile interval is visibly
    off-centre for a skewed statistic, and a class dataset is usually skewed.
    """
    a = np.asarray(values, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size < 3:
        return {"refused": True, "reason": f"n={a.size}; need at least 3 values to resample."}
    fn = _STATS.get(statistic)
    if fn is None:
        return {"refused": True, "reason": f"Unknown statistic {statistic!r}. "
                                           f"Choose from {sorted(_STATS)}."}
    rng = np.random.default_rng(seed)
    res = stats.bootstrap((a,), fn, confidence_level=confidence, n_resamples=n_resamples,
                          method=method if a.size > 3 else "percentile",
                          random_state=rng, vectorized=False)
    point = float(fn(a))
    lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    return {
        "statistic": statistic, "point_estimate": point,
        "ci": [lo, hi], "confidence": confidence,
        "standard_error": float(res.standard_error),
        "method": method if a.size > 3 else "percentile",
        "n": int(a.size), "n_resamples": int(n_resamples),
        "distribution_sample": [float(x) for x in
                                rng.choice(res.bootstrap_distribution.ravel(),
                                           size=min(2000, res.bootstrap_distribution.size),
                                           replace=False)],
        "plain_language": (
            f"The {statistic} of your {a.size} measurements is {point:.4g}. If you repeated the "
            f"whole experiment many times, the {statistic} would land between {lo:.4g} and {hi:.4g} "
            f"about {confidence * 100:.0f}% of the time. That width IS your uncertainty -- report it "
            f"instead of the single number."),
    }


@bound("shuffle_test", data_args=("group_a", "group_b"))
def shuffle_test(group_a: Sequence[float], group_b: Sequence[float],
                 statistic: str = "mean_difference", alternative: str = "two-sided",
                 n_shuffles: int = 9999, seed: int = 20260901) -> Dict[str, Any]:
    """The Grade 7 test: shuffle the labels and see how often chance wins.

    Returns the null distribution so the frontend can draw it. The p-value is
    "how many of the shuffled worlds beat the real one", which is a sentence a
    12-year-old can check by counting dots on the histogram.
    """
    a = np.asarray(group_a, dtype=float).ravel()
    b = np.asarray(group_b, dtype=float).ravel()
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return {"refused": True, "reason": "Need at least 2 measurements in each group."}

    fns = {
        "mean_difference": lambda x, y: float(np.mean(x) - np.mean(y)),
        "median_difference": lambda x, y: float(np.median(x) - np.median(y)),
        "ratio_of_means": lambda x, y: float(np.mean(x) / np.mean(y)) if np.mean(y) else np.nan,
    }
    fn = fns.get(statistic)
    if fn is None:
        return {"refused": True, "reason": f"Unknown statistic {statistic!r}."}

    observed = fn(a, b)
    pooled = np.concatenate([a, b])
    n_a = a.size
    rng = np.random.default_rng(seed)
    null = np.empty(n_shuffles)
    for i in range(n_shuffles):
        perm = rng.permutation(pooled)
        null[i] = fn(perm[:n_a], perm[n_a:])

    if alternative == "greater":
        count = int(np.sum(null >= observed))
    elif alternative == "less":
        count = int(np.sum(null <= observed))
    else:
        centre = 1.0 if statistic == "ratio_of_means" else 0.0
        count = int(np.sum(np.abs(null - centre) >= abs(observed - centre)))
    # +1 smoothing: a p-value of exactly 0 is a lie about resolution.
    p = (count + 1) / (n_shuffles + 1)

    return {
        "statistic_name": statistic, "observed": observed,
        "p_value": float(p), "n_shuffles": int(n_shuffles),
        "n_at_least_as_extreme": count, "alternative": alternative,
        "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
        "null_distribution": [float(x) for x in
                              rng.choice(null, size=min(2000, n_shuffles), replace=False)],
        "null_quantiles": {q: float(np.percentile(null, q)) for q in (2.5, 25, 50, 75, 97.5)},
        "effect_size": float(2 * np.mean((a[:, None] - b[None, :]) > 0) - 1),
        "effect_size_type": "rank_biserial",
        "n_per_group": {"a": int(n_a), "b": int(b.size)},
        "plain_language": (
            f"Your two groups differ by {observed:.4g}. We then mixed all {pooled.size} measurements "
            f"together and split them into two random groups {n_shuffles:,} times. "
            f"{count:,} of those random splits produced a difference at least as big as yours. "
            f"That is p = {p:.4g}. "
            + ("Chance rarely does this well, so the difference is probably real."
               if p < 0.05 else
               "Chance does this often enough that your result is not surprising.")),
        "resolution_note": (f"The smallest p-value {n_shuffles:,} shuffles can produce is "
                            f"{1 / (n_shuffles + 1):.5f}. A p of exactly 0 is never reported."),
    }


@bound("monte_carlo_p", data_args=("values",))
def monte_carlo_p(values: Sequence[float], null_distribution: str = "normal",
                  n_resamples: int = 9999, seed: int = 20260901) -> Dict[str, Any]:
    """Monte Carlo goodness-of-fit against a named null."""
    a = np.asarray(values, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size < 5:
        return {"refused": True, "reason": f"n={a.size}; need at least 5."}
    rvs = {"normal": lambda size: stats.norm.rvs(loc=a.mean(), scale=a.std(ddof=1), size=size,
                                                 random_state=np.random.default_rng(seed)),
           "poisson": lambda size: stats.poisson.rvs(mu=max(1e-9, a.mean()), size=size,
                                                     random_state=np.random.default_rng(seed)),
           "uniform": lambda size: stats.uniform.rvs(loc=a.min(), scale=float(np.ptp(a)), size=size,
                                                     random_state=np.random.default_rng(seed))}
    if null_distribution not in rvs:
        return {"refused": True, "reason": f"Unknown null {null_distribution!r}."}
    res = stats.monte_carlo_test(a, rvs[null_distribution], stats.skew,
                                 n_resamples=n_resamples, alternative="two-sided")
    return {
        "null_distribution": null_distribution, "statistic": float(res.statistic),
        "p_value": float(res.pvalue), "n": int(a.size),
        "plain_language": (
            f"Compared with {n_resamples:,} datasets simulated from a {null_distribution} "
            f"distribution, your data's shape is "
            f"{'unusual' if res.pvalue < 0.05 else 'unremarkable'} (p = {float(res.pvalue):.4g})."),
    }


@bound("jackknife", data_args=("values",))
def jackknife(values: Sequence[float], statistic: str = "mean") -> Dict[str, Any]:
    """Leave-one-out influence. Names the observation that is carrying the result.

    Used directly by the stability check, and exposed on its own because
    "which measurement is doing all the work?" is a question a Grade 8 can
    answer from a bar chart.
    """
    a = np.asarray(values, dtype=float).ravel()
    a = a[np.isfinite(a)]
    fn = _STATS.get(statistic)
    if fn is None or a.size < 3:
        return {"refused": True, "reason": "Need at least 3 values and a known statistic."}
    full = float(fn(a))
    loo = np.array([float(fn(np.delete(a, i))) for i in range(a.size)])
    infl = loo - full
    worst = int(np.argmax(np.abs(infl)))
    return {
        "full_sample": full, "leave_one_out": [float(x) for x in loo],
        "influence": [float(x) for x in infl],
        "most_influential_index": worst,
        "most_influential_value": float(a[worst]),
        "influence_of_worst": float(infl[worst]),
        "n": int(a.size),
        "plain_language": (
            f"Removing the measurement {a[worst]:.4g} moves the {statistic} from {full:.4g} to "
            f"{loo[worst]:.4g}. "
            + ("That one point is carrying a large share of the result -- check whether it is a "
               "real measurement or a mistake, and say which in your report."
               if abs(infl[worst]) > 0.2 * max(1e-12, abs(full)) else
               "No single measurement dominates, which is a good sign.")),
    }
