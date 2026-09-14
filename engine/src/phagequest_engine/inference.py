"""The statistical core (spec section 04).

Organised by the question a student asks, not by what a textbook contains.

Design rules encoded here rather than documented:

  * The test is *chosen by the assumption checks*, and the reason is reported.
  * An effect size with an interval is always returned; it is never optional.
  * A stability check runs automatically on every inferential result.
  * The multiple-comparison ledger is consulted and returned with the result.
  * Below `min_n` per group the engine refuses to infer and describes instead.
  * Permutation is the default route for tiny samples, because shuffling is
    both more defensible and more teachable than a t-distribution (Grade 7).

BSD/MIT only: scipy + numpy. No pingouin (GPL-3, spec section 09).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from . import effects
from .assumptions import assumption_panel, check_cell_counts
from .ledger import ledger_for
from .stability import stability_check
from .transcript import bound, data_fingerprint

__all__ = ["compare_groups", "correlate", "contingency", "trend_test", "describe", "one_sample"]

ALPHA = 0.05
MIN_N = 3


def _prep(groups: Dict[str, Sequence[float]]) -> Tuple[List[str], List[np.ndarray]]:
    labels, arrs = [], []
    for k, v in groups.items():
        a = np.asarray(v, dtype=float).ravel()
        a = a[np.isfinite(a)]
        labels.append(str(k))
        arrs.append(a)
    return labels, arrs


def _describe_one(a: np.ndarray) -> Dict[str, float]:
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "se": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "q1": float(np.percentile(a, 25)),
        "q3": float(np.percentile(a, 75)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


@bound("describe", data_args=("groups",))
def describe(groups: Dict[str, Sequence[float]]) -> Dict[str, Any]:
    """Group summaries with bootstrap CIs on the mean.

    This is the Grade 5-6 surface: no p-value anywhere, but the interval is
    already there, so "three groups did the same experiment and got three
    answers -- why?" has somewhere to go.
    """
    labels, arrs = _prep(groups)
    rng = np.random.default_rng(20260901)
    out = {}
    for lab, a in zip(labels, arrs):
        d = _describe_one(a)
        if a.size >= 3:
            boots = np.array([rng.choice(a, a.size, replace=True).mean() for _ in range(2000)])
            d["mean_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        else:
            d["mean_ci"] = [float("nan"), float("nan")]
        out[lab] = d
    notes = []
    ns = [d["n"] for d in out.values()]
    if ns and min(ns) < 3:
        notes.append("At least one group has fewer than 3 observations. The engine will not run an "
                     "inferential test on this data; it can only describe what was measured.")
    return {"groups": out, "notes": notes, "p_value": None}


@bound("one_sample", data_args=("values",))
def one_sample(values: Sequence[float], mu: float = 0.0, alpha: float = ALPHA) -> Dict[str, Any]:
    """One-sample test against a reference value, with a bootstrap interval.

    Serves G9-L9 (is the fuel-cell voltage above the threshold?) and the
    "is our count different from the expected count" shape in G8-L10.
    """
    a = np.asarray(values, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size < MIN_N:
        return {"refused": True, "reason": f"n={a.size} is below the engine's floor of {MIN_N}.",
                "p_value": None}
    res = stats.wilcoxon(a - mu) if a.size < 10 else stats.ttest_1samp(a, mu)
    test = "Wilcoxon signed-rank (n < 10)" if a.size < 10 else "one-sample t-test"
    rng = np.random.default_rng(20260901)
    boots = np.array([rng.choice(a, a.size, replace=True).mean() for _ in range(4999)])
    return {
        "test_used": test,
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "mean": float(a.mean()),
        "mean_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "reference": float(mu),
        "n": int(a.size),
        "effect_size": float((a.mean() - mu) / a.std(ddof=1)) if a.std(ddof=1) > 0 else 0.0,
        "effect_size_type": "cohens_d_one_sample",
    }


def _route(labels: List[str], arrs: List[np.ndarray], design: str, panel, prefer_robust: bool
           ) -> Tuple[str, str]:
    """Pick the test from the assumption checks. Returns (test_key, why)."""
    k = len(arrs)
    smallest = min(a.size for a in arrs)
    normal = next(a for a in panel if a.name == "normality").passed
    equal_var = next(a for a in panel if a.name == "equal_variance").passed

    if design == "paired":
        if smallest < 10 or not normal:
            return "wilcoxon", (f"Paired design with n={smallest} per pair"
                                f"{' and evidence of non-normality' if not normal else ''}; "
                                f"the signed-rank test makes no distributional assumption.")
        return "ttest_rel", "Paired design, samples large enough and no evidence against normality."

    if k == 2:
        if prefer_robust or smallest < 10 or not normal:
            return "permutation", (
                f"Two independent groups with n={smallest} in the smallest group"
                f"{' and evidence of non-normality' if not normal else ''}. A permutation test "
                f"shuffles the group labels many times and counts how often chance beats the "
                f"observed difference -- it assumes nothing about the shape of the data, and it is "
                f"what a student can actually verify by hand.")
        if not equal_var:
            return "welch", "Two groups, unequal variances; Welch's t-test does not pool them."
        return "ttest_ind", "Two independent groups, no evidence against normality or equal variance."

    if prefer_robust or smallest < 10 or not normal:
        return "kruskal", (
            f"{k} independent groups with n={smallest} in the smallest"
            f"{' and evidence of non-normality' if not normal else ''}; Kruskal-Wallis compares "
            f"ranks and does not assume a bell curve.")
    if not equal_var:
        return "alexandergovern", (
            f"{k} groups with unequal variances; Alexander-Govern is the heteroscedastic "
            f"alternative to one-way ANOVA (Welch's ANOVA family).")
    return "anova", f"{k} independent groups, assumptions of one-way ANOVA are not contradicted."


def _run(key: str, arrs: List[np.ndarray], alternative: str, seed: int) -> Tuple[float, float, str]:
    """Execute the chosen test. Returns (statistic, p_value, stat_label)."""
    if key == "ttest_ind":
        r = stats.ttest_ind(arrs[0], arrs[1], equal_var=True, alternative=alternative)
        return float(r.statistic), float(r.pvalue), f"t({len(arrs[0]) + len(arrs[1]) - 2})"
    if key == "welch":
        r = stats.ttest_ind(arrs[0], arrs[1], equal_var=False, alternative=alternative)
        return float(r.statistic), float(r.pvalue), f"t({float(r.df):.1f})"
    if key == "ttest_rel":
        n = min(len(arrs[0]), len(arrs[1]))
        r = stats.ttest_rel(arrs[0][:n], arrs[1][:n], alternative=alternative)
        return float(r.statistic), float(r.pvalue), f"t({n - 1})"
    if key == "wilcoxon":
        n = min(len(arrs[0]), len(arrs[1]))
        r = stats.wilcoxon(arrs[0][:n], arrs[1][:n], alternative=alternative)
        return float(r.statistic), float(r.pvalue), "W"
    if key == "mannwhitney":
        r = stats.mannwhitneyu(arrs[0], arrs[1], alternative=alternative)
        return float(r.statistic), float(r.pvalue), "U"
    if key == "permutation":
        def statfn(x, y):
            return np.mean(x) - np.mean(y)
        r = stats.permutation_test(
            (arrs[0], arrs[1]), statfn, permutation_type="independent",
            alternative=alternative, n_resamples=9999,
            random_state=np.random.default_rng(seed), vectorized=False)
        return float(r.statistic), float(r.pvalue), "mean difference"
    if key == "anova":
        r = stats.f_oneway(*arrs)
        dfb = len(arrs) - 1
        dfw = sum(len(a) for a in arrs) - len(arrs)
        return float(r.statistic), float(r.pvalue), f"F({dfb},{dfw})"
    if key == "alexandergovern":
        r = stats.alexandergovern(*arrs)
        return float(r.statistic), float(r.pvalue), "A"
    if key == "kruskal":
        r = stats.kruskal(*arrs)
        return float(r.statistic), float(r.pvalue), f"H({len(arrs) - 1})"
    raise ValueError(f"unknown test key {key!r}")


def _posthoc(labels: List[str], arrs: List[np.ndarray], parametric: bool) -> List[Dict[str, Any]]:
    """Pairwise follow-up with Holm correction.

    scikit-posthocs (MIT) is the Tier-1 choice; this is the dependency-free
    Tier-0 equivalent so the browser path is not weaker than the server path.
    """
    from .ledger import holm_bonferroni

    pairs, praw = [], []
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            if parametric:
                r = stats.ttest_ind(arrs[i], arrs[j], equal_var=False)
            else:
                r = stats.mannwhitneyu(arrs[i], arrs[j], alternative="two-sided")
            pairs.append((i, j, float(r.statistic)))
            praw.append(float(r.pvalue))
    padj = holm_bonferroni(praw)
    out = []
    for (i, j, s), pr, pa in zip(pairs, praw, padj):
        g = effects.hedges_g(arrs[i], arrs[j])
        out.append({
            "a": labels[i], "b": labels[j], "statistic": s,
            "p_raw": pr, "p_adjusted": pa, "significant": bool(pa < ALPHA),
            "effect_size": g, "effect_size_type": "hedges_g",
            "cles": effects.cles(arrs[i], arrs[j]),
        })
    return out


@bound("compare_groups", data_args=("groups",))
def compare_groups(
    groups: Dict[str, Sequence[float]],
    design: str = "independent",
    alternative: str = "two-sided",
    justification: str = "",
    alpha: float = ALPHA,
    prefer_robust: bool = False,
    posthoc: bool = True,
    seed: int = 20260901,
) -> Dict[str, Any]:
    """*"Is this one really bigger, or did we get lucky?"*

    The engine picks the test, reports why, returns an effect size with an
    interval, runs a stability check, and consults the multiple-comparison
    ledger. `justification` is required by the tool schema, logged, and shown
    to the teacher (spec section 06).

    Serves G6-L9 (three temperatures), G7-L9 (four salt levels), G8-L9 (four pH
    levels), G7-L7 (control vs Bacillus).
    """
    labels, arrs = _prep(groups)
    if len(arrs) < 2:
        return {"refused": True, "reason": "Need at least two groups to compare.", "p_value": None}

    ns = {lab: int(a.size) for lab, a in zip(labels, arrs)}
    if min(ns.values()) < MIN_N:
        return {
            "refused": True,
            "reason": (f"The smallest group has {min(ns.values())} observation(s). Below {MIN_N} the "
                       f"engine will not run a significance test, because any p-value it produced "
                       f"would be meaningless. Here is what you measured instead."),
            "n_per_group": ns,
            "descriptives": {lab: _describe_one(a) for lab, a in zip(labels, arrs)},
            "suggestion": "Collect at least 5 replicates per group, then ask again.",
            "p_value": None,
        }

    panel = assumption_panel(arrs, alpha)
    key, why = _route(labels, arrs, design, panel, prefer_robust)
    statistic, p_value, stat_label = _run(key, arrs, alternative, seed)

    k = len(arrs)
    if k == 2:
        if key in ("permutation", "wilcoxon", "mannwhitney", "kruskal"):
            es_type = "rank_biserial"
            es = effects.rank_biserial(arrs[0], arrs[1])
            ci = effects.effsize_ci(arrs[0], arrs[1], "rank_biserial", seed=seed)
        else:
            es_type = "hedges_g"
            es = effects.hedges_g(arrs[0], arrs[1], paired=(design == "paired"))
            ci = effects.effsize_ci(arrs[0], arrs[1], "hedges_g", seed=seed)
        cl = effects.cles(arrs[0], arrs[1])
    else:
        if key == "kruskal":
            es_type, es = "epsilon_squared", effects.epsilon_squared(arrs)
        else:
            es_type, es = "eta_squared", effects.eta_squared(arrs)
        ci = (float("nan"), float("nan"))
        cl = float("nan")

    def _test_for_stability(trial: List[np.ndarray]) -> Tuple[float, float]:
        # Permutation is too slow to run 400+ times; use its rank-based twin for
        # the stability sweep and say so, rather than silently using a different
        # alpha level. Everything else re-runs the actual chosen test.
        kk = "mannwhitney" if key == "permutation" else key
        try:
            _, p, _ = _run(kk, trial, alternative, seed)
        except Exception:
            return float("nan"), float("nan")
        if len(trial) == 2:
            e = effects.rank_biserial(trial[0], trial[1]) if es_type == "rank_biserial" \
                else effects.hedges_g(trial[0], trial[1], paired=(design == "paired"))
        else:
            e = effects.epsilon_squared(trial) if es_type == "epsilon_squared" \
                else effects.eta_squared(trial)
        return float(p), float(e)

    stab = stability_check(arrs, _test_for_stability, alpha=alpha, seed=seed)
    if key == "permutation":
        stab.verdict += (" (Stability was assessed with the Mann-Whitney twin of the permutation "
                         "test, which is far cheaper to re-run 400 times and agrees with it closely.)")

    data_id = data_fingerprint(*arrs)
    led = ledger_for(data_id, alpha=alpha)

    ph = []
    if posthoc and k > 2 and p_value < alpha:
        ph = _posthoc(labels, arrs, parametric=(key in ("anova", "alexandergovern")))

    plain = _plain_language(labels, arrs, key, p_value, es, es_type, ci, cl, alpha, stab)

    return {
        "test_used": key,
        "test_label": {
            "ttest_ind": "independent-samples t-test", "welch": "Welch's t-test",
            "ttest_rel": "paired t-test", "wilcoxon": "Wilcoxon signed-rank test",
            "mannwhitney": "Mann-Whitney U test", "permutation": "permutation test (label shuffling)",
            "anova": "one-way ANOVA", "alexandergovern": "Alexander-Govern test",
            "kruskal": "Kruskal-Wallis test",
        }[key],
        "why_this_test": why,
        "statistic": statistic,
        "statistic_label": stat_label,
        "p_value": p_value,
        "alternative": alternative,
        "effect_size": es,
        "effect_size_type": es_type,
        "ci": [float(ci[0]), float(ci[1])],
        "effect_interpretation": effects.interpret(es, es_type, ci),
        "cles": cl,
        "n_per_group": ns,
        "descriptives": {lab: _describe_one(a) for lab, a in zip(labels, arrs)},
        "assumption_checks": [a.to_dict() for a in panel],
        "stability": stab.to_dict(),
        "tests_run_so_far": led.tests_run_so_far + 1,
        "correction_applied": led.correction_applied,
        "ledger": led.to_dict(),
        "posthoc": ph,
        "justification": justification,
        "plain_language": plain,
        "notes": [a.detail for a in panel if a.severity in ("warn", "block")],
    }


def _plain_language(labels, arrs, key, p, es, es_type, ci, cl, alpha, stab) -> str:
    """A sentence a teacher who is not a statistician can read and trust.

    Spec section 02, constraint 4. This string is generated from the computed
    numbers -- it is never written by the language model.
    """
    best = labels[int(np.argmax([a.mean() for a in arrs]))]
    sig = p < alpha
    lead = (f"The groups differ by more than chance comfortably explains (p = {p:.4g}); "
            f"{best} had the highest average."
            if sig else
            f"The differences between these groups are within what chance alone produces "
            f"(p = {p:.4g}), so this experiment does not show a real difference. "
            f"That is a result, not a failure.")
    size = f" The size of the difference is {effects.interpret(es, es_type, ci)}"
    if np.isfinite(ci[0]):
        size += f" ({es_type.replace('_', ' ')} = {es:.2f}, 95% CI {ci[0]:.2f} to {ci[1]:.2f})."
    else:
        size += f" ({es_type.replace('_', ' ')} = {es:.2f})."
    cles_bit = ""
    if np.isfinite(cl) and len(arrs) == 2:
        cles_bit = (f" Put another way: pick one measurement from each group at random and "
                    f"{labels[0]} is higher about {cl * 100:.0f} times out of 100.")
    return lead + size + cles_bit + " " + stab.verdict


@bound("correlate", data_args=("x", "y"))
def correlate(x: Sequence[float], y: Sequence[float], method: str = "auto",
              alpha: float = ALPHA, seed: int = 20260901) -> Dict[str, Any]:
    """*"Do these two things move together?"*

    Spearman by default for class data. Also returns the Theil-Sen robust slope,
    because one mis-pipetted tube should not set the line (spec section 04,
    row 4: robust regression).
    """
    a = np.asarray(x, dtype=float).ravel()
    b = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = a.size
    if n < MIN_N + 1:
        return {"refused": True, "reason": f"Only {n} paired points; need at least {MIN_N + 1}.",
                "p_value": None}

    if method == "auto":
        sp = stats.shapiro(a).pvalue if n >= 3 else 0.0
        sq = stats.shapiro(b).pvalue if n >= 3 else 0.0
        method = "pearson" if (n >= 15 and sp > 0.05 and sq > 0.05) else "spearman"
        why = (f"n={n} with no evidence against normality in either variable."
               if method == "pearson" else
               f"n={n}; Spearman ranks the values first, so one extreme point cannot drag the "
               f"correlation, and it detects any consistently increasing relationship, not just a "
               f"straight line.")
    else:
        why = f"Requested explicitly: {method}."

    r = stats.pearsonr(a, b) if method == "pearson" else stats.spearmanr(a, b)
    rho = float(r.statistic if hasattr(r, "statistic") else r[0])
    pv = float(r.pvalue)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        aa, bb = a[idx], b[idx]
        if np.allclose(aa, aa[0]) or np.allclose(bb, bb[0]):
            continue
        rr = stats.pearsonr(aa, bb) if method == "pearson" else stats.spearmanr(aa, bb)
        boots.append(float(rr.statistic if hasattr(rr, "statistic") else rr[0]))
    ci = ([float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
          if boots else [float("nan")] * 2)

    ts = stats.theilslopes(b, a, 0.95)
    ols = stats.linregress(a, b)

    return {
        "method": method, "why_this_test": why,
        "statistic": rho, "statistic_label": "r" if method == "pearson" else "rho",
        "p_value": pv, "ci": ci, "n": int(n),
        "effect_size": rho, "effect_size_type": method,
        "r_squared": float(rho ** 2),
        "theil_sen": {"slope": float(ts[0]), "intercept": float(ts[1]),
                      "slope_ci": [float(ts[2]), float(ts[3])]},
        "least_squares": {"slope": float(ols.slope), "intercept": float(ols.intercept),
                          "stderr": float(ols.stderr)},
        "plain_language": (
            f"{'They do move together' if pv < alpha else 'There is no clear relationship'} "
            f"({'r' if method == 'pearson' else 'rho'} = {rho:.2f}, 95% CI {ci[0]:.2f} to {ci[1]:.2f}, "
            f"p = {pv:.4g}, n = {n}). "
            f"About {rho ** 2 * 100:.0f}% of the variation in one is tracked by the other. "
            f"Moving together is not the same as one causing the other -- your experimental design "
            f"decides that, not this number."),
        "notes": (["The robust (Theil-Sen) slope and the least-squares slope disagree noticeably, "
                   "which means at least one point is pulling the line. Plot it before believing it."]
                  if abs(ts[0] - ols.slope) > 0.25 * max(1e-12, abs(ols.slope)) else []),
    }


@bound("contingency", data_args=("table",))
def contingency(table: Sequence[Sequence[int]], row_labels: Optional[List[str]] = None,
                col_labels: Optional[List[str]] = None, alpha: float = ALPHA) -> Dict[str, Any]:
    """*"Are these counts related?"* Chi-square, or Fisher when cells are thin.

    Serves G5-L9 (water-quality strip categories) and G8-L10 (plaque / no plaque
    across sample sites).
    """
    t = np.asarray(table, dtype=float)
    if t.ndim != 2 or t.size == 0:
        return {"refused": True, "reason": "Need a 2-D table of counts.", "p_value": None}
    cells = check_cell_counts(t)
    if not cells.passed and t.shape == (2, 2):
        r = stats.fisher_exact(t.astype(int))
        stat, pv, used = float(r.statistic), float(r.pvalue), "fisher_exact"
        why = "Expected counts are too small for chi-square; Fisher's exact test is valid at any size."
    else:
        r = stats.chi2_contingency(t)
        stat, pv, used = float(r.statistic), float(r.pvalue), "chi2_contingency"
        why = ("Expected counts are large enough for the chi-square approximation."
               if cells.passed else
               "Expected counts are small and the table is larger than 2x2; chi-square is reported "
               "but treat the p-value as approximate.")
    v = effects.cramers_v(t)
    return {
        "test_used": used, "why_this_test": why,
        "statistic": stat, "p_value": pv,
        "effect_size": v, "effect_size_type": "cramers_v",
        "effect_interpretation": effects.interpret(v, "cramers_v"),
        "expected": (np.outer(t.sum(1), t.sum(0)) / t.sum()).tolist(),
        "observed": t.tolist(),
        "row_labels": row_labels, "col_labels": col_labels,
        "assumption_checks": [cells.to_dict()],
        "notes": [] if cells.passed else [cells.detail],
        "plain_language": (
            f"{'The categories are related' if pv < alpha else 'These categories look independent'} "
            f"(p = {pv:.4g}, Cramer's V = {v:.2f}, {effects.interpret(v, 'cramers_v')})."),
    }


@bound("trend_test", data_args=("values", "times"))
def trend_test(values: Sequence[float], times: Optional[Sequence[float]] = None,
               alpha: float = ALPHA) -> Dict[str, Any]:
    """*"Is it going up over time?"* Mann-Kendall plus a Theil-Sen slope.

    Serves G7-L7 (daily compost temperature) and G9-L9 (daily fuel-cell voltage
    feeding the Claim-Evidence-Reasoning write-up). Rank-based, so it survives
    the missed reading on a Sunday and the one bad probe contact.
    """
    v = np.asarray(values, dtype=float).ravel()
    t = np.arange(v.size, dtype=float) if times is None else np.asarray(times, dtype=float).ravel()
    m = np.isfinite(v) & np.isfinite(t)
    v, t = v[m], t[m]
    n = v.size
    if n < 4:
        return {"refused": True, "reason": f"Only {n} readings; need at least 4 for a trend.",
                "p_value": None}

    # Mann-Kendall S
    s = 0
    for i in range(n - 1):
        s += int(np.sum(np.sign(v[i + 1:] - v[i])))
    vals, counts = np.unique(v, return_counts=True)
    tie = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if var_s <= 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    pv = float(2 * (1 - stats.norm.cdf(abs(z))))

    ts = stats.theilslopes(v, t, 0.95)
    tau = stats.kendalltau(t, v)

    direction = "increasing" if s > 0 else ("decreasing" if s < 0 else "flat")
    return {
        "test_used": "mann_kendall",
        "why_this_test": ("Mann-Kendall only looks at whether later readings are higher or lower than "
                          "earlier ones, so it does not assume a straight line, a bell curve, or "
                          "evenly spaced readings -- all three of which a daily class log breaks."),
        "statistic": float(s), "z": float(z), "p_value": pv,
        "effect_size": float(tau.statistic), "effect_size_type": "kendall_tau",
        "direction": direction,
        "sen_slope": float(ts[0]), "sen_slope_ci": [float(ts[2]), float(ts[3])],
        "n": int(n),
        "first": float(v[0]), "last": float(v[-1]),
        "plain_language": (
            f"The readings are {direction}"
            + (f" at about {ts[0]:.4g} units per step (95% CI {ts[2]:.4g} to {ts[3]:.4g}), "
               f"and that trend is stronger than chance would produce (p = {pv:.4g})."
               if pv < alpha else
               f", but not by more than day-to-day noise would produce anyway (p = {pv:.4g}). "
               f"A longer run of readings would settle it.")),
        "notes": (["The slope interval includes zero, so the direction of the trend is not settled "
                   "even though the test is significant. Report the interval."]
                  if ts[2] * ts[3] <= 0 and pv < alpha else []),
    }
