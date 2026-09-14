"""Diversity and composition (spec section 04, row 7).

    "MUST ENCODE: relative abundances sum to 1, which violates the independence
    assumption of every test in row 1. Without CLR, the platform will teach
    students to produce spurious results with great confidence."

That warning is the design brief for this module. `compare_composition` refuses
to run an ordinary group comparison on proportions and routes to CLR + PERMANOVA
instead, and it says why in words a Grade 9 can follow.

scikit-bio is the Tier-1 choice (BSD) but is not in Pyodide. Everything here is
numpy/scipy, so the Winogradsky lesson (G9-L12) works offline in a browser on a
school PC, which was the whole point.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from .transcript import bound

__all__ = ["alpha_diversity", "beta_diversity", "clr_transform", "permanova",
           "compare_composition", "rarefaction_curve"]


def _counts_matrix(samples: Dict[str, Dict[str, float]]) -> tuple:
    taxa = sorted({t for s in samples.values() for t in s})
    names = list(samples)
    M = np.array([[float(samples[n].get(t, 0.0)) for t in taxa] for n in names])
    return names, taxa, M


@bound("alpha_diversity", data_args=("samples",))
def alpha_diversity(samples: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Richness, Shannon, Simpson, Chao1, Pielou's evenness -- per sample.

    Serves G9-L12: compare microbial communities across Winogradsky layers.

    Chao1 is included because it answers the question a student always asks
    next -- "but how many did we MISS?" -- from the singletons and doubletons
    they can see in their own count table.
    """
    names, taxa, M = _counts_matrix(samples)
    if M.size == 0:
        return {"refused": True, "reason": "No taxa found."}

    out = {}
    for name, row in zip(names, M):
        total = row.sum()
        obs = int((row > 0).sum())
        if total <= 0:
            out[name] = {"n_observed": 0, "total_count": 0}
            continue
        p = row[row > 0] / total
        shannon = float(-(p * np.log(p)).sum())
        simpson = float(1.0 - (p ** 2).sum())
        inv_simpson = float(1.0 / (p ** 2).sum())
        evenness = float(shannon / np.log(obs)) if obs > 1 else float("nan")
        f1 = int((row == 1).sum())
        f2 = int((row == 2).sum())
        chao1 = obs + (f1 * (f1 - 1)) / (2 * (f2 + 1))
        out[name] = {
            "n_observed": obs, "total_count": int(total),
            "shannon": shannon, "simpson": simpson, "inverse_simpson": inv_simpson,
            "pielou_evenness": evenness,
            "chao1": float(chao1),
            "singletons": f1, "doubletons": f2,
            "estimated_unseen": float(chao1 - obs),
        }

    best = max(out, key=lambda k: out[k].get("shannon", -1)) if out else None
    depths = [v.get("total_count", 0) for v in out.values()]
    uneven = max(depths) > 3 * max(1, min(depths)) if depths else False

    if best is None:
        headline = ""
    else:
        b = out[best]
        headline = (f"{best} is the most diverse (Shannon {b['shannon']:.2f}, "
                    f"{b['n_observed']} kinds seen")
        headline += (f", Chao1 estimates about {b['chao1']:.0f} really present)."
                     if b.get("estimated_unseen", 0) > 0.5 else ").")

    return {
        "samples": out, "taxa": taxa, "n_taxa": len(taxa),
        "most_diverse": best,
        "sequencing_depth_warning": (
            f"Your samples were counted to very different depths ({min(depths)} to {max(depths)}). "
            f"More counting finds more taxa, so a deeper sample looks more diverse whether or not "
            f"it is. Rarefy to the smallest depth before comparing, or compare evenness instead of "
            f"richness." if uneven else None),
        "teaches": (
            "Richness counts how many kinds there are. Shannon and Simpson also ask how EVENLY "
            "they are spread: a layer with one dominant organism and nine rare ones is less "
            "diverse than a layer with ten equal ones, even though both have ten kinds."),
        "plain_language": f"{len(out)} samples across {len(taxa)} taxa. {headline}",
        "p_value": None,
    }


@bound("clr_transform", data_args=("samples",))
def clr_transform(samples: Dict[str, Dict[str, float]], pseudocount: float = 0.5
                  ) -> Dict[str, Any]:
    """Centred log-ratio transform -- the fix for compositional data.

    Why it matters, in the terms a student can check: if one organism doubles,
    every OTHER organism's percentage goes down, even though nothing happened
    to them. Percentages are not independent measurements, so the t-tests and
    ANOVAs in `inference` are invalid on them. CLR divides each count by the
    geometric mean of its sample and takes a log, which breaks that forced
    dependence.

    The pseudocount is a real choice with a real consequence, so it is a named
    parameter and its effect is reported rather than hidden.
    """
    names, taxa, M = _counts_matrix(samples)
    if M.size == 0:
        return {"refused": True, "reason": "No taxa found."}
    zeros = int((M == 0).sum())
    X = M + float(pseudocount)
    gm = np.exp(np.log(X).mean(axis=1, keepdims=True))
    C = np.log(X / gm)
    return {
        "names": names, "taxa": taxa,
        "clr": C.tolist(),
        "pseudocount": float(pseudocount),
        "n_zeros_replaced": zeros,
        "zero_fraction": float(zeros / M.size),
        "note": ("scikit-bio renamed `multiplicative_replacement` to `multi_replace`; this "
                 "implementation depends on neither. The pseudocount above was added to "
                 f"{zeros} zero cells ({zeros / M.size:.0%} of the table). A different pseudocount "
                 f"gives different CLR values -- if your conclusion changes when you change it, "
                 f"your conclusion is about the pseudocount."),
        "teaches": (
            "Percentages of a whole cannot move independently. If the bacteria double, the "
            "percentage of everything else falls without anything happening to it. CLR is how you "
            "get back to numbers that can be compared."),
        "plain_language": (
            f"Transformed {len(names)} samples x {len(taxa)} taxa to centred log-ratios. "
            f"Positive means that taxon is above the sample's typical level; negative means below. "
            f"These values CAN be compared between samples; raw percentages cannot."),
        "p_value": None,
    }


@bound("beta_diversity", data_args=("samples",))
def beta_diversity(samples: Dict[str, Dict[str, float]], metric: str = "braycurtis"
                   ) -> Dict[str, Any]:
    """Between-sample distances, plus a PCoA ordination.

    `aitchison` is the compositionally correct metric (Euclidean distance on
    CLR values) and is what the platform recommends. Bray-Curtis is offered
    because it is what the literature a student will read uses.
    """
    names, taxa, M = _counts_matrix(samples)
    if M.shape[0] < 2:
        return {"refused": True, "reason": "Need at least two samples."}

    if metric == "aitchison":
        clr = clr_transform.__wrapped__(samples)
        X = np.asarray(clr["clr"])
        D = squareform(pdist(X, metric="euclidean"))
        why = ("Aitchison distance: Euclidean distance on CLR values. This is the metric that "
               "respects the fact that your data are proportions.")
    elif metric == "jaccard":
        B = (M > 0).astype(float)
        D = squareform(pdist(B, metric="jaccard"))
        why = "Jaccard: presence/absence only. Ignores how MUCH of each taxon there is."
    else:
        rel = M / np.maximum(M.sum(axis=1, keepdims=True), 1e-12)
        D = squareform(pdist(rel, metric="braycurtis"))
        why = ("Bray-Curtis on relative abundances. Widely used and easy to read, but it is a "
               "distance between proportions, so significance tests on it need PERMANOVA "
               "(permutation-based), never a t-test.")

    # Principal coordinates analysis: double-centre, then eigendecompose.
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    pos = vals > 1e-12
    coords = vecs[:, pos] * np.sqrt(vals[pos])
    explained = (vals[pos] / vals[pos].sum()).tolist() if pos.any() else []

    return {
        "names": names, "metric": metric, "why_this_metric": why,
        "distance_matrix": D.tolist(),
        "pcoa": {
            "coordinates": coords[:, :3].tolist() if coords.shape[1] >= 1 else [],
            "explained_variance": explained[:5],
            "axis_labels": [f"PCo{i + 1} ({e:.0%})" for i, e in enumerate(explained[:3])],
        },
        "most_similar_pair": _closest_pair(names, D),
        "most_different_pair": _farthest_pair(names, D),
        "plain_language": (
            f"Compared {n} samples using {metric}. "
            + (f"The first two ordination axes carry {sum(explained[:2]):.0%} of the differences. "
               if len(explained) >= 2 else "")
            + f"The two most similar samples are {_closest_pair(names, D)}; the two most different "
              f"are {_farthest_pair(names, D)}."),
        "p_value": None,
    }


def _closest_pair(names, D):
    d = D.copy()
    np.fill_diagonal(d, np.inf)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return f"{names[i]} and {names[j]}"


def _farthest_pair(names, D):
    i, j = np.unravel_index(np.argmax(D), D.shape)
    return f"{names[i]} and {names[j]}"


@bound("permanova", data_args=("distance_matrix", "groups"))
def permanova(distance_matrix: Sequence[Sequence[float]], groups: Sequence[str],
              n_permutations: int = 9999, seed: int = 20260901) -> Dict[str, Any]:
    """PERMANOVA: is the difference between groups of communities real?

    The right test for community data, and -- usefully -- it is the same idea
    as the Grade 7 shuffle test: compute a statistic, shuffle the labels many
    times, see how often chance does as well. Same lesson, bigger object.
    """
    D = np.asarray(distance_matrix, dtype=float)
    g = np.asarray([str(x) for x in groups])
    n = D.shape[0]
    if D.shape[0] != D.shape[1] or g.size != n:
        return {"refused": True, "reason": "Distance matrix and group labels must match in size."}
    levels = sorted(set(g.tolist()))
    if len(levels) < 2:
        return {"refused": True, "reason": "Need at least two groups."}
    if n < 6:
        return {"refused": True, "reason": f"Only {n} samples. PERMANOVA needs at least 6 to say "
                                           f"anything, and really wants 5 per group."}

    def pseudo_f(labels: np.ndarray) -> float:
        sst = float((D ** 2).sum() / (2 * n))
        ssw = 0.0
        for lv in levels:
            idx = np.nonzero(labels == lv)[0]
            k = idx.size
            if k < 2:
                continue
            sub = D[np.ix_(idx, idx)]
            ssw += float((sub ** 2).sum() / (2 * k))
        ssb = sst - ssw
        a, N = len(levels), n
        denom = ssw / (N - a) if N > a else np.nan
        return float((ssb / (a - 1)) / denom) if denom and denom > 0 else float("nan")

    observed = pseudo_f(g)
    rng = np.random.default_rng(seed)
    null = np.array([pseudo_f(rng.permutation(g)) for _ in range(int(n_permutations))])
    null = null[np.isfinite(null)]
    count = int((null >= observed).sum())
    p = (count + 1) / (null.size + 1)

    return {
        "test_used": "permanova", "pseudo_f": observed, "p_value": float(p),
        "n_permutations": int(null.size), "n_at_least_as_extreme": count,
        "groups": levels,
        "n_per_group": {lv: int((g == lv).sum()) for lv in levels},
        "null_distribution": [float(x) for x in
                              rng.choice(null, size=min(2000, null.size), replace=False)],
        "effect_size": float(observed / (observed + (n - len(levels)) / (len(levels) - 1)))
                       if np.isfinite(observed) else float("nan"),
        "effect_size_type": "pseudo_r_squared",
        "caveat": ("PERMANOVA is sensitive to differences in SPREAD as well as differences in "
                   "location. A significant result can mean 'the groups are in different places' "
                   "or 'one group is much more variable'. Look at the ordination plot before "
                   "deciding which you have."),
        "plain_language": (
            f"Shuffled the group labels {null.size:,} times. The real grouping separated the "
            f"communities better than {null.size - count:,} of those {null.size:,} random "
            f"groupings (pseudo-F = {observed:.3f}, p = {p:.4g}). "
            + ("The communities really do differ between groups."
               if p < 0.05 else
               "Chance rearrangements do about as well, so these communities are not "
               "distinguishable by this grouping.")),
    }


@bound("compare_composition", data_args=("samples",))
def compare_composition(samples: Dict[str, Dict[str, float]], groups: Dict[str, str],
                        metric: str = "aitchison", seed: int = 20260901) -> Dict[str, Any]:
    """The safe front door for "are these communities different?".

    This is the function the chatbot is allowed to call. It refuses to hand
    proportions to a t-test, runs the compositionally correct pipeline, and
    explains the refusal -- which is the lesson the spec says the platform must
    teach rather than document.
    """
    names, taxa, M = _counts_matrix(samples)
    labels = [groups.get(n, "ungrouped") for n in names]
    if len(set(labels)) < 2:
        return {"refused": True, "reason": "All samples are in the same group."}

    beta = beta_diversity.__wrapped__(samples, metric=metric)
    if beta.get("refused"):
        return beta
    perm = permanova.__wrapped__(beta["distance_matrix"], labels, seed=seed)
    alpha = alpha_diversity.__wrapped__(samples)

    # Alpha diversity CAN be compared with ordinary tests -- it is one number
    # per sample, not a composition. Route it through the normal engine.
    from .inference import compare_groups as _cg
    shannon_by_group: Dict[str, List[float]] = {}
    for name, lab in zip(names, labels):
        shannon_by_group.setdefault(lab, []).append(alpha["samples"][name]["shannon"])
    alpha_test = None
    if all(len(v) >= 3 for v in shannon_by_group.values()):
        alpha_test = _cg.__wrapped__(shannon_by_group,
                                     justification="Shannon diversity is one value per sample, so "
                                                   "an ordinary group comparison is valid on it.")

    return {
        "groups": sorted(set(labels)),
        "n_samples": len(names), "n_taxa": len(taxa),
        "alpha": alpha,
        "alpha_comparison": alpha_test,
        "beta": beta,
        "permanova": perm,
        "p_value": perm.get("p_value"),
        "why_not_a_t_test": (
            "You cannot run a t-test or ANOVA on the abundances themselves. Every sample's "
            "percentages are forced to add up to 100%, so the taxa are not independent: if one "
            "goes up, the others must go down whether or not anything happened to them. That "
            "breaks the assumption every one of those tests rests on, and it produces confident "
            "wrong answers rather than obvious errors. Two things ARE comparable with ordinary "
            "tests: a single diversity number per sample (done above), and CLR-transformed "
            "values. Everything else goes through PERMANOVA."),
        "plain_language": (
            perm.get("plain_language", "")
            + (" " + alpha_test["plain_language"].split(".")[0] + " (diversity level)."
               if alpha_test and not alpha_test.get("refused") else "")),
    }


@bound("rarefaction_curve", data_args=("counts",))
def rarefaction_curve(counts: Sequence[float], n_points: int = 20,
                      n_iter: int = 50, seed: int = 20260901) -> Dict[str, Any]:
    """How many kinds would you have found if you had counted fewer?

    The curve that answers "did we look hard enough?". If it is still climbing
    at your actual sample size, you did not.
    """
    c = np.asarray(counts, dtype=float).ravel()
    c = c[c > 0]
    total = int(c.sum())
    if total < 5:
        return {"refused": True, "reason": "Too few individuals counted."}
    pool = np.repeat(np.arange(c.size), c.astype(int))
    rng = np.random.default_rng(seed)
    depths = np.unique(np.linspace(1, total, int(n_points)).astype(int))
    means, los, his = [], [], []
    for d in depths:
        richness = [np.unique(rng.choice(pool, int(d), replace=False)).size for _ in range(n_iter)]
        means.append(float(np.mean(richness)))
        los.append(float(np.percentile(richness, 2.5)))
        his.append(float(np.percentile(richness, 97.5)))
    # Is the curve still climbing? Compare the last decile's slope to the first.
    slope_end = (means[-1] - means[-max(2, len(means) // 5)]) / max(1, depths[-1] - depths[-max(2, len(depths) // 5)])
    slope_start = (means[1] - means[0]) / max(1, depths[1] - depths[0]) if len(means) > 1 else 0.0
    saturated = bool(slope_end < 0.1 * max(slope_start, 1e-12))

    return {
        "depths": depths.tolist(), "mean_richness": means,
        "ci_low": los, "ci_high": his,
        "observed_richness": int(c.size), "total_counted": total,
        "saturated": saturated,
        "end_slope": float(slope_end),
        "plain_language": (
            f"You counted {total} individuals and found {c.size} kinds. "
            + ("The curve has flattened out, so counting more would find few new kinds -- you "
               "looked hard enough."
               if saturated else
               f"The curve is still rising at {slope_end:.3f} new kinds per individual counted, so "
               f"there are more kinds there that you did not see. Any comparison of richness with "
               f"another sample has to account for that.")),
        "p_value": None,
    }
