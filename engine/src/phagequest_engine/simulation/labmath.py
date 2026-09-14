"""Wet-lab arithmetic (spec section 03.3, last row).

    "Five lines of our own code, bound directly to G5-L6's 1:10 dilution chain
    so the student sees the same maths they did with coloured water, now
    counting plaques."

It is more than five lines because the honest version carries the counting
uncertainty with it. A titre from a single plate with 7 plaques is not
"7 x 10^6 PFU/mL"; it is a number with a 95% interval running from about
3 x 10^6 to 14 x 10^6, and that interval is the lesson.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from ..transcript import bound

__all__ = ["dilution_series", "pfu_per_ml", "moi", "titre_from_plates"]


@bound("dilution_series")
def dilution_series(start_concentration: float = 1e9, fold: float = 10.0,
                    steps: int = 8, volume_transferred: float = 0.1,
                    volume_diluent: float = 0.9) -> Dict[str, Any]:
    """The 1:10 chain from G5-L6, with the powers of ten made explicit.

    A Grade 5 sees the coloured-water version of exactly this table. The same
    function, unchanged, is what a Grade 9 uses to work out which tube to plate.
    """
    if fold <= 1:
        return {"refused": True, "reason": "The dilution fold must be greater than 1."}
    actual_fold = (volume_transferred + volume_diluent) / max(volume_transferred, 1e-12)
    rows = []
    c = float(start_concentration)
    for i in range(int(steps) + 1):
        rows.append({
            "tube": i,
            "dilution_factor": float(actual_fold ** i),
            "dilution_label": f"10^-{i}" if abs(actual_fold - 10) < 1e-9 else f"1:{actual_fold ** i:.0f}",
            "concentration": c,
            "log10_concentration": float(np.log10(c)) if c > 0 else float("-inf"),
            "expected_plaques_from_100uL": float(c * 0.1),
            "countable": bool(30 <= c * 0.1 <= 300),
        })
        c /= actual_fold

    countable = [r for r in rows if r["countable"]]
    return {
        "rows": rows,
        "requested_fold": float(fold), "actual_fold": float(actual_fold),
        "volumes": {"transferred_mL": volume_transferred, "diluent_mL": volume_diluent},
        "countable_tubes": [r["tube"] for r in countable],
        "fold_mismatch": (
            f"You asked for a 1:{fold:g} dilution but transferring {volume_transferred} mL into "
            f"{volume_diluent} mL gives 1:{actual_fold:.3g}. The total volume is what divides, not "
            f"the diluent. This is the single most common dilution mistake."
            if abs(actual_fold - fold) > 1e-6 else None),
        "plain_language": (
            f"Each step divides by {actual_fold:g}, so tube {steps} is {actual_fold ** steps:.3g} "
            f"times weaker than tube 0. "
            + (f"Plate 100 uL from tube {countable[0]['tube']}"
               + (f" or {countable[-1]['tube']}" if len(countable) > 1 else "")
               + " to land in the countable 30-300 plaque range."
               if countable else
               "None of these tubes lands in the countable 30-300 range -- extend the series.")),
        "p_value": None,
    }


@bound("pfu_per_ml")
def pfu_per_ml(plaque_count: int, dilution_factor: float, volume_plated_mL: float = 0.1
               ) -> Dict[str, Any]:
    """Titre from one plate, with its Poisson counting interval.

    The interval is the point. A plate with 7 plaques and a plate with 70 give
    titres differing by a factor of ten, but the 7-plaque titre is uncertain by
    a factor of two and the 70-plaque one by about 25%. That is why the
    countable range exists, and a student who has seen this table never asks
    "why can't I just use the plate I have?" again.
    """
    n = int(plaque_count)
    if n < 0 or dilution_factor <= 0 or volume_plated_mL <= 0:
        return {"refused": True, "reason": "Counts must be >= 0, and factors and volumes > 0."}

    titre = n * dilution_factor / volume_plated_mL
    # Exact Poisson interval on the count (Garwood), then propagated.
    lo_n = 0.0 if n == 0 else float(stats.chi2.ppf(0.025, 2 * n) / 2)
    hi_n = float(stats.chi2.ppf(0.975, 2 * (n + 1)) / 2)
    lo = lo_n * dilution_factor / volume_plated_mL
    hi = hi_n * dilution_factor / volume_plated_mL

    if n == 0:
        reliability = ("A count of zero does not give a titre. It gives an upper limit: the true "
                       "concentration is below the top of this interval. Report it that way.")
        grade = "upper limit only"
    elif n < 30:
        reliability = (f"{n} plaques is below the countable range of 30-300. The interval is "
                       f"{hi / max(lo, 1e-12):.1f}-fold wide -- plate a less dilute tube.")
        grade = "too few"
    elif n > 300:
        reliability = (f"{n} plaques is above 300; plaques overlap at that density and you have "
                       f"almost certainly undercounted. Plate a more dilute tube.")
        grade = "too many"
    else:
        reliability = (f"{n} plaques is in the countable range, and the interval is only "
                       f"{hi / max(lo, 1e-12):.2f}-fold wide. This is a usable titre.")
        grade = "good"

    return {
        "plaque_count": n, "dilution_factor": float(dilution_factor),
        "volume_plated_mL": float(volume_plated_mL),
        "titre_pfu_per_mL": float(titre),
        "ci": [float(lo), float(hi)],
        "log10_titre": float(np.log10(titre)) if titre > 0 else None,
        "relative_uncertainty": float(1 / np.sqrt(n)) if n > 0 else None,
        "count_quality": grade, "reliability_note": reliability,
        "formula": "titre = plaques x dilution factor / volume plated (mL)",
        "plain_language": (
            f"{n} plaques from {volume_plated_mL} mL of a 1:{dilution_factor:g} dilution gives "
            f"{titre:.3g} PFU/mL (95% CI {lo:.3g} to {hi:.3g}). " + reliability),
        "p_value": None,
    }


@bound("titre_from_plates", data_args=("plaque_counts",))
def titre_from_plates(plaque_counts: Sequence[int], dilution_factors: Sequence[float],
                      volume_plated_mL: float = 0.1) -> Dict[str, Any]:
    """Pool several plates into one titre -- the right way to do it.

    Averaging the per-plate titres is wrong: it weights a 5-plaque plate the
    same as a 200-plaque one. The correct estimate pools the counts against the
    total volume-equivalent plated, which is maximum likelihood for a Poisson.
    """
    counts = np.asarray(plaque_counts, dtype=float).ravel()
    dils = np.asarray(dilution_factors, dtype=float).ravel()
    if counts.size != dils.size or counts.size == 0:
        return {"refused": True, "reason": "Give one dilution factor per plate."}
    if np.any(counts < 0) or np.any(dils <= 0):
        return {"refused": True, "reason": "Counts must be >= 0 and dilutions > 0."}

    # Effective undiluted volume each plate represents.
    veq = volume_plated_mL / dils
    total_count = float(counts.sum())
    total_veq = float(veq.sum())
    titre = total_count / total_veq if total_veq > 0 else float("nan")

    lo_n = 0.0 if total_count == 0 else float(stats.chi2.ppf(0.025, 2 * total_count) / 2)
    hi_n = float(stats.chi2.ppf(0.975, 2 * (total_count + 1)) / 2)
    lo, hi = lo_n / total_veq, hi_n / total_veq

    per_plate = [float(c * d / volume_plated_mL) for c, d in zip(counts, dils)]
    naive_mean = float(np.mean(per_plate))

    # Overdispersion: if the plates disagree by more than Poisson allows, the
    # dilutions or the pipetting are the problem, not chance.
    expected = titre * veq
    chi2 = float(np.sum((counts - expected) ** 2 / np.maximum(expected, 1e-12)))
    df = max(1, counts.size - 1)
    p_consistency = float(stats.chi2.sf(chi2, df))

    return {
        "n_plates": int(counts.size),
        "plaque_counts": counts.astype(int).tolist(),
        "dilution_factors": dils.tolist(),
        "pooled_titre_pfu_per_mL": float(titre),
        "ci": [float(lo), float(hi)],
        "per_plate_titres": per_plate,
        "naive_mean_of_titres": naive_mean,
        "consistency_chi2": chi2, "consistency_df": df, "p_value": p_consistency,
        "plates_consistent": bool(p_consistency >= 0.05),
        "method_note": (
            "Counts are pooled against total volume-equivalent, not averaged. Averaging per-plate "
            "titres gives a 5-plaque plate the same weight as a 200-plaque one, which throws away "
            "most of your information."),
        "plain_language": (
            f"Pooling {counts.size} plates gives {titre:.3g} PFU/mL (95% CI {lo:.3g} to {hi:.3g}). "
            f"Averaging the individual titres would have given {naive_mean:.3g} -- "
            + ("close, but the pooled figure is the defensible one. "
               if abs(naive_mean - titre) < 0.2 * titre else "noticeably different. ")
            + (f"The plates agree with each other about as well as chance allows "
               f"(p = {p_consistency:.3g})."
               if p_consistency >= 0.05 else
               f"The plates disagree more than Poisson counting explains (p = {p_consistency:.3g}). "
               f"Check the dilution series and the pipetting before trusting this titre.")),
    }


@bound("moi")
def moi(phage_pfu_per_mL: float, bacteria_cfu_per_mL: float,
        volume_mL: float = 1.0) -> Dict[str, Any]:
    """Multiplicity of infection, plus what Poisson says actually happens at it.

    At MOI 1 a student expects every cell to be infected. Poisson says 37% of
    them are not touched at all. That gap between the intuition and the
    arithmetic is the whole lesson (G8-L10).
    """
    if bacteria_cfu_per_mL <= 0:
        return {"refused": True, "reason": "Bacterial count must be greater than zero."}
    m = float(phage_pfu_per_mL) / float(bacteria_cfu_per_mL)
    p0 = float(np.exp(-m))
    p1 = float(m * np.exp(-m))
    p2plus = float(1 - p0 - p1)
    return {
        "moi": m,
        "phage_total": float(phage_pfu_per_mL * volume_mL),
        "bacteria_total": float(bacteria_cfu_per_mL * volume_mL),
        "fraction_uninfected": p0,
        "fraction_singly_infected": p1,
        "fraction_multiply_infected": p2plus,
        "distribution": {int(k): float(stats.poisson.pmf(k, m)) for k in range(0, 8)},
        "regime": ("low MOI -- most cells escape, and the phage must go through several rounds "
                   "to clear the culture" if m < 0.1 else
                   "high MOI -- almost every cell is hit at once, so you see one round of lysis "
                   "and then nothing" if m > 5 else
                   "moderate MOI"),
        "plain_language": (
            f"MOI {m:.3g}: {phage_pfu_per_mL:.3g} phage per mL against {bacteria_cfu_per_mL:.3g} "
            f"cells per mL. Phage do not queue up politely -- they land at random, so Poisson says "
            f"{p0:.1%} of cells get no phage at all, {p1:.1%} get exactly one, and {p2plus:.1%} get "
            f"two or more. "
            + ("Even at MOI 1, more than a third of the cells are untouched."
               if 0.8 < m < 1.25 else "")),
        "p_value": None,
    }
