"""Stochastic simulation (spec section 03.3).

    "For the capstone question: why does the same sample sometimes give plaques
    and sometimes not? The curriculum's own 'no result is also a result'
    discussion has a mathematical answer, and it is Poisson."

GillesPy2 and StochPy are Tier-1 and not in Pyodide. The Gillespie direct
method is forty lines and exact, so it lives here in Tier-0 where it belongs:
this is the module that answers a Grade 8's question about their own plate, and
it must not require a server round-trip to do it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

from ..transcript import bound

__all__ = ["gillespie_lytic", "extinction_probability", "poisson_plaques"]


@bound("gillespie_lytic")
def gillespie_lytic(S0: int = 200, P0: int = 5, adsorption: float = 1e-3,
                    burst: int = 50, lysis_rate: float = 2.0, growth: float = 0.5,
                    decay: float = 0.05, t_max: float = 20.0, n_runs: int = 30,
                    max_events: int = 400_000, seed: int = 20260901) -> Dict[str, Any]:
    """Exact stochastic simulation (Gillespie direct method), repeated n_runs times.

    Small numbers are where chance actually bites: with 5 phage and 200 cells,
    the deterministic model says "infection happens" and the real world says
    "sometimes". Running the same experiment thirty times and drawing all thirty
    trajectories is the single clearest demonstration of that a student can see.

    Reactions:
        S            -> 2S          rate  growth * S
        S + P        -> I           rate  adsorption * S * P
        I            -> burst * P   rate  lysis_rate * I
        P            -> 0           rate  decay * P
    """
    rng = np.random.default_rng(seed)
    runs: List[Dict[str, Any]] = []
    extinctions = 0
    peak_phages: List[float] = []
    clear_times: List[float] = []

    for r in range(int(n_runs)):
        S, I, P = int(S0), 0, int(P0)
        t = 0.0
        ts, Ss, Is, Ps = [0.0], [S], [I], [P]
        events = 0
        while t < t_max and events < max_events:
            a1 = growth * S
            a2 = adsorption * S * P
            a3 = lysis_rate * I
            a4 = decay * P
            a0 = a1 + a2 + a3 + a4
            if a0 <= 0:
                break
            t += float(rng.exponential(1.0 / a0))
            if t > t_max:
                break
            u = rng.random() * a0
            if u < a1:
                S += 1
            elif u < a1 + a2:
                S -= 1
                I += 1
                P -= 1
            elif u < a1 + a2 + a3:
                I -= 1
                P += int(burst)
            else:
                P -= 1
            events += 1
            # Subsample the trace: 400k events is not a drawable curve.
            if events % 50 == 0:
                ts.append(t); Ss.append(S); Is.append(I); Ps.append(P)
        ts.append(min(t, t_max)); Ss.append(S); Is.append(I); Ps.append(P)

        died_out = (P == 0 and I == 0)
        extinctions += int(died_out)
        peak_phages.append(float(max(Ps)))
        if S == 0:
            idx = next((i for i, v in enumerate(Ss) if v == 0), None)
            if idx is not None:
                clear_times.append(float(ts[idx]))
        runs.append({
            "run": r, "time": ts, "S": Ss, "I": Is, "P": Ps,
            "phage_extinct": bool(died_out), "host_cleared": bool(S == 0),
            "final_S": int(S), "final_P": int(P), "events": events,
        })

    n = len(runs)
    p_ext = extinctions / n if n else float("nan")
    # Wilson interval -- correct at the small n and extreme proportions this
    # produces, where the normal approximation is simply wrong.
    lo, hi = _wilson(extinctions, n)

    return {
        "n_runs": n, "runs": runs,
        "extinction_count": extinctions,
        "extinction_probability": float(p_ext),
        "extinction_ci": [lo, hi],
        "host_cleared_count": sum(r["host_cleared"] for r in runs),
        "peak_phage_median": float(np.median(peak_phages)) if peak_phages else float("nan"),
        "peak_phage_range": [float(np.min(peak_phages)), float(np.max(peak_phages))]
                            if peak_phages else [],
        "clearance_time_median": float(np.median(clear_times)) if clear_times else None,
        "parameters": {"S0": int(S0), "P0": int(P0), "adsorption": adsorption, "burst": int(burst),
                       "lysis_rate": lysis_rate, "growth": growth, "decay": decay},
        "teaches": (
            "Every one of these runs used identical settings. They did not come out the same. "
            "That is not experimental error -- it is what small numbers do, and it is the reason "
            "your plate sometimes has no plaques."),
        "plain_language": (
            f"Ran the same experiment {n} times with {P0} phage and {S0} cells. The phage died out "
            f"completely in {extinctions} of them ({p_ext:.0%}, 95% CI {lo:.0%} to {hi:.0%}), and "
            f"cleared the culture in {sum(r['host_cleared'] for r in runs)}. "
            f"Identical settings, different outcomes."),
        "p_value": None,
    }


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + z ** 2 / n
    c = p + z ** 2 / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (float(max(0.0, (c - s) / d)), float(min(1.0, (c + s) / d)))


@bound("extinction_probability")
def extinction_probability(burst: int = 50, adsorption: float = 1e-3, S0: int = 200,
                           decay: float = 0.05, P0_values: Optional[List[int]] = None,
                           n_runs: int = 60, seed: int = 20260901) -> Dict[str, Any]:
    """How likely is it that a few phage go extinct before they get going?

    Also reports the branching-process prediction, so the student has a
    theoretical curve to compare their simulated points against -- which is the
    shape of a real scientific argument, not a demo.
    """
    P0s = list(P0_values or [1, 2, 3, 5, 8, 12, 20])
    rows = []
    for p0 in P0s:
        # growth=0: the branching-process prediction below assumes the host
        # pool is fixed. Letting the bacteria divide would make the simulation
        # and the theory answer subtly different questions.
        r = gillespie_lytic.__wrapped__(
            S0=S0, P0=int(p0), adsorption=adsorption, burst=burst, growth=0.0,
            decay=decay, n_runs=n_runs, t_max=15.0, seed=seed + p0)
        rows.append({"P0": int(p0), "extinction_probability": r["extinction_probability"],
                     "ci": r["extinction_ci"], "n_runs": r["n_runs"]})

    # Branching process: a single phage either adsorbs (and yields `burst`
    # offspring) or decays first. q = P(a lineage from one phage dies out).
    ads = adsorption * S0
    p_success = ads / (ads + decay)
    q_single = _extinction_root(p_success, int(burst))
    for row in rows:
        row["theory"] = float(q_single ** row["P0"])

    return {
        "rows": rows,
        "single_phage_extinction": float(q_single),
        "p_adsorb_before_decay": float(p_success),
        "theory_note": (
            "The theoretical curve is a branching process: one phage either finds a cell "
            "(probability p) and makes `burst` descendants, or falls apart first. The chance a "
            "whole lineage dies out is the smallest root of q = (1-p) + p*q^burst, and starting "
            "with n phage just raises that to the n-th power. The bacteria are held at a fixed "
            "number for this comparison; let them divide and the simulated points fall below the "
            "theory, because a growing host pool gives late phage more chances."),
        "plain_language": (
            f"A single phage in this culture has about a {q_single:.1%} chance of leaving no "
            f"descendants at all. Start with {P0s[-1]} phage instead and that drops to "
            f"{q_single ** P0s[-1]:.2%}. This is why 'we saw no plaques' does not mean "
            f"'there was no phage'."),
        "p_value": None,
    }


def _extinction_root(p: float, burst: int) -> float:
    """Smallest root in [0,1] of q = (1-p) + p q^burst, by bisection."""
    if p <= 0:
        return 1.0
    f = lambda q: (1 - p) + p * q ** burst - q
    if f(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 1.0 - 1e-12
    if f(hi) > 0:
        return 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


@bound("poisson_plaques")
def poisson_plaques(expected_plaques: float = 3.0, n_plates: int = 20,
                    seed: int = 20260901) -> Dict[str, Any]:
    """The Grade 8 answer to "why did my plate have none?" (G8-L10).

    Plaque counts are Poisson. At an expected 3 plaques per plate, one plate in
    twenty comes up empty by pure chance -- and a student who does not know that
    concludes their sample had no phage. This function simulates a class's worth
    of plates and puts the exact probability next to them.
    """
    lam = float(expected_plaques)
    if lam < 0:
        return {"refused": True, "reason": "Expected plaque count cannot be negative."}
    rng = np.random.default_rng(seed)
    counts = rng.poisson(lam, int(n_plates))
    p_zero = float(stats.poisson.pmf(0, lam))
    p_ge_double = float(stats.poisson.sf(2 * lam - 1, lam)) if lam > 0 else 0.0

    return {
        "expected": lam, "n_plates": int(n_plates),
        "simulated_counts": counts.tolist(),
        "observed_mean": float(counts.mean()), "observed_variance": float(counts.var(ddof=1)),
        "n_empty_plates": int((counts == 0).sum()),
        "probability_of_empty_plate": p_zero,
        "probability_of_double_or_more": p_ge_double,
        "pmf": {int(k): float(stats.poisson.pmf(k, lam)) for k in range(0, int(max(12, 3 * lam)))},
        "counting_interval_for_one_plate": [
            float(stats.chi2.ppf(0.025, 2 * max(1, int(lam))) / 2),
            float(stats.chi2.ppf(0.975, 2 * (int(lam) + 1)) / 2)],
        "teaches": (
            "For Poisson counts the variance equals the mean, so the uncertainty on a count of n "
            "is about the square root of n. Count 9 plaques and you know the number to about "
            "plus-or-minus 3 -- which is why you never report a titre from a single plate."),
        "plain_language": (
            f"If a plate should average {lam:g} plaques, then {p_zero:.1%} of plates come up "
            f"completely empty by chance alone. In this simulated set of {n_plates} plates, "
            f"{int((counts == 0).sum())} were empty. An empty plate is not proof of no phage -- "
            f"it is evidence, and weak evidence at that."),
        "p_value": None,
    }
