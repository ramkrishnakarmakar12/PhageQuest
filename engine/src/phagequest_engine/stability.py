"""The automatic stability check (spec section 06, point 5).

    "a model's own stated confidence is no help here: it is a fluency signal,
    not a stability measurement. The only thing that tells you whether a
    conclusion holds is perturbing the data and re-running it -- which is why
    the stability check below is automatic rather than optional."

So: it is not a flag the student can turn off, and not a separate menu item.
Every inferential result carries a StabilityReport, and if the conclusion flips
the frontend renders that *above* the p-value.

This is the Predictability-Computability-Stability sanity check, operationalised.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Sequence

import numpy as np

__all__ = ["StabilityReport", "stability_check"]


@dataclass
class StabilityReport:
    ran: bool
    n_perturbations: int
    conclusion: str                    # the conclusion on the full data
    agreement: float                   # fraction of perturbations agreeing
    flipped: bool
    p_range: List[float] = field(default_factory=list)
    effect_range: List[float] = field(default_factory=list)
    most_influential: Dict | None = None
    verdict: str = ""
    severity: str = "info"             # info | warn | block

    def to_dict(self) -> Dict:
        return asdict(self)


def _conclusion(p: float, alpha: float) -> str:
    if not np.isfinite(p):
        return "undetermined"
    return "difference" if p < alpha else "no difference"


def stability_check(
    groups: Sequence[Sequence[float]],
    test_fn: Callable[[List[np.ndarray]], tuple],
    alpha: float = 0.05,
    n_boot: int = 400,
    seed: int = 20260901,
) -> StabilityReport:
    """Three perturbations, run automatically on every inferential result.

    1. Leave-one-out -- does one observation carry the whole conclusion?
    2. Bootstrap resampling -- how often does the conclusion survive?
    3. The range of p and of the effect size across both.

    `test_fn` takes a list of arrays and returns (p_value, effect_size).
    """
    gs = [np.asarray(g, dtype=float).ravel() for g in groups]
    gs = [g[np.isfinite(g)] for g in gs]
    if len(gs) < 2 or min(len(g) for g in gs) < 3:
        return StabilityReport(
            ran=False, n_perturbations=0, conclusion="undetermined", agreement=float("nan"),
            flipped=False,
            verdict="Too few observations to perturb meaningfully. Treat this result as a "
                    "description of what happened, not as evidence about anything else.",
            severity="warn",
        )

    p0, e0 = test_fn(gs)
    base = _conclusion(p0, alpha)
    ps: List[float] = []
    es: List[float] = []
    agree = 0
    total = 0

    # 1. leave-one-out, and remember which single point moves the answer most
    worst_delta = -1.0
    worst: Dict | None = None
    for gi, g in enumerate(gs):
        for j in range(len(g)):
            trial = list(gs)
            trial[gi] = np.delete(g, j)
            if len(trial[gi]) < 2:
                continue
            p, e = test_fn(trial)
            ps.append(p)
            es.append(e)
            total += 1
            same = _conclusion(p, alpha) == base
            agree += int(same)
            d = abs(p - p0)
            if d > worst_delta:
                worst_delta = d
                worst = {
                    "group_index": gi, "observation_index": int(j),
                    "value": float(g[j]), "p_without_it": float(p),
                    "flips_conclusion": not same,
                }

    # 2. bootstrap
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        trial = [rng.choice(g, len(g), replace=True) for g in gs]
        if any(np.allclose(t, t[0]) for t in trial):
            continue
        try:
            p, e = test_fn(trial)
        except Exception:
            continue
        if not np.isfinite(p):
            continue
        ps.append(p)
        es.append(e)
        total += 1
        agree += int(_conclusion(p, alpha) == base)

    agreement = agree / total if total else float("nan")
    finite_ps = [p for p in ps if np.isfinite(p)]
    finite_es = [e for e in es if np.isfinite(e)]
    flipped = bool(np.isfinite(agreement) and agreement < 0.9)

    if not np.isfinite(agreement):
        verdict, sev = "Stability could not be assessed.", "warn"
    elif worst is not None and worst["flips_conclusion"]:
        verdict = (f"Removing a single observation (value {worst['value']:g}) changes the conclusion. "
                   f"This result rests on one data point. Collect more data before claiming anything.")
        sev = "block"
    elif agreement < 0.7:
        verdict = (f"The conclusion held in only {agreement:.0%} of perturbed re-runs. "
                   f"This is not a stable finding.")
        sev = "block"
    elif agreement < 0.9:
        verdict = (f"The conclusion held in {agreement:.0%} of perturbed re-runs -- shaky. "
                   f"Report the interval, not the p-value.")
        sev = "warn"
    else:
        verdict = (f"The conclusion held in {agreement:.0%} of perturbed re-runs, and no single "
                   f"observation changes it. This is a stable finding for this dataset.")
        sev = "info"

    return StabilityReport(
        ran=True,
        n_perturbations=total,
        conclusion=base,
        agreement=float(agreement),
        flipped=flipped,
        p_range=[float(min(finite_ps)), float(max(finite_ps))] if finite_ps else [],
        effect_range=[float(min(finite_es)), float(max(finite_es))] if finite_es else [],
        most_influential=worst,
        verdict=verdict,
        severity=sev,
    )
