"""Host-phage population dynamics (spec section 03.3).

The spec's finding, which shaped this module:

    "There is NO maintained, general-purpose phage-dynamics package anywhere.
    The field writes its own ODEs and solves them with generic solvers.
    ...that means this block is genuinely ours to define."

So this is our own model library over `scipy.integrate.solve_ivp`. Every model
is about thirty lines, runs in Pyodide, and costs nothing per student.

The one real modelling problem is the latent period. A one-step growth curve
needs an explicit delay between infection and lysis, which is formally a delay
differential equation that `solve_ivp` cannot do. The fix, per the spec, is an
Erlang chain: I1 -> I2 -> ... -> In -> lysis, which is ordinary ODEs. It is
also the better teaching object, because the student can *see* the latent
period as stages rather than as a hidden parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from ..transcript import bound

__all__ = ["MODELS", "simulate", "parameter_sweep", "one_step_growth", "PARAM_META"]


# --------------------------------------------------------------------------
# Parameter metadata. Drives the UI sliders, the units shown to a student, and
# the plausibility warnings -- so a Grade 8 cannot silently run a phage with a
# burst size of 100000 and believe the answer.
# --------------------------------------------------------------------------
PARAM_META: Dict[str, Dict[str, Any]] = {
    "mu":    {"label": "Bacterial growth rate", "unit": "per hour", "min": 0.0, "max": 3.0,
              "typical": [0.2, 1.5], "default": 0.7,
              "explain": "How fast the bacteria divide when nothing is stopping them. "
                         "0.7/hour is a doubling time of about an hour."},
    "K":     {"label": "Carrying capacity", "unit": "cells/mL", "min": 1e5, "max": 1e11,
              "typical": [1e8, 1e10], "default": 1e9, "log": True,
              "explain": "How many cells the flask can hold before it runs out of food."},
    "adsorption": {"label": "Adsorption rate", "unit": "mL/(phage x hour)", "min": 1e-12,
              "max": 1e-6, "typical": [1e-10, 1e-8], "default": 1e-9, "log": True,
              "explain": "How good the phage is at finding and sticking to a cell. This is the "
                         "single most important number in the whole model."},
    "burst": {"label": "Burst size", "unit": "phage per cell", "min": 1, "max": 500,
              "typical": [20, 200], "default": 100,
              "explain": "How many new phage come out when one infected cell bursts."},
    "latent": {"label": "Latent period", "unit": "hours", "min": 0.05, "max": 4.0,
              "typical": [0.3, 1.5], "default": 0.5,
              "explain": "The wait between a phage getting in and the cell bursting. During this "
                         "time nothing appears to happen -- which is exactly what makes it "
                         "interesting."},
    "decay": {"label": "Phage decay rate", "unit": "per hour", "min": 0.0, "max": 1.0,
              "typical": [0.0, 0.2], "default": 0.05,
              "explain": "Free phage falling apart on their own, with no bacteria involved."},
    "n_stages": {"label": "Latent-period stages", "unit": "compartments", "min": 1, "max": 40,
              "typical": [8, 25], "default": 15, "integer": True,
              "explain": "How finely the latent period is chopped up. With 1 stage the burst is "
                         "smeared out; with 20 it is sharp. This knob IS the Erlang trick."},
    "lysogeny": {"label": "Probability of lysogeny", "unit": "fraction", "min": 0.0, "max": 1.0,
              "typical": [0.0, 0.5], "default": 0.1,
              "explain": "The chance an infection hides in the genome instead of killing. "
                         "This is the 'does it kill, or does it hide?' question from G8-L10."},
    "induction": {"label": "Induction rate", "unit": "per hour", "min": 0.0, "max": 0.5,
              "typical": [1e-4, 0.01], "default": 0.001, "log": True,
              "explain": "How often a hidden prophage wakes up and kills its host anyway."},
    "resistance": {"label": "Resistance mutation rate", "unit": "per division", "min": 0.0,
              "max": 1e-4, "typical": [1e-9, 1e-6], "default": 1e-7, "log": True,
              "explain": "How often a dividing cell produces a daughter the phage cannot infect. "
                         "This is why the culture comes back."},
    "cost":  {"label": "Cost of resistance", "unit": "fraction of growth rate", "min": 0.0,
              "max": 0.9, "typical": [0.0, 0.3], "default": 0.1,
              "explain": "Resistant cells usually grow more slowly. If they did not, they would "
                         "already be everywhere."},
}


def _logistic(S: float, total: float, mu: float, K: float) -> float:
    return mu * S * (1.0 - total / K)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def _lytic_rhs(t, y, p):
    """Classic lytic model with an Erlang latent chain.

    State: [S, I1..In, P]
    """
    n = int(p["n_stages"])
    S = y[0]
    I = y[1:1 + n]
    P = y[1 + n]
    S, P = max(S, 0.0), max(P, 0.0)
    I = np.maximum(I, 0.0)

    total_cells = S + I.sum()
    infect = p["adsorption"] * S * P
    # Erlang: to get a mean latent period L from n stages, each stage runs at n/L.
    rate = n / p["latent"]

    dS = _logistic(S, total_cells, p["mu"], p["K"]) - infect
    dI = np.empty(n)
    dI[0] = infect - rate * I[0]
    for i in range(1, n):
        dI[i] = rate * I[i - 1] - rate * I[i]
    dP = p["burst"] * rate * I[-1] - infect - p["decay"] * P
    return np.concatenate([[dS], dI, [dP]])


def _lysogenic_rhs(t, y, p):
    """Lytic + lysogenic. State: [S, I1..In, L, P]

    A fraction `lysogeny` of infections become lysogens (L) that grow normally
    and carry the prophage; they induce back into the lytic cycle at
    `induction`. This is the model behind BACPHLIP's question.
    """
    n = int(p["n_stages"])
    S, I, L, P = y[0], y[1:1 + n], y[1 + n], y[2 + n]
    S, L, P = max(S, 0.0), max(L, 0.0), max(P, 0.0)
    I = np.maximum(I, 0.0)

    total = S + I.sum() + L
    infect = p["adsorption"] * S * P
    rate = n / p["latent"]
    f = p["lysogeny"]

    dS = _logistic(S, total, p["mu"], p["K"]) - infect
    dI = np.empty(n)
    dI[0] = (1 - f) * infect + p["induction"] * L - rate * I[0]
    for i in range(1, n):
        dI[i] = rate * I[i - 1] - rate * I[i]
    dL = f * infect + _logistic(L, total, p["mu"], p["K"]) - p["induction"] * L
    dP = p["burst"] * rate * I[-1] - infect - p["decay"] * P
    return np.concatenate([[dS], dI, [dL], [dP]])


def _resistance_rhs(t, y, p):
    """Lytic + an evolving resistant subpopulation. State: [S, I1..In, R, P]

    This is the model that produces the crash-and-rebound every student expects
    to see and no textbook figure ever shows them: the culture collapses, then
    comes back, and the phage cannot touch it the second time.
    """
    n = int(p["n_stages"])
    S, I, R, P = y[0], y[1:1 + n], y[1 + n], y[2 + n]
    S, R, P = max(S, 0.0), max(R, 0.0), max(P, 0.0)
    I = np.maximum(I, 0.0)

    total = S + I.sum() + R
    growth_S = _logistic(S, total, p["mu"], p["K"])
    growth_R = _logistic(R, total, p["mu"] * (1 - p["cost"]), p["K"])
    mutate = p["resistance"] * max(growth_S, 0.0)
    infect = p["adsorption"] * S * P
    rate = n / p["latent"]

    dS = growth_S - mutate - infect
    dI = np.empty(n)
    dI[0] = infect - rate * I[0]
    for i in range(1, n):
        dI[i] = rate * I[i - 1] - rate * I[i]
    dR = growth_R + mutate
    dP = p["burst"] * rate * I[-1] - infect - p["decay"] * P
    return np.concatenate([[dS], dI, [dR], [dP]])


def _chemostat_rhs(t, y, p):
    """Resource-explicit chemostat. State: [Resource, S, I1..In, P]

    Monod growth on an explicit nutrient with dilution. This is the model where
    "what is actually limiting?" has an answer, and it is the on-ramp to the
    real literature for a Grade 11-12 capstone.
    """
    n = int(p["n_stages"])
    Rn, S, I, P = y[0], y[1], y[2:2 + n], y[2 + n]
    Rn, S, P = max(Rn, 0.0), max(S, 0.0), max(P, 0.0)
    I = np.maximum(I, 0.0)

    monod = p["mu"] * Rn / (p["Ks"] + Rn)
    infect = p["adsorption"] * S * P
    rate = n / p["latent"]
    D, R0, yld = p["dilution"], p["inflow"], p["yield_coeff"]

    dR = D * (R0 - Rn) - monod * S / yld
    dS = monod * S - infect - D * S
    dI = np.empty(n)
    dI[0] = infect - rate * I[0] - D * I[0]
    for i in range(1, n):
        dI[i] = rate * I[i - 1] - rate * I[i] - D * I[i]
    dP = p["burst"] * rate * I[-1] - infect - p["decay"] * P - D * P
    return np.concatenate([[dR], [dS], dI, [dP]])


@dataclass
class ModelSpec:
    key: str
    label: str
    rhs: Any
    params: List[str]
    state_names: Any        # callable(p) -> list[str]
    initial: Any            # callable(p, y0) -> np.ndarray
    teaches: str
    grades: str


def _lytic_states(p):
    return ["Susceptible"] + [f"Infected (stage {i + 1})" for i in range(int(p["n_stages"]))] + ["Free phage"]


def _lytic_init(p, y0):
    n = int(p["n_stages"])
    return np.concatenate([[y0["S"]], np.zeros(n), [y0["P"]]])


MODELS: Dict[str, ModelSpec] = {
    "lytic": ModelSpec(
        "lytic", "Lytic infection (kill only)", _lytic_rhs,
        ["mu", "K", "adsorption", "burst", "latent", "decay", "n_stages"],
        _lytic_states, _lytic_init,
        "The phage wins, then starves. Watch what happens to the phage count AFTER the "
        "bacteria are gone -- nothing in the model creates more, so it can only fall.",
        "8-10",
    ),
    "lysogenic": ModelSpec(
        "lysogenic", "Lytic + lysogenic (kill or hide)", _lysogenic_rhs,
        ["mu", "K", "adsorption", "burst", "latent", "decay", "n_stages", "lysogeny", "induction"],
        lambda p: _lytic_states(p)[:-1] + ["Lysogens", "Free phage"],
        lambda p, y0: np.concatenate([[y0["S"]], np.zeros(int(p["n_stages"])), [0.0], [y0["P"]]]),
        "Turn the lysogeny probability up from 0 and the plaque goes cloudy instead of clear. "
        "That is the difference a student can see on a plate, explained.",
        "9-11",
    ),
    "resistance": ModelSpec(
        "resistance", "Lytic + evolving resistance", _resistance_rhs,
        ["mu", "K", "adsorption", "burst", "latent", "decay", "n_stages", "resistance", "cost"],
        lambda p: _lytic_states(p)[:-1] + ["Resistant", "Free phage"],
        lambda p, y0: np.concatenate([[y0["S"]], np.zeros(int(p["n_stages"])), [0.0], [y0["P"]]]),
        "The crash and the rebound. The culture comes back and the phage cannot touch it. "
        "This is the single most important thing to know before anyone says 'phage therapy'.",
        "9-12",
    ),
    "chemostat": ModelSpec(
        "chemostat", "Resource-explicit chemostat", _chemostat_rhs,
        ["mu", "adsorption", "burst", "latent", "decay", "n_stages",
         "Ks", "dilution", "inflow", "yield_coeff"],
        lambda p: ["Resource"] + _lytic_states(p),
        lambda p, y0: np.concatenate(
            [[y0.get("R", p["inflow"])], [y0["S"]], np.zeros(int(p["n_stages"])), [y0["P"]]]),
        "Now the food is in the model too. Coexistence and oscillation appear, and they are not "
        "a bug -- predator and prey cycling is what these equations do.",
        "11-12",
    ),
}

_EXTRA_DEFAULTS = {"Ks": 1e6, "dilution": 0.2, "inflow": 1e9, "yield_coeff": 1e6}


def _defaults(model: str) -> Dict[str, float]:
    spec = MODELS[model]
    out = {}
    for k in spec.params:
        if k in PARAM_META:
            out[k] = PARAM_META[k]["default"]
        else:
            out[k] = _EXTRA_DEFAULTS[k]
    return out


def _plausibility(p: Dict[str, float]) -> List[str]:
    warn = []
    for k, v in p.items():
        meta = PARAM_META.get(k)
        if not meta:
            continue
        lo, hi = meta["typical"]
        if v < lo or v > hi:
            warn.append(
                f"{meta['label']} = {v:g} {meta['unit']} is outside the range usually measured "
                f"for real phage ({lo:g} to {hi:g}). The simulation will still run -- that is the "
                f"point of a simulation -- but do not report the result as what a real phage does.")
    return warn


@bound("simulate")
def simulate(model: str = "lytic", params: Optional[Dict[str, float]] = None,
             S0: float = 1e6, P0: float = 1e4, hours: float = 24.0,
             n_points: int = 400, R0: Optional[float] = None) -> Dict[str, Any]:
    """Run one host-phage simulation.

    Uses LSODA, which switches between a non-stiff and a stiff method on its
    own. These systems go stiff the moment the susceptible population crashes,
    and a fixed non-stiff solver silently produces garbage there.
    """
    spec = MODELS.get(model)
    if spec is None:
        return {"refused": True, "reason": f"Unknown model {model!r}.",
                "available_models": list(MODELS)}

    p = _defaults(model)
    p.update({k: float(v) for k, v in (params or {}).items() if k in p})
    p["n_stages"] = max(1, int(round(p["n_stages"])))
    if p.get("latent", 1.0) <= 0:
        return {"refused": True, "reason": "The latent period must be greater than zero."}

    y0 = {"S": float(S0), "P": float(P0)}
    if R0 is not None:
        y0["R"] = float(R0)
    init = spec.initial(p, y0)
    t_eval = np.linspace(0.0, float(hours), int(n_points))

    sol = solve_ivp(spec.rhs, (0.0, float(hours)), init, args=(p,),
                    method="LSODA", t_eval=t_eval, rtol=1e-8, atol=1e-3, max_step=0.1)
    if not sol.success:
        return {"refused": True, "reason": f"The solver failed: {sol.message}. "
                                           f"This usually means a parameter is extreme."}

    names = spec.state_names(p)
    Y = np.maximum(sol.y, 0.0)
    n = p["n_stages"]

    # Collapse the Erlang chain for display -- a student wants one "infected"
    # curve, not fifteen. The stages stay available for the Grade 11 view.
    if model == "chemostat":
        series = {"Resource": Y[0], "Susceptible": Y[1],
                  "Infected": Y[2:2 + n].sum(axis=0), "Free phage": Y[2 + n]}
    elif model == "lytic":
        series = {"Susceptible": Y[0], "Infected": Y[1:1 + n].sum(axis=0), "Free phage": Y[1 + n]}
    elif model == "lysogenic":
        series = {"Susceptible": Y[0], "Infected": Y[1:1 + n].sum(axis=0),
                  "Lysogens": Y[1 + n], "Free phage": Y[2 + n]}
    else:
        series = {"Susceptible": Y[0], "Infected": Y[1:1 + n].sum(axis=0),
                  "Resistant": Y[1 + n], "Free phage": Y[2 + n]}

    t = sol.t
    host_total = sum(v for k, v in series.items() if k not in ("Free phage", "Resource"))
    phage = series["Free phage"]
    host0 = host_total[0]
    nadir_i = int(np.argmin(host_total))
    peak_phage_i = int(np.argmax(phage))

    collapsed = bool(host_total.min() < 0.01 * max(host0, 1e-12))
    rebounded = bool(nadir_i < len(t) - 2 and host_total[-1] > 10 * max(host_total[nadir_i], 1e-12))
    amplification = float(phage.max() / max(phage[0], 1e-12))

    summary = {
        "initial_moi": float(P0 / max(S0, 1e-12)),
        "host_collapsed": collapsed,
        "host_rebounded": rebounded,
        "nadir_time": float(t[nadir_i]), "nadir_host": float(host_total[nadir_i]),
        "final_host": float(host_total[-1]), "final_phage": float(phage[-1]),
        "peak_phage": float(phage.max()), "peak_phage_time": float(t[peak_phage_i]),
        "phage_amplification": amplification,
        "log10_amplification": float(np.log10(max(amplification, 1e-12))),
    }
    if model == "resistance":
        r = series["Resistant"]
        summary["final_resistant_fraction"] = float(r[-1] / max(host_total[-1], 1e-12))

    return {
        "model": model, "model_label": spec.label, "teaches": spec.teaches,
        "grades": spec.grades,
        "parameters": p, "initial": {"S0": float(S0), "P0": float(P0)},
        "time": t.tolist(),
        "series": {k: v.tolist() for k, v in series.items()},
        "stage_series": {f"Infected stage {i + 1}": Y[(1 if model != 'chemostat' else 2) + i].tolist()
                         for i in range(n)} if n <= 8 else {},
        "summary": summary,
        "solver": {"method": "LSODA", "rtol": 1e-8, "atol": 1e-3, "n_eval": int(t.size),
                   "message": sol.message},
        "notes": _plausibility(p),
        "plain_language": _narrate(model, summary, p),
        "p_value": None,
    }


def _narrate(model: str, s: Dict[str, Any], p: Dict[str, float]) -> str:
    bits = [f"Starting at MOI {s['initial_moi']:.3g} "
            f"({'more phage than bacteria' if s['initial_moi'] > 1 else 'fewer phage than bacteria'}):"]
    if s["host_collapsed"]:
        bits.append(f"the bacteria crashed by {s['nadir_time']:.2g} hours")
    else:
        bits.append(f"the bacteria were knocked back but never wiped out "
                    f"(lowest point {s['nadir_host']:.3g} cells/mL at {s['nadir_time']:.2g} h)")
    bits.append(f"and the phage multiplied {s['log10_amplification']:.1f} orders of magnitude, "
                f"peaking at {s['peak_phage']:.3g}/mL around {s['peak_phage_time']:.2g} h")
    if s.get("host_rebounded"):
        extra = ""
        if "final_resistant_fraction" in s:
            extra = (f", and {s['final_resistant_fraction'] * 100:.0f}% of the survivors are "
                     f"resistant -- the phage cannot clear them")
        bits.append(f"then the culture recovered{extra}")
    elif model == "resistance" and s.get("final_resistant_fraction", 0) > 0.5:
        bits.append("and the survivors are almost entirely resistant")
    return ". ".join([" ".join(bits[:2])] + bits[2:]) + "."


@bound("parameter_sweep")
def parameter_sweep(model: str = "lytic", parameter: str = "adsorption",
                    values: Optional[Sequence[float]] = None, n_values: int = 9,
                    log_scale: bool = True, base_params: Optional[Dict[str, float]] = None,
                    S0: float = 1e6, P0: float = 1e4, hours: float = 24.0) -> Dict[str, Any]:
    """Sweep one parameter and tabulate the outcome (spec section 03.3 / G8-L12).

    G8-L12 already establishes the entire pedagogy: set four parameters, sweep
    each one, compute a rate, plot it, write a conclusion. This is that lesson
    with a phage in the flask instead of an enzyme, and the output table is
    shaped so it drops straight into `fit_curve` -- the student sweeps, then
    fits, then gets an interval, without leaving the platform.
    """
    spec = MODELS.get(model)
    if spec is None:
        return {"refused": True, "reason": f"Unknown model {model!r}."}
    if parameter not in spec.params:
        return {"refused": True, "reason": f"{parameter!r} is not a parameter of the {model} model.",
                "available_parameters": spec.params}

    base = _defaults(model)
    base.update({k: float(v) for k, v in (base_params or {}).items() if k in base})

    if values is None:
        meta = PARAM_META.get(parameter)
        if meta is None:
            lo, hi = base[parameter] / 10, base[parameter] * 10
            use_log = True
        else:
            lo, hi = meta["typical"]
            use_log = bool(meta.get("log")) and log_scale
            lo = max(lo, meta["min"]) or meta["min"]
        vals = (np.logspace(np.log10(max(lo, 1e-300)), np.log10(max(hi, 1e-299)), n_values)
                if use_log else np.linspace(lo, hi, n_values))
        if PARAM_META.get(parameter, {}).get("integer"):
            vals = np.unique(np.round(vals)).astype(float)
    else:
        vals = np.asarray(values, dtype=float)

    rows = []
    for v in vals:
        p = dict(base)
        p[parameter] = float(v)
        r = simulate.__wrapped__(model=model, params=p, S0=S0, P0=P0, hours=hours, n_points=300)
        if r.get("refused"):
            continue
        s = r["summary"]
        rows.append({
            parameter: float(v),
            "log10_amplification": s["log10_amplification"],
            "peak_phage": s["peak_phage"],
            "time_to_nadir": s["nadir_time"],
            "final_host": s["final_host"],
            "host_collapsed": s["host_collapsed"],
            "clearance": float(1.0 - s["final_host"] / max(S0, 1e-12)),
        })

    if not rows:
        return {"refused": True, "reason": "Every simulation in the sweep failed."}

    x = np.array([r[parameter] for r in rows])
    y = np.array([r["log10_amplification"] for r in rows])
    collapsed = [r[parameter] for r in rows if r["host_collapsed"]]
    threshold = float(min(collapsed)) if collapsed else None

    return {
        "model": model, "parameter": parameter,
        "parameter_label": PARAM_META.get(parameter, {}).get("label", parameter),
        "parameter_unit": PARAM_META.get(parameter, {}).get("unit", ""),
        "values": x.tolist(), "rows": rows,
        "base_parameters": base,
        "response": {"x": x.tolist(), "y": y.tolist(), "y_label": "log10 phage amplification"},
        "collapse_threshold": threshold,
        "monotonic": bool(np.all(np.diff(y) >= -1e-9) or np.all(np.diff(y) <= 1e-9)),
        "plain_language": (
            f"Swept {PARAM_META.get(parameter, {}).get('label', parameter)} across "
            f"{len(rows)} values. Phage amplification ranged from {y.min():.1f} to {y.max():.1f} "
            f"orders of magnitude"
            + (f", and the bacteria were wiped out once {parameter} reached {threshold:.3g}."
               if threshold is not None else
               ", and the bacteria survived at every value tried.")
            + " Feed the response column into a curve fit to get the shape and its interval."),
        "next_step": {"tool": "fit_curve", "x": x.tolist(), "y": y.tolist(),
                      "suggested_model": "four_pl"},
        "p_value": None,
    }


@bound("one_step_growth")
def one_step_growth(latent: float = 0.5, burst: float = 100, n_stages: int = 20,
                    adsorption: float = 1e-8, S0: float = 1e8, P0: float = 1e4,
                    hours: float = 3.0) -> Dict[str, Any]:
    """The one-step growth curve, and why the Erlang chain exists.

    Runs the same experiment at n_stages = 1 and at the chosen n_stages, so the
    student sees directly that with one compartment the burst is smeared into an
    exponential and with twenty it is a step. The latent period becomes visible
    as a thing made of stages, which is the whole pedagogical point of using an
    ODE chain instead of hiding a delay in a solver.
    """
    out = {}
    for n in sorted({1, 3, int(n_stages)}):
        r = simulate.__wrapped__(
            model="lytic",
            params={"adsorption": adsorption, "burst": burst, "latent": latent,
                    "n_stages": n, "decay": 0.0, "mu": 0.0, "K": 1e12},
            S0=S0, P0=P0, hours=hours, n_points=400)
        if r.get("refused"):
            continue
        t = np.array(r["time"])
        P = np.array(r["series"]["Free phage"])
        # In a one-step experiment free phage first FALL, as they adsorb onto
        # cells, and only rise when the first cells burst. So the latent period
        # is the time from the minimum to the first clear rise off it -- not a
        # fraction of the (much later, much larger) plateau.
        dip = int(np.argmin(P))
        after = P[dip:]
        rose = np.nonzero(after > 1.5 * max(P[dip], 1e-12))[0]
        idx = dip + int(rose[0]) if rose.size else int(np.argmax(P))
        out[f"{n}_stages"] = {
            "n_stages": n, "time": t.tolist(), "free_phage": P.tolist(),
            "observed_rise_start": float(t[idx]),
            "plateau": float(P.max()),
            "adsorption_minimum": float(P.min()),
            "burst_size_estimate": float(P.max() / max(P.min(), 1e-12)),
        }

    ref = out.get(f"{int(n_stages)}_stages")
    return {
        "runs": out, "requested_stages": int(n_stages),
        "true_latent_period": float(latent), "true_burst_size": float(burst),
        "recovered_latent_period": ref["observed_rise_start"] if ref else None,
        "teaches": (
            "With 1 stage the phage start appearing immediately and the 'latent period' is "
            "invisible -- the model is wrong in a way you can see. Add stages and a real plateau "
            "appears before the rise. That plateau is the latent period, and it is made of "
            "compartments, not of a hidden delay."),
        "plain_language": (
            f"With {n_stages} stages the free-phage count stays flat until about "
            f"{ref['observed_rise_start']:.3g} h (true latent period {latent:g} h), then rises to "
            f"about {ref['burst_size_estimate']:.3g} times the starting count "
            f"(true burst size {burst:g}). Compare the 1-stage run to see why the chain matters."
            if ref else "The simulation did not complete."),
        "p_value": None,
    }
