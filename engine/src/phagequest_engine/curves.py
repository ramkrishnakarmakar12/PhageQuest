"""Curve fitting with honest uncertainty (spec section 04, row 5).

    "G8-L12 rate vs substrate, enzyme, pH, temperature -- four dose-response
    curves the students already tabulate by hand. CRITICAL: `pip install drc`
    installs a Django comment module, NOT the R dose-response package. There is
    no Python `drc`. Use `lmfit` with an explicit four-parameter logistic."

lmfit is the Tier-1 choice and is not in Pyodide, so Tier-0 uses
`scipy.optimize.curve_fit` with named parameters, bounds and bootstrap
intervals -- which is what we wanted lmfit for. Every model here is explicit;
nothing is auto-selected behind the student's back.

A fitted rate without an interval teaches the wrong lesson, so `conf_interval`
is not optional here either.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

from .transcript import bound

__all__ = ["MODELS", "fit_curve", "fit_growth_curve", "compare_models"]


def _four_pl(x, bottom, top, ec50, hill):
    """Four-parameter logistic. The dose-response workhorse."""
    return bottom + (top - bottom) / (1.0 + (x / np.maximum(ec50, 1e-12)) ** (-hill))


def _michaelis_menten(x, vmax, km):
    """Enzyme kinetics: rate vs substrate. G8-L12 sweep 1."""
    return vmax * x / (km + x)


def _logistic_growth(x, K, r, t0):
    """Population growth to a carrying capacity."""
    return K / (1.0 + np.exp(-r * (x - t0)))


def _gompertz(x, A, mu, lag):
    """Gompertz with an explicit lag phase -- the standard microbial growth form."""
    return A * np.exp(-np.exp((mu * np.e / A) * (lag - x) + 1.0))


def _exponential(x, a, k):
    return a * np.exp(k * x)


def _linear(x, slope, intercept):
    return slope * x + intercept


def _gaussian_optimum(x, peak, optimum, width):
    """Rate against a variable with an optimum -- pH and temperature sweeps.

    G8-L9 (four pH levels) and G6-L9 (three temperatures) both produce this
    shape, and fitting it is how a student gets a number for "the best pH" with
    an interval, rather than pointing at the tallest bar.
    """
    return peak * np.exp(-0.5 * ((x - optimum) / np.maximum(width, 1e-12)) ** 2)


MODELS: Dict[str, Dict[str, Any]] = {
    "four_pl": {
        "fn": _four_pl, "params": ["bottom", "top", "ec50", "hill"],
        "label": "Four-parameter logistic (dose-response)",
        "guess": lambda x, y: [float(np.min(y)), float(np.max(y)),
                               float(np.median(x[x > 0]) if np.any(x > 0) else 1.0), 1.0],
        "bounds": lambda x, y: ([-np.inf, -np.inf, 1e-9, -20], [np.inf, np.inf, np.inf, 20]),
        "teaches": "EC50 is the dose that gets you halfway. It is the number the curve exists to give you.",
    },
    "michaelis_menten": {
        "fn": _michaelis_menten, "params": ["vmax", "km"],
        "label": "Michaelis-Menten (enzyme kinetics)",
        "guess": lambda x, y: [float(np.max(y) * 1.1), float(np.median(x[x > 0]) if np.any(x > 0) else 1.0)],
        "bounds": lambda x, y: ([0, 1e-12], [np.inf, np.inf]),
        "teaches": "Vmax is the ceiling; Km is the substrate level that reaches half of it. "
                   "If your highest substrate is below Km, Vmax is an extrapolation, not a measurement.",
    },
    "logistic_growth": {
        "fn": _logistic_growth, "params": ["K", "r", "t0"],
        "label": "Logistic growth (S-curve to a carrying capacity)",
        "guess": lambda x, y: [float(np.max(y) * 1.05), 1.0, float(np.median(x))],
        "bounds": lambda x, y: ([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
        "teaches": "K is how many the flask can hold; r is how fast they get there.",
    },
    "gompertz": {
        "fn": _gompertz, "params": ["A", "mu", "lag"],
        "label": "Gompertz growth (with an explicit lag phase)",
        "guess": lambda x, y: [float(np.max(y)), 1.0, float(np.min(x))],
        "bounds": lambda x, y: ([1e-12, 1e-12, -np.inf], [np.inf, np.inf, np.inf]),
        "teaches": "The lag parameter is how long the culture sat there doing nothing before growing. "
                   "That waiting time is a real, measurable biological quantity.",
    },
    "exponential": {
        "fn": _exponential, "params": ["a", "k"],
        "label": "Exponential",
        "guess": lambda x, y: [float(max(1e-9, np.min(np.abs(y)))), 0.1],
        "bounds": lambda x, y: ([-np.inf, -np.inf], [np.inf, np.inf]),
        "teaches": "k is the growth rate; doubling time is ln(2)/k.",
    },
    "linear": {
        "fn": _linear, "params": ["slope", "intercept"],
        "label": "Straight line",
        "guess": lambda x, y: [1.0, 0.0],
        "bounds": lambda x, y: ([-np.inf, -np.inf], [np.inf, np.inf]),
        "teaches": "The simplest model. Always fit it too: if a curve is not clearly better than a "
                   "line, you have not earned the curve.",
    },
    "optimum": {
        "fn": _gaussian_optimum, "params": ["peak", "optimum", "width"],
        "label": "Peak with an optimum (pH / temperature response)",
        "guess": lambda x, y: [float(np.max(y)), float(x[np.argmax(y)]),
                               float(max(1e-6, np.ptp(x) / 4))],
        "bounds": lambda x, y: ([0, -np.inf, 1e-9], [np.inf, np.inf, np.inf]),
        "teaches": "The 'optimum' parameter is the best pH or temperature WITH an interval -- which "
                   "is a real answer, unlike pointing at whichever bar happened to be tallest.",
    },
}


def _fit_once(fn, x, y, p0, bounds, maxfev=20000):
    popt, pcov = optimize.curve_fit(fn, x, y, p0=p0, bounds=bounds, maxfev=maxfev)
    return popt, pcov


@bound("fit_curve", data_args=("x", "y"))
def fit_curve(x: Sequence[float], y: Sequence[float], model: str = "four_pl",
              n_boot: int = 800, seed: int = 20260901,
              p0: Optional[List[float]] = None) -> Dict[str, Any]:
    """Fit a named model and report every parameter with a bootstrap interval.

    Refuses to fit a model with more parameters than the data can support --
    four points cannot determine a four-parameter logistic, and a platform that
    lets a student do it anyway has taught them something false.
    """
    xa = np.asarray(x, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[m], ya[m]

    spec = MODELS.get(model)
    if spec is None:
        return {"refused": True, "reason": f"Unknown model {model!r}. Available: {sorted(MODELS)}.",
                "available_models": sorted(MODELS)}

    k = len(spec["params"])
    if xa.size < k + 1:
        return {
            "refused": True,
            "reason": (f"The {spec['label']} has {k} parameters and you have {xa.size} data points. "
                       f"A model needs more points than parameters, and comfortably more than that "
                       f"to mean anything. With {xa.size} points, try the 'linear' model or collect "
                       f"more concentrations."),
            "n": int(xa.size), "n_parameters": k,
        }

    guess = p0 if p0 is not None else spec["guess"](xa, ya)
    lo, hi = spec["bounds"](xa, ya)
    guess = [float(np.clip(g, l + 1e-12 if np.isfinite(l) else g, h - 1e-12 if np.isfinite(h) else g))
             for g, l, h in zip(guess, lo, hi)]
    try:
        popt, pcov = _fit_once(spec["fn"], xa, ya, guess, (lo, hi))
    except Exception as exc:
        return {"refused": True, "reason": f"The fit did not converge ({type(exc).__name__}). "
                                           f"This usually means the model is the wrong shape for "
                                           f"this data. Plot it first."}

    pred = spec["fn"](xa, *popt)
    resid = ya - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n = xa.size
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")
    aicc = (n * np.log(ss_res / n) + 2 * k + (2 * k * (k + 1)) / max(1, n - k - 1)
            if ss_res > 0 else -np.inf)

    # Bootstrap the parameters -- the covariance matrix from curve_fit assumes
    # normal residuals and large n, neither of which a class dataset has.
    rng = np.random.default_rng(seed)
    boots = np.full((n_boot, k), np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(xa[idx]).size < k + 1:
            continue
        try:
            b, _ = _fit_once(spec["fn"], xa[idx], ya[idx], list(popt), (lo, hi), maxfev=6000)
            boots[i] = b
        except Exception:
            continue
    ok = ~np.isnan(boots).any(axis=1)
    boots = boots[ok]

    params = {}
    for j, name in enumerate(spec["params"]):
        if boots.shape[0] >= 50:
            ci = [float(np.percentile(boots[:, j], 2.5)), float(np.percentile(boots[:, j], 97.5))]
            se = float(boots[:, j].std(ddof=1))
        else:
            se = float(np.sqrt(np.diag(pcov))[j]) if np.all(np.isfinite(pcov)) else float("nan")
            tcrit = stats.t.ppf(0.975, max(1, n - k))
            ci = [float(popt[j] - tcrit * se), float(popt[j] + tcrit * se)]
        params[name] = {"value": float(popt[j]), "se": se, "ci": ci,
                        "relative_width": float(abs(ci[1] - ci[0]) / max(1e-12, abs(popt[j])))}

    curve_x = np.linspace(float(xa.min()), float(xa.max()), 120)
    curve_y = spec["fn"](curve_x, *popt)
    if boots.shape[0] >= 50:
        band = np.array([spec["fn"](curve_x, *b) for b in boots])
        band_lo = np.percentile(band, 2.5, axis=0).tolist()
        band_hi = np.percentile(band, 97.5, axis=0).tolist()
    else:
        band_lo = band_hi = []

    wobbly = [p for p, v in params.items() if v["relative_width"] > 1.0]
    notes = []
    if wobbly:
        notes.append(
            f"The interval for {', '.join(wobbly)} is wider than the estimate itself. That parameter "
            f"is not actually determined by your data -- do not quote it as a result.")
    if r2 > 0.99 and n <= k + 2:
        notes.append(
            f"R-squared of {r2:.3f} with only {n} points and {k} parameters is not impressive -- a "
            f"model with almost as many knobs as points will always fit. Look at the intervals.")

    return {
        "model": model, "model_label": spec["label"], "teaches": spec["teaches"],
        "parameters": params,
        "parameter_names": spec["params"],
        "point_estimates": [float(v) for v in popt],
        "r_squared": float(r2), "adjusted_r_squared": float(adj_r2), "aicc": float(aicc),
        "rmse": float(np.sqrt(ss_res / n)),
        "n": int(n), "n_parameters": k,
        "n_bootstrap_fits": int(boots.shape[0]),
        "curve": {"x": curve_x.tolist(), "y": curve_y.tolist(),
                  "band_low": band_lo, "band_high": band_hi},
        "residuals": resid.tolist(),
        "notes": notes,
        "plain_language": (
            f"Fitted {spec['label']}. It explains {r2 * 100:.1f}% of the variation in your "
            f"{n} points. "
            + "; ".join(f"{name} = {v['value']:.4g} (95% CI {v['ci'][0]:.4g} to {v['ci'][1]:.4g})"
                        for name, v in params.items())
            + ". " + spec["teaches"]),
    }


@bound("compare_models", data_args=("x", "y"))
def compare_models(x: Sequence[float], y: Sequence[float],
                   models: Optional[List[str]] = None, seed: int = 20260901) -> Dict[str, Any]:
    """Fit several models to the same data and rank them by AICc.

    Always includes 'linear', because the honest question is never "does my
    curve fit?" but "does my curve fit BETTER THAN A STRAIGHT LINE, by enough
    to justify the extra parameters?" AICc, not R-squared, because R-squared
    always rewards more parameters.
    """
    candidates = list(models or ["linear", "exponential", "michaelis_menten", "four_pl"])
    if "linear" not in candidates:
        candidates.insert(0, "linear")

    fits = []
    for mname in candidates:
        try:
            r = fit_curve.__wrapped__(x, y, model=mname, n_boot=200, seed=seed)
        except Exception:
            continue
        if r.get("refused"):
            fits.append({"model": mname, "refused": True, "reason": r.get("reason")})
            continue
        fits.append({"model": mname, "model_label": r["model_label"], "aicc": r["aicc"],
                     "r_squared": r["r_squared"], "adjusted_r_squared": r["adjusted_r_squared"],
                     "n_parameters": r["n_parameters"],
                     "parameters": {k: v["value"] for k, v in r["parameters"].items()},
                     "refused": False})

    usable = [f for f in fits if not f["refused"] and np.isfinite(f["aicc"])]
    if not usable:
        return {"refused": True, "reason": "No model could be fitted to this data.", "fits": fits}

    best = min(usable, key=lambda f: f["aicc"])
    for f in usable:
        f["delta_aicc"] = float(f["aicc"] - best["aicc"])
        # Akaike weights: the probability each model is the best of this set.
        f["weight"] = float(np.exp(-0.5 * f["delta_aicc"]))
    tot = sum(f["weight"] for f in usable)
    for f in usable:
        f["weight"] = float(f["weight"] / tot)

    lin = next((f for f in usable if f["model"] == "linear"), None)
    verdict = ""
    if lin is not None and best["model"] != "linear":
        d = lin["delta_aicc"]
        verdict = (f"The curve beats a straight line by {d:.1f} AICc units"
                   + (" -- strong enough to be worth the extra parameters."
                      if d > 4 else
                      " -- which is not a decisive margin. With this much data, a straight line is "
                      "an honest description too."))
    elif lin is not None:
        verdict = "A straight line describes this data as well as any curve tried. Use the line."

    return {
        "fits": sorted(usable, key=lambda f: f["aicc"]) + [f for f in fits if f["refused"]],
        "best_model": best["model"], "best_model_label": best["model_label"],
        "plain_language": (
            f"Of {len(usable)} models tried, {best['model_label']} fits best "
            f"(AICc weight {best['weight']:.0%}). {verdict} "
            f"AICc is used rather than R-squared because R-squared always improves when you add "
            f"parameters, whether or not they mean anything."),
    }


@bound("fit_growth_curve", data_args=("time", "density"))
def fit_growth_curve(time: Sequence[float], density: Sequence[float],
                     model: str = "gompertz", seed: int = 20260901) -> Dict[str, Any]:
    """Growth-curve analysis with the phage-relevant parameters named.

    `gcplyr` (spec section 03.3) is R and GPL; this is the Tier-0 equivalent for
    the parameters a school actually reports: lag, maximum growth rate, and
    carrying capacity -- plus, for lysis-shaped curves, the time and depth of
    the crash, which is the whole point when a phage is in the flask.
    """
    t = np.asarray(time, dtype=float).ravel()
    d = np.asarray(density, dtype=float).ravel()
    m = np.isfinite(t) & np.isfinite(d)
    t, d = t[m], d[m]
    if t.size < 5:
        return {"refused": True, "reason": f"Only {t.size} readings; a growth curve needs at least 5."}

    order = np.argsort(t)
    t, d = t[order], d[order]

    peak_i = int(np.argmax(d))
    crashed = bool(peak_i < t.size - 2 and d[-1] < 0.6 * d[peak_i])

    if crashed:
        # Lysis: fit growth on the rising limb only, and report the crash separately.
        fit_t, fit_d = t[:peak_i + 1], d[:peak_i + 1]
        note = ("This culture rose and then crashed -- the signature of lysis. The growth model is "
                "fitted only to the rising part; fitting it through the crash would produce a "
                "meaningless carrying capacity. The crash itself is reported separately below, and "
                "it is the interesting half.")
    else:
        fit_t, fit_d = t, d
        note = ""

    res = fit_curve.__wrapped__(fit_t, fit_d, model=model, n_boot=500, seed=seed) \
        if fit_t.size >= 5 else {"refused": True, "reason": "Too few points on the rising limb."}

    # Model-free descriptors, always computed, because they need no fit to be true.
    with np.errstate(divide="ignore", invalid="ignore"):
        logd = np.log(np.maximum(d, 1e-12))
    dt = np.diff(t)
    slopes = np.divide(np.diff(logd), dt, out=np.full(dt.shape, np.nan), where=dt > 0)
    mu_max = float(np.nanmax(slopes)) if np.any(np.isfinite(slopes)) else float("nan")
    doubling = float(np.log(2) / mu_max) if np.isfinite(mu_max) and mu_max > 0 else float("nan")

    out = {
        "model": model, "lysis_detected": crashed,
        "empirical": {
            "max_density": float(d.max()), "time_of_max": float(t[peak_i]),
            "final_density": float(d[-1]),
            "max_specific_growth_rate": mu_max,
            "doubling_time": doubling,
            "auc": float(np.trapezoid(d, t)) if hasattr(np, "trapezoid") else float(np.trapz(d, t)),
        },
        "fit": res,
        "notes": [n for n in [note] if n],
        "p_value": None,
    }
    if crashed:
        drop = float((d[peak_i] - d[-1]) / max(1e-12, d[peak_i]))
        out["lysis"] = {
            "peak_time": float(t[peak_i]), "peak_density": float(d[peak_i]),
            "final_density": float(d[-1]), "fractional_drop": drop,
            "time_to_half_peak": float(
                next((t[i] for i in range(peak_i, t.size) if d[i] <= 0.5 * d[peak_i]), np.nan)),
        }
    out["plain_language"] = (
        f"Maximum growth rate {mu_max:.4g} per unit time"
        + (f" (doubling time {doubling:.3g})" if np.isfinite(doubling) else "")
        + f", peaking at {d.max():.4g} around t = {t[peak_i]:g}."
        + (f" The culture then fell by {out['lysis']['fractional_drop'] * 100:.0f}% -- that is lysis, "
           f"and its timing is what tells you about the phage." if crashed else ""))
    return out
