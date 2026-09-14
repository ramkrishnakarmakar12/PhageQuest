"""Agent-based plaque formation on a lattice (spec section 03.3).

    "A visual, spatial model of a plaque spreading through a lawn. This is the
    screen that makes G8-L10's clear circles click."

Mesa is the Tier-1 choice; it is not in Pyodide and, for a lattice of a few
hundred cells a side, it is also considerably slower than doing it with numpy.
So this is a vectorised cellular automaton: every step is array arithmetic, a
250x250 lawn runs 200 steps in well under a second in the browser, and the
frames stream straight to a canvas.

The model is deliberately simple enough that a student can state its rules in
one breath -- which is the difference between a simulation they learn from and
an animation they watch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..transcript import bound

__all__ = ["plaque_growth", "STATES"]

# Lattice states
UNINFECTED, INFECTED, LYSED, RESISTANT = 0, 1, 2, 3
STATES = {UNINFECTED: "lawn", INFECTED: "infected", LYSED: "plaque (lysed)", RESISTANT: "resistant"}


@bound("plaque_growth")
def plaque_growth(size: int = 151, steps: int = 120, latent_steps: int = 4,
                  spread_probability: float = 0.45, diffusion_radius: int = 1,
                  resistant_fraction: float = 0.0, n_seeds: int = 1,
                  capture_every: int = 4, seed: int = 20260901) -> Dict[str, Any]:
    """Grow plaques on a bacterial lawn and measure them.

    Rules, in full:
      1. Start with a lawn of susceptible cells, a few of them resistant.
      2. Drop `n_seeds` infected cells.
      3. An infected cell waits `latent_steps` steps, then lyses.
      4. When it lyses, each susceptible neighbour within `diffusion_radius`
         becomes infected with probability `spread_probability`.
      5. Resistant cells never become infected.

    What it measures: plaque radius over time, final area, and -- the point --
    the *edge roughness*, which is what actually distinguishes phage on a real
    plate and which no textbook diagram ever shows.
    """
    size = int(np.clip(size, 21, 401))
    steps = int(np.clip(steps, 5, 600))
    rng = np.random.default_rng(seed)

    grid = np.full((size, size), UNINFECTED, dtype=np.int8)
    if resistant_fraction > 0:
        grid[rng.random((size, size)) < float(resistant_fraction)] = RESISTANT

    timer = np.zeros((size, size), dtype=np.int16)
    centre = size // 2
    seeds = [(centre, centre)]
    for _ in range(max(0, int(n_seeds) - 1)):
        seeds.append((int(rng.integers(3, size - 3)), int(rng.integers(3, size - 3))))
    for r, c in seeds:
        grid[r, c] = INFECTED
        timer[r, c] = int(latent_steps)

    rad = int(np.clip(diffusion_radius, 1, 4))
    # Moore neighbourhood (Chebyshev distance). The von Neumann version -- four
    # neighbours only -- makes the process barely supercritical at realistic
    # spread probabilities, so a student's first run often dies out and looks
    # broken. Diagonals included, the automaton behaves like a plaque.
    offsets = [(dr, dc) for dr in range(-rad, rad + 1) for dc in range(-rad, rad + 1)
               if (dr or dc)]

    frames: List[List[List[int]]] = []
    series = {"step": [], "infected": [], "lysed": [], "uninfected": [],
              "radius": [], "perimeter": []}

    for step in range(steps):
        timer[grid == INFECTED] -= 1
        lysing = (grid == INFECTED) & (timer <= 0)

        if lysing.any():
            # Vectorised spread: shift the lysing mask in each direction, and
            # infect susceptible cells there with the given probability.
            newly = np.zeros_like(lysing)
            for dr, dc in offsets:
                shifted = np.zeros_like(lysing)
                r0, r1 = max(0, dr), size + min(0, dr)
                c0, c1 = max(0, dc), size + min(0, dc)
                shifted[r0:r1, c0:c1] = lysing[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
                hit = shifted & (grid == UNINFECTED) & (rng.random((size, size)) < spread_probability)
                newly |= hit
            grid[lysing] = LYSED
            grid[newly] = INFECTED
            timer[newly] = int(latent_steps)

        lysed_mask = grid == LYSED
        n_lysed = int(lysed_mask.sum())
        n_inf = int((grid == INFECTED).sum())

        if n_lysed:
            rr, cc = np.nonzero(lysed_mask)
            radius = float(np.sqrt(((rr - centre) ** 2 + (cc - centre) ** 2).max()))
            # Perimeter: lysed cells with at least one non-lysed 4-neighbour.
            pad = np.pad(lysed_mask, 1, constant_values=False)
            edge = lysed_mask & ~(pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
            perimeter = int(edge.sum())
        else:
            radius, perimeter = 0.0, 0

        series["step"].append(step)
        series["infected"].append(n_inf)
        series["lysed"].append(n_lysed)
        series["uninfected"].append(int((grid == UNINFECTED).sum()))
        series["radius"].append(radius)
        series["perimeter"].append(perimeter)

        if step % max(1, int(capture_every)) == 0:
            frames.append(grid.tolist())

        if n_inf == 0:
            break

    frames.append(grid.tolist())

    lysed_mask = grid == LYSED
    area = int(lysed_mask.sum())
    final_radius = series["radius"][-1] if series["radius"] else 0.0
    perimeter = series["perimeter"][-1] if series["perimeter"] else 0
    # Circularity: 1 for a perfect disc, lower for a ragged edge. Real plaques
    # are rarely above 0.8 and the students' own plates will not be either.
    circularity = float(4 * np.pi * area / perimeter ** 2) if perimeter else float("nan")

    steps_run = len(series["step"])
    growth_rate = float(np.polyfit(series["step"][-min(20, steps_run):],
                                   series["radius"][-min(20, steps_run):], 1)[0]) \
        if steps_run >= 4 else float("nan")

    stopped = series["infected"][-1] == 0 if series["infected"] else True
    fizzled = stopped and area < 20
    if not stopped:
        reason = "the simulation reached its step limit"
    elif fizzled:
        reason = (f"the infection fizzled out after only {area} cells -- with a spread probability "
                  f"of {spread_probability:.2f} each lysis produces about "
                  f"{spread_probability * len(offsets):.1f} new infections on average, and a chain "
                  f"that close to 1 often dies by chance. That is a real result about real phage, "
                  f"not a broken simulation: raise the spread probability and watch the threshold")
    elif resistant_fraction > 0:
        reason = "the infection ran out of susceptible neighbours"
    else:
        reason = "the infection died out"

    return {
        "size": size, "steps_run": steps_run,
        "parameters": {"latent_steps": int(latent_steps),
                       "spread_probability": float(spread_probability),
                       "diffusion_radius": rad, "resistant_fraction": float(resistant_fraction),
                       "n_seeds": len(seeds), "seed": int(seed)},
        "frames": frames, "frame_every": int(capture_every),
        "state_legend": {str(k): v for k, v in STATES.items()},
        "series": series,
        "final": {
            "plaque_area_cells": area,
            "plaque_radius_cells": float(final_radius),
            "perimeter_cells": perimeter,
            "circularity": circularity,
            "radial_growth_rate": growth_rate,
            "fraction_lawn_cleared": float(area / (size * size)),
            "stopped_because": reason,
        },
        "fizzled": bool(fizzled),
        "expected_offspring_per_lysis": float(spread_probability * len(offsets)),
        "teaches": (
            "A plaque is not a hole that appears -- it is a wave that travels. Its radius grows "
            "roughly linearly with time, which is why plaques are round and why a plate left "
            "overnight has bigger ones than a plate read after four hours."),
        "plain_language": (
            f"After {steps_run} steps the plaque is about {final_radius:.0f} cells across the "
            f"radius, covering {area:,} cells ({area / (size * size):.1%} of the lawn), and it grew "
            f"at roughly {growth_rate:.2f} cells of radius per step. Its circularity is "
            f"{circularity:.2f} (1.0 would be a perfect circle) -- the edge is ragged because "
            f"infection spreads by chance, not by rule. "
            + (f"With {resistant_fraction:.0%} of the lawn resistant, " if resistant_fraction > 0 else "")
            + f"the run ended because {reason}."),
        "p_value": None,
    }
