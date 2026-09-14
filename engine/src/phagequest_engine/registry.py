"""The typed tool registry (spec section 06, point 1).

    "Statistics are typed tools, not 'write some Python.' The model chooses the
    test and explains the output; it never computes."

Every tool here has a JSON schema, a declared grade band, and a declared
curriculum origin. The schema is what makes the choice auditable: a teacher can
read `compare_groups(design="independent", justification="...")` and see what
was asked, independently of what the model said about it.

`justification` is required on every inferential tool. It is logged, shown to
the teacher, and it is the thing the specification-search detector reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

from . import curves, diversity, geometry, inference, power, resampling, sequence, validation
from .simulation import labmath, ode, plaque, stochastic

__all__ = ["Tool", "TOOLS", "tool_schemas", "call_tool", "tools_for_grade"]


@dataclass
class Tool:
    name: str
    fn: Callable
    summary: str
    parameters: Dict[str, Any]        # JSON schema
    grades: str                       # e.g. "5-12"
    curriculum: str = ""
    inferential: bool = False
    category: str = "statistics"

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.summary,
                "input_schema": self.parameters, "grades": self.grades,
                "curriculum": self.curriculum, "category": self.category,
                "inferential": self.inferential}


def _p(**props) -> Dict[str, Any]:
    required = [k for k, v in props.items() if v.pop("_required", False)]
    return {"type": "object", "properties": props, "required": required,
            "additionalProperties": False}


_NUM_ARRAY = {"type": "array", "items": {"type": "number"}}
_JUSTIFY = {"type": "string", "_required": True,
            "description": "Why this test, on this data, right now. Logged and shown to the "
                           "teacher. One sentence about the DESIGN, not about the desired result."}
_GROUPS = {"type": "object", "_required": True,
           "additionalProperties": _NUM_ARRAY,
           "description": "Group label -> its measurements. One entry per condition."}


TOOLS: Dict[str, Tool] = {}


def _register(t: Tool) -> Tool:
    TOOLS[t.name] = t
    return t


# ---------------------------------------------------------------- statistics
_register(Tool(
    "describe", inference.describe,
    "Summarise each group: n, mean, spread and a bootstrap interval on the mean. "
    "No p-values. This is the Grade 5-6 surface and the safe first call for any dataset.",
    _p(groups=_GROUPS), grades="5-12", curriculum="G5-L9, G6-L9", category="statistics"))

_register(Tool(
    "compare_groups", inference.compare_groups,
    "Is one group really different, or did we get lucky? Chooses the test from assumption "
    "checks, always returns an effect size with an interval, runs an automatic stability check, "
    "and applies the multiple-comparison ledger. Never computes on fewer than 3 per group.",
    _p(groups=_GROUPS,
       design={"type": "string", "enum": ["independent", "paired"], "default": "independent"},
       alternative={"type": "string", "enum": ["two-sided", "less", "greater"],
                    "default": "two-sided"},
       prefer_robust={"type": "boolean", "default": False,
                      "description": "Force a rank/permutation route regardless of assumptions."},
       justification=_JUSTIFY),
    grades="7-12", curriculum="G6-L9, G7-L7, G7-L9, G8-L9", inferential=True))

_register(Tool(
    "shuffle_test", resampling.shuffle_test,
    "The Grade 7 test: mix the two groups together, split them at random many times, and count "
    "how often chance beats the real difference. Returns the null distribution so it can be "
    "drawn. Assumes nothing about the shape of the data.",
    _p(group_a=dict(_NUM_ARRAY, _required=True), group_b=dict(_NUM_ARRAY, _required=True),
       statistic={"type": "string", "enum": ["mean_difference", "median_difference",
                                             "ratio_of_means"], "default": "mean_difference"},
       alternative={"type": "string", "enum": ["two-sided", "less", "greater"],
                    "default": "two-sided"},
       justification=_JUSTIFY),
    grades="7-12", curriculum="G7-L4, G7-L9", inferential=True))

_register(Tool(
    "correlate", inference.correlate,
    "Do two measurements move together? Picks Spearman or Pearson from the data, returns a "
    "bootstrap interval, and reports the robust Theil-Sen slope alongside least squares.",
    _p(x=dict(_NUM_ARRAY, _required=True), y=dict(_NUM_ARRAY, _required=True),
       method={"type": "string", "enum": ["auto", "pearson", "spearman"], "default": "auto"},
       justification=_JUSTIFY),
    grades="8-12", curriculum="G8-L12", inferential=True))

_register(Tool(
    "contingency", inference.contingency,
    "Are two categorical things related? Chi-square, or Fisher's exact test when the expected "
    "counts are too small for it.",
    _p(table={"type": "array", "_required": True,
              "items": {"type": "array", "items": {"type": "integer"}}},
       row_labels={"type": "array", "items": {"type": "string"}},
       col_labels={"type": "array", "items": {"type": "string"}},
       justification=_JUSTIFY),
    grades="8-12", curriculum="G5-L9, G8-L10", inferential=True))

_register(Tool(
    "trend_test", inference.trend_test,
    "Is a series of daily readings going up or down? Mann-Kendall plus a Theil-Sen slope with an "
    "interval. Survives missed days and one bad probe contact.",
    _p(values=dict(_NUM_ARRAY, _required=True), times=_NUM_ARRAY, justification=_JUSTIFY),
    grades="9-12", curriculum="G7-L7, G9-L9", inferential=True))

_register(Tool(
    "bootstrap_ci", resampling.bootstrap_ci,
    "How uncertain is this number? Bias-corrected bootstrap interval on a mean, median, SD or IQR.",
    _p(values=dict(_NUM_ARRAY, _required=True),
       statistic={"type": "string", "enum": ["mean", "median", "sd", "max", "min", "iqr"],
                  "default": "mean"},
       confidence={"type": "number", "default": 0.95, "minimum": 0.5, "maximum": 0.999}),
    grades="6-12", curriculum="all measurement lessons"))

_register(Tool(
    "jackknife", resampling.jackknife,
    "Which single measurement is carrying the result? Leave-one-out influence.",
    _p(values=dict(_NUM_ARRAY, _required=True),
       statistic={"type": "string", "enum": ["mean", "median", "sd", "iqr"], "default": "mean"}),
    grades="8-12", curriculum="G8-L10 error analysis"))

# ---------------------------------------------------------------- power
_register(Tool(
    "n_for_ttest", power.n_for_ttest,
    "How many samples do we need? Ask BEFORE the experiment. Returns required n per group for a "
    "two-group comparison at a target power.",
    _p(effect_size={"type": "number", "_required": True,
                    "description": "Cohen's d -- how big a difference would actually matter to you."},
       power={"type": "number", "default": 0.8}, alpha={"type": "number", "default": 0.05}),
    grades="8-12", curriculum="G7-L7, G7-L9 (design stage)"))

_register(Tool(
    "n_for_anova", power.n_for_anova,
    "Required replicates per level for a multi-level experiment (three temperatures, four pH "
    "levels, four salt levels).",
    _p(effect_size_f={"type": "number", "_required": True},
       k_groups={"type": "integer", "_required": True, "minimum": 2},
       power={"type": "number", "default": 0.8}),
    grades="8-12", curriculum="G6-L9, G7-L9, G8-L9 (design stage)"))

_register(Tool(
    "achieved_power", power.achieved_power,
    "After the experiment: what is the smallest effect this design could reliably have detected? "
    "Reports the minimum detectable effect, not post-hoc power, and says why.",
    _p(observed_effect={"type": "number", "_required": True},
       n_per_group={"type": "integer", "_required": True},
       k_groups={"type": "integer", "default": 2},
       effect_size_type={"type": "string", "default": "hedges_g"}),
    grades="9-12", curriculum="reporting stage"))

# ---------------------------------------------------------------- curves
_register(Tool(
    "fit_curve", curves.fit_curve,
    "Fit a named model to x-y data and report every parameter with a bootstrap interval. "
    "Refuses to fit a model with more parameters than the data can support.",
    _p(x=dict(_NUM_ARRAY, _required=True), y=dict(_NUM_ARRAY, _required=True),
       model={"type": "string", "_required": True, "enum": sorted(curves.MODELS)},
       justification=_JUSTIFY),
    grades="8-12", curriculum="G8-L12"))

_register(Tool(
    "compare_models", curves.compare_models,
    "Fit several curves to the same data and rank by AICc. Always includes a straight line, "
    "because the honest question is whether the curve beats the line by enough to justify itself.",
    _p(x=dict(_NUM_ARRAY, _required=True), y=dict(_NUM_ARRAY, _required=True),
       models={"type": "array", "items": {"type": "string", "enum": sorted(curves.MODELS)}}),
    grades="9-12", curriculum="G8-L12"))

_register(Tool(
    "fit_growth_curve", curves.fit_growth_curve,
    "Growth-curve analysis: lag, maximum growth rate, carrying capacity -- and, if the culture "
    "crashed, the timing and depth of the lysis, which is the interesting half.",
    _p(time=dict(_NUM_ARRAY, _required=True), density=dict(_NUM_ARRAY, _required=True),
       model={"type": "string", "enum": ["gompertz", "logistic_growth"], "default": "gompertz"}),
    grades="9-12", curriculum="G6-L9, G7-L9, G8-L9"))

# ---------------------------------------------------------------- diversity
_register(Tool(
    "alpha_diversity", diversity.alpha_diversity,
    "Richness, Shannon, Simpson, Chao1 and evenness per sample, with a sequencing-depth warning.",
    _p(samples={"type": "object", "_required": True,
                "additionalProperties": {"type": "object",
                                         "additionalProperties": {"type": "number"}},
                "description": "sample name -> {taxon: count}"}),
    grades="9-12", curriculum="G9-L12", category="diversity"))

_register(Tool(
    "compare_composition", diversity.compare_composition,
    "Are these communities different? The SAFE front door: refuses to hand proportions to a "
    "t-test, runs CLR + PERMANOVA instead, and explains why. Use this, never compare_groups, on "
    "abundance data.",
    _p(samples={"type": "object", "_required": True,
                "additionalProperties": {"type": "object",
                                         "additionalProperties": {"type": "number"}}},
       groups={"type": "object", "_required": True,
               "additionalProperties": {"type": "string"},
               "description": "sample name -> group label"},
       metric={"type": "string", "enum": ["aitchison", "braycurtis", "jaccard"],
               "default": "aitchison"}),
    grades="9-12", curriculum="G9-L12, G8-L10", inferential=True, category="diversity"))

_register(Tool(
    "rarefaction_curve", diversity.rarefaction_curve,
    "Did we look hard enough? Richness against counting effort, with intervals.",
    _p(counts=dict(_NUM_ARRAY, _required=True)),
    grades="9-12", curriculum="G9-L12", category="diversity"))

# ---------------------------------------------------------------- lab maths
_register(Tool(
    "dilution_series", labmath.dilution_series,
    "The 1:10 chain with the powers of ten made explicit, and which tube to plate.",
    _p(start_concentration={"type": "number", "default": 1e9},
       fold={"type": "number", "default": 10}, steps={"type": "integer", "default": 8},
       volume_transferred={"type": "number", "default": 0.1},
       volume_diluent={"type": "number", "default": 0.9}),
    grades="5-12", curriculum="G5-L6", category="lab"))

_register(Tool(
    "pfu_per_ml", labmath.pfu_per_ml,
    "Titre from one plate WITH its Poisson counting interval, and whether the count was in the "
    "usable 30-300 range.",
    _p(plaque_count={"type": "integer", "_required": True, "minimum": 0},
       dilution_factor={"type": "number", "_required": True, "minimum": 1},
       volume_plated_mL={"type": "number", "default": 0.1}),
    grades="8-12", curriculum="G8-L10", category="lab"))

_register(Tool(
    "titre_from_plates", labmath.titre_from_plates,
    "Pool several plates into one titre correctly (not by averaging), and test whether the "
    "plates agree with each other better than chance.",
    _p(plaque_counts={"type": "array", "items": {"type": "integer"}, "_required": True},
       dilution_factors=dict(_NUM_ARRAY, _required=True),
       volume_plated_mL={"type": "number", "default": 0.1}),
    grades="9-12", curriculum="G8-L10", inferential=True, category="lab"))

_register(Tool(
    "moi", labmath.moi,
    "Multiplicity of infection, plus what Poisson says actually happens at it -- including that "
    "37% of cells are untouched at MOI 1.",
    _p(phage_pfu_per_mL={"type": "number", "_required": True},
       bacteria_cfu_per_mL={"type": "number", "_required": True},
       volume_mL={"type": "number", "default": 1.0}),
    grades="8-12", curriculum="G8-L10", category="lab"))

# ---------------------------------------------------------------- simulation
_register(Tool(
    "simulate", ode.simulate,
    "Run a host-phage population simulation: lytic, lysogenic, evolving resistance, or a "
    "resource-explicit chemostat. Erlang latent-period chain, LSODA solver.",
    _p(model={"type": "string", "enum": sorted(ode.MODELS), "default": "lytic"},
       params={"type": "object", "additionalProperties": {"type": "number"}},
       S0={"type": "number", "default": 1e6}, P0={"type": "number", "default": 1e4},
       hours={"type": "number", "default": 24}),
    grades="8-12", curriculum="G8-L12, G8-L10, G6-L8", category="simulation"))

_register(Tool(
    "parameter_sweep", ode.parameter_sweep,
    "Sweep one parameter across a range and tabulate the outcome. Output feeds straight into "
    "fit_curve. This is G8-L12's four-parameter-sweep pedagogy with a phage in the flask.",
    _p(model={"type": "string", "enum": sorted(ode.MODELS), "default": "lytic"},
       parameter={"type": "string", "_required": True},
       values=_NUM_ARRAY, n_values={"type": "integer", "default": 9},
       base_params={"type": "object", "additionalProperties": {"type": "number"}}),
    grades="8-12", curriculum="G8-L12", category="simulation"))

_register(Tool(
    "one_step_growth", ode.one_step_growth,
    "The one-step growth curve, run at 1, 3 and n latent-period stages side by side, so the "
    "latent period becomes visible as stages rather than a hidden delay.",
    _p(latent={"type": "number", "default": 0.5}, burst={"type": "number", "default": 100},
       n_stages={"type": "integer", "default": 20, "minimum": 1, "maximum": 40}),
    grades="9-12", curriculum="G8-L10", category="simulation"))

_register(Tool(
    "gillespie_lytic", stochastic.gillespie_lytic,
    "Exact stochastic simulation, repeated many times with identical settings, to show that they "
    "do not come out the same. The answer to 'why did my plate have no plaques?'.",
    _p(S0={"type": "integer", "default": 200}, P0={"type": "integer", "default": 5},
       adsorption={"type": "number", "default": 1e-3}, burst={"type": "integer", "default": 50},
       n_runs={"type": "integer", "default": 30, "minimum": 2, "maximum": 200}),
    grades="9-12", curriculum="G8-L10", category="simulation"))

_register(Tool(
    "poisson_plaques", stochastic.poisson_plaques,
    "Why an empty plate is not proof of no phage: the Poisson probability of zero, and a "
    "simulated set of plates to look at.",
    _p(expected_plaques={"type": "number", "default": 3.0},
       n_plates={"type": "integer", "default": 20}),
    grades="8-12", curriculum="G8-L10", category="simulation"))

_register(Tool(
    "plaque_growth", plaque.plaque_growth,
    "Grow plaques on a simulated lawn and measure radius, area and edge roughness over time.",
    _p(size={"type": "integer", "default": 151, "minimum": 21, "maximum": 401},
       steps={"type": "integer", "default": 120},
       latent_steps={"type": "integer", "default": 4},
       spread_probability={"type": "number", "default": 0.45, "minimum": 0, "maximum": 1},
       resistant_fraction={"type": "number", "default": 0.0, "minimum": 0, "maximum": 1}),
    grades="8-12", curriculum="G8-L10", category="simulation"))

# ---------------------------------------------------------------- sequence
_register(Tool(
    "translate", sequence.translate,
    "Translate DNA to protein codon by codon, showing the working, so a student can check their "
    "codon-wheel answer and see WHERE they went wrong.",
    _p(sequence={"type": "string", "_required": True},
       frame={"type": "integer", "default": 1, "enum": [1, 2, 3, -1, -2, -3]}),
    grades="9-12", curriculum="G9-L7", category="sequence"))

_register(Tool(
    "gc_content", sequence.gc_content,
    "GC content, melting temperature estimate, and an optional sliding window along the genome.",
    _p(sequence={"type": "string", "_required": True},
       window={"type": "integer", "default": 0}),
    grades="6-12", curriculum="G6-L10, G8-L11", category="sequence"))

_register(Tool(
    "find_orfs", sequence.find_orfs,
    "The simplest possible gene caller, with its limitations stated. Exists so a student can see "
    "what a naive caller does and where it fails -- the setup for comparing PHANOTATE to Prodigal.",
    _p(sequence={"type": "string", "_required": True},
       min_length_aa={"type": "integer", "default": 30},
       all_frames={"type": "boolean", "default": True}),
    grades="9-12", curriculum="G5-L11, G9-L7", category="sequence"))

_register(Tool(
    "codon_usage", sequence.codon_usage,
    "Codon usage bias (RSCU) -- which spelling of each amino acid this genome prefers.",
    _p(sequence={"type": "string", "_required": True}, frame={"type": "integer", "default": 1}),
    grades="11-12", curriculum="G9-L7", category="sequence"))

# ---------------------------------------------------------------- geometry
_register(Tool(
    "tetranucleotide_vector", geometry.tetranucleotide_vector,
    "The Fingerprint module: a genome's 256-dimensional 4-mer profile. G5-L11's barcode matching, "
    "grown up.",
    _p(sequence={"type": "string", "_required": True}, name={"type": "string", "default": "sequence"},
       canonical={"type": "boolean", "default": False}),
    grades="7-12", curriculum="G5-L11", category="geometry"))

_register(Tool(
    "component_sweep", geometry.component_sweep,
    "The Islands module: build the k-NN graph at every k and find the stability plateau. A "
    "structure that survives many values of k is real; one that appears at a single k is an "
    "artefact of the setting.",
    _p(distance_matrix={"type": "array", "_required": True,
                        "items": {"type": "array", "items": {"type": "number"}}},
       k_min={"type": "integer", "default": 3}, k_max={"type": "integer", "default": 40},
       names={"type": "array", "items": {"type": "string"}}),
    grades="9-12", curriculum="capstone", category="geometry"))

_register(Tool(
    "bridge_analysis", geometry.bridge_analysis,
    "The Bridges module: Ollivier-Ricci curvature, the negatively curved edges holding two clades "
    "together, and the hub genomes carrying them. Runs weighted AND unweighted and reports both.",
    _p(distance_matrix={"type": "array", "_required": True,
                        "items": {"type": "array", "items": {"type": "number"}}},
       clade_labels={"type": "array", "items": {"type": "integer"}},
       k={"type": "integer"}, alpha={"type": "number", "default": 0.5},
       names={"type": "array", "items": {"type": "string"}},
       method={"type": "string", "enum": ["exact", "sinkhorn"], "default": "exact"}),
    grades="11-12", curriculum="capstone / WP2", inferential=True, category="geometry"))

# ---------------------------------------------------------------- data
_register(Tool(
    "validate_table", validation.validate_table,
    "Check an uploaded table before any statistics touch it. Every problem names the spreadsheet "
    "row, the column, the value, and what to do about it.",
    _p(rows={"type": "array", "_required": True, "items": {"type": "object"}},
       template={"type": "string", "enum": sorted(validation.TEMPLATES)}),
    grades="5-12", curriculum="every data-collecting lesson", category="data"))

_register(Tool(
    "infer_schema", validation.infer_schema,
    "Work out what an unfamiliar upload contains and suggest the closest curriculum template.",
    _p(rows={"type": "array", "_required": True, "items": {"type": "object"}}),
    grades="5-12", category="data"))


# ==========================================================================
def tool_schemas(grade: Optional[int] = None,
                 categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Schemas for the tools a given grade is allowed to reach.

    The grade ladder (spec section 05) is enforced here rather than in the UI,
    so the model cannot be talked into a Grade 12 tool by a Grade 5 student.
    """
    out = []
    for t in TOOLS.values():
        if categories and t.category not in categories:
            continue
        if grade is not None:
            lo, _, hi = t.grades.partition("-")
            try:
                if not (int(lo) <= grade <= int(hi or lo)):
                    continue
            except ValueError:
                pass
        out.append(t.to_dict())
    return sorted(out, key=lambda d: (d["category"], d["name"]))


def tools_for_grade(grade: int) -> List[str]:
    return [d["name"] for d in tool_schemas(grade=grade)]


def call_tool(name: str, arguments: Dict[str, Any],
              grade: Optional[int] = None) -> Dict[str, Any]:
    """Invoke a registered tool. The only path by which a number may be produced.

    Returns the tool's bound result, which always carries a `transcript_id`.
    Anything that reaches a user without one is, by the platform's invariant,
    not renderable.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return {"error": "unknown_tool", "message": f"No tool named {name!r}.",
                "available": sorted(TOOLS)}

    if grade is not None and name not in tools_for_grade(grade):
        return {"error": "grade_restricted",
                "message": (f"'{name}' is a Grade {tool.grades} tool and this workspace is Grade "
                            f"{grade}. That is not a permissions problem to route around -- the "
                            f"tool would produce an answer this student has not yet been given the "
                            f"ideas to check."),
                "available": tools_for_grade(grade)}

    schema = tool.parameters

    # Checked first: "you did not say why" is a more useful error than
    # "missing required: ['justification']", and it is the one a student sees most.
    if tool.inferential and not str(arguments.get("justification", "")).strip():
        return {"error": "justification_required",
                "message": ("Every inferential test needs a written reason, which is logged and "
                            "shown to the teacher. Say why THIS test on THIS data.")}

    missing = [k for k in schema.get("required", []) if k not in arguments]
    if missing:
        return {"error": "missing_arguments", "message": f"Missing required: {missing}.",
                "schema": schema}

    known = set(schema.get("properties", {}))
    extra = [k for k in arguments if k not in known]
    if extra:
        return {"error": "unknown_arguments",
                "message": f"Not parameters of {name}: {extra}.", "schema": schema}

    try:
        result = tool.fn(**arguments)
    except Exception as exc:
        return {"error": "execution_failed", "message": f"{type(exc).__name__}: {exc}",
                "tool": name}

    result["tool"] = name
    result["tool_category"] = tool.category
    result["curriculum"] = tool.curriculum
    return result
