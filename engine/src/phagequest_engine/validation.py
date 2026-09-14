"""Upload validation (spec section 03.1).

    "Declarative schemas checked on every student upload. Catches the swapped
    column, the text in a numeric field, the impossible pH of 19 -- *before*
    the statistics run, with an error that names the row."

pandera is the Tier-1 choice and is not in Pyodide, and its error messages are
written for data engineers. A Grade 6 needs "row 14, the pH column says 19 --
pH only goes from 0 to 14; did you mean 1.9?". So this is a small declarative
schema checker whose entire output budget goes on the message.

It also ships the curriculum's own experiment templates as ready-made schemas,
so a teacher can upload the table their class already fills in by hand and have
it understood on the first try.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence

from .transcript import bound

__all__ = ["Column", "Schema", "TEMPLATES", "validate_table", "infer_schema", "coerce_table"]

_NUMERIC_RE = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$")
_MISSING = {"", "na", "n/a", "nan", "none", "null", "-", "--", "?", "nd", "n.d."}


@dataclass
class Column:
    name: str
    dtype: str = "number"          # number | integer | text | category | date
    required: bool = True
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    allowed: Optional[List[str]] = None
    unit: Optional[str] = None
    describes: str = ""
    hint: str = ""                 # shown when a value fails, in the student's language

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Schema:
    key: str
    label: str
    columns: List[Column]
    curriculum: str = ""
    min_rows: int = 3
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# --------------------------------------------------------------------------
# Curriculum templates. These are the tables the classes already produce.
# --------------------------------------------------------------------------
TEMPLATES: Dict[str, Schema] = {
    "serial_dilution": Schema(
        "serial_dilution", "Serial dilution series",
        [Column("tube", "integer", minimum=0, describes="Which tube in the chain (0 = undiluted)"),
         Column("dilution_factor", "number", minimum=1,
                describes="How many times weaker than the original",
                hint="A dilution factor is at least 1. If you wrote 0.1, you meant 10."),
         Column("count", "integer", required=False, minimum=0,
                describes="Plaques or colonies counted",
                hint="Counts are whole numbers and cannot be negative."),
         Column("volume_plated_mL", "number", required=False, minimum=1e-6, maximum=10,
                unit="mL", describes="Volume put on the plate")],
        curriculum="G5-L6"),
    "water_quality": Schema(
        "water_quality", "Water-quality strip readings",
        [Column("sample", "text", describes="Where the water came from"),
         Column("pH", "number", required=False, minimum=0, maximum=14,
                hint="pH runs from 0 to 14. A value of 19 is usually a typo for 1.9."),
         Column("nitrate_ppm", "number", required=False, minimum=0, unit="ppm"),
         Column("hardness_ppm", "number", required=False, minimum=0, unit="ppm"),
         Column("chlorine_ppm", "number", required=False, minimum=0, unit="ppm")],
        curriculum="G5-L9"),
    "growth_vs_factor": Schema(
        "growth_vs_factor", "Growth against one environmental factor",
        [Column("level", "text", describes="The condition: temperature, pH, salt level..."),
         Column("replicate", "integer", required=False, minimum=1,
                describes="Which repeat this row is",
                hint="Replicates are numbered from 1. Every level needs at least three."),
         Column("measurement", "number", describes="What you measured"),
         Column("units", "text", required=False)],
        curriculum="G6-L9 / G7-L9 / G8-L9",
        notes=["This is the one-way design behind three different lessons. Keep one row per "
               "replicate -- do not average before uploading, because the spread between "
               "replicates is most of the information."]),
    "daily_readings": Schema(
        "daily_readings", "Daily time series",
        [Column("day", "number", minimum=0, describes="Day number or date"),
         Column("reading", "number", describes="Today's measurement"),
         Column("group", "text", required=False, describes="Control or treatment"),
         Column("notes", "text", required=False)],
        curriculum="G7-L7 / G9-L9", min_rows=4,
        notes=["A missed day is fine -- leave the row out rather than writing 0. "
               "A 0 means you measured zero, and the trend test will believe you."]),
    "plaque_assay": Schema(
        "plaque_assay", "Plaque assay",
        [Column("sample", "text", describes="Which sample or site"),
         Column("dilution_factor", "number", minimum=1),
         Column("plaque_count", "integer", minimum=0,
                hint="Count whole plaques. 'TNTC' (too numerous to count) should be left blank, "
                     "not written as a number."),
         Column("volume_plated_mL", "number", required=False, minimum=1e-6, maximum=10, unit="mL"),
         Column("plaque_morphology", "text", required=False,
                describes="Clear, cloudy, large, small -- clear suggests lytic, cloudy suggests "
                          "the phage sometimes hides")],
        curriculum="G8-L10 / G9-L11"),
    "site_survey": Schema(
        "site_survey", "Sampling sites",
        [Column("site_id", "text"),
         Column("latitude", "number", required=False, minimum=-90, maximum=90,
                hint="Latitude runs -90 to 90. If your number is bigger, it is probably longitude."),
         Column("longitude", "number", required=False, minimum=-180, maximum=180),
         Column("habitat", "text", required=False),
         Column("soil_pH", "number", required=False, minimum=0, maximum=14),
         Column("moisture_percent", "number", required=False, minimum=0, maximum=100)],
        curriculum="G8-L10 / G9-L11"),
    "community_counts": Schema(
        "community_counts", "Community composition (counts per taxon)",
        [Column("sample", "text", describes="Which layer or site"),
         Column("taxon", "text", describes="What you saw"),
         Column("count", "integer", minimum=0)],
        curriculum="G9-L12",
        notes=["Long format: one row per taxon per sample. The diversity tools pivot it for you."]),
}


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    return str(v).strip().lower() in _MISSING


def _as_number(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    s = str(v).strip().replace(",", "")
    # Tolerate the units students type into the cell: "7.2 pH", "35 C", "12%"
    s = re.sub(r"[^\d.eE+\-]", "", s)
    if _NUMERIC_RE.match(s):
        try:
            return float(s)
        except ValueError:
            return None
    return None


@bound("validate_table", data_args=("rows",))
def validate_table(rows: Sequence[Dict[str, Any]], template: Optional[str] = None,
                   schema: Optional[Dict[str, Any]] = None,
                   max_issues: int = 60) -> Dict[str, Any]:
    """Check an uploaded table before any statistics touch it.

    Every issue names the row number the student can see in their spreadsheet
    (1-based, counting the header as row 1), the column, the offending value,
    and what to do about it. Nothing is silently coerced.
    """
    data = list(rows)
    if not data:
        return {"valid": False, "issues": [{"severity": "block", "row": None, "column": None,
                                            "message": "The file has no rows."}],
                "n_rows": 0, "p_value": None}

    sch: Optional[Schema] = None
    if schema is not None:
        sch = Schema(key=schema.get("key", "custom"), label=schema.get("label", "Custom schema"),
                     columns=[Column(**c) for c in schema.get("columns", [])],
                     curriculum=schema.get("curriculum", ""),
                     min_rows=int(schema.get("min_rows", 3)))
    elif template is not None:
        sch = TEMPLATES.get(template)
        if sch is None:
            return {"valid": False, "n_rows": len(data),
                    "issues": [{"severity": "block", "row": None, "column": None,
                                "message": f"Unknown template {template!r}."}],
                    "available_templates": sorted(TEMPLATES), "p_value": None}

    present = list({k for r in data for k in r.keys()})
    issues: List[Dict[str, Any]] = []
    add = issues.append

    if sch is not None:
        lower = {c.lower().strip(): c for c in present}
        for col in sch.columns:
            if col.name.lower() not in lower:
                if col.required:
                    close = _closest(col.name, present)
                    add({"severity": "block", "row": None, "column": col.name,
                         "message": (f"The required column '{col.name}' is missing."
                                     + (f" Did you mean '{close}'?" if close else "")
                                     + (f" It should hold: {col.describes}" if col.describes else ""))})
                continue
            actual = lower[col.name.lower()]
            for i, row in enumerate(data):
                rownum = i + 2  # header is row 1 in the student's spreadsheet
                v = row.get(actual)
                if _is_missing(v):
                    if col.required:
                        add({"severity": "warn", "row": rownum, "column": col.name, "value": v,
                             "message": f"Row {rownum} has no value for '{col.name}'. That row "
                                        f"will be left out of any calculation using this column."})
                    continue
                if col.dtype in ("number", "integer"):
                    num = _as_number(v)
                    if num is None:
                        add({"severity": "block", "row": rownum, "column": col.name, "value": v,
                             "message": (f"Row {rownum}, column '{col.name}': '{v}' is not a "
                                         f"number. Statistics cannot be computed on text. If this "
                                         f"means 'no reading', leave the cell empty instead.")})
                        continue
                    if col.dtype == "integer" and abs(num - round(num)) > 1e-9:
                        add({"severity": "warn", "row": rownum, "column": col.name, "value": v,
                             "message": f"Row {rownum}: '{col.name}' should be a whole number but "
                                        f"is {num}."})
                    if col.minimum is not None and num < col.minimum:
                        add({"severity": "block", "row": rownum, "column": col.name, "value": num,
                             "message": (f"Row {rownum}: '{col.name}' is {num}, below the minimum "
                                         f"of {col.minimum}. " + (col.hint or ""))})
                    if col.maximum is not None and num > col.maximum:
                        add({"severity": "block", "row": rownum, "column": col.name, "value": num,
                             "message": (f"Row {rownum}: '{col.name}' is {num}, above the maximum "
                                         f"of {col.maximum}. " + (col.hint or ""))})
                elif col.allowed:
                    if str(v).strip() not in col.allowed:
                        add({"severity": "warn", "row": rownum, "column": col.name, "value": v,
                             "message": f"Row {rownum}: '{col.name}' is '{v}', which is not one of "
                                        f"{col.allowed}."})
                if len(issues) >= max_issues:
                    break

        extra = [c for c in present if c.lower() not in {x.name.lower() for x in sch.columns}]
        if extra:
            add({"severity": "info", "row": None, "column": None,
                 "message": f"Extra columns kept as-is: {', '.join(sorted(extra))}."})
        if len(data) < sch.min_rows:
            add({"severity": "warn", "row": None, "column": None,
                 "message": (f"Only {len(data)} rows. This kind of experiment needs at least "
                             f"{sch.min_rows} before any analysis means anything.")})
    else:
        # Schema-free pass: still catch the things that break statistics.
        for col in present:
            vals = [r.get(col) for r in data if not _is_missing(r.get(col))]
            nums = [_as_number(v) for v in vals]
            n_num = sum(1 for x in nums if x is not None)
            if vals and 0 < n_num < len(vals) and n_num / len(vals) > 0.7:
                bad = [(i + 2, r.get(col)) for i, r in enumerate(data)
                       if not _is_missing(r.get(col)) and _as_number(r.get(col)) is None]
                for rownum, v in bad[:5]:
                    add({"severity": "block", "row": rownum, "column": col, "value": v,
                         "message": (f"Column '{col}' is mostly numbers, but row {rownum} says "
                                     f"'{v}'. One piece of text turns the whole column into text "
                                     f"and stops every calculation on it.")})

    # Duplicate-row check: a copy-paste accident that quietly doubles your n.
    seen: Dict[str, int] = {}
    for i, row in enumerate(data):
        key = "|".join(f"{k}={row[k]}" for k in sorted(row))
        if key in seen:
            add({"severity": "warn", "row": i + 2, "column": None,
                 "message": (f"Row {i + 2} is identical to row {seen[key] + 2}. If that is a "
                             f"copy-paste accident it inflates your sample size and makes every "
                             f"result look more certain than it is.")})
        else:
            seen[key] = i

    blocking = [i for i in issues if i["severity"] == "block"]
    return {
        "valid": not blocking,
        "template": sch.key if sch else None,
        "template_label": sch.label if sch else None,
        "curriculum": sch.curriculum if sch else None,
        "n_rows": len(data), "columns": present,
        "issues": issues[:max_issues],
        "n_issues": len(issues),
        "n_blocking": len(blocking),
        "schema_notes": sch.notes if sch else [],
        "plain_language": (
            f"{len(data)} rows checked. "
            + ("Everything looks usable."
               if not issues else
               f"{len(blocking)} problem{'s' if len(blocking) != 1 else ''} must be fixed before "
               f"the statistics will run"
               f"{f', and {len(issues) - len(blocking)} thing(s) worth a look' if len(issues) > len(blocking) else ''}.")),
        "p_value": None,
    }


def _closest(name: str, candidates: Sequence[str]) -> Optional[str]:
    """Cheap edit-distance suggestion for a mistyped column name."""
    def dist(a: str, b: str) -> int:
        a, b = a.lower(), b.lower()
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]
    best, bd = None, 99
    for c in candidates:
        d = dist(name, c)
        if d < bd:
            best, bd = c, d
    return best if bd <= max(2, len(name) // 3) else None


@bound("infer_schema", data_args=("rows",))
def infer_schema(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Guess a schema from an unfamiliar upload, and suggest the closest template."""
    data = list(rows)
    if not data:
        return {"refused": True, "reason": "No rows."}
    present = list({k for r in data for k in r.keys()})
    cols = []
    for c in present:
        vals = [r.get(c) for r in data if not _is_missing(r.get(c))]
        nums = [x for x in (_as_number(v) for v in vals) if x is not None]
        if vals and len(nums) == len(vals):
            integral = all(abs(x - round(x)) < 1e-9 for x in nums)
            cols.append(Column(c, "integer" if integral else "number",
                               required=len(vals) == len(data),
                               minimum=min(nums), maximum=max(nums)).to_dict())
        else:
            uniq = sorted({str(v).strip() for v in vals})
            cols.append(Column(c, "category" if len(uniq) <= max(2, len(data) // 3) else "text",
                               required=len(vals) == len(data),
                               allowed=uniq if len(uniq) <= 12 else None).to_dict())

    lowered = {c.lower() for c in present}
    scores = {k: len(lowered & {col.name.lower() for col in s.columns})
              for k, s in TEMPLATES.items()}
    best = max(scores, key=scores.get) if scores else None
    return {
        "columns": cols, "n_rows": len(data),
        "suggested_template": best if best and scores[best] >= 2 else None,
        "template_match_score": scores.get(best, 0) if best else 0,
        "plain_language": (
            f"Found {len(present)} columns across {len(data)} rows."
            + (f" This looks like the '{TEMPLATES[best].label}' template "
               f"({TEMPLATES[best].curriculum})." if best and scores[best] >= 2 else "")),
        "p_value": None,
    }


@bound("coerce_table", data_args=("rows",))
def coerce_table(rows: Sequence[Dict[str, Any]], numeric_columns: Sequence[str]) -> Dict[str, Any]:
    """Convert named columns to numbers, reporting every value it could not convert.

    Deliberately separate from validation and never automatic: a platform that
    quietly turns 'TNTC' into 0 has fabricated a measurement.
    """
    out, dropped = [], []
    for i, row in enumerate(rows):
        new = dict(row)
        ok = True
        for c in numeric_columns:
            if c not in new:
                continue
            if _is_missing(new[c]):
                new[c] = None
                continue
            v = _as_number(new[c])
            if v is None:
                dropped.append({"row": i + 2, "column": c, "value": row[c]})
                ok = False
            new[c] = v
        out.append(new) if ok else None
    return {
        "rows": out, "n_in": len(list(rows)), "n_out": len(out),
        "dropped": dropped,
        "plain_language": (
            f"Converted {len(numeric_columns)} column(s) to numbers; kept {len(out)} of "
            f"{len(list(rows))} rows."
            + (f" {len(dropped)} value(s) could not be converted and their rows were left out -- "
               f"they are listed so you can fix them rather than lose them silently."
               if dropped else "")),
        "p_value": None,
    }
