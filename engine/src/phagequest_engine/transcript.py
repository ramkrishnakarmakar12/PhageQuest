"""Execution records: the binding that makes a number renderable.

The platform invariant (spec section 06):

    No numeric statistical claim may be rendered to a user unless it is bound
    to a specific execution record -- code, engine version, raw output,
    timestamp -- produced by a real statistics engine.

This module is the mechanism. Every public engine function is wrapped by
`@bound`, which:

  1. records the exact arguments it was called with,
  2. records the engine versions in play,
  3. hashes the input data so the same data always yields the same data_id,
  4. stores the raw result,
  5. returns a `Bound` whose payload carries a `transcript_id`.

A result without a transcript_id is, by construction, not renderable. The
frontend enforces the other half: it refuses to display a numeric field whose
owning object has no transcript_id, and the chatbot guard (guards.py) refuses
to emit a number in prose that is not present in some transcript in the turn.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import numpy as np

__all__ = [
    "Transcript",
    "TranscriptStore",
    "Bound",
    "bound",
    "engine_versions",
    "data_fingerprint",
    "get_store",
    "set_store",
]


def engine_versions() -> Dict[str, str]:
    """Exact versions of every library that can influence a number."""
    import scipy

    v = {
        "phagequest-engine": "0.1.0",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": platform.python_version(),
    }
    # statsmodels is Tier-1 only; report it when present so server-side
    # transcripts are distinguishable from browser-side ones.
    try:  # pragma: no cover - depends on tier
        import statsmodels  # type: ignore

        v["statsmodels"] = statsmodels.__version__
    except Exception:  # pragma: no cover
        pass
    v["runtime"] = "pyodide" if _in_pyodide() else "cpython"
    return v


def _in_pyodide() -> bool:
    try:
        import sys

        return "pyodide" in sys.modules or platform.machine() == "wasm32"
    except Exception:  # pragma: no cover
        return False


def _canonical(obj: Any) -> Any:
    """Make an argument JSON-serialisable and stable across runs.

    Arrays are replaced by a fingerprint rather than inlined: a transcript must
    stay small enough to display, and the fingerprint is what makes the run
    reproducible.
    """
    if isinstance(obj, np.ndarray):
        return {"__array__": {"shape": list(obj.shape), "dtype": str(obj.dtype),
                              "sha256": data_fingerprint(obj)}}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        # Long numeric sequences are data, not parameters: fingerprint them.
        if len(obj) > 64 and all(isinstance(x, (int, float, np.number)) for x in obj):
            arr = np.asarray(obj, dtype=float)
            return {"__array__": {"shape": list(arr.shape), "dtype": "float64",
                                  "sha256": data_fingerprint(arr)}}
        return [_canonical(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        return _canonical(asdict(obj))
    return repr(obj)


def data_fingerprint(*arrays: Any) -> str:
    """Stable sha256 over one or more array-likes.

    Two runs on the same data produce the same fingerprint, which is how a
    teacher can tell that a student re-ran an analysis rather than changing it.
    """
    h = hashlib.sha256()
    for a in arrays:
        arr = np.asarray(a)
        if arr.dtype.kind in "OUS":
            h.update(json.dumps(arr.tolist(), sort_keys=True, default=str).encode())
        else:
            h.update(str(arr.shape).encode())
            h.update(np.ascontiguousarray(arr.astype(np.float64)).tobytes())
    return h.hexdigest()[:32]


@dataclass
class Transcript:
    """One execution record. This is the product (spec section 08)."""

    transcript_id: str
    tool: str
    call: str                      # human-readable reconstruction of the call
    params: Dict[str, Any]
    data_id: Optional[str]
    engine: Dict[str, str]
    started_at: float
    duration_ms: float
    result: Dict[str, Any]
    notes: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def engine_string(self) -> str:
        e = self.engine
        parts = [f"scipy {e.get('scipy')}", f"numpy {e.get('numpy')}"]
        if "statsmodels" in e:
            parts.append(f"statsmodels {e['statsmodels']}")
        return " / ".join(parts)


class TranscriptStore:
    """In-memory transcript log for one workspace session.

    Also carries the multiple-comparison ledger, because the ledger is only
    meaningful relative to a set of executions on the same data.
    """

    def __init__(self) -> None:
        self._records: Dict[str, Transcript] = {}
        self._order: List[str] = []

    def put(self, t: Transcript) -> Transcript:
        self._records[t.transcript_id] = t
        self._order.append(t.transcript_id)
        return t

    def get(self, transcript_id: str) -> Optional[Transcript]:
        return self._records.get(transcript_id)

    def all(self) -> List[Transcript]:
        return [self._records[i] for i in self._order]

    def since(self, marker: int) -> List[Transcript]:
        return [self._records[i] for i in self._order[marker:]]

    def marker(self) -> int:
        return len(self._order)

    def inferential_on(self, data_id: Optional[str]) -> List[Transcript]:
        """Every inferential test already run against this dataset.

        This is the raw material of the visible multiple-comparison ledger
        (spec section 06, point 4).
        """
        if data_id is None:
            return []
        return [
            t for t in self.all()
            if t.data_id == data_id
            and t.error is None
            and isinstance(t.result, dict)
            and t.result.get("p_value") is not None
        ]

    def numbers(self) -> List[float]:
        """Every numeric value this store has ever produced.

        The chatbot guard cross-checks model prose against exactly this set.

        Numbers inside the engine's own generated sentences count too. A result
        carries a `plain_language` field that the engine wrote -- "95% CI 0.12
        to 0.45", "held in 97% of 412 re-runs" -- and a model quoting that
        sentence is reporting, not inventing. Excluding those strings made the
        guard fire on the engine's own words, which trains people to ignore it,
        and a guard that is routinely ignored protects nobody.
        """
        import re as _re

        out: List[float] = []
        num_re = _re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

        def walk(v: Any) -> None:
            if isinstance(v, bool):
                return
            if isinstance(v, (int, float)) and np.isfinite(v):
                out.append(float(v))
            elif isinstance(v, str):
                for m in num_re.findall(v.replace(",", "")):
                    try:
                        f = float(m)
                    except ValueError:
                        continue
                    if np.isfinite(f):
                        out.append(f)
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)

        for t in self.all():
            walk(t.result)
        return out

    def clear(self) -> None:
        self._records.clear()
        self._order.clear()


_STORE = TranscriptStore()


def get_store() -> TranscriptStore:
    return _STORE


def set_store(store: TranscriptStore) -> TranscriptStore:
    global _STORE
    _STORE = store
    return _STORE


@dataclass
class Bound:
    """A result and the execution record that licenses its display."""

    value: Dict[str, Any]
    transcript: Transcript

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.value)
        out["transcript_id"] = self.transcript.transcript_id
        out["engine"] = self.transcript.engine_string
        out["transcript"] = self.transcript.to_dict()
        return out


def bound(tool_name: str, data_args: tuple = ()) -> Callable:
    """Decorator: bind a function's numeric output to an execution record.

    `data_args` names the parameters that carry the student's data, so the
    fingerprint is computed over the data and not over the knobs. Two analyses
    of the same table share a data_id, which is what lets the ledger know a
    fifth test has been run on it.
    """

    def deco(fn: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            import inspect

            sig = inspect.signature(fn)
            try:
                b = sig.bind(*args, **kwargs)
                b.apply_defaults()
                all_kw = dict(b.arguments)
            except TypeError:
                all_kw = dict(kwargs)

            data_id = None
            if data_args:
                present = [all_kw[n] for n in data_args if n in all_kw and all_kw[n] is not None]
                if present:
                    data_id = data_fingerprint(*present)

            params = {k: _canonical(v) for k, v in all_kw.items() if k not in data_args}
            started = time.time()
            tid = f"tx_{uuid.uuid4().hex[:16]}"
            err = None
            result: Dict[str, Any] = {}
            try:
                result = fn(*args, **kwargs)
                if not isinstance(result, dict):
                    result = {"value": result}
            except Exception as exc:  # record failures too -- they are evidence
                err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                t = Transcript(
                    transcript_id=tid,
                    tool=tool_name,
                    call=_render_call(tool_name, all_kw, data_args),
                    params=params,
                    data_id=data_id,
                    engine=engine_versions(),
                    started_at=started,
                    duration_ms=round((time.time() - started) * 1000, 3),
                    result=_canonical(result) if err is None else {},
                    notes=list(result.get("notes", [])) if isinstance(result, dict) else [],
                    error=err,
                )
                _STORE.put(t)

            out = dict(result)
            out["transcript_id"] = tid
            out["engine"] = t.engine_string
            return out

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
        wrapper._tool_name = tool_name  # type: ignore[attr-defined]
        wrapper._data_args = data_args  # type: ignore[attr-defined]
        return wrapper

    return deco


def _render_call(tool: str, kwargs: Dict[str, Any], data_args: tuple) -> str:
    bits = []
    for k, v in kwargs.items():
        if k in data_args:
            arr = np.asarray(v, dtype=object)
            bits.append(f"{k}=<data n={arr.size if arr.ndim else 1}>")
        elif isinstance(v, str):
            bits.append(f"{k}={v!r}")
        elif isinstance(v, (int, float, bool)) or v is None:
            bits.append(f"{k}={v}")
        else:
            bits.append(f"{k}=<{type(v).__name__}>")
    return f"{tool}({', '.join(bits)})"
