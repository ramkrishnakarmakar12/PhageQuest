"""PhageQuest engine.

The statistical, simulation and geometry core of the PhageQuest Discovery
Platform. Built statistics-first, per the design rule in the specification:

    "Build the statistics layer first and the bioinformatics on top of it --
    not the other way round. Every competitor did it the other way round, which
    is exactly why they all have the same hole."

The platform invariant, enforced in `transcript.py` and `guards.py`:

    No numeric statistical claim may be rendered to a user unless it is bound
    to a specific execution record -- code, engine version, raw output,
    timestamp -- produced by a real statistics engine.

Licence posture (specification section 09): this package depends only on numpy
and scipy (BSD). No GPL (pingouin, igraph, leidenalg), no AGPL (giotto-tda).
Everything here runs unchanged in Pyodide, which is what makes Tier 0 -- about
90% of the Grades 5-12 workload, in the student's own browser, at zero marginal
cost and with no student data reaching our servers -- possible at all.
"""

__version__ = "0.1.0"

from . import (
    assumptions,
    curves,
    diversity,
    effects,
    geometry,
    guards,
    inference,
    ledger,
    power,
    registry,
    resampling,
    sequence,
    simulation,
    stability,
    transcript,
    validation,
)
from .assistant import Assistant, MockProvider
from .registry import TOOLS, call_tool, tool_schemas, tools_for_grade
from .transcript import Transcript, TranscriptStore, get_store, set_store

__all__ = [
    "__version__",
    # modules
    "assumptions", "curves", "diversity", "effects", "geometry", "guards",
    "inference", "ledger", "power", "registry", "resampling", "sequence",
    "simulation", "stability", "transcript", "validation",
    # the tool boundary
    "TOOLS", "call_tool", "tool_schemas", "tools_for_grade",
    # transcripts
    "Transcript", "TranscriptStore", "get_store", "set_store",
    # assistant
    "Assistant", "MockProvider",
]
