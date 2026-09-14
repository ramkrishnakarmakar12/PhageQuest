"""The assistant: a chatbot that cannot invent a number (spec section 06).

The model's job is to choose a tool, explain a result, and ask the next
question. It never computes, and everything it says passes through the guards
in `guards.py` before a user sees it.

The turn pipeline, in order:

    user message
      -> PII redaction (DPDP, spec section 09)
      -> specification-search check on the phrasing
      -> pre-registration gate
      -> model proposes ONE tool call from the typed registry
      -> the engine executes it; a transcript is created
      -> model writes prose about the result
      -> claim guard        (UMAP distances, causation, accepting the null)
      -> numeric scan       (every number must trace to a transcript)
      -> behavioural specification-search check
      -> render, with the transcript attached

`LLMProvider` is an interface. `MockProvider` implements it deterministically,
so the whole harness -- including every guard -- runs and is tested without an
API key, in CI, and in a school with no internet.

Orchestration note (spec section 06): the spec picks DSPy for the reasoning
layer and MCP for the tool boundary, and warns against stacking LangChain and
LlamaIndex and MCP together. This module is the boundary itself: typed
signatures in, validated structures out, one orchestrator. A DSPy or MCP
adapter wraps it; neither is imported here, so the guarantees do not depend on
either staying maintained.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Protocol, Sequence

from . import guards, registry
from .transcript import TranscriptStore, get_store, set_store

__all__ = ["LLMProvider", "MockProvider", "Assistant", "Turn", "SYSTEM_PROMPT"]


SYSTEM_PROMPT = """\
You are the PhageQuest analysis assistant, working with a student in Grade {grade}.

HARD RULES -- these are enforced after you speak, so breaking them wastes the turn:

1. You never compute. Not arithmetic, not a p-value, not a mean, not a
   percentage. Every number you write must have come back from a tool call in
   this turn. A number you produce yourself is stripped before the student sees
   it, and the turn is regenerated.
2. You choose the test; the engine runs it. When you call an inferential tool
   you must supply `justification`: one sentence about the DESIGN -- why this
   test, on this data. It is logged and shown to the teacher.
3. Withhold the answer by default. Ask the probing question first. The evidence
   says structured practice, not conversational answer-giving, is what produces
   learning; a student who predicts before testing learns more than one who is
   told. Require a stated prediction before you run a test.
4. Never say a plot's distances mean something it cannot support, never turn a
   correlation into a cause, and never report a non-significant result as proof
   that there is no difference.
5. Below three observations per group, there is no test to run. Say so and
   describe the data instead. Below 100 samples there is no machine learning to
   do; steer back to description and inference, where the answer actually is.
6. When a stability check says the conclusion flips, lead with that, not with
   the p-value.

TONE: plain words, short sentences, no statistical jargon unless you define it
in the same breath. The student is {grade_words}. A teacher who is not a
statistician reads everything you write.

Available tools this grade may use: {tool_names}
"""

_GRADE_WORDS = {
    5: "ten or eleven and has met powers of ten but not averages",
    6: "eleven or twelve and is meeting the idea of variability for the first time",
    7: "twelve or thirteen and can understand shuffling but not a t-distribution",
    8: "thirteen or fourteen and is ready for rates, curves and quantified uncertainty",
    9: "fourteen or fifteen and can handle a hypothesis, a test and an interval",
    10: "fifteen or sixteen and is working toward a capstone",
    11: "sixteen or seventeen and is doing genuine research",
    12: "seventeen or eighteen and is doing genuine research",
}


class LLMProvider(Protocol):
    """Provider interface. Implement `complete` and the whole harness works."""

    name: str

    def complete(self, system: str, messages: List[Dict[str, str]],
                 tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return {"text": str, "tool_call": {"name": str, "arguments": dict} | None}."""
        ...


class MockProvider:
    """A deterministic, offline provider.

    It is not an imitation of a language model; it is a rule-based planner good
    enough to drive every path through the harness. That matters more than it
    sounds: it means the honesty guards are tested against real tool results in
    CI, on a school laptop with no internet, and in a country where sending a
    child's data to a foreign API is a compliance question (spec section 09).

    It also has a `misbehave` mode that fabricates statistics on purpose, so the
    numeric scan is tested against the failure it exists to catch.
    """

    name = "mock"

    def __init__(self, misbehave: bool = False) -> None:
        self.misbehave = misbehave

    def complete(self, system: str, messages: List[Dict[str, str]],
                 tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        last = messages[-1]["content"] if messages else ""
        tool_results = [m for m in messages if m.get("role") == "tool"]
        available = {t["name"] for t in tools}

        if tool_results:
            return {"text": self._explain(tool_results[-1], last), "tool_call": None}

        call = self._plan(last, available)
        if call is None:
            return {"text": self._ask_back(last), "tool_call": None}
        return {"text": "", "tool_call": call}

    # -- planning ----------------------------------------------------------
    def _plan(self, text: str, available: set) -> Optional[Dict[str, Any]]:
        t = text.lower()
        data = self._extract_groups(text)

        if re.search(r"how many (samples|replicates|tubes|do we need)|sample size|how big", t):
            if "n_for_ttest" in available:
                return {"name": "n_for_ttest",
                        "arguments": {"effect_size": 0.8, "power": 0.8}}
        if data and len(data) >= 2:
            if re.search(r"differ|bigger|better|significant|compare|which.*most|effect", t):
                if "compare_groups" in available:
                    return {"name": "compare_groups",
                            "arguments": {"groups": data, "design": "independent",
                                          "justification": "Comparing independent groups measured "
                                                           "under different conditions."}}
                if "shuffle_test" in available and len(data) == 2:
                    keys = list(data)
                    return {"name": "shuffle_test",
                            "arguments": {"group_a": data[keys[0]], "group_b": data[keys[1]],
                                          "justification": "Two small groups; shuffling assumes "
                                                           "nothing about the data's shape."}}
            if "describe" in available:
                return {"name": "describe", "arguments": {"groups": data}}
        if re.search(r"\bmoi\b|multiplicity", t) and "moi" in available:
            nums = self._numbers(text)
            if len(nums) >= 2:
                return {"name": "moi", "arguments": {"phage_pfu_per_mL": nums[0],
                                                     "bacteria_cfu_per_mL": nums[1]}}
        if re.search(r"pfu|titre|titer|plaque count", t) and "pfu_per_ml" in available:
            nums = self._numbers(text)
            if len(nums) >= 2:
                return {"name": "pfu_per_ml",
                        "arguments": {"plaque_count": int(nums[0]), "dilution_factor": nums[1]}}
        if re.search(r"simulat|infect.*culture|what happens if", t) and "simulate" in available:
            model = ("resistance" if "resist" in t else
                     "lysogenic" if ("lysogen" in t or "hide" in t) else "lytic")
            return {"name": "simulate", "arguments": {"model": model, "hours": 48}}
        if re.search(r"no plaques|empty plate|nothing grew|sometimes", t) and \
                "poisson_plaques" in available:
            return {"name": "poisson_plaques", "arguments": {"expected_plaques": 3.0}}
        return None

    def _extract_groups(self, text: str) -> Dict[str, List[float]]:
        """Pull 'label: 1, 2, 3' groups out of a message."""
        out: Dict[str, List[float]] = {}
        for m in re.finditer(r"([A-Za-z][\w .%-]{0,24}?)\s*[:=]\s*"
                             r"((?:[-+]?\d*\.?\d+\s*[, ]\s*){2,}[-+]?\d*\.?\d+)", text):
            label = re.sub(r"^(?:and|or|the|a|an|with|vs\.?|versus)\s+", "", m.group(1).strip(),
                           flags=re.IGNORECASE).strip()
            vals = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", m.group(2))]
            if len(vals) >= 2:
                out[label] = vals
        return out

    def _numbers(self, text: str) -> List[float]:
        return [float(x.replace(",", ""))
                for x in re.findall(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?", text)]

    # -- explaining --------------------------------------------------------
    def _explain(self, tool_msg: Dict[str, str], question: str) -> str:
        try:
            r = json.loads(tool_msg["content"])
        except Exception:
            return "I could not read that result."

        if self.misbehave:
            # Deliberately fabricated: the numeric scan must catch every one.
            return ("The difference is clearly significant, t(28) = 2.41, p = .023, with a large "
                    "effect size of d = 0.91 and a 47% improvement over the control.")

        if r.get("refused"):
            return f"I did not run that. {r.get('reason', '')}"
        if r.get("error"):
            return f"That did not work: {r.get('message', '')}"

        parts: List[str] = []
        stab = r.get("stability") or {}
        if stab.get("severity") in ("warn", "block"):
            parts.append(stab.get("verdict", ""))
        if r.get("plain_language"):
            parts.append(r["plain_language"])
        if r.get("why_this_test"):
            parts.append(f"Why this test: {r['why_this_test']}")
        led = r.get("ledger") or {}
        if led.get("severity") == "warn":
            parts.append(led.get("message", ""))
        for n in (r.get("notes") or [])[:2]:
            parts.append(n)
        if r.get("teaches"):
            parts.append(r["teaches"])
        parts.append(self._next_question(r))
        return " ".join(p for p in parts if p)

    def _next_question(self, r: Dict[str, Any]) -> str:
        if r.get("tool") == "compare_groups":
            return ("What would you expect to happen if you ran the whole experiment again "
                    "tomorrow -- the same result, or something different?")
        if r.get("tool") in ("simulate", "parameter_sweep"):
            return "Which single parameter do you think the outcome depends on most, and why?"
        if r.get("tool") == "describe":
            return ("Before we test anything: which group do you think is different, and what "
                    "would convince you that you are wrong?")
        return "What is the next thing you would measure to check this?"

    def _ask_back(self, text: str) -> str:
        return ("I need the numbers before I can do anything with them. Paste your table, or "
                "write it like this: control: 12, 14, 11, 13 and treatment: 18, 21, 19, 20. "
                "And tell me what you expected to happen before you looked.")


@dataclass
class Turn:
    user_message: str
    reply: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    transcripts: List[str] = field(default_factory=list)
    blocked: bool = False
    block_reasons: List[str] = field(default_factory=list)
    guard_reports: Dict[str, Any] = field(default_factory=dict)
    redactions: List[str] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Assistant:
    """The turn loop, with every guard wired in."""

    def __init__(self, provider: Optional[LLMProvider] = None, grade: int = 8,
                 store: Optional[TranscriptStore] = None,
                 mode: str = "block", student_names: Optional[Sequence[str]] = None,
                 require_prediction: bool = True) -> None:
        self.provider = provider or MockProvider()
        self.grade = int(grade)
        self.store = store or get_store()
        self.mode = mode
        self.student_names = list(student_names or [])
        self.require_prediction = require_prediction
        self.prereg = guards.PreRegistrationLog()
        self.detector = guards.SpecificationSearchDetector(self.store)
        self.history: List[Dict[str, str]] = []
        self.turns: List[Turn] = []
        self._active_dataset: Optional[str] = None

    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        names = ", ".join(registry.tools_for_grade(self.grade))
        return SYSTEM_PROMPT.format(
            grade=self.grade,
            grade_words=_GRADE_WORDS.get(self.grade, "a secondary student"),
            tool_names=names)

    def register_prediction(self, dataset_id: str, question: str, prediction: str,
                            direction: str = "two-sided", **kw) -> Dict[str, Any]:
        pre = guards.PreRegistration(dataset_id=dataset_id, question=question,
                                     prediction=prediction, direction=direction, **kw)
        self._active_dataset = dataset_id
        return self.prereg.register(pre)

    # ------------------------------------------------------------------
    def ask(self, message: str, dataset_id: Optional[str] = None) -> Turn:
        turn = Turn(user_message=message, reply="")
        ds = dataset_id or self._active_dataset

        # --- 1. PII redaction before anything leaves the device -----------
        red = guards.redact_pii(message, self.student_names)
        turn.redactions = red["redacted"]
        safe_message = red["text"]

        # --- 2. Specification search, by phrasing -------------------------
        spec = self.detector.check_message(safe_message)
        turn.guard_reports["specification_search_phrasing"] = spec
        if spec.get("detected"):
            turn.reply = spec["response"]
            turn.blocked = True
            turn.block_reasons.append("specification_search")
            self.turns.append(turn)
            return turn

        # --- 3. Pre-registration gate -------------------------------------
        wants_test = bool(re.search(
            r"\b(test|significant|differ|compare|bigger|better|prove|effect)\b",
            safe_message, re.IGNORECASE))
        if self.require_prediction and wants_test and ds:
            gate = self.prereg.gate(ds)
            turn.guard_reports["preregistration"] = gate
            if not gate["allowed"]:
                turn.reply = gate["reason"]
                turn.blocked = True
                turn.block_reasons.append("no_prediction")
                self.turns.append(turn)
                return turn

        # --- 4. Model proposes a tool call --------------------------------
        self.history.append({"role": "user", "content": safe_message})
        tools = registry.tool_schemas(grade=self.grade)
        mark = self.store.marker()

        step = self.provider.complete(self.system_prompt(), self.history, tools)
        if step.get("tool_call"):
            call = step["tool_call"]
            result = registry.call_tool(call["name"], call.get("arguments", {}),
                                        grade=self.grade)
            turn.tool_calls.append({"name": call["name"],
                                    "arguments": call.get("arguments", {}),
                                    "transcript_id": result.get("transcript_id"),
                                    "error": result.get("error")})
            if result.get("transcript_id"):
                turn.transcripts.append(result["transcript_id"])
            self.history.append({"role": "assistant",
                                 "content": f"[called {call['name']}]"})
            self.history.append({"role": "tool",
                                 "content": json.dumps(result, default=str)})
            step = self.provider.complete(self.system_prompt(), self.history, tools)

        text = step.get("text", "")

        # --- 5. Claim guard -----------------------------------------------
        claims = guards.claim_guard(text)
        turn.guard_reports["claims"] = claims
        if not claims["passed"]:
            turn.blocked = True
            turn.block_reasons.append("unsupported_claim")
            text = "\n\n".join(v["message"] for v in claims["violations"])

        # --- 6. Numeric scan ----------------------------------------------
        # Scoped to THIS turn's transcripts: a number from a computation three
        # questions ago is not evidence for a sentence written now.
        turn_numbers: List[float] = []
        scoped = TranscriptStore()
        for t in self.store.since(mark):
            scoped.put(t)
        turn_numbers = scoped.numbers()

        scan = guards.numeric_scan(text, allowed=turn_numbers, mode=self.mode)
        turn.guard_reports["numeric_scan"] = {
            "passed": scan.passed, "message": scan.message,
            "unmatched": scan.unmatched, "n_checked": len(scan.numbers_found),
        }
        if not scan.passed:
            turn.blocked = True
            turn.block_reasons.append("unverified_number")
            if self.mode == "redact":
                text = scan.redacted_text
            else:
                text = (
                    "I was about to give you numbers I had not actually computed, and the "
                    "platform stopped me -- which is exactly what it is there for. "
                    + scan.message
                    + " Ask me again and I will run the calculation properly first.")

        # --- 7. Specification search, by behaviour ------------------------
        behaviour = self.detector.check_behaviour(ds or self._current_data_id(turn))
        turn.guard_reports["specification_search_behaviour"] = behaviour
        if behaviour.get("detected"):
            text += "\n\n" + behaviour["response"]

        # --- 8. Pre-registration judgement --------------------------------
        if ds and turn.transcripts:
            last = self.store.get(turn.transcripts[-1])
            if last and last.result.get("p_value") is not None:
                judged = self.prereg.judge(ds, last.result)
                turn.guard_reports["prediction_judgement"] = judged
                if judged.get("judged"):
                    text += "\n\n" + judged["message"]

        turn.reply = text.strip()
        self.history.append({"role": "assistant", "content": turn.reply})
        self.turns.append(turn)
        return turn

    def _current_data_id(self, turn: Turn) -> Optional[str]:
        for tid in reversed(turn.transcripts):
            t = self.store.get(tid)
            if t and t.data_id:
                return t.data_id
        return None

    # ------------------------------------------------------------------
    def transcript_panel(self, turn: Turn) -> List[Dict[str, Any]]:
        """What the UI shows alongside the reply. The transcript IS the product."""
        out = []
        for tid in turn.transcripts:
            t = self.store.get(tid)
            if t:
                out.append({
                    "transcript_id": t.transcript_id, "tool": t.tool, "call": t.call,
                    "engine": t.engine_string, "duration_ms": t.duration_ms,
                    "data_id": t.data_id, "params": t.params,
                    "result": t.result, "at": t.started_at,
                })
        return out
