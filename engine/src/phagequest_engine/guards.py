"""The honesty machinery (spec section 06).

    "An LLM asked 'is this significant?' will happily produce a fluent,
    correctly formatted, entirely fabricated t(28) = 2.41, p = .023, d = 0.91.
    It will look right. It will go into a student's report. And unlike a
    hallucinated fact, a hallucinated p-value cannot be caught by a 15-year-old
    or, usually, by their teacher."

Six enforcement mechanisms, all in this module, all testable:

  1. numeric_scan          -- every number in model prose must trace to a transcript
  2. PreRegistration       -- a prediction must be logged before a test runs
  3. SpecificationSearch   -- the "try dropping that outlier" loop is detected
  4. claim_guard           -- sentences the method cannot support are blocked
  5. sample_size_guard     -- ML below a threshold is refused, not caveated
  6. NameCollisionGuard    -- the package-name traps, as code rather than prose

Note what is NOT here: asking the model to be careful. Every one of these runs
after the model has spoken, on its output, deterministically.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from .transcript import TranscriptStore, get_store

__all__ = [
    "numeric_scan", "ScanResult", "PreRegistration", "PreRegistrationLog",
    "SpecificationSearchDetector", "claim_guard", "sample_size_guard",
    "NAME_COLLISIONS", "check_package_names", "redact_pii",
]


# ==========================================================================
# 1. Post-generation numeric scan
# ==========================================================================

# Ordered: the most specific patterns first, so "p = .023" is captured as a
# statistic rather than as a bare decimal.
_STAT_PATTERNS = [
    (r"\bp\s*[=<>]\s*\.?\d*\.?\d+(?:[eE][-+]?\d+)?", "p_value"),
    (r"\b[tFrzFHUW]\s*\(\s*[\d.,\s]+\)\s*=\s*[-+]?\d*\.?\d+", "test_statistic"),
    (r"\b(?:chi2|chi-squared|χ²|χ\s*2)\s*(?:\([^)]*\))?\s*=\s*[-+]?\d*\.?\d+", "chi_square"),
    (r"\b(?:d|g|eta2|η²|r|rho|ρ|tau|τ|V)\s*=\s*[-+]?\d*\.?\d+", "effect_size"),
    (r"\b(?:CI|confidence interval)\s*[:=]?\s*\[?\s*[-+]?\d*\.?\d+\s*(?:,|to|-)\s*[-+]?\d*\.?\d+", "interval"),
    (r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*%", "percentage"),
    (r"[-+]?\d+(?:,\d{3})*\.\d+(?:[eE][-+]?\d+)?", "decimal"),
    (r"\b\d+(?:,\d{3})*(?:[eE][-+]?\d+)\b", "scientific"),
    (r"\b\d{2,}(?:,\d{3})*\b", "integer"),
]

# Numbers that are never claims about data and must not trip the scanner.
_SAFE_CONTEXT = re.compile(
    r"(?:grade|class|year|step|stage|tube|plate|day|week|page|figure|table|section|"
    r"question|part|version|python|scipy|numpy|k\s*=|n\s*=|alpha\s*=|level|row)\s*:?\s*$",
    re.IGNORECASE)


@dataclass
class ScanResult:
    passed: bool
    numbers_found: List[Dict[str, Any]]
    unmatched: List[Dict[str, Any]]
    matched: List[Dict[str, Any]]
    redacted_text: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_number(token: str) -> Optional[float]:
    m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", token.replace(",", ""))
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def _all_numbers_in(token: str) -> List[float]:
    out = []
    for m in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", token.replace(",", "")):
        try:
            out.append(float(m))
        except ValueError:
            continue
    return out


def numeric_scan(text: str, store: Optional[TranscriptStore] = None,
                 allowed: Optional[Iterable[float]] = None,
                 tolerance: float = 1e-6,
                 ignore_below: float = 10.0,
                 mode: str = "block") -> ScanResult:
    """Cross-check every number in model prose against the turn's tool results.

    Spec section 06, point 2:

        "Parse the model's prose for numeric patterns -- p =, p <, t(, F(, chi2,
        bare decimals, percentages -- and cross-check each against the turn's
        tool-result store. Unmatched number -> block and regenerate."

    `ignore_below` exempts small bare integers ("the four salt levels", "three
    replicates") which are prose, not claims. Statistical patterns -- anything
    matching `p =`, `t(...)`, an effect size, an interval or a percentage -- are
    NEVER exempted regardless of magnitude, because those are exactly the
    fabrications that matter.

    `mode="block"` fails the whole turn. `mode="redact"` replaces unmatched
    numbers with [UNVERIFIED] so the sentence survives and the number does not.
    """
    st = store or get_store()
    pool = list(allowed) if allowed is not None else st.numbers()
    pool_arr = np.asarray([p for p in pool if np.isfinite(p)], dtype=float)

    def known(v: float, written: str) -> bool:
        """Is `v` a faithful rendering of some number the engine actually produced?

        The tolerance is derived from HOW THE NUMBER WAS WRITTEN, not from a
        fixed epsilon. "p = .023" was written to three decimals, so it matches a
        stored value only within 0.0005 -- the rounding it claims. An earlier
        version of this function allowed a generous tolerance at every decimal
        place, which made 0.023 "match" a stored 0.0, and a fabricated p-value
        sailed straight through. That bug is the exact failure this whole module
        exists to prevent, so the rule is now strict: a written number must be
        the correct rounding of a real one.
        """
        if pool_arr.size == 0:
            return False

        m = re.search(r"(\d*)\.(\d+)", written.replace(",", ""))
        if m:
            dp = len(m.group(2))
        elif re.search(r"[eE][-+]?\d+", written):
            dp = 6
        else:
            dp = 0
        half = 0.5 * (10.0 ** -dp)

        # A written number may be the stored value itself, or its percentage
        # form -- a model writing "37%" from a stored 0.3679 is reporting, not
        # inventing. Both are checked at the written precision.
        for cand, scale in ((v, 1.0), (v / 100.0, 0.01)):
            if np.any(np.abs(pool_arr - cand) <= half * scale + tolerance):
                return True
        return False

    found: List[Dict[str, Any]] = []
    claimed_spans: List[Tuple[int, int]] = []

    for pattern, kind in _STAT_PATTERNS:
        for m in re.finditer(pattern, text):
            span = m.span()
            if any(s <= span[0] < e or s < span[1] <= e for s, e in claimed_spans):
                continue
            claimed_spans.append(span)
            statistical = kind not in ("integer", "decimal", "scientific", "percentage")
            for v in _all_numbers_in(m.group()):
                found.append({"token": m.group().strip(), "kind": kind, "value": v,
                              "start": span[0], "end": span[1],
                              "statistical": statistical})

    matched, unmatched = [], []
    for f in found:
        v = f["value"]
        exempt = (not f["statistical"]
                  and f["kind"] == "integer"
                  and abs(v) < ignore_below)
        if not exempt and f["kind"] in ("integer", "decimal"):
            prefix = text[max(0, f["start"] - 30):f["start"]]
            if _SAFE_CONTEXT.search(prefix):
                exempt = True
        if exempt:
            f["exempt"] = True
            matched.append(f)
        elif known(v, f["token"]):
            f["exempt"] = False
            matched.append(f)
        else:
            f["exempt"] = False
            unmatched.append(f)

    redacted = text
    for f in sorted(unmatched, key=lambda x: -x["start"]):
        redacted = redacted[:f["start"]] + "[UNVERIFIED]" + redacted[f["end"]:]

    passed = not unmatched
    if passed:
        msg = (f"All {len(matched)} number(s) in this response trace to a computed result."
               if matched else "No numeric claims in this response.")
    else:
        worst = [f for f in unmatched if f["statistical"]]
        # Mask the digits. Repeating a fabricated statistic back to the user --
        # even inside a warning -- puts it on screen where it can be copied, and
        # a number in a warning still reads as a number.
        def _mask(tok: str) -> str:
            return re.sub(r"\d", "#", tok)

        msg = (f"{len(unmatched)} number(s) could not be traced to any computed result: "
               + ", ".join(sorted({_mask(f['token']) for f in unmatched})[:6])
               + (". At least one is a statistical claim, which is the highest-severity case: "
                  "a fabricated p-value or effect size cannot be caught by a student or their "
                  "teacher." if worst else ".")
               + (" The response was blocked." if mode == "block" else
                  " The numbers were replaced with [UNVERIFIED]."))

    return ScanResult(passed=passed, numbers_found=found, unmatched=unmatched,
                      matched=matched, redacted_text=redacted, message=msg)


# ==========================================================================
# 2. Pre-registration (spec section 06, point 3)
# ==========================================================================

@dataclass
class PreRegistration:
    """A prediction, recorded before the data is looked at.

        "G7-L9 and G8-L9 both make students write a prediction before observing;
        G9-L9 makes them write a claim before evidence. The platform simply
        records it and holds them to it. This turns the p-hacking failure mode
        into the lesson."
    """
    dataset_id: str
    question: str
    prediction: str
    direction: str = "two-sided"      # greater | less | two-sided
    primary_outcome: str = ""
    groups: List[str] = field(default_factory=list)
    planned_test: str = ""
    planned_n: Optional[int] = None
    registered_at: float = field(default_factory=time.time)
    author: str = ""
    locked: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PreRegistrationLog:
    """Holds pre-registrations and judges results against them.

    The judgement is deliberately gentle about being WRONG and strict about
    changing the prediction afterwards. A wrong prediction honestly reported is
    a good piece of science; a prediction quietly rewritten to match the data is
    the thing the platform exists to prevent.
    """

    def __init__(self) -> None:
        self._by_dataset: Dict[str, List[PreRegistration]] = {}

    def register(self, prereg: PreRegistration) -> Dict[str, Any]:
        existing = self._by_dataset.setdefault(prereg.dataset_id, [])
        amended = bool(existing)
        existing.append(prereg)
        return {
            "registered": True, "amendment": amended, "version": len(existing),
            "message": ("Prediction recorded. It is now locked -- you can add a new one, but the "
                        "original stays in the record and your teacher sees both."
                        if not amended else
                        f"Recorded as amendment #{len(existing)}. The original prediction is still "
                        f"in the record and will be shown alongside this one. Amending after "
                        f"seeing data is not forbidden here, but it is always visible."),
        }

    def current(self, dataset_id: str) -> Optional[PreRegistration]:
        lst = self._by_dataset.get(dataset_id)
        return lst[0] if lst else None

    def all_for(self, dataset_id: str) -> List[PreRegistration]:
        return list(self._by_dataset.get(dataset_id, []))

    def gate(self, dataset_id: str) -> Dict[str, Any]:
        """Must a prediction be written before a test may run? Yes, once.

        The assistant is told to withhold the test and ask the probing question
        instead (spec section 06, pedagogy note). This is that rule as code.
        """
        if self.current(dataset_id) is None:
            return {
                "allowed": False,
                "reason": ("Before running a test on this data, write down what you expect to "
                           "happen and why. Not because it is a formality -- because a prediction "
                           "made after seeing the answer is not a prediction, and the difference "
                           "between those two is most of what makes something science."),
                "needs": ["question", "prediction", "direction"],
            }
        return {"allowed": True, "prereg": self.current(dataset_id).to_dict()}

    def judge(self, dataset_id: str, result: Dict[str, Any],
              alpha: float = 0.05) -> Dict[str, Any]:
        pre = self.current(dataset_id)
        if pre is None:
            return {"judged": False, "message": "No prediction was registered for this dataset."}
        p = result.get("p_value")
        es = result.get("effect_size")
        amendments = len(self.all_for(dataset_id)) - 1

        if p is None:
            outcome = "not a test"
            verdict = "This result is descriptive, so there is nothing to judge against a prediction yet."
        else:
            significant = p < alpha
            direction_ok = True
            if pre.direction in ("greater", "less") and es is not None and np.isfinite(es):
                direction_ok = (es > 0) if pre.direction == "greater" else (es < 0)
            if significant and direction_ok:
                outcome = "supported"
                verdict = ("The data went the way you predicted. Note that this is the easiest case "
                           "to over-read: say how big the effect was and how wide the interval, not "
                           "just that you were right.")
            elif significant and not direction_ok:
                outcome = "reversed"
                verdict = ("The effect is real but it went the OPPOSITE way to your prediction. "
                           "This is the most interesting result you can get. Do not rewrite the "
                           "prediction -- explain the surprise.")
            else:
                outcome = "not supported"
                verdict = ("Your prediction was not borne out. That is a result, and reporting it "
                           "is worth more than a significant finding you went looking for. Check "
                           "the power analysis: was this experiment big enough to have detected "
                           "the effect you expected?")

        return {
            "judged": True, "outcome": outcome,
            "prediction": pre.prediction, "direction": pre.direction,
            "registered_at": pre.registered_at,
            "n_amendments": amendments,
            "amendment_warning": (
                f"This prediction was amended {amendments} time(s) after registration. Both "
                f"versions are in the record." if amendments else None),
            "message": verdict,
        }


# ==========================================================================
# 3. Specification-search detection (spec section 06, point 6)
# ==========================================================================

_SPEC_SEARCH_PHRASES = [
    r"\btry\s+(?:a\s+few\s+)?(?:different|other|another)\b",
    r"\b(?:drop|remove|exclude|delete|ignore|leave out|take out)\s+(?:that|the|this|those|an?)?\s*"
    r"(?:outlier|point|value|row|reading|sample|measurement|tube|replicate)",
    r"\bwhat if we\s+(?:exclude|drop|remove|ignore|leave out)",
    r"\buntil it\s+(?:works|is significant|becomes significant)",
    r"\b(?:without|minus)\s+(?:the\s+)?(?:outlier|weird|odd|bad)\s*(?:one|point|value)?",
    r"\bregroup|\bregroup(?:ed|ing)?\b|\bdifferent\s+group(?:ing|s)\b",
    r"\bmake it significant\b|\bget(?:ting)? it (?:below|under) ?\.?0?5\b",
    r"\bp[- ]?hack",
    r"\breport(?:ing)? uncertainty across specifications\b",
    r"\brobustness across (?:all )?(?:model )?specifications\b",
]
_SPEC_SEARCH_RE = [re.compile(p, re.IGNORECASE) for p in _SPEC_SEARCH_PHRASES]


class SpecificationSearchDetector:
    """Detects the "keep trying until it works" loop, by phrasing AND by behaviour.

    The spec's finding is that framing, not intent, is what slipped past frontier
    models' guardrails:

        "under a reframing that recast specification search as 'reporting
        uncertainty across specifications,' both bypassed their guardrails
        entirely... The guardrails were sensitive to framing, not to intent."

    Which is why phrase-matching alone is not enough and is not what this class
    relies on. The behavioural signal -- repeated tests on the same dataset with
    shrinking n -- is framing-independent, and it is the one that fires on the
    polite version of the request.
    """

    def __init__(self, store: Optional[TranscriptStore] = None) -> None:
        self.store = store or get_store()
        self._exclusion_log: List[Dict[str, Any]] = []

    def check_message(self, text: str) -> Dict[str, Any]:
        hits = [p.pattern for p in _SPEC_SEARCH_RE if p.search(text)]
        if not hits:
            return {"detected": False}
        return {
            "detected": True, "patterns": hits, "signal": "phrasing",
            "response": (
                "Excluding a data point changes your answer, so the reason has to be recorded "
                "before it happens -- and it has to be a reason about the MEASUREMENT, not about "
                "the result. 'The tube cracked', 'the probe was not in the liquid', 'we read it a "
                "day late' are all fine. 'It is making the p-value too big' is not a reason, it is "
                "the thing we are trying to avoid. Which was it?"),
            "requires": ["written_justification"],
        }

    def check_behaviour(self, data_id: Optional[str], current_n: Optional[int] = None,
                        threshold: int = 3) -> Dict[str, Any]:
        """The framing-independent signal: many tests on one dataset, n shrinking."""
        tests = self.store.inferential_on(data_id)
        n = len(tests)
        if n < threshold:
            return {"detected": False, "tests_on_this_data": n}

        ns: List[int] = []
        for t in tests:
            per = t.result.get("n_per_group")
            if isinstance(per, dict) and per:
                vals = [v for v in per.values() if isinstance(v, (int, float))]
                if vals:
                    ns.append(int(sum(vals)))
            elif isinstance(t.result.get("n"), (int, float)):
                ns.append(int(t.result["n"]))
        shrinking = bool(len(ns) >= 2 and ns[-1] < ns[0])
        best_p = min((t.result.get("p_value", 1.0) for t in tests), default=1.0)
        first_p = tests[0].result.get("p_value")

        severity = "warn"
        if shrinking and n >= threshold + 1:
            severity = "block"
        elif n >= 6:
            severity = "block"

        return {
            "detected": True, "signal": "behaviour", "severity": severity,
            "tests_on_this_data": n,
            "sample_sizes": ns,
            "sample_size_shrinking": shrinking,
            "first_p": first_p, "best_p": best_p,
            "response": (
                f"{n} tests have now been run on this dataset"
                + (f", and the sample size has fallen from {ns[0]} to {ns[-1]} along the way"
                   if shrinking else "")
                + f". The best p-value found is {best_p:.4g}"
                + (f", against {first_p:.4g} on the first test." if first_p is not None else ".")
                + " Searching for the analysis that gives the answer you want is the single "
                  "commonest way honest people produce false results, and it does not feel "
                  "dishonest while you are doing it. The multiple-comparison ledger now applies "
                  "to every result from this dataset, and the count is shown to your teacher."),
        }

    def log_exclusion(self, data_id: str, description: str, justification: str,
                      author: str = "") -> Dict[str, Any]:
        about_measurement = not bool(
            re.search(r"\b(p[- ]?value|significan|result|outcome|too (?:big|small|high|low)|"
                      r"doesn'?t work|not working|ruin|spoil)\b", justification, re.IGNORECASE))
        entry = {
            "data_id": data_id, "description": description, "justification": justification,
            "author": author, "at": time.time(),
            "about_the_measurement": about_measurement,
        }
        self._exclusion_log.append(entry)
        return {
            "logged": True, "entry": entry,
            "accepted": about_measurement,
            "message": ("Recorded. This exclusion and its reason appear in your report and in your "
                        "teacher's review queue, and the analysis will be shown both with and "
                        "without the excluded point."
                        if about_measurement else
                        "Recorded, but flagged: this reason is about the RESULT, not about the "
                        "measurement. That is the definition of the practice this platform exists "
                        "to make visible. The exclusion is logged, the analysis will be shown both "
                        "ways, and your teacher will see this note."),
        }

    @property
    def exclusions(self) -> List[Dict[str, Any]]:
        return list(self._exclusion_log)


# ==========================================================================
# 4. Claim guard (spec section 03.4, "two hard guardrails to encode")
# ==========================================================================

_BLOCKED_CLAIMS = [
    {
        "id": "umap_distance",
        "pattern": re.compile(
            r"(umap|t-?sne|pacmap)[^.]{0,120}?\b(far|close|distan|nearer|further|apart|"
            r"tightly|loosely)\b|\b(far|close|distan|apart)\b[^.]{0,120}?(umap|t-?sne|pacmap)",
            re.IGNORECASE),
        "message": (
            "UMAP and t-SNE distances are not distances. Those methods preserve which points are "
            "neighbours, not how far apart anything is: the gap between two clusters on a UMAP "
            "plot carries no information, and neither does the size of a cluster. Saying two "
            "groups are 'far apart, therefore different' from such a plot asserts something the "
            "method cannot support. Say which points are neighbours, or use a method whose "
            "distances mean something -- PCoA on a real distance matrix does."),
    },
    {
        "id": "causation",
        "pattern": re.compile(
            r"\b(?:correlat\w+|associat\w+|r\s*=\s*[-+]?\d*\.?\d+)[^.]{0,100}?\b"
            r"(?:therefore|so it|which means|proves?|causes?|caused|causing)\b",
            re.IGNORECASE),
        "message": (
            "A correlation cannot establish that one thing causes the other, however strong it is. "
            "What settles causation is the design: did you CHANGE the thing and hold everything "
            "else the same? If you did, say so and the claim is fine. If you measured both as you "
            "found them, it is not."),
    },
    {
        "id": "accepting_null",
        "pattern": re.compile(
            # NOT [^.] -- a p-value contains a decimal point, so excluding periods
            # meant this rule could never fire on the sentence it exists to catch.
            r"\b(?:p\s*[=>]\s*\.?\d|not significant|no significant)[^!?;]{0,80}?\b"
            r"(?:proves?|shows? that there is no|means there is no|no difference exists|"
            r"are (?:the same|identical|equal))\b",
            re.IGNORECASE),
        "message": (
            "A non-significant result does not show that there is no difference. It shows that "
            "this experiment did not find one, which is a different statement -- and with a class-"
            "sized sample it is usually the expected outcome even when a real difference exists. "
            "Report the interval: 'the difference is somewhere between -2 and +5' says what you "
            "actually know. If you want to claim the groups ARE the same, that needs an "
            "equivalence test and a threshold you set beforehand."),
    },
    {
        "id": "cluster_reification",
        "pattern": re.compile(
            r"\b(?:k-?means|cluster\w*)[^.]{0,100}?\b(?:proves?|confirms?|shows? that there are "
            r"(?:exactly )?\d+ (?:real )?(?:types|kinds|species|groups))\b", re.IGNORECASE),
        "message": (
            "Clustering algorithms return clusters whether or not any exist -- ask k-means for "
            "four groups and it will give you four, from random noise included. The number of "
            "clusters is something you chose. Before claiming the groups are real, show that they "
            "survive changing that choice (the k sweep in the geometry module does exactly this)."),
    },
]


def claim_guard(text: str) -> Dict[str, Any]:
    """Block sentences the method cannot support, at the model layer.

    Spec section 03.4: *"Block that sentence at the model layer."* Not document
    it, not warn about it in a system prompt -- block it after generation, where
    it can be tested.
    """
    hits = []
    for rule in _BLOCKED_CLAIMS:
        m = rule["pattern"].search(text)
        if m:
            hits.append({"id": rule["id"], "matched": m.group()[:160],
                         "message": rule["message"]})
    return {
        "passed": not hits, "violations": hits,
        "message": ("No unsupported claims detected."
                    if not hits else
                    f"{len(hits)} claim(s) go beyond what the method can support and were blocked."),
    }


# ==========================================================================
# 5. Sample-size guard (spec section 03.4, second guardrail)
# ==========================================================================

def sample_size_guard(n: int, task: str = "machine_learning",
                      n_features: Optional[int] = None) -> Dict[str, Any]:
    """Refuse machine learning on a class dataset.

        "AutoML on n=30 produces a beautiful, meaningless model. The assistant
        must refuse or heavily caveat machine learning below a sample-size
        threshold and steer back to descriptive and inferential statistics --
        where the answer actually is."

    A refusal, not a caveat, below the floor. A caveat is read as permission.
    """
    floors = {
        "machine_learning": 100, "supervised": 100, "clustering": 50,
        "dimensionality_reduction": 30, "umap": 50, "correlation": 8,
        "inference": 3, "descriptive": 1,
    }
    floor = floors.get(task, 100)
    ratio_problem = bool(n_features and n_features >= n)

    if n >= floor and not ratio_problem:
        return {"allowed": True, "n": n, "floor": floor, "task": task,
                "message": f"n = {n} is adequate for {task.replace('_', ' ')}."}

    if ratio_problem:
        reason = (f"You have {n_features} measurements per sample and only {n} samples. With more "
                  f"columns than rows, a model can fit your data perfectly and predict nothing -- "
                  f"there is always a line through 2 points, a plane through 3, and so on. This is "
                  f"not a tuning problem; there is no fix that does not involve more samples or "
                  f"fewer measurements.")
    else:
        reason = (f"n = {n} is below the floor of {floor} for {task.replace('_', ' ')}. A model "
                  f"trained on this many points will report a high accuracy and mean nothing by "
                  f"it: with {n} points, cross-validation is itself too noisy to tell you whether "
                  f"the model is real.")

    return {
        "allowed": False, "n": n, "floor": floor, "task": task,
        "n_features": n_features,
        "message": reason,
        "instead": [
            "Describe the data: group means with intervals, and a plot.",
            "Compare the groups you actually care about, with an effect size and a stability check.",
            "Work out how many samples you WOULD need (the power module), and say so in your "
            "report -- 'this question needs n=64 per group and we had 6' is a genuine finding "
            "about your experiment.",
        ],
        "steer_to": ["describe", "compare_groups", "n_for_ttest", "bootstrap_ci"],
    }


# ==========================================================================
# 6. The name-collision guard list (spec section 06)
# ==========================================================================

NAME_COLLISIONS: Dict[str, Dict[str, str]] = {
    "drc": {"installs": "A Django/React comment module from 2015",
            "correct": "No Python port exists. Use `lmfit`, or this engine's `fit_curve` with an "
                       "explicit four-parameter logistic (`model='four_pl'`)."},
    "jellyfish": {"installs": "A string-distance library",
                  "correct": "The k-mer counter is `bioconda::kmer-jellyfish`."},
    "wish": {"installs": "An unrelated library",
             "correct": "The phage host-prediction tool WIsH is source/conda only."},
    "basico": {"installs": "An unrelated 2019 package",
               "correct": "COPASI's Python API is `copasi-basico`."},
    "graph-tool": {"installs": "A stale 2.11 placeholder",
                   "correct": "conda/apt only, and LGPL-3 with a heavy Boost/C++ build. Skip it -- "
                              "`scikit-network` is BSD."},
    "community": {"installs": "An unrelated package of that exact name",
                  "correct": "Louvain lives in `python-louvain` (imported as `community`) -- or "
                             "better, `networkx.algorithms.community.louvain_communities`."},
    "ete3": {"installs": "End-of-life", "correct": "Use `ete4`."},
    "dspy-ai": {"installs": "Legacy alias", "correct": "Use `dspy`."},
    "python-igraph": {"installs": "Legacy alias",
                      "correct": "Use `igraph` -- though it is GPL-2; prefer `scikit-network`."},
    "tetranucleotide": {"installs": "Does not exist -- a classic hallucination site",
                        "correct": "Compute it: `Counter` over a sliding window, or "
                                   "`CountVectorizer(analyzer='char', ngram_range=(4,4))`. This "
                                   "engine's `geometry.tetranucleotide_vector` does it."},
    "tnf": {"installs": "Does not exist",
            "correct": "See `geometry.tetranucleotide_vector`."},
    "phage-dynamics": {"installs": "Does not exist",
                       "correct": "There is no maintained general-purpose phage-dynamics package. "
                                  "Use `simulation.simulate`, which is scipy's solve_ivp."},
}

# Licences that cannot sit in the platform core (spec section 09).
COPYLEFT = {
    "pingouin": "GPL-3", "leidenalg": "GPL-3", "amiga": "GPL-3",
    "igraph": "GPL-2", "python-igraph": "GPL-2", "phate": "GPL-2",
    "vegan": "GPL-2", "growthcurver": "GPL-2",
    "giotto-tda": "AGPL-3", "phyloseq": "AGPL-3", "jamovi": "AGPL-3", "jasp": "AGPL-3",
    "graph-tool": "LGPL-3",
}

_INSTALL_RE = re.compile(
    r"(?:pip|pip3|uv pip|conda|mamba|micromamba)\s+(?:install|add)\s+((?:[-\w.\[\]=<>!,]+\s*)+)",
    re.IGNORECASE)
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([\w.]+)", re.MULTILINE)


def check_package_names(text: str) -> Dict[str, Any]:
    """CI-testable guard over any installation or import instruction.

        "An LLM writing installation code will get these wrong. They belong in
        the system prompt as hard constraints and in CI as tests."

    This is the CI test. It runs over generated code, over chatbot output, and
    over the repository's own dependency files.
    """
    collisions, licences = [], []
    seen: Set[str] = set()

    for m in _INSTALL_RE.finditer(text):
        for tok in m.group(1).split():
            name = re.split(r"[=<>!\[]", tok)[0].strip().lower()
            if not name or name.startswith("-"):
                continue
            seen.add(name)
    for m in _IMPORT_RE.finditer(text):
        seen.add(m.group(1).split(".")[0].lower())

    for name in sorted(seen):
        if name in NAME_COLLISIONS:
            collisions.append({"package": name, **NAME_COLLISIONS[name]})
        if name in COPYLEFT:
            licences.append({
                "package": name, "licence": COPYLEFT[name],
                "consequence": (
                    "AGPL-3: network use triggers source disclosure to your users. Fatal for a "
                    "hosted commercial product."
                    if COPYLEFT[name] == "AGPL-3" else
                    f"{COPYLEFT[name]}: linking it into a proprietary core is a copyleft "
                    f"obligation. Run it in a separate sandboxed process behind a JSON boundary, "
                    f"or substitute."),
            })

    return {
        "passed": not collisions and not licences,
        "packages_seen": sorted(seen),
        "collisions": collisions,
        "licence_issues": licences,
        "message": (
            "No package-name collisions or licence problems."
            if not collisions and not licences else
            "; ".join([f"{c['package']} actually installs: {c['installs']}" for c in collisions]
                      + [f"{l['package']} is {l['licence']}" for l in licences])),
    }


# ==========================================================================
# 7. PII redaction before any cross-border LLM call (spec section 09)
# ==========================================================================

_PII_PATTERNS = [
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b"), "[PHONE]"),
    (re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"), "[ID]"),          # Aadhaar-shaped
    (re.compile(r"\b\d{1,3}\.\d{4,},\s*\d{1,3}\.\d{4,}\b"), "[COORDS]"),
]


def redact_pii(text: str, student_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Strip identifiers before a prompt leaves the country.

        "Student prompts to a third-party model API is a cross-border transfer
        of children's personal data to a sub-processor. Needs zero-retention
        terms, a DPA, PII redaction before the call..."

    This is the redaction half. Precise GPS coordinates are included because a
    sampling site (G8-L10, G9-L11) recorded to five decimal places locates a
    child's school or home to within a metre.
    """
    out = text
    found: List[str] = []
    for pattern, tag in _PII_PATTERNS:
        if pattern.search(out):
            found.append(tag)
            out = pattern.sub(tag, out)
    for name in (student_names or []):
        if name and len(name) > 2:
            pat = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
            if pat.search(out):
                found.append("[NAME]")
                out = pat.sub("[STUDENT]", out)
    return {
        "text": out, "redacted": sorted(set(found)), "changed": bool(found),
        "note": ("Site coordinates are truncated as well as names: a sampling point recorded to "
                 "five decimal places locates a child to within a metre, and under DPDP Section 9 "
                 "that is exactly the kind of data that must not leave the platform."),
    }
