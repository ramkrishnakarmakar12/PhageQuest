"""Tests for the honesty machinery.

These are the most important tests in the repository. Everything else in the
platform is a convenience; this is the claim the specification says nobody else
can make, so it is the claim that has to be defended by tests rather than by
documentation.
"""

import numpy as np
import pytest

import phagequest_engine as pq
from phagequest_engine import guards, registry
from phagequest_engine.transcript import TranscriptStore, set_store


@pytest.fixture(autouse=True)
def clean_store():
    st = TranscriptStore()
    set_store(st)
    yield st


GROUPS = {
    "control": [12, 14, 11, 13, 12, 15],
    "treatment": [18, 21, 19, 20, 22, 19],
}


# ---------------------------------------------------------------- invariant
def test_every_result_carries_a_transcript():
    """The platform invariant. No transcript, no render."""
    r = pq.inference.compare_groups(GROUPS, justification="test")
    assert r["transcript_id"].startswith("tx_")
    assert "scipy" in r["engine"]


def test_transcript_records_engine_and_is_retrievable(clean_store):
    r = pq.inference.compare_groups(GROUPS, justification="test")
    t = clean_store.get(r["transcript_id"])
    assert t is not None
    assert t.tool == "compare_groups"
    assert t.data_id is not None
    assert "scipy" in t.engine
    assert t.result["p_value"] == r["p_value"]


def test_same_data_yields_same_data_id():
    a = pq.inference.compare_groups(GROUPS, justification="x")
    b = pq.inference.describe(GROUPS)
    st = pq.get_store()
    assert st.get(a["transcript_id"]).data_id == st.get(b["transcript_id"]).data_id


# ---------------------------------------------------------------- numeric scan
def test_numeric_scan_catches_fabricated_statistics(clean_store):
    """The headline failure mode from the specification, verbatim."""
    pq.inference.compare_groups(GROUPS, justification="test")
    fabricated = ("The difference is clearly significant, t(28) = 2.41, p = .023, "
                  "with a large effect size of d = 0.91.")
    scan = guards.numeric_scan(fabricated)
    assert not scan.passed
    tokens = {u["token"] for u in scan.unmatched}
    assert any("2.41" in t for t in tokens)
    assert any(".023" in t for t in tokens)
    assert any("0.91" in t for t in tokens)


def test_numeric_scan_accepts_real_numbers(clean_store):
    r = pq.inference.compare_groups(GROUPS, justification="test")
    honest = f"The p-value was p = {r['p_value']:.6g} and the effect size was {r['effect_size']:.3f}."
    scan = guards.numeric_scan(honest)
    assert scan.passed, scan.message


def test_numeric_scan_rejects_a_plausible_near_miss(clean_store):
    """A fabricated number close to a real one must still fail.

    This is the regression test for the bug that made the first version of the
    scan useless: a tolerance loose enough to forgive rounding also forgave
    invention.
    """
    r = pq.inference.compare_groups(GROUPS, justification="test")
    p = r["p_value"]
    wrong = p * 3 + 0.05
    scan = guards.numeric_scan(f"We found p = {wrong:.4f}.")
    assert not scan.passed


def test_numeric_scan_allows_correct_rounding(clean_store):
    r = pq.inference.compare_groups(GROUPS, justification="test")
    scan = guards.numeric_scan(f"p = {r['p_value']:.3f}")
    assert scan.passed, scan.message


def test_numeric_scan_ignores_small_prose_integers(clean_store):
    pq.inference.compare_groups(GROUPS, justification="test")
    scan = guards.numeric_scan("There were 4 salt levels and 3 replicates in each.")
    assert scan.passed, scan.message


def test_numeric_scan_redact_mode_preserves_the_sentence(clean_store):
    # Empty store: nothing has been computed, so every number is unverifiable.
    scan = guards.numeric_scan("The result was p = .0031 which is striking.", mode="redact")
    assert "[UNVERIFIED]" in scan.redacted_text
    assert "which is striking" in scan.redacted_text


def test_empty_store_rejects_every_statistic():
    scan = guards.numeric_scan("p = 0.04, d = 1.2")
    assert not scan.passed
    assert len(scan.unmatched) >= 2


# ---------------------------------------------------------------- claim guard
@pytest.mark.parametrize("text,rule", [
    ("The UMAP plot shows these two clusters are far apart, so they are different species.",
     "umap_distance"),
    ("There is a strong correlation (r = 0.82), therefore temperature causes the growth.",
     "causation"),
    ("p = 0.44, which proves there is no difference between the groups.",
     "accepting_null"),
    ("k-means confirms there are exactly 3 real types of phage here.",
     "cluster_reification"),
])
def test_claim_guard_blocks_unsupported_claims(text, rule):
    res = guards.claim_guard(text)
    assert not res["passed"]
    assert rule in {v["id"] for v in res["violations"]}


def test_claim_guard_allows_careful_language():
    ok = ("The two groups sit in different regions of the UMAP plot, so they are not each "
          "other's nearest neighbours. We changed the temperature and held everything else "
          "the same, so the difference is attributable to temperature.")
    assert guards.claim_guard(ok)["passed"]


# ---------------------------------------------------------------- sample size
def test_machine_learning_is_refused_on_a_class_dataset():
    res = guards.sample_size_guard(n=30, task="machine_learning")
    assert res["allowed"] is False
    assert "compare_groups" in res["steer_to"]


def test_more_features_than_samples_is_refused():
    res = guards.sample_size_guard(n=40, task="machine_learning", n_features=256)
    assert res["allowed"] is False
    assert "more columns than rows" in res["message"]


def test_inference_is_allowed_at_small_n():
    assert guards.sample_size_guard(n=6, task="inference")["allowed"] is True


# ---------------------------------------------------------------- prereg
def test_test_is_gated_until_a_prediction_is_written():
    log = guards.PreRegistrationLog()
    assert log.gate("ds1")["allowed"] is False
    log.register(guards.PreRegistration("ds1", "Does salt slow yeast?",
                                        "More salt will slow it down", "less"))
    assert log.gate("ds1")["allowed"] is True


def test_amendments_are_visible_not_forbidden():
    log = guards.PreRegistrationLog()
    log.register(guards.PreRegistration("ds1", "q", "first guess"))
    second = log.register(guards.PreRegistration("ds1", "q", "revised guess"))
    assert second["amendment"] is True
    judged = log.judge("ds1", {"p_value": 0.01, "effect_size": 1.0})
    assert judged["n_amendments"] == 1
    assert judged["amendment_warning"]
    # The ORIGINAL prediction is what the result is judged against.
    assert judged["prediction"] == "first guess"


def test_a_reversed_result_is_named_as_such():
    log = guards.PreRegistrationLog()
    log.register(guards.PreRegistration("ds1", "q", "A will be higher", direction="greater"))
    judged = log.judge("ds1", {"p_value": 0.001, "effect_size": -1.4})
    assert judged["outcome"] == "reversed"


# ---------------------------------------------------------------- spec search
def test_specification_search_phrasing_is_detected():
    d = guards.SpecificationSearchDetector()
    for msg in ["can we drop that outlier and try again",
                "what if we exclude the weird point",
                "try a few different groupings until it works"]:
        assert d.check_message(msg)["detected"], msg


def test_the_polite_reframing_is_caught_by_behaviour_not_phrasing(clean_store):
    """The specification's central finding: guardrails were sensitive to framing.

    'Reporting uncertainty across specifications' is the reframing that bypassed
    frontier-model guardrails. The behavioural signal does not care how it is
    phrased.
    """
    d = guards.SpecificationSearchDetector(clean_store)
    groups = {k: list(v) for k, v in GROUPS.items()}
    for i in range(4):
        groups["control"] = groups["control"][:-1] if i else groups["control"]
        pq.inference.compare_groups(groups, justification=f"run {i}")
    data_id = clean_store.all()[-1].data_id
    # A *different* data_id each time, so check the first dataset's ledger:
    first_id = clean_store.all()[0].data_id
    res = d.check_behaviour(first_id, threshold=1)
    assert res["detected"]


def test_repeated_tests_on_one_dataset_are_flagged(clean_store):
    d = guards.SpecificationSearchDetector(clean_store)
    for i in range(4):
        pq.inference.compare_groups(GROUPS, justification=f"run {i}")
    data_id = clean_store.all()[0].data_id
    res = d.check_behaviour(data_id)
    assert res["detected"]
    assert res["tests_on_this_data"] >= 4


def test_an_exclusion_justified_by_the_result_is_flagged():
    d = guards.SpecificationSearchDetector()
    bad = d.log_exclusion("ds", "row 4", "it was making the p-value too big")
    good = d.log_exclusion("ds", "row 4", "the tube cracked in the centrifuge")
    assert bad["accepted"] is False
    assert good["accepted"] is True


# ---------------------------------------------------------------- ledger
def test_the_ledger_counts_and_then_corrects(clean_store):
    for i in range(4):
        pq.inference.compare_groups(GROUPS, justification=f"run {i}")
    data_id = clean_store.all()[0].data_id
    led = pq.ledger.ledger_for(data_id)
    assert led.tests_run_so_far == 4
    assert led.correction_applied == "holm-bonferroni"
    assert all(e["p_adjusted"] >= e["p_raw"] for e in led.entries)


def test_holm_matches_a_hand_computed_example():
    adj = pq.ledger.holm_bonferroni([0.01, 0.02, 0.03])
    assert adj[0] == pytest.approx(0.03)
    assert adj[1] == pytest.approx(0.04)
    assert adj[2] == pytest.approx(0.04)   # step-down enforces monotonicity


# ---------------------------------------------------------------- name traps
def test_the_package_name_traps_are_caught():
    res = guards.check_package_names("pip install drc jellyfish basico")
    names = {c["package"] for c in res["collisions"]}
    assert {"drc", "jellyfish", "basico"} <= names
    assert not res["passed"]


def test_copyleft_packages_are_flagged():
    res = guards.check_package_names("pip install pingouin giotto-tda")
    licences = {l["package"]: l["licence"] for l in res["licence_issues"]}
    assert licences["pingouin"] == "GPL-3"
    assert licences["giotto-tda"] == "AGPL-3"


def test_the_engines_own_dependencies_are_clean():
    """The guard, applied to this repository. CI fails if a copyleft dep creeps in."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()
    res = guards.check_package_names(text)
    assert res["passed"], res["message"]


def test_no_engine_module_imports_a_copyleft_package():
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "phagequest_engine"
    offenders = []
    for f in src.rglob("*.py"):
        # guards.py names them on purpose, as data.
        if f.name == "guards.py":
            continue
        res = guards.check_package_names(f.read_text())
        if res["licence_issues"]:
            offenders.append((f.name, res["licence_issues"]))
    assert not offenders, offenders


# ---------------------------------------------------------------- PII
def test_pii_is_redacted_before_a_cross_border_call():
    res = guards.redact_pii(
        "I'm Priya, email priya@school.in, phone 9876543210, site 22.57234, 88.36389",
        student_names=["Priya"])
    assert "priya@school.in" not in res["text"]
    assert "9876543210" not in res["text"]
    assert "22.57234" not in res["text"]
    assert "Priya" not in res["text"]


# ---------------------------------------------------------------- assistant
def test_assistant_blocks_a_fabricating_model(clean_store):
    a = pq.Assistant(provider=pq.MockProvider(misbehave=True), grade=8,
                     store=clean_store, require_prediction=False)
    turn = a.ask("control: 12, 14, 11, 13, 12, 15 and treatment: 18, 21, 19, 20, 22, 19 "
                 "-- is it significant?")
    assert turn.blocked
    assert "unverified_number" in turn.block_reasons
    # The warning must not reproduce the fabricated statistics -- a number on
    # screen is copyable whatever the surrounding sentence says.
    assert "2.41" not in turn.reply
    assert ".023" not in turn.reply
    assert "0.91" not in turn.reply


def test_assistant_passes_an_honest_model(clean_store):
    a = pq.Assistant(grade=8, store=clean_store, require_prediction=False)
    turn = a.ask("control: 12, 14, 11, 13, 12, 15 and treatment: 18, 21, 19, 20, 22, 19 "
                 "-- is the treatment bigger?")
    assert not turn.blocked, turn.guard_reports["numeric_scan"]
    assert turn.transcripts
    assert a.transcript_panel(turn)[0]["tool"] == "compare_groups"


def test_assistant_requires_a_prediction_first(clean_store):
    a = pq.Assistant(grade=8, store=clean_store, require_prediction=True)
    turn = a.ask("a: 1,2,3,4 and b: 5,6,7,8 -- is that significant?", dataset_id="ds1")
    assert turn.blocked
    assert "no_prediction" in turn.block_reasons
    a.register_prediction("ds1", "Is b higher?", "I think b will be higher", "greater")
    turn2 = a.ask("a: 1,2,3,4 and b: 5,6,7,8 -- is that significant?", dataset_id="ds1")
    assert "no_prediction" not in turn2.block_reasons


def test_assistant_refuses_specification_search_in_conversation(clean_store):
    a = pq.Assistant(grade=8, store=clean_store, require_prediction=False)
    turn = a.ask("can we just drop that outlier and run it again?")
    assert turn.blocked
    assert "specification_search" in turn.block_reasons


# ---------------------------------------------------------------- grade ladder
def test_grade_five_cannot_reach_a_capstone_tool():
    assert "bridge_analysis" not in registry.tools_for_grade(5)
    res = registry.call_tool("bridge_analysis", {"distance_matrix": [[0, 1], [1, 0]]}, grade=5)
    assert res["error"] == "grade_restricted"


def test_grade_five_has_no_inferential_tools():
    for name in registry.tools_for_grade(5):
        assert registry.TOOLS[name].inferential is False, name


def test_inferential_tools_demand_a_justification():
    res = registry.call_tool("compare_groups", {"groups": GROUPS}, grade=9)
    assert res["error"] == "justification_required"
