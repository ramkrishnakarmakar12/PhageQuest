"""End-to-end API tests.

The important ones here are not CRUD: they are the places where the honesty
invariant could leak out of the engine and into a network boundary. A client
can post whatever JSON it likes, so the tests that matter are the ones showing
the server does not simply believe it.
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("PHAGEQUEST_SECRET_KEY", "test-secret")
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["PHAGEQUEST_DATABASE_URL"] = f"sqlite:///{_tmp.name}"

from app.main import app, engine  # noqa: E402
from app.models import Base  # noqa: E402

# TestClient only runs the app's lifespan inside a `with` block, and these
# tests share one module-scoped client, so create the schema explicitly rather
# than relying on startup having happened.
Base.metadata.create_all(engine)

client = TestClient(app)

GROUPS = {"control": [12, 14, 11, 13, 12, 15], "treatment": [18, 21, 19, 20, 22, 19]}


@pytest.fixture(scope="module")
def school():
    r = client.post("/auth/register-school", json={
        "school_name": "Test School", "admin_pseudonym": "admin-1",
        "password": "correct-horse"})
    assert r.status_code == 200
    return r.json()


@pytest.fixture(scope="module")
def admin_token(school):
    r = client.post("/auth/token", data={"username": "admin-1", "password": "correct-horse"})
    return r.json()["access_token"]


def _auth(tok):
    return {"Authorization": f"Bearer {tok}"}


# ---------------------------------------------------------------- DPDP gates
def test_students_cannot_be_enrolled_before_consent_is_confirmed(school, admin_token):
    r = client.post(f"/schools/{school['school_id']}/enrol",
                    json={"pseudonym": "s-001", "password": "pw-12345678",
                          "role": "student", "grade": 8},
                    headers=_auth(admin_token))
    assert r.status_code == 412
    assert "parental consent" in r.json()["detail"]


@pytest.fixture(scope="module")
def student_token(school, admin_token):
    client.post(f"/schools/{school['school_id']}/consent", headers=_auth(admin_token))
    r = client.post(f"/schools/{school['school_id']}/enrol",
                    json={"pseudonym": "s-001", "password": "pw-12345678",
                          "role": "student", "grade": 8},
                    headers=_auth(admin_token))
    assert r.status_code == 200
    return client.post("/auth/token",
                       data={"username": "s-001", "password": "pw-12345678"}
                       ).json()["access_token"]


@pytest.fixture(scope="module")
def trainer_token(school, admin_token):
    client.post(f"/schools/{school['school_id']}/enrol",
                json={"pseudonym": "t-001", "password": "pw-12345678",
                      "role": "trainer", "grade": 12},
                headers=_auth(admin_token))
    return client.post("/auth/token",
                       data={"username": "t-001", "password": "pw-12345678"}
                       ).json()["access_token"]


def test_unauthenticated_requests_are_refused():
    assert client.get("/workspaces").status_code == 401
    assert client.post("/run", json={"tool": "describe"}).status_code == 401


# ---------------------------------------------------------------- grade ladder
def test_the_grade_ladder_is_enforced_at_the_api(student_token):
    """A Grade 8 token cannot reach a Grade 11-12 tool, whatever it asks for."""
    r = client.get("/tools", params={"grade": 12}, headers=_auth(student_token))
    names = {t["name"] for t in r.json()["tools"]}
    assert "bridge_analysis" not in names
    assert r.json()["grade"] == 8

    run = client.post("/run", headers=_auth(student_token), json={
        "tool": "bridge_analysis", "arguments": {"distance_matrix": [[0, 1], [1, 0]]}})
    assert run.status_code == 400
    assert run.json()["detail"]["error"] == "grade_restricted"


def test_inferential_tools_need_a_justification_over_the_api(student_token):
    r = client.post("/run", headers=_auth(student_token),
                    json={"tool": "compare_groups", "arguments": {"groups": GROUPS}})
    assert r.status_code == 400
    assert r.json()["detail"]["error"] == "justification_required"


# ---------------------------------------------------------------- workspace
@pytest.fixture(scope="module")
def workspace(student_token):
    r = client.post("/workspaces", headers=_auth(student_token),
                    json={"title": "pH and yeast", "lesson_ref": "G8-L9"})
    return r.json()["id"]


def test_a_bad_upload_is_stored_with_errors_that_name_the_row(student_token, workspace):
    r = client.post(f"/workspaces/{workspace}/datasets", headers=_auth(student_token),
                    json={"name": "strips", "template": "water_quality",
                          "rows": [{"sample": "pond", "pH": 7.2},
                                   {"sample": "tap", "pH": 19},
                                   {"sample": "well", "pH": 7.0}]})
    v = r.json()["validation"]
    assert v["valid"] is False
    assert any(i.get("row") == 3 for i in v["issues"])


def test_coordinates_are_rounded_before_storage(student_token, workspace):
    r = client.post(f"/workspaces/{workspace}/specimens", headers=_auth(student_token),
                    json={"label": "soil-1", "latitude": 22.5723401, "longitude": 88.3638901})
    body = r.json()
    assert body["latitude"] == 22.572
    assert body["longitude"] == 88.364


# ---------------------------------------------------------------- transcripts
def test_a_fabricated_transcript_is_stored_flagged_not_believed(student_token, workspace):
    """The one place the invariant could leak: a lying client.

    The server re-runs the computation and records the mismatch. It does NOT
    quietly substitute the right answer, because hiding the event would hide
    the only thing worth seeing.
    """
    r = client.post(f"/workspaces/{workspace}/transcripts", headers=_auth(student_token),
                    json={"transcript_id": "tx_fake0000000001", "tool": "compare_groups",
                          "call": "compare_groups(...)", "data_id": "deadbeef",
                          "engine": {"scipy": "1.17.1"},
                          "result": {"p_value": 0.0001, "effect_size": 3.2},
                          "arguments": {"groups": GROUPS, "justification": "claimed"}})
    ver = r.json()["verification"]
    assert ver["attempted"] is True
    assert ver["matched"] is False
    assert ver["fields"]["p_value"]["claimed"] == 0.0001
    assert ver["fields"]["p_value"]["recomputed"] != 0.0001


def test_an_honest_transcript_verifies(student_token, workspace):
    run = client.post("/run", headers=_auth(student_token), json={
        "tool": "compare_groups",
        "arguments": {"groups": GROUPS, "justification": "Comparing two conditions."}})
    real = run.json()
    r = client.post(f"/workspaces/{workspace}/transcripts", headers=_auth(student_token),
                    json={"transcript_id": "tx_honest000000001", "tool": "compare_groups",
                          "call": "compare_groups(...)", "data_id": "abc",
                          "result": {"p_value": real["p_value"],
                                     "effect_size": real["effect_size"]},
                          "arguments": {"groups": GROUPS, "justification": "Comparing."}})
    assert r.json()["verification"]["matched"] is True


def test_the_ledger_survives_a_reload(student_token, workspace):
    """Run the same comparison repeatedly and the count follows the DATASET."""
    for i in range(4):
        run = client.post("/run", headers=_auth(student_token), json={
            "tool": "compare_groups",
            "arguments": {"groups": GROUPS, "justification": f"run {i}"}})
        client.post(f"/workspaces/{workspace}/transcripts", headers=_auth(student_token),
                    json={"transcript_id": f"tx_ledger{i:09d}", "tool": "compare_groups",
                          "call": "compare_groups(...)", "data_id": "ledger-data",
                          "result": {"p_value": run.json()["p_value"]}})
    led = client.get(f"/workspaces/{workspace}/ledger/ledger-data",
                     headers=_auth(student_token)).json()
    assert led["tests_run_so_far"] == 4
    assert led["correction_applied"] == "holm-bonferroni"


# ---------------------------------------------------------------- honesty
def test_an_exclusion_justified_by_the_result_is_flagged(student_token, workspace):
    r = client.post(f"/workspaces/{workspace}/exclusions", headers=_auth(student_token),
                    json={"data_id": "ledger-data", "description": "row 4",
                          "justification": "it was making the p-value too big"})
    assert r.json()["accepted"] is False


def test_amendments_keep_the_original_prediction(student_token, workspace):
    client.post(f"/workspaces/{workspace}/preregistrations", headers=_auth(student_token),
                json={"data_id": "pre-data", "question": "Does pH matter?",
                      "prediction": "pH 7 will be best", "direction": "two-sided"})
    second = client.post(f"/workspaces/{workspace}/preregistrations",
                         headers=_auth(student_token),
                         json={"data_id": "pre-data", "question": "Does pH matter?",
                               "prediction": "actually pH 5", "direction": "two-sided"})
    assert second.json()["version"] == 2
    rows = client.get(f"/workspaces/{workspace}/preregistrations",
                      params={"data_id": "pre-data"}, headers=_auth(student_token)).json()
    assert [p["prediction"] for p in rows][0] == "pH 7 will be best"


def test_a_claim_with_an_invented_number_is_flagged_to_the_trainer(
        student_token, trainer_token, workspace):
    sub = client.post(f"/workspaces/{workspace}/submit", headers=_auth(student_token),
                      json={"title": "pH result",
                            "claim": "The effect was huge, p = 0.0001 and d = 4.5.",
                            "transcript_ids": ["tx_honest000000001"]})
    kinds = {f["kind"] for f in sub.json()["flags"]}
    assert "unverified_number_in_claim" in kinds

    queue = client.get("/review", headers=_auth(trainer_token)).json()
    assert queue
    assert queue[0]["worst_flag"] == "block"
    assert queue[0]["student"] == "s-001"


def test_a_student_cannot_read_the_review_queue(student_token):
    assert client.get("/review", headers=_auth(student_token)).status_code == 403


# ---------------------------------------------------------------- jobs
def test_the_job_quota_refuses_rather_than_billing(student_token, workspace, admin_token, school):
    from sqlalchemy import select

    from app.main import SessionLocal
    from app.models import SchoolQuota
    with SessionLocal() as db:
        q = db.scalar(select(SchoolQuota).where(SchoolQuota.school_id == school["school_id"]))
        q.used_this_month = q.monthly_job_minutes - 1.0
        db.commit()

    r = client.post(f"/workspaces/{workspace}/jobs", headers=_auth(student_token),
                    json={"kind": "pharokka"})
    assert r.status_code == 429
    assert "NOT" in r.json()["detail"]["message"]


def test_job_kinds_declare_their_database_footprint():
    kinds = client.get("/jobs/kinds").json()
    assert kinds["total_shipped_database_mb"] < 1500
    assert "iPHoP" in kinds["note"]


# ---------------------------------------------------------------- chat
def test_chat_returns_transcripts_alongside_the_reply(student_token, workspace):
    r = client.post("/chat", headers=_auth(student_token), json={
        "message": "control: 12, 14, 11, 13, 12, 15 and treatment: 18, 21, 19, 20, 22, 19 "
                   "-- is the treatment bigger?",
        "workspace_id": workspace})
    body = r.json()
    assert body["transcripts"], body
    assert body["transcripts"][0]["tool"] == "compare_groups"
    assert body["guards"]["numeric_scan"]["passed"] is True


def test_chat_refuses_specification_search(student_token):
    r = client.post("/chat", headers=_auth(student_token),
                    json={"message": "can we drop that outlier and try again?"})
    assert r.json()["blocked"] is True
    assert "specification_search" in r.json()["block_reasons"]


def test_health_reports_the_engine_versions():
    h = client.get("/health").json()
    assert h["ok"] is True
    assert "scipy" in h["versions"]
    assert h["tools"] > 30
