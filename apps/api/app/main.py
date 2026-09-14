"""PhageQuest API.

A thin backend, deliberately. Spec section 07:

    "about 90% of the Grades 5-12 workload can run in the student's browser, at
    zero marginal compute cost and with no sandbox-escape surface on our
    infrastructure."

So this service does the things a browser cannot: identity and roles, shared
workspaces, the review queue, cached public data, the Tier-2 job queue, and the
chat proxy. It does NOT compute statistics for Tier 0 -- the browser does that,
using the same engine package, and posts the transcript up afterwards.

The transcript endpoint re-verifies what it is given: a client could post a
fabricated transcript, so the server re-executes cheap tools and compares. That
is the one place the invariant would otherwise have a hole.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

import phagequest_engine as pq
from phagequest_engine import guards, registry

from . import cache, security
from .models import (AuditEvent, Base, CacheEntry, Dataset, ExclusionRow, Job,
                     PreRegistrationRow, ReviewItem, Role, School, SchoolQuota,
                     Specimen, TranscriptRow, User, Workspace)

DATABASE_URL = os.getenv("PHAGEQUEST_DATABASE_URL", "sqlite:///./phagequest.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}
                       if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(engine)
    yield


app = FastAPI(
    title="PhageQuest API",
    version="0.1.0",
    description=("Identity, workspaces, transcripts, review queue, cached public data and the "
                 "Tier-2 job queue. Statistics run in the student's browser."),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("PHAGEQUEST_CORS", "http://localhost:5173").split(","),
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/token")


# ==========================================================================
# auth
# ==========================================================================
def current_user(token: str = Depends(oauth2), db: Session = Depends(get_db)) -> User:
    uid = security.decode_token(token)
    user = db.get(User, uid) if uid else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    return user


def require_role(*roles: Role):
    def dep(user: User = Depends(current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                f"This needs one of: {[r.value for r in roles]}")
        return user
    return dep


def audit(db: Session, user: User, action: str, target: str = "",
          detail: Optional[Dict] = None) -> None:
    db.add(AuditEvent(school_id=user.school_id, actor_id=user.id, action=action,
                      target=target, detail=detail or {}))


class RegisterSchool(BaseModel):
    school_name: str
    admin_pseudonym: str
    password: str = Field(min_length=8)


@app.post("/auth/register-school", tags=["auth"])
def register_school(body: RegisterSchool, db: Session = Depends(get_db)):
    """Bootstrap a school and its first admin.

    Note what is not asked for: no student names, no emails, no dates of birth.
    The school issues pseudonyms and holds the mapping (spec section 03.5).
    """
    school = School(name=body.school_name)
    db.add(school)
    db.flush()
    admin = User(school_id=school.id, pseudonym=body.admin_pseudonym,
                 role=Role.school_admin, grade=12,
                 password_hash=security.hash_password(body.password))
    db.add(admin)
    db.add(SchoolQuota(school_id=school.id))
    db.commit()
    return {"school_id": school.id, "admin_id": admin.id,
            "next": "Confirm parental consent (POST /schools/{id}/consent) before enrolling "
                    "students. DPDP Rule 10 requires verifiable parental consent with an audit "
                    "trail, and the school -- not the platform -- is the Data Fiduciary."}


class Enrol(BaseModel):
    pseudonym: str
    password: str = Field(min_length=8)
    role: Role = Role.student
    grade: int = Field(default=8, ge=5, le=12)


@app.post("/schools/{school_id}/consent", tags=["auth"])
def confirm_consent(school_id: str, admin: User = Depends(require_role(Role.school_admin)),
                    db: Session = Depends(get_db)):
    school = db.get(School, school_id)
    if school is None or admin.school_id != school_id:
        raise HTTPException(404, "School not found")
    school.consent_confirmed = True
    school.consent_confirmed_at = datetime.now(timezone.utc)
    audit(db, admin, "consent_confirmed", school_id)
    db.commit()
    return {"confirmed": True, "at": school.consent_confirmed_at}


@app.post("/schools/{school_id}/enrol", tags=["auth"])
def enrol(school_id: str, body: Enrol,
          admin: User = Depends(require_role(Role.school_admin)),
          db: Session = Depends(get_db)):
    if admin.school_id != school_id:
        raise HTTPException(403, "Not your school")
    school = db.get(School, school_id)
    if body.role == Role.student and not school.consent_confirmed:
        raise HTTPException(
            status.HTTP_412_PRECONDITION_FAILED,
            "This school has not confirmed that it holds verifiable parental consent. Under the "
            "DPDP Act 2023 every user under 18 is a child, and that is the entire Grades 5-12 "
            "roster. Students cannot be enrolled until the school confirms.")
    user = User(school_id=school_id, pseudonym=body.pseudonym, role=body.role,
                grade=body.grade, password_hash=security.hash_password(body.password))
    db.add(user)
    audit(db, admin, "enrol", body.pseudonym, {"role": body.role.value, "grade": body.grade})
    db.commit()
    return {"id": user.id, "pseudonym": user.pseudonym, "role": user.role, "grade": user.grade}


@app.post("/auth/token", tags=["auth"])
def token(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.pseudonym == form.username))
    if user is None or not security.verify_password(form.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Bad credentials")
    return {"access_token": security.make_token(user.id), "token_type": "bearer",
            "role": user.role, "grade": user.grade, "user_id": user.id}


@app.get("/me", tags=["auth"])
def me(user: User = Depends(current_user)):
    return {"id": user.id, "pseudonym": user.pseudonym, "role": user.role,
            "grade": user.grade, "school_id": user.school_id,
            "tools": registry.tools_for_grade(user.grade)}


# ==========================================================================
# tool catalogue -- the browser reads this to know what it may run
# ==========================================================================
@app.get("/tools", tags=["engine"])
def tools(grade: Optional[int] = None, user: User = Depends(current_user)):
    g = grade if grade is not None else user.grade
    if user.role == Role.student:
        g = user.grade          # a student cannot ask for a higher grade's tools
    return {"grade": g, "tools": registry.tool_schemas(grade=g),
            "engine": pq.transcript.engine_versions()}


class RunTool(BaseModel):
    tool: str
    arguments: Dict[str, Any] = {}
    workspace_id: Optional[str] = None


@app.post("/run", tags=["engine"])
def run_tool(body: RunTool, user: User = Depends(current_user),
             db: Session = Depends(get_db)):
    """Tier-1 execution, for the work Pyodide cannot do.

    The browser is the default path; this endpoint exists for the heavier
    geometry and fitting work, and for a device that cannot run WebAssembly.
    The result is identical either way, because it is the same package.
    """
    store = pq.TranscriptStore()
    pq.set_store(store)
    result = registry.call_tool(body.tool, body.arguments, grade=user.grade)
    if result.get("error"):
        raise HTTPException(400, result)
    if body.workspace_id:
        _store_transcripts(db, user, body.workspace_id, store, tier="1")
        db.commit()
    return result


def _store_transcripts(db: Session, user: User, workspace_id: str,
                       store: pq.TranscriptStore, tier: str) -> List[str]:
    ids = []
    for t in store.all():
        db.add(TranscriptRow(
            transcript_id=t.transcript_id, workspace_id=workspace_id, user_id=user.id,
            tool=t.tool, call=t.call, data_id=t.data_id, engine=t.engine,
            params=t.params, result=t.result, tier=tier, duration_ms=t.duration_ms))
        ids.append(t.transcript_id)
    return ids


# ==========================================================================
# workspaces, datasets, specimens
# ==========================================================================
class WorkspaceIn(BaseModel):
    title: str
    lesson_ref: str = ""
    members: List[str] = []


def _own_workspace(db: Session, user: User, workspace_id: str) -> Workspace:
    ws = db.get(Workspace, workspace_id)
    if ws is None:
        raise HTTPException(404, "Workspace not found")
    allowed = (ws.owner_id == user.id or user.id in (ws.members or [])
               or user.role in (Role.trainer, Role.school_admin))
    if not allowed or ws.school_id != user.school_id:
        raise HTTPException(403, "Not your workspace")
    return ws


@app.post("/workspaces", tags=["workspace"])
def create_workspace(body: WorkspaceIn, user: User = Depends(current_user),
                     db: Session = Depends(get_db)):
    ws = Workspace(owner_id=user.id, school_id=user.school_id, title=body.title,
                   grade=user.grade, lesson_ref=body.lesson_ref, members=body.members)
    db.add(ws)
    audit(db, user, "workspace_created", ws.id)
    db.commit()
    return {"id": ws.id, "title": ws.title, "grade": ws.grade}


@app.get("/workspaces", tags=["workspace"])
def list_workspaces(user: User = Depends(current_user), db: Session = Depends(get_db)):
    q = select(Workspace).where(Workspace.school_id == user.school_id)
    if user.role == Role.student:
        rows = [w for w in db.scalars(q)
                if w.owner_id == user.id or user.id in (w.members or [])]
    else:
        rows = list(db.scalars(q))
    return [{"id": w.id, "title": w.title, "grade": w.grade, "lesson_ref": w.lesson_ref,
             "updated_at": w.updated_at} for w in rows]


class DatasetIn(BaseModel):
    name: str
    rows: List[Dict[str, Any]]
    template: Optional[str] = None


@app.post("/workspaces/{workspace_id}/datasets", tags=["workspace"])
def upload_dataset(workspace_id: str, body: DatasetIn,
                   user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Validate first, store second. A table that fails validation is still
    stored -- with its errors -- because the student needs to see and fix it."""
    _own_workspace(db, user, workspace_id)
    store = pq.TranscriptStore()
    pq.set_store(store)
    report = pq.validation.validate_table(body.rows, template=body.template)
    ds = Dataset(workspace_id=workspace_id, name=body.name, template=body.template or "",
                 rows=body.rows, validation=report,
                 data_id=store.all()[-1].data_id if store.all() else "")
    db.add(ds)
    audit(db, user, "dataset_uploaded", ds.id,
          {"rows": len(body.rows), "valid": report.get("valid")})
    db.commit()
    return {"id": ds.id, "validation": report}


@app.get("/workspaces/{workspace_id}/datasets", tags=["workspace"])
def list_datasets(workspace_id: str, user: User = Depends(current_user),
                  db: Session = Depends(get_db)):
    _own_workspace(db, user, workspace_id)
    rows = db.scalars(select(Dataset).where(Dataset.workspace_id == workspace_id))
    return [{"id": d.id, "name": d.name, "template": d.template, "data_id": d.data_id,
             "n_rows": len(d.rows or []), "valid": (d.validation or {}).get("valid"),
             "rows": d.rows} for d in rows]


class SpecimenIn(BaseModel):
    label: str
    site_habitat: str = ""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    stage: str = "collected"
    plaque_morphology: str = ""
    notes: str = ""


@app.post("/workspaces/{workspace_id}/specimens", tags=["workspace"])
def add_specimen(workspace_id: str, body: SpecimenIn,
                 user: User = Depends(current_user), db: Session = Depends(get_db)):
    _own_workspace(db, user, workspace_id)
    sp = Specimen(workspace_id=workspace_id, **body.model_dump())
    sp.round_coords()      # DPDP: 3 dp is a habitat, 5 dp is a child's doorstep
    db.add(sp)
    audit(db, user, "specimen_added", sp.id)
    db.commit()
    return {"id": sp.id, "label": sp.label, "stage": sp.stage,
            "latitude": sp.latitude, "longitude": sp.longitude,
            "note": ("Coordinates are stored rounded to about 100 m. Precise enough for a "
                     "habitat map; not precise enough to locate a student.")}


@app.get("/workspaces/{workspace_id}/specimens", tags=["workspace"])
def list_specimens(workspace_id: str, user: User = Depends(current_user),
                   db: Session = Depends(get_db)):
    _own_workspace(db, user, workspace_id)
    rows = db.scalars(select(Specimen).where(Specimen.workspace_id == workspace_id))
    return [{"id": s.id, "label": s.label, "stage": s.stage, "habitat": s.site_habitat,
             "latitude": s.latitude, "longitude": s.longitude,
             "titre_pfu_per_mL": s.titre_pfu_per_mL,
             "plaque_morphology": s.plaque_morphology,
             "has_genome": bool(s.genome_fasta)} for s in rows]


# ==========================================================================
# transcripts -- including verification of client-computed ones
# ==========================================================================
class TranscriptIn(BaseModel):
    transcript_id: str
    tool: str
    call: str
    data_id: Optional[str] = None
    engine: Dict[str, str] = {}
    params: Dict[str, Any] = {}
    result: Dict[str, Any] = {}
    duration_ms: float = 0.0
    arguments: Optional[Dict[str, Any]] = None   # supplied so the server can re-run


@app.post("/workspaces/{workspace_id}/transcripts", tags=["transcripts"])
def post_transcript(workspace_id: str, body: TranscriptIn,
                    verify: bool = Query(True),
                    user: User = Depends(current_user), db: Session = Depends(get_db)):
    """Accept a transcript computed in the browser, and check it.

    A client can post anything. If the platform trusted whatever arrived, the
    invariant would hold only for numbers the server computed -- which is about
    10% of them. So for any tool whose arguments came with it, the server
    re-runs the computation and compares the p-value and effect size.

    A mismatch is not silently corrected: the row is stored flagged, and the
    trainer's review queue shows it. Quietly fixing it would hide the one event
    that actually matters.
    """
    _own_workspace(db, user, workspace_id)
    if db.scalar(select(TranscriptRow).where(
            TranscriptRow.transcript_id == body.transcript_id)):
        return {"stored": False, "reason": "Already recorded."}

    verification: Dict[str, Any] = {"attempted": False}
    if verify and body.arguments and body.tool in registry.TOOLS:
        vstore = pq.TranscriptStore()
        pq.set_store(vstore)
        check = registry.call_tool(body.tool, body.arguments, grade=user.grade)
        verification = {"attempted": True, "matched": True, "fields": {}}
        for field in ("p_value", "effect_size", "statistic"):
            claimed = body.result.get(field)
            actual = check.get(field)
            if isinstance(claimed, (int, float)) and isinstance(actual, (int, float)):
                ok = abs(claimed - actual) <= 1e-6 + 1e-6 * abs(actual)
                verification["fields"][field] = {"claimed": claimed, "recomputed": actual,
                                                 "matched": ok}
                verification["matched"] &= ok

    row = TranscriptRow(
        transcript_id=body.transcript_id, workspace_id=workspace_id, user_id=user.id,
        tool=body.tool, call=body.call, data_id=body.data_id, engine=body.engine,
        params=body.params,
        result={**body.result, "_server_verification": verification},
        tier="0", duration_ms=body.duration_ms)
    db.add(row)
    audit(db, user, "transcript_posted", body.transcript_id,
          {"tool": body.tool, "verified": verification.get("matched")})
    db.commit()
    return {"stored": True, "verification": verification}


@app.get("/workspaces/{workspace_id}/transcripts", tags=["transcripts"])
def list_transcripts(workspace_id: str, data_id: Optional[str] = None,
                     user: User = Depends(current_user), db: Session = Depends(get_db)):
    _own_workspace(db, user, workspace_id)
    q = select(TranscriptRow).where(TranscriptRow.workspace_id == workspace_id)
    if data_id:
        q = q.where(TranscriptRow.data_id == data_id)
    return [{"transcript_id": t.transcript_id, "tool": t.tool, "call": t.call,
             "data_id": t.data_id, "engine": t.engine, "params": t.params,
             "result": t.result, "tier": t.tier, "duration_ms": t.duration_ms,
             "created_at": t.created_at}
            for t in db.scalars(q.order_by(TranscriptRow.created_at))]


@app.get("/workspaces/{workspace_id}/ledger/{data_id}", tags=["transcripts"])
def ledger(workspace_id: str, data_id: str, user: User = Depends(current_user),
           db: Session = Depends(get_db)):
    """The multiple-comparison ledger, rebuilt from stored transcripts.

    Server-side so it survives a page reload, follows the dataset rather than
    the session, and cannot be reset by reopening the notebook.
    """
    _own_workspace(db, user, workspace_id)
    rows = db.scalars(select(TranscriptRow).where(
        TranscriptRow.workspace_id == workspace_id,
        TranscriptRow.data_id == data_id).order_by(TranscriptRow.created_at))
    store = pq.TranscriptStore()
    for r in rows:
        store.put(pq.Transcript(
            transcript_id=r.transcript_id, tool=r.tool, call=r.call, params=r.params,
            data_id=r.data_id, engine=r.engine, started_at=r.created_at.timestamp(),
            duration_ms=r.duration_ms, result=r.result))
    return pq.ledger.ledger_for(data_id, store=store).to_dict()


# ==========================================================================
# pre-registration and exclusions
# ==========================================================================
class PreRegIn(BaseModel):
    data_id: str
    question: str
    prediction: str
    direction: str = "two-sided"
    planned_test: str = ""
    planned_n: Optional[int] = None


@app.post("/workspaces/{workspace_id}/preregistrations", tags=["honesty"])
def prereg(workspace_id: str, body: PreRegIn, user: User = Depends(current_user),
           db: Session = Depends(get_db)):
    _own_workspace(db, user, workspace_id)
    existing = list(db.scalars(select(PreRegistrationRow).where(
        PreRegistrationRow.workspace_id == workspace_id,
        PreRegistrationRow.data_id == body.data_id)))
    row = PreRegistrationRow(workspace_id=workspace_id, user_id=user.id,
                             version=len(existing) + 1, **body.model_dump())
    db.add(row)
    audit(db, user, "prereg", body.data_id, {"version": row.version})
    db.commit()
    return {"id": row.id, "version": row.version, "amendment": bool(existing),
            "message": ("Locked. You can write another, but this one stays in the record and "
                        "your teacher sees both." if not existing else
                        f"Recorded as amendment #{row.version}. The original is still in the "
                        f"record and is what your result will be judged against.")}


@app.get("/workspaces/{workspace_id}/preregistrations", tags=["honesty"])
def list_prereg(workspace_id: str, data_id: Optional[str] = None,
                user: User = Depends(current_user), db: Session = Depends(get_db)):
    _own_workspace(db, user, workspace_id)
    q = select(PreRegistrationRow).where(PreRegistrationRow.workspace_id == workspace_id)
    if data_id:
        q = q.where(PreRegistrationRow.data_id == data_id)
    return [{"id": p.id, "data_id": p.data_id, "question": p.question,
             "prediction": p.prediction, "direction": p.direction, "version": p.version,
             "created_at": p.created_at}
            for p in db.scalars(q.order_by(PreRegistrationRow.version))]


class ExclusionIn(BaseModel):
    data_id: str
    description: str
    justification: str


@app.post("/workspaces/{workspace_id}/exclusions", tags=["honesty"])
def exclusion(workspace_id: str, body: ExclusionIn, user: User = Depends(current_user),
              db: Session = Depends(get_db)):
    _own_workspace(db, user, workspace_id)
    det = guards.SpecificationSearchDetector()
    res = det.log_exclusion(body.data_id, body.description, body.justification, user.id)
    row = ExclusionRow(workspace_id=workspace_id, user_id=user.id, data_id=body.data_id,
                       description=body.description, justification=body.justification,
                       about_the_measurement=res["accepted"])
    db.add(row)
    audit(db, user, "exclusion", body.data_id, {"accepted": res["accepted"]})
    db.commit()
    return {"id": row.id, **res}


# ==========================================================================
# review queue
# ==========================================================================
class SubmitIn(BaseModel):
    title: str
    claim: str
    transcript_ids: List[str] = []
    prereg_id: Optional[str] = None


@app.post("/workspaces/{workspace_id}/submit", tags=["review"])
def submit(workspace_id: str, body: SubmitIn, user: User = Depends(current_user),
           db: Session = Depends(get_db)):
    """Send work to the trainer, with the flags computed rather than declared.

    The trainer sees a stability warning, a ledger count, an unverified
    transcript or a result that contradicts the student's own prediction,
    because the server derives those from the record -- not because the student
    remembered to mention them.
    """
    _own_workspace(db, user, workspace_id)
    flags: List[Dict[str, Any]] = []
    rows = list(db.scalars(select(TranscriptRow).where(
        TranscriptRow.transcript_id.in_(body.transcript_ids))))

    for t in rows:
        st = (t.result or {}).get("stability") or {}
        if st.get("severity") in ("warn", "block"):
            flags.append({"kind": "stability", "severity": st["severity"],
                          "transcript_id": t.transcript_id, "detail": st.get("verdict")})
        ver = (t.result or {}).get("_server_verification") or {}
        if ver.get("attempted") and not ver.get("matched", True):
            flags.append({"kind": "verification_mismatch", "severity": "block",
                          "transcript_id": t.transcript_id, "detail": ver.get("fields")})
        led = (t.result or {}).get("ledger") or {}
        if led.get("tests_run_so_far", 0) >= 3:
            flags.append({"kind": "multiple_comparisons", "severity": "warn",
                          "transcript_id": t.transcript_id,
                          "detail": led.get("message")})

    data_ids = {t.data_id for t in rows if t.data_id}
    for did in data_ids:
        n_excl = len(list(db.scalars(select(ExclusionRow).where(
            ExclusionRow.data_id == did, ExclusionRow.about_the_measurement.is_(False)))))
        if n_excl:
            flags.append({"kind": "exclusion_justified_by_result", "severity": "block",
                          "detail": f"{n_excl} data point(s) excluded for a reason about the "
                                    f"result rather than the measurement."})
        if not db.scalar(select(PreRegistrationRow).where(
                PreRegistrationRow.data_id == did)):
            flags.append({"kind": "no_prediction", "severity": "warn",
                          "detail": "No prediction was written before this analysis."})

    scan = guards.numeric_scan(body.claim, allowed=[
        v for t in rows for v in _numbers_in(t.result)])
    if not scan.passed:
        flags.append({"kind": "unverified_number_in_claim", "severity": "block",
                      "detail": scan.message})

    item = ReviewItem(workspace_id=workspace_id, submitted_by=user.id, title=body.title,
                      claim=body.claim, transcript_ids=body.transcript_ids,
                      prereg_id=body.prereg_id, flags=flags)
    db.add(item)
    audit(db, user, "submitted", item.id, {"flags": len(flags)})
    db.commit()
    return {"id": item.id, "flags": flags,
            "message": ("Submitted." if not flags else
                        f"Submitted with {len(flags)} thing(s) your teacher will see. They are "
                        f"not accusations -- they are the parts of the record that need a human "
                        f"to look at them.")}


def _numbers_in(obj: Any) -> List[float]:
    st = pq.TranscriptStore()
    st.put(pq.Transcript(transcript_id="tmp", tool="", call="", params={}, data_id=None,
                         engine={}, started_at=0.0, duration_ms=0.0, result=obj or {}))
    return st.numbers()


@app.get("/review", tags=["review"])
def review_queue(status_filter: str = Query("pending", alias="status"),
                 trainer: User = Depends(require_role(Role.trainer, Role.school_admin)),
                 db: Session = Depends(get_db)):
    q = select(ReviewItem).join(Workspace, Workspace.id == ReviewItem.workspace_id).where(
        Workspace.school_id == trainer.school_id)
    if status_filter != "all":
        q = q.where(ReviewItem.status == status_filter)
    out = []
    for item in db.scalars(q.order_by(ReviewItem.created_at.desc())):
        ws = db.get(Workspace, item.workspace_id)
        student = db.get(User, item.submitted_by)
        out.append({
            "id": item.id, "title": item.title, "claim": item.claim,
            "workspace": {"id": ws.id, "title": ws.title, "grade": ws.grade,
                          "lesson_ref": ws.lesson_ref},
            "student": student.pseudonym, "status": item.status,
            "flags": item.flags, "transcript_ids": item.transcript_ids,
            "created_at": item.created_at,
            "worst_flag": max((f["severity"] for f in item.flags), default="none",
                              key=lambda s: {"none": 0, "warn": 1, "block": 2}.get(s, 0)),
        })
    return out


class ReviewDecision(BaseModel):
    status: str
    note: str = ""


@app.post("/review/{item_id}", tags=["review"])
def decide(item_id: str, body: ReviewDecision,
           trainer: User = Depends(require_role(Role.trainer, Role.school_admin)),
           db: Session = Depends(get_db)):
    item = db.get(ReviewItem, item_id)
    if item is None:
        raise HTTPException(404, "Not found")
    item.status = body.status
    item.reviewer_id = trainer.id
    item.reviewer_note = body.note
    item.reviewed_at = datetime.now(timezone.utc)
    audit(db, trainer, "reviewed", item_id, {"status": body.status})
    db.commit()
    return {"id": item.id, "status": item.status}


# ==========================================================================
# public data proxy -- cached, because a class of thirty will rate-limit us
# ==========================================================================
@app.get("/public/phagesdb/{path:path}", tags=["public-data"])
async def phagesdb(path: str, request: Request, user: User = Depends(current_user),
                   db: Session = Depends(get_db)):
    return await cache.fetch_cached(db, "phagesdb", path, dict(request.query_params))


@app.get("/public/ncbi/{path:path}", tags=["public-data"])
async def ncbi(path: str, request: Request, user: User = Depends(current_user),
               db: Session = Depends(get_db)):
    return await cache.fetch_cached(db, "ncbi", path, dict(request.query_params))


@app.get("/public/status", tags=["public-data"])
def public_status(db: Session = Depends(get_db)):
    return cache.status(db)


# ==========================================================================
# Tier-2 jobs -- queued, quota'd, and honest about waiting
# ==========================================================================
JOB_KINDS = {
    "pharokka": {"minutes": 6.0, "label": "Whole-genome phage annotation",
                 "wraps": "PHANOTATE, Prodigal-gv, tRNAscan-SE, ARAGORN, MinCED, MMseqs2, "
                          "PyHMMER against PHROGs, Mash against INPHARED",
                 "db_mb": 656},
    "taxmyphage": {"minutes": 4.0, "label": "ICTV genus and species assignment",
                   "wraps": "taxmyphage against ICTV MSL41", "db_mb": 300},
    "bacphlip": {"minutes": 1.0, "label": "Lytic or temperate?",
                 "wraps": "BACPHLIP (HMMs bundled, no database)", "db_mb": 0},
    "skani": {"minutes": 0.5, "label": "Nearest relatives by genome distance",
              "wraps": "skani / Mash sketches against INPHARED", "db_mb": 20},
}


class JobIn(BaseModel):
    kind: str
    specimen_id: Optional[str] = None
    params: Dict[str, Any] = {}


@app.post("/workspaces/{workspace_id}/jobs", tags=["jobs"])
def submit_job(workspace_id: str, body: JobIn, user: User = Depends(current_user),
               db: Session = Depends(get_db)):
    """Queue a Tier-2 job, refusing rather than overspending.

    Spec section 08 disqualifies Terra and DNAnexus for "consumption billing
    with no hard cap. A student's infinite loop produces a real invoice." So
    the quota here is a refusal, not a notification.
    """
    _own_workspace(db, user, workspace_id)
    spec = JOB_KINDS.get(body.kind)
    if spec is None:
        raise HTTPException(400, {"message": f"Unknown job kind {body.kind!r}.",
                                  "available": sorted(JOB_KINDS)})

    quota = db.get(SchoolQuota, user.school_id)
    if quota is None:
        quota = SchoolQuota(school_id=user.school_id)
        db.add(quota)
        db.flush()
    if datetime.now(timezone.utc) - quota.period_started.replace(tzinfo=timezone.utc) > timedelta(days=30):
        quota.used_this_month = 0.0
        quota.period_started = datetime.now(timezone.utc)

    remaining = quota.monthly_job_minutes - quota.used_this_month
    if spec["minutes"] > remaining:
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, {
            "message": (f"This school has {remaining:.0f} minutes of batch compute left this "
                        f"month and this job needs {spec['minutes']:.0f}. The job was NOT "
                        f"started. Nothing is billed beyond the ceiling -- a runaway job cannot "
                        f"produce a surprise invoice here."),
            "remaining_minutes": remaining, "required_minutes": spec["minutes"]})

    quota.used_this_month += spec["minutes"]
    job = Job(workspace_id=workspace_id, school_id=user.school_id, user_id=user.id,
              kind=body.kind, specimen_id=body.specimen_id, params=body.params,
              estimated_minutes=spec["minutes"], status="queued")
    db.add(job)
    audit(db, user, "job_queued", job.id, {"kind": body.kind})
    db.commit()
    return {
        "id": job.id, "status": job.status, "kind": body.kind, "label": spec["label"],
        "estimated_minutes": spec["minutes"],
        "wraps": spec["wraps"], "database_mb": spec["db_mb"],
        "remaining_minutes_this_month": quota.monthly_job_minutes - quota.used_this_month,
        "message": (f"Queued. {spec['label']} takes about {spec['minutes']:.0f} minutes, which is "
                    f"longer than a lesson -- come back to it next period rather than waiting."),
    }


@app.get("/workspaces/{workspace_id}/jobs", tags=["jobs"])
def list_jobs(workspace_id: str, user: User = Depends(current_user),
              db: Session = Depends(get_db)):
    _own_workspace(db, user, workspace_id)
    return [{"id": j.id, "kind": j.kind, "status": j.status,
             "estimated_minutes": j.estimated_minutes, "result": j.result,
             "error": j.error, "created_at": j.created_at, "finished_at": j.finished_at}
            for j in db.scalars(select(Job).where(Job.workspace_id == workspace_id))]


@app.get("/jobs/kinds", tags=["jobs"])
def job_kinds():
    return {"kinds": JOB_KINDS,
            "total_shipped_database_mb": sum(k["db_mb"] for k in JOB_KINDS.values()),
            "note": ("Deliberately excluded: iPHoP and VIBRANT (tens of GB), the full Bakta "
                     "database (31.9 GB -- use db-light at 1.3 GB), Phold (~8 GB, premium tier "
                     "only) and IMG/VR. The total shipped footprint is what makes a school "
                     "deployment possible at all.")}


# ==========================================================================
# chat
# ==========================================================================
class ChatIn(BaseModel):
    message: str
    workspace_id: Optional[str] = None
    dataset_id: Optional[str] = None


@app.post("/chat", tags=["chat"])
def chat(body: ChatIn, user: User = Depends(current_user), db: Session = Depends(get_db)):
    """One assistant turn, with every guard in `phagequest_engine.assistant`.

    The provider is pluggable and defaults to the offline mock, so a school with
    no internet -- or an organisation that has not yet signed a DPA for a
    cross-border model API (spec section 09) -- still gets a working assistant
    and every honesty guarantee.
    """
    store = pq.TranscriptStore()
    pq.set_store(store)
    assistant = pq.Assistant(grade=user.grade, store=store,
                             student_names=[user.pseudonym])

    if body.dataset_id:
        ds = db.get(Dataset, body.dataset_id)
        if ds and ds.data_id:
            for p in db.scalars(select(PreRegistrationRow).where(
                    PreRegistrationRow.data_id == ds.data_id).order_by(
                        PreRegistrationRow.version)):
                assistant.prereg.register(guards.PreRegistration(
                    dataset_id=ds.data_id, question=p.question, prediction=p.prediction,
                    direction=p.direction))

    turn = assistant.ask(body.message)
    if body.workspace_id:
        _own_workspace(db, user, body.workspace_id)
        _store_transcripts(db, user, body.workspace_id, store, tier="1")
        audit(db, user, "chat", body.workspace_id,
              {"blocked": turn.blocked, "reasons": turn.block_reasons})
        db.commit()

    return {"reply": turn.reply, "blocked": turn.blocked,
            "block_reasons": turn.block_reasons,
            "tool_calls": turn.tool_calls,
            "transcripts": assistant.transcript_panel(turn),
            "guards": turn.guard_reports,
            "redactions": turn.redactions}


@app.get("/health", tags=["meta"])
def health():
    return {"ok": True, "engine": pq.__version__,
            "versions": pq.transcript.engine_versions(),
            "tools": len(registry.TOOLS)}
