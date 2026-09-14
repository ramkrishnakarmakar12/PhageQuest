"""Persistence model.

Shaped around the three objects the live app already carries (spec section 07):

    "The student notebook is where experiment tables enter and where every
    analysis transcript lands. The specimen record -- the group's growing
    isolates -- is the natural key linking a soil sample through plaque counts
    to, eventually, a genome. The trainer's review queue is where the
    pre-registered hypothesis, the multiple-comparison ledger and the stability
    warning surface to a human before a claim is graded."

So Block 2 is services behind those three objects, not a new product.

DPDP posture (spec sections 02 and 09): students are identified by an opaque
`pseudonym` and a `school_ref` the school controls. There is no name, no email,
no date of birth and no free-text identity field on Student. The school holds
the mapping; we are the Data Processor and the school is the Data Fiduciary.
Site coordinates are stored rounded, because a sampling point to five decimals
locates a child.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (JSON, Boolean, Column, DateTime, Enum, Float, ForeignKey,
                        Integer, String, Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Role(str, enum.Enum):
    student = "student"
    trainer = "trainer"
    school_admin = "school_admin"


class School(Base):
    __tablename__ = "schools"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    # The school is the Data Fiduciary. This flag records that the school has
    # confirmed it holds verifiable parental consent (DPDP Rule 10) -- we never
    # collect that consent ourselves, because we never hold the identities.
    consent_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_confirmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    users = relationship("User", back_populates="school", cascade="all, delete-orphan")


class User(Base):
    """A participant.

    `pseudonym` is the only identifier. For a student it is issued by the school
    and means nothing to us; for a trainer or admin it may be an email, because
    staff are adults and outside the children's-data regime.
    """
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("school_id", "pseudonym"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    school_id: Mapped[str] = mapped_column(ForeignKey("schools.id"))
    pseudonym: Mapped[str] = mapped_column(String(120), index=True)
    role: Mapped[Role] = mapped_column(Enum(Role), default=Role.student)
    grade: Mapped[int] = mapped_column(Integer, default=8)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    school = relationship("School", back_populates="users")
    workspaces = relationship("Workspace", back_populates="owner",
                              cascade="all, delete-orphan")


class Workspace(Base):
    """The student notebook: the unit of work, of sharing and of provenance.

    Borrowed from KBase's Narrative (spec section 03.5): the analysis, the
    provenance record and the shareable report are one object.
    """
    __tablename__ = "workspaces"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    school_id: Mapped[str] = mapped_column(ForeignKey("schools.id"))
    title: Mapped[str] = mapped_column(String(200))
    grade: Mapped[int] = mapped_column(Integer, default=8)
    lesson_ref: Mapped[str] = mapped_column(String(64), default="")
    members: Mapped[list] = mapped_column(JSON, default=list)   # user ids, group work
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    owner = relationship("User", back_populates="workspaces")
    datasets = relationship("Dataset", back_populates="workspace",
                            cascade="all, delete-orphan")
    transcripts = relationship("TranscriptRow", back_populates="workspace",
                               cascade="all, delete-orphan")


class Dataset(Base):
    """A table the student uploaded or typed in.

    `data_id` is the engine's content fingerprint, which is what ties a dataset
    to its ledger and its pre-registration -- so re-uploading the same numbers
    under a new name does not reset the multiple-comparison count.
    """
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"))
    name: Mapped[str] = mapped_column(String(200))
    template: Mapped[str] = mapped_column(String(64), default="")
    data_id: Mapped[str] = mapped_column(String(64), index=True, default="")
    rows: Mapped[list] = mapped_column(JSON, default=list)
    validation: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    workspace = relationship("Workspace", back_populates="datasets")


class Specimen(Base):
    """The specimen record: soil sample -> plaque counts -> eventually a genome.

    Spec section 07: this chain "is exactly the chain section 11 depends on and
    which no competitor has". Coordinates are stored to 3 decimal places (about
    100 m) rather than as recorded, because that is enough for a habitat map and
    not enough to locate a child.
    """
    __tablename__ = "specimens"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"))
    label: Mapped[str] = mapped_column(String(120))
    site_habitat: Mapped[str] = mapped_column(String(120), default="")
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    collected_on: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    stage: Mapped[str] = mapped_column(String(40), default="collected")
    # collected -> enriched -> plaques -> purified -> sequenced -> annotated
    titre_pfu_per_mL: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    plaque_morphology: Mapped[str] = mapped_column(String(120), default="")
    genome_fasta: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    def round_coords(self) -> None:
        if self.latitude is not None:
            self.latitude = round(self.latitude, 3)
        if self.longitude is not None:
            self.longitude = round(self.longitude, 3)


class TranscriptRow(Base):
    """A stored execution record.

    Transcripts arrive from BOTH tiers: Tier 0 posts them up from the browser
    after computing locally, Tier 1 writes them directly. Either way the record
    is identical, which is what lets a teacher review work done offline.
    """
    __tablename__ = "transcripts"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    transcript_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    tool: Mapped[str] = mapped_column(String(64), index=True)
    call: Mapped[str] = mapped_column(Text)
    data_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    engine: Mapped[dict] = mapped_column(JSON, default=dict)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    tier: Mapped[str] = mapped_column(String(16), default="0")   # 0 browser, 1 sandbox, 2 batch
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    workspace = relationship("Workspace", back_populates="transcripts")


class PreRegistrationRow(Base):
    """A prediction, locked at the moment it was written.

    Amendments are new rows, never edits. The review queue shows all of them.
    """
    __tablename__ = "preregistrations"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    data_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text)
    prediction: Mapped[str] = mapped_column(Text)
    direction: Mapped[str] = mapped_column(String(16), default="two-sided")
    planned_test: Mapped[str] = mapped_column(String(64), default="")
    planned_n: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class ExclusionRow(Base):
    """A data point the student removed, and their stated reason.

    Kept forever, shown in the report, and flagged when the reason is about the
    result rather than about the measurement.
    """
    __tablename__ = "exclusions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    data_id: Mapped[str] = mapped_column(String(64), index=True)
    description: Mapped[str] = mapped_column(Text)
    justification: Mapped[str] = mapped_column(Text)
    about_the_measurement: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class ReviewItem(Base):
    """The trainer's queue entry.

    Carries the three things the spec says must reach a human before a claim is
    graded: the pre-registered hypothesis, the ledger, and the stability warning.
    """
    __tablename__ = "review_items"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"))
    submitted_by: Mapped[str] = mapped_column(ForeignKey("users.id"))
    title: Mapped[str] = mapped_column(String(200))
    claim: Mapped[str] = mapped_column(Text)
    transcript_ids: Mapped[list] = mapped_column(JSON, default=list)
    prereg_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    reviewer_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    reviewer_note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Job(Base):
    """A Tier-2 batch job: annotation, taxonomy, alignment.

    Spec section 07: "Minutes, not seconds -- so it is scheduled between
    lessons, not inside one. Hard spend ceilings per school." The ceiling is
    enforced here rather than being a billing alert, because Terra's
    consumption billing with no hard cap is listed as disqualifying.
    """
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id"))
    school_id: Mapped[str] = mapped_column(ForeignKey("schools.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    kind: Mapped[str] = mapped_column(String(40))   # pharokka | taxmyphage | bacphlip | skani
    specimen_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="queued")
    estimated_minutes: Mapped[float] = mapped_column(Float, default=5.0)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class SchoolQuota(Base):
    """The hard spend ceiling. Not an alert -- a refusal."""
    __tablename__ = "school_quotas"
    school_id: Mapped[str] = mapped_column(ForeignKey("schools.id"), primary_key=True)
    monthly_job_minutes: Mapped[float] = mapped_column(Float, default=600.0)
    used_this_month: Mapped[float] = mapped_column(Float, default=0.0)
    period_started: Mapped[datetime] = mapped_column(DateTime, default=_now)


class CacheEntry(Base):
    """Server-side cache for public data.

    Spec section 03.1: "PhagesDB serves JSON only, one gene per call, and
    ignores query filters -- cache aggressively server-side. NCBI E-utilities
    allows 3 req/s without a key; thirty students will trip that in the first
    minute."

    So the cache is not an optimisation here; it is what stops a class of
    thirty from being rate-limited into failure in the first five minutes of a
    forty-minute period.
    """
    __tablename__ = "cache"
    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    source: Mapped[str] = mapped_column(String(40), index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class AuditEvent(Base):
    """Audit trail.

    DPDP Rule 10 requires audit trails around children's data. This table
    records WHAT happened and to WHICH pseudonym -- never what was measured,
    and never the data itself.
    """
    __tablename__ = "audit"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    school_id: Mapped[str] = mapped_column(String(32), index=True)
    actor_id: Mapped[str] = mapped_column(String(32), index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str] = mapped_column(String(120), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    at: Mapped[datetime] = mapped_column(DateTime, default=_now)
