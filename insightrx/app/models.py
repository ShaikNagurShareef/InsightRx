"""Relational model. Every clinical row carries tenant_id; signed rows are never updated in place."""
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def now():
    return datetime.now(timezone.utc)


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(20))          # operator | referring | specialist | coordinator | admin | medinfo
    specialty: Mapped[str] = mapped_column(String(60), default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    prefs: Mapped[dict] = mapped_column(JSON, default=dict)  # {'mode': 'immediate'|'digest', 'quiet_start': 22, 'quiet_end': 7}


class Patient(Base):
    __tablename__ = "patients"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    ref: Mapped[str] = mapped_column(String(40))            # synthetic reference, never a real identity
    age: Mapped[float | None] = mapped_column(Float, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(10), nullable=True)
    dm_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    insulin: Mapped[str] = mapped_column(String(10), default="unknown")          # yes | no | unknown
    oral_treatment: Mapped[str] = mapped_column(String(10), default="unknown")
    # {condition_code: {'value': 'present'|'absent'|'unknown', 'source': str, 'date': str, 'verification': str}}
    conditions: Mapped[dict] = mapped_column(JSON, default=dict)
    medications: Mapped[list | None] = mapped_column(JSON, nullable=True)      # current medicines, generic names
    synthetic: Mapped[bool] = mapped_column(Boolean, default=True)


class Case(Base):
    __tablename__ = "cases"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    patient_id: Mapped[int] = mapped_column(ForeignKey("patients.id"))
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))         # referring HCP accountable for the case
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    encounter_date: Mapped[str] = mapped_column(String(20))
    device: Mapped[str] = mapped_column(String(80))
    symptoms: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(30), default="Draft")
    version: Mapped[int] = mapped_column(Integer, default=1)
    urgent_concern: Mapped[bool] = mapped_column(Boolean, default=False)   # clinician-entered, never model-set
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    patient: Mapped[Patient] = relationship()
    owner: Mapped[User] = relationship(foreign_keys=[owner_id])


class Image(Base):
    __tablename__ = "images"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    sha256: Mapped[str] = mapped_column(String(64))
    path: Mapped[str] = mapped_column(String(400), default="")       # filesystem store (local deployments)
    data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True, deferred=True)   # database store (Vercel)
    laterality: Mapped[str] = mapped_column(String(10))       # OD | OS | unknown
    view: Mapped[str] = mapped_column(String(30), default="macula-centred")
    source: Mapped[str] = mapped_column(String(80))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    superseded_by: Mapped[int | None] = mapped_column(Integer, nullable=True)
    uploaded_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ModelRun(Base):
    __tablename__ = "model_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    case_version: Mapped[int] = mapped_column(Integer)
    image_ids: Mapped[list] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(120))
    threshold_version: Mapped[str] = mapped_column(String(60))
    source: Mapped[str] = mapped_column(String(20))           # live | simulated
    status: Mapped[str] = mapped_column(String(20))           # completed | failed
    result: Mapped[dict] = mapped_column(JSON)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    case_version: Mapped[int] = mapped_column(Integer)
    model_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hcp_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    decision: Mapped[str] = mapped_column(String(30))         # accept | disagree | recapture | manual_review
    interpretation: Mapped[str] = mapped_column(Text)
    override_reason: Mapped[str] = mapped_column(Text, default="")
    next_action: Mapped[str] = mapped_column(String(60))
    systemic_notes: Mapped[dict] = mapped_column(JSON, default=dict)   # {condition: hcp assessment}
    signature: Mapped[str] = mapped_column(String(64))
    signed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    hcp: Mapped[User] = relationship()


class Referral(Base):
    __tablename__ = "referrals"
    __table_args__ = (UniqueConstraint("tenant_id", "sender_id", "idempotency_key"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    review_id: Mapped[int] = mapped_column(ForeignKey("reviews.id"))
    sender_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    recipient_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))         # accountable for the next action
    topic: Mapped[str] = mapped_column(String(60))                          # retinal | hypertension | nephropathy ...
    question: Mapped[str] = mapped_column(Text)
    package: Mapped[dict] = mapped_column(JSON)
    package_hash: Mapped[str] = mapped_column(String(64))
    signature: Mapped[str] = mapped_column(String(64))
    idempotency_key: Mapped[str] = mapped_column(String(80))
    stage: Mapped[str] = mapped_column(String(30), default="Sent")
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)   # SIMULATED demo policy
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    appointment: Mapped[dict] = mapped_column(JSON, default=dict)
    closure_code: Mapped[str] = mapped_column(String(30), default="")
    sent_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    case: Mapped[Case] = relationship()
    sender: Mapped[User] = relationship(foreign_keys=[sender_id])
    recipient: Mapped[User] = relationship(foreign_keys=[recipient_id])
    owner: Mapped[User] = relationship(foreign_keys=[owner_id])


class Message(Base):
    """Structured exchange on a referral: info requests, answers, signed specialist responses."""
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    referral_id: Mapped[int] = mapped_column(ForeignKey("referrals.id"))
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    kind: Mapped[str] = mapped_column(String(30))       # info_request | info_reply | response | decline | note
    body: Mapped[str] = mapped_column(Text)
    recommendation: Mapped[str] = mapped_column(String(120), default="")
    signature: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    author: Mapped[User] = relationship()


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    referral_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    kind: Mapped[str] = mapped_column(String(40))
    title: Mapped[str] = mapped_column(String(200))
    assignee_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(20), default="open")    # open | done | cancelled
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    assignee: Mapped[User] = relationship()


class Barrier(Base):
    __tablename__ = "barriers"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    referral_id: Mapped[int] = mapped_column(ForeignKey("referrals.id"))
    category: Mapped[str] = mapped_column(String(30))    # transport | language | affordability | contact | preference
    note: Mapped[str] = mapped_column(Text, default="")
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)


class Notification(Base):
    """In-app only; body never carries clinical detail."""
    __tablename__ = "notifications"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    body: Mapped[str] = mapped_column(String(200))
    link: Mapped[str] = mapped_column(String(200))
    read: Mapped[bool] = mapped_column(Boolean, default=False)
    digest: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AuditEvent(Base):
    """Append-only."""
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(Integer)
    case_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actor_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(60))
    detail: Mapped[str] = mapped_column(Text, default="")
    version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AppSetting(Base):
    """Small key/value settings, e.g. the currently registered vision-worker URL."""
    __tablename__ = "app_settings"
    key: Mapped[str] = mapped_column(String(60), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class TryImage(Base):
    """Photos uploaded to 'Try an image', kept briefly so details can be changed and the analysis re-run."""
    __tablename__ = "try_images"
    id: Mapped[int] = mapped_column(primary_key=True)
    token: Mapped[str] = mapped_column(String(40), index=True)
    tenant_id: Mapped[int] = mapped_column(Integer)
    user_id: Mapped[int] = mapped_column(Integer)
    idx: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(200))
    data: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class MedInfoRequest(Base):
    """Clinician question to a manufacturer's medical-information desk. Carries a de-identified context only (age band
    and finding labels): no patient reference, case link, images or free-text history ever reach the desk."""
    __tablename__ = "medinfo_requests"
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))          # requesting organisation
    requester_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    desk_tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))     # the manufacturer's desk
    drug: Mapped[str] = mapped_column(String(60))
    therapy_class: Mapped[str] = mapped_column(String(60), default="")
    question: Mapped[str] = mapped_column(Text)
    context: Mapped[str] = mapped_column(String(400), default="")
    status: Mapped[str] = mapped_column(String(20), default="Submitted")     # Submitted | Answered
    answer: Mapped[str] = mapped_column(Text, default="")
    answer_source: Mapped[str] = mapped_column(String(200), default="")
    answered_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    requester: Mapped["User"] = relationship(foreign_keys=[requester_id])


class ScreenResult(Base):
    """The latest analysis of a held Screen token (kept 1 hour like its photos), so reports need no re-run."""
    __tablename__ = "screen_results"
    token: Mapped[str] = mapped_column(String(40), primary_key=True)
    tenant_id: Mapped[int] = mapped_column(Integer)
    user_id: Mapped[int] = mapped_column(Integer)
    result: Mapped[dict] = mapped_column(JSON)
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
