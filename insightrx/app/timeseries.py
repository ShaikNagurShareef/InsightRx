"""
Finding and engagement trends on Tiger Data (TimescaleDB), MLH "Best Use of Tiger Data".

Every completed screening emits one de-identified event per finding (referable DR, macular edema, each systemic
research signal) and every signed review or sent consultation emits a workflow event. They land in a hypertable, and a
real-time continuous aggregate rolls them up per day. This is the "eye-detected demand by target over time" series the
life-science story rests on.

    signal_events(time, tenant_id, kind, target)            hypertable, partitioned by time
    signal_daily(day, tenant_id, kind, target, n)           continuous aggregate, refreshed every 15 min + real time

  TIGER_DATABASE_URL   postgres://... of a Tiger Cloud service; unset = trends are computed from the app database
  python -m insightrx.app.timeseries backfill              load past screenings from the app database

Events carry no patient, case or image identifiers. Writing never blocks clinical work: failures are logged.
"""
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

log = logging.getLogger("insightrx.timeseries")
CONNECT_TIMEOUT_S = 4
WORKFLOW = ("review_signed", "referral_sent")
SIGNALS = ("Research signal", "Exploratory signal")
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS signal_events (
         time timestamptz NOT NULL, tenant_id integer NOT NULL, kind text NOT NULL, target text NOT NULL)""",
    "SELECT create_hypertable('signal_events', by_range('time', INTERVAL '7 days'), if_not_exists => TRUE)",
    """CREATE MATERIALIZED VIEW IF NOT EXISTS signal_daily
         WITH (timescaledb.continuous, timescaledb.materialized_only = false) AS
         SELECT time_bucket(INTERVAL '1 day', time) AS day, tenant_id, kind, target, count(*) AS n
         FROM signal_events GROUP BY 1, 2, 3, 4 WITH NO DATA""",
    """SELECT add_continuous_aggregate_policy('signal_daily', start_offset => INTERVAL '90 days',
         end_offset => INTERVAL '1 hour', schedule_interval => INTERVAL '15 minutes', if_not_exists => TRUE)""",
]
_ready = False


def enabled() -> bool:
    return bool(os.environ.get("TIGER_DATABASE_URL"))


def findings(result: dict | None) -> list[str]:
    """Targets a completed analysis contributes: screened, referable_dr, macular_edema, systemic keys with a signal."""
    if not result or result.get("overall") in (None, "Unable to assess"):
        return []
    out = ["screened"]
    if result.get("overall") == "Referable DR signal":
        out.append("referable_dr")
    if any((result.get("eyes", {}).get(e) or {}).get("edema_signal") for e in ("OD", "OS")):
        out.append("macular_edema")
    out += sorted(k for k, v in (result.get("systemic") or {}).items() if v.get("status") in SIGNALS)
    return out


def _connect():
    import psycopg
    return psycopg.connect(os.environ["TIGER_DATABASE_URL"], connect_timeout=CONNECT_TIMEOUT_S, autocommit=True)


def setup(conn) -> None:
    global _ready
    if not _ready:
        for stmt in SCHEMA:
            conn.execute(stmt)
        _ready = True


def _write(rows: list[tuple]) -> bool:
    if not (enabled() and rows):
        return False
    try:
        with _connect() as conn:
            setup(conn)
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO signal_events (time, tenant_id, kind, target) VALUES (%s, %s, %s, %s)", rows)
        return True
    except Exception as e:                                   # noqa: BLE001 - analytics must never break a request
        log.warning("tiger write failed: %s", e)
        return False


def record_findings(tenant_id: int, result: dict | None, at: datetime | None = None) -> bool:
    at = at or datetime.now(timezone.utc)
    return _write([(at, tenant_id, "finding", t) for t in findings(result)])


def record_workflow(tenant_id: int, action: str) -> bool:
    return action in WORKFLOW and _write([(datetime.now(timezone.utc), tenant_id, "workflow", action)])


def _series(rows, days: int) -> dict:
    """[(day, target, n)] -> {target: [n per day, oldest first]}."""
    today = datetime.now(timezone.utc).date()
    out = defaultdict(lambda: [0] * days)
    for day, target, n in rows:
        i = days - 1 - (today - (day.date() if isinstance(day, datetime) else day)).days
        if 0 <= i < days:
            out[target][i] += int(n)
    return dict(out)


def trends(db, tenant_id: int, days: int = 30) -> dict:
    """{source, series: {target: [daily counts]}} from Tiger when configured, else from the app database."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    if enabled():
        try:
            with _connect() as conn:
                setup(conn)
                rows = conn.execute("SELECT day, target, sum(n) FROM signal_daily WHERE tenant_id = %s AND day >= %s "
                                    "GROUP BY 1, 2", (tenant_id, since)).fetchall()
            return {"source": "Tiger Data continuous aggregate", "series": _series(rows, days)}
        except Exception as e:                               # noqa: BLE001
            log.warning("tiger query failed: %s", e)
    from sqlalchemy import select
    from .models import AuditEvent, ModelRun
    rows = [(r.created_at, t, 1) for r in db.scalars(select(ModelRun).where(
        ModelRun.tenant_id == tenant_id, ModelRun.status == "completed", ModelRun.created_at >= since))
        for t in findings(r.result)]
    rows += [(e.created_at, e.action, 1) for e in db.scalars(select(AuditEvent).where(
        AuditEvent.tenant_id == tenant_id, AuditEvent.action.in_(WORKFLOW), AuditEvent.created_at >= since))]
    return {"source": "app database (set TIGER_DATABASE_URL for Tiger Data)", "series": _series(rows, days)}


def backfill() -> int:
    """Load every completed screening and workflow event already in the app database into Tiger."""
    from sqlalchemy import select
    from .db import SessionLocal
    from .models import AuditEvent, ModelRun
    with SessionLocal() as db:
        rows = [(r.created_at, r.tenant_id, "finding", t)
                for r in db.scalars(select(ModelRun).where(ModelRun.status == "completed")) for t in findings(r.result)]
        rows += [(e.created_at, e.tenant_id, "workflow", e.action)
                 for e in db.scalars(select(AuditEvent).where(AuditEvent.action.in_(WORKFLOW)))]
    return len(rows) if _write(rows) else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["backfill"]:
        if not enabled():
            sys.exit("TIGER_DATABASE_URL is not set.")
        print(f"backfilled {backfill()} events")
