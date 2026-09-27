"""Tiger Data trends: de-identified finding events, fallback to the app database, never breaking a request."""
from datetime import datetime, timedelta, timezone

from test_workflow import client_as, fundus_bytes, new_case, sign, upload  # noqa: E402  (seeds a temp workspace)

from insightrx.app import timeseries as ts  # noqa: E402

RESULT = {"overall": "Referable DR signal",
          "eyes": {"OD": {"edema_signal": True}, "OS": {"edema_signal": False}},
          "systemic": {"nephropathy": {"status": "Research signal"}, "obesity": {"status": "No research signal"},
                       "neuropathy": {"status": "Exploratory signal"}}}


def test_findings_are_targets_only():
    assert ts.findings(RESULT) == ["screened", "referable_dr", "macular_edema", "nephropathy", "neuropathy"]
    assert ts.findings({"overall": "Unable to assess"}) == [] and ts.findings(None) == []


def test_series_places_counts_by_day():
    today = datetime.now(timezone.utc)
    s = ts._series([(today, "screened", 2), (today - timedelta(days=2), "screened", 1), (today - timedelta(days=40), "x", 9)], 30)
    assert s["screened"][-1] == 2 and s["screened"][-3] == 1 and "x" not in s


def test_disabled_writes_nothing(monkeypatch):
    monkeypatch.delenv("TIGER_DATABASE_URL", raising=False)
    assert ts.record_findings(1, RESULT) is False and ts.record_workflow(1, "review_signed") is False


def test_unreachable_tiger_never_raises(monkeypatch):
    monkeypatch.setenv("TIGER_DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
    assert ts.record_findings(1, RESULT) is False


def test_writes_go_to_the_hypertable(monkeypatch):
    rows, stmts = [], []

    class Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def executemany(self, sql, data):
            rows.extend(data)

    class Conn(Cur):
        def execute(self, sql, *a):
            stmts.append(sql)

        def cursor(self):
            return Cur()

    monkeypatch.setenv("TIGER_DATABASE_URL", "postgresql://fake")
    monkeypatch.setattr(ts, "_connect", Conn)
    monkeypatch.setattr(ts, "_ready", False)
    assert ts.record_findings(7, RESULT)
    assert {r[3] for r in rows} == set(ts.findings(RESULT)) and all(r[1] == 7 and r[2] == "finding" for r in rows)
    assert any("create_hypertable" in s for s in stmts) and any("timescaledb.continuous" in s for s in stmts)


def test_trends_page_falls_back_to_app_database(monkeypatch):
    monkeypatch.delenv("TIGER_DATABASE_URL", raising=False)
    op, pcp = client_as("Sam Rivera"), client_as("Dr. Alex Morgan")
    cid = new_case(op, "RL-TREND1")
    upload(op, cid, "OD", fundus_bytes(51))
    op.post(f"/cases/{cid}/analyze")
    sign(pcp, cid)
    page = pcp.get("/performance?view=trends")
    assert page.status_code == 200 and "Patients screened" in page.text and "Reviews signed" in page.text
    assert "set TIGER_DATABASE_URL" in page.text
