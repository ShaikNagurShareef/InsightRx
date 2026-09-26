import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

_DEFAULT_ROOT = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/app_store"
APP_ROOT = os.environ.get("INSIGHTRX_APP_ROOT", _DEFAULT_ROOT if os.path.isdir(os.path.dirname(_DEFAULT_ROOT))
                          else os.path.join(os.path.expanduser("~"), ".insightrx"))
os.makedirs(os.path.join(APP_ROOT, "images"), exist_ok=True)
DATABASE_URL = (os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
                or f"sqlite:///{os.path.join(APP_ROOT, 'retilink.db')}")
if DATABASE_URL.startswith(("postgres://", "postgresql://")):          # Neon / Vercel Postgres URLs
    DATABASE_URL = "postgresql+psycopg://" + DATABASE_URL.split("://", 1)[1]
IMAGE_STORE = os.environ.get("INSIGHTRX_IMAGE_STORE", "db" if DATABASE_URL.startswith("postgresql") else "fs")

engine = create_engine(DATABASE_URL, pool_pre_ping=True,
                       connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_columns():
    """Additive, idempotent schema upgrades for databases created before a column existed (e.g. Neon in production)."""
    from sqlalchemy import inspect, text
    added = {"patients": {"medications": "JSON"}}
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, cols in added.items():
            if not insp.has_table(table):
                continue
            have = {c["name"] for c in insp.get_columns(table)}
            for col, typ in cols.items():
                if col not in have:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {typ}"))
