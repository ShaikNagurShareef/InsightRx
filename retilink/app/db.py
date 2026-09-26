import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

_DEFAULT_ROOT = "/data/users3/nshaik3/Projects/Oculomics/RetiLink/app_store"
APP_ROOT = os.environ.get("RETILINK_APP_ROOT", _DEFAULT_ROOT if os.path.isdir(os.path.dirname(_DEFAULT_ROOT))
                          else os.path.join(os.path.expanduser("~"), ".retilink"))
os.makedirs(os.path.join(APP_ROOT, "images"), exist_ok=True)
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(APP_ROOT, 'retilink.db')}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
