"""
Seed the Insight Rx demo workspace (see insightrx/app/seed.py).

  python scripts/seed_demo.py --reset [--no-images]

--no-images for any public deployment: scenario images come from the credentialed mBRSET test split.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from insightrx.app.db import Base, SessionLocal, engine  # noqa: E402
from insightrx.app.seed import seed  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    if args.reset:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        if not seed(db, with_images=not args.no_images):
            print("already seeded (use --reset)")
