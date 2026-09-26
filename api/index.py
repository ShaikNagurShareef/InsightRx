"""Vercel entry point: the RetiLink web app (models run on a separate vision worker)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("RETILINK_VISION", "remote")
os.environ.setdefault("RETILINK_APP_ROOT", "/tmp/retilink")          # explanation cache only; data lives in Postgres
os.environ.setdefault("RETILINK_MAX_IMAGE_MB", "4")                  # Vercel request body limit is 4.5 MB

from retilink.app.main import app  # noqa: E402,F401
