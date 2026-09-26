"""Insight Rx (formerly RetiLink). Legacy RETILINK_* environment variables are honoured as INSIGHTRX_* so existing
deployments and worker scripts keep working until their settings are renamed."""
import os

for _k, _v in list(os.environ.items()):
    if _k.startswith("RETILINK_"):
        os.environ.setdefault("INSIGHTRX_" + _k[len("RETILINK_"):], _v)
