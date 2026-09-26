"""Retrieval over a curated, versioned reference set (no browsing, no case-data indexing)."""
import json
import os
import re

REFS = json.load(open(os.path.join(os.path.dirname(__file__), "data", "references.json")))


def retrieve(question: str, topics=(), k=3):
    words = set(re.findall(r"[a-z_]+", question.lower()))
    scored = []
    for r in REFS:
        s = 3 * len(set(topics) & set(r["topics"])) + len(words & set(r["topics"]))
        if s > 0:
            scored.append((s, r))
    scored.sort(key=lambda t: -t[0])
    return [r for _, r in scored[:k]]
