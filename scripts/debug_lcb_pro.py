#!/usr/bin/env python3
"""Debug lcb_pro dataset — show actual field values."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset

TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
DATASET = "QAQAQAQAQ/LiveCodeBench-Pro"
SPLIT = "biannual_2025_1_6"

print(f"Loading {DATASET} split={SPLIT} ...")
ds = load_dataset(DATASET, split=SPLIT, token=TOKEN, trust_remote_code=True)
rows = list(ds)
print(f"Total rows: {len(rows)}")
if rows:
    print(f"Keys: {list(rows[0].keys())}")
    difficulties = sorted(set(r.get("difficulty", "N/A") for r in rows))
    print(f"Distinct difficulty values: {difficulties}")
    print(f"\nFirst row sample:")
    r = rows[0]
    for k in ["question_title", "platform", "difficulty", "question_content"]:
        v = str(r.get(k, "MISSING"))
        print(f"  {k}: {v[:120]}")
