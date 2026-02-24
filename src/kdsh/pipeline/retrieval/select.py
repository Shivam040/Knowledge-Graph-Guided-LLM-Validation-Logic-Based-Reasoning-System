from __future__ import annotations

from collections import defaultdict
import pandas as pd

def select_with_constraints(cand_df: pd.DataFrame, K: int, max_per_chapter: int = 2, enforce_buckets: bool = True) -> pd.DataFrame:
    selected = []
    chapter_counts = defaultdict(int)

    if enforce_buckets:
        for b in ["EARLY", "MID", "LATE"]:
            for _, r in cand_df.iterrows():
                if r["time_bucket"] != b:
                    continue
                ch = r["chapter_id"]
                if chapter_counts[ch] >= max_per_chapter:
                    continue
                selected.append(r)
                chapter_counts[ch] += 1
                break

    for _, r in cand_df.iterrows():
        if len(selected) >= K:
            break
        if any(r["chunk_id"] == s["chunk_id"] for s in selected):
            continue
        ch = r["chapter_id"]
        if chapter_counts[ch] >= max_per_chapter:
            continue
        selected.append(r)
        chapter_counts[ch] += 1

    return pd.DataFrame(selected)
