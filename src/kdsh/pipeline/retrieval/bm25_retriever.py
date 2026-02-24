from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from kdsh.common.utils import tokenize
from kdsh.pipeline.retrieval.bm25 import bm25_prepare, bm25_scores
from kdsh.pipeline.retrieval.interface import EvidenceChunk, Retriever
from kdsh.pipeline.retrieval.select import select_with_constraints

def parse_predicate(p: str) -> Optional[Tuple[str, List[str]]]:
    import re
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$", str(p))
    if not m:
        return None
    name = m.group(1)
    inside = m.group(2)
    args = [a.strip().strip('"').strip("'") for a in inside.split(",")]
    return name, args

def build_query_tokens(claim_row: Dict[str, Any]) -> List[str]:
    q = []
    q += tokenize(claim_row.get("claim_text", ""))
    for kw in claim_row.get("keywords", []) or []:
        q += tokenize(kw)
    q += tokenize(claim_row.get("char", ""))
    for p in claim_row.get("predicate_form", []) or []:
        parsed = parse_predicate(p)
        if not parsed:
            continue
        _name, args = parsed
        for a in args:
            q += tokenize(a)
    return q

class BM25Retriever:
    def __init__(self, k: int = 12, candidate_pool: int = 80, max_per_chapter: int = 2, enforce_buckets: bool = True):
        self.k = k
        self.candidate_pool = candidate_pool
        self.max_per_chapter = max_per_chapter
        self.enforce_buckets = enforce_buckets
        self.book_corp: Dict[str, Dict[str, Any]] = {}

    def build(self, chunks_df: pd.DataFrame) -> None:
        self.book_corp = {}
        for book, g in chunks_df.groupby("book_name"):
            g = g.reset_index(drop=True)
            docs_tokens = [tokenize(t) for t in g["chunk_text"].tolist()]
            prep = bm25_prepare(docs_tokens)
            self.book_corp[book] = {"chunks": g, "docs_tokens": docs_tokens, "prep": prep}

    def retrieve(self, claim_row: Dict[str, Any]) -> List[EvidenceChunk]:
        book = claim_row["book_name"]
        corp = self.book_corp.get(book)
        if corp is None:
            return []

        q_tokens = build_query_tokens(claim_row)
        scores = bm25_scores(q_tokens, corp["docs_tokens"], corp["prep"])
        g = corp["chunks"].copy()
        g["score_lex"] = scores
        g = g.sort_values("score_lex", ascending=False).head(self.candidate_pool)
        max_s = float(g["score_lex"].max()) if len(g) else 0.0
        min_s = float(g["score_lex"].min()) if len(g) else 0.0
        g["score_lex_norm"] = (g["score_lex"] - min_s) / (max_s - min_s) if max_s > min_s else 0.0

        selected = select_with_constraints(g, self.k, max_per_chapter=self.max_per_chapter, enforce_buckets=self.enforce_buckets)
        out: List[EvidenceChunk] = []
        for rank, (_, r) in enumerate(selected.iterrows(), start=1):
            out.append(
                EvidenceChunk(
                    chunk_id=str(r["chunk_id"]),
                    chapter_id=str(r["chapter_id"]),
                    chunk_pos=float(r["chunk_pos"]),
                    time_bucket=str(r["time_bucket"]),
                    text=str(r["chunk_text"]),
                    score_lex=float(r["score_lex"]),
                    score_lex_norm=float(r["score_lex_norm"]) if r.get("score_lex_norm") is not None else None,
                    rank=int(rank),
                )
            )
        return out
