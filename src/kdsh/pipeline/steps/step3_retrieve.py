from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from kdsh.pipeline.retrieval.bm25_retriever import BM25Retriever
from kdsh.pipeline.retrieval.pathway_retriever import PathwayRetriever

def step3_retrieve(
    chunks_df: pd.DataFrame,
    claims_path: Path,
    out_silver: Path,
    run_id: str,
    backend: str = "bm25",
    K: int = 12,
    candidate_pool: int = 80,
    max_per_chapter: int = 12,
    enforce_buckets: bool = False,
) -> Tuple[Path, Path]:
    claim_rows = [json.loads(line) for line in claims_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    claims_df = pd.DataFrame(claim_rows)

    if backend == "pathway":
        retriever = PathwayRetriever(K, candidate_pool, max_per_chapter, enforce_buckets)
    else:
        retriever = BM25Retriever(K, candidate_pool, max_per_chapter, enforce_buckets)

    retriever.build(chunks_df)

    retrieval_rows = []
    coverage_rows = []

    for _, crow in claims_df.iterrows():
        ev = retriever.retrieve(crow.to_dict())

        coverage_rows.append(
            dict(
                id=int(crow["id"]),
                claim_id=crow["claim_id"],
                book_name=crow["book_name"],
                k_found=int(len(ev)),
                buckets_covered=int(len({e.time_bucket for e in ev})) if ev else 0,
                unique_chapters=int(len({e.chapter_id for e in ev})) if ev else 0,
                split=crow["split"],
                run_id=run_id,
            )
        )

        for e in ev:
            retrieval_rows.append(
                dict(
                    id=int(crow["id"]),
                    claim_id=crow["claim_id"],
                    book_name=crow["book_name"],
                    chunk_id=e.chunk_id,
                    chapter_id=e.chapter_id,
                    chunk_pos=float(e.chunk_pos),
                    time_bucket=e.time_bucket,
                    retrieval_stage=("PATHWAY" if backend=="pathway" else "BM25"),
                    score_vec=e.score_vec,
                    score_lex=e.score_lex,
                    score_rerank=e.score_rerank,
                    score_lex_norm=e.score_lex_norm,
                    rank=e.rank,
                    is_disjoint=True,
                    run_id=run_id,
                )
            )

    retrieval_df = pd.DataFrame(retrieval_rows)
    coverage_df = pd.DataFrame(coverage_rows)

    retrieval_path = out_silver / "retrieval_candidates.csv"
    coverage_path = out_silver / "retrieval_coverage.csv"
    retrieval_df.to_csv(retrieval_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    return retrieval_path, coverage_path
