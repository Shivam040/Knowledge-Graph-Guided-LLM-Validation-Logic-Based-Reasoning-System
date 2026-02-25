from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple

import pandas as pd

from kdsh.pipeline.retrieval.bm25_retriever import BM25Retriever
from kdsh.pipeline.retrieval.pathway_retriever import PathwayRetriever


def _canonicalize_book_names(claims_df: pd.DataFrame, chunks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure claims_df.book_name matches chunks_df.book_name (case/whitespace robust).

    This prevents a common failure mode where retrieval returns k_found=0 for an entire book
    because claims use a different capitalization/spelling than chunks.
    """
    if "book_name" not in claims_df.columns or "book_name" not in chunks_df.columns:
        return claims_df

    # Canonical map from chunks (treat chunks.csv as source-of-truth)
    canon: dict[str, str] = {}
    for bn in chunks_df["book_name"].dropna().astype(str).map(str.strip).unique():
        canon[bn.lower()] = bn

    def fix(x: Any) -> Any:
        s = str(x).strip()
        return canon.get(s.lower(), s)

    out = claims_df.copy()
    out["book_name"] = out["book_name"].apply(fix)
    return out


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
    claim_rows = [
        json.loads(line)
        for line in claims_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    claims_df = pd.DataFrame(claim_rows)

    # Fix: robust book_name alignment to chunks.csv
    claims_df = _canonicalize_book_names(claims_df, chunks_df)

    # Make "K smaller than expected" obvious
    if max_per_chapter < K:
        print(
            f"[step3] WARNING: max_per_chapter({max_per_chapter}) < K({K}). "
            f"You may get <= {max_per_chapter} candidates/claim when evidence is concentrated in one chapter."
        )

    if backend == "pathway":
        retriever = PathwayRetriever(K, candidate_pool, max_per_chapter, enforce_buckets)
    else:
        retriever = BM25Retriever(K, candidate_pool, max_per_chapter, enforce_buckets)

    retriever.build(chunks_df)

    retrieval_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for _, crow in claims_df.iterrows():
        ev = retriever.retrieve(crow.to_dict())

        coverage_rows.append(
            dict(
                id=int(crow["id"]),
                claim_id=crow["claim_id"],
                book_name=crow.get("book_name", ""),
                k_found=int(len(ev)),
                buckets_covered=int(len({e.time_bucket for e in ev})) if ev else 0,
                unique_chapters=int(len({e.chapter_id for e in ev})) if ev else 0,
                split=crow.get("split", ""),
                backend=backend,
                K=int(K),
                candidate_pool=int(candidate_pool),
                max_per_chapter=int(max_per_chapter),
                enforce_buckets=bool(enforce_buckets),
                run_id=run_id,
            )
        )

        for e in ev:
            retrieval_rows.append(
                dict(
                    id=int(crow["id"]),
                    claim_id=crow["claim_id"],
                    book_name=crow.get("book_name", ""),
                    chunk_id=e.chunk_id,
                    chapter_id=e.chapter_id,
                    chunk_pos=float(e.chunk_pos),
                    time_bucket=e.time_bucket,
                    retrieval_stage=("PATHWAY" if backend == "pathway" else "BM25"),
                    score_vec=e.score_vec,
                    score_lex=e.score_lex,
                    score_rerank=e.score_rerank,
                    score_lex_norm=e.score_lex_norm,
                    rank=int(e.rank),
                    # Fix: do NOT hardcode. Use retriever-provided flag if present.
                    is_disjoint=bool(getattr(e, "is_disjoint", False)),
                    run_id=run_id,
                )
            )

    retrieval_df = pd.DataFrame(retrieval_rows)
    coverage_df = pd.DataFrame(coverage_rows)

    # Deterministic ordering for easier debugging/diffs
    if not retrieval_df.empty and "rank" in retrieval_df.columns:
        retrieval_df = retrieval_df.sort_values(["claim_id", "rank"], kind="mergesort")

    retrieval_path = out_silver / "retrieval_candidates.csv"
    coverage_path = out_silver / "retrieval_coverage.csv"
    retrieval_df.to_csv(retrieval_path, index=False)
    coverage_df.to_csv(coverage_path, index=False)
    return retrieval_path, coverage_path
