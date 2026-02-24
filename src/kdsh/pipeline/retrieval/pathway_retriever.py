# from __future__ import annotations

# from typing import Any, Dict, List, Optional
# import warnings
# import pandas as pd

# from kdsh.pipeline.retrieval.interface import EvidenceChunk
# from kdsh.pipeline.retrieval.bm25_retriever import BM25Retriever

# class PathwayRetriever:
#     """
#     Pathway-backed retriever adapter.

#     MVP design:
#     - If Pathway is installed, you can replace the internals of `build()` + `retrieve()` to query
#       a Pathway Vector Store / index (with metadata filters like book_name and time_bucket).
#     - Until then, this adapter **falls back to BM25** so the system stays runnable end-to-end.

#     This file exists so your system design cleanly satisfies Track A’s Pathway requirement once wired.
#     """

#     def __init__(self, k: int = 12, candidate_pool: int = 80, max_per_chapter: int = 2, enforce_buckets: bool = True):
#         self.k = k
#         self.candidate_pool = candidate_pool
#         self.max_per_chapter = max_per_chapter
#         self.enforce_buckets = enforce_buckets

#         self._bm25 = BM25Retriever(k=k, candidate_pool=candidate_pool, max_per_chapter=max_per_chapter, enforce_buckets=enforce_buckets)
#         self._chunks_df: Optional[pd.DataFrame] = None
#         self._pathway_available = False

#         try:
#             import pathway as pw  # noqa: F401
#             self._pathway_available = True
#         except Exception:
#             self._pathway_available = False

#     def build(self, chunks_df: pd.DataFrame) -> None:
#         self._chunks_df = chunks_df
#         # Always build BM25 as fallback (and as a high-recall lexical index)
#         self._bm25.build(chunks_df)

#         if not self._pathway_available:
#             warnings.warn("Pathway not installed; falling back to BM25 retriever. Install extras: `pip install -e '.[pathway]'`.")
#             return

#         # --- TODO (Hackathon wiring) ---
#         # Here you would:
#         # 1) Create/refresh a Pathway table from chunks_df with metadata columns.
#         # 2) Build a vector index (embedding model of your choice) + metadata index.
#         # 3) Optionally persist the index for reuse across runs.
#         #
#         # This repo keeps retrieval runnable even without Pathway.
#         # -------------------------------

#     def retrieve(self, claim_row: Dict[str, Any]) -> List[EvidenceChunk]:
#         if not self._pathway_available:
#             return self._bm25.retrieve(claim_row)

#         # --- TODO (Hackathon wiring) ---
#         # Use Pathway index to retrieve by:
#         # - book_name filter
#         # - query from claim_row (claim_text + keywords + predicate args)
#         # - enforce temporal buckets EARLY/MID/LATE by filtering/reranking
#         #
#         # For now, return BM25 results as a safe baseline while you implement Pathway.
#         return self._bm25.retrieve(claim_row)


from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from kdsh.pipeline.retrieval.bm25_retriever import BM25Retriever, build_query_tokens
from kdsh.pipeline.retrieval.interface import EvidenceChunk, Retriever
from kdsh.pipeline.retrieval.select import select_with_constraints

logger = logging.getLogger(__name__)


class PathwayRetriever:
    """
    Real Pathway-based lexical retrieval (Tantivy BM25) for Track A compliance.

    - Builds a Pathway table from chunks_df (via pw.debug.table_from_pandas)
    - Builds a Tantivy BM25 index with metadata_json for filtering
    - Queries per claim with metadata_filter enforcing book_name
    - Applies existing constraint-aware selection in Python
    - Falls back to BM25Retriever if Pathway isn't installed/working
    """

    def __init__(
        self,
        k: int = 12,
        candidate_pool: int = 80,
        max_per_chapter: int = 2,
        enforce_buckets: bool = True,
        strict_pathway: bool = False,
    ):
        self.k = k
        self.candidate_pool = candidate_pool
        self.max_per_chapter = max_per_chapter
        self.enforce_buckets = enforce_buckets
        self.strict_pathway = strict_pathway

        # Fallback backend (your current working retriever)
        self._fallback = BM25Retriever(
            k=k,
            candidate_pool=candidate_pool,
            max_per_chapter=max_per_chapter,
            enforce_buckets=enforce_buckets,
        )

        # Pathway objects (initialized in build)
        self._pw = None
        self._pw_index = None
        self._score_colname: Optional[str] = None

        self._pathway_ready = False

    def build(self, chunks_df: pd.DataFrame) -> None:
        """
        Build indexes.
        Always builds fallback BM25.
        Builds Pathway Tantivy BM25 index if Pathway is available.
        """
        # Always build fallback (so we can gracefully degrade)
        self._fallback.build(chunks_df)

        required = {"chunk_id", "book_name", "chapter_id", "time_bucket", "chunk_pos", "chunk_text"}
        missing = required - set(chunks_df.columns)
        if missing:
            raise ValueError(f"chunks_df missing required columns: {sorted(missing)}")

        try:
            import pathway as pw  # type: ignore
            from pathway.stdlib.indexing import DataIndex, TantivyBM25  # type: ignore
        except Exception as e:
            self._pathway_ready = False
            msg = f"Pathway not available, using BM25 fallback. Reason: {e}"
            if self.strict_pathway:
                raise RuntimeError(msg) from e
            logger.warning(msg)
            return

        # Score column name constant (best-effort)
        score_name = None
        try:
            from pathway.stdlib.indexing.colnames import _SCORE  # type: ignore

            # _SCORE is a string constant (column name)
            score_name = _SCORE
        except Exception:
            # fallback guesses; we’ll also probe runtime columns later
            score_name = "_pw_index_reply_score"

        # Build Pathway table from pandas
        df = chunks_df[
            ["chunk_id", "book_name", "chapter_id", "time_bucket", "chunk_pos", "chunk_text"]
        ].copy()

        # Metadata JSON for JMESPath filter. Use JSON-literal comparison with backticks.
        # (JMESPath spec supports literals via backticks: foo == `"bar"`)
        def _meta_row(r: pd.Series) -> str:
            meta = {
                "chunk_id": str(r["chunk_id"]),
                "book_name": str(r["book_name"]),
                "chapter_id": str(r["chapter_id"]),
                "time_bucket": str(r["time_bucket"]),
                "chunk_pos": float(r["chunk_pos"]),
            }
            return json.dumps(meta, ensure_ascii=False)

        df["metadata_json"] = df.apply(_meta_row, axis=1)

        # Define Pathway graph objects
        self._pw = pw
        chunks_tbl = pw.debug.table_from_pandas(df)

        inner = TantivyBM25(
            data_column=chunks_tbl.chunk_text,
            metadata_column=chunks_tbl.metadata_json,
        )
        self._pw_index = DataIndex(chunks_tbl, inner)

        self._score_colname = score_name
        self._pathway_ready = True
        logger.info("PathwayRetriever ready: Tantivy BM25 index built.")

    def retrieve(self, claim_row: Dict[str, Any]) -> List[EvidenceChunk]:
        """
        Retrieve EvidenceChunk list for a single claim.
        Uses Pathway if ready; else BM25 fallback.
        """
        if not self._pathway_ready or self._pw is None or self._pw_index is None:
            return self._fallback.retrieve(claim_row)

        pw = self._pw

        # Build query text (consistent with BM25Retriever token construction)
        q_tokens = build_query_tokens(claim_row)
        query_text = " ".join(t for t in q_tokens if t).strip()
        if not query_text:
            query_text = str(claim_row.get("claim_text", "")).strip()

        # Enforce book filter (Track A: scope to correct novel)
        book = str(claim_row.get("book_name", "")).strip()
        if not book:
            # If claim has no book_name, safest is fallback
            return self._fallback.retrieve(claim_row)

        # JMESPath literal comparison using backticks around JSON string literal.
        # Example: book_name == `"The Count of Monte Cristo"`
        book_json = json.dumps(book, ensure_ascii=False)
        metadata_filter = f"book_name == `{book_json}`"

        qdf = pd.DataFrame([{"query": query_text, "metadata_filter": metadata_filter}])
        qtbl = pw.debug.table_from_pandas(qdf)

        try:
            # One row per match is easier for downstream pandas selection
            res_tbl = self._pw_index.query_as_of_now(
                query_column=qtbl.query,
                number_of_matches=self.candidate_pool,
                collapse_rows=False,
                metadata_filter=qtbl.metadata_filter,
            )

            rows: List[Dict[str, Any]] = pw.debug.table_to_dicts(res_tbl)
        except Exception as e:
            msg = f"Pathway query failed; using BM25 fallback. Reason: {e}"
            if self.strict_pathway:
                raise RuntimeError(msg) from e
            logger.warning(msg)
            return self._fallback.retrieve(claim_row)

        if not rows:
            return []

        g = pd.DataFrame(rows)

        # Locate score column robustly
        score_col = None
        candidates = []
        if self._score_colname:
            candidates.append(self._score_colname)
        candidates += ["_pw_index_reply_score", "_score", "score"]
        for c in candidates:
            if c in g.columns:
                score_col = c
                break

        # Validate required chunk columns exist (Pathway join should expose them)
        needed_cols = ["chunk_id", "chapter_id", "time_bucket", "chunk_pos", "chunk_text"]
        for c in needed_cols:
            if c not in g.columns:
                msg = f"Pathway result missing column '{c}'. Columns={list(g.columns)}"
                if self.strict_pathway:
                    raise RuntimeError(msg)
                logger.warning(msg + " -> using BM25 fallback.")
                return self._fallback.retrieve(claim_row)

        if score_col is None:
            msg = f"Could not find score column in Pathway results. Columns={list(g.columns)}"
            if self.strict_pathway:
                raise RuntimeError(msg)
            logger.warning(msg + " -> using BM25 fallback.")
            return self._fallback.retrieve(claim_row)

        # Normalize and keep top candidate_pool (Pathway should already be top, but keep safe)
        g = g.copy()
        g["score_lex"] = pd.to_numeric(g[score_col], errors="coerce").fillna(0.0)
        g = g.sort_values("score_lex", ascending=False).head(self.candidate_pool)

        max_s = float(g["score_lex"].max()) if len(g) else 0.0
        min_s = float(g["score_lex"].min()) if len(g) else 0.0
        g["score_lex_norm"] = (
            (g["score_lex"] - min_s) / (max_s - min_s) if max_s > min_s else 0.0
        )

        # Apply your existing constraint-aware selector
        selected = select_with_constraints(
            g,
            self.k,
            max_per_chapter=self.max_per_chapter,
            enforce_buckets=self.enforce_buckets,
        )

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
                    score_lex_norm=float(r["score_lex_norm"])
                    if r.get("score_lex_norm") is not None
                    else None,
                    rank=int(rank),
                )
            )

        return out
