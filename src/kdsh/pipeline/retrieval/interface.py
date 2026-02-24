from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    chapter_id: str
    chunk_pos: float
    time_bucket: str
    text: str
    score_lex: Optional[float] = None
    score_vec: Optional[float] = None
    score_rerank: Optional[float] = None
    score_lex_norm: Optional[float] = None
    rank: Optional[int] = None

class Retriever(Protocol):
    def build(self, chunks_df) -> None:
        ...

    def retrieve(self, claim_row: Dict[str, Any]) -> List[EvidenceChunk]:
        ...
