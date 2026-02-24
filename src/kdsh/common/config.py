from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml

@dataclass
class ChunkingConfig:
    window_words: int = 350
    overlap_words: int = 50
    min_chunk_words: int = 120
    chapter_regex: str = r"(?im)^(chapter|chapitre)\b.*$"

@dataclass
class RetrievalConfig:
    k: int = 12
    candidate_pool: int = 80
    max_per_chapter: int = 2
    enforce_buckets: bool = True
    backend: str = "bm25"  # bm25 | pathway

@dataclass
class VerifierConfig:
    # heuristic | mnli
    backend: str = "heuristic"
    # MNLI options (used when backend == "mnli")
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    batch_size: int = 16
    max_length: int = 512
    device: str = "auto"  # auto|cpu|cuda
    # Confidence mixing: (1-alpha)*nli + alpha*retrieval_score_norm
    score_mix_alpha: float = 0.15

    min_fact_conf: float = 0.40

@dataclass
class KGConfig:
    min_fact_conf: float = 0.40

@dataclass
class AggregationConfig:
    contradiction_penalty: float = 2.0

@dataclass
class AppConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    kg: KGConfig = field(default_factory=KGConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)


def load_config(path: Optional[Path]) -> AppConfig:
    cfg = AppConfig()
    if path and path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for k, v in data.items():
            if not hasattr(cfg, k):
                continue
            section = getattr(cfg, k)
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if hasattr(section, kk):
                        setattr(section, kk, vv)

    # Env overrides (quick experiments)
    backend = os.getenv("KDSH_RETRIEVER_BACKEND")
    if backend:
        cfg.retrieval.backend = backend.strip().lower()

    vbackend = os.getenv("KDSH_VERIFIER_BACKEND")
    if vbackend:
        cfg.verifier.backend = vbackend.strip().lower()

    vmodel = os.getenv("KDSH_NLI_MODEL")
    if vmodel:
        cfg.verifier.model_name = vmodel.strip()

    vdevice = os.getenv("KDSH_NLI_DEVICE")
    if vdevice:
        cfg.verifier.device = vdevice.strip().lower()

    return cfg

# def load_config(path: Optional[Path]) -> AppConfig:
#     cfg = AppConfig()
#     if not path:
#         backend = os.getenv("KDSH_RETRIEVER_BACKEND")
#         if backend:
#             cfg.retrieval.backend = backend.strip().lower()
#         return cfg

#     raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
#     if "chunking" in raw:
#         for k, v in raw["chunking"].items():
#             setattr(cfg.chunking, k, v)
#     if "retrieval" in raw:
#         for k, v in raw["retrieval"].items():
#             setattr(cfg.retrieval, k, v)
#     if "verifier" in raw:
#         for k, v in raw["verifier"].items():
#             setattr(cfg.verifier, k, v)
#     if "kg" in raw:
#         for k, v in raw["kg"].items():
#             setattr(cfg.kg, k, v)
#     if "aggregation" in raw:
#         for k, v in raw["aggregation"].items():
#             setattr(cfg.aggregation, k, v)

#     backend = os.getenv("KDSH_RETRIEVER_BACKEND")
#     if backend:
#         cfg.retrieval.backend = backend.strip().lower()
#     return cfg
