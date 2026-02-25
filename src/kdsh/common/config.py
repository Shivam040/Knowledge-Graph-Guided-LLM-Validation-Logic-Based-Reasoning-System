from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os
import yaml


@dataclass
class ChunkingConfig:
    window_words: int = 300
    overlap_words: int = 50
    min_chunk_words: int = 120
    chapter_regex: str = r"(?im)^(chapter|chapitre)\b.*$"

    # Optional: if you applied the Step-1 pronoun-hint patch
    # (it does NOT modify chunk_text; only adds chunk_text_nli hint)
    pronoun_hint: bool = False  # maps to env KDSH_CHUNK_PRONOUN_HINT=1


@dataclass
class RetrievalConfig:
    k: int = 12
    candidate_pool: int = 80
    max_per_chapter: int = 5
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
    device: str = "auto"  # auto|cpu|cuda|cuda:0

    # Evidence scoring mix: (1-alpha)*nli + alpha*retrieval_score_norm
    score_mix_alpha: float = 0.15

    # --- NEW: sentence-level MNLI controls (used by patched Step 4) ---
    sentence_level: bool = True
    top_sentences: int = 5

    # --- NEW: label decision thresholds ---
    entail_threshold: float = 0.55
    contra_threshold: float = 0.80
    nli_margin: float = 0.05

    # --- NEW: gating + fallbacks ---
    enable_alias_gate: bool = False
    enable_contra_alias_gate: bool = True
    emit_supports_claim_fallback: bool = True

    # Facts emission filter
    min_fact_conf: float = 0.40


@dataclass
class KGConfig:
    min_fact_conf: float = 0.40


@dataclass
class LogicConfig:
    # Step-6 Claim-edge logic knobs (your new Step-6 uses env vars)
    use_nli: bool = True
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    contra_thr: float = 0.85
    entail_thr: float = 0.60
    margin: float = 0.05
    max_pairs_per_id: int = 400
    min_shared_tokens: int = 2
    min_jaccard: float = 0.20


@dataclass
class AggregationConfig:
    contradiction_penalty: float = 3.0

    # Step-7 v2 knobs (your patched Step-7 reads env vars)
    neutral_alpha: float = 0.15
    mode: str = "balanced"  # balanced | high_recall
    min_precision: float = 0.55  # only used in high_recall


@dataclass
class AppConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    kg: KGConfig = field(default_factory=KGConfig)
    logic: LogicConfig = field(default_factory=LogicConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)


def load_config(path: Optional[Path]) -> AppConfig:
    cfg = AppConfig()

    # YAML overrides
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

    # ---------------------------
    # Env overrides (quick tests)
    # ---------------------------
    # Retrieval
    backend = os.getenv("KDSH_RETRIEVER_BACKEND")
    if backend:
        cfg.retrieval.backend = backend.strip().lower()

    # Verifier
    vbackend = os.getenv("KDSH_VERIFIER_BACKEND")
    if vbackend:
        cfg.verifier.backend = vbackend.strip().lower()

    vmodel = os.getenv("KDSH_NLI_MODEL")
    if vmodel:
        cfg.verifier.model_name = vmodel.strip()

    vdevice = os.getenv("KDSH_NLI_DEVICE")
    if vdevice:
        # IMPORTANT: accept "cuda:0" and do NOT force numeric "0"
        cfg.verifier.device = vdevice.strip()

    vbs = os.getenv("KDSH_NLI_BATCH")
    if vbs:
        cfg.verifier.batch_size = int(vbs)

    vmax = os.getenv("KDSH_NLI_MAXLEN")
    if vmax:
        cfg.verifier.max_length = int(vmax)

    vtop = os.getenv("KDSH_NLI_TOP_SENTS")
    if vtop:
        cfg.verifier.top_sentences = int(vtop)

    # Threshold tuning
    ent = os.getenv("KDSH_NLI_ENTAIL_THR")
    if ent:
        cfg.verifier.entail_threshold = float(ent)

    con = os.getenv("KDSH_NLI_CONTRA_THR")
    if con:
        cfg.verifier.contra_threshold = float(con)

    mar = os.getenv("KDSH_NLI_MARGIN")
    if mar:
        cfg.verifier.nli_margin = float(mar)

    # Step-1 pronoun hint (if using patched step1)
    ph = os.getenv("KDSH_CHUNK_PRONOUN_HINT")
    if ph:
        cfg.chunking.pronoun_hint = (ph.strip() == "1")

    # Step-7 env passthrough (patched Step-7 reads env)
    # If you set YAML values, we also export env so Step-7 picks them up.
    os.environ.setdefault("KDSH_STEP7_NEUTRAL_ALPHA", str(cfg.aggregation.neutral_alpha))
    os.environ.setdefault("KDSH_STEP7_MODE", str(cfg.aggregation.mode))
    os.environ.setdefault("KDSH_STEP7_MIN_PRECISION", str(cfg.aggregation.min_precision))

    # Step-6 env passthrough (patched Step-6 reads env)
    os.environ.setdefault("KDSH_STEP6_USE_NLI", "1" if cfg.logic.use_nli else "0")
    os.environ.setdefault("KDSH_STEP6_NLI_MODEL", str(cfg.logic.nli_model))
    os.environ.setdefault("KDSH_STEP6_CONTRA_THR", str(cfg.logic.contra_thr))
    os.environ.setdefault("KDSH_STEP6_ENTAIL_THR", str(cfg.logic.entail_thr))
    os.environ.setdefault("KDSH_STEP6_MARGIN", str(cfg.logic.margin))
    os.environ.setdefault("KDSH_STEP6_MAX_PAIRS_PER_ID", str(cfg.logic.max_pairs_per_id))
    os.environ.setdefault("KDSH_STEP6_MIN_SHARED_TOKENS", str(cfg.logic.min_shared_tokens))
    os.environ.setdefault("KDSH_STEP6_MIN_JACCARD", str(cfg.logic.min_jaccard))

    return cfg

# from __future__ import annotations

# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Any, Dict, Optional
# import os
# import yaml

# @dataclass
# class ChunkingConfig:
#     window_words: int = 300 # 350
#     overlap_words: int = 50 # 50
#     min_chunk_words: int = 120 # 120
#     chapter_regex: str = r"(?im)^(chapter|chapitre)\b.*$"

# @dataclass
# class RetrievalConfig:
#     k: int = 12
#     candidate_pool: int = 80
#     max_per_chapter: int = 5
#     enforce_buckets: bool = True
#     backend: str = "bm25"  # bm25 | pathway

# @dataclass
# class VerifierConfig:
#     # heuristic | mnli
#     backend: str = "heuristic"
#     # MNLI options (used when backend == "mnli")
#     model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
#     batch_size: int = 16
#     max_length: int = 512
#     device: str = "auto"  # auto|cpu|cuda
#     # Confidence mixing: (1-alpha)*nli + alpha*retrieval_score_norm
#     score_mix_alpha: float = 0.15

#     min_fact_conf: float = 0.40

# @dataclass
# class KGConfig:
#     min_fact_conf: float = 0.40

# @dataclass
# class AggregationConfig:
#     contradiction_penalty: float = 2.0

# @dataclass
# class AppConfig:
#     chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
#     retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
#     verifier: VerifierConfig = field(default_factory=VerifierConfig)
#     kg: KGConfig = field(default_factory=KGConfig)
#     aggregation: AggregationConfig = field(default_factory=AggregationConfig)


# def load_config(path: Optional[Path]) -> AppConfig:
#     cfg = AppConfig()
#     if path and path.exists():
#         data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
#         for k, v in data.items():
#             if not hasattr(cfg, k):
#                 continue
#             section = getattr(cfg, k)
#             if isinstance(v, dict):
#                 for kk, vv in v.items():
#                     if hasattr(section, kk):
#                         setattr(section, kk, vv)

#     # Env overrides (quick experiments)
#     backend = os.getenv("KDSH_RETRIEVER_BACKEND")
#     if backend:
#         cfg.retrieval.backend = backend.strip().lower()

#     vbackend = os.getenv("KDSH_VERIFIER_BACKEND")
#     if vbackend:
#         cfg.verifier.backend = vbackend.strip().lower()

#     vmodel = os.getenv("KDSH_NLI_MODEL")
#     if vmodel:
#         cfg.verifier.model_name = vmodel.strip()

#     vdevice = os.getenv("KDSH_NLI_DEVICE")
#     if vdevice:
#         cfg.verifier.device = vdevice.strip().lower()

#     return cfg
