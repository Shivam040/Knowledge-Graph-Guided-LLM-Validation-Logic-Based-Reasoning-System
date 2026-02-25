# step4_verify.py (NLI-only, generalizable)
from __future__ import annotations

import os
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import regex as re2

from kdsh.common.utils import NEG_WORDS, tokenize
from kdsh.common.config import VerifierConfig


# ============================================================
# Text + alias utilities (general)
# ============================================================

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm_text(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    # normalize common Gutenberg punctuation/quotes
    s = (
        s.replace("’", "'")
         .replace("‘", "'")
         .replace("`", "'")
         .replace("“", '"')
         .replace("”", '"')
         .replace("—", " ")
         .replace("–", " ")
    )
    s = _strip_accents(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _alias_tokens(a: str) -> List[str]:
    a = _norm_text(a)
    a = re.sub(r"[^a-z0-9\s\-']", " ", a)
    a = a.replace("-", " ")
    a = re.sub(r"\s+", " ", a).strip()
    return [t for t in a.split(" ") if t]


def _expand_aliases(claim_row: Dict[str, Any]) -> List[str]:
    """
    Robust alias list:
      - include char + char_aliases
      - include first/last token for multi-token names (helps "Lord Edward Glenarvan" vs "Glenarvan")
    """
    char = str(claim_row.get("char", "")).strip()
    aliases = claim_row.get("char_aliases") or []
    aliases = [str(a).strip() for a in aliases if a and str(a).strip()]

    if char and all(_norm_text(char) != _norm_text(a) for a in aliases):
        aliases.insert(0, char)

    out: List[str] = []
    seen = set()
    for a in aliases:
        na = _norm_text(a)
        if not na or na in seen:
            continue
        seen.add(na)
        out.append(a)

        toks = _alias_tokens(a)
        if len(toks) >= 2:
            first, last = toks[0], toks[-1]
            for v in (first, last):
                if len(v) >= 3 and v not in seen:
                    seen.add(v)
                    out.append(v)

    return out


def _mentions_any_alias(text: str, aliases: List[str]) -> bool:
    if not aliases:
        return False
    tl = _norm_text(text)
    for a in aliases:
        toks = _alias_tokens(a)
        if not toks:
            continue
        patt = r"\b" + r"\s+".join(re.escape(t) for t in toks) + r"\b"
        if re.search(patt, tl):
            return True
    return False


# ============================================================
# Sentence selection for NLI (general)
# ============================================================

def _sent_spans(text: str):
    """
    Yield (start,end) sentence spans.
    Uses blingfire offsets if available, else regex fallback.
    """
    try:
        from blingfire import text_to_sentences_and_offsets
        _sents, offsets = text_to_sentences_and_offsets(text)
        for (a, b) in offsets:
            if a is None or b is None:
                continue
            yield int(a), int(b)
        return
    except Exception:
        pass

    last = 0
    for m in re2.finditer(r"[.!?]+(\s+|$)", text):
        end = m.end()
        if end > last:
            yield last, end
        last = end
    if last < len(text):
        yield last, len(text)


def _claim_anchor_tokens(claim_row: Dict[str, Any]) -> List[str]:
    """
    General, non-hardcoded anchors from claim text:
      - tokenize claim_text
      - drop very short tokens, NEG_WORDS, and alias tokens
    """
    claim_text = str(claim_row.get("claim_text", "") or "")
    if not claim_text.strip():
        return []

    aliases = _expand_aliases(claim_row)
    alias_toks = set()
    for a in aliases:
        alias_toks.update(_alias_tokens(a))

    toks = tokenize(claim_text)
    out: List[str] = []
    for t in toks:
        tn = _norm_text(t)
        if len(tn) < 3:
            continue
        if tn in NEG_WORDS:
            continue
        if tn in alias_toks:
            continue
        out.append(tn)
    # de-dup while preserving order
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq[:20]


def _rank_sentences_for_claim(passage: str, claim_row: Dict[str, Any], top_n: int = 5) -> List[str]:
    """
    Choose top-N sentences likely to be most informative for NLI.
    Scoring (general):
      - alias mention bonus
      - overlap with claim anchor tokens bonus
      - length tie-breaker
    """
    passage = str(passage or "")
    if not passage.strip():
        return []

    aliases = _expand_aliases(claim_row)
    anchors = _claim_anchor_tokens(claim_row)

    scored: List[Tuple[float, str]] = []
    for a, b in _sent_spans(passage):
        sent = passage[a:b].strip()
        if not sent:
            continue
        sent_norm = _norm_text(sent)

        score = 0.0
        if aliases and _mentions_any_alias(sent, aliases):
            score += 2.0

        if anchors:
            hit = 0
            for t in anchors:
                if re.search(rf"\b{re.escape(t)}\w*\b", sent_norm):
                    hit += 1
            score += min(2.0, 0.6 * hit)

        score += min(0.4, len(sent) / 600.0)
        scored.append((score, sent))

    if not scored:
        return [passage[:400].strip()]

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for sc, s in scored[: max(1, int(top_n))] if sc > 0.0]
    if not top:
        top = [scored[0][1]]
    return top


def _excerpt_for_span(text: str, span: tuple[int, int]) -> tuple[str, int, int]:
    s, e = span
    for a, b in _sent_spans(text):
        if a <= s < b:
            return text[a:b].strip(), a, b
    a = max(0, s - 120)
    b = min(len(text), e + 120)
    return text[a:b].strip(), a, b


def _add_provenance_to_fact(
    fact: Dict[str, Any],
    passage: str,
    claim_row: Dict[str, Any],
    require_alias_in_sentence: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Attach (excerpt, span_start, span_end) to a fact.
    If require_alias_in_sentence=True, we only keep facts if we can find a sentence
    that mentions an alias of the character.
    """
    aliases = _expand_aliases(claim_row)

    best = None
    first = None
    for a, b in _sent_spans(passage):
        sent = passage[a:b].strip()
        if not sent:
            continue
        if first is None:
            first = (sent, a, b)
        if aliases and _mentions_any_alias(sent, aliases):
            best = (sent, a, b)
            break

    if require_alias_in_sentence and aliases and best is None:
        return None

    excerpt, span_start, span_end = best if best is not None else (first if first is not None else ("", 0, 0))
    if not excerpt:
        excerpt = passage[: min(len(passage), 400)].strip()
        span_start, span_end = 0, min(len(passage), 400)

    out = dict(fact)
    out.update(dict(excerpt=excerpt, span_start=int(span_start), span_end=int(span_end)))
    return out


# ============================================================
# NLI helpers
# ============================================================

def _cfg_get(cfg: Optional[VerifierConfig], key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return getattr(cfg, key, default)


def _boolish(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _mnli_to_kdsh_label(mnli_label: str) -> str:
    lab = str(mnli_label).strip().lower()
    if "entail" in lab:
        return "SUPPORT"
    if "contrad" in lab:
        return "CONTRADICT"
    return "NEUTRAL"


def _mix_conf(nli_conf: float, score_norm: float, alpha: float) -> float:
    """
    Combine NLI confidence with retrieval score (normalized 0..1).
    """
    try:
        nli_conf = float(nli_conf)
    except Exception:
        nli_conf = 0.0
    try:
        score_norm = float(score_norm)
    except Exception:
        score_norm = 0.0
    alpha = float(alpha)
    alpha = min(1.0, max(0.0, alpha))
    return float(min(1.0, max(0.0, (1.0 - alpha) * nli_conf + alpha * score_norm)))


def _mix_fact_conf(base: float, evidence_conf: float) -> float:
    base = float(base)
    ev = float(evidence_conf)
    return float(min(1.0, max(0.0, 0.65 * base + 0.35 * ev)))


def _apply_alias_gate(
    label: str,
    conf: float,
    claim_row: Dict[str, Any],
    passage: str,
    downgraded_conf_cap: float = 0.65,
) -> Tuple[str, float]:
    """
    Optional safety: if label==SUPPORT but passage does not mention any alias,
    downgrade SUPPORT -> NEUTRAL.
    """
    if label != "SUPPORT":
        return label, float(conf)

    aliases = _expand_aliases(claim_row)
    if aliases and not _mentions_any_alias(passage, aliases):
        return "NEUTRAL", float(min(conf, downgraded_conf_cap))
    return label, float(conf)


def _apply_contra_gate(
    label: str,
    conf: float,
    claim_row: Dict[str, Any],
    passage: str,
    downgraded_conf_cap: float = 0.65,
) -> Tuple[str, float]:
    """
    Optional safety: if label==CONTRADICT but passage does not mention any alias,
    downgrade CONTRADICT -> NEUTRAL.

    Helpful in novels where retrieval may surface nearby text about other characters,
    producing spurious contradictions.
    """
    if label != "CONTRADICT":
        return label, float(conf)

    aliases = _expand_aliases(claim_row)
    if aliases and not _mentions_any_alias(passage, aliases):
        return "NEUTRAL", float(min(conf, downgraded_conf_cap))
    return label, float(conf)


# ============================================================
# Predicate-form facts (general; no relation-specific rules)
# ============================================================

def parse_predicate(p: str) -> Optional[Tuple[str, List[str]]]:
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$", str(p))
    if not m:
        return None
    name = m.group(1)
    inside = m.group(2)
    args = [a.strip().strip('"').strip("'") for a in inside.split(",")]
    return name, args


def extract_facts_from_predicate_form(
    claim_row: Dict[str, Any],
    passage: str,
    evidence_conf: float,
    base_conf: float = 0.86,
) -> List[Dict[str, Any]]:
    """
    Convert predicate_form into KG triples.
    This is schema-driven and does NOT hardcode relation semantics.

    Example predicate_form item:
      BornIn(Lord_Glenarvan, Scotland)
    """
    char = str(claim_row.get("char", "")).strip()
    preds = claim_row.get("predicate_form", []) or []
    out: List[Dict[str, Any]] = []

    for p in preds:
        parsed = parse_predicate(p)
        if not parsed:
            continue
        pred, args = parsed

        obj: Any = None
        if pred in ("Died",):
            obj = True
        elif len(args) >= 2:
            obj = args[1]
        else:
            continue

        # Try numeric casts for common numeric predicates (still schema-level, not semantics)
        if pred in ("Age", "BornYear"):
            try:
                obj = int(obj)
            except Exception:
                continue

        fact = dict(
            s=char,
            p=str(pred),
            o=obj,
            polarity="POS",
            fact_confidence=_mix_fact_conf(float(base_conf), evidence_conf),
        )

        pf = _add_provenance_to_fact(fact, passage, claim_row, require_alias_in_sentence=False)
        out.append(pf if pf is not None else fact)

    return out


def _fallback_support_fact(claim_row: Dict[str, Any], passage: str, evidence_conf: float) -> Dict[str, Any]:
    """
    If evidence is SUPPORT but predicate_form is missing/unparseable, record a generic support assertion.
    This keeps facts.jsonl non-empty without inventing a relation.
    """
    char = str(claim_row.get("char", "")).strip()
    claim_text = str(claim_row.get("claim_text", "")).strip()

    excerpt = passage[: min(len(passage), 400)].strip()
    return dict(
        s=char,
        p="SupportsClaim",
        o=claim_text if claim_text else str(claim_row.get("claim_id", "")),
        polarity="POS",
        fact_confidence=float(min(1.0, max(0.55, float(evidence_conf)))),
        excerpt=excerpt,
        span_start=0,
        span_end=min(len(passage), 400),
    )


# ============================================================
# Main Step 4
# ============================================================

def step4_verify(
    chunks_df: pd.DataFrame,
    claims_path: Path,
    retrieval_path: Path,
    out_silver: Path,
    run_id: str,
    min_fact_conf: float = 0.45,
    verifier_cfg: Optional[VerifierConfig] = None,
):
    """
    NLI-only verifier (generalizable):
      - label evidence using MNLI/NLI models (SUPPORT / CONTRADICT / NEUTRAL)
      - emit KG facts only from predicate_form (schema-driven)
      - optional alias safety gate (configurable)
      - optional SupportsClaim fallback when predicate_form is missing
    """

    backend = _cfg_get(verifier_cfg, "backend", None)
    if not backend:
        backend = os.getenv("KDSH_VERIFIER_BACKEND", "mnli")
    backend = str(backend).strip().lower()

    if backend != "mnli":
        raise ValueError(f"Unsupported verifier backend '{backend}' in NLI-only step4. Use backend=mnli.")

    model_name = _cfg_get(verifier_cfg, "model_name", None) or os.getenv(
        "KDSH_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )
    batch_size = int(_cfg_get(verifier_cfg, "batch_size", int(os.getenv("KDSH_NLI_BATCH", "16"))))
    device = _cfg_get(verifier_cfg, "device", os.getenv("KDSH_NLI_DEVICE", "auto"))
    max_length = int(_cfg_get(verifier_cfg, "max_length", int(os.getenv("KDSH_NLI_MAXLEN", "512"))))
    score_mix_alpha = float(_cfg_get(verifier_cfg, "score_mix_alpha", float(os.getenv("KDSH_NLI_ALPHA", "0.15"))))

    # sentence-level NLI is recommended for long passages
    sentence_level = _boolish(_cfg_get(verifier_cfg, "sentence_level", os.getenv("KDSH_NLI_SENTENCE_LEVEL", "1")), default=True)
    top_sentences = int(_cfg_get(verifier_cfg, "top_sentences", int(os.getenv("KDSH_NLI_TOP_SENTS", "5"))))

    # label decision knobs (recall-friendly defaults)
    entail_threshold = float(_cfg_get(verifier_cfg, "entail_threshold", float(os.getenv("KDSH_NLI_ENTAIL_THR", "0.55"))))
    contra_threshold = float(_cfg_get(verifier_cfg, "contra_threshold", float(os.getenv("KDSH_NLI_CONTRA_THR", "0.80"))))
    nli_margin = float(_cfg_get(verifier_cfg, "nli_margin", float(os.getenv("KDSH_NLI_MARGIN", "0.05"))))

    # safety knobs
    enable_alias_gate = _boolish(_cfg_get(verifier_cfg, "enable_alias_gate", os.getenv("KDSH_ALIAS_GATE", "0")), default=False)
    enable_contra_alias_gate = _boolish(_cfg_get(verifier_cfg, "enable_contra_alias_gate", os.getenv("KDSH_CONTRA_ALIAS_GATE", "1")), default=True)
    emit_supports_claim_fallback = _boolish(_cfg_get(verifier_cfg, "emit_supports_claim_fallback", os.getenv("KDSH_EMIT_SUPPORTS_CLAIM", "1")), default=True)

    # claims are jsonl
    claim_rows = [
        json.loads(line)
        for line in claims_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    claim_map = {str(r.get("claim_id")): r for r in claim_rows if r.get("claim_id") is not None}

    retrieval_df = pd.read_csv(retrieval_path)
    chunk_text_map = dict(zip(chunks_df["chunk_id"], chunks_df["chunk_text"]))

    work: List[Tuple[Dict[str, Any], Dict[str, Any], str, float]] = []
    for rr in retrieval_df.to_dict(orient="records"):
        cid = str(rr.get("claim_id"))
        claim = claim_map.get(cid)
        if not claim:
            continue
        passage = chunk_text_map.get(rr["chunk_id"], "") or ""
        score_norm = float(rr.get("score_lex_norm", 0.0) or 0.0)
        work.append((rr, claim, passage, score_norm))

    # If nothing to verify, still write empty outputs
    out_silver.mkdir(parents=True, exist_ok=True)

    if not work:
        evidence_path = out_silver / "evidence_labels.csv"
        pd.DataFrame([]).to_csv(evidence_path, index=False)

        facts_path = out_silver / "facts.jsonl"
        facts_path.write_text("", encoding="utf-8")
        return evidence_path, facts_path

    from kdsh.pipeline.verification.hf_mnli import HFNLIVerifier

    nli = HFNLIVerifier(
        model_name=str(model_name),
        device=str(device),
        batch_size=int(batch_size),
        max_length=int(max_length),
    )

    # Build inputs
    if sentence_level:
        flat_premises: List[str] = []
        flat_hypotheses: List[str] = []
        flat_sents: List[str] = []   # keep the actual sentence used as premise
        meta: List[int] = []         # map flat index -> work index

        for i, (_rr, claim, passage, _sn) in enumerate(work):
            sents = _rank_sentences_for_claim(passage, claim, top_n=top_sentences)
            if not sents:
                sents = [passage[:400]]
            hyp = str(claim.get("claim_text", ""))
            for s in sents:
                meta.append(i)
                flat_premises.append(s)
                flat_sents.append(s)
                flat_hypotheses.append(hyp)

        preds = nli.predict_batch(flat_premises, flat_hypotheses)

        # Aggregate best per class per work-item AND keep the sentence that achieved it
        best = [
            {"SUPPORT": (0.0, ""), "CONTRADICT": (0.0, ""), "NEUTRAL": (0.0, "")}
            for _ in range(len(work))
        ]
        for wi, sent, pred in zip(meta, flat_sents, preds):
            lab = _mnli_to_kdsh_label(getattr(pred, "label", "neutral"))
            c = float(getattr(pred, "confidence", 0.0) or 0.0)
            cur = best[wi].get(lab, (0.0, ""))
            if c > float(cur[0]):
                best[wi][lab] = (c, sent)

        per_pair: List[Tuple[str, float]] = []
        chosen_premise: List[str] = [""] * len(work)

        for i, b in enumerate(best):
            sup, sup_sent = b.get("SUPPORT", (0.0, ""))
            con, con_sent = b.get("CONTRADICT", (0.0, ""))
            neu, neu_sent = b.get("NEUTRAL", (0.0, ""))

            sup = float(sup); con = float(con); neu = float(neu)

            if sup >= entail_threshold and sup >= con + nli_margin:
                per_pair.append(("SUPPORT", sup))
                chosen_premise[i] = sup_sent or neu_sent or con_sent
            elif con >= contra_threshold and con >= sup + nli_margin:
                per_pair.append(("CONTRADICT", con))
                chosen_premise[i] = con_sent or neu_sent or sup_sent
            else:
                per_pair.append(("NEUTRAL", max(neu, sup, con)))
                # pick the most confident sentence among labels for provenance
                best_any = max(
                    [(neu, neu_sent), (sup, sup_sent), (con, con_sent)],
                    key=lambda x: float(x[0]),
                )
                chosen_premise[i] = best_any[1] or sup_sent or con_sent or neu_sent



    else:
        premises = [p for (_rr, _cl, p, _sn) in work]
        hypotheses = [str(cl.get("claim_text", "")) for (_rr, cl, _p, _sn) in work]
        preds = nli.predict_batch(premises, hypotheses)
        per_pair = [
            (_mnli_to_kdsh_label(getattr(pred, "label", "neutral")), float(getattr(pred, "confidence", 0.0) or 0.0))
            for pred in preds
        ]

    evidence_rows: List[Dict[str, Any]] = []
    facts_rows: List[Dict[str, Any]] = []

    for i, ((rr, claim, passage, score_norm), (label, nli_conf)) in enumerate(zip(work, per_pair)):
        conf = _mix_conf(float(nli_conf), float(score_norm), float(score_mix_alpha))
        evidence_text = ""
        if sentence_level:
            try:
                evidence_text = chosen_premise[i] or ""
            except Exception:
                evidence_text = ""

        if enable_alias_gate:
            label, conf = _apply_alias_gate(label, conf, claim, passage)
        if enable_contra_alias_gate:
            label, conf = _apply_contra_gate(label, conf, claim, passage)

        evidence_rows.append(
            dict(
                id=int(rr["id"]),
                claim_id=rr["claim_id"],
                book_name=rr["book_name"],
                chunk_id=rr["chunk_id"],
                evidence_text=evidence_text,
                label=label,
                confidence=float(conf),
                verifier_model=f"mnli:{model_name}",
                run_id=run_id,
            )
        )

        if label != "SUPPORT":
            continue

        facts = extract_facts_from_predicate_form(claim, passage, evidence_conf=conf)
        if evidence_text:
            start = passage.find(evidence_text)
            end = start + len(evidence_text) if start >= 0 else None
            for fct in facts:
                fct['excerpt'] = evidence_text
                if start >= 0 and end is not None:
                    fct['span_start'] = int(start)
                    fct['span_end'] = int(end)

        added_any = False
        for fct in facts:
            fc = float(fct.get("fact_confidence", 0.0) or 0.0)
            if fc < float(min_fact_conf):
                continue
            fct2 = dict(fct)
            fct2.update(
                dict(
                    id=int(rr["id"]),
                    claim_id=rr["claim_id"],
                    book_name=rr["book_name"],
                    chunk_id=rr["chunk_id"],
                    label=label,
                    confidence=float(conf),
                    run_id=run_id,
                )
            )
            facts_rows.append(fct2)
            added_any = True

        if (not added_any) and emit_supports_claim_fallback:
            fct2 = _fallback_support_fact(claim, passage, evidence_conf=conf)
            fct2.update(
                dict(
                    id=int(rr["id"]),
                    claim_id=rr["claim_id"],
                    book_name=rr["book_name"],
                    chunk_id=rr["chunk_id"],
                    label=label,
                    confidence=float(conf),
                    run_id=run_id,
                )
            )
            facts_rows.append(fct2)

    evidence_df = pd.DataFrame(evidence_rows)
    evidence_path = out_silver / "evidence_labels.csv"
    evidence_df.to_csv(evidence_path, index=False)

    facts_path = out_silver / "facts.jsonl"
    with facts_path.open("w", encoding="utf-8") as f:
        for r in facts_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return evidence_path, facts_path
