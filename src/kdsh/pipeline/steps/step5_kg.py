from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def normalize_mention(s: Any) -> str:
    """Aggressive but safe normalization for matching aliases."""
    if s is None:
        return ""
    s = str(s)
    s = s.strip().lower()
    # remove most punctuation but keep spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_alias_rows(ex_id: int, book_name: Optional[str], char: str, run_id: str) -> List[Dict[str, Any]]:
    """Generate stronger alias set for the target character."""
    rows: List[Dict[str, Any]] = []
    if not char:
        return rows

    char_clean = " ".join(str(char).strip().split())
    parts = [p for p in re.split(r"\s+", char_clean) if p]
    first = parts[0] if parts else None
    last = parts[-1] if len(parts) >= 2 else None
    first_last = f"{first} {last}" if first and last else None

    # Titles (with and without dot)
    titles = ["mr", "mrs", "ms", "miss", "count", "captain", "doctor", "dr", "sir", "madam", "monsieur", "mademoiselle"]
    title_variants = []
    for t in titles:
        title_variants.append(t)
        if t in ["mr", "mrs", "ms", "dr"]:
            title_variants.append(f"{t}.")

    def add(mention: str, conf: float):
        if mention and mention.strip():
            rows.append(dict(
                id=ex_id,
                book_name=book_name,
                mention=mention.strip(),
                canonical_entity=char_clean,
                entity_type="CHAR",
                confidence=float(conf),
                run_id=run_id,
            ))

    # Strong aliases
    add(char_clean, 1.00)
    if first_last and first_last.lower() != char_clean.lower():
        add(first_last, 0.85)
    if first:
        add(first, 0.55)
    if last:
        add(last, 0.65)

    # Title + last (preferred)
    if last:
        for t in title_variants:
            add(f"{t} {last}".title(), 0.40)

    # Title + full name (fallback)
    for t in title_variants:
        add(f"{t} {char_clean}".title(), 0.25)

    return rows


def excerpt_span(chunk_text: str, needle: str) -> Optional[List[int]]:
    if not chunk_text or not needle:
        return None
    tl = chunk_text.lower()
    nl = str(needle).lower()
    idx = tl.find(nl)
    if idx == -1:
        tok0 = nl.split()[0] if nl.split() else None
        if tok0 and tok0 in tl:
            idx = tl.find(tok0)
        else:
            return None
    start = max(0, idx - 40)
    end = min(len(chunk_text), idx + len(needle) + 80)
    return [int(start), int(end)]

def _load_evidence_map(out_silver: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Load Step-4 evidence labels (if available) and build a lookup keyed by (claim_id, chunk_id).
    This lets Step-5 attach:
      - support_label (SUPPORT/NEUTRAL/CONTRADICT)
      - evidence_confidence
      - evidence_text (best sentence used in sentence-level NLI, if present)
    Backwards compatible: returns empty dict if file not present.
    """
    p = out_silver / "evidence_labels.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for _, r in df.iterrows():
        cid = str(r.get("claim_id", "")).strip()
        chid = str(r.get("chunk_id", "")).strip()
        if not cid or not chid:
            continue
        out[(cid, chid)] = dict(
            label=str(r.get("label", "")).strip(),
            confidence=float(r.get("confidence", 0.0) or 0.0),
            evidence_text=str(r.get("evidence_text", "") or "").strip(),
        )
    return out


def _span_for_evidence(chunk_text: str, evidence_text: str) -> Optional[List[int]]:
    """
    Find span for evidence_text inside chunk_text. Falls back to excerpt_span behavior.
    """
    if not chunk_text or not evidence_text:
        return None
    tl = chunk_text
    idx = tl.find(evidence_text)
    if idx >= 0:
        return [int(idx), int(idx + len(evidence_text))]
    # try lower-cased match
    idx = tl.lower().find(evidence_text.lower())
    if idx >= 0:
        return [int(idx), int(idx + len(evidence_text))]
    return None




def step5_build_kg(
    chunks_df: pd.DataFrame,
    claims_path: Path,
    facts_path: Path,
    out_silver: Path,
    run_id: str,
    min_fact_conf: float = 0.65,
):
    id_to_char: Dict[int, str] = {}
    id_to_book: Dict[int, str] = {}
    for line in claims_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        i = int(r["id"])
        if i not in id_to_char:
            id_to_char[i] = str(r.get("char", "")).strip()
            id_to_book[i] = str(r.get("book_name", "")).strip()


    alias_rows: List[Dict[str, Any]] = []
    for ex_id, char in id_to_char.items():
        book_name = id_to_book.get(ex_id)
        alias_rows.extend(make_alias_rows(ex_id, book_name, char, run_id))

    aliases_df = pd.DataFrame(alias_rows).drop_duplicates(subset=["id", "mention", "canonical_entity"])
    aliases_path = out_silver / "aliases.csv"
    aliases_df.to_csv(aliases_path, index=False)

    # Build alias lookup per id for canonicalization
    alias_map_by_id: Dict[int, Dict[str, str]] = {}
    for ex_id, g in aliases_df.groupby("id"):
        m = {}
        for _, r in g.iterrows():
            m[normalize_mention(r["mention"])] = str(r["canonical_entity"])
        alias_map_by_id[int(ex_id)] = m


    facts = [json.loads(l) for l in facts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    facts_df = pd.DataFrame(facts) if facts else pd.DataFrame()

    chunks_meta = chunks_df.set_index("chunk_id")[["book_name","chapter_id","chunk_pos","time_bucket"]].to_dict("index")
    chunks_text = chunks_df.set_index("chunk_id")["chunk_text"].to_dict()

    evidence_map = _load_evidence_map(out_silver)

    triple_rows = []
    tid = 1
    if len(facts_df):
        for _, r in facts_df.iterrows():
            fc = float(r.get("fact_confidence", r.get("confidence", 0.0)))
            if fc < min_fact_conf:
                continue
            ex_id = int(r["id"])
            chunk_id = r["chunk_id"]
            meta = chunks_meta.get(chunk_id, {})
            txt = chunks_text.get(chunk_id, "")


            claim_id = str(r.get("claim_id", "")).strip()
            # Enrich with Step-4 evidence metadata if available
            em = evidence_map.get((claim_id, str(chunk_id).strip()), {})
            support_label = str(r.get("label") or em.get("label") or "").strip()
            evidence_conf = float(r.get("confidence", em.get("confidence", 0.0)) or 0.0)
            evidence_text = str(r.get("excerpt") or em.get("evidence_text") or "").strip()
            s = r.get("s")
            p = r.get("p")
            o = r.get("o")

            canonical = id_to_char.get(ex_id, "")
            alias_map = alias_map_by_id.get(ex_id, {})

            # Canonicalize subject if it matches target or any alias
            s_norm = normalize_mention(s)
            if canonical and (s_norm == normalize_mention(canonical) or s_norm in alias_map):
                s = canonical

            # Canonicalize object for person-relations if it matches aliases (helps consistency)
            if str(p) in {"MarriedTo", "ParentOf", "SiblingOf"}:
                o_norm = normalize_mention(o)
                if canonical and (o_norm == normalize_mention(canonical) or o_norm in alias_map):
                    o = canonical

            # ---- provenance span preference ----
            # If Step 4.5 wrote span_start/span_end, prefer them.
            # ---- provenance span preference ----
            # Preference order:
            # 1) If evidence_text is available (sentence-level NLI), locate it inside chunk_text.
            # 2) If Step-4 wrote span_start/span_end, prefer them.
            # 3) Fallback to fuzzy excerpt_span search, else head of chunk.
            span = None

            if evidence_text:
                span = _span_for_evidence(txt, evidence_text)

            if span is None and "span_start" in r and "span_end" in r:
                try:
                    a = int(r["span_start"])
                    b = int(r["span_end"])
                    if 0 <= a < b <= len(txt):
                        span = [a, b]
                except Exception:
                    span = None

            if span is None:
                # fallback to search for object / predicate / subject mentions
                span = (
                    excerpt_span(txt, str(o))
                    or excerpt_span(txt, str(p))
                    or excerpt_span(txt, str(s))
                    or [0, min(180, len(txt))]
                )

            # Ensure excerpt_text is aligned to the chosen span
            excerpt_text = evidence_text if evidence_text else (txt[span[0]:span[1]] if span else "")
            triple_rows.append(
                dict(
                    triple_id=f"t_{tid:05d}",
                    id=ex_id,
                    claim_id=claim_id,
                    book_name=meta.get("book_name", r.get("book_name")),
                    s=s,
                    p=p,
                    o=o,
                    time_bucket=meta.get("time_bucket"),
                    polarity=r.get("polarity", "POS"),
                    confidence=float(fc),
                    evidence_confidence=float(evidence_conf),
                    chunk_id=chunk_id,
                    excerpt_span=span,
                    excerpt_text=excerpt_text,
                    chapter_id=meta.get("chapter_id"),
                    chunk_pos=meta.get("chunk_pos"),
                    support_label=support_label,
                    run_id=run_id,
                )
            )
            tid += 1

    kg_df = pd.DataFrame(triple_rows)

    if len(kg_df):
        kg_df["best_rank"] = (
            kg_df.groupby(["id", "p", "polarity"])["confidence"]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        kg_df["is_best"] = kg_df["best_rank"] == 1
    else:
        kg_df["best_rank"] = pd.Series(dtype="int")
        kg_df["is_best"] = pd.Series(dtype="bool")


    kg_path = out_silver / "kg_triples.csv"
    kg_df.to_csv(kg_path, index=False)
    return aliases_path, kg_path
