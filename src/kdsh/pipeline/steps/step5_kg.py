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


# def excerpt_span(chunk_text: str, needle: str) -> Optional[List[int]]:
#     if not chunk_text or not needle:
#         return None
#     tl = chunk_text.lower()
#     nl = str(needle).lower()
#     idx = tl.find(nl)
#     if idx == -1:
#         tok0 = nl.split()[0] if nl.split() else None
#         if tok0 and tok0 in tl:
#             idx = tl.find(tok0)
#         else:
#             return None
#     start = max(0, idx - 40)
#     end = min(len(chunk_text), idx + len(needle) + 80)
#     return [int(start), int(end)]

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

    # alias_rows = []
    # for ex_id, char in id_to_char.items():
    #     if not char:
    #         continue
    #     parts = [p for p in re.split(r"\s+", char) if p]
    #     last = parts[-1] if len(parts) > 1 else None
    #     alias_rows.append(dict(id=ex_id, book_name=id_to_book.get(ex_id), mention=char, canonical_entity=char, entity_type="CHAR", confidence=1.0, run_id=run_id))
    #     if last:
    #         alias_rows.append(dict(id=ex_id, book_name=id_to_book.get(ex_id), mention=last, canonical_entity=char, entity_type="CHAR", confidence=0.65, run_id=run_id))
    #     for title in ["mr","mrs","ms","miss","count","captain","doctor","dr","sir","madam","monsieur","mademoiselle"]:
    #         alias_rows.append(dict(id=ex_id, book_name=id_to_book.get(ex_id), mention=f"{title} {last}".title() if last else f"{title} {char}".title(), canonical_entity=char, entity_type="CHAR", confidence=0.35, run_id=run_id))

    # aliases_df = pd.DataFrame(alias_rows).drop_duplicates(subset=["id","mention","canonical_entity"])
    # aliases_path = out_silver / "aliases.csv"
    # aliases_df.to_csv(aliases_path, index=False)

    facts = [json.loads(l) for l in facts_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    facts_df = pd.DataFrame(facts) if facts else pd.DataFrame()

    chunks_meta = chunks_df.set_index("chunk_id")[["book_name","chapter_id","chunk_pos","time_bucket"]].to_dict("index")
    chunks_text = chunks_df.set_index("chunk_id")["chunk_text"].to_dict()

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

            # s = r.get("s")
            # p = r.get("p")
            # o = r.get("o")
            # canonical = id_to_char.get(ex_id)
            # if canonical and isinstance(s, str) and s.strip().lower() == canonical.strip().lower():
            #     s = canonical

            # span = excerpt_span(txt, str(o)) or excerpt_span(txt, str(p)) or excerpt_span(txt, str(s)) or [0, min(180, len(txt))]

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
            span = None
            if "span_start" in r and "span_end" in r:
                try:
                    a = int(r["span_start"])
                    b = int(r["span_end"])
                    if 0 <= a < b <= len(txt):
                        span = [a, b]
                except Exception:
                    span = None

            # else fallback to search
            if span is None:
                span = excerpt_span(txt, str(o)) or excerpt_span(txt, str(p)) or excerpt_span(txt, str(s)) or [0, min(180, len(txt))]
            

            triple_rows.append(
                dict(
                    triple_id=f"t_{tid:05d}",
                    id=ex_id,
                    claim_id=r.get("claim_id"),
                    book_name=meta.get("book_name", r.get("book_name")),
                    s=s,
                    p=p,
                    o=o,
                    time_bucket=meta.get("time_bucket"),
                    polarity=r.get("polarity", "POS"),
                    confidence=float(fc),
                    chunk_id=chunk_id,
                    excerpt_span=span,
                    excerpt_text = r.get("excerpt"),
                    chapter_id=meta.get("chapter_id"),
                    chunk_pos=meta.get("chunk_pos"),
                    support_label=r.get("label"),
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
