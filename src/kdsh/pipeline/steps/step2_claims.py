from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import regex as re2

from kdsh.common.utils import STOPWORDS


def _build_book_name_map(out_silver: Path) -> Dict[str, str]:
    """
    Map case-insensitive book_name (from train/test) -> canonical book_name used in chunks/ingest.
    This prevents retrieval from returning 0 rows when casing differs (e.g. 'In Search...' vs 'In search...').
    """
    # Prefer novel_registry.csv (authoritative)
    nr = out_silver / "novel_registry.csv"
    if nr.exists():
        try:
            df = pd.read_csv(nr)
            if "book_name" in df.columns:
                return {str(b).casefold(): str(b) for b in df["book_name"].dropna().unique()}
        except Exception:
            pass

    # Fallback: chunks.csv
    cp = out_silver / "chunks.csv"
    if cp.exists():
        try:
            df = pd.read_csv(cp, usecols=["book_name"])
            return {str(b).casefold(): str(b) for b in df["book_name"].dropna().unique()}
        except Exception:
            pass

    return {}

def _canonical_book_name(raw_name: str, book_name_map: Dict[str, str]) -> str:
    if raw_name is None:
        return ""
    key = str(raw_name).casefold()
    return book_name_map.get(key, str(raw_name))


# IMPORTANT: Do NOT split on commas.
# Novels use commas for appositives that encode facts:
#   "Lord Glenarvan and his wife, Lady Helena, ..."
# If you split on comma, you lose the "wife + name" structure.
CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:;|:|\(|\)|—|–| - |\band\b|\bbut\b|\bwhile\b|\bbecause\b|\bso\b|\bhowever\b)\s*",
    re.IGNORECASE,
)

def normalize_ws(s: str) -> str:
    s = str(s or "")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()

def get_spacy_nlp():
    """Blank spaCy pipeline (offline-safe) for sentence splitting."""
    try:
        import spacy
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp
    except Exception:
        return None  # fallback to regex splitting

def split_sentences(text: str) -> List[str]:
    text = str(text or "").strip()
    if not text:
        return []
    nlp = get_spacy_nlp()
    if nlp is None:
        # fallback
        return [s for s in re.split(r"(?<=[\.\?\!])\s+|\n+", text) if s.strip()]
    doc = nlp(text)
    return [sent.text for sent in doc.sents if sent.text.strip()]

# ---------------------------
# Novel-aware “facty” detector
# ---------------------------

_FACTY_RE = re2.compile(
    r"\b("
    r"born|died|dead|killed|executed|passed\s+away|"
    r"married|wife|husband|"
    r"betrothed|engaged|"
    r"called|known\s+as|named|"
    r"son\s+of|daughter\s+of|child\s+of|"
    r"father\s+of|mother\s+of|parent\s+of|"
    r"brother\s+of|sister\s+of"
    r")\b",
    flags=re2.I,
)

# Unicode-aware Name patterns (works for Dantès, Mercédès, etc.)
# Allow particles: de, du, d', van, von, la, le
_NAME_BASE = r"(?:\p{Lu}[\p{L}\p{M}'\-]+(?:\s+(?:\p{Lu}[\p{L}\p{M}'\-]+|de|du|da|van|von|la|le|d')){0,4})"
_NAME = rf"(?-i:{_NAME_BASE})"

# Titles & abbreviations common in these novels
_TITLES = (
    r"Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Sir|Madam|"
    r"Monsieur|Madame|Mademoiselle|"
    r"M\.|Mme\.|Mlle\.|"
    r"Lord|Lady|Captain|Count|Comte|Abbé|Abbe"
)
_TITLED_NAME = rf"(?:{_TITLES}\s+{_NAME}|{_NAME})"

def char_aliases(char: str) -> List[str]:
    """
    Novel-friendly aliases:
    - supports titles + abbreviations: M., Mme., Mlle., Lord, Lady,
    - adds first + last tokens
    - adds both 'Mr Last' and 'Mr. Last' forms
    """
    c = normalize_ws(char)
    if not c:
        return []

    parts = c.split()
    first = parts[0]
    last = parts[-1] if len(parts) > 1 else ""

    aliases = set()
    aliases.add(c)
    aliases.add(first)

    if last:
        aliases.add(last)
        aliases.add(f"{first} {last}")

        # English titles
        for t in ["Mr", "Mrs", "Ms", "Miss", "Dr", "Sir", "Madam", "Captain", "Count", "Lord", "Lady"]:
            aliases.add(f"{t} {last}")
            aliases.add(f"{t}. {last}")

        # French/Gutenberg abbreviations
        for t in ["Monsieur", "Madame", "Mademoiselle", "M.", "Mme.", "Mlle.", "Comte", "Abbé", "Abbe"]:
            aliases.add(f"{t} {last}")

    # Normalize + filter
    out = []
    seen = set()
    for a in sorted(aliases):
        aa = normalize_ws(a)
        if not aa:
            continue
        low = aa.lower()
        if low == "none":
            continue
        if low in seen:
            continue
        seen.add(low)
        out.append(aa)
    return out

def split_into_claims(text: str, max_claims: int = 80) -> List[str]:
    """
    Novel-friendly claim splitting:
    - keep short clauses IF they look fact-like (married/betrothed/son of/called/etc.)
    - avoid over-pruning (old version dropped many key facts because they were <7 words)
    """
    text = str(text or "").strip()
    if not text:
        return []

    sents = split_sentences(text)
    claims: List[str] = []

    for s in sents:
        s = normalize_ws(s)
        if not s:
            continue

        clauses = [normalize_ws(c) for c in CLAUSE_SPLIT_RE.split(s) if normalize_ws(c)]
        if not clauses:
            clauses = [s]

        for c in clauses:
            words = c.split()

            is_facty = bool(_FACTY_RE.search(c))

            # Keep short factual sentences (e.g., “Edmond’s betrothed was Mercédès.”)
            if len(words) < 4 and not is_facty:
                continue

            # Light stopword pruning only for non-facty short clauses
            if not is_facty and len(words) <= 10:
                sw_ratio = sum(w.lower() in STOPWORDS for w in words) / max(1, len(words))
                if sw_ratio > 0.60:
                    continue

            claim = c if c.endswith((".", "!", "?")) else c + "."
            claims.append(claim)

            if len(claims) >= max_claims:
                return claims

    return claims[:max_claims]

def keywords_from_text(text: str, k: int = 12) -> List[str]:
    # Unicode-safe tokenization
    words = re2.findall(r"[\p{L}][\p{L}\p{M}\-']+", str(text or "").lower())
    freq: Dict[str, int] = {}
    for w in words:
        if w in STOPWORDS or len(w) < 4:
            continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:k]]

def _q(s: str) -> str:
    s = str(s or "").strip().replace('"', "'")
    return f'"{s}"'

def _mentions_any_alias(text: str, aliases: List[str]) -> bool:
    t = normalize_ws(text).lower()
    for a in aliases or []:
        aa = normalize_ws(a).lower()
        if not aa:
            continue
        # For single-token aliases, use word boundaries. For multi-token, substring match.
        if " " not in aa and len(aa) >= 2:
            if re.search(rf"\b{re.escape(aa)}\b", t):
                return True
        else:
            if aa in t:
                return True
    return False

_LEADING_POSSESSIVE = re.compile(r"^(his|her|their)\b\s*", re.IGNORECASE)
_LEADING_SUBJECT_PRON = re.compile(r"^(he|she|they)\b\s*", re.IGNORECASE)
_LEADING_OBJECT_PRON = re.compile(r"^(him|her|them)\b\s*", re.IGNORECASE)

# Nouny “fragment starters” common in summaries
_LEADING_NOUN_START = re.compile(r"^(boyhood|childhood|youth|mother|father|parents?)\b\s*", re.IGNORECASE)

def make_claim_standalone(char: str, aliases: List[str], claim: str) -> str:
    """
    Ensure the hypothesis is *self-contained* for NLI:
    - If an alias is already mentioned, keep as-is.
    - Else, resolve leading pronouns/fragments by anchoring to the character.
    This is intentionally conservative: it only rewrites the start of the clause.
    """
    c = normalize_ws(claim)
    if not c:
        return c
    if _mentions_any_alias(c, aliases):
        return c

    # Normalize leading pronouns/fragments
    m = _LEADING_POSSESSIVE.match(c)
    if m:
        # 'his mother ...' -> "<CHAR>'s mother ..."
        rest = c[m.end():].lstrip()
        anchored = f"{char}'s {rest}"
        return anchored

    m = _LEADING_SUBJECT_PRON.match(c)
    if m:
        # 'he ...' -> '<CHAR> ...'
        rest = c[m.end():].lstrip()
        anchored = f"{char} {rest}"
        return anchored

    m = _LEADING_OBJECT_PRON.match(c)
    if m:
        rest = c[m.end():].lstrip()
        anchored = f"{char} {rest}"
        return anchored

    # 'Mother died ...' / 'Boyhood was ...' -> '<CHAR>'s Mother ...' / '<CHAR>'s boyhood ...'
    m = _LEADING_NOUN_START.match(c)
    if m:
        rest = c[m.start():].lstrip()
        anchored = f"{char}'s {rest}"
        return anchored

    # Fallback: prefix with character name
    anchored = f"{char} {c}"
    return anchored

def predicate_form(char: str, claim: str) -> List[str]:
    """
    Refactor v1: NO relation extraction here.
    This step only emits a generic Claim(...) predicate.
    Downstream verification + extraction should decide what becomes a KG triple.
    """
    c = normalize_ws(claim).replace('"', "'")
    return [f'Claim({_q(char)}, "{c}")']

def step2_build_claims(train_df: pd.DataFrame, test_df: pd.DataFrame, out_silver: Path, run_id: str) -> Path:
    book_name_map = _build_book_name_map(out_silver)

    def build_rows(df: pd.DataFrame, split: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            ex_id = int(r["id"])
            raw_book_name = str(r["book_name"])
            book_name = _canonical_book_name(raw_book_name, book_name_map)
            char = str(r["char"]).strip()
            caption = None if pd.isna(r.get("caption", None)) else str(r.get("caption"))
            content = str(r["content"])

            aliases = char_aliases(char)
            cap = "" if caption is None else str(caption).strip()

            claims = split_into_claims(content, max_claims=80)
            for i, raw in enumerate(claims, start=1):
                cid = f"{ex_id}_c{i:02d}"

                claim_text = make_claim_standalone(char, aliases, raw)

                # Build retrieval keywords from the *standalone* claim (better for NLI + retrieval)
                kw = keywords_from_text(claim_text, k=12)
                if cap:
                    kw += keywords_from_text(cap, k=8)
                for a in aliases:
                    kw += keywords_from_text(a, k=3)

                rows.append(
                    dict(
                        id=ex_id,
                        book_name=book_name,
                        raw_book_name=raw_book_name,
                        char=char,
                        caption=caption,
                        char_aliases=aliases,
                        claim_id=cid,
                        raw_claim_text=normalize_ws(raw),
                        claim_text=claim_text,
                        predicate_form=predicate_form(char, claim_text),
                        keywords=sorted(set(kw)),
                        claim_type="UNK",
                        t_hint="UNK",
                        split=split,
                        run_id=run_id,
                    )
                )
        return rows

    rows = build_rows(train_df, "train") + build_rows(test_df, "test")
    out_path = out_silver / "claims.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out_path
