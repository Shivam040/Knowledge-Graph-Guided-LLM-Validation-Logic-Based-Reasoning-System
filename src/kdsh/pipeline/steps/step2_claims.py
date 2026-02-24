from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import regex as re2

from kdsh.common.utils import STOPWORDS

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
    - supports titles + abbreviations: M., Mme., Mlle., Lord, Lady, Count, etc.
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
            low = c.lower()

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

def guess_claim_type(claim: str) -> str:
    c = claim.lower()
    if any(k in c for k in ["believe", "thinks", "distrust", "fears", "hates", "loves", "wants", "refused"]):
        return "BELIEF"
    if any(k in c for k in ["married", "wife", "husband", "betrothed", "engaged",
                            "brother", "sister", "father", "mother", "son", "daughter", "child", "parent",
                            "known as", "called", "named"]):
        return "RELATION"
    if any(k in c for k in ["born", "raised", "grew up", "childhood", "origin", "background"]):
        return "BACKGROUND"
    return "EVENT"

def guess_time_hint(claim: str) -> str:
    c = claim.lower()
    if any(k in c for k in ["childhood", "as a child", "early", "before", "at first", "initially", "young"]):
        return "EARLY"
    if any(k in c for k in ["later", "eventually", "after years", "in the end", "finally", "towards the end"]):
        return "LATE"
    if any(k in c for k in ["midway", "in between", "during"]):
        return "MID"
    return "UNK"

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

def predicate_form(char: str, claim: str) -> List[str]:
    """
    Generate predicates that actually appear in these novels.
    (Unicode-safe names + M./Mme./Lord/Lady handling)
    """
    c = normalize_ws(claim)
    low = c.lower()
    preds: List[str] = []

    # BornIn / BornAt (rare but keep)
    m = re2.search(rf"\bborn\s+(?:in|at)\s+(?P<place>{_TITLED_NAME})", c, flags=re2.I | re2.V1)
    if m:
        place = normalize_ws(m.group("place")).rstrip(".,;:!")
        preds.append(f"BornIn({_q(char)}, {_q(place)})")

    # Died
    if any(k in low for k in [" died", "dead", "killed", "executed", "passed away"]):
        preds.append(f"Died({_q(char)})")

    # MarriedTo
    m2 = re2.search(rf"\bmarried\b(?:\s+to\s+)?(?P<o>{_TITLED_NAME})", c, flags=re2.I | re2.V1)
    if m2:
        other = normalize_ws(m2.group("o")).rstrip(".,;:!")
        preds.append(f"MarriedTo({_q(char)}, {_q(other)})")
    else:
        # "his wife, Lady Helena" / "her husband, X"
        m2b = re2.search(rf"\b(?:wife|husband)\b[^\p{{L}}]{{0,20}}(?P<o>{_TITLED_NAME})", c, flags=re2.I | re2.V1)
        if m2b:
            other = normalize_ws(m2b.group("o")).rstrip(".,;:!")
            preds.append(f"MarriedTo({_q(char)}, {_q(other)})")

    # BetrothedTo / EngagedTo (VERY common in Monte Cristo)
    m3 = re2.search(rf"\b(?:betrothed|engaged)\b(?:\s+to\s+)?(?P<o>{_TITLED_NAME})", c, flags=re2.I | re2.V1)
    if m3:
        other = normalize_ws(m3.group("o")).rstrip(".,;:!")
        preds.append(f"BetrothedTo({_q(char)}, {_q(other)})")
    else:
        # "<X>'s betrothed was <Y>"
        m3b = re2.search(
            rf"(?P<s>{_TITLED_NAME})\s*['’]s\s+betrothed\b.*?\b(?:was|is)\b\s+(?P<o>{_TITLED_NAME})",
            c,
            flags=re2.I | re2.V1,
        )
        if m3b:
            other = normalize_ws(m3b.group("o")).rstrip(".,;:!")
            preds.append(f"BetrothedTo({_q(char)}, {_q(other)})")

    # LivesIn / ResidesIn / Dwells (less common but ok)
    m4 = re2.search(rf"\b(lives|lived|resides|resided|dwells)\s+(?:in|at)\s+(?P<o>{_TITLED_NAME})", c, flags=re2.I | re2.V1)
    if m4:
        loc = normalize_ws(m4.group("o")).rstrip(".,;:!")
        preds.append(f"LivesIn({_q(char)}, {_q(loc)})")

    # ChildOf / ParentOf / SiblingOf
    m5 = re2.search(rf"\b(son|daughter|child)\s+of\s+(?P<o>{_TITLED_NAME})", c, flags=re2.I | re2.V1)
    if m5:
        parent = normalize_ws(m5.group("o")).rstrip(".,;:!")
        preds.append(f"ChildOf({_q(char)}, {_q(parent)})")

    m6 = re2.search(rf"\b(father|mother|parent)\s+of\s+(?P<o>{_TITLED_NAME})", c, flags=re2.I | re2.V1)
    if m6:
        child = normalize_ws(m6.group("o")).rstrip(".,;:!")
        preds.append(f"ParentOf({_q(char)}, {_q(child)})")

    m7 = re2.search(rf"\b(brother|sister)\s+of\s+(?P<o>{_TITLED_NAME})", c, flags=re2.I | re2.V1)
    if m7:
        sib = normalize_ws(m7.group("o")).rstrip(".,;:!")
        preds.append(f"SiblingOf({_q(char)}, {_q(sib)})")

    # KnownAs (EXTREMELY common in Monte Cristo)
    m8 = re2.search(rf"\b(?:was\s+called|called|known\s+as|named)\s+(?P<o>{_TITLED_NAME})", c, flags=re2.I | re2.V1)
    if m8:
        aka = normalize_ws(m8.group("o")).rstrip(".,;:!")
        preds.append(f"KnownAs({_q(char)}, {_q(aka)})")

    if not preds:
        safe = c.replace('"', "'")
        preds.append(f'Claim({_q(char)}, "{safe}")')

    return preds

def step2_build_claims(train_df: pd.DataFrame, test_df: pd.DataFrame, out_silver: Path, run_id: str) -> Path:
    def build_rows(df: pd.DataFrame, split: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            ex_id = int(r["id"])
            book_name = str(r["book_name"])
            char = str(r["char"]).strip()
            caption = None if pd.isna(r.get("caption", None)) else str(r.get("caption"))
            content = str(r["content"])

            aliases = char_aliases(char)
            cap = "" if caption is None else str(caption).strip()

            claims = split_into_claims(content, max_claims=80)
            for i, cl in enumerate(claims, start=1):
                cid = f"{ex_id}_c{i:02d}"

                kw = keywords_from_text(cl, k=12)
                if cap:
                    kw += keywords_from_text(cap, k=8)
                for a in aliases:
                    kw += keywords_from_text(a, k=3)

                rows.append(
                    dict(
                        id=ex_id,
                        book_name=book_name,
                        char=char,
                        caption=caption,
                        char_aliases=aliases,
                        claim_id=cid,
                        claim_text=cl,
                        predicate_form=predicate_form(char, cl),
                        keywords=sorted(set(kw)),
                        claim_type=guess_claim_type(cl),
                        t_hint=guess_time_hint(cl),
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
