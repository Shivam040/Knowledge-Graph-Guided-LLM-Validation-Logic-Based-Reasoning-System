# step4_verify.py
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


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    # normalize Gutenberg punctuation/quotes
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

def _get_aliases(claim_row: Dict[str, Any]) -> List[str]:
    """
    Safe default alias getter. Keeps behavior stable even if upstream omitted aliases.
    """
    char = str(claim_row.get("char", "")).strip()
    aliases = claim_row.get("char_aliases") or []
    aliases = [str(a).strip() for a in aliases if a and str(a).strip()]
    if char and all(_norm_text(char) != _norm_text(a) for a in aliases):
        aliases.insert(0, char)
    return aliases

def _expand_aliases(claim_row: Dict[str, Any]) -> List[str]:
    """
    Make alias matching robust:
    - include char + aliases
    - include first token + last token for multi-token names (e.g., "Edward", "Glenarvan")
    - keep originals (we normalize at match time)
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
            # add last name + first name variants (helps "Lord Edward Glenarvan" vs "Glenarvan")
            first = toks[0]
            last = toks[-1]
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

def _name_matches_alias(name: str, aliases: List[str]) -> bool:
    """
    Does the extracted subject name correspond to this claim's character?
    We allow partial (e.g., "Glenarvan" matches "Lord Edward Glenarvan").
    """
    nt = _alias_tokens(name)
    if not nt:
        return False

    for a in aliases:
        at = _alias_tokens(a)
        if not at:
            continue
        if nt == at:
            return True
        if len(at) == 1 and at[0] in nt:
            return True
        if len(nt) == 1 and nt[0] in at:
            return True
    return False


# ============================================================
# SUPPORT anchor gate: downgrade SUPPORT -> NEUTRAL unless the passage
# contains at least one non-name "claim anchor" token (e.g., barrels/vineyard/ciphered).
# Also ensure the passage mentions the target character (alias gate).
# ============================================================

_STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","by","from","as","at",
    "he","she","it","they","his","her","their","him","them","this","that","these","those",
    "was","were","is","are","be","been","being","do","did","does","have","has","had",
    "will","would","should","could","may","might","must","can","cannot","not","no",
    "into","over","under","after","before","during","while","then","than","very",
    "own","later","early","mid","late","secretly","quietly","young","old"
}

def _claim_proper_noun_tokens(claim_text: str) -> set[str]:
    """
    Extract tokens from Capitalized words in the original claim_text.
    We remove these from anchors so names like 'Villefort' don't satisfy the anchor gate.
    Unicode-safe via regex module.
    """
    claim_text = str(claim_text or "")
    # capture capitalized words (Unicode-aware), length>=3
    names = re2.findall(r"(?V1)\b\p{Lu}[\p{L}\p{M}'\-]{2,}\b", claim_text)
    out: set[str] = set()
    for n in names:
        for t in _alias_tokens(n):
            if t:
                out.add(t)
    return out

def _claim_anchor_tokens(claim_row: Dict[str, Any]) -> List[str]:
    """
    Anchor tokens = claim_text tokens (len>=4), minus stopwords, minus aliases,
    minus proper-noun tokens (names/places).
    """
    claim_text = str(claim_row.get("claim_text", "") or "").strip()
    if not claim_text:
        return []

    toks = _alias_tokens(claim_text)
    toks = [t for t in toks if len(t) >= 4 and (t not in _STOPWORDS) and (not t.isdigit())]

    # remove any alias tokens (target char + aliases)
    alias_toks: set[str] = set()
    for a in _expand_aliases(claim_row):
        for t in _alias_tokens(a):
            alias_toks.add(t)

    # remove proper nouns (Villefort/Napoleon/Marseille/etc.)
    proper = _claim_proper_noun_tokens(claim_text)

    toks = [t for t in toks if (t not in alias_toks) and (t not in proper)]

    # keep small for speed
    return toks[:20]

def _passage_has_any_anchor(passage: str, anchors: List[str]) -> bool:
    if not anchors:
        return True
    tl = _norm_text(passage)
    for t in anchors:
        # allow simple suffixes: cipher -> ciphered, barrel -> barrels
        if re.search(rf"\b{re.escape(t)}\w*\b", tl):
            return True
    return False

def _apply_support_gates(
    label: str,
    conf: float,
    claim_row: Dict[str, Any],
    passage: str,
    downgraded_conf_cap: float = 0.55,
) -> Tuple[str, float]:
    """
    If label==SUPPORT:
      1) alias gate: passage must mention the target character (any alias)
      2) anchor gate: passage must contain >=1 non-name anchor token
    Else: unchanged.
    """
    if label != "SUPPORT":
        return label, float(conf)

    aliases = _expand_aliases(claim_row)
    if aliases and not _mentions_any_alias(passage, aliases):
        return "NEUTRAL", float(min(conf, downgraded_conf_cap))

    anchors = _claim_anchor_tokens(claim_row)
    if anchors and not _passage_has_any_anchor(passage, anchors):
        return "NEUTRAL", float(min(conf, downgraded_conf_cap))

    return label, float(conf)



_NEG_RE = re2.compile(r"\b(?:not|never|no|none|n't|without|deny|denies|denied|refuse|refuses|refused)\b", re2.I)

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

def _excerpt_for_span(text: str, span: tuple[int, int]) -> tuple[str, int, int]:
    s, e = span
    for a, b in _sent_spans(text):
        if a <= s < b:
            return text[a:b].strip(), a, b
    a = max(0, s - 120)
    b = min(len(text), e + 120)
    return text[a:b].strip(), a, b

def _is_negated(sentence: str) -> bool:
    return bool(_NEG_RE.search(sentence))

def _add_provenance_to_fact(
    fact: Dict[str, Any],
    passage: str,
    claim_row: Dict[str, Any],
    require_alias_in_sentence: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Attach (excerpt, span_start, span_end) to a fact.
    If require_alias_in_sentence=True, we only keep facts if we can find a sentence
    that mentions an alias of the character (avoids cross-chunk hallucinations).
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

    sent, a, b = best if best is not None else (first if first is not None else ("", 0, 0))
    out = dict(fact)
    out["excerpt"] = sent
    out["span_start"] = int(a)
    out["span_end"] = int(b)
    return out

# ============================================================
# Unicode-aware NAME regex (works for Dantès, Mercédès, Kai-Koumou, etc.)
# IMPORTANT: We wrap NAME with (?-i:...) so IGNORECASE won't break \p{Lu} logic.
# ============================================================

_NAME_BASE = r"(?:\p{Lu}[\p{L}\p{M}'\-]+(?:\s+(?:\p{Lu}[\p{L}\p{M}'\-]+|de|du|da|van|von|la|le|d')){0,4})"
_NAME = rf"(?-i:{_NAME_BASE})"

# ============================================================
# Novel-style high-precision relation patterns
# ============================================================

_WIFE_HUSBAND_PAT = re2.compile(
    rf"(?P<s>{_NAME}).{{0,120}}\b(?:wife|husband)\b[^\p{{L}}]{{0,20}}(?P<o>{_NAME})",
    flags=re2.V1 | re2.I,
)
_MARRIED_TO_PAT = re2.compile(
    rf"(?P<s>{_NAME}).{{0,80}}\bmarried\b.{{0,40}}\b(?:to\s+)?(?P<o>{_NAME})",
    flags=re2.V1 | re2.I,
)
_BETROTHED_TO_PAT = re2.compile(
    rf"(?P<s>{_NAME}).{{0,80}}\b(?:betrothed|engaged)\b.{{0,40}}\bto\s+(?P<o>{_NAME})",
    flags=re2.V1 | re2.I,
)
_DIED_PAT = re2.compile(
    rf"(?P<s>{_NAME}).{{0,60}}\b(?:died|dead|killed|executed|passed\s+away)\b",
    flags=re2.V1 | re2.I,
)
_BORNIN_PAT = re2.compile(
    rf"(?P<s>{_NAME}).{{0,60}}\bborn\s+(?:in|at)\s+(?P<o>{_NAME})",
    flags=re2.V1 | re2.I,
)
_LIVESIN_PAT = re2.compile(
    rf"(?P<s>{_NAME}).{{0,60}}\b(?:lives|lived|resides|resided|dwells)\s+(?:in|at)\s+(?P<o>{_NAME})",
    flags=re2.V1 | re2.I,
)

def _mix_fact_conf(base: float, evidence_conf: float) -> float:
    """
    Facts are extracted from explicit surface patterns in the passage,
    so base confidence should dominate.
    """
    base = float(min(1.0, max(0.0, base)))
    ec = float(min(1.0, max(0.0, evidence_conf)))
    return float(min(1.0, 0.80 * base + 0.20 * ec))

def extract_novel_style_facts(
    claim_row: Dict[str, Any],
    passage: str,
    evidence_conf: float,
    require_alias_in_sentence: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extract facts that appear in Gutenberg-style narrative:
    - MarriedTo via "wife/husband, <Name>" and "married to"
    - BetrothedTo via "betrothed/engaged to"
    - Died via "<Name> died/dead/killed..."
    - BornIn/LivesIn when explicitly stated
    """
    char = str(claim_row.get("char", "")).strip()
    aliases = _expand_aliases(claim_row)

    facts: List[Dict[str, Any]] = []

    for a_s, b_s in _sent_spans(passage):
        sent = passage[a_s:b_s]
        if not sent.strip():
            continue
        if _is_negated(sent):
            continue
        if require_alias_in_sentence and aliases and not _mentions_any_alias(sent, aliases):
            continue

        for m in _WIFE_HUSBAND_PAT.finditer(sent):
            subj = m.group("s").strip()
            obj = m.group("o").strip()
            if aliases and not _name_matches_alias(subj, aliases):
                continue
            if _norm_text(obj) == _norm_text(subj):
                continue
            facts.append(dict(
                s=char, p="MarriedTo", o=obj, polarity="POS",
                fact_confidence=_mix_fact_conf(0.88, evidence_conf),
                excerpt=sent.strip(), span_start=a_s, span_end=b_s
            ))

        for m in _MARRIED_TO_PAT.finditer(sent):
            subj = m.group("s").strip()
            obj = m.group("o").strip()
            if aliases and not _name_matches_alias(subj, aliases):
                continue
            if _norm_text(obj) == _norm_text(subj):
                continue
            facts.append(dict(
                s=char, p="MarriedTo", o=obj, polarity="POS",
                fact_confidence=_mix_fact_conf(0.86, evidence_conf),
                excerpt=sent.strip(), span_start=a_s, span_end=b_s
            ))

        for m in _BETROTHED_TO_PAT.finditer(sent):
            subj = m.group("s").strip()
            obj = m.group("o").strip()
            if aliases and not _name_matches_alias(subj, aliases):
                continue
            if _norm_text(obj) == _norm_text(subj):
                continue
            facts.append(dict(
                s=char, p="BetrothedTo", o=obj, polarity="POS",
                fact_confidence=_mix_fact_conf(0.82, evidence_conf),
                excerpt=sent.strip(), span_start=a_s, span_end=b_s
            ))

        for m in _DIED_PAT.finditer(sent):
            subj = m.group("s").strip()
            if aliases and not _name_matches_alias(subj, aliases):
                continue
            facts.append(dict(
                s=char, p="Died", o=True, polarity="POS",
                fact_confidence=_mix_fact_conf(0.84, evidence_conf),
                excerpt=sent.strip(), span_start=a_s, span_end=b_s
            ))

        for m in _BORNIN_PAT.finditer(sent):
            subj = m.group("s").strip()
            place = m.group("o").strip()
            if aliases and not _name_matches_alias(subj, aliases):
                continue
            facts.append(dict(
                s=char, p="BornIn", o=place, polarity="POS",
                fact_confidence=_mix_fact_conf(0.80, evidence_conf),
                excerpt=sent.strip(), span_start=a_s, span_end=b_s
            ))

        for m in _LIVESIN_PAT.finditer(sent):
            subj = m.group("s").strip()
            place = m.group("o").strip()
            if aliases and not _name_matches_alias(subj, aliases):
                continue
            facts.append(dict(
                s=char, p="LivesIn", o=place, polarity="POS",
                fact_confidence=_mix_fact_conf(0.78, evidence_conf),
                excerpt=sent.strip(), span_start=a_s, span_end=b_s
            ))

    return facts

# ============================================================
# Step4.5 extractors (Age/BornYear/etc.)
# ============================================================

_BORN_YEAR_RE = re2.compile(r"\bborn\b.{0,40}?\b(?P<year>1[5-9]\d{2}|20\d{2})\b", re2.I)
_AGE_NUM_RE = re2.compile(
    r"\b(?:aged?\s*(?:about\s*)?|age\s*[:\-]?\s*)(?P<age>\d{1,3})\b|\b(?P<age2>\d{1,3})\s+(?:years?|yrs?)\s+old\b",
    re2.I,
)

def _parse_word_age(sentence: str) -> int | None:
    try:
        from word2number import w2n
    except Exception:
        return None
    m = re2.search(r"\b([a-z\- ]{3,30})\s+(?:years?|yrs?)\s+old\b", sentence, flags=re2.I)
    if not m:
        return None
    phrase = " ".join(m.group(1).split())
    try:
        return int(w2n.word_to_num(phrase))
    except Exception:
        return None

def extract_facts_step45(
    claim_row: Dict[str, Any],
    passage: str,
    evidence_conf: float,
    require_alias_in_sentence: bool = False,
) -> List[Dict[str, Any]]:
    char = str(claim_row.get("char", "")).strip()
    aliases = _expand_aliases(claim_row)

    def ok_sentence(sent: str) -> bool:
        if _is_negated(sent):
            return False
        if require_alias_in_sentence and aliases and not _mentions_any_alias(sent, aliases):
            return False
        return True

    facts: List[Dict[str, Any]] = []

    for m in _BORN_YEAR_RE.finditer(passage):
        sent, a, b = _excerpt_for_span(passage, m.span())
        if not ok_sentence(sent):
            continue
        year = m.group("year")
        facts.append(dict(
            s=char, p="BornYear", o=int(year), polarity="POS",
            fact_confidence=_mix_fact_conf(0.78, evidence_conf),
            excerpt=sent, span_start=a, span_end=b
        ))

    m = _AGE_NUM_RE.search(passage)
    if m:
        sent, a, b = _excerpt_for_span(passage, m.span())
        if ok_sentence(sent):
            age = m.group("age") or m.group("age2")
            if age:
                facts.append(dict(
                    s=char, p="Age", o=int(age), polarity="POS",
                    fact_confidence=_mix_fact_conf(0.76, evidence_conf),
                    excerpt=sent, span_start=a, span_end=b
                ))
    else:
        for a_s, b_s in _sent_spans(passage):
            sent = passage[a_s:b_s]
            if not ok_sentence(sent):
                continue
            w_age = _parse_word_age(sent)
            if w_age is not None:
                facts.append(dict(
                    s=char, p="Age", o=int(w_age), polarity="POS",
                    fact_confidence=_mix_fact_conf(0.74, evidence_conf),
                    excerpt=sent.strip(), span_start=a_s, span_end=b_s
                ))
                break

    return facts

# ============================================================
# Older predicate matchers (kept for backward compatibility)
# ============================================================

def contains_negation_near(text_lower: str, phrase_lower: str, window: int = 40) -> bool:
    idx = text_lower.find(phrase_lower)
    if idx == -1:
        return False
    start = max(0, idx - window)
    end = min(len(text_lower), idx + len(phrase_lower) + window)
    ctx = text_lower[start:end]
    return any(n in ctx.split() for n in NEG_WORDS)

def match_bornin(char: str, place: str, passage: str) -> Tuple[bool, bool, Optional[str]]:
    tl = _norm_text(passage)
    pl = _norm_text(place)
    support = bool(re.search(rf"\bborn\s+in\s+{re.escape(pl)}\b", tl))
    contradict = contains_negation_near(tl, f"born in {pl}")
    other_place = None
    m = re.search(r"\bborn\s+in\s+([a-z][a-z\s\-]{2,40})", tl)
    if m:
        cand = " ".join(m.group(1).strip().split()[0:5]).strip(" .,:;!?'\"")
        if cand and pl not in cand and cand != pl:
            other_place = cand.title()
    return support, contradict, other_place

def match_livesin(char: str, loc: str, passage: str) -> Tuple[bool, bool, Optional[str]]:
    tl = _norm_text(passage)
    ll = _norm_text(loc)
    support = bool(re.search(rf"\b(lives|lived|resides|resided)\s+in\s+{re.escape(ll)}\b", tl))
    contradict = contains_negation_near(tl, f"lives in {ll}") or contains_negation_near(tl, f"lived in {ll}")
    other = None
    m = re.search(r"\b(lives|lived|resides|resided)\s+in\s+([a-z][a-z\s\-]{2,40})", tl)
    if m:
        cand = " ".join(m.group(2).strip().split()[0:5]).strip(" .,:;!?'\"")
        if cand and ll not in cand and cand != ll:
            other = cand.title()
    return support, contradict, other

def match_marriedto(char: str, other: str, passage: str) -> Tuple[bool, bool]:
    tl = _norm_text(passage)
    ol = _norm_text(other)
    support = bool(re.search(rf"\bmarried\s+(to\s+)?{re.escape(ol)}\b", tl))
    contradict = contains_negation_near(tl, f"married to {ol}") or contains_negation_near(tl, "married")
    return support, contradict

def match_died(passage: str) -> Tuple[bool, bool]:
    tl = _norm_text(passage)
    support = any(w in tl for w in [" died", "dead", "killed", "executed", "passed away"])
    contradict = contains_negation_near(tl, "died") or contains_negation_near(tl, "dead")
    return support, contradict


def extract_facts_from_predicate_form(
    claim_row: Dict[str, Any],
    passage: str,
    evidence_conf: float,
) -> List[Dict[str, Any]]:
    """
    If a claim is SUPPORT, we can safely record its predicate_form as a fact.
    This guarantees KG triples even when surface patterns miss novel phrasing.
    """
    char = str(claim_row.get("char", "")).strip()
    preds = claim_row.get("predicate_form", []) or []
    out: List[Dict[str, Any]] = []

    for p in preds:
        parsed = parse_predicate(p)
        if not parsed:
            continue
        name, args = parsed

        # normalize predicate names to your KG schema
        pred = name

        # Build object
        obj: Any = None
        if pred in ("Died",):
            obj = True
        elif len(args) >= 2:
            obj = args[1]
        else:
            continue

        # Try numeric types where applicable
        if pred in ("Age", "BornYear"):
            try:
                obj = int(obj)
            except Exception:
                continue

        # strong base so it survives kg.min_fact_conf=0.65
        base = 0.86 if pred not in ("Age",) else 0.78
        fact = dict(
            s=char,
            p=pred,
            o=obj,
            polarity="POS",
            fact_confidence=_mix_fact_conf(base, evidence_conf),
        )

        # attach provenance even if alias sentence isn't found
        pf = _add_provenance_to_fact(fact, passage, claim_row, require_alias_in_sentence=False)
        out.append(pf if pf is not None else fact)

    return out


def parse_predicate(p: str) -> Optional[Tuple[str, List[str]]]:
    m = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$", str(p))
    if not m:
        return None
    name = m.group(1)
    inside = m.group(2)
    args = [a.strip().strip('"').strip("'") for a in inside.split(",")]
    return name, args

def generic_overlap(claim_text: str, passage: str, char: Optional[str] = None) -> float:
    ct = tokenize(claim_text)
    pt = tokenize(passage)
    if not ct or not pt:
        return 0.0
    cset, pset = set(ct), set(pt)
    inter = len(cset & pset)
    base = inter / max(1, len(cset))
    if char and _norm_text(char) in _norm_text(passage):
        base *= 1.15
    return min(1.0, base)

def match_childof(char: str, parent: str, passage: str) -> Tuple[bool, bool]:
    tl = passage.lower()
    pl = parent.lower()
    support = bool(re.search(rf"\b(son|daughter|child)\s+of\s+{re.escape(pl)}\b", tl))
    contradict = (
        contains_negation_near(tl, f"son of {pl}")
        or contains_negation_near(tl, f"daughter of {pl}")
        or contains_negation_near(tl, f"child of {pl}")
    )
    return support, contradict

def match_parentof(char: str, child: str, passage: str) -> Tuple[bool, bool]:
    tl = passage.lower()
    cl = child.lower()
    support = bool(re.search(rf"\b(father|mother|parent)\s+of\s+{re.escape(cl)}\b", tl))
    contradict = (
        contains_negation_near(tl, f"father of {cl}")
        or contains_negation_near(tl, f"mother of {cl}")
        or contains_negation_near(tl, f"parent of {cl}")
    )
    return support, contradict

def match_siblingof(char: str, sib: str, passage: str) -> Tuple[bool, bool]:
    tl = passage.lower()
    sl = sib.lower()
    support = bool(re.search(rf"\b(brother|sister)\s+of\s+{re.escape(sl)}\b", tl))
    contradict = contains_negation_near(tl, f"brother of {sl}") or contains_negation_near(tl, f"sister of {sl}")
    return support, contradict

def match_knownas(char: str, alias_name: str, passage: str) -> Tuple[bool, bool]:
    tl = passage.lower()
    al = alias_name.lower()
    support = bool(re.search(rf"\b(was\s+called|called|known\s+as|named)\s+{re.escape(al)}\b", tl))
    contradict = (
        contains_negation_near(tl, f"called {al}")
        or contains_negation_near(tl, f"known as {al}")
        or contains_negation_near(tl, f"named {al}")
    )
    return support, contradict

# ============================================================
# Label decision (returns extracted_facts too)
# ============================================================

def decide_label(claim_row: Dict[str, Any], passage: str, score_norm: float) -> Tuple[str, float, List[Dict[str, Any]]]:
    """
    Updated:
    - Supports more predicates used in novels (ChildOf/ParentOf/SiblingOf/KnownAs)
    - Avoids NEUTRAL->SUPPORT based only on overlap (that created SUPPORT with zero facts)
    - Extracts facts only when there is a concrete pattern match in the passage
    """
    char = str(claim_row.get("char", "")).strip()
    preds = claim_row.get("predicate_form", []) or []
    aliases = claim_row.get("char_aliases") or _get_aliases(claim_row)

    label = "NEUTRAL"
    conf = 0.20 + 0.30 * float(score_norm or 0.0)
    extracted_facts: List[Dict[str, Any]] = []

    def _safe_emit_fact(p: str, o: Any, base_conf: float) -> None:
        f = dict(s=char, p=p, o=o, polarity="POS", fact_confidence=float(min(1.0, base_conf)))
        pf = _add_provenance_to_fact(f, passage, claim_row, require_alias_in_sentence=True)
        if pf is not None:
            extracted_facts.append(pf)

    passage_mentions_char = _mentions_any_alias(passage, [str(a) for a in aliases if a])

    for p in preds:
        parsed = parse_predicate(p)
        if not parsed:
            continue
        name, args = parsed

        if name == "BornIn" and len(args) >= 2:
            place = args[1]
            sup, con, other = match_bornin(char, place, passage)
            if sup and passage_mentions_char:
                label = "SUPPORT"
                conf = max(conf, 0.82 + 0.12 * float(score_norm))
                _safe_emit_fact("BornIn", place, conf)
            elif con or other:
                label = "CONTRADICT"
                conf = max(conf, (0.78 if con else 0.62) + 0.15 * float(score_norm))
                if other:
                    _safe_emit_fact("BornIn", other, conf)

        elif name == "LivesIn" and len(args) >= 2:
            loc = args[1]
            sup, con, other = match_livesin(char, loc, passage)
            if sup and passage_mentions_char:
                label = "SUPPORT"
                conf = max(conf, 0.80 + 0.14 * float(score_norm))
                _safe_emit_fact("LivesIn", loc, conf)
            elif con or other:
                label = "CONTRADICT"
                conf = max(conf, (0.72 if con else 0.58) + 0.15 * float(score_norm))
                if other:
                    _safe_emit_fact("LivesIn", other, conf)

        elif name == "MarriedTo" and len(args) >= 2:
            other = args[1]
            sup, con = match_marriedto(char, other, passage)
            if sup and passage_mentions_char:
                label = "SUPPORT"
                conf = max(conf, 0.80 + 0.14 * float(score_norm))
                _safe_emit_fact("MarriedTo", other, conf)
            elif con:
                label = "CONTRADICT"
                conf = max(conf, 0.64 + 0.18 * float(score_norm))

        elif name == "Died":
            sup, con = match_died(passage)
            if sup and passage_mentions_char:
                label = "SUPPORT"
                conf = max(conf, 0.80 + 0.14 * float(score_norm))
                _safe_emit_fact("Died", True, conf)
            elif con:
                label = "CONTRADICT"
                conf = max(conf, 0.62 + 0.18 * float(score_norm))

        elif name in ("ChildOf", "SonOf", "DaughterOf") and len(args) >= 2:
            parent = args[1]
            sup, con = match_childof(char, parent, passage)
            if sup and passage_mentions_char:
                label = "SUPPORT"
                conf = max(conf, 0.84 + 0.10 * float(score_norm))
                _safe_emit_fact("ChildOf", parent, conf)
            elif con:
                label = "CONTRADICT"
                conf = max(conf, 0.64 + 0.18 * float(score_norm))

        elif name == "ParentOf" and len(args) >= 2:
            child = args[1]
            sup, con = match_parentof(char, child, passage)
            if sup and passage_mentions_char:
                label = "SUPPORT"
                conf = max(conf, 0.84 + 0.10 * float(score_norm))
                _safe_emit_fact("ParentOf", child, conf)
            elif con:
                label = "CONTRADICT"
                conf = max(conf, 0.64 + 0.18 * float(score_norm))

        elif name == "SiblingOf" and len(args) >= 2:
            sib = args[1]
            sup, con = match_siblingof(char, sib, passage)
            if sup and passage_mentions_char:
                label = "SUPPORT"
                conf = max(conf, 0.82 + 0.12 * float(score_norm))
                _safe_emit_fact("SiblingOf", sib, conf)
            elif con:
                label = "CONTRADICT"
                conf = max(conf, 0.64 + 0.18 * float(score_norm))

        elif name in ("KnownAs", "Called", "NamedAs") and len(args) >= 2:
            aka = args[1]
            sup, con = match_knownas(char, aka, passage)
            if sup and passage_mentions_char:
                label = "SUPPORT"
                conf = max(conf, 0.82 + 0.12 * float(score_norm))
                _safe_emit_fact("KnownAs", aka, conf)
            elif con:
                label = "CONTRADICT"
                conf = max(conf, 0.62 + 0.18 * float(score_norm))

        if label in ("SUPPORT", "CONTRADICT") and conf >= 0.90:
            break

    # IMPORTANT: do NOT promote to SUPPORT via overlap only
    if label == "NEUTRAL":
        ov = generic_overlap(str(claim_row.get("claim_text", "")), passage, char=char)

        # ✅ bring back NEUTRAL->SUPPORT, but ONLY if passage mentions the character
        if ov >= 0.22 and passage_mentions_char:
            label = "SUPPORT"
            conf = max(conf, 0.55 + 0.35 * ov + 0.10 * float(score_norm or 0.0))
        else:
            conf = max(conf, 0.25 + 0.50 * ov)


    return label, float(min(1.0, max(0.0, conf))), extracted_facts

# ============================================================
# Generic fallback fact: guarantee facts.jsonl isn't empty
# ============================================================

def _fallback_support_fact(
    claim_row: Dict[str, Any],
    passage: str,
    evidence_conf: float,
) -> Dict[str, Any]:
    """
    If a retrieval row is labeled SUPPORT but we didn't extract a concrete triple
    (MarriedTo/BornIn/etc.), emit a generic fact so facts.jsonl is never empty.

    This is safe: it does NOT invent a relation; it only records that this passage
    supports the claim text.
    """
    char = str(claim_row.get("char", "")).strip()
    claim_text = str(claim_row.get("claim_text", "")).strip()
    aliases = _expand_aliases(claim_row)

    excerpt = ""
    span_start, span_end = 0, min(len(passage), 400)

    first_sent = None
    for a, b in _sent_spans(passage):
        sent = passage[a:b].strip()
        if not sent:
            continue
        if first_sent is None:
            first_sent = (sent, a, b)
        if aliases and _mentions_any_alias(sent, aliases):
            excerpt, span_start, span_end = sent, a, b
            break

    if not excerpt and first_sent is not None:
        excerpt, span_start, span_end = first_sent

    if not excerpt:
        excerpt = passage[:span_end].strip()

    return dict(
        s=char,
        p="SupportsClaim",
        o=claim_text if claim_text else str(claim_row.get("claim_id", "")),
        polarity="POS",
        fact_confidence=float(min(1.0, max(0.55, evidence_conf))),
        excerpt=excerpt,
        span_start=int(span_start),
        span_end=int(span_end),
    )

# ============================================================
# Helpers for MNLI backend
# ============================================================

def _cfg_get(cfg: Optional[VerifierConfig], key: str, default: Any) -> Any:
    if cfg is None:
        return default
    return getattr(cfg, key, default)

def _mnli_to_kdsh_label(mnli_label: str) -> str:
    l = str(mnli_label or "").strip().lower()
    if "entail" in l:
        return "SUPPORT"
    if "contrad" in l:
        return "CONTRADICT"
    return "NEUTRAL"

def _mix_conf(nli_conf: float, score_norm: float, alpha: float) -> float:
    a = float(alpha)
    a = 0.0 if a < 0 else 1.0 if a > 1.0 else a
    conf = (1.0 - a) * float(nli_conf) + a * float(score_norm)
    return float(min(1.0, max(0.0, conf)))

# ============================================================
# Step 4
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
    Novel-friendly updates:
    - Unicode-safe alias matching (diacritics, curly apostrophes, hyphens)
    - Novel-style relation patterns (wife/husband/married/betrothed/died)
    - Fact confidence is pattern-driven
    - GUARANTEE: If label==SUPPORT but no pattern-facts fire, emit SupportsClaim.
    """

    backend = _cfg_get(verifier_cfg, "backend", None)
    if not backend:
        backend = os.getenv("KDSH_VERIFIER_BACKEND", "heuristic")
    backend = str(backend).strip().lower()

    model_name = _cfg_get(verifier_cfg, "model_name", None) or os.getenv(
        "KDSH_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    )
    batch_size = int(_cfg_get(verifier_cfg, "batch_size", int(os.getenv("KDSH_NLI_BATCH", "16"))))
    device = _cfg_get(verifier_cfg, "device", os.getenv("KDSH_NLI_DEVICE", "auto"))
    max_length = int(_cfg_get(verifier_cfg, "max_length", int(os.getenv("KDSH_NLI_MAXLEN", "512"))))
    score_mix_alpha = float(_cfg_get(verifier_cfg, "score_mix_alpha", float(os.getenv("KDSH_NLI_ALPHA", "0.15"))))

    claim_rows = [
        json.loads(line)
        for line in claims_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    claim_map = {r["claim_id"]: r for r in claim_rows}
    retrieval_df = pd.read_csv(retrieval_path)

    chunk_text_map = dict(zip(chunks_df["chunk_id"], chunks_df["chunk_text"]))

    evidence_rows: List[Dict[str, Any]] = []
    facts_rows: List[Dict[str, Any]] = []

    work: List[Tuple[Dict[str, Any], Dict[str, Any], str, float]] = []
    for rr in retrieval_df.to_dict(orient="records"):
        cid = rr["claim_id"]
        claim = claim_map.get(cid)
        if not claim:
            continue
        passage = chunk_text_map.get(rr["chunk_id"], "")
        score_norm = float(rr.get("score_lex_norm", 0.0) or 0.0)
        work.append((rr, claim, passage, score_norm))

    if backend == "mnli":
        from kdsh.pipeline.verification.hf_mnli import HFNLIVerifier

        nli = HFNLIVerifier(
            model_name=str(model_name),
            device=str(device),
            batch_size=int(batch_size),
            max_length=int(max_length),
        )

        premises = [p for (_rr, _cl, p, _sn) in work]
        hypotheses = [str(cl.get("claim_text", "")) for (_rr, cl, _p, _sn) in work]
        nli_out = nli.predict_batch(premises, hypotheses)

        for (rr, claim, passage, score_norm), pred in zip(work, nli_out):
            label = _mnli_to_kdsh_label(getattr(pred, "label", "neutral"))
            nli_conf = float(getattr(pred, "confidence", 0.0))
            conf = _mix_conf(nli_conf, score_norm, score_mix_alpha)

            # ✅ Apply gates BEFORE writing evidence_rows
            label, conf = _apply_support_gates(label, conf, claim, passage)

            evidence_rows.append(
                dict(
                    id=int(rr["id"]),
                    claim_id=rr["claim_id"],
                    book_name=rr["book_name"],
                    chunk_id=rr["chunk_id"],
                    label=label,
                    confidence=float(conf),
                    verifier_model=f"mnli:{model_name}",
                    run_id=run_id,
                )
            )

            # ✅ Only keep SUPPORT for facts.jsonl (your decision)
            if label != "SUPPORT":
                continue

            require_alias = False

            facts: List[Dict[str, Any]] = []
            facts += extract_novel_style_facts(claim, passage, evidence_conf=conf, require_alias_in_sentence=require_alias)
            facts += extract_facts_step45(claim, passage, evidence_conf=conf, require_alias_in_sentence=require_alias)

            if not facts:
                facts += extract_facts_from_predicate_form(claim, passage, evidence_conf=conf)

            _lbl_h, _conf_h, facts_pred = decide_label(claim, passage, score_norm=0.0)
            if facts_pred:
                for f in facts_pred:
                    f2 = dict(f)
                    f2["fact_confidence"] = float(min(1.0, max(f2.get("fact_confidence", 0.0), _mix_fact_conf(0.75, conf))))
                    facts.append(f2)

            added_any = False
            for fct in facts:
                fc = float(fct.get("fact_confidence", 0.0))
                if fc < min_fact_conf:
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

            if not added_any:
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


        # for (rr, claim, passage, score_norm), pred in zip(work, nli_out):
        #     label = _mnli_to_kdsh_label(getattr(pred, "label", "neutral"))
        #     nli_conf = float(getattr(pred, "confidence", 0.0))
        #     conf = _mix_conf(nli_conf, score_norm, score_mix_alpha)

        #     evidence_rows.append(
        #         dict(
        #             id=int(rr["id"]),
        #             claim_id=rr["claim_id"],
        #             book_name=rr["book_name"],
        #             chunk_id=rr["chunk_id"],
        #             label=label,
        #             confidence=float(conf),
        #             verifier_model=f"mnli:{model_name}",
        #             run_id=run_id,
        #         )
        #     )

        #     if label != "SUPPORT":
        #         continue

        #     require_alias = False

        #     facts: List[Dict[str, Any]] = []
        #     facts += extract_novel_style_facts(claim, passage, evidence_conf=conf, require_alias_in_sentence=require_alias)
        #     facts += extract_facts_step45(claim, passage, evidence_conf=conf, require_alias_in_sentence=require_alias)


        #     # ✅ If SUPPORT but patterns didn't fire, emit predicate_form facts
        #     if label == "SUPPORT" and not facts:
        #         facts += extract_facts_from_predicate_form(claim, passage, evidence_conf=conf)

        #     # Optional: also include predicate-derived facts (high precision)
        #     _lbl_h, _conf_h, facts_pred = decide_label(claim, passage, score_norm=0.0)
        #     if facts_pred:
        #         for f in facts_pred:
        #             f2 = dict(f)
        #             f2["fact_confidence"] = float(min(1.0, max(f2.get("fact_confidence", 0.0), _mix_fact_conf(0.75, conf))))
        #             facts.append(f2)

        #     added_any = False
        #     for fct in facts:
        #         fc = float(fct.get("fact_confidence", 0.0))
        #         if fc < min_fact_conf:
        #             continue
        #         fct2 = dict(fct)
        #         fct2.update(
        #             dict(
        #                 id=int(rr["id"]),
        #                 claim_id=rr["claim_id"],
        #                 book_name=rr["book_name"],
        #                 chunk_id=rr["chunk_id"],
        #                 label=label,
        #                 confidence=float(conf),
        #                 run_id=run_id,
        #             )
        #         )
        #         facts_rows.append(fct2)
        #         added_any = True

        #     if label == "SUPPORT" and not added_any:
        #         fct2 = _fallback_support_fact(claim, passage, evidence_conf=conf)
        #         fct2.update(
        #             dict(
        #                 id=int(rr["id"]),
        #                 claim_id=rr["claim_id"],
        #                 book_name=rr["book_name"],
        #                 chunk_id=rr["chunk_id"],
        #                 label=label,
        #                 confidence=float(conf),
        #                 run_id=run_id,
        #             )
        #         )
        #         facts_rows.append(fct2)

    else:
        for rr, claim, passage, score_norm in work:
            label, conf, facts_pred = decide_label(claim, passage, score_norm)

            # ✅ Apply gates BEFORE writing evidence_rows
            label, conf = _apply_support_gates(label, conf, claim, passage)

            evidence_rows.append(
                dict(
                    id=int(rr["id"]),
                    claim_id=rr["claim_id"],
                    book_name=rr["book_name"],
                    chunk_id=rr["chunk_id"],
                    label=label,
                    confidence=float(conf),
                    verifier_model="heuristic_v1",
                    run_id=run_id,
                )
            )

            if label != "SUPPORT":
                continue

            require_alias = False

            facts: List[Dict[str, Any]] = []
            facts += extract_novel_style_facts(claim, passage, evidence_conf=conf, require_alias_in_sentence=require_alias)
            facts += extract_facts_step45(claim, passage, evidence_conf=conf, require_alias_in_sentence=require_alias)

            if not facts:
                facts += extract_facts_from_predicate_form(claim, passage, evidence_conf=conf)

            if facts_pred:
                for f in facts_pred:
                    f2 = dict(f)
                    f2["fact_confidence"] = float(min(1.0, max(f2.get("fact_confidence", 0.0), _mix_fact_conf(0.75, conf))))
                    facts.append(f2)

            added_any = False
            for fct in facts:
                fc = float(fct.get("fact_confidence", 0.0))
                if fc < min_fact_conf:
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

            if not added_any:
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

        # --- heuristic backend ---
        # for rr, claim, passage, score_norm in work:
        #     label, conf, facts_pred = decide_label(claim, passage, score_norm)

        #     evidence_rows.append(
        #         dict(
        #             id=int(rr["id"]),
        #             claim_id=rr["claim_id"],
        #             book_name=rr["book_name"],
        #             chunk_id=rr["chunk_id"],
        #             label=label,
        #             confidence=float(conf),
        #             verifier_model="heuristic_v1",
        #             run_id=run_id,
        #         )
        #     )

        #     if label != "SUPPORT":
        #         continue

        #     require_alias = False

        #     facts: List[Dict[str, Any]] = []
        #     facts += extract_novel_style_facts(claim, passage, evidence_conf=conf, require_alias_in_sentence=require_alias)
        #     facts += extract_facts_step45(claim, passage, evidence_conf=conf, require_alias_in_sentence=require_alias)
        #     # ✅ If SUPPORT but patterns didn't fire, emit predicate_form facts
        #     if label == "SUPPORT" and not facts:
        #         facts += extract_facts_from_predicate_form(claim, passage, evidence_conf=conf)


        #     # ✅ CRITICAL: include facts returned by decide_label()
        #     if facts_pred:
        #         for f in facts_pred:
        #             f2 = dict(f)
        #             f2["fact_confidence"] = float(min(1.0, max(f2.get("fact_confidence", 0.0), _mix_fact_conf(0.75, conf))))
        #             facts.append(f2)

        #     added_any = False
        #     for fct in facts:
        #         fc = float(fct.get("fact_confidence", 0.0))
        #         if fc < min_fact_conf:
        #             continue
        #         fct2 = dict(fct)
        #         fct2.update(
        #             dict(
        #                 id=int(rr["id"]),
        #                 claim_id=rr["claim_id"],
        #                 book_name=rr["book_name"],
        #                 chunk_id=rr["chunk_id"],
        #                 label=label,
        #                 confidence=float(conf),
        #                 run_id=run_id,
        #             )
        #         )
        #         facts_rows.append(fct2)
        #         added_any = True

        #     # ✅ GUARANTEE: SUPPORT but no extracted facts -> emit SupportsClaim
        #     if label == "SUPPORT" and not added_any:
        #         fct2 = _fallback_support_fact(claim, passage, evidence_conf=conf)
        #         fct2.update(
        #             dict(
        #                 id=int(rr["id"]),
        #                 claim_id=rr["claim_id"],
        #                 book_name=rr["book_name"],
        #                 chunk_id=rr["chunk_id"],
        #                 label=label,
        #                 confidence=float(conf),
        #                 run_id=run_id,
        #             )
        #         )
        #         facts_rows.append(fct2)

    # Write outputs (unchanged contract)
    out_silver.mkdir(parents=True, exist_ok=True)

    evidence_df = pd.DataFrame(evidence_rows)
    evidence_path = out_silver / "evidence_labels.csv"
    evidence_df.to_csv(evidence_path, index=False)

    facts_path = out_silver / "facts.jsonl"
    with facts_path.open("w", encoding="utf-8") as f:
        for r in facts_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return evidence_path, facts_path
