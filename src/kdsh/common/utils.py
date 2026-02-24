from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

IST = timezone(timedelta(hours=5, minutes=30))
TOKEN_RE = re.compile(r"[a-z0-9']+")

STOP_ENTS = set("""
The A An And But Or Nor For So Yet He She I We You They His Her Their Our Your
This That These Those Chapter CHAPTER Book BOOK
""".split())

STOPWORDS = set("""
a an and are as at be been but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself
just me more most my myself no nor not of off on once only or other our ours ourselves out over own
same she should so some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which while who why will with
you your yours yourself yourselves
""".split())

NEG_WORDS = {"not", "never", "no", "none", "neither", "nor", "without", "n't"}

def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text).lower())

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_text(path: Path) -> Tuple[str, str]:
    try:
        return path.read_text(encoding="utf-8"), "utf-8"
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace"), "latin-1"

def normalize_book_key(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

def book_code_from_name(book_name: str) -> str:
    s = normalize_book_key(book_name)
    if "castaways" in s:
        return "castaways"
    if "montecristo" in s or "montecr" in s:
        return "montecristo"
    return s[:16] if s else "book"

def time_bucket(pos: float) -> str:
    if pos <= 0.33:
        return "EARLY"
    if pos <= 0.66:
        return "MID"
    return "LATE"

def extract_entities(chunk_text: str, max_entities: int = 12) -> List[str]:
    toks = re.findall(r"\b[A-Z][a-z]{2,}\b", chunk_text)
    freq: Dict[str, int] = {}
    for t in toks:
        if t in STOP_ENTS:
            continue
        freq[t] = freq.get(t, 0) + 1
    ents = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:max_entities]
    return [e for e, _ in ents]

def safe_fname(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(s))
