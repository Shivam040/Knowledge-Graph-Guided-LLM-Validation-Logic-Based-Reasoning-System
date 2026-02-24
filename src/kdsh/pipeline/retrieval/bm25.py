from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List

def bm25_prepare(docs_tokens: List[List[str]]) -> Dict[str, Any]:
    N = len(docs_tokens)
    df = Counter()
    doc_lens = []
    for toks in docs_tokens:
        doc_lens.append(len(toks))
        for t in set(toks):
            df[t] += 1
    avgdl = sum(doc_lens) / max(1, N)
    idf = {t: math.log(1 + (N - n_q + 0.5) / (n_q + 0.5)) for t, n_q in df.items()}
    return {"N": N, "df": df, "idf": idf, "doc_lens": doc_lens, "avgdl": avgdl}

def bm25_scores(query_tokens: List[str], docs_tokens: List[List[str]], prep: Dict[str, Any], k1=1.5, b=0.75) -> List[float]:
    idf = prep["idf"]
    avgdl = prep["avgdl"]
    scores = [0.0] * len(docs_tokens)
    qfreq = Counter(query_tokens)
    for i, toks in enumerate(docs_tokens):
        dl = prep["doc_lens"][i]
        tf = Counter(toks)
        s = 0.0
        for term, _qf in qfreq.items():
            if term not in tf:
                continue
            term_idf = idf.get(term, 0.0)
            f = tf[term]
            denom = f + k1 * (1 - b + b * (dl / avgdl))
            s += term_idf * (f * (k1 + 1) / denom)
        scores[i] = s
    return scores
