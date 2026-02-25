from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


# -----------------------------
# Simple tokenization utilities
# -----------------------------
_STOP = {
    "the","a","an","and","or","but","if","then","so","to","of","in","on","at","by","for","with","as",
    "is","are","was","were","be","been","being","do","does","did","done","have","has","had",
    "this","that","these","those","it","its","he","she","they","them","his","her","their",
    "i","you","we","us","my","your","our","me","him","hers","theirs",
}
_NEG = {"no","not","never","none","nobody","nothing","neither","nor","without","cannot","can't","won't","doesn't","didn't","isn't","aren't","wasn't","weren't"}


def _tok(text: str) -> List[str]:
    t = str(text or "").lower()
    toks = re.findall(r"[a-z0-9']+", t)
    return [x for x in toks if x and x not in _STOP]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / max(1, len(a | b))


def _has_neg(text: str) -> bool:
    toks = set(_tok(text))
    return any(n in toks for n in _NEG)


# -----------------------------
# Optional NLI (Transformers)
# -----------------------------
class _NLIPredictor:
    """
    Lightweight wrapper around HF transformers pipeline for MNLI-style labels.
    Returns dict with probs for: SUPPORT/CONTRADICT/NEUTRAL.
    """

    def __init__(self, model_name: str, device: str = "auto", max_len: int = 512, batch_size: int = 16):
        self.model_name = model_name
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)

        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:
            raise RuntimeError("transformers not available") from e

        if device == "auto":
            # pipeline uses -1 for cpu, integer GPU id for cuda
            try:
                import torch  # type: ignore
                dev = 0 if torch.cuda.is_available() else -1
            except Exception:
                dev = -1
        elif device.lower() in ("cpu", "-1"):
            dev = -1
        else:
            try:
                dev = int(device)
            except Exception:
                dev = 0

        # Return all scores so we can threshold on contradiction probability
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            device=dev,
            truncation=True,
            max_length=self.max_len,
            return_all_scores=True,
        )

    def predict_proba(self, premises: List[str], hypotheses: List[str]) -> List[Dict[str, float]]:
        assert len(premises) == len(hypotheses)
        inputs = [{"text": p, "text_pair": h} for p, h in zip(premises, hypotheses)]
        out: List[Dict[str, float]] = []
        # batch
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i : i + self.batch_size]
            preds = self.pipe(batch)
            for scores in preds:
                # scores: list[{label,score}]
                d = {s["label"].upper(): float(s["score"]) for s in scores}
                # map MNLI-ish labels
                sup = d.get("ENTAILMENT", d.get("LABEL_2", 0.0))
                con = d.get("CONTRADICTION", d.get("LABEL_0", 0.0))
                neu = d.get("NEUTRAL", d.get("LABEL_1", 0.0))
                out.append({"SUPPORT": float(sup), "CONTRADICT": float(con), "NEUTRAL": float(neu)})
        return out


# -----------------------------
# Step 6: Logic over Claim edges
# -----------------------------
def step6_logic(
    kg_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_gold: Path,
    run_id: str,
):
    """
    Rewritten Step-6:
      - Works when KG is mostly Claim edges: (s, p='Claim', o=<claim_text>)
      - Detects logical inconsistency as CONTRADICTING CLAIMS using NLI (optional) + overlap pruning
      - Emits the same outputs as before:
          constraint_runs.jsonl, constraints_grounded.jsonl

    Status meaning:
      - UNSAT: at least one HARD violation (high-confidence contradiction among supported claims)
      - SOFT_VIOLATIONS: duplicates / low-confidence contradictions
      - SAT: no issues detected
    """
    kg_df = pd.read_csv(kg_path) if kg_path.exists() else pd.DataFrame()
    all_ids = pd.concat([train_df[["id"]], test_df[["id"]]]).drop_duplicates()["id"].astype(int).tolist()

    # Config knobs
    use_nli = os.getenv("KDSH_STEP6_USE_NLI", "1") == "1"
    nli_model = os.getenv("KDSH_STEP6_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    nli_device = os.getenv("KDSH_STEP6_NLI_DEVICE", "auto")
    nli_maxlen = int(os.getenv("KDSH_STEP6_NLI_MAXLEN", "512"))
    nli_batch = int(os.getenv("KDSH_STEP6_NLI_BATCH", "16"))

    # Pair pruning
    min_shared = int(os.getenv("KDSH_STEP6_MIN_SHARED_TOKENS", "2"))
    min_jac = float(os.getenv("KDSH_STEP6_MIN_JACCARD", "0.20"))
    max_pairs = int(os.getenv("KDSH_STEP6_MAX_PAIRS_PER_ID", "400"))

    # Decision thresholds
    contra_thr = float(os.getenv("KDSH_STEP6_CONTRA_THR", "0.85"))
    entail_thr = float(os.getenv("KDSH_STEP6_ENTAIL_THR", "0.60"))
    margin = float(os.getenv("KDSH_STEP6_MARGIN", "0.05"))

    dup_jac = float(os.getenv("KDSH_STEP6_DUP_JACCARD", "0.92"))

    # Initialize NLI once
    nli = None
    if use_nli:
        try:
            nli = _NLIPredictor(nli_model, device=nli_device, max_len=nli_maxlen, batch_size=nli_batch)
        except Exception as e:
            print(f"[step6] WARNING: NLI disabled ({e}). Falling back to lexical-only checks.")
            nli = None

    def _candidate_pairs(claims: List[Dict[str, Any]]) -> List[Tuple[int, int, int]]:
        """
        Return list of (i, j, shared_token_count) candidate pairs using inverted index.
        """
        inv: Dict[str, List[int]] = defaultdict(list)
        toks: List[Set[str]] = []
        for i, c in enumerate(claims):
            ts = set(_tok(c["text"]))
            toks.append(ts)
            for t in ts:
                inv[t].append(i)

        seen = set()
        pairs: List[Tuple[int, int, int]] = []
        for i, ts in enumerate(toks):
            counter: Dict[int, int] = defaultdict(int)
            for t in ts:
                for j in inv.get(t, []):
                    if j <= i:
                        continue
                    counter[j] += 1
            for j, shared in counter.items():
                if shared >= min_shared:
                    key = (i, j)
                    if key in seen:
                        continue
                    seen.add(key)
                    # quick jaccard filter
                    jac = _jaccard(ts, toks[j])
                    if jac >= min_jac or (_has_neg(claims[i]["text"]) ^ _has_neg(claims[j]["text"])) and jac >= (min_jac * 0.6):
                        pairs.append((i, j, shared))

        # If too many, keep the most-overlapping pairs
        if len(pairs) > max_pairs:
            pairs.sort(key=lambda x: x[2], reverse=True)
            pairs = pairs[:max_pairs]
        return pairs

    def _structured_constraints(id_kg: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Keep backward-compatible structured constraints if schema predicates exist.
        """
        violations: List[Dict[str, Any]] = []
        by_p = defaultdict(list)
        for _, r in id_kg.iterrows():
            by_p[str(r.get("p", ""))].append(r)

        for p in ["BornIn", "BornYear"]:
            objs = defaultdict(list)
            for r in by_p.get(p, []):
                objs[str(r.get("o", ""))].append(r)
            if len(objs) >= 2:
                obj_best = max(objs.keys(), key=lambda o: max(float(x.get("confidence", 0.0)) for x in objs[o]))
                conflicting = [o for o in objs.keys() if o != obj_best]
                triples_best = [x.get("triple_id") for x in objs[obj_best]]
                triples_conf = [x.get("triple_id") for o in conflicting for x in objs[o]]
                conf = max(
                    max(float(x.get("confidence", 0.0)) for x in objs[obj_best]),
                    max(float(x.get("confidence", 0.0)) for o in conflicting for x in objs[o]),
                )
                violations.append(
                    dict(
                        rule=f"Unique{p}(A): cannot have multiple distinct {p} objects",
                        supporting_triples=sorted([t for t in (triples_best + triples_conf) if t]),
                        confidence=round(float(conf), 4),
                        severity="HARD",
                    )
                )

        spouses = defaultdict(list)
        for r in by_p.get("MarriedTo", []):
            spouses[str(r.get("o", ""))].append(r)
        if len(spouses) >= 2:
            all_triples = [x.get("triple_id") for os_ in spouses.values() for x in os_]
            conf = max(float(x.get("confidence", 0.0)) for os_ in spouses.values() for x in os_)
            violations.append(
                dict(
                    rule="MarriedTo(A,B) and MarriedTo(A,C) with B!=C (unless time-bucketed non-overlapping)",
                    supporting_triples=sorted([t for t in all_triples if t]),
                    confidence=round(float(conf), 4),
                    severity="SOFT",
                )
            )
        return violations

    def _claim_constraints(id_kg: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect contradictions and near-duplicates among supported Claim edges.
        """
        violations: List[Dict[str, Any]] = []

        # Focus on supported claims if label available
        if "support_label" in id_kg.columns:
            claim_df = id_kg[(id_kg["p"] == "Claim") & (id_kg["support_label"] == "SUPPORT")].copy()
        else:
            claim_df = id_kg[id_kg["p"] == "Claim"].copy()

        if len(claim_df) < 2:
            return violations

        claims: List[Dict[str, Any]] = []
        for _, r in claim_df.iterrows():
            claims.append(
                dict(
                    triple_id=str(r.get("triple_id", "")),
                    text=str(r.get("o", "")),
                    conf=float(r.get("confidence", 0.0) or 0.0),
                )
            )

        pairs = _candidate_pairs(claims)
        if not pairs:
            return violations

        # If no NLI, only do duplicates and simple negation contradictions
        if nli is None:
            for i, j, _shared in pairs:
                ti, tj = claims[i]["text"], claims[j]["text"]
                jac = _jaccard(set(_tok(ti)), set(_tok(tj)))
                if jac >= dup_jac:
                    violations.append(
                        dict(
                            rule="DuplicateClaims(A): near-identical supported claims",
                            supporting_triples=sorted([claims[i]["triple_id"], claims[j]["triple_id"]]),
                            confidence=round(float(max(claims[i]["conf"], claims[j]["conf"])), 4),
                            severity="SOFT",
                        )
                    )
                elif (_has_neg(ti) ^ _has_neg(tj)) and jac >= 0.35:
                    violations.append(
                        dict(
                            rule="ContradictoryClaims(A): lexical negation mismatch in otherwise similar claims",
                            supporting_triples=sorted([claims[i]["triple_id"], claims[j]["triple_id"]]),
                            confidence=0.65,
                            severity="SOFT",
                        )
                    )
            return violations

        # NLI: evaluate both directions to be robust
        premises: List[str] = []
        hypotheses: List[str] = []
        pair_meta: List[Tuple[int, int]] = []
        for i, j, _shared in pairs:
            premises.append(claims[i]["text"])
            hypotheses.append(claims[j]["text"])
            pair_meta.append((i, j))
            premises.append(claims[j]["text"])
            hypotheses.append(claims[i]["text"])
            pair_meta.append((j, i))

        probs = nli.predict_proba(premises, hypotheses)

        # Aggregate per unordered pair: take max contradiction prob across both directions
        agg: Dict[Tuple[int, int], Dict[str, float]] = {}
        for (a, b), pr in zip(pair_meta, probs):
            i, j = (a, b) if a < b else (b, a)
            cur = agg.setdefault((i, j), {"SUPPORT": 0.0, "CONTRADICT": 0.0, "NEUTRAL": 0.0})
            cur["SUPPORT"] = max(cur["SUPPORT"], float(pr["SUPPORT"]))
            cur["CONTRADICT"] = max(cur["CONTRADICT"], float(pr["CONTRADICT"]))
            cur["NEUTRAL"] = max(cur["NEUTRAL"], float(pr["NEUTRAL"]))

        for (i, j), pr in agg.items():
            ti, tj = claims[i]["text"], claims[j]["text"]
            jac = _jaccard(set(_tok(ti)), set(_tok(tj)))

            sup = float(pr["SUPPORT"])
            con = float(pr["CONTRADICT"])
            neu = float(pr["NEUTRAL"])

            # Duplicate: very high overlap or mutual entailment
            if jac >= dup_jac or (sup >= entail_thr and con < 0.2):
                violations.append(
                    dict(
                        rule="DuplicateClaims(A): near-identical supported claims",
                        supporting_triples=sorted([claims[i]["triple_id"], claims[j]["triple_id"]]),
                        confidence=round(float(max(claims[i]["conf"], claims[j]["conf"], sup)), 4),
                        severity="SOFT",
                    )
                )
                continue

            # Contradiction: high contradiction, and stronger than entail by margin
            if con >= contra_thr and con >= sup + margin:
                # Mix with underlying fact confidences so weak claims don't dominate
                base = max(claims[i]["conf"], claims[j]["conf"])
                conf = 0.6 * con + 0.4 * base
                violations.append(
                    dict(
                        rule="ContradictoryClaims(A): NLI contradiction between supported claims",
                        supporting_triples=sorted([claims[i]["triple_id"], claims[j]["triple_id"]]),
                        confidence=round(float(conf), 4),
                        severity="HARD",
                    )
                )
            # Soft contradiction: medium contradiction and high overlap / negation mismatch
            elif con >= 0.70 and (jac >= 0.30 or (_has_neg(ti) ^ _has_neg(tj))):
                conf = 0.5 * con + 0.5 * max(claims[i]["conf"], claims[j]["conf"])
                violations.append(
                    dict(
                        rule="ContradictoryClaims(A): possible contradiction among similar supported claims",
                        supporting_triples=sorted([claims[i]["triple_id"], claims[j]["triple_id"]]),
                        confidence=round(float(conf), 4),
                        severity="SOFT",
                    )
                )

        return violations

    constraint_rows: List[Dict[str, Any]] = []
    grounded_rows: List[Dict[str, Any]] = []

    for ex_id in all_ids:
        id_kg = kg_df[kg_df["id"] == ex_id] if len(kg_df) else kg_df.iloc[0:0]

        violations: List[Dict[str, Any]] = []
        violations.extend(_structured_constraints(id_kg))
        violations.extend(_claim_constraints(id_kg))

        if any(v.get("severity") == "HARD" for v in violations):
            status = "UNSAT"
        elif violations:
            status = "SOFT_VIOLATIONS"
        else:
            status = "SAT"

        if violations:
            supporting_triples = sorted({t for v in violations for t in v.get("supporting_triples", []) if t})
            conf = max(float(v.get("confidence", 0.0)) for v in violations)
        else:
            supporting_triples = (
                id_kg.sort_values("confidence", ascending=False).head(3)["triple_id"].astype(str).tolist()
                if len(id_kg) and "triple_id" in id_kg.columns
                else []
            )
            conf = float(id_kg["confidence"].max()) if len(id_kg) and "confidence" in id_kg.columns else 0.0

        constraint_rows.append(
            dict(
                id=int(ex_id),
                status=status,
                violations=violations,
                supporting_triples=supporting_triples,
                confidence=round(float(conf), 4),
                solver="claim_nli_rule_engine_v1",
                run_id=run_id,
            )
        )

        grounded_rows.append(
            dict(
                id=int(ex_id),
                constraints=[
                    {"predicate": "Claim", "type": "nli_contradiction", "severity": "HARD"},
                    {"predicate": "Claim", "type": "near_duplicate", "severity": "SOFT"},
                    {"predicate": "BornIn", "type": "unique_object", "severity": "HARD"},
                    {"predicate": "BornYear", "type": "unique_object", "severity": "HARD"},
                    {"predicate": "MarriedTo", "type": "unique_spouse", "severity": "SOFT"},
                ],
                run_id=run_id,
            )
        )

    out_gold.mkdir(parents=True, exist_ok=True)
    constraint_path = out_gold / "constraint_runs.jsonl"
    grounded_path = out_gold / "constraints_grounded.jsonl"

    with constraint_path.open("w", encoding="utf-8") as f:
        for r in constraint_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with grounded_path.open("w", encoding="utf-8") as f:
        for r in grounded_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return constraint_path, grounded_path
