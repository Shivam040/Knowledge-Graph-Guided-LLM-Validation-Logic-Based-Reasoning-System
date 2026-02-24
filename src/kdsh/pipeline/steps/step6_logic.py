from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

def step6_logic(
    kg_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_gold: Path,
    run_id: str,
):
    kg_df = pd.read_csv(kg_path) if kg_path.exists() else pd.DataFrame()
    all_ids = pd.concat([train_df[["id"]], test_df[["id"]]]).drop_duplicates()["id"].astype(int).tolist()

    def build_violations_for_id(id_kg: pd.DataFrame) -> List[Dict[str, Any]]:
        violations = []
        by_p = defaultdict(list)
        for _, r in id_kg.iterrows():
            by_p[str(r["p"])].append(r)

        for p in ["BornIn", "BornYear"]:
            objs = defaultdict(list)
            for r in by_p.get(p, []):
                objs[str(r["o"])].append(r)
            if len(objs) >= 2:
                obj_best = max(objs.keys(), key=lambda o: max(float(x["confidence"]) for x in objs[o]))
                conflicting = [o for o in objs.keys() if o != obj_best]
                triples_best = [x["triple_id"] for x in objs[obj_best]]
                triples_conf = [x["triple_id"] for o in conflicting for x in objs[o]]
                conf = max(
                    max(float(x["confidence"]) for x in objs[obj_best]),
                    max(float(x["confidence"]) for o in conflicting for x in objs[o]),
                )
                violations.append(
                    dict(
                        rule=f"Unique{p}(A): cannot have multiple distinct {p} objects",
                        supporting_triples=triples_best + triples_conf,
                        confidence=round(float(conf), 4),
                        severity="HARD",
                    )
                )

        spouses = defaultdict(list)
        for r in by_p.get("MarriedTo", []):
            spouses[str(r["o"])].append(r)
        if len(spouses) >= 2:
            all_triples = [x["triple_id"] for os_ in spouses.values() for x in os_]
            conf = max(float(x["confidence"]) for os_ in spouses.values() for x in os_)
            violations.append(
                dict(
                    rule="MarriedTo(A,B) and MarriedTo(A,C) with B!=C (unless time-bucketed non-overlapping)",
                    supporting_triples=all_triples,
                    confidence=round(float(conf), 4),
                    severity="SOFT",
                )
            )
        return violations

    constraint_rows = []
    grounded_rows = []
    for ex_id in all_ids:
        id_kg = kg_df[kg_df["id"] == ex_id] if len(kg_df) else kg_df.iloc[0:0]
        violations = build_violations_for_id(id_kg)
        if any(v["severity"] == "HARD" for v in violations):
            status = "UNSAT"
        elif violations:
            status = "SOFT_VIOLATIONS"
        else:
            status = "SAT"

        supporting_triples = sorted({t for v in violations for t in v["supporting_triples"]}) if violations else (
            id_kg.sort_values("confidence", ascending=False).head(3)["triple_id"].tolist() if len(id_kg) else []
        )

        conf = float(id_kg["confidence"].max()) if len(id_kg) else 0.0
        constraint_rows.append(
            dict(
                id=int(ex_id),
                status=status,
                violations=violations,
                supporting_triples=supporting_triples,
                confidence=round(conf, 4),
                solver="rule_engine_v1",
                run_id=run_id,
            )
        )

        grounded_rows.append(
            dict(
                id=int(ex_id),
                constraints=[
                    {"predicate": "BornIn", "type": "unique_object", "severity": "HARD"},
                    {"predicate": "BornYear", "type": "unique_object", "severity": "HARD"},
                    {"predicate": "Died", "type": "implies_not_alive_after", "severity": "HARD"},
                    {"predicate": "MarriedTo", "type": "unique_spouse", "severity": "SOFT"},
                ],
                run_id=run_id,
            )
        )

    constraint_path = out_gold / "constraint_runs.jsonl"
    grounded_path = out_gold / "constraints_grounded.jsonl"
    with constraint_path.open("w", encoding="utf-8") as f:
        for r in constraint_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with grounded_path.open("w", encoding="utf-8") as f:
        for r in grounded_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return constraint_path, grounded_path
