from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd


def map_y(lbl: Any) -> float:
    s = str(lbl).strip().lower()
    if s.startswith("cons"):
        return 1.0
    if s.startswith("contr"):
        return 0.0
    return float("nan")


def step7_aggregate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    evidence_path: Path,
    constraint_path: Path,
    out_gold: Path,
    run_id: str,
    contradiction_penalty: float = 2.0,
) -> Tuple[Path, Path, float]:
    evidence_df = pd.read_csv(evidence_path)

    # --- load constraints (may be empty) ---
    constraints = []
    if constraint_path.exists():
        txt = constraint_path.read_text(encoding="utf-8")
        constraints = [json.loads(l) for l in txt.splitlines() if l.strip()]
    constraints_df = pd.DataFrame(constraints) if constraints else pd.DataFrame(columns=["id", "status", "confidence"])

    # --- id sets ---
    train_ids = pd.Index(train_df["id"].astype(int).unique()) if "id" in train_df.columns else pd.Index([])
    test_ids = pd.Index(test_df["id"].astype(int).unique()) if "id" in test_df.columns else pd.Index([])

    # ✅ IMPORTANT: limit universe to train ∪ test (don’t add stray evidence-only ids)
    all_ids = train_ids.union(test_ids)
    base = pd.DataFrame({"id": all_ids.astype(int)})

    # --- aggregate weights ---
    if len(evidence_df) == 0:
        agg = base.copy()
        agg["support_weight"] = 0.0
        agg["contradict_weight"] = 0.0
        agg["neutral_weight"] = 0.0
        cnt = base.copy()
        cnt["support_n"] = 0
        cnt["contradict_n"] = 0
        cnt["neutral_n"] = 0
    else:
        # sum of confidences per label
        agg = evidence_df.pivot_table(
            index="id", columns="label", values="confidence", aggfunc="sum", fill_value=0.0
        )
        for col in ["SUPPORT", "CONTRADICT", "NEUTRAL"]:
            if col not in agg.columns:
                agg[col] = 0.0
        agg = (
            agg.rename(
                columns={
                    "SUPPORT": "support_weight",
                    "CONTRADICT": "contradict_weight",
                    "NEUTRAL": "neutral_weight",
                }
            )
            .reset_index()
        )

        # counts per label
        cnt = evidence_df.groupby(["id", "label"]).size().unstack(fill_value=0).reset_index()
        for col in ["SUPPORT", "CONTRADICT", "NEUTRAL"]:
            if col not in cnt.columns:
                cnt[col] = 0
        cnt = cnt.rename(
            columns={
                "SUPPORT": "support_n",
                "CONTRADICT": "contradict_n",
                "NEUTRAL": "neutral_n",
            }
        )

        # ✅ outer merge onto base (so ids with no evidence still appear)
        agg = base.merge(agg, on="id", how="left").fillna(
            {"support_weight": 0.0, "contradict_weight": 0.0, "neutral_weight": 0.0}
        )
        cnt = base.merge(cnt, on="id", how="left").fillna(
            {"support_n": 0, "contradict_n": 0, "neutral_n": 0}
        )
        cnt[["support_n", "contradict_n", "neutral_n"]] = cnt[
            ["support_n", "contradict_n", "neutral_n"]
        ].astype(int)

    features = agg.merge(cnt, on="id", how="left")

    # --- attach logic constraints ---
    if len(constraints_df) > 0:
        c2 = constraints_df[["id", "status", "confidence"]].rename(columns={"confidence": "logic_conf"}).copy()
        c2["id"] = c2["id"].astype(int)
        features = features.merge(c2, on="id", how="left")
    else:
        features["status"] = np.nan
        features["logic_conf"] = np.nan

    features["status"] = features["status"].fillna("SAT")
    features["logic_conf"] = features["logic_conf"].fillna(0.0)
    features["hard_violations"] = features["status"].eq("UNSAT").astype(int)
    features["soft_violations"] = features["status"].eq("SOFT_VIOLATIONS").astype(int)

    # --- score ---
    # --- score ---
    # Allow NEUTRAL to contribute as weak evidence (helpful when verifier is conservative).
    # Tunable via env:
    #   KDSH_STEP7_NEUTRAL_ALPHA (default 0.15)
    #   KDSH_STEP7_MODE = balanced | high_recall (default balanced)
    neutral_alpha = float(os.getenv("KDSH_STEP7_NEUTRAL_ALPHA", "0.15"))
    mode = str(os.getenv("KDSH_STEP7_MODE", "balanced")).strip().lower()
    features["evidence_score"] = (
        features["support_weight"] + neutral_alpha * features["neutral_weight"]
        - float(contradiction_penalty) * features["contradict_weight"]
    )

    # --- attach train labels to calibrate threshold ---
    train_labels = train_df[["id", "label"]].copy()
    train_labels["id"] = train_labels["id"].astype(int)
    train_labels["y"] = train_labels["label"].apply(map_y)

    features = features.merge(train_labels[["id", "label", "y"]], on="id", how="left", suffixes=("", "_train"))
    features["split"] = np.where(features["y"].isna(), "test", "train")

    # --- threshold calibration on TRAIN only (using evidence_score) ---
    # balanced: maximize train accuracy
    # high_recall: maximize recall under an optional precision floor
    train_feat = features[features["split"] == "train"].copy()
    ys = train_feat["y"].dropna().values
    if len(ys) == 0:
        thr = 0.0
    else:
        scores = train_feat["evidence_score"].values
        cands = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 37)))
        if len(cands) == 0:
            thr = 0.0
        else:
            if mode == "high_recall":
                min_prec = float(os.getenv("KDSH_STEP7_MIN_PRECISION", "0.55"))
                best = None  # (recall, precision, thr)
                for t in cands:
                    pred = (scores >= t).astype(int)
                    tp = int(((pred == 1) & (ys == 1)).sum())
                    fp = int(((pred == 1) & (ys == 0)).sum())
                    fn = int(((pred == 0) & (ys == 1)).sum())
                    prec = tp / max(1, (tp + fp))
                    rec = tp / max(1, (tp + fn))
                    if prec < min_prec:
                        continue
                    cand = (rec, prec, float(t))
                    if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]) or (cand[0] == best[0] and cand[1] == best[1] and cand[2] > best[2]):
                        best = cand
                if best is None:
                    # fall back to max recall without precision constraint
                    best_rec, best_thr = -1.0, float(cands[0])
                    for t in cands:
                        pred = (scores >= t).astype(int)
                        tp = int(((pred == 1) & (ys == 1)).sum())
                        fn = int(((pred == 0) & (ys == 1)).sum())
                        rec = tp / max(1, (tp + fn))
                        if rec > best_rec or (rec == best_rec and float(t) > best_thr):
                            best_rec, best_thr = rec, float(t)
                    thr = float(best_thr)
                else:
                    thr = float(best[2])
            else:
                best_acc, best_thr = -1.0, float(cands[0])
                for t in cands:
                    pred = (scores >= t).astype(int)
                    acc = float((pred == ys).mean())
                    if acc > best_acc or (acc == best_acc and float(t) > best_thr):
                        best_acc, best_thr = acc, float(t)
                thr = float(best_thr)

    # --- final prediction rule ---
    # 1) hard UNSAT -> 0
    # 2) require at least some evidence (SUPPORT or NEUTRAL) -> else 0
    # 3) score vs calibrated threshold
    def decide_pred(r) -> int:
        if int(r.get("hard_violations", 0)) == 1:
            return 0
        # If verifier is conservative, NEUTRAL can still be useful weak evidence.
        if (int(r.get("support_n", 0)) + int(r.get("neutral_n", 0))) == 0:
            return 0

        es = float(r.get("evidence_score", 0.0))
        # High-recall mode is allowed to use negative thresholds; don't clamp.
        return int(es >= float(thr))

    features["prediction"] = features.apply(decide_pred, axis=1).astype(int)

    def short_rationale(r) -> str:
        if r["status"] == "UNSAT":
            return "Hard contradiction detected by Step-6 (UNSAT)."
        if (int(r.get("support_n", 0)) + int(r.get("neutral_n", 0))) == 0 and int(r.get("contradict_n", 0)) == 0:
            return "No evidence retrieved for this id."
        if (int(r.get("support_n", 0)) + int(r.get("neutral_n", 0))) == 0 and int(r.get("contradict_n", 0)) > 0:
            return "Only contradicting evidence found; predicting 0."
        if int(r.get("contradict_n", 0)) > 0 and int(r.get("support_n", 0)) == 0:
            return "Neutral/contradict mix; score penalizes contradictions."
        if int(r.get("contradict_n", 0)) > 0:
            return "Mixed evidence (support/neutral with contradictions); prediction uses score/threshold."
        if int(r.get("support_n", 0)) == 0 and int(r.get("neutral_n", 0)) > 0:
            return f"Only neutral evidence; treated as weak support (alpha={neutral_alpha:.2f})."
        return "Supporting evidence present; no hard contradictions detected."

    features["rationale"] = features.apply(short_rationale, axis=1)

    out_gold.mkdir(parents=True, exist_ok=True)
    decision_path = out_gold / "decision_scores.csv"
    results_path = out_gold / "results.csv"
    results_train_path = out_gold / "results_train.csv"  # optional, handy for debugging

    out_cols = [
        "id",
        "split",
        "prediction",
        "evidence_score",
        "support_weight",
        "contradict_weight",
        "neutral_weight",
        "support_n",
        "contradict_n",
        "neutral_n",
        "status",
        "hard_violations",
        "soft_violations",
        "logic_conf",
        "rationale",
    ]
    features[out_cols].sort_values(["split", "id"]).to_csv(decision_path, index=False)

    # ✅ ONLY TEST IDS in results.csv
    test_out = features[features["id"].isin(test_ids)][["id", "prediction"]].sort_values("id")
    test_out.to_csv(results_path, index=False)
    root_results_path = Path.cwd() / "results.csv"
    test_out.to_csv(root_results_path, index=False)

    # (Optional) train predictions file too
    train_out = features[features["id"].isin(train_ids)][["id", "prediction"]].sort_values("id")
    train_out.to_csv(results_train_path, index=False)

    return decision_path, results_path, float(thr)
