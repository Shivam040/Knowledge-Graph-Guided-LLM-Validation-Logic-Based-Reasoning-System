# src/scripts/eval_train.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

LABELS = ["SUPPORT", "CONTRADICT", "NEUTRAL"]
AUTO_SECTION_START = "<!-- AUTO:TRAIN_EVAL_START -->"
AUTO_SECTION_END = "<!-- AUTO:TRAIN_EVAL_END -->"


# ----------------------------
# helpers
# ----------------------------

def _norm_label(x: Any) -> str:
    s = str(x or "").strip().upper()
    if s in ("ENTAILMENT", "ENTAIL", "E"):
        return "SUPPORT"
    if s in ("CONTRADICTION", "CONTRADICT", "C"):
        return "CONTRADICT"
    if s in ("NEUTRAL", "N", "UNKNOWN", "UNVERIFIED"):
        return "NEUTRAL"
    if s in LABELS:
        return s
    return s


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _validate_labels(series: pd.Series, what: str) -> None:
    bad = sorted(set(series.unique()) - set(LABELS))
    if bad:
        raise ValueError(f"{what} has unexpected labels: {bad}. Expected only {LABELS}.")


def _parse_int_label_map(s: str) -> Dict[int, str]:
    """
    "0=CONTRADICT,1=NEUTRAL,2=SUPPORT" -> {0:"CONTRADICT",1:"NEUTRAL",2:"SUPPORT"}
    """
    out: Dict[int, str] = {}
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            raise ValueError(f"Bad label map part: {p}")
        k, v = p.split("=", 1)
        k = int(k.strip())
        v = _norm_label(v.strip())
        out[k] = v
    return out


def _infer_run_id(gold_dir: Path, silver_dir: Path, explicit: str) -> str:
    if explicit.strip():
        return explicit.strip()
    for p in (gold_dir, silver_dir):
        name = p.name
        if "run_id=" in name:
            return name.split("run_id=", 1)[1]
        if name.startswith("run_"):
            return name
    return ""


@dataclass
class PredSourceInfo:
    source: str
    path: Path
    aggregation: str
    note: str = ""


# ----------------------------
# loading gold
# ----------------------------

def load_gold_labels(gold_labels_path: Path) -> pd.DataFrame:
    """
    Gold labels file (dataset train labels).
    Must contain claim_id (or id) and label column.
    Accepts CSV or JSONL.
    """
    if not gold_labels_path.exists():
        raise FileNotFoundError(f"gold_labels not found: {gold_labels_path}")

    if gold_labels_path.suffix.lower() == ".jsonl":
        df = pd.DataFrame(_read_jsonl(gold_labels_path))
    else:
        df = pd.read_csv(gold_labels_path)

    id_col = "claim_id" if "claim_id" in df.columns else ("id" if "id" in df.columns else None)
    if not id_col:
        raise ValueError(f"Gold labels must contain claim_id or id. Got columns: {list(df.columns)}")

    label_col = None
    for c in ("label", "gold_label", "y", "target", "truth"):
        if c in df.columns:
            label_col = c
            break
    if not label_col:
        raise ValueError(
            "Gold labels must contain one of: label/gold_label/y/target/truth. "
            f"Got columns: {list(df.columns)}"
        )

    out = df[[id_col, label_col]].rename(columns={id_col: "claim_id", label_col: "gold_label"}).copy()
    out["gold_label"] = out["gold_label"].apply(_norm_label)
    _validate_labels(out["gold_label"], "gold_label")
    return out


# ----------------------------
# loading predictions
# ----------------------------

def load_pred_labels(
    gold_dir: Path,
    silver_dir: Path,
    aggregation: str,
    int_label_map: Dict[int, str],
) -> Tuple[pd.DataFrame, PredSourceInfo]:
    """
    Preference:
      1) gold/results.csv (claim-level, string labels)
      2) gold/decision_scores.csv (numeric prediction -> labels)
      3) silver/evidence_labels.csv (aggregate)
    Returns claim_id + pred_label
    """
    # (1) results.csv
    results_path = gold_dir / "results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        if ("claim_id" in df.columns or "id" in df.columns):
            id_col = "claim_id" if "claim_id" in df.columns else "id"
            lab_col = None
            for c in ("label", "pred_label", "prediction_label", "final_label", "y_pred", "decision"):
                if c in df.columns:
                    lab_col = c
                    break
            if lab_col:
                out = df[[id_col, lab_col]].rename(columns={id_col: "claim_id", lab_col: "pred_label"}).copy()
                out["pred_label"] = out["pred_label"].apply(_norm_label)
                _validate_labels(out["pred_label"], "pred_label")
                return out, PredSourceInfo("gold_results", results_path, "none")

    # (2) decision_scores.csv
    dec_path = gold_dir / "decision_scores.csv"
    if dec_path.exists():
        df = pd.read_csv(dec_path)
        if ("claim_id" in df.columns or "id" in df.columns) and "prediction" in df.columns:
            id_col = "claim_id" if "claim_id" in df.columns else "id"
            out = df[[id_col, "prediction"]].rename(columns={id_col: "claim_id"}).copy()

            def map_pred(v: Any) -> str:
                try:
                    iv = int(v)
                except Exception:
                    return _norm_label(v)
                if iv not in int_label_map:
                    raise ValueError(
                        f"decision_scores.csv has prediction={iv} but it's not in --int_label_map. "
                        f"Map provided: {int_label_map}"
                    )
                return int_label_map[iv]

            out["pred_label"] = out["prediction"].apply(map_pred)
            out = out[["claim_id", "pred_label"]]
            _validate_labels(out["pred_label"], "pred_label")
            return out, PredSourceInfo(
                "gold_decision_scores",
                dec_path,
                "none",
                note=f"mapped numeric prediction via {int_label_map}",
            )

    # (3) evidence_labels.csv
    ev_path = silver_dir / "evidence_labels.csv"
    if not ev_path.exists():
        raise FileNotFoundError(
            "Could not find predictions. Tried:\n"
            f"- {results_path}\n"
            f"- {dec_path}\n"
            f"- {ev_path}\n"
        )
    ev = pd.read_csv(ev_path)
    if "claim_id" not in ev.columns or "label" not in ev.columns:
        raise ValueError(f"evidence_labels.csv must have claim_id,label. Got columns: {list(ev.columns)}")
    if "confidence" not in ev.columns:
        raise ValueError("evidence_labels.csv must have confidence for robust aggregation.")
    ev = ev.copy()
    ev["label"] = ev["label"].apply(_norm_label)
    _validate_labels(ev["label"], "evidence label")

    if aggregation == "max_conf":
        idx = ev.groupby("claim_id")["confidence"].idxmax()
        agg = ev.loc[idx, ["claim_id", "label"]].rename(columns={"label": "pred_label"}).reset_index(drop=True)
        return agg, PredSourceInfo("silver_evidence", ev_path, "max_conf")
    elif aggregation == "vote":
        def vote_one(g: pd.DataFrame) -> str:
            counts = g["label"].value_counts()
            top = counts.max()
            top_labels = counts[counts == top].index.tolist()
            if len(top_labels) == 1:
                return top_labels[0]
            # tie-break by best confidence among tied labels
            best_lab = top_labels[0]
            best_conf = -1.0
            for lab in top_labels:
                c = float(g.loc[g["label"] == lab, "confidence"].max())
                if c > best_conf:
                    best_conf = c
                    best_lab = lab
            return best_lab

        agg = ev.groupby("claim_id").apply(vote_one).reset_index(name="pred_label")
        return agg, PredSourceInfo("silver_evidence", ev_path, "vote")
    else:
        raise ValueError("aggregation must be max_conf or vote")


# ----------------------------
# metrics
# ----------------------------

def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> List[List[int]]:
    li = {lab: i for i, lab in enumerate(labels)}
    m = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        m[li[t]][li[p]] += 1
    return m


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def per_class_metrics(cm: List[List[int]], labels: List[str]) -> Dict[str, Dict[str, float]]:
    n = len(labels)
    out: Dict[str, Dict[str, float]] = {}
    for i, lab in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n) if r != i)
        fn = sum(cm[i][c] for c in range(n) if c != i)
        support = sum(cm[i][c] for c in range(n))

        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0

        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": float(support)}
    return out


def macro_avg(per_class: Dict[str, Dict[str, float]], labels: List[str]) -> Dict[str, float]:
    def avg(k: str) -> float:
        return float(sum(per_class[l][k] for l in labels) / len(labels))
    return {"precision": avg("precision"), "recall": avg("recall"), "f1": avg("f1")}


def render_cm_md(cm: List[List[int]], labels: List[str]) -> str:
    header = "| true\\pred | " + " | ".join(labels) + " |\n"
    sep = "|---|" + "|".join(["---"] * len(labels)) + "|\n"
    rows = []
    for i, lab in enumerate(labels):
        rows.append("| " + lab + " | " + " | ".join(str(x) for x in cm[i]) + " |")
    return header + sep + "\n".join(rows) + "\n"


# ----------------------------
# reporting outputs
# ----------------------------

def build_report_md(
    run_id: str,
    metrics: Dict[str, Any],
    cm: List[List[int]],
    per_class: Dict[str, Dict[str, float]],
    pred_info: PredSourceInfo,
    verifier_used: str,
    logic_backend: str,
    thresholds: Dict[str, Any],
    gold_labels_path: Path,
) -> str:
    lines: List[str] = []
    lines.append(f"# Train Evaluation{(' — ' + run_id) if run_id else ''}\n\n")
    lines.append(f"- Generated: **{metrics['timestamp']}**\n")
    lines.append(f"- Gold labels: `{gold_labels_path}`\n")
    lines.append(f"- Pred source: `{pred_info.path.name}` ({pred_info.source})\n")
    lines.append(f"- n_eval: **{metrics['n_eval']}**\n\n")

    lines.append("## Configuration (MVP)\n")
    lines.append(f"- verifier used: **{verifier_used}**\n")
    lines.append(f"- logic backend: **{logic_backend}**\n")
    for k, v in thresholds.items():
        lines.append(f"- {k}: **{v}**\n")
    lines.append(f"- pred aggregation: **{pred_info.aggregation}**\n")
    if pred_info.note:
        lines.append(f"- note: {pred_info.note}\n")
    lines.append("\n")

    lines.append("## Metrics (train)\n")
    lines.append(f"- accuracy: **{metrics['accuracy']:.4f}**\n")
    lines.append(f"- macro precision: **{metrics['macro_precision']:.4f}**\n")
    lines.append(f"- macro recall: **{metrics['macro_recall']:.4f}**\n")
    lines.append(f"- macro f1: **{metrics['macro_f1']:.4f}**\n\n")

    lines.append("## Confusion matrix (train)\n")
    lines.append(render_cm_md(cm, LABELS))
    lines.append("\n")

    lines.append("## Per-class (train)\n")
    lines.append("| class | precision | recall | f1 | support |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for lab in LABELS:
        pc = per_class[lab]
        lines.append(
            f"| {lab} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {int(pc['support'])} |\n"
        )
    return "".join(lines)


def write_metrics_json(gold_dir: Path, metrics: Dict[str, Any]) -> Path:
    out = gold_dir / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return out


def write_eval_report(gold_dir: Path, report_md: str) -> Path:
    out = gold_dir / "eval_train_report.md"
    out.write_text(report_md, encoding="utf-8")
    return out


def append_into_report_run(gold_dir: Path, report_md: str) -> Optional[Path]:
    candidates = sorted(gold_dir.glob("report_run_*.md"))
    if not candidates:
        return None
    path = candidates[0]
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    section = (
        f"\n\n{AUTO_SECTION_START}\n"
        "## Train evaluation (auto)\n\n"
        + report_md
        + f"\n{AUTO_SECTION_END}\n"
    )
    if AUTO_SECTION_START in old and AUTO_SECTION_END in old:
        pre = old.split(AUTO_SECTION_START, 1)[0]
        post = old.split(AUTO_SECTION_END, 1)[1]
        new = pre + section + post
    else:
        new = old.rstrip() + section
    path.write_text(new, encoding="utf-8")
    return path


# ----------------------------
# main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Train evaluation + metrics.json + short markdown report (MVP).")
    ap.add_argument("--gold_dir", type=str, required=True, help="Run gold folder (your screenshot folder with report_run_*.md).")
    ap.add_argument("--silver_dir", type=str, required=True, help="Run silver folder (contains evidence_labels.csv etc.)")

    # IMPORTANT: provide dataset gold labels (train truth)
    ap.add_argument("--gold_labels", type=str, required=True, help="Path to dataset gold labels (CSV or JSONL). Must contain claim_id/id and label.")

    ap.add_argument("--aggregation", type=str, default="max_conf", choices=["max_conf", "vote"],
                    help="If using evidence_labels.csv, aggregate to claim-level.")
    ap.add_argument("--run_id", type=str, default="", help="Optional run id override.")

    # mapping for numeric predictions in decision_scores.csv
    ap.add_argument("--int_label_map", type=str, default="0=CONTRADICT,1=NEUTRAL,2=SUPPORT",
                    help='Mapping for numeric predictions, e.g. "0=CONTRADICT,1=NEUTRAL,2=SUPPORT".')

    # report knobs
    ap.add_argument("--verifier_used", type=str, default=os.getenv("KDSH_VERIFIER_BACKEND", "heuristic"))
    ap.add_argument("--logic_backend", type=str, default=os.getenv("KDSH_LOGIC_BACKEND", "rules"))
    ap.add_argument("--min_fact_conf", type=float, default=float(os.getenv("KDSH_MIN_FACT_CONF", "0.45")))
    ap.add_argument("--score_mix_alpha", type=float, default=float(os.getenv("KDSH_NLI_ALPHA", "0.15")))
    ap.add_argument("--retrieval_k", type=int, default=int(os.getenv("KDSH_RETRIEVAL_K", "12")))

    args = ap.parse_args()

    gold_dir = Path(args.gold_dir)
    silver_dir = Path(args.silver_dir)
    gold_labels_path = Path(args.gold_labels)

    if not gold_dir.exists():
        raise FileNotFoundError(f"gold_dir not found: {gold_dir}")
    if not silver_dir.exists():
        raise FileNotFoundError(f"silver_dir not found: {silver_dir}")

    run_id = _infer_run_id(gold_dir, silver_dir, args.run_id)
    int_label_map = _parse_int_label_map(args.int_label_map)

    gold = load_gold_labels(gold_labels_path)
    pred, pred_info = load_pred_labels(gold_dir, silver_dir, args.aggregation, int_label_map)

    merged = gold.merge(pred, on="claim_id", how="inner")
    if merged.empty:
        raise RuntimeError(
            "No overlap between gold and pred claim_id.\n"
            f"- gold: {len(gold)} rows from {gold_labels_path}\n"
            f"- pred: {len(pred)} rows from {pred_info.path}\n"
            "Check whether gold uses claim_id vs id, and whether you evaluated the same split."
        )

    y_true = merged["gold_label"].tolist()
    y_pred = merged["pred_label"].tolist()

    cm = confusion_matrix(y_true, y_pred, LABELS)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = _safe_div(correct, len(y_true))

    per_class = per_class_metrics(cm, LABELS)
    macro = macro_avg(per_class, LABELS)

    metrics: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_eval": int(len(y_true)),
        "accuracy": float(acc),
        "macro_precision": float(macro["precision"]),
        "macro_recall": float(macro["recall"]),
        "macro_f1": float(macro["f1"]),
        "per_class": per_class,
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
        "inputs": {
            "gold_labels": str(gold_labels_path.resolve()),
            "pred_source": pred_info.source,
            "pred_path": str(pred_info.path.resolve()),
            "aggregation": pred_info.aggregation,
        },
        "run_config": {
            "verifier_used": args.verifier_used,
            "logic_backend": args.logic_backend,
            "min_fact_conf": float(args.min_fact_conf),
            "score_mix_alpha": float(args.score_mix_alpha),
            "retrieval_k": int(args.retrieval_k),
            "int_label_map": args.int_label_map,
        },
    }

    metrics_path = write_metrics_json(gold_dir, metrics)

    thresholds = {
        "min_fact_conf": float(args.min_fact_conf),
        "score_mix_alpha": float(args.score_mix_alpha),
        "retrieval_k": int(args.retrieval_k),
    }

    report_md = build_report_md(
        run_id=run_id,
        metrics=metrics,
        cm=cm,
        per_class=per_class,
        pred_info=pred_info,
        verifier_used=args.verifier_used,
        logic_backend=args.logic_backend,
        thresholds=thresholds,
        gold_labels_path=gold_labels_path,
    )

    report_path = write_eval_report(gold_dir, report_md)
    updated_run_report = append_into_report_run(gold_dir, report_md)

    print(f"[eval_train] wrote metrics: {metrics_path}")
    print(f"[eval_train] wrote report:  {report_path}")
    if updated_run_report:
        print(f"[eval_train] updated:       {updated_run_report}")
    print(f"[eval_train] n_eval={metrics['n_eval']} acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")



if __name__ == "__main__":
    main()


# # scripts/eval_train.py
# from __future__ import annotations

# import argparse
# import json
# import os
# from dataclasses import dataclass
# from datetime import datetime
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple

# import pandas as pd


# LABELS = ["SUPPORT", "CONTRADICT", "NEUTRAL"]
# AUTO_SECTION_START = "<!-- AUTO:TRAIN_EVAL_START -->"
# AUTO_SECTION_END = "<!-- AUTO:TRAIN_EVAL_END -->"


# def _norm_label(x: Any) -> str:
#     s = str(x or "").strip().upper()
#     # common NLI variants
#     if s in ("ENTAILMENT", "ENTAIL", "E"):
#         return "SUPPORT"
#     if s in ("CONTRADICTION", "CONTRADICT", "C"):
#         return "CONTRADICT"
#     if s in ("NEUTRAL", "N", "UNKNOWN", "UNVERIFIED"):
#         return "NEUTRAL"
#     # already in our schema
#     if s in LABELS:
#         return s
#     return s  # keep; we'll validate later


# def _validate_labels(series: pd.Series, what: str) -> None:
#     bad = sorted(set(series.unique()) - set(LABELS))
#     if bad:
#         raise ValueError(f"{what} has unexpected labels: {bad}. Expected only {LABELS}.")


# def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
#     rows: List[Dict[str, Any]] = []
#     for line in path.read_text(encoding="utf-8").splitlines():
#         line = line.strip()
#         if not line:
#             continue
#         rows.append(json.loads(line))
#     return rows


# def _find_first_existing(paths: List[Path]) -> Optional[Path]:
#     for p in paths:
#         if p.exists():
#             return p
#     return None


# @dataclass
# class PredSourceInfo:
#     source: str  # "gold_results" or "silver_evidence"
#     path: Path
#     aggregation: str
#     verifier_model_hint: str = ""


# def load_gold_labels_from_claims(silver_dir: Path) -> pd.DataFrame:
#     """
#     Train gold truth is usually present in silver/claims.jsonl.
#     Expected each row has claim_id and one of: label/gold_label/y/target.
#     """
#     claims_path = silver_dir / "claims.jsonl"
#     if not claims_path.exists():
#         raise FileNotFoundError(f"Missing claims.jsonl in silver_dir: {claims_path}")

#     rows = _read_jsonl(claims_path)
#     df = pd.DataFrame(rows)

#     if "claim_id" not in df.columns:
#         raise ValueError(f"claims.jsonl must include claim_id. Got columns: {list(df.columns)}")

#     label_col = None
#     for c in ("label", "gold_label", "y", "target"):
#         if c in df.columns:
#             label_col = c
#             break

#     if label_col is None:
#         raise ValueError(
#             "claims.jsonl does not contain a gold label column. "
#             "For train evaluation, it must contain one of: label/gold_label/y/target."
#         )

#     out = df[["claim_id", label_col]].rename(columns={label_col: "gold_label"}).copy()
#     out["gold_label"] = out["gold_label"].apply(_norm_label)
#     _validate_labels(out["gold_label"], "gold_label")
#     return out


# def load_pred_labels(
#     gold_dir: Path,
#     silver_dir: Path,
#     aggregation: str = "max_conf",
# ) -> Tuple[pd.DataFrame, PredSourceInfo]:
#     """
#     Preference order:
#       1) gold/results.csv if it has claim_id + label-like column
#       2) silver/evidence_labels.csv aggregated to claim level
#     """
#     # ---- (1) gold/results.csv ----
#     gold_results = gold_dir / "results.csv"
#     if gold_results.exists():
#         df = pd.read_csv(gold_results)
#         if "claim_id" in df.columns:
#             pred_col = None
#             for c in ("label", "pred_label", "prediction", "decision", "final_label", "y_pred"):
#                 if c in df.columns:
#                     pred_col = c
#                     break
#             if pred_col is not None:
#                 out = df[["claim_id", pred_col]].rename(columns={pred_col: "pred_label"}).copy()
#                 out["pred_label"] = out["pred_label"].apply(_norm_label)
#                 _validate_labels(out["pred_label"], "pred_label")
#                 return out, PredSourceInfo(
#                     source="gold_results",
#                     path=gold_results,
#                     aggregation="none",
#                     verifier_model_hint="",
#                 )

#     # ---- (2) silver/evidence_labels.csv ----
#     ev_path = silver_dir / "evidence_labels.csv"
#     if not ev_path.exists():
#         raise FileNotFoundError(
#             "Could not find predictions. Tried:\n"
#             f"- {gold_results}\n"
#             f"- {ev_path}\n"
#         )

#     ev = pd.read_csv(ev_path)
#     if "claim_id" not in ev.columns or "label" not in ev.columns:
#         raise ValueError(f"evidence_labels.csv must contain claim_id and label. Got: {list(ev.columns)}")
#     if "confidence" not in ev.columns:
#         # still can aggregate by first row per claim, but that's weak; enforce confidence for now
#         raise ValueError("evidence_labels.csv must contain confidence for robust aggregation.")

#     ev = ev.copy()
#     ev["label"] = ev["label"].apply(_norm_label)
#     _validate_labels(ev["label"], "evidence label")

#     verifier_hint = ""
#     if "verifier_model" in ev.columns:
#         # most common / first non-empty
#         s = ev["verifier_model"].dropna().astype(str)
#         s = s[s.str.strip() != ""]
#         if len(s) > 0:
#             verifier_hint = str(s.value_counts().idxmax())

#     if aggregation == "max_conf":
#         idx = ev.groupby("claim_id")["confidence"].idxmax()
#         agg = ev.loc[idx, ["claim_id", "label"]].rename(columns={"label": "pred_label"}).reset_index(drop=True)
#     elif aggregation == "vote":
#         def vote_one(g: pd.DataFrame) -> str:
#             counts = g["label"].value_counts()
#             top = counts.max()
#             top_labels = counts[counts == top].index.tolist()
#             if len(top_labels) == 1:
#                 return top_labels[0]
#             # tie-break by max confidence among tied labels
#             best_lab = top_labels[0]
#             best_conf = -1.0
#             for lab in top_labels:
#                 c = float(g.loc[g["label"] == lab, "confidence"].max())
#                 if c > best_conf:
#                     best_conf = c
#                     best_lab = lab
#             return best_lab

#         agg = ev.groupby("claim_id").apply(vote_one).reset_index(name="pred_label")
#     else:
#         raise ValueError("aggregation must be one of: max_conf, vote")

#     _validate_labels(agg["pred_label"], "pred_label")
#     return agg, PredSourceInfo(
#         source="silver_evidence",
#         path=ev_path,
#         aggregation=aggregation,
#         verifier_model_hint=verifier_hint,
#     )


# def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> List[List[int]]:
#     li = {lab: i for i, lab in enumerate(labels)}
#     m = [[0 for _ in labels] for _ in labels]
#     for t, p in zip(y_true, y_pred):
#         m[li[t]][li[p]] += 1
#     return m


# def _safe_div(a: float, b: float) -> float:
#     return float(a / b) if b else 0.0


# def per_class_metrics(cm: List[List[int]], labels: List[str]) -> Dict[str, Dict[str, float]]:
#     # rows=true, cols=pred
#     n = len(labels)
#     out: Dict[str, Dict[str, float]] = {}
#     for i, lab in enumerate(labels):
#         tp = cm[i][i]
#         fp = sum(cm[r][i] for r in range(n) if r != i)
#         fn = sum(cm[i][c] for c in range(n) if c != i)
#         support = sum(cm[i][c] for c in range(n))

#         prec = _safe_div(tp, tp + fp)
#         rec = _safe_div(tp, tp + fn)
#         f1 = _safe_div(2 * prec * rec, prec + rec) if (prec + rec) else 0.0

#         out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": float(support)}
#     return out


# def macro_avg(per_class: Dict[str, Dict[str, float]], labels: List[str]) -> Dict[str, float]:
#     def avg(k: str) -> float:
#         return float(sum(per_class[l][k] for l in labels) / len(labels))
#     return {"precision": avg("precision"), "recall": avg("recall"), "f1": avg("f1")}


# def render_cm_md(cm: List[List[int]], labels: List[str]) -> str:
#     header = "| true\\pred | " + " | ".join(labels) + " |\n"
#     sep = "|---|" + "|".join(["---"] * len(labels)) + "|\n"
#     rows = []
#     for i, lab in enumerate(labels):
#         rows.append("| " + lab + " | " + " | ".join(str(x) for x in cm[i]) + " |")
#     return header + sep + "\n".join(rows) + "\n"


# def _infer_run_id(gold_dir: Path, silver_dir: Path, explicit: str) -> str:
#     if explicit.strip():
#         return explicit.strip()
#     # common layout: .../run_id=run_YYYYMMDD_HHMMSS
#     for p in (gold_dir, silver_dir):
#         name = p.name
#         if "run_id=" in name:
#             return name.split("run_id=", 1)[1]
#         if name.startswith("run_"):
#             return name
#     return ""


# def write_metrics_json(gold_dir: Path, metrics: Dict[str, Any]) -> Path:
#     out = gold_dir / "metrics.json"
#     out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
#     return out


# def build_eval_report_md(
#     run_id: str,
#     metrics: Dict[str, Any],
#     cm: List[List[int]],
#     per_class: Dict[str, Dict[str, float]],
#     pred_info: PredSourceInfo,
#     verifier_used: str,
#     logic_backend: str,
#     thresholds: Dict[str, Any],
# ) -> str:
#     lines: List[str] = []
#     lines.append(f"# Train Evaluation{(' — ' + run_id) if run_id else ''}\n\n")
#     lines.append(f"- Generated: **{metrics['timestamp']}**\n")
#     lines.append(f"- n_eval: **{metrics['n_eval']}**\n\n")

#     lines.append("## Run configuration (MVP)\n")
#     lines.append(f"- verifier used: **{verifier_used}**\n")
#     if pred_info.verifier_model_hint:
#         lines.append(f"- verifier model (hint from evidence): **{pred_info.verifier_model_hint}**\n")
#     lines.append(f"- logic backend: **{logic_backend}**\n")
#     for k, v in thresholds.items():
#         lines.append(f"- {k}: **{v}**\n")
#     lines.append("\n")

#     lines.append("## Prediction source\n")
#     lines.append(f"- source: **{pred_info.source}**\n")
#     lines.append(f"- file: `{pred_info.path.name}`\n")
#     lines.append(f"- aggregation: **{pred_info.aggregation}**\n\n")

#     lines.append("## Metrics (train)\n")
#     lines.append(f"- accuracy: **{metrics['accuracy']:.4f}**\n")
#     lines.append(f"- macro precision: **{metrics['macro_precision']:.4f}**\n")
#     lines.append(f"- macro recall: **{metrics['macro_recall']:.4f}**\n")
#     lines.append(f"- macro f1: **{metrics['macro_f1']:.4f}**\n\n")

#     lines.append("## Confusion matrix (train)\n")
#     lines.append(render_cm_md(cm, LABELS))
#     lines.append("\n")

#     lines.append("## Per-class (train)\n")
#     lines.append("| class | precision | recall | f1 | support |\n")
#     lines.append("|---|---:|---:|---:|---:|\n")
#     for lab in LABELS:
#         pc = per_class[lab]
#         lines.append(
#             f"| {lab} | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {int(pc['support'])} |\n"
#         )

#     return "".join(lines)


# def write_eval_report(gold_dir: Path, run_id: str, report_md: str) -> Path:
#     # Keep it simple + stable name
#     out = gold_dir / "eval_train_report.md"
#     out.write_text(report_md, encoding="utf-8")
#     return out


# def maybe_append_to_existing_run_report(gold_dir: Path, report_md: str) -> Optional[Path]:
#     """
#     If gold_dir contains report_run_*.md, append/replace an AUTO section.
#     """
#     candidates = sorted(gold_dir.glob("report_run_*.md"))
#     if not candidates:
#         return None
#     path = candidates[0]

#     old = path.read_text(encoding="utf-8") if path.exists() else ""
#     section = (
#         f"\n\n{AUTO_SECTION_START}\n"
#         "## Train evaluation (auto)\n\n"
#         + report_md
#         + f"\n{AUTO_SECTION_END}\n"
#     )

#     if AUTO_SECTION_START in old and AUTO_SECTION_END in old:
#         pre = old.split(AUTO_SECTION_START, 1)[0]
#         post = old.split(AUTO_SECTION_END, 1)[1]
#         new = pre + section + post
#     else:
#         new = old.rstrip() + section

#     path.write_text(new, encoding="utf-8")
#     return path


# def main() -> None:
#     ap = argparse.ArgumentParser(
#         description="MVP-grade train evaluation + reporting. Writes metrics.json into gold folder."
#     )
#     ap.add_argument("--gold_dir", type=str, required=True, help="Path to run gold folder (contains results.csv, dossier/, report_run_*.md etc.)")
#     ap.add_argument("--silver_dir", type=str, required=True, help="Path to run silver folder (contains claims.jsonl, evidence_labels.csv etc.)")

#     ap.add_argument("--aggregation", type=str, default="max_conf", choices=["max_conf", "vote"],
#                     help="If using evidence_labels.csv, how to aggregate to claim-level.")
#     ap.add_argument("--run_id", type=str, default="", help="Optional run id override.")

#     # Report knobs (pass from pipeline OR env)
#     ap.add_argument("--verifier_used", type=str, default=os.getenv("KDSH_VERIFIER_BACKEND", "heuristic"))
#     ap.add_argument("--logic_backend", type=str, default=os.getenv("KDSH_LOGIC_BACKEND", "rules"))

#     # Key thresholds / knobs you want printed
#     ap.add_argument("--min_fact_conf", type=float, default=float(os.getenv("KDSH_MIN_FACT_CONF", "0.45")))
#     ap.add_argument("--score_mix_alpha", type=float, default=float(os.getenv("KDSH_NLI_ALPHA", "0.15")))
#     ap.add_argument("--retrieval_k", type=int, default=int(os.getenv("KDSH_RETRIEVAL_K", "12")))

#     args = ap.parse_args()

#     gold_dir = Path(args.gold_dir)
#     silver_dir = Path(args.silver_dir)

#     if not gold_dir.exists():
#         raise FileNotFoundError(f"gold_dir not found: {gold_dir}")
#     if not silver_dir.exists():
#         raise FileNotFoundError(f"silver_dir not found: {silver_dir}")

#     run_id = _infer_run_id(gold_dir, silver_dir, args.run_id)

#     # Gold truth labels (train) from silver/claims.jsonl
#     gold = load_gold_labels_from_claims(silver_dir)

#     # Predictions from gold/results.csv OR silver/evidence_labels.csv
#     pred, pred_info = load_pred_labels(gold_dir, silver_dir, aggregation=args.aggregation)

#     merged = gold.merge(pred, on="claim_id", how="inner")
#     if merged.empty:
#         raise RuntimeError(
#             "No overlap between gold and pred claim_id.\n"
#             f"- gold rows: {len(gold)} (from {silver_dir/'claims.jsonl'})\n"
#             f"- pred rows:  {len(pred)} (from {pred_info.path})\n"
#         )

#     y_true = merged["gold_label"].tolist()
#     y_pred = merged["pred_label"].tolist()

#     cm = confusion_matrix(y_true, y_pred, LABELS)
#     correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
#     acc = _safe_div(correct, len(y_true))

#     per_class = per_class_metrics(cm, LABELS)
#     macro = macro_avg(per_class, LABELS)

#     metrics: Dict[str, Any] = {
#         "run_id": run_id,
#         "timestamp": datetime.now().isoformat(timespec="seconds"),
#         "n_eval": int(len(y_true)),
#         "accuracy": float(acc),
#         "macro_precision": float(macro["precision"]),
#         "macro_recall": float(macro["recall"]),
#         "macro_f1": float(macro["f1"]),
#         "per_class": per_class,
#         "confusion_matrix": {"labels": LABELS, "matrix": cm},
#         "inputs": {
#             "gold_truth": str((silver_dir / "claims.jsonl").resolve()),
#             "pred_source": pred_info.source,
#             "pred_path": str(pred_info.path.resolve()),
#             "aggregation": pred_info.aggregation,
#         },
#         "run_config": {
#             "verifier_used": args.verifier_used,
#             "logic_backend": args.logic_backend,
#             "min_fact_conf": float(args.min_fact_conf),
#             "score_mix_alpha": float(args.score_mix_alpha),
#             "retrieval_k": int(args.retrieval_k),
#         },
#     }

#     metrics_path = write_metrics_json(gold_dir, metrics)

#     thresholds = {
#         "min_fact_conf": float(args.min_fact_conf),
#         "score_mix_alpha": float(args.score_mix_alpha),
#         "retrieval_k": int(args.retrieval_k),
#     }

#     report_md = build_eval_report_md(
#         run_id=run_id,
#         metrics=metrics,
#         cm=cm,
#         per_class=per_class,
#         pred_info=pred_info,
#         verifier_used=args.verifier_used,
#         logic_backend=args.logic_backend,
#         thresholds=thresholds,
#     )

#     report_path = write_eval_report(gold_dir, run_id, report_md)
#     appended = maybe_append_to_existing_run_report(gold_dir, report_md)

#     print(f"[eval_train] wrote metrics: {metrics_path}")
#     print(f"[eval_train] wrote report:  {report_path}")
#     if appended:
#         print(f"[eval_train] updated existing run report: {appended}")
#     print(f"[eval_train] n_eval={metrics['n_eval']} acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")


# if __name__ == "__main__":
#     main()
