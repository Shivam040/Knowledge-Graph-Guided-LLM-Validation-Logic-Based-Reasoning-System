from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

def normalize_keywords(x: Any) -> List[str]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            y = json.loads(x)
            if isinstance(y, list):
                return y
        except Exception:
            pass
        return [w.strip() for w in re.split(r"[,\s]+", x) if w.strip()]
    return []

def make_excerpt(text: str, terms: List[str], max_len: int = 320) -> str:
    if not text:
        return ""
    tl = text.lower()
    idx = -1
    for t in terms:
        if not t:
            continue
        j = tl.find(str(t).lower())
        if j != -1:
            idx = j
            break
    if idx == -1:
        snippet = text[:max_len]
        return snippet + ("…" if len(text) > max_len else "")
    start = max(0, idx - 120)
    end = min(len(text), idx + 120)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    if len(snippet) > max_len:
        snippet = snippet[: max_len - 1] + "…"
    return snippet

def step8_dossier(
    chunks_df: pd.DataFrame,
    claims_path: Path,
    retrieval_path: Path,
    evidence_path: Path,
    decision_path: Path,
    out_gold: Path,
    run_id: str,
    key_claims_per_id: int = 4,
    evidence_rows_per_claim: int = 3,
) -> Tuple[Path, Path]:
    dossier_dir = out_gold / "dossier"
    dossier_dir.mkdir(parents=True, exist_ok=True)

    retrieval_df = pd.read_csv(retrieval_path)
    evidence_df = pd.read_csv(evidence_path)
    decision_df = pd.read_csv(decision_path)

    claim_rows = [json.loads(line) for line in claims_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    claims_df = pd.DataFrame(claim_rows)

    chunk_text_map = dict(zip(chunks_df["chunk_id"], chunks_df["chunk_text"]))
    chunk_meta_map = chunks_df.set_index("chunk_id")[["chapter_id", "time_bucket"]].to_dict("index")

    evidence_grp = evidence_df.groupby(["id", "claim_id"])
    retrieval_grp = retrieval_df.groupby(["id", "claim_id"])
    claims_by_id = claims_df.groupby("id")

    written = []
    for _, drow in decision_df.iterrows():
        ex_id = int(drow["id"])
        pred = int(drow["prediction"])
        status = str(drow.get("status", "SAT"))
        rationale = str(drow.get("rationale", ""))
        split = str(drow.get("split", ""))

        if ex_id not in claims_by_id.groups:
            continue
        cdf = claims_by_id.get_group(ex_id).copy()
        if len(cdf) == 0:
            continue

        per_claim = []
        for _, crow in cdf.iterrows():
            cid = crow["claim_id"]
            if (ex_id, cid) in evidence_grp.groups:
                ev = evidence_grp.get_group((ex_id, cid))
                sup_sum = float(ev.loc[ev["label"] == "SUPPORT", "confidence"].sum())
                con_sum = float(ev.loc[ev["label"] == "CONTRADICT", "confidence"].sum())
                neu_sum = float(ev.loc[ev["label"] == "NEUTRAL", "confidence"].sum())
                per_claim.append((cid, sup_sum, con_sum, neu_sum))
            else:
                per_claim.append((cid, 0.0, 0.0, 0.0))
        per_claim_df = pd.DataFrame(per_claim, columns=["claim_id", "support_sum", "contradict_sum", "neutral_sum"])

        if (per_claim_df["contradict_sum"] > 0).any():
            key = per_claim_df.sort_values(["contradict_sum", "support_sum"], ascending=False).head(key_claims_per_id)
        else:
            key = per_claim_df.sort_values(["support_sum", "neutral_sum"], ascending=False).head(key_claims_per_id)

        book_name = str(cdf.iloc[0].get("book_name", ""))
        char = str(cdf.iloc[0].get("char", ""))
        caption = cdf.iloc[0].get("caption", "")
        caption = "" if pd.isna(caption) else str(caption)

        lines = []
        lines.append(f"# Evidence Dossier — id {ex_id}\n")
        lines.append(f"**book_name:** {book_name}  ")
        lines.append(f"**target char:** {char}  ")
        if caption:
            lines.append(f"**caption:** {caption}  ")
        lines.append(f"**split:** {split}  ")
        lines.append(f"**prediction:** {pred}  ")
        lines.append(f"**logic_status:** {status}  ")
        if rationale:
            lines.append(f"**rationale:** {rationale}")
        lines.append("\n---\n")
        lines.append("## Key claims & linked evidence\n")

        for _, krow in key.iterrows():
            cid = krow["claim_id"]
            crow = cdf[cdf["claim_id"] == cid].iloc[0]
            claim_text = str(crow["claim_text"])
            claim_type = str(crow.get("claim_type", ""))
            t_hint = str(crow.get("t_hint", "UNK"))
            kws = normalize_keywords(crow.get("keywords", []))

            lines.append(f"### {cid}  ")
            lines.append(f"**type:** {claim_type} | **t_hint:** {t_hint}\n")
            lines.append(f"**claim:** {claim_text}\n")

            if (ex_id, cid) in evidence_grp.groups:
                ev = evidence_grp.get_group((ex_id, cid)).copy()
                if (ex_id, cid) in retrieval_grp.groups:
                    rr = retrieval_grp.get_group((ex_id, cid))[["chunk_id", "score_lex_norm", "rank"]]
                    ev = ev.merge(rr, on="chunk_id", how="left")

                if (ev["label"] == "CONTRADICT").any():
                    order = {"CONTRADICT": 0, "SUPPORT": 1, "NEUTRAL": 2}
                    ev["label_order"] = ev["label"].map(order).fillna(9)
                    ev = ev.sort_values(["label_order", "confidence"], ascending=[True, False])
                else:
                    ev = ev.sort_values("confidence", ascending=False)

                ev = ev.head(evidence_rows_per_claim)

                lines.append("| label | conf | rank | score_lex_norm | chunk_id | chapter | bucket | excerpt |")
                lines.append("|---|---:|---:|---:|---|---|---|---|")
                for _, er in ev.iterrows():
                    chunk_id = er["chunk_id"]
                    meta = chunk_meta_map.get(chunk_id, {})
                    txt = chunk_text_map.get(chunk_id, "")
                    excerpt = make_excerpt(txt, [char] + kws[:6]).replace("|", "\\|").replace("\n", " ")
                    rank = "" if pd.isna(er.get("rank")) else int(er.get("rank"))
                    sc = "" if pd.isna(er.get("score_lex_norm")) else float(er.get("score_lex_norm"))
                    lines.append(
                        f"| {er['label']} | {float(er['confidence']):.3f} | {rank} | "
                        f"{'' if sc=='' else f'{sc:.3f}'} | `{chunk_id}` | {meta.get('chapter_id','')} | {meta.get('time_bucket','')} | {excerpt} |"
                    )
                lines.append("\n---\n")
            else:
                lines.append("_No evidence found for this claim._\n---\n")

        out_path = dossier_dir / f"{ex_id}.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        written.append({"id": ex_id, "prediction": pred, "split": split, "dossier_path": str(out_path)})

    index_df = pd.DataFrame(written).sort_values("id")
    index_path = dossier_dir / "dossier_index.csv"
    index_df.to_csv(index_path, index=False)

    zip_path = out_gold / f"dossier_{run_id}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in dossier_dir.rglob("*"):
            z.write(p, arcname=str(p.relative_to(out_gold)))

    return index_path, zip_path
