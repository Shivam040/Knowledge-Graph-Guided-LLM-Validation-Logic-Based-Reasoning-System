from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from kdsh.common.config import AppConfig
from kdsh.common.utils import IST, sha256_file, normalize_book_key
from kdsh.pipeline.steps.step1_ingest import step1_ingest_and_chunk
from kdsh.pipeline.steps.step2_claims import step2_build_claims
from kdsh.pipeline.steps.step3_retrieve import step3_retrieve
from kdsh.pipeline.steps.step4_verify import step4_verify
from kdsh.pipeline.steps.step5_kg import step5_build_kg
from kdsh.pipeline.steps.step6_logic import step6_logic
from kdsh.pipeline.steps.step7_aggregate import step7_aggregate
from kdsh.pipeline.steps.step8_dossier import step8_dossier
from kdsh.pipeline.steps.step9_package import step9_package

def build_run_dirs(outdir: Path, run_id: str) -> Tuple[Path, Path, Path, Path]:
    bronze = outdir / "data" / "bronze"
    silver = outdir / "data" / "silver" / f"run_id={run_id}"
    gold = outdir / "data" / "gold" / f"run_id={run_id}"
    manifests = outdir / "data" / "run_manifests"
    for d in [bronze, silver, gold, manifests]:
        d.mkdir(parents=True, exist_ok=True)
    (bronze / "novels").mkdir(exist_ok=True)
    return bronze, silver, gold, manifests

def write_manifest(manifest_path: Path, payload: Dict[str, Any]) -> None:
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def remap_book_names(train_df: pd.DataFrame, test_df: pd.DataFrame, novels: List[Tuple[str, Path]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    novel_key_to_pretty = {normalize_book_key(n[0]): n[0] for n in novels}

    def remap_book_name(df: pd.DataFrame) -> pd.DataFrame:
        def remap(x):
            k = normalize_book_key(x)
            if k in novel_key_to_pretty:
                return novel_key_to_pretty[k]
            for kk, pretty in novel_key_to_pretty.items():
                if kk in k or k in kk:
                    return pretty
            return x
        df = df.copy()
        df["book_name"] = df["book_name"].apply(remap)
        return df

    return remap_book_name(train_df), remap_book_name(test_df)

def run_pipeline(
    train_csv: Path,
    test_csv: Path,
    novel_paths: List[Path],
    outdir: Path,
    cfg: AppConfig,
    use_pathway: bool = False,
) -> Dict[str, Any]:
    now_ist = datetime.now(IST)
    run_id = f"run_{now_ist.strftime('%Y%m%d_%H%M%S')}"
    bronze, silver, gold, manifests = build_run_dirs(outdir, run_id)

    # Copy bronze inputs
    shutil.copy2(train_csv, bronze / "train.csv")
    shutil.copy2(test_csv, bronze / "test.csv")
    bronze_novels_dir = bronze / "novels"
    # bronze_novels_dir = "novels"

    novels: List[Tuple[str, Path]] = []
    for p in novel_paths:
        shutil.copy2(p, bronze_novels_dir / f"{p.name}")
    for p in novel_paths:
        novels.append((p.stem.strip(), p))

    train_df = pd.read_csv(bronze / "train.csv")
    test_df = pd.read_csv(bronze / "test.csv")
    train_df, test_df = remap_book_names(train_df, test_df, novels)

    manifest_path = manifests / f"{run_id}.json"
    manifest = {
        "run_id": run_id,
        "created_at_ist": now_ist.isoformat(),
        "inputs": {
            "train_csv": {"path": str(bronze / "train.csv"), "sha256": sha256_file(train_csv)},
            "test_csv": {"path": str(bronze / "test.csv"), "sha256": sha256_file(test_csv)},
            "novels": [{"book_name": bn, "file": str(bronze_novels_dir / p.name), "sha256": sha256_file(p)} for bn, p in novels],
        },
        "parameters": {"config": {
            "chunking": asdict(cfg.chunking),
            "retrieval": asdict(cfg.retrieval),
            "verifier": asdict(cfg.verifier),
            "kg": asdict(cfg.kg),
            "aggregation": asdict(cfg.aggregation),
            "use_pathway": bool(use_pathway),
        }},
        "outputs": {},
    }

    # Step 1
    registry_path, chunks_path = step1_ingest_and_chunk(
        novels=[(bn, p) for (bn, p) in novels],
        out_silver=silver,
        out_bronze_novels=bronze_novels_dir,
        run_id=run_id,
        cfg=cfg.chunking,
    )
    manifest["outputs"]["novel_registry"] = str(registry_path)
    manifest["outputs"]["chunks"] = str(chunks_path)

    chunks_df = pd.read_csv(chunks_path)

    # Step 2
    claims_path = step2_build_claims(train_df, test_df, silver, run_id)
    manifest["outputs"]["claims"] = str(claims_path)

    # Step 3
    backend = "pathway" if use_pathway else cfg.retrieval.backend
    retrieval_path, coverage_path = step3_retrieve(
        chunks_df,
        claims_path,
        silver,
        run_id,
        backend=backend,
        K=cfg.retrieval.k,
        candidate_pool=cfg.retrieval.candidate_pool,
        max_per_chapter=cfg.retrieval.max_per_chapter,
        enforce_buckets=cfg.retrieval.enforce_buckets,
    )
    manifest["outputs"]["retrieval_candidates"] = str(retrieval_path)
    manifest["outputs"]["retrieval_coverage"] = str(coverage_path)

    # Step 4
    evidence_path, facts_path = step4_verify(
        chunks_df, claims_path, retrieval_path, silver, run_id, min_fact_conf=cfg.verifier.min_fact_conf, verifier_cfg=cfg.verifier,
    )
    manifest["outputs"]["evidence_labels"] = str(evidence_path)
    manifest["outputs"]["facts"] = str(facts_path)

    # Step 5
    aliases_path, kg_path = step5_build_kg(
        chunks_df, claims_path, facts_path, silver, run_id, min_fact_conf=cfg.kg.min_fact_conf
    )
    manifest["outputs"]["aliases"] = str(aliases_path)
    manifest["outputs"]["kg_triples"] = str(kg_path)

    # Step 6
    constraint_path, grounded_path = step6_logic(kg_path, train_df, test_df, gold, run_id)
    manifest["outputs"]["constraint_runs"] = str(constraint_path)
    manifest["outputs"]["constraints_grounded"] = str(grounded_path)

    # Step 7
    decision_path, results_path, thr = step7_aggregate(
        train_df,
        test_df,
        evidence_path,
        constraint_path,
        gold,
        run_id,
        contradiction_penalty=cfg.aggregation.contradiction_penalty,
    )
    manifest["parameters"]["aggregation_threshold_calibrated_on_train"] = thr
    manifest["outputs"]["decision_scores"] = str(decision_path)
    manifest["outputs"]["results"] = str(results_path)

    # Step 8
    dossier_index, dossier_zip = step8_dossier(
        chunks_df, claims_path, retrieval_path, evidence_path, decision_path, gold, run_id
    )
    manifest["outputs"]["dossier_index"] = str(dossier_index)
    manifest["outputs"]["dossier_zip"] = str(dossier_zip)

    write_manifest(manifest_path, manifest)

    # Step 9
    submission_zip = step9_package(gold, manifest_path, run_id)
    manifest["outputs"]["submission_zip"] = str(submission_zip)
    write_manifest(manifest_path, manifest)

    return {
        "run_id": run_id,
        "results_csv": str(results_path),
        "submission_zip": str(submission_zip),
        "manifest": str(manifest_path),
        "outdir": str(outdir),
    }
