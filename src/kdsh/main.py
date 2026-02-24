from __future__ import annotations

import argparse
from pathlib import Path

from kdsh.common.config import load_config
from kdsh.pipeline.orchestrator import run_pipeline

def main():
    ap = argparse.ArgumentParser(description="KDSH Track A MVP pipeline (Pathway-ready).")
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--test", required=True, type=Path)
    ap.add_argument("--novels", required=True, nargs="+", type=Path)
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--run-all", action="store_true", help="kept for backward compatibility; pipeline always runs end-to-end")
    ap.add_argument("--config", type=Path, default=Path("configs/dev.yaml"))
    ap.add_argument("--use-pathway", action="store_true", help="force Pathway retrieval backend (if installed)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    result = run_pipeline(
        train_csv=args.train,
        test_csv=args.test,
        novel_paths=args.novels,
        outdir=args.outdir,
        cfg=cfg,
        use_pathway=args.use_pathway,
    )

    print("\nDONE ✅")
    print("run_id:", result["run_id"])
    print("results.csv:", result["results_csv"])
    print("submission zip:", result["submission_zip"])
    print("manifest:", result["manifest"])

if __name__ == "__main__":
    main()
