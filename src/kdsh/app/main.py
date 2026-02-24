from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd

from kdsh.common.config import load_config
from kdsh.pipeline.orchestrator import run_pipeline

app = FastAPI(title="KDSH Track A MVP API", version="0.1.0")

class ClassifyRequest(BaseModel):
    train_csv: str = Field(..., description="Path to train.csv (used for threshold calibration)")
    test_csv: str = Field(..., description="Path to test.csv (can be the same as a 1-row temp file)")
    novels: List[str] = Field(..., description="Paths to novel .txt files")
    outdir: str = Field(..., description="Output directory")
    use_pathway: bool = False

@app.post("/run")
def run(req: ClassifyRequest) -> Dict[str, Any]:
    cfg = load_config(Path("configs/dev.yaml"))
    result = run_pipeline(
        train_csv=Path(req.train_csv),
        test_csv=Path(req.test_csv),
        novel_paths=[Path(p) for p in req.novels],
        outdir=Path(req.outdir),
        cfg=cfg,
        use_pathway=req.use_pathway,
    )
    return result
