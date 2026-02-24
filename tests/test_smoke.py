from __future__ import annotations

from pathlib import Path
import pandas as pd

from kdsh.common.config import load_config
from kdsh.pipeline.orchestrator import run_pipeline

def test_smoke(tmp_path: Path):
    # tiny novels
    novel1 = tmp_path / "Castaways.txt"
    novel1.write_text("Chapter 1\nJohn was born in Paris. Later he lived in London.\n", encoding="utf-8")
    novel2 = tmp_path / "Monte.txt"
    novel2.write_text("Chapter 1\nEdmond married Mercedes.\n", encoding="utf-8")

    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"

    train_df = pd.DataFrame([
        {"id": 1, "book_name": "Castaways", "char": "John", "content": "John was born in Paris and lived in London.", "label": "consistent"},
    ])
    test_df = pd.DataFrame([
        {"id": 2, "book_name": "Monte", "char": "Edmond", "content": "Edmond married Mercedes.", },
    ])
    train_df.to_csv(train, index=False)
    test_df.to_csv(test, index=False)

    cfg = load_config(None)
    outdir = tmp_path / "out"
    result = run_pipeline(train, test, [novel1, novel2], outdir, cfg, use_pathway=False)

    results_csv = Path(result["results_csv"])
    assert results_csv.exists()
    df = pd.read_csv(results_csv)
    assert set(df.columns) == {"id", "prediction"}
