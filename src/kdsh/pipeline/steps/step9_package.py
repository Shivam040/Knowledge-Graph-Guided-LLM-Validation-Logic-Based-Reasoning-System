from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, Any

def step9_package(out_gold: Path, manifest_path: Path, run_id: str) -> Path:
    results_path = out_gold / "results.csv"
    dossier_zip = out_gold / f"dossier_{run_id}.zip"
    report_path = out_gold / f"report_{run_id}.md"

    lines = []
    lines.append(f"# Run Report — {run_id}\n\n")
    lines.append("This package contains the baseline Track A pipeline outputs.\n\n")
    lines.append("## Included\n")
    lines.append("- results.csv (submission)\n")
    lines.append("- run_manifest.json\n")
    if dossier_zip.exists():
        lines.append("- dossier zip (optional explainability)\n")
    report_path.write_text("".join(lines), encoding="utf-8")

    submission_zip = out_gold / f"submission_{run_id}.zip"
    with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if results_path.exists():
            z.write(results_path, arcname="results.csv")
        if manifest_path.exists():
            z.write(manifest_path, arcname="run_manifest.json")
        if dossier_zip.exists():
            z.write(dossier_zip, arcname=dossier_zip.name)
        z.write(report_path, arcname=report_path.name)

    return submission_zip
