"""Recalculate the evidence pack's formulas with LibreOffice and compare the
formula-computed PSI values with the Python values."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from openpyxl import load_workbook


def recalculate(xlsx: str | Path, outdir: str | Path) -> Path:
    """Open the workbook in headless LibreOffice, recalculate and save a copy
    with cached formula results. Returns the path of the copy."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        raise RuntimeError("LibreOffice (soffice) is required to recalculate formulas")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="e23mon_lo_"))
    try:
        subprocess.run(
            [soffice, "--headless", "--norestore", f"-env:UserInstallation={profile.as_uri()}",
             "--convert-to", "xlsx:Calc Office Open XML", "--outdir", str(outdir), str(xlsx)],
            check=True, capture_output=True, timeout=300,
        )
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    return outdir / Path(xlsx).name


def recalculate_and_compare(xlsx: str | Path, checks: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix="e23mon_recalc_") as tmp:
        path = recalculate(xlsx, tmp)
        wb = load_workbook(path, data_only=True)
        diffs = []
        for c in checks:
            v = wb[c["sheet"]][c["cell"]].value
            diffs.append(float("inf") if v is None else abs(float(v) - c["python"]))
    return {"n_checked": len(checks), "max_abs_diff": max(diffs) if diffs else 0.0, "diffs": diffs}
