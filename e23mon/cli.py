"""Command line interface.

    python3 -m e23mon validate INVENTORY [--as-of DATE] [--out FILE] [--strict]
    python3 -m e23mon monitor --sample --out out/
    python3 -m e23mon monitor --config monitoring.yaml --out out/
    python3 -m e23mon monitor --reference ref.csv --slices jan.csv feb.csv \\
        --score-col pd --target-col default --model-id CR-PD-001 --out out/
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from .excel import build_workbook
from .formula_check import recalculate_and_compare
from .inventory import load_inventory, resolve_as_of, validate, violations_frame
from .monitor import Config, Reference, Sample, evaluate_slice, summarize

SAMPLE_DIR = Path(__file__).resolve().parent / "sample"


def _date(s: str) -> date:
    return date.fromisoformat(s)


def cmd_validate(a) -> int:
    inv = load_inventory(a.inventory)
    as_of = resolve_as_of(inv, a.as_of)
    v = violations_frame(validate(inv, as_of))
    print(f"Inventory {a.inventory}: {len(inv)} models, as of {as_of}")
    if v.empty:
        print("No violations.")
    else:
        print(f"{len(v)} violation(s) in {v.model_id.nunique()} model(s):")
        print(v.to_string(index=False))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        v.to_csv(a.out, index=False)
    return 1 if (a.strict and not v.empty) else 0


def _load_config(a) -> tuple[dict, Path]:
    if a.sample:
        return yaml.safe_load((SAMPLE_DIR / "monitoring.yaml").read_text()), SAMPLE_DIR
    if a.config:
        p = Path(a.config)
        return yaml.safe_load(p.read_text()), p.parent
    if not (a.reference and a.slices and a.score_col):
        raise SystemExit("monitor needs --sample, --config, or --reference/--slices/--score-col")
    cfg = {
        "reference": str(Path(a.reference).resolve()),
        "target_col": a.target_col,
        "features": a.features.split(",") if a.features else None,
        "models": [{"model_id": a.model_id, "score_col": a.score_col}],
        "slices": [{"period": Path(s).stem, "file": str(Path(s).resolve())} for s in a.slices],
    }
    return cfg, Path(".")


def run_monitoring(cfg: dict, base: Path, out: Path, config: Config, inventory_path=None, as_of=None,
                   verify: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    ref_df = pd.read_csv(base / cfg["reference"])
    target = cfg.get("target_col", "default")
    score_cols = [m["score_col"] for m in cfg["models"]]
    features = cfg.get("features") or [c for c in ref_df.columns if c != target and c not in score_cols]
    slices = [(s["period"], pd.read_csv(base / s["file"])) for s in cfg["slices"]]
    all_rows, summary_rows, model_results = [], [], {}
    for m in cfg["models"]:
        ref = Reference(Sample.from_frame(ref_df, m["score_col"], target, features), config)
        results = []
        for period, df in slices:
            cur = Sample.from_frame(df, m["score_col"], target, features)
            res = evaluate_slice(ref, cur)
            results.append((period, cur, res))
            all_rows.append(res.assign(model_id=m["model_id"], period=period, slice_size=len(cur)))
            summary_rows.append({"model_id": m["model_id"], "period": period, "slice_size": len(cur),
                                 "default_rate": float(cur.y.mean()), **summarize(res)})
        model_results[m["model_id"]] = (ref, results)
    report = pd.concat(all_rows, ignore_index=True)
    lead = ["model_id", "period", "slice_size"]
    report = report[lead + [c for c in report.columns if c not in lead]]
    summary = pd.DataFrame(summary_rows)
    report.to_csv(out / "breach_report.csv", index=False)
    summary.to_csv(out / "breach_summary.csv", index=False)

    inv = viol = None
    if inventory_path:
        inv = load_inventory(inventory_path)
        as_of = resolve_as_of(inv, as_of)
        viol = violations_frame(validate(inv, as_of))
        viol.to_csv(out / "inventory_violations.csv", index=False)
    xlsx = out / "evidence_pack.xlsx"
    checks = build_workbook(summary, model_results, inv, viol, as_of, xlsx)
    check_result = None
    if verify:
        check_result = recalculate_and_compare(xlsx, checks)
    _write_markdown(out / "breach_report.md", summary, report, check_result, config)
    return {"summary": summary, "report": report, "checks": checks, "check_result": check_result,
            "workbook": xlsx}


def _write_markdown(path: Path, summary: pd.DataFrame, report: pd.DataFrame, check, config: Config):
    lines = ["# Monitoring breach report", "",
             f"Calibrated alarms: Holm-adjusted p <= {config.alpha} across all metrics of a model and period "
             "(family-wise false-alarm rate per model per period at the nominal level).",
             "Rule-of-thumb alarms: PSI/CSI >= 0.25, AUC or KS drop >= 0.05, O/E outside [0.8, 1.2], "
             "HL p < 0.05 with no multiplicity correction.", "",
             "| model | period | n | default rate | calibrated alarm | calibrated breaches | rule-of-thumb alarm | rule-of-thumb breaches |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r.model_id} | {r.period} | {r.slice_size} | {r.default_rate:.3f} | "
                     f"{'BREACH' if r.calibrated_alarm else 'ok'} | {r.calibrated_breaches or '-'} | "
                     f"{'BREACH' if r.rule_of_thumb_alarm else 'ok'} | {r.rule_of_thumb_breaches or '-'} |")
    br = report[report.calibrated_breach]
    if len(br):
        lines += ["", "## Calibrated breaches", "", "| model | period | metric | value | threshold (single test) | Holm p |",
                  "|---|---|---|---|---|---|"]
        for _, r in br.iterrows():
            thr = "-" if pd.isna(r.calibrated_threshold) else f"{r.calibrated_threshold:.4f}"
            lines.append(f"| {r.model_id} | {r.period} | {r.metric} | {r.value:.4f} | {thr} | {r.p_holm:.2e} |")
    if check is not None:
        lines += ["", "## Excel formula check", "",
                  f"{check['n_checked']} PSI/CSI formulas recalculated; max |Excel - Python| = {check['max_abs_diff']:.2e}."]
    path.write_text("\n".join(lines) + "\n")


def cmd_monitor(a) -> int:
    cfg, base = _load_config(a)
    config = Config(n_bins=a.bins, alpha=a.alpha, n_null=a.null_draws, n_boot=a.boot, seed=a.seed)
    inv_path = a.inventory or (str(SAMPLE_DIR / "inventory.csv") if a.sample else None)
    verify = not a.no_verify and shutil.which("soffice") is not None
    res = run_monitoring(cfg, base, Path(a.out), config, inv_path, a.as_of, verify=verify)
    s = res["summary"]
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(s[["model_id", "period", "slice_size", "calibrated_alarm", "calibrated_breaches",
                 "rule_of_thumb_alarm"]].to_string(index=False))
    if res["check_result"] is not None:
        c = res["check_result"]
        print(f"Excel formula check: {c['n_checked']} PSI cells, max |Excel - Python| = {c['max_abs_diff']:.2e}")
    elif not a.no_verify:
        print("Excel formula check skipped: LibreOffice (soffice) not found")
    print(f"Wrote {Path(a.out) / 'breach_report.md'}, breach_report.csv and {res['workbook']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="e23mon", description="E-23 model inventory checks and drift monitoring")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate", help="validate a model inventory (CSV or YAML)")
    v.add_argument("inventory")
    v.add_argument("--as-of", type=_date, default=None, help="evaluation date (default: inventory as_of or today)")
    v.add_argument("--out", help="write violations to this CSV")
    v.add_argument("--strict", action="store_true", help="exit with status 1 if any violation is found")
    v.set_defaults(func=cmd_validate)

    m = sub.add_parser("monitor", help="run monitoring on scored slice CSVs")
    m.add_argument("--sample", action="store_true", help="use the bundled synthetic sample")
    m.add_argument("--config", help="YAML file listing reference, slices and models")
    m.add_argument("--reference", help="reference CSV (scores, outcomes, features)")
    m.add_argument("--slices", nargs="+", help="slice CSVs, one per period")
    m.add_argument("--score-col", help="column with the predicted PD")
    m.add_argument("--target-col", default="default", help="column with the 0/1 outcome")
    m.add_argument("--features", help="comma-separated feature columns (default: all other columns)")
    m.add_argument("--model-id", default="MODEL")
    m.add_argument("--inventory", help="inventory to validate and include in the workbook")
    m.add_argument("--as-of", type=_date, default=None)
    m.add_argument("--out", default="out")
    m.add_argument("--bins", type=int, default=10)
    m.add_argument("--alpha", type=float, default=0.05)
    m.add_argument("--null-draws", type=int, default=500)
    m.add_argument("--boot", type=int, default=200)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--no-verify", action="store_true", help="skip the LibreOffice formula recalculation check")
    m.set_defaults(func=cmd_monitor)
    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
