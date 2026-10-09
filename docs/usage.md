# Usage

## validate

```bash
python3 -m e23mon validate INVENTORY [--as-of DATE] [--out FILE] [--strict]
```

| Flag | Meaning |
|---|---|
| `INVENTORY` | inventory file, CSV or YAML |
| `--as-of DATE` | evaluation date (default: the inventory's `as_of` column, otherwise today) |
| `--out FILE` | write violations to this CSV |
| `--strict` | exit with status 1 if any violation is found (for scheduled jobs); without it the exit status is 0 |

## monitor

```bash
python3 -m e23mon monitor --sample --out out/
python3 -m e23mon monitor --config monitoring.yaml --out out/
python3 -m e23mon monitor --reference ref.csv --slices jan.csv feb.csv \
    --score-col pd --target-col default --model-id CR-PD-001 --out out/
```

One of `--sample`, `--config`, or `--reference`/`--slices`/`--score-col` is required.

| Flag | Default | Meaning |
|---|---|---|
| `--sample` | | use the bundled synthetic sample (and its inventory) |
| `--config` | | YAML file listing reference, slices and models |
| `--reference` | | reference CSV (scores, outcomes, features) |
| `--slices` | | slice CSVs, one per period; the period name is the file stem |
| `--score-col` | | column with the predicted PD |
| `--target-col` | `default` | column with the 0/1 outcome |
| `--features` | all other columns | comma-separated feature columns |
| `--model-id` | `MODEL` | model id used in reports |
| `--inventory` | | inventory to validate and include in the workbook |
| `--as-of` | | evaluation date for the inventory checks |
| `--out` | `out` | output directory |
| `--bins` | 10 | number of score bins |
| `--alpha` | 0.05 | family-wise false-alarm level per model per period |
| `--null-draws` | 1000 | resampled null draws per slice size |
| `--boot` | 200 | bootstrap draws for AUC/KS confidence intervals |
| `--seed` | 0 | random seed |
| `--no-verify` | | skip the LibreOffice formula recalculation check |

If LibreOffice (`soffice`) is not on the path, the formula check is skipped and the CLI says so.

## Config file

Several models scored on the same files can be listed in one YAML file. See `e23mon/sample/monitoring.yaml`. Keys: `reference` (CSV path), `target_col`, optional `features`, `models` (a list of `model_id` and `score_col`) and `slices` (a list of `period` and `file`). Paths are relative to the config file.

## Outputs

- `breach_report.md`: per model and period, calibrated and rule-of-thumb alarms, the calibrated breaches and the Excel formula check result
- `breach_report.csv`: every metric, slice and model, with value, threshold, p-value, Holm p-value and CI
- `breach_summary.csv`: one row per model and period
- `inventory_violations.csv`: only when an inventory is given
- `evidence_pack.xlsx`: see [monitoring.md](monitoring.md)
