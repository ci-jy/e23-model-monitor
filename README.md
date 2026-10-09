# E-23 Model Risk Monitor

This tool checks two things for a bank's model-risk team getting ready for OSFI Guideline E-23. Does the model inventory follow its own review and monitoring rules, and have a model's scores or outcomes drifted since development? The drift part started from one question: how often does a fixed rule like "PSI above 0.25 is a breach" fire when nothing has changed?

On a credit-card default model, the answer was never. That sounds good until you see that the same rule also missed a credit-limit shift that the calibrated thresholds here caught in 100% of periods.

## Running it

```bash
pip install -r requirements.txt && sudo apt-get install -y libreoffice-calc   # LibreOffice recalculates the Excel formulas
python3 -m e23mon validate examples/inventory.csv                               # inventory check: flags the planted violations
python3 -m e23mon monitor --sample --out out/                                   # breach report + Excel evidence pack
```

`out/` then holds a Markdown and CSV breach report, the inventory violations and `evidence_pack.xlsx`. To run it on your own model, export one CSV for the reference period and one per monitoring period, each with the predicted PD, the 0/1 outcome and the model inputs:

```bash
python3 -m e23mon monitor --reference dev_sample.csv --slices 2026-07.csv 2026-08.csv \
    --score-col pd --target-col default --model-id CR-PD-001 --inventory inventory.csv --out out/
```

Every flag, the YAML config for several models and the output files are in [docs/usage.md](docs/usage.md). The tests run with `python3 -m pytest -q` and take under a minute.

## What it found

I trained a scikit-learn gradient-boosting PD model on the UCI credit-card default data (30,000 accounts) and simulated thousands of monitoring periods, with and without drift. `scripts/run_simulation.py` reproduces every number. It takes about 21 minutes on 3 cores.

![False-alarm rate and detection power](docs/false_alarm_power.png)

Two things stood out. Fixed PSI cut-offs are far too loose for decile bins and a 10,000-account reference: the largest no-drift score PSI was 0.073, so 0.25 is almost never reached. And checking 28 metrics per model without correction goes the other way. Some metric looks significant in about 70% of periods where nothing changed. With calibrated thresholds and Holm's correction that drops to 3.9-5.9%, close to the 5% target. The full tables are in [docs/simulation.md](docs/simulation.md).

## How it works

The inventory is a CSV or YAML file with owner, tiering criteria, risk rating and review and monitoring dates. The validator checks the rating against the criteria, overdue reviews and monitoring, and decommissioned models that still produce scores. The example inventory has seven planted violations and the validator finds all of them. The rules are in [docs/inventory.md](docs/inventory.md).

For monitoring, each period gets a PSI on the score, a CSI per input, drops in AUC and KS, and two calibration tests. The thresholds come from resampling. A monitoring slice is compared with a reference that is itself a sample, so the null draws both a pseudo-reference and a pseudo-slice from the reference and refits the bins each time. Holm's correction then runs across all metrics of a model and period. Details are in [docs/monitoring.md](docs/monitoring.md).

The tricky part was the threshold itself. My first run took the empirical 95th percentile of 1,000 null draws, and the false-alarm rate came out between 3.6% and 6.8%, with three slice sizes outside the binomial band. The null wasn't biased. The tail quantile was just noisy with that few draws. Fitting a scaled chi-square (or normal) to the draws and reading the threshold off the fit fixed it.

The Excel evidence pack has live formulas for every PSI table. The CLI recalculates it in headless LibreOffice and compares against Python. On the sample, the largest difference over 120 tables is 2.8e-16.

## Known issues

- Thresholds assume the reference and slice are independent draws. Overlapping accounts or seasonal portfolios would need a reference per season.
- The scaled chi-square tail is an approximation, so Holm p-values for very sparse categorical features can be slightly off far out in the tail.
- The rating rule and review frequencies are an example policy, not OSFI's. The tool produces inventory checks and monitoring evidence, not full E-23 documentation.
- The bundled sample is synthetic. Only the simulation study uses real (public) data.

Project period: 2026-07-27 to 2026-10-13.
