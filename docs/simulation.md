# Simulation study

`scripts/run_simulation.py` (about 21 minutes on 3 cores) downloads the UCI Default of Credit Card Clients data (30,000 accounts), trains a gradient-boosting PD model on a random 30% and treats the other 21,000 scored accounts as the population. For each slice size (500 to 20,000) it builds the calibrated null from one reference of 10,000 accounts, then draws 1,000 no-drift periods, each with a fresh reference of 10,000 and a fresh slice, and 200 periods for each drift type and magnitude:

- covariate drift: resample with weights exp(-d z), z the standardized log credit limit
- label drift: prior shift that multiplies the default rate by (1 + d), keeping the features given the outcome
- calibration drift: a share d of non-defaulters become defaulters, so PD is understated and ranking weakens

`python3 scripts/make_figure.py` turns `docs/results/*.csv` into the figure and these tables.

## False-alarm rate

Per model per period, 1,000 no-drift periods per slice size. The 95% binomial band around 5% is 3.7-6.4%. Source: `docs/results/false_alarm.csv`.

| slice size | score PSI >= 0.10 | score PSI >= 0.25 | score PSI > calibrated 5% threshold | any metric, no correction | rule-of-thumb policy | calibrated + Holm |
|---|---|---|---|---|---|---|
| 500 | 0.0% | 0.0% | 4.4% | 70.2% | 12.1% | 5.4% |
| 1,000 | 0.0% | 0.0% | 4.1% | 67.4% | 8.3% | 5.9% |
| 2,000 | 0.0% | 0.0% | 5.1% | 69.9% | 6.0% | 3.9% |
| 5,000 | 0.0% | 0.0% | 5.8% | 66.7% | 5.2% | 5.3% |
| 10,000 | 0.0% | 0.0% | 5.6% | 69.4% | 5.4% | 5.9% |
| 20,000 | 0.0% | 0.0% | 5.5% | 71.5% | 5.2% | 5.4% |

## Detection power

Share of 200 drifted periods with a model-level alarm, calibrated + Holm / rule-of-thumb policy. Source: `docs/results/power.csv`.

| drift | magnitude | n=500 | n=1,000 | n=2,000 | n=5,000 | n=10,000 | n=20,000 |
|---|---|---|---|---|---|---|---|
| calibration | 0.01 | 5.0% / 23.0% | 6.5% / 14.0% | 5.5% / 14.5% | 8.5% / 16.0% | 17.5% / 22.5% | 17.0% / 23.0% |
| calibration | 0.02 | 8.0% / 31.5% | 12.0% / 25.0% | 19.5% / 39.5% | 41.0% / 57.5% | 65.5% / 76.0% | 75.5% / 81.5% |
| calibration | 0.05 | 41.0% / 67.5% | 65.5% / 84.5% | 90.0% / 96.5% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| calibration | 0.1 | 96.5% / 99.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| covariate | 0.05 | 5.5% / 13.5% | 5.5% / 6.5% | 11.5% / 9.5% | 26.0% / 4.5% | 39.0% / 6.5% | 49.5% / 7.0% |
| covariate | 0.1 | 9.0% / 13.0% | 26.0% / 8.0% | 50.0% / 4.5% | 93.0% / 7.0% | 99.0% / 5.5% | 100.0% / 9.0% |
| covariate | 0.2 | 57.0% / 9.0% | 98.0% / 8.0% | 100.0% / 6.0% | 100.0% / 4.5% | 100.0% / 7.0% | 100.0% / 5.5% |
| covariate | 0.4 | 100.0% / 14.0% | 100.0% / 10.0% | 100.0% / 5.5% | 100.0% / 9.5% | 100.0% / 9.5% | 100.0% / 10.0% |
| label | 0.05 | 6.0% / 17.5% | 5.5% / 7.5% | 7.5% / 14.0% | 13.5% / 18.5% | 13.5% / 14.0% | 21.0% / 20.0% |
| label | 0.1 | 10.5% / 18.5% | 12.5% / 19.5% | 20.0% / 31.0% | 40.5% / 46.0% | 62.5% / 63.5% | 80.0% / 78.0% |
| label | 0.2 | 24.0% / 52.0% | 45.5% / 55.0% | 76.5% / 78.0% | 99.0% / 97.0% | 100.0% / 100.0% | 100.0% / 100.0% |
| label | 0.4 | 86.5% / 97.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |

## Notes on the results

- With deciles and a 10,000-account reference, the largest no-drift score PSI was 0.073 at n = 500 and 0.004 at n = 20,000. The 0.10/0.25 rules never alarmed without drift, and they also missed a credit-limit shift that moved the score PSI to about 0.05. The rule-of-thumb policy caught it in at most 14% of periods; the calibrated threshold (0.0025 to 0.035 depending on slice size) caught it in 100%.
- With 24 PSI/CSI values, two discrimination drops and two calibration tests per period, some metric is significant at 5% in about 70% of no-drift periods. Holm's correction brings that to 3.9-5.9%.
- The rule-of-thumb policy's false-alarm rate depends on slice size (12.1% at n = 500, 5.2% at n = 20,000), driven by the uncorrected HL test and KS drops on small slices. Its higher power on label and calibration drift at small n comes from that higher false-alarm rate.
- Calibrated score PSI alone holds its level: 4.1-5.8% at every size, inside the binomial band.
- A first run used the empirical 95th percentile of 1,000 null draws and gave 3.6-6.8% for the single PSI test, three sizes just outside the band (`docs/results/pilot_empirical_b1000/`). The null mean matched the simulated mean within 3%, so the spread came from estimating a tail quantile from few draws. Thresholds now come from the moment-matched fit, with 4,000 draws in the study and 1,000 by default in the CLI.

## Data

UCI Machine Learning Repository, Default of Credit Card Clients (Yeh, I.-C., 2009), CC BY 4.0: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients. `scripts/download_data.py` fetches it. The tests use only the bundled synthetic portfolio.
