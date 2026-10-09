# Monitoring metrics and thresholds

## Metrics per model and period

| Metric | Statistic | Null distribution |
|---|---|---|
| `psi:score` | PSI of the PD over reference deciles | resampled from the reference |
| `csi:<feature>` | CSI per input; features with 20 or fewer values are binned as categories, with an OTHER bin for unseen categories and a MISSING bin | resampled from the reference |
| `auc_drop`, `ks_drop` | reference AUC/KS minus slice AUC/KS, with percentile bootstrap CIs | resampled from the reference |
| `hl` | Hosmer-Lemeshow-style chi-square of slice default counts per score decile against the reference default rate of that decile | chi-square, one df per band |
| `oe` | slice observed/expected default ratio against the reference ratio | normal (Poisson-binomial variance) |

PSI/CSI use a 0.5 pseudo-count per bin, so empty bins and new categories give finite values. Bins empty in both samples are dropped. The pseudo-count is part of the PSI definition everywhere, which is why Excel reproduces the Python value exactly.

HL and O/E compare the slice with the reference default rates, not with the raw PD. A model that was slightly off at development does not alarm every period, but a change in calibration does.

## Calibrated thresholds

For a slice of size n, the null is built by drawing (1,000 times by default) a pseudo-reference of the reference size and a pseudo-slice of size n from the reference, refitting the bins on the pseudo-reference and recomputing every drift metric. Resampling only the slice against the fixed reference understates the spread once the slice is as large as the reference, and refitting the bins means the threshold accounts for noise in the decile edges too.

P-values come from a moment-matched scaled chi-square (PSI/CSI) or normal (AUC/KS drop) fitted to the null draws, so they stay smooth below 1/1,000. The calibrated single-test threshold reported next to each value is the 95th percentile of the same fit, which is less noisy than an empirical tail quantile.

Holm's step-down correction runs across all metrics of a model and period, keeping the probability of any false alarm for that model in that period at or below 5%.

## Rule-of-thumb policy

Reported alongside for comparison: PSI/CSI of 0.25 or more, AUC or KS drop of 0.05 or more, O/E outside [0.8, 1.2], and HL p < 0.05, all without multiplicity correction.

## Evidence pack

`evidence_pack.xlsx` has:

- a Summary sheet with the calibrated and rule-of-thumb alarm per model and period
- an Inventory sheet with validator flags, conditional formatting and a live `OVERDUE` formula against the as-of date
- one `MON_<model>` sheet per model with every metric, threshold, p-value, Holm p-value and CI, followed by the bin counts of every PSI/CSI with live formulas next to the Python value:

```
share = (count + 0.5)/(total + 0.5*bins)
term  = (q - p)*LN(q/p)
PSI   = SUM(terms)
```

The CLI recalculates the workbook in headless LibreOffice and reports the largest difference from Python. On the sample it is 2.8e-16 over 120 tables. The tests require agreement within 1e-9.
