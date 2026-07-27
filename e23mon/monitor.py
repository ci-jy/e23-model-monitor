"""Monitoring engine: per-slice metrics, calibrated null distributions and alarms.

A slice is compared with the reference (development or validation) sample on:

* ``psi:score``      PSI of the model score
* ``csi:<feature>``  characteristic stability index of each input feature
* ``auc_drop``       reference AUC minus slice AUC
* ``ks_drop``        reference KS minus slice KS
* ``hl``             Hosmer-Lemeshow-style test of default rates by score band
                     against the reference rates
* ``oe``             test of the slice observed-to-expected default ratio
                     against the reference ratio

Drift metrics have no closed-form null, so their null distribution is built by
resampling the reference: each replicate draws a pseudo-reference of the
reference size and a pseudo-slice of the slice size from the reference, refits
the bins on the pseudo-reference and recomputes every metric. Calibrated
thresholds are quantiles of that distribution; p-values come from a
moment-matched scaled chi-square (PSI/CSI) or normal (AUC/KS drop) fit so that
Holm's correction can work below the resolution of the resampling. The
calibration tests already have analytic nulls.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import stats as st

# Fixed thresholds commonly used in monitoring standards, kept for comparison.
RULE_OF_THUMB = {
    "psi_amber": 0.10,
    "psi_red": 0.25,
    "auc_drop": 0.05,
    "ks_drop": 0.05,
    "oe_low": 0.80,
    "oe_high": 1.20,
    "hl_p": 0.05,
}


@dataclass
class Sample:
    """Scored data for one period: PD score, binary outcome and features."""

    score: np.ndarray
    y: np.ndarray
    features: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, score_col: str, target_col: str, feature_cols=None) -> "Sample":
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c not in (score_col, target_col)]
        feats = {}
        for c in feature_cols:
            col = df[c]
            feats[c] = col.to_numpy(dtype=float) if pd.api.types.is_numeric_dtype(col) else col.to_numpy(dtype=object)
        return cls(df[score_col].to_numpy(dtype=float), df[target_col].to_numpy(dtype=float), feats)

    def __len__(self) -> int:
        return self.score.size

    def take(self, idx: np.ndarray) -> "Sample":
        return Sample(self.score[idx], self.y[idx], {k: v[idx] for k, v in self.features.items()})


@dataclass
class Config:
    n_bins: int = 10
    smoothing: float = st.DEFAULT_SMOOTHING
    max_categories: int = 20
    alpha: float = 0.05
    n_null: int = 500
    n_boot: int = 200
    seed: int = 0


class Reference:
    """Reference sample with its fitted bins and discrimination metrics."""

    def __init__(self, sample: Sample, config: Config | None = None):
        self.sample = sample
        self.config = config or Config()
        c = self.config
        self.binnings = {"score": st.fit_binning(sample.score, c.n_bins, c.max_categories, categorical=False)}
        for name, x in sample.features.items():
            self.binnings[name] = st.fit_binning(x, c.n_bins, c.max_categories)
        self.ref_counts = {k: b.counts(self._column(sample, k)) for k, b in self.binnings.items()}
        self.band_counts, self.band_defaults = self.score_bands(sample)
        self.auc = st.auc(sample.y, sample.score)
        self.ks = st.ks(sample.y, sample.score)
        self._nulls: dict[int, "NullModel"] = {}

    @staticmethod
    def _column(sample: Sample, key: str) -> np.ndarray:
        return sample.score if key == "score" else sample.features[key]

    def score_bands(self, cur: Sample) -> tuple[np.ndarray, np.ndarray]:
        """Counts and defaults of ``cur`` in the reference score bins."""
        b = self.binnings["score"]
        counts = b.counts(cur.score)[:-1]
        idx = np.searchsorted(b.edges, cur.score, side="left")
        defaults = np.bincount(idx, weights=cur.y, minlength=counts.size)
        return counts, defaults

    def calibration_metrics(self, cur: Sample) -> dict[str, float]:
        n, d = self.score_bands(cur)
        hl_stat, hl_p, _ = st.hl_drift(self.band_counts, self.band_defaults, n, d)
        oe, oe_z, oe_p = st.oe_drift(self.sample.y, self.sample.score, cur.y, cur.score)
        return {"hl_stat": hl_stat, "hl_p": hl_p, "oe": oe, "oe_z": oe_z, "oe_p": oe_p}

    def drift_metrics(self, cur: Sample) -> dict[str, float]:
        """PSI/CSI and discrimination drops of ``cur`` against this reference."""
        sm = self.config.smoothing
        out = {}
        for k, b in self.binnings.items():
            name = "psi:score" if k == "score" else f"csi:{k}"
            out[name] = st.psi_from_counts(self.ref_counts[k], b.counts(self._column(cur, k)), sm)
        out["auc_drop"] = self.auc - st.auc(cur.y, cur.score)
        out["ks_drop"] = self.ks - st.ks(cur.y, cur.score)
        return out

    def null(self, n: int) -> "NullModel":
        if n not in self._nulls:
            self._nulls[n] = NullModel.build(self.sample, n, self.config)
        return self._nulls[n]


@dataclass
class NullModel:
    """Resampled null distribution of the drift metrics for one slice size."""

    n: int
    draws: dict[str, np.ndarray]

    @classmethod
    def build(cls, ref: Sample, n: int, config: Config) -> "NullModel":
        rng = np.random.default_rng([config.seed, n])
        m = len(ref)
        rows: dict[str, list[float]] = {}
        for _ in range(config.n_null):
            pseudo_ref = ref.take(rng.integers(0, m, m))
            pseudo_cur = ref.take(rng.integers(0, m, n))
            r = Reference(pseudo_ref, config)
            for k, v in r.drift_metrics(pseudo_cur).items():
                rows.setdefault(k, []).append(v)
        return cls(n, {k: np.asarray(v) for k, v in rows.items()})

    def threshold(self, metric: str, alpha: float) -> float:
        d = self.draws[metric]
        d = d[~np.isnan(d)]
        return float(np.quantile(d, 1 - alpha)) if d.size else float("nan")

    def pvalue(self, metric: str, value: float) -> float:
        if np.isnan(value):
            return float("nan")
        d = self.draws[metric]
        d = d[~np.isnan(d)]
        if d.size < 10:
            return float("nan")
        mu, var = d.mean(), d.var(ddof=1)
        if metric.startswith(("psi:", "csi:")):
            if var <= 0 or mu <= 0:
                return float(np.mean(d >= value))
            # Satterthwaite: value ~ c * chi2(df) with matching mean and variance.
            c = var / (2 * mu)
            df = 2 * mu**2 / var
            return float(sps.chi2.sf(value / c, df))
        sd = np.sqrt(var)
        if sd <= 0:
            return float(np.mean(d >= value))
        return float(sps.norm.sf((value - mu) / sd))


def evaluate_slice(ref: Reference, cur: Sample, with_ci: bool = True) -> pd.DataFrame:
    """All metrics for one slice with calibrated p-values, Holm decisions and
    the rule-of-thumb decision side by side. One row per metric."""
    c = ref.config
    drift = ref.drift_metrics(cur)
    null = ref.null(len(cur))
    rows = []
    for k, v in drift.items():
        rows.append({
            "metric": k,
            "value": v,
            "reference": {"auc_drop": ref.auc, "ks_drop": ref.ks}.get(k, np.nan),
            "calibrated_threshold": null.threshold(k, c.alpha),
            "p_value": null.pvalue(k, v),
            "rule_of_thumb_breach": _rot(k, v),
        })
    cal = ref.calibration_metrics(cur)
    rows.append({"metric": "hl", "value": cal["hl_stat"], "reference": np.nan, "calibrated_threshold": np.nan,
                 "p_value": cal["hl_p"], "rule_of_thumb_breach": bool(cal["hl_p"] < RULE_OF_THUMB["hl_p"])})
    rows.append({"metric": "oe", "value": cal["oe"], "reference": 1.0, "calibrated_threshold": np.nan,
                 "p_value": cal["oe_p"], "rule_of_thumb_breach": _rot("oe", cal["oe"])})
    df = pd.DataFrame(rows)
    adj, rej = st.holm(df["p_value"].to_numpy(), c.alpha)
    df["p_holm"] = adj
    df["calibrated_breach"] = rej
    df["testable"] = ~df["p_value"].isna()
    if with_ci:
        lo, hi = st.bootstrap_ci(cur.y, cur.score, st.auc, c.n_boot, seed=c.seed)
        df["ci_low"] = np.nan
        df["ci_high"] = np.nan
        df.loc[df.metric == "auc_drop", ["ci_low", "ci_high"]] = [ref.auc - hi, ref.auc - lo]
        lo, hi = st.bootstrap_ci(cur.y, cur.score, st.ks, c.n_boot, seed=c.seed)
        df.loc[df.metric == "ks_drop", ["ci_low", "ci_high"]] = [ref.ks - hi, ref.ks - lo]
    return df


def _rot(metric: str, v: float) -> bool:
    if np.isnan(v):
        return False
    r = RULE_OF_THUMB
    if metric.startswith(("psi:", "csi:")):
        return bool(v >= r["psi_red"])
    if metric == "auc_drop":
        return bool(v >= r["auc_drop"])
    if metric == "ks_drop":
        return bool(v >= r["ks_drop"])
    if metric == "oe":
        return bool(v < r["oe_low"] or v > r["oe_high"])
    return False


def summarize(df: pd.DataFrame) -> dict:
    """Model-level decision for one slice."""
    return {
        "calibrated_alarm": bool(df["calibrated_breach"].any()),
        "rule_of_thumb_alarm": bool(df["rule_of_thumb_breach"].any()),
        "calibrated_breaches": ", ".join(df.loc[df.calibrated_breach, "metric"]),
        "rule_of_thumb_breaches": ", ".join(df.loc[df.rule_of_thumb_breach, "metric"]),
        "untestable": ", ".join(df.loc[~df.testable, "metric"]),
    }
