"""Simulation study: false-alarm rate and detection power of rule-of-thumb
thresholds against calibrated thresholds with Holm correction.

Population: the scored UCI rows not used for training. Every replicate draws a
fresh reference sample (size ``ref_size``) and a monitoring slice (size ``n``)
from the population, with or without injected drift, refits the bins on that
reference and evaluates every metric. Calibrated thresholds and p-values come
from the null model built once per slice size from a separate canonical
reference, exactly as the monitoring CLI does for an analyst's own reference.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import stats as st
from .data import DRIFTS, FEATURES, TARGET, load_uci, score, train_models
from .monitor import RULE_OF_THUMB, Config, Reference, Sample, _rot

SCORE_COL = "pd_gbm"


def binom_ci(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval for a binomial proportion."""
    from scipy.stats import beta

    a = 1 - level
    lo = beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def nominal_interval(n: int, p: float = 0.05, level: float = 0.95) -> tuple[float, float]:
    """Central binomial interval for the count of alarms out of ``n`` slices
    when the true rate is ``p``, expressed as rates."""
    from scipy.stats import binom

    a = 1 - level
    return float(binom.ppf(a / 2, n, p) / n), float(binom.ppf(1 - a / 2, n, p) / n)


def prepare_population(df: pd.DataFrame, train_frac: float = 0.3, seed: int = 0) -> pd.DataFrame:
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train = int(len(df) * train_frac)
    features = [c for c in df.columns if c != TARGET]
    models = train_models(df.iloc[:n_train], features, seed=seed)
    return score(models, df.iloc[n_train:]).reset_index(drop=True)


def evaluate_replicate(ref: Reference, cur: Sample, null) -> dict:
    drift = ref.drift_metrics(cur)
    cal = ref.calibration_metrics(cur)
    names = list(drift) + ["hl", "oe"]
    p = [null.pvalue(k, v) for k, v in drift.items()] + [cal["hl_p"], cal["oe_p"]]
    _, reject = st.holm(p, ref.config.alpha)
    rot = [_rot(k, v) for k, v in drift.items()] + [cal["hl_p"] < RULE_OF_THUMB["hl_p"], _rot("oe", cal["oe"])]
    psi = drift["psi:score"]
    return {
        "psi_score": psi,
        "psi_rot_010": psi >= RULE_OF_THUMB["psi_amber"],
        "psi_rot_025": psi >= RULE_OF_THUMB["psi_red"],
        "psi_calibrated": psi > null.threshold("psi:score", ref.config.alpha),
        "model_rot": bool(np.any(rot)),
        "model_holm": bool(np.any(reject)),
        "model_unadjusted": bool(np.nanmin(np.where(np.isnan(p), 1.0, p)) <= ref.config.alpha),
        "holm_metrics": ";".join(n for n, r in zip(names, reject) if r),
        "rot_metrics": ";".join(n for n, r in zip(names, rot) if r),
    }


def _sample(df: pd.DataFrame) -> Sample:
    return Sample.from_frame(df, SCORE_COL, TARGET, FEATURES)


def run_cell(args) -> list[dict]:
    """One slice size: build the null, then run no-drift and drift replicates."""
    pop, canon_idx, n, ref_size, n_null, n_reps, n_power, grid, seed = args
    cfg = Config(n_null=n_null, seed=seed)
    canonical = Reference(_sample(pop.iloc[canon_idx]), cfg)
    null = canonical.null(n)
    rng = np.random.default_rng([seed, n, 1])
    rows = []
    jobs = [("none", 0.0, n_reps)] + [(k, d, n_power) for k, ds in grid.items() for d in ds]
    for kind, delta, reps in jobs:
        for r in range(reps):
            ref_df = pop.iloc[rng.integers(0, len(pop), ref_size)]
            if kind == "none":
                cur_df = pop.iloc[rng.integers(0, len(pop), n)]
            else:
                cur_df = DRIFTS[kind](pop, n, delta, rng)
            ref = Reference(_sample(ref_df), cfg)
            out = evaluate_replicate(ref, _sample(cur_df), null)
            out.update(slice_size=n, drift=kind, magnitude=delta, rep=r)
            rows.append(out)
    return rows


DEFAULT_GRID = {
    "covariate": [0.05, 0.1, 0.2, 0.4],
    "label": [0.05, 0.1, 0.2, 0.4],
    "calibration": [0.01, 0.02, 0.05, 0.1],
}


def run(sizes=(500, 1000, 2000, 5000, 10000, 20000), ref_size: int = 10000, n_null: int = 1000,
        n_reps: int = 1000, n_power: int = 200, grid=None, workers: int = 3, seed: int = 0,
        data_path: str = "data/uci_default.csv") -> pd.DataFrame:
    grid = DEFAULT_GRID if grid is None else grid
    pop = prepare_population(load_uci(data_path), seed=seed)
    canon_idx = np.random.default_rng(seed).choice(len(pop), ref_size, replace=False)
    tasks = [(pop, canon_idx, n, ref_size, n_null, n_reps, n_power, grid, seed) for n in sizes]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(run_cell, tasks):
            rows.extend(part)
    return pd.DataFrame(rows)


def summarize_false_alarms(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, g in raw[raw.drift == "none"].groupby("slice_size"):
        reps = len(g)
        lo, hi = nominal_interval(reps)
        row = {"slice_size": n, "slices": reps, "nominal_low": lo, "nominal_high": hi,
               "median_psi_score": g.psi_score.median()}
        for col in ["psi_rot_010", "psi_rot_025", "psi_calibrated", "model_rot", "model_unadjusted", "model_holm"]:
            row[col] = g[col].mean()
        k = int(g.psi_calibrated.sum())
        row["psi_calibrated_ci_low"], row["psi_calibrated_ci_high"] = binom_ci(k, reps)
        row["psi_calibrated_in_interval"] = lo <= g.psi_calibrated.mean() <= hi
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_power(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw[raw.drift != "none"]
    return (d.groupby(["drift", "magnitude", "slice_size"])
             .agg(slices=("rep", "size"), psi_rot_025=("psi_rot_025", "mean"),
                  psi_calibrated=("psi_calibrated", "mean"), model_rot=("model_rot", "mean"),
                  model_holm=("model_holm", "mean"))
             .reset_index())
