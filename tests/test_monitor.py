import numpy as np
import pandas as pd
import pytest

from e23mon.data import FEATURES, TARGET, calibration_drift, covariate_drift, label_drift
from e23mon.monitor import Config, Reference, Sample, evaluate_slice, summarize
from e23mon.simulate import nominal_interval

FEATS = ["LIMIT_BAL", "SEX", "EDUCATION", "PAY_0", "AGE"]


def _sample(df, score_col="pd_gbm"):
    return Sample.from_frame(df, score_col, TARGET, FEATS)


@pytest.fixture(scope="module")
def reference(scored_population):
    return Reference(_sample(scored_population.iloc[:3000]), Config(n_null=200, n_boot=50))


def test_no_drift_slice_is_quiet(reference, scored_population):
    cur = _sample(scored_population.iloc[3000:5000])
    df = evaluate_slice(reference, cur)
    assert set(df.metric) >= {"psi:score", "csi:LIMIT_BAL", "auc_drop", "ks_drop", "hl", "oe"}
    assert not summarize(df)["calibrated_alarm"]
    lo, hi = df.loc[df.metric == "auc_drop", ["ci_low", "ci_high"]].iloc[0]
    assert lo < 0 < hi


@pytest.mark.parametrize("drift,delta,expected", [
    (covariate_drift, 0.6, "psi:score"),
    (label_drift, 0.5, "oe"),
    (calibration_drift, 0.1, "auc_drop"),
])
def test_injected_drift_is_detected(reference, scored_population, drift, delta, expected):
    pool = scored_population.iloc[3000:].reset_index(drop=True)
    cur = _sample(drift(pool, 2000, delta, np.random.default_rng(0)))
    df = evaluate_slice(reference, cur, with_ci=False)
    assert df.set_index("metric").loc[expected, "calibrated_breach"]


def test_single_class_slice_is_flagged_untestable(reference, scored_population):
    cur_df = scored_population.iloc[3000:3400]
    cur_df = cur_df[cur_df[TARGET] == 0]
    df = evaluate_slice(reference, _sample(cur_df))
    row = df.set_index("metric")
    assert not row.loc["auc_drop", "testable"] and not row.loc["ks_drop", "testable"]
    assert "auc_drop" in summarize(df)["untestable"]
    # Calibration tests still work: zero defaults is a strong O/E signal.
    assert row.loc["oe", "calibrated_breach"]


def test_unseen_category_in_slice(reference, scored_population):
    cur_df = scored_population.iloc[3000:4000].copy()
    cur_df.loc[cur_df.index[:100], "EDUCATION"] = 9
    df = evaluate_slice(reference, _sample(cur_df), with_ci=False)
    assert df.set_index("metric").loc["csi:EDUCATION", "calibrated_breach"]


def test_null_thresholds_shrink_with_slice_size(reference):
    t_small = reference.null(500).threshold("psi:score", 0.05)
    t_large = reference.null(5000).threshold("psi:score", 0.05)
    assert t_small > t_large > 0
    # Far below the 0.10 / 0.25 rules of thumb at these sizes.
    assert t_small < 0.1


def test_calibrated_false_alarm_rate_within_binomial_interval(scored_population):
    """Seeded check: over 400 no-drift slices drawn from the population with a
    fresh reference each time, the calibrated 5% PSI threshold fires at a rate
    inside the 95% binomial interval of the nominal 5%."""
    cfg = Config(n_null=1000)
    feats = ["LIMIT_BAL"]
    pop = scored_population
    rng = np.random.default_rng(2024)
    m, n, reps = 2000, 500, 400
    canonical = Reference(Sample.from_frame(pop.iloc[:m], "pd_gbm", TARGET, feats), cfg)
    thr = canonical.null(n).threshold("psi:score", 0.05)
    alarms = 0
    for _ in range(reps):
        ref = Reference(Sample.from_frame(pop.iloc[rng.integers(0, len(pop), m)], "pd_gbm", TARGET, feats), cfg)
        cur = Sample.from_frame(pop.iloc[rng.integers(0, len(pop), n)], "pd_gbm", TARGET, feats)
        alarms += ref.drift_metrics(cur)["psi:score"] > thr
    lo, hi = nominal_interval(reps)
    assert lo <= alarms / reps <= hi, alarms / reps
