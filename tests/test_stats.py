import math

import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from e23mon import stats as st


def test_psi_hand_computed_no_smoothing():
    # p = (.5, .3, .2), q = (.4, .4, .2)
    expected = (0.4 - 0.5) * math.log(0.4 / 0.5) + (0.4 - 0.3) * math.log(0.4 / 0.3)
    assert st.psi_from_counts([50, 30, 20], [40, 40, 20], smoothing=0) == pytest.approx(expected, abs=1e-12)
    assert expected == pytest.approx(0.0510826, abs=1e-6)


def test_psi_hand_computed_with_smoothing():
    p = np.array([50.5, 30.5, 20.5]) / 101.5
    q = np.array([40.5, 40.5, 20.5]) / 101.5
    expected = sum((qi - pi) * math.log(qi / pi) for pi, qi in zip(p, q))
    assert st.psi_from_counts([50, 30, 20], [40, 40, 20], smoothing=0.5) == pytest.approx(expected, abs=1e-12)


def test_psi_identical_is_zero_and_symmetric():
    assert st.psi_from_counts([10, 20, 30], [10, 20, 30]) == 0.0
    a = st.psi_from_counts([10, 20, 30], [30, 20, 10])
    b = st.psi_from_counts([30, 20, 10], [10, 20, 30])
    assert a == pytest.approx(b)


def test_psi_empty_bin_is_finite():
    # Without smoothing the empty reference bin would give infinity.
    v = st.psi_from_counts([100, 0], [50, 50], smoothing=0.5)
    p = np.array([100.5, 0.5]) / 101
    q = np.array([50.5, 50.5]) / 101
    assert np.isfinite(v)
    assert v == pytest.approx(np.sum((q - p) * np.log(q / p)), abs=1e-12)


def test_psi_bins_empty_in_both_samples_are_dropped():
    assert st.psi_from_counts([50, 30, 0, 20], [40, 40, 0, 20]) == pytest.approx(
        st.psi_from_counts([50, 30, 20], [40, 40, 20]))


def test_numeric_binning_quantiles_and_missing():
    ref = np.arange(1, 101, dtype=float)
    b = st.fit_binning(ref, n_bins=4)
    assert b.kind == "numeric"
    np.testing.assert_allclose(b.edges, [25.75, 50.5, 75.25])
    c = b.counts(np.array([1, 30, 60, 90, 1000, np.nan, -5]))
    np.testing.assert_array_equal(c, [2, 1, 1, 2, 1])  # last bin is MISSING
    assert len(b.labels()) == 5


def test_categorical_unseen_category_goes_to_other():
    b = st.fit_binning(np.array(["a", "b", "b", "c"], dtype=object))
    c = b.counts(np.array(["a", "z", "z", None, "c"], dtype=object))
    np.testing.assert_array_equal(c, [1, 0, 1, 2, 1])  # a, b, c, OTHER, MISSING
    v = st.psi(np.array(["a", "b", "b", "c"], dtype=object), np.array(["a", "z", "z", "c"], dtype=object))
    assert np.isfinite(v) and v > 0


def test_discrete_numeric_codes_binned_as_categories():
    ref = np.array([1, 2, 2, 3, 3, 3], dtype=float)
    b = st.fit_binning(ref)
    assert b.kind == "categorical"
    np.testing.assert_array_equal(b.counts(np.array([1, 2, 7, np.nan, 3.0])), [1, 1, 1, 1, 1])


def test_auc_and_ks_hand_computed():
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    # pairs (bad, good): (.35,.1) win, (.35,.4) loss, (.8,.1) win, (.8,.4) win -> 3/4
    assert st.auc(y, s) == pytest.approx(0.75)
    # CDFs at .1: bad 0 good .5; .35: .5/.5; .4: .5/1; .8: 1/1 -> max .5
    assert st.ks(y, s) == pytest.approx(0.5)


def test_auc_ties_count_half():
    assert st.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    assert st.ks([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)


def test_auc_ks_match_reference_implementations():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(y * 0.7, 1), 2)  # rounding creates ties
    assert st.auc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert st.ks(y, s) == pytest.approx(ks_2samp(s[y == 1], s[y == 0]).statistic, abs=1e-12)


def test_single_class_slice_gives_nan():
    y = np.zeros(50)
    s = np.linspace(0, 1, 50)
    assert np.isnan(st.auc(y, s))
    assert np.isnan(st.ks(y, s))
    assert all(np.isnan(st.bootstrap_ci(y, s, st.auc, n_boot=50)))


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 1000)
    s = rng.normal(y, 1)
    lo, hi = st.bootstrap_ci(y, s, st.auc, n_boot=200)
    assert lo < st.auc(y, s) < hi
    assert hi - lo < 0.1


def test_hosmer_lemeshow_hand_computed():
    y = np.array([0, 0, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    stat, pval, g = st.hosmer_lemeshow(y, p, n_groups=2)
    # group 1: p .1 .2 .3 -> E=.6, O=1, n=3 ; group 2: p .6 .7 .8 -> E=2.1, O=2
    e1, e2 = 0.6, 2.1
    expected = (1 - e1) ** 2 / (e1 * (1 - e1 / 3)) + (2 - e2) ** 2 / (e2 * (1 - e2 / 3))
    assert g == 2
    assert stat == pytest.approx(expected)
    from scipy.stats import chi2
    assert pval == pytest.approx(chi2.sf(expected, 2))


def test_observed_expected_hand_computed():
    y = np.array([1, 0, 0, 1])
    p = np.array([0.2, 0.2, 0.3, 0.3])
    oe, z, pval = st.observed_expected(y, p)
    assert oe == pytest.approx(2 / 1.0)
    assert z == pytest.approx((2 - 1.0) / math.sqrt(0.16 + 0.16 + 0.21 + 0.21))


def test_hl_drift_hand_computed():
    stat, pval, g = st.hl_drift([100, 100], [10, 30], [50, 50], [10, 15])
    r1, r2 = 10.5 / 101, 30.5 / 101
    e1, e2 = 50 * r1, 50 * r2
    v1 = 50 * r1 * (1 - r1) * 1.5
    v2 = 50 * r2 * (1 - r2) * 1.5
    assert g == 2
    assert stat == pytest.approx((10 - e1) ** 2 / v1 + (15 - e2) ** 2 / v2)


def test_oe_drift_zero_when_ratio_unchanged():
    p = np.full(100, 0.2)
    y = np.r_[np.ones(20), np.zeros(80)]
    oe, z, pval = st.oe_drift(y, p, y, p)
    assert oe == pytest.approx(1.0)
    assert z == pytest.approx(0.0)
    assert pval == pytest.approx(1.0)


def test_holm_hand_computed():
    adj, rej = st.holm([0.01, 0.04, 0.03, 0.005], alpha=0.05)
    # sorted .005 .01 .03 .04 -> 4*.005=.02, 3*.01=.03, 2*.03=.06, 1*.04=.04 -> monotone .06
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    np.testing.assert_array_equal(rej, [True, False, False, True])


def test_holm_ignores_nan():
    adj, rej = st.holm([0.01, np.nan], alpha=0.05)
    assert adj[0] == pytest.approx(0.01)
    assert np.isnan(adj[1]) and not rej[1]
