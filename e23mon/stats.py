"""Monitoring statistics: PSI/CSI, discrimination (AUC, KS) and calibration tests.

All functions are plain numpy so they can be called many thousands of times in the
simulation study without pandas overhead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sps

# Pseudo-count added to every bin before converting counts to proportions.
# It keeps PSI finite when a bin is empty in either sample and is reproduced
# exactly by the Excel formulas in the evidence pack.
DEFAULT_SMOOTHING = 0.5
MISSING = "__MISSING__"
OTHER = "__OTHER__"


@dataclass
class Binning:
    """Bin definition fitted on the reference sample.

    Numeric binning uses interior quantile edges; categorical binning uses the
    reference categories plus an OTHER bucket that receives unseen categories.
    A MISSING bin is added when the reference contains missing values; missing
    values in a slice with no such bin are counted in it as well (the bin is
    created on demand by ``counts``)."""

    kind: str  # "numeric" or "categorical"
    edges: np.ndarray | None = None  # interior edges for numeric
    categories: list = field(default_factory=list)

    def labels(self) -> list[str]:
        if self.kind == "numeric":
            e = self.edges
            bounds = [-np.inf, *e, np.inf]
            return [f"({_fmt(bounds[i])}, {_fmt(bounds[i + 1])}]" for i in range(len(bounds) - 1)] + [MISSING]
        return [str(c) for c in self.categories] + [OTHER, MISSING]

    def counts(self, values) -> np.ndarray:
        """Count values per bin. The last bin is always MISSING (often zero)."""
        if self.kind == "numeric":
            x = np.asarray(values, dtype=float)
            miss = np.isnan(x)
            idx = np.searchsorted(self.edges, x[~miss], side="left")
            c = np.bincount(idx, minlength=len(self.edges) + 1).astype(float)
            return np.append(c, miss.sum())
        k = len(self.categories)
        x = np.asarray(values)
        if x.dtype.kind in "fiub" and all(isinstance(c, (int, float, np.number)) for c in self.categories):
            # Fast path for numeric codes: match against the sorted category list.
            xf = x.astype(float)
            miss = np.isnan(xf)
            cats = np.asarray(self.categories, dtype=float)
            v = xf[~miss]
            pos = np.clip(np.searchsorted(cats, v), 0, max(k - 1, 0))
            idx = np.where((k > 0) & (cats[pos] == v) if k else np.zeros(v.size, bool), pos, k)
            c = np.bincount(idx, minlength=k + 1).astype(float)
            return np.append(c, miss.sum())
        x = x.astype(object)
        miss = _is_missing(x)
        lookup = {c: i for i, c in enumerate(self.categories)}
        idx = np.fromiter((lookup.get(v, k) for v in x[~miss]), dtype=int, count=int((~miss).sum()))
        c = np.bincount(idx, minlength=k + 1).astype(float)
        return np.append(c, miss.sum())


def _fmt(v: float) -> str:
    if np.isinf(v):
        return "-inf" if v < 0 else "inf"
    return f"{v:.6g}"


def _is_missing(x: np.ndarray) -> np.ndarray:
    return np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in x], dtype=bool)


def fit_binning(reference, n_bins: int = 10, max_categories: int = 20, categorical: bool | None = None) -> Binning:
    """Fit bins on a reference sample.

    Numeric columns with at most ``max_categories`` distinct values are binned as
    categories, which avoids collapsed quantile edges on discrete features."""
    ref = np.asarray(reference)
    if categorical is None:
        if ref.dtype.kind in "fiub":
            vals = ref[~np.isnan(ref.astype(float))]
            categorical = np.unique(vals).size <= max_categories
        else:
            categorical = True
    if categorical:
        miss = _is_missing(ref.astype(object))
        cats = sorted(set(ref[~miss].tolist()), key=lambda v: (str(type(v)), v))
        return Binning(kind="categorical", categories=cats)
    x = ref.astype(float)
    x = x[~np.isnan(x)]
    qs = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = np.unique(qs)
    return Binning(kind="numeric", edges=edges)


def proportions(counts, smoothing: float = DEFAULT_SMOOTHING) -> np.ndarray:
    c = np.asarray(counts, dtype=float)
    return (c + smoothing) / (c.sum() + smoothing * c.size)


def psi_from_counts(ref_counts, cur_counts, smoothing: float = DEFAULT_SMOOTHING) -> float:
    """Population Stability Index from bin counts.

    PSI = sum_i (q_i - p_i) * ln(q_i / p_i), with p (reference) and q (current)
    smoothed by a pseudo-count. Bins that are empty in both samples are dropped
    so an unused MISSING or OTHER bin contributes nothing."""
    r = np.asarray(ref_counts, dtype=float)
    c = np.asarray(cur_counts, dtype=float)
    keep = (r + c) > 0
    p = proportions(r[keep], smoothing)
    q = proportions(c[keep], smoothing)
    return float(np.sum((q - p) * np.log(q / p)))


def psi(reference, current, n_bins: int = 10, smoothing: float = DEFAULT_SMOOTHING, binning: Binning | None = None) -> float:
    b = binning or fit_binning(reference, n_bins=n_bins)
    return psi_from_counts(b.counts(reference), b.counts(current), smoothing)


def auc(y, score) -> float:
    """Area under the ROC curve via the Mann-Whitney statistic (ties count 1/2).
    Returns NaN when the sample contains a single class."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=float)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = sps.rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ks(y, score) -> float:
    """Kolmogorov-Smirnov discrimination statistic: max |F_bad(s) - F_good(s)|.
    Returns NaN when the sample contains a single class."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=float)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]
    cum_bad = np.cumsum(y_sorted) / n1
    cum_good = np.cumsum(~y_sorted) / n0
    # Evaluate only at the last position of each tied score value.
    last = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    return float(np.max(np.abs(cum_bad[last] - cum_good[last])))


def bootstrap_ci(y, score, metric, n_boot: int = 200, level: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for a discrimination metric. Resamples drawing a
    single class are discarded; returns (nan, nan) if fewer than 20 remain."""
    y = np.asarray(y)
    s = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    n = y.size
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        v = metric(y[i], s[i])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 20:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def hosmer_lemeshow(y, pd_, n_groups: int = 10) -> tuple[float, float, int]:
    """Hosmer-Lemeshow-style calibration test for an externally fitted model.

    Observations are grouped by deciles of predicted PD; the statistic
    sum_g (O_g - E_g)^2 / (E_g (1 - E_g / n_g)) is compared with chi-square on
    ``g`` degrees of freedom (no parameters are estimated on the slice).
    Returns (statistic, p-value, groups)."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(pd_, dtype=float), 1e-12, 1 - 1e-12)
    order = np.argsort(p, kind="mergesort")
    groups = np.array_split(order, min(n_groups, p.size))
    stat = 0.0
    g_used = 0
    for g in groups:
        if g.size == 0:
            continue
        o = y[g].sum()
        e = p[g].sum()
        v = e * (1 - e / g.size)
        if v <= 0:
            continue
        stat += (o - e) ** 2 / v
        g_used += 1
    if g_used == 0:
        return float("nan"), float("nan"), 0
    return float(stat), float(sps.chi2.sf(stat, g_used)), g_used


def observed_expected(y, pd_) -> tuple[float, float, float]:
    """Observed-to-expected default ratio with a two-sided z-test.

    Under correct calibration the default count is Poisson-binomial with mean
    sum(p) and variance sum(p(1-p)). Returns (O/E ratio, z, p-value)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(pd_, dtype=float)
    o = y.sum()
    e = p.sum()
    v = np.sum(p * (1 - p))
    if e <= 0 or v <= 0:
        return float("nan"), float("nan"), float("nan")
    z = (o - e) / np.sqrt(v)
    return float(o / e), float(z), float(2 * sps.norm.sf(abs(z)))


def holm(pvalues, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm step-down adjustment. NaN p-values are ignored (not rejected).
    Returns (adjusted p-values, reject flags)."""
    p = np.asarray(pvalues, dtype=float)
    adj = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    m = pv.size
    if m:
        order = np.argsort(pv, kind="mergesort")
        scaled = (m - np.arange(m)) * pv[order]
        scaled = np.minimum(np.maximum.accumulate(scaled), 1.0)
        out = np.empty(m)
        out[order] = scaled
        adj[ok] = out
    reject = np.where(np.isnan(adj), False, adj <= alpha)
    return adj, reject


def hl_drift(ref_counts, ref_defaults, cur_counts, cur_defaults) -> tuple[float, float, int]:
    """Hosmer-Lemeshow-style calibration drift test on score bands.

    The expected default count of each band in the slice is the band size times
    the reference default rate of that band, r_g = (d_g + 0.5) / (n_g + 1), so a
    model that was already slightly off at development does not alarm forever.
    The variance n_g r_g (1 - r_g) (1 + n_g / N_g) includes the sampling error of
    the reference rate. Statistic ~ chi-square with one degree of freedom per
    non-empty band. Returns (statistic, p-value, bands)."""
    N = np.asarray(ref_counts, dtype=float)
    D = np.asarray(ref_defaults, dtype=float)
    n = np.asarray(cur_counts, dtype=float)
    o = np.asarray(cur_defaults, dtype=float)
    keep = (n > 0) & (N > 0)
    if not keep.any():
        return float("nan"), float("nan"), 0
    r = (D[keep] + 0.5) / (N[keep] + 1)
    e = n[keep] * r
    v = n[keep] * r * (1 - r) * (1 + n[keep] / N[keep])
    stat = float(np.sum((o[keep] - e) ** 2 / v))
    g = int(keep.sum())
    return stat, float(sps.chi2.sf(stat, g)), g


def oe_drift(ref_y, ref_pd, y, pd_) -> tuple[float, float, float]:
    """Test whether the slice observed-to-expected ratio differs from the
    reference ratio. Expected slice defaults are (O/E)_ref * sum(p); the
    variance adds the Poisson-binomial variance of the slice count and the
    sampling variance of (O/E)_ref. Returns (slice O/E, z, two-sided p)."""
    ref_y = np.asarray(ref_y, dtype=float)
    ref_pd = np.asarray(ref_pd, dtype=float)
    y = np.asarray(y, dtype=float)
    p = np.asarray(pd_, dtype=float)
    e_ref = ref_pd.sum()
    e = p.sum()
    if e_ref <= 0 or e <= 0:
        return float("nan"), float("nan"), float("nan")
    oe_ref = ref_y.sum() / e_ref
    var_oe_ref = np.sum(ref_pd * (1 - ref_pd)) * oe_ref / e_ref**2
    expected = oe_ref * e
    var = oe_ref * np.sum(p * (1 - p)) + e**2 * var_oe_ref
    if var <= 0:
        return float(y.sum() / e), float("nan"), float("nan")
    z = (y.sum() - expected) / np.sqrt(var)
    return float(y.sum() / e), float(z), float(2 * sps.norm.sf(abs(z)))
