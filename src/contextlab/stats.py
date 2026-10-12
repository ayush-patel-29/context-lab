import numpy as np


def bootstrap_ci(values, n_boot=1000, seed=0, alpha=0.05):
    """Resample queries with replacement; return (mean, lower, upper)."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a, b, **kw):
    """CI for mean(a - b) where a[i], b[i] are the same query. Tighter than comparing two separate CIs."""
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), **kw)