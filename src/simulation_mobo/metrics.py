"""Metrics for sampled scalar response curves."""

import numpy as np


def threshold_metrics(frequency, response, threshold=-10.0):
    """Sum below-threshold intervals using linear interpolation at crossings.

    Bandwidth uses the input frequency unit. Outside excess is a sample-wise
    arithmetic mean, not a frequency-weighted integral.
    """
    x, y = np.asarray(frequency, dtype=float), np.asarray(response, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError("A curve requires at least two paired one-dimensional samples.")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(threshold):
        raise ValueError("Curve values and threshold must be finite.")
    order = np.argsort(x)
    x, y = x[order], y[order] - threshold
    if np.any(np.diff(x) <= 0):
        raise ValueError("Frequency samples must be distinct.")
    bandwidth = 0.0
    for left, right, a, b in zip(x[:-1], x[1:], y[:-1], y[1:]):
        width = right - left
        if a <= 0 and b <= 0:
            bandwidth += width
        elif (a < 0) != (b < 0):
            bandwidth += width * (abs(a) if a < 0 else abs(b)) / abs(b - a)
    outside = y[y > 0]
    return {"bandwidth": float(bandwidth), "outside_excess": float(outside.mean()) if len(outside) else 0.0}
