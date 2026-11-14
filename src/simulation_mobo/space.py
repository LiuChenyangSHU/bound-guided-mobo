"""Bounded continuous designs and feasible Latin hypercube sampling."""

from collections.abc import Callable, Mapping

import numpy as np
from scipy.stats import qmc

Feasibility = Callable[[dict[str, float]], bool]


class DesignSpace:
    def __init__(self, bounds: Mapping[str, tuple[float, float]]):
        self.names = tuple(bounds)
        self.bounds = np.asarray(list(bounds.values()), dtype=float)
        if not self.names or self.bounds.shape != (len(self.names), 2):
            raise ValueError("At least one named lower/upper bound is required.")
        if not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Bounds must be finite with lower < upper.")

    def parameters(self, row):
        return dict(zip(self.names, map(float, row), strict=True))

    def encode(self, x):
        return (np.asarray(x) - self.bounds[:, 0]) / np.diff(self.bounds, axis=1).ravel()

    def decode(self, x):
        return self.bounds[:, 0] + np.asarray(x) * np.diff(self.bounds, axis=1).ravel()

    def sample(self, n, seed=0, feasible: Feasibility | None = None, max_attempts=100):
        if n < 1:
            raise ValueError("Sample count must be positive.")
        sampler = qmc.LatinHypercube(d=len(self.names), seed=seed)
        accepted = []
        for _ in range(max_attempts):
            points = self.decode(sampler.random(n))
            accepted.extend(row for row in points if feasible is None or feasible(self.parameters(row)))
            if len(accepted) >= n:
                return np.asarray(accepted[:n])
        raise ValueError("Could not obtain enough feasible designs within the sampling budget.")
