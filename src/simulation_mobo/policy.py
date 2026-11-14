"""Optional staged search-window policy for a derived design quantity."""

from dataclasses import dataclass, replace
import math

import numpy as np


@dataclass(frozen=True)
class WindowState:
    stage: int = 0
    entered_round: int = 0
    frozen: bool = False


@dataclass(frozen=True)
class WindowPolicy:
    initial_lower: float = 5.3
    width: float = 5.0
    step: float = 2.0
    cap: float = 20.0
    target: float = 16.0
    max_rounds: int = 20
    plateau_rounds: int = 5
    progress_threshold: float = 0.3
    ehvi_relative_range: float = 0.4
    hv_relative_range: float = 0.08
    ehvi_low: float = 1.2e-3

    def __post_init__(self):
        values = [self.initial_lower, self.width, self.step, self.cap, self.target,
                  self.progress_threshold, self.ehvi_relative_range, self.hv_relative_range, self.ehvi_low]
        if not np.isfinite(values).all() or self.width <= 0 or self.step <= 0:
            raise ValueError("Window parameters must be finite with positive width and step.")
        if self.initial_lower + self.width > self.cap or self.plateau_rounds < 2 or self.max_rounds < 1:
            raise ValueError("Invalid window cap or round limits.")

    def bounds(self, state):
        lower = self.initial_lower + max(0, state.stage) * self.step
        upper = min(lower + self.width, self.cap)
        return upper - self.width, upper

    def plateau(self, history):
        if len(history) < self.plateau_rounds:
            return False
        recent = history[-self.plateau_rounds:]
        e, h, b = (np.array([r[key] for r in recent], dtype=float) for key in ("ehvi", "hypervolume", "best"))
        if not all(np.isfinite(a).all() for a in (e, h, b)):
            return False
        relative = lambda a: float(np.ptp(a) / max(abs(a.mean()), 1e-8))
        hv_stable = relative(h) <= self.hv_relative_range
        soft = e[-1] <= e[0] and np.sum(np.diff(e) > 0) <= max(1, self.plateau_rounds // 2)
        ehvi_stable = relative(e) <= self.ehvi_relative_range and (np.all(np.diff(e) <= 0) or (soft and hv_stable))
        low = e.max() <= self.ehvi_low and hv_stable
        return bool(np.ptp(b) < self.progress_threshold and (ehvi_stable or low))

    def update(self, state, round_index, best_value, history):
        if state.frozen:
            return state
        if self.plateau(history) or round_index - state.entered_round >= self.max_rounds:
            maximum = math.ceil((self.cap - self.initial_lower - self.width) / self.step)
            stage = min(state.stage + 1, maximum)
            if stage != state.stage:
                state = replace(state, stage=stage, entered_round=round_index)
        return replace(state, frozen=best_value >= self.target - 1e-6)
