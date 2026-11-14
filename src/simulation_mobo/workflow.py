"""Sampling, simulation, validation, recommendation, and append-only records."""

from collections.abc import Callable
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

from .optimizer import recommend
from .space import DesignSpace, Feasibility

Evaluator = Callable[[dict[str, float]], dict[str, float]]


@dataclass(frozen=True)
class WorkflowConfig:
    initial_samples: int = 12
    batch_size: int = 2
    rounds: int = 3
    candidate_count: int = 256
    mc_samples: int = 64
    seed: int = 42
    target: dict[str, float] | None = None

    def __post_init__(self):
        if self.initial_samples < 2 or self.batch_size < 1 or self.rounds < 0:
            raise ValueError("Need at least two initial samples, a positive batch, and nonnegative rounds.")
        if self.candidate_count < self.batch_size or self.mc_samples < 1:
            raise ValueError("Candidate pool must cover the batch; MC sample count must be positive.")


class Workflow:
    def __init__(self, space: DesignSpace, objectives, reference, evaluate: Evaluator,
                 config: WorkflowConfig | None = None, feasible: Feasibility | None = None):
        self.space, self.objectives = space, tuple(objectives)
        self.reference = np.asarray(reference, dtype=float)
        self.evaluate, self.config, self.feasible = evaluate, config or WorkflowConfig(), feasible
        if not self.objectives or len(set(self.objectives)) != len(self.objectives):
            raise ValueError("Objective names must be nonempty and unique.")
        if self.reference.shape != (len(self.objectives),) or not np.isfinite(self.reference).all():
            raise ValueError("A finite reference value is required for each objective.")
        if self.config.target is not None:
            if not self.config.target or not set(self.config.target) <= set(self.objectives):
                raise ValueError("Targets must name existing objectives.")
            if not np.isfinite(list(self.config.target.values())).all():
                raise ValueError("Targets must be finite.")

    def run(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise FileExistsError("Output directory must be empty; choose a new run directory.")
        manifest = {"config": asdict(self.config), "bounds": dict(zip(self.space.names, self.space.bounds.tolist())),
                    "objectives": list(self.objectives), "reference": self.reference.tolist(), "schema_version": 1}
        (output / "config.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        x, y, history = [], [], []
        cfg = self.config
        designs = self.space.sample(cfg.initial_samples, cfg.seed, self.feasible)
        stop_reason = "round_budget"
        for round_index in range(cfg.rounds + 1):
            for row in designs:
                parameters = self.space.parameters(row)
                values = self.evaluate(parameters)
                if set(values) != set(self.objectives):
                    raise ValueError("Simulator output keys must match the objective names.")
                objectives = [float(values[name]) for name in self.objectives]
                if not np.isfinite(objectives).all():
                    raise ValueError("Simulator objectives must be finite.")
                record = {"evaluation": len(x), "round": round_index, "parameters": parameters,
                          "objectives": dict(zip(self.objectives, objectives))}
                with (output / "observations.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                x.append(row.tolist())
                y.append(objectives)
            ty = torch.tensor(y, dtype=torch.double)
            pareto = is_non_dominated(ty)
            hv = Hypervolume(torch.tensor(self.reference, dtype=torch.double)).compute(ty[pareto])
            history.append({"round": round_index, "evaluations": len(x), "hypervolume": hv})
            target = cfg.target
            if target and any(all(row[self.objectives.index(name)] >= value for name, value in target.items()) for row in y):
                stop_reason = "target_reached"
                break
            if round_index == cfg.rounds:
                break
            pool = self.space.sample(cfg.candidate_count, cfg.seed + round_index + 1, self.feasible)
            designs = self.space.decode(recommend(self.space.encode(x), y, self.space.encode(pool),
                                                  self.reference, cfg.batch_size, cfg.seed + round_index, cfg.mc_samples))
        result = {"stop_reason": stop_reason, "evaluations": len(x),
                  "pareto_indices": torch.where(pareto)[0].tolist(), "history": history}
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        return result
