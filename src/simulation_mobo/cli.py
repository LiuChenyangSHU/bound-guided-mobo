"""Run a small analytic optimization example."""

import argparse
import json

import torch

from . import DesignSpace, Workflow, WorkflowConfig
from .demo import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/demo")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    workflow = Workflow(DesignSpace({"x": (0, 1), "y": (0, 1)}),
                        ["quality_a", "quality_b"], [-1.1, -1.1], evaluate,
                        WorkflowConfig(rounds=args.rounds, seed=args.seed))
    print(json.dumps(workflow.run(args.output), indent=2))


if __name__ == "__main__":
    main()
