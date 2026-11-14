# Simulation MOBO Framework

A compact, simulator-agnostic framework for closed-loop multi-objective Bayesian
optimization. It connects bounded design sampling, a replaceable evaluation
function, Gaussian-process surrogates, batch candidate selection, and auditable
optimization records.

## Workflow

```text
Design space and feasibility rules
                 |
        Latin hypercube sampling
                 |
       Simulation / evaluation <-------------------+
                 |                                 |
       Validate and record objectives              |
                 |                                 |
       Fit Gaussian-process surrogates             |
                 |                                 |
       Select a batch using log-EHVI ---------------+
                 |
       Pareto set and run summary
```

## Installation

Python 3.10 or later is required. A virtual environment is recommended.

```bash
git clone https://github.com/LiuChenyangSHU/simulation-mobo-framework.git
cd simulation-mobo-framework
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## Quick Start

```bash
simulation-mobo --output runs/demo --rounds 3 --seed 42
```

The analytic example optimizes two competing quadratic objectives on the unit
square. The default run evaluates 12 initial designs and three batches of two
designs, for a total of 18 evaluations. Use a new output directory for each run.

The output directory contains:

| File | Description |
| --- | --- |
| `config.json` | Bounds, objective order, reference point, random seed, and budgets |
| `observations.jsonl` | Append-only records of validated evaluations |
| `summary.json` | Stop reason, evaluation count, Pareto indices, and hypervolume history |

## Connect an Evaluation Module

The computational interface is a Python callable. It receives named physical
parameters and returns named scalar objectives. Geometry generation, solver
execution, and response extraction can be encapsulated behind this interface.

```python
from simulation_mobo import DesignSpace, Workflow, WorkflowConfig

def evaluate(parameters: dict[str, float]) -> dict[str, float]:
    x, y = parameters["x"], parameters["y"]
    return {
        "performance": -(x**2 + y**2),
        "negative_cost": -((1 - x)**2 + (1 - y)**2),
    }

workflow = Workflow(
    space=DesignSpace({"x": (0.0, 1.0), "y": (0.0, 1.0)}),
    objectives=["performance", "negative_cost"],
    reference=[-2.1, -2.1],
    evaluate=evaluate,
    config=WorkflowConfig(initial_samples=12, batch_size=2, rounds=3, seed=42),
    feasible=lambda p: p["x"] + p["y"] <= 1.5,
)
result = workflow.run("runs/custom")
```

**All objectives are maximized.** Negate a quantity that should be minimized.
Choose a fixed reference point in the same transformed objective units, below
the region of interest. Objective names returned by the evaluation module must
match the configured names exactly; values must be finite.

## Method

- Inputs are mapped to the unit hypercube using their configured bounds.
- Initial designs and candidate pools use SciPy Latin hypercube sampling.
  A feasibility predicate filters the sampled designs.
- BoTorch `SingleTaskGP` fits independent standardized objective outputs with
  exact marginal likelihood optimization, using double precision on the CPU.
- A finite candidate pool is scored using Monte Carlo log-stabilized expected
  hypervolume improvement (`qLogExpectedHypervolumeImprovement`). A batch is
  selected greedily with pending-point conditioning.
- Previously evaluated points and duplicate pool entries are excluded from
  recommendation. Pareto membership and hypervolume are computed by BoTorch.

The workflow stops after the configured recommendation-round budget, or after
an explicitly configured target is satisfied by a single evaluated design.
Evaluation failures raise an exception; earlier validated observations remain
available in the run directory.

The optional `WindowPolicy` utility supports staged search intervals and
plateau-triggered updates. It is caller-controlled and is not enabled in the
default workflow. See [Methods](docs/methods.md) for its exact rules and the
response-curve metric definitions.

## Tests

```bash
python -m pytest -q
```

Tests cover feasibility sampling, normalization, interpolated response metrics,
window updates, GP-based recommendation, output validation, and stopping rules.
Dependency versions are listed in [requirements-tested.txt](requirements-tested.txt).

## Citation

For manuscript citations, cite the versioned release rather than an unversioned
branch. Software metadata is available in [CITATION.cff](CITATION.cff).

```bibtex
@software{simulation_mobo_framework_2026,
  author  = {{LiuChenyangSHU}},
  title   = {Simulation MOBO Framework},
  year    = {2026},
  version = {0.1.1},
  url     = {https://github.com/LiuChenyangSHU/simulation-mobo-framework}
}
```

## License

MIT. See [LICENSE](LICENSE).
