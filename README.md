# Bound-Guided MOBO

**Physics-informed search domains for autonomous simulation-driven design.**

Bound-Guided MOBO is a modular framework for multi-objective design discovery
with computationally expensive evaluations. It organizes bound-informed design
spaces, Gaussian-process surrogates, batch Bayesian recommendation, and
feedback-driven search-window policies around a replaceable simulation module.

The central idea is to use physical knowledge to choose where the search begins,
then use optimization feedback to decide when a search window should expand.
This connects physically motivated initialization with staged exploration and
closed-loop evaluation.

## Core Ideas

- **Start from physical knowledge.** Translate a physical bound or engineering
  constraint into an initial design interval and feasible parameter space.
- **Learn from each evaluation.** Update Gaussian-process surrogates and select
  batches through expected hypervolume improvement to balance competing objectives.
- **Expand with feedback.** Use the staged-window policy to advance the search
  interval when improvement plateaus or a stage reaches its evaluation-round budget.
- **Keep the computational module replaceable.** Connect geometry generation,
  numerical evaluation, and objective extraction through one callable interface.

For broadband absorber design, a thickness-bandwidth bound such as the Rozanov
bound can motivate the initial thickness interval. The framework accepts that
interval as a design-space input, while the window policy supplies the staged
expansion logic. The same organization applies to other bound-informed design
problems with competing performance and resource objectives.

## Workflow

```text
Physical bounds and design knowledge
                 |
Initial design space and feasibility rules
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

For adaptive-domain studies, the caller connects optimization history to
`WindowPolicy` and applies the resulting interval to subsequent candidate
sampling. The standard workflow provides the fixed-domain optimization loop;
the policy provides the domain-update component.

## Installation

Python 3.10 or later is required. A virtual environment is recommended.

```bash
git clone https://github.com/LiuChenyangSHU/bound-guided-mobo.git
cd bound-guided-mobo
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

## Quick Start

```bash
bound-guided-mobo --output runs/demo --rounds 3 --seed 42
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

## License

MIT. See [LICENSE](LICENSE).
