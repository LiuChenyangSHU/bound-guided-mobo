# Methods and Interface Contract

## Design and Evaluation

A design is an ordered vector of bounded continuous parameters. The order is
the insertion order of the `DesignSpace` bounds mapping. An optional feasibility
predicate accepts a named-parameter dictionary. The sampling budget is finite;
an infeasible search space raises an error rather than looping indefinitely.

The evaluation module is responsible for translating a design into a valid
computation and returning scalar objectives. The optimizer does not interpret
solver completion as scientific acceptance. The module should raise an
exception when geometry, solver, or numerical-quality checks fail. Only finite,
schema-matching objective dictionaries enter the observation log.

Objectives should be scaled to meaningful comparable numerical ranges by the
evaluation module. The GP standardizes each output internally and transforms
its posterior back to the supplied units. The hypervolume reference point and
any stopping targets must use those supplied objective units. The reference
point remains fixed throughout a workflow run, making its recorded
hypervolumes comparable across rounds.

## Candidate Recommendation

For each recommendation round, a fresh feasible Latin hypercube pool is drawn.
The model is refitted to all completed observations. The acquisition uses
Sobol quasi-Monte Carlo posterior samples. Greedy selection adds each chosen
point to the pending set before choosing the next point. Candidate selection
is discrete and approximate; it does not certify a global acquisition maximum
or a globally optimal design.

`WorkflowConfig.rounds` counts recommendation batches after initialization.
Without an early target, the evaluation count is
`initial_samples + rounds * batch_size`. When targets are supplied, one
evaluated design must meet all named targets. The check occurs after completing
each batch, including initialization. A target is optional and independent of
the staged-window freeze rule below.

## Optional Staged Search Window

`WindowPolicy` is a small state-transition utility for a derived design quantity
such as total thickness. It does not impose a physical model. The caller owns
the state, objective history, and feasibility predicate; the standard workflow
does not invoke the policy automatically.

Default settings are an initial interval of 5.3-10.3, a width of 5, a stage step
of 2, and an upper cap of 20, in caller-defined units. A stage advances by one
after 20 rounds in that stage or after a plateau. The terminal interval is
15-20. For the default five-entry history window, define
`relative_range(z) = (max(z) - min(z)) / max(abs(mean(z)), 1e-8)`.

A plateau requires the range of the recorded best objective to be below 0.3
and either of the following:

1. EHVI relative range is at most 0.40, and EHVI is non-increasing; alternatively,
   the last EHVI is no larger than the first, at most two adjacent increases
   occur, and HV relative range is at most 0.08.
2. Maximum EHVI is at most 0.0012 and HV relative range is at most 0.08.

After a possible stage update, a best value of at least `target - 1e-6` freezes
the interval. The default target is 16. Freezing does not stop optimization.
An objective improvement alone does not advance the interval.

```python
from simulation_mobo.policy import WindowPolicy, WindowState

policy = WindowPolicy()
state = WindowState()
history = []
state = policy.update(state, round_index=20, best_value=12.0, history=history)
lower, upper = policy.bounds(state)  # 7.3, 12.3
```

History entries have the keys `ehvi`, `hypervolume`, and `best`. The `ehvi` field
must contain ordinary, nonnegative EHVI values, **not log-EHVI scores**. The
log-EHVI acquisition used by the generic optimizer is not automatically fed to
this utility. Comparisons across changing reference points or search domains
require separate interpretation. Plateau thresholds are heuristic settings,
not a statistical convergence guarantee.

## Sampled Response Metrics

`threshold_metrics(frequency, response, threshold=-10)` sorts distinct frequency
samples and interpolates threshold crossings linearly. It returns:

- `bandwidth`: the sum of all below-threshold intervals within the sampled
  frequency span. Disconnected intervals are added; this is not the longest
  contiguous bandwidth.
- `outside_excess`: the arithmetic mean of `response - threshold` at sampled
  points above threshold, or zero if no such points exist. It is not an
  integrated or frequency-weighted measure.

Frequency units are preserved. The caller must validate frequency coverage,
resolution, and the physical meaning of the response. Finite curve samples
alone do not establish adequate spectral resolution.

## Run Records

Observation indices are zero-based and identify the corresponding entries in
`observations.jsonl`. Round zero denotes initialization. The final Pareto
indices refer to these observation indices. The run directory must be empty
at startup to prevent accidental overwriting or mixing of experiments.
Automatic resume and parallel evaluation are not implemented.
