# Changelog

## 0.1.0 - 2026-11-13

- Bounded design spaces, unit-cube normalization, and feasible Latin hypercube sampling.
- A callable simulation interface with strict objective-schema and finiteness checks.
- Independent-output Gaussian-process modeling and greedy batch log-EHVI selection.
- Append-only evaluation records, Pareto summaries, and explicit stopping budgets.
- Optional staged-window policy and linearly interpolated response-curve metrics.
- An executable analytic example, English method documentation, and citation metadata.

Local validation: 25 tests passed on Windows with Python 3.13.9 and the numerical
versions in `requirements-tested.txt`. The three-round analytic example completed
18 evaluations. Two dependency-level PyTorch deprecation warnings were emitted
during testing. These checks establish software behavior for the tested cases,
not validation of a particular physical model.
