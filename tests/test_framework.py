import json

import numpy as np
import pytest

from simulation_mobo import DesignSpace, Workflow, WorkflowConfig
from simulation_mobo.demo import evaluate
from simulation_mobo.metrics import threshold_metrics
from simulation_mobo.optimizer import recommend
from simulation_mobo.policy import WindowPolicy, WindowState


def test_sampling_is_reproducible_and_feasible():
    space = DesignSpace({"x": (0.0, 1.0), "y": (2.0, 3.0)})
    a = space.sample(12, seed=7, feasible=lambda p: p["x"] < 0.4)
    assert a.shape == (12, 2)
    assert np.all(a[:, 0] < 0.4)
    np.testing.assert_array_equal(a, space.sample(12, seed=7, feasible=lambda p: p["x"] < 0.4))
    np.testing.assert_allclose(space.decode(space.encode(a)), a)


def test_impossible_constraint_fails():
    with pytest.raises(ValueError, match="feasible"):
        DesignSpace({"x": (0, 1)}).sample(2, feasible=lambda p: False, max_attempts=2)


@pytest.mark.parametrize("bounds", [{}, {"x": (1, 1)}, {"x": (0, float("inf"))}])
def test_bad_bounds_rejected(bounds):
    with pytest.raises(ValueError):
        DesignSpace(bounds)


def test_threshold_metric_sums_disconnected_bands():
    result = threshold_metrics([0, 1, 2, 3, 4], [0, -20, 0, -20, 0], threshold=-10)
    assert result["bandwidth"] == pytest.approx(2.0)
    assert result["outside_excess"] == pytest.approx(10.0)


@pytest.mark.parametrize("x,y", [([0, 0], [1, 2]), ([0, 1], [1, np.nan]), ([0], [1])])
def test_invalid_curve_rejected(x, y):
    with pytest.raises(ValueError):
        threshold_metrics(x, y)


def test_window_advances_at_budget_and_freezes_at_target():
    policy = WindowPolicy(initial_lower=5.3, width=5, step=2, cap=20, target=16)
    state = WindowState()
    state = policy.update(state, round_index=20, best_value=12, history=[])
    assert policy.bounds(state) == pytest.approx((7.3, 12.3))
    state = policy.update(state, round_index=21, best_value=16, history=[])
    assert state.frozen
    assert policy.update(state, round_index=99, best_value=16, history=[]).stage == 1


def test_window_plateau_and_cap():
    policy = WindowPolicy()
    history = [{"ehvi": 0.001, "hypervolume": 1, "best": 12} for _ in range(5)]
    state = policy.update(WindowState(), round_index=5, best_value=12, history=history)
    assert state.stage == 1
    assert policy.bounds(WindowState(stage=100)) == pytest.approx((15, 20))


def test_progress_alone_does_not_advance_window():
    state = WindowPolicy().update(WindowState(), round_index=1, best_value=15, history=[])
    assert state.stage == 0


def test_recommendation_has_correct_shape_and_avoids_observed_points():
    space = DesignSpace({"x": (0, 1), "y": (0, 1)})
    x = space.sample(6, seed=3)
    y = np.array([list(evaluate(space.parameters(row)).values()) for row in x])
    candidates = space.sample(16, seed=4)
    proposed = recommend(space.encode(x), y, space.encode(candidates), [-1.1, -1.1], 2, seed=5, mc_samples=8)
    assert proposed.shape == (2, 2)
    assert len(np.unique(proposed, axis=0)) == 2
    assert not np.any(np.all(np.isclose(proposed[:, None], space.encode(x)[None]), axis=2))


def test_workflow_records_validated_evaluations_and_budget(tmp_path):
    cfg = WorkflowConfig(initial_samples=6, batch_size=2, rounds=1, candidate_count=16, mc_samples=8)
    result = Workflow(DesignSpace({"x": (0, 1), "y": (0, 1)}), ["quality_a", "quality_b"], [-1.1, -1.1], evaluate, cfg).run(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "observations.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert result["stop_reason"] == "round_budget"
    assert result["evaluations"] == 8
    assert result["pareto_indices"]
    assert (tmp_path / "summary.json").exists()
    with pytest.raises(FileExistsError):
        Workflow(DesignSpace({"x": (0, 1), "y": (0, 1)}), ["quality_a", "quality_b"], [-1.1, -1.1], evaluate, cfg).run(tmp_path)


def test_invalid_simulation_results_are_not_recorded(tmp_path):
    workflow = Workflow(DesignSpace({"x": (0, 1)}), ["f"], [-2], lambda p: {"f": float("nan")}, WorkflowConfig(initial_samples=2, rounds=0))
    with pytest.raises(ValueError, match="finite"):
        workflow.run(tmp_path)
    assert not (tmp_path / "observations.jsonl").exists()


def test_target_stop_is_explicit(tmp_path):
    cfg = WorkflowConfig(initial_samples=2, rounds=2, target={"f": -1})
    result = Workflow(DesignSpace({"x": (0, 1)}), ["f"], [-2], lambda p: {"f": 0.0}, cfg).run(tmp_path)
    assert result["evaluations"] == 2
    assert result["stop_reason"] == "target_reached"


@pytest.mark.parametrize("kwargs", [{"initial_samples": 1}, {"rounds": -1}, {"batch_size": 0}, {"candidate_count": 1}, {"mc_samples": 0}])
def test_invalid_budgets_rejected(kwargs):
    with pytest.raises(ValueError):
        WorkflowConfig(**kwargs)


def test_targets_must_be_met_by_same_design(tmp_path):
    def competing(p):
        return {"a": p["x"], "b": 1 - p["x"]}
    workflow = Workflow(DesignSpace({"x": (0, 1)}), ["a", "b"], [-1, -1], competing,
                        WorkflowConfig(initial_samples=6, rounds=0, target={"a": 0.8, "b": 0.8}))
    assert workflow.run(tmp_path)["stop_reason"] == "round_budget"


def test_wrong_simulator_schema_rejected(tmp_path):
    workflow = Workflow(DesignSpace({"x": (0, 1)}), ["a", "b"], [-1, -1], lambda p: {"wrong": 0}, WorkflowConfig(rounds=0))
    with pytest.raises(ValueError, match="keys"):
        workflow.run(tmp_path)


def test_duplicate_candidate_pool_rejected():
    with pytest.raises(ValueError, match="distinct"):
        recommend([[0], [1]], [[0, 1], [1, 0]], [[0], [0], [1]], [-1, -1], 1)


def test_same_seed_reproduces_run(tmp_path):
    cfg = WorkflowConfig(initial_samples=4, rounds=0)
    workflow = Workflow(DesignSpace({"x": (0, 1), "y": (0, 1)}), ["quality_a", "quality_b"], [-1.1, -1.1], evaluate, cfg)
    assert workflow.run(tmp_path / "a") == workflow.run(tmp_path / "b")
    assert (tmp_path / "a" / "observations.jsonl").read_bytes() == (tmp_path / "b" / "observations.jsonl").read_bytes()
