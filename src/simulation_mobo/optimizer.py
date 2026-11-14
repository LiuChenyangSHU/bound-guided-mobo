"""Gaussian-process modeling and discrete-pool batch EHVI selection."""

import numpy as np
import torch
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood


def recommend(x, y, candidates, reference, batch_size, seed=0, mc_samples=64):
    """Return unit-cube designs; every objective is maximized.

    Select greedily from a finite pool, conditioning the batch acquisition on
    previously selected pending designs. The acquisition is log-stabilized EHVI.
    """
    x, y, candidates = [np.asarray(v, dtype=float) for v in (x, y, candidates)]
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
        raise ValueError("GP training requires at least two paired input/output rows.")
    if candidates.ndim != 2 or candidates.shape[1] != x.shape[1]:
        raise ValueError("Candidate and training dimensions must match.")
    if any(not np.isfinite(v).all() for v in (x, y, candidates, np.asarray(reference))):
        raise ValueError("Optimization values must be finite.")
    if len(reference) != y.shape[1] or batch_size < 1:
        raise ValueError("Reference dimension and positive batch size are required.")
    if np.any((x < 0) | (x > 1)) or np.any((candidates < 0) | (candidates > 1)):
        raise ValueError("Inputs must lie in the unit cube.")
    candidates = np.unique(candidates, axis=0)
    observed = np.any(np.all(np.isclose(candidates[:, None], x[None], rtol=0, atol=1e-10), axis=2), axis=1)
    candidates = candidates[~observed]
    if len(candidates) < batch_size:
        raise ValueError("Not enough distinct unobserved candidates.")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        tx, ty, pool = [torch.as_tensor(v, dtype=torch.double) for v in (x, y, candidates)]
        model = SingleTaskGP(tx, ty, outcome_transform=Standardize(m=ty.shape[1]))
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        partition = NondominatedPartitioning(ref_point=torch.tensor(reference, dtype=torch.double), Y=ty)
        acq = qLogExpectedHypervolumeImprovement(
            model=model, ref_point=list(reference), partitioning=partition,
            sampler=SobolQMCNormalSampler(torch.Size([mc_samples]), seed=seed),
        )
        selected = []
        for _ in range(batch_size):
            with torch.no_grad():
                scores = torch.cat([acq(chunk.unsqueeze(1)) for chunk in pool.split(256)])
            index = int(scores.argmax())
            selected.append(pool[index].clone())
            pool = torch.cat((pool[:index], pool[index + 1:]))
            acq.set_X_pending(torch.stack(selected))
        return torch.stack(selected).numpy()
