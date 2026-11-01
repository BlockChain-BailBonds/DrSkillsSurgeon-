"""Weight, spectral, and activation descriptors computed on identical probes.

Only the training dataset and model parameters may vary between skill labels;
feature extraction uses an independent, fixed-seed, task-agnostic input array.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .tasks import fixed_inputs


def _moments(values: np.ndarray, name: str) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).ravel()
    if not v.size or not np.isfinite(v).all():
        raise ValueError(f"{name} contains empty or nonfinite values")
    return {
        f"{name}.mean": float(v.mean()),
        f"{name}.std": float(v.std()),
        f"{name}.absmean": float(np.abs(v).mean()),
        f"{name}.rms": float(np.sqrt(np.mean(v * v))),
        f"{name}.q10": float(np.quantile(v, .10)),
        f"{name}.q50": float(np.quantile(v, .50)),
        f"{name}.q90": float(np.quantile(v, .90)),
    }


def weight_features(model) -> dict[str, float]:
    result: dict[str, float] = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            a = parameter.detach().float().cpu().numpy()
            tag = name.replace(".", "_")
            result.update(_moments(a, f"raw.{tag}"))
            if a.ndim != 2:
                continue
            # Keep raw statistics and normalized spectra separate: normalizing
            # singular values deliberately erases the absolute weight scale.
            s = np.linalg.svd(a.astype(np.float64), compute_uv=False)
            norm = s / max(float(s[0]), 1e-12)
            result.update(_moments(norm, f"spectral.{tag}"))
            for i in range(8):
                result[f"spectral.{tag}.sv{i}"] = float(norm[i]) if i < len(norm) else 0.0
            p = s / max(float(s.sum()), 1e-12)
            entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
            result[f"spectral.{tag}.effective_rank"] = float(math.exp(entropy))
            result[f"spectral.{tag}.fro_norm"] = float(np.linalg.norm(s))
            result[f"spectral.{tag}.nuclear_norm"] = float(s.sum())
    return result


def activation_features(model, device: torch.device, *, seed: int = 14791) -> dict[str, float]:
    result: dict[str, float] = {}
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        x = fixed_inputs(seed=seed).to(device)
        _, activations = model(x, capture=True)
        for site, t in activations.items():
            a = t.float().cpu().numpy()
            result.update(_moments(a, f"activation.{site}"))
            result[f"activation.{site}.nonzero_fraction"] = float((a > 0).mean())
            neuron_means = a.mean(axis=0)
            neuron_stds = a.std(axis=0)
            for i, q in enumerate((0., .1, .25, .5, .75, .9, 1.)):
                result[f"activation.{site}.neuron_mean_q{i}"] = float(np.quantile(neuron_means, q))
                result[f"activation.{site}.neuron_std_q{i}"] = float(np.quantile(neuron_stds, q))
    model.train(was_training)
    return result


def describe_model(model, device: torch.device) -> dict[str, float]:
    return {**weight_features(model), **activation_features(model, device)}
