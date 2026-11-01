"""Held-out bidirectional activation interventions on a controlled toy model.

This is node-level patching, not edge-level path patching, skill extraction, or
unlearning. Ranking uses activation differences on disjoint calibration pairs.
"""
from __future__ import annotations

import numpy as np
import torch

from .tasks import LENGTH, INPUT_VOCAB, targets


def matched_pairs(model, task: str, device: torch.device, *, seed: int,
                  desired: int = 32, attempts: int = 25):
    """Select counterfactual pairs where one input change changes a target.

    For causal skill-expression evaluation, both target tokens must be predicted
    correctly. Return a smaller sample if the model is not sufficiently trained.
    """
    if desired < 1 or attempts < 1:
        raise ValueError("desired and attempts must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows: list[tuple[torch.Tensor, torch.Tensor, int, int, int]] = []
    model.eval()
    with torch.inference_mode():
        for _ in range(attempts):
            x = torch.randint(0, INPUT_VOCAB, (256, LENGTH), generator=generator)
            bad = x.clone()
            pos = torch.randint(0, LENGTH, (256,), generator=generator)
            diff = torch.randint(1, INPUT_VOCAB, (256,), generator=generator)
            row = torch.arange(256)
            bad[row, pos] = (bad[row, pos] + diff) % INPUT_VOCAB
            y1, mask = targets(task, x)
            y0, _ = targets(task, bad)
            changed = (y1 != y0) & mask
            eligible = changed.any(dim=1)
            output_pos = changed.to(torch.int32).argmax(dim=1)
            if not bool(eligible.any()):
                continue
            x_device, bad_device = x.to(device), bad.to(device)
            p1 = model(x_device).argmax(dim=-1).cpu()
            p0 = model(bad_device).argmax(dim=-1).cpu()
            correct = eligible & (p1[row, output_pos] == y1[row, output_pos]) & (
                p0[row, output_pos] == y0[row, output_pos])
            for idx in correct.nonzero(as_tuple=False).view(-1).tolist():
                p = int(output_pos[idx]); t1 = int(y1[idx, p]); t0 = int(y0[idx, p])
                rows.append((x[idx], bad[idx], p, t1, t0))
                if len(rows) == desired:
                    break
            if len(rows) == desired:
                break
    if not rows:
        return None
    return (
        torch.stack([r[0] for r in rows]).to(device),
        torch.stack([r[1] for r in rows]).to(device),
        torch.tensor([r[2] for r in rows], device=device, dtype=torch.long),
        torch.tensor([r[3] for r in rows], device=device, dtype=torch.long),
        torch.tensor([r[4] for r in rows], device=device, dtype=torch.long),
    )


def _margins(logits: torch.Tensor, pos: torch.Tensor,
             true: torch.Tensor, foil: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(pos.numel(), device=logits.device)
    return logits[rows, pos, true] - logits[rows, pos, foil]


def patch_sites(model, task: str, device: torch.device, *, seed: int,
                top_k: int = 12, minimum_pairs: int = 8) -> list[dict]:
    """Use calibration and evaluation pairs with different random seeds.

    Returns records for each activation site. Calibration determines only the
    candidate neuron indices. All reported effects are measured on held-out
    counterfactual pairs, and a matched-size random intervention is included.
    """
    if top_k < 1 or minimum_pairs < 1:
        raise ValueError("top_k and minimum_pairs must be positive")
    cal = matched_pairs(model, task, device, seed=seed + 300001, desired=32)
    test = matched_pairs(model, task, device, seed=seed + 700003, desired=48)
    records = []
    if cal is None or test is None or len(cal[0]) < minimum_pairs or len(test[0]) < minimum_pairs:
        return [{"task": task, "site": site, "status": "insufficient_correct_pairs",
                 "calibration_pairs": 0 if cal is None else len(cal[0]),
                 "test_pairs": 0 if test is None else len(test[0])}
                for site in ("hidden1", "hidden2")]

    cal_x, cal_corrupted = cal[0], cal[1]
    x, corrupted, output_pos, target, foil = test
    model.eval()
    with torch.inference_mode():
        _, cal_clean_act = model(cal_x, capture=True)
        _, cal_bad_act = model(cal_corrupted, capture=True)
        clean_logits, clean_act = model(x, capture=True)
        bad_logits, bad_act = model(corrupted, capture=True)
        m_clean = _margins(clean_logits, output_pos, target, foil)
        m_bad = _margins(bad_logits, output_pos, target, foil)
        # Since both outputs are correct, the difference should be positive,
        # but we preserve the signed denominator and report its magnitude.
        denominator = float((m_clean - m_bad).mean().item())
        for site in ("hidden1", "hidden2"):
            dim = clean_act[site].shape[1]
            k = min(top_k, max(1, dim // 2))
            differences = (cal_clean_act[site] - cal_bad_act[site]).abs().mean(dim=0)
            ranked = torch.argsort(differences, descending=True, stable=True)
            top = ranked[:k]
            remainder = ranked[k:]
            rng = np.random.default_rng(seed + (1 if site == "hidden1" else 2))
            random_indices = rng.choice(len(remainder), size=k, replace=False)
            random = remainder[torch.as_tensor(random_indices, device=device, dtype=torch.long)]

            full = model(corrupted, patch={"site": site, "values": clean_act[site]})
            top_logits = model(corrupted, patch={"site": site, "values": clean_act[site], "indices": top})
            random_logits = model(corrupted, patch={"site": site, "values": clean_act[site], "indices": random})
            reverse = model(x, patch={"site": site, "values": bad_act[site], "indices": top})
            m_full = _margins(full, output_pos, target, foil)
            m_top = _margins(top_logits, output_pos, target, foil)
            m_random = _margins(random_logits, output_pos, target, foil)
            m_reverse = _margins(reverse, output_pos, target, foil)
            full_abs_error = float((full - clean_logits).abs().max().item())
            if full_abs_error > 1e-4:
                raise RuntimeError(f"full patch at {site} failed architectural consistency test")
            # Paired uncertainty interval for the top-vs-random control effect.
            paired = (m_top - m_random).detach().cpu().numpy().astype(np.float64)
            bootstrap_rng = np.random.default_rng(seed + (3 if site == "hidden1" else 4))
            draws = bootstrap_rng.integers(0, len(paired), size=(1000, len(paired)))
            bootstrap = paired[draws].mean(axis=1)
            ci_low, ci_high = (float(value) for value in np.quantile(bootstrap, [.025, .975]))
            records.append({
                "task": task, "site": site, "status": "measured",
                "calibration_pairs": len(cal_x), "test_pairs": len(x), "top_k": k,
                "candidate_indices": top.cpu().tolist(),
                "control_indices": random.cpu().tolist(),
                "clean_margin": float(m_clean.mean().item()),
                "corrupted_margin": float(m_bad.mean().item()),
                "full_patch_margin": float(m_full.mean().item()),
                "top_patch_margin": float(m_top.mean().item()),
                "random_patch_margin": float(m_random.mean().item()),
                "reverse_top_margin": float(m_reverse.mean().item()),
                "top_improvement": float((m_top-m_bad).mean().item()),
                "top_minus_random_mean": float(paired.mean()),
                "top_minus_random_bootstrap_ci95_low": ci_low,
                "top_minus_random_bootstrap_ci95_high": ci_high,
                "top_beats_random_at_ci95": bool(ci_low > 0),
                "random_improvement": float((m_random-m_bad).mean().item()),
                "reverse_necessity_drop": float((m_clean-m_reverse).mean().item()),
                "mean_clean_bad_gap": denominator,
                "top_recovery_fraction": float((m_top-m_bad).mean().item() / denominator) if abs(denominator)>1e-8 else None,
                "full_patch_max_abs_error": full_abs_error,
                "interpretation": "hypothesis_only_single_model_toy_intervention",
            })
    return records
