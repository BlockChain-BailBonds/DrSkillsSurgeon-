"""Seed-held-out evaluation of weight statistics, spectra, and activations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import describe_model, weight_features
from .model import SkillMLP
from .patching import patch_sites
from .tasks import TASKS, batch, fixed_inputs, targets


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, task: str, device: torch.device, *, examples: int = 384,
             data_seed: int = 908127) -> float:
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(data_seed)
    successes = 0
    total = 0
    with torch.inference_mode():
        remaining = examples
        while remaining:
            size = min(256, remaining)
            x, y, mask = batch(task, size, g, device)
            guesses = model(x).argmax(dim=-1)
            successes += int(((guesses == y) & mask).sum().item())
            total += int(mask.sum().item())
            remaining -= size
    return successes / total


def train_one(task: str, seed: int, device: torch.device, *, steps: int,
              batch_size: int, lr: float, hidden: int):
    seed_everything(seed)
    model = SkillMLP(hidden=hidden).to(device)
    initial_features = weight_features(model)
    initial_accuracy = evaluate(model, task, device)
    g = torch.Generator(device="cpu").manual_seed(200000 + seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    last_loss = None
    for _ in range(steps):
        model.train()
        x, y, mask = batch(task, batch_size, g, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits[mask], y[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
    score = evaluate(model, task, device)
    features = describe_model(model, device)
    for name, value in initial_features.items():
        features[f"init_{name}"] = value
    buffer = io.BytesIO()
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, buffer)
    digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    return model, {"task": task, "seed": seed, "steps": steps, "eval_accuracy": score,
                   "init_accuracy": initial_accuracy, "final_training_loss": last_loss,
                   "sha256_pytorch_state_dict": digest, **features}


def heldout_seed_prediction(rows: list[dict]) -> dict:
    """Each entire initialization seed is kept out of classifier training."""
    if len(set(r["seed"] for r in rows)) < 2:
        raise ValueError("at least two distinct model seeds are required")
    feature_names = sorted(k for k in rows[0] if k.startswith(("raw.", "spectral.",
                                                                "activation.", "init_")))
    groups = {
        "raw_weight": [n for n in feature_names if n.startswith("raw.")],
        "spectral": [n for n in feature_names if n.startswith("spectral.")],
        "activation": [n for n in feature_names if n.startswith("activation.")],
        "weight_combined": [n for n in feature_names if n.startswith(("raw.", "spectral."))],
        "all": [n for n in feature_names if not n.startswith("init_")],
        "untrained_control": [n for n in feature_names if n.startswith("init_")],
    }
    labels = np.asarray([r["task"] for r in rows])
    seeds = np.asarray([r["seed"] for r in rows])
    performance = np.asarray([r["eval_accuracy"] for r in rows])
    baseline = []
    predictions = []
    outcomes = {}
    for label, names in groups.items():
        x = np.asarray([[float(r[n]) for n in names] for r in rows], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"non-finite feature values for {label}")
        predicted = np.empty_like(labels)
        for seed in sorted(set(seeds)):
            train = seeds != seed
            test = seeds == seed
            if len(set(labels[train])) < 2:
                raise ValueError("training fold needs at least two task classes")
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1500))
            clf.fit(x[train], labels[train])
            predicted[test] = clf.predict(x[test])
            majority = sorted(set(labels[train]), key=lambda k: (-sum(labels[train]==k), k))[0]
            if label == "raw_weight":
                baseline.extend([{"seed": int(seed), "task": str(truth),
                                  "prediction": str(majority)} for truth in labels[test]])
            predictions.extend({"feature_group": label, "heldout_seed": int(seed),
                                "task": str(truth), "prediction": str(guess)}
                               for truth, guess in zip(labels[test], predicted[test]))
        rng = np.random.default_rng(10918)
        per_seed_scores = np.asarray([
            accuracy_score(labels[seeds==s], predicted[seeds==s])
            for s in sorted(set(seeds))], dtype=float)
        draws = rng.integers(0, len(per_seed_scores), size=(1000, len(per_seed_scores)))
        boot = per_seed_scores[draws].mean(axis=1)
        outcomes[label] = {"accuracy": float(accuracy_score(labels, predicted)),
                           "seed_bootstrap_ci95": [float(v) for v in np.quantile(boot, [.025, .975])],
                           "features": len(names), "predictions": len(labels),
                           "per_seed_accuracy": {str(int(s)): float(accuracy_score(labels[seeds==s],predicted[seeds==s]))
                                                 for s in sorted(set(seeds))}}
    # Performance prediction is distinct from task-label classification.
    x = np.asarray([[float(r[n]) for n in groups["spectral"]] for r in rows])
    yhat = np.zeros(len(rows))
    for seed in sorted(set(seeds)):
        train, test = seeds != seed, seeds == seed
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(x[train], performance[train])
        yhat[test] = model.predict(x[test])
    return {"sample_count": len(rows), "seed_count": len(set(seeds)),
            "task_count": len(set(labels)), "chance_balanced": 1/len(set(labels)),
            "majority_baseline_accuracy": float(accuracy_score(
                [r["task"] for r in baseline], [r["prediction"] for r in baseline])),
            "feature_groups": outcomes,
            "spectral_regression_R2_heldout_seed": float(r2_score(performance, yhat)),
            "spectral_regression_note": "R2 may be negative; task/accuracy and training-progress confounds remain",
            "predictions": predictions,
            "spectral_regression_predictions": yhat.tolist()}


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: json.dumps(row[name]) if isinstance(row.get(name), (dict, list))
                             else row.get(name) for name in keys})


def run(*, seeds: int, steps: int, batch_size: int, hidden: int, lr: float,
        device_name: str, output: Path, patch_task: str, patch_top_k: int) -> dict:
    if seeds < 2 or steps < 1 or batch_size < 1 or hidden < 4 or not 0 < lr < 1:
        raise ValueError("seeds >= 2, steps >= 1, batch >= 1, hidden >= 4, and 0 < lr < 1 required")
    if patch_task not in TASKS:
        raise ValueError(f"patch task must be one of {TASKS}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output folder is not empty: {output}; choose a new path")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else "cpu" if device_name == "auto" else device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    data = []
    patch_model = None
    for seed in range(seeds):
        for task in TASKS:
            model, record = train_one(task, seed+1000, device, steps=steps,
                                      batch_size=batch_size, lr=lr, hidden=hidden)
            record["seed"] = seed
            data.append(record)
            print(f"seed={seed} task={task} accuracy={record['eval_accuracy']:.3f}", flush=True)
            if seed == seeds-1 and task == patch_task:
                patch_model = model
    summary = heldout_seed_prediction(data)
    patching = patch_sites(patch_model, patch_task, device, seed=1776, top_k=patch_top_k)
    _write_csv(output / "models_features.csv", data)
    _write_csv(output / "predictions.csv", summary.pop("predictions"))
    _write_csv(output / "causal_routes.csv", patching)
    summary["spectral_regression_predictions_ordered_as_models_csv"] = summary.pop("spectral_regression_predictions")
    summary["patching"] = patching
    summary["config"] = {"seeds": seeds, "steps": steps, "batch_size": batch_size,
                         "hidden": hidden, "learning_rate": lr, "device": str(device),
                         "patch_task": patch_task, "patch_top_k": patch_top_k}
    summary["environment"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                              "numpy": np.__version__, "platform": platform.platform(),
                              "cuda_available": torch.cuda.is_available()}
    summary["interpretation"] = (
        "Independent task-specific toy models. Held-out initialization seed evaluation. "
        "Feature predictivity is not a causal skill map; activation interventions "
        "do not demonstrate skill extraction, removal, or retention.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n",encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6, help="independent seeds, each covers all four tasks")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--patch-task", choices=TASKS, default="copy")
    parser.add_argument("--patch-top-k", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("runs/seed_heldout"))
    args = parser.parse_args(argv)
    result = run(seeds=args.seeds, steps=args.steps, batch_size=args.batch_size,
                 hidden=args.hidden, lr=args.learning_rate, device_name=args.device,
                 output=args.output, patch_task=args.patch_task, patch_top_k=args.patch_top_k)
    print(json.dumps({"output": str(args.output), "classification": result["feature_groups"],
                      "regression_R2": result["spectral_regression_R2_heldout_seed"],
                      "patching": result["patching"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
