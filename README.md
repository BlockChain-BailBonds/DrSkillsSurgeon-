# ROOTFORGE Skill Surgery

A staged research program for causal localization, selective extraction, and evaluation of model capability edits.

**Status:** Stage 0 contract validation and a Stage 1/2 **toy research implementation** are available. The experiment trains fresh small models, measures held-out feature predictions, and performs controlled activation interventions; it is not a real-model weight-editing or skill-unlearning system.

## Research principles

1. Use independent *skill*, *contrast*, *retain*, and *adversarial* datasets.
2. Separate correlation, causal intervention, and parameter attribution.
3. Measure shared-use contamination before changing any weights.
4. Compare against matched random-mask and unedited baselines.
5. Treat resistance to tested recovery attacks as bounded evidence, not proof of complete erasure.

See [architecture](docs/ARCHITECTURE.md) and [validation protocol](docs/VALIDATION.md).

## Stage 0: run the contract validator

Requires Python 3.10 or later; no third-party packages are required for this initial stage.

```sh
PYTHONPATH=src python -m rootforge_skill_surgery validate examples/toy-contract.json
PYTHONPATH=src python -m unittest discover -s tests -v
```

For real experimental enrollment, validate minimum sample sizes:

```sh
PYTHONPATH=src python -m rootforge_skill_surgery validate examples/toy-contract.json --minimum-probes 500
```

The toy example is intentionally too small to pass that experimental gate.

## Reproducible weight/activation research (optional dependencies)

The [weight-image experiment](docs/WEIGHT_IMAGE_RESEARCH.md) reproduces a controlled comparison of unnormalized weight statistics, singular-value spectra, fixed-input activations, and untrained controls. It evaluates classifiers on **unseen initialization seeds**, and uses bidirectional activation patching with same-size random-site controls on held-out counterfactual pairs.

```sh
pip install -r requirements-research.txt
PYTHONPATH=src python -m rootforge_skill_surgery.research.experiment \
  --seeds 8 --steps 160 --batch-size 96 --device auto \
  --output runs/my_first_experiment
```

The toy experiment is separate from the original user-supplied model screenshots. See the [evidence notes](docs/WEIGHT_IMAGE_RESEARCH.md), [reproduced confusion matrix](evidence/confusion-matrix.svg), and [controlled run](evidence/runs/20261101-seed8-summary.json) for provenance and limitations.

## Repository roadmap

- **Stage 0:** Behavioral contract schema and validation (**implemented**).
- **Stage 1:** Toy controlled behavioral fingerprints, weight/spectrum features, and held-out-seed prediction (**implemented as a research baseline**); full natural-language adversarial coverage **planned**.
- **Stage 2:** Toy MLP neuron-level bidirectional activation patching with holdout and random controls (**implemented as a research baseline**); production transformer patching **planned**.
- **Stage 3:** Path patching and minimal circuit proposals (**planned**).
- **Stage 4:** Forget/retain-aware weight attribution and matched baselines (**planned**).
- **Stage 5:** Reversible contamination rehearsal and statistical equivalence (**planned**).
- **Stage 6:** Extract or edit only on disposable checkpoints (**planned**).
- **Stage 7:** Independent adversarial, latent-probe, and retention evaluation (**planned**).

**Model safety:** never mutate an original checkpoint. Preserve originals and record SHA-256 digests for derived artifacts. A successful behavioral suppression test does not establish complete unlearning.
