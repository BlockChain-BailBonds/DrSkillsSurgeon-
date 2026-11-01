# ROOTFORGE Skill Surgery

A staged research program for causal localization, selective extraction, and evaluation of model capability edits.

**Status:** Stage 0 contract validation is implemented. Causal tracing, weight editing, extraction, and unlearning are research milestones, not implemented or verified by this scaffold.

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

## Repository roadmap

- **Stage 0:** Behavioral contract schema and validation (**implemented**).
- **Stage 1:** Baseline behavior, paraphrases, held-out attacks, and activation fingerprints (**planned**).
- **Stage 2:** Forward/backward activation patching with controls (**planned**).
- **Stage 3:** Path patching and minimal circuit proposals (**planned**).
- **Stage 4:** Forget/retain-aware weight attribution and matched baselines (**planned**).
- **Stage 5:** Reversible contamination rehearsal and statistical equivalence (**planned**).
- **Stage 6:** Extract or edit only on disposable checkpoints (**planned**).
- **Stage 7:** Independent adversarial, latent-probe, and retention evaluation (**planned**).

**Model safety:** never mutate an original checkpoint. Preserve originals and record SHA-256 digests for derived artifacts. A successful behavioral suppression test does not establish complete unlearning.
