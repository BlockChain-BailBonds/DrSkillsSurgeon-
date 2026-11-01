# Research follow-up gates

## Gate A — Replication

Run at least eight initialization seeds with identical step and architecture budgets across four tasks; review seed-cluster uncertainty. Repeat with matched *task accuracy* budgets to test whether features track learning stage instead of skill identity. Distinct model architectures and optimizer seeds should be held out separately.

## Gate B — Feature controls

Compare raw parameters, singular spectra and fixed-input activation statistics against untrained weights and matched random features. Run negative-label permutation controls that preserve seed/task grouping. Report all model failures and folds, not only the best feature family.

## Gate C — Causal localization

Use held-out counterfactual pairs, score top-k versus matched random sites with paired confidence intervals, test reverse interventions, multi-neuron interactions and redundant pathways. Large full-site patch recovery is a structural sanity check only. Do not infer unique causal circuits from it.

## Gate D — Retention and editing

Before editing weights, pre-register a measured retain set with equivalence margins. Evaluate reversible ablations on disposable model copies. Only then compare sparse edits to matched random and norm-matched weight masks. Include independent adversarial recovery checks and report their budget. Never report universal erasure based on finite tests.
