# Research architecture

## Stage 0 — Define the skill contract

Create distinct skill probes, matched contrast prompts, retain prompts, and held-out adversarial prompts. Define scoring direction, floors, and retention tolerances before probing. Each prompt ID must be unique, and each contrast pair must reference valid groups. The contract validator enforces these structural properties.

**Gate:** pre-registered scoring and sufficient independent samples. The provided toy example is only a schema demonstration.

## Stage 1 — Behavioral characterization

Evaluate direct probes, paraphrases, format and language transfer (where meaningful), and adversarial attempts. Record outputs, model revision, tokenizer revision, sampling seed, and environment fingerprint. Distinguish elicitation robustness from generalization.

**Gate:** a reproducible behavior fingerprint, holdout split, and a shortlist of candidate activation sites. Correlation does not imply causal necessity.

## Stage 2 — Node interventions

Use matched clean/corrupted prompts, patch activations in both directions, and compare against same-size random and irrelevant-site interventions. Rank with attribution patching only when its approximation is checked by true patching. Redundant components may fail single-site necessity tests even when jointly important.

**Gate:** reproducible intervention effects, uncertainty estimates, and interaction tests for candidate components.

## Stage 3 — Edge interventions

Intervene on individual contributions, not all downstream effects of a node. Search backwards from the output, then stress-test the circuit on independent prompt families. A graph minimal with respect to tested removals need not be globally minimal.

**Gate:** directed circuit hypothesis with held-out causal confirmation and explicit uncovered cases.

## Stage 4 — Weight attribution

Combine forget gradients, retain gradients, and known task deltas if available. Compare saliency-selected masks with equal-sized random and weight-magnitude-matched masks. Activation sites do not give unique mappings to parameters.

**Gate:** a sparse candidate mask with measured confidence, controls, and explicit shared-weight warnings.

## Stage 5 — Contamination rehearsal

Edit only a disposable model copy. Probe adjacent skills and unrelated skills, not just global perplexity. Use equivalence tests with a predeclared retention margin; failure to reject degradation is not evidence of equivalence.

**Gate:** target suppression and retention equivalence on validation data with bounded uncertainty.

## Stage 6 — Selective editing / extraction

Escalate from reversible activation interventions to task-vector editing or constrained training only after gates pass. A parameter mask is not automatically a portable LoRA adapter; adapters must be trained and separately evaluated.

**Gate:** a versioned derivative artifact plus immutable baseline checksums and complete provenance.

## Stage 7 — Independent verification

Evaluate direct, adversarial, paraphrase, multilingual, transfer, latent-knowledge, and membership-inference tests, plus retention and perplexity controls. Separate test data from selection data. A failure to recover information provides only a bounded claim about the tested attack family.

**Gate:** publish a test-by-test result table, statistical uncertainties, unresolved recovery paths, and release decision; reroute failures to causal mapping or contamination analysis as appropriate.
