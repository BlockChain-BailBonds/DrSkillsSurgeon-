# Evaluation contract

## Quantities

- `S_F(theta)`: preregistered target-skill expression score, with direction recorded in the contract.
- `U_R(theta)`: per-domain retained performance, reported as a vector and not only an aggregate.
- `D(theta, theta_prime)`: number of edited parameters, edit norm, and optional compute cost.
- `R_A(theta_prime)`: strongest tested attack recovery under a specified attack budget.

The optimization objective should limit edit cost subject to a target-suppression threshold, per-domain retention equivalence margins, and attack-bounded recovery limits. This is not an information-theoretic proof of erasure.

## Experimental design

1. Freeze datasets, scoring scripts, and intervention budgets before selecting a mask.
2. Split data by semantic task family so paraphrase copies cannot leak across train/validation/test.
3. Compare baseline, matched random masks, magnitude-matched masks, reversible edits, and proposed mask under equal budgets.
4. Use paired bootstrap confidence intervals or another predeclared uncertainty procedure and multiple-seed reruns where stochasticity matters.
5. Test retention with statistical equivalence margins, not a nonsignificant difference test.
6. Run the final adversarial suite on untouched test examples and report best successful attacks explicitly.
7. Preserve hashes of baseline, dataset, probes, masks, checkpoints, and scored results.

## Claim discipline

Allowed: 'No recovery was observed on N pre-registered attacks at the stated budget.'

Not allowed: 'The skill was completely erased from every internal representation.'
