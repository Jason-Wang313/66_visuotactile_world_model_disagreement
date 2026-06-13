# Novelty Boundary Map

## Crowded Territory
- Bigger data/model scaling.
- New benchmark only.
- Generic active learning or uncertainty.
- Combining a planner with a learned policy without a new state/action object.

## Claimed Boundary
Visuotactile world model disagreement keeps action-critical alternatives explicit until a physical observation collapses them.

## What Would Falsify The Claim
If observed-only baselines match the adverse-mode coverage and closed-loop success of the proposed branch-aware mechanism, the paper should be revised or killed.

## v4 Falsification
The real MuJoCo rebuild falsifies the current claim. On combined shift, `vt_disagreement_branch_mpc` reaches 0.450 success, while `mean_fusion_mpc` reaches 0.667 and `ensemble_uncertainty_mpc` reaches 0.733. No-branch and no-disagreement ablations also match or beat the full method.
