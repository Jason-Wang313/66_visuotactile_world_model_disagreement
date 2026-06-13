# Claims

- Mechanism claim: Visuotactile world model disagreement keeps action-critical alternatives explicit until a physical observation collapses them.
- Real-evidence result: the v4 MuJoCo benchmark falsifies the mechanism as implemented. On combined shift, `vt_disagreement_branch_mpc` reaches 0.450 success, below `mean_fusion_mpc` at 0.667 and `ensemble_uncertainty_mpc` at 0.733.
- Ablation result: no-branch, no-disagreement-trigger, no-risk-penalty, and no-tactile-residual ablations match or beat the full branch method.
- Scope claim: the results support archiving this specific mechanism, not deployment.
- Unsupported claim explicitly avoided: no claim of SOTA robot performance.
