# Paper 66 ICLR-Main Execution Plan

Date: 2026-06-15

Paper: `66_visuotactile_world_model_disagreement`

Goal: verify whether the current real MuJoCo visuotactile-disagreement evidence can honestly support an ICLR-main-target submission, or whether the paper must remain `KILL_ARCHIVE` as a falsified mechanism-level negative result.

## Execution Gates

1. Reproducibility gate:
   - Compile `src/run_experiment.py`.
   - Confirm main, seed, pairwise, ablation, stress-sweep, negative-case, and compatibility CSV outputs exist.
   - Confirm all CSV outputs are non-empty and finite.
   - Rebuild the PDF from `paper/main.tex` with BibTeX.

2. Evidence gate:
   - Confirm the benchmark uses real MuJoCo contact rollouts rather than synthetic probability tables.
   - Confirm five seeds, seven hidden physical splits, nine main methods, confidence intervals, paired comparisons, ablations, and stress sweeps.
   - Confirm baselines include random push, vision-only MPC, tactile-only MPC, mean-fusion MPC, ensemble-uncertainty MPC, conformal risk filtering, diagnostic probing, and oracle mode MPC.

3. Negative-claim gate:
   - Compare `vt_disagreement_branch_mpc` against mean fusion, ensemble uncertainty, diagnostic probing, and oracle mode MPC under combined shift.
   - Check whether no-branch, no-diagnostic-value, no-disagreement-trigger, no-risk-penalty, no-tactile-residual-update, and small-branch ablations degrade performance.
   - Fix stale documentation that still presents the archive reason as synthetic-only evidence rather than the current real MuJoCo falsification.

4. Artifact gate:
   - Rebuild `paper/main.pdf`.
   - Copy only `C:/Users/wangz/Downloads/66.pdf`.
   - Confirm `C:/Users/wangz/Desktop/66.pdf` is absent.
   - Confirm the GitHub repository is public, clean, and pushed.

## Decision Rule

Upgrade only if the branch-preserving visuotactile planner clearly beats strong non-oracle baselines and its ablations show the claimed disagreement mechanism is necessary. If it loses to mean fusion or ensemble uncertainty, or if no-mechanism ablations match or beat the full method, keep the terminal decision as `KILL_ARCHIVE` and document the failure as real negative evidence rather than a formatting or packaging issue.
