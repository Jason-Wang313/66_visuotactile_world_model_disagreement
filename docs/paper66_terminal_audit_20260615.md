# Paper 66 Terminal Audit

Date: 2026-06-15

Paper: `66_visuotactile_world_model_disagreement`

Decision: `KILL_ARCHIVE`

ICLR-main ready: no

## Commands Executed

- `python -m py_compile src\run_experiment.py`
- CSV finite/schema audit over `results/vt_disagreement_raw.csv`, `results/vt_disagreement_metrics.csv`, `results/vt_disagreement_pairwise.csv`, `results/vt_disagreement_ablation.csv`, `results/vt_disagreement_ablation_raw.csv`, `results/raw_seed_metrics.csv`, `results/negative_cases.csv`, compatibility CSVs, and `results/stress_sweep.csv`.
- `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` in `paper`
- `Copy-Item paper\main.pdf C:\Users\wangz\Downloads\66.pdf -Force`

## Verified Evidence

- Real MuJoCo planar contact-manipulation rollouts are implemented in `src/run_experiment.py`.
- Main evidence contains 3,780 MuJoCo episodes: 7 hidden physical splits, 5 seeds, 12 episodes per seed/split/method, and 9 methods.
- Ablation evidence contains 420 combined-shift episodes.
- Stress-sweep evidence contains 1,200 episodes.
- Baselines include random push, vision-only MPC, tactile-only MPC, mean-fusion MPC, ensemble-uncertainty MPC, conformal risk filtering, diagnostic probing, and oracle mode MPC.
- CSV outputs are present, non-empty, and finite.
- BibTeX warnings from missing prior-work sort keys were fixed without inventing authors.
- The rebuilt PDF is `C:/Users/wangz/Downloads/66.pdf`.
- `C:/Users/wangz/Desktop/66.pdf` is absent.

## Fatal Results

The branch-preserving visuotactile-disagreement mechanism is falsified by the current evidence:

- Combined shift: `vt_disagreement_branch_mpc` reaches `0.450 +/- 0.127` success.
- Combined shift: `mean_fusion_mpc` reaches `0.667 +/- 0.120` success.
- Combined shift: `ensemble_uncertainty_mpc` reaches `0.733 +/- 0.113` success.
- Combined shift: `oracle_mode_mpc` reaches `0.800 +/- 0.102` success.
- Pairwise comparisons are significant in the wrong direction for the proposed method versus mean fusion (`diff=-0.2166`, `p=0.0080`) and ensemble uncertainty (`diff=-0.2833`, `p=0.0083`).
- Combined-shift ablations are fatal: `no_branch_preservation`, `no_diagnostic_value`, `no_disagreement_trigger`, `no_risk_penalty`, and `no_tactile_residual_update` all match or beat the full method.

## Gate Decision

This paper satisfies the local evidence-package requirements for a real negative result: high-fidelity simulator evidence, paired baselines, ablations, stress tests, uncertainty, negative cases, rebuilt PDF, corrected BibTeX metadata, corrected hostile-review documentation, and public repository.

It does not satisfy `STRONG_REVISE` because the proposed mechanism is not merely under-validated; it loses to strong non-oracle baselines and is contradicted by ablations. The correct terminal state remains `KILL_ARCHIVE`.

Required revival work:

- invent a substantially different visuotactile latent-state mechanism that clears mean fusion and ensemble uncertainty;
- prove the mechanism is necessary through ablations where branch preservation and disagreement triggers matter;
- validate on hardware or a public contact-rich manipulation benchmark;
- train or release a learned world-action model rather than only analytic planners;
- perform a manual full-paper related-work synthesis.
