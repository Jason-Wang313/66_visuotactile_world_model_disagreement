# Paper 66 Rebuild Plan: Visuotactile World-Model Disagreement

## Terminal Objective
Rebuild `66_visuotactile_world_model_disagreement` as a real evidence package. The paper may become submission-ready only if the evidence shows that keeping vision-touch disagreement as explicit latent physical branches improves closed-loop manipulation beyond strong uncertainty, risk-filter, and diagnostic-probe baselines. If not, archive it honestly.

## Central Claim Under Test
Vision-only and mean-fusion world models collapse ambiguous physical modes too early. A visuotactile disagreement planner should preserve multiple action-critical hypotheses until touch or motion evidence resolves them, then choose safer pushes, probes, or abstentions.

## High-Fidelity Benchmark
- Build a lightweight MuJoCo planar manipulation benchmark with a pusher, a sliding object, a target, contacts, friction, sensor noise, and hidden physical modes.
- Hidden modes:
  - nominal friction and visible object pose
  - high-friction object
  - low-friction object
  - vision-biased/occluded pose
  - tactile-noisy contact
  - sticky contact / object drag
  - combined vision+tactile shift
- Each episode is a closed-loop manipulation trial: observe vision and tactile signals, optionally run diagnostic probe actions, plan a push, execute in MuJoCo, and evaluate target success, contact violation, energy, and belief calibration.

## Methods And Baselines
- `random_push`: sanity-check lower bound.
- `vision_only_mpc`: plans from visual object pose and a single nominal friction model.
- `tactile_only_mpc`: ignores visual correction and infers mode from contact force only.
- `mean_fusion_mpc`: fuses vision/touch into a single collapsed state estimate.
- `ensemble_uncertainty_mpc`: samples model parameters and selects conservative action by expected utility.
- `conformal_risk_filter`: rejects high-risk actions under calibration residual thresholds.
- `diagnostic_probe_then_mpc`: performs a fixed probe before planning.
- `vt_disagreement_branch_mpc`: proposed method; keeps branch hypotheses when vision and touch disagree, values diagnostic actions only when branch entropy matters, and plans under branch-weighted risk.
- `oracle_mode_mpc`: non-submission upper bound with access to the hidden mode.

## Required Experiments
- Main benchmark: at least 5 seeds, 10-12 episodes per seed per split and method, with real MuJoCo rollouts for every executed action.
- Splits: nominal, high friction, low friction, vision bias, tactile noise, sticky contact, combined shift.
- Stress sweep: continuous levels for visual bias, tactile noise, and friction shift.
- Ablations:
  - no branch preservation
  - no disagreement trigger
  - no diagnostic value
  - no tactile residual update
  - no risk penalty
  - small branch set
- Pairwise comparisons against `mean_fusion_mpc`, `ensemble_uncertainty_mpc`, `conformal_risk_filter`, and `diagnostic_probe_then_mpc`.
- Negative cases: adversarial simultaneous vision and tactile corruption, unmodeled actuator weakness, and target ambiguity.

## Submission-Readiness Gate
To be ICLR-main ready, the proposed method must:
- beat every non-oracle baseline on combined shift and at least four of six non-nominal splits
- show non-overlapping or clearly favorable seed uncertainty against the best robust baseline
- retain the advantage under ablations, with the no-branch and no-disagreement variants clearly worse
- avoid merely trading success for unsafe contact, excessive energy, or abstention
- include a hostile prior-work discussion and honest limitations

## Terminal Decision Rules
- `SUBMISSION_READY_CANDIDATE`: only if all gates above pass and the paper can be written as a strong empirical contribution.
- `STRONG_REVISE`: if the method has real advantages but not enough breadth, related-work depth, or hardware/public-benchmark validation for ICLR main.
- `KILL_ARCHIVE`: if mean fusion, uncertainty/risk filters, diagnostic probing, or no-branch ablations match the proposed method.

## Resource Discipline
Keep RAM light by using deterministic MuJoCo rollouts, compact CSV outputs, small branch sets, and process-level parallelism capped to a few workers. Do not reduce experimental rigor: preserve seeds, baselines, ablations, uncertainty, and stress tests.

## Deliverables
- Rewritten `src/run_experiment.py` with real MuJoCo rollouts and implemented baselines.
- Updated requirements, README, child status, claims, gate, readiness, audit, and evidence docs.
- CSV results, figures, pairwise stats, stress sweep, negative cases, and a terminal evidence summary.
- Rewritten paper and compiled `C:/Users/wangz/Downloads/66.pdf` only.
- Public GitHub repo pushed with the final commit.
- Root reports updated before moving to Paper 67.
