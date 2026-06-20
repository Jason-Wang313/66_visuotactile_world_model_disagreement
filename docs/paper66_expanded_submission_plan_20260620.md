# Paper 66 Expanded-Standard Execution Plan

Date: 2026-06-20

Paper: `66_visuotactile_world_model_disagreement`

Target standard: 25+ page ICLR-style manuscript, CPU-only/RAM-light execution, bright boxed clickable citations, numbered PDF in Downloads only, public GitHub repository updated, and honest terminal decision.

## Current State

The v4 artifact is a real MuJoCo planar visuotactile pushing benchmark, but it is far below the expanded standard:

- 3,780 main rows.
- 420 ablation rows.
- 5 seeds.
- 12 episodes per seed/split/method.
- 7 main splits.
- 9 main methods.
- 8 paired rows, only on `combined_shift`.
- 4-page PDF.
- Terminal decision: `KILL_ARCHIVE`.

The old negative result is strong. `vt_disagreement_branch_mpc` beats only `random_push` in aggregate and loses to mean fusion, ensemble uncertainty, conformal risk filtering, diagnostic probing, and oracle references. On `combined_shift`, it reaches 0.450 success versus 0.667 for mean fusion, 0.733 for ensemble uncertainty, and 0.800 for the oracle. Ablations are fatal: removing tactile residual update reaches 0.767 on combined shift, far above the full branch method.

## Core Failure To Attack

The old mechanism preserves branches, but it does not know when a branch is sensor-corrupted rather than physically informative. The branch planner often over-trusts noisy tactile residuals and high-force branches. The v5 rebuild must test a sharper hypothesis:

> Explicit disagreement is useful only when paired with calibrated sensor-health and action-value gating. Branch preservation alone is not enough.

The v5 method must therefore separate:

- physical mode uncertainty: friction, drag, actuator response, sticky contact;
- sensor-health uncertainty: vision bias, tactile noise/dropout, delayed touch, contact ambiguity;
- value of information: whether probing changes the action choice enough to pay its cost.

If this calibrated branch-and-probe mechanism still loses to mean fusion, ensemble uncertainty, robust/minimax MPC, or no-branch ablations, the paper remains `KILL_ARCHIVE`.

## Proposed v5 Method

Working method name: Calibrated Visuotactile Branch-and-Probe MPC (CVTB-MPC).

CVTB-MPC extends the old branch planner with:

1. A compact joint branch set over physical mode and sensor-health mode.
2. Reliability-calibrated branch weights from vision residual, tactile contact confidence, force consistency, and disagreement magnitude.
3. Branch pruning when tactile and vision disagreement is likely a sensor fault rather than a physical mode.
4. Value-of-information gating: run a diagnostic probe only when branch-conditioned best actions disagree and the expected regret reduction exceeds probe cost.
5. A minimax-CVaR term over the worst branch losses.
6. A contact-safety term for peak force and repeated contact.
7. A conservative fallback to robust mean-fusion when branch posterior entropy is high but sensor reliability is low.

This remains CPU/RAM light: no neural training, no GPU, compact branch sets, deterministic MuJoCo rollouts, and at most four workers by default.

## Main Baselines

The frozen main run should include at least:

- `random_push`
- `vision_only_mpc`
- `tactile_only_mpc`
- `mean_fusion_mpc`
- `ensemble_uncertainty_mpc`
- `conformal_risk_filter`
- `diagnostic_probe_then_mpc`
- `robust_minimax_mpc`
- `particle_belief_mpc`
- `old_vt_disagreement_branch_mpc`
- `cvtb_mpc_v5`
- `cvtb_no_probe`
- `oracle_mode_mpc`

`old_vt_disagreement_branch_mpc` must stay in the frozen run so we can prove whether v5 improves the old mechanism honestly rather than erasing it.

## Ablations

Ablations should run on at least `combined_shift`, `sensor_conflict`, and `delayed_touch_sticky`.

Required ablations:

- `cvtb_mpc_v5`
- `cvtb_no_probe`
- `no_sensor_health`
- `no_branch_preservation`
- `no_value_of_information`
- `no_cvar_tail`
- `no_contact_safety`
- `no_reliability_fallback`
- `mean_only_fusion`
- `tactile_trust_high`
- `small_branch_set`
- `old_vt_disagreement_branch_mpc`
- `ensemble_uncertainty_mpc`
- `robust_minimax_mpc`
- `oracle_mode_mpc`

## Stress Splits

The frozen main run should include 10 splits:

- `nominal`
- `high_friction`
- `low_friction`
- `vision_bias`
- `tactile_noise`
- `sticky_contact`
- `combined_shift`
- `sensor_conflict`
- `contact_dropout`
- `delayed_touch_sticky`

The last three are designed to distinguish physical mode disagreement from sensor corruption.

## Pre-Freeze Development Protocol

1. Compile the current runner.
2. Run the current v4 runner only if needed for a smoke baseline; do not treat v4 results as final.
3. Implement output-dir arguments, ablation-split arguments, deterministic partial writes, and plotting that works for subset smoke runs.
4. Implement v5 splits, baselines, branch reliability, value-of-information gating, CVaR/minimax scoring, and extra metrics.
5. Run a tiny smoke test:
   - 1 seed.
   - 1 episode.
   - splits: `nominal`.
   - ablation splits: `combined_shift`.
   - workers: 1.
6. Run a medium development run:
   - 2 seeds.
   - 6 episodes.
   - splits: `nominal`, `combined_shift`, `sensor_conflict`, `delayed_touch_sticky`.
   - ablation splits: `combined_shift`, `sensor_conflict`.
   - workers: 4.
7. Inspect:
   - whether CVTB changes actions relative to mean fusion, ensemble, robust minimax, and old VT;
   - whether the action changes improve success/error/energy without raising contact violations;
   - whether sensor-health and value-of-information ablations hurt on hostile splits;
   - whether old VT remains worse than v5.
8. Repair recoverable implementation/design failures before freeze.
9. Write a development log.
10. Freeze the final protocol and do no tuning after freeze.

## Frozen Target Scale

Target full run:

- Seeds: 8.
- Episodes per seed/split: 24.
- Main splits: 10.
- Main methods: 13.
- Expected main rows: 24,960.
- Ablation splits: 3.
- Ablation methods: 15.
- Expected ablation rows: 8,640.
- Stress sweep: at least 6 levels, selected methods, 8 seeds, 12 episodes per seed/level.
- CPU-only execution.
- RAM-light execution: compact branch sets, no GPU, no large learned models.

If runtime becomes unreasonable, reduce stress-sweep episodes before protocol freeze and document why. Do not reduce strong baselines or mechanism ablations to make the result look better.

## Decision Gates

Weak gate:

- CVTB-MPC must beat random, vision-only, tactile-only, and old VT in aggregate success and final error.

Strong gate:

- CVTB-MPC must beat or tie mean fusion, ensemble uncertainty, conformal risk filtering, diagnostic probing, robust minimax, and particle belief MPC in aggregate success, combined-shift success, and at least two of the three new hostile splits.
- Contact violation must not increase relative to the safest strong baseline by more than 0.02 absolute.

Mechanism gate:

- CVTB-MPC must change action choices relative to mean fusion and old VT in a nontrivial fraction of hostile episodes.
- `no_sensor_health`, `no_value_of_information`, `no_cvar_tail`, and `no_reliability_fallback` must be worse on pre-registered hostile splits.
- `cvtb_no_probe` may tie on easy splits, but should lose when probe value is high.

Terminal decision:

- `ACCEPTABLE_SUBMISSION_CANDIDATE`: all gates pass, validation passes, and the manuscript is genuinely submission-grade.
- `STRONG_REVISE`: weak and some strong evidence is promising, but external validation, related-work depth, or ablations remain insufficient.
- `KILL_ARCHIVE`: strong baselines or ablations match/beat CVTB-MPC, or sensor-health branch mechanism remains non-identifiable.

Given v4 evidence, the prior expectation is `KILL_ARCHIVE` unless v5 creates a real frozen separation.

## Manuscript Requirements

The final paper must be 25+ pages and contain:

- Formal visuotactile belief/branch problem setup.
- Theory explaining when branch preservation helps and when it collapses into ensemble uncertainty or robust fusion.
- Sensor-health identifiability analysis.
- Full frozen protocol.
- Generated tables from CSVs.
- Main, ablation, pairwise, seed, stress, and failure analyses.
- Action-change and probe-use diagnostics.
- Honest limitations and terminal decision.
- Bright boxed clickable citations.

## Validation Requirements

The final pass must verify:

- `python -m py_compile src\run_experiment.py`.
- Frozen row counts.
- Expected result CSVs and figures exist.
- Manuscript compiles.
- Downloads-only `C:\Users\wangz\Downloads\66.pdf`.
- PDF is 25+ pages.
- No `C:\Users\wangz\Desktop\66.pdf`.
- Public GitHub repo is updated and remains public.
- Root ledgers are updated.
