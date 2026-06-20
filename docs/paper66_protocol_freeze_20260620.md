# Paper 66 Frozen Protocol

Date frozen: 2026-06-20

This document freezes the final expanded-standard evidence protocol for Paper 66. No method tuning is allowed after this point.

## Primary Method

Primary method: `cvtb_mpc_v5`

Manuscript name: Calibrated Visuotactile Branch-and-Probe MPC (CVTB-MPC)

CVTB-MPC scores actions using:

- reliability-calibrated vision/tactile/physical/sensor-fault branches;
- compact physical-mode branches over friction, drag, and sticky contact;
- sensor-health weighting from contact confidence, tactile noise, dropout, force, displacement, and disagreement;
- CVaR-style tail branch loss;
- contact-safety penalty;
- conservative fallback under suspected sensor corruption;
- value-of-information diagnostic probing, frozen after development.

## Final Command

```powershell
python src\run_experiment.py --seeds 8 --episodes 24 --ablation-episodes 24 --stress-episodes 12 --splits nominal high_friction low_friction vision_bias tactile_noise sticky_contact combined_shift sensor_conflict contact_dropout delayed_touch_sticky --ablation-splits combined_shift sensor_conflict delayed_touch_sticky --stress-levels 0.0 0.2 0.4 0.6 0.8 1.0 --workers 4 --results-dir results --figures-dir figures
```

## Frozen Scale

- Seeds: 8.
- Episodes per seed/split: 24.
- Main splits: 10.
- Main methods: 13.
- Expected main rows: 24,960.
- Ablation splits: 3.
- Ablation methods: 15.
- Expected ablation rows: 8,640.
- Stress levels: 6.
- Stress methods: 6.
- Stress episodes per seed/level/method: 12.
- Expected stress rows: 3,456.
- CPU-only execution.
- RAM-light execution: compact branch sets, deterministic MuJoCo rollouts, no GPU, no large neural model.

## Main Splits

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

## Main Methods

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

## Ablation Methods

Frozen on `combined_shift`, `sensor_conflict`, and `delayed_touch_sticky`:

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

## Metrics

Primary:

- Success rate.
- Final target error.
- Energy/contact-effort proxy.
- Contact violation rate.

Mechanism diagnostics:

- Action difference rate versus baselines.
- Diagnostic probe rate.
- Probe value.
- Branch entropy.
- Fallback rate.
- Pairwise success/error/energy deltas on shared episodes.

## Decision Gates

Weak gate:

- CVTB-MPC must beat random, tactile-only, and old VT in aggregate success and final error.

Strong gate:

- CVTB-MPC must beat or tie mean fusion, ensemble uncertainty, conformal risk filtering, diagnostic probing, robust minimax, particle belief MPC, and vision-only in aggregate success and final error.
- CVTB-MPC must not lose on `combined_shift`, `sensor_conflict`, `contact_dropout`, or `delayed_touch_sticky`.
- Contact violation must remain within 0.02 absolute of the safest strong baseline.

Mechanism gate:

- CVTB-MPC must change actions relative to mean fusion and old VT in a nontrivial fraction of hostile episodes.
- `cvtb_no_probe`, `no_sensor_health`, `no_value_of_information`, `no_cvar_tail`, and `no_reliability_fallback` should be worse on pre-registered hostile splits.
- If `cvtb_no_probe` matches CVTB-MPC, the value-of-information mechanism fails.

Terminal decision:

- `STRONG_REVISE` only if weak and most strong gates pass but external validation remains missing.
- `KILL_ARCHIVE` if strong baselines or ablations match/beat CVTB-MPC, or if sensor-health/value-of-information mechanisms remain non-identifiable.

Prior expectation from development is `KILL_ARCHIVE`.

## Artifact Rules

- Final PDF must be `C:\Users\wangz\Downloads\66.pdf`.
- No PDF may be copied to visible Desktop.
- Manuscript must be at least 25 pages.
- Manuscript must use bright boxed clickable citations.
- Public GitHub repo must be pushed after validation.
