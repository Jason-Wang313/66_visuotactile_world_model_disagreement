# Paper 66 Terminal Evidence

Decision: `KILL_ARCHIVE`

Date: 2026-06-20

## Expanded Real-Evidence Rebuild

The v5 rebuild tests Calibrated Visuotactile Branch-and-Probe MPC (CVTB-MPC) in a MuJoCo planar contact-manipulation benchmark. It extends the old branch-preservation mechanism with sensor-health branches, value-of-information probe gating, CVaR-style tail scoring, contact-safety penalties, and conservative fallback under suspected sensor corruption.

Frozen command:

```powershell
python src\run_experiment.py --seeds 8 --episodes 24 --ablation-episodes 24 --stress-episodes 12 --splits nominal high_friction low_friction vision_bias tactile_noise sticky_contact combined_shift sensor_conflict contact_dropout delayed_touch_sticky --ablation-splits combined_shift sensor_conflict delayed_touch_sticky --stress-levels 0.0 0.2 0.4 0.6 0.8 1.0 --workers 4 --results-dir results --figures-dir figures
```

Generated evidence:

- 24,960 main MuJoCo episode-method rows.
- 8,640 ablation rows.
- 3,456 stress-sweep rows.
- 8 seeds, 24 main episodes per seed/split/method, 10 main splits, 13 main methods.
- 1,040 seed summaries and 120 paired comparisons.
- CSVs: raw rollouts, metrics, seed metrics, pairwise comparisons, ablations, stress sweep, negative cases.
- Figures: success by split, energy by split, ablation success, stress sweep.
- PDF: 28 pages at `C:\Users\wangz\Downloads\66.pdf`.

## Aggregate Result

| Method | Aggregate success | Aggregate error | Aggregate energy |
|---|---:|---:|---:|
| `mean_fusion_mpc` | 0.796 | 0.052 | 0.492 |
| `particle_belief_mpc` | 0.790 | 0.053 | 0.491 |
| `robust_minimax_mpc` | 0.784 | 0.059 | 0.522 |
| `ensemble_uncertainty_mpc` | 0.778 | 0.055 | 0.489 |
| `cvtb_mpc_v5` | 0.777 | 0.056 | 0.487 |
| `cvtb_no_probe` | 0.777 | 0.056 | 0.487 |
| `old_vt_disagreement_branch_mpc` | 0.752 | 0.058 | 0.513 |
| `oracle_mode_mpc` | 0.908 | 0.040 | 0.520 |

## Hostile-Split Failures

- `combined_shift`: CVTB-MPC success 0.5469; vision-only 0.6094; robust minimax 0.5833; mean fusion 0.5729; particle belief 0.5729.
- `sensor_conflict`: CVTB-MPC success 0.3125; vision-only 0.3802; mean fusion 0.3750; particle belief 0.3438.
- `contact_dropout`: CVTB-MPC success 0.8385; vision-only 0.9271; mean fusion 0.9115; particle belief 0.8906.
- `delayed_touch_sticky`: CVTB-MPC success 0.6250; robust minimax 0.6458; old VT 0.6406; mean fusion and ensemble 0.6354.

## Ablation Failures

- `cvtb_no_probe` ties CVTB-MPC on all three pre-registered ablation splits.
- On `combined_shift`, `mean_only_fusion`, `no_sensor_health`, `tactile_trust_high`, and `small_branch_set` beat CVTB-MPC.
- On `sensor_conflict`, `mean_only_fusion`, `no_branch_preservation`, and `no_reliability_fallback` beat CVTB-MPC.
- On `delayed_touch_sticky`, robust minimax, old VT, ensemble uncertainty, `no_sensor_health`, and `no_reliability_fallback` beat or match CVTB-MPC.

## Terminal Rationale

The central claim requires calibrated branch-and-probe planning to beat branch-collapsing, robust, belief-space, and no-mechanism alternatives under hidden visuotactile shifts. It does not. Strong baselines match or beat CVTB-MPC, and ablations without the claimed mechanisms match or beat the full method. The honest action is `KILL_ARCHIVE`.
