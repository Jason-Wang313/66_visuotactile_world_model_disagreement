# Paper 66 Terminal Evidence

Decision: `KILL_ARCHIVE`

## Real-Evidence Rebuild
The v4 rebuild replaces the synthetic scaffold with a MuJoCo planar contact-manipulation benchmark. Each episode executes real contact rollouts for a pusher, object, tactile probe, and final push under hidden physical modes.

Run command:

```powershell
python src\run_experiment.py
```

Generated evidence:
- 3,780 main MuJoCo episode rows.
- 420 ablation rows.
- 1,200 stress-sweep rows.
- 5 seeds, 12 main episodes per seed, 7 splits, 9 main methods.
- CSVs: raw rollouts, metrics, seed metrics, pairwise comparisons, ablations, stress sweep, negative cases.
- Figures: success by split, energy by split, ablation success, stress sweep.

## Combined-Shift Results

| Method | Success | CI95 | Error | Energy |
|---|---:|---:|---:|---:|
| `random_push` | 0.083 | 0.071 | 0.151 | 0.415 |
| `vision_only_mpc` | 0.550 | 0.127 | 0.073 | 0.527 |
| `tactile_only_mpc` | 0.633 | 0.123 | 0.074 | 0.501 |
| `mean_fusion_mpc` | 0.667 | 0.120 | 0.064 | 0.502 |
| `ensemble_uncertainty_mpc` | 0.733 | 0.113 | 0.064 | 0.513 |
| `conformal_risk_filter` | 0.583 | 0.126 | 0.077 | 0.483 |
| `diagnostic_probe_then_mpc` | 0.633 | 0.123 | 0.070 | 0.601 |
| `vt_disagreement_branch_mpc` | 0.450 | 0.127 | 0.094 | 0.564 |
| `oracle_mode_mpc` | 0.800 | 0.102 | 0.052 | 0.554 |

Pairwise combined-shift comparisons show `vt_disagreement_branch_mpc` is below mean fusion by 0.217 success and below ensemble uncertainty by 0.283 success.

## Ablation Results

| Ablation | Success | CI95 | Energy |
|---|---:|---:|---:|
| `full_vt_disagreement_branch_mpc` | 0.500 | 0.128 | 0.532 |
| `no_branch_preservation` | 0.517 | 0.128 | 0.547 |
| `no_diagnostic_value` | 0.550 | 0.127 | 0.481 |
| `no_disagreement_trigger` | 0.583 | 0.126 | 0.496 |
| `no_risk_penalty` | 0.600 | 0.125 | 0.523 |
| `no_tactile_residual_update` | 0.767 | 0.108 | 0.519 |
| `small_branch_set` | 0.500 | 0.128 | 0.527 |

## Terminal Rationale
The central claim requires branch-preserving visuotactile disagreement to beat branch-collapsing and uncertainty-based alternatives. It does not. Strong non-oracle baselines outperform it, and ablations without the claimed mechanism match or beat the full method. The honest action is `KILL_ARCHIVE`.
