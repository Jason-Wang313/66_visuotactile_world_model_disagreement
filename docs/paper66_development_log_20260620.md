# Paper 66 Development Log

Date: 2026-06-20

This log records pre-freeze development for Paper 66, "Visuotactile World-Model Disagreement." These runs are not final evidence. They were used to expose implementation issues and method weaknesses before freezing the final protocol.

## Starting Point

The v4 benchmark was a real MuJoCo planar contact-pushing setup with 3,780 main rows and 420 ablation rows. It reached `KILL_ARCHIVE` because `vt_disagreement_branch_mpc` lost to mean fusion, ensemble uncertainty, and no-mechanism ablations.

## v5 Changes Implemented Before Freeze

- Added CLI controls for seeds, episodes, splits, ablation splits, stress sweep, workers, and output directories.
- Expanded main methods to 13, including robust minimax, particle belief, old VT branch planner, CVTB-MPC, no-probe CVTB, and oracle.
- Added three sensor-health hostile splits:
- `sensor_conflict`
- `contact_dropout`
- `delayed_touch_sticky`
- Added calibrated sensor-health branch weighting from disagreement, contact confidence, tactile noise, contact dropout, and force/displacement consistency.
- Added CVaR/minimax-style scoring and contact-safety scoring.
- Added value-of-information probe gating.
- Added reliability fallback to conservative mean fusion under suspected sensor corruption.
- Added action/probe/fallback diagnostics and all-split paired comparisons.
- Fixed the v4 comparability bug where the environment seed included the method name. In v5, all methods share the same split/seed/episode world; method randomness is separate.

## Smoke Run

Command:

```powershell
python src\run_experiment.py --seeds 1 --episodes 1 --ablation-episodes 1 --splits nominal --ablation-splits combined_shift --skip-stress --workers 1 --results-dir results\dev_smoke_v5 --figures-dir figures\dev_smoke_v5
```

Outcome:

- Completed successfully.
- Produced 13 main rows and 15 ablation rows.
- Verified v5 I/O, plotting, raw rows, pairwise rows, and action diagnostics.

## Medium Development Run: Initial CVTB

Command:

```powershell
python src\run_experiment.py --seeds 2 --episodes 6 --ablation-episodes 6 --splits nominal combined_shift sensor_conflict delayed_touch_sticky --ablation-splits combined_shift sensor_conflict --skip-stress --workers 4 --results-dir results\dev_medium_v5 --figures-dir figures\dev_medium_v5
```

Outcome:

- Completed successfully.
- CVTB probed too often and underperformed `cvtb_no_probe`.
- Aggregate CVTB success: 0.5625.
- Aggregate no-probe CVTB success: 0.6250.
- Combined-shift CVTB success: 0.2500.
- Interpretation: the value-of-information gate was too permissive.

## Probe-Gate Repair

Change:

- Raised the probe-value threshold.
- Required enough tactile reliability and sticky-contact evidence before probing.
- Made reliability fallback less timid.

Command:

```powershell
python src\run_experiment.py --seeds 2 --episodes 6 --ablation-episodes 6 --splits nominal combined_shift sensor_conflict delayed_touch_sticky --ablation-splits combined_shift sensor_conflict --skip-stress --workers 4 --results-dir results\dev_medium_v5_repaired --figures-dir figures\dev_medium_v5_repaired
```

Outcome:

- Probes disappeared in the medium grid, avoiding the worst harmful probes.
- CVTB still underperformed mean fusion and vision-only on key splits.
- The run exposed a more serious comparability issue: method-specific environment seeds.

## Shared-Environment Seed Repair

Change:

- Environment generation now depends only on split, seed, episode, and stress level.
- Method-specific randomness is separated into a policy RNG.
- This makes paired comparisons meaningful and makes ablations face the same episodes.

Command:

```powershell
python src\run_experiment.py --seeds 2 --episodes 6 --ablation-episodes 6 --splits nominal combined_shift sensor_conflict delayed_touch_sticky --ablation-splits combined_shift sensor_conflict --skip-stress --workers 4 --results-dir results\dev_medium_v5_shared --figures-dir figures\dev_medium_v5_shared
```

Outcome:

- Completed successfully.
- CVTB aggregate success: 0.6250.
- Mean fusion aggregate success: 0.6875.
- Vision-only aggregate success: 0.7083.
- CVTB matched `cvtb_no_probe`, so the mechanism gate remained in danger.

## Mean-Anchor Repair

Change:

- Added the reliability-weighted mean anchor as an explicit CVTB candidate.
- This is a legitimate design repair because robust/mean baselines were solving cases that branch-only anchors missed.

Command:

```powershell
python src\run_experiment.py --seeds 2 --episodes 6 --ablation-episodes 6 --splits nominal combined_shift sensor_conflict delayed_touch_sticky --ablation-splits combined_shift sensor_conflict --skip-stress --workers 4 --results-dir results\dev_medium_v5_mean_anchor --figures-dir figures\dev_medium_v5_mean_anchor
```

Outcome:

- Completed successfully.
- CVTB aggregate success improved to 0.6667.
- CVTB improved combined-shift success to 0.5833, tying mean fusion and robust minimax and beating old VT.
- CVTB still tied `cvtb_no_probe` exactly and lost to vision-only in aggregate.
- Sensor-conflict ablations such as `no_sensor_health`, `no_value_of_information`, and old VT matched or beat CVTB.

## Pre-Freeze Decision

Freeze the mean-anchor CVTB scorer. Do not tune further.

Prior expectation for the frozen full run is `KILL_ARCHIVE`, because:

- CVTB still ties `cvtb_no_probe`.
- Sensor-health and value-of-information ablations are not reliably worse.
- Vision-only, mean fusion, robust minimax, and oracle remain dangerous.
- The mechanism is still not clearly identifiable on hostile sensor-conflict splits.

The final run must report these facts honestly.
