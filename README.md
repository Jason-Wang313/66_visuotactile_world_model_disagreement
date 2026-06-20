# 66 Visuotactile World-Model Disagreement

Expanded-standard rebuild: v5 calibrated MuJoCo evidence package.

Terminal decision: `KILL_ARCHIVE` for ICLR main conference.

This repository is retained as a reproducible archive of a falsified robotics mechanism. The v5 rebuild tests Calibrated Visuotactile Branch-and-Probe MPC (CVTB-MPC), a stronger version of the original disagreement-branch idea with sensor-health branches, value-of-information probe gating, CVaR-style tail scoring, contact-safety penalties, and conservative fallback under suspected sensor corruption.

The method does not survive hostile review. The frozen full run contains 24,960 main MuJoCo episode-method rows, 8,640 ablation rows, 3,456 stress rows, 1,040 seed summaries, and 120 paired comparisons. CVTB-MPC reaches aggregate success 0.777, while mean fusion reaches 0.796, particle belief reaches 0.790, robust minimax reaches 0.784, and the oracle reaches 0.908. The no-probe variant matches CVTB-MPC, and ablations without the claimed sensor-health/value-of-information mechanism match or beat it on hostile splits.

## Reproduce Frozen Evidence

```powershell
python src\run_experiment.py --seeds 8 --episodes 24 --ablation-episodes 24 --stress-episodes 12 --splits nominal high_friction low_friction vision_bias tactile_noise sticky_contact combined_shift sensor_conflict contact_dropout delayed_touch_sticky --ablation-splits combined_shift sensor_conflict delayed_touch_sticky --stress-levels 0.0 0.2 0.4 0.6 0.8 1.0 --workers 4 --results-dir results --figures-dir figures
```

The run writes raw MuJoCo rollouts, split metrics, seed metrics, pairwise comparisons, ablations, stress sweeps, negative cases, and figures into `results/` and `figures/`.

## Build and Validate PDF

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_submission_pdf.ps1
python scripts\validate_submission_artifacts.py
```

Canonical numbered PDF: `C:\Users\wangz\Downloads\66.pdf`.

Do not copy numbered PDFs to the visible Desktop.
