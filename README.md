# 66 Visuotactile World-Model Disagreement

Submission-hardening version: v4 real MuJoCo rebuild

Terminal decision: KILL_ARCHIVE for ICLR main conference.

The repository is retained as an archive of a falsified mechanism-level robotics idea. The v4 rebuild replaces the synthetic probability scaffold with a MuJoCo planar contact-manipulation benchmark containing hidden friction, vision-bias, tactile-noise, sticky-contact, and combined-shift modes.

The proposed visuotactile disagreement branch planner does not survive the ICLR-main gate. On combined shift it reaches 0.450 success, while mean fusion reaches 0.667, ensemble uncertainty reaches 0.733, and the oracle reaches 0.800. Ablations without the claimed branch mechanism match or beat the full method.

## Reproduce Real Evidence

```powershell
python src\run_experiment.py
```

The run writes raw MuJoCo rollouts, seed metrics, pairwise comparisons, ablations, stress sweeps, negative cases, and figures into `results/` and `figures/`.

## Rebuild Archive PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/66.pdf`
