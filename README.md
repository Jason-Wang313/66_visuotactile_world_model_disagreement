# 66 Visuotactile World-Model Disagreement

Submission-hardening version: v2

This repository contains a hardened, honest workshop-only robotics paper draft for `visuotactile_world_model_disagreement`.

## Reproduce Experiments

```powershell
python src\run_experiment.py
```

The script uses only the Python standard library and writes:

- `results/metrics.csv`
- `results/raw_seed_metrics.csv`
- `results/ablation_metrics.csv`
- `results/stress_sweep.csv`
- `results/negative_cases.csv`
- `figures/stress_curve_data.csv`

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF path: `C:/Users/wangz/Downloads/66.pdf`

## Readiness

Decision: workshop-only / strong-revise. The evidence is multi-seed and reviewer-hardened, but it remains synthetic and should not be overstated as real-robot state of the art.
