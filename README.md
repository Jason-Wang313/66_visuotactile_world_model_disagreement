# 66 Visuotactile World-Model Disagreement

Submission-hardening version: v3

Terminal decision: KILL_ARCHIVE for ICLR main conference.

The repository is retained as an archive of the generated idea, hostile review, synthetic stress-test scaffold, and reproducibility files. It is not an ICLR main-conference-ready robotics paper because it lacks real-robot/high-fidelity evidence and implemented learned baselines.

## Reproduce Synthetic Scaffold

```powershell
python src\run_experiment.py
```

## Rebuild Archive PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/66.pdf`
