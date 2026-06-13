# Experiment Rigor Checklist

## v2 Synthetic Rigor
- [x] Multiple seeds.
- [x] Error bars.
- [x] Stronger synthetic baselines.
- [x] Ablations.
- [x] Stress tests.
- [x] Negative cases.

## ICLR Main Bar
- [ ] Real-robot validation.
- [x] High-fidelity simulator benchmark.
- [ ] Implemented learned model.
- [x] Implemented real competing baselines.
- [ ] Manual related-work synthesis.
- [x] Paper-specific empirical figures.
- [x] Multiple seeds, uncertainty, ablations, stress sweep, pairwise comparisons, and negative cases.

Decision: fail ICLR main empirical-rigor gate for a stronger reason than v3. Real MuJoCo evidence now exists, but it falsifies the mechanism because branch preservation is not better than mean fusion, ensemble uncertainty, or no-branch/no-disagreement ablations.
