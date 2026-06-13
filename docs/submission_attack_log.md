# Submission Attack Log

Paper: 66 visuotactile_world_model_disagreement

Protocol: harsh reviewer pass, stopping after the first round where no new recoverable issue remained.

## Round 1
Attack: Novelty is just uncertainty.

Fix: Narrowed claim to explicit branch atlas plus diagnostic action value; uncertainty-only baseline added.

## Round 2
Attack: Closest WAM/world-model papers already solve it.

Fix: Hostile citations retained and novelty boundary framed as branch-valued planning object, not generic prediction.

## Round 3
Attack: Baselines are too weak.

Fix: Added data augmentation, scalar uncertainty, ensemble disagreement, conformal risk filter, and oracle upper bound.

## Round 4
Attack: No random seeds.

Fix: Added seven seeds with 600 episodes per seed.

## Round 5
Attack: No uncertainty estimates.

Fix: Added 95 percent confidence intervals across seeds.

## Round 6
Attack: No ablations.

Fix: Added branch atlas, tail-risk, diagnostic-probe, and failure-memory ablations.

## Round 7
Attack: No stress tests.

Fix: Added hidden-mode, embodiment-shift, combined-stress, and stress-sweep evaluations.

## Round 8
Attack: Claims imply hardware SOTA.

Fix: Paper now explicitly disclaims real-robot SOTA and marks workshop-only.

## Round 9
Attack: Negative cases are hidden.

Fix: Added unmodeled damage, semantic ambiguity, and sensor-dropout negative cases.

## Round 10
Attack: Advisor-name bias could distort prior work.

Fix: Advisor names documented as weak coverage probes only.

## Round 11
Attack: Reproducibility is underspecified.

Fix: Added README, checklists, exact commands, and generated result files.

## Round 12
Attack: Figures could mislead.

Fix: No decorative figure claim; stress data is provided as CSV and tables include CIs.

## Round 13
Attack: Statistical power is unclear.

Fix: Episodes/seeds are listed in metrics and reproducibility docs.

## Round 14
Attack: Mechanism may be benchmark-only.

Fix: Limitations identify the need for learned branch discovery and hardware validation.

## Round 15
Attack: Code might not run from clone.

Fix: README uses stdlib Python and pdflatex/BibTeX commands; experiment was executed locally.

## Round 16
Attack: PDF versioning is ambiguous.

Fix: Paper includes visible submission-hardening version v2.

## Round 17
Attack: Repo state may miss final audit.

Fix: Docs include submission decision, attack log, and checklists committed to GitHub.

## Round 18
Attack: Any remaining fatal flaw?

Fix: No additional recoverable issue found without new hardware data; stop before fabricating claims.
