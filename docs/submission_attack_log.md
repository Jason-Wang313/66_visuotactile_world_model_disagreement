# Submission Attack Log

Paper: 66 visuotactile_world_model_disagreement

This v4 pass applies the ICLR main-conference bar. The result is an honest archive decision, not a workshop resubmission.

## 2026-06-15 Continuation Note

The earlier v3 objections about synthetic-only evidence are no longer the active reason for archive. The repository now contains a real MuJoCo contact-manipulation benchmark with 3,780 main episodes, 420 ablation episodes, 1,200 stress-sweep episodes, strong baselines, pairwise tests, figures, and a rebuilt PDF. The current fatal problem is stronger: the branch-preserving visuotactile-disagreement mechanism loses to mean fusion and ensemble uncertainty on combined shift, while no-mechanism ablations match or beat the full method. Terminal decision remains `KILL_ARCHIVE`.

## ICLR Main Gate Round 1
Attack: No real-robot validation.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 2
Attack: No high-fidelity simulator validation.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 3
Attack: Synthetic benchmark is generated from a shared template.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 4
Attack: The mechanism is not empirically learned from real robot data.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 5
Attack: Baselines are synthetic probability models, not implemented competing systems.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 6
Attack: Prior-work threat set is metadata-derived and not a full manual related-work synthesis.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 7
Attack: All papers share nearly identical experiment code, weakening paper-specific novelty.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 8
Attack: No external benchmark comparison such as LIBERO, Meta-World, RLBench, BridgeData, or real manipulation suite.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 9
Attack: No hardware failure modes are measured.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 10
Attack: No learned representation, training curves, or model architecture is implemented.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 11
Attack: No ablation is attached to a real model component; ablations are synthetic knobs.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 12
Attack: No reviewer can reproduce a robotics system, only a diagnostic simulation.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 13
Attack: No statistical test on real deployment outcomes.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 14
Attack: No compute/data/model card for a trained WAM.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 15
Attack: No evidence that the branch atlas can be inferred from observations.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 16
Attack: No proof that the proposed mechanism beats strong real baselines.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 17
Attack: Potential novelty collision with world models, uncertainty planning, conformal filters, and model-based RL remains unresolved.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 18
Attack: The paper text is template-like across the batch.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 19
Attack: The PDF is better framed as an archive memo than an ICLR submission.

Verdict: Recoverable by rewriting honesty, not by claiming readiness.

Action: Rewrite as ICLR main gate archive.

## ICLR Main Gate Round 20
Attack: Main-conference claim validity fails.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 21
Attack: Advisor-name policy is respected but does not rescue technical evidence.

Verdict: Coverage probe only.

Action: Keep names weak and do not rank by them.

## ICLR Main Gate Round 22
Attack: Reproducibility is adequate for synthetic code but inadequate for robotics claims.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 23
Attack: No figures from real rollouts or model predictions.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 24
Attack: No dataset release beyond generated CSVs.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 25
Attack: No causal identification of the mechanism.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 26
Attack: No theoretical guarantee strong enough to replace empirical validation.

Verdict: Fatal for ICLR main unless new external evidence is produced.

Action: Not recoverable within the existing local artifacts; archive rather than overclaim.

## ICLR Main Gate Round 27
Attack: No meaningful recoverable ICLR-main issue remains after archiving.

Verdict: Terminal condition reached.

Action: Mark KILL_ARCHIVE and stop.
