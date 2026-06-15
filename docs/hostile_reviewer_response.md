# Hostile Reviewer Response

Paper: 66 Visuotactile World-Model Disagreement

## Strongest Technical Threats

- Multimodal AI: PaLM-E's Role in Vision-Language-Robotics & the Future of Efficient Fine-Tuning (2026)
- MIRSAM: multimodal vision-language segment anything model for infrared small target detection (2025)
- Vision Language Models in Healthcare Through a Multimodal Approach to Medical Imaging and Clinical Applications (2026)
- EveryDayVLA: A Vision-Language-Action Model for Affordable Robotic Manipulation (2025)
- VLA-MP: A Vision-Language-Action Framework for Multimodal Perception and Physics-Constrained Action Generation in Autonomous Driving (2025)
- OmDet: Large-scale vision-language multi-dataset pre-training with multimodal detection network (2024)
- Application of multimodal learning in robotic perception: an intelligent perception framework integrating vision, sound, and touch (2026)
- ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model (2025)

## ICLR Main Response

A hostile ICLR reviewer would be correct to reject this as a main-conference submission. The v4 rebuild contains a real MuJoCo contact-manipulation benchmark with hidden friction, visual bias, tactile noise, sticky contact, combined shift, multiple seeds, strong baselines, ablations, stress sweeps, and uncertainty. That stronger evidence does not rescue the paper: the proposed branch-preserving planner loses to mean fusion and ensemble uncertainty on the combined shift, and no-mechanism ablations match or beat the full method.

## Honest Action

The paper is marked `KILL_ARCHIVE`. This avoids converting a falsified mechanism into an overstated main-conference claim.

## What Would Be Needed To Revive

- A substantially different mechanism that clears mean fusion, ensemble uncertainty, diagnostic probing, and no-branch ablations.
- Real robot or public high-fidelity contact-rich manipulation experiments.
- A learned world-action model or released checkpoint showing the representation is useful beyond analytic planner knobs.
- Manual full-paper related-work audit.
- Evidence that the core mechanism is necessary under deployment shift.
