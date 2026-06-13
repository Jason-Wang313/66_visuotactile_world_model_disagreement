# Submission Readiness Decision

Decision: KILL_ARCHIVE

ICLR main-conference readiness: NO.

Reason: v4 adds a real MuJoCo contact-manipulation benchmark, but the evidence is negative. The proposed visuotactile branch planner is worse than mean fusion and ensemble uncertainty on combined shift, and ablations that remove the claimed branch mechanism match or beat the full method.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: invent and test a substantially different mechanism that clears ensemble uncertainty, mean fusion, diagnostic probing, and no-branch ablations on real robot or public high-fidelity benchmarks.
