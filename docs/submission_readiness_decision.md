# Submission Readiness Decision

Decision: `KILL_ARCHIVE`

ICLR main-conference readiness: no.

Reason: the expanded v5 MuJoCo rebuild gives the idea a stronger calibrated implementation, but the frozen evidence is still negative. CVTB-MPC is matched or beaten by strong baselines in aggregate and on hostile splits. Its no-probe variant ties the full method, and mechanism ablations without sensor-health, reliability fallback, CVaR tail, or mean-only fusion match or beat CVTB-MPC on the pre-registered hostile splits.

Evidence scale:

- Main raw rows: 24,960.
- Ablation raw rows: 8,640.
- Stress rows: 3,456.
- Seed summaries: 1,040.
- Paired comparisons: 120.
- Final PDF: 28 pages at `C:\Users\wangz\Downloads\66.pdf`.

Honest terminal action: archive/kill for ICLR main. Do not submit this paper to ICLR main in its current form.

Revival condition: develop a substantially different mechanism and freeze a new protocol that clears mean fusion, ensemble uncertainty, robust minimax, particle belief, diagnostic probing, no-probe, no-sensor-health, no-VOI, and no-tail-risk ablations on a public benchmark or real-robot validation setting.
