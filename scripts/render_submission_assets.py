"""Render Paper 66 CSV evidence into LaTeX assets and audit summaries."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"
GENERATED = PAPER / "generated"
DOCS = ROOT / "docs"


METHOD_ORDER = [
    "random_push",
    "vision_only_mpc",
    "tactile_only_mpc",
    "mean_fusion_mpc",
    "ensemble_uncertainty_mpc",
    "conformal_risk_filter",
    "diagnostic_probe_then_mpc",
    "robust_minimax_mpc",
    "particle_belief_mpc",
    "old_vt_disagreement_branch_mpc",
    "cvtb_mpc_v5",
    "cvtb_no_probe",
    "oracle_mode_mpc",
]

SELECTED_METHODS = [
    "vision_only_mpc",
    "mean_fusion_mpc",
    "ensemble_uncertainty_mpc",
    "robust_minimax_mpc",
    "particle_belief_mpc",
    "old_vt_disagreement_branch_mpc",
    "cvtb_mpc_v5",
    "cvtb_no_probe",
    "oracle_mode_mpc",
]

HOSTILE_SPLITS = ["combined_shift", "sensor_conflict", "contact_dropout", "delayed_touch_sticky"]
ABLATED_MECHANISMS = [
    "cvtb_no_probe",
    "no_sensor_health",
    "no_value_of_information",
    "no_cvar_tail",
    "no_reliability_fallback",
    "mean_only_fusion",
    "old_vt_disagreement_branch_mpc",
]

LABELS = {
    "random_push": "Random",
    "vision_only_mpc": "Vision only",
    "tactile_only_mpc": "Tactile only",
    "mean_fusion_mpc": "Mean fusion",
    "ensemble_uncertainty_mpc": "Ensemble uncertainty",
    "conformal_risk_filter": "Conformal risk",
    "diagnostic_probe_then_mpc": "Diagnostic probe",
    "robust_minimax_mpc": "Robust minimax",
    "particle_belief_mpc": "Particle belief",
    "old_vt_disagreement_branch_mpc": "Old VT branch",
    "cvtb_mpc_v5": "CVTB-MPC",
    "cvtb_no_probe": "CVTB no-probe",
    "oracle_mode_mpc": "Oracle",
    "no_sensor_health": "No sensor health",
    "no_branch_preservation": "No branch preservation",
    "no_value_of_information": "No VOI gate",
    "no_cvar_tail": "No CVaR tail",
    "no_contact_safety": "No contact safety",
    "no_reliability_fallback": "No reliability fallback",
    "mean_only_fusion": "Mean-only fusion",
    "tactile_trust_high": "High tactile trust",
    "small_branch_set": "Small branch set",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def summary_stress_rows() -> int:
    path = RESULTS / "summary.txt"
    if not path.exists():
        return 0
    match = re.search(r"stress rows:\s*(\d+)", path.read_text(encoding="utf-8"), re.IGNORECASE)
    return int(match.group(1)) if match else 0


def f4(value: object) -> str:
    return f"{float(value):.4f}"


def f3(value: object) -> str:
    return f"{float(value):.3f}"


def pct(value: object) -> str:
    return f"{100.0 * float(value):.1f}"


def esc(text: object) -> str:
    out = str(text)
    return (
        out.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def label(method: str) -> str:
    return LABELS.get(method, method)


def display(text: object) -> str:
    return esc(str(text).replace("_", " "))


def method_rank(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def aggregate_by_method(metrics: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metrics:
        groups[row["method"]].append(row)
    out: list[dict[str, object]] = []
    for method, rows in groups.items():
        out.append(
            {
                "method": method,
                "splits": len(rows),
                "episodes": sum(int(float(row["episodes"])) for row in rows),
                "success": mean(float(row["mean_success"]) for row in rows),
                "error": mean(float(row["mean_final_error"]) for row in rows),
                "energy": mean(float(row["mean_energy"]) for row in rows),
                "contact": mean(float(row["mean_contact_violation"]) for row in rows),
                "entropy": mean(float(row["mean_branch_entropy"]) for row in rows),
                "diagnostic": mean(float(row["diagnostic_rate"]) for row in rows),
                "probe": mean(float(row["mean_probe_value"]) for row in rows),
                "fallback": mean(float(row["fallback_rate"]) for row in rows),
            }
        )
    return sorted(out, key=lambda row: method_rank(str(row["method"])))


def render_aggregate(metrics: list[dict[str, str]]) -> str:
    rows = aggregate_by_method(metrics)
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Aggregate frozen results across all ten visuotactile shifts. Success is higher-is-better; error, energy, and contact violation are lower-is-better.}",
        "\\label{tab:aggregate-main}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Method & Episodes & Success & Error & Energy & Contact & Entropy & Probe & Fallback \\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            f"{esc(label(str(row['method'])))} & {row['episodes']} & {f3(row['success'])} & "
            f"{f3(row['error'])} & {f3(row['energy'])} & {f3(row['contact'])} & "
            f"{f3(row['entropy'])} & {f3(row['probe'])} & {f3(row['fallback'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_selected_split_table(metrics: list[dict[str, str]]) -> str:
    by_key = {(row["split"], row["method"]): row for row in metrics}
    splits = sorted({row["split"] for row in metrics})
    header = "Split & Vision & Mean & Ensemble & Robust & Particle & Old VT & CVTB & No-probe & Oracle \\\\"
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Split-level success for strong baselines, the old branch planner, CVTB-MPC, its no-probe variant, and the oracle.}",
        "\\label{tab:selected-splits}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{lrrrrrrrrr}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for split in splits:
        vals = [f3(by_key[(split, method)]["mean_success"]) for method in SELECTED_METHODS]
        body.append(f"{esc(split)} & " + " & ".join(vals) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_diagnostic_table(metrics: list[dict[str, str]]) -> str:
    by_key = {(row["split"], row["method"]): row for row in metrics}
    splits = sorted({row["split"] for row in metrics})
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{CVTB-MPC mechanism diagnostics by split. Probe and fallback rates are frozen outputs, not post-hoc annotations.}",
        "\\label{tab:diagnostics}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Split & Success & Error & Entropy & Probe value & Diagnostic & Fallback \\\\",
        "\\midrule",
    ]
    for split in splits:
        row = by_key[(split, "cvtb_mpc_v5")]
        body.append(
            f"{esc(split)} & {f3(row['mean_success'])} & {f3(row['mean_final_error'])} & "
            f"{f3(row['mean_branch_entropy'])} & {f3(row['mean_probe_value'])} & "
            f"{f3(row['diagnostic_rate'])} & {f3(row['fallback_rate'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_ablation(ablation: list[dict[str, str]]) -> str:
    rows = sorted(ablation, key=lambda row: (row["split"], method_rank(row["method"]), row["method"]))
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Frozen ablation results on the three pre-registered hostile splits. Ablations matching CVTB-MPC are mechanism failures, not robustness wins.}",
        "\\label{tab:ablation}",
        "\\scriptsize",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Split & Method & Episodes & Success & Error & Energy & Contact & Fallback \\\\",
        "\\midrule",
    ]
    for row in rows:
        method = row["method"]
        body.append(
            f"{esc(row['split'])} & {esc(label(method))} & {int(float(row['episodes']))} & "
            f"{f3(row['mean_success'])} & {f3(row['mean_final_error'])} & "
            f"{f3(row['mean_energy'])} & {f3(row['mean_contact_violation'])} & "
            f"{f3(row['fallback_rate'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_pairwise(pairwise: list[dict[str, str]]) -> str:
    keep = {
        "vision_only_mpc",
        "mean_fusion_mpc",
        "ensemble_uncertainty_mpc",
        "conformal_risk_filter",
        "diagnostic_probe_then_mpc",
        "robust_minimax_mpc",
        "particle_belief_mpc",
        "old_vt_disagreement_branch_mpc",
        "cvtb_no_probe",
    }
    rows = [row for row in pairwise if row["split"] in HOSTILE_SPLITS and row["baseline"] in keep]
    rows.sort(key=lambda row: (HOSTILE_SPLITS.index(row["split"]), method_rank(row["baseline"])))
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Paired CVTB-MPC deltas on hostile splits. Positive success/error/energy columns favor CVTB-MPC; contact delta is CVTB minus baseline.}",
        "\\label{tab:hostile-pairwise}",
        "\\scriptsize",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Split & Baseline & $\\Delta$ succ. & Error impr. & Energy impr. & Contact $\\Delta$ & Action diff. & $p$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        body.append(
            f"{esc(row['split'])} & {esc(label(row['baseline']))} & "
            f"{f3(row['success_delta_mean'])} & {f3(row['final_error_improvement_mean'])} & "
            f"{f3(row['energy_improvement_mean'])} & {f3(row['contact_violation_delta_mean'])} & "
            f"{pct(row['action_diff_rate'])}\\% & {f3(row['normal_approx_p'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_stress(stress: list[dict[str, str]]) -> str:
    levels = sorted({row["stress_level"] for row in stress}, key=float)
    methods = ["mean_fusion_mpc", "ensemble_uncertainty_mpc", "robust_minimax_mpc", "old_vt_disagreement_branch_mpc", "cvtb_mpc_v5", "oracle_mode_mpc"]
    by_key = {(row["method"], row["stress_level"]): row for row in stress}
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Stress-sweep success as visual bias, tactile noise, and friction shift increase together.}",
        "\\label{tab:stress}",
        "\\scriptsize",
        "\\begin{tabular}{l" + "r" * len(levels) + "}",
        "\\toprule",
        "Method & " + " & ".join(esc(level) for level in levels) + " \\\\",
        "\\midrule",
    ]
    for method in methods:
        vals = [f3(by_key[(method, level)]["mean_success"]) if (method, level) in by_key else "--" for level in levels]
        body.append(f"{esc(label(method))} & " + " & ".join(vals) + " \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def render_full_metrics(metrics: list[dict[str, str]]) -> str:
    rows = sorted(metrics, key=lambda row: (row["split"], method_rank(row["method"])))
    body = [
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{longtable}{@{}p{0.15\\linewidth}p{0.17\\linewidth}rrrrrrrr@{}}",
        "\\caption{Complete frozen split-method metric table used for the terminal decision.}\\label{tab:full-metrics}\\\\",
        "\\toprule",
        "Split & Method & Ep. & Succ. & Err. & Eng. & Viol. & Ent. & Diag. & Fb. \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Split & Method & Ep. & Succ. & Err. & Eng. & Viol. & Ent. & Diag. & Fb. \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in rows:
        body.append(
            f"{display(row['split'])} & {display(label(row['method']))} & {int(float(row['episodes']))} & "
            f"{f3(row['mean_success'])} & {f3(row['mean_final_error'])} & {f3(row['mean_energy'])} & "
            f"{f3(row['mean_contact_violation'])} & {f3(row['mean_branch_entropy'])} & "
            f"{f3(row['diagnostic_rate'])} & {f3(row['fallback_rate'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{longtable}", "}", ""]
    return "\n".join(body)


def render_full_pairwise(pairwise: list[dict[str, str]]) -> str:
    rows = sorted(pairwise, key=lambda row: (row["split"], method_rank(row["baseline"])))
    body = [
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{longtable}{@{}p{0.15\\linewidth}p{0.18\\linewidth}rrrrrr@{}}",
        "\\caption{Complete paired CVTB-MPC comparison table.}\\label{tab:full-pairwise}\\\\",
        "\\toprule",
        "Split & Baseline & Pairs & $\\Delta$ succ. & Err. impr. & Eng. impr. & Viol. $\\Delta$ & Diff. \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Split & Baseline & Pairs & $\\Delta$ succ. & Err. impr. & Eng. impr. & Viol. $\\Delta$ & Diff. \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in rows:
        body.append(
            f"{display(row['split'])} & {display(label(row['baseline']))} & {int(float(row['paired_episodes']))} & "
            f"{f3(row['success_delta_mean'])} & {f3(row['final_error_improvement_mean'])} & "
            f"{f3(row['energy_improvement_mean'])} & {f3(row['contact_violation_delta_mean'])} & "
            f"{pct(row['action_diff_rate'])}\\% \\\\"
        )
    body += ["\\bottomrule", "\\end{longtable}", "}", ""]
    return "\n".join(body)


def render_seed_metrics(seed_metrics: list[dict[str, str]]) -> str:
    keep = {"mean_fusion_mpc", "ensemble_uncertainty_mpc", "robust_minimax_mpc", "old_vt_disagreement_branch_mpc", "cvtb_mpc_v5", "cvtb_no_probe", "oracle_mode_mpc"}
    rows = [row for row in seed_metrics if row["method"] in keep]
    rows.sort(key=lambda row: (row["split"], int(float(row["seed"])), method_rank(row["method"])))
    body = [
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{longtable}{@{}p{0.15\\linewidth}p{0.18\\linewidth}rrrrrr@{}}",
        "\\caption{Seed-level robustness table for selected strong baselines, CVTB-MPC, its no-probe variant, and the oracle.}\\label{tab:seed-metrics}\\\\",
        "\\toprule",
        "Split & Method & Seed & Ep. & Succ. & Err. & Eng. & Fb. \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\toprule",
        "Split & Method & Seed & Ep. & Succ. & Err. & Eng. & Fb. \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for row in rows:
        body.append(
            f"{display(row['split'])} & {display(label(row['method']))} & {int(float(row['seed']))} & "
            f"{int(float(row['episodes']))} & {f3(row['mean_success'])} & "
            f"{f3(row['mean_final_error'])} & {f3(row['mean_energy'])} & {f3(row['fallback_rate'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{longtable}", "}", ""]
    return "\n".join(body)


def render_failure_cases(cases: list[dict[str, str]]) -> str:
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Pre-registered limitations and negative cases.}",
        "\\label{tab:negative-cases}",
        "\\small",
        "\\begin{tabular}{p{0.27\\linewidth}p{0.35\\linewidth}p{0.28\\linewidth}}",
        "\\toprule",
        "Case & Observed failure mode & Submission implication \\\\",
        "\\midrule",
    ]
    for row in cases:
        body.append(
            f"{display(row['case'])} & {esc(row['observed_failure_mode'])} & {esc(row['submission_implication'])} \\\\"
        )
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body)


def decision_gates(metrics: list[dict[str, str]], ablation: list[dict[str, str]]) -> tuple[list[tuple[str, str, str]], str, str]:
    aggregates = {str(row["method"]): row for row in aggregate_by_method(metrics)}
    proposed = aggregates["cvtb_mpc_v5"]
    weak_methods = ["random_push", "tactile_only_mpc", "old_vt_disagreement_branch_mpc"]
    weak_failures = [
        label(method)
        for method in weak_methods
        if not (
            float(proposed["success"]) > float(aggregates[method]["success"]) + 1e-9
            and float(proposed["error"]) < float(aggregates[method]["error"]) - 1e-9
        )
    ]
    strong_methods = [
        "vision_only_mpc",
        "mean_fusion_mpc",
        "ensemble_uncertainty_mpc",
        "conformal_risk_filter",
        "diagnostic_probe_then_mpc",
        "robust_minimax_mpc",
        "particle_belief_mpc",
    ]
    strong_failures = [
        label(method)
        for method in strong_methods
        if float(aggregates[method]["success"]) > float(proposed["success"]) + 1e-9
        or float(aggregates[method]["error"]) < float(proposed["error"]) - 1e-9
    ]
    by_split_method = {(row["split"], row["method"]): row for row in metrics}
    hostile_failures = []
    for split in HOSTILE_SPLITS:
        p = by_split_method.get((split, "cvtb_mpc_v5"))
        if not p:
            continue
        for method in strong_methods:
            b = by_split_method.get((split, method))
            if b and float(b["mean_success"]) > float(p["mean_success"]) + 1e-9:
                hostile_failures.append(f"{label(method)} on {split}")
                break
    safest = min(float(aggregates[method]["contact"]) for method in strong_methods)
    contact_fail = float(proposed["contact"]) > safest + 0.02
    ab_by_key = {(row["split"], row["method"]): row for row in ablation}
    mechanism_failures = []
    for split in sorted({row["split"] for row in ablation}):
        p = ab_by_key.get((split, "cvtb_mpc_v5"))
        if not p:
            continue
        for method in ABLATED_MECHANISMS:
            b = ab_by_key.get((split, method))
            if b and float(b["mean_success"]) >= float(p["mean_success"]) - 1e-9:
                mechanism_failures.append(f"{label(method)} on {split}")
    failures = weak_failures + strong_failures + hostile_failures + mechanism_failures
    decision = "KILL_ARCHIVE" if failures or contact_fail else "STRONG_REVISE"
    reason = (
        "strong baselines or ablations match/beat CVTB-MPC: " + "; ".join((strong_failures + hostile_failures + mechanism_failures)[:8])
        if decision == "KILL_ARCHIVE"
        else "local frozen gates pass, but the package still lacks public benchmark or real-robot validation"
    )
    rows = [
        (
            "Weak baseline gate",
            "FAIL" if weak_failures else "PASS",
            "Failures: " + ", ".join(weak_failures) if weak_failures else "CVTB-MPC beats random, tactile-only, and old VT in aggregate success/error.",
        ),
        (
            "Strong baseline gate",
            "FAIL" if strong_failures else "PASS",
            "Matched or beaten by " + ", ".join(strong_failures[:6]) if strong_failures else "No aggregate strong-baseline failure.",
        ),
        (
            "Hostile split gate",
            "FAIL" if hostile_failures else "PASS",
            "Lost hostile splits: " + "; ".join(hostile_failures[:6]) if hostile_failures else "No strong baseline has higher success on the four hostile splits.",
        ),
        (
            "Contact safety gate",
            "FAIL" if contact_fail else "PASS",
            f"CVTB contact violation {f3(proposed['contact'])}; safest strong baseline {f3(safest)}.",
        ),
        (
            "Mechanism ablation gate",
            "FAIL" if mechanism_failures else "PASS",
            "Non-identifying ablations: " + "; ".join(mechanism_failures[:7]) if mechanism_failures else "Sensor-health, VOI, and tail-risk terms are separated.",
        ),
        ("Terminal decision", decision, reason),
    ]
    return rows, decision, reason


def render_gate_table(metrics: list[dict[str, str]], ablation: list[dict[str, str]]) -> tuple[str, str, str]:
    rows, decision, reason = decision_gates(metrics, ablation)
    body = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Frozen decision gates. The decision is evidence-driven rather than presentation-driven.}",
        "\\label{tab:gates}",
        "\\small",
        "\\begin{tabular}{p{0.22\\linewidth}p{0.13\\linewidth}p{0.55\\linewidth}}",
        "\\toprule",
        "Gate & Status & Evidence \\\\",
        "\\midrule",
    ]
    for gate, status, evidence in rows:
        body.append(f"{esc(gate)} & \\textbf{{{esc(status.replace('_', '-'))}}} & {esc(evidence)} \\\\")
    body += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(body), decision, reason


def render_macros(metrics: list[dict[str, str]], ablation: list[dict[str, str]], pairwise: list[dict[str, str]], stress: list[dict[str, str]], decision: str) -> str:
    aggregates = {str(row["method"]): row for row in aggregate_by_method(metrics)}
    cvtb = aggregates["cvtb_mpc_v5"]
    mean_fusion = aggregates["mean_fusion_mpc"]
    ensemble = aggregates["ensemble_uncertainty_mpc"]
    robust = aggregates["robust_minimax_mpc"]
    particle = aggregates["particle_belief_mpc"]
    old_vt = aggregates["old_vt_disagreement_branch_mpc"]
    oracle = aggregates["oracle_mode_mpc"]
    return "\n".join(
        [
            "% Auto-generated by scripts/render_submission_assets.py",
            f"\\newcommand{{\\PaperDecision}}{{\\textsc{{{decision.replace('_', '-')}}}}}",
            f"\\newcommand{{\\MainRows}}{{{count_rows(RESULTS / 'vt_disagreement_raw.csv'):,}}}",
            f"\\newcommand{{\\AblationRows}}{{{count_rows(RESULTS / 'vt_disagreement_ablation_raw.csv'):,}}}",
            f"\\newcommand{{\\StressRows}}{{{summary_stress_rows():,}}}",
            f"\\newcommand{{\\MetricRows}}{{{len(metrics)}}}",
            f"\\newcommand{{\\SeedRows}}{{{count_rows(RESULTS / 'raw_seed_metrics.csv'):,}}}",
            f"\\newcommand{{\\PairwiseRows}}{{{len(pairwise)}}}",
            f"\\newcommand{{\\AblationSummaryRows}}{{{len(ablation)}}}",
            f"\\newcommand{{\\StressSummaryRows}}{{{len(stress)}}}",
            f"\\newcommand{{\\CVTBAggregateSuccess}}{{{f3(cvtb['success'])}}}",
            f"\\newcommand{{\\MeanAggregateSuccess}}{{{f3(mean_fusion['success'])}}}",
            f"\\newcommand{{\\EnsembleAggregateSuccess}}{{{f3(ensemble['success'])}}}",
            f"\\newcommand{{\\RobustAggregateSuccess}}{{{f3(robust['success'])}}}",
            f"\\newcommand{{\\ParticleAggregateSuccess}}{{{f3(particle['success'])}}}",
            f"\\newcommand{{\\OldVTAggregateSuccess}}{{{f3(old_vt['success'])}}}",
            f"\\newcommand{{\\OracleAggregateSuccess}}{{{f3(oracle['success'])}}}",
            f"\\newcommand{{\\CVTBAggregateError}}{{{f3(cvtb['error'])}}}",
            f"\\newcommand{{\\MeanAggregateError}}{{{f3(mean_fusion['error'])}}}",
            f"\\newcommand{{\\EnsembleAggregateError}}{{{f3(ensemble['error'])}}}",
            f"\\newcommand{{\\RobustAggregateError}}{{{f3(robust['error'])}}}",
            f"\\newcommand{{\\ParticleAggregateError}}{{{f3(particle['error'])}}}",
            f"\\newcommand{{\\OldVTAggregateError}}{{{f3(old_vt['error'])}}}",
            f"\\newcommand{{\\CVTBAggregateEnergy}}{{{f3(cvtb['energy'])}}}",
            f"\\newcommand{{\\MeanAggregateEnergy}}{{{f3(mean_fusion['energy'])}}}",
            f"\\newcommand{{\\EnsembleAggregateEnergy}}{{{f3(ensemble['energy'])}}}",
            f"\\newcommand{{\\RobustAggregateEnergy}}{{{f3(robust['energy'])}}}",
            "",
        ]
    )


def write_decision_markdown(metrics: list[dict[str, str]], ablation: list[dict[str, str]], pairwise: list[dict[str, str]], stress: list[dict[str, str]], decision: str, reason: str) -> None:
    aggregates = {str(row["method"]): row for row in aggregate_by_method(metrics)}
    cvtb = aggregates["cvtb_mpc_v5"]
    lines = [
        "# Paper 66 Expanded-Standard Terminal Decision",
        "",
        "Date: 2026-06-20",
        "",
        f"Decision: `{decision}`",
        "",
        f"Reason: {reason}",
        "",
        "## Evidence Scale",
        "",
        f"- Main raw rows: {count_rows(RESULTS / 'vt_disagreement_raw.csv'):,}.",
        f"- Ablation raw rows: {count_rows(RESULTS / 'vt_disagreement_ablation_raw.csv'):,}.",
        f"- Stress raw rows recorded in summary: {summary_stress_rows():,}.",
        f"- Main split-method summaries: {len(metrics)}.",
        f"- Seed-level summaries: {count_rows(RESULTS / 'raw_seed_metrics.csv'):,}.",
        f"- Paired comparisons: {len(pairwise)}.",
        f"- Stress summary rows: {len(stress)}.",
        "",
        "## Aggregate Result",
        "",
        f"- CVTB-MPC aggregate success: {f3(cvtb['success'])}.",
        f"- CVTB-MPC aggregate final error: {f3(cvtb['error'])}.",
        f"- CVTB-MPC aggregate energy: {f3(cvtb['energy'])}.",
        "",
        "## Artifact Rule",
        "",
        "- The final PDF must remain `C:\\Users\\wangz\\Downloads\\66.pdf` only.",
        "- No PDF should be placed on the visible Desktop.",
        "",
    ]
    write(DOCS / "paper66_expanded_terminal_decision_20260620.md", "\n".join(lines))


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(RESULTS / "vt_disagreement_metrics.csv")
    seed_rows = read_csv(RESULTS / "raw_seed_metrics.csv")
    ablation = read_csv(RESULTS / "vt_disagreement_ablation.csv")
    pairwise = read_csv(RESULTS / "vt_disagreement_pairwise.csv")
    stress = read_csv(RESULTS / "stress_sweep.csv")
    cases = read_csv(RESULTS / "negative_cases.csv")
    gate_table, decision, reason = render_gate_table(metrics, ablation)
    write(GENERATED / "aggregate_metrics_table.tex", render_aggregate(metrics))
    write(GENERATED / "selected_split_table.tex", render_selected_split_table(metrics))
    write(GENERATED / "diagnostic_table.tex", render_diagnostic_table(metrics))
    write(GENERATED / "ablation_table.tex", render_ablation(ablation))
    write(GENERATED / "hostile_pairwise_table.tex", render_pairwise(pairwise))
    write(GENERATED / "stress_table.tex", render_stress(stress))
    write(GENERATED / "full_metrics_longtable.tex", render_full_metrics(metrics))
    write(GENERATED / "full_pairwise_longtable.tex", render_full_pairwise(pairwise))
    write(GENERATED / "seed_metrics_selected_longtable.tex", render_seed_metrics(seed_rows))
    write(GENERATED / "negative_cases_table.tex", render_failure_cases(cases))
    write(GENERATED / "gate_table.tex", gate_table)
    write(GENERATED / "result_macros.tex", render_macros(metrics, ablation, pairwise, stress, decision))
    write_decision_markdown(metrics, ablation, pairwise, stress, decision, reason)
    print(f"Rendered Paper 66 submission assets with decision={decision}")


if __name__ == "__main__":
    main()
