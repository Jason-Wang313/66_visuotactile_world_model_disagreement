"""Validate Paper 66 expanded-standard artifacts."""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"
DOWNLOADS = Path.home() / "Downloads"
DESKTOP = Path.home() / "Desktop"


EXPECTED_COUNTS = {
    "vt_disagreement_raw.csv": 24960,
    "raw_seed_metrics.csv": 1040,
    "vt_disagreement_metrics.csv": 130,
    "vt_disagreement_pairwise.csv": 120,
    "vt_disagreement_ablation_raw.csv": 8640,
    "vt_disagreement_ablation.csv": 45,
    "stress_sweep.csv": 36,
    "negative_cases.csv": 3,
}

EXPECTED_FIGURES = [
    "vt_disagreement_success_by_split.png",
    "vt_disagreement_energy_by_split.png",
    "vt_disagreement_ablation_success.png",
    "vt_disagreement_stress_sweep.png",
    "stress_curve_data.csv",
]


def fail(message: str) -> None:
    raise SystemExit(f"VALIDATION FAILED: {message}")


def count_rows(path: Path) -> int:
    if not path.exists():
        fail(f"missing {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return sum(1 for _ in reader)


def validate_counts() -> None:
    for name, expected in EXPECTED_COUNTS.items():
        actual = count_rows(RESULTS / name)
        if actual != expected:
            fail(f"{name} row count {actual} != {expected}")
    summary = RESULTS / "summary.txt"
    if not summary.exists():
        fail("missing summary.txt")
    match = re.search(r"stress rows:\s*(\d+)", summary.read_text(encoding="utf-8"), re.IGNORECASE)
    if not match or int(match.group(1)) != 3456:
        fail("summary.txt does not record 3456 stress rows")


def validate_figures() -> None:
    for name in EXPECTED_FIGURES:
        path = FIGURES / name
        if not path.exists() or path.stat().st_size < 1000:
            fail(f"missing or tiny figure/data artifact {path}")


def pdf_page_count(path: Path) -> int:
    try:
        from pypdf import PdfReader  # type: ignore

        return len(PdfReader(str(path)).pages)
    except Exception:
        pass
    try:
        from PyPDF2 import PdfReader  # type: ignore

        return len(PdfReader(str(path)).pages)
    except Exception:
        pass
    data = path.read_bytes()
    return len(re.findall(rb"/Type\s*/Page\b", data))


def validate_pdf() -> None:
    pdf = DOWNLOADS / "66.pdf"
    if not pdf.exists() or pdf.stat().st_size < 100000:
        fail(f"missing or tiny final PDF {pdf}")
    pages = pdf_page_count(pdf)
    if pages < 25:
        fail(f"final PDF has {pages} pages, expected at least 25")
    desktop_pdf = DESKTOP / "66.pdf"
    if desktop_pdf.exists():
        fail(f"Desktop PDF is forbidden: {desktop_pdf}")


def validate_tex() -> None:
    tex = PAPER / "main.tex"
    if not tex.exists():
        fail("paper/main.tex missing")
    body = tex.read_text(encoding="utf-8")
    required = [
        "citebordercolor",
        "pdfborder",
        "\\bibliography{references}",
        "\\input{generated/result_macros.tex}",
        "\\input{generated/full_metrics_longtable.tex}",
    ]
    missing = [item for item in required if item not in body]
    if missing:
        fail("main.tex missing citation/link hardening or generated tables: " + ", ".join(missing))


def validate_compile() -> None:
    subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / "src" / "run_experiment.py")], check=True)
    subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / "scripts" / "render_submission_assets.py")], check=True)


def main() -> None:
    validate_compile()
    validate_counts()
    validate_figures()
    validate_tex()
    validate_pdf()
    print("Paper 66 validation passed: counts, figures, TeX links, Downloads PDF, and Desktop hygiene are OK.")


if __name__ == "__main__":
    main()
