#!/usr/bin/env python3
"""
Read model-level p-values from elastic net nohup logs and compute
Benjamini-Hochberg FDR-adjusted p-values.
"""

from pathlib import Path
import re
from typing import Iterable, List


LOG_GLOB_PATTERNS = ["nohup_*_t1.out", "nohup_*_t1_slope.out"]
TRAINING_PATTERN = re.compile(
    r"Training EN on (?P<feature_set>[^ ]+) features .* connectivity_type='(?P<connectivity_type>[^']+)'"
)
P_VALUE_PATTERN = re.compile(r"Model-level p-value:\s*(?P<p_value>[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def bh_fdr(pvals: Iterable[float]) -> List[float]:
    """Return BH FDR adjusted p-values in original order."""
    pvals = list(pvals)
    m = len(pvals)
    if m == 0:
        return []

    for p in pvals:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"All p-values must be in [0, 1]. Found: {p}")

    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted_sorted = [0.0] * m

    for rank, (_, p) in enumerate(indexed, start=1):
        adjusted_sorted[rank - 1] = p * m / rank

    for i in range(m - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    adjusted = [0.0] * m
    for i, (original_idx, _) in enumerate(indexed):
        adjusted[original_idx] = min(adjusted_sorted[i], 1.0)

    return adjusted


def collect_p_values(log_dir: Path):
    """Collect model-level p-values and labels from elastic net nohup logs.

    Args:
        log_dir: Directory containing the nohup log files.

    Returns:
        A list of dictionaries with the parsed label, p-value, and source file.
    """
    rows = []
    for pattern in LOG_GLOB_PATTERNS:
        for path in sorted(log_dir.glob(pattern)):
            text = path.read_text()
            training_match = TRAINING_PATTERN.search(text)
            p_value_match = P_VALUE_PATTERN.search(text)
            if training_match is None or p_value_match is None:
                continue

            connectivity_type = training_match.group("connectivity_type")
            feature_set = training_match.group("feature_set")
            label = f"{connectivity_type} {feature_set}"

            rows.append({
                "label": label,
                "p_value": float(p_value_match.group("p_value")),
                "file": path.name,
            })

    return rows


def main() -> None:
    """Parse nohup logs and print raw and FDR-adjusted model p-values."""
    log_dir = Path(__file__).resolve().parent / "results"
    rows = collect_p_values(log_dir)
    if not rows:
        raise SystemExit("No matching nohup logs with model-level p-values were found.")

    qvals = bh_fdr([row["p_value"] for row in rows])

    print(f"{'Index':>5}  {'Label':<12}  {'Raw p-value':>12}  {'FDR-adjusted':>12}  File")
    print("-" * 78)
    for i, (row, qval) in enumerate(zip(rows, qvals), start=1):
        print(
            f"{i:>5}  {row['label']:<12}  {row['p_value']:>12.6g}  {qval:>12.6g}  {row['file']}"
        )


if __name__ == "__main__":
    main()
