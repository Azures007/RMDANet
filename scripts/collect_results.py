#!/usr/bin/env python3
"""Collect OA/AA/Kappa from result JSON/log files into CSV + Markdown table.

Usage:
    python scripts/collect_results.py <results_root> [--out results_comparison/summary]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

OA_RE = re.compile(r"(?i)\bOA\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")
AA_RE = re.compile(r"(?i)\bAA\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")
KAPPA_RE = re.compile(r"(?i)kappa\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")


def parse_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = {}
    for key in ("OA", "oa", "overall_accuracy", "overall accuracy"):
        if key in payload:
            metrics["OA"] = float(payload[key])
            break
    for key in ("AA", "aa", "average_accuracy"):
        if key in payload:
            metrics["AA"] = float(payload[key])
            break
    for key in ("kappa", "Kappa", "cohen_kappa"):
        if key in payload:
            metrics["Kappa"] = float(payload[key])
            break
    return metrics or None


def parse_log(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="ignore")[-200000:]
    out = {}
    for name, regex in (("OA", OA_RE), ("AA", AA_RE), ("Kappa", KAPPA_RE)):
        matches = regex.findall(text)
        if matches:
            out[name] = float(matches[-1])
    return out or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--out", default="results_comparison/summary")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in sorted(root.rglob("*")):
        if path.suffix == ".json":
            metrics = parse_json(path)
        elif path.suffix == ".log":
            metrics = parse_log(path)
        else:
            continue
        if not metrics:
            continue
        relative = path.relative_to(root)
        rows.append({
            "file": str(relative),
            "method": relative.parts[0] if len(relative.parts) > 1 else relative.stem,
            "OA": metrics.get("OA", ""),
            "AA": metrics.get("AA", ""),
            "Kappa": metrics.get("Kappa", ""),
        })

    csv_path = out_base.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "method", "OA", "AA", "Kappa"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = out_base.with_suffix(".md")
    lines = ["| method | OA | AA | Kappa | file |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['OA']} | {row['AA']} | {row['Kappa']} | {row['file']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"collected {len(rows)} records -> {csv_path} / {md_path}")


if __name__ == "__main__":
    main()
