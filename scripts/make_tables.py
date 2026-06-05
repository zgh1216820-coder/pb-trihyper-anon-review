#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def markdown_table(rows):
    if not rows:
        return "No rows.\n"
    fields = list(rows[0].keys())
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build lightweight Markdown tables from expected/result CSVs.")
    parser.add_argument("--expected-dir", default="results/expected")
    parser.add_argument("--output-dir", default="outputs/tables")
    args = parser.parse_args()
    expected_dir = Path(args.expected_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(expected_dir.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSV files found in {expected_dir}")
    for csv_path in csv_paths:
        rows = read_csv(csv_path)
        out_path = out_dir / f"{csv_path.stem}.md"
        out_path.write_text(f"# {csv_path.stem}\n\n" + markdown_table(rows), encoding="utf-8")
        print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
