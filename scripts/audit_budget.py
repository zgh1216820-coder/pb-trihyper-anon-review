#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

FORBIDDEN_DIRS = {"data/raw", "out", "outputs", "checkpoints", "runs", "w" "andb", "model_saved", "__pycache__"}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".pkl", ".pyc", ".ipynb"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an anonymous release tree for large/generated artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-mb", type=float, default=50.0)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    problems = []
    total = 0
    for path in root.rglob("*"):
        rel = path.relative_to(root).as_posix()
        if ".git/" in rel or rel == ".git":
            continue
        if path.is_dir():
            if rel in FORBIDDEN_DIRS or path.name in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"}:
                problems.append(f"forbidden directory: {rel}")
            continue
        size = path.stat().st_size
        total += size
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden artifact: {rel}")
        if size > args.max_mb * 1024 * 1024:
            problems.append(f"large file > {args.max_mb:.1f}MB: {rel} ({size / 1024 / 1024:.1f}MB)")
    print(f"release_root={root}")
    print(f"total_file_size_mb={total / 1024 / 1024:.2f}")
    if problems:
        print("FAIL")
        for item in problems:
            print(item)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
