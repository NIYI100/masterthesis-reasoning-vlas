#!/usr/bin/env python3
"""
Move or merge misplaced reasoning-trace shard files.

Workers launched with --reasoning_trace_jsonl under experiments/logs/<DATE>/ while
--local_log_dir was extended to .../<DATE>/<suite>/<rollout>/ left shards at the date
root. This script moves them (or appends into the canonical copy if both exist).

Safe to run while jobs are still writing: re-run after jobs finish to catch stragglers.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

DATE_RE = re.compile(r"^\d{4}_\d{2}_\d{2}$")
SHARD_RE = re.compile(r"^(.+)_reasoning_trace\.shard\d+\.jsonl$")


def repair_logs_root(logs_root: Path, dry_run: bool) -> tuple[int, int, int]:
    moved = merged = skipped = 0
    for path in sorted(logs_root.rglob("*_reasoning_trace.shard*.jsonl")):
        if not path.is_file():
            continue
        parent = path.parent
        if not DATE_RE.match(parent.name):
            continue
        date_dir = parent
        m = SHARD_RE.match(path.name)
        if not m:
            continue
        rollout_name = m.group(1)
        dest_dir: Path | None = None
        for child in sorted(date_dir.iterdir()):
            if not child.is_dir() or DATE_RE.match(child.name):
                continue
            cand = child / rollout_name
            if cand.is_dir():
                dest_dir = cand
                break
        if dest_dir is None:
            dest_dir = date_dir / "libero_90" / rollout_name
            if not dry_run:
                dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / path.name
        if dest.resolve() == path.resolve():
            skipped += 1
            continue
        if not dest.exists():
            if dry_run:
                print(f"mv {path} -> {dest}")
            else:
                shutil.move(str(path), str(dest))
            moved += 1
            continue
        if dry_run:
            print(f"append {path} -> {dest}")
        else:
            extra = path.read_text(encoding="utf-8")
            if extra.strip():
                with open(dest, "a", encoding="utf-8") as out:
                    if not extra.endswith("\n"):
                        extra += "\n"
                    out.write(extra)
            path.unlink()
        merged += 1
    return moved, merged, skipped


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--logs-root",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "experiments" / "logs",
        help="experiments/logs directory (default: openvla-mini/experiments/logs)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions only.")
    args = p.parse_args()
    root = args.logs_root
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)
    moved, merged, skipped = repair_logs_root(root, args.dry_run)
    print(f"moved={moved} merged_appended={merged} skipped_already_ok={skipped}")


if __name__ == "__main__":
    main()
