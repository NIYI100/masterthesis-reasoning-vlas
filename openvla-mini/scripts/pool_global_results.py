#!/usr/bin/env python3
"""
Pool *GLOBAL-RESULTS.json from sibling rollout folders named ``*_20_rollouts`` and ``*_30_rollouts``.

For each pair that shares the same parent directory and the same basename (e.g.
``vanilla_eval_20_rollouts`` + ``vanilla_eval_30_rollouts`` → base ``vanilla_eval``),
merges per-task episodes/successes and reasoning metrics.

Writes ``{base}-pooled.json`` next to those folders (under the shared parent), unless
``--out-dir`` is set (flat layout).

Example:
  python scripts/pool_global_results.py --logs-root ~/.../z_final_results
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_OPENVLA_MINI_ROOT = _SCRIPT_DIR.parent

DEFAULT_LOGS_ROOT = _OPENVLA_MINI_ROOT / "experiments" / "logs" / "z_final_results"

_DIR_20 = re.compile(r"^(?P<base>.+)_20_rollouts$", re.IGNORECASE)
_DIR_30 = re.compile(r"^(?P<base>.+)_30_rollouts$", re.IGNORECASE)

# --- Copied from prismatic.util.reasoning_metrics (avoids importing prismatic / torch stack) ---


def _merge_running_stats_dict(x: dict, y: dict) -> dict:
    def _to_totals(d: dict) -> tuple[int, float, float]:
        c = int(d.get("count", 0))
        if c <= 0:
            return 0, 0.0, 0.0
        if "sum" in d and "sum_sq" in d:
            return c, float(d["sum"]), float(d["sum_sq"])
        mean = float(d.get("mean", 0.0))
        std = float(d.get("std", 0.0))
        total = mean * c
        total_sq = ((std**2) * (c - 1)) + (c * (mean**2)) if c > 1 else total * mean
        return c, total, total_sq

    cx, sx, sx2 = _to_totals(x)
    cy, sy, sy2 = _to_totals(y)
    count = cx + cy
    if count <= 0:
        return {"count": 0, "mean": None, "std": None, "sum": 0.0, "sum_sq": 0.0}
    total = sx + sy
    total_sq = sx2 + sy2
    mean = total / count
    if count > 1:
        variance = (total_sq - count * (mean**2)) / float(count - 1)
        std = float(max(0.0, variance) ** 0.5)
    else:
        std = 0.0
    return {"count": count, "mean": float(mean), "std": std, "sum": float(total), "sum_sq": float(total_sq)}


def merge_text_rouge_payloads(a: dict, b: dict) -> dict:
    keys = set(a.keys()) | set(b.keys())
    out = {}
    for k in keys:
        out[k] = _merge_running_stats_dict(a.get(k, {}), b.get(k, {}))
    return out


def merge_reasoning_metrics_payloads(acc: Optional[dict], worker: dict) -> dict:
    if not worker:
        return dict(acc or {})
    base = dict(acc or {})
    for k, bv in worker.items():
        if k == "text_rouge_l" and isinstance(bv, dict):
            base["text_rouge_l"] = merge_text_rouge_payloads(base.get("text_rouge_l", {}), bv)
        elif isinstance(bv, dict) and "count" in bv:
            base[k] = _merge_running_stats_dict(base.get(k, {}), bv)
    return base


FINGERPRINT_KEYS: tuple[str, ...] = (
    "task_suite_name",
    "experiment_type",
    "perturbation_type",
    "perturbation_level",
    "noise_sigma",
    "pretrained_checkpoint",
    "reasoning_modifier_fn_str",
    "ablation_components",
    "perturbation",
    "distractors",
    "model_family",
    "reasoning_gt_source",
    "metrics_camera_name",
)


def _normalize_fingerprint_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {str(k): _normalize_fingerprint_value(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list):
        if value and isinstance(value[0], (int, float, str)):
            return sorted(value, key=lambda x: (str(type(x)), str(x)))
        return [_normalize_fingerprint_value(v) for v in value]
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def fingerprint_payload(payload: dict[str, Any]) -> str:
    parts = [(k, _normalize_fingerprint_value(payload.get(k))) for k in FINGERPRINT_KEYS]
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def parse_rollout_folder(name: str) -> tuple[str, str] | None:
    """Return (base_name, '20'|'30') if folder is *_{20|30}_rollouts."""
    m = _DIR_20.match(name)
    if m:
        return (m.group("base"), "20")
    m = _DIR_30.match(name)
    if m:
        return (m.group("base"), "30")
    return None


def safe_pooled_filename_stem(base: str) -> str:
    s = "".join(c if c.isalnum() or c in "-_" else "_" for c in base)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:200] if s else "results"


def collect_global_result_paths(root: Path, pattern: str) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob(pattern)):
        if not p.is_file():
            continue
        name = p.name.lower()
        if "pooled" in name:
            continue
        if not p.name.endswith(".json"):
            continue
        out.append(p)
    return out


def discover_20_30_pairs(
    logs_root: Path, pattern: str
) -> list[tuple[str, str, Path, Path]]:
    """
    Each tuple: (parent_rel, base_name, path_20_rollouts_json, path_30_rollouts_json)
    parent_rel is path from logs_root to the directory containing both rollout folders.
    """
    root = logs_root.resolve()
    paths = collect_global_result_paths(root, pattern)
    slots: dict[tuple[str, str], dict[str, Path]] = defaultdict(dict)

    for p in paths:
        rel = p.resolve().relative_to(root)
        folder = rel.parent.name
        parsed = parse_rollout_folder(folder)
        if not parsed:
            continue
        base, kind = parsed
        parent = rel.parent.parent
        parent_rel = parent.as_posix() if parent.parts else ""
        key = (parent_rel, base)
        if kind in slots[key]:
            raise ValueError(
                f"Duplicate *_{kind}_rollouts GLOBAL under {key!r}: {slots[key][kind]} and {p}"
            )
        slots[key][kind] = p

    pairs: list[tuple[str, str, Path, Path]] = []
    for (parent_rel, base), d in sorted(slots.items()):
        if "20" in d and "30" in d:
            pairs.append((parent_rel, base, d["20"], d["30"]))
    return pairs


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object at root: {path}")
    return data


def merge_two_task_rows(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ts = int(a["task_successes"]) + int(b["task_successes"])
    te = int(a["task_episodes"]) + int(b["task_episodes"])
    out = dict(a)
    out["task_successes"] = ts
    out["task_episodes"] = te
    out["task_success_rate"] = float(ts) / float(te) if te > 0 else 0.0

    rm_keys = ("bbox_iou", "gripper_distance", "text_rouge_l")
    sub_a = {k: a[k] for k in rm_keys if k in a and isinstance(a[k], dict)}
    sub_b = {k: b[k] for k in rm_keys if k in b and isinstance(b[k], dict)}
    merged_sub = merge_reasoning_metrics_payloads(sub_a, sub_b)
    for k, v in merged_sub.items():
        out[k] = v
    return out


def infer_num_tasks_in_suite(payload: dict[str, Any]) -> int:
    n = int(payload.get("num_tasks_in_suite") or 0)
    if n > 0:
        return n
    ts = str(payload.get("task_suite_name", ""))
    m = re.search(r"_(\d+)\s*$", ts)
    if m:
        return int(m.group(1))
    ids = payload.get("evaluated_task_ids")
    if isinstance(ids, list) and ids:
        return len(ids)
    nut = payload.get("num_unique_tasks_evaluated")
    if nut is not None:
        try:
            return int(nut)
        except (TypeError, ValueError):
            pass
    return 0


def _int_or_default(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def pool_payloads(paths: list[Path], pool_label: str) -> tuple[dict[str, Any], list[str]]:
    payloads = [load_json(p) for p in paths]
    warnings: list[str] = []

    suite = str(payloads[0].get("task_suite_name", ""))
    for i, p in enumerate(payloads[1:], start=1):
        if str(p.get("task_suite_name", "")) != suite:
            raise ValueError(
                f"task_suite_name mismatch: {paths[0]} has {suite!r}, "
                f"{paths[i]} has {p.get('task_suite_name')!r}"
            )

    fp0 = fingerprint_payload(payloads[0])
    for i, p in enumerate(payloads[1:], start=1):
        fpi = fingerprint_payload(p)
        if fpi != fp0:
            warnings.append(
                f"JSON metadata fingerprint differs but merging anyway: {paths[0].name} vs {paths[i].name}"
            )

    combined_per_task: dict[int, dict[str, Any]] = {}

    for path, payload in zip(paths, payloads):
        ptm = payload.get("per_task_metrics")
        if not isinstance(ptm, dict):
            raise ValueError(f"Missing per_task_metrics in {path}")

        for _metrics_key, row in ptm.items():
            if not isinstance(row, dict):
                continue
            tid = int(row["task_id"])
            if tid in combined_per_task:
                combined_per_task[tid] = merge_two_task_rows(combined_per_task[tid], row)
            else:
                combined_per_task[tid] = dict(row)

    # Per-task rows in GLOBAL-RESULTS can be incomplete (key collisions / missing rows),
    # so pooled top-line totals must come from source top-line totals, not from per-task sums.
    per_task_total_episodes = sum(int(combined_per_task[t]["task_episodes"]) for t in combined_per_task)
    per_task_total_successes = sum(int(combined_per_task[t]["task_successes"]) for t in combined_per_task)
    source_total_episodes = sum(_int_or_default(p.get("total_episodes"), 0) for p in payloads)
    source_total_successes = sum(_int_or_default(p.get("total_successes"), 0) for p in payloads)

    if source_total_episodes <= 0:
        warnings.append(
            "Source total_episodes missing/invalid; falling back to sum of pooled per_task_metrics."
        )
        source_total_episodes = per_task_total_episodes
    if source_total_successes < 0:
        warnings.append(
            "Source total_successes missing/invalid; falling back to sum of pooled per_task_metrics."
        )
        source_total_successes = per_task_total_successes

    if source_total_episodes != per_task_total_episodes or source_total_successes != per_task_total_successes:
        warnings.append(
            "Top-line totals differ from pooled per_task_metrics sums "
            f"(source episodes/successes={source_total_episodes}/{source_total_successes}, "
            f"per_task episodes/successes={per_task_total_episodes}/{per_task_total_successes}). "
            "This usually means per_task_metrics is incomplete in one or more source files."
        )

    trials_set = sorted(
        {int(p.get("num_trials_per_task", -1)) for p in payloads if p.get("num_trials_per_task") is not None}
    )
    trials_set = [x for x in trials_set if x != -1]

    agg_reasoning: dict[str, Any] = {}
    for p in payloads:
        agg_reasoning = merge_reasoning_metrics_payloads(agg_reasoning, p.get("reasoning_metrics") or {})

    evaluated_union: set[int] = set()
    for p in payloads:
        for x in p.get("evaluated_task_ids") or []:
            evaluated_union.add(int(x))
        for x in p.get("task_ids") or []:
            if x is not None:
                try:
                    evaluated_union.add(int(x))
                except (TypeError, ValueError):
                    pass

    out: dict[str, Any] = dict(payloads[0])
    # Key pooled per-task rows by task_id string to avoid collisions on repeated descriptions.
    out["per_task_metrics"] = {str(tid): combined_per_task[tid] for tid in sorted(combined_per_task)}
    out["total_episodes"] = int(source_total_episodes)
    out["total_successes"] = int(source_total_successes)
    out["total_success_rate"] = (
        float(source_total_successes) / float(source_total_episodes) if source_total_episodes else 0.0
    )
    out["reasoning_metrics"] = agg_reasoning
    out["evaluated_task_ids"] = sorted(evaluated_union) if evaluated_union else sorted(combined_per_task.keys())
    out["num_unique_tasks_evaluated"] = len(out["evaluated_task_ids"])
    out["pooling"] = {
        "pool_label": pool_label,
        "pooled_from": ["*_20_rollouts", "*_30_rollouts"],
        "metadata_fingerprint": fp0,
        "source_paths": [str(p.resolve()) for p in paths],
        "source_num_trials_per_task": trials_set,
        "num_source_files": len(paths),
        "per_task_row_count": len(combined_per_task),
        "per_task_totals": {
            "episodes": int(per_task_total_episodes),
            "successes": int(per_task_total_successes),
        },
        "source_totals": {
            "episodes": int(source_total_episodes),
            "successes": int(source_total_successes),
        },
    }
    if trials_set:
        out["num_trials_per_task"] = int(max(trials_set))
    out["log_path"] = None

    for drop in ("num_shards", "shard_rank", "input_metric_paths", "worker_registry", "num_shards_found"):
        out.pop(drop, None)

    n_suite = infer_num_tasks_in_suite(payloads[0])
    if n_suite > 0:
        for path, p in zip(paths, payloads):
            nt = int(p.get("num_trials_per_task") or 0)
            expected_file = nt * n_suite
            got = int(p.get("total_episodes") or 0)
            if got != expected_file:
                warnings.append(
                    f"Source {path.name}: total_episodes {got} != expected {expected_file} "
                    f"(num_trials_per_task={nt} × tasks_in_suite={n_suite})"
                )

    return out, warnings


def output_dir_for_pair(logs_root: Path, parent_rel: str, flat_out: Path | None) -> Path:
    if flat_out is not None:
        return flat_out.resolve()
    root = logs_root.resolve()
    if not parent_rel:
        return root
    return root / parent_rel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--logs-root",
        type=Path,
        default=DEFAULT_LOGS_ROOT,
        help=f"Root to search (default: {DEFAULT_LOGS_ROOT}).",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="*GLOBAL-RESULTS.json",
        help="Glob under logs-root (recursive).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="If set, write all *-pooled.json files here (flat). Otherwise write under each pair's parent.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned merges only.",
    )
    args = ap.parse_args()

    root = args.logs_root
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    try:
        pairs = discover_20_30_pairs(root, args.pattern)
    except ValueError as e:
        raise SystemExit(str(e)) from e

    if not pairs:
        raise SystemExit(
            f"No matching sibling pairs (*_20_rollouts + *_30_rollouts) with {args.pattern!r} under {root}"
        )

    flat_out: Path | None = args.out_dir.resolve() if args.out_dir is not None else None
    if flat_out is not None and not args.dry_run:
        flat_out.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    n_written = 0

    for parent_rel, base, p20, p30 in pairs:
        gpaths = [p20, p30]
        pool_label = f"{parent_rel}/{base}" if parent_rel else base
        out_dir = output_dir_for_pair(root, parent_rel, flat_out)
        stem = safe_pooled_filename_stem(base)
        out_name = f"{stem}-pooled.json"
        out_path = out_dir / out_name

        try:
            pooled, warns = pool_payloads(gpaths, pool_label=pool_label)
        except ValueError as e:
            raise SystemExit(f"Pool failed for {pool_label!r}: {e}") from e

        manifest.append(
            {
                "output": str(out_path),
                "pool_label": pool_label,
                "base_name": base,
                "parent_rel": parent_rel,
                "sources": [str(p20), str(p30)],
                "warnings": warns,
            }
        )
        for w in warns:
            print(f"Warning [{pool_label}]: {w}")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(pooled, indent=2), encoding="utf-8")
            n_written += 1
        print(f"Pooled {base!r} -> {out_path}")

    manifest_path = (flat_out if flat_out is not None else root.resolve()) / "pool_manifest.json"
    if not args.dry_run:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {manifest_path}")
    print(f"Pairs pooled: {len(pairs)}  outputs written: {n_written}")


if __name__ == "__main__":
    main()
