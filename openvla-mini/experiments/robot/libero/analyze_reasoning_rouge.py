#!/usr/bin/env python3
"""
Offline ROUGE-L analysis for LIBERO reasoning traces against libero_reasonings.json.

Supports two join modes:
1) Direct join (preferred): trace row has ep_path + demo_id + step.
2) Fallback join: trace row has task_id + episode_idx + env_step, and we infer
   ep_path/demo_id from LIBERO suite metadata + libero_reasonings keys.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

@dataclass(frozen=True)
class FieldSpec:
    out_key: str
    cot_tag: str
    gt_key: str


FIELD_SPECS: tuple[FieldSpec, ...] = (
    FieldSpec("plan", "PLAN:", "plan"),
    FieldSpec("subtask_reasoning", "SUBTASK REASONING:", "subtask_reasoning"),
    FieldSpec("subtask", "SUBTASK:", "subtask"),
    FieldSpec("move_reasoning", "MOVE REASONING:", "movement_reasoning"),
    FieldSpec("move", "MOVE:", "movement"),
)

COT_TAGS = [
    "PLAN:",
    "VISIBLE OBJECTS:",
    "SUBTASK REASONING:",
    "SUBTASK:",
    "MOVE REASONING:",
    "MOVE:",
    "GRIPPER POSITION:",
    "ACTION:",
]


DIRECT_EP_KEYS = ("ep_path", "file_path", "episode_file_path", "rlds_file_path")
DIRECT_DEMO_KEYS = ("demo_id", "episode_id", "rlds_demo_id")
DIRECT_STEP_KEYS = ("step", "step_idx", "rlds_step")


@dataclass
class RunningStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0

    def add(self, value: Optional[float]) -> None:
        if value is None:
            return
        v = float(value)
        self.count += 1
        self.total += v
        self.total_sq += v * v

    def to_dict(self) -> dict[str, Any]:
        if self.count <= 0:
            return {"count": 0, "mean": None, "std": None}
        mean = self.total / float(self.count)
        if self.count > 1:
            var = (self.total_sq - self.count * (mean**2)) / float(self.count - 1)
            std = max(0.0, var) ** 0.5
        else:
            std = 0.0
        return {"count": int(self.count), "mean": float(mean), "std": float(std)}


def tokenize_text(text: str) -> list[str]:
    if text is None:
        return []
    return re.findall(r"[a-z0-9]+", str(text).lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        ai = a[i - 1]
        row_prev, row = dp[i - 1], dp[i]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                row[j] = row_prev[j - 1] + 1
            else:
                row[j] = row_prev[j] if row_prev[j] > row[j - 1] else row[j - 1]
    return dp[n][m]


def rouge_l_f1(candidate: str, reference: str) -> Optional[float]:
    c = tokenize_text(candidate)
    r = tokenize_text(reference)
    if not c and not r:
        return None
    if not c or not r:
        return 0.0
    lcs = lcs_length(c, r)
    if lcs == 0:
        return 0.0
    rec = lcs / len(r)
    prec = lcs / len(c)
    if rec + prec == 0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def split_reasoning(text: str) -> dict[Optional[str], str]:
    new_parts: dict[Optional[str], str] = {None: text}
    for tag in COT_TAGS:
        parts = new_parts
        new_parts = {}
        for k, v in parts.items():
            if tag in v:
                left, right = v.split(tag, 1)
                if left.strip():
                    new_parts[k] = left
                new_parts[tag] = right
            else:
                new_parts[k] = v
    if None in new_parts and not new_parts[None].strip():
        del new_parts[None]
    return new_parts


def parse_text_field_from_reasoning(reasoning: str, cot_tag: str) -> Optional[str]:
    if not isinstance(reasoning, str):
        return None
    parts = split_reasoning(reasoning.replace("@", " "))
    text = parts.get(cot_tag)
    if not isinstance(text, str):
        return None
    text = text.strip()
    return text if text else None


def _first_present(record: dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _as_nonempty_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def _as_nonneg_int(value: Any) -> Optional[int]:
    try:
        out = int(value)
    except Exception:
        return None
    return out if out >= 0 else None


def _normalize_gt_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        items: list[tuple[int, str]] = []
        fallback_items: list[str] = []
        for k, v in value.items():
            try:
                items.append((int(k), str(v)))
            except Exception:
                fallback_items.append(str(v))
        if items:
            items.sort(key=lambda x: x[0])
            text = " ".join(v for _, v in items).strip()
            return text if text else None
        text = " ".join(fallback_items).strip()
        return text if text else None
    if isinstance(value, (list, tuple)):
        text = " ".join(str(v) for v in value).strip()
        return text if text else None
    text = str(value).strip()
    return text if text else None


def _resolve_trace_paths(inputs: list[str]) -> list[Path]:
    out: set[Path] = set()
    for raw in inputs:
        p = Path(raw)
        if p.is_file():
            out.add(p.resolve())
            continue
        if p.is_dir():
            for cand in sorted(p.rglob("*.jsonl")):
                out.add(cand.resolve())
            continue
        for cand in sorted(Path.cwd().glob(raw)):
            if cand.is_file():
                out.add(cand.resolve())
    return sorted(out)


def _iter_trace_rows(trace_paths: list[Path]) -> Iterable[tuple[Path, int, dict[str, Any]]]:
    for path in trace_paths:
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield path, line_no, row


def _extract_task_name_from_ep_path(ep_path: str) -> str:
    name = Path(ep_path).name
    if name.endswith("_demo.hdf5"):
        name = name[: -len("_demo.hdf5")]
    if "SCENE" in name:
        m = re.match(r"^(?:[A-Za-z]+_)?SCENE\d+_(.*)$", name)
        if m:
            name = m.group(1)
    return name.lower()


def _normalize_task_text(text: str) -> str:
    s = str(text).lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _task_names_from_trace_rows(
    rows: list[tuple[Path, int, dict[str, Any]]], needed_task_ids: set[int]
) -> dict[int, str]:
    out: dict[int, str] = {}
    for _, _, row in rows:
        task_id = _as_nonneg_int(row.get("task_id"))
        if task_id is None or task_id not in needed_task_ids:
            continue
        if task_id in out:
            continue
        td = row.get("task_description")
        if isinstance(td, str) and td.strip():
            out[task_id] = _normalize_task_text(td)
    return out


def _load_libero_task_names_optional(
    libero_root: Path, suite_name: str, needed_task_ids: set[int]
) -> dict[int, str]:
    if not needed_task_ids:
        return {}
    try:
        sys.path.append(str(libero_root))
        from libero.libero import benchmark  # type: ignore
    except Exception:
        return {}

    out: dict[int, str] = {}
    try:
        bench_cls = benchmark.get_benchmark_dict()[suite_name]
        suite = bench_cls()
        for task_id in sorted(needed_task_ids):
            out[task_id] = _normalize_task_text(suite.get_task(task_id).name)
    except Exception:
        return {}
    return out


def _load_reasonings_subset(reasonings_path: Path, needed_ep_paths: Optional[set[str]]) -> dict[str, Any]:
    # Use optional streaming parser if available to keep memory lower on huge files.
    try:
        import ijson  # type: ignore

        subset: dict[str, Any] = {}
        with reasonings_path.open("rb") as f:
            for ep_path, ep_payload in ijson.kvitems(f, ""):
                if needed_ep_paths is None or ep_path in needed_ep_paths:
                    subset[ep_path] = ep_payload
        return subset
    except Exception:
        with reasonings_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if needed_ep_paths is None:
            return payload
        return {k: v for k, v in payload.items() if k in needed_ep_paths}


def _to_direct_key(record: dict[str, Any], step_offset: int) -> Optional[tuple[str, str, str]]:
    ep_path = _as_nonempty_str(_first_present(record, DIRECT_EP_KEYS))
    demo_id = _as_nonempty_str(_first_present(record, DIRECT_DEMO_KEYS))
    step_any = _first_present(record, DIRECT_STEP_KEYS)
    if step_any is None:
        # If explicit step is not present, allow env_step as fallback.
        step_any = record.get("env_step")
    step_id = _as_nonneg_int(step_any)
    if ep_path is None or demo_id is None or step_id is None:
        return None
    step_id = step_id + int(step_offset)
    if step_id < 0:
        return None
    return ep_path, demo_id, str(step_id)


def _task_name_match(task_name: str, ep_path_task_name: str) -> bool:
    t1 = _normalize_task_text(task_name)
    t2 = _normalize_task_text(ep_path_task_name)
    return t1 in t2 or t2 in t1


def _build_fallback_episode_map(
    reasonings_by_ep: dict[str, Any],
    task_names_by_id: dict[int, str],
) -> dict[int, list[tuple[str, str]]]:
    ep_demo_by_task_name: dict[str, list[tuple[str, str]]] = {}
    for ep_path, demos in reasonings_by_ep.items():
        if not isinstance(demos, dict):
            continue
        ep_task_name = _extract_task_name_from_ep_path(ep_path)
        matched = [tn for tn in task_names_by_id.values() if _task_name_match(tn, ep_task_name)]
        if not matched:
            continue
        best = max(matched, key=len)
        lst = ep_demo_by_task_name.setdefault(best, [])
        for demo_id in demos.keys():
            lst.append((ep_path, str(demo_id)))

    for task_name in ep_demo_by_task_name.keys():
        ep_demo_by_task_name[task_name] = sorted(
            ep_demo_by_task_name[task_name],
            key=lambda x: (x[0], int(x[1]) if str(x[1]).isdigit() else str(x[1])),
        )

    out: dict[int, list[tuple[str, str]]] = {}
    for task_id, task_name in task_names_by_id.items():
        out[task_id] = ep_demo_by_task_name.get(task_name, [])
    return out


def _to_fallback_key(
    record: dict[str, Any],
    fallback_episode_map: dict[int, list[tuple[str, str]]],
    step_offset: int,
) -> Optional[tuple[str, str, str]]:
    task_id = _as_nonneg_int(record.get("task_id"))
    episode_idx = _as_nonneg_int(record.get("episode_idx"))
    env_step = _as_nonneg_int(record.get("env_step"))
    if task_id is None or episode_idx is None or env_step is None:
        return None
    episodes = fallback_episode_map.get(task_id, [])
    if episode_idx >= len(episodes):
        return None
    ep_path, demo_id = episodes[episode_idx]
    step = env_step + int(step_offset)
    if step < 0:
        return None
    return ep_path, demo_id, str(step)


def _compute_summary_payload(stats: dict[str, RunningStats]) -> dict[str, dict[str, Any]]:
    return {k: v.to_dict() for k, v in stats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ROUGE-L by reasoning field from LIBERO traces.")
    parser.add_argument(
        "--trace-input",
        nargs="+",
        required=True,
        help="Trace JSONL files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--reasonings-json",
        default="/home/hk-project-p0024638/uvrfq/hkfswork/ecot-lite_data/libero_reasonings.json",
        help="Path to libero_reasonings.json.",
    )
    parser.add_argument(
        "--task-suite-name",
        default="libero_90",
        help="LIBERO suite name for fallback mapping (e.g., libero_90).",
    )
    parser.add_argument(
        "--libero-root",
        default=os.environ.get("LIBERO_ROOT", "/home/hk-project-p0024638/uvrfq/LIBERO"),
        help="Path to LIBERO repo root for fallback mapping.",
    )
    parser.add_argument(
        "--step-offset",
        type=int,
        default=0,
        help="Offset added to step index before lookup. Example: -10 when env_step includes wait steps.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional output summary JSON path.",
    )
    parser.add_argument(
        "--summary-csv",
        default=None,
        help="Optional output summary CSV path.",
    )
    parser.add_argument(
        "--samples-jsonl",
        default=None,
        help="Optional detailed per-sample scores JSONL path.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional cap on processed trace rows (0 = all).",
    )
    args = parser.parse_args()

    trace_paths = _resolve_trace_paths(args.trace_input)
    if not trace_paths:
        raise SystemExit("No trace JSONL files found from --trace-input.")

    rows: list[tuple[Path, int, dict[str, Any]]] = []
    for item in _iter_trace_rows(trace_paths):
        rows.append(item)
        if args.max_rows > 0 and len(rows) >= args.max_rows:
            break
    if not rows:
        raise SystemExit("No valid JSON rows found in trace inputs.")

    direct_candidates: list[tuple[int, tuple[str, str, str]]] = []
    fallback_needed_task_ids: set[int] = set()
    fallback_rows: set[int] = set()
    for idx, (_, _, row) in enumerate(rows):
        dk = _to_direct_key(row, step_offset=args.step_offset)
        if dk is not None:
            direct_candidates.append((idx, dk))
            continue
        task_id = _as_nonneg_int(row.get("task_id"))
        episode_idx = _as_nonneg_int(row.get("episode_idx"))
        env_step = _as_nonneg_int(row.get("env_step"))
        if task_id is not None and episode_idx is not None and env_step is not None:
            fallback_needed_task_ids.add(task_id)
            fallback_rows.add(idx)

    needed_ep_paths = {ep for _, (ep, _, _) in direct_candidates}
    reasonings_path = Path(args.reasonings_json).resolve()
    if not reasonings_path.exists():
        raise SystemExit(f"Reasonings file not found: {reasonings_path}")

    reasonings_by_ep = _load_reasonings_subset(
        reasonings_path,
        needed_ep_paths if not fallback_needed_task_ids else None,
    )

    fallback_episode_map: dict[int, list[tuple[str, str]]] = {}
    if fallback_needed_task_ids:
        task_names_by_id = _task_names_from_trace_rows(rows, fallback_needed_task_ids)
        missing_ids = set(fallback_needed_task_ids) - set(task_names_by_id.keys())
        if missing_ids:
            task_names_by_id.update(
                _load_libero_task_names_optional(
                    libero_root=Path(args.libero_root).resolve(),
                    suite_name=args.task_suite_name,
                    needed_task_ids=missing_ids,
                )
            )

        fallback_episode_map = _build_fallback_episode_map(reasonings_by_ep, task_names_by_id)
        for missing_task_id in fallback_needed_task_ids:
            fallback_episode_map.setdefault(missing_task_id, [])

    stats = {spec.out_key: RunningStats() for spec in FIELD_SPECS}
    stats["whole"] = RunningStats()

    matched = 0
    unmatched = 0
    direct_matched = 0
    fallback_matched = 0
    unmatched_reasons: dict[str, int] = {}
    sample_rows_out: list[dict[str, Any]] = []

    for idx, (trace_path, line_no, row) in enumerate(rows):
        key = _to_direct_key(row, step_offset=args.step_offset)
        mode = "direct"
        if key is None:
            key = _to_fallback_key(row, fallback_episode_map, step_offset=args.step_offset)
            mode = "fallback"
        if key is None:
            reason = "no_join_key"
            if mode == "fallback":
                task_id = _as_nonneg_int(row.get("task_id"))
                if task_id is not None and task_id in fallback_episode_map and len(fallback_episode_map[task_id]) == 0:
                    reason = "fallback_task_unmapped"
            unmatched += 1
            unmatched_reasons[reason] = unmatched_reasons.get(reason, 0) + 1
            continue

        ep_path, demo_id, step = key
        leaf = (
            reasonings_by_ep.get(ep_path, {})
            .get(str(demo_id), {})
            .get(str(step))
        )
        if not isinstance(leaf, dict):
            unmatched += 1
            unmatched_reasons["missing_reasoning_leaf"] = unmatched_reasons.get("missing_reasoning_leaf", 0) + 1
            continue

        pred_reasoning = row.get("reasoning")
        if not isinstance(pred_reasoning, str) or not pred_reasoning.strip():
            unmatched += 1
            unmatched_reasons["missing_pred_reasoning"] = unmatched_reasons.get("missing_pred_reasoning", 0) + 1
            continue

        matched += 1
        if mode == "direct":
            direct_matched += 1
        else:
            fallback_matched += 1

        pred_whole_parts: list[str] = []
        ref_whole_parts: list[str] = []
        row_scores: dict[str, Optional[float]] = {}

        for spec in FIELD_SPECS:
            pred_t = parse_text_field_from_reasoning(pred_reasoning, spec.cot_tag)
            ref_t = _normalize_gt_text(leaf.get(spec.gt_key))
            if pred_t:
                pred_whole_parts.append(pred_t)
            if ref_t:
                ref_whole_parts.append(ref_t)
            score = None
            if pred_t and ref_t:
                score = rouge_l_f1(pred_t, ref_t)
                if score is not None:
                    stats[spec.out_key].add(score)
            row_scores[spec.out_key] = score

        pred_whole = " ".join(pred_whole_parts).strip()
        ref_whole = " ".join(ref_whole_parts).strip()
        whole_score = None
        if pred_whole and ref_whole:
            whole_score = rouge_l_f1(pred_whole, ref_whole)
            if whole_score is not None:
                stats["whole"].add(whole_score)
        row_scores["whole"] = whole_score

        if args.samples_jsonl:
            sample_rows_out.append(
                {
                    "trace_file": str(trace_path),
                    "trace_line": line_no,
                    "join_mode": mode,
                    "task_id": row.get("task_id"),
                    "episode_idx": row.get("episode_idx"),
                    "env_step": row.get("env_step"),
                    "ep_path": ep_path,
                    "demo_id": str(demo_id),
                    "step": str(step),
                    "scores": row_scores,
                }
            )

    summary = {
        "trace_files": [str(p) for p in trace_paths],
        "reasonings_json": str(reasonings_path),
        "task_suite_name": args.task_suite_name,
        "step_offset": int(args.step_offset),
        "rows_total": len(rows),
        "rows_matched": matched,
        "rows_unmatched": unmatched,
        "rows_matched_direct": direct_matched,
        "rows_matched_fallback": fallback_matched,
        "unmatched_reasons": unmatched_reasons,
        "rouge_l": _compute_summary_payload(stats),
    }

    print(json.dumps(summary, indent=2))

    if args.summary_json:
        out_json = Path(args.summary_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.summary_csv:
        out_csv = Path(args.summary_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["field", "count", "mean", "std"])
            writer.writeheader()
            for key in ["plan", "subtask_reasoning", "subtask", "move_reasoning", "move", "whole"]:
                payload = summary["rouge_l"][key]
                writer.writerow(
                    {
                        "field": key,
                        "count": payload["count"],
                        "mean": payload["mean"],
                        "std": payload["std"],
                    }
                )

    if args.samples_jsonl:
        out_samples = Path(args.samples_jsonl).resolve()
        out_samples.parent.mkdir(parents=True, exist_ok=True)
        with out_samples.open("w", encoding="utf-8") as f:
            for row in sample_rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
