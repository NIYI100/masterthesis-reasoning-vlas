"""Deterministic in-task instruction rewriting for LIBERO-90 counterfactual eval."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TaskSwapDecision:
    task_id: int
    task_name: str
    task_language: str
    swap_status: str
    swapped_task_id: Optional[int]
    swapped_task_name: Optional[str]
    swapped_task_language: Optional[str]
    swap_type: Optional[str]
    swap_rule_applied: Optional[str]
    candidate_count: int
    skip_reason: Optional[str]


_DIRECTION_PAIRS = (
    ("_to_the_left_of_the_", "_to_the_right_of_the_"),
    ("_left_compartment_", "_right_compartment_"),
    ("_on_the_left_", "_on_the_right_"),
    ("_left_plate_", "_right_plate_"),
    ("_at_the_front_", "_at_the_back_"),
    ("_front_compartment_", "_back_compartment_"),
    ("_on_top_of_the_", "_under_the_"),
    ("_open_the_", "_close_the_"),
    ("_turn_on_the_", "_turn_off_the_"),
    ("_left_", "_right_"),
    ("_front_", "_back_"),
)

_OBJECT_REL_PATTERNS = (
    "_and_put_it_in_the_",
    "_and_put_it_on_the_",
    "_and_put_it_to_the_left_of_the_",
    "_and_put_it_to_the_right_of_the_",
    "_and_place_it_in_the_",
    "_and_place_it_on_the_",
    "_and_place_it_to_the_left_of_the_",
    "_and_place_it_to_the_right_of_the_",
    "_on_the_",
    "_in_the_",
    "_to_the_left_of_the_",
    "_to_the_right_of_the_",
    "_to_the_front_of_the_",
    "_to_the_back_of_the_",
    "_under_the_",
)
_DEFAULT_TASK_ROLLOUTS_CSV = Path("/home/hk-project-p0024638/uvrfq/libero_reasonings_task_rollouts.csv")


def _split_scene_and_rest(task_name: str) -> Tuple[str, str]:
    parts = task_name.split("_")
    if len(parts) < 3:
        return "", task_name
    return "_".join(parts[:2]), "_".join(parts[2:])


def _swap_pairs(text: str, pairs: Tuple[Tuple[str, str], ...]) -> str:
    out = f"_{text}_"
    placeholders: List[Tuple[str, str]] = []
    for idx, (left, right) in enumerate(pairs):
        ph = f"__TMP_SWAP_{idx}__"
        if left in out:
            out = out.replace(left, ph)
            placeholders.append((ph, right))
    for left, right in pairs:
        if right in out:
            out = out.replace(right, left)
    for ph, right in placeholders:
        out = out.replace(ph, right)
    return out.strip("_")


def _to_language(task_name_no_scene: str) -> str:
    return task_name_no_scene.replace("_", " ")


def _load_csv_instruction_map() -> Dict[int, str]:
    if not _DEFAULT_TASK_ROLLOUTS_CSV.exists():
        return {}
    out: Dict[int, str] = {}
    with _DEFAULT_TASK_ROLLOUTS_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                task_id = int(row.get("task_id", "").strip())
            except Exception:
                continue
            instruction = str(row.get("instruction", "")).strip()
            if instruction:
                out[task_id] = instruction
    return out


def _swap_objects_in_rest(task_rest: str) -> Optional[str]:
    # Handle pick_up / put / place templates by swapping source and target object phrases.
    src_prefixes = ("pick_up_the_", "put_the_", "stack_the_", "push_the_")
    src_prefix = None
    for pfx in src_prefixes:
        if task_rest.startswith(pfx):
            src_prefix = pfx
            break
    if src_prefix is None:
        return None

    rel_idx = -1
    rel = None
    for rel_pattern in _OBJECT_REL_PATTERNS:
        idx = task_rest.find(rel_pattern, len(src_prefix))
        if idx >= 0 and (rel_idx < 0 or idx < rel_idx):
            rel_idx = idx
            rel = rel_pattern
    if rel_idx < 0 or rel is None:
        return None

    src = task_rest[len(src_prefix) : rel_idx]
    tail = task_rest[rel_idx + len(rel) :]
    if src == "" or tail == "":
        return None

    # Avoid degenerate rewrites.
    if src == tail:
        return None

    return f"{src_prefix}{tail}{rel}{src}"


def resolve_libero90_swaps(tasks: List[object]) -> List[TaskSwapDecision]:
    """
    Resolve in-task rewritten instructions for LIBERO-90 with deterministic priority:
    object rewrite first, then direction rewrite, then skip.
    """
    names: Dict[int, str] = {idx: str(task.name) for idx, task in enumerate(tasks)}
    languages: Dict[int, str] = {idx: str(task.language) for idx, task in enumerate(tasks)}
    csv_instruction_map = _load_csv_instruction_map()

    decisions: List[TaskSwapDecision] = []
    for task_id in sorted(names.keys()):
        task_name = names[task_id]
        task_lang = csv_instruction_map.get(task_id, languages[task_id])
        scene, rest = _split_scene_and_rest(task_name)

        swapped_rest_obj = _swap_objects_in_rest(rest)
        if swapped_rest_obj is not None:
            swapped_name = f"{scene}_{swapped_rest_obj}" if scene else swapped_rest_obj
            decisions.append(
                TaskSwapDecision(
                    task_id=task_id,
                    task_name=task_name,
                    task_language=task_lang,
                    swap_status="swapped",
                    swapped_task_id=None,
                    swapped_task_name=swapped_name,
                    swapped_task_language=_to_language(swapped_rest_obj),
                    swap_type="object",
                    swap_rule_applied="in_task_source_target_object_swap",
                    candidate_count=1,
                    skip_reason=None,
                )
            )
            continue

        swapped_rest_dir = _swap_pairs(rest, _DIRECTION_PAIRS)
        if swapped_rest_dir != rest:
            swapped_name = f"{scene}_{swapped_rest_dir}" if scene else swapped_rest_dir
            decisions.append(
                TaskSwapDecision(
                    task_id=task_id,
                    task_name=task_name,
                    task_language=task_lang,
                    swap_status="swapped",
                    swapped_task_id=None,
                    swapped_task_name=swapped_name,
                    swapped_task_language=_to_language(swapped_rest_dir),
                    swap_type="direction",
                    swap_rule_applied="in_task_direction_token_inversion",
                    candidate_count=1,
                    skip_reason=None,
                )
            )
            continue

        decisions.append(
            TaskSwapDecision(
                task_id=task_id,
                task_name=task_name,
                task_language=task_lang,
                swap_status="not_swapped",
                swapped_task_id=None,
                swapped_task_name=None,
                swapped_task_language=None,
                swap_type=None,
                swap_rule_applied=None,
                candidate_count=0,
                skip_reason="no_in_task_object_or_direction_rewrite",
            )
        )
    return decisions


def decisions_to_serializable(decisions: List[TaskSwapDecision]) -> List[Dict[str, object]]:
    return [
        {
            "task_id": int(d.task_id),
            "task_name": d.task_name,
            "task_language": d.task_language,
            "swap_status": d.swap_status,
            "swapped_task_id": (None if d.swapped_task_id is None else int(d.swapped_task_id)),
            "swapped_task_name": d.swapped_task_name,
            "swapped_task_language": d.swapped_task_language,
            "swap_type": d.swap_type,
            "swap_rule_applied": d.swap_rule_applied,
            "candidate_count": int(d.candidate_count),
            "skip_reason": d.skip_reason,
        }
        for d in decisions
    ]

