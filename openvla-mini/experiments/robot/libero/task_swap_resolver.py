"""Deterministic LIBERO task counterpart resolution for counterfactual eval."""

from __future__ import annotations

from dataclasses import dataclass
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

_OBJECT_PATTERNS = (
    ("pick_up_the_", "_and_place_it_"),
    ("pick_up_the_", "_and_put_it_"),
    ("pick_up_the_", "_and_place_them_"),
    ("pick_up_the_", "_and_put_them_"),
    ("put_the_", "_on_"),
    ("put_the_", "_in_"),
    ("put_the_", "_to_"),
    ("put_the_", "_under_"),
    ("stack_the_", "_on_"),
    ("push_the_", "_to_"),
)


def _scene_prefix(task_name: str) -> str:
    parts = task_name.split("_")
    if len(parts) < 3:
        return task_name
    return "_".join(parts[:2])


def _strip_scene_prefix(task_name: str) -> str:
    parts = task_name.split("_")
    if len(parts) < 3:
        return task_name
    return "_".join(parts[2:])


def _extract_object_mask(rest: str) -> Optional[Tuple[str, str]]:
    for prefix, suffix in _OBJECT_PATTERNS:
        if prefix not in rest or suffix not in rest:
            continue
        start = rest.find(prefix)
        if start < 0:
            continue
        obj_start = start + len(prefix)
        obj_end = rest.find(suffix, obj_start)
        if obj_end <= obj_start:
            continue
        obj = rest[obj_start:obj_end]
        if obj == "":
            continue
        masked = f"{rest[:obj_start]}<OBJ>{rest[obj_end:]}"
        return obj, masked
    return None


def _extract_direction_signature(rest: str) -> Dict[str, str]:
    sig: Dict[str, str] = {}
    if "_to_the_left_of_the_" in rest:
        sig["rel_lr"] = "left_of"
    elif "_to_the_right_of_the_" in rest:
        sig["rel_lr"] = "right_of"
    if "_left_compartment_" in rest:
        sig["comp_lr"] = "left_compartment"
    elif "_right_compartment_" in rest:
        sig["comp_lr"] = "right_compartment"
    elif "_front_compartment_" in rest:
        sig["comp_fb"] = "front_compartment"
    elif "_back_compartment_" in rest:
        sig["comp_fb"] = "back_compartment"
    if "_at_the_front_" in rest:
        sig["front_back"] = "front"
    elif "_at_the_back_" in rest:
        sig["front_back"] = "back"
    elif "_on_the_left_" in rest:
        sig["left_right"] = "left"
    elif "_on_the_right_" in rest:
        sig["left_right"] = "right"
    if "_left_plate_" in rest:
        sig["plate_lr"] = "left"
    elif "_right_plate_" in rest:
        sig["plate_lr"] = "right"
    if "_on_top_of_the_" in rest:
        sig["vertical"] = "top"
    elif "_under_the_" in rest:
        sig["vertical"] = "under"
    if "_open_the_" in f"_{rest}_":
        sig["open_close"] = "open"
    elif "_close_the_" in f"_{rest}_":
        sig["open_close"] = "close"
    if "_turn_on_the_" in f"_{rest}_":
        sig["on_off"] = "on"
    elif "_turn_off_the_" in f"_{rest}_":
        sig["on_off"] = "off"
    return sig


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


def _find_object_swap(
    task_id: int,
    task_name: str,
    task_language: str,
    same_scene_ids: List[int],
    names: Dict[int, str],
    languages: Dict[int, str],
) -> Optional[TaskSwapDecision]:
    src_rest = _strip_scene_prefix(task_name)
    parsed = _extract_object_mask(src_rest)
    if parsed is None:
        return None

    src_object, src_masked = parsed
    src_sig = _extract_direction_signature(src_rest)
    matches: List[int] = []
    for cand_id in same_scene_ids:
        if cand_id == task_id:
            continue
        cand_rest = _strip_scene_prefix(names[cand_id])
        cand_parsed = _extract_object_mask(cand_rest)
        if cand_parsed is None:
            continue
        cand_object, cand_masked = cand_parsed
        if cand_masked != src_masked:
            continue
        if cand_object == src_object:
            continue
        if _extract_direction_signature(cand_rest) != src_sig:
            continue
        matches.append(cand_id)

    if not matches:
        return None

    swap_id = min(matches)
    return TaskSwapDecision(
        task_id=task_id,
        task_name=task_name,
        task_language=task_language,
        swap_status="swapped",
        swapped_task_id=swap_id,
        swapped_task_name=names[swap_id],
        swapped_task_language=languages[swap_id],
        swap_type="object",
        swap_rule_applied="same_scene_same_masked_template_different_object",
        candidate_count=len(matches),
        skip_reason=None,
    )


def _find_direction_swap(
    task_id: int,
    task_name: str,
    task_language: str,
    same_scene_name_to_id: Dict[str, int],
    names: Dict[int, str],
    languages: Dict[int, str],
) -> Optional[TaskSwapDecision]:
    src_rest = _strip_scene_prefix(task_name)
    swapped_rest = _swap_pairs(src_rest, _DIRECTION_PAIRS)
    if swapped_rest == src_rest:
        return None

    scene = _scene_prefix(task_name)
    swapped_name = f"{scene}_{swapped_rest}"
    swap_id = same_scene_name_to_id.get(swapped_name)
    if swap_id is None or swap_id == task_id:
        return None

    return TaskSwapDecision(
        task_id=task_id,
        task_name=task_name,
        task_language=task_language,
        swap_status="swapped",
        swapped_task_id=swap_id,
        swapped_task_name=names[swap_id],
        swapped_task_language=languages[swap_id],
        swap_type="direction",
        swap_rule_applied="same_scene_directional_token_inversion",
        candidate_count=1,
        skip_reason=None,
    )


def resolve_libero90_swaps(tasks: List[object]) -> List[TaskSwapDecision]:
    """
    Resolve task counterparts online for LIBERO-90 with deterministic priority:
    object swap first, then direction swap, then skip.
    """
    names: Dict[int, str] = {idx: str(task.name) for idx, task in enumerate(tasks)}
    languages: Dict[int, str] = {idx: str(task.language) for idx, task in enumerate(tasks)}
    scene_to_ids: Dict[str, List[int]] = {}
    scene_to_name_to_id: Dict[str, Dict[str, int]] = {}
    for task_id, task_name in names.items():
        scene = _scene_prefix(task_name)
        scene_to_ids.setdefault(scene, []).append(task_id)
        scene_to_name_to_id.setdefault(scene, {})[task_name] = task_id

    decisions: List[TaskSwapDecision] = []
    for task_id in sorted(names.keys()):
        task_name = names[task_id]
        task_lang = languages[task_id]
        scene = _scene_prefix(task_name)
        same_scene_ids = scene_to_ids[scene]

        object_decision = _find_object_swap(
            task_id=task_id,
            task_name=task_name,
            task_language=task_lang,
            same_scene_ids=same_scene_ids,
            names=names,
            languages=languages,
        )
        if object_decision is not None:
            decisions.append(object_decision)
            continue

        direction_decision = _find_direction_swap(
            task_id=task_id,
            task_name=task_name,
            task_language=task_lang,
            same_scene_name_to_id=scene_to_name_to_id[scene],
            names=names,
            languages=languages,
        )
        if direction_decision is not None:
            decisions.append(direction_decision)
            continue

        parsed = _extract_object_mask(_strip_scene_prefix(task_name))
        skip_reason = "no_object_match_and_no_direction_match"
        if parsed is None:
            skip_reason = "no_object_pattern_and_no_direction_match"
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
                skip_reason=skip_reason,
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

