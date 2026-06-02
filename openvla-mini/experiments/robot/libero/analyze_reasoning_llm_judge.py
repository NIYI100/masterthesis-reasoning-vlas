#!/usr/bin/env python3
"""
Offline LLM-judge analysis for LIBERO reasoning traces against libero_reasonings.json.

This mirrors the row-join behavior from analyze_reasoning_rouge.py, but scores semantic
equivalence with an LLM judge instead of ROUGE-L.

Expected output score range per field: [0.0, 1.0]
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import glob
import io
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None  # type: ignore

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


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
ANALYSIS_KEYS = ("strict", "mean", "weighted_mean", "max")

DEFAULT_JUDGE_SYSTEM_PROMPT = """You are a strict semantic equivalence evaluator for robot reasoning.
You compare CANDIDATE reasoning against GROUND_TRUTH reasoning.

Output format (mandatory):
- Return only one JSON object with exactly these keys:
{"plan": number|null, "subtask_reasoning": number|null, "subtask": number|null, "move_reasoning": number|null, "move": number|null, "whole": number|null}
- Each number must be in [0.0, 1.0].
- Use null only if one or both texts for that field are missing/empty.
- No prose, no markdown, no extra keys.

Critical semantics rules:
1) Contradictions in action intent must score very low (0.0-0.15), e.g.:
   - "pick up / grasp / close gripper" vs "put down / lay down / release / open gripper"
   - opposite directions that change intent (left vs right, up vs down, forward vs backward)
2) Equivalent wording/paraphrase should score very high (0.95-1.0), e.g.:
   - "move slightly left" vs "move left slightly"
   - "grasp the mug" vs "pick up the mug"
3) Word order differences alone should not be penalized much.
4) Wrong target object, wrong spatial relation, or wrong manipulation stage should be strongly penalized.
5) If candidate is heavily garbled/ungrammatical and intent cannot be recovered, score <= 0.25.
6) For PLAN: compare ordered intent sequence. Missing major steps or reversed stage order should lower score sharply.
7) For WHOLE: aggregate semantic equivalence across all available text fields.

Rubric anchors:
- 1.00: semantically identical and fully correct
- 0.95: near-identical paraphrase
- 0.75: mostly same intent, minor omissions
- 0.50: mixed; some overlap but key info differs
- 0.25: mostly different / uncertain intent
- 0.00: opposite or contradictory intent
"""

# Tailored for Gemma 4 family (including MoE) as a JSON-only LLM judge: emphasize
# phase tolerance, object/goal priority, and format-agnostic plan comparison. Inputs
# may be pre-normalized (plan as ordered step string; move axis order canonicalized).
GEMMA_JUDGE_SYSTEM_PROMPT = """You are a precise, fair semantic equivalence judge for **robot task reasoning** in simulation (LIBERO-style).
You compare a CANDIDATE to GROUND_TRUTH for the *same* reasoning fields. The goal is to score how well the candidate expresses the same intended task and reasoning, not identical wording.

## Multimodal context
- You may receive one scene image together with structured text payload.
- Use the image to interpret spatial relations, object identity, manipulation stage, and motion plausibility.
- Ground-truth text remains the primary supervision target; image context is used to disambiguate or verify semantics.
- If image evidence and candidate text conflict, penalize the candidate accordingly.
- If image evidence and ground-truth text conflict, trust ground-truth text for scoring but reduce certainty (avoid extreme scores unless contradiction is clear).

## Output (mandatory, strict)
Return **only** one JSON object, no other text, with exactly these keys:
{"plan": number|null, "subtask_reasoning": number|null, "subtask": number|null, "move_reasoning": number|null, "move": number|null, "whole": number|null}
- Each value is a number in [0.0, 1.0] or null.
- Use null if and only if that field is missing/empty in **both** or **one** of the sides (treat "cannot score" as null).
- No markdown, no code fences, no comments, no trailing text.

## Pre-normalization (already applied before you see the text)
- **PLAN** may be shown as a single ordered list of steps separated by " | " (e.g. step0 | step1 | ...), even if the original candidate was a dict or inline string. Judge **semantics and order of intent**, not braces or key syntax.
- **MOVE** may have axis conjunctions in a canonical order (e.g. "back and right" vs "right and back" normalized). Treat that as the same *intent* if axis words match.

## Scoring principles (Gemma-4-friendly)
1) **Object and goal over wording**: If both refer to the same manipuland and task goal, prefer high scores even when nouns differ ("cabinet" vs "top drawer of the cabinet", "ketchup bottle" vs "ketchup") **unless** the candidate picks a *different* object or wrong relation (e.g. wrong bowl, wrong surface).
2) **Adjacent phase tolerance (subtask / reasoning)**: Manipulation often progresses as: *approach / move-to / align* → *grasp* → *move / transport* → *place / release* → *stop / reset*. If the candidate is **one** logical phase before or after but still the same object and end goal, score **subtask** in about **0.55–0.80** and **subtask_reasoning** similarly, **not** near zero. Reserve very low subtask scores for wrong object, wrong goal, or contradictory stage (e.g. "release" when GT is "grasp").
3) **Movement equivalence**: Paraphrases and reorderings of the same low-level move are near-perfect (0.9–1.0): e.g. "move slightly left" / "move left slowly"; "down and forward" / "forward and down" (after normalization). Penalize strongly only for **incompatible** motions (e.g. left vs right, up vs down, rotate vs straight translation when that changes the intent) or **stop** vs non-stop when that matters.
4) **MOVE REASONING**: Judge spatial rationale consistency with the MOVE string; allow paraphrase. Penalize if the *explained* direction contradicts the move or the GT situation.
5) **PLAN**: Compare the **ordered** sequence of high-level intentions. If the candidate omits, adds, or reorders a **major** stage relative to GT, lower the plan score. Minor wording differences in the same step are fine.
6) **WHOLE (aggregate)**: Down-weight tiny motion phrasing. Weight intent roughly: plan + subtask + subtask_reasoning (combined ~70%), move + move_reasoning (~30%). If the high-level intent matches well, **whole** should stay **moderate to high** even if **move** alone is imperfect.
7) **Image-grounded tie-break**: When text-only comparison is ambiguous, use image evidence to choose between close scores. For example: wrong object instance, wrong side/direction, impossible gripper stage, or mismatched drawer/cabinet state should lower score.

## Contrastive anchors
- 1.0: Same intent, object, and stage; wording may differ.
- 0.85–0.95: Same goal/object; small omissions or one adjacent phase.
- 0.55–0.75: Same overall task; phase lag/lead or partial mismatch in motion.
- 0.25–0.50: Partly related but key intent differs; risky interpretation.
- 0.0–0.2: Wrong object, wrong goal, or clearly contradictory action.

## Hard fail (0.0–0.15)
Opposite manipulation (pick vs place), clear direction contradiction (left vs right, up vs down) when that flips the intended motion, or wrong target object/location.
"""


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


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _try_parse_structured_plan_string(text: str) -> Optional[Any]:
    """
    Parse dict/list plan representation from a string (JSON, Python literal, etc.).
    """
    s = text.strip()
    if not s:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            out = parser(s)
        except Exception:
            continue
        if isinstance(out, (dict, list, tuple)):
            return out
    return None


def _flatten_plan_value(value: Any) -> str:
    """Turn dict/list plan into a single ordered string for comparison."""
    if value is None:
        return ""
    if isinstance(value, dict):
        items: list[tuple[Any, str]] = []
        for k, v in value.items():
            try:
                ik = int(k)  # type: ignore[arg-type]
            except Exception:
                ik = k
            sv = str(v).strip()
            if sv:
                items.append((ik, sv))
        if not items:
            return ""
        if all(isinstance(a[0], int) for a in items):
            items.sort(key=lambda x: x[0])
        else:
            items.sort(key=lambda x: (str(x[0]),))
        return " | ".join(t[1] for t in items)
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return " | ".join(parts)
    return str(value).strip()


def normalize_plan_for_judge(text: Optional[str]) -> Optional[str]:
    """
    Unify plan formatting: dict-like strings and JSON become 'step | step | ...';
    plain strings get whitespace collapsed.
    """
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    parsed = _try_parse_structured_plan_string(raw)
    if parsed is not None:
        flat = _flatten_plan_value(parsed)
        if flat:
            return flat
    return _collapse_whitespace(raw) or None


def _move_conjunction_canonical(text: str) -> str:
    """
    Canonicalize commutative 'and' in simple move commands (axis order only).
    """
    t = text.strip().lower()
    if not t.startswith("move "):
        return t
    rest = t[len("move ") :].strip()
    if " and " not in rest:
        return t
    # Only rewrite short axis-style conjunctions (avoid breaking "open gripper and move up")
    parts = [p.strip() for p in rest.split(" and ") if p.strip()]
    if len(parts) < 2:
        return t
    axis_tokens = {
        "back",
        "forward",
        "left",
        "right",
        "up",
        "down",
        "clockwise",
        "counterclockwise",
        "counter-clockwise",
    }
    first_ok = all(
        (p.split()[0] in axis_tokens)
        or p.startswith(("rotate", "turn"))
        or (p in axis_tokens)
        for p in parts
    )
    if not first_ok:
        return t
    norm = sorted(parts, key=lambda x: x.lower())
    return "move " + " and ".join(norm)


def normalize_move_for_judge(text: Optional[str]) -> Optional[str]:
    """
    Lowercase, collapse whitespace, normalize common punctuation; canonicalize
    commutative axis 'and' lists for simple 'move ...' templates.
    """
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    t = t.lower()
    t = re.sub(r"\s*,\s*", " ", t)
    t = _collapse_whitespace(t)
    t = _move_conjunction_canonical(t)
    return t or None


def normalize_subtask_for_judge(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    t = _collapse_whitespace(str(text).strip().lower())
    return t or None


def normalize_free_text_for_judge(text: Optional[str]) -> Optional[str]:
    """Light cleanup for reasoning paragraphs (no aggressive synonym rewrite)."""
    if text is None:
        return None
    t = _collapse_whitespace(str(text).strip())
    return t or None


def apply_judge_field_normalization(
    out_key: str,
    candidate: Optional[str],
    ground_truth: Optional[str],
    *,
    enabled: bool,
) -> tuple[Optional[str], Optional[str]]:
    if not enabled:
        return candidate, ground_truth
    if out_key == "plan":
        return normalize_plan_for_judge(candidate), normalize_plan_for_judge(ground_truth)
    if out_key == "move":
        return normalize_move_for_judge(candidate), normalize_move_for_judge(ground_truth)
    if out_key == "subtask":
        return normalize_subtask_for_judge(candidate), normalize_subtask_for_judge(ground_truth)
    if out_key in ("subtask_reasoning", "move_reasoning"):
        return (
            normalize_free_text_for_judge(candidate),
            normalize_free_text_for_judge(ground_truth),
        )
    return candidate, ground_truth


def _resolve_judge_system_prompt(
    prompt_file: Optional[str], judge_preset: str
) -> str:
    if prompt_file:
        prompt_path = Path(prompt_file).expanduser().resolve()
        if not prompt_path.exists():
            raise SystemExit(f"Prompt file not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")
    if judge_preset == "gemma":
        return GEMMA_JUDGE_SYSTEM_PROMPT
    return DEFAULT_JUDGE_SYSTEM_PROMPT


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
        # Support both relative and absolute wildcard patterns (Path.glob does not
        # accept absolute patterns).
        has_wildcard = any(tok in raw for tok in ("*", "?", "["))
        if has_wildcard:
            for cand_str in sorted(glob.glob(raw, recursive=True)):
                cand = Path(cand_str)
                if cand.is_file():
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
        step_any = record.get("env_step")
    step_id = _as_nonneg_int(step_any)
    if ep_path is None or demo_id is None or step_id is None:
        return None
    step_id = step_id + int(step_offset)
    if step_id < 0:
        return None
    return ep_path, demo_id, str(step_id)


def _build_task_id_to_ep_path_map(
    reasonings_by_ep: dict[str, Any],
    libero_root: Path,
    suite_name: str,
) -> dict[int, str]:
    try:
        sys.path.append(str(libero_root))
        from libero.libero import benchmark  # type: ignore
    except Exception:
        return {}

    try:
        suite_cls = benchmark.get_benchmark_dict()[suite_name]
        suite = suite_cls()
    except Exception:
        return {}

    available_eps = set(reasonings_by_ep.keys())
    out: dict[int, str] = {}
    for task_id in range(int(getattr(suite, "n_tasks", 0))):
        task = suite.get_task(task_id)
        expected_ep = f"{task.name}_demo.hdf5"
        if expected_ep in available_eps:
            out[int(task_id)] = expected_ep
    return out


def _to_inferred_direct_key(
    record: dict[str, Any],
    task_id_to_ep_path: dict[int, str],
    step_offset: int,
) -> Optional[tuple[str, str, str]]:
    task_id = _as_nonneg_int(record.get("task_id"))
    episode_idx = _as_nonneg_int(record.get("episode_idx"))
    env_step = _as_nonneg_int(record.get("env_step"))
    if task_id is None or episode_idx is None or env_step is None:
        return None
    ep_path = task_id_to_ep_path.get(task_id)
    if ep_path is None:
        return None
    step = env_step + int(step_offset)
    if step < 0:
        return None
    return ep_path, str(episode_idx), str(step)


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


def _stats_payload_running(stats: RunningStats) -> dict[str, float]:
    return {
        "count": int(stats.count),
        "total": float(stats.total),
        "total_sq": float(stats.total_sq),
    }


def _running_stats_from_payload(payload: dict[str, Any]) -> RunningStats:
    return RunningStats(
        count=int(payload.get("count", 0)),
        total=float(payload.get("total", 0.0)),
        total_sq=float(payload.get("total_sq", 0.0)),
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _is_local_endpoint(endpoint: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(str(endpoint))
    except Exception:
        return False
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _extract_first_json_object(text: str) -> Optional[dict[str, Any]]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    snippet = m.group(0)
    try:
        parsed = json.loads(snippet)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _init_stats_by_analysis() -> dict[str, dict[str, RunningStats]]:
    stats_by_analysis: dict[str, dict[str, RunningStats]] = {}
    for analysis_key in ANALYSIS_KEYS:
        stats_by_analysis[analysis_key] = {spec.out_key: RunningStats() for spec in FIELD_SPECS}
        stats_by_analysis[analysis_key]["whole"] = RunningStats()
    return stats_by_analysis


def _stats_payload_running_all(
    stats_by_analysis: dict[str, dict[str, RunningStats]],
) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for analysis_key in ANALYSIS_KEYS:
        out[analysis_key] = {
            field_key: _stats_payload_running(stats_by_analysis[analysis_key][field_key])
            for field_key in ("plan", "subtask_reasoning", "subtask", "move_reasoning", "move", "whole")
        }
    return out


def _summary_payload_all(
    stats_by_analysis: dict[str, dict[str, RunningStats]],
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for analysis_key in ANALYSIS_KEYS:
        out[analysis_key] = _compute_summary_payload(stats_by_analysis[analysis_key])
    return out


def _load_running_stats_all(
    payload: dict[str, Any],
    stats_by_analysis: dict[str, dict[str, RunningStats]],
) -> None:
    if not isinstance(payload, dict):
        return
    for analysis_key in ANALYSIS_KEYS:
        per_analysis = payload.get(analysis_key)
        if not isinstance(per_analysis, dict):
            continue
        for field_key in ("plan", "subtask_reasoning", "subtask", "move_reasoning", "move", "whole"):
            if isinstance(per_analysis.get(field_key), dict):
                stats_by_analysis[analysis_key][field_key] = _running_stats_from_payload(per_analysis[field_key])


def _window_steps(
    base_step: int,
    row: dict[str, Any],
    *,
    window_mode: str,
    window_max_steps: int,
) -> list[int]:
    if window_mode == "strict":
        return [base_step]
    requested = _as_nonneg_int(row.get("control_actions_from_forward"))
    if requested is None or requested <= 0:
        requested = 1
    if window_max_steps > 0:
        requested = min(requested, int(window_max_steps))
    return [base_step + d for d in range(requested)]


def _weighted_mean(values: list[float], deltas: list[int], alpha: float) -> Optional[float]:
    if not values or not deltas or len(values) != len(deltas):
        return None
    alpha_safe = float(alpha)
    if alpha_safe <= 0.0:
        alpha_safe = 1e-6
    num = 0.0
    den = 0.0
    for v, d in zip(values, deltas):
        w = alpha_safe ** max(0, int(d))
        num += w * float(v)
        den += w
    if den <= 0.0:
        return None
    return num / den


def _aggregate_step_scores(
    per_step_scores: list[tuple[int, dict[str, Optional[float]]]],
    *,
    base_step: int,
    alpha: float,
) -> dict[str, dict[str, Optional[float]]]:
    out: dict[str, dict[str, Optional[float]]] = {k: {} for k in ANALYSIS_KEYS}
    strict_by_step = {int(step): scores for step, scores in per_step_scores}
    strict_scores = strict_by_step.get(int(base_step), {})
    for field_key in ("plan", "subtask_reasoning", "subtask", "move_reasoning", "move", "whole"):
        strict_v = strict_scores.get(field_key)
        out["strict"][field_key] = strict_v if strict_v is None else float(strict_v)

        vals: list[float] = []
        deltas: list[int] = []
        for step_i, step_scores in per_step_scores:
            v = step_scores.get(field_key)
            if v is None:
                continue
            vals.append(float(v))
            deltas.append(int(step_i) - int(base_step))
        if vals:
            out["mean"][field_key] = float(sum(vals) / float(len(vals)))
            out["max"][field_key] = float(max(vals))
            out["weighted_mean"][field_key] = _weighted_mean(vals, deltas, alpha)
        else:
            out["mean"][field_key] = None
            out["max"][field_key] = None
            out["weighted_mean"][field_key] = None
    return out


def _image_to_data_url(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        if not raw:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    if np is None:
        return None
    arr = np.asarray(value)
    if arr.size <= 0:
        return None
    if arr.dtype == np.object_:
        return None
    if arr.ndim >= 4:
        arr = arr[0]
    if arr.ndim == 1 and arr.dtype == np.uint8:
        raw = arr.tobytes()
        if not raw:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    if Image is None:
        return None
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype("uint8"), mode="L")
    elif arr.ndim == 3:
        if arr.shape[-1] == 1:
            img = Image.fromarray(arr[..., 0].astype("uint8"), mode="L")
        else:
            arr_rgb = arr[..., :3].astype("uint8")
            img = Image.fromarray(arr_rgb, mode="RGB")
    else:
        return None
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    raw = buf.getvalue()
    if not raw:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")


class RldsPrimaryImageResolver:
    def __init__(self, *, data_dir: Path, camera_keys: list[str], dataset_name: str = "libero_lm_90") -> None:
        self.data_dir = data_dir
        self.dataset_name = str(dataset_name).strip() or "libero_lm_90"
        self.camera_keys = [k.strip() for k in camera_keys if str(k).strip()]
        if not self.camera_keys:
            self.camera_keys = ["agentview_rgb", "image", "rgb"]
        self._h5_cache: dict[str, Any] = {}
        self._basename_to_path: Optional[dict[str, Path]] = None
        self._tfds_dataset = None
        self._tfds_index: Optional[dict[tuple[str, str], int]] = None
        self._tfds_episode_key: Optional[tuple[str, str]] = None
        self._tfds_episode_frames: Optional[list[Optional[str]]] = None
        self.fetch_attempts = 0
        self.fetch_hits = 0
        self.fetch_misses = 0
        self.fetch_errors = 0

    def _build_h5_index(self) -> dict[str, Path]:
        if self._basename_to_path is not None:
            return self._basename_to_path
        index: dict[str, Path] = {}
        try:
            if self.data_dir.exists():
                for p in self.data_dir.rglob("*.hdf5"):
                    index[p.name] = p.resolve()
        except Exception:
            pass
        self._basename_to_path = index
        return index

    def _resolve_h5_path(self, ep_path: str) -> Optional[Path]:
        p = Path(ep_path)
        if p.is_absolute() and p.exists():
            return p
        cand = (self.data_dir / ep_path).resolve()
        if cand.exists():
            return cand
        by_name = self._build_h5_index().get(p.name)
        if by_name is not None and by_name.exists():
            return by_name
        return None

    def _open_h5(self, h5_path: Path) -> Optional[Any]:
        if h5py is None:
            return None
        key = str(h5_path)
        if key in self._h5_cache:
            return self._h5_cache[key]
        h5f = h5py.File(str(h5_path), "r")
        self._h5_cache[key] = h5f
        return h5f

    def _demo_paths(self, demo_id: str) -> list[str]:
        did = str(demo_id).strip()
        return [
            f"data/demo_{did}/obs",
            f"demo_{did}/obs",
            f"data/{did}/obs",
            f"{did}/obs",
            f"data/demo_{did}/observations",
            f"demo_{did}/observations",
        ]

    def _pick_image_dataset(self, obs_group: Any) -> Optional[Any]:
        for camera_key in self.camera_keys:
            if camera_key in obs_group:
                return obs_group[camera_key]
        for k in obs_group.keys():
            lk = str(k).lower()
            if any(c in lk for c in ("wrist", "hand")):
                continue
            if "image" in lk or "rgb" in lk:
                return obs_group[k]
        return None

    @staticmethod
    def _to_str(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _pick_image_from_obs_dict(self, obs_dict: dict[str, Any]) -> Optional[Any]:
        for camera_key in self.camera_keys:
            if camera_key in obs_dict:
                return obs_dict[camera_key]
        for k, v in obs_dict.items():
            lk = str(k).lower()
            if "wrist" in lk or "hand" in lk:
                continue
            if "image" in lk or "rgb" in lk:
                return v
        return None

    def _ensure_tfds_index(self) -> Optional[dict[tuple[str, str], int]]:
        if self._tfds_index is not None:
            return self._tfds_index
        try:
            import tensorflow_datasets as tfds  # type: ignore
        except Exception:
            return None
        try:
            ds = tfds.load(self.dataset_name, split="train", data_dir=str(self.data_dir))
        except Exception:
            return None
        index: dict[tuple[str, str], int] = {}
        for idx, ep in enumerate(tfds.as_numpy(ds)):
            meta = ep.get("episode_metadata")
            if not isinstance(meta, dict):
                tm = ep.get("traj_metadata")
                if isinstance(tm, dict):
                    meta = tm.get("episode_metadata")
            if not isinstance(meta, dict):
                continue
            ep_path = self._to_str(meta.get("file_path", "")).strip()
            demo_id = self._to_str(meta.get("demo_id", "")).strip()
            if ep_path and demo_id:
                index[(ep_path, demo_id)] = idx
        self._tfds_dataset = ds
        self._tfds_index = index
        return index

    def _load_tfds_episode_frames(self, *, ep_path: str, demo_id: str) -> Optional[list[Optional[str]]]:
        index = self._ensure_tfds_index()
        if index is None or self._tfds_dataset is None:
            return None
        key = (str(ep_path), str(demo_id))
        episode_idx = index.get(key)
        if episode_idx is None:
            # try basename fallback for ep_path
            ep_base = Path(ep_path).name
            for (k_ep, k_demo), k_idx in index.items():
                if Path(k_ep).name == ep_base and str(k_demo) == str(demo_id):
                    episode_idx = k_idx
                    key = (k_ep, k_demo)
                    break
        if episode_idx is None:
            return None
        if self._tfds_episode_key == key and self._tfds_episode_frames is not None:
            return self._tfds_episode_frames
        try:
            import tensorflow_datasets as tfds  # type: ignore

            episode = next(iter(tfds.as_numpy(self._tfds_dataset.skip(int(episode_idx)).take(1))))
            steps = episode.get("steps")
            if steps is None:
                return None
            frames: list[Optional[str]] = []
            for step in steps:
                if isinstance(step, dict):
                    step_np = step
                else:
                    step_np = tfds.as_numpy(step)
                obs = step_np.get("observation")
                if not isinstance(obs, dict):
                    frames.append(None)
                    continue
                img = self._pick_image_from_obs_dict(obs)
                frames.append(_image_to_data_url(img))
            self._tfds_episode_key = key
            self._tfds_episode_frames = frames
            return frames
        except Exception:
            return None

    def fetch_data_url(self, *, ep_path: str, demo_id: str, step: int) -> Optional[str]:
        self.fetch_attempts += 1
        try:
            h5_path = self._resolve_h5_path(ep_path)
            step_i = int(step)
            if h5_path is not None:
                h5f = self._open_h5(h5_path)
                if h5f is None:
                    self.fetch_errors += 1
                    return None
                obs_group = None
                for p in self._demo_paths(demo_id):
                    if p in h5f:
                        obs_group = h5f[p]
                        break
                if obs_group is not None:
                    ds = self._pick_image_dataset(obs_group)
                    if ds is not None and 0 <= step_i < int(ds.shape[0]):
                        payload = _image_to_data_url(ds[step_i])
                        if payload is not None:
                            self.fetch_hits += 1
                            return payload
            frames = self._load_tfds_episode_frames(ep_path=str(ep_path), demo_id=str(demo_id))
            if frames is not None and 0 <= step_i < len(frames):
                payload = frames[step_i]
                if payload is not None:
                    self.fetch_hits += 1
                    return payload
            self.fetch_misses += 1
            return None
        except Exception:
            self.fetch_errors += 1
            return None


class LLMJudgeClient:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str,
        timeout_s: float,
        temperature: float,
        max_retries: int,
        retry_sleep_s: float,
        system_prompt: str,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.max_retries = int(max_retries)
        self.retry_sleep_s = float(retry_sleep_s)
        self.system_prompt = str(system_prompt)
        self.cache: dict[str, dict[str, Optional[float]]] = {}
        self.calls_total = 0
        self.calls_cached = 0
        self.calls_failed = 0

    def score(self, payload: dict[str, Any], *, image_data_url: Optional[str] = None) -> dict[str, Optional[float]]:
        cache_key = json.dumps(
            {"payload": payload, "image_data_url": image_data_url},
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self.cache:
            self.calls_cached += 1
            return dict(self.cache[cache_key])

        self.calls_total += 1

        user_prompt = json.dumps(payload, ensure_ascii=False)

        user_content: Any = user_prompt
        if image_data_url:
            user_content = [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": user_prompt},
            ]
        req_body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
        }

        url = f"{self.endpoint}/chat/completions"
        raw_content: Optional[str] = None
        for attempt in range(self.max_retries + 1):
            try:
                headers = {"Content-Type": "application/json"}
                if str(self.api_key).strip():
                    headers["Authorization"] = f"Bearer {self.api_key}"
                req = urllib.request.Request(
                    url=url,
                    data=json.dumps(req_body).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(body)
                raw_content = (
                    parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if isinstance(parsed, dict)
                    else ""
                )
                break
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt >= self.max_retries:
                    self.calls_failed += 1
                    out = {
                        "plan": None,
                        "subtask_reasoning": None,
                        "subtask": None,
                        "move_reasoning": None,
                        "move": None,
                        "whole": None,
                    }
                    self.cache[cache_key] = dict(out)
                    return out
                time.sleep(self.retry_sleep_s)

        score_obj = _extract_first_json_object(raw_content or "") or {}
        out: dict[str, Optional[float]] = {}
        for key in ("plan", "subtask_reasoning", "subtask", "move_reasoning", "move", "whole"):
            v = score_obj.get(key)
            if v is None:
                out[key] = None
                continue
            try:
                fv = float(v)
            except Exception:
                out[key] = None
                continue
            if fv < 0.0:
                fv = 0.0
            if fv > 1.0:
                fv = 1.0
            out[key] = fv

        self.cache[cache_key] = dict(out)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute LLM-judge similarity by reasoning field from LIBERO traces.")
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
    parser.add_argument(
        "--row-offset",
        type=int,
        default=0,
        help="Skip the first N loaded rows before evaluation (for manual resume/chunking).",
    )
    parser.add_argument(
        "--checkpoint-json",
        default=None,
        help="Optional checkpoint JSON path for periodic state saves.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N processed rows (0 disables periodic checkpoints).",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume counters/state from --checkpoint-json and continue after last processed row.",
    )
    parser.add_argument(
        "--samples-append",
        action="store_true",
        help="Append to --samples-jsonl instead of overwriting (useful for resumes).",
    )
    parser.add_argument(
        "--llm-endpoint",
        default=os.environ.get("LLM_JUDGE_ENDPOINT", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL (without trailing /chat/completions).",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("LLM_JUDGE_MODEL", "qwen3.5"),
        help="Judge model name (example: Qwen/Qwen3.5-72B-Instruct).",
    )
    parser.add_argument(
        "--llm-api-key",
        default=os.environ.get("LLM_JUDGE_API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        help="API key (or set LLM_JUDGE_API_KEY / OPENAI_API_KEY).",
    )
    parser.add_argument("--llm-timeout-s", type=float, default=60.0, help="LLM request timeout (seconds).")
    parser.add_argument("--llm-temperature", type=float, default=0.0, help="LLM sampling temperature.")
    parser.add_argument("--llm-max-retries", type=int, default=3, help="Retries on request errors.")
    parser.add_argument("--llm-retry-sleep-s", type=float, default=1.5, help="Sleep between retries (seconds).")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print live progress every N processed rows (0 disables periodic updates).",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional path to a custom system prompt file for the judge.",
    )
    parser.add_argument(
        "--judge-preset",
        choices=("default", "gemma"),
        default="default",
        help="Built-in system prompt: 'default' (original rubric) or 'gemma' (Gemma-4-oriented: phase tolerance, plan/move normalization awareness). Ignored if --prompt-file is set.",
    )
    parser.add_argument(
        "--no-field-normalization",
        action="store_true",
        help="Disable pre-LLM normalization of plan/move/subtask text (dict plan flattening, move axis order, whitespace).",
    )
    parser.add_argument(
        "--use-images",
        action="store_true",
        help="Include primary observation image in the judge request when resolvable.",
    )
    parser.add_argument(
        "--image-source",
        choices=("rlds",),
        default="rlds",
        help="Image source backend for --use-images.",
    )
    parser.add_argument(
        "--rlds-data-dir",
        default="/home/hk-project-p0024638/uvrfq/hkfswork/ecot-lite_data",
        help="Directory containing LIBERO HDF5 files referenced by ep_path.",
    )
    parser.add_argument(
        "--rlds-primary-image-keys",
        default="agentview_rgb,image,rgb",
        help="Comma-separated primary camera dataset names to try in order.",
    )
    parser.add_argument(
        "--rlds-dataset-name",
        default="libero_lm_90",
        help="TFDS RLDS dataset name for image fallback (default: libero_lm_90).",
    )
    parser.add_argument(
        "--window-mode",
        choices=("chunk", "strict"),
        default="chunk",
        help="Scoring window mode: strict uses base step only, chunk uses control_actions_from_forward.",
    )
    parser.add_argument(
        "--window-max-steps",
        type=int,
        default=0,
        help="Optional cap for window length when --window-mode=chunk (0 = no cap).",
    )
    parser.add_argument(
        "--weight-decay-alpha",
        type=float,
        default=0.85,
        help="Decay alpha for weighted_mean over window steps (weight for delta d is alpha^d).",
    )
    args = parser.parse_args()

    if not args.llm_api_key and not _is_local_endpoint(args.llm_endpoint):
        raise SystemExit(
            "Missing API key. Pass --llm-api-key (or env var) for non-local endpoints. "
            "Local endpoints (localhost/127.0.0.1/0.0.0.0) are allowed without a key."
        )

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

    needed_ep_paths = {ep for _, (ep, _, _) in direct_candidates}
    reasonings_path = Path(args.reasonings_json).resolve()
    if not reasonings_path.exists():
        raise SystemExit(f"Reasonings file not found: {reasonings_path}")

    reasonings_by_ep = _load_reasonings_subset(
        reasonings_path,
        needed_ep_paths if not fallback_needed_task_ids else None,
    )

    task_id_to_ep_path = _build_task_id_to_ep_path_map(
        reasonings_by_ep,
        libero_root=Path(args.libero_root).resolve(),
        suite_name=args.task_suite_name,
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

    system_prompt = _resolve_judge_system_prompt(args.prompt_file, str(args.judge_preset))
    field_norm = not bool(args.no_field_normalization)

    judge = LLMJudgeClient(
        endpoint=args.llm_endpoint,
        model=args.llm_model,
        api_key=args.llm_api_key,
        timeout_s=args.llm_timeout_s,
        temperature=args.llm_temperature,
        max_retries=args.llm_max_retries,
        retry_sleep_s=args.llm_retry_sleep_s,
        system_prompt=system_prompt,
    )

    stats_by_analysis = _init_stats_by_analysis()

    matched = 0
    unmatched = 0
    direct_matched = 0
    direct_explicit_matched = 0
    direct_inferred_matched = 0
    fallback_matched = 0
    unmatched_reasons: dict[str, int] = {}
    checkpoint_path: Optional[Path] = Path(args.checkpoint_json).resolve() if args.checkpoint_json else None
    checkpoint_loaded = False
    effective_row_offset = max(0, int(args.row_offset))
    image_resolver: Optional[RldsPrimaryImageResolver] = None
    if args.use_images and args.image_source == "rlds":
        image_resolver = RldsPrimaryImageResolver(
            data_dir=Path(args.rlds_data_dir).resolve(),
            camera_keys=str(args.rlds_primary_image_keys).split(","),
            dataset_name=str(args.rlds_dataset_name),
        )

    if args.resume_from_checkpoint:
        if checkpoint_path is None:
            raise SystemExit("--resume-from-checkpoint requires --checkpoint-json.")
        if checkpoint_path.exists():
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            effective_row_offset = max(effective_row_offset, int(ckpt.get("last_processed_row_idx", 0)))
            matched = int(ckpt.get("rows_matched", matched))
            unmatched = int(ckpt.get("rows_unmatched", unmatched))
            direct_matched = int(ckpt.get("rows_matched_direct", direct_matched))
            direct_explicit_matched = int(
                ckpt.get("rows_matched_direct_explicit", direct_explicit_matched)
            )
            direct_inferred_matched = int(
                ckpt.get("rows_matched_direct_inferred", direct_inferred_matched)
            )
            fallback_matched = int(ckpt.get("rows_matched_fallback", fallback_matched))
            unmatched_reasons = dict(ckpt.get("unmatched_reasons", unmatched_reasons))

            stats_payload = ckpt.get("stats_running", {})
            if isinstance(stats_payload, dict):
                for key in ("plan", "subtask_reasoning", "subtask", "move_reasoning", "move", "whole"):
                    if isinstance(stats_payload.get(key), dict):
                        stats_by_analysis["strict"][key] = _running_stats_from_payload(stats_payload[key])
            _load_running_stats_all(ckpt.get("stats_running_analyses", {}), stats_by_analysis)

            judge_payload = ckpt.get("judge_call_stats", {})
            judge.calls_total = int(judge_payload.get("calls_total", 0))
            judge.calls_cached = int(judge_payload.get("calls_cached", 0))
            judge.calls_failed = int(judge_payload.get("calls_failed", 0))
            checkpoint_loaded = True

    total_rows_to_process = max(0, len(rows) - effective_row_offset)
    if total_rows_to_process == 0:
        print(
            f"All rows already covered (len(rows)={len(rows)}, effective_row_offset={effective_row_offset}).",
            file=sys.stderr,
        )

    samples_handle = None
    samples_flush_counter = 0
    if args.samples_jsonl:
        out_samples = Path(args.samples_jsonl).resolve()
        out_samples.parent.mkdir(parents=True, exist_ok=True)
        samples_mode = "a" if (args.samples_append or checkpoint_loaded) else "w"
        samples_handle = out_samples.open(samples_mode, encoding="utf-8")

    started_at = time.time()
    last_progress_chars = 0
    processed_count = 0
    last_processed_row_idx = effective_row_offset

    def _print_progress(force: bool = False, processed: int = 0) -> None:
        nonlocal last_progress_chars
        if not force and args.progress_every <= 0:
            return
        if not force and processed % args.progress_every != 0:
            return
        msg = (
            f"[llm-judge] rows {processed}/{total_rows_to_process} | matched {matched} | "
            f"unmatched {unmatched} | judge_calls {judge.calls_total} | "
            f"cache_hits {judge.calls_cached} | failed {judge.calls_failed} | "
            f"elapsed {_format_elapsed(time.time() - started_at)}"
        )
        pad = max(0, last_progress_chars - len(msg))
        sys.stderr.write("\r" + msg + (" " * pad))
        sys.stderr.flush()
        last_progress_chars = len(msg)

    def _write_checkpoint(force: bool, processed_row_idx: int) -> None:
        if checkpoint_path is None:
            return
        if not force:
            if args.checkpoint_every <= 0:
                return
            if processed_count % int(args.checkpoint_every) != 0:
                return
        payload = {
            "last_processed_row_idx": int(processed_row_idx),
            "rows_total_loaded": int(len(rows)),
            "row_offset_effective": int(effective_row_offset),
            "rows_matched": int(matched),
            "rows_unmatched": int(unmatched),
            "rows_matched_direct": int(direct_matched),
            "rows_matched_direct_explicit": int(direct_explicit_matched),
            "rows_matched_direct_inferred": int(direct_inferred_matched),
            "rows_matched_fallback": int(fallback_matched),
            "unmatched_reasons": unmatched_reasons,
            "stats_running": {
                key: _stats_payload_running(stats_by_analysis["strict"][key])
                for key in ("plan", "subtask_reasoning", "subtask", "move_reasoning", "move", "whole")
            },
            "stats_running_analyses": _stats_payload_running_all(stats_by_analysis),
            "judge_call_stats": {
                "calls_total": int(judge.calls_total),
                "calls_cached": int(judge.calls_cached),
                "calls_failed": int(judge.calls_failed),
            },
            "updated_at_unix_s": time.time(),
        }
        _write_json_atomic(checkpoint_path, payload)

    for row_idx, (trace_path, line_no, row) in enumerate(rows, start=1):
        if row_idx <= effective_row_offset:
            continue

        processed_count += 1
        last_processed_row_idx = row_idx
        _print_progress(force=False, processed=processed_count)
        key = _to_direct_key(row, step_offset=args.step_offset)
        mode = "direct"
        if key is None:
            key = _to_inferred_direct_key(
                row,
                task_id_to_ep_path=task_id_to_ep_path,
                step_offset=args.step_offset,
            )
            mode = "direct_inferred"
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
            _write_checkpoint(force=False, processed_row_idx=row_idx)
            continue

        ep_path, demo_id, step = key
        step_base = _as_nonneg_int(step)
        if step_base is None:
            unmatched += 1
            unmatched_reasons["invalid_step"] = unmatched_reasons.get("invalid_step", 0) + 1
            _write_checkpoint(force=False, processed_row_idx=row_idx)
            continue
        demo_leaf = reasonings_by_ep.get(ep_path, {}).get(str(demo_id), {})
        strict_leaf = demo_leaf.get(str(step_base)) if isinstance(demo_leaf, dict) else None
        if not isinstance(strict_leaf, dict):
            unmatched += 1
            unmatched_reasons["missing_reasoning_leaf"] = unmatched_reasons.get("missing_reasoning_leaf", 0) + 1
            _write_checkpoint(force=False, processed_row_idx=row_idx)
            continue

        pred_reasoning = row.get("reasoning")
        if not isinstance(pred_reasoning, str) or not pred_reasoning.strip():
            unmatched += 1
            unmatched_reasons["missing_pred_reasoning"] = unmatched_reasons.get("missing_pred_reasoning", 0) + 1
            _write_checkpoint(force=False, processed_row_idx=row_idx)
            continue

        matched += 1
        if mode == "direct":
            direct_matched += 1
            direct_explicit_matched += 1
        elif mode == "direct_inferred":
            direct_matched += 1
            direct_inferred_matched += 1
        else:
            fallback_matched += 1

        candidate_steps = _window_steps(
            step_base,
            row,
            window_mode=str(args.window_mode),
            window_max_steps=int(args.window_max_steps),
        )
        valid_step_payloads: list[tuple[int, dict[str, Any]]] = []
        for step_i in candidate_steps:
            leaf_i = demo_leaf.get(str(step_i)) if isinstance(demo_leaf, dict) else None
            if not isinstance(leaf_i, dict):
                continue
            compare_payload: dict[str, Any] = {"fields": {}, "whole": {}}
            pred_whole_parts: list[str] = []
            ref_whole_parts: list[str] = []
            for spec in FIELD_SPECS:
                pred_t = parse_text_field_from_reasoning(pred_reasoning, spec.cot_tag)
                raw_ref = leaf_i.get(spec.gt_key)
                if spec.out_key == "plan" and isinstance(raw_ref, (dict, list, tuple)):
                    ref_t = _flatten_plan_value(raw_ref) or None
                else:
                    ref_t = _normalize_gt_text(raw_ref)
                pred_t, ref_t = apply_judge_field_normalization(
                    spec.out_key,
                    pred_t,
                    ref_t,
                    enabled=field_norm,
                )
                compare_payload["fields"][spec.out_key] = {
                    "ground_truth": ref_t,
                    "candidate": pred_t,
                }
                if pred_t:
                    pred_whole_parts.append(pred_t)
                if ref_t:
                    ref_whole_parts.append(ref_t)
            compare_payload["whole"] = {
                "ground_truth": " ".join(ref_whole_parts).strip(),
                "candidate": " ".join(pred_whole_parts).strip(),
            }
            valid_step_payloads.append((int(step_i), compare_payload))

        if not valid_step_payloads:
            unmatched += 1
            unmatched_reasons["missing_reasoning_leaf_window"] = (
                unmatched_reasons.get("missing_reasoning_leaf_window", 0) + 1
            )
            _write_checkpoint(force=False, processed_row_idx=row_idx)
            continue

        per_step_scores: list[tuple[int, dict[str, Optional[float]]]] = []
        row_has_image = False
        for step_i, compare_payload in valid_step_payloads:
            image_data_url: Optional[str] = None
            if image_resolver is not None:
                image_data_url = image_resolver.fetch_data_url(
                    ep_path=str(ep_path), demo_id=str(demo_id), step=int(step_i)
                )
                if image_data_url:
                    row_has_image = True
            judged = judge.score(compare_payload, image_data_url=image_data_url)
            per_step_scores.append((int(step_i), judged))

        row_scores = _aggregate_step_scores(
            per_step_scores,
            base_step=int(step_base),
            alpha=float(args.weight_decay_alpha),
        )
        for analysis_key in ANALYSIS_KEYS:
            for field_key in ("plan", "subtask_reasoning", "subtask", "move_reasoning", "move", "whole"):
                stats_by_analysis[analysis_key][field_key].add(row_scores[analysis_key].get(field_key))

        if samples_handle is not None:
            sample_row = {
                "trace_file": str(trace_path),
                "trace_line": line_no,
                "join_mode": mode,
                "task_id": row.get("task_id"),
                "episode_idx": row.get("episode_idx"),
                "env_step": row.get("env_step"),
                "ep_path": ep_path,
                "demo_id": str(demo_id),
                "step": str(step_base),
                "window_size_requested": len(candidate_steps),
                "window_size_valid": len(valid_step_payloads),
                "window_steps_valid": [int(s) for s, _ in valid_step_payloads],
                "row_has_image": bool(row_has_image),
                "scores": row_scores,
                "compare_payload": valid_step_payloads[0][1],
            }
            samples_handle.write(json.dumps(sample_row, ensure_ascii=False) + "\n")
            samples_flush_counter += 1
            if samples_flush_counter >= 50:
                samples_handle.flush()
                samples_flush_counter = 0

        _write_checkpoint(force=False, processed_row_idx=row_idx)

    _write_checkpoint(force=True, processed_row_idx=last_processed_row_idx)
    _print_progress(force=True, processed=processed_count)
    if last_progress_chars > 0:
        sys.stderr.write("\n")
        sys.stderr.flush()

    summary = {
        "trace_files": [str(p) for p in trace_paths],
        "reasonings_json": str(reasonings_path),
        "task_suite_name": args.task_suite_name,
        "step_offset": int(args.step_offset),
        "row_offset_effective": int(effective_row_offset),
        "rows_total": len(rows),
        "rows_matched": matched,
        "rows_unmatched": unmatched,
        "rows_matched_direct": direct_matched,
        "rows_matched_direct_explicit": direct_explicit_matched,
        "rows_matched_direct_inferred": direct_inferred_matched,
        "rows_matched_fallback": fallback_matched,
        "unmatched_reasons": unmatched_reasons,
        "llm_judge": _compute_summary_payload(stats_by_analysis["strict"]),
        "llm_judge_analyses": _summary_payload_all(stats_by_analysis),
        "judge_call_stats": {
            "calls_total": int(judge.calls_total),
            "calls_cached": int(judge.calls_cached),
            "calls_failed": int(judge.calls_failed),
        },
        "image_fetch_stats": {
            "enabled": bool(args.use_images),
            "source": str(args.image_source),
            "attempts": int(image_resolver.fetch_attempts) if image_resolver is not None else 0,
            "hits": int(image_resolver.fetch_hits) if image_resolver is not None else 0,
            "misses": int(image_resolver.fetch_misses) if image_resolver is not None else 0,
            "errors": int(image_resolver.fetch_errors) if image_resolver is not None else 0,
        },
        "judge_config": {
            "endpoint": args.llm_endpoint,
            "model": args.llm_model,
            "temperature": float(args.llm_temperature),
            "prompt_file": args.prompt_file,
            "judge_preset": str(args.judge_preset),
            "field_normalization": field_norm,
            "window_mode": str(args.window_mode),
            "window_max_steps": int(args.window_max_steps),
            "weight_decay_alpha": float(args.weight_decay_alpha),
            "use_images": bool(args.use_images),
            "image_source": str(args.image_source),
            "rlds_data_dir": str(Path(args.rlds_data_dir).resolve()),
            "rlds_dataset_name": str(args.rlds_dataset_name),
        },
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
            writer = csv.DictWriter(f, fieldnames=["analysis", "field", "count", "mean", "std"])
            writer.writeheader()
            for analysis_key in ANALYSIS_KEYS:
                block = summary["llm_judge_analyses"][analysis_key]
                for key in ("plan", "subtask_reasoning", "subtask", "move_reasoning", "move", "whole"):
                    payload = block[key]
                    writer.writerow(
                        {
                            "analysis": analysis_key,
                            "field": key,
                            "count": payload["count"],
                            "mean": payload["mean"],
                            "std": payload["std"],
                        }
                    )

    if args.samples_jsonl:
        if samples_handle is not None:
            samples_handle.flush()
            samples_handle.close()


if __name__ == "__main__":
    main()
