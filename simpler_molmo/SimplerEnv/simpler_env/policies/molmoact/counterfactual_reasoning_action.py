"""
Counterfactual second-pass action generation (same logic as MolmoAct repo).
Bundled here so SimplerEnv does not require PYTHONPATH to MolmoAct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch

_ACTION_ANCHOR_RE = re.compile(
    r"the action that the robot should take is\s*",
    flags=re.IGNORECASE | re.DOTALL,
)

_ACTION_OPEN_BRACKET_RE = re.compile(
    r"the action that the robot should take is\s*",
    re.IGNORECASE,
)


def repair_missing_action_open_bracket(text: str) -> str:
    """
    Continuation generation sometimes emits `... take is tok1, tok2, ...]` without a leading `[`,
    so bracket-based action parsing finds nothing. Insert `[` before the first `]` after the anchor.
    """

    for m in reversed(list(_ACTION_OPEN_BRACKET_RE.finditer(text))):
        pos = m.end()
        if pos >= len(text) or text[pos] == "[":
            continue
        end = text.find("]", pos)
        if end == -1:
            continue
        inner_region = text[pos:end]
        if "," not in inner_region:
            continue
        return text[:pos] + "[" + inner_region + "]" + text[end + 1 :]
    return text


def find_action_anchor_span(generated_assistant_text: str) -> Optional[tuple[int, int]]:
    m = _ACTION_ANCHOR_RE.search(generated_assistant_text)
    if not m:
        return None
    return m.start(), m.end()


def split_reasoning_and_action_anchor(
    generated_assistant_text: str,
) -> Optional[tuple[str, str]]:
    span = find_action_anchor_span(generated_assistant_text)
    if span is None:
        return None
    start, end = span
    reasoning = generated_assistant_text[:start]
    anchor = generated_assistant_text[start:end]
    return reasoning, anchor


def build_assistant_prefix_for_regeneration(
    original_generated_text: str,
    perturb_reasoning_fn: Callable[[str], str],
) -> Optional[str]:
    parts = split_reasoning_and_action_anchor(original_generated_text)
    if parts is None:
        return None
    reasoning, anchor = parts
    perturbed = perturb_reasoning_fn(reasoning)
    if perturbed is None:
        return None
    return str(perturbed) + anchor


def regenerate_action_under_perturbed_reasoning(
    *,
    model: Any,
    processor: Any,
    images: Any,
    user_prompt: str,
    original_generated_assistant_text: str,
    perturb_reasoning_fn: Callable[[str], str],
    unnorm_key: Optional[str],
    max_new_tokens: int = 256,
    device: Optional[torch.device] = None,
) -> tuple[Optional[List[List[float]]], Optional[str], Optional[str]]:
    """
    Returns (actions, error_message, full_assistant_text_on_success).
    ``full_assistant_text_on_success`` is the perturbed-prefix + continued decode (after bracket repair),
    for overlays and logging; None if regen did not produce a parsed action.
    """
    assistant_prefix = build_assistant_prefix_for_regeneration(
        original_generated_assistant_text, perturb_reasoning_fn
    )
    if assistant_prefix is None:
        return None, "counterfactual_skip:no_action_anchor_or_empty_perturbation", None

    messages = [
        {"role": "user", "content": [dict(type="text", text=user_prompt)]},
        {"role": "assistant", "content": [dict(type="text", text=assistant_prefix)]},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    inputs = processor(
        images=images,
        text=text,
        padding=True,
        return_tensors="pt",
    )
    if device is None:
        device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    new_tokens = generated_ids[:, inputs["input_ids"].size(1) :]
    suffix = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    full_text = repair_missing_action_open_bracket(assistant_prefix + suffix)
    actions = model.parse_action(full_text, unnorm_key=unnorm_key)
    if not actions:
        return None, "counterfactual_skip:parse_action_empty", None
    return actions, None, full_text


@dataclass
class MolmoActStepResult:
    action: Any
    annotated: Any
    trace: Any
    generated_text: str = ""
    action_under_perturbed_reasoning: Optional[List[Any]] = None
    counterfactual_error: Optional[str] = None


def identity_reasoning(reasoning: str) -> str:
    return reasoning


# Matches MolmoAct-style 2D points like [140, 225] inside the assistant reasoning string.
_XY_PAIR_RE = re.compile(r"\[(\d+)\s*,\s*(\d+)\]")


def _make_shift_xy_pairs_perturb(dx: float, dy: float) -> Callable[[str], str]:
    """Add (dx, dy) to every [x, y] integer pair; clip to [0, 255]."""

    def _fn(reasoning: str) -> str:
        def repl(m) -> str:
            x = int(max(0, min(255, round(int(m.group(1)) + dx))))
            y = int(max(0, min(255, round(int(m.group(2)) + dy))))
            return f"[{x},{y}]"

        return _XY_PAIR_RE.sub(repl, reasoning)

    return _fn


def _make_append_suffix_perturb(suffix: str) -> Callable[[str], str]:
    """Append a literal suffix (smoke test that the second pass sees a changed prefix)."""

    def _fn(reasoning: str) -> str:
        return reasoning + suffix

    return _fn


def get_reasoning_perturb_fn(fn_name: Optional[str]) -> Optional[Callable[[str], str]]:
    """
    Registry (CLI: ``--counterfactual-perturb-fn NAME``).

    Built-ins:

    - ``identity`` — no change.
    - ``append_suffix:TEXT`` — append TEXT to the reasoning prefix (use underscores, no spaces, e.g.
      ``append_suffix:_PTEST``), or quote in the shell if you need spaces.
    - ``shift_xy_pairs:DX,DY`` — add DX,DY to every ``[x, y]`` integer pair (model 0–255 space), clipped.
    """
    if fn_name is None:
        return None
    s = fn_name.strip()
    if not s or s.lower() in ("none", "null"):
        return None
    low = s.lower()
    if low == "identity":
        return identity_reasoning

    if low.startswith("append_suffix:"):
        suf = s.split(":", 1)[1]
        if not suf:
            raise ValueError("append_suffix: requires non-empty TEXT after the colon")
        return _make_append_suffix_perturb(suf)

    if low.startswith("shift_xy_pairs:"):
        rest = s.split(":", 1)[1].strip()
        parts = [p.strip() for p in rest.split(",")]
        if len(parts) != 2:
            raise ValueError("shift_xy_pairs: expects shift_xy_pairs:DX,DY")
        dx, dy = float(parts[0]), float(parts[1])
        return _make_shift_xy_pairs_perturb(dx, dy)

    raise ValueError(f"Unknown counterfactual reasoning perturbation: {fn_name!r}")
