import json
import re
import ast
import numpy as np
import os
import random
from pathlib import Path

"""
Reasoning looks like this:

PLAN:@{'0': 'move to the black bowl', '1': 'grasp the black bowl', '2': 'move the black bowl to the plate', '3': 'release the black bowl onto the plate'}
@VISIBLE OBJECTS:@{'akita black bowl 1': [[147, 70], [176, 104]], 'akita black bowl 2': [[105, 148], [119, 178]], 'new salad dressing 1': [[71, 70], [109, 87]], 'chocolate pudding 1': [[144, 126], [169, 141]], 'wooden cabinet 1': [[77, 0], [176, 59]]}
@SUBTASK REASONING:@The robot needs to move to the black bowl because it is currently positioned far away from it, while the bowl is located to the right of the plate.
@SUBTASK:@move to the black bowl
@MOVE REASONING:@The black bowl is positioned on the left side of the table, while the plate is on the right, so the robot should move back to create space and reach the bowl.
@MOVE:@move back slowly
@GRIPPER POSITION:@[37, 110] 
ACTION: <|extra_178|><|extra_142|><|extra_106|><|extra_9|><|extra_134|><|extra_254|><|extra_25|><|im_end|><|endoftext|>
"""

# Short registry keys -> CoT tag prefix (for word dropout / sentence shuffle / phrase swaps).
NL_SHORT_TO_TAG = {
    "plan": "PLAN:",
    "subtask_reasoning": "SUBTASK REASONING:",
    "subtask": "SUBTASK:",
    "move_reasoning": "MOVE REASONING:",
    "move": "MOVE:",
}

# Default per-task average steps file used by temporal noise modifiers.
# Prefer the shared hkfswork export by default.
DEFAULT_TEMPORAL_AVG_STEPS_JSON = (
    Path(__file__).resolve().parents[4]
    / "hkfswork"
    / "logs"
    / "z_final_results"
    / "full_reasoning_model"
    / "noise"
    / "all_modalities"
    / "avg_steps_executed_low_noise.json"
)
DEFAULT_TEMPORAL_AVG_STEPS_JSON_FALLBACK = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "logs"
    / "z_final_results"
    / "full_reasoning_model"
    / "noise"
    / "all_modalities"
    / "avg_steps_executed_low_noise.json"
)

# Emit at most one warning per process when bbox noise expected a VISIBLE OBJECTS
# block but could not match/modify it (format drift / regex mismatch safeguard).
_GAUSS_BBOX_NO_MATCH_WARNED = False

def _canonicalize_bidirectional_pairs(raw):
    """Drop duplicate undirected edges; sort by longest phrase first (regex alternation order)."""
    seen = set()
    out = []
    for a, b in raw:
        a, b = a.strip(), b.strip()
        if not a or not b:
            continue
        key = tuple(sorted((a.lower(), b.lower())))
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    out.sort(key=lambda ab: max(len(ab[0]), len(ab[1])), reverse=True)
    return tuple(out)


# Raw bidirectional (a<->b) opposites: multi-word LIBERO-style primitives + narrative synonyms.
# `invert_motion_phrases` matches longest keys first (word-boundary safe).
_MOTION_PHRASE_PAIR_RAW = (
    # --- Frequent primitives (dataset-informed opposites) ---
    ("stop", "start"),
    ("close gripper", "open gripper"),
    ("move down", "move up"),
    ("move left", "move right"),
    ("move forward", "move backward"),
    ("rotate clockwise", "rotate counterclockwise"),
    # --- Compound moves (comma-separated) ---
    ("move up, open gripper", "move down, close gripper"),
    ("move up, close gripper", "move down, open gripper"),
    ("move down, close gripper", "move up, open gripper"),
    ("move down, open gripper", "move up, close gripper"),
    ("move backward, open gripper", "move forward, close gripper"),
    ("move backward, close gripper", "move forward, open gripper"),
    ("move forward, open gripper", "move backward, close gripper"),
    ("move forward, close gripper", "move backward, open gripper"),
    # --- Diagonal / diagonal + rotate ---
    ("move forward right", "move backward left"),
    ("move forward left", "move backward right"),
    ("move backward left", "move forward right"),
    ("move backward right", "move forward left"),
    ("move left down", "move right up"),
    ("move right down", "move left up"),
    ("move left up", "move right down"),
    ("move right up", "move left down"),
    ("move forward down", "move backward up"),
    ("move backward up", "move forward down"),
    ("move forward up", "move backward down"),
    ("move backward down", "move forward up"),
    # --- Move + rotate (single comma) ---
    ("move right, rotate clockwise", "move left, rotate counterclockwise"),
    ("move left, rotate counterclockwise", "move right, rotate clockwise"),
    ("move left, rotate clockwise", "move right, rotate counterclockwise"),
    ("move right, rotate counterclockwise", "move left, rotate clockwise"),
    ("move up, rotate clockwise", "move down, rotate counterclockwise"),
    ("move up, rotate counterclockwise", "move down, rotate clockwise"),
    ("move down, rotate clockwise", "move up, rotate counterclockwise"),
    ("move down, rotate counterclockwise", "move up, rotate clockwise"),
    ("move forward, rotate clockwise", "move backward, rotate counterclockwise"),
    ("move forward, rotate counterclockwise", "move backward, rotate clockwise"),
    ("move backward, rotate clockwise", "move forward, rotate counterclockwise"),
    ("move backward, rotate counterclockwise", "move forward, rotate clockwise"),
    # --- Move + gripper (single comma) ---
    ("move right, open gripper", "move left, close gripper"),
    ("move left, open gripper", "move right, close gripper"),
    ("move right, close gripper", "move left, open gripper"),
    ("move left, close gripper", "move right, open gripper"),
    # --- Triple clauses ---
    ("move backward right, rotate clockwise", "move forward left, rotate counterclockwise"),
    ("move backward left, rotate counterclockwise", "move forward right, rotate clockwise"),
    ("move right up, open gripper", "move left down, close gripper"),
    ("move right up, close gripper", "move left down, open gripper"),
    ("move left up, open gripper", "move right down, close gripper"),
    ("move forward right, rotate clockwise", "move backward left, rotate counterclockwise"),
    ("rotate clockwise, close gripper", "rotate counterclockwise, open gripper"),
    ("rotate counterclockwise, close gripper", "rotate clockwise, open gripper"),
    # --- Pick / place wording (CoT narrative) ---
    ("pick up", "put down"),
    ("grasp", "release"),
    ("higher", "lower"),
    ("upward", "downward"),
)

_MOTION_PHRASE_PAIRS: tuple[tuple[str, str], ...] = _canonicalize_bidirectional_pairs(_MOTION_PHRASE_PAIR_RAW)


def get_reasoning_fn(fn_name: str):
    """
    Looks up a function defined in this file by its string name.
    """
    if not fn_name or fn_name.lower() in ["none", "null", ""]:
        return None

    if fn_name.startswith("ablate:"):
        component_list = [c.strip() for c in fn_name.split(":", maxsplit=1)[1].split(",") if c.strip()]
        return build_ablation_fn(component_list)

    if fn_name.startswith("gaussian_bbox_sigma:"):
        sigma = float(fn_name.split(":", maxsplit=1)[1])
        return make_gaussian_bbox_noise(sigma)

    if fn_name.startswith("gaussian_gripper_sigma:"):
        sigma = float(fn_name.split(":", maxsplit=1)[1])
        return make_gaussian_gripper_noise(sigma)

    if fn_name.startswith("word_dropout:"):
        rest = fn_name.split(":", maxsplit=1)[1]
        if ":" in rest:
            p_str, keys_str = rest.split(":", maxsplit=1)
            p = float(p_str)
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        else:
            p = float(rest)
            keys = None
        return make_word_dropout(p, keys)

    if fn_name.startswith("noise_text_subset:"):
        # Word dropout only on listed NL fields (see ``make_word_dropout`` / ``NL_SHORT_TO_TAG``).
        # Unlike ``noise_all_modalities:...``, this does **not** add Gaussian noise to
        # VISIBLE OBJECTS or GRIPPER POSITION.
        #
        # Format: noise_text_subset:<p>:<key1,key2,...>
        # Example: noise_text_subset:0.4:subtask,move_reasoning,move
        rest = fn_name.split(":", maxsplit=1)[1]
        return get_reasoning_fn(f"word_dropout:{rest}")

    if fn_name.startswith("noise_all_modalities:"):
        # Format:
        #   noise_all_modalities:<bbox_sigma>:<word_dropout_p>
        #   noise_all_modalities:<bbox_sigma>:<word_dropout_p>:<key1,key2,...>
        rest = fn_name.split(":", maxsplit=1)[1]
        if ":" not in rest:
            print(
                "Warning: noise_all_modalities expects 'noise_all_modalities:<bbox_sigma>:<word_dropout_p>'"
            )
            return None
        parts = rest.split(":", maxsplit=2)
        sigma_str, p_str = parts[0], parts[1]
        custom_keys = None
        if len(parts) == 3 and parts[2].strip():
            custom_keys = [k.strip() for k in parts[2].split(",") if k.strip()]
        sigma = float(sigma_str)
        p = float(p_str)
        return make_noise_all_modalities(
            bbox_sigma=sigma,
            word_dropout_p=p,
            word_dropout_keys=custom_keys
            if custom_keys is not None
            else ["plan", "subtask", "subtask_reasoning", "move", "move_reasoning"],
        )

    if fn_name.startswith("temporal_noise_all_modalities:"):
        # Format:
        # temporal_noise_all_modalities:<phase>:<bbox_sigma>:<word_dropout_p>
        # (avg steps path is hardcoded via DEFAULT_TEMPORAL_AVG_STEPS_JSON)
        # Optional boundaries:
        # temporal_noise_all_modalities:<phase>:<bbox_sigma>:<word_dropout_p>:<early_end>:<middle_end>:<late_end>
        #
        # Backward-compatible legacy format (explicit avg steps path) is still accepted:
        # temporal_noise_all_modalities:<phase>:<bbox_sigma>:<word_dropout_p>:<avg_steps_json>
        # temporal_noise_all_modalities:<phase>:<bbox_sigma>:<word_dropout_p>:<avg_steps_json>:<early_end>:<middle_end>:<late_end>
        parts = fn_name.split(":")
        if len(parts) not in {4, 7, 5, 8}:
            print(
                "Warning: temporal_noise_all_modalities expects "
                "'temporal_noise_all_modalities:<phase>:<bbox_sigma>:<word_dropout_p>' "
                "or with optional boundaries "
                "'temporal_noise_all_modalities:<phase>:<bbox_sigma>:<word_dropout_p>:<early_end>:<middle_end>:<late_end>'"
            )
            return None

        _, phase, sigma_str, p_str, *rest = parts
        bbox_sigma = float(sigma_str)
        word_dropout_p = float(p_str)

        avg_steps_path = str(DEFAULT_TEMPORAL_AVG_STEPS_JSON)
        bounds = []
        # New format: no explicit avg path.
        if len(parts) == 7:
            bounds = rest
        # Legacy format: explicit avg path and maybe bounds.
        elif len(parts) == 5:
            avg_steps_path = rest[0]
        elif len(parts) == 8:
            avg_steps_path = rest[0]
            bounds = rest[1:]

        if bounds:
            early_end, middle_end, late_end = [float(x) for x in bounds[:3]]
        else:
            early_end, middle_end, late_end = 0.4, 0.6, 1.0
        return TemporalNoiseAllModalitiesModifier(
            phase=phase,
            bbox_sigma=bbox_sigma,
            word_dropout_p=word_dropout_p,
            avg_steps_json_path=avg_steps_path,
            early_end=early_end,
            middle_end=middle_end,
            late_end=late_end,
            word_dropout_keys=["plan", "subtask", "subtask_reasoning", "move", "move_reasoning"],
        )

    if fn_name.startswith("temporal_noise_all_modalities_prob:"):
        # Format:
        # temporal_noise_all_modalities_prob:<phase>:<bbox_sigma>:<word_dropout_p>:<apply_p>
        # Optional boundaries:
        # temporal_noise_all_modalities_prob:<phase>:<bbox_sigma>:<word_dropout_p>:<apply_p>:<early_end>:<middle_end>:<late_end>
        parts = fn_name.split(":")
        if len(parts) not in {5, 8}:
            print(
                "Warning: temporal_noise_all_modalities_prob expects "
                "'temporal_noise_all_modalities_prob:<phase>:<bbox_sigma>:<word_dropout_p>:<apply_p>' "
                "or with optional boundaries "
                "'temporal_noise_all_modalities_prob:<phase>:<bbox_sigma>:<word_dropout_p>:<apply_p>:<early_end>:<middle_end>:<late_end>'"
            )
            return None

        _, phase, sigma_str, dropout_p_str, apply_p_str, *rest = parts
        bbox_sigma = float(sigma_str)
        word_dropout_p = float(dropout_p_str)
        apply_p = float(apply_p_str)
        if rest:
            early_end, middle_end, late_end = [float(x) for x in rest[:3]]
        else:
            early_end, middle_end, late_end = 0.4, 0.6, 1.0
        return TemporalNoiseAllModalitiesModifier(
            phase=phase,
            bbox_sigma=bbox_sigma,
            word_dropout_p=word_dropout_p,
            avg_steps_json_path=str(DEFAULT_TEMPORAL_AVG_STEPS_JSON),
            early_end=early_end,
            middle_end=middle_end,
            late_end=late_end,
            word_dropout_keys=["plan", "subtask", "subtask_reasoning", "move", "move_reasoning"],
            apply_prob=apply_p,
        )

    if fn_name == "sentence_shuffle" or fn_name == "sentence_shuffle_subtask_move_reasoning":
        return make_sentence_shuffle(["subtask_reasoning", "move_reasoning"])

    if fn_name.startswith("sentence_shuffle:"):
        keys = [k.strip() for k in fn_name.split(":", maxsplit=1)[1].split(",") if k.strip()]
        return make_sentence_shuffle(keys)

    if fn_name == "invert_motion_phrases":
        return make_invert_motion_phrases(list(NL_SHORT_TO_TAG.keys()))

    if fn_name.startswith("invert_motion_phrases_prob:"):
        # Format: invert_motion_phrases_prob:<p>[:key1,key2,...]
        # Example: invert_motion_phrases_prob:0.3:move
        rest = fn_name.split(":", maxsplit=1)[1]
        if ":" in rest:
            p_str, keys_str = rest.split(":", maxsplit=1)
            p = float(p_str)
            keys = [k.strip() for k in keys_str.split(",") if k.strip()]
        else:
            p = float(rest)
            keys = list(NL_SHORT_TO_TAG.keys())
        return make_invert_motion_phrases_prob(p, keys)

    if fn_name.startswith("invert_motion_phrases:"):
        keys = [k.strip() for k in fn_name.split(":", maxsplit=1)[1].split(",") if k.strip()]
        return make_invert_motion_phrases(keys)

    # Metadata-only name: return None so OpenVLA stays single-pass (a no-op lambda would still trigger
    # two-stage generate). Pair with --reasoning_trace_jsonl in run_libero_eval.py for NDJSON logging.
    if fn_name == "knowledge_index_trace":
        return None

    if fn_name in {"shuffle_subtask", "shuffle_subtask_words"}:
        return shuffle_subtask_words

    # Get the dictionary of everything defined in THIS file
    current_file_namespace = globals()
    
    if fn_name in current_file_namespace:
        return current_file_namespace[fn_name]
    else:
        print(f"Warning: Function '{fn_name}' not found in reasoning_manipulation.py")
        return None

def no_reasoning(reasoning):
    return ""


def reasoning_dropout_50(reasoning):
    """
    ECoT-Lite: Randomly drop reasoning with 50% probability during training.
    At inference, use reasoning_modifier_fn_str="None" so no reasoning is generated.
    This matches the paper's best LIBERO-90 results (~89%).
    """
    return "" if random.random() < 0.5 else reasoning

def swap_x_y(reasoning):
    # Replacement function for the VISIBLE OBJECTS dictionary
    def swap_x_y_objects(match):
        dict_str = match.group(1)
        objects_dict = ast.literal_eval(dict_str)
        
        for key, coords in objects_dict.items():
            # Swap [x, y] to [y, x]
            objects_dict[key] = [[c[1], c[0]] for c in coords]
            
        # Reconstruct the matched portion
        return f"VISIBLE OBJECTS:@{str(objects_dict)}@"

    # Replacement function for the GRIPPER POSITION list
    def swap_x_y_gripper(match):
        list_str = match.group(1)
        gripper_list = ast.literal_eval(list_str)
        
        # Swap [x, y] to [y, x]
        if len(gripper_list) == 2:
            gripper_list = [gripper_list[1], gripper_list[0]]
            
        # Reconstruct the matched portion
        return f"GRIPPER POSITION:@{str(gripper_list)}"

    # Apply the replacements
    # Match everything between 'VISIBLE OBJECTS:@' and the next '@'
    updated_string = re.sub(r'VISIBLE OBJECTS:@(\{.*?\})@', swap_x_y_objects, reasoning)
    
    # Match the coordinate list after 'GRIPPER POSITION:@'
    updated_string = re.sub(r'GRIPPER POSITION:@(\[.*?\])', swap_x_y_gripper, updated_string)

    return updated_string

def _move_bboxes(reasoning, offset):
    def adjust_objects(match):
        dict_str = match.group(1)
        objects_dict = ast.literal_eval(dict_str)
        
        for key, coords in objects_dict.items():
            # Add the offset to x and y for both points [[x1, y1], [x2, y2]]
            objects_dict[key] = [[c[0] + offset, c[1] + offset] for c in coords]
            
        # Reconstruct the matched portion
        return f"VISIBLE OBJECTS:@{str(objects_dict)}@"

    def adjust_gripper(match):
        list_str = match.group(1)
        gripper_list = ast.literal_eval(list_str)
        
        # Add the offset to x and y
        if len(gripper_list) == 2:
            gripper_list = [gripper_list[0] + offset, gripper_list[1] + offset]
            
        # Reconstruct the matched portion
        return f"GRIPPER POSITION:@{str(gripper_list)}"

    updated_string = re.sub(r'VISIBLE OBJECTS:@(\{.*?\})@', adjust_objects, reasoning)
    updated_string = re.sub(r'GRIPPER POSITION:@(\[.*?\])', adjust_gripper, updated_string)

    return updated_string

def move_bboxes_15(reasoning):
    return _move_bboxes(reasoning, 15)

def _cut_out_of_reasoning(reasoning, reasoning_tag):
    """
    Removes a specific tag and its contents from the reasoning string.
    """
    pattern = r"@?" + re.escape(reasoning_tag) + r":@?.*?(?=@|\sACTION:|$)"
    
    cleaned_text = re.sub(pattern, "", reasoning)
    return cleaned_text.lstrip('@')

def cut_out_gripper(reasoning):
    return _cut_out_of_reasoning(reasoning, "GRIPPER POSITION")

def cut_out_move(reasoning):
    return _cut_out_of_reasoning(reasoning, "MOVE")

def cut_out_move_reasoning(reasoning):
    return _cut_out_of_reasoning(reasoning, "MOVE REASONING")

def cut_out_subtask(reasoning):
    return _cut_out_of_reasoning(reasoning, "SUBTASK")

def cut_out_subtask_reasoning(reasoning):
    return _cut_out_of_reasoning(reasoning, "SUBTASK REASONING")

def cut_out_visible_objects(reasoning):
    return _cut_out_of_reasoning(reasoning, "VISIBLE OBJECTS")

def cut_out_plan(reasoning):
    return _cut_out_of_reasoning(reasoning, "PLAN")

def gauß_50(reasoning):
    return _gauß_on_bboxes(reasoning, 50, "/home/hk-project-p0024638/uvrfq/shifts_sigma50.jsonl")

def make_gaussian_bbox_noise(sigma, folder_path=None, max_val=224):
    sigma = float(sigma)

    def _apply(reasoning):
        return _gauß_on_bboxes(reasoning, sigma=sigma, folder_path=folder_path, max_val=max_val)

    return _apply


def _gauß_on_bboxes(reasoning, sigma, folder_path=None, max_val=224):
    global _GAUSS_BBOX_NO_MATCH_WARNED
    actual_dx = []
    actual_dy = []
    # Helper function to apply Gaussian shift and track it
    def apply_gauss_to_point(x, y):
        intended_dx = int(np.round(np.random.normal(0, sigma)))
        intended_dy = int(np.round(np.random.normal(0, sigma)))
        
        new_x = max(0, min(x + intended_dx, max_val))
        new_y = max(0, min(y + intended_dy, max_val))

        actual_dx.append(new_x - x)
        actual_dy.append(new_y - y)
        
        return new_x, new_y

    # Replacement function for the VISIBLE OBJECTS dictionary.
    # Group 1 = prefix ("VISIBLE OBJECTS:" with optional whitespace/@),
    # Group 2 = dict payload, Group 3 = optional trailing '@'.
    def shift_objects(match):
        prefix = match.group(1)
        dict_str = match.group(2)
        suffix = match.group(3)
        objects_dict = ast.literal_eval(dict_str)
        
        for key, coords in objects_dict.items():
            new_coords = []
            for c in coords:
                # Calculate the Gaussian shift for every individual point
                new_x, new_y = apply_gauss_to_point(c[0], c[1])
                new_coords.append([new_x, new_y])
            objects_dict[key] = new_coords

        return f"{prefix}{str(objects_dict)}{suffix}"

    # Replacement function for the GRIPPER POSITION list.
    # Group 1 = prefix ("GRIPPER POSITION:" with optional whitespace/@),
    # Group 2 = list payload.
    def shift_gripper(match):
        prefix = match.group(1)
        list_str = match.group(2)
        gripper_list = ast.literal_eval(list_str)
        
        if len(gripper_list) == 2:
            new_x, new_y = apply_gauss_to_point(gripper_list[0], gripper_list[1])
            gripper_list = [new_x, new_y]

        return f"{prefix}{str(gripper_list)}"

    # Robust matching for slight formatting drift:
    # - optional whitespace around colon and '@'
    # - optional trailing '@' after dict
    # - stops before next CoT tag/end to avoid over-capture.
    visible_pattern = r"(VISIBLE OBJECTS:\s*@?\s*)(\{.*?\})(\s*@?)(?=\s*@?[A-Z ]+?:@?|$)"
    gripper_pattern = r"(GRIPPER POSITION:\s*@?\s*)(\[.*?\])"

    updated_string, n_visible = re.subn(visible_pattern, shift_objects, reasoning, flags=re.DOTALL)
    updated_string, _n_gripper = re.subn(gripper_pattern, shift_gripper, updated_string, flags=re.DOTALL)

    # Sanity check: if a reasoning trace mentions visible objects but no block matched,
    # warn once so failed corruption cannot go unnoticed.
    if n_visible == 0 and not _GAUSS_BBOX_NO_MATCH_WARNED and "VISIBLE OBJECTS" in str(reasoning):
        print(
            f"[gaussian_bbox_noise] Warning: no VISIBLE OBJECTS block matched (sigma={sigma}). "
            "Reasoning format may have drifted; bbox noise may be skipped."
        )
        _GAUSS_BBOX_NO_MATCH_WARNED = True

    if folder_path and actual_dx and actual_dy:
        update_running_sigma(folder_path, sigma, actual_dx, actual_dy)

    return updated_string


def make_gaussian_gripper_noise(sigma, folder_path=None, max_val=224):
    sigma = float(sigma)

    def _apply(reasoning):
        return _gauß_on_gripper_only(reasoning, sigma=sigma, folder_path=folder_path, max_val=max_val)

    return _apply


def _gauß_on_gripper_only(reasoning, sigma, folder_path=None, max_val=224):
    actual_dx = []
    actual_dy = []

    def apply_gauss_to_point(x, y):
        intended_dx = int(np.round(np.random.normal(0, sigma)))
        intended_dy = int(np.round(np.random.normal(0, sigma)))
        new_x = max(0, min(x + intended_dx, max_val))
        new_y = max(0, min(y + intended_dy, max_val))
        actual_dx.append(new_x - x)
        actual_dy.append(new_y - y)
        return new_x, new_y

    def shift_gripper(match):
        list_str = match.group(1)
        gripper_list = ast.literal_eval(list_str)
        if len(gripper_list) == 2:
            new_x, new_y = apply_gauss_to_point(gripper_list[0], gripper_list[1])
            gripper_list = [new_x, new_y]
        return f"GRIPPER POSITION:@{str(gripper_list)}"

    updated_string = re.sub(r"GRIPPER POSITION:@(\[.*?\])", shift_gripper, reasoning)
    if folder_path and actual_dx and actual_dy:
        update_running_sigma(folder_path, sigma, actual_dx, actual_dy)
    return updated_string


def _map_tag_body(reasoning: str, tag_colon: str, fn):
    esc = re.escape(tag_colon)
    pattern = rf"({esc}@)(.*?)(?=@[A-Z ]+?:@?|$)"

    def repl(m):
        return m.group(1) + fn(m.group(2))

    return re.sub(pattern, repl, reasoning, flags=re.DOTALL)


def _word_dropout_words(text: str, p: float) -> str:
    words = text.split()
    if not words:
        return text
    kept = [w for w in words if random.random() >= p]
    if not kept:
        return random.choice(words)
    return " ".join(kept)


def _shuffle_sentences(text: str) -> str:
    raw = text.strip()
    if not raw:
        return text
    pieces = re.split(r"(?<=[.!?])\s+", raw)
    pieces = [s for s in pieces if s.strip()]
    if len(pieces) <= 1:
        return text
    random.shuffle(pieces)
    return " ".join(pieces)


def _invert_motion_phrases_in_text(text: str) -> str:
    swap = {}
    for a, b in _MOTION_PHRASE_PAIRS:
        swap[a.lower()] = b
        swap[b.lower()] = a
    keys = sorted(swap.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(rf"\b{re.escape(k)}\b" for k in keys), flags=re.IGNORECASE)

    def repl(m):
        frag = m.group(0)
        return swap.get(frag.lower(), frag)

    return pattern.sub(repl, text)


def _corrupt_plan_dict(reasoning: str, value_fn) -> str:
    """
    Corrupt values in the PLAN dict while being tolerant to minor formatting variants.

    Supported examples:
      PLAN:@{...}@
      PLAN:{...}
      PLAN: @{...}
    and with/without a trailing '@' before the next tag.
    """
    def repl(m):
        prefix = m.group(1)
        dict_str = m.group(2)
        suffix = m.group(3)
        d = ast.literal_eval(dict_str)
        newd = {}
        for k, v in d.items():
            if isinstance(v, str):
                newd[k] = value_fn(v)
            else:
                newd[k] = v
        return f"{prefix}{str(newd)}{suffix}"

    return re.sub(
        r"(PLAN:\s*@?)(\{.*?\})(@?)(?=\s*@?[A-Z ]+?:@?|$)",
        repl,
        reasoning,
        flags=re.DOTALL,
    )


def plan_step_shuffle(reasoning: str) -> str:
    """
    Permute PLAN step descriptions across step indices: keys stay in sorted order but values are
    shuffled (breaks temporal / causal ordering of the dict). No-op if fewer than two steps.
    Complements sentence_shuffle:plan, which only shuffles sentences inside each step string.
    """

    def repl(m):
        dict_str = m.group(1)
        try:
            d = ast.literal_eval(dict_str)
        except (ValueError, SyntaxError):
            return m.group(0)
        if not isinstance(d, dict) or len(d) < 2:
            return m.group(0)

        def key_sort_key(k):
            sk = str(k)
            if sk.isdigit():
                return (0, int(sk))
            return (1, sk)

        keys = sorted(d.keys(), key=key_sort_key)
        vals = [d[k] for k in keys]
        random.shuffle(vals)
        newd = {keys[i]: vals[i] for i in range(len(keys))}
        return f"PLAN:@{str(newd)}@"

    return re.sub(r"PLAN:@(\{.*?\})@", repl, reasoning, flags=re.DOTALL)


def make_word_dropout(p: float, tag_keys=None):
    if tag_keys is None:
        keys_set = set(NL_SHORT_TO_TAG.keys())
    else:
        keys_set = set(tag_keys)
        unknown = keys_set - set(NL_SHORT_TO_TAG.keys())
        if unknown:
            print(f"Warning: word_dropout unknown tag keys (ignored): {unknown}")
        keys_set &= set(NL_SHORT_TO_TAG.keys())

    def apply(reasoning):
        r = reasoning
        if "plan" in keys_set:
            r = _corrupt_plan_dict(r, lambda s: _word_dropout_words(s, p))
        for short, tag in NL_SHORT_TO_TAG.items():
            if short == "plan" or short not in keys_set:
                continue
            r = _map_tag_body(r, tag, lambda b, pp=p: _word_dropout_words(b, pp))
        return r

    return apply


def make_sentence_shuffle(tag_keys: list):
    keys_set = set(tag_keys)
    unknown = keys_set - set(NL_SHORT_TO_TAG.keys())
    if unknown:
        print(f"Warning: sentence_shuffle unknown tag keys (ignored): {unknown}")
    keys_set &= set(NL_SHORT_TO_TAG.keys())

    def apply(reasoning):
        r = reasoning
        if "plan" in keys_set:
            r = _corrupt_plan_dict(r, _shuffle_sentences)
        for short, tag in NL_SHORT_TO_TAG.items():
            if short == "plan" or short not in keys_set:
                continue
            r = _map_tag_body(r, tag, _shuffle_sentences)
        return r

    return apply


def make_invert_motion_phrases(tag_keys: list):
    """Apply phrase inversion only to listed short keys (see NL_SHORT_TO_TAG)."""
    keys_set = set(tag_keys)
    unknown = keys_set - set(NL_SHORT_TO_TAG.keys())
    if unknown:
        print(f"Warning: invert_motion_phrases unknown tag keys (ignored): {unknown}")
    keys_set &= set(NL_SHORT_TO_TAG.keys())

    def apply(reasoning):
        r = reasoning
        if "plan" in keys_set:
            r = _corrupt_plan_dict(r, _invert_motion_phrases_in_text)
        for short, tag in NL_SHORT_TO_TAG.items():
            if short == "plan" or short not in keys_set:
                continue
            r = _map_tag_body(r, tag, _invert_motion_phrases_in_text)
        return r

    return apply


def make_invert_motion_phrases_prob(p: float, tag_keys: list):
    """
    Probabilistic variant of motion inversion.
    For each targeted text field, invert motion phrases with probability ``p``.
    """
    p = float(p)
    if p <= 0.0:
        return lambda reasoning: reasoning
    if p >= 1.0:
        return make_invert_motion_phrases(tag_keys)

    keys_set = set(tag_keys)
    unknown = keys_set - set(NL_SHORT_TO_TAG.keys())
    if unknown:
        print(f"Warning: invert_motion_phrases_prob unknown tag keys (ignored): {unknown}")
    keys_set &= set(NL_SHORT_TO_TAG.keys())

    def _maybe_invert(text: str) -> str:
        if random.random() < p:
            return _invert_motion_phrases_in_text(text)
        return text

    def apply(reasoning):
        r = reasoning
        if "plan" in keys_set:
            r = _corrupt_plan_dict(r, _maybe_invert)
        for short, tag in NL_SHORT_TO_TAG.items():
            if short == "plan" or short not in keys_set:
                continue
            r = _map_tag_body(r, tag, _maybe_invert)
        return r

    return apply


def invert_motion_phrases(reasoning):
    """Backward-compatible: invert on all NL_SHORT_TO_TAG fields."""
    return make_invert_motion_phrases(list(NL_SHORT_TO_TAG.keys()))(reasoning)


def noise_all_modalities_dropout20_sigma20(reasoning):
    """
    Composite training-time perturbation used for all-modality noise training:
    - word dropout p=0.2 on plan/subtask/subtask_reasoning/move/move_reasoning
    - Gaussian sigma=20 on visible objects and gripper position

    Note: make_gaussian_bbox_noise(...) already perturbs both VISIBLE OBJECTS and
    GRIPPER POSITION tags, so a separate gripper-only pass is not needed.
    """
    r = reasoning
    r = make_word_dropout(
        0.2, ["plan", "subtask", "subtask_reasoning", "move", "move_reasoning"]
    )(r)
    r = make_gaussian_bbox_noise(20)(r)
    return r


def make_noise_all_modalities(
    bbox_sigma: float,
    word_dropout_p: float,
    word_dropout_keys=None,
):
    """
    Build a composite perturbation:
    1) Word dropout on selected language fields
    2) Gaussian bbox noise (also perturbs gripper in this codebase)
    """
    if word_dropout_keys is None:
        word_dropout_keys = ["plan", "subtask", "subtask_reasoning", "move", "move_reasoning"]

    bbox_fn = make_gaussian_bbox_noise(float(bbox_sigma))
    dropout_fn = make_word_dropout(float(word_dropout_p), word_dropout_keys)

    def apply(reasoning):
        r = reasoning
        r = dropout_fn(r)
        r = bbox_fn(r)
        return r

    return apply


class TemporalNoiseAllModalitiesModifier:
    """
    Applies all-modalities noise only in one temporal phase of an episode.
    Temporal progress is estimated as env_step / avg_steps_for_task.
    """

    def __init__(
        self,
        phase: str,
        bbox_sigma: float,
        word_dropout_p: float,
        avg_steps_json_path: str,
        early_end: float = 0.4,
        middle_end: float = 0.6,
        late_end: float = 1.0,
        word_dropout_keys=None,
        apply_prob: float = 1.0,
    ):
        normalized_phase = str(phase).strip().lower()
        if normalized_phase not in {"early", "middle", "late"}:
            raise ValueError(
                f"Invalid phase '{phase}'. Expected one of: early, middle, late."
            )
        self.phase = normalized_phase
        self.early_end = float(early_end)
        self.middle_end = float(middle_end)
        self.late_end = float(late_end)
        if not (0.0 < self.early_end <= self.middle_end <= self.late_end):
            raise ValueError(
                "Temporal boundaries must satisfy 0.0 < early_end <= middle_end <= late_end."
            )
        self.apply_prob = float(apply_prob)
        if not (0.0 <= self.apply_prob <= 1.0):
            raise ValueError("apply_prob must be in [0.0, 1.0].")

        self.noise_fn = make_noise_all_modalities(
            bbox_sigma=float(bbox_sigma),
            word_dropout_p=float(word_dropout_p),
            word_dropout_keys=word_dropout_keys,
        )

        self.avg_steps_by_task, self.default_avg_steps = self._load_avg_steps(avg_steps_json_path)
        self.current_task_id = None
        self.current_env_step = 0

    @staticmethod
    def _load_avg_steps(path: str):
        # Prefer the explicit path, but gracefully fall back to known shared export locations.
        raw = Path(path).expanduser()
        candidates = [raw]
        fallback = DEFAULT_TEMPORAL_AVG_STEPS_JSON_FALLBACK
        if raw != fallback:
            candidates.append(fallback)

        payload = None
        resolved = None
        for candidate in candidates:
            candidate_resolved = candidate.resolve()
            if not candidate_resolved.exists():
                continue
            with candidate_resolved.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            resolved = candidate_resolved
            break

        if payload is None or resolved is None:
            checked = ", ".join(str(c.resolve()) for c in candidates)
            raise FileNotFoundError(
                f"avg-steps json not found. Checked: {checked}"
            )

        avg_steps_by_task = {}
        values = []

        # Preferred schema: {"per_task": {"6": {"avg_steps_executed": ...}, ...}}
        per_task = payload.get("per_task") if isinstance(payload, dict) else None
        if isinstance(per_task, dict):
            for key, row in per_task.items():
                try:
                    task_id = int(key)
                except Exception:
                    continue
                if isinstance(row, dict):
                    val = row.get("avg_steps_executed")
                else:
                    val = row
                try:
                    avg_steps = float(val)
                except Exception:
                    continue
                if avg_steps > 0:
                    avg_steps_by_task[task_id] = avg_steps
                    values.append(avg_steps)

        # Fallback schema: {"task_avg_steps": {"6": 123.4, ...}}
        task_avg_steps = payload.get("task_avg_steps") if isinstance(payload, dict) else None
        if isinstance(task_avg_steps, dict):
            for key, val in task_avg_steps.items():
                try:
                    task_id = int(key)
                    avg_steps = float(val)
                except Exception:
                    continue
                if avg_steps > 0:
                    avg_steps_by_task[task_id] = avg_steps
                    values.append(avg_steps)

        if not avg_steps_by_task:
            raise ValueError(
                f"No per-task average steps found in {resolved}. "
                "Expected keys like per_task.<task_id>.avg_steps_executed."
            )

        default_avg_steps = float(np.mean(values)) if values else 1.0
        return avg_steps_by_task, max(default_avg_steps, 1.0)

    def set_context(self, task_id: int, env_step: int):
        self.current_task_id = int(task_id)
        self.current_env_step = max(0, int(env_step))

    def _current_phase(self) -> str:
        avg_steps = self.avg_steps_by_task.get(self.current_task_id, self.default_avg_steps)
        avg_steps = max(float(avg_steps), 1.0)
        progress = float(self.current_env_step) / avg_steps
        if progress <= self.early_end:
            return "early"
        if progress <= self.middle_end:
            return "middle"
        return "late"

    def __call__(self, reasoning):
        if self._current_phase() != self.phase:
            return reasoning
        if self.apply_prob < 1.0 and random.random() >= self.apply_prob:
            return reasoning
        return self.noise_fn(reasoning)


def _shuffle_words(text: str) -> str:
    words = text.split()
    if len(words) <= 1:
        return text
    random.shuffle(words)
    return " ".join(words)


def shuffle_subtask_words(reasoning):
    def _replace(match):
        prefix = match.group(1)
        subtask_text = match.group(2).strip()
        return f"{prefix}{_shuffle_words(subtask_text)}"

    return re.sub(r"(SUBTASK:@)(.*?)(?=@[A-Z ]+?:@?|$)", _replace, reasoning)


def build_ablation_fn(components):
    component_to_fn = {
        "plan": cut_out_plan,
        "visible_objects": cut_out_visible_objects,
        "subtask_reasoning": cut_out_subtask_reasoning,
        "subtask": cut_out_subtask,
        "move_reasoning": cut_out_move_reasoning,
        "move": cut_out_move,
        "gripper": cut_out_gripper,
    }

    selected_fns = []
    for component in components:
        key = component.strip().lower()
        if key in component_to_fn:
            selected_fns.append(component_to_fn[key])

    def _ablate(reasoning):
        updated = reasoning
        for fn in selected_fns:
            updated = fn(updated)
        return updated

    return _ablate

def initialize_shift_log(filepath, target_sigma):
    """
    Creates the directory and initializes a .jsonl file with the target sigma.
    """
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Open in 'w' mode to create a fresh file (overwrites existing)
    with open(filepath, 'w') as f:
        metadata = {"target_sigma": target_sigma, "type": "metadata"}
        f.write(json.dumps(metadata) + '\n')
        
    print(f"Initialized log file at: {filepath}")

def calculate_real_sigma_from_log(filepath):
    """
    Reads the .jsonl file line by line and calculates the true mean and std dev.
    """
    all_dx = []
    all_dy = []
    target_sigma = None

    with open(filepath, 'r') as f:
        for line_number, line in enumerate(f):
            data = json.loads(line.strip())
            
            # The first line contains our metadata
            if line_number == 0 and data.get("type") == "metadata":
                target_sigma = data.get("target_sigma")
                continue
            
            # Extend our flattened lists with the values from this step
            all_dx.extend(data.get("dx", []))
            all_dy.extend(data.get("dy", []))

    # Convert to numpy arrays for fast math
    dx_array = np.array(all_dx)
    dy_array = np.array(all_dy)

    # Calculate statistics
    real_mean_x = np.mean(dx_array) if len(dx_array) > 0 else 0
    real_sigma_x = np.std(dx_array, ddof=1) if len(dx_array) > 1 else 0

    real_mean_y = np.mean(dy_array) if len(dy_array) > 0 else 0
    real_sigma_y = np.std(dy_array, ddof=1) if len(dy_array) > 1 else 0

    print(f"--- Results for {filepath} ---")
    print(f"Target Sigma: {target_sigma}")
    print(f"Total points shifted: {len(dx_array)}")
    print(f"Real X -> Mean: {real_mean_x:.2f}, Sigma: {real_sigma_x:.2f}")
    print(f"Real Y -> Mean: {real_mean_y:.2f}, Sigma: {real_sigma_y:.2f}")
    
    return real_sigma_x, real_sigma_y

def update_running_sigma(folder_path, target_sigma, actual_dx, actual_dy):
    os.makedirs(folder_path, exist_ok=True)
    filepath = os.path.join(folder_path, f"running_sigma_{target_sigma}.json")
    
    # 1. Load existing state or initialize a new one
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            state = json.load(f)
    else:
        state = {
            "target_sigma": target_sigma,
            "n": 0,
            "sum_dx": 0.0,
            "sum_dy": 0.0,
            "sum_sq_dx": 0.0,
            "sum_sq_dy": 0.0,
            "real_sigma_x": 0.0,
            "real_sigma_y": 0.0
        }
        
    # 2. If no shifts occurred in this call, just return current sigmas
    if not actual_dx or not actual_dy:
        return state["real_sigma_x"], state["real_sigma_y"]
        
    # 3. Update running totals
    n_new = len(actual_dx)
    dx_arr = np.array(actual_dx)
    dy_arr = np.array(actual_dy)
    
    state["n"] += n_new
    state["sum_dx"] += float(np.sum(dx_arr))
    state["sum_dy"] += float(np.sum(dy_arr))
    state["sum_sq_dx"] += float(np.sum(dx_arr**2))
    state["sum_sq_dy"] += float(np.sum(dy_arr**2))
    
    # 4. Calculate the new running standard deviation
    n = state["n"]
    if n > 1:
        mean_x = state["sum_dx"] / n
        mean_y = state["sum_dy"] / n
        
        # Population variance
        var_x = (state["sum_sq_dx"] / n) - (mean_x ** 2)
        var_y = (state["sum_sq_dy"] / n) - (mean_y ** 2)
        
        # Apply Bessel's correction (n / (n-1)) for Sample variance
        var_x = var_x * (n / (n - 1))
        var_y = var_y * (n / (n - 1))
        
        # max(0, var) prevents crashing from microscopic negative floats due to precision limits
        state["real_sigma_x"] = float(np.sqrt(max(0, var_x)))
        state["real_sigma_y"] = float(np.sqrt(max(0, var_y)))
        
    # 5. Save the updated state safely
    with open(filepath, 'w') as f:
        json.dump(state, f, indent=4)
        
    return state["real_sigma_x"], state["real_sigma_y"]

# TODO - Eigentlich sollte jetzt
if __name__ == "__main__":
    log_filepath = "./shifts_sigma50.jsonl"
    target_sigma = 50
    initialize_shift_log(log_filepath, target_sigma)
    reasoning_output = "VISIBLE OBJECTS:@{'apple': [[0, 5], [150, 150]]}@ GRIPPER POSITION:@[100, 0]"
    altered_reasoning = _gauß_on_bboxes(reasoning_output, target_sigma, log_filepath)
    print(reasoning_output)
    print(altered_reasoning)
    print(calculate_real_sigma_from_log(log_filepath))