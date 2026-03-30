import ast
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from prismatic.util.cot_utils import CotTag, split_reasoning

# (json_key, cot_tag_value) for NL sections used in ROUGE-L (excludes VISIBLE OBJECTS, GRIPPER, ACTION).
TEXT_ROUGE_FIELD_SPECS: Tuple[Tuple[str, str], ...] = (
    ("plan", CotTag.PLAN.value),
    ("subtask_reasoning", CotTag.SUBTASK_REASONING.value),
    ("subtask", CotTag.SUBTASK.value),
    ("move_reasoning", CotTag.MOVE_REASONING.value),
    ("move", CotTag.MOVE.value),
)


def normalize_entity_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Drop trailing instance index ("... 1") for easier matching.
    cleaned = re.sub(r"\s+\d+$", "", cleaned)
    return cleaned


def tokenize_text(text: str) -> list[str]:
    if text is None:
        return []
    return re.findall(r"[a-z0-9]+", str(text).lower())


def token_jaccard_similarity(text_a: str, text_b: str) -> Optional[float]:
    set_a = set(tokenize_text(text_a))
    set_b = set(tokenize_text(text_b))
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if len(union) == 0:
        return None
    return float(len(set_a & set_b)) / float(len(union))


def lcs_length(a: List[str], b: List[str]) -> int:
    """Length of longest common subsequence over token sequences."""
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
    """
    Token-level ROUGE-L F1 using the same tokenization as tokenize_text (lowercase alphanumeric tokens).
    Returns None only if both sides tokenize empty (undefined); if one side empty and the other not, returns 0.0.
    """
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
    if prec + rec == 0:
        return 0.0
    return float(2.0 * prec * rec / (prec + rec))


def _parse_literal(raw: str):
    try:
        return ast.literal_eval(raw.strip())
    except Exception:
        return None


def parse_reasoning_sections(reasoning: str) -> dict:
    if reasoning is None:
        return {}
    parsed = split_reasoning(reasoning.replace("@", " "))
    return {k: str(v).strip() for k, v in parsed.items() if isinstance(k, str)}


def parse_bboxes_from_reasoning(reasoning: str) -> Dict[str, list]:
    sections = parse_reasoning_sections(reasoning)
    raw = sections.get(CotTag.VISIBLE_OBJECTS.value)
    if raw is None:
        return {}
    bboxes = _parse_literal(raw)
    return bboxes if isinstance(bboxes, dict) else {}


def parse_gripper_from_reasoning(reasoning: str) -> Optional[list[int]]:
    sections = parse_reasoning_sections(reasoning)
    raw = sections.get(CotTag.GRIPPER_POSITION.value)
    if raw is None:
        return None
    value = _parse_literal(raw)
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return [int(round(float(value[0]))), int(round(float(value[1])))]
    return None


def parse_text_field_from_reasoning(reasoning: str, cot_tag: str) -> Optional[str]:
    sections = parse_reasoning_sections(reasoning)
    text = sections.get(cot_tag)
    return text.strip() if isinstance(text, str) and text.strip() != "" else None


def concat_nl_reasoning_sections(reasoning: Optional[str]) -> str:
    """Single-space join of NL CoT fields in fixed order (for whole-trace ROUGE)."""
    if not reasoning:
        return ""
    parts = []
    for _, tag in TEXT_ROUGE_FIELD_SPECS:
        t = parse_text_field_from_reasoning(reasoning, tag)
        if t:
            parts.append(t)
    return " ".join(parts)


def compute_iou(bbox_a, bbox_b) -> Optional[float]:
    if bbox_a is None or bbox_b is None:
        return None
    try:
        ax1, ay1 = bbox_a[0]
        ax2, ay2 = bbox_a[1]
        bx1, by1 = bbox_b[0]
        bx2, by2 = bbox_b[1]
    except Exception:
        return None

    ax1, ax2 = sorted([float(ax1), float(ax2)])
    ay1, ay2 = sorted([float(ay1), float(ay2)])
    bx1, bx2 = sorted([float(bx1), float(bx2)])
    by1, by2 = sorted([float(by1), float(by2)])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return None
    return float(inter_area / union)


def _build_gt_name_buckets(gt_bboxes: Dict[str, list]) -> Dict[str, list[str]]:
    buckets = {}
    for gt_name in gt_bboxes.keys():
        norm = normalize_entity_name(gt_name)
        buckets.setdefault(norm, []).append(gt_name)
    return buckets


def compute_bbox_iou_stats(pred_bboxes: Dict[str, list], gt_bboxes: Dict[str, list]) -> dict:
    if not pred_bboxes or not gt_bboxes:
        return {"count": 0, "mean": None, "std": None, "per_object_iou": {}}

    gt_name_buckets = _build_gt_name_buckets(gt_bboxes)
    used_gt = set()
    per_object_iou = {}

    for pred_name, pred_bbox in pred_bboxes.items():
        norm_pred = normalize_entity_name(pred_name)
        candidates = [c for c in gt_name_buckets.get(norm_pred, []) if c not in used_gt]
        if len(candidates) == 0:
            continue
        best_name = None
        best_iou = -1.0
        for gt_name in candidates:
            iou = compute_iou(pred_bbox, gt_bboxes[gt_name])
            if iou is None:
                continue
            if iou > best_iou:
                best_iou = iou
                best_name = gt_name
        if best_name is not None:
            used_gt.add(best_name)
            per_object_iou[pred_name] = float(best_iou)

    values = list(per_object_iou.values())
    if len(values) == 0:
        return {"count": 0, "mean": None, "std": None, "per_object_iou": {}}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "per_object_iou": per_object_iou,
    }


def compute_gripper_distance(pred_gripper_xy, gt_gripper_xy) -> Optional[float]:
    if pred_gripper_xy is None or gt_gripper_xy is None:
        return None
    try:
        pred = np.asarray(pred_gripper_xy[:2], dtype=np.float32)
        gt = np.asarray(gt_gripper_xy[:2], dtype=np.float32)
    except Exception:
        return None
    return float(np.linalg.norm(pred - gt))


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

    def add_many(self, values: Iterable[Optional[float]]) -> None:
        for value in values:
            self.add(value)

    def to_dict(self) -> dict:
        if self.count == 0:
            return {"count": 0, "mean": None, "std": None}
        mean = self.total / self.count
        if self.count > 1:
            variance = (self.total_sq - self.count * (mean**2)) / (self.count - 1)
            std = float(np.sqrt(max(0.0, variance)))
        else:
            std = 0.0
        return {"count": int(self.count), "mean": float(mean), "std": float(std)}


def create_text_rouge_running_stats() -> Dict[str, RunningStats]:
    """RunningStats buckets: one per TEXT_ROUGE_FIELD_SPECS key plus 'whole'."""
    out = {key: RunningStats() for key, _ in TEXT_ROUGE_FIELD_SPECS}
    out["whole"] = RunningStats()
    return out


def add_text_rouge_l_samples(
    stats_by_key: Dict[str, RunningStats],
    pred_reasoning: Optional[str],
    reference_reasoning_or_task: Optional[str],
    use_task_as_single_reference: bool,
) -> None:
    """
    use_task_as_single_reference: True -> every field and whole string compared to reference_reasoning_or_task
    (the task instruction). False -> per-field pred vs same field in reference string; whole vs concat(reference).
    """
    if not pred_reasoning:
        return
    ref_ctx = reference_reasoning_or_task or ""

    for key, tag in TEXT_ROUGE_FIELD_SPECS:
        pred_t = parse_text_field_from_reasoning(pred_reasoning, tag)
        if use_task_as_single_reference:
            ref_t = ref_ctx
        else:
            ref_t = parse_text_field_from_reasoning(ref_ctx, tag) if ref_ctx else None
        if pred_t is None:
            continue
        if ref_t is None or (isinstance(ref_t, str) and ref_t.strip() == ""):
            continue
        score = rouge_l_f1(pred_t, ref_t)
        if score is not None:
            stats_by_key[key].add(score)

    pred_whole = concat_nl_reasoning_sections(pred_reasoning)
    ref_whole = ref_ctx if use_task_as_single_reference else concat_nl_reasoning_sections(ref_ctx)
    if pred_whole.strip() == "" or not ref_whole.strip():
        return
    w = rouge_l_f1(pred_whole, ref_whole)
    if w is not None:
        stats_by_key["whole"].add(w)


def text_rouge_stats_to_payload(stats_by_key: Dict[str, RunningStats]) -> Dict[str, dict]:
    """Serialize text ROUGE RunningStats for JSON (includes sum/sum_sq for shard merging)."""
    return {k: _stats_payload_running(v) for k, v in stats_by_key.items()}


def _stats_payload_running(stats: RunningStats) -> dict:
    payload = stats.to_dict()
    payload["sum"] = float(stats.total)
    payload["sum_sq"] = float(stats.total_sq)
    return payload


def merge_text_rouge_payloads(a: dict, b: dict) -> dict:
    """Merge two text_rouge_l dicts from workers (same structure as text_rouge_stats_to_payload)."""
    keys = set(a.keys()) | set(b.keys())
    out = {}
    for k in keys:
        out[k] = _merge_running_stats_dict(a.get(k, {}), b.get(k, {}))
    return out


def merge_reasoning_metrics_payloads(acc: Optional[dict], worker: dict) -> dict:
    """Merge worker ``reasoning_metrics`` into accumulator (bbox_iou, gripper_distance, text_rouge_l, ...)."""
    if not worker:
        return dict(acc or {})
    base = dict(acc or {})
    for k, bv in worker.items():
        if k == "text_rouge_l" and isinstance(bv, dict):
            base["text_rouge_l"] = merge_text_rouge_payloads(base.get("text_rouge_l", {}), bv)
        elif isinstance(bv, dict) and "count" in bv:
            base[k] = _merge_running_stats_dict(base.get(k, {}), bv)
    return base


def _merge_running_stats_dict(x: dict, y: dict) -> dict:
    """Combine two _stats_payload_running outputs into one."""

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
