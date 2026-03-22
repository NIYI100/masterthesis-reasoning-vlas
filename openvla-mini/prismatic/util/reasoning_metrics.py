import ast
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from prismatic.util.cot_utils import CotTag, split_reasoning


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
