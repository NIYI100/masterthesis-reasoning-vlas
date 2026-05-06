#!/usr/bin/env python3
"""
Visualize GT bounding boxes and gripper pixel position using the same extraction
as reasoning metrics (extract_gt_bboxes_from_obs, extract_gt_gripper_pixel_from_obs)
on top of get_libero_image — same flip/resolution as IoU / gripper distance.

Saves a PNG snapshot and optionally an MP4 while stepping the sim with a dummy action.

Run from repo root with the same env as eval, e.g.:
  conda activate minivla
  cd openvla-mini && python experiments/robot/libero/visualize_gt_metrics_overlay.py --output_dir experiments/logs/gt_metrics_viz_demo
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import imageio
import numpy as np

# Repo root = openvla-mini (parent of experiments/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.append("/home/hk-project-p0024638/uvrfq/LIBERO")

from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    extract_gt_bboxes_from_obs,
    extract_gt_gripper_pixel_from_obs,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    safe_env_reset,
)


def render_overlay_rgb(
    img_rgb: np.ndarray,
    gt_boxes: dict,
    gt_grip: list | None,
    *,
    box_color_bgr=(0, 255, 0),
    gripper_color_bgr=(0, 0, 255),
    gripper_ring_bgr=(255, 255, 255),
) -> np.ndarray:
    """Draw GT boxes and gripper on RGB image; returns RGB uint8."""
    img_bgr = cv2.cvtColor(np.ascontiguousarray(img_rgb), cv2.COLOR_RGB2BGR)
    for name, bbox in gt_boxes.items():
        try:
            (x1, y1), (x2, y2) = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        except (TypeError, ValueError, IndexError):
            continue
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), box_color_bgr, 2)
        label = str(name)[:50]
        cv2.putText(
            img_bgr,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            box_color_bgr,
            1,
            cv2.LINE_AA,
        )
    if gt_grip is not None and len(gt_grip) >= 2:
        gx, gy = int(gt_grip[0]), int(gt_grip[1])
        cv2.circle(img_bgr, (gx, gy), 8, gripper_color_bgr, -1)
        cv2.circle(img_bgr, (gx, gy), 10, gripper_ring_bgr, 1)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    p = argparse.ArgumentParser(description="Save GT bbox + gripper overlays (metrics-aligned).")
    p.add_argument("--output_dir", type=str, default="experiments/logs/gt_metrics_viz")
    p.add_argument("--task_suite_name", type=str, default="libero_90")
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--episode_idx", type=int, default=0, help="Which init state from the task suite.")
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--video_frames", type=int, default=60, help="Extra frames after wait (dummy actions); 0 skips MP4.")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--model_family", type=str, default="minivla", help="Only affects dummy action layout.")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    resize_size = args.resize
    image_key = f"{args.camera}_image"

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)

    env, task_description = get_libero_env(
        task,
        model_family=args.model_family,
        resolution=resize_size,
        enable_segmentation_metrics=True,
        hard_reset=True,
        placement_seed=0,
    )
    try:
        _run_capture_loop(
            env,
            args,
            resize_size,
            image_key,
            out_dir,
            stamp,
            task_description,
            initial_states,
        )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass


def _run_capture_loop(
    env,
    args,
    resize_size,
    image_key,
    out_dir: Path,
    stamp: str,
    task_description: str,
    initial_states: list,
) -> None:
    obs = safe_env_reset(env)
    ep_idx = min(args.episode_idx, len(initial_states) - 1)
    obs = env.set_init_state(initial_states[ep_idx])

    dummy = get_libero_dummy_action(args.model_family)

    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(dummy)

    def capture_overlay() -> tuple[np.ndarray, dict, list | None]:
        img_rgb = get_libero_image(obs, resize_size, key=image_key)
        boxes = extract_gt_bboxes_from_obs(
            env, obs, camera_name=args.camera, flip_vertical=True
        )
        grip = extract_gt_gripper_pixel_from_obs(
            env, obs, camera_name=args.camera, flip_vertical=True
        )
        return render_overlay_rgb(img_rgb, boxes, grip), boxes, grip

    frames_rgb: list[np.ndarray] = []
    gt_boxes: dict = {}
    gt_grip = None

    if args.video_frames <= 0:
        overlay, gt_boxes, gt_grip = capture_overlay()
        frames_rgb = [overlay]
    else:
        for _ in range(args.video_frames):
            overlay, gt_boxes, gt_grip = capture_overlay()
            frames_rgb.append(overlay)
            obs, _, _, _ = env.step(dummy)

    snapshot = frames_rgb[-1]

    png_path = out_dir / f"gt_overlay_{args.task_suite_name}_task{args.task_id}_ep{ep_idx}_{stamp}.png"
    imageio.imwrite(str(png_path), snapshot)

    meta_lines = [
        f"task_suite={args.task_suite_name}",
        f"task_id={args.task_id}",
        f"episode_idx={ep_idx}",
        f"task_description={task_description!r}",
        f"resize={resize_size}",
        f"camera={args.camera}",
        f"num_gt_boxes={len(gt_boxes)}",
        f"gt_gripper_pixel={gt_grip}",
        f"instance_to_id_keys_sample={list(getattr(env, 'instance_to_id', {}).keys())[:12]}",
    ]
    txt_path = out_dir / f"gt_overlay_{args.task_suite_name}_task{args.task_id}_ep{ep_idx}_{stamp}.txt"
    txt_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    mp4_path = None
    if args.video_frames > 0 and len(frames_rgb) > 0:
        mp4_path = out_dir / f"gt_overlay_{args.task_suite_name}_task{args.task_id}_ep{ep_idx}_{stamp}.mp4"
        w = imageio.get_writer(str(mp4_path), fps=args.fps)
        for fr in frames_rgb:
            w.append_data(fr)
        w.close()

    print(f"Wrote PNG: {png_path}")
    print(f"Wrote meta: {txt_path}")
    if mp4_path:
        print(f"Wrote MP4 ({len(frames_rgb)} frames @ {args.fps} fps): {mp4_path}")
    print(f"Final frame: n_boxes={len(gt_boxes)}, gripper={gt_grip}")


if __name__ == "__main__":
    main()
