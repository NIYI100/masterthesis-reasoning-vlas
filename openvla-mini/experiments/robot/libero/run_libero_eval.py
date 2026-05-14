"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from typing import Optional, Union
import json

import draccus
import numpy as np
import tqdm

sys.path.append("/home/hk-project-p0024638/uvrfq/LIBERO")
from libero.libero import benchmark, get_libero_path

from prismatic.util.draw_cot import draw_cot
from prismatic.util.reasoning_manipulation import *

import wandb

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../../")
from experiments.robot.libero.libero_utils import (
    apply_perturbation_to_env,
    create_bddl_with_distractors,
    extract_gt_bboxes_from_obs,
    extract_gt_gripper_pixel_from_obs,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    reset_with_perturbation,
    safe_env_reset,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from prismatic.util.reasoning_metrics import (
    TEXT_ROUGE_FIELD_SPECS,
    RunningStats,
    add_text_rouge_l_samples,
    concat_nl_reasoning_sections,
    compute_bbox_iou_stats,
    compute_gripper_distance,
    create_text_rouge_running_stats,
    parse_bboxes_from_reasoning,
    parse_gripper_from_reasoning,
    parse_text_field_from_reasoning,
    rouge_l_f1,
    text_rouge_stats_to_payload,
)
from prismatic.util.reasoning_trace_log import append_reasoning_trace_line, resolve_reasoning_trace_path
from experiments.robot.libero.task_swap_resolver import decisions_to_serializable, resolve_libero90_swaps
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "minivla"                    # Model family
    hf_token: str = Path(".hf_token")                       # Model family
    pretrained_checkpoint: Union[str, Path] = "/home/hk-project-p0024638/uvrfq/hkfswork/minivla_reasoning_run_dir/vanilla_model/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-lm-90+n1+b16+x7/checkpoints/step-200000-epoch-45-loss=0.0165.pt"
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization
    load_in_8bit: bool = False

    center_crop: bool = False                         # Center crop? (if trained w/ random crop image aug)
    obs_history: int = 1                             # Number of images to pass in from history
    use_wrist_image: bool = False                    # Use wrist images (doubles the number of input images)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_90"          # Task suite.                                      Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 20                    # Number of rollouts per task

    #################################################################################################################
    # Generalization challenge parameters (perturbation & distractors)
    #################################################################################################################
    perturbation: bool = True                       # Expand initial object regions by 1.2× and require at least one task-relevant object in the expanded portion
    distractors: bool = True                        # Add 1-2 random distractor objects from the LIBERO object suite into the scene

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs
    prefix: str = ''

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "mini-vla-libero-reasoning_eval_spatial"        # Name of W&B project to log to (use default!)
    wandb_entity: Optional[str] = "ambrosius-karlsruhe-institute-of-technology"     # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    #################################################################################################################
    # Parallel evaluation parameters
    #################################################################################################################
    num_shards: int = 1                              # Number of parallel workers for task sharding
    shard_rank: int = 0                              # Worker ID in [0, num_shards - 1]
    task_ids: Optional[str] = None                   # Optional explicit task IDs (e.g. "0,3,7-10"); overrides sharding
    save_metrics_json: bool = True                   # Save per-run metrics to JSON for easy aggregation

    
    rollout_dir_name: Optional[str] = None             # Optional name of the rollout directory
    reasoning_modifier_fn_str: str = "None"
    experiment_type: str = "vanilla"                   # vanilla | ablation | noise_bbox | noise_text | custom
    perturbation_type: str = "none"                    # Metadata for aggregation/filtering
    perturbation_level: str = "none"                   # Metadata (e.g., sigma value bucket)
    noise_sigma: float = 0.0                           # Used by parameterized noise modifiers
    ablation_components: str = ""                      # Comma-separated reasoning parts removed
    enable_reasoning_metrics: bool = True              # Compute IoU/gripper/text metrics online
    reasoning_gt_source: str = "runtime_seg"           # runtime_seg (primary implemented source)
    metrics_camera_name: str = "agentview"             # Camera used for segmentation/projection metrics
    # NDJSON: one JSON object per line with task_id, episode_idx, env_step, reasoning, … (for offline GT ROUGE).
    # With num_shards>1, path becomes base.shard{RANK}.jsonl automatically.
    reasoning_trace_jsonl: Optional[str] = None
    # NDJSON: one JSON object per executed step with raw metric values (IoU per object, gripper distance,
    # text ROUGE fields) and identifiers for later custom aggregation at step/episode/task/global levels.
    # With num_shards>1, path becomes base.shard{RANK}.jsonl automatically.
    reasoning_step_metrics_jsonl: Optional[str] = None
    # Counterfactual task/reasoning arm:
    # none | intact_task_swapped_reasoning | swapped_task_intact_reasoning
    counterfactual_arm: str = "none"
    # Optional per-step trace for counterfactual pass-1/pass-2 metadata.
    counterfactual_trace_jsonl: Optional[str] = None
    # Optional task-level audit JSON path with swap decisions (swapped + unswapped).
    counterfactual_swap_audit_json: Optional[str] = None
    # fmt: on


def _parse_task_ids(task_ids_spec: Optional[str], num_tasks_in_suite: int) -> Optional[list[int]]:
    if task_ids_spec is None:
        return None
    spec = task_ids_spec.strip()
    if spec == "":
        return None

    selected = set()
    for token in spec.split(","):
        token = token.strip()
        if token == "":
            continue
        if "-" in token:
            start_str, end_str = token.split("-", maxsplit=1)
            start_idx = int(start_str.strip())
            end_idx = int(end_str.strip())
            if end_idx < start_idx:
                raise ValueError(f"Invalid task range '{token}': end < start.")
            for task_id in range(start_idx, end_idx + 1):
                selected.add(task_id)
        else:
            selected.add(int(token))

    selected_task_ids = sorted(selected)
    for task_id in selected_task_ids:
        if task_id < 0 or task_id >= num_tasks_in_suite:
            raise ValueError(
                f"Task id {task_id} is out of range for suite with {num_tasks_in_suite} tasks "
                f"(expected in [0, {num_tasks_in_suite - 1}])."
            )
    return selected_task_ids


def _resolve_eval_task_ids(cfg: GenerateConfig, num_tasks_in_suite: int) -> list[int]:
    explicit_task_ids = _parse_task_ids(cfg.task_ids, num_tasks_in_suite)
    if explicit_task_ids is not None:
        return explicit_task_ids

    if cfg.num_shards <= 0:
        raise ValueError(f"`num_shards` must be >= 1, but got {cfg.num_shards}.")
    if cfg.shard_rank < 0 or cfg.shard_rank >= cfg.num_shards:
        raise ValueError(
            f"`shard_rank` must satisfy 0 <= shard_rank < num_shards, "
            f"but got shard_rank={cfg.shard_rank}, num_shards={cfg.num_shards}."
        )
    return [task_id for task_id in range(num_tasks_in_suite) if task_id % cfg.num_shards == cfg.shard_rank]


def _stats_payload(stats: RunningStats) -> dict:
    payload = stats.to_dict()
    payload["sum"] = float(stats.total)
    payload["sum_sq"] = float(stats.total_sq)
    return payload


def _unique_task_metrics_key(base_name: str, existing_keys: set[str]) -> str:
    """
    Return a unique per-task key by appending numeric suffixes.
    Example: task, task_1, task_2, ...
    """
    if base_name not in existing_keys:
        return base_name

    suffix = 1
    while True:
        candidate = f"{base_name}_{suffix}"
        if candidate not in existing_keys:
            return candidate
        suffix += 1


def _resolve_optional_json_path(base_path: Optional[str], num_shards: int, shard_rank: int) -> Optional[str]:
    if not base_path or not str(base_path).strip():
        return None
    p = str(base_path).strip()
    if num_shards > 1:
        root, ext = os.path.splitext(p)
        if ext.lower() == ".json":
            return f"{root}.shard{int(shard_rank)}{ext}"
        return f"{p}.shard{int(shard_rank)}.json"
    return p


def _subset_success_summary(per_task_metrics: dict, eligible_ids: set[int]) -> dict:
    rows = [row for row in per_task_metrics.values() if int(row.get("task_id", -1)) in eligible_ids]
    if len(rows) == 0:
        return {
            "num_tasks": 0,
            "num_episodes": 0,
            "num_successes": 0,
            "macro_task_success_rate": 0.0,
            "micro_episode_success_rate": 0.0,
            "per_task_success_rates": [],
        }

    per_task_success_rates = []
    total_episodes = 0
    total_successes = 0
    for row in rows:
        task_episodes = int(row.get("task_episodes", 0))
        task_successes = int(row.get("task_successes", 0))
        task_sr = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        total_episodes += task_episodes
        total_successes += task_successes
        per_task_success_rates.append(
            {
                "task_id": int(row.get("task_id", -1)),
                "task_description": row.get("task_description", ""),
                "task_episodes": task_episodes,
                "task_successes": task_successes,
                "task_success_rate": task_sr,
            }
        )

    macro = float(np.mean([row["task_success_rate"] for row in per_task_success_rates]))
    micro = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    return {
        "num_tasks": len(per_task_success_rates),
        "num_episodes": int(total_episodes),
        "num_successes": int(total_successes),
        "macro_task_success_rate": macro,
        "micro_episode_success_rate": micro,
        "per_task_success_rates": per_task_success_rates,
    }


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name
    reasoning_modifier_fn = get_reasoning_fn(cfg.reasoning_modifier_fn_str)
    valid_counterfactual_arms = {"none", "intact_task_swapped_reasoning", "swapped_task_intact_reasoning"}
    if cfg.counterfactual_arm not in valid_counterfactual_arms:
        raise ValueError(
            f"Invalid counterfactual_arm='{cfg.counterfactual_arm}'. "
            f"Expected one of {sorted(valid_counterfactual_arms)}."
        )
    counterfactual_enabled = cfg.counterfactual_arm != "none"

    reasoning_trace_resolved = resolve_reasoning_trace_path(
        cfg.reasoning_trace_jsonl, cfg.num_shards, cfg.shard_rank
    )
    if reasoning_trace_resolved:
        print(f"Reasoning trace log (NDJSON): {reasoning_trace_resolved}")
    reasoning_step_metrics_resolved = resolve_reasoning_trace_path(
        cfg.reasoning_step_metrics_jsonl, cfg.num_shards, cfg.shard_rank
    )
    if reasoning_step_metrics_resolved:
        print(f"Reasoning step-metrics log (NDJSON): {reasoning_step_metrics_resolved}")
    counterfactual_trace_resolved = resolve_reasoning_trace_path(
        cfg.counterfactual_trace_jsonl, cfg.num_shards, cfg.shard_rank
    )
    if counterfactual_trace_resolved:
        print(f"Counterfactual trace log (NDJSON): {counterfactual_trace_resolved}")
    counterfactual_swap_audit_resolved = _resolve_optional_json_path(
        cfg.counterfactual_swap_audit_json, cfg.num_shards, cfg.shard_rank
    )

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family in ["openvla", "prismatic"]:
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"
    if cfg.model_family == "minivla":
        cfg.unnorm_key = "libero_lm_90"
    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"{cfg.prefix}EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.task_ids is not None:
        run_id += "--tasks-custom"
    elif cfg.num_shards > 1:
        run_id += f"--shard-{cfg.shard_rank}-of-{cfg.num_shards}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    eval_task_ids = _resolve_eval_task_ids(cfg, num_tasks_in_suite)
    if len(eval_task_ids) == 0:
        raise ValueError(
            "No tasks selected for this worker. Check `task_ids`, `num_shards`, and `shard_rank` configuration."
        )
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")
    print(f"Evaluating {len(eval_task_ids)} task(s): {eval_task_ids}")
    log_file.write(f"Evaluating {len(eval_task_ids)} task(s): {eval_task_ids}\n")

    swap_decision_by_task_id = {}
    serializable_decisions = []
    object_swapped_task_ids: set[int] = set()
    direction_swapped_task_ids: set[int] = set()
    skipped_task_ids: set[int] = set()
    skipped_reason_counts: dict[str, int] = {}
    if counterfactual_enabled:
        if cfg.task_suite_name != "libero_90":
            raise ValueError("Counterfactual task/reasoning swap mode is currently implemented for task_suite_name=libero_90.")
        all_tasks = [task_suite.get_task(i) for i in range(num_tasks_in_suite)]
        all_decisions = resolve_libero90_swaps(all_tasks)
        serializable_decisions = decisions_to_serializable(all_decisions)
        swap_decision_by_task_id = {int(row["task_id"]): row for row in serializable_decisions}

        if counterfactual_swap_audit_resolved:
            audit_parent = os.path.dirname(os.path.abspath(counterfactual_swap_audit_resolved))
            if audit_parent:
                os.makedirs(audit_parent, exist_ok=True)
            audit_payload = {
                "task_suite_name": cfg.task_suite_name,
                "counterfactual_arm": cfg.counterfactual_arm,
                "num_tasks_in_suite": int(num_tasks_in_suite),
                "num_shards": int(cfg.num_shards),
                "shard_rank": int(cfg.shard_rank),
                "decisions": serializable_decisions,
            }
            with open(counterfactual_swap_audit_resolved, "w", encoding="utf-8") as f:
                json.dump(audit_payload, f, indent=2)
            print(f"Saved counterfactual swap audit JSON: {counterfactual_swap_audit_resolved}")

        filtered_eval_task_ids = []
        for task_id in eval_task_ids:
            row = swap_decision_by_task_id[int(task_id)]
            if row["swap_status"] == "swapped":
                filtered_eval_task_ids.append(task_id)
                if row["swap_type"] == "object":
                    object_swapped_task_ids.add(int(task_id))
                elif row["swap_type"] == "direction":
                    direction_swapped_task_ids.add(int(task_id))
            else:
                skipped_task_ids.add(int(task_id))
                reason = str(row.get("skip_reason") or "unknown")
                skipped_reason_counts[reason] = skipped_reason_counts.get(reason, 0) + 1
        eval_task_ids = filtered_eval_task_ids
        print(
            f"Counterfactual coverage: swapped={len(eval_task_ids)} "
            f"(object={len(object_swapped_task_ids)}, direction={len(direction_swapped_task_ids)}), "
            f"skipped={len(skipped_task_ids)}"
        )
        log_file.write(
            f"Counterfactual coverage: swapped={len(eval_task_ids)} "
            f"(object={len(object_swapped_task_ids)}, direction={len(direction_swapped_task_ids)}), "
            f"skipped={len(skipped_task_ids)}\n"
        )
        if skipped_reason_counts:
            log_file.write(f"Counterfactual skipped reason counts: {skipped_reason_counts}\n")
            print(f"Counterfactual skipped reason counts: {skipped_reason_counts}")
        if len(eval_task_ids) == 0:
            raise ValueError("No swappable tasks selected for counterfactual eval on this worker.")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    global_bbox_iou_stats = RunningStats()
    global_gripper_dist_stats = RunningStats()
    global_text_rouge_stats = create_text_rouge_running_stats()
    per_episode_metrics = []
    per_task_metrics = {}
    for task_id in tqdm.tqdm(eval_task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Optionally create a modified BDDL file with distractor objects
        bddl_file_override = None
        if cfg.distractors:
            original_bddl = os.path.join(
                get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
            )
            distractor_rng = np.random.RandomState(cfg.seed + task_id * 1000)
            bddl_file_override = create_bddl_with_distractors(original_bddl, rng=distractor_rng)

        # Initialize LIBERO environment and task description
        # Placement RNG: avoid env.seed(0) on every task (identical first reset across tasks).
        _placement_seed = (
            int(cfg.seed) + int(task_id) * 1000
            if (cfg.perturbation or cfg.distractors)
            else 0
        )
        env, task_description = get_libero_env(
            task,
            cfg.model_family,
            resolution=resize_size,
            enable_segmentation_metrics=cfg.enable_reasoning_metrics and cfg.reasoning_gt_source == "runtime_seg",
            bddl_file_override=bddl_file_override,
            # Perturbation mutates sampler ranges in memory; hard_reset reloads MJCF and
            # rebuilds samplers from BDDL, wiping those edits.
            hard_reset=not cfg.perturbation,
            placement_seed=_placement_seed,
        )
        counterfactual_swap_row = swap_decision_by_task_id.get(int(task_id)) if counterfactual_enabled else None
        counterfactual_swapped_description = None
        if counterfactual_enabled and counterfactual_swap_row is not None:
            counterfactual_swapped_description = str(counterfactual_swap_row.get("swapped_task_language") or "")

        # Optionally apply perturbation (expand placement regions by 1.2×)
        perturbation_info = None
        if cfg.perturbation:
            perturbation_info = apply_perturbation_to_env(env, factor=1.2)

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_bbox_iou_stats = RunningStats()
        task_gripper_dist_stats = RunningStats()
        task_text_rouge_stats = create_text_rouge_running_stats()
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            done = False
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment and set initial state
            if cfg.perturbation:
                obs = reset_with_perturbation(env, perturbation_info)
            elif cfg.distractors:
                obs = safe_env_reset(env)
            else:
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            episode_bbox_iou_stats = RunningStats()
            episode_gripper_dist_stats = RunningStats()
            episode_text_rouge_stats = create_text_rouge_running_stats()
            replay_images = []
            replay_images_annotated = []
            replay_wrist_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait and not done:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # use_wrist_image
                    if cfg.use_wrist_image:
                        wrist_img = get_libero_image(obs, resize_size, key="robot0_eye_in_hand_image")
                        replay_wrist_images.append(wrist_img)

                    # buffering #obs_history images, optionally
                    image_history = replay_images[-cfg.obs_history :]
                    if len(image_history) < cfg.obs_history:
                        image_history.extend([replay_images[-1]] * (cfg.obs_history - len(image_history)))

                    # same but for optional wrist images
                    if cfg.use_wrist_image:
                        wrist_image_history = replay_wrist_images[-cfg.obs_history :]
                        if len(wrist_image_history) < cfg.obs_history:
                            wrist_image_history.extend(
                                [replay_wrist_images[-1]] * (cfg.obs_history - len(wrist_image_history))
                            )
                        # interleaved images [... image_t, wrist_t ...]
                        image_history = [val for tup in zip(image_history, wrist_image_history) for val in tup]

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": image_history,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    if reasoning_modifier_fn is not None and hasattr(reasoning_modifier_fn, "set_context"):
                        # Temporal modifiers can use runtime step/task context to decide whether to perturb.
                        reasoning_modifier_fn.set_context(task_id=int(task_id), env_step=int(t))
                    reasoning_source = "default_generation"
                    pass1_reasoning = None
                    pass1_clean_reasoning = None
                    pass1_task_description = None
                    pass2_task_description = task_description
                    if counterfactual_enabled:
                        if not counterfactual_swapped_description:
                            raise RuntimeError(
                                f"Missing swapped task description for task_id={task_id} in counterfactual mode."
                            )
                        if cfg.counterfactual_arm == "intact_task_swapped_reasoning":
                            pass1_task_description = counterfactual_swapped_description
                            pass2_task_description = task_description
                            reasoning_source = "reasoning_from_swapped_task"
                        else:
                            pass1_task_description = task_description
                            pass2_task_description = counterfactual_swapped_description
                            reasoning_source = "reasoning_from_intact_task"

                        _, pass1_reasoning, pass1_clean_reasoning = get_action(
                            cfg,
                            model,
                            observation,
                            pass1_task_description,
                            reasoning_modifier_fn=reasoning_modifier_fn,
                            processor=processor,
                        )
                        forced_reasoning = pass1_reasoning if isinstance(pass1_reasoning, str) else ""
                        action, reasoning, clean_reasoning = get_action(
                            cfg,
                            model,
                            observation,
                            pass2_task_description,
                            reasoning_modifier_fn=None,
                            forced_reasoning=forced_reasoning,
                            processor=processor,
                        )
                    else:
                        action, reasoning, clean_reasoning = get_action(
                            cfg,
                            model,
                            observation,
                            task_description,
                            reasoning_modifier_fn=reasoning_modifier_fn,
                            processor=processor,
                        )

                    if counterfactual_trace_resolved and counterfactual_enabled:
                        append_reasoning_trace_line(
                            counterfactual_trace_resolved,
                            {
                                "task_suite_name": cfg.task_suite_name,
                                "task_id": int(task_id),
                                "task_description": task_description,
                                "episode_idx": int(episode_idx),
                                "env_step": int(t),
                                "counterfactual_arm": cfg.counterfactual_arm,
                                "swap_type": None if counterfactual_swap_row is None else counterfactual_swap_row.get("swap_type"),
                                "swapped_task_id": None
                                if counterfactual_swap_row is None
                                else counterfactual_swap_row.get("swapped_task_id"),
                                "swapped_task_description": counterfactual_swapped_description,
                                "pass1_task_description": pass1_task_description,
                                "pass1_reasoning": pass1_reasoning,
                                "pass1_clean_reasoning": pass1_clean_reasoning,
                                "pass2_task_description": pass2_task_description,
                                "pass2_reasoning_used_for_action": reasoning,
                                "reasoning_source": reasoning_source,
                                "shard_rank": int(cfg.shard_rank),
                                "num_shards": int(cfg.num_shards),
                                "seed": int(cfg.seed),
                            },
                        )

                    if reasoning_trace_resolved and isinstance(reasoning, str):
                        ac = action
                        n_ctrl = 1
                        if isinstance(ac, np.ndarray) and ac.ndim == 2 and ac.shape[0] > 1:
                            n_ctrl = int(ac.shape[0])
                        append_reasoning_trace_line(
                            reasoning_trace_resolved,
                            {
                                "task_suite_name": cfg.task_suite_name,
                                "task_id": int(task_id),
                                "task_description": task_description,
                                "episode_idx": int(episode_idx),
                                "env_step": int(t),
                                "shard_rank": int(cfg.shard_rank),
                                "num_shards": int(cfg.num_shards),
                                "seed": int(cfg.seed),
                                "reasoning": reasoning,
                                "clean_reasoning": clean_reasoning,
                                "reasoning_modifier_fn_str": cfg.reasoning_modifier_fn_str,
                                "model_family": cfg.model_family,
                                "control_actions_from_forward": n_ctrl,
                                "counterfactual_arm": cfg.counterfactual_arm,
                                "reasoning_source": reasoning_source,
                                "swapped_task_id": None
                                if counterfactual_swap_row is None
                                else counterfactual_swap_row.get("swapped_task_id"),
                                "swapped_task_description": counterfactual_swapped_description,
                                "pass1_reasoning": pass1_reasoning,
                            },
                        )

                    if action.shape == (1, 7):
                        action = [action]
                    for action_idx_in_forward, predicted_action in enumerate(action):
                        #print(f"{reasoning} ACTION: {predicted_action}")
                        #TODO - for loop für mehrere actions
                        if cfg.enable_reasoning_metrics and isinstance(reasoning, str):
                            # Collect one metric sample per executed action (important for chunked action outputs).
                            gt_bboxes = extract_gt_bboxes_from_obs(env, obs, camera_name=cfg.metrics_camera_name)
                            pred_bboxes = parse_bboxes_from_reasoning(reasoning)
                            iou_stats = {"count": 0, "mean": None, "std": None, "per_object_iou": {}}
                            if gt_bboxes:
                                iou_stats = compute_bbox_iou_stats(pred_bboxes, gt_bboxes)
                                if iou_stats["count"] > 0:
                                    iou_values = list(iou_stats["per_object_iou"].values())
                                    episode_bbox_iou_stats.add_many(iou_values)
                                    task_bbox_iou_stats.add_many(iou_values)
                                    global_bbox_iou_stats.add_many(iou_values)

                            pred_gripper = parse_gripper_from_reasoning(reasoning)
                            gt_gripper = extract_gt_gripper_pixel_from_obs(env, obs, camera_name=cfg.metrics_camera_name)
                            gripper_dist = compute_gripper_distance(pred_gripper, gt_gripper)
                            episode_gripper_dist_stats.add(gripper_dist)
                            task_gripper_dist_stats.add(gripper_dist)
                            global_gripper_dist_stats.add(gripper_dist)

                            use_task_ref = clean_reasoning is None
                            ref_for_rouge = task_description if use_task_ref else clean_reasoning
                            add_text_rouge_l_samples(
                                episode_text_rouge_stats,
                                reasoning,
                                ref_for_rouge,
                                use_task_ref,
                            )
                            add_text_rouge_l_samples(
                                task_text_rouge_stats,
                                reasoning,
                                ref_for_rouge,
                                use_task_ref,
                            )
                            add_text_rouge_l_samples(
                                global_text_rouge_stats,
                                reasoning,
                                ref_for_rouge,
                                use_task_ref,
                            )

                            # Optional step-level raw metrics log for flexible offline aggregation.
                            if reasoning_step_metrics_resolved:
                                text_scores: dict[str, float] = {}
                                for key, tag in TEXT_ROUGE_FIELD_SPECS:
                                    pred_t = parse_text_field_from_reasoning(reasoning, tag)
                                    if use_task_ref:
                                        ref_t = ref_for_rouge
                                    else:
                                        ref_t = parse_text_field_from_reasoning(ref_for_rouge, tag) if ref_for_rouge else None
                                    if pred_t is None:
                                        continue
                                    if ref_t is None or (isinstance(ref_t, str) and ref_t.strip() == ""):
                                        continue
                                    score = rouge_l_f1(pred_t, ref_t)
                                    if score is not None:
                                        text_scores[key] = float(score)

                                pred_whole = concat_nl_reasoning_sections(reasoning)
                                ref_whole = (
                                    ref_for_rouge
                                    if use_task_ref
                                    else concat_nl_reasoning_sections(ref_for_rouge)
                                )
                                whole_score = None
                                if pred_whole.strip() and ref_whole.strip():
                                    ws = rouge_l_f1(pred_whole, ref_whole)
                                    if ws is not None:
                                        whole_score = float(ws)

                                append_reasoning_trace_line(
                                    reasoning_step_metrics_resolved,
                                    {
                                        "task_suite_name": cfg.task_suite_name,
                                        "task_id": int(task_id),
                                        "task_description": task_description,
                                        "episode_idx": int(episode_idx),
                                        "env_step": int(t),
                                        "step_in_forward": int(action_idx_in_forward),
                                        "control_actions_from_forward": int(n_ctrl),
                                        "shard_rank": int(cfg.shard_rank),
                                        "num_shards": int(cfg.num_shards),
                                        "seed": int(cfg.seed),
                                        "reasoning": reasoning,
                                        "clean_reasoning": clean_reasoning,
                                        "reasoning_modifier_fn_str": cfg.reasoning_modifier_fn_str,
                                        "reasoning_source": reasoning_source,
                                        "reasoning_gt_source": cfg.reasoning_gt_source,
                                        "metrics_camera_name": cfg.metrics_camera_name,
                                        "pred_bboxes": pred_bboxes,
                                        "gt_bboxes": gt_bboxes,
                                        "bbox_iou_count": int(iou_stats.get("count", 0) or 0),
                                        "bbox_iou_per_object": iou_stats.get("per_object_iou", {}),
                                        "bbox_iou_mean": iou_stats.get("mean"),
                                        "pred_gripper": pred_gripper,
                                        "gt_gripper": gt_gripper,
                                        "gripper_distance": gripper_dist,
                                        "text_rouge_l": {"whole": whole_score, **text_scores},
                                    },
                                )

                        # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                        predicted_action = normalize_gripper_action(predicted_action, binarize=True)
                        # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                        # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                        if cfg.model_family in ["openvla", "prismatic", "minivla"]:
                            predicted_action = invert_gripper_action(predicted_action)

                        annotated_obs = draw_cot(reasoning, predicted_action, img)
                        replay_images_annotated.append(annotated_obs)

                        # Execute action in environment
                        obs, reward, done, info = env.step(predicted_action.tolist())
                        img = get_libero_image(obs, resize_size)
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1
            per_episode_metrics.append(
                {
                    "task_id": int(task_id),
                    "task_description": task_description,
                    "episode_idx": int(episode_idx),
                    "success": bool(done),
                    "steps_executed": int(t),
                    "bbox_iou": _stats_payload(episode_bbox_iou_stats),
                    "gripper_distance": _stats_payload(episode_gripper_dist_stats),
                    "text_rouge_l": text_rouge_stats_to_payload(episode_text_rouge_stats),
                }
            )

            # Save a replay video of the episode
            rollout_idx = task_id * cfg.num_trials_per_task + episode_idx + 1
            save_rollout_video(
                replay_images_annotated, rollout_idx, success=done, task_description=task_description, log_file=log_file, rollout_dir_name=cfg.rollout_dir_name
            )

            # Save the videos to wandb
            if cfg.use_wandb and (task_successes < 10 or task_episodes - task_successes < 10):
                group = "success" if done else "failure"
                idx = task_successes if done else task_episodes - task_successes
                wandb.log(
                    {f"{task_description}/{group}/{idx}": wandb.Video(np.array(replay_images).transpose(0, 3, 1, 2))}
                )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )
        metrics_key = _unique_task_metrics_key(task_description, set(per_task_metrics.keys()))

        per_task_metrics[metrics_key] = {
            "task_id": int(task_id),
            "task_description": task_description,
            "task_successes": int(task_successes),
            "task_episodes": int(task_episodes),
            "task_success_rate": float(task_successes) / float(task_episodes),
            "bbox_iou": _stats_payload(task_bbox_iou_stats),
            "gripper_distance": _stats_payload(task_gripper_dist_stats),
            "text_rouge_l": text_rouge_stats_to_payload(task_text_rouge_stats),
            "counterfactual_swap_type": None
            if counterfactual_swap_row is None
            else counterfactual_swap_row.get("swap_type"),
            "counterfactual_swapped_task_id": None
            if counterfactual_swap_row is None
            else counterfactual_swap_row.get("swapped_task_id"),
            "counterfactual_swapped_task_description": counterfactual_swapped_description,
        }

        # Clean up temporary distractor BDDL file
        if bddl_file_override is not None and bddl_file_override != os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        ):
            try:
                os.unlink(bddl_file_override)
            except OSError:
                pass

    # Save local log file
    log_file.close()

    # Push total metrics and local log file to wandb
    if cfg.use_wandb:
        wandb.log(
            {
                "success_rate/total": float(total_successes) / float(total_episodes),
                "num_episodes/total": total_episodes,
            }
        )
        wandb.save(local_log_filepath)

    counterfactual_summary = None
    if counterfactual_enabled:
        swapped_eval_task_ids = object_swapped_task_ids | direction_swapped_task_ids
        counterfactual_summary = {
            "arm": cfg.counterfactual_arm,
            "coverage": {
                "total_selected_tasks": int(len(eval_task_ids) + len(skipped_task_ids)),
                "swapped_tasks": int(len(swapped_eval_task_ids)),
                "object_swapped_tasks": int(len(object_swapped_task_ids)),
                "direction_swapped_tasks": int(len(direction_swapped_task_ids)),
                "not_swapped_tasks": int(len(skipped_task_ids)),
                "not_swapped_task_ids": sorted(int(task_id) for task_id in skipped_task_ids),
                "not_swapped_by_reason": skipped_reason_counts,
            },
            "swap_decisions": serializable_decisions,
            "success_rates": {
                "overall_swapped": _subset_success_summary(per_task_metrics, swapped_eval_task_ids),
                "object_swapped": _subset_success_summary(per_task_metrics, object_swapped_task_ids),
                "direction_swapped": _subset_success_summary(per_task_metrics, direction_swapped_task_ids),
            },
        }

    if cfg.save_metrics_json:
        metrics_payload = {
            "experiment_type": cfg.experiment_type,
            "perturbation_type": cfg.perturbation_type,
            "perturbation_level": cfg.perturbation_level,
            "perturbation": cfg.perturbation,
            "distractors": cfg.distractors,
            "noise_sigma": float(cfg.noise_sigma),
            "ablation_components": cfg.ablation_components,
            "reasoning_modifier_fn_str": cfg.reasoning_modifier_fn_str,
            "counterfactual_arm": cfg.counterfactual_arm,
            "counterfactual_trace_jsonl": cfg.counterfactual_trace_jsonl,
            "counterfactual_trace_jsonl_resolved": counterfactual_trace_resolved,
            "counterfactual_swap_audit_json": cfg.counterfactual_swap_audit_json,
            "counterfactual_swap_audit_json_resolved": counterfactual_swap_audit_resolved,
            "reasoning_gt_source": cfg.reasoning_gt_source,
            "metrics_camera_name": cfg.metrics_camera_name,
            "reasoning_trace_jsonl": cfg.reasoning_trace_jsonl,
            "reasoning_trace_jsonl_resolved": reasoning_trace_resolved,
            "reasoning_step_metrics_jsonl": cfg.reasoning_step_metrics_jsonl,
            "reasoning_step_metrics_jsonl_resolved": reasoning_step_metrics_resolved,
            "task_suite_name": cfg.task_suite_name,
            "model_family": cfg.model_family,
            "pretrained_checkpoint": str(cfg.pretrained_checkpoint),
            "num_trials_per_task": int(cfg.num_trials_per_task),
            "num_tasks_in_suite": int(num_tasks_in_suite),
            "evaluated_task_ids": [int(task_id) for task_id in eval_task_ids],
            "num_shards": int(cfg.num_shards),
            "shard_rank": int(cfg.shard_rank),
            "task_ids": cfg.task_ids,
            "total_episodes": int(total_episodes),
            "total_successes": int(total_successes),
            "total_success_rate": float(total_successes) / float(total_episodes),
            "reasoning_metrics": {
                "bbox_iou": _stats_payload(global_bbox_iou_stats),
                "gripper_distance": _stats_payload(global_gripper_dist_stats),
                "text_rouge_l": text_rouge_stats_to_payload(global_text_rouge_stats),
            },
            "per_task_metrics": per_task_metrics,
            "per_episode_metrics": per_episode_metrics,
            "counterfactual_summary": counterfactual_summary,
            "log_path": local_log_filepath,
        }
        metrics_json_filepath = os.path.join(cfg.local_log_dir, run_id + ".json")
        with open(metrics_json_filepath, "w") as metrics_file:
            json.dump(metrics_payload, metrics_file, indent=2)
        print(f"Saved metrics JSON: {metrics_json_filepath}")


if __name__ == "__main__":
    eval_libero()
