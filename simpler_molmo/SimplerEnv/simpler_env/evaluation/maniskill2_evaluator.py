"""
Evaluate a model on ManiSkill2 environment.
"""

import math
import os
import time

import json
from numpy.linalg import norm
import numpy as np
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video

from datetime import datetime
from collections import defaultdict

import random

import re


def _truncate_task(s: str, max_len: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."

def _json_safe(obj):
    """Convert numpy / bool_ types for json.dump."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _swap_objects_positional(instruction):
    """
    Parses strings like 'Move the [X] near the [Y]' 
    and returns 'Move the [Y] near the [X]'
    """
    
    # Regex Explanation:
    # (?i)       -> Case insensitive (matches "Move" or "move")
    # Move the   -> The start anchor
    # \s+        -> Matches spaces
    # (.*?)      -> Capture Group 1: The First Object (non-greedy)
    # \s+near the\s+ -> The middle anchor
    # (.*)       -> Capture Group 2: The Second Object (rest of string)
    
    pattern = r"(?i)move\s+(.*?)\s+near\s+(.*)"
    match = re.search(pattern, instruction)
    
    if match:
        obj_a = match.group(1).strip() # e.g., "coke can"
        obj_b = match.group(2).strip() # e.g., "apple"
        
        obj_b = obj_b.rstrip(".,!")
        
        return f"move {obj_b} near {obj_a}"
    return instruction

def _replace_objects():
    """
    Parses strings like 'Move the [X] near the [Y]' 
    and returns 'Move the [Y] near the [X]'
    """
        
    return f"move object near object"


def write_logs(
    video_path,
    video_name_original,
    real_actions,
    manipulated_actions,
    args,
    extension_count,
    verbose=False,
    episode_context=None,
):
    """
    Compare first-pass vs second-pass world_vector actions (per-step), write JSON next to the video
    (same basename as ``video_name_original``, e.g. ``*_original_*.json``).

    ``episode_context``: optional dict merged into ``meta`` (rollout_index, success, env, ckpt, …).

    Returns the log dict (also written to disk).
    """
    video_dir = os.path.dirname(video_path)
    video_basename = os.path.splitext(os.path.basename(video_name_original))[0]
    json_path = os.path.join(video_dir, f"{video_basename}.json")

    _cf_fn = (episode_context or {}).get("counterfactual_perturb_fn") if episode_context else None
    if _cf_fn:
        _delta_help = (
            "counterfactual_control mode (--counterfactual-perturb-fn set): real_actions hold the "
            "vanilla (original first-pass) world XYZ; manipulated_actions hold the executed world XYZ "
            "(counterfactual regen when parsing succeeds). The robot was stepped with the executed "
            "full action. The saved rollout video uses the first-pass frame with trajectory drawn from "
            "the perturbed-reasoning assistant text (aligned with the counterfactual action). A second "
            "HF pass still runs for verbose logs / legacy comparison. Euclidean error is L2(vanilla_xyz - executed_xyz)."
        )
    else:
        _delta_help = (
            "Per timestep: world_vector from the first model.step(update_state=False) minus "
            "world_vector from the second model.step(update_state=True, evaluator-injected reasoning). "
            "Euclidean error is L2 norm of that difference."
        )
    interpretation = {
        "log_format_version": 3,
        "action_delta_first_vs_second_pass": _delta_help,
    }

    def _base_meta(steps_n):
        m = {
            "task": args.task,
            "video_name": os.path.basename(video_path),
            "total_steps": int(steps_n),
            "coordinate extensions": extension_count,
            "coordinate_extensions": extension_count,
            "analysis_json_path": json_path,
            "json_basename": os.path.basename(json_path),
        }
        if episode_context:
            for k, v in _json_safe(episode_context).items():
                if v is not None:
                    m[k] = v
        return m

    n = len(real_actions)
    if n == 0:
        log_data = {
            "interpretation": interpretation,
            "meta": _base_meta(0),
            "summary_metrics": {
                "mean_euclidean_error_xyz": None,
                "median_euclidean_error_xyz": None,
                "max_euclidean_error_xyz": None,
                "std_euclidean_error_xyz": None,
                "mean_error_x": None,
                "mean_error_y": None,
                "mean_error_z": None,
                "mean_abs_error_x": None,
                "mean_abs_error_y": None,
                "mean_abs_error_z": None,
                "mean_cosine_similarity": None,
            },
            "steps": [],
        }
        if verbose:
            print("Writing analysis log file (0 steps)")
        with open(json_path, "w") as f:
            json.dump(log_data, f, indent=4)
        return log_data

    real_xyz = np.array(real_actions)
    man_xyz = np.array(manipulated_actions)

    euclidean_dists = norm(real_xyz - man_xyz, axis=1)
    mean_euclidean = float(np.mean(euclidean_dists))

    differences = real_xyz - man_xyz
    avg_diff_per_axis = np.mean(differences, axis=0)
    mean_abs = np.mean(np.abs(differences), axis=0)

    epsilon = 1e-8
    real_norm = norm(real_xyz, axis=1) + epsilon
    man_norm = norm(man_xyz, axis=1) + epsilon
    dot_products = np.sum(real_xyz * man_xyz, axis=1)
    cosine_sims = dot_products / (real_norm * man_norm)
    mean_cosine = float(np.mean(cosine_sims))

    log_data = {
        "interpretation": interpretation,
        "meta": _base_meta(n),
        "summary_metrics": {
            "mean_euclidean_error_xyz": mean_euclidean,
            "median_euclidean_error_xyz": float(np.median(euclidean_dists)),
            "max_euclidean_error_xyz": float(np.max(euclidean_dists)),
            "std_euclidean_error_xyz": float(np.std(euclidean_dists)),
            "mean_error_x": float(avg_diff_per_axis[0]),
            "mean_error_y": float(avg_diff_per_axis[1]),
            "mean_error_z": float(avg_diff_per_axis[2]),
            "mean_abs_error_x": float(mean_abs[0]),
            "mean_abs_error_y": float(mean_abs[1]),
            "mean_abs_error_z": float(mean_abs[2]),
            "mean_cosine_similarity": mean_cosine,
        },
        "steps": [],
    }

    for i in range(n):
        step_info = {
            "step": i + 1,
            "euclidean_error": float(euclidean_dists[i]),
            "error_x": float(differences[i][0]),
            "error_y": float(differences[i][1]),
            "error_z": float(differences[i][2]),
            "cosine_similarity": float(cosine_sims[i]),
        }
        log_data["steps"].append(step_info)

    if verbose:
        print("Writing analysis log file")

    with open(json_path, "w") as f:
        json.dump(log_data, f, indent=4)
    return log_data


def _build_experiment_summary(episode_records, args):
    """Aggregate per-rollout records into one JSON-serializable dict."""
    n = len(episode_records)
    successes = sum(1 for e in episode_records if e["success"])
    eucs = [
        e["summary_metrics"]["mean_euclidean_error_xyz"]
        for e in episode_records
        if e["summary_metrics"].get("mean_euclidean_error_xyz") is not None
    ]
    cosines = [
        e["summary_metrics"]["mean_cosine_similarity"]
        for e in episode_records
        if e["summary_metrics"].get("mean_cosine_similarity") is not None
    ]

    by_task = defaultdict(lambda: {"n": 0, "successes": 0})
    for e in episode_records:
        t = e.get("task_final") or ""
        by_task[t]["n"] += 1
        by_task[t]["successes"] += int(e["success"])

    by_task_out = {
        task: {
            "n": v["n"],
            "success_rate": v["successes"] / v["n"] if v["n"] else 0.0,
        }
        for task, v in sorted(by_task.items(), key=lambda x: (-x[1]["n"], x[0]))
    }

    episodes_out = []
    log_root = os.path.abspath(args.logging_dir)
    for e in episode_records:
        sm = e["summary_metrics"]
        row = {
            "rollout_index": e["rollout_index"],
            "success": e["success"],
            "task_final": e.get("task_final"),
            "env_name": e.get("env_name"),
            "scene_name": e.get("scene_name"),
            "experiment": e.get("experiment"),
            "total_steps": e.get("total_steps"),
            "coordinate_extensions": e.get("coordinate_extensions"),
            "episode_stats": e.get("episode_stats"),
            "mean_euclidean_error_xyz_first_vs_second_pass": sm.get(
                "mean_euclidean_error_xyz"
            ),
            "mean_cosine_similarity_first_vs_second_pass": sm.get(
                "mean_cosine_similarity"
            ),
            "mean_error_xyz_first_minus_second_pass": {
                "x": sm.get("mean_error_x"),
                "y": sm.get("mean_error_y"),
                "z": sm.get("mean_error_z"),
            },
        }
        vp = e.get("video_path")
        if vp:
            try:
                row["video_relpath"] = os.path.relpath(
                    os.path.abspath(vp), start=log_root
                )
            except ValueError:
                row["video_relpath"] = vp
        aj = e.get("analysis_json_path")
        if aj:
            try:
                row["analysis_json_relpath"] = os.path.relpath(
                    os.path.abspath(aj), start=log_root
                )
            except ValueError:
                row["analysis_json_relpath"] = aj
        episodes_out.append(row)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "policy_model": getattr(args, "policy_model", None),
            "ckpt_path": getattr(args, "ckpt_path", None),
            "env_name": getattr(args, "env_name", None),
            "scene_name": getattr(args, "scene_name", None),
            "robot": getattr(args, "robot", None),
            "experiment": getattr(args, "experiment", None),
            "logging_dir": getattr(args, "logging_dir", None),
            "obj_variation_mode": getattr(args, "obj_variation_mode", None),
            "counterfactual_perturb_fn": getattr(
                args, "counterfactual_perturb_fn", None
            ),
            "counterfactual_max_new_tokens": getattr(
                args, "counterfactual_max_new_tokens", None
            ),
        },
        "aggregate": {
            "n_rollouts": n,
            "success_rate": successes / n if n else 0.0,
            "mean_mean_euclidean_error_xyz_first_vs_second_pass": float(np.mean(eucs))
            if eucs
            else None,
            "median_mean_euclidean_error_xyz_first_vs_second_pass": float(np.median(eucs))
            if eucs
            else None,
            "mean_mean_cosine_similarity_first_vs_second_pass": float(np.mean(cosines))
            if cosines
            else None,
        },
        "by_task": by_task_out,
        "episodes": episodes_out,
    }


def _write_experiment_summary_json(path, summary, verbose=False):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"Wrote experiment summary to {path}")


def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
    experiment="coke_can",
    verbose=False,
    rollout_index: int = 0,
    quiet_episodes: bool = False,
    policy_model=None,
    counterfactual_perturb_fn=None,
    counterfactual_max_new_tokens=None,
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    episode_t0 = time.time()
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask() 

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    if verbose:
        print(task_description)

    _ckpt = ckpt_path or "none"
    ckpt_path_basename = _ckpt[:-1] if _ckpt.endswith("/") else _ckpt
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]

    if not quiet_episodes:
        if obj_episode_id is not None:
            obj_info = f"episode_id={obj_episode_id}"
        else:
            obj_info = f"obj_xy=({obj_init_x},{obj_init_y})"
        build_bits = []
        if additional_env_build_kwargs:
            for bk, bv in additional_env_build_kwargs.items():
                build_bits.append(f"{bk}={bv}")
        build_str = ",".join(build_bits) if build_bits else "-"
        tags_str = additional_env_save_tags or "-"
        print(
            f"[rollout {rollout_index}] start | env={env_name} | scene={scene_name} | "
            f"experiment={experiment} | ckpt={ckpt_path_basename}"
        )
        print(f"  task: {_truncate_task(task_description)}")
        print(
            f"  robot=({robot_init_x:.3f},{robot_init_y:.3f}) | {obj_info} | "
            f"build=[{build_str}] | save_tags={tags_str}"
        )

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    annotated_frames = []
    annotated_frames_2 = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"
    real_actions = []
    manipulated_actions = []

    args = type('Args', (), {
        "env_name": "simpler_env",
        "unnorm_key": "fractal20220817_data",
        "task": task_description,
    })()

    x_shift = 0
    y_shift = 0
    i = 0
    extension_count = 0
    distance_threshold = 40.0
    custom_middle_points = [[200, 180], [100, 180], [55, 110]]

    # Step the environment
    while not (predicted_terminated or truncated):
        args.task = task_description

        # Normal inference
        args.image = image
        args.update_state = False
        args.reasoning = None

        raw_output, raw_action, action, annotated_image = model.step(args)
        baseline_wv = getattr(model, "last_baseline_world_vector", None)
        executed_action = None
        if baseline_wv is not None:
            real_actions.append(np.asarray(baseline_wv, dtype=np.float64))
            executed_action = action
        else:
            real_actions.append(action["world_vector"])
        predicted_actions.append(raw_action)
        annotated_frames.append(annotated_image)

        depth, trace, _ = model.get_reasoning(raw_output)
        trace = trace[0]

        

        if experiment == "coke_can":
            first_coord = trace[0]
            last_coord = trace[-1]

            dist_to_last_point = math.sqrt((last_coord[0] - first_coord[0])**2 + (last_coord[1] - first_coord[1])**2)
            # Custom trace, if distance is still large
            if dist_to_last_point > distance_threshold:
                extension_count += 1

                filtered_middle_points = []
                if len(custom_middle_points) > 0:
                    first_custom_point = custom_middle_points[0]
                    distance_to_first_custom_point = math.sqrt((first_custom_point[0] - first_coord[0])**2 + (first_custom_point[1] - first_coord[1])**2)
                    # Pop custom_points, when in close proximity
                    if distance_to_first_custom_point < distance_threshold:
                        custom_middle_points.pop(0)

                new_coords = [first_coord] + custom_middle_points + [last_coord]
            else:
                new_coords = trace
            new_coords = str(new_coords).replace(" ", "")
            reasoning = f"The depth map of the image is {depth[0]}. The trajectory of the end effector is {new_coords}."

        elif experiment == "move_near":
            #x_shift = 15
            #y_shift = 15
            #y_shift = random.randint(0, 30)
            #first_coord = trace[0]
            #new_coords = [[x - x_shift, y - y_shift] for x, y in trace]
            #new_coords[0] = first_coord
            #new_task = _replace_objects()
            #args.task = new_task
            #new_coords = [trace]
            #new_coords = str(new_coords).replace(" ", "")
            swapped_task = _replace_objects()
            args.task = swapped_task
            reasoning = f"The depth map of the image is {depth[0]}. The trajectory of the end effector is {trace}."


        # Manipulated inference
        args.image = image
        args.update_state = True
        #args.reasoning = None
        #args.task = task_description
        args.reasoning = reasoning
        manipulated_raw_output, manipulated_raw_action, manipulated_action, manipulated_annotated_image = model.step(
            args
        )
        if executed_action is not None:
            manipulated_actions.append(executed_action["world_vector"])
            # Primary video: perturbed-reasoning trajectory overlay matches env action (counterfactual).
            annotated_frames_2.append(annotated_image)
        else:
            manipulated_actions.append(manipulated_action["world_vector"])
            annotated_frames_2.append(manipulated_annotated_image)

        if verbose and i % 10 == 0:
            print(f"Task:             {task_description}")
            print(f"Manipulated Task: {args.task}")
            print("")
            print(f"Original output:    {raw_output}")
            print(f"Manipulated Output: {manipulated_raw_output}")
        i += 1

        if executed_action is not None:
            env_action = executed_action
            predicted_terminated = bool(executed_action["terminate_episode"][0] > 0)
        else:
            env_action = manipulated_action
            predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.advance_to_next_subtask()

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate(
                [
                    env_action["world_vector"],
                    env_action["rot_axangle"],
                    env_action["gripper"],
                ]
            ),
        )
        
        success = "success" if done else "failure"
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            #print(task_description)
            task_description = new_task_description
            #print(task_description)
        is_final_subtask = env.is_final_subtask()

        if verbose:
            print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # save video
    if verbose:
        print("Saving video")
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    now = datetime.now()
    save_file_name = now.strftime("%d%m%Y_%H%M%S")
    video_name_original = save_file_name + "_original_" + video_name + ".mp4"
    video_name_manipulated = save_file_name + video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    rpy = quat2euler(robot_init_quat)
    r, p, y = rpy
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name_manipulated}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, annotated_frames_2, fps=5)

    if not quiet_episodes:
        ok = success == "success"
        try:
            rel_video = os.path.relpath(
                os.path.abspath(video_path), start=os.path.abspath(logging_dir)
            )
        except ValueError:
            rel_video = video_path
        print(
            f"[rollout {rollout_index}] done | success={ok} | env_steps={timestep} | "
            f"task={_truncate_task(task_description, max_len=80)}"
        )
        print(f"  video: {rel_video}")
    #write_video(video_path, annotated_frames, fps=5)
    
    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    # save logs
    args.task = task_description
    episode_context = {
        "rollout_index": rollout_index,
        "success": success == "success",
        "env_name": env_name,
        "scene_name": scene_name,
        "experiment": experiment,
        "robot": robot_name,
        "control_mode": control_mode,
        "episode_stats": _json_safe(dict(episode_stats)),
        "robot_init_xy": [float(robot_init_x), float(robot_init_y)],
        "robot_init_rot_rpy": [float(rpy[0]), float(rpy[1]), float(rpy[2])],
        "obj_variation_mode": obj_variation_mode,
        "obj_episode_id": obj_episode_id,
        "obj_init_x": float(obj_init_x) if obj_init_x is not None else None,
        "obj_init_y": float(obj_init_y) if obj_init_y is not None else None,
        "additional_env_build_kwargs": _json_safe(dict(additional_env_build_kwargs)),
        "additional_env_save_tags": additional_env_save_tags,
        "ckpt_path": ckpt_path,
        "ckpt_basename": ckpt_path_basename,
        "policy_model": policy_model,
        "counterfactual_perturb_fn": counterfactual_perturb_fn,
        "counterfactual_max_new_tokens": counterfactual_max_new_tokens,
        "logging_dir": logging_dir,
        "max_episode_steps": max_episode_steps,
        "control_freq": control_freq,
        "sim_freq": sim_freq,
        "obs_camera_name": obs_camera_name,
        "rgb_overlay_basename": (
            os.path.splitext(os.path.basename(rgb_overlay_path))[0]
            if rgb_overlay_path
            else None
        ),
        "video_manipulated_basename": os.path.basename(video_path),
        "video_original_basename": os.path.basename(video_name_original),
        "episode_wall_time_s": round(time.time() - episode_t0, 3),
    }
    if hasattr(model, "last_counterfactual_error"):
        episode_context["last_counterfactual_error"] = model.last_counterfactual_error
    if hasattr(model, "last_action_under_perturbed_reasoning"):
        episode_context["counterfactual_action_under_perturbed_reasoning_available"] = (
            model.last_action_under_perturbed_reasoning is not None
        )

    log_data = write_logs(
        video_path,
        video_name_original,
        real_actions,
        manipulated_actions,
        args,
        extension_count,
        verbose=verbose,
        episode_context=episode_context,
    )

    return {
        "success": success == "success",
        "rollout_index": rollout_index,
        "task_final": task_description,
        "episode_stats": dict(episode_stats),
        "env_name": env_name,
        "scene_name": scene_name,
        "experiment": experiment,
        "video_path": video_path,
        "analysis_json_path": log_data["meta"]["analysis_json_path"],
        "summary_metrics": log_data["summary_metrics"],
        "total_steps": log_data["meta"]["total_steps"],
        "coordinate_extensions": log_data["meta"]["coordinate extensions"],
    }


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    episode_records = []
    rollout_index = 0
    quiet_episodes = getattr(args, "quiet_episodes", False)

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                    experiment=args.experiment,
                    verbose=getattr(args, "verbose", False),
                    quiet_episodes=quiet_episodes,
                    policy_model=getattr(args, "policy_model", None),
                    counterfactual_perturb_fn=getattr(
                        args, "counterfactual_perturb_fn", None
                    ),
                    counterfactual_max_new_tokens=getattr(
                        args, "counterfactual_max_new_tokens", None
                    ),
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            rollout_index += 1
                            episode_records.append(
                                run_maniskill2_eval_single_episode(
                                    rollout_index=rollout_index,
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        rollout_index += 1
                        episode_records.append(
                            run_maniskill2_eval_single_episode(
                                rollout_index=rollout_index,
                                obj_episode_id=obj_episode_id,
                                **kwargs,
                            )
                        )
                else:
                    raise NotImplementedError()

    if not getattr(args, "no_experiment_summary", False):
        summary = _build_experiment_summary(episode_records, args)
        summary_path = getattr(args, "experiment_summary_path", None) or os.path.join(
            args.logging_dir, "experiment_summary.json"
        )
        _write_experiment_summary_json(
            summary_path,
            summary,
            verbose=getattr(args, "verbose", False),
        )
        ag = summary["aggregate"]
        mean_e = ag["mean_mean_euclidean_error_xyz_first_vs_second_pass"]
        mean_e_str = f"{mean_e:.6g}" if mean_e is not None else "n/a"
        print(
            f"experiment_summary: n_rollouts={ag['n_rollouts']} success_rate={ag['success_rate']:.4f} "
            f"mean_euclidean_first_vs_second_pass={mean_e_str} -> {summary_path}"
        )

    return episode_records