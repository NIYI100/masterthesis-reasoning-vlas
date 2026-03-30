import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.stats import kruskal


import argparse
import glob
from pathlib import Path

import numpy as np
from scipy.stats import kruskal
def get_dir_stats(
    dir_name: str,
    extra_pattern_require,
    succ_fail_pattern,
):
    if dir_name[-1] == "/":
        dir_name = dir_name[:-1]

    results = []
    fnames = glob.glob(dir_name + "/**/*.mp4", recursive=True)
    for fname in fnames:
        flag = True
        for pattern in extra_pattern_require:
            if pattern not in fname:
                flag = False
                break
        if not flag:
            continue
        fname = Path(fname)
        if fname.suffix != ".mp4":
            continue
        fname = fname.stem
        if succ_fail_pattern[0] in fname:
            results.append(1)
        elif succ_fail_pattern[1] in fname:
            results.append(0)

    return results


succ_fail_pattern = ["success_", "failure_"]

def safe_mean(data, path):
    """Return the mean of data if available; otherwise print a warning and return 0.0."""
    if len(data) == 0:
        print(f"WARNING: No simulation data found for {path}")
        return 0.0
    return np.mean(data)
def calc_pick_coke_can_stats(root_result_dir):

    print("***Pick coke can simulation results***")
    # Checkpoint keys used for simulation results.
    ckpt_alias_keys = [DEFAULT_CKPT_ALIAS]
    coke_can_orientation_map_dict = {
        "horizontal": "lr_switch",
        "vertical": "laid_vertically",
        "standing": "upright",
    }
    n_trials_per_ckpt_per_orientation = 25  # number of trials per checkpoint per orientation

    # Extra patterns required for sim results.
    extra_pattern_require_sim_variants = ["rgb_overlay_None"]
    extra_pattern_require_visual_matching = ["rgb_overlay_google_coke_can_real_eval_1"]

    # Variant groups
    variant_groups = {
        "base": [
            "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True",
        ],
        "background": [
            "google_pick_coke_can_1_v4_alt_background/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True",
            "google_pick_coke_can_1_v4_alt_background_2/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True",
        ],
        "lighting": [
            "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True_slightly_brighter_lighting_True",
            "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True_slightly_darker_lighting_True",
        ],
        "distractor": [
            "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanDistractorInScene-v0_{}_True",
            "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanDistractorInScene-v0_{}_True_distractor_config_more",
        ],
        "table_texture": [
            "Baked_sc1_staging_objaverse_cabinet1_h870/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True",
            "Baked_sc1_staging_objaverse_cabinet2_h870/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{}_True",
        ],
    }

    # Collect simulation results
    coke_sim_results = {orient: {ckpt: {} for ckpt in ckpt_alias_keys} for orient in coke_can_orientation_map_dict}
    for group_name, variants in variant_groups.items():
        print(f"--- Results for {group_name} variants ---")
        for orient, orient_str in coke_can_orientation_map_dict.items():
            for ckpt in ckpt_alias_keys:
                total_succ = 0
                total_trials = 0
                rates = []
                for var in variants:
                    path = var.format(orient_str)
                    stats = get_dir_stats(f"{root_result_dir}/{CKPT_MAPPING[ckpt]}/{path}", extra_pattern_require=extra_pattern_require_sim_variants, succ_fail_pattern=succ_fail_pattern)
                    succ = sum(stats)
                    trials = len(stats)
                    rate = safe_mean(stats, f"{root_result_dir}/{CKPT_MAPPING[ckpt]}/{path}")
                    rates.append(rate)
                    total_succ += succ
                    total_trials += trials
                    printable = path.split('/')[-1]
                    print(f"{group_name} - {orient} - {printable}: {succ}/{trials} (avg {rate:.2%})")
                mean_rate = np.mean(rates)
                coke_sim_results[orient][ckpt][group_name] = mean_rate
                print(f"=> {group_name} aggregate for {orient}: {total_succ}/{total_trials} (avg {mean_rate:.2%})")
        print("--------------------")

    # Aggregated over variant groups per orientation
    aggregated = {}
    for orient in coke_can_orientation_map_dict:
        aggregated[orient] = {}
        for ckpt in ckpt_alias_keys:
            group_rates = list(coke_sim_results[orient][ckpt].values())
            aggregated[orient][ckpt] = np.mean(group_rates)
    print("Aggregated sim results per orientation:", aggregated)

    # Overall sim summary
    rates = [v[next(iter(v))] for v in aggregated.values()]
    overall_rate = sum(rates) / len(rates)
    n_vars = sum(len(v) for v in variant_groups.values())
    total_trials_sim = n_vars * n_trials_per_ckpt_per_orientation * len(coke_can_orientation_map_dict)
    total_succ_sim = int(overall_rate * total_trials_sim)
    print(f"Overall sim: {total_succ_sim}/{total_trials_sim} successes (avg {overall_rate:.2%})")
    print("--------------------")

    # Visual matching
    vis_variants = []
    for ver in ["None","recolor_tabletop_visual_matching_1","recolor_tabletop_visual_matching_2","recolor_cabinet_visual_matching_1"]:
        vis_variants.append(f"google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_{{}}_True_urdf_version_{ver}")

    print("--- Visual Matching Variants ---")
    coke_vis_results = {orient: {ckpt: [] for ckpt in ckpt_alias_keys} for orient in coke_can_orientation_map_dict}
    for orient, orient_str in coke_can_orientation_map_dict.items():
        for ckpt in ckpt_alias_keys:
            total_vis_succ = 0
            total_vis_trials = 0
            rates = []
            for var in vis_variants:
                path = var.format(orient_str)
                stats = get_dir_stats(f"{root_result_dir}/{CKPT_MAPPING[ckpt]}/{path}", extra_pattern_require=extra_pattern_require_visual_matching, succ_fail_pattern=succ_fail_pattern)
                succ = sum(stats)
                trials = len(stats)
                rate = safe_mean(stats, f"{root_result_dir}/{CKPT_MAPPING[ckpt]}/{path}")
                rates.append(rate)
                total_vis_succ += succ
                total_vis_trials += trials
                printable = path.split('/')[-1]
                print(f"Vis - {orient} - {printable}: {succ}/{trials} (avg {rate:.2%})")
            mean_vis = np.mean(rates)
            coke_vis_results[orient][ckpt] = mean_vis
            print(f"=> Visual matching aggregate for {orient}: {total_vis_succ}/{total_vis_trials} (avg {mean_vis:.2%})")
        print("--------------------")

    # Overall visual summary
    all_rates = [coke_vis_results[orient][ckpt] for orient in coke_can_orientation_map_dict for ckpt in ckpt_alias_keys]
    overall_vis_rate = np.mean(all_rates)
    n_vis = len(vis_variants) * len(coke_can_orientation_map_dict)
    total_vis_trials = n_trials_per_ckpt_per_orientation * n_vis
    total_vis_succ = int(overall_vis_rate * total_vis_trials)
    print(f"Overall visual: {total_vis_succ}/{total_vis_trials} successes (avg {overall_vis_rate:.2%})")
    print("****************************")
    print()


def calc_move_near_stats(root_result_dir):
    print("***Move Near simulation results***")
    ckpt_alias_keys = [DEFAULT_CKPT_ALIAS]
    n_trials_per_ckpt = 60

    extra_pattern_require_sim_variants = ["rgb_overlay_None"]
    extra_pattern_require_visual_matching = ["rgb_overlay_google_move_near_real_eval_1"]

    # Simulation variant success container
    move_near_sim_variant_success = {ckpt: [] for ckpt in ckpt_alias_keys}

    # Variant lists
    base_variants = [
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0",
    ]
    background_variants = [
        "google_pick_coke_can_1_v4_alt_background/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0",
        "google_pick_coke_can_1_v4_alt_background_2/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0",
    ]
    lighting_variants = [
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0_slightly_brighter_lighting_True",
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0_slightly_darker_lighting_True",
    ]
    distractor_variants = [
        "google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0_no_distractor_True",
    ]
    table_texture_variants = [
        "Baked_sc1_staging_objaverse_cabinet1_h870/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0",
        "Baked_sc1_staging_objaverse_cabinet2_h870/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleInScene-v0",
    ]

    all_variants = base_variants + background_variants + lighting_variants + distractor_variants + table_texture_variants

    # Process sim variants
    for ckpt in ckpt_alias_keys:
        for variant in all_variants:
            variant_path = f"{root_result_dir}/{CKPT_MAPPING[ckpt]}/{variant}"
            stats = get_dir_stats(variant_path, extra_pattern_require=extra_pattern_require_sim_variants, succ_fail_pattern=succ_fail_pattern)
            success_count = sum(stats)
            trials = len(stats)
            avg_success = safe_mean(stats, variant_path)
            move_near_sim_variant_success[ckpt].append(avg_success)
            # print only last path segment
            printable = variant.split('/')[-1]
            print(f"Sim Variant: {printable}\n  Successes: {success_count}/{trials} (avg {avg_success:.2%})")

        # aggregate sim variants
        agg_rate = np.mean(move_near_sim_variant_success[ckpt])
        total_trials = len(all_variants) * n_trials_per_ckpt
        total_successes = int(agg_rate * total_trials)
        print("--------------------")
        print(f"Move Near simulation variant avg success: {agg_rate:.2%}")
        print(f"Total successes vs total trials (sim variants): {total_successes}/{total_trials} ({agg_rate:.2%})")
        print("--------------------")

    # Visual matching variants
    move_near_sim_visual_matching_success = {ckpt: [] for ckpt in ckpt_alias_keys}
    base_visual_matching_variants = []
    for ver in [
        "None",
        "recolor_tabletop_visual_matching_1",
        "recolor_tabletop_visual_matching_2",
        "recolor_cabinet_visual_matching_1",
    ]:
        base_visual_matching_variants.append(
            f"google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleBakedTexInScene-v0_urdf_version_{ver}_baked_except_bpb_orange"
        )

    for ckpt in ckpt_alias_keys:
        for variant in base_visual_matching_variants:
            variant_path = f"{root_result_dir}/{CKPT_MAPPING[ckpt]}/{variant}"
            stats = get_dir_stats(variant_path, extra_pattern_require=extra_pattern_require_visual_matching, succ_fail_pattern=succ_fail_pattern)
            success_count = sum(stats)
            trials = len(stats)
            avg_success = safe_mean(stats, variant_path)
            move_near_sim_visual_matching_success[ckpt].append(avg_success)
            printable = variant.split('/')[-1]
            print(f"Vis Variant: {printable}\n  Successes: {success_count}/{trials} (avg {avg_success:.2%})")

        # aggregate visual matching
        agg_vis = np.mean(move_near_sim_visual_matching_success[ckpt])
        total_vis_trials = len(base_visual_matching_variants) * n_trials_per_ckpt
        total_vis_successes = int(agg_vis * total_vis_trials)
        print("--------------------")
        print(f"Move Near visual matching avg success: {agg_vis:.2%}")
        print(f"Total successes vs total trials (visual matching): {total_vis_successes}/{total_vis_trials} ({agg_vis:.2%})")
        print("****************************")
        print()


def calc_drawer_stats(root_result_dir):
    print("***Drawer simulation results***")
    ckpt_alias_keys = [DEFAULT_CKPT_ALIAS]
    n_trials_per_ckpt_per_task = 27  # number of trials per checkpoint per task

    extra_pattern_require_sim_variants = ["rgb_overlay_None"]
    extra_pattern_require_visual_matching = ["rgb_overlay_open_drawer"]

    # Drawer tasks
    drawer_task_map_dict = {
        "open": [
            "OpenTopDrawerCustomInScene-v0",
            "OpenMiddleDrawerCustomInScene-v0",
            "OpenBottomDrawerCustomInScene-v0",
        ],
        "close": [
            "CloseTopDrawerCustomInScene-v0",
            "CloseMiddleDrawerCustomInScene-v0",
            "CloseBottomDrawerCustomInScene-v0",
        ],
    }

    # Variant directory templates
    base_variants = [
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt",
    ]
    background_variants = [
        "modern_bedroom_no_roof/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt",
        "modern_office_no_roof/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt",
    ]
    lighting_variants = [
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_light_mode_brighter",
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_light_mode_darker",
    ]
    table_texture_variants = [
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_station_name_mk_station2",
        "frl_apartment_stage_simple/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{}_shader_dir_rt_station_name_mk_station3",
    ]
    sim_variants = base_variants + background_variants + lighting_variants + table_texture_variants
    n_sim_variants = len(sim_variants)

    # Simulation variant results
    drawer_sim_variant_success = {task: {ckpt: 0 for ckpt in ckpt_alias_keys} for task in drawer_task_map_dict}

    for task_name, scenes in drawer_task_map_dict.items():
        print(f"--- Results for {task_name} variants ---")
        for ckpt in ckpt_alias_keys:
            total_succ = 0
            total_trials = 0
            rates = []
            for scene in scenes:
                for variant in sim_variants:
                    variant_formatted = variant.format(scene)
                    variant_path = f"{root_result_dir}/{CKPT_MAPPING[ckpt]}/{variant_formatted}"
                    stats = get_dir_stats(variant_path, extra_pattern_require=extra_pattern_require_sim_variants, succ_fail_pattern=succ_fail_pattern)
                    succ = sum(stats)
                    trials = len(stats)
                    rate = safe_mean(stats, variant_path)
                    rates.append(rate)
                    total_succ += succ
                    total_trials += trials
                    # print only the scene and suffix, not full path
                    printable = variant_formatted.split('/')[-1]
                    print(f"{task_name} - {scene} - {printable}: {succ}/{trials} ({rate:.2%})")
            mean_rate = np.mean(rates)
            drawer_sim_variant_success[task_name][ckpt] = mean_rate
            print(f"{task_name} aggregate: {total_succ}/{total_trials} successes (avg {mean_rate:.2%})")
        print("--------------------")

    # Overall simulation aggregate
    overall_succ = 0
    overall_trials = 0
    overall_rates = []
    for task_name, ckpt_dict in drawer_sim_variant_success.items():
        scenes = drawer_task_map_dict[task_name]
        for ckpt, rate in ckpt_dict.items():
            overall_rates.append(rate)
            overall_trials += len(scenes) * n_sim_variants * n_trials_per_ckpt_per_task
            overall_succ += int(rate * (len(scenes) * n_sim_variants * n_trials_per_ckpt_per_task))
    overall_rate = np.mean(overall_rates)
    print(f"Overall sim variants: {overall_succ}/{overall_trials} successes (avg {overall_rate:.2%})")
    print("====================")

    # Visual matching variant results
    base_visual_matching_variants = []
    for ver in ["None", "recolor_tabletop_visual_matching_1", "recolor_tabletop_visual_matching_2", "recolor_cabinet_visual_matching_1"]:
        base_visual_matching_variants.append(
            f"dummy_drawer/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/{{}}_shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_{ver}"
        )
    n_vis_variants = len(base_visual_matching_variants)

    drawer_sim_visual_matching_success = {task: {ckpt: 0 for ckpt in ckpt_alias_keys} for task in drawer_task_map_dict}

    for task_name, scenes in drawer_task_map_dict.items():
        print(f"--- Visual Matching for {task_name} ---")
        for ckpt in ckpt_alias_keys:
            total_vis_succ = 0
            total_vis_trials = 0
            vis_rates = []
            for scene in scenes:
                for variant in base_visual_matching_variants:
                    variant_formatted = variant.format(scene)
                    variant_path = f"{root_result_dir}/{CKPT_MAPPING[ckpt]}/{variant_formatted}"
                    stats = get_dir_stats(variant_path, extra_pattern_require=extra_pattern_require_visual_matching, succ_fail_pattern=succ_fail_pattern)
                    succ = sum(stats)
                    trials = len(stats)
                    rate = safe_mean(stats, variant_path)
                    vis_rates.append(rate)
                    total_vis_succ += succ
                    total_vis_trials += trials
                    printable = variant_formatted.split('/')[-1]
                    print(f"Vis {task_name} - {scene} - {printable}: {succ}/{trials} ({rate:.2%})")
            mean_vis_rate = np.mean(vis_rates)
            drawer_sim_visual_matching_success[task_name][ckpt] = mean_vis_rate
            print(f"{task_name} visual matching aggregate: {total_vis_succ}/{total_vis_trials} successes (avg {mean_vis_rate:.2%})")
        print("--------------------")

    # Overall visual matching aggregate
    overall_vis_succ = 0
    overall_vis_trials = 0
    overall_vis_rates = []
    for task_name, ckpt_dict in drawer_sim_visual_matching_success.items():
        scenes = drawer_task_map_dict[task_name]
        for rate in ckpt_dict.values():
            overall_vis_rates.append(rate)
            overall_vis_trials += len(scenes) * n_vis_variants * n_trials_per_ckpt_per_task
            overall_vis_succ += int(rate * (len(scenes) * n_vis_variants * n_trials_per_ckpt_per_task))
    overall_vis_rate = np.mean(overall_vis_rates)
    print(f"Overall visual matching: {overall_vis_succ}/{overall_vis_trials} successes (avg {overall_vis_rate:.2%})")
    print("****************************")
    print()


# Define checkpoint alias-to-directory mapping; If you use a new checkpoint, please update the dict


DEFAULT_TASK = "pick_coke_can"
# DEFAULT_TASK = "move_near"  
# DEFAULT_TASK = "drawer"

DEFAULT_CKPT_ALIAS="./results/MolmoAct-7B-D-Pretrain-0812"
DEFAULT_CKPT_ALIAS="./results/MolmoAct-7B-D-Pretrain-Fractal-0812"


CKPT_MAPPING = {
    DEFAULT_CKPT_ALIAS: DEFAULT_CKPT_ALIAS,
}

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="task name")
parser.add_argument("--log-dir-root", type=str, default="", help="log directory")

args = parser.parse_args()

if args.task == "pick_coke_can":
    calc_pick_coke_can_stats(args.log_dir_root)
elif args.task == "move_near":
    calc_move_near_stats(args.log_dir_root)
elif args.task == "drawer":
    calc_drawer_stats(args.log_dir_root)
else:
    raise ValueError(f"Unknown task: {args.task}")

exit(0)

"""
octo-base variant aggregation:
pick coke can ([horizontal, vertical, standing, avg]): default urdf [0.00, 0.00, 0.00, 0.00]; recolor_sim urdf [0.009, 0.00, 0.0267, 0.012]
move near: default urdf 0.03125; recolor_sim urdf 0.033
drawer ([open, close, avg]): default urdf [0.00, 0.021, 0.011]; recolor_sim urdf [0.00, 0.016, 0.008]
"""
