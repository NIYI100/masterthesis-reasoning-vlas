import argparse

import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat

from simpler_env.utils.io import DictAction


def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))


def get_args():
    # parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-model",
        type=str,
        default="rt1",
        help="Policy model type; e.g., 'rt1', 'octo-base', 'octo-small'",
    )
    parser.add_argument(
        "--policy-setup",
        type=str,
        default="google_robot",
        help="Policy model setup; e.g., 'google_robot', 'widowx_bridge'",
    )
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument(
        "--additional-env-save-tags",
        type=str,
        default=None,
        help="Additional tags to save the environment eval results",
    )
    parser.add_argument("--scene-name", type=str, default="google_pick_coke_can_1_v4")
    parser.add_argument("--enable-raytracing", action="store_true")
    parser.add_argument("--robot", type=str, default="google_robot_static")
    parser.add_argument(
        "--obs-camera-name",
        type=str,
        default=None,
        help="Obtain image observation from this camera for policy input. None = default",
    )
    parser.add_argument("--action-scale", type=float, default=1.0)

    parser.add_argument("--control-freq", type=int, default=3)
    parser.add_argument("--sim-freq", type=int, default=513)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--rgb-overlay-path", type=str, default=None)
    parser.add_argument(
        "--robot-init-x-range",
        type=float,
        nargs=3,
        default=[0.35, 0.35, 1],
        help="[xmin, xmax, num]",
    )
    parser.add_argument(
        "--robot-init-y-range",
        type=float,
        nargs=3,
        default=[0.20, 0.20, 1],
        help="[ymin, ymax, num]",
    )
    parser.add_argument(
        "--robot-init-rot-quat-center",
        type=float,
        nargs=4,
        default=[1, 0, 0, 0],
        help="[x, y, z, w]",
    )
    parser.add_argument(
        "--robot-init-rot-rpy-range",
        type=float,
        nargs=9,
        default=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        help="[rmin, rmax, rnum, pmin, pmax, pnum, ymin, ymax, ynum]",
    )
    parser.add_argument(
        "--obj-variation-mode",
        type=str,
        default="xy",
        choices=["xy", "episode"],
        help="Whether to vary the xy position of a single object, or to vary predetermined episodes",
    )
    parser.add_argument("--obj-episode-range", type=int, nargs=2, default=[0, 60], help="[start, end]")
    parser.add_argument(
        "--obj-init-x-range",
        type=float,
        nargs=3,
        default=[-0.35, -0.12, 5],
        help="[xmin, xmax, num]",
    )
    parser.add_argument(
        "--obj-init-y-range",
        type=float,
        nargs=3,
        default=[-0.02, 0.42, 5],
        help="[ymin, ymax, num]",
    )

    parser.add_argument(
        "--additional-env-build-kwargs",
        nargs="+",
        action=DictAction,
        help="Additional env build kwargs in xxx=yyy format. If the value "
        'is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--logging-dir", type=str, default="./results")
    parser.add_argument("--tf-memory-limit", type=int, default=3072, help="Tensorflow memory limit")
    parser.add_argument("--octo-init-rng", type=int, default=0, help="Octo init rng seed")
    parser.add_argument("--experiment", type=str, default="coke_can", help="The experiment name to do a specific experiment for the task")
    parser.add_argument(
        "--counterfactual-perturb-fn",
        type=str,
        default=None,
        help=(
            "HF policy_model=molmoact (not vllm, not custom): after the first full decode, run a second "
            "decode with a perturbed reasoning prefix before the action anchor. "
            "When parsing succeeds, the counterfactual 7D action drives the robot; the rollout video shows the "
            "trajectory from perturbed reasoning; the vanilla first-pass world XYZ is logged as real_actions and "
            "the executed XYZ as manipulated_actions (delta = perturbation effect). "
            "Built-ins: identity; append_suffix:TEXT; shift_xy_pairs:DX,DY (see counterfactual_reasoning_action). "
            "Omit or use 'none' to disable. vLLM backend ignores this (warning only)."
        ),
    )
    parser.add_argument(
        "--counterfactual-max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens when completing the action tail in the counterfactual / manipulated second pass.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step eval details (task text, model raw outputs, timestep info). Default is quiet.",
    )
    parser.add_argument(
        "--quiet-episodes",
        action="store_true",
        help="Disable per-rollout summary lines (task, rollout index, success, video path).",
    )
    parser.add_argument(
        "--experiment-summary-path",
        type=str,
        default=None,
        help="Where to write experiment_summary.json (default: <logging-dir>/experiment_summary.json).",
    )
    parser.add_argument(
        "--no-experiment-summary",
        action="store_true",
        help="Do not write experiment_summary.json or print the aggregate line.",
    )

    args = parser.parse_args()

    # env args: robot pose
    args.robot_init_xs = parse_range_tuple(args.robot_init_x_range)
    args.robot_init_ys = parse_range_tuple(args.robot_init_y_range)
    args.robot_init_quats = []
    for r in parse_range_tuple(args.robot_init_rot_rpy_range[:3]):
        for p in parse_range_tuple(args.robot_init_rot_rpy_range[3:6]):
            for y in parse_range_tuple(args.robot_init_rot_rpy_range[6:]):
                args.robot_init_quats.append((Pose(q=euler2quat(r, p, y)) * Pose(q=args.robot_init_rot_quat_center)).q)
    # env args: object position
    if args.obj_variation_mode == "xy":
        args.obj_init_xs = parse_range_tuple(args.obj_init_x_range)
        args.obj_init_ys = parse_range_tuple(args.obj_init_y_range)
    # update logging info (args.additional_env_save_tags) if using a different camera from default
    if args.obs_camera_name is not None:
        if args.additional_env_save_tags is None:
            args.additional_env_save_tags = f"obs_camera_{args.obs_camera_name}"
        else:
            args.additional_env_save_tags = args.additional_env_save_tags + f"_obs_camera_{args.obs_camera_name}"

    return args
