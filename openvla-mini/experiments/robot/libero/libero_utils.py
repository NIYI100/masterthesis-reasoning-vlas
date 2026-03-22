"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import tempfile

import imageio
import numpy as np
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv
from PIL import Image
from robosuite.utils.errors import RandomizationError

import libero.libero.envs.bddl_utils as BDDLUtils

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)

def get_libero_env(
    task,
    model_family,
    resolution=256,
    enable_segmentation_metrics=False,
    bddl_file_override=None,
    hard_reset=True,
    placement_seed=0,
):
    """Initializes and returns the LIBERO environment, along with the task description.

    When using in-memory placement perturbation (``apply_perturbation_to_env``), pass
    ``hard_reset=False``. Otherwise each ``env.reset()`` rebuilds the model and recreates
    samplers from the BDDL, discarding expanded ``x_ranges`` / ``y_ranges``.
    """
    task_description = task.language
    if bddl_file_override is not None:
        task_bddl_file = bddl_file_override
    else:
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "hard_reset": hard_reset,
    }
    last_exc = None
    for _env_attempt in range(10):
        try:
            if enable_segmentation_metrics:
                env_args["camera_segmentations"] = "instance"
                env = SegmentationRenderEnv(**env_args)
            else:
                env = OffScreenRenderEnv(**env_args)
            last_exc = None
            break
        except RandomizationError:
            last_exc = RandomizationError("Cannot place all objects during env init")
            continue
    if last_exc is not None:
        raise last_exc
    # When randomizing placements, vary seed by task (caller passes placement_seed); hardcoded 0
    # makes the *first* reset of every new env identical across tasks (same np.random state).
    env.seed(placement_seed)
    return env, task_description


def _get_segmentation_key(obs, camera_name="agentview"):
    camera_name = str(camera_name).lower()
    for key in obs.keys():
        key_l = key.lower()
        if "segmentation" in key_l and camera_name in key_l:
            return key
    return None


def _to_segmentation_2d(seg):
    seg = np.asarray(seg)
    if seg.ndim == 2:
        return seg
    if seg.ndim == 3:
        # Robosuite can return HxWx1 or HxWx3 for segmentation.
        return seg[..., 0]
    return None


def _bbox_from_mask(mask_2d):
    ys, xs = np.where(mask_2d)
    if len(xs) == 0:
        return None
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return [[x_min, y_min], [x_max, y_max]]


def extract_gt_bboxes_from_obs(env, obs, camera_name="agentview", flip_vertical=True):
    """
    Extract per-instance GT bboxes from segmentation observations.
    Returns a dict: {instance_name: [[x1, y1], [x2, y2]]}
    """
    seg_key = _get_segmentation_key(obs, camera_name=camera_name)
    if seg_key is None:
        return {}

    seg = _to_segmentation_2d(obs[seg_key])
    if seg is None:
        return {}
    seg = np.asarray(seg)
    if flip_vertical:
        # Match image orientation used by get_libero_image().
        seg = np.flipud(seg)

    if not hasattr(env, "instance_to_id"):
        return {}

    gt_bboxes = {}
    for instance_name, instance_id in env.instance_to_id.items():
        bbox = _bbox_from_mask(seg == instance_id)
        if bbox is not None:
            gt_bboxes[instance_name] = bbox
    return gt_bboxes


def project_world_point_to_image(env, world_xyz, camera_name="agentview", image_height=224, image_width=224):
    """
    Project a 3D world point to image coordinates for a named MuJoCo camera.
    Returns [x, y] in pixel space or None when projection is invalid.
    """
    try:
        cam_id = env.sim.model.camera_name2id(camera_name)
        cam_pos = np.array(env.sim.data.cam_xpos[cam_id])
        cam_rot = np.array(env.sim.data.cam_xmat[cam_id]).reshape(3, 3)
        rel_world = np.asarray(world_xyz) - cam_pos
        rel_cam = cam_rot.T @ rel_world
        # MuJoCo camera convention looks down negative z.
        if rel_cam[2] >= 0:
            return None

        fovy_rad = np.deg2rad(float(env.sim.model.cam_fovy[cam_id]))
        fy = 0.5 * float(image_height) / np.tan(0.5 * fovy_rad)
        fx = fy * (float(image_width) / float(image_height))
        cx = 0.5 * float(image_width)
        cy = 0.5 * float(image_height)

        x_pix = fx * (rel_cam[0] / -rel_cam[2]) + cx
        y_pix = fy * (rel_cam[1] / -rel_cam[2]) + cy
        x_pix = int(np.clip(np.round(x_pix), 0, image_width - 1))
        y_pix = int(np.clip(np.round(y_pix), 0, image_height - 1))
        return [x_pix, y_pix]
    except Exception:
        return None


def extract_gt_gripper_pixel_from_obs(env, obs, camera_name="agentview", flip_vertical=True):
    """
    Estimate GT gripper center pixel from `robot0_eef_pos` via camera projection.
    Returns [x, y] or None when unavailable.
    """
    if "robot0_eef_pos" not in obs:
        return None

    image_key = f"{camera_name}_image"
    if image_key not in obs:
        return None

    height, width = obs[image_key].shape[:2]
    px = project_world_point_to_image(
        env,
        obs["robot0_eef_pos"],
        camera_name=camera_name,
        image_height=height,
        image_width=width,
    )
    if px is None:
        return None
    if flip_vertical:
        px[1] = int(height - 1 - px[1])
    return px


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size, key="agentview_image"):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs[key]
    img = np.flipud(img)
    # img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = Image.fromarray(img)
    img = img.resize(resize_size, Image.Resampling.LANCZOS)  # resize to size seen at train time
    img = img.convert("RGB")
    return np.array(img)


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, rollout_dir_name=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    if rollout_dir_name is not None:
        rollout_dir = os.path.join(rollout_dir, rollout_dir_name)
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/episode={idx}--{DATE_TIME}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# ---------------------------------------------------------------------------
#  Perturbation & Distractor utilities
# ---------------------------------------------------------------------------

# Small manipulable objects suitable as distractors.
DISTRACTOR_OBJECT_CATEGORIES = [
    "akita_black_bowl", "white_bowl", "plate", "glazed_rim_porcelain_ramekin",
    "alphabet_soup", "bbq_sauce", "butter", "cherries", "chocolate_pudding",
    "cookies", "corn", "cream_cheese", "ketchup", "macaroni_and_cheese",
    "mayo", "milk", "orange_juice", "popcorn", "salad_dressing",
    "new_salad_dressing", "tomato_sauce",
    "moka_pot", "red_coffee_mug", "porcelain_mug", "white_yellow_mug",
    "wine_bottle", "black_book", "yellow_book",
]

LARGE_OBJECT_CATEGORIES = {
    "microwave", "wooden_cabinet", "white_cabinet", "short_cabinet",
    "short_fridge", "flat_stove", "slide_cabinet", "window",
    "faucet", "basin_faucet", "rack", "basket", "wooden_tray",
    "white_storage_box", "wooden_shelf", "wooden_two_layer_shelf",
    "wine_rack", "bowl_drainer", "chefmate_8_frypan", "desk_caddy",
    "dining_set_group",
}


def _find_section_bounds(text, section_keyword):
    """Return (start, end) character indices for a top-level BDDL section.

    *end* points to the closing ')' of the section.
    """
    marker = f"(:{section_keyword}"
    start = text.find(marker)
    if start == -1:
        return -1, -1
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return start, i
    return start, -1


def _candidate_asset_paths_for_category(category):
    base_assets = get_libero_path("assets")
    return [
        os.path.join(base_assets, "stable_hope_objects", category, f"{category}.xml"),
        os.path.join(base_assets, "stable_scanned_objects", category, f"{category}.xml"),
        os.path.join(base_assets, "turbosquid_objects", category, f"{category}.xml"),
    ]


def create_bddl_with_distractors(original_bddl_path, rng, num_distractors=None):
    """Create a temporary BDDL file that includes 1-2 distractor objects.

    Distractor objects are chosen from the LIBERO object suite, excluding
    objects already present in the scene and large objects (e.g. microwave).
    They are assigned random placement regions on the workspace table so the
    placement sampler will find collision-free positions at reset time.

    Args:
        original_bddl_path: Path to the original task BDDL file.
        rng: ``np.random.RandomState`` for reproducibility.
        num_distractors: How many distractors to add (sampled from {1, 2} if ``None``).

    Returns:
        Path to a temporary BDDL file (caller should delete when done).
    """
    with open(original_bddl_path) as f:
        content = f.read()

    parsed = BDDLUtils.robosuite_parse_problem(original_bddl_path)

    existing_categories = set()
    for cat in parsed["objects"]:
        existing_categories.add(cat.lower())
    for cat in parsed["fixtures"]:
        existing_categories.add(cat.lower())

    existing_instances = set()
    for instances in parsed["objects"].values():
        for inst in instances:
            existing_instances.add(inst)
    for instances in parsed["fixtures"].values():
        for inst in instances:
            existing_instances.add(inst)

    available = [
        c for c in DISTRACTOR_OBJECT_CATEGORIES
        if c not in existing_categories
        and c not in LARGE_OBJECT_CATEGORIES
        and any(os.path.exists(p) for p in _candidate_asset_paths_for_category(c))
    ]

    if not available:
        print("[distractors] No suitable distractor objects available for this task, skipping.")
        return original_bddl_path

    if num_distractors is None:
        num_distractors = int(rng.randint(1, 3))  # 1 or 2
    num_distractors = min(num_distractors, len(available))
    selected = list(rng.choice(available, size=num_distractors, replace=False))

    workspace_name = None
    for cat in parsed["fixtures"]:
        for inst in parsed["fixtures"][cat]:
            if any(t in inst for t in ("table", "floor")):
                workspace_name = inst
                break
        if workspace_name:
            break

    if workspace_name is None:
        print("[distractors] Could not determine workspace table, skipping.")
        return original_bddl_path

    region_parts, object_parts, init_parts = [], [], []
    for cat_name in selected:
        inst_name = f"distractor_{cat_name}_1"
        region_short = f"distractor_{cat_name}_init_region"
        full_region = f"{workspace_name}_{region_short}"

        region_parts.append(
            f"      ({region_short}\n"
            f"          (:target {workspace_name})\n"
            f"          (:ranges (\n"
            f"              (-0.2500 -0.3500 0.2500 0.3500)\n"
            f"            )\n"
            f"          )\n"
            f"          (:yaw_rotation (\n"
            f"              (0.0 0.0)\n"
            f"            )\n"
            f"          )\n"
            f"      )\n"
        )
        object_parts.append(f"    {inst_name} - {cat_name}\n")
        init_parts.append(f"    (On {inst_name} {full_region})\n")

    # Insert into BDDL content (order matters – work from end to start so
    # earlier insertions don't shift later indices).
    insertions = []  # (position, text)

    _, regions_end = _find_section_bounds(content, "regions")
    if regions_end > 0:
        insertions.append((regions_end, "\n" + "".join(region_parts)))

    _, objects_end = _find_section_bounds(content, "objects")
    if objects_end > 0:
        insertions.append((objects_end, "\n" + "".join(object_parts)))

    _, init_end = _find_section_bounds(content, "init")
    if init_end > 0:
        insertions.append((init_end, "\n" + "".join(init_parts)))

    for pos, text in sorted(insertions, key=lambda x: x[0], reverse=True):
        content = content[:pos] + text + content[pos:]

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".bddl", delete=False)
    tmp.write(content)
    tmp.close()

    print(f"[distractors] Created BDDL with {num_distractors} distractor(s) {selected}: {tmp.name}")
    return tmp.name


def apply_perturbation_to_env(env, factor=1.2):
    """Expand placement regions for task-relevant objects by *factor* (width & height).

    Only samplers in ``placement_initializer`` whose ``mujoco_objects`` intersect
    ``obj_of_interest`` (from the BDDL) are scaled about each rectangle's center.

    Requires ``hard_reset=False`` on the underlying env so ``reset()`` does not
    reload the model and recreate samplers (see ``get_libero_env``).

    Returns a dict of original regions (for debugging / optional checks).
    """
    inner_env = env.env
    obj_of_interest = set(inner_env.obj_of_interest)
    original_regions = {}

    for sampler_name, sampler in inner_env.placement_initializer.samplers.items():
        obj_names = [obj.name for obj in sampler.mujoco_objects]
        if not any(name in obj_of_interest for name in obj_names):
            continue

        orig_x = [[r[0], r[1]] for r in sampler.x_ranges]
        orig_y = [[r[0], r[1]] for r in sampler.y_ranges]
        original_regions[sampler_name] = {
            "original_x_ranges": orig_x,
            "original_y_ranges": orig_y,
            "obj_names": obj_names,
            "reference_pos": list(sampler.reference_pos),
        }

        for i in range(len(sampler.x_ranges)):
            cx = (sampler.x_ranges[i][0] + sampler.x_ranges[i][1]) / 2.0
            hx = (sampler.x_ranges[i][1] - sampler.x_ranges[i][0]) / 2.0
            sampler.x_ranges[i] = [cx - hx * factor, cx + hx * factor]

            cy = (sampler.y_ranges[i][0] + sampler.y_ranges[i][1]) / 2.0
            hy = (sampler.y_ranges[i][1] - sampler.y_ranges[i][0]) / 2.0
            sampler.y_ranges[i] = [cy - hy * factor, cy + hy * factor]

    print(f"[perturbation] Expanded regions (×{factor}) for samplers: {list(original_regions.keys())}")
    return original_regions


def safe_env_reset(env, max_retries=100):
    """Call ``env.reset()`` with retry logic for ``RandomizationError``."""
    for _ in range(max_retries):
        try:
            return env.reset()
        except RandomizationError:
            continue
    raise RuntimeError(f"Could not reset environment after {max_retries} retries")


def reset_with_perturbation(env, original_regions=None, max_retries=100):
    """Reset with expanded placement regions already applied (see ``apply_perturbation_to_env``).

    Sampling + collision checks are handled by LIBERO/robosuite placement samplers; we only
    retry on ``RandomizationError``. ``original_regions`` is optional (unused); kept for
    call-site compatibility.
    """
    return safe_env_reset(env, max_retries=max_retries)
