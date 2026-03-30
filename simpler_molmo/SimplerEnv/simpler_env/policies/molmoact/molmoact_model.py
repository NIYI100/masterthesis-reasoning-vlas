from typing import Optional, Sequence
import os
import ast
import json
import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from transforms3d.euler import euler2axangle

from simpler_env.policies.molmoact.counterfactual_reasoning_action import (
    get_reasoning_perturb_fn,
    regenerate_action_under_perturbed_reasoning,
    repair_missing_action_open_bracket,
)
# Appended to evaluator-provided partial reasoning strings before regenerating the action tail.
_REASONING_TO_ACTION_GLUE = " Based on this information, the action that the robot should take is "


class MolmoActInference:
    def __init__(
        self,
        saved_model_path: str = "",
        unnorm_key: Optional[str] = None,
        policy_setup: str = "google_robot",
        image_size: list[int] = [256, 256],
        action_scale: float = 1.0,
        counterfactual_perturb_fn_str: Optional[str] = None,
        counterfactual_max_new_tokens: int = 256,
        verbose: bool = False,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        self.processor = AutoProcessor.from_pretrained(
            saved_model_path,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map="auto",
            padding_side="left",
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            saved_model_path,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map="auto",
        )


        self.image_size = image_size
        self.action_scale = action_scale


        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None


        stats_path = './simpler_env/policies/molmoact/dataset_statistics.json'
        

        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Dataset statistics file not found at {stats_path}")
        with open(stats_path, 'r') as f:
            self.dataset_stats = json.load(f)

        self._counterfactual_perturb_fn = get_reasoning_perturb_fn(counterfactual_perturb_fn_str)
        self._counterfactual_max_new_tokens = counterfactual_max_new_tokens
        self.last_action_under_perturbed_reasoning: Optional[list] = None
        self.last_counterfactual_error: Optional[str] = None
        # Set on first-pass step when counterfactual regen succeeds: vanilla XYZ (scaled) for delta logs.
        self.last_baseline_world_vector: Optional[np.ndarray] = None
        # Full assistant decode after perturbed-prefix regen (for trajectory overlay + raw_output).
        self.last_counterfactual_assistant_text: Optional[str] = None
        self._verbose = verbose

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    @staticmethod
    def _unpack_step_args(first, *rest, **kwargs):
        """`maniskill2_evaluator` calls `step(args_ns)`; also supports `step(image, task_description=...)`."""
        if hasattr(first, "image"):
            ea = first
            return (
                ea.image,
                getattr(ea, "task", None),
                bool(getattr(ea, "update_state", False)),
                getattr(ea, "reasoning", None),
            )
        task_description = kwargs.get("task_description")
        if task_description is None and rest:
            task_description = rest[0]
        return first, task_description, False, None

    def _build_user_prompt(self, language_instruction: str) -> str:
        return (
            f"The task is {language_instruction}. "
            "What is the action that the robot should take. "
            f"To figure out the action that the robot should take to {language_instruction}, "
            "let's think through it step by step. "
            "First, what is the depth map for this image? "
            "Second, what is the trajectory of the end effector? "
            "Based on the depth map of the image and the trajectory of the end effector, "
            "what is the action that the robot should take?"
        )

    def _hf_generate_from_chat(
        self, img: Image.Image, user_prompt: str, assistant_prefix: Optional[str], max_new_tokens: int
    ) -> str:
        if assistant_prefix is None:
            messages = [
                {"role": "user", "content": [dict(type="text", text=user_prompt)]},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            messages = [
                {"role": "user", "content": [dict(type="text", text=user_prompt)]},
                {"role": "assistant", "content": [dict(type="text", text=assistant_prefix)]},
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        inputs = self.processor(
            images=[img],
            text=text,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        gen_tokens = generated_ids[:, inputs["input_ids"].size(1) :]
        return self.processor.batch_decode(
            gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def get_reasoning(self, raw_output: str):
        """Used by `maniskill2_evaluator` after the first `step` (raw_output = assistant text)."""
        depth = self.model.parse_depth(raw_output)
        trace = self.model.parse_trace(raw_output)
        return depth, trace, None

    def scale_pt(self, pt, w, h):
        """
        Convert a point whose coordinates are in 0–255 space
        to image-pixel space (0‥w-1, 0‥h-1).
        """
        x, y = pt
        return (int(round(x / 255.0 * (w - 1))),
                int(round(y / 255.0 * (h - 1))))

    def step(self, first, *rest, **kwargs):
        """
        Supports `model.step(env_args)` (SimplerEnv evaluator) and legacy `step(image, task_description=...)`.

        Returns:
            generated_text, raw_action, action, annotated_image
        """
        image, task_override, update_state, reasoning_override = self._unpack_step_args(
            first, *rest, **kwargs
        )
        if task_override is not None and task_override != self.task_description:
            self.reset(task_override)

        assert image.dtype == np.uint8
        orig_h, orig_w = image.shape[:2]
        image = self._resize_image(image)
        new_h, new_w = image.shape[:2]
        img = Image.fromarray(image)
        language_instruction = self.task_description
        assert language_instruction is not None, "Call model.reset(task_description) before step."
        user_prompt = self._build_user_prompt(language_instruction)

        self.last_action_under_perturbed_reasoning = None
        self.last_counterfactual_error = None
        self.last_baseline_world_vector = None
        self.last_counterfactual_assistant_text = None

        if update_state and reasoning_override:
            assistant_prefix = reasoning_override.rstrip() + _REASONING_TO_ACTION_GLUE
            suffix = self._hf_generate_from_chat(
                img, user_prompt, assistant_prefix, max_new_tokens=self._counterfactual_max_new_tokens
            )
            generated_text = assistant_prefix + suffix
        else:
            generated_text = self._hf_generate_from_chat(
                img, user_prompt, assistant_prefix=None, max_new_tokens=256
            )
            if self._counterfactual_perturb_fn is not None:
                acf, err, cf_assistant = regenerate_action_under_perturbed_reasoning(
                    model=self.model,
                    processor=self.processor,
                    images=[img],
                    user_prompt=user_prompt,
                    original_generated_assistant_text=generated_text,
                    perturb_reasoning_fn=self._counterfactual_perturb_fn,
                    unnorm_key=self.unnorm_key,
                    max_new_tokens=self._counterfactual_max_new_tokens,
                )
                self.last_action_under_perturbed_reasoning = acf
                self.last_counterfactual_error = err
                if err is None and cf_assistant:
                    self.last_counterfactual_assistant_text = cf_assistant
                if self._verbose and acf is not None:
                    print(f"counterfactual action (analysis): {acf}")
                if self._verbose and err:
                    print(f"counterfactual note: {err}")

        generated_text = repair_missing_action_open_bracket(generated_text)

        depth = self.model.parse_depth(generated_text)
        if self._verbose:
            print(f"Depth: {depth}")

        trace = self.model.parse_trace(generated_text)
        if self._verbose:
            print(f"Trace: {trace}")

        action = self.model.parse_action(generated_text, unnorm_key=self.unnorm_key)
        if self._verbose:
            print(f"Action: {action}")

        unnormalized_action = action[0] if isinstance(action, list) and len(action) == 1 else action
        unnormalized_action = np.array(unnormalized_action, dtype=np.float64)

        # Use counterfactual 7D action for control; keep vanilla XYZ in last_baseline_world_vector for logging.
        if (
            (not update_state)
            and self._counterfactual_perturb_fn is not None
            and self.last_action_under_perturbed_reasoning is not None
            and len(self.last_action_under_perturbed_reasoning) > 0
            and self.last_counterfactual_error is None
        ):
            self.last_baseline_world_vector = (
                unnormalized_action[:3].astype(np.float64) * self.action_scale
            ).copy()
            unnormalized_action = np.array(
                self.last_action_under_perturbed_reasoning[0], dtype=np.float64
            )

        text_for_trajectory = (
            self.last_counterfactual_assistant_text
            if self.last_counterfactual_assistant_text is not None
            else generated_text
        )
        assistant_text_out = text_for_trajectory

        annotated_image = image.copy()

        if "The trajectory of the end effector is" in text_for_trajectory:
            try:
                traj_part = text_for_trajectory.split("The trajectory of the end effector is")[-1]
                traj_part = traj_part.split("Based on")[0].strip()
                traj_str = traj_part.rstrip(".").strip()
                trajectory = ast.literal_eval(traj_str)

                for i in range(len(trajectory) - 1):
                    pt1 = tuple(map(int, trajectory[i]))
                    pt2 = tuple(map(int, trajectory[i + 1]))
                    pt1 = self.scale_pt(pt1, new_w, new_h)
                    pt2 = self.scale_pt(pt2, new_w, new_h)
                    cv.line(
                        annotated_image,
                        pt1,
                        pt2,
                        (0, 255, 255),
                        thickness=2,
                        lineType=cv.LINE_AA,
                    )

            except Exception as e:
                if self._verbose:
                    print("Failed to parse trajectory:", e)
        elif self._verbose:
            print("No trajectory found in generated text.")

        raw_action = {
            "world_vector": unnormalized_action[:3],
            "rotation_delta": unnormalized_action[3:6],
            "open_gripper": unnormalized_action[6:7],
        }
        annotated_image = cv.resize(annotated_image, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])

        return assistant_text_out, raw_action, action, annotated_image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image
    
   
        
    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        # images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]
    
        img_strip = np.concatenate(np.array(images[::3]), axis=1)
    
        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])
    
        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")
    
        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)












