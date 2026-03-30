import os
os.environ["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"

import numpy as np

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator



try:
    from simpler_env.policies.octo.octo_model import OctoInference
except ImportError as e:
    print("Octo is not correctly imported.")
    print(e)


try:
    from simpler_env.policies.molmoact.molmoact_model import MolmoActInference
    from simpler_env.policies.molmoact.molmoact_model_vllm import MolmoActInferenceVLLM
except ImportError as e:
    print("MolmoAct is not correctly imported.")
    print(e)


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    # prevent a single jax process from taking up all the GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    
    # CRITICAL FIX: For molmoact-vllm models, we need to initialize PyTorch/vLLM BEFORE TensorFlow
    # Otherwise TensorFlow fails to register GPU devices and leaves CUDA in a bad state
    if "molmoact" in args.policy_model and "vllm" in args.policy_model:
        # Initialize PyTorch CUDA first
        import torch
        if torch.cuda.is_available():
            print(f"Initializing PyTorch CUDA first: {torch.cuda.device_count()} device(s) available")
            _ = torch.cuda.current_device()
    
    # Now import TensorFlow
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        # prevent a single tf process from taking up all the GPU memory
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
        )

    # policy model creation; update this if you are using a new policy model
    if args.policy_model == "rt1":
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        assert args.ckpt_path is not None
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif "octo" in args.policy_model:
        from simpler_env.policies.octo.octo_server_model import OctoServerInference
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif "molmoact" in args.policy_model:
        if args.additional_env_build_kwargs is None:
            args.additional_env_build_kwargs = {}
        args.additional_env_build_kwargs["renderer_kwargs"] = {"offscreen_only": True}

        if "custom" in args.policy_model:
            import sys
            # Directory that contains the ``Models`` package (sibling layout:
            #   <root>/Models/...  and  <root>/simpler_molmo/SimplerEnv/...
            # ). When you vendor only SimplerEnv under your thesis repo, set e.g.
            #   export MOLMO_ACT_CUSTOM_ROOT=/path/to/parent_of_Models
            custom_root = os.environ.get("MOLMO_ACT_CUSTOM_ROOT")
            if custom_root:
                parent_dir = os.path.abspath(custom_root)
            else:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(
                    os.path.dirname(os.path.dirname(current_dir))
                )
            if parent_dir not in sys.path:
                sys.path.append(parent_dir)
            from Models.MolmoAct.molmo_act_manipulated import MolmoActManipulated
            model = MolmoActManipulated()
        
        # This ensures env.make(..., renderer_kwargs={"offscreen_only": True}) is called internally
        elif "hf" in args.policy_model or "vllm" not in args.policy_model:
            model = MolmoActInference(
                saved_model_path=args.ckpt_path,
                policy_setup=args.policy_setup,
                counterfactual_perturb_fn_str=getattr(args, "counterfactual_perturb_fn", None),
                counterfactual_max_new_tokens=getattr(args, "counterfactual_max_new_tokens", 256),
                verbose=getattr(args, "verbose", False),
            )
        elif "vllm" in args.policy_model:
            model = MolmoActInferenceVLLM(
                saved_model_path=args.ckpt_path,
                policy_setup=args.policy_setup,
                counterfactual_perturb_fn_str=getattr(args, "counterfactual_perturb_fn", None),
                counterfactual_max_new_tokens=getattr(args, "counterfactual_max_new_tokens", 256),
                verbose=getattr(args, "verbose", False),
            )
    else:
        raise NotImplementedError()

    # run real-to-sim evaluation (returns list of per-episode record dicts)
    episode_records = maniskill2_evaluator(model, args)
    success_arr = [bool(e["success"]) for e in episode_records]
    print(args)
    print(" " * 10, "Average success", float(np.mean(success_arr)) if success_arr else 0.0)
