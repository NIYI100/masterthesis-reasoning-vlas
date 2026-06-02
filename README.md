# Masterthesis Reasoning VLAs

This repository is prepared for **using the OpenVLA-mini workflow** with reasoning:

- train models,
- evaluate models,
- perturb reasoning during training or evaluation.

## Project Scope

Use this folder as main entry point:

- `openvla-mini/`

Core files:

- training: `openvla-mini/vla-scripts/train.py`
- evaluation: `openvla-mini/experiments/robot/libero/run_libero_eval.py`
- reasoning perturbations: `openvla-mini/prismatic/util/reasoning_manipulation.py`



### Training command (full parameterized template)

```bash
cd openvla-mini
torchrun \
  --nnodes <NUM_NODES> \
  --nproc-per-node <PROCS_PER_NODE> \
  --rdzv_id "<RENDEZVOUS_ID>" \
  --rdzv_backend c10d \
  --rdzv_endpoint "<MASTER_ADDR:MASTER_PORT>" \
  vla-scripts/train.py \
  --config_path "train_config.yaml" \
  --vla.type "<VLA_TYPE>" \
  --data_root_dir "<PATH_TO_DATA>" \
  --run_root_dir "<PATH_TO_RUNS>" \
  --wandb_project "<WANDB_PROJECT>" \
  --wandb_entity "<WANDB_ENTITY>" \
  --reasoning_modifier_fn_str "<REASONING_MODIFIER_OR_None>" \
  --run_id_note "<RUN_TAG>"
```

## Training Parameters

These are the user-controlled parameters used in the training setup:

- `--nnodes`: number of compute nodes for distributed training.
- `--nproc-per-node`: number of GPU worker processes per node.
- `--rdzv_id`: rendezvous job id used to coordinate all workers.
- `--rdzv_backend`: rendezvous backend (`c10d` in my setup).
- `--rdzv_endpoint`: master address and port for rendezvous.
- `--config_path`: base YAML config file (`train_config.yaml`).
  - CLI parameters can be used too
- `--vla.type`: model/training recipe identifier.
- `--data_root_dir`: root folder for training data.
- `--run_root_dir`: root folder where runs/checkpoints are saved.
- `--wandb_project`: Weights & Biases project name.
- `--wandb_entity`: Weights & Biases entity/team name.
- `--reasoning_modifier_fn_str`: reasoning perturbation method (or `None`).
- `--run_id_note`: short suffix appended to run id for experiment tracking.

Example values from your current setup:

- `<VLA_TYPE>`: `prism-qwen25-dinosiglip-224px+0_5b+mx-libero-lm-90`
- `<REASONING_MODIFIER_OR_None>`: `None` or `ablate:plan,visible_objects,subtask_reasoning,subtask,move_reasoning,gripper`

### Evaluation command (parallel execution)

```bash
cd openvla-mini
python experiments/robot/libero/launch_parallel_libero_eval.py \
  --rollout_dir_name "<ROLLOUT_DIR_NAME>" \
  --gpus_per_node "<NUM_GPUS>" \
  --workers_per_gpu "<WORKERS_PER_GPU>" \
  --base_seed "<BASE_SEED>" \
  -- \
  --task_suite_name "<TASK_SUITE_NAME>" \
  --local_log_dir "<PATH_TO_LOG_DIR>" \
  --num_trials_per_task "<NUM_TRIALS_PER_TASK>" \
  --pretrained_checkpoint "<PATH_TO_CHECKPOINT>" \
  --run_id_note "<RUN_TAG>" \
  --reasoning_modifier_fn_str "<REASONING_MODIFIER_OR_None>" \
  --experiment_type "<EXPERIMENT_TYPE>" \
  --perturbation_type "<PERTURBATION_TYPE>" \
  --perturbation_level "<PERTURBATION_LEVEL>" \
  --noise_sigma "<NOISE_SIGMA>"
```

### Evaluation command (normal, non-parallel execution)

Use this when you want a single-process run without `launch_parallel_libero_eval.py`:

```bash
cd openvla-mini
python experiments/robot/libero/run_libero_eval.py \
  --model_family "<MODEL_FAMILY>" \
  --pretrained_checkpoint "<PATH_TO_CHECKPOINT>" \
  --task_suite_name "<TASK_SUITE_NAME>" \
  --num_trials_per_task "<NUM_TRIALS_PER_TASK>" \
  --seed "<SEED>" \
  --local_log_dir "<PATH_TO_LOG_DIR>" \
  --run_id_note "<RUN_TAG>" \
  --reasoning_modifier_fn_str "<REASONING_MODIFIER_OR_None>" \
  --experiment_type "<EXPERIMENT_TYPE>" \
  --perturbation_type "<PERTURBATION_TYPE>" \
  --perturbation_level "<PERTURBATION_LEVEL>" \
  --noise_sigma "<NOISE_SIGMA>"
```

## Evaluation Parameters

These are the user-controlled parameters used in your evaluation sbatch scripts:

- `--rollout_dir_name`: output experiment name/folder for this evaluation run.
- `--gpus_per_node`: number of local GPUs used for parallel evaluation.
- `--workers_per_gpu`: number of evaluation workers launched per GPU.
- `--base_seed`: base random seed for workers (worker seeds are offset from this).
- `--task_suite_name`: LIBERO suite to evaluate (for example `libero_90`).
- `--local_log_dir`: local directory for evaluation logs/results.
- `--num_trials_per_task`: number of rollouts per task.
- `--pretrained_checkpoint`: checkpoint file to evaluate.
- `--run_id_note`: experiment tag stored in logs/metrics.
- `--reasoning_modifier_fn_str`: reasoning perturbation method (or `None`).
- logs:
  - `--experiment_type`: high-level label (for example `noise_text`, `ablation`, `vanilla`).
  - `--perturbation_type`: perturbation family label (for example `word_dropout`, `invert_motion_prob`).
  - `--perturbation_level`: perturbation intensity label (for example `p30_move`).
  - `--noise_sigma`: numeric perturbation strength metadata value.



Examples from your current setup:

- vanilla: `--reasoning_modifier_fn_str None --experiment_type vanilla --perturbation_type none --perturbation_level none --noise_sigma 0`
- word dropout: `--reasoning_modifier_fn_str "word_dropout:0.3:move" --experiment_type noise_text --perturbation_type word_dropout --perturbation_level p30_move --noise_sigma 0.3`
- invert-motion prob: `--reasoning_modifier_fn_str "invert_motion_phrases_prob:0.3:move" --experiment_type noise_text --perturbation_type invert_motion_prob --perturbation_level p30_move --noise_sigma 0.3`

## Reasoning Perturbations

The repository uses one string flag:

- `--reasoning_modifier_fn_str`

This works in **both** training and evaluation.

Perturbations used in your experiments (generic form -> example):

- No perturbation:
  - Generic: `None`
  - Example: `None`
- Modality ablation:
  - Generic: `ablate:<TAG1>,<TAG2>,...`
  - Example: `ablate:plan,visible_objects,subtask_reasoning,subtask,move_reasoning,gripper`
- No-reasoning baseline:
  - Generic: `no_reasoning`
  - Example: `no_reasoning`
- BBox Gaussian noise:
  - Generic: `gaussian_bbox_sigma:<SIGMA>`
  - Example: `gaussian_bbox_sigma:20`
- Gripper Gaussian noise:
  - Generic: `gaussian_gripper_sigma:<SIGMA>`
  - Example: `gaussian_gripper_sigma:50`
- Word dropout (field-specific):
  - Generic: `word_dropout:<P>:<FIELD>`
  - Example: `word_dropout:0.3:move`
- Text-subset dropout:
  - Generic: `noise_text_subset:<P>:<FIELD1,FIELD2,...>`
  - Example: `noise_text_subset:0.3:subtask,move_reasoning,move`
- Combined bbox + text dropout:
  - Generic: `noise_all_modalities:<SIGMA>:<P>[:FIELD1,FIELD2,...]`
  - Example: `noise_all_modalities:20:0.3:subtask,move_reasoning,move`
- Temporal combined noise:
  - Generic: `temporal_noise_all_modalities_prob:<PHASE>:<SIGMA>:<P_DROPOUT>:<APPLY_P>:<EARLY_END>:<MIDDLE_END>:<LATE_END>`
  - Example: `temporal_noise_all_modalities_prob:early:20:0.3:0.4:0.4:0.6:1.0`
- Sentence shuffle:
  - Generic: `sentence_shuffle[:FIELD1,FIELD2,...]`
  - Example: `sentence_shuffle:move_reasoning`
- Plan-step shuffle:
  - Generic: `plan_step_shuffle`
  - Example: `plan_step_shuffle`
- Subtask shuffle alias:
  - Generic: `shuffle_subtask`
  - Example: `shuffle_subtask`
- Motion inversion:
  - Generic: `invert_motion_phrases[:FIELD1,FIELD2,...]`
  - Example: `invert_motion_phrases:move`
- Probabilistic motion inversion:
  - Generic: `invert_motion_phrases_prob:<P>[:FIELD1,FIELD2,...]`
  - Example: `invert_motion_phrases_prob:0.3:move`
- Knowledge-index trace mode:
  - Generic: `knowledge_index_trace`
  - Example: `knowledge_index_trace`

All implemented modifiers are defined in:

- `openvla-mini/prismatic/util/reasoning_manipulation.py`

## Where perturbation is applied

- Training-time: `vla-scripts/train.py` resolves `--reasoning_modifier_fn_str` and passes it into the dataset/batch transform path before tokenization.
- Eval-time: `run_libero_eval.py` passes the modifier to model inference, where reasoning is generated, perturbed, and then reused to requery action generation.

## Modify or Add Your Own Perturbation

1. Open `openvla-mini/prismatic/util/reasoning_manipulation.py`.
2. Add a new function (string -> string transform).
3. Register it in `get_reasoning_fn(...)`.
4. Call it from CLI via `--reasoning_modifier_fn_str "<your_modifier_name>"`.

