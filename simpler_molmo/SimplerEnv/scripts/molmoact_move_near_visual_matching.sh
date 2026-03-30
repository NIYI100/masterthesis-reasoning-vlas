# --------------------------
# Args & policy selection
# --------------------------
# Use plain molmoact (HF Transformers) for the stock policy + counterfactual second pass.
# molmoact_custom -> MolmoActManipulated (separate code path; ignores --counterfactual-*).
# molmoact + --vllm -> faster inference; counterfactual flags are ignored (warning at import).
POLICY_MODEL="molmoact"
USE_VLLM=0

# Repo root = parent of scripts/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# Extra args forwarded to `python simpler_env/main_inference.py` (quote-sensitive).
# Example: EXTRA_MOLMO_ARGS='--counterfactual-perturb-fn shift_xy_pairs:15,0'
# Example: EXTRA_MOLMO_ARGS='--counterfactual-perturb-fn append_suffix:_PTEST --counterfactual-max-new-tokens 128'
# Aggregates (success_rate, per-episode action deltas, by_task): see --logging-dir /experiment_summary.json
#   or --experiment-summary-path /path/to/run.json; disable with --no-experiment-summary
EXTRA_MOLMO_ARGS="${EXTRA_MOLMO_ARGS:-}"
# Args after `--` are appended to the python command (same as extra CLI flags).
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm)
      USE_VLLM=1
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--vllm] [-- EXTRA_PY_ARGS...]"
      echo ""
      echo "Pass flags to main_inference.py in either way:"
      echo "  1) Environment: EXTRA_MOLMO_ARGS='--counterfactual-perturb-fn identity' $0"
      echo "  2) After --:     $0 -- --counterfactual-perturb-fn identity --verbose"
      echo ""
      echo "With xvfb-run:"
      echo "  EXTRA_MOLMO_ARGS='--counterfactual-perturb-fn shift_xy_pairs:10,0' \\"
      echo "    xvfb-run -a bash scripts/molmoact_move_near_visual_matching.sh"
      exit 0
      ;;
    --)
      shift
      PASSTHROUGH+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1 (use -- before flags for main_inference.py, or set EXTRA_MOLMO_ARGS)"
      exit 1
      ;;
  esac
done

if [[ "${USE_VLLM}" -eq 1 ]]; then
  POLICY_MODEL="${POLICY_MODEL}-vllm"
fi

gpu_id=0

declare -a arr=("allenai/MolmoAct-7B-D-Pretrain-0812") #\
                #"allenai/MolmoAct-7B-D-Pretrain-RT-1-0812")

env_name=MoveNearGoogleBakedTexInScene-v0
# env_name=MoveNearGoogleBakedTexInScene-v1
scene_name=google_pick_coke_can_1_v4
rgb_overlay_path=./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png

# URDF variations
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

for ckpt_path in "${arr[@]}"; do echo "$ckpt_path"; done


for urdf_version in "${urdf_version_arr[@]}";

do for ckpt_path in "${arr[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${POLICY_MODEL} --ckpt-path ${ckpt_path} \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 10 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
  --additional-env-build-kwargs urdf_version=${urdf_version} \
  --additional-env-save-tags baked_except_bpb_orange \
  --experiment move_near \
  ${EXTRA_MOLMO_ARGS} \
  "${PASSTHROUGH[@]}";

done

done
