"""
launch_parallel_libero_eval.py

Launches multiple `run_libero_eval.py` workers in parallel with task sharding.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import List

import torch


def _parse_gpus(gpu_str: str) -> List[str]:
    gpus = [token.strip() for token in gpu_str.split(",") if token.strip() != ""]
    if len(gpus) == 0:
        raise ValueError("No GPU IDs provided. Example: --gpus 0,1,2,3")
    return gpus


def _get_arg_value(args_list: List[str], arg_name: str, default_value: str) -> str:
    if arg_name in args_list:
        idx = args_list.index(arg_name)
        if idx + 1 < len(args_list):
            return args_list[idx + 1]
    return default_value


def _upsert_arg(args_list: List[str], arg_name: str, arg_value: str) -> List[str]:
    updated = list(args_list)
    if arg_name in updated:
        idx = updated.index(arg_name)
        if idx + 1 < len(updated):
            updated[idx + 1] = arg_value
        else:
            updated.append(arg_value)
    else:
        updated.extend([arg_name, arg_value])
    return updated


def _aggregate_worker_metrics(metric_paths: List[str]) -> dict:
    if len(metric_paths) == 0:
        raise ValueError("No worker metric files found to aggregate.")

    payloads = []
    for path in sorted(metric_paths):
        with open(path, "r") as f:
            payload = json.load(f)
        payload["_source_path"] = path
        payloads.append(payload)

    task_suite_name = payloads[0]["task_suite_name"]
    num_trials_per_task = payloads[0]["num_trials_per_task"]
    for payload in payloads[1:]:
        if payload["task_suite_name"] != task_suite_name:
            raise ValueError(
                f"Inconsistent task_suite_name in {payload['_source_path']}: {payload['task_suite_name']} != {task_suite_name}"
            )
        if payload["num_trials_per_task"] != num_trials_per_task:
            raise ValueError(
                f"Inconsistent num_trials_per_task in {payload['_source_path']}: "
                f"{payload['num_trials_per_task']} != {num_trials_per_task}"
            )

    combined_total_episodes = 0
    combined_total_successes = 0
    combined_task_ids = set()
    combined_per_task = {}

    for payload in payloads:
        combined_total_episodes += int(payload["total_episodes"])
        combined_total_successes += int(payload["total_successes"])
        for task_desc, task_data in payload["per_task_metrics"].items():
            task_id = int(task_data["task_id"])
            if task_id in combined_task_ids:
                raise ValueError(
                    f"Duplicate task_id {task_id} found across worker metrics. "
                    "Each task should be evaluated by exactly one worker."
                )
            combined_task_ids.add(task_id)
            combined_per_task[task_desc] = task_data

    return {
        "task_suite_name": task_suite_name,
        "num_trials_per_task": int(num_trials_per_task),
        "num_shards_found": len(payloads),
        "input_metric_paths": sorted(metric_paths),
        "evaluated_task_ids": sorted(int(tid) for tid in combined_task_ids),
        "num_unique_tasks_evaluated": len(combined_task_ids),
        "total_episodes": int(combined_total_episodes),
        "total_successes": int(combined_total_successes),
        "total_success_rate": float(combined_total_successes) / float(combined_total_episodes),
        "per_task_metrics": combined_per_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch parallel LIBERO eval workers.")
    parser.add_argument("--gpus", type=str, required=False, help="Comma-separated GPU IDs (e.g. '0,1').")
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="How many eval workers per GPU.")
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=None,
        help="Alternative to --gpus: number of local GPUs to use (IDs 0..gpus_per_node-1).",
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="Total nodes in the job.")
    parser.add_argument("--node_rank", type=int, default=0, help="Current node rank in [0, num_nodes-1].")
    parser.add_argument("--base_seed", type=int, default=7, help="Base seed; worker i uses base_seed + i.")
    parser.add_argument("--python_executable", type=str, default=sys.executable, help="Python executable for workers.")
    parser.add_argument("eval_args", nargs=argparse.REMAINDER, help="Arguments passed through to run_libero_eval.py.")
    args = parser.parse_args()

    if args.gpus is not None and args.gpus_per_node is not None:
        raise ValueError("Specify either --gpus or --gpus_per_node, not both.")
    if args.gpus is not None:
        gpus = _parse_gpus(args.gpus)
    elif args.gpus_per_node is not None:
        if args.gpus_per_node <= 0:
            raise ValueError(f"--gpus_per_node must be >= 1, got {args.gpus_per_node}")
        gpus = [str(i) for i in range(args.gpus_per_node)]
    else:
        gpus = [str(i) for i in range(max(1, torch.cuda.device_count()))]

    if args.workers_per_gpu <= 0:
        raise ValueError(f"--workers_per_gpu must be >= 1, got {args.workers_per_gpu}")
    if args.num_nodes <= 0:
        raise ValueError(f"--num_nodes must be >= 1, got {args.num_nodes}")
    if args.node_rank < 0 or args.node_rank >= args.num_nodes:
        raise ValueError(
            f"--node_rank must satisfy 0 <= node_rank < num_nodes, got node_rank={args.node_rank}, num_nodes={args.num_nodes}"
        )

    eval_args = args.eval_args
    if len(eval_args) > 0 and eval_args[0] == "--":
        eval_args = eval_args[1:]

    launch_id = time.strftime("%Y_%m_%d-%H_%M_%S")
    launch_prefix = f"PARALLEL-{launch_id}-"
    eval_args = _upsert_arg(eval_args, "--prefix", f"{launch_prefix}{_get_arg_value(eval_args, '--prefix', '')}")

    local_num_workers = len(gpus) * args.workers_per_gpu
    global_num_workers = args.num_nodes * local_num_workers
    global_rank_offset = args.node_rank * local_num_workers
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, "run_libero_eval.py")

    print(
        f"Launching {local_num_workers} local workers on node {args.node_rank}/{args.num_nodes - 1} "
        f"(global workers: {global_num_workers})"
    )
    print(f"Local GPUs: {gpus}")
    print(f"Eval script: {eval_script}")
    print(f"Pass-through args: {eval_args}")

    procs = []
    for local_worker_rank in range(local_num_workers):
        global_worker_rank = global_rank_offset + local_worker_rank
        assigned_gpu = gpus[local_worker_rank % len(gpus)]
        worker_seed = args.base_seed + global_worker_rank

        cmd = [
            args.python_executable,
            eval_script,
            "--num_shards",
            str(global_num_workers),
            "--shard_rank",
            str(global_worker_rank),
            "--seed",
            str(worker_seed),
            "--run_id_note",
            f"parallel-node-{args.node_rank}-worker-{local_worker_rank}-global-{global_worker_rank}",
            *eval_args,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

        print(f"[Local worker {local_worker_rank} | global rank {global_worker_rank}] GPU {assigned_gpu} | seed {worker_seed}")
        procs.append((global_worker_rank, assigned_gpu, subprocess.Popen(cmd, env=env)))

    failures = []
    for worker_rank, assigned_gpu, proc in procs:
        return_code = proc.wait()
        if return_code != 0:
            failures.append((worker_rank, assigned_gpu, return_code))
    if failures:
        print("\nSome workers failed:")
        for worker_rank, assigned_gpu, return_code in failures:
            print(f"  worker={worker_rank} gpu={assigned_gpu} exit_code={return_code}")
        raise SystemExit(1)

    local_log_dir = _get_arg_value(eval_args, "--local_log_dir", "./experiments/logs")
    local_log_dir_abs = local_log_dir if os.path.isabs(local_log_dir) else os.path.abspath(local_log_dir)
    worker_metric_paths = sorted(glob.glob(os.path.join(local_log_dir_abs, f"{launch_prefix}*.json")))
    aggregate = _aggregate_worker_metrics(worker_metric_paths)
    aggregate["launch_id"] = launch_id
    aggregate["launch_prefix"] = launch_prefix
    aggregate["num_nodes"] = int(args.num_nodes)
    aggregate["node_rank"] = int(args.node_rank)
    aggregate["gpus"] = gpus
    aggregate["workers_per_gpu"] = int(args.workers_per_gpu)
    aggregate["global_num_workers"] = int(global_num_workers)

    aggregate_json_path = os.path.join(local_log_dir_abs, f"{launch_prefix}GLOBAL-RESULTS.json")
    with open(aggregate_json_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nWrote global aggregate JSON: {aggregate_json_path}")
    print("\nAll workers finished successfully.")


if __name__ == "__main__":
    main()
