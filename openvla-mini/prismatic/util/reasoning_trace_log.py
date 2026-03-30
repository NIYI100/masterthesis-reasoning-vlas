"""
Append-only NDJSON logging of model reasoning during LIBERO eval for offline joins
(e.g. ROUGE-L vs libero_reasonings.json).

Each line is one JSON object. Fields (enable with GenerateConfig.reasoning_trace_jsonl):

- task_suite_name (str)
- task_id (int): LIBERO suite task index
- task_description (str): language instruction (may duplicate across tasks)
- episode_idx (int): trial index within this task for this run (0 .. num_trials_per_task-1)
- env_step (int): environment step counter ``t`` in the eval loop (includes num_steps_wait)
- shard_rank, num_shards (int): parallel worker id / total shards
- seed (int): eval seed
- reasoning (str): final CoT string used for metrics and action (post-modifier if any)
- clean_reasoning (str | null): pre-modifier CoT from pass-1 when a reasoning_modifier_fn is set; else null
- reasoning_modifier_fn_str (str): config string
- model_family (str)
- control_actions_from_forward (int): how many env actions this forward may execute (chunked policy)
- wall_time_utc (str): ISO-8601 UTC timestamp added at write time

Joining to RLDS/GT: use (task_suite_name, task_id, episode_idx, env_step) once you confirm how
your supervision keys map to LIBERO trials and frames.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def resolve_reasoning_trace_path(base_path: Optional[str], num_shards: int, shard_rank: int) -> Optional[str]:
    """
    When num_shards > 1, write to a shard-specific file to avoid concurrent writes
    from parallel workers (e.g. name.shard3.jsonl).
    """
    if not base_path or not str(base_path).strip():
        return None
    p = str(base_path).strip()
    if num_shards > 1:
        root, ext = os.path.splitext(p)
        if ext.lower() == ".jsonl":
            return f"{root}.shard{int(shard_rank)}{ext}"
        return f"{p}.shard{int(shard_rank)}.jsonl"
    return p


def append_reasoning_trace_line(path: Optional[str], record: Dict[str, Any]) -> None:
    """Append one JSON object per line. Adds wall_time_utc (ISO-8601, Z)."""
    if not path:
        return
    out = dict(record)
    out["wall_time_utc"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
