#!/usr/bin/env python3
"""
One-off script: add task_success_vector to metadata.json for each agent under a DGM run directory
that has overall_performance but no task_success_vector.
Run from dgm/ directory: python ignore/add_task_success_vector_to_metadata.py output_dgm/<run_id>
"""
import argparse
import json
import os
import sys

# Run from dgm/ so that utils can be imported
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DGM_DIR = os.path.dirname(SCRIPT_DIR)
if DGM_DIR not in sys.path:
    sys.path.insert(0, DGM_DIR)

from utils.common_utils import load_json_file
from utils.evo_utils import build_task_success_vector


def main():
    parser = argparse.ArgumentParser(description="Add task_success_vector to existing agent metadata.json")
    parser.add_argument("dgm_run_dir", type=str, help="DGM run directory (e.g. output_dgm/20260130175344_606722)")
    parser.add_argument("--task_list", type=str, default=None, help="Path to task.json (default: dgm/swe_bench/subsets/task.json)")
    args = parser.parse_args()

    dgm_run_dir = os.path.abspath(args.dgm_run_dir) if not os.path.isabs(args.dgm_run_dir) else args.dgm_run_dir
    task_list_path = args.task_list or os.path.join(DGM_DIR, "swe_bench", "subsets", "task.json")

    if not os.path.exists(task_list_path):
        print(f"Error: task list not found: {task_list_path}")
        return 1

    if not os.path.isdir(dgm_run_dir):
        print(f"Error: not a directory: {dgm_run_dir}")
        return 1

    updated = 0
    skipped_no_perf = 0
    skipped_has_vector = 0

    for name in sorted(os.listdir(dgm_run_dir)):
        agent_dir = os.path.join(dgm_run_dir, name)
        if not os.path.isdir(agent_dir):
            continue
        meta_path = os.path.join(agent_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        if "task_success_vector" in metadata:
            skipped_has_vector += 1
            continue

        overall_performance = metadata.get("overall_performance")
        if not overall_performance:
            skipped_no_perf += 1
            continue

        vector = build_task_success_vector(overall_performance, task_list_path)
        if vector is None:
            skipped_no_perf += 1
            continue

        metadata["task_success_vector"] = vector
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)
        updated += 1
        print(f"Updated: {name} (vector length {len(vector)}, sum={sum(vector)})")

    print(f"\nDone. Updated {updated}, skipped (already has vector) {skipped_has_vector}, skipped (no overall_performance) {skipped_no_perf}")
    return 0


if __name__ == "__main__":
    exit(main())
