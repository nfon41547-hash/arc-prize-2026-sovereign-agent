import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, total_tasks, end_time, nprocs):
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        gpu_id = rank % num_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(gpu_id)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue across ranks
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(2)
    
    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")
    
    print(f"[Rank {rank} (GPU {rank % max(1, num_gpus)})] starting worker...")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {rank}] completed all assigned tasks!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

    if os.path.exists("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"):
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    task_keys = sorted(data.keys())
    for key in task_keys:
        queue.put(key)
    
    num_gpus = torch.cuda.device_count()
    nprocs = max(1, num_gpus)
    print(f"🚀 Sovereign TTT Engine launched: {len(task_keys)} puzzles, {nprocs} workers on {num_gpus} GPUs...")

    for _ in range(nprocs):
        queue.put(None)
    
    if nprocs > 1:
        mp.spawn(local_worker, args=(queue, len(task_keys), args.end_time, nprocs), nprocs=nprocs)
    else:
        local_worker(0, queue, len(task_keys), args.end_time, 1)
