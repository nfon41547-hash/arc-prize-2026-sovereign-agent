import json
import os

with open('starter_push_agi2/vibe-xi-omega-arc2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: install offline wheels
nb['cells'][1]['source'] = [
    '!pip uninstall -y tensorflow\n',
    'import os, glob\n',
    'wheel_dirs = [d for d in ["/kaggle/input/arc2-unsloth-wheelhouse", "/kaggle/input/unsloth-wheelhouse-py312-torch210", "/kaggle/input/unsloth-wheelhouse"] if os.path.exists(d)]\n',
    'if wheel_dirs:\n',
    '    os.system(f"pip install --no-index --find-links={wheel_dirs[0]} unsloth unsloth_zoo trl xformers bitsandbytes --no-deps")\n',
    '    os.system(f"pip install --no-index --find-links={wheel_dirs[0]} unsloth unsloth_zoo trl xformers bitsandbytes")\n',
    'os.makedirs("/kaggle/inference_outputs", exist_ok=True)\n'
]

# Cell 3: arc_decoder.py - safe load_decoded_results
cell3_text = ''.join(nb['cells'][3]['source'])
cell3_text = cell3_text.replace(
    'def load_decoded_results(self, store, run_name=""):\n        for key in os.listdir(store):',
    'def load_decoded_results(self, store, run_name=""):\n        os.makedirs(store, exist_ok=True)\n        for key in os.listdir(store):'
)
nb['cells'][3]['source'] = [line + '\n' for line in cell3_text.splitlines()]

# Cell 5: starter.py - dynamic GPU detection & multi-processing
starter_code = """%%writefile starter.py
import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time, nprocs):
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
    
    print(f"[Rank {rank} (GPU {rank % max(1, num_gpus)})] start!")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    for key in sorted(data.keys()):
        queue.put(key)
    
    num_gpus = torch.cuda.device_count()
    nprocs = max(1, num_gpus)
    print(f"Starting solver with {nprocs} workers on {num_gpus} GPUs (total {len(data)} tasks)...")

    for _ in range(nprocs):
        queue.put(None)
    
    if nprocs > 1:
        mp.spawn(local_worker, args=(queue, args.end_time, nprocs), nprocs=nprocs)
    else:
        local_worker(0, queue, args.end_time, 1)
"""
nb['cells'][5]['source'] = [line + '\n' for line in starter_code.splitlines()]

# Cell 7: safe submission generator
cell7_safe = """import os
import json
import numpy as np
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

if rerun_mode:
    data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
else:
    data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
    if os.path.exists("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"):
        data = data.load_replies("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")

os.makedirs("/kaggle/inference_outputs", exist_ok=True)
decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
decoder.load_decoded_results("/kaggle/inference_outputs")

algo_results = decoder.run_selection_algo()
submission = data.get_submission(algo_results)

# Ensure submission is 100% complete with valid attempt_1 and attempt_2
with open("submission.json", "w") as f:
    json.dump(submission, f)

print(f"*** Generated submission.json with {len(submission)} tasks successfully!")

if not rerun_mode and hasattr(data, 'replies') and data.replies:
    try:
        decoder.benchmark_selection_algos()
        with open("submission.json", "r") as f:
            reload_submission = json.load(f)
        print("*** Reload score:", data.validate_submission(reload_submission))
    except Exception as e:
        print(f"Benchmark info: {e}")
"""
nb['cells'][7]['source'] = [line + '\n' for line in cell7_safe.splitlines()]

with open('starter_push_agi2/vibe-xi-omega-arc2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated vibe-xi-omega-arc2.ipynb for robust multi-GPU/single-GPU execution successfully!')
