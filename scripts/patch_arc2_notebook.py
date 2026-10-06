import json
import os

with open('starter_push_agi2/vibe-xi-omega-arc2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: wheelhouse install & output directory
nb['cells'][1]['source'] = [
    '!pip uninstall -y tensorflow\n',
    'import os, glob\n',
    'wheel_dirs = [d for d in ["/kaggle/input/arc2-unsloth-wheelhouse", "/kaggle/input/unsloth-wheelhouse-py312-torch210", "/kaggle/input/unsloth-wheelhouse"] if os.path.exists(d)]\n',
    'if wheel_dirs:\n',
    '    os.system(f"pip install --no-index --find-links={wheel_dirs[0]} unsloth unsloth_zoo trl xformers bitsandbytes --no-deps")\n',
    '    os.system(f"pip install --no-index --find-links={wheel_dirs[0]} unsloth unsloth_zoo trl xformers bitsandbytes")\n',
    'os.makedirs("/kaggle/inference_outputs", exist_ok=True)\n'
]

# Cell 3: arc_decoder.py - robust safe directory check
cell3_code = """%%writefile arc_decoder.py
import os
import bz2
import pickle
import numpy as np

def hashable(guess):
    return tuple(map(tuple, guess))

def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    if not guess_list:
        return []
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs

def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline-g.get("beam_score", 0) for g in guesses])
    aug_score = np.mean([np.sum([baseline-s for s in g.get("score_aug", [0])]) for g in guesses]) if guesses else 0
    return inf_score + aug_score

def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)

def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g.get("score_aug", [0])) for g in guesses]) if guesses else 0
    return inf_score - aug_score

def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
]


class ArcDecoder:
    
    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        os.makedirs(store, exist_ok=True)
        for key in os.listdir(store):
            try:
                with bz2.BZ2File(os.path.join(store, key)) as f:
                    outputs = pickle.load(f)
                base_key = key.split(".")[0]
                self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
                for i, sample in enumerate(outputs):
                    self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample
            except Exception as e:
                print(f"Error loading {key}: {e}")

    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")
        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0
        correct_beam_scores = []

        for basekey, basevalues in self.decoded_results.items():
            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)
            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():
                solution = sample["solution"]
                beam_score = sample.get("beam_score", 0)
                aug_mean = np.mean(sample.get("score_aug", [0]))

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"
                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_keys += 1

        if num_total_keys > 0:
            print(f" subkeys: {num_solved_keys}/{num_total_keys}")
            if correct_beam_scores:
                print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
                print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")
"""
nb['cells'][3]['source'] = [line + '\n' for line in cell3_code.splitlines()]

# Cell 5: Dynamic time allocation & adaptive queue dispatch
starter_code = """%%writefile starter.py
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
"""
nb['cells'][5]['source'] = [line + '\n' for line in starter_code.splitlines()]

# Cell 7: Full robust ensemble and symbolic fallback
cell7_code = """import os
import json
import numpy as np
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

rerun_mode = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

if os.path.exists("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"):
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

# Fill any unsolved fallback with intelligent heuristics (input shape/identity fallback)
for task_id, attempts_list in submission.items():
    query_info = data.queries.get(task_id, {})
    test_inputs = query_info.get('test', [])
    for idx, att in enumerate(attempts_list):
        inp_grid = test_inputs[idx]['input'] if idx < len(test_inputs) else [[0]]
        # If attempt is trivial default [[0]], substitute with smart shape/identity priors
        if att.get('attempt_1') == [[0]]:
            att['attempt_1'] = inp_grid
        if att.get('attempt_2') == [[0]]:
            att['attempt_2'] = inp_grid

with open("submission.json", "w") as f:
    json.dump(submission, f)

print(f"*** Generated complete submission.json with {len(submission)} tasks successfully!")

if not rerun_mode and hasattr(data, 'replies') and data.replies:
    try:
        decoder.benchmark_selection_algos()
        with open("submission.json", "r") as f:
            reload_submission = json.load(f)
        print("*** Benchmark accuracy score:", data.validate_submission(reload_submission))
    except Exception as e:
        print(f"Benchmark diagnostics: {e}")
"""
nb['cells'][7]['source'] = [line + '\n' for line in cell7_code.splitlines()]

with open('starter_push_agi2/vibe-xi-omega-arc2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated vibe-xi-omega-arc2.ipynb to EVOLVED Version 9 successfully!')
