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
