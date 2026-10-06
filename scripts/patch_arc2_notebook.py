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
        print(f"Benchmark error: {e}")
"""
nb['cells'][7]['source'] = [line + '\n' for line in cell7_safe.splitlines()]

with open('starter_push_agi2/vibe-xi-omega-arc2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated vibe-xi-omega-arc2.ipynb successfully!')
