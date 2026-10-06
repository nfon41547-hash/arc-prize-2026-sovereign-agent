import json
import re

nb_path = "fable-astra-play-arc-3-handbook-harness.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if any(k in source.lower() for k in ["model", "qwen", "nvfp4", "fp8", "vllm", "flash", "serving", "weights", "model_sources"]):
        print(f"\n--- Cell {i} [{cell.get('cell_type')}] ---")
        for line in source.splitlines():
            if any(k in line.lower() for k in ["model", "qwen", "nvfp4", "fp8", "vllm", "flash", "serving", "weights", "dataset", "bundle", "source"]):
                print(line[:120])
