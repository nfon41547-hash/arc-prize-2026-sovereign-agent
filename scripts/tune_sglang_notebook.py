"""Tune SGLang notebook configurations for maximum stability and anti-timeout resilience."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def tune_notebook(nb_path):
    full_path = os.path.join(ROOT, nb_path)
    if not os.path.isfile(full_path):
        print(f"File not found: {full_path}")
        return
    with open(full_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    modified = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            new_source = []
            for line in source:
                line_mod = line
                line_mod = line_mod.replace("MEMFRAC=0.96", "MEMFRAC=0.93")
                line_mod = line_mod.replace("MAXREQ=10", "MAXREQ=4")
                line_mod = line_mod.replace("CUDAGRAPH_MAXBS=10", "CUDAGRAPH_MAXBS=4")
                line_mod = line_mod.replace("CHUNK=8192", "CHUNK=4096")
                line_mod = line_mod.replace('"--watchdog-timeout", "1800"', '"--watchdog-timeout", "3600"')
                line_mod = line_mod.replace('"watchdog-timeout 1800', '"watchdog-timeout 3600')
                if line_mod != line:
                    modified = True
                new_source.append(line_mod)
            cell["source"] = new_source

    if modified:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully tuned {nb_path}")
    else:
        print(f"No changes needed in {nb_path}")

if __name__ == "__main__":
    tune_notebook("arc-prize-2026-sovereign-agent.ipynb")
    tune_notebook("starter_push/arc-prize-2026-sovereign-agent.ipynb")
