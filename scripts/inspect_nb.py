import json
import sys

nb_path = r"C:\Users\gemin\Downloads\arc-agi-3-milestone-2-solution (2).ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("Kernelspec:", nb.get("metadata", {}).get("kernelspec"))

for i, cell in enumerate(nb["cells"]):
    ctype = cell.get("cell_type")
    src = "".join(cell.get("source", []))
    lines = src.splitlines()
    first_few = " | ".join(lines[:3]) if lines else "<empty>"
    print(f"Cell {i:02d} [{ctype}]: length={len(src)} chars, lines={len(lines)} :: {first_few[:120]}")
