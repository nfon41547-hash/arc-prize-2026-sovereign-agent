import json
import sys

nb_path = r"C:\Users\gemin\Downloads\fable-astra-play-arc-3-handbook-harness.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print(f"Kernel metadata: {nb.get('metadata', {})}")

for i, cell in enumerate(nb["cells"]):
    cell_type = cell.get("cell_type", "unknown")
    source = "".join(cell.get("source", []))
    first_lines = "\n".join([line.strip() for line in source.split("\n")[:3] if line.strip()])
    print(f"\n--- Cell {i} [{cell_type}] (chars: {len(source)}) ---")
    print(first_lines[:200])
