import json

nb_path = r"C:\Users\gemin\Downloads\fable-astra-play-arc-3-handbook-harness.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i in range(40, len(nb["cells"])):
    cell = nb["cells"][i]
    cell_type = cell.get("cell_type", "unknown")
    source = "".join(cell.get("source", []))
    print(f"\n==================== Cell {i} [{cell_type}] (length: {len(source)}) ====================")
    print(source[:1000])
