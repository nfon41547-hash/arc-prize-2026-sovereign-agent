import json

nb_path = r"C:\Users\gemin\Downloads\fable-astra-play-arc-3-handbook-harness.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        print(f"\n=== Code Cell {i} (lines: {len(source.splitlines())}, chars: {len(source)}) ===")
        print("\n".join(source.splitlines()[:6]))
