import json
from pathlib import Path

nb_path = r"C:\Users\gemin\Downloads\fable-astra-play-arc-3-handbook-harness.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

c41 = "".join(nb["cells"][41]["source"])
c42 = "".join(nb["cells"][42]["source"])

out_dir = Path("arc3sdk")
(out_dir / "fable_layer_core.py").write_text(c41, encoding="utf-8")
(out_dir / "fable_layer_patch.py").write_text(c42, encoding="utf-8")

print("Wrote arc3sdk/fable_layer_core.py (bytes:", len(c41), ")")
print("Wrote arc3sdk/fable_layer_patch.py (bytes:", len(c42), ")")
