import json
import os
import re

nb_path = r"C:\Users\gemin\Downloads\arc-agi-3-milestone-2-solution (2).ipynb"
out_dir = r"d:\โฟลเดอร์ใหม่ (10)\ARC-AGI-3-Kaggle-Starter-main\extracted_m2"

os.makedirs(out_dir, exist_ok=True)

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

patch_src = "".join(nb['cells'][2]['source'])
patch_file = os.path.join(out_dir, "harness-changes.patch")
with open(patch_file, "w", encoding="utf-8") as f:
    f.write(patch_src)
print(f"Saved full patch to {patch_file} ({len(patch_src)} chars)")

# Let's inspect what base reference repos exist locally in reference_kernels or elsewhere
