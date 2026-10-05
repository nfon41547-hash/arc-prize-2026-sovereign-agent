import json
import re

nb = json.load(open(r"C:\Users\gemin\Downloads\arc-agi-3-sovereign-v6-full.ipynb", encoding="utf-8"))
src = "".join(nb["cells"][20]["source"])

with open(r"d:\โฟลเดอร์ใหม่ (10)\ARC-AGI-3-Kaggle-Starter-main\extracted_m2\v6_cell20_source.py", "w", encoding="utf-8") as f:
    f.write(src)

print("Saved cell 20 source to extracted_m2/v6_cell20_source.py")
