import json
import re

nb_path = r"C:\Users\gemin\Downloads\arc-agi-3-milestone-2-solution (2).ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

patch_src = "".join(nb['cells'][2]['source'])

# Split patch by diff --git
diffs = re.split(r"(?=diff --git )", patch_src)
print(f"Total diff sections: {len(diffs)}")

for d in diffs:
    lines = d.splitlines()
    if not lines: continue
    header = lines[0]
    print(f"\n>>> {header}")
    added = sum(1 for l in lines if l.startswith('+') and not l.startswith('+++'))
    removed = sum(1 for l in lines if l.startswith('-') and not l.startswith('---'))
    print(f"    (+{added}, -{removed} lines)")
    # print first 5 added lines that are meaningful
    sample_adds = [l for l in lines if l.startswith('+') and not l.startswith('+++') and len(l.strip()) > 2][:5]
    for sa in sample_adds:
        print(f"      {sa[:90]}")
