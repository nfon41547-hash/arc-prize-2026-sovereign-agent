import json

nb_path = r"C:\Users\gemin\Downloads\arc-agi-3-milestone-2-solution (2).ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 0 markdown
print("=== CELL 0 ===")
print("".join(nb['cells'][0]['source']))

# Check cell 2 patch header and files modified
print("=== CELL 2 PATCH SUMMARY ===")
patch_src = "".join(nb['cells'][2]['source'])
lines = patch_src.splitlines()
diff_files = [line for line in lines if line.startswith("diff --git")]
print("Total diff lines:", len(lines))
print(f"Total modified files in patch: {len(diff_files)}")
for df in diff_files[:25]:
    print(" -", df)
if len(diff_files) > 25:
    print(f" ... and {len(diff_files) - 25} more files")

# Check cell 4 environment & models
print("=== CELL 4 (ENV & MODELS) ===")
print("".join(nb['cells'][4]['source'])[:1500])

# Check cell 12 serving config
print("=== CELL 12 (SERVING HEAD) ===")
print("".join(nb['cells'][12]['source'])[:1000])

# Check cell 16 customization hook
print("=== CELL 16 (CUSTOMIZATION HOOK) ===")
print("".join(nb['cells'][16]['source']))
