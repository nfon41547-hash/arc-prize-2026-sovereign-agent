import os
import shutil
import subprocess

src_base = r"d:\โฟลเดอร์ใหม่ (10)\ARC-AGI-3-Kaggle-Starter-main\reference_kernels\duck-harness"
target_dir = r"d:\โฟลเดอร์ใหม่ (10)\ARC-AGI-3-Kaggle-Starter-main\extracted_m2\duck-harness-m2"
patch_path = r"d:\โฟลเดอร์ใหม่ (10)\ARC-AGI-3-Kaggle-Starter-main\extracted_m2\harness-changes.patch"

if os.path.exists(target_dir):
    shutil.rmtree(target_dir)

print(f"Copying {src_base} to {target_dir}...")
shutil.copytree(src_base, target_dir)

# Clean first line if it's %%writefile
with open(patch_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

if lines[0].startswith("%%writefile"):
    clean_patch = "".join(lines[1:])
    clean_patch_path = r"d:\โฟลเดอร์ใหม่ (10)\ARC-AGI-3-Kaggle-Starter-main\extracted_m2\clean-changes.patch"
    with open(clean_patch_path, "w", encoding="utf-8") as f:
        f.write(clean_patch)
else:
    clean_patch_path = patch_path

print(f"Applying patch {clean_patch_path} to {target_dir}...")
# git apply
cmd = ["git", "apply", "--ignore-whitespace", "--whitespace=nowarn", clean_patch_path]
res = subprocess.run(cmd, cwd=target_dir, capture_output=True, text=True)
print("Git apply exit code:", res.returncode)
print("Stdout:", res.stdout)
print("Stderr:", res.stderr)
