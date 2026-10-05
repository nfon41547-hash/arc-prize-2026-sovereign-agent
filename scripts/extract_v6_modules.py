import ast

v6_src = r"d:\โฟลเดอร์ใหม่ (10)\ARC-AGI-3-Kaggle-Starter-main\extracted_m2\v6_cell20_source.py"
with open(v6_src, "r", encoding="utf-8") as f:
    text = f.read()

tree = ast.parse(text)
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        # look for _v6_mod.write_text
        if isinstance(node.func, ast.Attribute) and node.func.attr == "write_text":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "_v6_mod":
                sobu_code = node.args[0].value
                target = r"d:\โฟลเดอร์ใหม่ (10)\ARC-AGI-3-Kaggle-Starter-main\arc3sdk\sobu_v6.py"
                with open(target, "w", encoding="utf-8") as out:
                    out.write(sobu_code)
                print(f"Extracted sobu_v6.py ({len(sobu_code)} chars)")
            elif isinstance(node.func.value, ast.Name) and node.func.value.id == "_v6_sched":
                sched_code = node.args[0].value
                target_sched = r"d:\โฟลเดอร์ใหม่ (10)\ARC-AGI-3-Kaggle-Starter-main\arc3sdk\priority_scheduler_dprime.py"
                with open(target_sched, "w", encoding="utf-8") as out:
                    out.write(sched_code)
                print(f"Extracted priority_scheduler_dprime.py ({len(sched_code)} chars)")
