import json
import ast
import sys

nb_path = r"starter_push/arc-agi-3-starter-kernel-v32-profile-3.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
syntax_errors = 0
for idx, cell in enumerate(nb["cells"]):
    ctype = cell.get("cell_type")
    src = "".join(cell.get("source", []))
    if ctype == "code":
        py_lines = []
        for line in src.splitlines():
            sline = line.strip()
            if sline.startswith("%") or sline.startswith("!") or sline.startswith("?"):
                py_lines.append("# " + line)
            else:
                py_lines.append(line)
        clean_code = "\n".join(py_lines)
        try:
            ast.parse(clean_code)
            status = "VALID_AST"
        except SyntaxError as e:
            status = f"SYNTAX_ERROR: {e}"
            syntax_errors += 1
        first_line = clean_code.splitlines()[0] if clean_code.splitlines() else "<empty>"
        print(f"Cell {idx:02d} [code] ({len(src):>6} chars) :: {status} | {first_line[:60]}")
    else:
        first_line = src.splitlines()[0] if src.splitlines() else "<empty>"
        print(f"Cell {idx:02d} [markdown] ({len(src):>6} chars) :: {first_line[:60]}")

print("Total syntax errors:", syntax_errors)
if syntax_errors > 0:
    sys.exit(1)
print("ALL CELLS 100% VALID!")
