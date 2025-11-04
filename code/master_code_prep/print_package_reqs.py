import os, ast

def get_imports_from_file(file_path):
    """Extract top-level imported package names from one .py file."""
    imports = set()
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            node = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            return imports  # skip files that don't parse
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for name in n.names:
                imports.add(name.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                imports.add(n.module.split(".")[0])
    return imports

def collect_all_imports(root="."):
    """Recursively collect imports from all .py files under root."""
    all_imports = set()
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = os.path.join(dirpath, filename)
                all_imports |= get_imports_from_file(file_path)
    return all_imports

# Scan the values_pipeline folder 3 levels up
TARGET_ROOT = os.path.join("..", "..", "..", "values_pipeline")
packages = collect_all_imports(TARGET_ROOT)

# Overwrite requirements.txt with sorted package names
with open("requirements.txt", "w", encoding="utf-8") as f:
    for pkg in sorted(packages):
        f.write(pkg + "\n")

print("requirements.txt written with the following packages:")
print("\n".join(sorted(packages)))
