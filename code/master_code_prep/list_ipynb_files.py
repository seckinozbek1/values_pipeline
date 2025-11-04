import os

base_dir = os.path.join("..", "..", "..", "values_pipeline")

ipynb_files = []
for dirpath, _, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith(".ipynb"):
            # Build a path relative to base_dir (starts with "values_pipeline/...")
            rel_path = os.path.relpath(os.path.join(dirpath, filename), os.path.dirname(base_dir))
            ipynb_files.append(rel_path)

print("Found the following .ipynb files (paths relative to values_pipeline):")
for path in ipynb_files:
    print(path)
