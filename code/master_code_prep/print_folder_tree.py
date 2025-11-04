import os

def print_tree(start_path, prefix=""):
    """
    Recursively print the directory contents of a given path as a tree structure,
    omitting any directories named 'cache' or 'TXT' (and all their contents).
    """
    try:
        items = sorted(os.listdir(start_path))
    except FileNotFoundError:
        print(f"Path not found: {start_path}")
        return

    for i, name in enumerate(items):
        path = os.path.join(start_path, name)

        # Skip entire cache or TXT directories
        if name == "cache" or name == "TXT":
            continue

        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + name)

        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(path, prefix + extension)

# Start printing from values_pipeline two levels up
base_dir = os.path.join("..", "..", "..", "values_pipeline")
print_tree(base_dir)
