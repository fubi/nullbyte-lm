import os
from pathlib import Path

# ROOT = Path(".").resolve().parent if Path(".").resolve().name == "data-pipeline" else Path(".").resolve()
# ^ adjust ROOT below if this doesn't land on your actual project root - see note after the script
ROOT = Path(".").resolve()

OUTPUT_FILE = "repo_inventory.txt"

# directories to skip entirely (never descend into these)
SKIP_DIRS = {".venv", "__pycache__", ".git", ".pytest_cache", "node_modules", ".idea", ".vscode"}

# file extensions whose full content should be dumped
TEXT_EXTENSIONS = {".py", ".md", ".toml", ".txt", ".cfg", ".ini"}

# extensions/files to always just list with size, never dump content
SKIP_CONTENT_EXTENSIONS = {".bin", ".pt", ".json", ".pyc", ".npy"}

MAX_CONTENT_CHARS = 20_000  # truncate any single file's dumped content beyond this

def human_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TB"

def main():
    root = ROOT
    print(f"Scanning from: {root}")

    lines = []
    lines.append(f"REPO INVENTORY — root: {root}")
    lines.append("=" * 80)
    lines.append("")

    # ---------- Section 1: directory tree ----------
    lines.append("DIRECTORY TREE")
    lines.append("-" * 80)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        rel_dir = Path(dirpath).relative_to(root)
        depth = len(rel_dir.parts)
        indent = "  " * depth
        if str(rel_dir) != ".":
            lines.append(f"{indent}{rel_dir.name}/")
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            size = fpath.stat().st_size
            lines.append(f"{indent}  {fname}  ({human_size(size)})")
    lines.append("")

    # ---------- Section 2: file contents ----------
    lines.append("FILE CONTENTS")
    lines.append("=" * 80)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in sorted(filenames):
            fpath = Path(dirpath) / fname
            rel_path = fpath.relative_to(root)
            ext = fpath.suffix.lower()

            if ext in SKIP_CONTENT_EXTENSIONS:
                continue
            if ext not in TEXT_EXTENSIONS:
                continue

            lines.append("")
            lines.append(f"--- {rel_path} ---")
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
                if len(content) > MAX_CONTENT_CHARS:
                    content = content[:MAX_CONTENT_CHARS] + f"\n... [truncated, {len(content)} total chars]"
                lines.append(content)
            except Exception as e:
                lines.append(f"[could not read: {e}]")

    output_path = root / OUTPUT_FILE
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote inventory to: {output_path}")
    print(f"Total size: {human_size(output_path.stat().st_size)}")

if __name__ == "__main__":
    main()