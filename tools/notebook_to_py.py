#!/usr/bin/env python3
"""Convert a Python notebook to a script without Jupyter/nbconvert.

Intentionally fails on IPython magics to avoid silently changing training behavior.
"""
import argparse
import json
from pathlib import Path


def convert(source: Path, output: Path) -> None:
    notebook = json.loads(source.read_text(encoding="utf-8"))
    if notebook.get("nbformat", 0) < 4:
        raise ValueError("Expected notebook format >= 4")
    chunks = ["# Generated from {}. Do not edit; modify the notebook.\n".format(source.name)]
    for cell_no, cell in enumerate(notebook.get("cells", []), 1):
        if cell.get("cell_type") != "code":
            continue
        raw = cell.get("source", [])
        code = "".join(raw) if isinstance(raw, list) else raw
        for line in code.splitlines():
            stripped = line.lstrip()
            if stripped.startswith(("%", "!", "?")) or stripped.startswith("get_ipython("):
                raise ValueError(
                    f"Notebook cell {cell_no} contains IPython-only syntax: {stripped[:90]}. "
                    "Move it out of the production notebook or use nbconvert."
                )
        chunks.extend([f"\n# --- notebook cell {cell_no} ---\n", code, "\n"])
    script = "".join(chunks)
    compile(script, str(output), "exec")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(script, encoding="utf-8")
    print(f"Converted {source} -> {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    convert(args.notebook, args.output)


if __name__ == "__main__":
    main()
