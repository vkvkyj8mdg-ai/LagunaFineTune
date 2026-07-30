#!/usr/bin/env python3
"""Convert jupytext-style percent scripts (notebooks_src/*.py) to Colab .ipynb.

Stdlib only. Cell markers:  `# %%` (code)  and  `# %% [markdown]`.
Markdown cell bodies are comment lines with the leading `# ` stripped.

Usage: python tools/py2ipynb.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebooks_src"
DST = ROOT / "notebooks"


def parse_cells(text):
    cells, kind, lines = [], None, []

    def flush():
        nonlocal lines
        body = "\n".join(lines).strip("\n")
        if body.strip():
            if kind == "markdown":
                body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#")
                                 for l in body.split("\n"))
            cells.append({"kind": kind, "source": body})
        lines = []

    for line in text.split("\n"):
        # column-0 only: indented "# %%" inside code/strings must not split cells
        if line.startswith("# %%"):
            if kind is not None:
                flush()
            kind = "markdown" if "[markdown]" in line else "code"
        elif kind is not None:
            lines.append(line)
    if kind is not None:
        flush()
    return cells


def to_ipynb(cells):
    nb_cells = []
    for c in cells:
        src = [l + "\n" for l in c["source"].split("\n")]
        src[-1] = src[-1].rstrip("\n")
        if c["kind"] == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
        else:
            nb_cells.append({"cell_type": "code", "metadata": {}, "source": src,
                             "outputs": [], "execution_count": None})
    return {
        "nbformat": 4, "nbformat_minor": 4,  # minor 5 would require per-cell ids
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
        },
        "cells": nb_cells,
    }


def main():
    DST.mkdir(exist_ok=True)
    py_files = sorted(SRC.glob("*.py"))
    if not py_files:
        sys.exit(f"no sources in {SRC}")
    for py in py_files:
        cells = parse_cells(py.read_text())
        out = DST / (py.stem + ".ipynb")
        out.write_text(json.dumps(to_ipynb(cells), indent=1))
        print(f"{py.name} -> {out.relative_to(ROOT)} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
