"""Build notebooks/A2_main.ipynb from notebooks/A2_main.py, so the notebook never drifts from the code.

A2_main.py is a VS Code "percent" script: "# %%" starts a code cell, "# %% [markdown]" a text cell.
The notebook is those cells plus an "Open in Colab" badge: Introduction, Data Preparation, one section per model,
hyperparameter tuning and Final Findings. It is self-contained (numpy, pandas, matplotlib, scipy; the data snapshot is
read from the GitHub repository), as the A2 specification requires.
Run from the repo root:  python notebooks/build_notebook.py          (add --execute to store the outputs)
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REPO = "tungtran0911/vol-har-conformal"
STORY = "notebooks/A2_main.py"


def percent_cells(path: str) -> list:
    """Cells of a percent script. Markdown lines lose their leading "# "; the module docstring before the first
    marker is skipped."""
    cells, kind, lines = [], None, []

    def flush():
        body = "\n".join(lines).strip("\n")
        if kind == "markdown":
            cells.append(nbf.v4.new_markdown_cell(body))
        elif kind == "code" and body:
            cells.append(nbf.v4.new_code_cell(body))

    for line in (ROOT / path).read_text(encoding="utf8").splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        elif kind == "markdown":
            lines.append(line[2:] if line.startswith("# ") else line.lstrip("#"))
        elif kind == "code":
            lines.append(line)
    flush()
    return cells


def build() -> nbf.NotebookNode:
    badge = nbf.v4.new_markdown_cell(
        f'<a href="https://colab.research.google.com/github/{REPO}/blob/main/notebooks/A2_main.ipynb" '
        'target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>')
    badge.metadata.update({"id": "view-in-github", "colab_type": "text"})
    return nbf.v4.new_notebook(cells=[badge] + percent_cells(STORY), metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    })


def main():
    nb = build()
    out = ROOT / "notebooks" / "A2_main.ipynb"
    out.parent.mkdir(exist_ok=True)
    if "--execute" in sys.argv:
        from nbclient import NotebookClient

        # run from the repo root: the local data snapshot is used and the figures land in figures/
        NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        errors = [c for c in nb.cells if c.cell_type == "code" and any(o.get("output_type") == "error" for o in c.outputs)]
        print(f"executed: {len(errors)} cell(s) with errors")
    nbf.write(nb, out)
    print(f"wrote {out} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
