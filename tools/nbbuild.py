"""Build a notebook from a percent-format source (`# %%` code cells, `# %% [markdown]` cells).

    uv run python tools/nbbuild.py SRC.py OUT.ipynb      # then: make notebooks (executes in place)
"""

import sys

import nbformat

src, out = sys.argv[1], sys.argv[2]
cells, kind, buf = [], None, []


def flush():
    if kind is None:
        return
    text = "\n".join(buf).strip("\n")
    if kind == "markdown":
        text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in text.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    elif text:
        cells.append(nbformat.v4.new_code_cell(text))


for line in open(src, encoding="utf-8").read().splitlines():
    if line.startswith("# %%"):
        flush()
        kind, buf = ("markdown" if "[markdown]" in line else "code"), []
    else:
        buf.append(line)
flush()
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3 (novaretail)", "language": "python", "name": "python3"}
nbformat.write(nb, out)
print(f"{out}: {len(cells)} cells")
