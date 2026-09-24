"""Execute the original notebook in a fresh kernel and preserve verified outputs."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in ("ipython", "jupyter", "matplotlib"):
    (ROOT / ".cache" / directory).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".cache/ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".cache/jupyter"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import nbformat
from nbclient import NotebookClient

notebook = nbformat.read(ROOT / "AutoEncoder.ipynb", as_version=4)
client = NotebookClient(notebook, timeout=180, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(notebook, ROOT / "AutoEncoder.ipynb")
for cell in notebook.cells:
    for output in cell.get("outputs", []):
        if output.output_type == "stream" and "Initial loss:" in output.text:
            print(output.text, end="")
print("All notebook cells executed successfully; outputs saved in AutoEncoder.ipynb.")
