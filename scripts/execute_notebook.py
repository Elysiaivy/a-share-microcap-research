"""Execute only the public result-recomputation notebook, using this Python."""
from pathlib import Path
import os
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('IPYTHONDIR',str(Path(tempfile.gettempdir())/'microcap-ipython'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.build/matplotlib'))
path = ROOT/'research.ipynb'
nb = nbformat.read(path,as_version=4)
km = KernelManager(kernel_name='python3')
km._kernel_spec = KernelSpec(argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
                             display_name='Python 3',language='python')
try:
    NotebookClient(nb,timeout=180,km=km,resources={'metadata':{'path':str(ROOT)}}).execute()
finally:
    if km.has_kernel:
        km.shutdown_kernel(now=True)
nbformat.write(nb,path)
print(f'Executed {sum(c.cell_type == "code" for c in nb.cells)} public code cells without errors')
