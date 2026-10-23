"""Refresh checksums only after reviewing and verifying intentional release edits."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
ignored={'.git','.venv','.build','.doc_build','__pycache__','.ipynb_checkpoints','data'}
files={}
for p in sorted(ROOT.rglob('*')):
    rel=p.relative_to(ROOT)
    if not p.is_file() or any(part in ignored for part in rel.parts):continue
    if p.name in ['PUBLIC_MANIFEST.json','.DS_Store'] or p.suffix in ['.pyc','.log']:continue
    files[str(rel)]=hashlib.sha256(p.read_bytes()).hexdigest()
(ROOT/'PUBLIC_MANIFEST.json').write_text(json.dumps({'scope':'public release files, not omitted historical raw data','files':files},ensure_ascii=False,indent=2)+'\n')
print('Public files:',len(files))
