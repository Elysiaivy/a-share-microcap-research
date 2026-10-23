"""Fail before a full-data run can overwrite published research artifacts."""
from pathlib import Path
import json
import hashlib


def require_full_inputs(root, stage='all', online=False):
    root = Path(root)
    required = ['data/raw/manifest.json', 'data/repair_sources/early/market_cap.csv']
    if stage not in ('prepare', 'all'):
        required += ['data/prepared/manifest.json', 'data/prepared/panel.csv',
                     'data/prepared/universe_coverage.csv',
                     'data/prepared/selection_universe.csv',
                     'data/prepared/daily_market.npz',
                     'data/prepared/daily_market_manifest.json']
    missing = [name for name in required if not (root/name).is_file()]
    if not (root/'data/repair_sources/point_in_time').is_dir():
        missing.append('data/repair_sources/point_in_time/')
    if missing:
        raise FileNotFoundError(
            'Full historical-input bundle is required; the public checkout supports result recomputation only. '
            'See RUN_GUIDE.md. Missing: ' + ', '.join(missing))
    manifest = json.loads((root/'data/raw/manifest.json').read_text())
    inputs = manifest.get('inputs', [])
    if not inputs:
        raise ValueError('Empty frozen raw-input manifest')
    for item in inputs:
        path = (root/item['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Raw manifest path escapes the project root')
        if not path.is_file():
            raise FileNotFoundError(f'Missing frozen raw input: {item["path"]}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Raw input checksum mismatch: {item["path"]}')
    # This verifies presence and hashes, not vendor licensing or point-in-time validity.
    return len(inputs)
