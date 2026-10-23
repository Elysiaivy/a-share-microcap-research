"""Verified reuse of immutable Tushare responses; optional official API repairs.

No reads of a data platform's private storage. Tokens never enter the request
manifest. Existing caches are read-only; every consumed response is copied here.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime, timezone

import pandas as pd
import requests


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False))


def default_sources():
    """Optional external frozen manifests, separated by the OS path separator."""
    return [Path(x).expanduser() for x in os.environ.get('MICROCAP_SOURCE_MANIFESTS', '').split(os.pathsep) if x]


class FrozenStore:
    def __init__(self, root, sources=None, online=False):
        self.root = Path(root)
        self.raw = self.root/'data/raw'
        self.raw.mkdir(parents=True, exist_ok=True)
        self.index = {}
        self.used = {}
        self.online = online
        self.token = None
        self.last_request = 0
        self.network_requests = 0
        # An already copied input takes precedence over mutable external caches.
        self.old_manifest = self.raw/'manifest.json'
        for manifest in (default_sources() if sources is None else sources):
            manifest = Path(manifest)
            if not manifest.exists():
                continue
            data = json.loads(manifest.read_text())
            for item in data.get('inputs', data.get('frozen_inputs', [])):
                req = item.get('request', {})
                if req.get('provider') != 'tushare':
                    continue
                body = req.get('body') or {}
                path = manifest.parent/item['path']
                # Project manifests store paths relative to their project root;
                # older provider manifests use paths relative to the manifest.
                if not path.exists() and manifest.parent.name == 'raw' and manifest.parent.parent.name == 'data':
                    path = manifest.parent.parent.parent/item['path']
                if not path.exists():
                    continue
                entry = dict(item, source_path=str(path), source_manifest=str(manifest))
                self.index[self.key(body)] = entry
        if self.old_manifest.exists():
            for item in json.loads(self.old_manifest.read_text())['inputs']:
                entry = dict(item, source_path=str(self.root/item['path']))
                self.index[self.key(item['request']['body'])] = entry

    @staticmethod
    def key(body):
        # Fields are a part of a query. Do not substitute subtly different schemas.
        return canonical(body).decode()

    def find(self, api, **params):
        entries = []
        for entry in self.index.values():
            body = entry['request']['body']
            if body.get('api_name') == api and all(str(body.get('params', {}).get(k)) == str(v) for k,v in params.items()):
                entries.append(entry)
        return sorted(entries, key=lambda e: (e.get('acquired_at', ''), e['source_path']))

    def read(self, entry):
        path = Path(entry['source_path'])
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if entry.get('sha256') and digest != entry['sha256']:
            raise ValueError(f'Raw file hash mismatch: {path}')
        obj = json.loads(content)
        if obj['response'].get('code') != 0:
            raise ValueError(f'Provider response not successful: {path}')
        response_hash = hashlib.sha256(canonical(obj['response'])).hexdigest()
        if response_hash != obj.get('response_sha256'):
            raise ValueError(f'Response hash mismatch: {path}')
        target = self.raw/(digest+'.json')
        if not target.exists():
            target.write_bytes(content)
        elif sha(target) != digest:
            raise ValueError(f'Frozen destination hash mismatch: {target}')
        meta = dict(path=str(target.relative_to(self.root)), sha256=digest,
                    response_sha256=response_hash, request=obj['request'],
                    acquired_at=obj.get('acquired_at'), source_path=str(path),
                    source_manifest=entry.get('source_manifest'))
        self.used[digest] = meta
        dat = obj['response']['data']
        frame = pd.DataFrame(dat.get('items', []), columns=dat.get('fields', []))
        frame.attrs['source_sha256'] = digest
        return frame

    def query(self, api, params, fields):
        body = dict(api_name=api, params=params, fields=fields)
        if self.key(body) in self.index:
            return self.read(self.index[self.key(body)])
        if not self.online:
            raise FileNotFoundError(f'Uncached official API query: {body}')
        if self.token is None:
            self.token = os.environ.get('TUSHARE_TOKEN')
            if not self.token:
                raise RuntimeError('Set TUSHARE_TOKEN in your local environment for authorized API access; never commit it.')
        delay = max(0, .6-(time.monotonic()-self.last_request))
        if delay:
            time.sleep(delay)
        response = requests.post('https://api.tushare.pro', json=dict(body, token=self.token), timeout=60)
        self.last_request = time.monotonic()
        self.network_requests += 1
        response.raise_for_status()
        result = response.json()
        if result.get('code') != 0:
            # Provider diagnostics can include sensitive strings; only expose code.
            raise RuntimeError(f'{api} failed with provider code {result.get("code")}')
        obj = dict(acquired_at=datetime.now(timezone.utc).isoformat(),
                   request=dict(provider='tushare',method='POST',url='https://api.tushare.pro',headers={},params={},body=body),
                   response=result,response_sha256=hashlib.sha256(canonical(result)).hexdigest())
        content = canonical(obj)
        path = self.raw/(hashlib.sha256(content).hexdigest()+'.json')
        path.write_bytes(content)
        entry = dict(source_path=str(path), sha256=sha(path), request=obj['request'], acquired_at=obj['acquired_at'])
        self.index[self.key(body)] = entry
        return self.read(entry)

    def date_query(self, api, day, fields, limit=6000):
        matches = self.find(api, trade_date=str(day), offset=0)
        matches = [x for x in matches if set(fields.split(',')).issubset(set(x['request']['body'].get('fields','').split(',')))]
        if matches:
            entry = matches[-1]
            frame = self.read(entry)
            if len(frame) >= int(entry['request']['body']['params'].get('limit', limit)):
                raise ValueError(f'Pagination required for {api} {day}')
            return frame
        return self.query(api, dict(trade_date=str(day),limit=limit,offset=0),fields)

    def save_manifest(self):
        previous = {}
        if self.old_manifest.exists():
            previous = {e['sha256']:e for e in json.loads(self.old_manifest.read_text())['inputs']}
        previous.update(self.used)
        dump(self.old_manifest, dict(version=1, inputs=sorted(previous.values(),key=lambda x:x['path']),
                                    network_requests_this_run=self.network_requests))
