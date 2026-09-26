#!/usr/bin/env python3
"""One-time, explicit network provisioning. Standard library only.
Runtime never calls this script. --verify is entirely local.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_manifest import ASSETS, Asset, DETECTOR_ASSETS, assets_for
from awareness.common import atomic_json


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def asset_files(target: Path) -> list[Path]:
    if target.is_file() and not target.is_symlink():
        return [target]
    if target.is_dir() and not target.is_symlink():
        paths = sorted(target.rglob('*'))
        if any(p.is_symlink() for p in paths):
            raise ValueError('Refusing symlinks in a model directory')
        return [p for p in paths if p.is_file()]
    return []


def safe_extract(archive: Path, destination: Path, expected_root: str,
                 maximum_bytes: int = 512*1024*1024) -> None:
    """Reject traversal, symlinks, extra top-level folders and oversized archives."""
    with zipfile.ZipFile(archive) as zipped:
        items = zipped.infolist()
        if len(items) > 20000 or sum(i.file_size for i in items) > maximum_bytes:
            raise ValueError('Model archive exceeds extraction limits')
        names = set()
        for item in items:
            name = PurePosixPath(item.filename)
            if ('\\' in item.filename or name.is_absolute() or '..' in name.parts
                    or not name.parts or name.parts[0] != expected_root
                    or ':' in item.filename or '\x00' in item.filename
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError('Unsafe model archive member: '+item.filename)
            normalized = str(name)
            if normalized in names:
                raise ValueError('Duplicate model archive member: '+item.filename)
            names.add(normalized)
        for item in items:
            path = destination.joinpath(*PurePosixPath(item.filename).parts)
            # destination is always a newly created private temporary directory.
            if item.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(item) as source, path.open('xb') as target:
                    shutil.copyfileobj(source, target, length=1024*1024)


def download(asset: Asset, path: Path) -> str:
    if not asset.url.startswith('https://'):
        raise ValueError('Only HTTPS model URLs are accepted')
    error = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(asset.url, headers={'User-Agent': 'PI_AWARENESS-provision/1.1'})
            total, digest = 0, hashlib.sha256()
            with urllib.request.urlopen(req, timeout=60) as response, path.open('wb') as stream:
                if not response.geturl().startswith('https://'):
                    raise ValueError('Refusing HTTPS-to-HTTP redirect')
                content_type = response.headers.get('Content-Type', '').lower()
                if 'text/html' in content_type or 'application/json' in content_type:
                    raise ValueError('Server returned a page/error instead of model bytes')
                for chunk in iter(lambda: response.read(1024*1024), b''):
                    total += len(chunk)
                    if total > asset.maximum_bytes:
                        raise ValueError('Download exceeded size limit')
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            if total < 100000:
                raise ValueError('Downloaded asset is too small to be the requested model')
            actual = digest.hexdigest()
            if asset.sha256 and actual != asset.sha256:
                raise ValueError(f'SHA256 mismatch for {asset.name}; expected {asset.sha256}, got {actual}')
            print(f'  received {total/1024/1024:.1f} MiB; SHA256 {actual}', flush=True)
            return actual
        except Exception as exc:
            error = exc
            path.unlink(missing_ok=True)
            print(f'  attempt {attempt+1}/3 failed: {exc}', file=sys.stderr, flush=True)
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError(f'Could not fetch {asset.name}: {error}')


def is_valid(asset: Asset, destination: Path, record: dict) -> bool:
    target = destination / asset.target
    try:
        files = asset_files(target)
        if not files:
            return False
        if asset.archive_root and not (target/'am/final.mdl').is_file():
            return False
        if asset.sha256 and not asset.archive_root:
            return sha256(target) == asset.sha256
        previous = record.get(asset.target, {})
        saved = previous.get('files', {})
        if previous.get('source_url') != asset.url or not saved:
            return False
        current = {str(p.relative_to(destination)): sha256(p) for p in files}
        return current == saved
    except (OSError, ValueError):
        return False


def provision(destination: Path, verify: bool = False, force: bool = False,
              assets: tuple[Asset, ...] | None = None) -> int:
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    lockfile = destination/'installed_manifest.json'
    try:
        record = json.loads(lockfile.read_text(encoding='utf-8'))
        if not isinstance(record, dict):
            record = {}
    except (OSError, ValueError):
        record = {}
    failures = []
    selected = ASSETS if assets is None else assets
    for asset in selected:
        if is_valid(asset, destination, record) and not (force and not verify):
            print('OK   '+asset.name)
            continue
        if verify:
            failures.append(asset.name)
            print('FAIL '+asset.name+' is missing, changed, or has no installation hash record')
            continue
        print('GET  '+asset.name, flush=True)
        try:
            with tempfile.TemporaryDirectory(prefix='.install-', dir=destination) as td:
                staging = Path(td)
                payload = staging/'payload.bin'
                source_hash = download(asset, payload)
                target = destination/asset.target
                if asset.archive_root:
                    safe_extract(payload, staging/'unpack', asset.archive_root)
                    source = staging/'unpack'/asset.archive_root
                    if not (source/'am/final.mdl').is_file():
                        raise ValueError('Archive lacks the Vosk acoustic model am/final.mdl')
                    if target.exists():
                        if target.is_symlink():
                            raise ValueError('Refusing to replace a symlink')
                        shutil.rmtree(target)
                    os.replace(source, target)
                else:
                    if target.is_symlink():
                        raise ValueError('Refusing to replace a symlink')
                    os.replace(payload, target)
                record[asset.target] = {
                    'source_url': asset.url, 'source_sha256': source_hash,
                    'upstream_sha256_pinned': bool(asset.sha256),
                    'files': {str(p.relative_to(destination)): sha256(p) for p in asset_files(target)},
                    'installed_unix': time.time(),
                }
                atomic_json(lockfile, record)
        except Exception as exc:
            failures.append(asset.name)
            print(f'FAIL {asset.name}: {exc}', file=sys.stderr)
    if failures:
        print('Incomplete assets: '+', '.join(failures), file=sys.stderr)
        print('Retry this script with a network connection; inference will NOT silently download models.', file=sys.stderr)
        return 1
    print(f'All {len(selected)} selected assets verified locally. Next: .venv/bin/python server.py --doctor --deep')
    return 0


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT/'models')
    p.add_argument('--verify',action='store_true',help='Check local files; make no network requests')
    p.add_argument('--force',action='store_true',help='Explicitly re-download all assets')
    p.add_argument('--list',action='store_true',help='Print public URLs and targets only')
    p.add_argument('--detector', choices=tuple(DETECTOR_ASSETS), help='Model to provision; defaults to settings.py')
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--only-detector', action='store_true', help='Download/verify only the selected detector, not OCR/Vosk')
    selection.add_argument('--skip-detector', action='store_true', help='OCR/Vosk only; keep a separately supplied native detector export')
    args=p.parse_args()
    from settings import CONFIG
    try:
        selected=assets_for(args.detector or CONFIG.vision.detector, args.only_detector, args.skip_detector)
    except ValueError as exc:
        p.error(str(exc))
    if args.list:
        for a in selected:
            print(f'{a.target}\n  {a.url}\n  upstream SHA256: {a.sha256 or "not pinned; local integrity record only"}')
        return 0
    return provision(args.directory,args.verify,args.force,selected)

if __name__=='__main__':
    raise SystemExit(main())
