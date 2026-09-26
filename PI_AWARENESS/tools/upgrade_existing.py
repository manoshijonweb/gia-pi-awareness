#!/usr/bin/env python3
"""Apply this source release to a separate existing PI_AWARENESS installation.
Stop the service first. Preserves camera/audio/OCR settings, .venv, model weights,
logs and saved calibration. Switches detector settings to YOLO26m at 640.
Does NOT install packages, download models, or restart any service.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile

SOURCE=Path(__file__).resolve().parents[1]
DETECTOR_LINES={
 'detector':'    detector: str = "yolo26m"',
 'object_model':'    object_model: Path = ROOT / "models/yolo26m_640.onnx"',
 'object_input_size':'    object_input_size: int = 640',
 'object_format':'    object_format: str = "community_yolos"',
 'object_threads':'    object_threads: int = 3',
}


def updated_settings(text: str) -> str:
    tree=ast.parse(text)
    classes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='VisionSettings']
    if len(classes)!=1:
        raise ValueError('Expected one VisionSettings class; refusing to guess how to edit this configuration')
    cls=classes[0];seen=set();edits=[]
    for node in cls.body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id in DETECTOR_LINES:
            name=node.target.id;seen.add(name)
            edits.append((node.lineno-1,node.end_lineno,[DETECTOR_LINES[name]+'\n']))
    for required in ('detector','object_model','object_input_size'):
        if required not in seen:raise ValueError(f'Missing expected VisionSettings field {required}; no files changed')
    missing=[line+'\n' for name,line in DETECTOR_LINES.items() if name not in seen]
    if missing:edits.append((cls.end_lineno,cls.end_lineno,missing))
    lines=text.splitlines(keepends=True)
    for start,end,replacement in sorted(edits,reverse=True):lines[start:end]=replacement
    result=''.join(lines);ast.parse(result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',required=True,type=Path,help='Existing repo directory, not this extracted update directory')
    args=p.parse_args();target=args.target.expanduser().resolve()
    if target==SOURCE:p.error('Extract this release separately and point --target at the existing repo')
    if not (target/'server.py').is_file() or not (target/'settings.py').is_file():p.error('Target is not an existing PI_AWARENESS repo')
    merged=updated_settings((target/'settings.py').read_text(encoding='utf-8'))
    sources=[]
    for file in sorted(SOURCE.rglob('*')):
        rel=file.relative_to(SOURCE)
        if not file.is_file() or file.is_symlink():continue
        if any(part in ('.venv','__pycache__','.git','state','logs') for part in rel.parts):continue
        if rel.parts[0]=='models' and rel!=Path('models/README.md'):continue
        if rel==Path('settings.py') or file.suffix=='.pyc':continue
        dest=target/rel
        if dest.is_symlink() or any(parent.is_symlink() for parent in dest.parents if parent!=target.parent):
            p.error(f'Refusing symlink destination {dest}')
        sources.append((file,rel))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=target/'state'/'upgrade_backups'/stamp;backup.mkdir(parents=True)
    shutil.copy2(target/'settings.py',backup/'settings.py')
    for file,rel in sources:
        dest=target/rel
        if dest.is_file():
            old=backup/rel;old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest,old)
        dest.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent,delete=False) as temp:
            staged=Path(temp.name)
        shutil.copy2(file,staged);staged.replace(dest)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=target,delete=False) as temp:
        temp.write(merged);staged=Path(temp.name)
    staged.chmod((target/'settings.py').stat().st_mode & 0o777)
    staged.replace(target/'settings.py')
    print(f'Source update complete. Previous changed files backed up at {backup}')
    print('Preserved hardware settings, calibration, .venv and installed models. Detector now YOLO26m/640.')
    print('From the target directory, run:')
    print('  .venv/bin/python tools/download_models.py --only-detector')
    print('  .venv/bin/python server.py --doctor --deep --hardware')
    print('  .venv/bin/python tools/benchmark.py --features scan --iterations 20')
    print('Stop the service before hardware tests; restart it only after successful acceptance.')

if __name__=='__main__':main()
