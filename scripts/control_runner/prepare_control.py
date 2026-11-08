#!/usr/bin/env python3
"""Prepare an isolated, private FBroker run without evaluating or using network."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ADAPTERS = ('run.py', 'retry_errors.py', 'retry_consistency.py', 'run_control.py')
CRITERIA = {'contact', 'needs', 'identity', 'consultation', 'empathy', 'management', 'closing', 'gross_error'}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepared_files(source, checklist_path, project):
    # Import only core data validation. Do not import inference adapters here.
    sys.path.insert(0, str(project))
    from cqa.domain import normalize_conversation, validate_checklist
    payload = json.loads(source.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict) or payload.get('isSuccess') is not True:
        raise ValueError('Need the original successful FBroker API object')
    data = payload.get('data')
    if not isinstance(data, dict) or data.get('dataset_group_id') != 'FBroker':
        raise ValueError('Only FBroker is accepted; SWAP must not be evaluated here')
    records = data.get('records')
    if type(data.get('total_records')) is not int or data['total_records'] != 10 or not isinstance(records, list) or len(records) != 10:
        raise ValueError('FBroker must contain exactly 10 source records')
    normalized = [normalize_conversation(row) for row in records]
    if len({row['conversation_id'] for row in normalized}) != 10:
        raise ValueError('Duplicate conversation IDs')
    if any(row['channel'] != 'agent chat' for row in normalized):
        raise ValueError('FBroker control input requires 10 chats')
    checklist_bytes = checklist_path.read_bytes()
    checklist = validate_checklist(json.loads(checklist_bytes))
    if {rule['id'] for rule in checklist['criteria']} != CRITERIA:
        raise ValueError('Need the original eight-criterion main checklist snapshot')
    files = {
        'conversations.json': json.dumps(normalized, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'),
        'checklist.json': checklist_bytes,
    }
    for name in ADAPTERS:
        path = HERE / name
        if path.is_symlink(): raise ValueError('Symlink adapter refused')
        files[name] = path.read_bytes()
    for folder in ('cqa', 'config'):
        for path in sorted((project/folder).rglob('*')):
            if '__pycache__' in path.parts or path.suffix not in ('.py', '.json'): continue
            if path.is_symlink(): raise ValueError('Symlink source refused')
            if path.is_file(): files[str(path.relative_to(project))] = path.read_bytes()
    # Pin code to the actual submitted control protocol, not silently to a later checkout.
    frozen = json.loads((HERE/'frozen_code_sha256.json').read_text())
    for name, expected in frozen['sha256'].items():
        if name not in files or sha256(files[name]) != expected:
            raise ValueError('Code differs from frozen control version: '+name)
    return files, frozen


def prepare(source, checklist, output, project):
    files, frozen = prepared_files(source, checklist, project)
    if output.is_symlink(): raise ValueError('Symlink output refused')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError('Output must be absent or empty; no files will be overwritten')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'-', dir=output.parent))
    try:
        os.chmod(staging, 0o700)
        for name, content in files.items():
            path = staging/name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.write_bytes(content); os.chmod(path, 0o600)
        manifest = {'model':frozen['model'], 'review':'auto', 'num_ctx':32768,
                    'sha256':{name:sha256(content) for name,content in sorted(files.items())}}
        path = staging/'manifest.json'
        path.write_text(json.dumps(manifest, indent=2)+'\n'); os.chmod(path, 0o600)
        # rename refuses a concurrently populated destination instead of replacing its files.
        staging.replace(output)
    finally:
        if staging.exists(): shutil.rmtree(staging)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checklist', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--project', type=Path, default=HERE.parents[1])
    parser.add_argument('--validate-only', action='store_true', help='Validate source/checklist/frozen code, write nothing, no model calls')
    args = parser.parse_args()
    if args.validate_only:
        files,_ = prepared_files(args.source,args.checklist,args.project.resolve())
        print(json.dumps({'offline_validated':True,'records':10,'files':len(files),'network_calls':0}))
        return
    if args.output is None: parser.error('--output is required unless --validate-only')
    result = prepare(args.source,args.checklist,args.output,args.project.resolve())
    print(json.dumps({'prepared':str(args.output),'records':10,'frozen_files':len(result['sha256']),'network_calls':0}))


if __name__ == '__main__':
    try: main()
    except (ValueError, OSError, TypeError, KeyError) as error:
        raise SystemExit('Preparation refused: '+str(error))
