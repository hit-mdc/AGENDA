"""Configuration-driven registration of external Python adapters."""
import hashlib
import importlib
import json
import os
from pathlib import Path


def read_config(path=None):
    path=Path(path or os.environ.get('AGENDA_CONFIG','config.json')).resolve()
    with path.open(encoding='utf-8') as source: config=json.load(source)
    config['_base']=str(path.parent)
    config['_fingerprint']=hashlib.sha256(path.read_bytes()).hexdigest()
    for field in ('datasets','methods','indexes'):
        entries=config.get(field,[])
        if not entries or len({e['id'] for e in entries})!=len(entries):
            raise ValueError(f'{field} must contain unique nonempty registrations')
        for entry in entries:
            if not entry['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in entry['id']):
                raise ValueError('Registration IDs must use letters, digits, underscores, or hyphens')
    return config


def resolve(config, value):
    path=Path(value)
    return path if path.is_absolute() else Path(config['_base'])/path


def create(entry):
    module, name=entry['factory'].split(':',1)
    return getattr(importlib.import_module(module),name)(**entry.get('options',{}))
