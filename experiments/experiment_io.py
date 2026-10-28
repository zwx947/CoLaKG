"""Atomic experiment files on Windows; no RNG use in retry or serialization."""
import json
import os
from lastfm_windows import replace_with_retry


def atomic_save(torch, path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path, path.parent / 'replace-retries.jsonl')


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    replace_with_retry(temporary, path, path.parent / 'replace-retries.jsonl')
