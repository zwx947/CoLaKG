"""Create a NEW replay freeze after verifying canonical text against Git.

Windows checkout can convert LF/CRLF. Never rewrite data, source, or historical
freeze records. Reject any semantic source/data text change or binary asset change.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

ROOT=Path(__file__).resolve().parents[1]
BASE_COMMIT='06c2035bd5261fba7a0442ab6110727ce6d1f605'
HISTORICAL='experiments/frozen_pair_geometry_v1.json'


def digest(data):return hashlib.sha256(data).hexdigest()


def git_bytes(path):
    return subprocess.check_output(['git','-C',str(ROOT),'show',BASE_COMMIT+':'+path.replace('\\','/')])


def canonical(data):return data.replace(b'\r\n',b'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    old=json.loads(git_bytes(HISTORICAL).decode('utf-8'))
    assert json.loads((ROOT/HISTORICAL).read_text(encoding='utf-8'))==old
    record=copy.deepcopy(old)
    equivalence={}
    for name in old['source_sha256']:
        actual=(ROOT/name).read_bytes();original=git_bytes(name)
        assert canonical(actual)==canonical(original),'Source changed beyond line endings: '+name
        record['source_sha256'][name]=digest(actual)
        equivalence[name]=digest(canonical(actual))
    for name,expected in old['original_asset_sha256'].items():
        actual=(ROOT/name).read_bytes()
        if Path(name).suffix=='.txt':
            original=git_bytes(name)
            assert canonical(actual)==canonical(original),'Data changed beyond line endings: '+name
            equivalence[name]=digest(canonical(actual))
        else:
            assert digest(actual)==expected,'Binary semantic asset changed: '+name
        record['original_asset_sha256'][name]=digest(actual)
    protocol=Path(r'D:\CoLaKGResearch\protocol-v1\manifest.json')
    if protocol.exists():
        derived=json.loads(protocol.read_text(encoding='utf-8'))
        assert derived['original_train_sha256']==record['original_asset_sha256'][str(Path('data/lastfm/train.txt'))], \
            'Existing derived protocol has different train bytes; preserve it and restore the recorded data checkout format; never overwrite it'
    record['created_at']=datetime.now().isoformat()
    record['code_commit_before_freeze']=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    record['replay_provenance']={'historical_commit':BASE_COMMIT,'historical_freeze_git_blob_sha256':digest(git_bytes(HISTORICAL)),
                                 'canonical_lf_sha256':equivalence,
                                 'allowed_change':'LF/CRLF representation only; hyperparameters copied unchanged; no file rewritten'}
    out=args.output or Path(r'D:\CoLaKGResearch')/('frozen-replay-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.json')
    out=out.resolve()
    with out.open('x',encoding='utf-8') as stream:json.dump(record,stream,ensure_ascii=False,indent=2)
    print('REPLAY_FREEZE='+str(out));print('SHA256='+digest(out.read_bytes()))


if __name__=='__main__':main()
