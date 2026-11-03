"""Resolve validated immutable task mapping. No registry writes inside workers."""
import argparse
import json
from pathlib import Path
from tools.t04_06_registry import read, sha
from train_utils.t04_preflight import verify


def resolve(manifest, index):
    m=json.loads(Path(manifest).read_text())
    if m.get('schema') != 1 or m.get('kind') != 't04-06-e5-array' or len(m.get('entries',[])) != 8:
        raise ValueError('Invalid immutable task manifest')
    if index not in range(8):
        raise ValueError('Invalid task index')
    e=m['entries'][index]
    doc=read(m['registry'])
    registered=doc['entries'][e['id']]
    if registered['run'] != e['run'] or registered['config_sha256'] != e['config_sha256']:
        raise ValueError('Registry task identity mismatch')
    if registered['status'] not in ('submitted','running'):
        raise ValueError('Task not submitted/running in registry')
    verify(e['run'])
    return e['run']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--index',type=int,required=True)
    a=p.parse_args()
    print(resolve(a.manifest,a.index))

if __name__=='__main__':main()
