import json
from pathlib import Path
import pytest
from tools.t04_06_launcher import selected, intent_path, array_manifest


def test_selected_order():
    doc={'entries':{f'{m}_{s.lower()}_e5_seed42':{'id':f'{m}_{s.lower()}_e5_seed42','run':f'/r/{m}/{s}','config_sha256':'a'*64} for m in ('clip','siglip') for s in ('S1','S2','S3','S4')}}
    batch=selected(doc)
    assert len(batch)==8
    assert [x['id'] for x in batch]==['clip_s1_e5_seed42','clip_s2_e5_seed42','clip_s3_e5_seed42','clip_s4_e5_seed42','siglip_s1_e5_seed42','siglip_s2_e5_seed42','siglip_s3_e5_seed42','siglip_s4_e5_seed42']
    m=array_manifest(Path('/tmp/registry.runtime.json'),batch)
    assert m['entries'][7]['id']=='siglip_s4_e5_seed42'


def test_missing_experiment_fail_closed():
    with pytest.raises(ValueError,match='Missing'):
        selected({'entries':{}})


def test_intent_is_sibling():
    assert intent_path('/r/registry.runtime.json')==Path('/r/stage3.e5.seed42.intent.json')
