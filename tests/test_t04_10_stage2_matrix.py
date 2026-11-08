import json
import yaml
import pytest
from tools.t04_10_generate_matrix import generate


def fixture(path, model):
    c = {'model':model, 'moment_loss':{'type':'cross_entropy'},
         'training_protocol':'fixed_budget_v2','epochs':80,
         'checkpoint':model,'checkpoint_revision':'f'*40,
         'strategy':'S4','moments':[1,2,3],'precision':'bf16',
         'optimizer':{'type':'adamw','weight_decay':0.01},
         'learning_rates':{'backbone':1e-5,'adapter':1e-4,'heads':1e-4},
         'batch_size':8,'num_workers':2,'dataset_root':'/dataset',
         'source_audit_report':'/audit','split_protocol':'aggregate_disjoint',
         'gpu_profile':'a100_mig_1g_10gb',
         'scheduler':{'type':'ReduceLROnPlateau','factor':0.5,'patience':4,'min_lr':1e-8}}
    path.write_text(yaml.safe_dump(c))
    return path


def test_matrix_has_twelve_candidates_and_six_blocked(tmp_path):
    c = fixture(tmp_path/'clip.yml','clip')
    s = fixture(tmp_path/'siglip.yml','siglip')
    result = generate(c,s,tmp_path/'matrix')
    assert len(result['entries']) == 18
    assert len(list((tmp_path/'matrix').glob('*.yml'))) == 12
    assert {x['status'] for x in result['entries']} == {'NOT_PREPARED','BLOCKED_VIT_PARITY'}
    for x in result['entries']:
        if x['status'] == 'NOT_PREPARED':
            config = yaml.safe_load(open(x['config']))
            assert config['task'] in ('alpha_only','q0_only')
            assert config['epochs'] == 80
            assert config['output_root'] != '/dataset'


def test_generator_does_not_overwrite(tmp_path):
    c = fixture(tmp_path/'clip.yml','clip')
    s = fixture(tmp_path/'siglip.yml','siglip')
    out = tmp_path/'matrix'
    generate(c,s,out)
    with pytest.raises(FileExistsError):
        generate(c,s,out)


def test_generator_rejects_non_ce(tmp_path):
    c = fixture(tmp_path/'clip.yml','clip')
    s = fixture(tmp_path/'siglip.yml','siglip')
    cfg = yaml.safe_load(c.read_text())
    cfg['moment_loss'] = {'type':'ce_ordinal','ordinal_weight':0.1}
    c.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='CE dual-head'):
        generate(c,s,tmp_path/'matrix')
