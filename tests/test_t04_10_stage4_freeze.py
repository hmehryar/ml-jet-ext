"""CPU-only Stage 4 contract tests; no HPC directory or model loads."""
import csv
import importlib.util
from pathlib import Path
import pytest
import yaml

MODULE=Path(__file__).resolve().parents[1]/'tools/t04_10_stage4_freeze.py'
spec=importlib.util.spec_from_file_location('t04_10_stage4_freeze',MODULE)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def config(model='clip',strategy='S1',k=1,seed=42):
    return {'model':model,'strategy':strategy,'moments':list(range(1,k+1)),
            'seed':seed,'epochs':80,'training_protocol':'fixed_budget_v2',
            'moment_loss':{'type':'cross_entropy'},'split_protocol':'aggregate_disjoint',
            'checkpoint':'openai/clip-vit-base-patch16' if model=='clip' else 'google/siglip-base-patch16-224',
            'checkpoint_revision':'1'*40,'precision':'bf16',
            'optimizer':{'type':'adamw','weight_decay':0.01},
            'learning_rates':{'backbone':1e-5,'adapter':1e-4,'heads':1e-4},
            'batch_size':8,'num_workers':0,'dataset_root':'/fixture/dataset',
            'source_audit_report':'/fixture/audit.json','gpu_profile':'a100_mig_1g_10gb',
            'scheduler':{'type':'ReduceLROnPlateau','factor':0.5,'patience':4,'min_lr':1e-8}}


def test_identity_requires_ce_and_complete_protocol():
    assert m.identity(config())==('clip','S1',1,42)
    for field,value in [('epochs',40),('training_protocol','early_stop_v1'),('moment_loss',{'type':'physical_mae'})]:
        c=config();c[field]=value
        with pytest.raises(ValueError):m.identity(c)


def test_validation_history_requires_full_contiguous_epochs(tmp_path):
    path=tmp_path/'moment_epoch_metrics.csv'
    fields=list(m.REQUIRED_HISTORY)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for ep in range(1,81):
            row=dict.fromkeys(fields,'0.2');row['epoch']=str(ep)
            row['val_joint_accuracy']=str(0.5 if ep==20 else 0.4)
            w.writerow(row)
    rows=m.read_history(path)
    assert len(rows)==80 and max(range(80),key=lambda i:rows[i]['val_joint_accuracy'])==19
    with path.open('a') as f:f.write('81,0.1,0.1,0.1,0.1,0.1,0.1,0.1\n')
    with pytest.raises(ValueError,match='80 consecutive'):m.read_history(path)


def test_fails_closed_without_complete_120_run_evidence(tmp_path):
    cfgs=tmp_path/'configs'; cfgs.mkdir()
    path=cfgs/'one.yml';path.write_text(yaml.safe_dump(config()))
    audit,results,cfg=m.collect(cfgs,tmp_path/'runs')
    assert audit['complete']==0
    assert len(audit['missing_configs'])==119
    with pytest.raises(RuntimeError,match='120'):m.select(audit,results,cfg)
    assert m.freeze(cfgs,tmp_path/'runs',None,preview=True)['status']=='AUDIT'


def test_task_specific_selection_and_ties():
    result={}; cfg={}
    hashes={'train':'a'*64,'val':'b'*64,'test':'c'*64,'source_audit':'d'*64}
    for arch in m.ARCHS:
        for strategy in m.STRATEGIES:
            for k in range(1,6):
                for seed in m.SEEDS:
                    key=(arch,strategy,k,seed)
                    # Alpha prefers S2/E2; Q0 prefers S3/E4. No fake multi-task winner.
                    a=0.9 if (strategy,k)==('S2',2) else 0.5
                    q=0.85 if (strategy,k)==('S3',4) else 0.4
                    result[key]={'metrics':{'val_alpha_macro_f1':a,'val_q0_macro_f1':q,
                         'val_alpha_accuracy':a,'val_q0_accuracy':q,
                         'val_alpha_mae':1-a,'val_q0_mae':1-q},'input_hashes':hashes,'source_overlap':103221}
                    cfg[key]=(None,config(arch,strategy,k,seed))
    audit={'complete':120,'missing_configs':[],'extra_configs':[],
           'missing_runs':[],'bad_runs':[],'config_errors':[]}
    _,winners,_=m.select(audit,result,cfg)
    for arch in m.ARCHS:
        assert (winners[(arch,'alpha_only')]['strategy'],winners[(arch,'alpha_only')]['E'])==('S2',2)
        assert (winners[(arch,'q0_only')]['strategy'],winners[(arch,'q0_only')]['E'])==('S3',4)
    result[('clip','S1',1,42)]['input_hashes']={'train':'bad'}
    with pytest.raises(RuntimeError,match='SHA mismatch'):m.select(audit,result,cfg)


def test_translated_run_is_isolated_and_preserves_protocol():
    c=config('siglip','S3',4,42)
    d=m.translate(c,'siglip','q0_only',123)
    assert d['seed']==123 and d['task']=='q0_only'
    assert d['moments']==[1,2,3,4]
    assert 't04_10_single_head' in d['output_root']
    assert d['epochs']==80 and d['split_protocol']=='aggregate_disjoint'
    assert set(d)=={'model','checkpoint','checkpoint_revision','strategy','moments','precision',
                    'optimizer','learning_rates','batch_size','num_workers','dataset_root',
                    'source_audit_report','split_protocol','gpu_profile','epochs',
                    'training_protocol','task','seed','output_root','scheduler'}


def test_inspect_run_scores_actual_joint_checkpoint_epoch(tmp_path):
    import json
    path=tmp_path/'source.yml';c=config();c['fold']=None
    root=tmp_path/'runs'
    c['output_root']=str(root.resolve())
    path.write_text(yaml.safe_dump(c))

    # Reproduce the T04-06 production directory layout.
    run=root/'clip'/'S1'/'E1'/'seed-42'/'synthetic-run'
    (run/'source').mkdir(parents=True)

    # Historical run used the relative output directory.
    original_cfg=dict(c)
    original_cfg['output_root']='training_output/T04/t04_06_v2_production'
    original=run/'source/config.original.yml'
    original.write_text(yaml.safe_dump(original_cfg))
    resolved=run/'config.resolved.json'
    resolved.write_text(json.dumps(original_cfg))
    inp={key:{'sha256':value*64} for key,value in
         [('train','a'),('val','b'),('test','c'),('source_audit','d')]}
    manifest={'schema_version':2,'test_set_evaluated':False,
              'source_hashes':{'config.original.yml':m.sha(original)},
              'configuration_sha256':m.sha(resolved),'configuration':original_cfg,
              'git':{'commit':'a'*40},
              'inputs':{'input_records':inp,'global_cross_split_overlap':103221,
                        'source_overlap_independence_claim':False}}
    (run/'run_manifest.json').write_text(json.dumps(manifest))
    with (run/'moment_epoch_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=m.REQUIRED_HISTORY);writer.writeheader()
        for ep in range(1,81):
            row={col:'0.2' for col in m.REQUIRED_HISTORY}
            row.update({'epoch':ep,'val_joint_accuracy':0.9 if ep==15 else 0.3,
                        'val_alpha_macro_f1':0.4 if ep==15 else 0.99})
            writer.writerow(row)
    ckpt=run/'sessions/s1/checkpoints/best_epoch_000015.pth'
    ckpt.parent.mkdir(parents=True);ckpt.write_bytes(b'synthetic checkpoint')
    result=m.inspect_run(run,path,c)
    assert result['best_epoch']==15
    assert result['metrics']['val_alpha_macro_f1']==0.4  # not cherry-picked .99
    assert result['source_overlap']==103221
    assert result['input_hashes']['train']=='a'*64
    # Manifest hash tampering must fail closed.
    manifest['configuration_sha256']='f'*64
    (run/'run_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='hash mismatch'):
        m.inspect_run(run,path,c)
