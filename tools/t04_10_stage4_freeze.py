"""T04-10 Stage 4: validation-only, fail-closed T04-06 baseline audit and freeze.

No GPU imports, no job submissions, and no mutations of T04-06/T04-07 runs.
Requires a complete 120-run 3-seed T04-06 CE grid to freeze. Python 3.9.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

SEEDS = (42, 123, 2026)
ARCHS = ('clip', 'siglip')
TASKS = ('alpha_only', 'q0_only')
STRATEGIES = ('S1', 'S2', 'S3', 'S4')
OUTPUT = '/rs/rs_grp_mljet/ml-jet-ext/training_output/T04/t04_10_single_head'
REQUIRED_HISTORY = ('epoch','val_joint_accuracy','val_alpha_macro_f1','val_q0_macro_f1',
                    'val_alpha_accuracy','val_q0_accuracy','val_alpha_mae','val_q0_mae')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda:f.read(1024*1024), b''):
            h.update(buf)
    return h.hexdigest()


def load_yml(path):
    d = yaml.safe_load(Path(path).read_text())
    if not isinstance(d, dict):
        raise ValueError('Config is not a mapping: '+str(path))
    return d


def identity(cfg):
    k = len(cfg['moments'])
    if (cfg.get('model') not in ARCHS or cfg.get('strategy') not in STRATEGIES
        or cfg.get('seed') not in SEEDS or cfg['moments'] != list(range(1,k+1))
        or k not in range(1,6) or cfg.get('epochs') != 80
        or cfg.get('training_protocol') != 'fixed_budget_v2'
        or cfg.get('moment_loss') != {'type':'cross_entropy'}
        or cfg.get('split_protocol') not in ('aggregate_disjoint','source_disjoint')):
        raise ValueError('Not eligible T04-06 CE fixed-budget configuration')
    return (cfg['model'], cfg['strategy'], k, cfg['seed'])


def read_history(path):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        if not set(REQUIRED_HISTORY) <= set(reader.fieldnames or ()):
            raise ValueError('Missing validation columns: '+str(path))
        rows = list(reader)
    if len(rows)!=80 or [int(row['epoch']) for row in rows] != list(range(1,81)):
        raise ValueError('Not 80 consecutive epochs: '+str(path))
    for row in rows:
        for key in REQUIRED_HISTORY[1:]:
            x=float(row[key])
            if not math.isfinite(x):
                raise ValueError('Non-finite '+key+': '+str(path))
            row[key]=x
    return rows


def inspect_run(run, cfg_path, cfg):
    run = Path(run)
    original = run/'source/config.original.yml'
    manifest_file = run/'run_manifest.json'
    resolved = run/'config.resolved.json'
    history_path = run/'moment_epoch_metrics.csv'
    for p in (original,manifest_file,resolved,history_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    # T04-06 original snapshots used a relative output root.
    # The published configs later relocated outputs to /rs.
    # Permit exactly this operational difference, and nothing else.
    original_cfg = load_yml(original)
    expected_root = run.parents[4].resolve()
    published_root = Path(cfg['output_root']).expanduser().resolve()
    if (
        original_cfg.get('output_root')
        != 'training_output/T04/t04_06_v2_production'
        or published_root != expected_root
    ):
        raise ValueError(
            'Unexpected output-root provenance: '
            f'run={run}, original={original_cfg.get("output_root")!r}, '
            f'published={cfg.get("output_root")!r}, '
            f'expected={str(expected_root)!r}'
        )

    old_normalized = dict(original_cfg)
    new_normalized = dict(cfg)
    old_normalized.pop('output_root')
    new_normalized.pop('output_root')

    if old_normalized != new_normalized:
        raise ValueError('Training configuration differs: '+str(run))

    manifest=json.loads(manifest_file.read_text())
    original_sha = sha(original)
    recorded_sha = manifest.get('source_hashes', {}).get(
        'config.original.yml'
    )
    if original_sha != recorded_sha:
        raise ValueError('Immutable source YAML hash mismatch: '+str(run))
    if manifest.get('schema_version')!=2 or manifest.get('test_set_evaluated') is not False:
        raise ValueError('Untrusted manifest/test-evaluation state: '+str(run))
    if manifest.get('configuration_sha256') != sha(resolved):
        raise ValueError('Resolved config hash mismatch: '+str(run))
    rconfig=json.loads(resolved.read_text())
    if (
        identity(rconfig) != identity(cfg)
        or rconfig != original_cfg
        or manifest.get('configuration') != rconfig
    ):
        raise ValueError('Run config differs from immutable source: '+str(run))
    if not isinstance(manifest.get('git',{}).get('commit'), str) or not re.fullmatch('[0-9a-f]{40}',manifest['git']['commit']):
        raise ValueError('Missing run commit provenance: '+str(run))
    inputs=manifest.get('inputs',{})
    records=inputs.get('input_records',{})
    if inputs.get('source_overlap_independence_claim') is not False:
        raise ValueError('Source independence not explicitly disclosed as false: '+str(run))
    if not isinstance(inputs.get('global_cross_split_overlap'),int):
        raise ValueError('Missing source overlap count: '+str(run))
    # T04-06 records hashes under inputs.input_records, not directly under inputs.
    hashes={}
    for key in ('train','val','test','source_audit'):
        r=records.get(key)
        if not isinstance(r,dict) or not re.fullmatch('[0-9a-f]{64}',str(r.get('sha256',''))):
            raise ValueError('Missing input hash '+key+': '+str(run))
        hashes[key]=r['sha256']
    rows=read_history(history_path)
    # Existing joint-head training chooses first epoch with the highest val joint accuracy.
    best=max(range(80),key=lambda i: rows[i]['val_joint_accuracy'])
    checkpoint=run/'sessions'
    best_name='best_epoch_%06d.pth'%(best+1)
    best_paths=list(checkpoint.glob('*/checkpoints/'+best_name)) if checkpoint.is_dir() else []
    if not best_paths:
        raise ValueError('Missing original joint-selected best checkpoint '+best_name+': '+str(run))
    row=rows[best]
    return {'run':str(run.resolve()), 'source_config':str(Path(cfg_path).resolve()),
            'source_config_sha256':sha(cfg_path),'history_sha256':sha(history_path),
            'manifest_sha256':sha(manifest_file),'best_checkpoint':str(best_paths[0].resolve()),
            'best_checkpoint_sha256':sha(best_paths[0]),'best_epoch':best+1,
            'metrics':{key:row[key] for key in REQUIRED_HISTORY[1:]},
            'input_hashes':hashes,'git_commit':manifest['git']['commit'],
            'source_overlap':inputs['global_cross_split_overlap']}


def collect(config_dir, root):
    config_dir,root=Path(config_dir),Path(root)
    configs={}; errors=[]
    for path in sorted(config_dir.glob('*.yml')):
        try:
            cfg=load_yml(path); key=identity(cfg)
            if key in configs:
                raise ValueError('Duplicate identity '+str(key))
            configs[key]=(path,cfg)
        except Exception as e:
            errors.append({'config':str(path),'error':str(e)})
    expected={(a,s,k,seed) for a in ARCHS for s in STRATEGIES for k in range(1,6) for seed in SEEDS}
    missing_configs=sorted(expected-set(configs))
    extra_configs=sorted(set(configs)-expected)
    results={}; missing_runs=[]; bad_runs=[]
    for key,(path,cfg) in sorted(configs.items()):
        if key not in expected:
            continue
        base=root/key[0]/key[1]/('E'+str(key[2]))/('seed-'+str(key[3]))
        candidates=list(base.glob('*/moment_epoch_metrics.csv')) if base.is_dir() else []
        # Fail closed if there are multiple candidate trajectories. Don't guess a winner.
        if not candidates:
            missing_runs.append({'identity':key,'directory':str(base)})
            continue
        valid=[]; rejected=[]
        for h in candidates:
            try:
                valid.append(inspect_run(h.parent,path,cfg))
            except Exception as e:
                rejected.append({'run':str(h.parent),'error':str(e)})
        if len(valid)!=1 or rejected:
            bad_runs.append({'identity':key,'valid_count':len(valid),'rejected':rejected,
                             'reason':'Requires exactly one complete and unambiguous run'})
        else:
            results[key]=valid[0]
    return {'config_count':len(configs),'complete':len(results),'missing_configs':missing_configs,
            'extra_configs':extra_configs,'missing_runs':missing_runs,'bad_runs':bad_runs,
            'config_errors':errors},results,configs


def select(audit, results, configs):
    if (audit['complete'] != 120 or any(audit[k] for k in
        ('missing_configs','extra_configs','missing_runs','bad_runs','config_errors'))):
        raise RuntimeError('Selection requires 120 unambiguous completed CE runs; audit first')
    common_hashes={tuple(sorted(r['input_hashes'].items())) for r in results.values()}
    overlap_counts={r['source_overlap'] for r in results.values()}
    if len(overlap_counts)!=1:
        raise RuntimeError('Cross-split source overlap count differs between runs')
    if len(common_hashes)!=1:
        raise RuntimeError('Split/audit SHA mismatch among candidate trajectories')
    groups=[]; selected={}
    for arch in ARCHS:
        for strat in STRATEGIES:
            for k in range(1,6):
                runs=[results[(arch,strat,k,seed)] for seed in SEEDS]
                for task in TASKS:
                    prefix='alpha' if task=='alpha_only' else 'q0'
                    mean=lambda field:sum(r['metrics'][field] for r in runs)/3
                    groups.append({'model':arch,'task':task,'strategy':strat,'E':k,
                                   'macro_f1_mean':mean('val_'+prefix+'_macro_f1'),
                                   'accuracy_mean':mean('val_'+prefix+'_accuracy'),
                                   'physical_mae_mean':mean('val_'+prefix+'_mae'),
                                   'runs':runs})
    for arch in ARCHS:
        for task in TASKS:
            subset=[r for r in groups if r['model']==arch and r['task']==task]
            # Fixed deterministic tie break: macro F1, accuracy, minus MAE, strategy, E.
            subset.sort(key=lambda r:(-r['macro_f1_mean'],-r['accuracy_mean'],
                                      r['physical_mae_mean'],r['strategy'],r['E']))
            selected[(arch,task)]=subset[0]
    return groups,selected,dict(common_hashes.pop())


def translate(cfg, model, task, seed):
    # Directly matches Stage 2 T04-10 strict YAML schema.
    keys=('model','checkpoint','checkpoint_revision','strategy','moments',
          'precision','optimizer','learning_rates','batch_size','num_workers',
          'dataset_root','source_audit_report','split_protocol','gpu_profile',
          'epochs','training_protocol')
    out={k:cfg[k] for k in keys}
    out['task']=task
    out['seed']=seed
    out['output_root']=OUTPUT
    out['scheduler']={k:cfg['scheduler'][k] for k in
                      ('type','factor','patience','min_lr')}
    return out


def freeze(config_dir, root, output, *, preview=False):
    audit,results,configs=collect(config_dir,root)
    if preview:
        return {'status':'AUDIT','audit':audit,'selected':None}
    groups,winners,hashes=select(audit,results,configs)
    output=Path(output)
    if output.exists():
        raise FileExistsError('Freeze destination already exists: '+str(output))
    # Validate all resulting YAML configs before creating a frozen output.
    planned=[]
    from train_utils.t04_10_preflight import load_settings
    import tempfile
    for arch in ARCHS:
        for task in TASKS:
            best=winners[(arch,task)]
            for seed in SEEDS:
                original=configs[(arch,best['strategy'],best['E'],seed)][1]
                cfg=translate(original,arch,task,seed)
                with tempfile.TemporaryDirectory() as tmp:
                    tmpfile=Path(tmp)/'config.yml'
                    tmpfile.write_text(yaml.safe_dump(cfg,sort_keys=False))
                    load_settings(tmpfile)
                ident=f't04_10_{arch}_{task}_seed{seed}'
                planned.append((ident,cfg,best,results[(arch,best['strategy'],best['E'],seed)]))
    output.mkdir(parents=True,exist_ok=False)
    try:
        entries=[]
        for ident,cfg,best,reference in planned:
            file=output/(ident+'.yml')
            file.write_text(yaml.safe_dump(cfg,sort_keys=False))
            entries.append({'id':ident,'status':'FROZEN_NOT_PREPARED',
                'config':file.name,'config_sha256':sha(file),
                'source_config':reference['source_config'],
                'source_config_sha256':reference['source_config_sha256'],
                'reference_joint_run':reference['run'],
                'reference_joint_best_epoch':reference['best_epoch'],
                'reference_joint_checkpoint_sha256':reference['best_checkpoint_sha256'],
                'selection_group':{'strategy':best['strategy'],'E':best['E']}})
        for task in TASKS:
            for seed in SEEDS:
                entries.append({'id':f't04_10_vit_{task}_seed{seed}',
                                'status':'BLOCKED_VIT_PARITY'})
        winners_out=[{'model':a,'task':t,'strategy':winners[(a,t)]['strategy'],
                      'E':winners[(a,t)]['E'],
                      'macro_f1_mean':winners[(a,t)]['macro_f1_mean'],
                      'accuracy_mean':winners[(a,t)]['accuracy_mean'],
                      'physical_mae_mean':winners[(a,t)]['physical_mae_mean']}
                     for a in ARCHS for t in TASKS]
        plan={'schema':1,'issue':30,'status':'FROZEN_NO_JOBS_SUBMITTED',
              'frozen_utc':datetime.now(timezone.utc).isoformat(),
              'selection_rule':'3-seed validation macro-F1 at T04-06 joint-selected best checkpoint epoch; accuracy then MAE tie-break',
              'selection_basis':'validation_only','test_used_for_selection':False,
              'candidates_compared':120,'matching_input_sha256':hashes,
              'source_disjoint_claim':False,
              'source_overlap_occurrences':next(iter({r['source_overlap'] for r in results.values()})),
              'selected':winners_out,'entries':entries,
              'eligible_single_head':12,'vit_blocked':6}
        (output/'pilot_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        with (output/'ranking.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['model','task','strategy','E','macro_f1_mean','accuracy_mean','physical_mae_mean'])
            writer.writeheader()
            for row in sorted(groups,key=lambda r:(r['model'],r['task'],-r['macro_f1_mean'])):
                writer.writerow({key:row[key] for key in writer.fieldnames})
        return {'status':'FROZEN','output':str(output),'selected':winners_out,
                'eligible_single_head':12,'vit_blocked':6}
    except Exception:
        # Do not silently adopt a partial freeze or delete forensic evidence.
        (output/'FREEZE_FAILED').write_text('Partial output: inspect and use a new destination.\n')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs',required=True,help='T04-06 v2 120 YAML folder')
    p.add_argument('--runs',required=True,help='T04-06 v2 production output root')
    p.add_argument('--output',help='New dedicated T04-10 freeze directory')
    p.add_argument('--audit',action='store_true',help='Read-only coverage check')
    args=p.parse_args()
    if not args.audit and not args.output:
        p.error('--output required for freeze')
    result=freeze(args.configs,args.runs,args.output,preview=args.audit)
    print(json.dumps(result,indent=2,default=str))

if __name__=='__main__':
    main()
