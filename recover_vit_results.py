#!/usr/bin/env python3
"""Read-only extraction of ViT Exp02 and Exp03 run metrics (no inference/training)."""
import csv, json, math, re
from pathlib import Path
from collections import Counter

ROOTS = [
 ('exp02', Path('experiments/exp_02_transfer_learning_with_vit_backbone/training_output')),
 ('exp03', Path('experiments/exp_03_cumulative_moment_ablation')),
 ('exp03_v2', Path('experiments/exp_03_cumulative_moment_ablation_v2')),
 ('exp03_v3', Path('experiments/exp_03_cumulative_moment_ablation_v3')),
]
OUT = Path('/tmp/mljet_vit_results_audit')
OUT.mkdir(parents=True, exist_ok=True)


def value(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):
        return None


def classify(p):
    path=str(p).lower()
    if 'double_loading' in path: return '1.b double-loading'
    if '1b' in path or 'branch' in path: return '1.b branching'
    if '2b' in path or 'parallel' in path: return '2.b backbone'
    return 'unclassified'


def init(p):
    s=str(p).lower()
    if 'gaussian' in s: return 'gaussian-labelled; verify effective load'
    if 'unfrozen' in s: return 'preload, unfrozen-labelled'
    if 'frozen' in s: return 'preload, frozen-labelled'
    if 'preload' in s: return 'preload-labelled; verify freeze mask'
    return 'unknown'


def moments(p):
    s=str(p)
    m=re.search(r'(?:^|[/_-])E([1-5])(?:[_/.-]|$)',s,re.I)
    if m: return 'E'+m.group(1)
    m=re.search(r'm1[_-]?m([2-5])',s,re.I)
    if m: return 'E'+m.group(1)
    return ''


def read_summary(path):
    if not path.is_file(): return {}
    try:
        d=json.loads(path.read_text())
        return d if isinstance(d,dict) else {'error':'summary_not_mapping'}
    except Exception as e: return {'error':str(e)}


def read_metrics(path):
    try:
        with path.open(newline='') as fh:
            reader=csv.DictReader(fh)
            return list(reader),reader.fieldnames or [],''
    except Exception as e:
        return [],[],str(e)


def choose(rows,metric):
    valid=[]
    for r in rows:
        v=value(r.get(metric))
        e=value(r.get('epoch'))
        if v is not None and e is not None:
            valid.append((int(e),v,r))
    if not valid: return None
    return max(valid,key=lambda t:(t[1],-t[0]))


def scan_root(tag,root):
    candidates = []
    if not root.exists(): return candidates
    if tag=='exp02':
        candidates=list(root.rglob('epoch_metrics.csv'))
    else:
        candidates=list(root.rglob('moment_epoch_metrics.csv'))
        # Legacy T03 artifacts might have other metric filenames. Include but mark.
        known={p.parent for p in candidates}
        candidates += [p for p in root.rglob('epoch_metrics.csv') if p.parent not in known]
    return sorted(set(candidates))

records=[]
for family,root in ROOTS:
    paths=scan_root(family,root)
    print(f'{family}: root={root} metric_files={len(paths)}')
    for path in paths:
        summary=read_summary(path.parent/'training_summary.json')
        rows,cols,err=read_metrics(path)
        is03=family!='exp02'
        metric='val_joint_accuracy' if is03 else 'val_acc'
        selected=choose(rows,metric)
        epochs=[int(e) for r in rows if (e:=value(r.get('epoch'))) is not None and e.is_integer()]
        continuity=(len(epochs)==len(rows) and len(set(epochs))==len(epochs)
                    and epochs==list(range(1,len(epochs)+1)))
        best_epoch=selected[0] if selected else None
        if not is03 and value(summary.get('best_epoch')) is not None:
            # Exp02 model selection best_epoch from summary; don't silently substitute a new choice.
            best_epoch=int(float(summary['best_epoch']))
        selected_row=next((r for r in rows if value(r.get('epoch'))==best_epoch),None)
        best_checkpoint=path.parent/'best_model.pth'
        if not best_checkpoint.exists():
            best_checkpoint=path.parent/'best_model.pt'
        split_match=re.search(r'fold[_-]?(\d+)',str(path),re.I)
        out={
            'family':family,'run_path':str(path.parent),'design':classify(path),
            'initialization_label':init(path),'moment':moments(path),
            'fold':split_match.group(1) if split_match else '',
            'history_file':path.name,'history_rows':len(rows),'history_continuous':continuity,
            'last_epoch':epochs[-1] if epochs else '',
            'summary_status':summary.get('summary_status',''),
            'summary_configured_epochs':summary.get('epochs',''),
            'summary_best_epoch':summary.get('best_epoch',''),
            'summary_best_accuracy':summary.get('best_accuracy',''),
            'selected_epoch':best_epoch or '',
            'selection_metric':('summary best_epoch / val_acc' if not is03 else metric),
            'best_checkpoint_present':best_checkpoint.is_file(),
            'validation_metric_source':('best_epoch in summary' if not is03 and summary.get('best_epoch') is not None else 'earliest history maximum'),
            'issue':err or summary.get('error',''),
        }
        for src,target in [('val_acc','validation_total_accuracy'),('val_joint_accuracy','validation_joint_accuracy'),
                           ('val_alpha_acc','validation_alpha_accuracy'),('val_alpha_accuracy','validation_alpha_accuracy'),
                           ('val_q0_acc','validation_q0_accuracy'),('val_q0_accuracy','validation_q0_accuracy'),
                           ('val_alpha_f1','validation_alpha_f1'),('val_alpha_macro_f1','validation_alpha_f1'),
                           ('val_q0_f1','validation_q0_f1'),('val_q0_macro_f1','validation_q0_f1'),
                           ('val_loss','validation_loss')]:
            if selected_row is not None and src in selected_row:
                out[target]=selected_row[src]
        if not selected_row:out['issue']+=' | missing selected epoch row'
        if family=='exp02' and summary.get('best_epoch') is not None and selected:
            if best_epoch!=selected[0]:out['issue']+=f' | summary epoch differs from max {metric} epoch {selected[0]}'
        if not continuity:out['issue']+=' | discontinuous or non-1-based epochs'
        if not best_checkpoint.is_file():out['issue']+=' | selected checkpoint not found'
        out['column_names']=';'.join(cols)
        records.append(out)

output=OUT/'vit_run_inventory.csv'
if records:
    fields=list(dict.fromkeys(k for row in records for k in row))
    with output.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
print('\nTOTAL RECORDED METRIC HISTORIES:',len(records))
print('INVENTORY:',output)
for family in ['exp02','exp03','exp03_v2','exp03_v3']:
    block=[r for r in records if r['family']==family]
    print('\n'+'='*72+'\n'+family.upper()+' — '+str(len(block))+' histories')
    for r in block:
        metric=('validation_joint_accuracy' if family!='exp02' else 'validation_total_accuracy')
        val=value(r.get(metric))
        a=value(r.get('validation_alpha_accuracy'))
        q=value(r.get('validation_q0_accuracy'))
        print(f"  {r['design']:<20} {r['moment']:<3} epoch={str(r['selected_epoch']):>4}/{str(r['last_epoch']):<4} "
              f"total/joint={val if val is not None else 'NA'} "
              f"alpha={a if a is not None else 'NA'} q0={q if q is not None else 'NA'} "
              f"checkpoint={'YES' if r['best_checkpoint_present'] else 'NO'}")
        print('   ',r['run_path'])
        if r['issue'].strip(' |'): print('    REVIEW:',r['issue'])
    if block:
        print('COLUMNS EXAMPLE:',block[0]['column_names'][:350])

print('\nNOTE: Exp02 val_acc is legacy total accuracy and is not assumed identical to Exp03 joint accuracy.')
print('NOTE: Filenames are hints about initialization, not proof of actual weights loaded.')
print('NOTE: Metrics reflect validation histories; not independently verified held-out test scores.')
