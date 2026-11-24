#!/usr/bin/env python3
"""T04-11 deterministic, sample-aligned, cache-based evaluation.
Input NPZ: ids, y_alpha, y_q0, logits_alpha and/or logits_q0.
Optional metadata JSON alongside NPZ with sha256 and provenance fields.
"""
import argparse, csv, hashlib, itertools, json
from pathlib import Path
import numpy as np

ALPHA=np.round(np.arange(0.20,0.501,0.05),2)
Q0=np.arange(1.,3.01,.5)

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def load(path, expected_split=None):
    path=Path(path)
    with np.load(path,allow_pickle=False) as d:
        a={k:d[k].copy() for k in d.files}
    for k in ('ids','y_alpha','y_q0'):
        if k not in a:raise ValueError(f'{path}: missing {k}')
    ids=a['ids'].astype(str)
    if len(set(ids))!=len(ids):raise ValueError(f'{path}: duplicate sample IDs')
    if not len(ids):raise ValueError(f'{path}: empty cache')
    for k,n in [('y_alpha',7),('y_q0',5)]:
        y=a[k]
        if y.shape!=(len(ids),) or y.dtype.kind not in 'iu' or np.any((y<0)|(y>=n)):
            raise ValueError(f'{path}: invalid {k}')
    for k,n in [('logits_alpha',7),('logits_q0',5)]:
        if k in a and (a[k].shape!=(len(ids),n) or not np.isfinite(a[k]).all()):
            raise ValueError(f'{path}: invalid {k} shape/nonfinite')
    if not any(k in a for k in ('logits_alpha','logits_q0')):raise ValueError('No logits')
    meta_path=path.with_suffix('.json')
    if not meta_path.exists():raise ValueError(f'Missing provenance: {meta_path}')
    meta=json.loads(meta_path.read_text())
    for k in ('checkpoint_sha256','split_sha256','seed','split','class_mapping','input_moments','run_id'):
        if k not in meta:raise ValueError(f'{meta_path}: missing {k}')
    if expected_split and meta['split']!=expected_split:raise ValueError('Wrong split')
    if meta['class_mapping']!={'alpha':ALPHA.tolist(),'q0':Q0.tolist()}:
        raise ValueError('Incorrect physical class mapping')
    a['ids']=ids
    a['meta']=meta
    a['file_sha256']=digest(path)
    return a

def align(*caches):
    ref=caches[0]
    target=ref['ids'].tolist()
    out=[]
    for c in caches:
        if c['meta']['split_sha256']!=ref['meta']['split_sha256'] or c['meta']['split']!=ref['meta']['split']:
            raise ValueError('Split hash/name mismatch')
        if c['meta']['seed']!=ref['meta']['seed']:raise ValueError('Seed mismatch')
        if set(c['ids'])!=set(target):raise ValueError('Missing/extra sample IDs')
        index={s:i for i,s in enumerate(c['ids'])}
        order=np.array([index[s] for s in target])
        d={k:(v[order] if isinstance(v,np.ndarray) and len(v)==len(order) else v) for k,v in c.items()}
        for k in ('y_alpha','y_q0'):
            if not np.array_equal(d[k],ref[k]):raise ValueError('Ground truth mismatch')
        out.append(d)
    return out

def ranked(logits):
    # Stable ties: lower class index first.
    return np.argsort(-logits,axis=1,kind='stable')

def topk(logits,y,ks=(1,2,3,5)):
    order=ranked(logits)
    return {f'top{k}':float(np.mean(np.any(order[:,:k]==y[:,None],axis=1))) for k in ks if k<=logits.shape[1]}

def pair_topk(a,q,ya,yq,ks=(1,2,3,5)):
    # Sum of logits ranks exactly as sum of log-softmaxes (constant per row).
    score=(a[:,:,None]+q[:,None,:]).reshape(len(ya),35)
    return topk(score,ya*5+yq,ks)

def confusion(y,p,n):
    m=np.zeros((n,n),dtype=int)
    np.add.at(m,(y,p),1)
    support=m.sum(axis=1)
    normalized=np.divide(m,support[:,None],out=np.zeros_like(m,dtype=float),where=support[:,None]!=0)
    return m,normalized

def evaluate(a,q):
    a,q=align(a,q)
    ya,yq=a['y_alpha'],a['y_q0']
    la,lq=a['logits_alpha'],q['logits_q0']
    pa,pq=ranked(la)[:,0],ranked(lq)[:,0]
    da=np.abs(pa-ya);dq=np.abs(pq-yq)
    results={'n':len(ya),'alpha_accuracy':float(np.mean(da==0)),
      'q0_accuracy':float(np.mean(dq==0)),
      'joint_accuracy':float(np.mean((da==0)&(dq==0))),
      'alpha_mae_physical':float(np.mean(da)*.05),
      'q0_mae_physical':float(np.mean(dq)*.5)}
    results.update({f'alpha_{k}':v for k,v in topk(la,ya).items()})
    results.update({f'q0_{k}':v for k,v in topk(lq,yq).items()})
    results.update({f'pair_{k}':v for k,v in pair_topk(la,lq,ya,yq).items()})
    for k in (0,1,2):
        results[f'alpha_within{k}']=float(np.mean(da<=k))
        results[f'q0_within{k}']=float(np.mean(dq<=k))
        results[f'joint_within{k}']=float(np.mean((da<=k)&(dq<=k)))
    if abs(results['pair_top1']-results['joint_accuracy'])>1e-12:
        raise ValueError('Pair Top-1 mismatch; investigate ties')
    matrices={}
    for name,y,p,n in [('alpha',ya,pa,7),('q0',yq,pq,5)]:
        raw,norm=confusion(y,p,n)
        matrices[name]=(raw,norm)
        results[f'{name}_per_class_accuracy']=[float(norm[i,i]) if raw[i].sum() else None for i in range(n)]
    return results,matrices

def write_figures(matrices,out,prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,(raw,norm) in matrices.items():
        labels=ALPHA if name=='alpha' else Q0
        for kind,m in [('counts',raw),('normalized',norm)]:
            fig,ax=plt.subplots(figsize=(7,6))
            im=ax.imshow(m,cmap='Blues',vmin=0,vmax=(1 if kind=='normalized' else None))
            ax.set_xticks(range(len(labels)),[f'{v:.2f}' if name=='alpha' else f'{v:.1f}' for v in labels])
            ax.set_yticks(range(len(labels)),[f'{v:.2f}' if name=='alpha' else f'{v:.1f}' for v in labels])
            ax.set_xlabel('Predicted');ax.set_ylabel('True')
            ax.set_title(f'{prefix}: {name}, {kind}')
            fig.colorbar(im,ax=ax)
            fig.tight_layout()
            for ext in ('pdf','png'):fig.savefig(out/f'{prefix}_{name}_{kind}.{ext}',dpi=300)
            plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--alpha-cache',required=True);p.add_argument('--q0-cache',required=True)
    p.add_argument('--split',choices=['val','test'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--name',required=True)
    p.add_argument('--figures',action='store_true')
    args=p.parse_args()
    a=load(args.alpha_cache,args.split);q=load(args.q0_cache,args.split)
    if 'logits_alpha' not in a or 'logits_q0' not in q:raise ValueError('Required head logits missing')
    r,m=evaluate(a,q)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    payload={'metrics':r,'inputs':{'alpha':{'path':args.alpha_cache,'sha256':a['file_sha256'],'meta':a['meta']},
      'q0':{'path':args.q0_cache,'sha256':q['file_sha256'],'meta':q['meta']}}}
    (out/f'{args.name}.json').write_text(json.dumps(payload,indent=2))
    for name,(raw,norm) in m.items():
        np.savetxt(out/f'{args.name}_{name}_counts.csv',raw,fmt='%d',delimiter=',')
        np.savetxt(out/f'{args.name}_{name}_normalized.csv',norm,fmt='%.9f',delimiter=',')
    if args.figures:write_figures(m,out,args.name)
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()
