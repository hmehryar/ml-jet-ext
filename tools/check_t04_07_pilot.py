"""Read-only HPC preflight for 21 proposed T04-07 YAMLs; no run reservations."""
from pathlib import Path
import argparse,json
from train_utils.t04_preflight import preflight
from train_utils.loss_config import resolve_moment_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path.cwd())
    args=p.parse_args();d=args.repo.resolve()/'experiments/t04/config/t04_07_pilot'
    files=sorted(d.glob('*.yml'))
    if len(files)!=21: raise RuntimeError(f'Expected 21 configs, got {len(files)} at {d}')
    successful=[]; errors=[]
    for f in files:
        try:
            c,inputs=preflight(f)
            resolve_moment_loss(c.moment_loss)
            if c.model!='clip' or c.strategy!='S4' or c.moments!=[1,2,3] or c.epochs!=80:
                raise ValueError('Unexpected pilot training protocol')
            successful.append({'file':f.name,'seed':c.seed,'loss':c.moment_loss,
                               'split_protocol':c.split_protocol,
                               'overlap':inputs['global_cross_split_overlap']})
            print('PASS',f.name)
        except Exception as e:
            errors.append({'file':f.name,'error':str(e)})
            print('FAIL',f.name,repr(e))
    summary={'expected':21,'passed':len(successful),'failed':errors,
             'source_disjoint_claim':False,'runs_reserved':0,'jobs_submitted':0}
    print(json.dumps(summary,indent=2))
    if errors or len(successful)!=21: raise SystemExit(1)

if __name__=='__main__':main()
