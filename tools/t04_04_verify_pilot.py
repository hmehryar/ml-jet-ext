#!/usr/bin/env python3
"""Check prepared source, session chain, metric rows, telemetry and resume states."""
import argparse,csv,hashlib,json
from pathlib import Path
import torch
from train_utils.t04_preflight import verify
from train_utils.t04_sessions import latest_checkpoint

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def chk(ok,msg):
 if not ok:raise AssertionError(msg)
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();run=Path(args.run).expanduser().resolve()
 verify(run);config=json.loads((run/'config.resolved.json').read_text());chk(config['precision']=='bf16','not bf16');chk(config['optimizer']['type']=='adamw','not AdamW');chk(config['gradient_accumulation']==4,'not accumulation=4')
 sessions=[run/'sessions'/('session-%03d'%i) for i in (1,2)]
 for i,s in enumerate(sessions):
  m=json.loads((s/'session_manifest.json').read_text());chk(m['status']=='completed','session incomplete');chk(m['mode']==('fresh' if i==0 else 'resume'),'wrong mode');chk(m['last_committed_epoch']==i+1,'wrong committed epoch');
  chk((s/'result.json').is_file(),'missing result');chk((s/'moment_epoch_metrics.csv').is_file(),'missing session CSV')
  with (s/'moment_epoch_metrics.csv').open() as f: rows=list(csv.DictReader(f))
  chk(len(rows)==1 and int(rows[0]['epoch'])==i+1,'incorrect session CSV epoch')
  tel=s/'telemetry'/('epoch_%06d.json'%(i+1));chk(tel.is_file(),'missing epoch telemetry '+str(tel)); t=json.loads(tel.read_text());
  print('TELEMETRY',s.name, t,flush=True)
  chk(t['gradient_accumulation']==4,'bad telemetry accumulation');chk(isinstance(t['learning_rates'],dict) and t['learning_rates'],'missing LR groups')
  chk(t.get('gpu_name'),'missing GPU name')
  if i:chk(m['parent_checkpoint']['epoch']==1,'wrong resume parent epoch')
 pointer=latest_checkpoint(run);chk(pointer['epoch']==2,'latest epoch is not 2');chk(digest(pointer['absolute_path'])==pointer['sha256'],'checkpoint hash mismatch')
 with (run/'moment_epoch_metrics.csv').open() as f: rows=list(csv.DictReader(f))
 chk([int(r['epoch']) for r in rows]==[1,2],'epoch continuity failed')
 state=torch.load(pointer['absolute_path'],map_location='cpu',weights_only=False)
 for key in ('optimizer_state_dict','scheduler_state_dict','model_state_dict','rng_state','metrics_row'):
  chk(state.get(key) is not None,'missing checkpoint '+key)
 chk(state['epoch']==2,'wrong checkpoint epoch');chk(state.get('scaler_state_dict') is None,'bf16 must not have GradScaler')
 groups=state['optimizer_state_dict']['param_groups'];chk(len(groups)>=2,'expected multiple AdamW groups');chk(all('name' in g for g in groups),'unnamed AdamW groups')
 print('PASS: frozen source; two completed sessions; resume parent; epochs 1,2; BF16; AdamW groups; telemetry; checkpoint and CSV')
if __name__=='__main__':main()
