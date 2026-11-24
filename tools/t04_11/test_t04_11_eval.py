import numpy as np
import pytest
from t04_11_eval import evaluate,align,confusion,topk

def cache(ids,ya,yq,la=None,lq=None):
    d={'ids':np.array(ids),'y_alpha':np.array(ya),'y_q0':np.array(yq),
       'meta':{'split_sha256':'same','split':'val','seed':42}}
    if la is not None:d['logits_alpha']=np.asarray(la)
    if lq is not None:d['logits_q0']=np.asarray(lq)
    return d

def test_alignment_and_metrics():
    a=cache(['x','y'],[1,2],[3,4],la=np.eye(7)[[1,2]])
    q=cache(['y','x'],[2,1],[4,3],lq=np.eye(5)[[4,3]])
    r,_=evaluate(a,q)
    assert r['joint_accuracy']==1 and r['pair_top1']==1
    assert r['joint_within0']==r['joint_accuracy']

def test_duplicate_or_missing_ids():
    a=cache(['x','y'],[1,2],[3,4]);q=cache(['x','z'],[1,2],[3,4])
    with pytest.raises(ValueError):align(a,q)

def test_zero_support():
    raw,norm=confusion(np.array([0]),np.array([1]),3)
    assert norm[2].sum()==0 and norm[0].sum()==1

def test_topk_monotonic():
    logits=np.array([[0.,2.,1.],[1.,0.,2.]])
    r=topk(logits,np.array([2,1]),(1,2,3))
    assert r['top1']<=r['top2']<=r['top3']
