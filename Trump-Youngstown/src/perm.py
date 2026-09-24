"""Permutation null for the fixed-effects deference direction.

Shuffle the stance labels ACROSS OPINIONS WITHIN EACH CASE, keeping the number
of opinions and paragraphs per opinion fixed, then refit. This preserves case
structure and opinion lengths and destroys only the stance assignment, so the
resulting distribution is the right null for "delta carries stance information".
"""
import json,os,sys,numpy as np
sys.path.insert(0,"src")
from fe import load, labelled, fit_delta
from sklearn.metrics import roc_auc_score
rng=np.random.default_rng(11)
rows,V=load(); idx,s,case=labelled(rows)
# opinion identity per labelled paragraph
opid=np.array([f'{rows[i]["case_key"]}|{rows[i]["author"]}' for i in idx])

def loo_auc(svec):
    yt,ys=[],[]
    for held in sorted(set(case)):
        d,n=fit_delta(V,idx[case!=held],svec[case!=held],case[case!=held])
        if d is None: continue
        m=case==held
        sc=(V[idx[m]]-V[idx[m]].mean(0))@d
        y=(svec[m]==1).astype(int)
        if len(set(y))<2: continue
        yt.append(y); ys.append(sc)
    return roc_auc_score(np.concatenate(yt),np.concatenate(ys))

obs=loo_auc(s)
B=200; null=np.empty(B)
for b in range(B):
    sp=s.copy()
    for c in set(case):
        m=case==c
        ops=sorted(set(opid[m]))
        # one stance value per opinion, permuted among that case's opinions
        vals=[s[m&(opid==o)][0] for o in ops]
        rng.shuffle(vals)
        for o,v in zip(ops,vals): sp[m&(opid==o)]=v
    null[b]=loo_auc(sp)
p=float((np.abs(null-.5)>=abs(obs-.5)).mean())
print(f"observed LOO AUC        = {obs:.3f}")
print(f"permutation null        = {null.mean():.3f} +/- {null.std():.3f}  (B={B})")
print(f"null 2.5-97.5 pct       = [{np.percentile(null,2.5):.3f}, {np.percentile(null,97.5):.3f}]")
print(f"two-sided p             = {p:.3f}")
print(f"\n-> observed is {'INSIDE' if p>0.05 else 'OUTSIDE'} the null: "
      f"{'no recoverable stance direction' if p>0.05 else 'signal present'}")
json.dump({"obs":float(obs),"null_mean":float(null.mean()),"null_sd":float(null.std()),"p":p},
          open("data/proc/perm_results.json","w"))
