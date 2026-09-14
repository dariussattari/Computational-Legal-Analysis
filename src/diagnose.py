"""Is the pipeline broken, or is doctrinal stance just not learnable across cases?

Same features, same leave-one-case-out protocol, three different targets.
If style/era targets are learnable and stance is not, the pipeline is fine and
the negative result is about the construct, not the code.
"""
import json,os,sys,numpy as np
sys.path.insert(0,"src")
from labels import LABELS, HELD_OUT_CASE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, accuracy_score
rows=[json.loads(l) for l in open("data/proc/paragraphs.jsonl")]
V=np.load("data/proc/embeddings.npy")
keep=[i for i,r in enumerate(rows) if not r["is_footnote"] and r["n_words"]>=25]
rows=[rows[i] for i in keep]

RES={}
def run(name, idx, y, grp):
    y=np.array(y); grp=np.array(grp); texts=[rows[i]["text"] for i in idx]
    probs=np.zeros(len(y)); preds=np.zeros(len(y))
    for trn,tst in LeaveOneGroupOut().split(np.zeros(len(y)),y,grp):
        if len(set(y[trn]))<2: probs[tst]=y[trn].mean(); preds[tst]=round(y[trn].mean()); continue
        vec=TfidfVectorizer(ngram_range=(1,2),min_df=3,max_features=60000,sublinear_tf=True)
        A=vec.fit_transform([texts[i] for i in trn]); B=vec.transform([texts[i] for i in tst])
        clf=LogisticRegression(max_iter=4000,class_weight="balanced").fit(A,y[trn])
        probs[tst]=clf.predict_proba(B)[:,1]; preds[tst]=clf.predict(B)
    base=max(y.mean(),1-y.mean())
    RES[name.split()[0]]=float(roc_auc_score(y,probs))
    print(f"  {name:<34} acc={accuracy_score(y,preds):.3f} (baseline {base:.3f})  AUC={roc_auc_score(y,probs):.3f}")

# target 1: doctrinal stance (the real question)
i1=[];y1=[];g1=[]
for i,r in enumerate(rows):
    if r["case_key"]==HELD_OUT_CASE: continue
    l=LABELS.get((r["case_key"],r["author"]))
    if l is None: continue
    i1.append(i);y1.append(l);g1.append(r["case_key"])
run("stance (expansive vs restrictive)", i1,y1,g1)

# target 2: majority vs dissent - pure style/role, should be easier
i2=[];y2=[];g2=[]
for i,r in enumerate(rows):
    if r["opinion_type"] in ("majority","dissent"):
        i2.append(i); y2.append(1 if r["opinion_type"]=="dissent" else 0); g2.append(r["case_key"])
run("majority vs dissent", i2,y2,g2)

# target 3: era - vocabulary drift, should be easy
i3=[];y3=[];g3=[]
for i,r in enumerate(rows):
    i3.append(i); y3.append(1 if r["year"]>=2000 else 0); g3.append(r["case_key"])
run("era (pre-2000 vs post-2000)", i3,y3,g3)

import json as _j
_j.dump({"stance":RES["stance"],"role":RES["majority"],"era":RES["era"]},open("data/proc/diagnostics.json","w"))
print("saved diagnostics.json")
