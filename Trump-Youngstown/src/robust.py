"""Robustness of the C1/C2 contrasts to (a) quotation, (b) my section boundaries."""
import json,os,sys,numpy as np
sys.path.insert(0,"src")
from jackson import JACKSON_MAIN_END, section_of, strip_quotes
from embed import Embedder
PROC="data/proc"; RNG=np.random.default_rng(7); B=2000
rows=[json.loads(l) for l in open(f"{PROC}/paragraphs.jsonl")]
V=np.load(f"{PROC}/embeddings.npy")
idx=[i for i,r in enumerate(rows) if not r["is_footnote"] and r["n_words"]>=25]
jk=[i for i in idx if rows[i]["case_key"]=="youngstown_1952" and rows[i]["author"]=="Jackson"][:JACKSON_MAIN_END]
J=V[jk]/np.linalg.norm(V[jk],axis=1,keepdims=True)
lab=[section_of(k) for k in range(len(jk))]
tr=[i for i in idx if rows[i]["case_key"]=="trump_us_2024"]

def contrasts(Vx, cz, cw, boot=0):
    by={}
    for i in tr: by.setdefault(rows[i]["author"],[]).append(i)
    order=[a for a in ["Roberts","Thomas","Barrett","Sotomayor","Jackson"] if a in by]
    mats=[Vx[by[a]] for a in order]
    iM,iS,iJ=order.index("Roberts"),order.index("Sotomayor"),order.index("Jackson")
    def one(ms):
        S=np.vstack([((M/np.linalg.norm(M,axis=1,keepdims=True))@J.T).mean(0) for M in ms])
        R=S-S.mean(1,keepdims=True)-S.mean(0,keepdims=True)+S.mean()
        return R[:,cw].mean(1), R[:,cz].mean(1)
    w,z=one(mats)
    c1=w[iM]-(w[iS]+w[iJ])/2; c2=z[iM]-(z[iS]+z[iJ])/2
    if not boot: return c1,c2,None,None
    a1=np.empty(boot); a2=np.empty(boot)
    for b in range(boot):
        ms=[M[RNG.integers(0,len(M),len(M))] for M in mats]
        w,z=one(ms); a1[b]=w[iM]-(w[iS]+w[iJ])/2; a2[b]=z[iM]-(z[iS]+z[iJ])/2
    return c1,c2,a1,a2

cz=[k for k,l in enumerate(lab) if l.startswith("Zone")]
cw=[k for k,l in enumerate(lab) if l=="Structural warning"]

print("=== (a) quotation control ===")
emb=Embedder(); Vq=V.copy(); keep=[];txt=[]
for i in tr:
    s=strip_quotes(rows[i]["text"])
    if len(s.split())>=20: keep.append(i); txt.append(s)
E=emb.encode(txt)
for i,v in zip(keep,E): Vq[i]=v
for nm,Vx in (("as written",V),("quotes removed",Vq)):
    c1,c2,a1,a2=contrasts(Vx,cz,cw,B)
    print(f"  {nm:<16} C1={1000*c1:+6.2f} CI[{1000*np.percentile(a1,2.5):+.2f},{1000*np.percentile(a1,97.5):+.2f}]"
          f"   C2={1000*c2:+6.2f} CI[{1000*np.percentile(a2,2.5):+.2f},{1000*np.percentile(a2,97.5):+.2f}]")

print("\n=== (b) leave-one-paragraph-out of each section ===")
c1v=[];c2v=[]
for drop in cw:
    c1,c2,_,_=contrasts(V,cz,[k for k in cw if k!=drop]); c1v.append(c1)
for drop in cz:
    c1,c2,_,_=contrasts(V,[k for k in cz if k!=drop],cw); c2v.append(c2)
print(f"  C1 over {len(c1v)} warning-paragraph drops: min={1000*min(c1v):+.2f} max={1000*max(c1v):+.2f} (all negative: {all(x<0 for x in c1v)})")
print(f"  C2 over {len(c2v)} zone-paragraph drops:    min={1000*min(c2v):+.2f} max={1000*max(c2v):+.2f} (all positive: {all(x>0 for x in c2v)})")
