"""Can pre-2024 presidential-power jurisprudence predict the reasoning in Trump?

Paragraphs inherit their opinion's stance label (distant supervision).  Two
feature sets are compared - TF-IDF bigrams and the sentence embeddings - and
both are evaluated leave-one-CASE-out, so a fold is never scored on a case it
trained on.  That matters more than usual here: paragraphs inside one opinion
share vocabulary, facts, and an author, so a random split would be badly
optimistic.

Trump v. United States is excluded from every fit, then scored at the end.
"""

import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from labels import HELD_OUT_CASE, LABELS

PROC = os.path.join(HERE, "..", "data", "proc")


def load():
    rows = [json.loads(l) for l in open(os.path.join(PROC, "paragraphs.jsonl"))]
    V = np.load(os.path.join(PROC, "embeddings.npy"))
    keep = [i for i, r in enumerate(rows) if not r["is_footnote"] and r["n_words"] >= 25]
    return [rows[i] for i in keep], V[keep]


def main():
    rows, V = load()
    tr_i, y, grp = [], [], []
    for i, r in enumerate(rows):
        if r["case_key"] == HELD_OUT_CASE:
            continue
        lab = LABELS.get((r["case_key"], r["author"]))
        if lab is None:
            continue
        tr_i.append(i)
        y.append(lab)
        grp.append(r["case_key"])
    tr_i = np.array(tr_i)
    y = np.array(y)
    grp = np.array(grp)

    print(f"training paragraphs: {len(y):,}  expansive={y.mean():.1%}  "
          f"cases={len(set(grp))}  opinions={len(set((rows[i]['case_key'],rows[i]['author']) for i in tr_i))}")
    print(f"majority-class baseline: {max(y.mean(), 1-y.mean()):.1%}\n")

    texts = [rows[i]["text"] for i in tr_i]
    Xemb = V[tr_i]

    logo = LeaveOneGroupOut()
    results = {}
    for name in ("tfidf", "embeddings"):
        preds = np.zeros(len(y))
        probs = np.zeros(len(y))
        for trn, tst in logo.split(np.zeros(len(y)), y, grp):
            if len(set(y[trn])) < 2:
                probs[tst] = y[trn].mean()
                preds[tst] = round(y[trn].mean())
                continue
            if name == "tfidf":
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000,
                                      sublinear_tf=True, stop_words=None)
                A = vec.fit_transform([texts[i] for i in trn])
                Bm = vec.transform([texts[i] for i in tst])
            else:
                sc = StandardScaler().fit(Xemb[trn])
                A, Bm = sc.transform(Xemb[trn]), sc.transform(Xemb[tst])
            clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
            clf.fit(A, y[trn])
            probs[tst] = clf.predict_proba(Bm)[:, 1]
            preds[tst] = clf.predict(Bm)
        acc = accuracy_score(y, preds)
        auc = roc_auc_score(y, probs)
        results[name] = (acc, auc, probs)
        print(f"{name:<12} leave-one-case-out  accuracy={acc:.3f}  AUC={auc:.3f}")

    # case-level: does the model recover each case's majority stance?
    print("\nper-case accuracy (tfidf), lowest first:")
    _, _, p = results["tfidf"]
    per = {}
    for j, i in enumerate(tr_i):
        per.setdefault(rows[i]["case_key"], []).append((p[j] > .5) == bool(y[j]))
    for k, v in sorted(per.items(), key=lambda kv: np.mean(kv[1]))[:6]:
        print(f"   {k:<20} {np.mean(v):.2f}  (n={len(v)})")

    # ---- score the held-out case ----
    best = "tfidf" if results["tfidf"][1] >= results["embeddings"][1] else "embeddings"
    print(f"\nrefitting {best} on all labelled pre-2024 data, scoring {HELD_OUT_CASE}\n")
    ho = [i for i, r in enumerate(rows) if r["case_key"] == HELD_OUT_CASE]
    if best == "tfidf":
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True)
        A = vec.fit_transform(texts)
        Bm = vec.transform([rows[i]["text"] for i in ho])
    else:
        sc = StandardScaler().fit(Xemb)
        A, Bm = sc.transform(Xemb), sc.transform(V[ho])
    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced").fit(A, y)
    ph = clf.predict_proba(Bm)[:, 1]

    by = {}
    for i, pr in zip(ho, ph):
        by.setdefault(rows[i]["author"], []).append(pr)
    print("   P(EXPANSIVE) by opinion in Trump v. United States")
    for a in ("Roberts", "Thomas", "Barrett", "Sotomayor", "Jackson"):
        if a in by:
            v = np.array(by[a])
            print(f"     {a:<10} mean={v.mean():.3f}  share of paragraphs >0.5 = {(v>.5).mean():.0%}  (n={len(v)})")

    # reference points from the training era
    print("\n   for reference, the same model on labelled opinions it trained on:")
    tp = clf.predict_proba(A)[:, 1]
    ref = {}
    for j, i in enumerate(tr_i):
        ref.setdefault((rows[i]["case_key"], rows[i]["author"]), []).append(tp[j])
    for key in (("youngstown_1952", "Jackson"), ("youngstown_1952", "Vinson"),
                ("morrison_1988", "Scalia"), ("nixon_fitz_1982", "Powell"),
                ("clinton_jones_1997", "Stevens"), ("seila_law_2020", "Roberts")):
        if key in ref:
            print(f"     {key[0]}/{key[1]:<12} mean={np.mean(ref[key]):.3f}  (label={LABELS[key]})")

    json.dump({"trump_scores": {k: list(map(float, v)) for k, v in by.items()}},
              open(os.path.join(PROC, "clf_trump.json"), "w"))


if __name__ == "__main__":
    main()
