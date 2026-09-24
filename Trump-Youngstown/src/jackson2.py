"""Relative engagement with Jackson's concurrence, double-centred.

Nearest-neighbour attribution failed: a handful of Jackson's paragraphs are
semantic hubs that attract everything, so all five Trump opinions looked
identical.  The signal is not "which section is closest" in absolute terms - it
is which opinion leans toward which passage *more than the others do*.

So build a similarity matrix S[opinion, jackson-paragraph] of mean cosine
similarity and double-centre it:

    R = S - rowmean - colmean + grandmean

Row means absorb how Youngstown-flavoured an opinion is overall; column means
absorb hub paragraphs.  What is left is relative affinity, which is the
comparative question the doctrinal claim actually turns on.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from embed import Embedder
from jackson import JACKSON_MAIN_END, SECTIONS, section_of, strip_quotes

PROC = os.path.join(HERE, "..", "data", "proc")
ORDER = ["Roberts", "Thomas", "Barrett", "Sotomayor", "Jackson"]


def build(rows, V, tr_idx, J):
    by = {}
    for i in tr_idx:
        by.setdefault(rows[i]["author"], []).append(i)
    S, authors = [], []
    for a in ORDER:
        if a not in by:
            continue
        M = V[by[a]]
        M = M / np.linalg.norm(M, axis=1, keepdims=True)
        S.append((M @ J.T).mean(0))
        authors.append(a)
    return np.vstack(S), authors


def double_center(S):
    return S - S.mean(1, keepdims=True) - S.mean(0, keepdims=True) + S.mean()


def report(title, R, authors, jlabels):
    print(f"\n=== {title} ===")
    print("   relative affinity (x1000); + = this opinion leans on that passage "
          "more than the other opinions do")
    cols = ["Zone 1 - maximum power", "Zone 2 - twilight", "Zone 3 - lowest ebb",
            "Structural warning", "Framework preamble"]
    print("   " + " " * 10 + "".join(f"{c[:13]:>15}" for c in cols))
    for r, a in enumerate(authors):
        cells = []
        for c in cols:
            js = [j for j in range(R.shape[1]) if jlabels[j] == c]
            cells.append(f"{1000*R[r, js].mean():>14.1f}")
        print(f"   {a:<10}" + "".join(cells))


def main():
    rows = [json.loads(l) for l in open(os.path.join(PROC, "paragraphs.jsonl"))]
    V = np.load(os.path.join(PROC, "embeddings.npy"))
    idx = [i for i, r in enumerate(rows) if not r["is_footnote"] and r["n_words"] >= 25]
    jk = [i for i in idx if rows[i]["case_key"] == "youngstown_1952"
          and rows[i]["author"] == "Jackson"][:JACKSON_MAIN_END]
    J = V[jk] / np.linalg.norm(V[jk], axis=1, keepdims=True)
    jlabels = [section_of(k) for k in range(len(jk))]

    tr = [i for i in idx if rows[i]["case_key"] == "trump_us_2024"]

    S, authors = build(rows, V, tr, J)
    report("AS WRITTEN", double_center(S), authors, jlabels)

    # quotation control
    emb = Embedder()
    keep, texts = [], []
    for i in tr:
        s = strip_quotes(rows[i]["text"])
        if len(s.split()) >= 20:
            keep.append(i)
            texts.append(s)
    Vq = emb.encode(texts)
    Vall = V.copy()
    for i, v in zip(keep, Vq):
        Vall[i] = v
    S2, a2 = build(rows, Vall, keep, J)
    report("QUOTES REMOVED", double_center(S2), a2, jlabels)

    # per-paragraph detail for the two zones that matter
    R = double_center(S)
    print("\n=== which single Jackson paragraph each opinion leans on most ===")
    for r, a in enumerate(authors):
        j = int(R[r].argmax())
        print(f"   {a:<10} -> [{j}] {jlabels[j]:<26} {rows[jk[j]]['text'][:70]}...")


if __name__ == "__main__":
    main()
