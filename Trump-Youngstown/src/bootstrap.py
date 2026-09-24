"""Bootstrap the majority/dissent contrast in engagement with Jackson.

The double-centred affinities are differences of a few thousandths of a cosine.
Before any of it goes in a paper we need to know whether resampling the
paragraphs would move the sign.  Paragraphs are resampled with replacement
within each opinion; the whole double-centring is redone inside every replicate,
because the centring itself depends on the sample.

Two pre-registered contrasts:
  C1  majority - dissents on Jackson's STRUCTURAL WARNING passages
  C2  majority - dissents on Jackson's ZONE machinery (zones 1-3)
The thesis predicts C1 < 0 and C2 > 0.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from jackson import JACKSON_MAIN_END, section_of

PROC = os.path.join(HERE, "..", "data", "proc")
B = 4000
RNG = np.random.default_rng(20240701)

ZONES = {"Zone 1 - maximum power", "Zone 2 - twilight", "Zone 3 - lowest ebb"}
WARN = {"Structural warning"}


def affinity(groups, J, cols_zone, cols_warn):
    # one row per opinion: mean similarity of that opinion's paragraphs to each
    # Jackson paragraph
    S = np.vstack([ ((M / np.linalg.norm(M, axis=1, keepdims=True)) @ J.T).mean(0)
                    for M in groups ])
    R = S - S.mean(1, keepdims=True) - S.mean(0, keepdims=True) + S.mean()
    return R[:, cols_zone].mean(1), R[:, cols_warn].mean(1)


def main():
    rows = [json.loads(l) for l in open(os.path.join(PROC, "paragraphs.jsonl"))]
    V = np.load(os.path.join(PROC, "embeddings.npy"))
    idx = [i for i, r in enumerate(rows) if not r["is_footnote"] and r["n_words"] >= 25]
    jk = [i for i in idx if rows[i]["case_key"] == "youngstown_1952"
          and rows[i]["author"] == "Jackson"][:JACKSON_MAIN_END]
    J = V[jk] / np.linalg.norm(V[jk], axis=1, keepdims=True)
    lab = [section_of(k) for k in range(len(jk))]
    cz = [k for k, l in enumerate(lab) if l in ZONES]
    cw = [k for k, l in enumerate(lab) if l in WARN]

    tr = [i for i in idx if rows[i]["case_key"] == "trump_us_2024"]
    by = {}
    for i in tr:
        by.setdefault(rows[i]["author"], []).append(i)
    order = ["Roberts", "Thomas", "Barrett", "Sotomayor", "Jackson"]
    order = [a for a in order if a in by]
    mats = [V[by[a]] for a in order]
    iM, iS, iJ = order.index("Roberts"), order.index("Sotomayor"), order.index("Jackson")

    z, w = affinity(mats, J, cz, cw)
    obs_c1 = w[iM] - (w[iS] + w[iJ]) / 2
    obs_c2 = z[iM] - (z[iS] + z[iJ]) / 2

    c1s, c2s = np.empty(B), np.empty(B)
    for b in range(B):
        res = [M[RNG.integers(0, len(M), len(M))] for M in mats]
        z, w = affinity(res, J, cz, cw)
        c1s[b] = w[iM] - (w[iS] + w[iJ]) / 2
        c2s[b] = z[iM] - (z[iS] + z[iJ]) / 2

    for name, obs, d, pred in (("C1 warning (predict < 0)", obs_c1, c1s, "<"),
                               ("C2 zones   (predict > 0)", obs_c2, c2s, ">")):
        lo, hi = np.percentile(d, [2.5, 97.5])
        frac = float((d < 0).mean() if pred == "<" else (d > 0).mean())
        print(f"{name}: observed {1000*obs:+.2f}  95% CI [{1000*lo:+.2f}, {1000*hi:+.2f}]"
              f"  P(sign as predicted) = {frac:.3f}")

    import json as _j
    _j.dump({"c1":{"obs":1000*float(obs_c1),"lo":1000*float(np.percentile(c1s,2.5)),"hi":1000*float(np.percentile(c1s,97.5))},
             "c2":{"obs":1000*float(obs_c2),"lo":1000*float(np.percentile(c2s,2.5)),"hi":1000*float(np.percentile(c2s,97.5))}},
            open(os.path.join(PROC,"contrasts.json"),"w"))
    print(f"\nB = {B} bootstrap replicates; paragraphs resampled within opinion.")
    print("Opinion sizes:", {a: len(by[a]) for a in order})


if __name__ == "__main__":
    main()
