"""Constraint <-> autonomy axis: construction, validation, and case scores.

The axis is the difference between the mean embedding of the autonomy anchors
and the mean of the constraint anchors, applied to corpus-centred paragraph
vectors.  Centring matters: raw cosine similarity in this space is dominated by
"this is judicial prose about presidential power", which every paragraph shares.

Nothing here is trustworthy without the validation block at the bottom, which
checks the axis against passages whose direction is not in doubt.
"""

import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from anchors import AUTONOMY, CONSTRAINT
from embed import Embedder

PROC = os.path.join(HERE, "..", "data", "proc")
OUT = os.path.join(HERE, "..", "out")

MIN_WORDS = 25


def load():
    rows = [json.loads(l) for l in open(os.path.join(PROC, "paragraphs.jsonl"))]
    V = np.load(os.path.join(PROC, "embeddings.npy"))
    keep = [i for i, r in enumerate(rows)
            if not r["is_footnote"] and r["n_words"] >= MIN_WORDS]
    return [rows[i] for i in keep], V[keep], rows, V


def build_axis(emb, constraint=CONSTRAINT, autonomy=AUTONOMY):
    A = emb.encode(list(autonomy))
    C = emb.encode(list(constraint))
    ax = A.mean(0) - C.mean(0)
    return ax / np.linalg.norm(ax)


def project(V, axis, center):
    s = (V - center) @ axis
    return s


def main():
    rows, V, all_rows, all_V = load()
    center = all_V.mean(0)
    emb = Embedder()
    axis = build_axis(emb)

    raw = project(V, axis, center)
    mu, sd = raw.mean(), raw.std()
    z = (raw - mu) / sd
    for r, s in zip(rows, z):
        r["axis_z"] = float(s)

    print(f"analysis set: {len(rows):,} paragraphs (body, >={MIN_WORDS} words)\n")

    # ---------------- validation 1: Jackson's own zones ---------------- #
    jk = [r for r in rows if r["case_key"] == "youngstown_1952" and r["author"] == "Jackson"]
    z1 = [r for r in jk if r["text"].startswith("1. When the President acts pursuant")]
    z3 = [r for r in jk if r["text"].startswith("3. When the President takes measures")]
    print("VALIDATION 1 - Jackson's own categories (same author, same page):")
    if z1 and z3:
        print(f"   Zone 1 (maximum power)  axis_z = {z1[0]['axis_z']:+.2f}")
        print(f"   Zone 3 (lowest ebb)     axis_z = {z3[0]['axis_z']:+.2f}")
        print(f"   -> ordered correctly: {z1[0]['axis_z'] > z3[0]['axis_z']}")

    # ---------------- validation 2: majority vs dissent ---------------- #
    print("\nVALIDATION 2 - direction within the immunity/removal cases:")
    for k in ("trump_us_2024", "seila_law_2020", "morrison_1988"):
        sub = [r for r in rows if r["case_key"] == k]
        maj = [r["axis_z"] for r in sub if r["opinion_type"] == "majority"]
        dis = [r["axis_z"] for r in sub if r["opinion_type"] == "dissent"]
        if maj and dis:
            t, p = stats.mannwhitneyu(maj, dis, alternative="two-sided")
            print(f"   {k:<18} majority {np.mean(maj):+.2f} (n={len(maj)})  "
                  f"dissent {np.mean(dis):+.2f} (n={len(dis)})  p={p:.1e}")

    # ---------------- validation 3: leave-one-anchor-out ---------------- #
    print("\nVALIDATION 3 - leave-one-anchor-out stability of the case ranking:")
    base = case_scores(rows)
    keys = [k for k, _ in base]
    rhos = []
    for drop_pole, lst in (("A", CONSTRAINT), ("B", AUTONOMY)):
        for i in range(len(lst)):
            c = [x for j, x in enumerate(CONSTRAINT) if not (drop_pole == "A" and j == i)]
            a = [x for j, x in enumerate(AUTONOMY) if not (drop_pole == "B" and j == i)]
            ax2 = build_axis(emb, c, a)
            r2 = project(V, ax2, center)
            r2 = (r2 - r2.mean()) / r2.std()
            tmp = [dict(r, axis_z=float(s)) for r, s in zip(rows, r2)]
            sc = dict(case_scores(tmp))
            rhos.append(stats.spearmanr([sc[k] for k in keys],
                                        [v for _, v in base]).statistic)
    print(f"   {len(rhos)} refits, Spearman rho vs full axis: "
          f"min={min(rhos):.3f} mean={np.mean(rhos):.3f}")

    # ---------------- case scores ---------------- #
    print("\nCASE SCORES (mean axis_z, majority opinions only):")
    for k, v in sorted(case_scores(rows, majority_only=True), key=lambda x: x[1]):
        yr = next(r["year"] for r in rows if r["case_key"] == k)
        nm = next(r["case_name"] for r in rows if r["case_key"] == k)
        print(f"   {v:+.3f}  {yr}  {nm}")

    with open(os.path.join(PROC, "scored.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    np.save(os.path.join(PROC, "axis.npy"), axis)
    np.save(os.path.join(PROC, "center.npy"), center)
    print(f"\nwrote scored.jsonl ({len(rows):,} rows)")


def case_scores(rows, majority_only=False):
    agg = {}
    for r in rows:
        if majority_only and r["opinion_type"] != "majority":
            continue
        agg.setdefault(r["case_key"], []).append(r["axis_z"])
    return sorted(((k, float(np.mean(v))) for k, v in agg.items()), key=lambda x: x[0])


if __name__ == "__main__":
    main()
