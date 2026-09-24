"""Attempts to break the within-case result.

Five challenges, in rough order of how badly each would hurt:

 1. PLACEBO PARTITION.  The strongest objection is that this has nothing to do
    with zones and warnings: majorities apply tests and dissents warn about
    consequences, so ANY split of Jackson's text into a "doctrinal" part and a
    "rhetorical" part might produce a positive Delta.  Replace the real
    partition with random ones of identical sizes (3 and 5 paragraphs) and see
    where the real Delta falls in that distribution.
 2. LABEL PERMUTATION.  Shuffle which opinions in a case count as majority and
    dissent, preserving opinion sizes.
 3. QUOTATION.  Delete quoted spans and re-embed; both sides quote Youngstown.
 4. SECTION BOUNDARIES.  Drop each paragraph of each section in turn.
 5. THRESHOLD.  Vary the minimum opinion length.
"""

import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from embed import Embedder
from fe import load
from jackson import JACKSON_MAIN_END, section_of, strip_quotes

PROC = os.path.join(HERE, "..", "data", "proc")
ZONES = {"Zone 1 - maximum power", "Zone 2 - twilight", "Zone 3 - lowest ebb"}
WARN = {"Structural warning"}
RNG = np.random.default_rng(1789)


def deltas(rows, V, U, cz, cw, min_para=10):
    by_case = {}
    for i, r in enumerate(rows):
        if r["case_key"] == "youngstown_1952":
            continue
        by_case.setdefault(r["case_key"], {}).setdefault(
            (r["opinion_type"], r["author"]), []).append(i)
    out, names = [], []
    for key, ops in sorted(by_case.items()):
        ops = {k: v for k, v in ops.items() if len(v) >= min_para}
        maj = [k for k in ops if k[0] == "majority"]
        dis = [k for k in ops if k[0] == "dissent"]
        if not maj or not dis or len(ops) < 2:
            continue
        keys = list(ops)
        S = np.vstack([
            ((V[ops[k]] / np.linalg.norm(V[ops[k]], axis=1, keepdims=True)) @ U.T).mean(0)
            for k in keys])
        R = S - S.mean(1, keepdims=True) - S.mean(0, keepdims=True) + S.mean()
        F = (R[:, cz].mean(1) - R[:, cw].mean(1)) * 1000
        out.append(np.mean([F[keys.index(k)] for k in maj])
                   - np.mean([F[keys.index(k)] for k in dis]))
        names.append(key)
    return np.array(out), names


def main():
    rows, V = load()
    jk = [i for i, r in enumerate(rows)
          if r["case_key"] == "youngstown_1952" and r["author"] == "Jackson"][:JACKSON_MAIN_END]
    U = V[jk] / np.linalg.norm(V[jk], axis=1, keepdims=True)
    lab = [section_of(k) for k in range(len(jk))]
    cz = [k for k, l in enumerate(lab) if l in ZONES]
    cw = [k for k, l in enumerate(lab) if l in WARN]

    d0, names = deltas(rows, V, U, cz, cw)
    ti = names.index("trump_us_2024")
    print(f"observed: mean Delta = {d0.mean():+.2f}, {int((d0>0).sum())}/{len(d0)} positive, "
          f"Trump = {d0[ti]:+.1f}\n")

    # ---- 1. placebo partitions ----
    J = len(jk)
    B = 4000
    null = np.empty(B)
    nullT = np.empty(B)
    for b in range(B):
        perm = RNG.permutation(J)
        pz, pw = list(perm[:len(cz)]), list(perm[len(cz):len(cz) + len(cw)])
        dd, _ = deltas(rows, V, U, pz, pw)
        null[b] = dd.mean()
        nullT[b] = dd[ti]
    p_mean = float((null >= d0.mean()).mean())
    p_trump = float((nullT >= d0[ti]).mean())
    print("1. PLACEBO PARTITION (random 3-vs-5 splits of Jackson's paragraphs)")
    print(f"   null mean Delta: {null.mean():+.2f} +/- {null.std():.2f}  "
          f"[{np.percentile(null,2.5):+.2f}, {np.percentile(null,97.5):+.2f}]")
    print(f"   observed {d0.mean():+.2f}  ->  p = {p_mean:.4f}")
    print(f"   Trump observed {d0[ti]:+.1f} vs null {nullT.mean():+.1f}"
          f" +/- {nullT.std():.1f}  ->  p = {p_trump:.4f}")

    # ---- 2. majority/dissent label permutation ----
    B2 = 2000
    n2 = np.empty(B2)
    for b in range(B2):
        shuffled = []
        by_case = {}
        for i, r in enumerate(rows):
            if r["case_key"] == "youngstown_1952":
                continue
            by_case.setdefault(r["case_key"], {}).setdefault(
                (r["opinion_type"], r["author"]), []).append(i)
        vals = []
        for key, ops in by_case.items():
            ops = {k: v for k, v in ops.items() if len(v) >= 10}
            roles = [k[0] for k in ops]
            if roles.count("majority") == 0 or roles.count("dissent") == 0 or len(ops) < 2:
                continue
            keys = list(ops)
            perm_roles = list(roles)
            RNG.shuffle(perm_roles)
            S = np.vstack([
                ((V[ops[k]] / np.linalg.norm(V[ops[k]], axis=1, keepdims=True)) @ U.T).mean(0)
                for k in keys])
            R = S - S.mean(1, keepdims=True) - S.mean(0, keepdims=True) + S.mean()
            F = (R[:, cz].mean(1) - R[:, cw].mean(1)) * 1000
            m = [F[i] for i, rr in enumerate(perm_roles) if rr == "majority"]
            dd = [F[i] for i, rr in enumerate(perm_roles) if rr == "dissent"]
            if m and dd:
                vals.append(np.mean(m) - np.mean(dd))
        n2[b] = np.mean(vals) if vals else np.nan
    print("\n2. ROLE PERMUTATION (shuffle which opinions are majority/dissent)")
    print(f"   null {np.nanmean(n2):+.2f} +/- {np.nanstd(n2):.2f}  ->  "
          f"p = {float((n2 >= d0.mean()).mean()):.4f}")

    # ---- 3. quotation ----
    emb = Embedder()
    Vq = V.copy()
    keep, txt = [], []
    for i, r in enumerate(rows):
        s = strip_quotes(r["text"])
        if len(s.split()) >= 20:
            keep.append(i)
            txt.append(s)
    E = emb.encode(txt)
    for i, v in zip(keep, E):
        Vq[i] = v
    Uq = Vq[jk] / np.linalg.norm(Vq[jk], axis=1, keepdims=True)
    dq, nq = deltas(rows, Vq, Uq, cz, cw)
    print("\n3. QUOTATION CONTROL (quoted spans deleted, everything re-embedded)")
    print(f"   mean Delta {dq.mean():+.2f}   positive {int((dq>0).sum())}/{len(dq)}   "
          f"Trump {dq[nq.index('trump_us_2024')]:+.1f}")

    # ---- 4. section boundaries ----
    lo = []
    for drop in cz:
        dd, _ = deltas(rows, V, U, [k for k in cz if k != drop], cw)
        lo.append(dd.mean())
    for drop in cw:
        dd, _ = deltas(rows, V, U, cz, [k for k in cw if k != drop])
        lo.append(dd.mean())
    print("\n4. LEAVE-ONE-PARAGRAPH-OUT of either section")
    print(f"   mean Delta range [{min(lo):+.2f}, {max(lo):+.2f}]  "
          f"all positive: {all(x > 0 for x in lo)}")

    # ---- 5. threshold ----
    print("\n5. MINIMUM OPINION LENGTH")
    for mp in (5, 10, 15, 20, 30):
        dd, nn = deltas(rows, V, U, cz, cw, min_para=mp)
        w = stats.wilcoxon(dd, alternative="greater").pvalue
        print(f"   >= {mp:>2} paragraphs: n={len(dd):>2}  mean {dd.mean():+6.2f}  "
              f"positive {int((dd>0).sum())}/{len(dd)}  Wilcoxon p={w:.4f}")

    json.dump({"placebo_p_mean": p_mean, "placebo_p_trump": p_trump,
               "placebo_null_mean": float(null.mean()), "placebo_null_sd": float(null.std()),
               "role_perm_p": float((n2 >= d0.mean()).mean()),
               "quote_mean": float(dq.mean()),
               "quote_trump": float(dq[nq.index("trump_us_2024")]),
               "loo_range": [float(min(lo)), float(max(lo))]},
              open(os.path.join(PROC, "within_robust.json"), "w"))


if __name__ == "__main__":
    main()
