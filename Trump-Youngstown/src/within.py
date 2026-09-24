"""Majority and dissent inside the same case: who keeps Jackson's warning?

The across-case version of this question is confounded and should not be run.
Jackson's zone paragraphs are about the President's relationship with Congress;
his warning paragraphs are about concentrated power and free government.  So a
detention case resembles the warning register and an allocation-of-authority
case resembles the zones for reasons that have nothing to do with doctrinal
choice.  Comparing Medellin to Hamdi measures subject matter.

The fix is the one the earlier failure already pointed to: hold the case fixed.
Majority and dissent in the SAME case share facts, statute, era, vocabulary and
the question presented.  Whatever differs between them is not subject matter.

MODEL.  Within case c, over its opinions o and Jackson's paragraphs j,

    S_c[o,j]  =  mu_c + rho_{c,o} + gamma_{c,j} + R_c[o,j]                 (8)

double-centred inside the case, so rho (an opinion's overall Youngstown
flavour) and gamma (hub paragraphs) are both swept out.  Then

    F_c(o)  =  mean_{j in ZONES} R_c[o,j]  -  mean_{j in WARNING} R_c[o,j]  (9)

    Delta_c =  F_c(majority)  -  mean over dissents F_c(dissent)          (10)

Delta_c is a difference-in-differences: (zones - warning) for the majority
minus the same quantity for the dissent, inside one case.  Case-level effects
cancel in (10) by construction.

TEST.  H0: Delta_c has median zero - majorities and dissents divide Jackson's
concurrence the same way.  Reported with a Wilcoxon signed-rank test, a sign
test, and a bootstrap CI on the mean.  Then: where does Trump fall?
"""

import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from domains import DOMAIN
from fe import load
from jackson import JACKSON_MAIN_END, section_of

PROC = os.path.join(HERE, "..", "data", "proc")
ZONES = {"Zone 1 - maximum power", "Zone 2 - twilight", "Zone 3 - lowest ebb"}
WARN = {"Structural warning"}
MIN_PARA = 10
RNG = np.random.default_rng(343)


def main():
    rows, V = load()
    jk = [i for i, r in enumerate(rows)
          if r["case_key"] == "youngstown_1952" and r["author"] == "Jackson"][:JACKSON_MAIN_END]
    U = V[jk] / np.linalg.norm(V[jk], axis=1, keepdims=True)
    lab = [section_of(k) for k in range(len(jk))]
    cz = [k for k, l in enumerate(lab) if l in ZONES]
    cw = [k for k, l in enumerate(lab) if l in WARN]

    by_case = {}
    for i, r in enumerate(rows):
        if r["case_key"] == "youngstown_1952":
            continue
        by_case.setdefault(r["case_key"], {}).setdefault(
            (r["opinion_type"], r["author"]), []).append(i)

    recs = []
    for key, ops in sorted(by_case.items()):
        ops = {k: v for k, v in ops.items() if len(v) >= MIN_PARA}
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
        fm = np.mean([F[keys.index(k)] for k in maj])
        fd = np.mean([F[keys.index(k)] for k in dis])
        recs.append({"case": key, "year": rows[ops[keys[0]][0]]["year"],
                     "name": rows[ops[keys[0]][0]]["case_name"],
                     "domain": DOMAIN.get(key, "?"),
                     "F_maj": float(fm), "F_dis": float(fd),
                     "delta": float(fm - fd), "n_op": len(keys)})

    recs.sort(key=lambda r: r["year"])
    d = np.array([r["delta"] for r in recs])

    print(f"{len(recs)} cases with both a majority and a dissent "
          f"(opinions of >= {MIN_PARA} paragraphs)\n")
    print("Delta = (zones - warning) for majority, minus the same for dissent")
    print("positive = the majority keeps more of Jackson's machinery,")
    print("           the dissent keeps more of his warning\n")
    print(f"   {'year':<6}{'Delta':>9}   {'domain':<10} case")
    for r in recs:
        star = "  <--" if r["case"] == "trump_us_2024" else ""
        print(f"   {r['year']:<6}{r['delta']:>+9.1f}   {r['domain']:<10}{r['name']}{star}")

    pos = int((d > 0).sum())
    w = stats.wilcoxon(d, alternative="greater")
    sign_p = stats.binomtest(pos, len(d), .5, alternative="greater").pvalue
    bs = np.array([RNG.choice(d, len(d), replace=True).mean() for _ in range(20000)])

    print(f"\n   cases with Delta > 0 : {pos}/{len(d)}   sign test p = {sign_p:.4f}")
    print(f"   mean Delta           : {d.mean():+.2f}  "
          f"95% CI [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]")
    print(f"   median Delta         : {np.median(d):+.2f}")
    print(f"   Wilcoxon signed-rank : W = {w.statistic:.0f}, p = {w.pvalue:.4f} (one-sided)")

    ti = next(i for i, r in enumerate(recs) if r["case"] == "trump_us_2024")
    z = (d[ti] - d.mean()) / d.std(ddof=1)
    rank = int((d < d[ti]).sum()) + 1
    print(f"\n   Trump v. United States: Delta = {d[ti]:+.1f}  "
          f"z = {z:+.2f}  rank {rank} of {len(d)}")
    print(f"   empirical p(Delta >= Trump) = {(d >= d[ti]).mean():.3f}")

    print("\n   mean Delta by domain:")
    for g in ("security", "structure", "immunity"):
        s = [r["delta"] for r in recs if r["domain"] == g]
        if s:
            print(f"      {g:<10} {np.mean(s):+6.2f}  (n={len(s)})")

    json.dump({"recs": recs, "mean": float(d.mean()),
               "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               "wilcoxon_p": float(w.pvalue), "sign_p": float(sign_p),
               "pos": pos, "n": len(d), "trump_delta": float(d[ti]),
               "trump_z": float(z), "trump_rank": rank},
              open(os.path.join(PROC, "within.json"), "w"))


if __name__ == "__main__":
    main()
