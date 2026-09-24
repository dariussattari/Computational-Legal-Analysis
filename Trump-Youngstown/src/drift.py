"""Which half of Youngstown does the Court keep using?

Jackson's concurrence does two separable things.  It supplies a CLASSIFICATION
SCHEME - the three categories at 343 U.S. 635-638 - and it supplies a
STRUCTURAL WARNING about concentrated executive power, at 653-655.  The
question is whether later majorities draw on those two parts in stable
proportion, or whether the scheme survives while the warning falls away.

MODEL.  Let u_j, j = 1..J index the paragraphs of Jackson's main text and let
M_c be the majority opinion of case c.  Define

    S[c,j]  =  mean_{p in M_c} cos(v_p, u_j)                               (5)

and decompose it as a two-way layout,

    S[c,j]  =  mu  +  rho_c  +  gamma_j  +  R[c,j]                         (6)

rho_c absorbs how Youngstown-flavoured case c is overall (long opinions, shared
separation-of-powers vocabulary, era).  gamma_j absorbs hub paragraphs of
Jackson's that are close to everything.  The interaction R[c,j] is what is
left: SELECTIVE engagement, case c's pull toward passage j beyond what the two
main effects predict.  R is the double-centred residual and sums to zero along
both margins by construction, so it cannot be inflated by a case simply citing
Youngstown more.

Aggregate R over the two parts of the concurrence,

    Z_c = mean_{j in ZONES} R[c,j],   W_c = mean_{j in WARNING} R[c,j]

and define the FRAMEWORK PREFERENCE of case c

    F_c  =  Z_c  -  W_c                                                    (7)

F_c > 0: the majority leans on Jackson's machinery relative to his warning.
F_c < 0: the reverse.  Because R is doubly centred, F is a contrast of
contrasts - a difference-in-differences in similarity space.

TEST.  Regress F_c on year, F_c = a + b * year + e_c, and ask whether b > 0.
Inference is by permutation: reassign years to cases (which preserves the
multiset of F values and of years, and destroys only their pairing) and refit,
B times.
"""

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from domains import DOMAIN
from fe import load
from jackson import JACKSON_MAIN_END, section_of

PROC = os.path.join(HERE, "..", "data", "proc")
ZONES = {"Zone 1 - maximum power", "Zone 2 - twilight", "Zone 3 - lowest ebb"}
WARN = {"Structural warning"}
B = 10000
RNG = np.random.default_rng(1952)


def build():
    rows, V = load()
    jk = [i for i, r in enumerate(rows)
          if r["case_key"] == "youngstown_1952" and r["author"] == "Jackson"][:JACKSON_MAIN_END]
    U = V[jk] / np.linalg.norm(V[jk], axis=1, keepdims=True)
    lab = [section_of(k) for k in range(len(jk))]
    cz = [k for k, l in enumerate(lab) if l in ZONES]
    cw = [k for k, l in enumerate(lab) if l in WARN]

    cases, S, meta = [], [], []
    for key in sorted({r["case_key"] for r in rows}):
        if key == "youngstown_1952":
            continue                         # Jackson cannot engage himself
        maj = [i for i, r in enumerate(rows)
               if r["case_key"] == key and r["opinion_type"] == "majority"]
        if len(maj) < 15:
            continue
        M = V[maj] / np.linalg.norm(V[maj], axis=1, keepdims=True)
        S.append((M @ U.T).mean(0))
        cases.append(key)
        yr = rows[maj[0]]["year"]
        cites = sum(len(re.findall(r"Youngstown", rows[i]["text"])) for i in maj)
        meta.append({"year": yr, "n": len(maj), "youngstown_mentions": cites,
                     "name": rows[maj[0]]["case_name"], "domain": DOMAIN.get(key, "?")})
    S = np.vstack(S)
    R = S - S.mean(1, keepdims=True) - S.mean(0, keepdims=True) + S.mean()
    F = R[:, cz].mean(1) - R[:, cw].mean(1)
    return cases, meta, F * 1000, R, cz, cw


def perm_slope(x, y, b_obs):
    cnt = 0
    for _ in range(B):
        yp = RNG.permutation(y)
        s = np.polyfit(x, yp, 1)[0]
        if s >= b_obs:
            cnt += 1
    return (cnt + 1) / (B + 1)


def main():
    cases, meta, F, R, cz, cw = build()
    yrs = np.array([m["year"] for m in meta], float)

    print(f"{len(cases)} majority opinions, {int(yrs.min())}-{int(yrs.max())}\n")
    print("FRAMEWORK PREFERENCE  F = (zone machinery) - (structural warning)")
    print("positive = leans on Jackson's categories; negative = on his warning\n")
    print(f"   {'year':<6}{'F':>8}   {'domain':<10}{'cites':>6}  case")
    for i in np.argsort(yrs):
        m = meta[i]
        print(f"   {m['year']:<6}{F[i]:>+8.1f}   {m['domain']:<10}"
              f"{m['youngstown_mentions']:>6}  {m['name']}")

    b, a = np.polyfit(yrs, F, 1)
    r = np.corrcoef(yrs, F)[0, 1]
    p = perm_slope(yrs, F, b)
    print(f"\nOLS  F = {a:.1f} + {b:+.4f} * year     r = {r:+.3f}")
    print(f"     slope = {b*10:+.2f} per decade")
    print(f"     permutation p (one-sided, B={B}) = {p:.4f}")

    # Trump specifically
    ti = cases.index("trump_us_2024")
    z = (F[ti] - F.mean()) / F.std()
    rank = int((F < F[ti]).sum()) + 1
    print(f"\nTrump v. United States: F = {F[ti]:+.1f}  "
          f"(z = {z:+.2f}, rank {rank} of {len(F)})")

    # by domain
    print("\nmean F by domain:")
    for g in ("security", "structure", "immunity"):
        sel = [i for i, m in enumerate(meta) if m["domain"] == g]
        if sel:
            print(f"   {g:<10} {np.mean(F[sel]):+6.1f}  (n={len(sel)})")

    # robustness: drop cases that never mention Youngstown
    keep = [i for i, m in enumerate(meta) if m["youngstown_mentions"] > 0]
    b2 = np.polyfit(yrs[keep], F[keep], 1)[0]
    print(f"\nrobustness - only the {len(keep)} cases that actually cite Youngstown:")
    print(f"   slope = {b2*10:+.2f} per decade "
          f"(p = {perm_slope(yrs[keep], F[keep], b2):.4f})")

    json.dump({"cases": cases, "meta": meta, "F": F.tolist(),
               "slope_per_decade": float(b * 10), "r": float(r), "p": float(p),
               "trump_F": float(F[ti]), "trump_z": float(z)},
              open(os.path.join(PROC, "drift.json"), "w"))


if __name__ == "__main__":
    main()
