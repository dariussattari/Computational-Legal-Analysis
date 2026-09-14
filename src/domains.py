"""Is "pro-executive" one direction, or several?

The pooled fixed-effects fit came in at AUC 0.485 - below chance - and the
per-case pattern was not random: removal and appointments cases all landed
around 0.37-0.40 while detention and foreign-affairs cases landed at 0.55-0.67.
A direction estimated mostly on one family points the wrong way in the other.

That is the signature of a misspecified model.  Equation (1) assumed a single
delta.  Relax it to one direction per doctrinal domain,

    v_p  =  alpha_{c(p)}  +  delta_{g(c(p))} * s_{o(p)}  +  eps_p          (4)

where g(c) is the domain of case c, and test three things:

  (i)   cos(delta_g, delta_h) for each pair of domains.  If "expansive" were
        one rhetorical posture these would be near 1.
  (ii)  within-domain leave-one-case-out AUC.  If (4) is right, each domain
        should predict its own held-out cases.
  (iii) cross-domain transfer.  Fit on g, test on h.  Under (4) with
        near-orthogonal directions this should collapse to chance, and under a
        sign flip it should fall below it.

The claim being tested is substantive, not technical: whether the language of
executive power is a single dimension across national security, internal
structure, and personal immunity.
"""

import itertools
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from fe import fit_delta, labelled, load
from labels import LABELS

PROC = os.path.join(HERE, "..", "data", "proc")

DOMAIN = {
    # war powers, detention, foreign affairs, secrecy
    "youngstown_1952": "security", "dames_moore_1981": "security",
    "egan_1988": "security", "hamdi_2004": "security", "rasul_2004": "security",
    "hamdan_2006": "security", "boumediene_2008": "security",
    "medellin_2008": "security", "zivotofsky_2015": "security",
    "trump_hawaii_2018": "security",
    # internal structure: removal, appointments, the legislative veto, delegation
    "chadha_1983": "structure", "bowsher_1986": "structure",
    "morrison_1988": "structure", "mistretta_1989": "structure",
    "clinton_nyc_1998": "structure", "free_ent_2010": "structure",
    "noel_canning_2014": "structure", "seila_law_2020": "structure",
    "collins_2021": "structure", "biden_neb_2023": "structure",
    # the President personally: privilege, immunity, compulsory process
    "us_v_nixon_1974": "immunity", "nixon_gsa_1977": "immunity",
    "nixon_fitz_1982": "immunity", "clinton_jones_1997": "immunity",
    "trump_vance_2020": "immunity", "trump_mazars_2020": "immunity",
    "trump_us_2024": "immunity",
}


def loo_auc(V, idx, s, case, fit_cases, test_cases):
    """Fit on fit_cases (minus the held-out one), score each test case."""
    yt_all, ys_all, per = [], [], {}
    for held in sorted(set(test_cases)):
        allowed = set(fit_cases) - {held}
        d, n = fit_delta(V, idx, s, case, cases_allowed=allowed)
        if d is None or n == 0:
            continue
        m = case == held
        if m.sum() == 0:
            continue
        cm = V[idx[m]].mean(0)
        sc = (V[idx[m]] - cm) @ d
        yt = (s[m] == 1).astype(int)
        if len(set(yt)) < 2:
            continue
        per[held] = roc_auc_score(yt, sc)
        yt_all.append(yt)
        ys_all.append(sc)
    if not yt_all:
        return None, {}
    return roc_auc_score(np.concatenate(yt_all), np.concatenate(ys_all)), per


def main():
    rows, V = load()
    idx, s, case = labelled(rows)
    dom = np.array([DOMAIN[c] for c in case])
    groups = ["security", "structure", "immunity"]

    print("cases per domain (identifying ones only):")
    deltas = {}
    for g in groups:
        cs = sorted({c for c in set(case) if DOMAIN[c] == g
                     and len(set(s[case == c])) == 2})
        d, n = fit_delta(V, idx, s, case, cases_allowed=set(cs))
        deltas[g] = d
        print(f"   {g:<10} n={n:<3} {cs}")

    # ---- (i) angles between the domain directions ----
    print("\n(i) cosine between domain deference directions")
    print("    if 'pro-executive' were one posture these would be near 1")
    for a, b in itertools.combinations(groups, 2):
        c = float(deltas[a] @ deltas[b])
        print(f"    cos(delta_{a}, delta_{b}) = {c:+.3f}   "
              f"angle = {np.degrees(np.arccos(np.clip(c,-1,1))):.0f} deg")

    # ---- (ii) within-domain LOO ----
    print("\n(ii) within-domain leave-one-case-out AUC")
    for g in groups:
        cs = {c for c in set(case) if DOMAIN[c] == g}
        a, per = loo_auc(V, idx, s, case, cs, cs)
        if a is not None:
            print(f"    fit {g:<10} -> test {g:<10} AUC = {a:.3f}  "
                  f"(cases: {len(per)})")

    # ---- (iii) cross-domain transfer ----
    print("\n(iii) cross-domain transfer AUC")
    print("     rows = domain the direction was fitted on")
    hdr = "".join(f"{g[:9]:>12}" for g in groups)
    print(f"     {'':<11}{hdr}")
    M = np.full((3, 3), np.nan)
    for i, gf in enumerate(groups):
        cells = []
        for j, gt in enumerate(groups):
            fit_cs = {c for c in set(case) if DOMAIN[c] == gf}
            test_cs = {c for c in set(case) if DOMAIN[c] == gt}
            a, _ = loo_auc(V, idx, s, case, fit_cs, test_cs)
            M[i, j] = a if a is not None else np.nan
            cells.append(f"{a:>12.3f}" if a is not None else f"{'-':>12}")
        print(f"     {gf:<11}{''.join(cells)}")

    json.dump({"cos": {f"{a}|{b}": float(deltas[a] @ deltas[b])
                       for a, b in itertools.combinations(groups, 2)},
               "transfer": M.tolist(), "groups": groups},
              open(os.path.join(PROC, "domain_results.json"), "w"))


if __name__ == "__main__":
    main()
