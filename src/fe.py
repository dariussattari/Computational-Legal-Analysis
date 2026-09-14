"""Within-case (fixed-effects) estimation of a judicial deference direction.

MODEL.  Let v_p in R^d be the embedding of paragraph p, written in opinion o(p)
of case c(p).  Let s_o = +1 if opinion o argues for broader independent
executive authority and -1 if it argues for greater legislative or judicial
control.  Posit

    v_p  =  alpha_{c(p)}  +  delta * s_{o(p)}  +  eps_p                    (1)

alpha_c is a case fixed effect - everything about subject matter, era, facts
and vocabulary that every paragraph in a case shares.  delta in R^d is the
common deference direction we want.

WHY POOLED ESTIMATION FAILS.  If delta is estimated by pooling paragraphs
across cases, alpha_c is in the error term and is correlated with s_o whenever
the government tended to win (or lose) in a given kind of case.  It does: the
detention cases cluster one way, the removal cases another.  The pooled
estimator is therefore biased toward case identity, which is exactly what the
earlier diagnostic showed - an axis correlating +0.48 with executive-power word
density, and a classifier at AUC 0.54.

IDENTIFICATION.  Difference (1) within a case.  alpha_c is constant inside a
case, so it drops out:

    dbar_c  =  mean(v_p | c, s=+1)  -  mean(v_p | c, s=-1)  =  2*delta + noise

Only cases carrying BOTH signs identify delta - a case where every opinion
points the same way contributes nothing, which is the usual fixed-effects
trade: we buy consistency with the within variation and discard the between.

    delta_hat  =  (1/2) * (1/|C2|) * sum_{c in C2} dbar_c                  (2)

Weighting each case equally rather than each paragraph keeps a single long
opinion (Vinson's 231-paragraph Youngstown dissent) from setting the direction.

SCORING.  A paragraph's deference score is its case-demeaned projection,

    y_p  =  (v_p - vbar_{c(p)}) . delta_hat / ||delta_hat||                (3)

Demeaning at scoring time uses no labels, so it is legitimate for a held-out
case.

EVALUATION.  Leave-one-case-out: delta_hat is estimated on the other cases and
applied to the held-out one.  This is the same protocol the pooled classifier
faced, so the numbers are directly comparable.
"""

import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from labels import HELD_OUT_CASE, LABELS

PROC = os.path.join(HERE, "..", "data", "proc")

# The national-security core of the corpus: war powers, detention, foreign
# affairs, secrecy. Used for a subgroup estimate, not for the main fit.
NATSEC = {"youngstown_1952", "dames_moore_1981", "egan_1988", "hamdi_2004",
          "rasul_2004", "hamdan_2006", "boumediene_2008", "medellin_2008",
          "zivotofsky_2015", "trump_hawaii_2018"}


def load():
    rows = [json.loads(l) for l in open(os.path.join(PROC, "paragraphs.jsonl"))]
    V = np.load(os.path.join(PROC, "embeddings.npy"))
    keep = [i for i, r in enumerate(rows)
            if not r["is_footnote"] and r["n_words"] >= 25]
    return [rows[i] for i in keep], V[keep]


def labelled(rows):
    """Indices of labelled paragraphs, with stance in {+1,-1} and case key."""
    idx, s, case = [], [], []
    for i, r in enumerate(rows):
        if r["case_key"] == HELD_OUT_CASE:
            continue
        lab = LABELS.get((r["case_key"], r["author"]))
        if lab is None:
            continue
        idx.append(i)
        s.append(1 if lab == 1 else -1)
        case.append(r["case_key"])
    return np.array(idx), np.array(s), np.array(case)


def case_means(V, idx, case):
    """vbar_c over the paragraphs available for that case."""
    out = {}
    for c in set(case):
        out[c] = V[idx[case == c]].mean(0)
    return out


def fit_delta(V, idx, s, case, cases_allowed=None):
    """Equation (2): mean over cases of the within-case stance difference."""
    diffs = []
    for c in sorted(set(case)):
        if cases_allowed is not None and c not in cases_allowed:
            continue
        m = case == c
        pos, neg = idx[m & (s == 1)], idx[m & (s == -1)]
        if len(pos) == 0 or len(neg) == 0:
            continue                      # no within-case contrast: no signal
        diffs.append(V[pos].mean(0) - V[neg].mean(0))
    if not diffs:
        return None, 0
    d = np.mean(diffs, axis=0) / 2.0
    return d / np.linalg.norm(d), len(diffs)


def score(V, i_all, delta, cmean, case_of):
    """Equation (3)."""
    Vc = np.vstack([V[i] - cmean[case_of[k]] for k, i in enumerate(i_all)])
    return Vc @ delta


def main():
    rows, V = load()
    idx, s, case = labelled(rows)
    cases = sorted(set(case))
    usable = [c for c in cases
              if len(set(s[case == c])) == 2]
    print(f"labelled paragraphs {len(idx):,} across {len(cases)} cases")
    print(f"cases with a within-case contrast (identifying): {len(usable)}")
    print(f"cases discarded by the fixed effect: "
          f"{sorted(set(cases) - set(usable))}\n")

    # ---------------- leave-one-case-out ----------------
    y_true, y_hat = [], []
    per_case = {}
    for held in cases:
        trn = case != held
        d, n = fit_delta(V, idx[trn], s[trn], case[trn])
        if d is None:
            continue
        m = case == held
        cm = {held: V[idx[m]].mean(0)}
        sc = score(V, idx[m], d, cm, [held] * m.sum())
        yt = (s[m] == 1).astype(int)
        if len(set(yt)) == 2:
            per_case[held] = roc_auc_score(yt, sc)
        y_true.append(yt)
        y_hat.append(sc)
    y_true = np.concatenate(y_true)
    y_hat = np.concatenate(y_hat)
    auc = roc_auc_score(y_true, y_hat)
    acc = ((y_hat > 0) == (y_true == 1)).mean()

    print("LEAVE-ONE-CASE-OUT, within-case fixed-effects estimator")
    print(f"   paragraph-level AUC = {auc:.3f}   accuracy = {acc:.3f}")
    print(f"   (pooled logistic baseline from the earlier run: AUC 0.540)\n")

    print("   per-case AUC, weakest first:")
    for c, a in sorted(per_case.items(), key=lambda kv: kv[1])[:6]:
        print(f"      {c:<20} {a:.3f}")
    print("   strongest:")
    for c, a in sorted(per_case.items(), key=lambda kv: -kv[1])[:4]:
        print(f"      {c:<20} {a:.3f}")

    # opinion-level: average the paragraph scores inside each opinion
    op = {}
    for k, i in enumerate(idx):
        key = (rows[i]["case_key"], rows[i]["author"])
        op.setdefault(key, []).append(k)
    print("\n   opinion-level accuracy (mean score sign vs. label):")
    ok = tot = 0
    pos_idx = {tuple(x): j for j, x in enumerate(zip(case, range(len(case)))) }
    # recompute LOO scores aligned to idx order
    loo = np.full(len(idx), np.nan)
    cursor = 0
    for held in cases:
        m = case == held
        n = m.sum()
        loo[np.where(m)[0]] = y_hat[cursor:cursor + n]
        cursor += n
    for key, ks in op.items():
        mean = np.nanmean(loo[ks])
        lab = LABELS[key]
        tot += 1
        ok += int((mean > 0) == (lab == 1))
    print(f"      {ok}/{tot} = {ok/tot:.3f}")

    np.save(os.path.join(PROC, "loo_scores.npy"), loo)
    json.dump({"auc": float(auc), "acc": float(acc),
               "opinion_acc": ok / tot, "n_identifying": len(usable),
               "per_case": {k: float(v) for k, v in per_case.items()}},
              open(os.path.join(PROC, "fe_results.json"), "w"))


if __name__ == "__main__":
    main()
