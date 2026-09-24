"""Is Trump an outlier, or just the largest of twenty draws?

Being rank 20 of 20 is weak evidence on its own: in any sample of twenty,
something is the maximum, and the chance that it happens to be the case you
care about is 1/20 = .05 under pure exchangeability.  That is the whole content
of the empirical p = .050 reported in within.py, and it is not a test of
whether Trump is UNUSUAL - only of whether it is first.

The sharper question is whether the GAP between Trump and the rest is bigger
than a single sample of twenty should produce.  Statistic:

    G  =  ( Delta_Trump  -  mean of the other 19 )  /  sd of the other 19

G must be compared against the distribution of the LARGEST such gap in a sample
of twenty, not against a standard normal - otherwise the fact that Trump was
selected for being extreme is double-counted.  Two nulls, because the answer
should not depend on a normality assumption:

  (a) parametric   - draw 20 from N(mean_19, sd_19)
  (b) nonparametric- resample 20 with replacement from the other 19

Under each, compute the max studentised gap and see how often it reaches G.
"""

import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
PROC = os.path.join(HERE, "..", "data", "proc")
B = 200000
RNG = np.random.default_rng(1952)


def max_gap(sample):
    """Largest studentised gap in a sample, leaving the max out of the moments."""
    i = int(np.argmax(sample))
    rest = np.delete(sample, i)
    sd = rest.std(ddof=1)
    if sd == 0:
        return 0.0
    return (sample[i] - rest.mean()) / sd


def main():
    W = json.load(open(os.path.join(PROC, "within.json")))
    recs = W["recs"]
    d = np.array([r["delta"] for r in recs])
    names = [r["name"] for r in recs]
    ti = names.index("Trump v. United States")

    rest = np.delete(d, ti)
    G = (d[ti] - rest.mean()) / rest.std(ddof=1)

    print(f"n = {len(d)} cases")
    print(f"Trump             Delta = {d[ti]:+.2f}")
    print(f"other 19          mean  = {rest.mean():+.2f}   sd = {rest.std(ddof=1):.2f}")
    print(f"next highest      {sorted(d)[-2]:+.2f}  ({names[int(np.argsort(d)[-2])]})")
    print(f"\nstudentised gap   G = {G:.3f}")

    # ---- null (a): normal ----
    sim = RNG.normal(rest.mean(), rest.std(ddof=1), size=(B, len(d)))
    ga = np.array([max_gap(s) for s in sim[:20000]])
    pa = float((ga >= G).mean())

    # ---- null (b): resample the other 19 ----
    simb = RNG.choice(rest, size=(20000, len(d)), replace=True)
    gb = np.array([max_gap(s) for s in simb])
    pb = float((gb >= G).mean())

    print(f"\nnull distribution of the LARGEST studentised gap in a sample of {len(d)}:")
    print(f"   (a) normal        mean {ga.mean():.2f}  95th pct {np.percentile(ga,95):.2f}"
          f"   ->  p = {pa:.4f}")
    print(f"   (b) resampled     mean {gb.mean():.2f}  95th pct {np.percentile(gb,95):.2f}"
          f"   ->  p = {pb:.4f}")

    # Grubbs, for the reader who wants a named test
    N = len(d)
    t = stats.t.ppf(1 - .05 / (2 * N), N - 2)
    crit = (N - 1) / np.sqrt(N) * np.sqrt(t ** 2 / (N - 2 + t ** 2))
    g_grubbs = (d[ti] - d.mean()) / d.std(ddof=1)
    print(f"\nGrubbs' test (single outlier, alpha = .05):")
    print(f"   G_grubbs = {g_grubbs:.3f}   critical = {crit:.3f}   "
          f"-> {'REJECT exchangeability' if g_grubbs > crit else 'do not reject'}")

    print(f"""
Reading:
   The corpus-wide pattern (majorities keep the categories) is solid.
   Whether Trump is a genuine OUTLIER rather than simply the top of a
   continuum is the weaker of the two claims, and these numbers say how
   much weight it will bear.""")

    json.dump({"G": float(G), "p_normal": pa, "p_resampled": pb,
               "grubbs": float(g_grubbs), "grubbs_crit": float(crit),
               "trump": float(d[ti]), "rest_mean": float(rest.mean()),
               "rest_sd": float(rest.std(ddof=1))},
              open(os.path.join(PROC, "outlier.json"), "w"))


if __name__ == "__main__":
    main()
