"""A full audit trail of the computation for Trump v. United States.

Prints every intermediate quantity between raw text and the final Delta, with
real numbers, so the claim can be checked by hand rather than taken on faith.
Mirrors within.py exactly; the last line asserts the two agree.

Run:  python3 src/walkthrough.py
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from fe import load
from jackson import JACKSON_MAIN_END, section_of

PROC = os.path.join(HERE, "..", "data", "proc")
ZONES = {"Zone 1 - maximum power", "Zone 2 - twilight", "Zone 3 - lowest ebb"}
WARN = {"Structural warning"}
MIN_PARA = 10
BAR = "=" * 78


def main():
    rows, V = load()

    # ------------------------------------------------------------------ 0 --
    print(BAR)
    print("STEP 0   THE UNIT OF ANALYSIS IS ONE PARAGRAPH -> ONE VECTOR")
    print(BAR)
    tr = [i for i, r in enumerate(rows) if r["case_key"] == "trump_us_2024"]
    ex = next(i for i in tr if rows[i]["author"] == "Roberts"
              and "conclusive and preclusive" in rows[i]["text"])
    print(f"\nOne paragraph of the Trump majority ({rows[ex]['n_words']} words):\n")
    print("   " + rows[ex]["text"][:300] + " ...\n")
    v = V[ex]
    print(f"becomes ONE vector of {v.shape[0]} numbers:\n")
    print("   [" + ", ".join(f"{x:+.4f}" for x in v[:8]) + ", ... ]")
    print(f"\n   length (L2 norm) = {np.linalg.norm(v):.4f}   <- every vector is unit length")
    print("""
   The 384 numbers have NO individual meaning. Dimension 17 is not
   "deference" and dimension 200 is not "Article II". They are only
   useful through the dot product of two whole vectors, which for unit
   vectors is the cosine of the angle between them. That single number -
   "how alike are these two paragraphs" - is the ONLY thing taken from
   the model. No features are extracted. Nothing is trained.""")

    # ------------------------------------------------------------------ 1 --
    print("\n" + BAR)
    print("STEP 1   'CATEGORIES' AND 'WARNING' ARE ADDRESSES I CHOSE, NOT MODEL OUTPUT")
    print(BAR)
    jk = [i for i, r in enumerate(rows)
          if r["case_key"] == "youngstown_1952" and r["author"] == "Jackson"][:JACKSON_MAIN_END]
    print(f"\nJackson's main text = {len(jk)} paragraphs, indexed 0..{len(jk)-1}.")
    print("I read the opinion and wrote down which index is which:\n")
    cz = [k for k in range(len(jk)) if section_of(k) in ZONES]
    cw = [k for k in range(len(jk)) if section_of(k) in WARN]
    for k in cz:
        print(f"   ZONES   [{k:>2}]  {rows[jk[k]]['text'][:86]}")
    print()
    for k in cw:
        print(f"   WARNING [{k:>2}]  {rows[jk[k]]['text'][:86]}")
    print(f"""
   That is the whole definition: ZONES = paragraphs {cz}, WARNING = {cw}.
   The model never decides what counts as a category or a warning. It is
   only ever asked "how close is this paragraph to Jackson's paragraph 5".
   This is exactly why the placebo test is the load-bearing check: swap in
   random index sets of the same sizes and the effect disappears.""")

    # ------------------------------------------------------------------ 2 --
    print("\n" + BAR)
    print("STEP 2   THE ONE NUMBER PER PAIR:  S[opinion, jackson-paragraph]")
    print(BAR)
    by = {}
    for i in tr:
        by.setdefault((rows[i]["opinion_type"], rows[i]["author"]), []).append(i)
    by = {k: v for k, v in by.items() if len(v) >= MIN_PARA}
    keys = list(by)
    U = V[jk] / np.linalg.norm(V[jk], axis=1, keepdims=True)

    print(f"\nTrump has {len(keys)} opinions of >= {MIN_PARA} paragraphs:")
    for k in keys:
        print(f"   {k[1]:<10} {k[0]:<11} {len(by[k]):>3} paragraphs")

    S = np.vstack([
        ((V[by[k]] / np.linalg.norm(V[by[k]], axis=1, keepdims=True)) @ U.T).mean(0)
        for k in keys])
    print(f"""
S is {S.shape[0]} x {S.shape[1]}. S[o,j] = average cosine similarity between EVERY
paragraph of opinion o and Jackson's paragraph j. For Roberts that is
{len(by[keys[0]])} paragraphs x {len(jk)} Jackson paragraphs = {len(by[keys[0]])*len(jk)} cosines, averaged down
columns to {len(jk)} numbers.

Raw S, restricted to the 8 columns that matter (x1000):\n""")
    hdr = "".join(f"{('Z'+str(i+1)) if k in cz else 'W'+str(k-cw[0]+1):>8}" for i, k in enumerate(cz + cw))
    print(f"   {'':<11}{hdr}")
    for r, k in enumerate(keys):
        print(f"   {k[1]:<11}" + "".join(f"{1000*S[r, c]:>8.1f}" for c in cz + cw))
    print("""
   Notice these are all large and packed into a narrow band - roughly
   .58 to .75, a spread of about 17 points against a floor of 580. Any two
   paragraphs of judicial prose about presidential power look alike, so the
   raw numbers are nearly useless on their own. Everything interesting is
   in the small deviations, which is what Step 3 isolates.""")

    # ------------------------------------------------------------------ 3 --
    print("\n" + BAR)
    print("STEP 3   DOUBLE-CENTRING: SUBTRACT WHAT IS NOT ABOUT CHOICE")
    print(BAR)
    print(f"""
Two nuisances live in S:

  ROW effect   - Sotomayor's dissent quotes Youngstown constantly, so her
                 whole row runs high. That is verbosity, not selectivity.
  COLUMN effect- some of Jackson's paragraphs are generic and sit close to
                 everything. That is hubness, not selectivity.

Remove both:   R[o,j] = S[o,j] - rowmean(o) - colmean(j) + grandmean

Row means (x1000), i.e. overall Youngstown-flavour of each opinion:""")
    for r, k in enumerate(keys):
        print(f"   {k[1]:<11}{1000*S[r].mean():>8.1f}")
    print("\nColumn means (x1000) for our 8 columns, i.e. how hub-like each is:")
    print(f"   {'':<11}{hdr}")
    print(f"   {'':<11}" + "".join(f"{1000*S[:, c].mean():>8.1f}" for c in cz + cw))

    R = S - S.mean(1, keepdims=True) - S.mean(0, keepdims=True) + S.mean()
    print(f"\nGrand mean (x1000) = {1000*S.mean():.1f}")
    print("\nR after double-centring (x1000) - now signed, and sums to ~0 both ways:\n")
    print(f"   {'':<11}{hdr}")
    for r, k in enumerate(keys):
        print(f"   {k[1]:<11}" + "".join(f"{1000*R[r, c]:>+8.1f}" for c in cz + cw))
    print(f"\n   check: row sums {np.abs(R.sum(1)).max():.2e}, "
          f"col sums {np.abs(R.sum(0)).max():.2e}  (both ~0 by construction)")

    # ------------------------------------------------------------------ 4 --
    print("\n" + BAR)
    print("STEP 4   F = (average over ZONES) - (average over WARNING)")
    print(BAR)
    F = (R[:, cz].mean(1) - R[:, cw].mean(1)) * 1000
    print()
    for r, k in enumerate(keys):
        z, w = 1000 * R[r, cz].mean(), 1000 * R[r, cw].mean()
        print(f"   {k[1]:<11} {k[0]:<11} zones {z:>+7.1f}   warning {w:>+7.1f}"
              f"   F = {z:+.1f} - ({w:+.1f}) = {F[r]:>+7.1f}")

    # ------------------------------------------------------------------ 5 --
    print("\n" + BAR)
    print("STEP 5   DELTA = F(majority) - F(dissent)")
    print(BAR)
    maj = [k for k in keys if k[0] == "majority"]
    dis = [k for k in keys if k[0] == "dissent"]
    fm = np.mean([F[keys.index(k)] for k in maj])
    fd = np.mean([F[keys.index(k)] for k in dis])
    print(f"""
   majority : {', '.join(k[1] for k in maj)}  ->  F = {fm:+.1f}
   dissents : {', '.join(k[1] for k in dis)}  ->  F = {fd:+.1f}  (averaged)

   Delta = {fm:+.1f} - ({fd:+.1f}) = {fm-fd:+.1f}

   Concurrences (Thomas, Barrett) take part in the centring - they help
   define what "typical for this case" means - but are not in Delta, which
   is majority against dissent only.

   This is the within-case comparison. Roberts and Sotomayor are writing
   about the same indictment, the same statutes, the same term, in the same
   register. Whatever separates their rows of R is not subject matter.""")

    w = json.load(open(os.path.join(PROC, "within.json")))
    ref = w["trump_delta"]
    print(f"\n   within.py reports {ref:+.4f};  this walkthrough {fm-fd:+.4f}")
    assert abs((fm - fd) - ref) < 1e-6, "walkthrough disagrees with within.py"
    print("   MATCH\n")

    # ------------------------------------------------------------------ 6 --
    print(BAR)
    print("STEP 6   WHAT 'SIMILARITY' ACTUALLY LOOKS LIKE")
    print(BAR)
    z3 = [k for k in range(len(jk)) if section_of(k) == "Zone 3 - lowest ebb"][0]
    w1 = cw[0]
    for jidx, name in ((z3, "Jackson's Zone 3 (lowest ebb)"),
                       (w1, "Jackson's structural warning")):
        print(f"\n--- closest paragraph in each Trump opinion to {name} ---")
        for k in keys:
            sims = (V[by[k]] / np.linalg.norm(V[by[k]], axis=1, keepdims=True)) @ U[jidx]
            b = int(sims.argmax())
            print(f"   {k[1]:<10} cos={sims[b]:.3f}  {rows[by[k][b]]['text'][:88]}")
    print("""
   Read those and judge for yourself whether "similarity" is picking up
   something real. It is topical and lexical, not logical: the model has no
   idea who is agreeing with Jackson. That is precisely why the paper
   claims ENGAGEMENT and never ENDORSEMENT.""")


if __name__ == "__main__":
    main()
