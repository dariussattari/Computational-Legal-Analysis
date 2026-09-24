"""Which part of Jackson's concurrence does each Trump v. United States opinion engage?

This plays to what sentence embeddings actually do well - topical retrieval -
rather than asking them to judge doctrinal stance, which the axis experiment
showed they cannot do.

For every paragraph of every opinion in Trump, we find its nearest neighbour
among the paragraphs of Jackson's Youngstown concurrence and record which
section of the concurrence that neighbour belongs to.  Because Roberts and
Sotomayor both quote Jackson directly, the whole thing is run twice: once on
the text as written, once with quoted spans removed, so we can see how much of
any result is just people copying Jackson's words.
"""

import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from embed import Embedder

PROC = os.path.join(HERE, "..", "data", "proc")
OUT = os.path.join(HERE, "..", "out")

# Jackson's main text runs to index 37; after that CAP's paragraph stream is his
# footnotes, which are citation dumps and would pollute the retrieval targets.
JACKSON_MAIN_END = 38

SECTIONS = [
    ("Framework preamble",      range(0, 3)),
    ("Zone 1 - maximum power",  range(3, 4)),
    ("Zone 2 - twilight",       range(4, 5)),
    ("Zone 3 - lowest ebb",     range(5, 6)),
    ("Applying the zones",      range(6, 9)),
    ("Rejecting Art. II claims", range(9, 22)),
    ("Inherent power / emergency", range(22, 33)),
    ("Structural warning",      range(33, 38)),
]


def section_of(i):
    for name, rng in SECTIONS:
        if i in rng:
            return name
    return "Footnotes"


def strip_quotes(t):
    """Remove quoted spans so similarity is not driven by verbatim borrowing."""
    t = re.sub(r"[“\"][^”\"]{8,}[”\"]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main():
    rows = [json.loads(l) for l in open(os.path.join(PROC, "paragraphs.jsonl"))]
    V = np.load(os.path.join(PROC, "embeddings.npy"))

    idx = [i for i, r in enumerate(rows)
           if not r["is_footnote"] and r["n_words"] >= 25]
    jk = [i for i in idx
          if rows[i]["case_key"] == "youngstown_1952" and rows[i]["author"] == "Jackson"]
    jk_main = jk[:JACKSON_MAIN_END]
    J = V[jk_main]
    J = J / np.linalg.norm(J, axis=1, keepdims=True)

    tr = [i for i in idx if rows[i]["case_key"] == "trump_us_2024"]
    emb = Embedder()

    # Re-embed Trump paragraphs with quotations removed.
    stripped = [strip_quotes(rows[i]["text"]) for i in tr]
    keepq = [n for n, s in enumerate(stripped) if len(s.split()) >= 20]
    Vq = emb.encode([stripped[n] for n in keepq])

    print("JACKSON SECTION MAP")
    for name, rng in SECTIONS:
        print(f"   {name:<28} paragraphs {rng.start}-{rng.stop-1}")

    for label, mat, owners in (("AS WRITTEN", V[tr], tr),
                               ("QUOTES REMOVED", Vq, [tr[n] for n in keepq])):
        M = mat / np.linalg.norm(mat, axis=1, keepdims=True)
        sim = M @ J.T
        best = sim.argmax(1)
        print(f"\n=== {label} - nearest Jackson section, share of each opinion ===")
        by = {}
        for k, b in zip(owners, best):
            a = rows[k]["author"]
            by.setdefault(a, []).append(section_of(b))
        order = ["Roberts", "Thomas", "Barrett", "Sotomayor", "Jackson"]
        names = [s for s, _ in SECTIONS]
        hdr = "   " + " " * 11 + "".join(f"{n[:11]:>13}" for n in names)
        print(hdr)
        for a in order:
            if a not in by:
                continue
            tot = len(by[a])
            cells = "".join(f"{100*by[a].count(n)/tot:>12.0f}%" for n in names)
            print(f"   {a:<11}{cells}   (n={tot})")

        # zone-level summary
        print(f"   {'-'*len(hdr)}")
        for a in order:
            if a not in by:
                continue
            tot = len(by[a])
            z1 = by[a].count("Zone 1 - maximum power") / tot
            z3 = by[a].count("Zone 3 - lowest ebb") / tot
            warn = by[a].count("Structural warning") / tot
            print(f"   {a:<11} Zone1={100*z1:4.1f}%  Zone3={100*z3:4.1f}%  Warning={100*warn:4.1f}%")


if __name__ == "__main__":
    main()
