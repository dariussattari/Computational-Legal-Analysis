# Method — Youngstown's Divided Inheritance

The formal companion to [the essay](../README.md): the model, the full statistics,
every robustness check, the null results, and the judgment calls. For the route by
which this design was reached — and the four designs it replaced — see
[PROVENANCE.md](PROVENANCE.md).

A computational study of how *Trump v. United States*, 603 U.S. 593 (2024), inherits
Justice Jackson's concurrence in *Youngstown Sheet & Tube Co. v. Sawyer*,
343 U.S. 579 (1952).

**Claim.** Jackson's concurrence supplies two separable resources: a scheme for
classifying presidential power (the three categories, 343 U.S. 635–638) and a
structural warning about concentrating it (653–655). Across twenty divided
presidential-power cases, majorities draw on the scheme and dissents on the
warning. *Trump* is the most extreme instance in the corpus.

| | |
|---|---|
| Cases / opinions / paragraphs | 27 / 97 / 7,531 |
| Analysis set (body, ≥25 words) | 5,557 |
| Divided cases with both a majority and a dissent | 20 |
| Cases with Δ > 0 | 16 / 20 |
| Mean Δ | +6.27, 95% CI [+2.40, +10.23] |
| Sign test / Wilcoxon | p = .0059 / p = .0021 |
| Placebo partition (corpus / *Trump*) | p = .043 / p = .0035 |
| *Trump v. United States* | Δ = +24.4, rank 20 of 20, z = +1.98 |
| *Trump* as outlier (Grubbs, and max-gap MC) | **not supported**: G = 2.29 vs expected max-gap 2.03–2.21, p = .24–.37 |

---

## Quick start

```bash
make setup          # python deps + the 133 MB ONNX embedding model
make result         # prints the headline table
```

`data/proc/` ships pre-built in the working copy but is **git-ignored**; if you
cloned this fresh, run `make all` (about four minutes, most of it embedding).
Run every target from the study directory (`Trump-Youngstown/`) — some scripts
resolve paths relative to the working directory.

---

## The model

For case *c*, opinion *o*, and paragraph *j* of Jackson's main text, let
`S_c[o,j]` be the mean cosine similarity between the paragraphs of opinion *o*
and Jackson's paragraph *j*. Decompose it as a two-way layout inside the case:

```
S_c[o,j]  =  μ_c + ρ_{c,o} + γ_{c,j} + R_c[o,j]                        (1)

F_c(o)    =  mean_{j ∈ ZONES} R_c[o,j]  −  mean_{j ∈ WARNING} R_c[o,j]  (2)

Δ_c       =  F_c(majority)  −  F_c(dissent)                            (3)
```

- `ρ` absorbs how Youngstown-flavoured an opinion is overall.
- `γ` absorbs hub paragraphs of Jackson's that sit close to everything.
- `R` is the double-centred residual — *selective* engagement. It sums to zero
  along both margins, so no opinion can inflate it by citing *Youngstown* more.
- `Δ` is therefore a contrast of contrasts; case-level effects cancel by
  construction.

**Why the comparison must be made inside a case.** Jackson's zone paragraphs are
about the President's relationship with Congress; his warning paragraphs are
about concentrated power and free government. A detention case therefore
resembles the warning register, and an allocation-of-authority case resembles
the categories, *for reasons unrelated to doctrinal choice*. Comparing
*Medellín* to *Hamdi* measures subject matter. Majority and dissent in the same
case share the facts, statute, era, question presented and vocabulary;
differencing inside the case removes all of it.

This is not a stylistic preference — it is forced by the null results below.
Three across-case designs were tried first and all failed.

---

## Repository layout

```
README.md       the essay - the paper itself, rendered for GitHub
src/            all code (flat, so imports resolve without packaging)
data/raw/       Caselaw Access Project JSON, pre-2011 opinions
data/raw_pdf/   supremecourt.gov PDFs + extracted text, 2014-2024
data/proc/      derived; git-ignored; rebuilt by `make all`
out/            figures (PNG + PDF) and findings.html (the write-up)
paper/          the manuscript (.docx) and the hand-drawn Figure 1
scripts/        model fetcher
docs/           METHOD.md (this file) and PROVENANCE.md - the route to the
                method, for your methods section
```

### `src/` by role

**Pipeline** — run in this order.

| File | Does |
|---|---|
| `manifest.py` | the 27-case list with reporter citations |
| `fetch_cap.py` | pre-2011 opinions from `static.case.law` |
| `fetch_modern.py` | 2014-2024 slip opinions from supremecourt.gov |
| `refetch_pdfs.py` | re-extracts PDFs with PyMuPDF (see *Ligatures* below) |
| `segment.py` | paragraph dataset with case/opinion/author/type metadata |
| `embed.py` | bge-small-en-v1.5 via onnxruntime, CLS pooling, L2-normalised |

**The result.**

| File | Does |
|---|---|
| `within.py` | equations (1)–(3); the headline table |
| `walkthrough.py` | `make walkthrough` — prints every intermediate number for *Trump* and asserts it matches `within.py` |
| `within_robust.py` | placebo partition, role permutation, quotation, boundaries, thresholds |
| `outlier.py` | whether *Trump* is a genuine outlier or the top of a continuum — **it is the latter** |
| `figure3.py` | the paper's figure |
| `jackson.py` | section map of Jackson's concurrence; `strip_quotes` |
| `labels.py` | opinion-level stance labels (see *Judgment calls*) |
| `fe.py` | shared loaders; also the pooled fixed-effects null result |

**Null results the paper reports.**

| File | Result |
|---|---|
| `anchors.py`, `anchors2.py`, `axis.py` | semantic constraint↔autonomy axis; orders Jackson's zones backwards |
| `classify.py` | stance classifier, leave-one-case-out AUC 0.540 |
| `diagnose.py` | same features predict role (0.756) and era (0.820) |
| `fe.py`, `perm.py` | pooled deference direction AUC 0.485, permutation p = .62 |
| `domains.py` | per-domain directions; near-orthogonality is the high-dimensional null, not a finding |
| `drift.py` | Δ has no time trend, r = +0.10, p = .31 |

**Known wart.** `diagnose.py`, `perm.py`, `robust.py` and `refetch_pdfs.py`
were written as one-off scripts and execute on import rather than behind a
`__main__` guard. They are correct and the Makefile invokes them as scripts;
they were left unrefactored so the published numbers stay byte-reproducible.

**Superseded but cited.** `jackson2.py`, `bootstrap.py`, `robust.py`, `figure.py`
are the earlier single-case version of the analysis (Trump's five opinions
only). `within.py` generalises it from n = 1 case to n = 20. `figure.py` still
produces `figure2_limits`. `fetch_cl.py` is a dead end kept for provenance —
CourtListener's API returned 401 and its HTML is bot-gated.

---

## What would break the result, and what happened

| Challenge | Outcome |
|---|---|
| **Placebo partition** — 4,000 random splits of Jackson's paragraphs into groups of the same sizes (3 and 5) | null −0.1 ± 3.7; corpus p = .043, *Trump* p = .0035 |
| **Role permutation** — shuffle which opinions count as majority/dissent | null −0.0 ± 2.7; p = .008 |
| **Quotation** — delete every quoted span and re-embed | +7.0 (from +6.3); *Trump* +28.3 |
| **Section boundaries** — drop each paragraph of each section in turn | +4.0 to +7.8, all positive |
| **Length threshold** — minimum opinion 5 to 30 paragraphs | +6.1 to +6.3, p ≤ .005 throughout |
| **Domain** — security / structure / immunity separately | +6.4 / +5.3 / +7.3 |

The placebo is the one that matters. Had a random split of Jackson's text
produced the same gap, the result would have been about the genre of majorities
and dissents rather than about *Youngstown*.

---

## Judgment calls, flagged

Three places where a human decision enters, all reversible in one file:

1. **`labels.py`** — whether each opinion argues for a broader or narrower
   presidency. Contested cases (*Mistretta*, Rehnquist in *Chadha*, Breyer in
   *Medellín*) are marked `None` and excluded rather than guessed. **These
   labels are not load-bearing for the headline result** — Δ uses only the
   majority/dissent distinction, which is a matter of record. They matter only
   to the null results.
2. **`jackson.py: SECTIONS`** — the partition of the concurrence into zones,
   warning, and the rest. Survives leave-one-out and beats random partitions,
   but was drawn by hand.
3. **`domains.py: DOMAIN`** — assignment of cases to security / structure /
   immunity, used only for the subgroup table.

## Known data issues

- **Ligatures.** The official *preliminary print* PDFs for *Trump*, *Collins*
  and *Biden v. Nebraska* have broken font encoding that silently drops the
  `fi` ligature: "official" extracts as "offcial" 336 times in *Trump* alone —
  the central term of the case. `refetch_pdfs.py` uses the clean slip-opinion
  PDFs instead. Check any new case with
  `grep -c 'offcial' data/raw_pdf/<case>.txt`.
- **Footnotes.** Separated by font size in the PDF path, but *not* in the CAP
  path — pre-2011 cases carry footnotes inline in the body stream. The analysis
  set is body paragraphs of ≥25 words, which removes most but not all of them.
- **Two cases were recovered from the Wayback Machine** (*Noel Canning*,
  *Zivotofsky*): the Court has removed those slip PDFs from its site.
- **OCR.** CAP's text has occasional artefacts — `Scaiia` for Scalia in
  *Clinton v. City of New York*, `de jacto` for *de facto* in Jackson's
  concurrence. Neither affects the analysis; both are visible in `data/raw/`.

## Limits to state in the paper

- Δ measures **emphasis, not endorsement**. It shows which part of Jackson's
  text an opinion works with, and deliberately claims nothing about agreement.
  The null results are why.
- n = 20 cases. The placebo and permutation tests address whether the statistic
  is meaningful, not whether twenty cases represent the doctrine.
- Four cases are excluded for lacking a dissent of sufficient length, including
  *United States v. Nixon* — unanimous, and therefore uninformative under a
  design built on disagreement.

---

## Sources

Caselaw Access Project (`static.case.law`) and supremecourt.gov, both public.
Supreme Court opinions are federal government works and not subject to
copyright. Embeddings: `BAAI/bge-small-en-v1.5`, Apache-2.0.
All inference is by permutation and bootstrap; no asymptotic standard errors
are claimed anywhere.
