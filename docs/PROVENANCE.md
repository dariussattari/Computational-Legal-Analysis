# How this analysis reached its method

A record of what was tried, in order, and why each step forced the next. The
final design is not the first thing attempted; it is what survived four
failures. The failures are reportable, and the paper is stronger for stating
them than for presenting the last attempt as though it were the first.

---

## 0. Getting the text

Three sources were tried before two worked.

| Source | Outcome |
|---|---|
| CourtListener API | 401 on `/clusters/` and `/opinions/`; needs a token |
| CourtListener HTML | HTTP 202 with an empty body — bot-gated |
| Cornell LII | Cloudflare interstitial |
| **Caselaw Access Project** | works; U.S. Reports through **vol. 561** (2010) |
| **supremecourt.gov** | works; slip opinions 2014 onward |

CAP was the better find: it already separates each case into majority,
concurrence and dissent with an author string, which is the metadata the whole
design turns on. The Court's PDFs required recovering that structure by hand
(`segment.py`), using the fact that the Court indents the first line of every
paragraph and nothing else in the body is indented.

Two gaps had to be filled from the Wayback Machine — the Court has removed the
*Noel Canning* and *Zivotofsky* slip PDFs, and its term-19 index page has lost
its PDF links entirely.

**The ligature bug.** PyPDF silently dropped the `fi` ligature from the official
*preliminary print* PDFs: "official" became "offcial" **336 times in *Trump***.
Since "official act" is the central term of that case, this would have
corrupted every measurement involving it. PyMuPDF fixed most files; for
*Trump*, *Collins* and *Biden v. Nebraska* the preliminary prints are encoded
badly enough that the slip opinions had to be used instead.

**Segmentation bugs caught by auditing rather than by the code running.** Both
were silent:
- *Hamdan* lost Stevens's entire majority — 103,000 characters — because CAP
  filed six opinions under one record whose lead header ("announced the
  judgment … with respect to Parts I through IV, VI through VI-D-iii …") was
  too long to match.
- *Seila Law* lost its majority because Roberts "delivered the opinion of the
  Court **with respect to Parts I, II, and III**" does not end where the regex
  expected a period.

Neither produced an error. Both were found by printing the opinion inventory
per case and checking that every case had a majority.

---

## 1. First attempt: a semantic constraint ↔ autonomy axis

Build an axis from anchor phrases, project paragraph embeddings onto it, read
off where each case falls. This is the obvious design and it is wrong.

Validated against passages whose direction is not in doubt, it placed Jackson's
**lowest ebb** paragraph *further toward autonomy* than his **maximum power**
paragraph, and put *Youngstown* itself — the canonical limiting case — on the
autonomy side. Inspecting the poles explained it: the constraint extreme filled
with statutory citation boilerplate (*Chadha*'s list of veto provisions), and
the autonomy extreme included Sotomayor's dissent in *Trump v. Hawaii*, which
attacks executive power.

The axis correlated **+0.48** with nothing more than the density of
executive-power vocabulary. It measured *how much* a passage is about exclusive
presidential power, not whether it endorsed it.

Rebuilding the anchors as minimal pairs — identical subject, opposite holding,
so the difference cancels topic — cut the confound to +0.37 and **still ordered
the zones backwards**. Sentence embeddings of this class do not encode
negation or stance reliably.

## 2. Second attempt: a supervised stance classifier

Label opinions expansive or restrictive, train on pre-2024 cases, hold out
*Trump*. Evaluated leave-one-**case**-out so no fold is scored on a case it
trained on: **AUC 0.540** against a 0.513 majority-class baseline.

The diagnostic that made this interpretable: the same features and the same
protocol reach **0.756** for majority-vs-dissent and **0.820** for pre- vs
post-2000. The pipeline was not broken. Doctrinal commitment is simply what
fails to transfer across subject matters — stance lives in argument structure
and the handling of precedent, not in paragraph vocabulary that carries from
steel seizures to security clearances to student loans.

The classifier's *Trump* predictions happened to rank in the intuitive order
(Roberts highest, dissents lowest). At AUC 0.54 that ordering is not reportable
and is not reported.

## 3. Third attempt: a within-case fixed-effects deference direction

If pooling fails because case identity confounds stance, difference it out.
Posit `v_p = α_c + δ·s_o + ε_p` and estimate δ from within-case contrasts only.

**AUC 0.485** — below chance — against a permutation null of 0.499 ± 0.029,
**p = .62**. The observed value sits inside the null.

Splitting δ by doctrinal domain did not rescue it: within-domain AUCs were
0.47–0.53, and although the three domain directions came out nearly orthogonal,
**that is what random vectors look like in 384 dimensions** (sd ≈ 1/√384 ≈
0.05). The cosines of −0.07 and +0.09 are within noise of zero and were not
treated as a finding.

Conclusion: no linear deference direction is recoverable from these embeddings.

## 4. Fourth attempt: across-case drift

Measure, for each case's majority, its relative engagement with Jackson's
categories versus his warning, and regress on year. No trend: r = +0.10,
p = .31.

More importantly, this design **reintroduces the confound from step 1**.
Jackson's zone paragraphs are about President–Congress relations; his warning
paragraphs are about concentrated power and free government. A detention case
resembles the warning register and an allocation case resembles the categories
for reasons that have nothing to do with doctrinal choice. Comparing *Medellín*
to *Hamdi* measures subject matter.

## 5. What worked

Keep the quantity from step 4 but make the comparison **inside a case**, where
majority and dissent share facts, statute, era, question presented and
vocabulary. Δ becomes a difference-in-differences and case effects cancel by
construction.

Δ > 0 in 16 of 20 cases; mean +6.27; sign test p = .0059; Wilcoxon p = .0021.
*Trump* is rank 20 of 20 at Δ = +24.4.

The decisive check is the **placebo partition**: replace the real
zones/warning split with 4,000 random splits of Jackson's paragraphs into
groups of the same sizes. Null = −0.08 ± 3.66, so corpus p = .043 and *Trump*
p = .0035. Without this test the result could have been about the genre of
majorities and dissents — majorities apply tests, dissents warn about
consequences — rather than about *Youngstown* specifically.

---

## The methodological point for the paper

Embedding similarity reliably indicates **which passages an opinion works
with**. It carries no usable information about **whether the opinion agrees**.
Steps 1–3 establish the second claim empirically rather than as a caveat; the
final design uses only the first.

That is why the paper measures *engagement*, not *endorsement*, and why the
strongest doctrinal claim in it — that "conclusive and preclusive" is a phrase
from Jackson's lowest-ebb passage, where it describes a claim that "must be
scrutinized with caution" — rests on a citation rather than on a model. The
measurement then shows that the surrounding warning is exactly the material
this majority engages least.
