# Computational Legal Analysis

Applying the machinery behind large language models — embedding models, attention,
vector geometry — to questions in legal doctrine.

The aim is deliberately narrow. A model cannot say what the Constitution means. It can
say whether a particular reading of a case is unusual *within the case law*, and it can
do so on evidence that has no notion of partisan bias in it. Each study here is meant as
an additional data point alongside traditional legal research, not a replacement for it.

---

## Studies

| Study | Question | Finding | Status |
|---|---|---|---|
| [**Trump-Youngstown**](Trump-Youngstown/) — *A Selective Precedent?* | Did the majority in *Trump v. United States* (2024) use *Youngstown* more selectively than other Supreme Court majorities? | Yes. Across 20 divided presidential-power cases the majority leans to Jackson's three categories and the dissent to his structural warning (16 of 20, mean Δ = +6.3). *Trump* is the extreme at Δ = +24.4, rank 20 of 20. | Complete |

More studies are planned; each gets its own top-level directory on the pattern below.

---

## How a study is organised

Every study directory is self-contained and reproducible on its own. The layout is
described in [`docs/ADDING-A-STUDY.md`](docs/ADDING-A-STUDY.md); in short:

```
<Study-Name>/
  README.md          the paper itself, written for GitHub — prose, figures, headline numbers
  docs/METHOD.md     the formal model, full statistics, robustness checks, judgment calls
  docs/PROVENANCE.md what was tried in order, including the designs that failed
  paper/             the manuscript as submitted, and any hand-made figures
  src/               all analysis code
  data/raw*/         primary sources, committed; derived data is git-ignored
  out/               generated figures and write-ups
  Makefile           every step, in order — `make all` rebuilds from raw text
```

Read the study README first. It is the argument. `docs/METHOD.md` is where you go to
check it, and `docs/PROVENANCE.md` is where you go to see what it replaced.

---

## Method commitments

These hold across studies, and are the reason the results are worth reading:

- **Reproducible from raw text.** A `Makefile` rebuilds every number from the primary
  sources. Derived artefacts are git-ignored so the repository stays reviewable.
- **Public primary sources only.** Court opinions from the Caselaw Access Project and
  the Court's own slip opinions — federal government works, not subject to copyright.
- **Permutation and bootstrap inference.** No asymptotic standard errors are claimed
  anywhere. Where a result depends on a hand-drawn partition, a placebo test asks
  whether a random partition would have done as well.
- **Null results are reported.** The designs that failed are documented alongside the
  one that worked, because the failures are what justify the final design.
- **Judgment calls are isolated and flagged.** Wherever a human decision enters, it
  lives in one file, is named in the README, and can be reversed by a reader who
  disagrees.
- **Measurement claims stay modest.** These methods show *which passages an opinion
  works with*. They do not show whether the opinion is right, and nothing here is
  presented as though they did.

---

## Citing this work

> Darius Sattari, *A Selective Precedent? Youngstown and Presidential Immunity*,
> Computational Legal Analysis (2026).
> https://github.com/dariussattari/Computational-Legal-Analysis

---

## License

- **Code** — MIT. See [`LICENSE`](LICENSE).
- **Writing and figures** (the study READMEs, `docs/`, `paper/`, `out/`) — Creative
  Commons Attribution 4.0 International. See [`LICENSE-TEXT.md`](LICENSE-TEXT.md).
- **Court opinions** in `data/` are federal government works and are not subject to
  copyright.
