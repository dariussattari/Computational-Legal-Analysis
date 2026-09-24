"""Opinion-level stance labels.

1 = EXPANSIVE  : the position argued would leave the President with broader
                 independent authority (or more insulation from control by
                 Congress or the courts) than the competing position.
0 = RESTRICTIVE: the position argued would subject the President to greater
                 legislative or judicial control.
None           = excluded: the opinion's executive-power valence is genuinely
                 contested or orthogonal (delegation, severability, standing).

These are my readings, not facts recovered from the text, and they are the one
place where the author's priors enter the pipeline.  They are listed one line
per opinion so a reader can disagree with any single call and re-run.  Labels
attach to OPINIONS, never to paragraphs: paragraph labels are inherited, which
makes this distant supervision, and the accuracy figures should be read with
that in mind.

Trump v. United States carries no label at all - it is the held-out case.
"""

LABELS = {
    # Youngstown: seizure invalid.
    ("youngstown_1952", "Black"): 0,
    ("youngstown_1952", "Frankfurter"): 0,
    ("youngstown_1952", "Douglas"): 0,
    ("youngstown_1952", "Jackson"): 0,
    ("youngstown_1952", "Burton"): 0,
    ("youngstown_1952", "Clark"): 0,
    ("youngstown_1952", "Vinson"): 1,          # would uphold the seizure

    ("us_v_nixon_1974", "Burger"): 0,          # privilege yields to criminal process

    # Nixon v. GSA: recordings statute upheld against the ex-President.
    ("nixon_gsa_1977", "Brennan"): 0,
    ("nixon_gsa_1977", "Stevens"): 0,
    ("nixon_gsa_1977", "White"): 0,
    ("nixon_gsa_1977", "Blackmun"): 0,
    ("nixon_gsa_1977", "Powell"): 0,
    ("nixon_gsa_1977", "Burger"): 1,
    ("nixon_gsa_1977", "Rehnquist"): 1,

    ("dames_moore_1981", "Rehnquist"): 1,      # claims settlement sustained
    ("dames_moore_1981", "Powell"): 1,

    ("nixon_fitz_1982", "Powell"): 1,          # absolute civil immunity
    ("nixon_fitz_1982", "Burger"): 1,
    ("nixon_fitz_1982", "White"): 0,
    ("nixon_fitz_1982", "Blackmun"): 0,

    # Chadha: invalidating the legislative veto removes a congressional control.
    ("chadha_1983", "Burger"): 1,
    ("chadha_1983", "Powell"): None,           # concurs on adjudication grounds
    ("chadha_1983", "White"): 0,
    ("chadha_1983", "Rehnquist"): None,        # severability only

    ("bowsher_1986", "Burger"): 1,
    ("bowsher_1986", "Stevens"): None,         # legislative-power rationale
    ("bowsher_1986", "White"): 0,
    ("bowsher_1986", "Blackmun"): None,

    ("egan_1988", "Blackmun"): 1,              # clearance decisions unreviewable
    ("egan_1988", "White"): 0,

    ("morrison_1988", "Rehnquist"): 0,         # independent counsel upheld
    ("morrison_1988", "Scalia"): 1,            # unitary executive dissent

    ("mistretta_1989", "Blackmun"): None,      # nondelegation, not executive power
    ("mistretta_1989", "Scalia"): None,

    ("clinton_jones_1997", "Stevens"): 0,      # no temporary immunity
    ("clinton_jones_1997", "Breyer"): 1,       # would protect presidential time

    ("clinton_nyc_1998", "Stevens"): 0,        # Line Item Veto Act invalid
    ("clinton_nyc_1998", "Kennedy"): 0,
    ("clinton_nyc_1998", "Scaiia"): 1,         # sic: CAP mis-OCRs "Scalia"
    ("clinton_nyc_1998", "Breyer"): 1,

    ("hamdi_2004", "O’Connor"): 0,             # due process required
    ("hamdi_2004", "Souter"): 0,
    ("hamdi_2004", "Scalia"): 0,               # charge or suspend the writ
    ("hamdi_2004", "Thomas"): 1,

    ("rasul_2004", "Stevens"): 0,
    ("rasul_2004", "Kennedy"): 0,
    ("rasul_2004", "Scalia"): 1,

    ("hamdan_2006", "Stevens"): 0,
    ("hamdan_2006", "Breyer"): 0,
    ("hamdan_2006", "Kennedy"): 0,
    ("hamdan_2006", "Scalia"): 1,
    ("hamdan_2006", "Thomas"): 1,
    ("hamdan_2006", "Alito"): 1,

    ("boumediene_2008", "Kennedy"): 0,
    ("boumediene_2008", "Souter"): 0,
    ("boumediene_2008", "Roberts"): 1,
    ("boumediene_2008", "Scalia"): 1,

    ("medellin_2008", "Roberts"): 0,           # memorandum could not bind states
    ("medellin_2008", "Stevens"): 0,
    ("medellin_2008", "Breyer"): None,         # self-execution, not presidential power

    ("free_ent_2010", "Roberts"): 1,           # dual for-cause removal invalid
    ("free_ent_2010", "Breyer"): 0,

    # Noel Canning: labels track the scope each opinion gives the Clause,
    # not the outcome for the appointments actually at issue.
    ("noel_canning_2014", "Breyer"): 1,
    ("noel_canning_2014", "Scalia"): 0,

    ("zivotofsky_2015", "Kennedy"): 1,         # exclusive recognition power
    ("zivotofsky_2015", "Breyer"): 1,
    ("zivotofsky_2015", "Thomas"): 1,          # broad Vesting Clause reading
    ("zivotofsky_2015", "Roberts"): 0,
    ("zivotofsky_2015", "Scalia"): 0,

    ("trump_hawaii_2018", "Roberts"): 1,
    ("trump_hawaii_2018", "Kennedy"): 1,
    ("trump_hawaii_2018", "Breyer"): 0,
    ("trump_hawaii_2018", "Sotomayor"): 0,

    ("seila_law_2020", "Roberts"): 1,
    ("seila_law_2020", "Thomas"): 1,           # would overrule Humphrey's Executor
    ("seila_law_2020", "Kagan"): 0,

    ("trump_mazars_2020", "Roberts"): 1,       # special limits on congressional subpoenas
    ("trump_mazars_2020", "Thomas"): 1,
    ("trump_mazars_2020", "Alito"): 1,

    ("trump_vance_2020", "Roberts"): 0,        # no absolute immunity from state process
    ("trump_vance_2020", "Kavanaugh"): 1,
    ("trump_vance_2020", "Thomas"): 1,
    ("trump_vance_2020", "Alito"): 1,

    ("collins_2021", "Alito"): 1,
    ("collins_2021", "Thomas"): 1,
    ("collins_2021", "Gorsuch"): 1,
    ("collins_2021", "Kagan"): 0,
    ("collins_2021", "Sotomayor"): 0,

    ("biden_neb_2023", "Roberts"): 0,          # executive action struck down
    ("biden_neb_2023", "Barrett"): 0,
    ("biden_neb_2023", "Kagan"): 1,
}

HELD_OUT_CASE = "trump_us_2024"
