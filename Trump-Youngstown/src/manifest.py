# Supreme Court presidential-power corpus, 1952-2024.
# Each entry: key, case name, year, candidate reporter cites (tried in order).
# `held_out` marks cases excluded from classifier training.

CASES = [
    ("youngstown_1952",  "Youngstown Sheet & Tube Co. v. Sawyer", 1952, [("us","343","579")]),
    ("us_v_nixon_1974",  "United States v. Nixon",                1974, [("us","418","683")]),
    ("nixon_gsa_1977",   "Nixon v. Administrator of General Services", 1977, [("us","433","425")]),
    ("dames_moore_1981", "Dames & Moore v. Regan",                1981, [("us","453","654")]),
    ("nixon_fitz_1982",  "Nixon v. Fitzgerald",                   1982, [("us","457","731")]),
    ("chadha_1983",      "INS v. Chadha",                         1983, [("us","462","919")]),
    ("bowsher_1986",     "Bowsher v. Synar",                      1986, [("us","478","714")]),
    ("egan_1988",        "Department of the Navy v. Egan",        1988, [("us","484","518")]),
    ("morrison_1988",    "Morrison v. Olson",                     1988, [("us","487","654")]),
    ("mistretta_1989",   "Mistretta v. United States",            1989, [("us","488","361")]),
    ("clinton_jones_1997","Clinton v. Jones",                     1997, [("us","520","681")]),
    ("clinton_nyc_1998", "Clinton v. City of New York",           1998, [("us","524","417")]),
    ("rasul_2004",       "Rasul v. Bush",                         2004, [("us","542","466")]),
    ("hamdi_2004",       "Hamdi v. Rumsfeld",                     2004, [("us","542","507")]),
    ("hamdan_2006",      "Hamdan v. Rumsfeld",                    2006, [("us","548","557")]),
    ("medellin_2008",    "Medellin v. Texas",                     2008, [("us","552","491")]),
    ("boumediene_2008",  "Boumediene v. Bush",                    2008, [("us","553","723")]),
    ("free_ent_2010",    "Free Enterprise Fund v. PCAOB",         2010, [("us","561","477")]),
    ("noel_canning_2014","NLRB v. Noel Canning",                  2014, [("us","573","513"),("s-ct","134","2550")]),
    ("zivotofsky_2015",  "Zivotofsky v. Kerry",                   2015, [("us","576","1"),("s-ct","135","2076")]),
    ("trump_hawaii_2018","Trump v. Hawaii",                       2018, [("us","585","667"),("s-ct","138","2392")]),
    ("seila_law_2020",   "Seila Law LLC v. CFPB",                 2020, [("us","591","197"),("s-ct","140","2183")]),
    ("trump_vance_2020", "Trump v. Vance",                        2020, [("us","591","786"),("s-ct","140","2412")]),
    ("trump_mazars_2020","Trump v. Mazars USA",                   2020, [("us","591","848"),("s-ct","140","2019")]),
    ("collins_2021",     "Collins v. Yellen",                     2021, [("us","594","220"),("s-ct","141","1761")]),
    ("biden_neb_2023",   "Biden v. Nebraska",                     2023, [("us","600","477"),("s-ct","143","2355")]),
    ("trump_us_2024",    "Trump v. United States",                2024, [("us","603","593"),("s-ct","144","2312")]),
]

HELD_OUT = {"trump_us_2024"}
