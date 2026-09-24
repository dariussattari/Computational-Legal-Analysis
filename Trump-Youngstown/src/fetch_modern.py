"""Fetch modern (post-2010) opinions as official PDFs from supremecourt.gov.

CAP's U.S. Reports coverage stops at volume 561, so anything later comes from
the Court's own site.  Several dockets have more than one PDF (revised slip
opinions, orders, preliminary prints), so we download every candidate and keep
the one that actually looks like the merits opinion.
"""

import os
import re
import sys

import requests
from pypdf import PdfReader

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "data", "raw_pdf")
BASE = "https://www.supremecourt.gov"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

# key -> (case name tokens that must appear, [candidate paths])
CANDIDATES = {
    "noel_canning_2014": (["noel canning"], [
        "/opinions/13pdf/12-1281_6k47.pdf", "/opinions/13pdf/12-1281_bodg.pdf",
        "/opinions/13pdf/12-1281_mc8p.pdf"]),
    "zivotofsky_2015": (["zivotofsky"], [
        "/opinions/14pdf/13-628_1b7d.pdf", "/opinions/14pdf/13-628_3dq3.pdf",
        "/opinions/14pdf/13-628_l5gm.pdf"]),
    "trump_hawaii_2018": (["hawaii"], ["/opinions/17pdf/17-965_h315.pdf"]),
    "seila_law_2020": (["seila"], [
        "/opinions/19pdf/19-7_5i36.pdf", "/opinions/19pdf/19-7_6k47.pdf",
        "/opinions/19pdf/19-7_n6io.pdf"]),
    "trump_vance_2020": (["vance"], ["/opinions/19pdf/19-635_o7jq.pdf"]),
    "trump_mazars_2020": (["mazars"], [
        "/opinions/19pdf/19-715_f63d.pdf", "/opinions/19pdf/19-715_febh.pdf",
        "/opinions/19pdf/19-715_i425.pdf"]),
    "collins_2021": (["collins"], ["/opinions/20pdf/594us1r55_5468.pdf"]),
    "biden_neb_2023": (["nebraska"], ["/opinions/22pdf/600us1r56_1o13.pdf"]),
    "trump_us_2024": (["trump"], ["/opinions/23pdf/603us1r57_2dp3.pdf"]),
}

MERITS = re.compile(r"delivered the opinion of the Court", re.I)


def pdf_text(path):
    try:
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception as e:  # noqa: BLE001
        print(f"      ! parse: {e}")
        return ""


def main():
    os.makedirs(RAW, exist_ok=True)
    for key, (tokens, paths) in CANDIDATES.items():
        out = os.path.join(RAW, f"{key}.txt")
        if os.path.exists(out):
            print(f"[skip] {key}")
            continue
        best, best_score = None, -1
        for p in paths:
            tmp = os.path.join(RAW, f"_tmp_{key}_{p.split('/')[-1]}")
            try:
                r = requests.get(BASE + p, headers=UA, timeout=90)
                if r.status_code != 200 or not r.content.startswith(b"%PDF"):
                    print(f"      x {p} HTTP={r.status_code}")
                    continue
                open(tmp, "wb").write(r.content)
            except Exception as e:  # noqa: BLE001
                print(f"      x {p}: {e}")
                continue
            txt = pdf_text(tmp)
            os.remove(tmp)
            low = txt.lower()
            score = len(txt) if (MERITS.search(txt) and all(t in low for t in tokens)) else -1
            print(f"      . {p.split('/')[-1]:<24} chars={len(txt):>7} merits={bool(MERITS.search(txt))} score={score}")
            if score > best_score:
                best, best_score = txt, score
        if best_score > 0:
            open(out, "w").write(best)
            print(f"[ok]   {key}: {best_score:,} chars")
        else:
            print(f"[MISS] {key}")


if __name__ == "__main__":
    main()
