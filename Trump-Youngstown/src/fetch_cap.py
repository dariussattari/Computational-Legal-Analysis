"""Fetch pre-2011 opinions from the Caselaw Access Project static archive.

CAP already splits each case into separate opinions tagged majority /
concurrence / dissent with an author string, which is the metadata the whole
experiment turns on.  We locate a case by volume + first page via the volume's
CasesMetadata.json rather than guessing the file's ordinal suffix.
"""

import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
from manifest import CASES

HERE = os.path.dirname(__file__)
RAW = os.path.join(HERE, "..", "data", "raw")
CAP = "https://static.case.law"
_meta_cache = {}


def volume_meta(vol):
    if vol not in _meta_cache:
        r = requests.get(f"{CAP}/us/{vol}/CasesMetadata.json", timeout=90)
        r.raise_for_status()
        _meta_cache[vol] = r.json()
    return _meta_cache[vol]


def find_case(vol, page):
    for c in volume_meta(vol):
        if str(c.get("first_page")) == str(page):
            return c
    return None


def main():
    os.makedirs(RAW, exist_ok=True)
    got = miss = 0
    for key, name, year, cites in CASES:
        out = os.path.join(RAW, f"{key}.json")
        if os.path.exists(out):
            print(f"[skip] {key}")
            got += 1
            continue
        us = next((c for c in cites if c[0] == "us"), None)
        if not us:
            continue
        _, vol, page = us
        try:
            meta = find_case(vol, page)
        except Exception as e:  # noqa: BLE001
            print(f"[MISS] {key}: volume {vol} unavailable ({e})")
            miss += 1
            continue
        if not meta:
            print(f"[MISS] {key}: no case at {vol} U.S. {page}")
            miss += 1
            continue

        r = requests.get(f"{CAP}/us/{vol}/cases/{meta['file_name']}.json", timeout=90)
        r.raise_for_status()
        d = r.json()
        subs = []
        for o in d.get("casebody", {}).get("opinions", []):
            txt = o.get("text") or ""
            if len(txt) < 400:
                continue
            subs.append({
                "type": o.get("type"),
                "author_str": (o.get("author") or "").strip(),
                "text_field": "cap",
                "text": txt,
            })
        rec = {
            "key": key, "case_name": name, "year": year,
            "source": "cap", "cap_id": d.get("id"),
            "cl_case_name": d.get("name_abbreviation"),
            "date_filed": d.get("decision_date"),
            "sub_opinions": subs,
        }
        with open(out, "w") as f:
            json.dump(rec, f)
        types = {}
        for s in subs:
            types[s["type"]] = types.get(s["type"], 0) + 1
        print(f"[ok]   {key:<20} {len(subs)} opinions {types} {sum(len(s['text']) for s in subs):,} chars")
        got += 1
        time.sleep(0.4)

    print(f"\nCAP: {got} ok, {miss} missing")


if __name__ == "__main__":
    main()
