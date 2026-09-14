"""Fetch presidential-power opinions from CourtListener.

For each case we resolve a reporter citation to a cluster, then pull every
sub-opinion in that cluster.  CourtListener stores concurrences and dissents as
separate opinion records, which gives us the majority/concurrence/dissent
metadata for free instead of having to parse it out of a slip PDF.
"""

import json
import os
import re
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
from manifest import CASES

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
UA = {"User-Agent": "academic-research-script/1.0"}
API = "https://www.courtlistener.com/api/rest/v4"


def resolve_cluster(reporter, vol, page):
    """Follow the /c/ redirect to an /opinion/<cluster_id>/ URL."""
    url = f"https://www.courtlistener.com/c/{reporter}/{vol}/{page}/"
    r = requests.get(url, headers=UA, allow_redirects=True, timeout=30)
    m = re.search(r"/opinion/(\d+)/", r.url)
    return int(m.group(1)) if m else None


def get_cluster(cid):
    r = requests.get(f"{API}/clusters/{cid}/", headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def get_opinion(url_or_id):
    url = url_or_id if str(url_or_id).startswith("http") else f"{API}/opinions/{url_or_id}/"
    r = requests.get(url, headers=UA, timeout=30)
    r.raise_for_status()
    return r.json()


def best_text(op):
    """CourtListener fills different text fields depending on the source."""
    for field in ("plain_text", "html_with_citations", "html", "xml_harvard", "html_lawbox", "html_columbia"):
        t = op.get(field)
        if t and len(t) > 500:
            return t, field
    return None, None


def main():
    os.makedirs(RAW, exist_ok=True)
    index = []
    for key, name, year, cites in CASES:
        out = os.path.join(RAW, f"{key}.json")
        if os.path.exists(out):
            print(f"[skip] {key}")
            index.append(key)
            continue

        cid = None
        for reporter, vol, page in cites:
            cid = resolve_cluster(reporter, vol, page)
            if cid:
                print(f"[ok]   {key}: {reporter} {vol} {page} -> cluster {cid}")
                break
            time.sleep(0.4)
        if not cid:
            print(f"[MISS] {key}: no cluster for {cites}")
            continue

        cl = get_cluster(cid)
        subs = []
        for ref in cl.get("sub_opinions", []):
            try:
                op = get_opinion(ref)
            except Exception as e:  # noqa: BLE001
                print(f"       ! sub-opinion {ref}: {e}")
                continue
            text, field = best_text(op)
            if not text:
                continue
            subs.append({
                "opinion_id": op.get("id"),
                "type": op.get("type"),
                "author_str": op.get("author_str") or "",
                "text_field": field,
                "text": text,
            })
            time.sleep(0.3)

        rec = {
            "key": key, "case_name": name, "year": year, "cluster_id": cid,
            "cl_case_name": cl.get("case_name"), "date_filed": cl.get("date_filed"),
            "sub_opinions": subs,
        }
        with open(out, "w") as f:
            json.dump(rec, f)
        print(f"       {len(subs)} sub-opinions, {sum(len(s['text']) for s in subs):,} chars")
        index.append(key)
        time.sleep(0.6)

    print(f"\nfetched {len(index)}/{len(CASES)}")


if __name__ == "__main__":
    main()
