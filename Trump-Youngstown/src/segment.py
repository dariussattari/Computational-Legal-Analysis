"""Turn raw opinions into a paragraph-level dataset.

Two very different inputs have to end up in the same shape:

  * CAP JSON (pre-2011) - already split per opinion, paragraphs are '\\n'.
  * Slip PDFs (2014-2024) - one blob per case; we recover paragraph breaks from
    layout, because the Court indents the first line of every paragraph and
    nothing else in the body is indented.

Output: one row per paragraph with case/opinion/author/type metadata, plus the
share of the paragraph that sits inside quotation marks (needed later to test
whether similarity to Jackson is driven by people quoting Jackson).
"""

import json
import os
import re
import statistics
import sys

import pymupdf

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
from manifest import CASES

RAW_JSON = os.path.join(HERE, "..", "data", "raw")
RAW_PDF = os.path.join(HERE, "..", "data", "raw_pdf")
PROC = os.path.join(HERE, "..", "data", "proc")

BODY_MIN_SIZE = 10.4          # below this is a footnote or a superscript marker
FOOT_MIN_SIZE = 8.0
INDENT_MIN = 4.0              # pt beyond the body margin that marks a new paragraph
INDENT_MAX = 45.0             # beyond this we are looking at a centred heading

# Running heads and other page furniture that must never enter a paragraph.
JUNK = re.compile(
    r"^\s*(?:"
    r"\d{1,4}"
    r"|Cite as:.*"
    r"|(?:Opinion|Syllabus|Per Curiam)(?:\s+of\s+.*)?"
    r"|[A-Z][A-Za-z'’]+(?:,\s*(?:C\.\s*)?J\.)?,\s*(?:concurring|dissenting).*"
    r"|Opinion of [A-Z][A-Za-z'’]*(?:,\s*(?:C\.\s*)?J\.)?.*"
    r"|[A-Z][A-Z .,'’&()\-]{4,}\s+v\.\s+[A-Z][A-Z .,'’&()\-]{2,}"
    r"|NOTICE:.*|Page Proof Pending Publication"
    r")\s*$"
)

# Start of a separate opinion inside a slip PDF, e.g.
#   CHIEF JUSTICE ROBERTS delivered the opinion of the Court.
#   JUSTICE SOTOMAYOR, with whom JUSTICE KAGAN and JUSTICE JACKSON join, dissenting.
OPINION_START = re.compile(
    r"(?:CHIEF\s+JUSTICE|JUSTICE)\s+([A-Z][A-Z’']{2,})"
    r"(?P<mid>(?:,\s*(?:with\s+whom|and)?[^.]{0,220}?)?)"
    r",?\s*(?P<role>delivered the opinion of the Court[^.]{0,200}?"
    r"|announced the judgment of the Court[^.]{0,200}?"
    r"|concurring(?:\s+in[^.]{0,120})?"
    r"|dissenting(?:\s+in[^.]{0,120})?"
    r"|concurring in the judgment[^.]{0,80})\.",
    re.S,
)

# Same idea for CAP blobs that were not split (Hamdan, Boumediene).
CAP_OPINION_START = re.compile(
    r"(?:^|\n)\s*((?:Mr\.\s+)?(?:Chief\s+)?Justice\s+[A-Z][a-zA-Z’']+"
    r"(?:,\s*(?:with\s+whom[^.]{0,260}?)?)?"
    r",?\s*(?:delivered the opinion|announced the judgment|concurring|dissenting)[^.]{0,400}?\.)"
)


def role_to_type(role: str) -> str:
    r = role.lower()
    if "delivered" in r or "announced" in r:
        return "majority"
    if r.startswith("concurring in the judgment"):
        return "concurrence"
    if "concurring" in r and "dissenting" in r:
        return "mixed"
    if "concurring" in r:
        return "concurrence"
    if "dissenting" in r:
        return "dissent"
    return "other"


def dehyphenate(text: str) -> str:
    # A soft hyphen marks a word broken across lines; the lines were joined with a
    # space, so the space has to go too ("be\u00ad half" -> "behalf").
    text = re.sub(r"\u00ad\s*", "", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text


def quote_share(p: str) -> float:
    """Fraction of characters inside curly or straight double quotes."""
    inside = sum(len(m.group(0)) for m in re.finditer(r"[“\"][^”\"]{8,}[”\"]", p))
    return round(inside / max(len(p), 1), 4)


# --------------------------------------------------------------------------- #
# PDF path
# --------------------------------------------------------------------------- #

def pdf_paragraphs(pdf_path):
    """Recover (text, is_footnote) paragraphs using indentation."""
    doc = pymupdf.open(pdf_path)
    paras, cur, cur_foot = [], [], False

    def flush():
        nonlocal cur, cur_foot
        if cur:
            t = dehyphenate(" ".join(cur))
            t = re.sub(r"\s+", " ", t).strip()
            if len(t) > 1:
                paras.append((t, cur_foot))
        cur = []

    for page in doc:
        lines = []
        for b in page.get_text("dict")["blocks"]:
            for l in b.get("lines", []):
                txt = "".join(s["text"] for s in l["spans"])
                if not txt.strip():
                    continue
                size = max(s["size"] for s in l["spans"] if s["text"].strip())
                lines.append({"x0": l["bbox"][0], "y0": l["bbox"][1], "t": txt, "sz": size})
        if not lines:
            continue

        body = [l for l in lines if l["sz"] >= BODY_MIN_SIZE]
        if not body:
            continue
        try:
            margin = statistics.mode([round(l["x0"], 0) for l in body])
        except statistics.StatisticsError:
            margin = min(round(l["x0"], 0) for l in body)

        for l in lines:
            txt = l["t"].strip()
            if l["sz"] < FOOT_MIN_SIZE or not txt:
                continue
            if JUNK.match(txt):
                flush()
                continue
            is_foot = l["sz"] < BODY_MIN_SIZE
            dx = l["x0"] - margin
            centred = dx > INDENT_MAX
            if centred and len(txt) <= 40:      # section heading: I, II, A, B, 1
                flush()
                continue
            starts_para = (INDENT_MIN < dx <= INDENT_MAX) or (is_foot and re.match(r"^\d+\s", txt))
            if starts_para or is_foot != cur_foot:
                flush()
                cur_foot = is_foot
            cur.append(l["t"])
        flush()
    flush()
    return paras


def split_pdf_opinions(pdf_path):
    """Split a slip PDF into opinions, dropping the syllabus."""
    paras = pdf_paragraphs(pdf_path)
    joined, offsets = [], []
    pos = 0
    for t, foot in paras:
        offsets.append((pos, t, foot))
        joined.append(t)
        pos += len(t) + 1
    blob = "\n".join(joined)

    starts = []
    for m in OPINION_START.finditer(blob):
        author = m.group(1).title()
        typ = role_to_type(m.group("role"))
        starts.append((m.start(), author, typ, re.sub(r"\s+", " ", m.group(0))[:130]))
    if not starts:
        return []

    # Everything before the first opinion header is syllabus / caption matter.
    # Report it so an over-long syllabus (i.e. a missed header) is visible.
    print(f"       syllabus dropped: {starts[0][0]:,} of {len(blob):,} chars "
          f"({100*starts[0][0]/max(len(blob),1):.0f}%)")
    out = []
    for i, (s, author, typ, hdr) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else len(blob)
        chunk = [(t, f) for (o, t, f) in offsets if s <= o < e]
        if chunk:
            out.append({"author_str": author, "type": typ, "header": hdr, "paras": chunk})
    return out


# --------------------------------------------------------------------------- #
# CAP path
# --------------------------------------------------------------------------- #

def split_cap_opinion(text, parent_author="", parent_type="majority"):
    """Split a CAP blob that bundles several opinions into one record.

    Anything before the first matched header is kept as its own opinion; CAP
    sometimes files several opinions under one record whose lead header is too
    long to match (Hamdan's "announced the judgment ... with respect to Parts
    I through IV, VI through VI-D-iii ..." runs for hundreds of characters).
    """
    marks = list(CAP_OPINION_START.finditer(text))
    if len(marks) <= 1:
        return None
    out = []
    if marks[0].start(1) > 500:
        head = text[:marks[0].start(1)]
        out.append({
            "author_str": parent_author,
            "type": parent_type,
            "header": re.sub(r"\s+", " ", head[:130]),
            "text": head,
        })
    for i, m in enumerate(marks):
        s = m.start(1)
        e = marks[i + 1].start(1) if i + 1 < len(marks) else len(text)
        hdr = re.sub(r"\s+", " ", m.group(1))
        am = re.search(r"Justice\s+([A-Z][a-zA-Z’']+)", hdr)
        role = re.search(r"(delivered the opinion|announced the judgment|concurring[^.]*|dissenting[^.]*)", hdr)
        out.append({
            "author_str": am.group(1) if am else "",
            "type": role_to_type(role.group(1)) if role else "other",
            "header": hdr[:130],
            "text": text[s:e],
        })
    return out


def cap_paragraphs(text):
    return [(re.sub(r"\s+", " ", p).strip(), False)
            for p in text.split("\n") if len(p.strip()) > 1]


# --------------------------------------------------------------------------- #

def main():
    os.makedirs(PROC, exist_ok=True)
    rows, opinions = [], []
    pid = 0
    names = {k: (n, y) for k, n, y, _ in CASES}

    for key, name, year, _ in CASES:
        jp = os.path.join(RAW_JSON, f"{key}.json")
        pp = os.path.join(RAW_PDF, f"{key}.pdf")
        ops = []

        if os.path.exists(jp):
            d = json.load(open(jp))
            for sub in d["sub_opinions"]:
                pa = re.sub(r"^(Mr\.\s+)?(Chief\s+)?Justice\s+", "", sub["author_str"]).strip(" ,.")
                pa = pa.split()[0] if pa else ""
                pieces = split_cap_opinion(sub["text"], pa, sub["type"])
                if pieces and len(pieces) > 1:
                    for pc in pieces:
                        ops.append({"author_str": pc["author_str"], "type": pc["type"],
                                    "header": pc["header"], "paras": cap_paragraphs(pc["text"]),
                                    "src": "cap-split"})
                else:
                    a = re.sub(r"^(Mr\.\s+)?(Chief\s+)?Justice\s+", "", sub["author_str"]).strip(" ,.")
                    ops.append({"author_str": a.split()[0] if a else "", "type": sub["type"],
                                "header": sub["author_str"][:130],
                                "paras": cap_paragraphs(sub["text"]), "src": "cap"})
            source = "cap"
        elif os.path.exists(pp):
            for o in split_pdf_opinions(pp):
                o["src"] = "pdf"
                ops.append(o)
            source = "scotus-pdf"
        else:
            print(f"[MISS] {key}")
            continue

        for oi, o in enumerate(ops):
            body = [p for p in o["paras"] if not p[1]]
            foot = [p for p in o["paras"] if p[1]]
            opinions.append({"case_key": key, "case_name": name, "year": year,
                             "opinion_idx": oi, "author": o["author_str"], "type": o["type"],
                             "source": source, "header": o["header"],
                             "n_body": len(body), "n_foot": len(foot)})
            for t, is_foot in o["paras"]:
                words = len(t.split())
                rows.append({"para_id": pid, "case_key": key, "case_name": name, "year": year,
                             "opinion_idx": oi, "author": o["author_str"], "opinion_type": o["type"],
                             "source": source, "is_footnote": bool(is_foot),
                             "n_words": words, "quote_share": quote_share(t), "text": t})
                pid += 1

    with open(os.path.join(PROC, "paragraphs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(PROC, "opinions.json"), "w") as f:
        json.dump(opinions, f, indent=1)

    print(f"{len(rows):,} paragraphs across {len(opinions)} opinions, {len(set(r['case_key'] for r in rows))} cases")


if __name__ == "__main__":
    main()
