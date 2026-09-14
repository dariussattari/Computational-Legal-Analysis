import os, requests, pymupdf
UA={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"}
URLS={
 "trump_hawaii_2018":"/opinions/17pdf/17-965_h315.pdf",
 "seila_law_2020":"/opinions/19pdf/19-7_n6io.pdf",
 "trump_vance_2020":"/opinions/19pdf/19-635_o7jq.pdf",
 "trump_mazars_2020":"/opinions/19pdf/19-715_febh.pdf",
 "collins_2021":"/opinions/20pdf/594us1r55_5468.pdf",
 "biden_neb_2023":"/opinions/22pdf/600us1r56_1o13.pdf",
 "trump_us_2024":"/opinions/23pdf/603us1r57_2dp3.pdf",
}
RAW="data/raw_pdf"
for k,p in URLS.items():
    f=f"{RAW}/{k}.pdf"
    if not os.path.exists(f):
        r=requests.get("https://www.supremecourt.gov"+p,headers=UA,timeout=120)
        open(f,"wb").write(r.content)
    d=pymupdf.open(f)
    t="".join(pg.get_text() for pg in d)
    open(f"{RAW}/{k}.txt","w").write(t)
    print(f"{k:<20} pages={len(d):<4} chars={len(t):>8}  'official'={t.count('official')} 'offcial'={t.count('offcial')}")
# re-extract the two wayback PDFs too
for k in ("noel_canning_2014","zivotofsky_2015"):
    d=pymupdf.open(f"{RAW}/{k}.pdf")
    t="".join(pg.get_text() for pg in d)
    open(f"{RAW}/{k}.txt","w").write(t)
    print(f"{k:<20} pages={len(d):<4} chars={len(t):>8}  'official'={t.count('official')} 'offcial'={t.count('offcial')}")
