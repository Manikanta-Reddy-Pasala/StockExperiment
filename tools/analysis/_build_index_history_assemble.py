import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import os,glob,re,json,csv,io
import pdfplumber
WORK="/tmp/idx_all"; WB="src/data/symbols/wayback_snapshots"
REVIEWS=[("2021","03"),("2021","09"),("2022","03"),("2022","09"),("2023","03"),
         ("2023","09"),("2024","03"),("2024","09"),("2025","03"),("2025","09"),("2026","03")]
def rdate(y,m): return f"{y}{m}{'31' if m=='03' else '30'}"
def pdf_syms(p):
    out=set()
    try:
        with pdfplumber.open(p) as pdf:
            for pg in pdf.pages:
                for tb in (pg.extract_tables() or []):
                    for row in tb:
                        c0=(row[0] or "").strip() if row else ""
                        if c0 and c0.lower()!="symbol" and re.fullmatch(r'[A-Z0-9][A-Z0-9&.\-]{0,14}',c0): out.add(c0)
    except: pass
    return out
def per_to_rev(per):  # Mar2021 -> 2021,03
    return per[3:], ('03' if per[:3]=='Mar' else '09')
# --- N50, Next50, Smallcap, Midcap from factsheet zips ---
zips={"n50":{}, "next50":{}, "smallcap250":{}, "midcap150":{}}
zp={"n50":"NIFTY_50_","next50":"NIFTY_Next_50_","smallcap250":"NIFTY_Smallcap_250_","midcap150":"NIFTY_Midcap_150_"}
for per in os.listdir(WORK):
    d=f"{WORK}/{per}"
    if not os.path.isdir(d): continue
    y,m=per_to_rev(per)
    for k,pp in zp.items():
        fs=glob.glob(f"{d}/{pp}*.pdf")
        if fs:
            s=pdf_syms(fs[0])
            if len(s)>15: zips[k][rdate(y,m)]=s
# --- wayback CSV captures ---
def load_csv(p):
    with open(p) as f:
        return {r["Symbol"].strip() for r in csv.DictReader(f) if r.get("Symbol") and r.get("Series","EQ").strip() in("EQ","")}
wb_caps={"n100":{},"next50":{},"n500":{},"smallcap250":{}}
for p in glob.glob(f"{WB}/*.csv"):
    b=os.path.basename(p)[:-4]; k="_".join(b.split("_")[:-1]); ts=b.split("_")[-1]
    if k in wb_caps and ts.isdigit(): wb_caps[k][ts]=load_csv(p)
def nearest_rev(ts):  # map YYYYMMDD capture to nearest review (y, m)
    y=ts[:4]; mm=int(ts[4:6]); return (y, '03' if mm<=6 else '09')
# merge wayback into zips by nearest review (only fill if zip missing)
for k in ("next50","smallcap250"):
    for ts,s in wb_caps.get(k,{}).items():
        y,m=nearest_rev(ts); rd=rdate(y,m)
        if rd not in zips[k] and len(s)>15: zips[k][rd]=s
# --- assemble per review with carry-forward ---
def series(raw):  # raw {rdate:set} -> filled for all REVIEWS via carry-forward
    out={}; last=None
    for y,m in REVIEWS:
        rd=rdate(y,m)
        if rd in raw: last=raw[rd]
        if last is not None: out[rd]=set(last)
    return out
n50=series(zips["n50"]); next50=series(zips["next50"])
n100={rd:(n50.get(rd,set())|next50.get(rd,set())) for rd in n50}
# N500 from xlsx-delta build
n500_raw={k:set(v) for k,v in json.load(open("/tmp/n500_semi.json")).items() if k[:4]<="2026" and not k.endswith("0930") or k[:4]<"2026"}
n500={}
for y,m in REVIEWS:
    rd=rdate(y,m); src=f"{y}{m}{'31' if m=='03' else '30'}"
    # n500_semi keys are YYYY0331/YYYY0930
    key=f"{y}03{'31'}" if m=='03' else f"{y}09{'30'}"
    n500[rd]=set(json.load(open("/tmp/n500_semi.json")).get(key,[]))
smallcap=series(zips["smallcap250"])
# report coverage
print("index        "+" ".join(f"{y}{m}" for y,m in REVIEWS))
for nm,dd,rawk in [("n50",n50,zips["n50"]),("n100",n100,None),("next50",next50,zips["next50"]),("n500",n500,None),("smallcap250",smallcap,zips["smallcap250"])]:
    row=[]
    for y,m in REVIEWS:
        rd=rdate(y,m); n=len(dd.get(rd,set()))
        real = (rawk is not None and rd in rawk) or (nm in ("n100","n500"))
        row.append(f"{n:3}")
    print(f"{nm:12} "+" ".join(row))
json.dump({"n50":{k:sorted(v) for k,v in n50.items()},"n100":{k:sorted(v) for k,v in n100.items()},
           "n500":{k:sorted(v) for k,v in n500.items()},"smallcap250":{k:sorted(v) for k,v in smallcap.items()},
           "next50":{k:sorted(v) for k,v in next50.items()}}, open("/tmp/all_membership.json","w"))
print("saved /tmp/all_membership.json")
