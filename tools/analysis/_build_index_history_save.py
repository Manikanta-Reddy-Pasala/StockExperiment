import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import json,csv,os
from datetime import date
data=json.load(open("/tmp/all_membership.json"))
HIST="src/data/symbols/index_history"; os.makedirs(HIST,exist_ok=True)
SENT="2099-12-31"
def rd_iso(rd): return f"{rd[:4]}-{rd[4:6]}-{rd[6:]}"
# 1. by-year files (one CSV per index per review)
nfiles=0
for idx,snaps in data.items():
    for rd,syms in snaps.items():
        with open(f"{HIST}/{idx}_{rd}.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["symbol"]); [w.writerow([s]) for s in syms]
        nfiles+=1
print(f"wrote {nfiles} by-year files to {HIST}/")
# 2. rebuild interval membership CSVs (n100, n500) from clean snapshots
def build_intervals(snaps):
    dates=sorted(snaps); rows=[]
    alls=set().union(*[set(snaps[d]) for d in dates])
    for sym in sorted(alls):
        run=None
        for i,d in enumerate(dates):
            present=sym in snaps[d]
            if present and run is None: run=d
            if run is not None and not present: rows.append((sym,rd_iso(run),rd_iso(d))); run=None
            if present and i==len(dates)-1: rows.append((sym,rd_iso(run),SENT)); run=None
    return rows
import shutil
for idx in ("n100","n500"):
    rows=build_intervals(data[idx])
    p=f"src/data/symbols/{idx}_membership.csv"
    shutil.copy(p,p+".prev.bak")
    with open(p,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["symbol","start_date","end_date"]); w.writerows(rows)
    print(f"rebuilt {p}: {len(rows)} intervals, {len(set(r[0] for r in rows))} syms")
# 3. generate DB load SQL (delete + insert) for nifty_index_membership
sql=["BEGIN;","DELETE FROM nifty_index_membership WHERE index_name IN ('n50','n100','n200','n500','smallcap250','next50');"]
for idx,snaps in data.items():
    for rd,syms in snaps.items():
        for s in syms:
            ss=s.replace("'","''")
            sql.append(f"INSERT INTO nifty_index_membership(index_name,symbol,review_date,captured_at) VALUES('{idx}','{ss}','{rd_iso(rd)}',now());")
sql.append("COMMIT;")
open("/tmp/load_membership.sql","w").write("\n".join(sql))
print(f"wrote {len(sql)} SQL lines -> /tmp/load_membership.sql")
