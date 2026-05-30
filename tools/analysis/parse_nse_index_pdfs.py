"""Parse NSE index-constituent factsheet PDFs (indices_dataMar2021-2026) into
clean PIT membership + REAL historical index market cap.

Each PDF (e.g. NIFTY_50_Mar2022.pdf) is a full constituent table:
  Symbol | Security Name | Industry | Close Price | Index Mcap (Rs Cr) | Weight %

Coverage in the dump:
  NIFTY 50      : every review (Mar/Sep 2021 .. Mar2026)
  NIFTY Next 50 : recent reviews (Sep2024+) -> NIFTY 100 = N50 + Next50
  NIFTY 100/500 : only Mar2021, Sep2021, Mar2022 (explicit PDFs)

Output:
  exports/index_constituents/<INDEX>_<PERIOD>.csv  (symbol, close, mcap_cr, weight)
  + a combined exports/index_constituents/ALL_constituents.csv with review_date.

Run: python3 tools/analysis/parse_nse_index_pdfs.py [--src <dir>]
"""
from __future__ import annotations
import sys, re, csv, argparse
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "exports" / "index_constituents"
DEFAULT_SRC = Path("/Users/manip/Downloads/indices_dataMar2021-2026")

# constituent line: SYMBOL ... <close> <mcap_int> <weight.xx>   (mcap = integer)
LINE = re.compile(r"^([A-Z][A-Z0-9&\-]{0,14})\s+.*?([\d,]+\.\d{1,2})\s+([\d,]+)\s+(\d+\.\d{1,2})\s*$")
MONTHS = {"Mar": 3, "Sep": 9}


def period_to_date(period: str):
    m = re.match(r"(Mar|Sep)(\d{4})", period)
    if not m:
        return None
    mon, yr = m.group(1), int(m.group(2))
    # NSE reviews are EFFECTIVE end of Mar / end of Sep
    return date(yr, 3, 31) if mon == "Mar" else date(yr, 9, 30)


def parse_pdf(path: Path):
    import pdfplumber
    rows = []
    with pdfplumber.open(path) as pdf:
        txt = "\n".join(pg.extract_text() or "" for pg in pdf.pages)
    for line in txt.splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        sym = m.group(1)
        # skip obvious non-tickers (header fragments)
        if sym in {"NIFTY", "INDEX", "TOTAL", "RS"} or len(sym) < 2:
            continue
        close = float(m.group(2).replace(",", ""))
        mcap = float(m.group(3).replace(",", ""))
        wt = float(m.group(4))
        if mcap < 100 or wt > 100:           # sanity: mcap in ₹Cr, weight a %
            continue
        rows.append({"symbol": sym, "close": close, "mcap_cr": mcap, "weight": wt})
    # dedup by symbol (keep first)
    seen, out = set(), []
    for r in rows:
        if r["symbol"] not in seen:
            seen.add(r["symbol"]); out.append(r)
    return out


IDX_MAP = {"NIFTY_50": "n50", "NIFTY_100": "n100", "NIFTY_200": "n200",
           "NIFTY_500": "n500", "NIFTY_NEXT_50": "next50"}


def load_db(combined):
    """Mark constituents into Postgres: nifty_index_membership (PIT membership)
    + market_cap_history (REAL free-float index mcap per review). Reconstructs
    n100 = n50 + next50 for review dates lacking an explicit NIFTY_100 table."""
    sys.path.insert(0, str(ROOT))
    from sqlalchemy import text
    from tools.shared.ohlcv_cache import _get_engine
    from tools.analysis.mcap_db import init_tables
    eng = _get_engine()
    init_tables(eng)

    memb, mcap = [], {}
    by_date_idx: dict = {}
    for r in combined:
        short = IDX_MAP.get(r["index_name"])
        if not short:
            continue
        rev = r["review_date"]; sym = f"NSE:{r['symbol']}-EQ"
        by_date_idx.setdefault((rev, short), set()).add(r["symbol"])
        if short in ("n50", "n100", "n200", "n500"):
            memb.append({"index_name": short, "symbol": r["symbol"], "review_date": rev})
        # real FF index mcap — dedupe per (symbol, review): value is index-agnostic
        mcap[(r["symbol"], rev)] = r["mcap_cr"]

    # reconstruct n100 = n50 ∪ next50 where explicit n100 missing for that review
    revs = {d for (d, i) in by_date_idx}
    for rev in revs:
        if (rev, "n100") in by_date_idx:
            continue
        n50 = by_date_idx.get((rev, "n50"), set())
        nx = by_date_idx.get((rev, "next50"), set())
        if n50 and nx:
            for s in (n50 | nx):
                memb.append({"index_name": "n100", "symbol": s, "review_date": rev})

    with eng.begin() as c:
        c.execute(text("""INSERT INTO nifty_index_membership (index_name,symbol,review_date)
                          VALUES (:index_name,:symbol,:review_date)
                          ON CONFLICT (index_name,symbol,review_date) DO NOTHING"""), memb)
        c.execute(text("""INSERT INTO market_cap_history
                          (symbol,snapshot_date,ff_mcap_cr,source)
                          VALUES (:symbol,:snapshot_date,:ff_mcap_cr,'nse_index_factsheet')
                          ON CONFLICT (symbol,snapshot_date) DO UPDATE SET
                            ff_mcap_cr=EXCLUDED.ff_mcap_cr, source=EXCLUDED.source"""),
                  [{"symbol": f"NSE:{s}-EQ", "snapshot_date": d, "ff_mcap_cr": v}
                   for (s, d), v in mcap.items()])
    print(f"DB: {len(memb)} membership rows, {len(mcap)} mcap snapshots "
          f"(n100 reconstructed where needed)")


def load_from_committed_csv():
    """Rebuild the `combined` list from the committed exports/index_constituents/
    ALL_constituents.csv (so we can load the DB without the source PDFs)."""
    p = OUT / "ALL_constituents.csv"
    if not p.exists():
        print(f"ERROR: {p} not found"); return []
    out = []
    for r in csv.DictReader(open(p)):
        try:
            out.append({"index_name": r["index_name"], "review_date": r["review_date"],
                        "period": r.get("period", ""), "symbol": r["symbol"],
                        "close": float(r.get("close") or 0), "mcap_cr": float(r["mcap_cr"]),
                        "weight": float(r.get("weight") or 0)})
        except (ValueError, KeyError):
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--load-db", action="store_true")
    ap.add_argument("--from-csv", action="store_true",
                    help="load DB from committed ALL_constituents.csv (no PDFs needed)")
    a = ap.parse_args()
    if a.from_csv:
        combined = load_from_committed_csv()
        print(f"loaded {len(combined)} rows from committed CSV")
        if combined:
            load_db(combined)
        return
    src = Path(a.src)
    OUT.mkdir(parents=True, exist_ok=True)
    combined = []
    want = re.compile(r"NIFTY_(50|100|200|500|Next_50)_(Mar|Sep)\d{4}\.pdf$", re.I)
    pdfs = sorted(src.glob("indices_data*/*.pdf"))
    summary = {}
    for p in pdfs:
        mm = want.search(p.name)
        if not mm:
            continue
        idx = "NIFTY_" + mm.group(1).upper()
        period = re.search(r"(Mar|Sep\d{0,4})\d{4}", p.name)
        period = re.search(r"(Mar|Sep)(\d{4})", p.name).group(0)
        rev = period_to_date(period)
        try:
            rows = parse_pdf(p)
        except Exception as e:
            print(f"  ERR {p.name}: {type(e).__name__}: {e}"); continue
        if not rows:
            print(f"  WARN {p.name}: 0 rows parsed"); continue
        (OUT / f"{idx}_{period}.csv").write_text(
            "symbol,close,mcap_cr,weight\n" +
            "\n".join(f"{r['symbol']},{r['close']},{r['mcap_cr']},{r['weight']}" for r in rows))
        for r in rows:
            combined.append({"index_name": idx, "review_date": rev.isoformat(),
                             "period": period, **r})
        summary[f"{idx} {period}"] = len(rows)
    # combined file
    if combined:
        with open(OUT / "ALL_constituents.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["index_name", "review_date", "period",
                                              "symbol", "close", "mcap_cr", "weight"])
            w.writeheader(); w.writerows(combined)
    print(f"parsed {len(pdfs)} candidate PDFs -> {len(summary)} index/period tables, "
          f"{len(combined)} constituent rows")
    for k in sorted(summary):
        print(f"  {k}: {summary[k]}")
    if a.load_db and combined:
        load_db(combined)


if __name__ == "__main__":
    main()
