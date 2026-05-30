"""Build authoritative PIT index-membership CSVs from official NSE factsheets.

Replaces the old Wayback-snapshot membership (which was ~18% wrong on Nifty 100 —
missing real members like ADANIGREEN/BANKBARODA/CHOLAFIN and carrying non-members
like ABB/BHEL/IDEA and even a "DUMMYREL" placeholder) with the official NSE
semi-annual factsheet constituents.

Sources (supplied by the user, 2026-05-31):
  - indices_dataMar2021-2026/  : per-rebalance ZIPs (Mar + Sep, 2021-2026) of
    official NSE "Constituents of NIFTY <index>" factsheet PDFs. Only the three
    big zips (Mar2021/Sep2021/Mar2022) carry NIFTY_100 + NIFTY_500 PDFs; all 11
    carry NIFTY_50; several carry NIFTY_Next_50.
  - Nifty500_Rebalancing_v2.xlsx : a VERIFIED derived Nifty-500 membership matrix
    (symbol x year 2021-2026), cross-checked against the official N500 PDFs.

Method:
  N100 = NIFTY 50 ∪ NIFTY Next 50 (exact NSE construction). Where the direct
         NIFTY_100 PDF exists it matches the union exactly (validated). Periods
         with only N50 (no Next50: Sep2022..Mar2024) are skipped — the half-open
         interval from the prior snapshot carries that membership forward, which
         is the documented last-known-state fallback.
  N500 = the xlsx year-wise matrix, one snapshot per year (anchored Mar 31).

Snapshots -> half-open intervals [start, end) consumed by
tools/shared/index_membership.py. Symbols are stored period-correct (old tickers);
index_membership applies _TICKER_ALIAS on read to resolve to price-DB symbols.

Usage:
  python tools/analysis/build_membership_from_nse_factsheets.py \
    --indices-dir ~/Downloads/indices_dataMar2021-2026 \
    --xlsx ~/Downloads/Nifty500_Rebalancing_v2.xlsx
  (writes src/data/symbols/n{100,500}_membership.csv, backing up the old ones to
   *.wayback.bak; --dry-run to only print the validation table.)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import shutil
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[2]
SYM_DIR = ROOT / "src" / "data" / "symbols"
SENT = "2099-12-31"
_SYM_RE = re.compile(r"[A-Z0-9][A-Z0-9&.\-]{0,14}$")


def _period_to_date(period: str) -> date:
    """'Mar2021' -> 2021-03-31 ; 'Sep2021' -> 2021-09-30 (rebalance effective)."""
    mon, yr = period[:3], int(period[3:])
    return date(yr, 3, 31) if mon == "Mar" else date(yr, 9, 30)


def _pdf_symbols(pdf_path: str) -> set[str]:
    """First-column (Symbol) of every constituent row across all table pages."""
    out: set[str] = set()
    with pdfplumber.open(pdf_path) as pdf:
        for pg in pdf.pages:
            for tb in (pg.extract_tables() or []):
                for row in tb:
                    if not row:
                        continue
                    c0 = (row[0] or "").strip()
                    if c0 and c0.lower() != "symbol" and _SYM_RE.fullmatch(c0):
                        out.add(c0)
    return out


def extract_snapshots(indices_dir: Path) -> tuple[dict, dict]:
    """Unzip every rebalance ZIP and pull N50 / Next50 / N100 / N500 constituents.

    Returns (n100_snaps, n500_pdf_snaps): {date: set(symbols)}.

    N100 per period (NSE construction = N50 ∪ Next-50):
      - direct NIFTY_100 PDF if present; else NIFTY_50 ∪ NIFTY_Next_50.
      - GAP periods that ship only NIFTY_50 (Sep2022..Mar2024 — the partial zips
        with no Next-50 PDF): N100 = that period's ACTUAL N50 ∪ the most recent
        prior Next-50 (carried forward). Keeps the N50 half period-correct and
        only freezes the Next-50 half across the gap (option 1, 2026-05-31).
    n500_pdf_snaps = only the periods shipping a NIFTY_500 PDF (xlsx cross-check).
    """
    raw: dict[date, dict] = {}
    n500_pdf: dict[date, set[str]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for z in glob.glob(str(indices_dir / "*.zip")):
            period = os.path.basename(z)[len("indices_data"):-len(".zip")]
            dst = os.path.join(tmp, period)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dst)

            def find(pat):
                fs = glob.glob(os.path.join(dst, pat + "*.pdf"))
                return _pdf_symbols(fs[0]) if fs else None

            d = _period_to_date(period)
            raw[d] = {"n50": find("NIFTY_50_"), "next50": find("NIFTY_Next_50_"),
                      "n100_pdf": find("NIFTY_100_")}
            n5 = find("NIFTY_500_")
            if n5:
                n500_pdf[d] = n5

    # Assemble N100 CHRONOLOGICALLY so a gap period can borrow the last Next-50.
    n100: dict[date, set[str]] = {}
    last_next50: set[str] | None = None
    for d in sorted(raw):
        r = raw[d]
        if r["next50"]:
            last_next50 = r["next50"]
        if r["n100_pdf"]:
            n100[d] = r["n100_pdf"]
        elif r["n50"] and r["next50"]:
            n100[d] = r["n50"] | r["next50"]
        elif r["n50"] and last_next50:           # gap: actual N50 + carried Next-50
            n100[d] = r["n50"] | last_next50
    return n100, n500_pdf


def n500_from_xlsx(xlsx: Path) -> dict:
    """Year-wise N500 membership matrix from the verified workbook -> {date: set}.
    One snapshot per year, anchored at Mar 31 (NSE annual reconstitution)."""
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb["Year-wise Constituents"]
    rows = list(ws.iter_rows(values_only=True))
    hi = next(i for i, r in enumerate(rows) if r and "Symbol" in [str(c) for c in r])
    hdr = rows[hi]
    sym_j = hdr.index("Symbol")
    yr_cols = {str(c).strip(): j for j, c in enumerate(hdr)
               if c and str(c).strip() in ("2021", "2022", "2023", "2024", "2025", "2026")}
    snaps: dict[date, set[str]] = {}
    for y, j in yr_cols.items():
        members = set()
        for r in rows[hi + 1:]:
            if not r or not r[sym_j]:
                continue
            v = str(r[j]).strip() if r[j] is not None else ""
            if v and v not in ("—", "-"):
                members.add(str(r[sym_j]).strip())
        snaps[date(int(y), 3, 31)] = members
    return snaps


def build_intervals(snaps: dict) -> list[tuple[str, str, str]]:
    """{date: set} -> [(symbol, start_iso, end_iso)] half-open. A symbol holds
    [d_i, d_{i+1}) per snapshot it appears in; consecutive snapshots merge; present
    in the last snapshot -> end = SENT (still a member)."""
    dates = sorted(snaps)
    rows: list[tuple[str, str, str]] = []
    for sym in sorted(set().union(*snaps.values())):
        run_start = None
        for i, d in enumerate(dates):
            present = sym in snaps[d]
            if present and run_start is None:
                run_start = d
            if run_start is not None and not present:
                rows.append((sym, run_start.isoformat(), d.isoformat()))
                run_start = None
            if present and i == len(dates) - 1:
                rows.append((sym, run_start.isoformat(), SENT))
                run_start = None
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices-dir", required=True, help="indices_dataMar2021-2026 folder")
    ap.add_argument("--xlsx", required=True, help="Nifty500_Rebalancing_v2.xlsx")
    ap.add_argument("--dry-run", action="store_true", help="print validation only, don't write CSVs")
    a = ap.parse_args()

    n100_snaps, n500_pdf = extract_snapshots(Path(a.indices_dir).expanduser())
    n500_snaps = n500_from_xlsx(Path(a.xlsx).expanduser())

    print("N100 snapshots:", sorted(d.isoformat() for d in n100_snaps),
          "| sizes:", [len(n100_snaps[d]) for d in sorted(n100_snaps)])
    print("N500 snapshots:", sorted(d.isoformat() for d in n500_snaps),
          "| sizes:", [len(n500_snaps[d]) for d in sorted(n500_snaps)])
    # cross-check xlsx N500 vs the 3 official N500 PDFs
    for d, pdfset in sorted(n500_pdf.items()):
        xls = n500_snaps.get(date(d.year, 3, 31), set())
        if xls:
            print(f"  N500 xlsx vs PDF {d}: |xlsx∩pdf|={len(xls & pdfset)} "
                  f"xls_only={len(xls - pdfset)} pdf_only={len(pdfset - xls)}")

    if a.dry_run:
        return 0
    for idx, snaps in [("n100", n100_snaps), ("n500", n500_snaps)]:
        rows = build_intervals(snaps)
        dst = SYM_DIR / f"{idx}_membership.csv"
        if dst.exists():
            shutil.copy(dst, dst.with_suffix(".csv.wayback.bak"))
        with open(dst, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "start_date", "end_date"])
            w.writerows(rows)
        print(f"wrote {dst}: {len(rows)} intervals, "
              f"{len(set(r[0] for r in rows))} unique symbols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
