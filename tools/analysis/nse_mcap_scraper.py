"""Scrape NSE per-stock Total + Free-Float market cap via headless browser.

NSE's /api/ is WAF-blocked (403) for scripts AND datacenter IPs; only a real
browser from a residential IP gets through. So we drive headless Chromium, load
each get-quotes page, and regex Total + Free-Float mcap from the rendered DOM.

Resumable: appends to OUT_CSV; on restart, skips symbols already saved.
Run (background): python3 tools/analysis/nse_mcap_scraper.py
Input: /tmp/mcap_candidates.json (list of plain symbols). Output: OUT_CSV.
"""
from __future__ import annotations
import json, csv, re, time, sys, os
from pathlib import Path
from playwright.sync_api import sync_playwright

CAND = "/tmp/mcap_candidates.json"
OUT = Path(__file__).resolve().parents[2] / "exports" / "nse_mcap.csv"
FF_RE = re.compile(r"Free Float Market Cap \(₹ Cr\.\)[\s\n]*([\d,]+\.?\d*)", re.I)
TOT_RE = re.compile(r"Total Market Cap[^0-9]{0,40}([\d,]+\.?\d*)", re.I)
LTP_RE = re.compile(r"Last Traded Price[\s\n₹()]*([\d,]+\.?\d*)", re.I)


def num(s):
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def main():
    syms = json.load(open(CAND))
    # drop index/derivative pseudo-symbols; keep equities
    syms = [s for s in syms if not s.endswith("-INDEX") and "NIFTY" not in s]
    done = set()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        for r in csv.DictReader(open(OUT)):
            done.add(r["symbol"])
    todo = [s for s in syms if s not in done]
    print(f"total={len(syms)} done={len(done)} todo={len(todo)}", flush=True)
    new_file = not OUT.exists()
    import glob
    _exe = sorted(glob.glob(os.path.expanduser(
        "~/Library/Caches/ms-playwright/chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium")))
    exe = _exe[-1] if _exe else None  # FULL chromium, NOT headless_shell (NSE blocks the shell)
    with sync_playwright() as p:
        br = p.chromium.launch(headless=True, executable_path=exe,
                               args=["--disable-blink-features=AutomationControlled"])
        ctx = br.new_context(user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                             "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
        pg = ctx.new_page()
        f = open(OUT, "a", newline="")
        w = csv.writer(f)
        if new_file:
            w.writerow(["symbol", "total_mcap_cr", "ff_mcap_cr", "ltp"]); f.flush()
        ok = 0
        for i, sym in enumerate(todo, 1):
            try:
                pg.goto(f"https://www.nseindia.com/get-quotes/equity?symbol={sym}",
                        wait_until="domcontentloaded", timeout=25000)
                pg.wait_for_timeout(1800)
                txt = pg.inner_text("body")
                ff = FF_RE.search(txt); tot = TOT_RE.search(txt); ltp = LTP_RE.search(txt)
                ffv = num(ff.group(1)) if ff else None
                totv = num(tot.group(1)) if tot else None
                ltpv = num(ltp.group(1)) if ltp else None
                w.writerow([sym, totv, ffv, ltpv]); f.flush()
                if ffv:
                    ok += 1
                if i % 25 == 0:
                    print(f"  {i}/{len(todo)} ok={ok} last={sym} ff={ffv}", flush=True)
            except Exception as e:
                w.writerow([sym, None, None, None]); f.flush()
                print(f"  ERR {sym}: {type(e).__name__}", flush=True)
            time.sleep(0.8)
        f.close(); br.close()
        print(f"DONE: {ok} with FF-mcap of {len(todo)} attempted -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
