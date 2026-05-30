"""Harvest historical NSE index constituents from the Wayback Machine.

The live niftyindices.com only serves TODAY's constituent list. For point-in-time
backtest membership we use the official factsheet zips (indices_dataMar2021-2026),
but those have gaps (e.g. no Nifty Next-50 in the Sep2022–Mar2024 partial zips).
This scrapes web.archive.org captures of the stable niftyindices CSV URLs to fill
gaps where a real historical snapshot exists.

Reality check: the open web has only SPARSE, arbitrary-dated captures (1–7 per index
over 2021–2026) — NOT clean semi-annual coverage. So this is supplementary, not a
replacement for the factsheets + verified N500 workbook.

Writes usable (HTTP-200, well-formed) captures to
src/data/symbols/wayback_snapshots/{index}_{YYYYMMDD}.csv, consumed by
build_membership_from_nse_factsheets.py --wayback-dir.

Usage: python tools/analysis/harvest_wayback_snapshots.py
"""
from __future__ import annotations

import json
import ssl
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "src" / "data" / "symbols" / "wayback_snapshots"
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE
_UA = {"User-Agent": "Mozilla/5.0"}
INDEX_FILES = {
    "n100": "ind_nifty100list",
    "next50": "ind_niftynext50list",
    "n500": "ind_nifty500list",
    "smallcap250": "ind_niftysmallcap250list",
}


def _get(url: str) -> str:
    try:
        return urllib.request.urlopen(
            urllib.request.Request(url, headers=_UA), timeout=45, context=_CTX
        ).read().decode("utf-8", "ignore")
    except Exception as e:
        return f"ERR {e}"


def harvest() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for key, f in INDEX_FILES.items():
        cdx = _get(f"http://web.archive.org/cdx/search/cdx?url=niftyindices.com/"
                   f"IndexConstituent/{f}.csv&output=json&from=20210101&to=20260601"
                   f"&fl=timestamp&filter=statuscode:200&collapse=digest")
        try:
            stamps = [r[0] for r in json.loads(cdx)[1:]]
        except Exception:
            stamps = []
        print(f"{key}: {len(stamps)} distinct 200-captures")
        for ts in stamps:
            body = _get(f"https://web.archive.org/web/{ts}id_/"
                        f"https://niftyindices.com/IndexConstituent/{f}.csv")
            if body.startswith("Company Name") and body.count("\n") > 10:
                (OUT / f"{key}_{ts[:8]}.csv").write_text(body)
                print(f"  saved {key}_{ts[:8]}.csv ({body.count(chr(10))} rows)")


if __name__ == "__main__":
    harvest()
