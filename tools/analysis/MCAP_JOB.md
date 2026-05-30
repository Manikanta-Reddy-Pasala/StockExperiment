# NSE Market-Cap Pre-Inclusion Job

Anticipates Nifty index inclusions: buy stocks whose **free-float market cap**
already ranks top-100 / top-500 but which NSE hasn't officially added yet (NSE
reviews semi-annually → there's a lag). When NSE promotes them, inclusion flows
+ momentum lift; we enter ahead.

All code lives in this repo (`tools/analysis/`). The launchd job runs on a
residential-IP Mac because NSE's WAF 403s the VM and plain scripts.

## Files

| File | Role |
|------|------|
| `download_niftyindices.py` | **REPLACES the old get-quotes scraper.** Downloads current index constituents (n50/100/200/500) from niftyindices.com (NOT WAF-blocked → runs anywhere, incl the VM) → `exports/index_constituents/current/` + marks `nifty_index_membership`. |
| `parse_nse_index_pdfs.py` | Parses NSE factsheet PDFs (the `indices_dataMar2021-2026` dump) → full constituent tables w/ **real free-float mcap** → CSVs + `nifty_index_membership` + `market_cap_history`. |
| `mcap_inclusion_model.py` | PIT backtest. `ff_shares = ff_mcap/ltp`; `ff_mcap[t] = ff_shares × close[t]`; monthly rank; CANDIDATE = mcap rank ≤ cutoff AND not yet in `eligible_at(target)` AND 30d-ret>0. `--target n100\|n500 --k 5`. |
| `refresh_mcap.sh` | Rebuild candidate list (**full NSE equity universe**, no ADV cap) → scrape → **persist to Postgres** → rsync CSV to VM + `docker cp` into app. |
| `mcap_db.py` | Postgres persistence: `market_cap_history` (every run) + `nifty_index_membership` (Apr & Sep reviews). CLI: `init` / `load-mcap` / `snapshot-membership` / `status`. |
| `com.stockexp.mcaprefresh.plist` | launchd template — 02:30 on the **1st of every month** (monthly mcap track). |

## Postgres tables (historical track)

| Table | Grain | Filled |
|-------|-------|--------|
| `market_cap_history` | (symbol, snapshot_date) — total/FF mcap ₹Cr, LTP, derived FF shares | every run = **monthly** (`load-mcap`) |
| `nifty_index_membership` | (index_name, symbol, review_date) — full n100/n500 constituent list | Apr & Sep runs (`snapshot-membership`) |

The CSV (`exports/nse_mcap.csv`) stays the working file the model reads; the DB
is the permanent append-only archive so we can see how mcap + membership drift
over time. Seeded 2026-04-01: n100=105, n500=519.

## Universe

Full NSE equity set (`NSE:%-EQ` with data in the last 120 days, ~2,300 symbols).
No top-N-ADV cap — a stock climbing toward N500 can sit below top liquidity.

## Data note

`exports/nse_mcap.csv` is **gitignored** (data, not code). Free-float *shares*
are reconstructed as `ff_mcap / latest DB close` (the scrape's LTP column is
unreliable) and applied to historical close — the only lookahead approximation
(shares drift slowly vs price). Because shares are frozen at one snapshot, the
reconstructed "mcap climb" is really **cross-sectional price relative-strength
vs the whole market**, not fundamental mcap growth — true mcap-drift needs the
accumulating monthly `market_cap_history`. Index membership (`eligible_at`) is
fully point-in-time historical.

## Source: niftyindices.com (NOT the old NSE get-quotes scraper)

The old approach scraped NSE `get-quotes` pages with headless full Chromium —
NSE's WAF blocks datacenter IPs (VM got HTTP 302) so it was laptop-only + slow.
**Retired 2026-05-30.** niftyindices.com is NOT WAF-blocked (plain HTTPS works
from any IP, verified incl the VM), so `download_niftyindices.py` pulls the
constituent CSVs directly — can run on the laptop OR the VM, no Chromium.

Mcap nuance: the niftyindices constituent CSVs give MEMBERSHIP only (no per-stock
mcap). Real free-float mcap comes from the NSE index factsheets
(`parse_nse_index_pdfs.py`, loaded to `market_cap_history`); the climber overlay
otherwise uses a price-derived proxy off `exports/nse_mcap.csv` + DB closes.

## Install the cron (one-time)

```bash
cp tools/analysis/com.stockexp.mcaprefresh.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.stockexp.mcaprefresh.plist 2>/dev/null
launchctl load   ~/Library/LaunchAgents/com.stockexp.mcaprefresh.plist
launchctl list | grep mcaprefresh   # verify
```

## Manual run

```bash
python3 tools/analysis/download_niftyindices.py --load-db   # current constituents -> membership
python3 tools/analysis/parse_nse_index_pdfs.py --load-db    # factsheet PDFs -> real mcap (when you drop a new dump)
bash    tools/analysis/refresh_mcap.sh                       # full monthly cycle (download + mark + push)
```
