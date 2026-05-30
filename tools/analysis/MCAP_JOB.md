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
| `nse_mcap_scraper.py` | Scrapes Total + Free-Float mcap + LTP per stock via headless **full Chromium** (NSE blocks `headless_shell` + datacenter IPs). Resumable → `exports/nse_mcap.csv`. |
| `mcap_inclusion_model.py` | PIT backtest. `ff_shares = ff_mcap/ltp`; `ff_mcap[t] = ff_shares × close[t]`; monthly rank; CANDIDATE = mcap rank ≤ cutoff AND not yet in `eligible_at(target)` AND 30d-ret>0. `--target n100\|n500 --k 5`. |
| `refresh_mcap.sh` | Rebuild candidate list (**full NSE equity universe**, no ADV cap) → scrape → **persist to Postgres** → rsync CSV to VM + `docker cp` into app. |
| `mcap_db.py` | Postgres persistence: `market_cap_history` (every run) + `nifty_index_membership` (Apr & Sep reviews). CLI: `init` / `load-mcap` / `snapshot-membership` / `status`. |
| `com.stockexp.mcaprefresh.plist` | launchd template — 02:30 on 1 Jan/Apr/Jul/**Sep**/Oct (Sep added for the NSE Sep review). |

## Postgres tables (historical track)

| Table | Grain | Filled |
|-------|-------|--------|
| `market_cap_history` | (symbol, snapshot_date) — total/FF mcap ₹Cr, LTP, derived FF shares | every run (`load-mcap`) |
| `nifty_index_membership` | (index_name, symbol, review_date) — full n100/n500 constituent list | Apr & Sep runs (`snapshot-membership`) |

The CSV (`exports/nse_mcap.csv`) stays the working file the model reads; the DB
is the permanent append-only archive so we can see how mcap + membership drift
over time. Seeded 2026-04-01: n100=105, n500=519.

## Universe

Full NSE equity set (`NSE:%-EQ` with data in the last 120 days, ~2,300 symbols).
No top-N-ADV cap — a stock climbing toward N500 can sit below top liquidity.

## Data note

`exports/nse_mcap.csv` is **gitignored** (data, not code). Free-float *shares*
are reconstructed from current `ff_mcap/ltp` and applied to historical close —
the only lookahead approximation (shares drift slowly vs price). Index
membership (`eligible_at`) is fully point-in-time historical.

## Install the cron (one-time, on the residential Mac)

```bash
cp tools/analysis/com.stockexp.mcaprefresh.plist ~/Library/LaunchAgents/
launchctl unload ~/Library/LaunchAgents/com.stockexp.mcaprefresh.plist 2>/dev/null
launchctl load   ~/Library/LaunchAgents/com.stockexp.mcaprefresh.plist
launchctl list | grep mcaprefresh   # verify
```

## Manual run

```bash
python3 tools/analysis/nse_mcap_scraper.py            # scrape (background)
python3 tools/analysis/mcap_inclusion_model.py --target n100
python3 tools/analysis/mcap_inclusion_model.py --target n500
bash    tools/analysis/refresh_mcap.sh                 # full quarterly cycle
```
