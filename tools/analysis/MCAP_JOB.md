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

## Why it runs on the laptop, NOT the VM (load-bearing)

NSE's WAF blocks **datacenter IPs**. The Hetzner VM (77.42.45.12) cannot pull
the quote pages — verified 2026-05-30:

| From | Result |
|------|--------|
| VM (`2a01:4f9:…` Hetzner IPv6) → NSE get-quotes | **HTTP 302** (WAF bounce, no data) — even with a browser User-Agent |
| Laptop (residential IP) + **full Chromium** headless | **200**, real data ✅ |

Also dead: plain `requests`/`curl` from anywhere (403, TLS fingerprint), and
`headless_shell` Chromium (HTTP2 error / timeout). The scraper must use the
**full Chromium binary** via `executable_path` (see `nse_mcap_scraper.py`).

So the flow is: **scrape on the laptop → load local Postgres → rsync CSV →
`docker cp` into the VM app container.** The VM never scrapes; it only receives
the finished CSV.

⚠️ **Laptop-wake caveat:** the launchd job fires 02:30 on the 1st only if the
Mac is awake. launchd runs a *missed* job at next wake, so a short sleep is
fine, but if the Mac is off for days the monthly snapshot lands late. For
bulletproof timing: keep the Mac awake overnight on the 1st, or route the
scrape through a residential proxy (more setup). Monthly cadence tolerates slack.

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
