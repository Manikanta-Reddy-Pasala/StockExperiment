# Data Pipeline — Fetching & Populating

Single reference for **every place data enters the system**: what it is, where
it comes from, the code that fetches it, where it lands (Postgres table / CSV /
JSON), and when it runs. All schedule times are **IST** (the box runs on IST;
jobs use the Python `schedule` lib in `data_scheduler.py` + per-model `cron.py`).

> **Golden rule:** prices come from **Fyers only** (never yfinance). If Fyers
> fails, refresh the token (TOTP) — do not switch source. Index membership for
> backtests must go through `eligible_at()` (PIT), never today's CSV.

---

## 0. Database connection

| Item | Detail |
|------|--------|
| Engine helper | `tools/shared/ohlcv_cache.py::_get_engine()` → `src/models/database.get_database_manager()` (memoized SQLAlchemy engine; returns `None` if DB unreachable, callers degrade) |
| URL source | `DATABASE_URL` → `POSTGRES_URL` → local default |
| Local dev DB | `stockexperiment` @ localhost (a SUBSET of tables) |
| Production DB | `trading_system` in `trading_system_db` container on the VM (77.42.45.12); full schema incl. all `audit_*` / `model_*` / `historical_options` tables |

Local and prod DBs differ — a table can exist in prod but not in the local dev
DB. The "populated by" code below is the source of truth for schema.

---

## 1. Fyers OHLCV — daily / intraday prices

| | |
|---|---|
| **What** | Daily (D), 1h, 15m OHLCV candles for NSE equities (`NSE:TICKER-EQ`) |
| **Source** | Fyers API |
| **Fetcher** | `tools/shared/prefetch_ohlcv.py` (bulk; `--universe n50/n100/n500/n100_union/n500_union/all --days N --intervals D,1h,15m`) |
| **Writer** | `tools/shared/ohlcv_cache.py::write_rows()` — upsert `ON CONFLICT` (self-heals split adjustments); `read_cached()` / `get_or_fetch()` for reads |
| **Tables** | `historical_data` (PK symbol,date), `historical_data_1h`, `historical_data_15m` |
| **Triggers** | **Sun 03:00** `backfill_full_history()` (`--universe all --days 1500`); on boot if `BACKFILL_ON_BOOT=true`. Per-model nightly incremental: `tools/models/<m>/data_pull.py::pull_daily_ohlcv()` at **20:30** (last ~5 days, for correction backfill) |

## 2. Fyers access token (TOTP refresh)

| | |
|---|---|
| **What** | Fresh `access_token` (~24h TTL); SEBI killed the v3 refresh-token tier, so this uses headless TOTP login |
| **Source** | Fyers OAuth (TOTP) |
| **Fetcher** | `tools/refresh_fyers_token.py` → `src/services/brokers/fyers_token_refresh.py` |
| **Table** | `broker_configurations` (client_id, access_token, refresh_token, connection_status) |
| **Env** | `FYERS_PIN`, `FYERS_TOTP_KEY`, `FYERS_CLIENT_ID_LOGIN` (≠ app_id), `FYERS_CLIENT_ID`, `FYERS_API_SECRET`. `appType` is parsed from the client_id suffix (`…-200` ⇒ 200). |
| **Trigger** | **Daily 03:30** `refresh_fyers_token_job()` (3 retries, Telegram alert on failure). Manual: `docker exec trading_system_app python /app/tools/refresh_fyers_token.py --force` |

## 3. Stock master + fundamentals

| | |
|---|---|
| **What** | All NSE-EQ symbols + Fyers tokens, sector, fundamentals, volatility, tradeable flags |
| **Source** | Fyers symbol master API (+ derived volatility) |
| **Fetcher** | `src/services/data/fyers_symbol_service.py::refresh_all_symbols(sync_to_database=True)`; volatility via `src/services/data/volatility_service.py` |
| **Tables** | `stocks` (master + fundamentals), `symbol_master` (raw Fyers `fytoken`, lot/tick, isin) |
| **Trigger** | **Mon 06:00** `update_symbol_master()` |

## 4. Technical indicators

| | |
|---|---|
| **What** | SMA 50/200 etc. per symbol (also EMA200/400 cached on the 1h table for that strategy) |
| **Source** | Computed from `historical_data*` |
| **Table** | `technical_indicators` |
| **Trigger** | Nightly, as part of model data pulls / export job |

---

## 5. NSE index universe lists (current)

| | |
|---|---|
| **What** | Current Nifty 100 / 500 / Midcap 150 / Smallcap 250 constituents |
| **Source** | NSE official `ind_nifty*list.csv` (browser User-Agent fetch) |
| **Fetchers** | `tools/refresh_nifty100.py`, `…500.py`, `…midcap150.py`, `…smallcap250.py` |
| **Files** | `src/data/symbols/nifty{100,500,_midcap150,_smallcap250}.csv` |
| **Triggers** | **Sat 06:00** `refresh_universe_csvs()` (all four); **daily 06:00** `daily_universe_csv_check()` (re-fetch if stale >7d). Per-model monthly: `tools/models/<m>/cron.py::_monthly_universe()` at **06:30** (1st of month) → `build_universe.py` → `logs/momrot/universes/n100_current.json` |

## 6. NSE index membership (point-in-time, survivorship-free)

| | |
|---|---|
| **What** | Half-open membership intervals `(symbol, start_date, end_date)` so backtests use the index as it WAS on each date |
| **Source** | Wayback NSE snapshots in `/tmp/n_snapshots/n{100,500}/…csv` (`tools/analysis/fetch_wayback_index_snapshots.py`, setup-only) |
| **Builder** | `tools/analysis/build_membership_table.py` (LAST-KNOWN-STATE rule; `_TICKER_ALIAS` maps renames e.g. TATAMOTORS→TMPV, ZOMATO→ETERNAL) |
| **Loader API** | `tools/shared/index_membership.py::eligible_at(index, date)` + `universe_union(index)` |
| **Files** | `src/data/symbols/n{100,500}_membership.csv` |
| **DB archive** | `nifty_index_membership` (index_name, symbol, review_date) — see §8 |

---

## 7. NSE free-float market cap (pre-inclusion model)

| | |
|---|---|
| **What** | Per-stock total + free-float market cap (₹ Cr) + LTP, for the "joining Nifty 100/500" anticipation model |
| **Source** | NSE get-quotes pages — headless **full Chromium** (NSE WAF 403s plain scripts, datacenter IPs, and `headless_shell`; only full-Chromium from a residential IP works) |
| **Downloader** | `tools/analysis/download_niftyindices.py` (current n50/100/200/500 constituent CSVs from niftyindices.com — NOT WAF-blocked, runs anywhere). REPLACED the old `nse_mcap_scraper.py` (NSE get-quotes headless Chromium, deleted 2026-05-30). Real per-stock FF-mcap from `parse_nse_index_pdfs.py` (factsheet PDFs). |
| **Working file** | `exports/nse_mcap.csv` (symbol, total_mcap_cr, ff_mcap_cr, ltp) — the file the model reads |
| **Model** | `tools/analysis/mcap_inclusion_model.py` (`--target n100\|n500`) — rank by reconstructed FF-mcap, buy names above the cutoff not yet in the index |
| **Job** | `tools/analysis/refresh_mcap.sh`: rebuild candidates → scrape → **load to Postgres** → rsync CSV to VM + `docker cp` into app |
| **Schedule** | launchd `~/Library/LaunchAgents/com.stockexp.mcaprefresh.plist` (tracked template: `tools/analysis/com.stockexp.mcaprefresh.plist`) — 02:30 on **1 Jan / Apr / Jul / Sep / Oct**. Runs LOCALLY (the VM can't reach NSE). See `tools/analysis/MCAP_JOB.md`. |

## 8. Market-cap + membership history (permanent track)

| | |
|---|---|
| **What** | Append-only archive so mcap + index membership drift is queryable over time |
| **Code** | `tools/analysis/mcap_db.py` (`init` / `load-mcap` / `snapshot-membership` / `status`) |
| **`market_cap_history`** | PK (symbol, snapshot_date): total/FF mcap ₹Cr, LTP, derived `ff_shares`, source. Filled **every** mcap run (`load-mcap`, from `exports/nse_mcap.csv`) |
| **`nifty_index_membership`** | PK (index_name, symbol, review_date): full n100/n500 list. Snapshotted on **Apr & Sep** runs (`snapshot-membership`, from `eligible_at`). Seeded 2026-04-01: n100=105, n500=519 |

## 9. Options bhavcopy (F&O)

| | |
|---|---|
| **What** | NSE F&O daily bhavcopy (OHLC, OI, volume) — OPTIDX/OPTSTK/FUTIDX/FUTSTK |
| **Source** | NSE archives (URL format switched 2024-07-07; 404 ⇒ exchange holiday) |
| **Fetcher** | `tools/shared/prefetch_bhav.py` (`parse_old`/`parse_new`; derives Fyers option symbol; stamps 15:30 IST) |
| **Tables** | `historical_options` (PK symbol,interval,timestamp), `option_universe` |
| **Trigger** | Manual backfill: `python tools/shared/prefetch_bhav.py --from … --to … --underlying NIFTY,BANKNIFTY --instrument OPTIDX` |

## 10. NSE trading holidays

| | |
|---|---|
| **What** | Holiday calendar for `is_trading_day()` |
| **Source** | NSE API (cookie-primed); hardcoded offline fallback if it fails |
| **Code** | `tools/shared/nse_calendar.py` |
| **File** | `logs/nse_holidays.json` (a.k.a. `/app/logs/nse_holidays.json`) |
| **Trigger** | **Daily 04:00** `refresh_nse_holidays_monthly()` (1st-of-month gate) + on boot; failure alerts but never blocks (fallback stays in effect) |

---

## 11. Pre-market data-quality gate

| | |
|---|---|
| **What** | Fail-closed freshness marker the executor checks before trading |
| **Source** | Computed from `historical_data` (latest date + distinct-symbol count; stale if >4 days, needs ≥400 symbols) |
| **Code** | `data_scheduler.py::pre_market_data_quality_gate()` (also re-stamped on scheduler boot, so a mid-day restart heals a stale gate) |
| **File** | `logs/data_quality_gate.json` (`ok`, `msg`, `ts`) |
| **Consumer** | `tools/live/fyers_executor.py` pre-flight — aborts (rc=3) if `ok=false`/stale/missing |
| **Trigger** | **Daily 09:00** + boot |

## 12. Model signals, rankings, picks (daily live flow)

| | |
|---|---|
| **What** | Per-model ranking, the emitted signal, the order placed, the rebalance decision |
| **Signal** | `tools/models/<m>/live_signal.py` (shares the model's `strategy.py` core with backtest — no drift) → `logs/momrot/signals/YYYY-MM-DD_*.json` |
| **Executor** | `tools/live/fyers_executor.py` reads the signal file, places orders, writes audit + ledger |
| **Tables** | `daily_suggested_stocks` (UI picks), `audit_model_rankings`, `audit_model_signals`, `audit_orders` (+ `charges_breakdown` JSONB via `tools/live/broker_charges.py`), `audit_rebalance_decisions` |
| **Schedule** (n100 example) | **09:25** emit_signal (rebalance-gated) · **09:30** execute_orders · **09:27/09:35** mid-month signal/execute. Registered via each model's `cron.py::register_trading_jobs()` |

## 13. Model ledger / capital state

| | |
|---|---|
| **What** | Per-model cash, open position(s), realized PnL, principal, enable + signals_only flags |
| **Tables** | `model_settings` (enabled, signals_only, invested_amount), `model_ledger` (single-position: open_symbol/qty/entry), `model_holdings` (multi-position e.g. retest K=3), `model_trades` (BUY/SELL/DEPOSIT/WITHDRAW audit) |
| **Writers** | `tools/live/fyers_executor.py` (single) / `fyers_executor_multi.py` (multi) on fill; `tools/live/position_reconciler.py` daily reconcile vs Fyers; `tools/live/daily_summary.py` MTM |

## 14. Model backtest artifacts (committed reference)

| | |
|---|---|
| **What** | Full-detail trade ledger + summary per model |
| **Source** | `tools/models/<m>/backtest.py` (reads `historical_data` via the shared `strategy.py` core) |
| **Full window (3yr)** | `tools/models/<m>/{summary.json, trade_ledger.json}` — ledger rows carry sym, entry/exit date, qty, entry_px, exit_px, pnl, ret_pct, cap_after, exit_reason |
| **14-month regime view** | `exports/models/<m>/{summary.json, trade_ledger.json}` + human `SUMMARY.md`/`TRADE_LEDGER.md` |
| **Doc generator** | `tools/analysis/refresh_export_docs.py` (DESC-driven; renders Entry₹/Exit₹/Qty/PnL columns + open positions). Run after a backtest writes new json. |

---

## 15. Daily CSV exports + VM sync

| | |
|---|---|
| **Exporter** | `data_scheduler.py::export_daily_csv()` → `exports/{stocks, historical_30d, technical_indicators, suggested_stocks}_YYYY-MM-DD.csv` (purges >90d) |
| **Trigger** | **Daily 22:00** (+ `validate_data_quality` 22:00, `snapshot_data_quality_audit` 22:05) |
| **VM sync** | `exports/` is excluded from the source rsync; CSVs the VM needs are pushed directly (`refresh_mcap.sh` does `rsync exports/nse_mcap.csv` → `docker cp` into `trading_system_app`). Deploy = rsync source to `/opt/trading_system/` (NO git on the VM), then docker rebuild. |

---

## Scheduler job map (data_scheduler.py, IST)

| Time | Job | Touches |
|------|-----|---------|
| Sun 03:00 | `backfill_full_history` | `historical_data*` (full 1500d) |
| 03:30 | `refresh_fyers_token_job` | `broker_configurations` |
| 04:00 | `refresh_nse_holidays_monthly` | `logs/nse_holidays.json` |
| daily 06:00 | `daily_universe_csv_check` | `src/data/symbols/*.csv` |
| Sat 06:00 | `refresh_universe_csvs` | `src/data/symbols/*.csv` |
| Mon 06:00 | `update_symbol_master` | `stocks`, `symbol_master` |
| per-model 06:30 | `_monthly_universe` (1st) | `logs/momrot/universes/*.json` |
| 09:00 | `pre_market_data_quality_gate` | `logs/data_quality_gate.json` |
| per-model 09:25–09:35 | emit/execute (+ mid-month) | signals, `audit_*`, `model_*`, `daily_suggested_stocks` |
| per-model 20:30 | `pull_daily_ohlcv` | `historical_data` (incremental) |
| 22:00 | `export_daily_csv`, `validate_data_quality` | `exports/*.csv` |
| 22:05 | `snapshot_data_quality_audit` | data-quality audit |
| 02:30 (launchd, quarterly+Sep) | `refresh_mcap.sh` | `exports/nse_mcap.csv`, `market_cap_history`, `nifty_index_membership` |

## Key env vars

`DATABASE_URL`/`POSTGRES_URL`, `FYERS_PIN`, `FYERS_TOTP_KEY`, `FYERS_CLIENT_ID_LOGIN`,
`FYERS_CLIENT_ID`, `FYERS_API_SECRET`, `BACKFILL_ON_BOOT`, `TELEGRAM_BOT_TOKEN`.

---
*Related: `README.md` (overview/setup), `STRATEGY.md` (models), `tools/analysis/MCAP_JOB.md` (mcap job detail).*
