# StockExperiment — Multi-Model Live Trading + Audit

NSE momentum & breakout trading system running **4 active equity models** in parallel against a single Fyers brokerage account. Each model has its own capital pool, own ledger, own ranking signal, own rebalance cadence. Live orders fire daily via cron 09:30 IST; every decision is captured in a 7-table audit trail.

**Charges are approximate** (formula-based — Fyers SEBI rates), deducted from per-model cash at fill time. Not chased to broker-exact.

Production: `77.42.45.12` · App: <https://stock.oneshell.in> · Bot: `@stocks_momrot_bot`

---

## Recent Changes (2026-05)

- **n40 daily → WEEKLY rebalance** (the fix). Daily rotation churned (55% of trades held ≤3 days = whipsaw). Rebalancing weekly (1st trading day of each ISO week, shared `rebalance_calendar.build_weekly_calendar` / `is_week_rebalance_day`) lifts CAGR + cuts DD on BOTH windows: 2023-05→2026 **+59.3% / 24.4% DD** (was +55.4% / 25.4%), full-cycle 2021→2026 **+20.6% / 55.5%** (was +13.7% / 59%). retain-1 kept (wider bands hurt); stop-loss / min-ADV tested + rejected (n40 already top-ADV). backtest + live share the weekly rule.
- **All models unified backtest↔live** — each model's `backtest.py` + `live_signal.py` import ONE shared core (per-model `strategy.py` + `tools/shared/rebalance_calendar.py`); params can't drift. Canonical numbers regenerated on VM postgres.
- **n100 backtest was under-reporting** — the CLI defaulted `--mid-month-check` OFF, so the committed summary showed +43% CAGR. Live actually runs the mid-month job (cron 09:27), and with mid-month ON (the real live config) n100 = **+87.5% CAGR / 34% DD** (2023-05→2026-05). CLI now defaults mid-month ON to match live; summary regenerated.
- **mcap-climber** shipped to emerging (real free-float-mcap filter, +98%→+111% CAGR same DD). Backtest + live share it via `strategy.py`.

- **Model-aware position reconciler** (`tools/live/position_reconciler.py`). The shared Fyers account merges per-symbol holdings across models, so when two equity models (e.g. n100 + n20 in ADANIPOWER) both hold the same name, the broker reports ONE merged qty with no per-model tag. The previous reconciler AUTO_MIRRORed that merged qty into BOTH `model_ledger` rows on every pass → each model thought it owned 2× the position → cash math drifted. New behaviour: ration the broker net by subtracting sibling ledger claims before compare, and refuse to overwrite `entry_px` under overlap (broker avg is a blend, not any one model's truth). See [Cross-Model Overlap](#cross-model-overlap) below. 14-test pure-unit coverage in `tests/test_position_reconciler.py`.
- **Combined-portfolio sim** (`tools/backtests/combined_portfolio_sim.py`) runs the 3 large-cap models as one portfolio under `{allow, block, rank2}` overlap policies. Verdict (2023-05→2026-05, ₹10L per bucket): `allow` 136.56% CAGR / 26.18% DD / 5.22 Calmar BEATS `block` (77.20% / 17.02% / 4.54) and `rank2` (79.76% / 22.85% / 3.49). Models converge on consensus winners — concentrating into agreement is the alpha, the reconciler change above is the corresponding safety fix. Pair tool: `tools/backtests/analyze_model_overlap.py` — quantifies historical (symbol, date-range) collisions across model trade ledgers.
- **Market-status banner on `/dashboard`** — visible on weekends and NSE holidays (e.g. Bakri Id 2026-05-28). Shows the next trading day; auto-hides on trading days. Source = existing `/api/nse-holidays` (no new backend), fail-safe (weekend check works even if the API fetch fails).
- **Dead code removed (campaign, commits `eba33542`→`f7b8722b`):** the daily-loss kill-switch (`RiskConfig.max_daily_loss_pct` / `MAX_DAILY_LOSS_PCT` / `can_enter day_pnl`) was vestigial — the executor never passed `day_pnl`, so the gate could never fire. Removed. Realistic loss control now = rank exit + midcap stop + margin gate + per-model cap. Also dropped: `Position.sl/target` (never written by models, never read), `_save_ledger` / `_append_history` no-ops, dup imports, dead universes helpers.
- **Startup pipeline → opt-in.** The legacy `pipeline_saga` used to run every app boot (opt-out via `SKIP_STARTUP_PIPELINE`). Flipped to default-skip — runs only when `RUN_STARTUP_PIPELINE=true`. The 21:00 cron that called the legacy pipeline was also killed: live models pull their own daily OHLCV at 20:30-45 (`prefetch_ohlcv` → `historical_data`), the saga only fed admin dashboards.
- **`n100` lookback 30 → 15 trading days** (6-year-validated sweep). Pseudo stays at 30. See model table below.
- **6-year data backfill.** `historical_data` for the `n100` universe now spans 2019-04 → present (was 2022-04); `ON CONFLICT DO UPDATE` upsert self-heals split discontinuities.
- **NSE holiday auto-source** (`tools/shared/nse_calendar.py`): primary = NSE `holiday-master?type=trading` (cookie-primed session, cached `/app/logs/nse_holidays.json`, refreshed on boot + daily 04:00), falls back to a hardcoded list. Same data feeds executor's holiday guard, picks UI's "trading days left", and the new dashboard banner.

---

## Models

| Model | Universe | Cadence | Product | Hold | Signal |
|-------|----------|---------|---------|------|--------|
| `momentum_n100_top5_max1` | Real Nifty 100 | Monthly (1st weekday) + mid-month | CNC delivery | until it drops below rank-1 | rank by **15-trading-day** return, hold rank-1 (top-1 rotation) |
| `momentum_pseudo_n100_adv` | Top-100 ADV from N500 minus Smallcap-250, yearly PIT rebuild, close > 200d SMA | **Monthly + mid-month** (day-15 lead check) | CNC | while in top-5 | rank by 30d return, hold while in top-5 + uptrend + ≤₹3K + 3pp mid-month lead gate |
| `midcap_narrow_60d_breakout` | ~100 NSE midcaps (top-100 ADV minus Nifty 100) | Event-driven (daily check) | CNC | up to 120d / target +100% / trail -20% from peak | 40d-high + vol >2× + 200d SMA, ALL must fire |
| `n20_daily_large_only` | **Top-40** ADV ∩ Nifty 100 (n40; dir keeps legacy n20 name) | **Weekly** (1st trading day of ISO week) | CNC | until it drops below rank-1 | rank by 30d return + 200d SMA uptrend filter (PIT) |

**Capital model (per model):**
- `Allocated / Invested` = user-deposited principal (`ModelSettings.invested_amount`). Default ₹30,000 per active model.
- `Available Cash` = idle un-invested cash + cumulative realized P&L (`ModelLedger.cash`). Approx broker charges already deducted at fill time.
- `Position Value (Live)` = held_qty × live Fyers LTP.
- `Realized P&L` = sum of closed-trade P&L (`ModelLedger.realized_pnl`). Approximate (formula-based charges).
- `Unrealized P&L` = position_value − cost_basis.
- `Available (Net Worth) / NAV` = cash + position_value.
- `Total P&L` = NAV − Invested.

Each model gets a single `model_settings` row + single `model_ledger` row + N `model_trades` rows. No cross-talk; one model's BUY never touches another's cash.

---

## Logic — End-to-end Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  Container TZ = Asia/Kolkata (IST). All cron times below are IST.    │
└──────────────────────────────────────────────────────────────────────┘

06:30-35  Universe refresh (monthly/yearly/quarterly per model)
            └─ tools/models/<m>/data_pull.py refresh_universe()

09:25     emit_signal — every enabled model
            └─ tools/models/<m>/live_signal.py
                 - Loads OHLCV from Postgres (historical_data)
                 - Computes rank by model's scoring rule
                 - If held in top-N → emit HOLD; else SELL old + BUY rank-1
                 - Writes signals JSON + ranking JSON to /app/logs/<m>/
                 - Audit hook: audit_model_rankings + audit_model_signals

09:30-32  execute_orders (always live)
            └─ tools/live/fyers_executor.py
                 1. RiskManager.from_model(name)
                    → capital = model_ledger.cash (live truth)
                    → max_total_buy_inr = allocated + realized (hard ceiling)
                 2. PASS 1 — exits first, block on each fill
                 3. RiskManager rebuilt (SELL proceeds in cash now)
                 4. PASS 2 — entries with refreshed sizing
                    → size_position clamps if breach max_total_buy_inr
                    → _placeorder() → CNC LIMIT @ tol → MARKET fallback
                 5. record_buy / record_sell → updates model_ledger
                 6. Audit hooks: audit_orders + audit_rebalance_decisions
                 7. Telegram notify: ✅ BUY <model> <sym> x<qty> @ <px>

15:30     Market close. CNC holdings persist overnight.

20:30-45  Daily OHLCV pull per model (post-close, naturally tomorrow's data)
            └─ tools/models/<m>/data_pull.py pull_daily_ohlcv()
               (21:00 legacy saga cron removed 2026-05-28 — live models pull
                their own daily OHLCV here; admin /admin/pipeline still runs
                the legacy saga manually if the admin UI needs a refresh.)

22:00     CSV exports + data-quality validation
22:05     audit_data_quality snapshot (90d retention for trending)
Sun 03:00 Full 4-year backfill (every NSE-EQ symbol, ~2400 stocks)
Mon 06:00 Symbol master refresh from Fyers

App boot: startup pipeline is OPT-IN — set RUN_STARTUP_PIPELINE=true to run
          the legacy saga at boot. Default = skip (the daily cron keeps data
          fresh on its own). Position.sl/target fields removed; the executor
          ignores any sl/target values models still emit in signal JSON.

Re-run / force: each model's Rebalance button (portfolio page) chains
                 live_signal + executor synchronously with progress modal.
```

### Sizing rules (every BUY)

```
slot_alloc = min(cash / slots_left, max_per_trade_inr)
qty        = floor(slot_alloc / price)
# Pre-deduct approx broker charges; if qty*price + charges > cash,
# shrink qty by 1% per iter until cost fits (bounded loop, max 200 iters).
# Hard guardrail: qty *= cap by (max_total_buy_inr - used_value) / price
# qty = 0 if no fit — Telegram alert fires with shortfall details.
```

Per-model `Rebalance` clicks are serialized: if a prior rebalance for the same model is still pending/running (in-process lock OR cross-worker DB check), a duplicate click returns HTTP 409.

---

## Audit — 7-Table Forensics

All audit tables auto-created at app boot via `Base.metadata.create_all()`. Helpers in `src/services/audit_service.py` — never raise (trading must never break on audit failure).

| Table | What it captures | Written by |
|-------|------------------|------------|
| `audit_orders` | Every Fyers `placeorder` request + response, fill price, slippage, order ID, raw JSON | `tools/live/fyers_executor.py::_placeorder()` |
| `audit_rebalance_decisions` | Reasoning per entry attempt: HOLD/ROTATE/OPEN/SKIP_CANNOT_ENTER/SKIP_QTY_ZERO, held vs rank-1, qty before/after clamp | `fyers_executor.main()` entry loop |
| `audit_model_rankings` | Daily top-N snapshot per model — rank, symbol, score, price, universe_size, qualifying_count | All 4 `live_signal.py` files |
| `audit_model_signals` | Every signal emitted including HOLD days — type (ENTRY/EXIT/HOLD), side, price, reason | All 4 `live_signal.py` files |
| `audit_config_changes` | Settings + ledger field deltas — old vs new, reason, who | SQLAlchemy `set` listeners on `ModelSettings` + `ModelLedger` |
| `audit_data_quality` | Daily snapshot of `/admin/system/models-status` — coverage %, stale days, universe age | Scheduler 22:05 IST |
| `audit_system_events` | BOOT, CRON_FIRED, TOKEN_REFRESH, DEPLOY markers | `src/web/app.py` + `data_scheduler.py` startup |

**UI:** single audit dashboard at `/admin/audit` — 7 lazy-loaded tabs, IST timestamps via `window.fmtIST`, slippage colouring, model + days filters.

**API:** read-only JSON endpoints under `/admin/audit/*` (orders, rankings, signals, decisions, config-changes, data-quality, system-events) — accept `?model=` + `?days=` for filtering.

Retention:
- Rankings, signals, orders, decisions, config-changes — forever.
- Data quality — 365 days (rotate after).
- System events — 90 days (high volume).

---

## Risk Controls

1. **Per-model capital cap** — `max_total_buy_inr = allocated + cumulative_realized_pnl` (default ₹30k allocated per active model). `size_position` clamps any BUY that would breach. Logs WARNING line on every clamp. This is the primary boundary on the shared Fyers account.
2. **Concurrency lock** — rebalance endpoint rejects 409 if same-model task already in-flight (in-process threading.Lock + DB-backed cross-worker check).
3. **Account-margin gate + broker rejection** — pre-trade gate checks live Fyers margin; broker is the final backstop on cash. (Replaces the removed `MAX_DAILY_LOSS_PCT` kill-switch, which was dead code — see Recent Changes.)
4. **Model-aware position reconciler** — cron-driven drift detection vs Fyers truth. Auto-mirrors safe drifts (extra fills, corporate actions) into the right per-model row; alerts on `QTY_REDUCED` (external sell), `LEDGER_AHEAD` (broker shows nothing), `FYERS_NET_NEGATIVE` (over-sold), and `SIBLING_OVERCLAIM` (some ledger row lies). Correctly attributes merged Fyers holdings back to each claiming model under cross-model overlap.
5. **Per-trade cap** — `MAX_PER_TRADE_INR` (default capital / max_concurrent).
6. **Always-live** — no env kill switch. `LIVE_TRADING` env key is vestigial (read by zero code). Every signal during market hours places real Fyers orders within the per-model cap. CLI `--dry-run` flag for manual paper runs only. Holiday guard short-circuits both signal emit and order execute on non-trading days.
7. **Agent reconciliation actions need explicit confirmation** — per repo memory rule, the agent will not auto-zero ledgers, force-sell positions, or apply destructive reconciliation without the user explicitly approving the action.

---

## Cross-Model Overlap

The 3 large-cap equity models (`n100`, `pseudo`, `n20`) share a single Fyers account and frequently agree on the biggest momentum winners (ADANIPOWER, IRFC, PFC, SHRIRAMFIN, ADANIGREEN). Over 2023-05 → 2026-05 they collided on the same name for 58 distinct events / 775 overlap-days. `midcap` is structurally safe — its universe (Nifty-100 excluded) cannot collide with the other three.

**Why the broker can't attribute slices.** Fyers reports one `positions[symbol]` and one `holdings[symbol]` per account, both with merged qty and a blended avg price. No model tag. So the same ADANIPOWER 49 806 (n100) + 65 141 (n20) shows up at the broker as a single row with qty 114 947.

**Per-model attribution at fill time.** `record_buy` / `record_sell` (in `src/services/trading/model_ledger_service.py`) stamp each model's own qty + entry_px into its own `model_ledger` row directly from the order response. This is the source of truth for per-model accounting and is always correct.

**Where overlap used to bite.** The cron `position_reconciler` compared each model's expected qty against the broker NET. With overlap, `actual_qty > expected_qty` → AUTO_MIRROR → both ledgers wrote the merged qty → each thought it owned 2× → cash recomputation drifted on every pass.

**The fix.** Two pure helpers in `tools/live/position_reconciler.py`:

| Helper | Returns |
|---|---|
| `sibling_qty_for(ledgers, model_name, symbol)` | Σ `open_qty` across OTHER ledger rows holding the same (normalized) symbol. |
| `decide_drift(expected_qty, expected_px, actual_qty, actual_px, sibling_qty=0)` | `(kind, fix_qty, fix_px)` — `kind` ∈ {`NO_DRIFT`, `AUTO_MIRROR`, `QTY_REDUCED`, `SIBLING_OVERCLAIM`}. |

Under overlap, `my_share = actual_qty - sibling_qty` is compared to `expected_qty`. When AUTO_MIRROR fires under overlap, `fix_px` returns None — the reconciler refuses to write `entry_px` because the broker avg is a cross-model blend. The ledger keeps its own per-model entry_px, which keeps that model's cash math consistent. When `sibling_qty == 0` the helper is byte-for-byte equivalent to the original branches, so the no-overlap path is preserved.

**Why we DON'T dedup at entry.** The combined-portfolio sim (see Recent Changes) shows `allow overlap` beats `block` and `rank2` decisively on both return and Calmar. Models converging on consensus winners IS the alpha — forcing diversification destroys ~60pp of CAGR to shave ~9pp of DD. The right move is to let overlap happen and make the reconciler model-aware.

---

## UI

| Page | URL | Purpose |
|------|-----|---------|
| Dashboard | `/` | Per-model cards: allocated, position, realized, unrealized, NAV, P&L. Yellow market-status banner on weekends / NSE holidays with the next trading day. |
| Today's Picks | `/picks` | Per-model collapsible card with top-5 ranking + Re-calculate button |
| Portfolio | `/portfolio` | Aggregate table + live Fyers funds widget + open positions + per-model Rebalance |
| Model Detail | `/admin/models/<m>/detail` | Balance sheet + trade history per model |
| Admin Triggers | `/admin` | Per-Model Data Status (4 models) + manual pulls |
| Audit | `/admin/audit` | 7-tab forensics dashboard (orders, decisions, rankings, signals, config, data quality, system) |
| History | `/history` | Closed-trade ledger across all models |
| Settings | `/settings` | Per-model enable/disable, capital top-up/withdraw, seed/clear position |
| Users | `/admin/users` | Manage app users |

All timestamps render in IST via `window.fmtIST` helper in `base.html` (naive ISO from backend treated as IST literal, no double-shift).

---

## Data

📄 **Full reference: [DATA_PIPELINE.md](DATA_PIPELINE.md)** — every data fetch + populate path (source → fetcher code → table/file → schedule), incl. Fyers OHLCV, token refresh, universe lists, PIT membership, NSE market-cap scrape + `market_cap_history` / `nifty_index_membership` tables, options bhavcopy, holidays, the data-quality gate, signals/audit/ledger tables, and the full scheduler job map.

- **Postgres** historical_data table — daily OHLCV for ~2400 NSE-EQ stocks (n100 universe back to 2019-04)
- Weekly Sunday 03:00 IST full backfill via `tools/shared/prefetch_ohlcv.py --universe all`
- Per-model daily incremental pulls at 20:30 IST
- Symbol master refresh Mon 06:00 IST
- NSE free-float market cap: quarterly + Sep scrape → `exports/nse_mcap.csv` + `market_cap_history` (see `tools/analysis/MCAP_JOB.md`)
- Live LTP overlaid on every UI display via Fyers `quotes_multiple` API (`_resolve_live_prices` in `admin_routes.py`)

---

## Quick Commands

```bash
# Latest model rankings
curl -s https://stock.oneshell.in/admin/audit/rankings?days=1 | jq .

# Recent Fyers orders
curl -s https://stock.oneshell.in/admin/audit/orders?days=7 | jq .

# Data coverage
curl -s https://stock.oneshell.in/admin/data/coverage | jq .

# Force rebalance one model (UI does this on Rebalance button)
curl -X POST https://stock.oneshell.in/admin/n20_daily_large_only/rebalance \
  -H 'Content-Type: application/json' -d '{"dry_run":false}'

# Force fresh signal recompute
curl -s 'https://stock.oneshell.in/admin/midcap_narrow_60d_breakout/ranking?recalc=1' | jq .

# Disable scheduler for a model (no real orders)
# Settings UI → toggle Enabled → ledger persists, cron skips
```

---

## Runtime + Mobile PWA

- **WSGI:** `gunicorn -w 4 -b 0.0.0.0:5001 wsgi:app` (replaces single-thread Werkzeug). True concurrency across tabs.
- **Session:** `SECRET_KEY` persisted in `.env` (64 hex chars). Container restart / rebuild **does NOT** re-login users.
- **Service Worker (`/static/sw.js` v10):**
  - Static (`/static/*`) — cache-first
  - HTML tab routes (`/dashboard`, `/picks`, `/portfolio`, `/history`, `/settings`) — stale-while-revalidate, precached on install. Tab nav feels instant.
  - APIs (`/api/*`, `/admin/*`, `/login`) — network-only, never cached.
- **Mobile UI:** bottom tab bar with inline Lucide SVGs (no JS lib). Top-right username/Settings/Logout dropdown hidden on mobile (`d-none d-md-flex`). Pull-to-refresh gesture in PWA standalone mode.
- **No CDN JS bloat:** Chart.js + chartjs-plugin-datalabels removed from v2 base (were 225KB of unused payload).
- **DB resources:** `database` container 1G mem, `data_scheduler` 1G, app 1G, dragonfly 512M.

## Deployment

`/app/src` is baked into image at build time (NOT volume-mounted). Code/template changes need rebuild:

```bash
ssh root@77.42.45.12 'cd /opt/trading_system && \
  git pull --ff-only && \
  docker compose build trading_system && \
  docker compose up -d trading_system'
```

Env-only changes: `docker compose up -d --force-recreate trading_system`.

## Tags / Releases

- `v1.0.0` — Initial single-model momentum rotation
- `v2.0-btc-rules` — BTC-correlated regime filter (rejected)
- `v2.1-slope50` — EMA-50 slope refinement
- `v3.0-multi-model-audit` — 4 equity models + 7-table audit trail + capital guardrails
- `v3.1-approx-charges` — Approximate broker charges, live-backtest parity sync, insufficient-cash alerts, per-model ledger reconciler
- `v3.2-always-live` — LIVE_TRADING gate removed (always live), gunicorn 4-worker, persistent SECRET_KEY, SW v10 SWR PWA, inline Lucide nav, dropped chart.js bloat, purged dead `historical_data_15m/1h` tables (4.4G), deleted dead `run_daily.sh`
- **`v3.3-overlap-safe`** — n100 lookback 30 → 15 trading days (6-year-validated); NSE holiday auto-source + dashboard banner; daily-loss kill-switch removed; startup pipeline opt-in; 21:00 legacy cron killed; `Position.sl/target` dropped; **model-aware position reconciler** (sibling-qty subtraction); combined-portfolio sim infrastructure (current)

---

## 10-Year Backtest Appendix (2016-05 → 2026-05)

Run after the 2026-05-28 Fyers backfill that extended `historical_data` back to 2016-04-11 across 504 N500 symbols (~10yr daily, 951K rows added).

**Methodology + caveats** (read first):

- **Universes are today's CSV snapshots** (`nifty100.csv`, `nifty500.csv`, `nifty_smallcap250.csv`). pseudo's yearly-PIT ADV-rank picks top-100 *from today's N500* each year; n100 / n20 / midcap also use today's lists.
- **Survivorship bias is real and material**. Names like IRFC (IPO 2021), MAZDOCK (2020), ETERNAL/Zomato (2021), LIC (2022), POWERINDIA, ADANIPOWER (post-restructure) didn't trade in 2016-2019 — but their winning post-2023 contribution is still in the headline. True 10yr CAGR is likely **5-15pp lower** than reported below. Drawdown numbers are still reliable (crashes hit the names that DID exist).
- **2016 partial year** — backtest starts 2016-05-15. **2026 partial year** — ends 2026-05-12.
- **Pseudo / n20 / midcap need 200d SMA warmup**, so they're effectively flat until ~Feb 2017.
- **Capital ₹10L per model** (research scale); combined sim re-run at live ₹30k confirmed direction (see [Cross-Model Overlap](#cross-model-overlap)).

### Per-model 10yr headline

| Model | CAGR | Max DD | Calmar | Trades | WR |
|---|---:|---:|---:|---:|---:|
| `momentum_n100_top5_max1` (15td lookback) | **+43.31%** | 59.98% | 0.72 | 113 | 57.5% |
| `momentum_pseudo_n100_adv` (30d + SMA200 + ≤₹3k, mid-month + RET5) | +16.88% | **81.30%** | 0.21 | 125 | 49.6% |
| `n40` (weekly, top-40 ADV ∩ N100) | +6.51% | 67.91% | 0.10 | 251 | 49.4% |
| `midcap_narrow_60d_breakout` (40d-high + 2× vol) | +21.00% | 53.14% | 0.40 | 29 | 65.5% |

Calmar < 1 across the board — no model is "amazing" risk-adjusted over a full decade. n100's 0.72 is the strongest. Pseudo's 81% DD is the catastrophic outlier — the 200d SMA + MAX_PRICE filters were tuned on recent data and whipsawed badly in 2017-2020 (see below). The `n40` row is the current PIT weekly engine (apples-to-apples daily would be +0.7%/76%DD — weekly beats daily on every window 2016-26); the other rows predate the PIT rewrite, so don't over-compare across rows.

### Year-by-year breakdown (return %, true MTM)

NAV at each year-end = cash + held_qty × close_price_on_Dec_31 (queried from `historical_data`). This corrects the cash-only proxy — positions held across Dec 31 contribute their unrealized P&L to the year they span, not deferred until the close trade. Especially material for midcap (60-120d holds straddle year-ends frequently) and any final-year open position. See `tools/analysis/yearly_breakdown_mtm.py`.

|  Year | n100 | pseudo | n20 | midcap |
|---:|---:|---:|---:|---:|
| 2016 (partial) | +39.08 | 0.00 | 0.00 | 0.00 |
| 2017 | +8.32 | **−60.22** | −5.62 | −3.93 |
| 2018 | −29.71 | −22.42 | +3.20 | **−34.52** |
| 2019 | −1.74 | −23.07 | −20.67 | +14.62 |
| 2020 | **+153.41** | −12.95 | +37.52 | +26.36 |
| 2021 | −20.46 | +15.50 | +79.13 | +58.24 |
| 2022 | +50.72 | −21.36 | +52.20 | −45.43 |
| 2023 | **+193.67** | **+108.41** | **+209.16** | +34.78 |
| 2024 | +144.08 | +137.51 | **+191.69** | +18.63 |
| 2025 | +25.28 | +90.73 | +38.58 | **+214.06** |
| 2026 (partial, to 05-12) | +28.29 | +58.11 | +15.32 | +69.98 |

### Honest takeaways

Compound returns 2017-2019 (3-year, MTM-corrected): n100 **−25.2%**, pseudo **−76.3%**, n20 **−22.7%**, midcap **−27.9%**. The 2017-2019 stretch was hostile to all four, with pseudo crushed. Detail:

1. **Pseudo 2017 −60.22% is the catastrophe.** SMA200 + MAX_PRICE ₹3k filters force concentrated bets in a flat regime; the model never recovers — it limps along through 2022 before the 2023+ momentum era pulls it back. Worth a research pass on regime-detection or removing the SMA200 gate for 2017-style markets, but parity with live wins for now.
2. **2020+ drove the headline.** 2023 alone (MTM): +194% / +108% / +209% / +35%. The 3yr combined +136% CAGR was the 2023-2025 fat-tail halo; honest 10yr combined is 39%.
3. **Midcap broke down in 2022 (−45.43%) and 2018 (−34.52%).** 40d-high breakouts misfire in declining mid-cap regimes; the −20% stop limits per-trade damage but cumulative whipsaw bleeds equity.
4. **n100 holds up better than the others in 2017-2019** (just −25% cumulative). Pure 15td rank-1 rotation on real Nifty 100 has the least filter-induced fragility.
5. **n20 highest churn (519 trades)** absorbs regime noise; daily rotation rotates out of losers fast, which is why it survives 2017-2019 with only −22.7% cumulative.
6. **Calmar < 1 on all 4 models.** Even n20's best Calmar 0.85 means a year of CAGR-equivalent gain matched by a year of equivalent DD risk. Real expectation: chunky equity curves with multi-year recoveries from regime mismatches.

### Combined-portfolio 10yr (3-bucket, ₹30k each, live cap)

| Policy | CAGR | Max DD | Calmar | Final ₹ |
|---|---:|---:|---:|---:|
| **allow** | **+39.12%** | 65.68% | 0.60 | 2,437,313 |
| block | +29.01% | 55.61% | 0.52 | 1,146,223 |
| rank2 | +33.10% | 60.49% | 0.55 | 1,566,549 |

`allow` still wins on CAGR + Calmar over the full decade. Margin shrinks vs the 3yr sim (136% → 39%) because dedup's missed concentration in 2023-2025 winners is offset by smaller advantages in the regime-hostile 2017-2019 period — but the strategy verdict holds.

**Honest combined DD = 65.68%** (vs 25.91% on 3yr) — this is what you should plan for, not the 3yr figure.

### Artefacts

- `exports/bt10yr/<model>/{summary,trade_ledger}.json` — per-model full ledgers
- `exports/bt10yr/combined_30k.json` — 3-policy combined sim output
- `tools/analysis/yearly_breakdown.py` — post-processor (reproduce any year table)
- `tools/backtests/combined_portfolio_sim.py` — combined-portfolio engine

To reproduce:

```bash
# inside trading_system_app container (where the DB is reachable)
for m in momentum_n100_top5_max1 momentum_pseudo_n100_adv \
         n20_daily_large_only midcap_narrow_60d_breakout; do
  python tools/models/$m/backtest.py --from 2016-05-15 --to 2026-05-12 \
    --capital 1000000 --out /app/exports/bt10yr/$m
done
python tools/backtests/combined_portfolio_sim.py \
  --from 2016-05-15 --to 2026-05-12 --capital 30000 \
  --out /app/exports/bt10yr/combined_30k.json
```

---

## Realistic Caveats

- Backtests are 3-year samples — 2018-style momentum crashes underrepresented. n100 has a 6-year backfill but its honest max DD is ~46%, not the 14.89% 3-year figure.
- All 4 equity models are momentum-correlated; they draw down together in a regime shift. Cross-model overlap means concentration into the consensus winner is up to 2-3× the per-model cap on the shared account — accepted because the combined-portfolio sim shows allowing it dominates dedup on return AND Calmar.
- Live forward expectation: 25-40% CAGR after slippage / STT / STCG, not the 80%+ headline backtest figures.
- Per-model ₹30k cap is SOFTWARE-only on one shared Fyers account (C-FAD51080). Account-margin gate + broker rejection are the hard backstops; the reconciler is the slow-burn drift safety net.
- Fyers MIS auto-square-off at 3:20 IST → equity models use **CNC** to allow multi-day hold (matches backtest).
- TOTP-based token refresh: see `feedback-no-yfinance.md` for the recovery flow.
- NSE holiday fallback list is manual (Python + JS). Auto-pull from `nseindia.com/api/holiday-master?type=trading` runs on boot + monthly 1st-of-month; refresh the fallback list each December in case the API ever goes dark.
