# StockExperiment — Multi-Model Live Trading + Audit

NSE momentum & breakout trading system running **4 active equity models** + 1 disabled options scaffold in parallel against a single Fyers brokerage account. Each model has its own capital pool, own ledger, own ranking signal, own rebalance cadence. Live orders fire daily via cron 09:30 IST; every decision is captured in a 7-table audit trail.

**Charges are approximate** (formula-based — Fyers SEBI rates), deducted from per-model cash at fill time. Not chased to broker-exact.

Production: `77.42.45.12` · App: <https://stock.oneshell.in> · Bot: `@stocks_momrot_bot`

---

## Models

| Model | Universe | Cadence | Product | Hold | Signal |
|-------|----------|---------|---------|------|--------|
| `momentum_n100_top5_max1` | Real Nifty 100 | Monthly (1st weekday) | CNC delivery | until rank-1 changes | top-5 by 30d return, hold rank-1 |
| `momentum_pseudo_n100_adv` | Top-100 ADV from N500 minus Smallcap-250, yearly PIT rebuild, close > 200d SMA | Monthly | CNC | until rank-1 changes | top-5 by 30d return + uptrend gate |
| `midcap_narrow_60d_breakout` | ~100 NSE midcaps (top-100 ADV minus Nifty 100) | Event-driven (daily check) | CNC | up to 120d / target +100% / trail -20% from peak | 40d-high + vol >2× + 200d SMA, ALL must fire |
| `n20_daily_large_only` | Top-20 ADV ∩ Nifty 100 | Daily | CNC | until rank-1 changes | rank by 30d return + 200d SMA uptrend filter (PIT) |
| `finnifty_ic_otm4_w300_lots5` | FinNifty weekly | Weekly expiry | Options multi-leg | weekly | OTM4 iron condor, 300pt wing, 5 lots (executor not yet wired — currently DISABLED) |

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

09:30-32  execute_orders (LIVE_TRADING-gated)
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

21:00     Legacy 4-step saga (technical indicators + market cap refresh)
22:00     CSV exports + data-quality validation
22:05     audit_data_quality snapshot (90d retention for trending)
Sun 03:00 Full 4-year backfill (every NSE-EQ symbol, ~2400 stocks)
Mon 06:00 Symbol master refresh from Fyers

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

1. **Per-model capital cap** — `max_total_buy_inr = allocated + cumulative_realized_pnl`. Size_position clamps any BUY that would breach. Logs WARNING line on every clamp.
2. **Concurrency lock** — rebalance endpoint rejects 409 if same-model task already in-flight (in-process threading.Lock + DB-backed cross-worker check).
3. **Daily loss kill-switch** — `MAX_DAILY_LOSS_PCT = -5.0` blocks new entries.
4. **No agent trading** — per repo memory rule (`feedback-no-real-trades.md`): the agent never invokes placeorder, only the user via UI buttons or the scheduler.
5. **LIVE_TRADING gate** — executor falls back to dry-run if `LIVE_TRADING != true`.
6. **Per-trade cap** — `MAX_PER_TRADE_INR` (default capital / max_concurrent).

---

## UI

| Page | URL | Purpose |
|------|-----|---------|
| Dashboard | `/` | Per-model cards: allocated, position, realized, unrealized, NAV, P&L |
| Today's Picks | `/picks` | Per-model collapsible card with top-5 ranking + Re-calculate button |
| Portfolio | `/portfolio` | Aggregate table + live Fyers funds widget + open positions + per-model Rebalance |
| Model Detail | `/admin/models/<m>/detail` | Balance sheet + trade history per model |
| Admin Triggers | `/admin` | Per-Model Data Status (5 models) + manual pulls |
| Audit | `/admin/audit` | 7-tab forensics dashboard (orders, decisions, rankings, signals, config, data quality, system) |
| History | `/history` | Closed-trade ledger across all models |
| Settings | `/settings` | Per-model enable/disable, capital top-up/withdraw, seed/clear position |
| Users | `/admin/users` | Manage app users |

All timestamps render in IST via `window.fmtIST` helper in `base.html` (naive ISO from backend treated as IST literal, no double-shift).

---

## Data

- **Postgres** historical_data table — 4 years of daily OHLCV for ~2400 NSE-EQ stocks
- Weekly Sunday 03:00 IST full backfill via `tools/shared/prefetch_ohlcv.py --universe all`
- Per-model daily incremental pulls at 20:30-45 IST
- Symbol master refresh Mon 06:00 IST
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

## Tags / Releases

- `v1.0.0` — Initial single-model momentum rotation
- `v2.0-btc-rules` — BTC-correlated regime filter (rejected)
- `v2.1-slope50` — EMA-50 slope refinement
- `v3.0-multi-model-audit` — 5 isolated models + 7-table audit trail + capital guardrails
- **`v3.1-approx-charges`** — Approximate broker charges (formula), live-backtest parity sync (midcap V2 + pseudo_n100 smallcap exclude + 200d uptrend), insufficient-cash Telegram alerts, per-model ledger reconciler (current)

---

## Realistic Caveats

- Backtests are 3-year samples — 2018-style momentum crashes underrepresented.
- All 4 equity models are momentum-correlated; they will draw down together in a regime shift.
- Live forward expectation: 25-40% CAGR after slippage / STT / STCG, not the 80%+ headline backtest figures.
- Fyers MIS auto-square-off at 3:20 IST → equity models use **CNC** to allow multi-day hold (matches backtest).
- TOTP-based token refresh: see `feedback-no-yfinance.md` for the recovery flow.
