# Per-Model "Invest More" — Design Spec

Date: 2026-06-10
Status: Approved (design), pending implementation

## Goal

On the `/v2/portfolio` per-model breakdown, add a per-model **"Invest more"** action that:
1. On click, **suggests** how to deploy the model's idle cash (read-only, no orders).
2. On **Approve**, places the suggested live BUY order(s) and updates the model ledger.

This lets the operator deploy spare allocated cash mid-cadence without waiting for the model's scheduled rebalance, while staying true to each model's own pick logic.

## Context

- Models hold a software allocation (`allocated_capital`) and deploy on their own cadence (monthly / weekly / daily-retest). Newly deposited cash sits idle until the next scheduled rebalance.
- Existing machinery to reuse:
  - `_fyers_available_cash(user_id)` (momrot_routes) — broker free cash.
  - Model ledger service (`src/services/trading/model_ledger_service.py`) — `deposit()`, allocation/invested/cash per model.
  - Per-model ranking / live-signal (each model's `run-signal` produces current picks).
  - Live buy path: `tools/live/fyers_executor.py` (used by `/admin/<model>/run-execute`), and `/momrot/buy-now`.
  - pg-advisory trading lock (`src/services/trading/trade_lock.py`).
  - Telegram `_tg_safe`.

## Decisions (locked)

- **Suggest logic:** deploy idle cash per the model's OWN current pick logic — single-position models (Nifty 100 Momentum, Weekly Top-40) top up their current rank-1; Retest Momentum fills empty slots with its eligible top-4. No selling. Sized to `min(model idle cash, broker free cash)`.
- **Order timing:** market hours only (09:15–15:30 IST). Approve disabled / blocked when market closed ("market closed — deploys next session").

## Architecture

### New: `src/services/trading/model_invest_service.py`
Pure logic over existing services. Two functions:

- `preview(model_name, user_id=1) -> dict`
  - `allocated`, `invested` from ledger → `idle = max(0, allocated - invested)`.
  - `broker_free = _fyers_available_cash(user_id)`.
  - `deployable = min(idle, broker_free)`.
  - `targets = model_current_targets(model_name)` — the model's intended holdings now:
    - single-position model: its current rank-1 (symbol + LTP). If already holding rank-1, target = same symbol (top-up).
    - Retest (multi-slot, top_n=4): the eligible top-4 targets for currently EMPTY slots only.
  - Size buys: split `deployable` across target slots needing cash, `qty = floor(slot_budget / ltp)`, drop zero-qty. Ensure `sum(qty*ltp) <= deployable`.
  - Returns `{ idle, broker_free, deployable, market_open: bool, buys: [{symbol, ltp, qty, amount}], token }`.
  - `token` = idempotency token (hash of model+date+buys), stored to guard double-submit.
- `execute(model_name, buys, token, user_id=1) -> dict`
  - Reject if `token` already consumed (idempotency) or market closed.
  - Re-read broker free cash; re-validate `sum(qty*ltp) <= min(idle, broker_free)` (stale-safe; shrink/abort if changed).
  - Acquire per-model trading lock.
  - Place live market BUY for each `{symbol, qty}` via the existing executor/order entrypoint, tagged to `model_name`.
  - On fills: update ledger (`invested += filled_value`, `cash` adjust), record trades.
  - Telegram notify. Return fills + new ledger snapshot.

`model_current_targets(model_name)` — thin helper that reads the model's latest ranking/signal (reuse `run-signal` output file or the model's ranking module). Exact source confirmed at build time; isolated behind this helper so the routes/service don't depend on it directly.

### New routes (admin_routes.py or small blueprint)
- `GET  /api/models/<model_name>/invest-preview` → `model_invest_service.preview(...)`. Read-only.
- `POST /api/models/<model_name>/invest-execute` → body `{buys, token}` → `model_invest_service.execute(...)`. Live order.

### UI: `src/web/templates/v2/portfolio.html`
- Per-model row/card: **"Invest more"** button, rendered only when `idle > 0`.
- Click → fetch invest-preview → modal: idle · broker free · deployable · buy list (symbol, LTP, qty, ₹) · market-open status.
- **Approve** button in modal (disabled when market closed) → POST invest-execute → show fills, refresh portfolio.
- SW `CACHE_VERSION` bump.

## Data flow

```
[Invest more] -> GET invest-preview -> {idle, deployable, buys[], market_open, token}
   -> modal renders buys
[Approve] -> POST invest-execute {buys, token}
   -> re-validate (market open, cash, token unused) -> trade lock
   -> live BUY orders -> ledger update -> Telegram -> {fills}
   -> UI refresh
```

## Error handling

- `idle <= 0` → button hidden; preview returns `deployable=0`, empty buys.
- Market closed → preview `market_open=false`; execute rejects with clear message; Approve disabled in UI.
- Broker cash dropped since preview → execute shrinks to affordable qty or aborts with message (never overdraws).
- Token reused / double-submit → execute rejects (idempotent).
- No current target (model flat + no rank, or all slots filled) → empty buys, "nothing to deploy".
- Order/API failure → surfaced per-symbol; partial fills recorded honestly.
- Concurrency with cron rebalance → per-model pg-advisory lock; execute waits/abort rather than collide.

## Testing

- Unit (`model_invest_service`):
  - sizing: idle-cash cap, broker-cash cap, qty floor, multi-slot split (Retest) vs single top-up.
  - market-hours gate (open/closed).
  - idempotency token consume/reject.
  - cash-shrink-at-execute path.
- Dry-run execute path (no real order placed) — assert order intents, no Fyers live call.
- Manual on VM: dry-run preview for each of the 3 models, confirm suggested symbol == model's actual current rank-1 / Retest empty-slot targets; confirm deployable == min(idle, broker free).

## Out of scope (YAGNI)

- Selling / full rebalance (decision: deploy idle cash only).
- After-hours queuing (decision: market-hours only).
- Editing the suggested qty in the modal (approve-as-suggested only; can add later if needed).
