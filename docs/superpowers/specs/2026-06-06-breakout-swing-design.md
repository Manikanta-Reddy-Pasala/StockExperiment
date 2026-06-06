# breakout_swing_n500 — Momentum-Burst Short-Hold (1–5d) Model

**Date:** 2026-06-06
**Status:** Design approved, pending backtest validation
**Author:** brainstorming session (user + Claude)

## Context & Motivation

User asked for "HFT / day trading". Hard reality established up front:

- **True HFT is infeasible on this stack.** Fyers retail REST API (100–500ms
  round-trip, rate-limited, no streaming L2 depth, no DMA, no co-location,
  ₹30k/model cap). Latency alone disqualifies any microsecond strategy.
- **Intraday already proven dead.** ORB (opening-range breakout) was built,
  found to be lookahead-biased; after the fix it returned −63% total / −72% CAGR /
  67% DD, negative every year, any selection/direction (−60% to −95%). Archived
  and removed from the system (commit ac7c6e6e).

Chosen achievable target: a **short-hold swing model** — faster than the current
monthly/weekly rotation models, holding 1–5 trading days. This is the
intermediate band between proven multi-day rotation and the dead intraday space.

This model is **experimental** and must justify itself against the existing best
unleveraged model (emerging_momentum, ~+114% CAGR full-cycle). If the backtest
shows no edge after an honest no-lookahead run, it will NOT be shipped — same
discipline applied to ORB.

## Goals

- Capture momentum bursts: stocks breaking out of consolidation on volume surge.
- Hold 1–5 days, exit on target / stop / trail / time.
- Reuse the existing shared breakout cores so backtest and live cannot drift.
- Backtest honestly (no lookahead) and only ship if a config beats a sane bar.

## Non-Goals (YAGNI)

- No intraday entries/exits (proven dead, and high slippage on retail latency).
- No new executor — reuse `tools/live/fyers_executor.py` (sequential SELL→BUY,
  limit-with-fallback, blocking fills).
- No options, no leverage, no overnight gap trading (separate alpha, higher risk).
- No new shared-core abstractions unless the sweep proves the model worth keeping.

## Architecture

Mirrors the existing model layout exactly (one folder under `tools/models/`):

```
tools/models/breakout_swing_n500/
  strategy.py      # constants + thin wrappers over shared cores (single source)
  backtest.py      # historical sim, real fyers data, no-lookahead
  live_signal.py   # emits ENTRY/EXIT signal dicts for fyers_executor
  cron.py          # daily emit + daily exit-check scheduling
  SUMMARY.md       # results writeup (after backtest)
```

Three-layer separation (same as the rest of the system):
- **SELECTION** (per-model, in this folder): universe build + breakout ranking.
- **RULE** (shared): `tools/shared/breakout_strategy.py` — `is_breakout` +
  `breakout_exit_reason`. Already exists; reused verbatim.
- **EXECUTION**: backtest engine path + live `fyers_executor` (shared).

### Shared-core reuse (parity guarantee)

- Entry qualification: `breakout_strategy.is_breakout(close, prior_high, sma_long,
  vol, vol_avg20, vol_mult)` — used by both backtest.py and live_signal.py.
- Exit decision: `breakout_strategy.breakout_exit_reason(entry_px, close, peak,
  age, target, stop, trail, profit_trigger, max_hold)` — used by both.

Because both sides import the same pure functions, the decision cannot diverge.
This is the same pattern that already keeps midcap_narrow_60d_breakout in parity.

## Selection Detail

### Universe
- PIT N500 via `eligible_at("n500", date)` (backtest) / current N500 list (live),
  both PIT-filtered to avoid survivorship bias (the documented retest fix pattern).
- Rank by 20d ADV, take **top 150** (breakout breadth without illiquid tail).
- Price filter: **₹100 ≤ close ≤ ₹3000** (ORB lesson: sub-₹100 pennies whipsaw
  into fake breakouts; ₹3000 cap consistent with other models for share-granularity).
- Uptrend gate: `close > 200d SMA`.

### Entry ranking — NO LOOKAHEAD
- For each universe symbol at decision index `di`, evaluate `is_breakout` using the
  **observed completed close at `di`** and the prior-40d high (`shift(1)`, excludes
  today). Rank qualifiers by **volume-surge ratio** (vol / 20d-avg-vol), highest first.
- **Transact at the SAME observed bar's close `di`** (not `di+1` open, not a future
  bar). Ranking input and transaction price are the same observed close → live can
  reproduce exactly. This is the explicit guard against the ORB sin (which ranked
  `di` then traded `di`'s open).
- Multi-position: fill free slots in rank order, equal-weight by available cash.

## Exit Detail (the short-hold constraint)

Daily check on every held position via `breakout_exit_reason`:
- **Target**: +X% from entry (sweep).
- **Stop**: −Y% from entry, checked on the day's LOW (sweep).
- **Trail**: Z% off the running peak, peak seeded at entry (sweep).
- **MAX_HOLD**: **5 trading days** — the defining swing constraint; force-exit at
  close if no other exit fired. (Also swept: 1/2/3/5 to find the hold sweet spot.)
- Stop has priority over target when both touched intrabar (conservative).

## Cadence

- Daily (every trading day, weekday-gated inside live_signal like other models):
  1. Exit-check each held position → emit EXIT (STOP_HIT / TARGET_HIT / etc).
  2. Scan for new breakouts → emit ENTRY into any free slots.
- cron.py: `emit_signal` ~09:29 daily (entries + exits in one pass). Self-gates on
  weekend/holiday via `nse_calendar` + model-enabled flag.

## Position Sizing

- Equal-weight across N concurrent slots, N swept ∈ {1, 3, 5}.
- Single (N=1) = all-in strongest breakout; multi spreads across clustered breakouts.
- Sized from `model_ledger.cash` (multi-holding via `model_holdings`, already supported).

## Backtest Sweep (honest validation)

- Period: 2021-03-01 → 2026-05-29, real fyers `data_source='fyers'`.
- Charges: net 0.15%/side (consistent with other backtests).
- Grid:
  - N (slots): {1, 3, 5}
  - target: {6%, 10%, 15%}
  - stop: {4%, 6%, 8%}
  - trail: {off, 5%, 8%}
  - max_hold: {2, 3, 5} trading days
  - vol_mult: {1.5, 2.0}
- Output per config: CAGR, total return, MaxDD, Calmar, trades, win-rate, per-year
  return + DD. Pick the config with the best Calmar that also has a positive return
  every year (robustness over peak CAGR).

### KILL GATE
Ship ONLY if the winning config:
1. Survives the no-lookahead check (rank input == transact bar), AND
2. Is positive every calendar year, AND
3. Has a defensible Calmar (≥ ~2) and a return that justifies adding a 7th model
   alongside emerging (+114% CAGR).

If no config clears the gate → archive the experiment (do not enable), document
the negative result in SUMMARY.md, like ORB. A negative result is a valid outcome.

## Testing

- `tests/test_breakout_swing_parity.py`: assert live `check_exit` == core
  `breakout_exit_reason` for a battery of cases; assert live entry qualification ==
  `is_breakout` on the same synthetic bars (mirrors existing test_breakout_parity).
- No-lookahead assertion: a test that the backtest ranks and transacts on the same
  index (regression guard against re-introducing the ORB bug).
- Full suite must stay green (currently 221 passing).

## Risks & Open Questions

- **Edge may not exist.** Short-hold breakout in NSE equity with retail
  latency/charges may not beat buy-and-hold-the-leader. The sweep + kill gate
  handle this honestly.
- **Slippage on fast exits.** 5d hold means more round-trips than monthly rotation
  → charges/slippage drag is higher. Modelled via 0.15%/side; live may be worse.
- **Capital churn** vs the ₹30k/model cap — multi-position with fast rotation needs
  the cap honored per the capital-realloc rules. Live sizing already reads ledger cash.

## Out of Scope / Deferred

- Live deployment is a SEPARATE step, only after backtest clears the kill gate and
  user explicitly approves (per project no-auto-deploy rule).
- UI cards / picks wiring / Telegram notify: added only if the model ships.
