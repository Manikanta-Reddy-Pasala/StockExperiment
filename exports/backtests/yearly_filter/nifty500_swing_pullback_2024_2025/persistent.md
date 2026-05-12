# Persistent Systems Ltd. (PERSISTENT)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-11 00:00:00 (663 bars)
- **Last close:** 5097.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 0 |
| ALERT1 | 0 |
| ALERT2 | 0 |
| ALERT2_SKIP | 0 |
| ALERT3 | 0 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -0.38% / 2.52%
- **Sum % (uncompounded):** -1.53%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.38% | -1.5% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.38% | -1.5% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.38% | -1.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 00:00:00 | 5170.85 | 4020.66 | 4878.52 | Stage2 pullback-breakout RSI=68 vol=3.1x ATR=147.37 |
| Stop hit — per-position SL triggered | 2024-09-16 00:00:00 | 5301.00 | 4150.26 | 5142.49 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2024-10-11 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 00:00:00 | 5469.55 | 4345.08 | 5300.81 | Stage2 pullback-breakout RSI=59 vol=1.6x ATR=162.10 |
| Stop hit — per-position SL triggered | 2024-10-21 00:00:00 | 5226.41 | 4413.02 | 5388.22 | SL hit (bars_held=6) |

### Cycle 3 — BUY (started 2024-10-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-23 00:00:00 | 5718.75 | 4433.35 | 5399.88 | Stage2 pullback-breakout RSI=61 vol=5.2x ATR=202.88 |
| Stop hit — per-position SL triggered | 2024-10-31 00:00:00 | 5414.44 | 4502.08 | 5490.08 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 5917.70 | 4663.54 | 5655.69 | Stage2 pullback-breakout RSI=65 vol=2.2x ATR=169.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 00:00:00 | 6256.17 | 4791.98 | 5899.81 | T1 booked 50% @ 6256.17 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-08-30 00:00:00 | 5170.85 | 2024-09-16 00:00:00 | 5301.00 | STOP_HIT | 1.00 | 2.52% |
| BUY | retest1 | 2024-10-11 00:00:00 | 5469.55 | 2024-10-21 00:00:00 | 5226.41 | STOP_HIT | 1.00 | -4.45% |
| BUY | retest1 | 2024-10-23 00:00:00 | 5718.75 | 2024-10-31 00:00:00 | 5414.44 | STOP_HIT | 1.00 | -5.32% |
| BUY | retest1 | 2024-11-25 00:00:00 | 5917.70 | 2024-12-09 00:00:00 | 6256.17 | PARTIAL | 0.50 | 5.72% |
