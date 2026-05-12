# NCC Ltd. (NCC)

## Backtest Summary

- **Window:** 2023-09-05 00:00:00 → 2026-05-08 00:00:00 (662 bars)
- **Last close:** 170.04
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
| ENTRY1 | 3 |
| ENTRY2 | 0 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 0 / 3 / 1
- **Avg / median % per leg:** -0.81% / 2.29%
- **Sum % (uncompounded):** -3.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.81% | -3.2% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.81% | -3.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 0 | 3 | 1 | -0.81% | -3.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-03 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 00:00:00 | 341.00 | 236.07 | 318.35 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=14.72 |
| Stop hit — per-position SL triggered | 2024-07-18 00:00:00 | 318.91 | 245.16 | 325.97 | SL hit (bars_held=10) |

### Cycle 2 — BUY (started 2024-07-23 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 00:00:00 | 337.75 | 247.41 | 324.97 | Stage2 pullback-breakout RSI=58 vol=6.2x ATR=16.78 |
| Stop hit — per-position SL triggered | 2024-08-06 00:00:00 | 312.58 | 255.95 | 331.67 | SL hit (bars_held=10) |

### Cycle 3 — BUY (started 2024-11-25 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-25 00:00:00 | 299.60 | 281.57 | 292.41 | Stage2 pullback-breakout RSI=53 vol=1.9x ATR=12.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 00:00:00 | 324.81 | 283.86 | 302.47 | T1 booked 50% @ 324.81 |
| Stop hit — per-position SL triggered | 2024-12-11 00:00:00 | 306.45 | 284.66 | 304.59 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-07-03 00:00:00 | 341.00 | 2024-07-18 00:00:00 | 318.91 | STOP_HIT | 1.00 | -6.48% |
| BUY | retest1 | 2024-07-23 00:00:00 | 337.75 | 2024-08-06 00:00:00 | 312.58 | STOP_HIT | 1.00 | -7.45% |
| BUY | retest1 | 2024-11-25 00:00:00 | 299.60 | 2024-12-06 00:00:00 | 324.81 | PARTIAL | 0.50 | 8.42% |
| BUY | retest1 | 2024-11-25 00:00:00 | 299.60 | 2024-12-11 00:00:00 | 306.45 | STOP_HIT | 0.50 | 2.29% |
