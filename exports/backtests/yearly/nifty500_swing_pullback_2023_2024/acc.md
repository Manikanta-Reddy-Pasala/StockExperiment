# ACC Ltd. (ACC)

## Backtest Summary

- **Window:** 2022-09-05 05:30:00 → 2026-05-08 05:30:00 (911 bars)
- **Last close:** 1391.90
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
| TARGET_HIT | 1 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 4 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 2
- **Target hits / Stop hits / Partials:** 1 / 2 / 1
- **Avg / median % per leg:** 0.57% / 4.28%
- **Sum % (uncompounded):** 2.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.57% | 2.3% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.57% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 0.57% | 2.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-01-25 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-25 05:30:00 | 2467.65 | 2043.02 | 2274.95 | Stage2 pullback-breakout RSI=69 vol=5.5x ATR=79.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-09 05:30:00 | 2625.95 | 2088.83 | 2435.87 | T1 booked 50% @ 2625.95 |
| Target hit | 2024-02-28 05:30:00 | 2573.25 | 2158.00 | 2595.81 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-04-01 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-01 05:30:00 | 2544.55 | 2229.19 | 2510.53 | Stage2 pullback-breakout RSI=53 vol=1.5x ATR=72.23 |
| Stop hit — per-position SL triggered | 2024-04-15 05:30:00 | 2436.21 | 2257.29 | 2528.06 | SL hit (bars_held=9) |

### Cycle 3 — BUY (started 2024-04-24 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-24 05:30:00 | 2556.40 | 2268.28 | 2493.54 | Stage2 pullback-breakout RSI=56 vol=2.5x ATR=70.77 |
| Stop hit — per-position SL triggered | 2024-05-07 05:30:00 | 2450.24 | 2287.54 | 2503.65 | SL hit (bars_held=8) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-01-25 05:30:00 | 2467.65 | 2024-02-09 05:30:00 | 2625.95 | PARTIAL | 0.50 | 6.42% |
| BUY | retest1 | 2024-01-25 05:30:00 | 2467.65 | 2024-02-28 05:30:00 | 2573.25 | TARGET_HIT | 0.50 | 4.28% |
| BUY | retest1 | 2024-04-01 05:30:00 | 2544.55 | 2024-04-15 05:30:00 | 2436.21 | STOP_HIT | 1.00 | -4.26% |
| BUY | retest1 | 2024-04-24 05:30:00 | 2556.40 | 2024-05-07 05:30:00 | 2450.24 | STOP_HIT | 1.00 | -4.15% |
