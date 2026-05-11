# Persistent Systems Ltd. (PERSISTENT)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
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
- **Avg / median % per leg:** 1.60% / 5.00%
- **Sum % (uncompounded):** 6.40%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.60% | 6.4% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.60% | 6.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 4 | 2 | 50.0% | 1 | 2 | 1 | 1.60% | 6.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-30 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-30 00:00:00 | 2505.53 | 2196.32 | 2467.73 | Stage2 pullback-breakout RSI=56 vol=1.6x ATR=55.58 |
| Stop hit — per-position SL triggered | 2023-07-04 00:00:00 | 2422.16 | 2200.99 | 2461.31 | SL hit (bars_held=2) |

### Cycle 2 — BUY (started 2023-07-14 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-14 00:00:00 | 2462.00 | 2216.37 | 2427.59 | Stage2 pullback-breakout RSI=54 vol=2.3x ATR=59.02 |
| Stop hit — per-position SL triggered | 2023-07-21 00:00:00 | 2373.47 | 2229.59 | 2449.68 | SL hit (bars_held=5) |

### Cycle 3 — BUY (started 2023-10-19 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 00:00:00 | 2924.48 | 2458.26 | 2871.07 | Stage2 pullback-breakout RSI=59 vol=2.1x ATR=73.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-27 00:00:00 | 3070.70 | 2482.57 | 2906.32 | T1 booked 50% @ 3070.70 |
| Target hit | 2023-12-06 00:00:00 | 3167.90 | 2644.69 | 3169.77 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-06-30 00:00:00 | 2505.53 | 2023-07-04 00:00:00 | 2422.16 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest1 | 2023-07-14 00:00:00 | 2462.00 | 2023-07-21 00:00:00 | 2373.47 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest1 | 2023-10-19 00:00:00 | 2924.48 | 2023-10-27 00:00:00 | 3070.70 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2023-10-19 00:00:00 | 2924.48 | 2023-12-06 00:00:00 | 3167.90 | TARGET_HIT | 0.50 | 8.32% |
