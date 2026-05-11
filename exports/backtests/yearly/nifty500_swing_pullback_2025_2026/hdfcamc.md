# HDFC Asset Management Company Ltd. (HDFCAMC)

## Backtest Summary

- **Window:** 2024-09-03 05:30:00 → 2026-05-08 05:30:00 (416 bars)
- **Last close:** 2854.00
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
| PARTIAL | 2 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 2
- **Avg / median % per leg:** 2.01% / 4.55%
- **Sum % (uncompounded):** 12.07%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.01% | 12.1% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.01% | 12.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 1 | 3 | 2 | 2.01% | 12.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-07-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 05:30:00 | 2603.75 | 2227.88 | 2526.12 | Stage2 pullback-breakout RSI=64 vol=1.8x ATR=59.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 05:30:00 | 2722.22 | 2247.81 | 2573.33 | T1 booked 50% @ 2722.22 |
| Target hit | 2025-08-11 05:30:00 | 2750.50 | 2335.14 | 2764.17 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2025-10-15 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 05:30:00 | 2882.00 | 2512.78 | 2819.37 | Stage2 pullback-breakout RSI=57 vol=3.2x ATR=61.79 |
| Stop hit — per-position SL triggered | 2025-10-16 05:30:00 | 2789.31 | 2516.65 | 2827.21 | SL hit (bars_held=1) |

### Cycle 3 — BUY (started 2025-12-18 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 05:30:00 | 2722.90 | 2575.00 | 2637.89 | Stage2 pullback-breakout RSI=57 vol=3.7x ATR=62.79 |
| Stop hit — per-position SL triggered | 2025-12-29 05:30:00 | 2628.71 | 2580.27 | 2649.61 | SL hit (bars_held=6) |

### Cycle 4 — BUY (started 2026-02-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-03 05:30:00 | 2671.30 | 2574.54 | 2547.43 | Stage2 pullback-breakout RSI=59 vol=2.1x ATR=83.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 05:30:00 | 2838.94 | 2588.22 | 2666.70 | T1 booked 50% @ 2838.94 |
| Stop hit — per-position SL triggered | 2026-02-19 05:30:00 | 2732.20 | 2599.16 | 2722.59 | Time-stop (10d <3%) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-07-10 05:30:00 | 2603.75 | 2025-07-17 05:30:00 | 2722.22 | PARTIAL | 0.50 | 4.55% |
| BUY | retest1 | 2025-07-10 05:30:00 | 2603.75 | 2025-08-11 05:30:00 | 2750.50 | TARGET_HIT | 0.50 | 5.64% |
| BUY | retest1 | 2025-10-15 05:30:00 | 2882.00 | 2025-10-16 05:30:00 | 2789.31 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest1 | 2025-12-18 05:30:00 | 2722.90 | 2025-12-29 05:30:00 | 2628.71 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest1 | 2026-02-03 05:30:00 | 2671.30 | 2026-02-12 05:30:00 | 2838.94 | PARTIAL | 0.50 | 6.28% |
| BUY | retest1 | 2026-02-03 05:30:00 | 2671.30 | 2026-02-19 05:30:00 | 2732.20 | STOP_HIT | 0.50 | 2.28% |
