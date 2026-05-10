# Garden Reach Shipbuilders & Engineers Ltd. (GRSE)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 3051.30
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
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 7 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 2
- **Target hits / Stop hits / Partials:** 1 / 3 / 3
- **Avg / median % per leg:** 2.61% / 1.15%
- **Sum % (uncompounded):** 18.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 7 | 5 | 71.4% | 1 | 3 | 3 | 2.61% | 18.3% |
| BUY @ 2nd Alert (retest1) | 7 | 5 | 71.4% | 1 | 3 | 3 | 2.61% | 18.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 7 | 5 | 71.4% | 1 | 3 | 3 | 2.61% | 18.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-09-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 05:30:00 | 2581.90 | 2255.46 | 2474.87 | Stage2 pullback-breakout RSI=55 vol=4.8x ATR=109.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-25 05:30:00 | 2800.15 | 2286.71 | 2571.18 | T1 booked 50% @ 2800.15 |
| Stop hit — per-position SL triggered | 2025-09-26 05:30:00 | 2611.50 | 2289.94 | 2575.02 | Time-stop (10d <3%) |

### Cycle 2 — BUY (started 2025-11-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 05:30:00 | 2682.50 | 2368.75 | 2598.36 | Stage2 pullback-breakout RSI=59 vol=4.3x ATR=79.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 05:30:00 | 2841.56 | 2377.32 | 2635.31 | T1 booked 50% @ 2841.56 |
| Target hit | 2025-11-24 05:30:00 | 2711.90 | 2412.40 | 2742.09 | Trail-exit close<EMA20 |

### Cycle 3 — BUY (started 2026-01-28 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 05:30:00 | 2518.70 | 2423.59 | 2402.02 | Stage2 pullback-breakout RSI=58 vol=2.2x ATR=99.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 05:30:00 | 2718.25 | 2428.57 | 2452.28 | T1 booked 50% @ 2718.25 |
| Stop hit — per-position SL triggered | 2026-02-01 05:30:00 | 2518.70 | 2429.50 | 2458.94 | SL hit (bars_held=3) |

### Cycle 4 — BUY (started 2026-03-06 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-06 05:30:00 | 2523.20 | 2435.52 | 2442.60 | Stage2 pullback-breakout RSI=55 vol=4.6x ATR=105.51 |
| Stop hit — per-position SL triggered | 2026-03-13 05:30:00 | 2364.94 | 2435.36 | 2435.48 | SL hit (bars_held=5) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-09-12 05:30:00 | 2581.90 | 2025-09-25 05:30:00 | 2800.15 | PARTIAL | 0.50 | 8.45% |
| BUY | retest1 | 2025-09-12 05:30:00 | 2581.90 | 2025-09-26 05:30:00 | 2611.50 | STOP_HIT | 0.50 | 1.15% |
| BUY | retest1 | 2025-11-10 05:30:00 | 2682.50 | 2025-11-12 05:30:00 | 2841.56 | PARTIAL | 0.50 | 5.93% |
| BUY | retest1 | 2025-11-10 05:30:00 | 2682.50 | 2025-11-24 05:30:00 | 2711.90 | TARGET_HIT | 0.50 | 1.10% |
| BUY | retest1 | 2026-01-28 05:30:00 | 2518.70 | 2026-01-30 05:30:00 | 2718.25 | PARTIAL | 0.50 | 7.92% |
| BUY | retest1 | 2026-01-28 05:30:00 | 2518.70 | 2026-02-01 05:30:00 | 2518.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-06 05:30:00 | 2523.20 | 2026-03-13 05:30:00 | 2364.94 | STOP_HIT | 1.00 | -6.27% |
