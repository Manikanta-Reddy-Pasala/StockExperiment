# BSE Ltd. (BSE)

## Backtest Summary

- **Window:** 2024-09-02 05:30:00 → 2026-05-08 05:30:00 (417 bars)
- **Last close:** 3907.40
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
| ENTRY1 | 6 |
| ENTRY2 | 0 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 4 / 4
- **Target hits / Stop hits / Partials:** 1 / 4 / 3
- **Avg / median % per leg:** 1.14% / 2.72%
- **Sum % (uncompounded):** 9.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 8 | 4 | 50.0% | 1 | 4 | 3 | 1.14% | 9.1% |
| BUY @ 2nd Alert (retest1) | 8 | 4 | 50.0% | 1 | 4 | 3 | 1.14% | 9.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 8 | 4 | 50.0% | 1 | 4 | 3 | 1.14% | 9.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-10-09 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 05:30:00 | 2330.20 | 2117.09 | 2187.15 | Stage2 pullback-breakout RSI=61 vol=1.7x ATR=74.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-13 05:30:00 | 2479.76 | 2123.30 | 2231.71 | T1 booked 50% @ 2479.76 |
| Stop hit — per-position SL triggered | 2025-10-31 05:30:00 | 2330.20 | 2165.85 | 2404.88 | SL hit (bars_held=15) |

### Cycle 2 — BUY (started 2025-11-03 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 05:30:00 | 2548.30 | 2169.65 | 2418.54 | Stage2 pullback-breakout RSI=66 vol=1.5x ATR=77.67 |
| Stop hit — per-position SL triggered | 2025-11-06 05:30:00 | 2431.79 | 2175.72 | 2428.79 | SL hit (bars_held=2) |

### Cycle 3 — BUY (started 2025-11-07 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-07 05:30:00 | 2678.30 | 2180.72 | 2452.55 | Stage2 pullback-breakout RSI=69 vol=3.6x ATR=90.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 05:30:00 | 2859.01 | 2219.58 | 2611.49 | T1 booked 50% @ 2859.01 |
| Target hit | 2025-12-03 05:30:00 | 2751.10 | 2286.56 | 2775.45 | Trail-exit close<EMA20 |

### Cycle 4 — BUY (started 2026-01-12 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-12 05:30:00 | 2790.60 | 2381.68 | 2695.27 | Stage2 pullback-breakout RSI=58 vol=2.1x ATR=81.77 |
| Stop hit — per-position SL triggered | 2026-01-20 05:30:00 | 2667.95 | 2400.65 | 2721.54 | SL hit (bars_held=5) |

### Cycle 5 — BUY (started 2026-02-10 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 05:30:00 | 3174.20 | 2459.26 | 2834.31 | Stage2 pullback-breakout RSI=70 vol=3.2x ATR=121.29 |
| Stop hit — per-position SL triggered | 2026-02-16 05:30:00 | 2992.26 | 2481.88 | 2896.20 | SL hit (bars_held=4) |

### Cycle 6 — BUY (started 2026-04-08 05:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 05:30:00 | 3163.60 | 2577.48 | 2876.77 | Stage2 pullback-breakout RSI=64 vol=2.3x ATR=124.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 05:30:00 | 3413.45 | 2606.15 | 3020.87 | T1 booked 50% @ 3413.45 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-10-09 05:30:00 | 2330.20 | 2025-10-13 05:30:00 | 2479.76 | PARTIAL | 0.50 | 6.42% |
| BUY | retest1 | 2025-10-09 05:30:00 | 2330.20 | 2025-10-31 05:30:00 | 2330.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-03 05:30:00 | 2548.30 | 2025-11-06 05:30:00 | 2431.79 | STOP_HIT | 1.00 | -4.57% |
| BUY | retest1 | 2025-11-07 05:30:00 | 2678.30 | 2025-11-18 05:30:00 | 2859.01 | PARTIAL | 0.50 | 6.75% |
| BUY | retest1 | 2025-11-07 05:30:00 | 2678.30 | 2025-12-03 05:30:00 | 2751.10 | TARGET_HIT | 0.50 | 2.72% |
| BUY | retest1 | 2026-01-12 05:30:00 | 2790.60 | 2026-01-20 05:30:00 | 2667.95 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest1 | 2026-02-10 05:30:00 | 3174.20 | 2026-02-16 05:30:00 | 2992.26 | STOP_HIT | 1.00 | -5.73% |
| BUY | retest1 | 2026-04-08 05:30:00 | 3163.60 | 2026-04-15 05:30:00 | 3413.45 | PARTIAL | 0.50 | 7.90% |
