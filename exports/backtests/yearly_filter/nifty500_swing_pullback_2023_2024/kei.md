# KEI Industries Ltd. (KEI)

## Backtest Summary

- **Window:** 2022-09-05 00:00:00 → 2026-05-11 00:00:00 (912 bars)
- **Last close:** 5112.10
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
| TARGET_HIT | 2 |
| STOP_HIT | 2 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 2
- **Target hits / Stop hits / Partials:** 2 / 2 / 2
- **Avg / median % per leg:** 5.22% / 8.79%
- **Sum % (uncompounded):** 31.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 6 | 4 | 66.7% | 2 | 2 | 2 | 5.22% | 31.3% |
| BUY @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 5.22% | 31.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 5.22% | 31.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-11-02 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-02 00:00:00 | 2585.70 | 2213.62 | 2566.20 | Stage2 pullback-breakout RSI=51 vol=2.6x ATR=99.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-17 00:00:00 | 2784.09 | 2252.87 | 2589.29 | T1 booked 50% @ 2784.09 |
| Target hit | 2024-01-11 00:00:00 | 2980.95 | 2496.71 | 3131.73 | Trail-exit close<EMA20 |

### Cycle 2 — BUY (started 2024-01-24 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-24 00:00:00 | 3314.75 | 2548.01 | 3115.73 | Stage2 pullback-breakout RSI=62 vol=3.0x ATR=144.26 |
| Stop hit — per-position SL triggered | 2024-01-30 00:00:00 | 3098.36 | 2565.67 | 3122.53 | SL hit (bars_held=3) |

### Cycle 3 — BUY (started 2024-03-04 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-04 00:00:00 | 3420.40 | 2715.74 | 3239.29 | Stage2 pullback-breakout RSI=63 vol=3.4x ATR=113.11 |
| Stop hit — per-position SL triggered | 2024-03-06 00:00:00 | 3250.74 | 2728.68 | 3262.78 | SL hit (bars_held=2) |

### Cycle 4 — BUY (started 2024-03-20 00:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 00:00:00 | 3420.10 | 2773.47 | 3256.52 | Stage2 pullback-breakout RSI=58 vol=1.7x ATR=150.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-08 00:00:00 | 3720.69 | 2850.69 | 3444.70 | T1 booked 50% @ 3720.69 |
| Target hit | 2024-05-06 00:00:00 | 3798.15 | 3012.46 | 3811.77 | Trail-exit close<EMA20 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-11-02 00:00:00 | 2585.70 | 2023-11-17 00:00:00 | 2784.09 | PARTIAL | 0.50 | 7.67% |
| BUY | retest1 | 2023-11-02 00:00:00 | 2585.70 | 2024-01-11 00:00:00 | 2980.95 | TARGET_HIT | 0.50 | 15.29% |
| BUY | retest1 | 2024-01-24 00:00:00 | 3314.75 | 2024-01-30 00:00:00 | 3098.36 | STOP_HIT | 1.00 | -6.53% |
| BUY | retest1 | 2024-03-04 00:00:00 | 3420.40 | 2024-03-06 00:00:00 | 3250.74 | STOP_HIT | 1.00 | -4.96% |
| BUY | retest1 | 2024-03-20 00:00:00 | 3420.10 | 2024-04-08 00:00:00 | 3720.69 | PARTIAL | 0.50 | 8.79% |
| BUY | retest1 | 2024-03-20 00:00:00 | 3420.10 | 2024-05-06 00:00:00 | 3798.15 | TARGET_HIT | 0.50 | 11.05% |
