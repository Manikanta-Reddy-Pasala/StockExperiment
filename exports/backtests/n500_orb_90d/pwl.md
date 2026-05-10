# Physicswallah Ltd. (PWL)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 108.35
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
| ENTRY1 | 10 |
| ENTRY2 | 0 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 6 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 6
- **Target hits / Stop hits / Partials:** 4 / 6 / 5
- **Avg / median % per leg:** 0.45% / 0.55%
- **Sum % (uncompounded):** 6.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.28% | 1.4% |
| BUY @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 1 | 3 | 1 | 0.28% | 1.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 10 | 7 | 70.0% | 3 | 3 | 4 | 0.53% | 5.3% |
| SELL @ 2nd Alert (retest1) | 10 | 7 | 70.0% | 3 | 3 | 4 | 0.53% | 5.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 15 | 9 | 60.0% | 4 | 6 | 5 | 0.45% | 6.7% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2026-03-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-16 09:30:00 | 83.73 | 84.16 | 0.00 | ORB-short ORB[83.74,84.74] vol=1.6x ATR=0.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:35:00 | 83.07 | 83.86 | 0.00 | T1 1.5R @ 83.07 |
| Target hit | 2026-03-16 10:00:00 | 82.32 | 82.07 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — BUY (started 2026-04-07 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:05:00 | 93.81 | 92.72 | 0.00 | ORB-long ORB[91.32,92.69] vol=1.9x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 15:00:00 | 94.53 | 93.39 | 0.00 | T1 1.5R @ 94.53 |
| Target hit | 2026-04-07 15:20:00 | 95.66 | 93.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — BUY (started 2026-04-13 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 10:55:00 | 102.24 | 99.85 | 0.00 | ORB-long ORB[98.50,99.89] vol=2.2x ATR=0.46 |
| Stop hit — per-position SL triggered | 2026-04-13 11:05:00 | 101.78 | 100.04 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-04-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:55:00 | 108.98 | 109.53 | 0.00 | ORB-short ORB[109.00,110.27] vol=1.8x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 15:10:00 | 108.38 | 109.14 | 0.00 | T1 1.5R @ 108.38 |
| Target hit | 2026-04-22 15:20:00 | 108.35 | 109.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-04-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 10:45:00 | 107.82 | 108.63 | 0.00 | ORB-short ORB[107.95,109.50] vol=1.7x ATR=0.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-23 10:55:00 | 107.37 | 108.54 | 0.00 | T1 1.5R @ 107.37 |
| Stop hit — per-position SL triggered | 2026-04-23 11:05:00 | 107.82 | 108.45 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-04-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:40:00 | 106.79 | 107.14 | 0.00 | ORB-short ORB[107.04,108.44] vol=1.7x ATR=0.48 |
| Stop hit — per-position SL triggered | 2026-04-24 09:45:00 | 107.27 | 107.13 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 09:35:00 | 105.92 | 106.52 | 0.00 | ORB-short ORB[106.43,107.30] vol=2.0x ATR=0.37 |
| Stop hit — per-position SL triggered | 2026-04-29 10:00:00 | 106.29 | 106.26 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 11:15:00 | 110.92 | 110.12 | 0.00 | ORB-long ORB[109.23,110.58] vol=2.9x ATR=0.37 |
| Stop hit — per-position SL triggered | 2026-05-04 11:25:00 | 110.55 | 110.15 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-05-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 10:45:00 | 107.71 | 108.63 | 0.00 | ORB-short ORB[108.41,109.86] vol=5.5x ATR=0.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 11:35:00 | 107.11 | 108.40 | 0.00 | T1 1.5R @ 107.11 |
| Target hit | 2026-05-05 15:20:00 | 106.05 | 107.15 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 10 — BUY (started 2026-05-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 09:30:00 | 110.79 | 109.90 | 0.00 | ORB-long ORB[108.06,109.50] vol=6.4x ATR=0.60 |
| Stop hit — per-position SL triggered | 2026-05-07 09:40:00 | 110.19 | 110.02 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2026-03-16 09:30:00 | 83.73 | 2026-03-16 09:35:00 | 83.07 | PARTIAL | 0.50 | 0.78% |
| SELL | retest1 | 2026-03-16 09:30:00 | 83.73 | 2026-03-16 10:00:00 | 82.32 | TARGET_HIT | 0.50 | 1.68% |
| BUY | retest1 | 2026-04-07 10:05:00 | 93.81 | 2026-04-07 15:00:00 | 94.53 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-04-07 10:05:00 | 93.81 | 2026-04-07 15:20:00 | 95.66 | TARGET_HIT | 0.50 | 1.97% |
| BUY | retest1 | 2026-04-13 10:55:00 | 102.24 | 2026-04-13 11:05:00 | 101.78 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-22 09:55:00 | 108.98 | 2026-04-22 15:10:00 | 108.38 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-04-22 09:55:00 | 108.98 | 2026-04-22 15:20:00 | 108.35 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2026-04-23 10:45:00 | 107.82 | 2026-04-23 10:55:00 | 107.37 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2026-04-23 10:45:00 | 107.82 | 2026-04-23 11:05:00 | 107.82 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:40:00 | 106.79 | 2026-04-24 09:45:00 | 107.27 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-04-29 09:35:00 | 105.92 | 2026-04-29 10:00:00 | 106.29 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-05-04 11:15:00 | 110.92 | 2026-05-04 11:25:00 | 110.55 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-05-05 10:45:00 | 107.71 | 2026-05-05 11:35:00 | 107.11 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-05-05 10:45:00 | 107.71 | 2026-05-05 15:20:00 | 106.05 | TARGET_HIT | 0.50 | 1.54% |
| BUY | retest1 | 2026-05-07 09:30:00 | 110.79 | 2026-05-07 09:40:00 | 110.19 | STOP_HIT | 1.00 | -0.54% |
