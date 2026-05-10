# RBL Bank Ltd. (RBLBANK)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 343.65
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
| ENTRY1 | 20 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 26 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 16
- **Target hits / Stop hits / Partials:** 4 / 16 / 6
- **Avg / median % per leg:** 0.13% / -0.29%
- **Sum % (uncompounded):** 3.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.19% | 2.7% |
| BUY @ 2nd Alert (retest1) | 14 | 7 | 50.0% | 3 | 7 | 4 | 0.19% | 2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 12 | 3 | 25.0% | 1 | 9 | 2 | 0.07% | 0.8% |
| SELL @ 2nd Alert (retest1) | 12 | 3 | 25.0% | 1 | 9 | 2 | 0.07% | 0.8% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 26 | 10 | 38.5% | 4 | 16 | 6 | 0.13% | 3.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-12 10:15:00 | 310.10 | 307.95 | 0.00 | ORB-long ORB[306.35,309.90] vol=2.7x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 10:20:00 | 311.59 | 308.53 | 0.00 | T1 1.5R @ 311.59 |
| Target hit | 2026-02-12 11:30:00 | 311.10 | 311.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 2 — BUY (started 2026-02-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:30:00 | 317.40 | 315.63 | 0.00 | ORB-long ORB[312.25,316.75] vol=3.8x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 316.27 | 316.58 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:10:00 | 322.15 | 324.19 | 0.00 | ORB-short ORB[322.40,326.85] vol=1.6x ATR=1.19 |
| Stop hit — per-position SL triggered | 2026-02-18 10:25:00 | 323.34 | 324.05 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:55:00 | 318.25 | 321.21 | 0.00 | ORB-short ORB[321.00,324.80] vol=1.7x ATR=1.11 |
| Stop hit — per-position SL triggered | 2026-02-27 11:00:00 | 319.36 | 321.04 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2026-03-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 10:45:00 | 306.20 | 308.42 | 0.00 | ORB-short ORB[308.35,310.10] vol=1.8x ATR=1.03 |
| Stop hit — per-position SL triggered | 2026-03-06 11:00:00 | 307.23 | 308.31 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-03-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:45:00 | 306.10 | 307.52 | 0.00 | ORB-short ORB[307.85,309.85] vol=1.6x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 11:05:00 | 304.94 | 306.90 | 0.00 | T1 1.5R @ 304.94 |
| Target hit | 2026-03-11 15:20:00 | 297.15 | 301.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2026-03-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:55:00 | 297.40 | 295.21 | 0.00 | ORB-long ORB[292.90,296.00] vol=1.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-03-12 10:15:00 | 296.22 | 295.79 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2026-03-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-16 11:15:00 | 296.25 | 293.75 | 0.00 | ORB-long ORB[292.00,294.85] vol=2.4x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-03-16 11:50:00 | 294.96 | 293.94 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 09:45:00 | 293.15 | 293.83 | 0.00 | ORB-short ORB[293.20,297.40] vol=3.4x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 294.43 | 293.84 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2026-03-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-19 09:40:00 | 293.25 | 295.15 | 0.00 | ORB-short ORB[294.20,297.25] vol=1.7x ATR=1.25 |
| Stop hit — per-position SL triggered | 2026-03-19 10:10:00 | 294.50 | 294.68 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 10:40:00 | 308.25 | 303.91 | 0.00 | ORB-long ORB[299.00,303.60] vol=2.5x ATR=1.29 |
| Stop hit — per-position SL triggered | 2026-03-25 10:55:00 | 306.96 | 304.45 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-04-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-09 09:30:00 | 316.95 | 319.53 | 0.00 | ORB-short ORB[318.10,322.15] vol=1.6x ATR=1.30 |
| Stop hit — per-position SL triggered | 2026-04-09 09:35:00 | 318.25 | 319.06 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2026-04-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:40:00 | 322.05 | 321.01 | 0.00 | ORB-long ORB[318.70,321.80] vol=2.4x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:50:00 | 323.82 | 322.44 | 0.00 | T1 1.5R @ 323.82 |
| Target hit | 2026-04-10 11:00:00 | 324.40 | 324.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 14 — SELL (started 2026-04-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 11:05:00 | 318.55 | 321.57 | 0.00 | ORB-short ORB[320.25,324.40] vol=1.6x ATR=0.92 |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 319.47 | 321.37 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2026-04-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:40:00 | 313.40 | 315.97 | 0.00 | ORB-short ORB[314.90,318.15] vol=2.5x ATR=1.03 |
| Stop hit — per-position SL triggered | 2026-04-17 09:55:00 | 314.43 | 315.55 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2026-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:35:00 | 322.35 | 320.36 | 0.00 | ORB-long ORB[317.85,320.40] vol=1.7x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 321.22 | 322.08 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2026-04-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 10:20:00 | 311.25 | 314.08 | 0.00 | ORB-short ORB[312.80,317.25] vol=2.0x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:30:00 | 309.90 | 313.70 | 0.00 | T1 1.5R @ 309.90 |
| Stop hit — per-position SL triggered | 2026-04-24 11:10:00 | 311.25 | 312.50 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2026-04-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:05:00 | 336.60 | 332.03 | 0.00 | ORB-long ORB[321.80,326.80] vol=3.1x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 10:55:00 | 339.74 | 334.65 | 0.00 | T1 1.5R @ 339.74 |
| Target hit | 2026-04-29 15:20:00 | 341.10 | 339.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2026-05-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 09:40:00 | 343.65 | 341.51 | 0.00 | ORB-long ORB[339.00,342.35] vol=2.5x ATR=1.37 |
| Stop hit — per-position SL triggered | 2026-05-04 10:05:00 | 342.28 | 341.99 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2026-05-05 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:50:00 | 331.90 | 328.65 | 0.00 | ORB-long ORB[326.15,330.90] vol=1.6x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-05 09:55:00 | 334.20 | 330.91 | 0.00 | T1 1.5R @ 334.20 |
| Stop hit — per-position SL triggered | 2026-05-05 10:25:00 | 331.90 | 331.37 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-12 10:15:00 | 310.10 | 2026-02-12 10:20:00 | 311.59 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-12 10:15:00 | 310.10 | 2026-02-12 11:30:00 | 311.10 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-17 09:30:00 | 317.40 | 2026-02-17 10:15:00 | 316.27 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-18 10:10:00 | 322.15 | 2026-02-18 10:25:00 | 323.34 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2026-02-27 10:55:00 | 318.25 | 2026-02-27 11:00:00 | 319.36 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-03-06 10:45:00 | 306.20 | 2026-03-06 11:00:00 | 307.23 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-03-11 10:45:00 | 306.10 | 2026-03-11 11:05:00 | 304.94 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-03-11 10:45:00 | 306.10 | 2026-03-11 15:20:00 | 297.15 | TARGET_HIT | 0.50 | 2.92% |
| BUY | retest1 | 2026-03-12 09:55:00 | 297.40 | 2026-03-12 10:15:00 | 296.22 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-03-16 11:15:00 | 296.25 | 2026-03-16 11:50:00 | 294.96 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2026-03-17 09:45:00 | 293.15 | 2026-03-17 10:15:00 | 294.43 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2026-03-19 09:40:00 | 293.25 | 2026-03-19 10:10:00 | 294.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-03-25 10:40:00 | 308.25 | 2026-03-25 10:55:00 | 306.96 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2026-04-09 09:30:00 | 316.95 | 2026-04-09 09:35:00 | 318.25 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2026-04-10 09:40:00 | 322.05 | 2026-04-10 09:50:00 | 323.82 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-04-10 09:40:00 | 322.05 | 2026-04-10 11:00:00 | 324.40 | TARGET_HIT | 0.50 | 0.73% |
| SELL | retest1 | 2026-04-15 11:05:00 | 318.55 | 2026-04-15 11:15:00 | 319.47 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-04-17 09:40:00 | 313.40 | 2026-04-17 09:55:00 | 314.43 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2026-04-21 09:35:00 | 322.35 | 2026-04-21 10:15:00 | 321.22 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-04-24 10:20:00 | 311.25 | 2026-04-24 10:30:00 | 309.90 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2026-04-24 10:20:00 | 311.25 | 2026-04-24 11:10:00 | 311.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-29 10:05:00 | 336.60 | 2026-04-29 10:55:00 | 339.74 | PARTIAL | 0.50 | 0.93% |
| BUY | retest1 | 2026-04-29 10:05:00 | 336.60 | 2026-04-29 15:20:00 | 341.10 | TARGET_HIT | 0.50 | 1.34% |
| BUY | retest1 | 2026-05-04 09:40:00 | 343.65 | 2026-05-04 10:05:00 | 342.28 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-05-05 09:50:00 | 331.90 | 2026-05-05 09:55:00 | 334.20 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2026-05-05 09:50:00 | 331.90 | 2026-05-05 10:25:00 | 331.90 | STOP_HIT | 0.50 | 0.00% |
