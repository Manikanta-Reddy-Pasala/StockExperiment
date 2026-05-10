# Exide Industries Ltd. (EXIDEIND)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 361.75
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
| ENTRY1 | 18 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 6 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 28 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 12
- **Target hits / Stop hits / Partials:** 6 / 12 / 10
- **Avg / median % per leg:** 0.23% / 0.32%
- **Sum % (uncompounded):** 6.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 11 | 73.3% | 5 | 4 | 6 | 0.28% | 4.1% |
| BUY @ 2nd Alert (retest1) | 15 | 11 | 73.3% | 5 | 4 | 6 | 0.28% | 4.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.18% | 2.4% |
| SELL @ 2nd Alert (retest1) | 13 | 5 | 38.5% | 1 | 8 | 4 | 0.18% | 2.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 28 | 16 | 57.1% | 6 | 12 | 10 | 0.23% | 6.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:35:00 | 337.50 | 335.65 | 0.00 | ORB-long ORB[333.40,335.90] vol=1.6x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 13:00:00 | 339.36 | 336.99 | 0.00 | T1 1.5R @ 339.36 |
| Target hit | 2026-02-09 15:20:00 | 340.90 | 338.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 10:15:00 | 338.35 | 336.58 | 0.00 | ORB-long ORB[332.20,336.70] vol=1.9x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 13:10:00 | 340.23 | 337.71 | 0.00 | T1 1.5R @ 340.23 |
| Target hit | 2026-02-16 15:20:00 | 340.35 | 338.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 11:00:00 | 340.15 | 340.96 | 0.00 | ORB-short ORB[340.30,343.50] vol=1.8x ATR=0.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:30:00 | 339.06 | 340.68 | 0.00 | T1 1.5R @ 339.06 |
| Target hit | 2026-02-19 15:20:00 | 333.00 | 336.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — BUY (started 2026-02-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:50:00 | 335.30 | 334.27 | 0.00 | ORB-long ORB[332.40,334.85] vol=1.5x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 12:20:00 | 336.63 | 334.95 | 0.00 | T1 1.5R @ 336.63 |
| Target hit | 2026-02-20 15:15:00 | 336.20 | 336.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 335.15 | 336.13 | 0.00 | ORB-short ORB[335.30,338.40] vol=1.9x ATR=0.79 |
| Stop hit — per-position SL triggered | 2026-02-24 09:45:00 | 335.94 | 335.93 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:15:00 | 339.50 | 338.67 | 0.00 | ORB-long ORB[336.60,339.35] vol=1.7x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 10:35:00 | 340.75 | 338.97 | 0.00 | T1 1.5R @ 340.75 |
| Stop hit — per-position SL triggered | 2026-02-25 12:45:00 | 339.50 | 340.48 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 11:10:00 | 336.30 | 336.42 | 0.00 | ORB-short ORB[336.50,339.60] vol=1.8x ATR=0.71 |
| Stop hit — per-position SL triggered | 2026-02-27 13:45:00 | 337.01 | 336.36 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-11 10:40:00 | 314.00 | 317.94 | 0.00 | ORB-short ORB[316.40,320.65] vol=2.3x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-11 13:55:00 | 312.43 | 315.48 | 0.00 | T1 1.5R @ 312.43 |
| Stop hit — per-position SL triggered | 2026-03-11 14:25:00 | 314.00 | 315.24 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 09:50:00 | 303.80 | 305.59 | 0.00 | ORB-short ORB[305.30,308.80] vol=7.4x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 302.03 | 304.80 | 0.00 | T1 1.5R @ 302.03 |
| Stop hit — per-position SL triggered | 2026-03-13 10:50:00 | 303.80 | 303.69 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 09:30:00 | 299.00 | 297.39 | 0.00 | ORB-long ORB[295.25,297.85] vol=3.0x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 09:35:00 | 300.89 | 298.40 | 0.00 | T1 1.5R @ 300.89 |
| Target hit | 2026-03-17 11:25:00 | 299.45 | 300.83 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2026-03-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-18 09:30:00 | 307.15 | 305.69 | 0.00 | ORB-long ORB[302.95,306.20] vol=2.5x ATR=1.02 |
| Stop hit — per-position SL triggered | 2026-03-18 09:45:00 | 306.13 | 305.98 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2026-03-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:55:00 | 291.80 | 294.66 | 0.00 | ORB-short ORB[294.55,297.65] vol=1.8x ATR=1.10 |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 292.90 | 294.54 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2026-03-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-30 10:55:00 | 292.05 | 295.12 | 0.00 | ORB-short ORB[294.60,298.10] vol=1.7x ATR=1.13 |
| Stop hit — per-position SL triggered | 2026-03-30 11:35:00 | 293.18 | 294.71 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-04-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-16 11:10:00 | 330.45 | 331.40 | 0.00 | ORB-short ORB[331.00,335.80] vol=2.6x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-16 11:25:00 | 329.20 | 331.26 | 0.00 | T1 1.5R @ 329.20 |
| Stop hit — per-position SL triggered | 2026-04-16 14:45:00 | 330.45 | 330.14 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2026-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-21 09:40:00 | 333.00 | 331.12 | 0.00 | ORB-long ORB[328.60,331.60] vol=2.1x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-21 10:05:00 | 334.36 | 332.41 | 0.00 | T1 1.5R @ 334.36 |
| Target hit | 2026-04-21 12:45:00 | 334.05 | 334.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — BUY (started 2026-04-29 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-29 10:30:00 | 362.95 | 359.74 | 0.00 | ORB-long ORB[356.90,361.40] vol=3.1x ATR=1.33 |
| Stop hit — per-position SL triggered | 2026-04-29 10:40:00 | 361.62 | 359.99 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2026-05-04 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-04 10:30:00 | 370.75 | 367.33 | 0.00 | ORB-long ORB[363.55,368.70] vol=1.7x ATR=1.54 |
| Stop hit — per-position SL triggered | 2026-05-04 10:35:00 | 369.21 | 367.47 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2026-05-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-06 11:00:00 | 359.40 | 362.81 | 0.00 | ORB-short ORB[361.00,366.20] vol=2.0x ATR=1.03 |
| Stop hit — per-position SL triggered | 2026-05-06 11:05:00 | 360.43 | 362.73 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:35:00 | 337.50 | 2026-02-09 13:00:00 | 339.36 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2026-02-09 10:35:00 | 337.50 | 2026-02-09 15:20:00 | 340.90 | TARGET_HIT | 0.50 | 1.01% |
| BUY | retest1 | 2026-02-16 10:15:00 | 338.35 | 2026-02-16 13:10:00 | 340.23 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2026-02-16 10:15:00 | 338.35 | 2026-02-16 15:20:00 | 340.35 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2026-02-19 11:00:00 | 340.15 | 2026-02-19 11:30:00 | 339.06 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2026-02-19 11:00:00 | 340.15 | 2026-02-19 15:20:00 | 333.00 | TARGET_HIT | 0.50 | 2.10% |
| BUY | retest1 | 2026-02-20 10:50:00 | 335.30 | 2026-02-20 12:20:00 | 336.63 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2026-02-20 10:50:00 | 335.30 | 2026-02-20 15:15:00 | 336.20 | TARGET_HIT | 0.50 | 0.27% |
| SELL | retest1 | 2026-02-24 09:30:00 | 335.15 | 2026-02-24 09:45:00 | 335.94 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-25 10:15:00 | 339.50 | 2026-02-25 10:35:00 | 340.75 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2026-02-25 10:15:00 | 339.50 | 2026-02-25 12:45:00 | 339.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 11:10:00 | 336.30 | 2026-02-27 13:45:00 | 337.01 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2026-03-11 10:40:00 | 314.00 | 2026-03-11 13:55:00 | 312.43 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2026-03-11 10:40:00 | 314.00 | 2026-03-11 14:25:00 | 314.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-13 09:50:00 | 303.80 | 2026-03-13 10:15:00 | 302.03 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2026-03-13 09:50:00 | 303.80 | 2026-03-13 10:50:00 | 303.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 09:30:00 | 299.00 | 2026-03-17 09:35:00 | 300.89 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-03-17 09:30:00 | 299.00 | 2026-03-17 11:25:00 | 299.45 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2026-03-18 09:30:00 | 307.15 | 2026-03-18 09:45:00 | 306.13 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-03-24 10:55:00 | 291.80 | 2026-03-24 11:15:00 | 292.90 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2026-03-30 10:55:00 | 292.05 | 2026-03-30 11:35:00 | 293.18 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-04-16 11:10:00 | 330.45 | 2026-04-16 11:25:00 | 329.20 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-04-16 11:10:00 | 330.45 | 2026-04-16 14:45:00 | 330.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-21 09:40:00 | 333.00 | 2026-04-21 10:05:00 | 334.36 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2026-04-21 09:40:00 | 333.00 | 2026-04-21 12:45:00 | 334.05 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2026-04-29 10:30:00 | 362.95 | 2026-04-29 10:40:00 | 361.62 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2026-05-04 10:30:00 | 370.75 | 2026-05-04 10:35:00 | 369.21 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2026-05-06 11:00:00 | 359.40 | 2026-05-06 11:05:00 | 360.43 | STOP_HIT | 1.00 | -0.29% |
