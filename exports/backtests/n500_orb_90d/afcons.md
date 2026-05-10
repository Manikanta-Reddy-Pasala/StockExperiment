# Afcons Infrastructure Ltd. (AFCONS)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 340.40
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
| ENTRY1 | 13 |
| ENTRY2 | 0 |
| PARTIAL | 6 |
| TARGET_HIT | 4 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 9
- **Target hits / Stop hits / Partials:** 4 / 9 / 6
- **Avg / median % per leg:** 0.79% / 0.24%
- **Sum % (uncompounded):** 15.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 8 | 80.0% | 3 | 2 | 5 | 1.67% | 16.7% |
| BUY @ 2nd Alert (retest1) | 10 | 8 | 80.0% | 3 | 2 | 5 | 1.67% | 16.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 2 | 22.2% | 1 | 7 | 1 | -0.18% | -1.6% |
| SELL @ 2nd Alert (retest1) | 9 | 2 | 22.2% | 1 | 7 | 1 | -0.18% | -1.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 10 | 52.6% | 4 | 9 | 6 | 0.79% | 15.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:30:00 | 337.05 | 334.48 | 0.00 | ORB-long ORB[332.20,335.95] vol=1.6x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 14:35:00 | 339.63 | 337.43 | 0.00 | T1 1.5R @ 339.63 |
| Target hit | 2026-02-09 15:20:00 | 341.75 | 338.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — SELL (started 2026-02-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:35:00 | 329.75 | 331.82 | 0.00 | ORB-short ORB[332.10,333.80] vol=4.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 330.93 | 331.74 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2026-02-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 10:10:00 | 323.15 | 323.98 | 0.00 | ORB-short ORB[323.80,325.75] vol=1.7x ATR=0.58 |
| Stop hit — per-position SL triggered | 2026-02-18 10:30:00 | 323.73 | 323.96 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2026-02-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-19 10:40:00 | 321.35 | 322.87 | 0.00 | ORB-short ORB[322.60,325.20] vol=2.2x ATR=0.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-19 11:20:00 | 320.57 | 322.41 | 0.00 | T1 1.5R @ 320.57 |
| Target hit | 2026-02-19 15:20:00 | 320.05 | 321.22 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2026-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-20 09:30:00 | 316.40 | 317.58 | 0.00 | ORB-short ORB[317.00,320.50] vol=3.1x ATR=0.75 |
| Stop hit — per-position SL triggered | 2026-02-20 09:35:00 | 317.15 | 317.50 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2026-02-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-23 10:00:00 | 315.95 | 317.82 | 0.00 | ORB-short ORB[317.30,321.85] vol=1.8x ATR=0.99 |
| Stop hit — per-position SL triggered | 2026-02-23 10:05:00 | 316.94 | 317.72 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2026-02-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:50:00 | 289.45 | 292.00 | 0.00 | ORB-short ORB[292.50,296.85] vol=1.6x ATR=0.82 |
| Stop hit — per-position SL triggered | 2026-02-27 10:55:00 | 290.27 | 291.94 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2026-03-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-09 10:00:00 | 272.00 | 274.20 | 0.00 | ORB-short ORB[273.15,276.95] vol=1.5x ATR=1.45 |
| Stop hit — per-position SL triggered | 2026-03-09 11:10:00 | 273.45 | 273.50 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2026-03-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-10 09:30:00 | 278.55 | 279.77 | 0.00 | ORB-short ORB[279.15,282.00] vol=1.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2026-03-10 09:50:00 | 279.51 | 279.15 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-03-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 09:30:00 | 279.55 | 276.92 | 0.00 | ORB-long ORB[274.50,278.40] vol=1.5x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 10:15:00 | 281.44 | 279.66 | 0.00 | T1 1.5R @ 281.44 |
| Stop hit — per-position SL triggered | 2026-03-13 11:20:00 | 279.55 | 279.61 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2026-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 10:15:00 | 279.65 | 278.21 | 0.00 | ORB-long ORB[276.00,279.15] vol=5.6x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 10:20:00 | 281.13 | 278.54 | 0.00 | T1 1.5R @ 281.13 |
| Stop hit — per-position SL triggered | 2026-03-17 11:25:00 | 279.65 | 279.34 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 314.90 | 314.35 | 0.00 | ORB-long ORB[310.85,314.75] vol=4.6x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-10 09:50:00 | 316.58 | 314.59 | 0.00 | T1 1.5R @ 316.58 |
| Target hit | 2026-04-10 15:20:00 | 324.90 | 322.51 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 315.00 | 313.23 | 0.00 | ORB-long ORB[310.00,314.40] vol=2.1x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 09:50:00 | 317.43 | 313.82 | 0.00 | T1 1.5R @ 317.43 |
| Target hit | 2026-04-15 12:25:00 | 342.90 | 348.75 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:30:00 | 337.05 | 2026-02-09 14:35:00 | 339.63 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2026-02-09 10:30:00 | 337.05 | 2026-02-09 15:20:00 | 341.75 | TARGET_HIT | 0.50 | 1.39% |
| SELL | retest1 | 2026-02-13 09:35:00 | 329.75 | 2026-02-13 09:40:00 | 330.93 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-18 10:10:00 | 323.15 | 2026-02-18 10:30:00 | 323.73 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-02-19 10:40:00 | 321.35 | 2026-02-19 11:20:00 | 320.57 | PARTIAL | 0.50 | 0.24% |
| SELL | retest1 | 2026-02-19 10:40:00 | 321.35 | 2026-02-19 15:20:00 | 320.05 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-20 09:30:00 | 316.40 | 2026-02-20 09:35:00 | 317.15 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2026-02-23 10:00:00 | 315.95 | 2026-02-23 10:05:00 | 316.94 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-02-27 10:50:00 | 289.45 | 2026-02-27 10:55:00 | 290.27 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-03-09 10:00:00 | 272.00 | 2026-03-09 11:10:00 | 273.45 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest1 | 2026-03-10 09:30:00 | 278.55 | 2026-03-10 09:50:00 | 279.51 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-13 09:30:00 | 279.55 | 2026-03-13 10:15:00 | 281.44 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-03-13 09:30:00 | 279.55 | 2026-03-13 11:20:00 | 279.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-03-17 10:15:00 | 279.65 | 2026-03-17 10:20:00 | 281.13 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-03-17 10:15:00 | 279.65 | 2026-03-17 11:25:00 | 279.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-10 09:30:00 | 314.90 | 2026-04-10 09:50:00 | 316.58 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2026-04-10 09:30:00 | 314.90 | 2026-04-10 15:20:00 | 324.90 | TARGET_HIT | 0.50 | 3.18% |
| BUY | retest1 | 2026-04-15 09:40:00 | 315.00 | 2026-04-15 09:50:00 | 317.43 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2026-04-15 09:40:00 | 315.00 | 2026-04-15 12:25:00 | 342.90 | TARGET_HIT | 0.50 | 8.86% |
