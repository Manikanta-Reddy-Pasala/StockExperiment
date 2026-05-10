# Petronet LNG Ltd. (PETRONET)

## Backtest Summary

- **Window:** 2026-02-09 09:15:00 → 2026-05-08 15:25:00 (4425 bars)
- **Last close:** 282.50
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
| ENTRY1 | 14 |
| ENTRY2 | 0 |
| PARTIAL | 8 |
| TARGET_HIT | 5 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 9
- **Target hits / Stop hits / Partials:** 5 / 9 / 8
- **Avg / median % per leg:** 0.29% / 0.29%
- **Sum % (uncompounded):** 6.43%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 12 | 75.0% | 5 | 4 | 7 | 0.44% | 7.0% |
| BUY @ 2nd Alert (retest1) | 16 | 12 | 75.0% | 5 | 4 | 7 | 0.44% | 7.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.10% | -0.6% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 5 | 1 | -0.10% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 22 | 13 | 59.1% | 5 | 9 | 8 | 0.29% | 6.4% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-02-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:25:00 | 298.95 | 297.16 | 0.00 | ORB-long ORB[295.35,297.75] vol=4.2x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 11:05:00 | 300.93 | 299.01 | 0.00 | T1 1.5R @ 300.93 |
| Stop hit — per-position SL triggered | 2026-02-09 11:15:00 | 298.95 | 299.01 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2026-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-10 11:15:00 | 301.40 | 300.14 | 0.00 | ORB-long ORB[297.35,301.20] vol=2.4x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 11:20:00 | 302.27 | 300.31 | 0.00 | T1 1.5R @ 302.27 |
| Target hit | 2026-02-10 15:20:00 | 303.25 | 302.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-11 11:15:00 | 301.35 | 302.47 | 0.00 | ORB-short ORB[302.50,304.00] vol=1.6x ATR=0.54 |
| Stop hit — per-position SL triggered | 2026-02-11 11:30:00 | 301.89 | 302.43 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2026-02-18 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:45:00 | 300.10 | 299.00 | 0.00 | ORB-long ORB[296.60,299.65] vol=2.6x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 12:40:00 | 300.95 | 299.84 | 0.00 | T1 1.5R @ 300.95 |
| Target hit | 2026-02-18 15:20:00 | 303.90 | 303.40 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — BUY (started 2026-02-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-24 09:35:00 | 310.30 | 309.32 | 0.00 | ORB-long ORB[306.45,309.85] vol=3.7x ATR=0.73 |
| Stop hit — per-position SL triggered | 2026-02-24 09:40:00 | 309.57 | 309.31 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2026-02-26 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:55:00 | 316.20 | 315.17 | 0.00 | ORB-long ORB[313.50,315.70] vol=3.6x ATR=0.69 |
| Stop hit — per-position SL triggered | 2026-02-26 11:05:00 | 315.51 | 315.23 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2026-02-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-27 09:35:00 | 322.20 | 319.76 | 0.00 | ORB-long ORB[316.10,320.45] vol=3.1x ATR=1.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:20:00 | 323.74 | 321.31 | 0.00 | T1 1.5R @ 323.74 |
| Target hit | 2026-02-27 15:00:00 | 323.30 | 323.64 | 0.00 | Trail-exit close<VWAP |

### Cycle 8 — BUY (started 2026-03-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 09:40:00 | 292.10 | 290.61 | 0.00 | ORB-long ORB[288.15,291.85] vol=2.0x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:50:00 | 293.91 | 291.44 | 0.00 | T1 1.5R @ 293.91 |
| Target hit | 2026-03-12 11:45:00 | 295.00 | 295.37 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — SELL (started 2026-04-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-17 09:40:00 | 270.54 | 273.33 | 0.00 | ORB-short ORB[271.40,275.23] vol=1.5x ATR=0.96 |
| Stop hit — per-position SL triggered | 2026-04-17 09:55:00 | 271.50 | 272.97 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 11:15:00 | 279.33 | 277.93 | 0.00 | ORB-long ORB[275.88,278.11] vol=1.5x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 11:55:00 | 280.05 | 278.28 | 0.00 | T1 1.5R @ 280.05 |
| Stop hit — per-position SL triggered | 2026-04-22 13:00:00 | 279.33 | 278.62 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2026-04-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:45:00 | 273.81 | 275.08 | 0.00 | ORB-short ORB[274.17,276.90] vol=1.6x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 10:10:00 | 272.35 | 274.72 | 0.00 | T1 1.5R @ 272.35 |
| Stop hit — per-position SL triggered | 2026-04-24 11:20:00 | 273.81 | 273.99 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2026-04-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:20:00 | 276.87 | 275.21 | 0.00 | ORB-long ORB[273.36,276.44] vol=4.0x ATR=0.90 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 10:50:00 | 278.22 | 276.17 | 0.00 | T1 1.5R @ 278.22 |
| Target hit | 2026-04-27 15:20:00 | 280.14 | 278.86 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2026-04-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-30 10:10:00 | 274.46 | 276.84 | 0.00 | ORB-short ORB[276.70,278.45] vol=1.8x ATR=0.74 |
| Stop hit — per-position SL triggered | 2026-04-30 11:05:00 | 275.20 | 276.08 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2026-05-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-07 09:55:00 | 278.85 | 280.32 | 0.00 | ORB-short ORB[280.50,284.05] vol=2.1x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-05-07 10:15:00 | 279.75 | 279.89 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2026-02-09 10:25:00 | 298.95 | 2026-02-09 11:05:00 | 300.93 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2026-02-09 10:25:00 | 298.95 | 2026-02-09 11:15:00 | 298.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-10 11:15:00 | 301.40 | 2026-02-10 11:20:00 | 302.27 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2026-02-10 11:15:00 | 301.40 | 2026-02-10 15:20:00 | 303.25 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2026-02-11 11:15:00 | 301.35 | 2026-02-11 11:30:00 | 301.89 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2026-02-18 10:45:00 | 300.10 | 2026-02-18 12:40:00 | 300.95 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2026-02-18 10:45:00 | 300.10 | 2026-02-18 15:20:00 | 303.90 | TARGET_HIT | 0.50 | 1.27% |
| BUY | retest1 | 2026-02-24 09:35:00 | 310.30 | 2026-02-24 09:40:00 | 309.57 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2026-02-26 10:55:00 | 316.20 | 2026-02-26 11:05:00 | 315.51 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2026-02-27 09:35:00 | 322.20 | 2026-02-27 10:20:00 | 323.74 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2026-02-27 09:35:00 | 322.20 | 2026-02-27 15:00:00 | 323.30 | TARGET_HIT | 0.50 | 0.34% |
| BUY | retest1 | 2026-03-12 09:40:00 | 292.10 | 2026-03-12 09:50:00 | 293.91 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-03-12 09:40:00 | 292.10 | 2026-03-12 11:45:00 | 295.00 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2026-04-17 09:40:00 | 270.54 | 2026-04-17 09:55:00 | 271.50 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-04-22 11:15:00 | 279.33 | 2026-04-22 11:55:00 | 280.05 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2026-04-22 11:15:00 | 279.33 | 2026-04-22 13:00:00 | 279.33 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-24 09:45:00 | 273.81 | 2026-04-24 10:10:00 | 272.35 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2026-04-24 09:45:00 | 273.81 | 2026-04-24 11:20:00 | 273.81 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-27 10:20:00 | 276.87 | 2026-04-27 10:50:00 | 278.22 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-04-27 10:20:00 | 276.87 | 2026-04-27 15:20:00 | 280.14 | TARGET_HIT | 0.50 | 1.18% |
| SELL | retest1 | 2026-04-30 10:10:00 | 274.46 | 2026-04-30 11:05:00 | 275.20 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-05-07 09:55:00 | 278.85 | 2026-05-07 10:15:00 | 279.75 | STOP_HIT | 1.00 | -0.32% |
