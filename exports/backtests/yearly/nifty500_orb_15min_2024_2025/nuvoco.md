# Nuvoco Vistas Corporation Ltd. (NUVOCO)

## Backtest Summary

- **Window:** 2025-02-05 09:15:00 → 2026-05-08 15:25:00 (21463 bars)
- **Last close:** 328.90
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
| TARGET_HIT | 2 |
| STOP_HIT | 11 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 8 / 11
- **Target hits / Stop hits / Partials:** 2 / 11 / 6
- **Avg / median % per leg:** 0.21% / 0.00%
- **Sum % (uncompounded):** 3.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.20% | 2.0% |
| BUY @ 2nd Alert (retest1) | 10 | 4 | 40.0% | 1 | 6 | 3 | 0.20% | 2.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.22% | 2.0% |
| SELL @ 2nd Alert (retest1) | 9 | 4 | 44.4% | 1 | 5 | 3 | 0.22% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 19 | 8 | 42.1% | 2 | 11 | 6 | 0.21% | 4.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-02-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 10:50:00 | 361.45 | 357.36 | 0.00 | ORB-long ORB[352.50,354.90] vol=9.2x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-06 10:55:00 | 363.32 | 359.52 | 0.00 | T1 1.5R @ 363.32 |
| Stop hit — per-position SL triggered | 2025-02-06 11:15:00 | 361.45 | 362.24 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-02-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:50:00 | 351.50 | 353.69 | 0.00 | ORB-short ORB[353.90,358.90] vol=1.9x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 12:00:00 | 349.25 | 353.09 | 0.00 | T1 1.5R @ 349.25 |
| Target hit | 2025-02-10 15:20:00 | 347.35 | 351.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 3 — SELL (started 2025-02-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-18 11:10:00 | 312.60 | 314.93 | 0.00 | ORB-short ORB[313.55,318.20] vol=3.3x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 11:25:00 | 311.09 | 314.16 | 0.00 | T1 1.5R @ 311.09 |
| Stop hit — per-position SL triggered | 2025-02-18 12:00:00 | 312.60 | 313.78 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-02-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 11:05:00 | 323.30 | 323.03 | 0.00 | ORB-long ORB[318.30,323.00] vol=1.6x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-02-20 11:25:00 | 322.49 | 323.03 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-02-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:55:00 | 320.25 | 322.29 | 0.00 | ORB-short ORB[320.85,324.25] vol=2.0x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-02-21 10:00:00 | 321.30 | 322.16 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-02-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 11:05:00 | 317.40 | 316.58 | 0.00 | ORB-long ORB[314.40,317.15] vol=1.7x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-02-25 11:30:00 | 316.46 | 316.85 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-03-07 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 10:25:00 | 319.20 | 317.04 | 0.00 | ORB-long ORB[315.30,318.05] vol=2.0x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-07 10:40:00 | 320.69 | 317.54 | 0.00 | T1 1.5R @ 320.69 |
| Stop hit — per-position SL triggered | 2025-03-07 12:10:00 | 319.20 | 318.76 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2025-03-12 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:40:00 | 298.20 | 301.53 | 0.00 | ORB-short ORB[301.35,305.45] vol=3.5x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-03-12 11:15:00 | 299.27 | 300.76 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-03-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-26 10:00:00 | 323.40 | 322.12 | 0.00 | ORB-long ORB[318.00,322.80] vol=15.3x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-03-26 10:10:00 | 321.72 | 322.12 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-03-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-27 09:30:00 | 311.25 | 313.16 | 0.00 | ORB-short ORB[312.40,316.15] vol=4.6x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 10:00:00 | 309.03 | 311.95 | 0.00 | T1 1.5R @ 309.03 |
| Stop hit — per-position SL triggered | 2025-03-27 10:40:00 | 311.25 | 311.24 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-04-09 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 10:25:00 | 303.60 | 305.10 | 0.00 | ORB-short ORB[303.70,307.40] vol=1.9x ATR=1.12 |
| Stop hit — per-position SL triggered | 2025-04-09 10:30:00 | 304.72 | 305.07 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-04-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:00:00 | 326.60 | 325.04 | 0.00 | ORB-long ORB[321.70,325.95] vol=7.0x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-04-21 10:10:00 | 325.73 | 325.14 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-04-24 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:30:00 | 334.15 | 333.42 | 0.00 | ORB-long ORB[327.65,332.05] vol=16.7x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 10:35:00 | 336.13 | 333.46 | 0.00 | T1 1.5R @ 336.13 |
| Target hit | 2025-04-24 15:20:00 | 340.00 | 335.51 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-02-06 10:50:00 | 361.45 | 2025-02-06 10:55:00 | 363.32 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-02-06 10:50:00 | 361.45 | 2025-02-06 11:15:00 | 361.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-02-10 09:50:00 | 351.50 | 2025-02-10 12:00:00 | 349.25 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-02-10 09:50:00 | 351.50 | 2025-02-10 15:20:00 | 347.35 | TARGET_HIT | 0.50 | 1.18% |
| SELL | retest1 | 2025-02-18 11:10:00 | 312.60 | 2025-02-18 11:25:00 | 311.09 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-02-18 11:10:00 | 312.60 | 2025-02-18 12:00:00 | 312.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-20 11:05:00 | 323.30 | 2025-02-20 11:25:00 | 322.49 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-02-21 09:55:00 | 320.25 | 2025-02-21 10:00:00 | 321.30 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-02-25 11:05:00 | 317.40 | 2025-02-25 11:30:00 | 316.46 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-07 10:25:00 | 319.20 | 2025-03-07 10:40:00 | 320.69 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-03-07 10:25:00 | 319.20 | 2025-03-07 12:10:00 | 319.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-12 10:40:00 | 298.20 | 2025-03-12 11:15:00 | 299.27 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-03-26 10:00:00 | 323.40 | 2025-03-26 10:10:00 | 321.72 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2025-03-27 09:30:00 | 311.25 | 2025-03-27 10:00:00 | 309.03 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2025-03-27 09:30:00 | 311.25 | 2025-03-27 10:40:00 | 311.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-04-09 10:25:00 | 303.60 | 2025-04-09 10:30:00 | 304.72 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-04-21 10:00:00 | 326.60 | 2025-04-21 10:10:00 | 325.73 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-04-24 10:30:00 | 334.15 | 2025-04-24 10:35:00 | 336.13 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-04-24 10:30:00 | 334.15 | 2025-04-24 15:20:00 | 340.00 | TARGET_HIT | 0.50 | 1.75% |
