# Authum Investment & Infrastructure Ltd. (AIIL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (35296 bars)
- **Last close:** 494.80
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
| ENTRY1 | 27 |
| ENTRY2 | 0 |
| PARTIAL | 14 |
| TARGET_HIT | 9 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 41 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 18
- **Target hits / Stop hits / Partials:** 9 / 18 / 14
- **Avg / median % per leg:** 0.51% / 0.47%
- **Sum % (uncompounded):** 20.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 17 | 12 | 70.6% | 5 | 5 | 7 | 0.92% | 15.6% |
| BUY @ 2nd Alert (retest1) | 17 | 12 | 70.6% | 5 | 5 | 7 | 0.92% | 15.6% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 24 | 11 | 45.8% | 4 | 13 | 7 | 0.22% | 5.4% |
| SELL @ 2nd Alert (retest1) | 24 | 11 | 45.8% | 4 | 13 | 7 | 0.22% | 5.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 41 | 23 | 56.1% | 9 | 18 | 14 | 0.51% | 21.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:30:00 | 167.15 | 166.25 | 0.00 | ORB-long ORB[164.20,166.56] vol=6.6x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 09:45:00 | 168.80 | 166.71 | 0.00 | T1 1.5R @ 168.80 |
| Stop hit — per-position SL triggered | 2024-05-15 09:55:00 | 167.15 | 167.01 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-06-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:20:00 | 229.20 | 231.39 | 0.00 | ORB-short ORB[229.80,233.00] vol=1.6x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:35:00 | 227.23 | 231.03 | 0.00 | T1 1.5R @ 227.23 |
| Target hit | 2024-06-27 15:15:00 | 228.38 | 227.40 | 0.00 | Trail-exit close>VWAP |

### Cycle 3 — SELL (started 2024-07-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 10:00:00 | 229.00 | 230.62 | 0.00 | ORB-short ORB[230.83,233.20] vol=3.2x ATR=1.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 10:10:00 | 227.48 | 230.33 | 0.00 | T1 1.5R @ 227.48 |
| Stop hit — per-position SL triggered | 2024-07-03 10:25:00 | 229.00 | 230.05 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-07-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:25:00 | 228.66 | 229.51 | 0.00 | ORB-short ORB[228.70,231.20] vol=1.9x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-07-11 12:45:00 | 229.70 | 229.31 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-07-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-15 09:30:00 | 223.61 | 224.24 | 0.00 | ORB-short ORB[223.64,226.86] vol=3.9x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-07-15 09:35:00 | 224.65 | 223.85 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-09-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:00:00 | 340.00 | 341.62 | 0.00 | ORB-short ORB[340.01,344.80] vol=1.9x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 10:25:00 | 337.68 | 341.04 | 0.00 | T1 1.5R @ 337.68 |
| Stop hit — per-position SL triggered | 2024-09-25 11:20:00 | 340.00 | 340.85 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-10-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 09:35:00 | 369.74 | 368.79 | 0.00 | ORB-long ORB[365.71,369.60] vol=3.5x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-10-16 13:45:00 | 368.20 | 369.28 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:05:00 | 367.08 | 367.92 | 0.00 | ORB-short ORB[368.01,369.76] vol=2.5x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 11:25:00 | 365.34 | 367.81 | 0.00 | T1 1.5R @ 365.34 |
| Stop hit — per-position SL triggered | 2024-10-17 12:30:00 | 367.08 | 367.57 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-31 11:15:00 | 325.36 | 328.82 | 0.00 | ORB-short ORB[328.21,331.01] vol=2.9x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-10-31 15:00:00 | 327.23 | 326.75 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-11-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:30:00 | 340.20 | 337.82 | 0.00 | ORB-long ORB[334.44,339.00] vol=2.9x ATR=1.74 |
| Stop hit — per-position SL triggered | 2024-11-06 10:20:00 | 338.46 | 338.66 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-12-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:55:00 | 326.41 | 324.05 | 0.00 | ORB-long ORB[320.60,324.12] vol=1.7x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 10:00:00 | 328.64 | 324.93 | 0.00 | T1 1.5R @ 328.64 |
| Target hit | 2024-12-06 14:15:00 | 331.03 | 331.27 | 0.00 | Trail-exit close<VWAP |

### Cycle 12 — BUY (started 2024-12-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 10:10:00 | 337.85 | 333.02 | 0.00 | ORB-long ORB[328.62,333.59] vol=2.9x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 10:30:00 | 340.78 | 334.94 | 0.00 | T1 1.5R @ 340.78 |
| Target hit | 2024-12-09 15:20:00 | 341.99 | 340.21 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 363.80 | 365.59 | 0.00 | ORB-short ORB[363.90,368.81] vol=1.7x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 11:20:00 | 361.57 | 363.51 | 0.00 | T1 1.5R @ 361.57 |
| Target hit | 2024-12-17 15:20:00 | 355.62 | 360.45 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2024-12-18 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-18 11:10:00 | 357.96 | 355.34 | 0.00 | ORB-long ORB[353.31,355.14] vol=1.9x ATR=1.33 |
| Stop hit — per-position SL triggered | 2024-12-18 11:40:00 | 356.63 | 355.51 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-12-26 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 10:10:00 | 332.80 | 336.29 | 0.00 | ORB-short ORB[336.27,339.29] vol=2.8x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-12-26 10:15:00 | 333.88 | 336.16 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-12-31 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-31 09:50:00 | 337.21 | 338.35 | 0.00 | ORB-short ORB[338.42,341.90] vol=4.8x ATR=1.50 |
| Stop hit — per-position SL triggered | 2024-12-31 10:00:00 | 338.71 | 338.42 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-01-20 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 09:55:00 | 356.30 | 353.55 | 0.00 | ORB-long ORB[351.00,353.62] vol=1.5x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 10:05:00 | 359.34 | 354.91 | 0.00 | T1 1.5R @ 359.34 |
| Target hit | 2025-01-20 15:20:00 | 376.84 | 371.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2025-01-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:45:00 | 367.60 | 361.56 | 0.00 | ORB-long ORB[353.00,357.75] vol=1.6x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:15:00 | 371.86 | 366.00 | 0.00 | T1 1.5R @ 371.86 |
| Target hit | 2025-01-23 11:40:00 | 369.07 | 369.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — SELL (started 2025-01-30 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-30 10:35:00 | 339.62 | 343.06 | 0.00 | ORB-short ORB[340.80,344.92] vol=3.8x ATR=1.94 |
| Stop hit — per-position SL triggered | 2025-01-30 10:40:00 | 341.56 | 342.83 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-02-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:00:00 | 339.62 | 340.10 | 0.00 | ORB-short ORB[340.40,343.79] vol=4.5x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-02-01 11:10:00 | 340.75 | 340.10 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-02-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-05 10:35:00 | 347.47 | 349.39 | 0.00 | ORB-short ORB[350.00,354.00] vol=3.1x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-02-05 10:45:00 | 349.13 | 349.43 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-06 09:30:00 | 351.15 | 352.84 | 0.00 | ORB-short ORB[352.87,355.73] vol=4.2x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-02-06 09:35:00 | 352.76 | 352.81 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-03-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 10:00:00 | 289.79 | 292.13 | 0.00 | ORB-short ORB[289.99,293.38] vol=2.0x ATR=1.57 |
| Stop hit — per-position SL triggered | 2025-03-12 10:40:00 | 291.36 | 291.01 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 09:35:00 | 289.45 | 291.55 | 0.00 | ORB-short ORB[291.00,295.19] vol=2.5x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 10:10:00 | 286.52 | 290.54 | 0.00 | T1 1.5R @ 286.52 |
| Target hit | 2025-03-17 15:20:00 | 285.01 | 286.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2025-04-21 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:25:00 | 360.98 | 357.16 | 0.00 | ORB-long ORB[352.24,357.60] vol=1.5x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-21 10:35:00 | 363.39 | 357.75 | 0.00 | T1 1.5R @ 363.39 |
| Target hit | 2025-04-21 15:20:00 | 369.00 | 365.75 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 350.82 | 354.97 | 0.00 | ORB-short ORB[355.12,359.74] vol=3.0x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:45:00 | 347.77 | 353.95 | 0.00 | T1 1.5R @ 347.77 |
| Target hit | 2025-04-25 12:55:00 | 348.40 | 347.89 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — BUY (started 2025-04-29 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-29 10:35:00 | 359.14 | 356.87 | 0.00 | ORB-long ORB[356.20,359.08] vol=1.6x ATR=1.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 10:50:00 | 361.60 | 358.05 | 0.00 | T1 1.5R @ 361.60 |
| Stop hit — per-position SL triggered | 2025-04-29 12:20:00 | 359.14 | 359.74 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:30:00 | 167.15 | 2024-05-15 09:45:00 | 168.80 | PARTIAL | 0.50 | 0.99% |
| BUY | retest1 | 2024-05-15 09:30:00 | 167.15 | 2024-05-15 09:55:00 | 167.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-27 10:20:00 | 229.20 | 2024-06-27 10:35:00 | 227.23 | PARTIAL | 0.50 | 0.86% |
| SELL | retest1 | 2024-06-27 10:20:00 | 229.20 | 2024-06-27 15:15:00 | 228.38 | TARGET_HIT | 0.50 | 0.36% |
| SELL | retest1 | 2024-07-03 10:00:00 | 229.00 | 2024-07-03 10:10:00 | 227.48 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-07-03 10:00:00 | 229.00 | 2024-07-03 10:25:00 | 229.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 10:25:00 | 228.66 | 2024-07-11 12:45:00 | 229.70 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-07-15 09:30:00 | 223.61 | 2024-07-15 09:35:00 | 224.65 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-09-25 10:00:00 | 340.00 | 2024-09-25 10:25:00 | 337.68 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-09-25 10:00:00 | 340.00 | 2024-09-25 11:20:00 | 340.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-16 09:35:00 | 369.74 | 2024-10-16 13:45:00 | 368.20 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-17 11:05:00 | 367.08 | 2024-10-17 11:25:00 | 365.34 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-10-17 11:05:00 | 367.08 | 2024-10-17 12:30:00 | 367.08 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-31 11:15:00 | 325.36 | 2024-10-31 15:00:00 | 327.23 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-11-06 09:30:00 | 340.20 | 2024-11-06 10:20:00 | 338.46 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-12-06 09:55:00 | 326.41 | 2024-12-06 10:00:00 | 328.64 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-12-06 09:55:00 | 326.41 | 2024-12-06 14:15:00 | 331.03 | TARGET_HIT | 0.50 | 1.42% |
| BUY | retest1 | 2024-12-09 10:10:00 | 337.85 | 2024-12-09 10:30:00 | 340.78 | PARTIAL | 0.50 | 0.87% |
| BUY | retest1 | 2024-12-09 10:10:00 | 337.85 | 2024-12-09 15:20:00 | 341.99 | TARGET_HIT | 0.50 | 1.23% |
| SELL | retest1 | 2024-12-17 09:30:00 | 363.80 | 2024-12-17 11:20:00 | 361.57 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-12-17 09:30:00 | 363.80 | 2024-12-17 15:20:00 | 355.62 | TARGET_HIT | 0.50 | 2.25% |
| BUY | retest1 | 2024-12-18 11:10:00 | 357.96 | 2024-12-18 11:40:00 | 356.63 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-12-26 10:10:00 | 332.80 | 2024-12-26 10:15:00 | 333.88 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-31 09:50:00 | 337.21 | 2024-12-31 10:00:00 | 338.71 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-01-20 09:55:00 | 356.30 | 2025-01-20 10:05:00 | 359.34 | PARTIAL | 0.50 | 0.85% |
| BUY | retest1 | 2025-01-20 09:55:00 | 356.30 | 2025-01-20 15:20:00 | 376.84 | TARGET_HIT | 0.50 | 5.76% |
| BUY | retest1 | 2025-01-23 09:45:00 | 367.60 | 2025-01-23 10:15:00 | 371.86 | PARTIAL | 0.50 | 1.16% |
| BUY | retest1 | 2025-01-23 09:45:00 | 367.60 | 2025-01-23 11:40:00 | 369.07 | TARGET_HIT | 0.50 | 0.40% |
| SELL | retest1 | 2025-01-30 10:35:00 | 339.62 | 2025-01-30 10:40:00 | 341.56 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2025-02-01 11:00:00 | 339.62 | 2025-02-01 11:10:00 | 340.75 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-02-05 10:35:00 | 347.47 | 2025-02-05 10:45:00 | 349.13 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-02-06 09:30:00 | 351.15 | 2025-02-06 09:35:00 | 352.76 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-03-12 10:00:00 | 289.79 | 2025-03-12 10:40:00 | 291.36 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2025-03-17 09:35:00 | 289.45 | 2025-03-17 10:10:00 | 286.52 | PARTIAL | 0.50 | 1.01% |
| SELL | retest1 | 2025-03-17 09:35:00 | 289.45 | 2025-03-17 15:20:00 | 285.01 | TARGET_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2025-04-21 10:25:00 | 360.98 | 2025-04-21 10:35:00 | 363.39 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-04-21 10:25:00 | 360.98 | 2025-04-21 15:20:00 | 369.00 | TARGET_HIT | 0.50 | 2.22% |
| SELL | retest1 | 2025-04-25 09:35:00 | 350.82 | 2025-04-25 09:45:00 | 347.77 | PARTIAL | 0.50 | 0.87% |
| SELL | retest1 | 2025-04-25 09:35:00 | 350.82 | 2025-04-25 12:55:00 | 348.40 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2025-04-29 10:35:00 | 359.14 | 2025-04-29 10:50:00 | 361.60 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-04-29 10:35:00 | 359.14 | 2025-04-29 12:20:00 | 359.14 | STOP_HIT | 0.50 | 0.00% |
