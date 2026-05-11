# NTPC Ltd. (NTPC)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 402.10
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
| ENTRY1 | 100 |
| ENTRY2 | 0 |
| PARTIAL | 37 |
| TARGET_HIT | 17 |
| STOP_HIT | 83 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 54 / 83
- **Target hits / Stop hits / Partials:** 17 / 83 / 37
- **Avg / median % per leg:** 0.10% / 0.00%
- **Sum % (uncompounded):** 13.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 77 | 27 | 35.1% | 6 | 50 | 21 | 0.12% | 9.1% |
| BUY @ 2nd Alert (retest1) | 77 | 27 | 35.1% | 6 | 50 | 21 | 0.12% | 9.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 60 | 27 | 45.0% | 11 | 33 | 16 | 0.08% | 4.7% |
| SELL @ 2nd Alert (retest1) | 60 | 27 | 45.0% | 11 | 33 | 16 | 0.08% | 4.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 137 | 54 | 39.4% | 17 | 83 | 37 | 0.10% | 13.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-20 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-20 10:35:00 | 348.50 | 346.05 | 0.00 | ORB-long ORB[344.20,347.00] vol=2.4x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-20 10:50:00 | 349.69 | 346.86 | 0.00 | T1 1.5R @ 349.69 |
| Stop hit — per-position SL triggered | 2025-05-20 10:55:00 | 348.50 | 346.94 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-23 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 10:45:00 | 343.45 | 341.93 | 0.00 | ORB-long ORB[340.15,343.00] vol=1.7x ATR=0.83 |
| Stop hit — per-position SL triggered | 2025-05-23 11:45:00 | 342.62 | 342.59 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 10:20:00 | 346.20 | 348.23 | 0.00 | ORB-short ORB[347.15,351.20] vol=2.6x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-26 11:35:00 | 344.83 | 347.75 | 0.00 | T1 1.5R @ 344.83 |
| Target hit | 2025-05-26 15:20:00 | 343.35 | 346.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2025-06-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 11:00:00 | 330.15 | 331.72 | 0.00 | ORB-short ORB[331.80,334.75] vol=2.7x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 11:10:00 | 329.13 | 331.50 | 0.00 | T1 1.5R @ 329.13 |
| Target hit | 2025-06-03 15:20:00 | 328.85 | 329.07 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2025-06-04 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:50:00 | 326.85 | 328.14 | 0.00 | ORB-short ORB[327.50,329.65] vol=1.6x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-06-04 10:05:00 | 327.70 | 327.76 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 10:00:00 | 334.80 | 334.09 | 0.00 | ORB-long ORB[333.00,334.35] vol=2.4x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-06-09 11:05:00 | 334.17 | 334.60 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-11 10:35:00 | 341.60 | 340.98 | 0.00 | ORB-long ORB[340.10,341.50] vol=1.6x ATR=0.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 11:55:00 | 342.54 | 341.37 | 0.00 | T1 1.5R @ 342.54 |
| Stop hit — per-position SL triggered | 2025-06-11 13:15:00 | 341.60 | 341.59 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-17 09:30:00 | 336.65 | 335.40 | 0.00 | ORB-long ORB[333.10,336.00] vol=1.5x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-06-17 09:40:00 | 335.91 | 335.58 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-19 10:40:00 | 331.30 | 331.96 | 0.00 | ORB-short ORB[331.70,333.15] vol=1.8x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 11:55:00 | 330.46 | 331.66 | 0.00 | T1 1.5R @ 330.46 |
| Stop hit — per-position SL triggered | 2025-06-19 12:00:00 | 331.30 | 331.62 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2025-07-01 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-01 11:00:00 | 333.20 | 333.79 | 0.00 | ORB-short ORB[333.60,335.25] vol=1.6x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-07-01 11:05:00 | 333.67 | 333.78 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-02 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-02 09:35:00 | 335.35 | 334.49 | 0.00 | ORB-long ORB[332.60,334.50] vol=2.4x ATR=0.59 |
| Stop hit — per-position SL triggered | 2025-07-02 09:40:00 | 334.76 | 334.51 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 09:40:00 | 338.75 | 337.16 | 0.00 | ORB-long ORB[333.75,335.90] vol=3.6x ATR=0.86 |
| Stop hit — per-position SL triggered | 2025-07-07 09:45:00 | 337.89 | 337.27 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-07-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-08 10:00:00 | 342.05 | 340.47 | 0.00 | ORB-long ORB[337.00,340.50] vol=1.7x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-07-08 10:30:00 | 341.33 | 340.83 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 11:15:00 | 345.05 | 344.03 | 0.00 | ORB-long ORB[343.10,344.80] vol=2.1x ATR=0.38 |
| Stop hit — per-position SL triggered | 2025-07-09 11:45:00 | 344.67 | 344.17 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-15 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-15 10:45:00 | 342.40 | 342.51 | 0.00 | ORB-short ORB[342.75,343.85] vol=1.7x ATR=0.69 |
| Stop hit — per-position SL triggered | 2025-07-15 11:00:00 | 343.09 | 342.55 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 11:05:00 | 340.55 | 341.08 | 0.00 | ORB-short ORB[341.00,342.60] vol=1.8x ATR=0.40 |
| Stop hit — per-position SL triggered | 2025-07-22 11:10:00 | 340.95 | 341.07 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-23 11:00:00 | 343.95 | 342.77 | 0.00 | ORB-long ORB[340.95,343.20] vol=2.2x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-07-23 11:05:00 | 343.51 | 342.80 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-24 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:50:00 | 340.95 | 341.16 | 0.00 | ORB-short ORB[341.40,343.60] vol=2.6x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-07-24 11:00:00 | 341.44 | 341.15 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-25 10:00:00 | 336.15 | 337.33 | 0.00 | ORB-short ORB[338.35,339.90] vol=1.6x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 10:15:00 | 335.09 | 337.02 | 0.00 | T1 1.5R @ 335.09 |
| Target hit | 2025-07-25 12:45:00 | 335.80 | 335.79 | 0.00 | Trail-exit close>VWAP |

### Cycle 20 — BUY (started 2025-07-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 10:50:00 | 335.30 | 334.47 | 0.00 | ORB-long ORB[330.35,334.75] vol=1.8x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-07-28 11:20:00 | 334.58 | 334.70 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:55:00 | 331.20 | 331.56 | 0.00 | ORB-short ORB[332.00,334.35] vol=1.8x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-07-29 11:00:00 | 331.98 | 331.54 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-31 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 11:00:00 | 339.15 | 337.77 | 0.00 | ORB-long ORB[335.80,338.40] vol=1.7x ATR=0.67 |
| Stop hit — per-position SL triggered | 2025-07-31 11:30:00 | 338.48 | 338.14 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:00:00 | 329.70 | 330.81 | 0.00 | ORB-short ORB[330.35,332.45] vol=1.8x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-08-07 11:35:00 | 330.32 | 330.51 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-08 11:00:00 | 333.10 | 331.42 | 0.00 | ORB-long ORB[329.65,332.55] vol=2.9x ATR=0.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-08 11:15:00 | 334.20 | 331.51 | 0.00 | T1 1.5R @ 334.20 |
| Stop hit — per-position SL triggered | 2025-08-08 11:20:00 | 333.10 | 331.55 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2025-08-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-12 10:10:00 | 339.20 | 337.89 | 0.00 | ORB-long ORB[335.75,337.50] vol=2.1x ATR=0.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 10:15:00 | 340.22 | 338.49 | 0.00 | T1 1.5R @ 340.22 |
| Stop hit — per-position SL triggered | 2025-08-12 10:25:00 | 339.20 | 338.67 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-08-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-14 10:55:00 | 337.35 | 337.50 | 0.00 | ORB-short ORB[337.65,340.45] vol=1.6x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-08-14 11:55:00 | 337.81 | 337.37 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-19 11:15:00 | 335.05 | 337.04 | 0.00 | ORB-short ORB[337.80,340.65] vol=1.8x ATR=0.55 |
| Stop hit — per-position SL triggered | 2025-08-19 11:55:00 | 335.60 | 336.71 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-20 11:15:00 | 339.80 | 339.06 | 0.00 | ORB-long ORB[336.50,339.60] vol=3.8x ATR=0.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 12:40:00 | 340.71 | 339.45 | 0.00 | T1 1.5R @ 340.71 |
| Target hit | 2025-08-20 15:20:00 | 341.80 | 340.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 29 — SELL (started 2025-08-21 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-21 10:10:00 | 340.85 | 341.61 | 0.00 | ORB-short ORB[341.25,344.35] vol=7.4x ATR=0.74 |
| Stop hit — per-position SL triggered | 2025-08-21 10:15:00 | 341.59 | 341.59 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-22 11:00:00 | 336.55 | 336.71 | 0.00 | ORB-short ORB[337.00,338.90] vol=3.3x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-08-22 11:05:00 | 337.01 | 336.72 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:30:00 | 335.25 | 333.66 | 0.00 | ORB-long ORB[331.10,333.95] vol=1.5x ATR=0.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 10:05:00 | 336.30 | 334.63 | 0.00 | T1 1.5R @ 336.30 |
| Stop hit — per-position SL triggered | 2025-09-02 10:10:00 | 335.25 | 334.74 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-05 10:45:00 | 328.20 | 329.19 | 0.00 | ORB-short ORB[329.15,331.80] vol=2.4x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-09-05 10:55:00 | 328.77 | 329.10 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2025-09-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-08 10:30:00 | 328.30 | 329.63 | 0.00 | ORB-short ORB[328.75,330.95] vol=1.9x ATR=0.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 11:10:00 | 327.47 | 328.96 | 0.00 | T1 1.5R @ 327.47 |
| Stop hit — per-position SL triggered | 2025-09-08 12:35:00 | 328.30 | 328.62 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 11:05:00 | 327.10 | 326.30 | 0.00 | ORB-long ORB[325.15,326.40] vol=4.3x ATR=0.48 |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 326.62 | 326.34 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:30:00 | 335.25 | 333.35 | 0.00 | ORB-long ORB[331.20,333.80] vol=2.6x ATR=0.73 |
| Stop hit — per-position SL triggered | 2025-09-16 09:45:00 | 334.52 | 333.97 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-22 11:15:00 | 338.45 | 339.30 | 0.00 | ORB-short ORB[338.80,340.35] vol=1.8x ATR=0.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-22 11:25:00 | 337.82 | 339.13 | 0.00 | T1 1.5R @ 337.82 |
| Target hit | 2025-09-22 12:50:00 | 338.10 | 338.09 | 0.00 | Trail-exit close>VWAP |

### Cycle 37 — BUY (started 2025-09-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-23 10:55:00 | 340.90 | 339.61 | 0.00 | ORB-long ORB[338.10,339.80] vol=2.4x ATR=0.65 |
| Stop hit — per-position SL triggered | 2025-09-23 11:15:00 | 340.25 | 339.85 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-09-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 10:05:00 | 346.50 | 344.43 | 0.00 | ORB-long ORB[342.25,344.80] vol=2.1x ATR=0.75 |
| Stop hit — per-position SL triggered | 2025-09-24 10:10:00 | 345.75 | 344.61 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-09-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 11:00:00 | 338.20 | 338.74 | 0.00 | ORB-short ORB[338.50,340.75] vol=2.1x ATR=0.62 |
| Stop hit — per-position SL triggered | 2025-09-26 11:10:00 | 338.82 | 338.72 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-09-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:55:00 | 340.95 | 340.25 | 0.00 | ORB-long ORB[338.00,340.05] vol=8.5x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-09-29 11:40:00 | 340.31 | 340.52 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-06 09:45:00 | 338.35 | 339.18 | 0.00 | ORB-short ORB[339.50,341.80] vol=1.6x ATR=0.68 |
| Stop hit — per-position SL triggered | 2025-10-06 09:55:00 | 339.03 | 339.12 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-10-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-07 10:30:00 | 338.25 | 339.48 | 0.00 | ORB-short ORB[338.85,341.20] vol=2.6x ATR=0.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 10:50:00 | 337.34 | 339.25 | 0.00 | T1 1.5R @ 337.34 |
| Stop hit — per-position SL triggered | 2025-10-07 12:05:00 | 338.25 | 338.02 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-10-08 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:45:00 | 336.95 | 337.37 | 0.00 | ORB-short ORB[337.15,339.25] vol=1.7x ATR=0.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-08 10:55:00 | 336.12 | 337.11 | 0.00 | T1 1.5R @ 336.12 |
| Target hit | 2025-10-08 15:20:00 | 333.30 | 334.34 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2025-10-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-10 10:45:00 | 341.25 | 339.67 | 0.00 | ORB-long ORB[335.95,338.75] vol=1.8x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-10-10 11:05:00 | 340.47 | 340.09 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-10-14 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:00:00 | 340.95 | 341.61 | 0.00 | ORB-short ORB[341.50,342.90] vol=3.3x ATR=0.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-14 10:30:00 | 340.01 | 341.25 | 0.00 | T1 1.5R @ 340.01 |
| Target hit | 2025-10-14 15:20:00 | 336.75 | 338.24 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2025-10-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 10:55:00 | 341.45 | 340.49 | 0.00 | ORB-long ORB[336.75,341.00] vol=4.7x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-10-15 11:10:00 | 340.85 | 340.60 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-16 11:15:00 | 342.50 | 341.39 | 0.00 | ORB-long ORB[339.65,341.70] vol=4.3x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-10-16 11:20:00 | 342.03 | 341.42 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 11:00:00 | 343.90 | 342.56 | 0.00 | ORB-long ORB[340.15,342.10] vol=2.1x ATR=0.54 |
| Stop hit — per-position SL triggered | 2025-10-17 11:35:00 | 343.36 | 343.12 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-10-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-24 10:20:00 | 340.65 | 341.80 | 0.00 | ORB-short ORB[340.90,343.25] vol=1.6x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-10-24 10:50:00 | 341.43 | 341.43 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-10-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:55:00 | 342.15 | 341.67 | 0.00 | ORB-long ORB[340.65,341.70] vol=1.6x ATR=0.56 |
| Stop hit — per-position SL triggered | 2025-10-27 10:55:00 | 341.59 | 341.99 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-10-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:25:00 | 346.55 | 343.98 | 0.00 | ORB-long ORB[339.15,341.80] vol=1.8x ATR=0.98 |
| Stop hit — per-position SL triggered | 2025-10-29 10:35:00 | 345.57 | 344.44 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-11-04 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 10:05:00 | 330.70 | 332.99 | 0.00 | ORB-short ORB[333.60,336.00] vol=2.1x ATR=0.72 |
| Stop hit — per-position SL triggered | 2025-11-04 10:10:00 | 331.42 | 332.76 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2025-11-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-06 10:55:00 | 329.05 | 329.52 | 0.00 | ORB-short ORB[329.60,332.95] vol=1.5x ATR=0.66 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-06 11:30:00 | 328.07 | 329.25 | 0.00 | T1 1.5R @ 328.07 |
| Target hit | 2025-11-06 15:20:00 | 327.10 | 326.92 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — SELL (started 2025-11-11 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 10:45:00 | 323.65 | 324.13 | 0.00 | ORB-short ORB[324.50,326.75] vol=4.7x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-11-11 10:50:00 | 324.18 | 324.30 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2025-11-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-12 10:55:00 | 325.80 | 326.62 | 0.00 | ORB-short ORB[327.15,328.20] vol=1.8x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-11-12 11:05:00 | 326.27 | 326.57 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-11-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:45:00 | 328.25 | 327.26 | 0.00 | ORB-long ORB[325.55,327.40] vol=4.7x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 11:00:00 | 328.94 | 327.64 | 0.00 | T1 1.5R @ 328.94 |
| Stop hit — per-position SL triggered | 2025-11-13 12:35:00 | 328.25 | 328.30 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:40:00 | 326.50 | 327.65 | 0.00 | ORB-short ORB[327.00,328.65] vol=1.7x ATR=0.53 |
| Stop hit — per-position SL triggered | 2025-11-14 10:45:00 | 327.03 | 327.63 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-17 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-17 11:00:00 | 330.60 | 330.48 | 0.00 | ORB-long ORB[328.65,330.45] vol=1.9x ATR=0.47 |
| Stop hit — per-position SL triggered | 2025-11-17 11:20:00 | 330.13 | 330.48 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-11-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:00:00 | 325.90 | 327.26 | 0.00 | ORB-short ORB[326.30,328.40] vol=1.6x ATR=0.63 |
| Stop hit — per-position SL triggered | 2025-11-21 10:50:00 | 326.53 | 326.72 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-11-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 11:00:00 | 325.45 | 325.95 | 0.00 | ORB-short ORB[325.50,327.10] vol=1.6x ATR=0.44 |
| Stop hit — per-position SL triggered | 2025-11-24 11:20:00 | 325.89 | 325.83 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-11-25 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-25 10:55:00 | 324.90 | 324.36 | 0.00 | ORB-long ORB[322.50,324.25] vol=6.9x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-11-25 12:30:00 | 324.33 | 324.63 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2025-11-26 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-26 10:50:00 | 326.60 | 325.60 | 0.00 | ORB-long ORB[324.00,325.85] vol=1.7x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-26 11:25:00 | 327.45 | 325.97 | 0.00 | T1 1.5R @ 327.45 |
| Stop hit — per-position SL triggered | 2025-11-26 12:25:00 | 326.60 | 326.53 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 11:15:00 | 323.90 | 324.70 | 0.00 | ORB-short ORB[325.35,329.00] vol=5.1x ATR=0.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 11:40:00 | 323.16 | 324.29 | 0.00 | T1 1.5R @ 323.16 |
| Target hit | 2025-12-03 15:20:00 | 322.95 | 323.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2025-12-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-05 11:00:00 | 323.40 | 322.48 | 0.00 | ORB-long ORB[322.00,323.20] vol=3.1x ATR=0.57 |
| Stop hit — per-position SL triggered | 2025-12-05 11:55:00 | 322.83 | 322.71 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:15:00 | 319.30 | 317.70 | 0.00 | ORB-long ORB[317.15,318.90] vol=3.1x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 12:05:00 | 320.38 | 318.23 | 0.00 | T1 1.5R @ 320.38 |
| Stop hit — per-position SL triggered | 2025-12-09 12:55:00 | 319.30 | 318.61 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-12-10 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 10:45:00 | 323.00 | 321.76 | 0.00 | ORB-long ORB[319.50,321.40] vol=1.6x ATR=0.60 |
| Stop hit — per-position SL triggered | 2025-12-10 10:55:00 | 322.40 | 321.85 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-12-11 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:50:00 | 322.75 | 321.53 | 0.00 | ORB-long ORB[319.70,322.40] vol=2.3x ATR=0.64 |
| Stop hit — per-position SL triggered | 2025-12-11 10:55:00 | 322.11 | 321.64 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-12-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 10:55:00 | 318.20 | 318.79 | 0.00 | ORB-short ORB[318.40,322.15] vol=1.7x ATR=0.49 |
| Stop hit — per-position SL triggered | 2025-12-18 11:55:00 | 318.69 | 318.54 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-12-19 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:10:00 | 317.95 | 318.64 | 0.00 | ORB-short ORB[318.65,319.70] vol=4.5x ATR=0.46 |
| Stop hit — per-position SL triggered | 2025-12-19 11:55:00 | 318.41 | 318.30 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-12-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 10:50:00 | 322.05 | 321.25 | 0.00 | ORB-long ORB[319.50,321.65] vol=1.7x ATR=0.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:25:00 | 322.76 | 321.48 | 0.00 | T1 1.5R @ 322.76 |
| Stop hit — per-position SL triggered | 2025-12-22 12:30:00 | 322.05 | 321.80 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-12-23 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 11:05:00 | 324.05 | 322.79 | 0.00 | ORB-long ORB[320.25,322.45] vol=3.3x ATR=0.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 11:25:00 | 324.74 | 323.26 | 0.00 | T1 1.5R @ 324.74 |
| Stop hit — per-position SL triggered | 2025-12-23 11:30:00 | 324.05 | 323.31 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-12-26 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-26 11:00:00 | 324.60 | 323.64 | 0.00 | ORB-long ORB[322.30,324.20] vol=3.5x ATR=0.45 |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 324.15 | 323.67 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2025-12-31 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 09:55:00 | 327.95 | 327.26 | 0.00 | ORB-long ORB[324.70,327.20] vol=1.9x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 10:20:00 | 329.23 | 327.88 | 0.00 | T1 1.5R @ 329.23 |
| Target hit | 2025-12-31 15:10:00 | 329.15 | 329.34 | 0.00 | Trail-exit close<VWAP |

### Cycle 74 — BUY (started 2026-01-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-01 11:10:00 | 334.40 | 332.87 | 0.00 | ORB-long ORB[329.60,331.45] vol=2.7x ATR=0.69 |
| Stop hit — per-position SL triggered | 2026-01-01 11:20:00 | 333.71 | 332.93 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-01-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:10:00 | 340.30 | 338.64 | 0.00 | ORB-long ORB[335.95,339.65] vol=2.1x ATR=0.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-02 10:20:00 | 341.64 | 339.05 | 0.00 | T1 1.5R @ 341.64 |
| Target hit | 2026-01-02 15:20:00 | 352.80 | 347.12 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 76 — BUY (started 2026-01-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 10:00:00 | 354.50 | 352.57 | 0.00 | ORB-long ORB[348.65,352.00] vol=4.4x ATR=1.26 |
| Stop hit — per-position SL triggered | 2026-01-05 10:10:00 | 353.24 | 352.64 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2026-01-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 10:45:00 | 349.30 | 350.83 | 0.00 | ORB-short ORB[349.40,354.10] vol=1.8x ATR=0.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-06 11:15:00 | 348.16 | 350.43 | 0.00 | T1 1.5R @ 348.16 |
| Stop hit — per-position SL triggered | 2026-01-06 13:05:00 | 349.30 | 349.15 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-01-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-07 10:50:00 | 348.00 | 348.63 | 0.00 | ORB-short ORB[348.60,351.35] vol=3.1x ATR=0.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-07 11:10:00 | 347.12 | 348.49 | 0.00 | T1 1.5R @ 347.12 |
| Target hit | 2026-01-07 12:50:00 | 347.50 | 347.47 | 0.00 | Trail-exit close>VWAP |

### Cycle 79 — SELL (started 2026-01-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:10:00 | 346.35 | 347.78 | 0.00 | ORB-short ORB[346.75,349.80] vol=2.6x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:25:00 | 345.37 | 347.48 | 0.00 | T1 1.5R @ 345.37 |
| Target hit | 2026-01-08 14:35:00 | 345.20 | 345.17 | 0.00 | Trail-exit close>VWAP |

### Cycle 80 — BUY (started 2026-01-14 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 10:05:00 | 343.15 | 341.28 | 0.00 | ORB-long ORB[337.45,341.35] vol=2.0x ATR=0.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 10:20:00 | 344.38 | 342.46 | 0.00 | T1 1.5R @ 344.38 |
| Target hit | 2026-01-14 15:20:00 | 349.30 | 347.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 81 — BUY (started 2026-01-23 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 10:25:00 | 345.00 | 343.55 | 0.00 | ORB-long ORB[340.50,343.95] vol=5.1x ATR=0.79 |
| Stop hit — per-position SL triggered | 2026-01-23 10:50:00 | 344.21 | 344.00 | 0.00 | SL hit |

### Cycle 82 — BUY (started 2026-01-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-27 09:55:00 | 342.00 | 340.70 | 0.00 | ORB-long ORB[337.05,341.30] vol=1.7x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-27 10:20:00 | 343.57 | 341.40 | 0.00 | T1 1.5R @ 343.57 |
| Stop hit — per-position SL triggered | 2026-01-27 10:50:00 | 342.00 | 341.87 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2026-02-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 10:25:00 | 363.55 | 362.90 | 0.00 | ORB-long ORB[361.00,362.95] vol=2.5x ATR=1.05 |
| Stop hit — per-position SL triggered | 2026-02-06 10:35:00 | 362.50 | 362.90 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 11:15:00 | 364.20 | 365.82 | 0.00 | ORB-short ORB[364.65,368.00] vol=2.4x ATR=0.69 |
| Stop hit — per-position SL triggered | 2026-02-13 11:25:00 | 364.89 | 365.64 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2026-02-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-20 10:30:00 | 369.05 | 366.87 | 0.00 | ORB-long ORB[362.70,366.35] vol=2.2x ATR=0.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 11:05:00 | 370.27 | 368.08 | 0.00 | T1 1.5R @ 370.27 |
| Target hit | 2026-02-20 15:20:00 | 373.15 | 371.70 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 86 — BUY (started 2026-03-11 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 11:00:00 | 382.05 | 381.67 | 0.00 | ORB-long ORB[376.45,382.00] vol=2.3x ATR=0.96 |
| Stop hit — per-position SL triggered | 2026-03-11 11:15:00 | 381.09 | 381.65 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2026-03-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-12 10:10:00 | 383.25 | 380.36 | 0.00 | ORB-long ORB[375.65,380.55] vol=2.9x ATR=1.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 10:30:00 | 385.26 | 381.95 | 0.00 | T1 1.5R @ 385.26 |
| Target hit | 2026-03-12 15:20:00 | 390.40 | 387.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — SELL (started 2026-03-18 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 11:00:00 | 381.90 | 383.16 | 0.00 | ORB-short ORB[384.00,386.35] vol=2.9x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-18 11:30:00 | 380.73 | 382.90 | 0.00 | T1 1.5R @ 380.73 |
| Target hit | 2026-03-18 13:30:00 | 381.30 | 381.00 | 0.00 | Trail-exit close>VWAP |

### Cycle 89 — SELL (started 2026-03-23 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 10:55:00 | 371.50 | 374.60 | 0.00 | ORB-short ORB[375.55,379.50] vol=1.5x ATR=1.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 12:05:00 | 369.75 | 373.56 | 0.00 | T1 1.5R @ 369.75 |
| Stop hit — per-position SL triggered | 2026-03-23 12:35:00 | 371.50 | 373.17 | 0.00 | SL hit |

### Cycle 90 — SELL (started 2026-03-24 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 11:00:00 | 371.90 | 374.23 | 0.00 | ORB-short ORB[374.20,378.00] vol=2.3x ATR=1.28 |
| Stop hit — per-position SL triggered | 2026-03-24 11:10:00 | 373.18 | 373.89 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2026-03-25 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 11:10:00 | 378.45 | 377.17 | 0.00 | ORB-long ORB[375.60,378.05] vol=2.4x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 12:10:00 | 379.71 | 377.62 | 0.00 | T1 1.5R @ 379.71 |
| Stop hit — per-position SL triggered | 2026-03-25 12:50:00 | 378.45 | 377.83 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2026-04-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 11:05:00 | 367.85 | 367.32 | 0.00 | ORB-long ORB[362.25,366.00] vol=3.4x ATR=1.27 |
| Stop hit — per-position SL triggered | 2026-04-07 11:40:00 | 366.58 | 367.36 | 0.00 | SL hit |

### Cycle 93 — BUY (started 2026-04-09 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-09 10:50:00 | 381.65 | 380.45 | 0.00 | ORB-long ORB[376.50,381.30] vol=4.4x ATR=1.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 11:20:00 | 383.30 | 380.81 | 0.00 | T1 1.5R @ 383.30 |
| Stop hit — per-position SL triggered | 2026-04-09 13:10:00 | 381.65 | 381.21 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2026-04-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-13 09:40:00 | 380.90 | 378.37 | 0.00 | ORB-long ORB[374.35,379.40] vol=2.7x ATR=1.53 |
| Stop hit — per-position SL triggered | 2026-04-13 09:50:00 | 379.37 | 379.03 | 0.00 | SL hit |

### Cycle 95 — BUY (started 2026-04-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 11:00:00 | 397.45 | 395.51 | 0.00 | ORB-long ORB[390.65,395.45] vol=1.5x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-20 11:05:00 | 398.82 | 396.28 | 0.00 | T1 1.5R @ 398.82 |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 397.45 | 397.96 | 0.00 | SL hit |

### Cycle 96 — SELL (started 2026-04-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 11:10:00 | 400.95 | 402.71 | 0.00 | ORB-short ORB[402.50,405.90] vol=6.3x ATR=0.87 |
| Stop hit — per-position SL triggered | 2026-04-23 11:45:00 | 401.82 | 402.39 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2026-04-24 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 11:10:00 | 399.80 | 400.92 | 0.00 | ORB-short ORB[402.40,405.25] vol=2.0x ATR=0.84 |
| Stop hit — per-position SL triggered | 2026-04-24 11:30:00 | 400.64 | 400.84 | 0.00 | SL hit |

### Cycle 98 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 10:15:00 | 410.60 | 406.76 | 0.00 | ORB-long ORB[402.60,406.80] vol=1.9x ATR=1.17 |
| Stop hit — per-position SL triggered | 2026-04-27 10:25:00 | 409.43 | 407.28 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2026-05-05 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-05 11:05:00 | 396.10 | 398.03 | 0.00 | ORB-short ORB[397.25,400.50] vol=1.8x ATR=0.90 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 397.00 | 397.91 | 0.00 | SL hit |

### Cycle 100 — BUY (started 2026-05-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-07 10:50:00 | 400.00 | 396.85 | 0.00 | ORB-long ORB[393.15,397.05] vol=1.8x ATR=1.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-07 12:35:00 | 402.17 | 398.43 | 0.00 | T1 1.5R @ 402.17 |
| Stop hit — per-position SL triggered | 2026-05-07 13:15:00 | 400.00 | 398.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-20 10:35:00 | 348.50 | 2025-05-20 10:50:00 | 349.69 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-05-20 10:35:00 | 348.50 | 2025-05-20 10:55:00 | 348.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-23 10:45:00 | 343.45 | 2025-05-23 11:45:00 | 342.62 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-05-26 10:20:00 | 346.20 | 2025-05-26 11:35:00 | 344.83 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-05-26 10:20:00 | 346.20 | 2025-05-26 15:20:00 | 343.35 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2025-06-03 11:00:00 | 330.15 | 2025-06-03 11:10:00 | 329.13 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-06-03 11:00:00 | 330.15 | 2025-06-03 15:20:00 | 328.85 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2025-06-04 09:50:00 | 326.85 | 2025-06-04 10:05:00 | 327.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-06-09 10:00:00 | 334.80 | 2025-06-09 11:05:00 | 334.17 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-06-11 10:35:00 | 341.60 | 2025-06-11 11:55:00 | 342.54 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-06-11 10:35:00 | 341.60 | 2025-06-11 13:15:00 | 341.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-17 09:30:00 | 336.65 | 2025-06-17 09:40:00 | 335.91 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-06-19 10:40:00 | 331.30 | 2025-06-19 11:55:00 | 330.46 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-06-19 10:40:00 | 331.30 | 2025-06-19 12:00:00 | 331.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-01 11:00:00 | 333.20 | 2025-07-01 11:05:00 | 333.67 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-07-02 09:35:00 | 335.35 | 2025-07-02 09:40:00 | 334.76 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-07 09:40:00 | 338.75 | 2025-07-07 09:45:00 | 337.89 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-07-08 10:00:00 | 342.05 | 2025-07-08 10:30:00 | 341.33 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-07-09 11:15:00 | 345.05 | 2025-07-09 11:45:00 | 344.67 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest1 | 2025-07-15 10:45:00 | 342.40 | 2025-07-15 11:00:00 | 343.09 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-07-22 11:05:00 | 340.55 | 2025-07-22 11:10:00 | 340.95 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest1 | 2025-07-23 11:00:00 | 343.95 | 2025-07-23 11:05:00 | 343.51 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest1 | 2025-07-24 10:50:00 | 340.95 | 2025-07-24 11:00:00 | 341.44 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-07-25 10:00:00 | 336.15 | 2025-07-25 10:15:00 | 335.09 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-07-25 10:00:00 | 336.15 | 2025-07-25 12:45:00 | 335.80 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2025-07-28 10:50:00 | 335.30 | 2025-07-28 11:20:00 | 334.58 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-07-29 10:55:00 | 331.20 | 2025-07-29 11:00:00 | 331.98 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-31 11:00:00 | 339.15 | 2025-07-31 11:30:00 | 338.48 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-08-07 11:00:00 | 329.70 | 2025-08-07 11:35:00 | 330.32 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-08-08 11:00:00 | 333.10 | 2025-08-08 11:15:00 | 334.20 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-08-08 11:00:00 | 333.10 | 2025-08-08 11:20:00 | 333.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-12 10:10:00 | 339.20 | 2025-08-12 10:15:00 | 340.22 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-08-12 10:10:00 | 339.20 | 2025-08-12 10:25:00 | 339.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-14 10:55:00 | 337.35 | 2025-08-14 11:55:00 | 337.81 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-08-19 11:15:00 | 335.05 | 2025-08-19 11:55:00 | 335.60 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-08-20 11:15:00 | 339.80 | 2025-08-20 12:40:00 | 340.71 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2025-08-20 11:15:00 | 339.80 | 2025-08-20 15:20:00 | 341.80 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2025-08-21 10:10:00 | 340.85 | 2025-08-21 10:15:00 | 341.59 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-08-22 11:00:00 | 336.55 | 2025-08-22 11:05:00 | 337.01 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-09-02 09:30:00 | 335.25 | 2025-09-02 10:05:00 | 336.30 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-09-02 09:30:00 | 335.25 | 2025-09-02 10:10:00 | 335.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-05 10:45:00 | 328.20 | 2025-09-05 10:55:00 | 328.77 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-09-08 10:30:00 | 328.30 | 2025-09-08 11:10:00 | 327.47 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-09-08 10:30:00 | 328.30 | 2025-09-08 12:35:00 | 328.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-10 11:05:00 | 327.10 | 2025-09-10 11:15:00 | 326.62 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-09-16 09:30:00 | 335.25 | 2025-09-16 09:45:00 | 334.52 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-22 11:15:00 | 338.45 | 2025-09-22 11:25:00 | 337.82 | PARTIAL | 0.50 | 0.18% |
| SELL | retest1 | 2025-09-22 11:15:00 | 338.45 | 2025-09-22 12:50:00 | 338.10 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2025-09-23 10:55:00 | 340.90 | 2025-09-23 11:15:00 | 340.25 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-24 10:05:00 | 346.50 | 2025-09-24 10:10:00 | 345.75 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-26 11:00:00 | 338.20 | 2025-09-26 11:10:00 | 338.82 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-29 10:55:00 | 340.95 | 2025-09-29 11:40:00 | 340.31 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-10-06 09:45:00 | 338.35 | 2025-10-06 09:55:00 | 339.03 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-10-07 10:30:00 | 338.25 | 2025-10-07 10:50:00 | 337.34 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-10-07 10:30:00 | 338.25 | 2025-10-07 12:05:00 | 338.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 10:45:00 | 336.95 | 2025-10-08 10:55:00 | 336.12 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-10-08 10:45:00 | 336.95 | 2025-10-08 15:20:00 | 333.30 | TARGET_HIT | 0.50 | 1.08% |
| BUY | retest1 | 2025-10-10 10:45:00 | 341.25 | 2025-10-10 11:05:00 | 340.47 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-14 10:00:00 | 340.95 | 2025-10-14 10:30:00 | 340.01 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-10-14 10:00:00 | 340.95 | 2025-10-14 15:20:00 | 336.75 | TARGET_HIT | 0.50 | 1.23% |
| BUY | retest1 | 2025-10-15 10:55:00 | 341.45 | 2025-10-15 11:10:00 | 340.85 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-10-16 11:15:00 | 342.50 | 2025-10-16 11:20:00 | 342.03 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-10-17 11:00:00 | 343.90 | 2025-10-17 11:35:00 | 343.36 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-10-24 10:20:00 | 340.65 | 2025-10-24 10:50:00 | 341.43 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-27 09:55:00 | 342.15 | 2025-10-27 10:55:00 | 341.59 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-10-29 10:25:00 | 346.55 | 2025-10-29 10:35:00 | 345.57 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-11-04 10:05:00 | 330.70 | 2025-11-04 10:10:00 | 331.42 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-11-06 10:55:00 | 329.05 | 2025-11-06 11:30:00 | 328.07 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-11-06 10:55:00 | 329.05 | 2025-11-06 15:20:00 | 327.10 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2025-11-11 10:45:00 | 323.65 | 2025-11-11 10:50:00 | 324.18 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-11-12 10:55:00 | 325.80 | 2025-11-12 11:05:00 | 326.27 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-11-13 10:45:00 | 328.25 | 2025-11-13 11:00:00 | 328.94 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-11-13 10:45:00 | 328.25 | 2025-11-13 12:35:00 | 328.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-14 10:40:00 | 326.50 | 2025-11-14 10:45:00 | 327.03 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2025-11-17 11:00:00 | 330.60 | 2025-11-17 11:20:00 | 330.13 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest1 | 2025-11-21 10:00:00 | 325.90 | 2025-11-21 10:50:00 | 326.53 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-11-24 11:00:00 | 325.45 | 2025-11-24 11:20:00 | 325.89 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest1 | 2025-11-25 10:55:00 | 324.90 | 2025-11-25 12:30:00 | 324.33 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-11-26 10:50:00 | 326.60 | 2025-11-26 11:25:00 | 327.45 | PARTIAL | 0.50 | 0.26% |
| BUY | retest1 | 2025-11-26 10:50:00 | 326.60 | 2025-11-26 12:25:00 | 326.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-03 11:15:00 | 323.90 | 2025-12-03 11:40:00 | 323.16 | PARTIAL | 0.50 | 0.23% |
| SELL | retest1 | 2025-12-03 11:15:00 | 323.90 | 2025-12-03 15:20:00 | 322.95 | TARGET_HIT | 0.50 | 0.29% |
| BUY | retest1 | 2025-12-05 11:00:00 | 323.40 | 2025-12-05 11:55:00 | 322.83 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-12-09 11:15:00 | 319.30 | 2025-12-09 12:05:00 | 320.38 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-12-09 11:15:00 | 319.30 | 2025-12-09 12:55:00 | 319.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-10 10:45:00 | 323.00 | 2025-12-10 10:55:00 | 322.40 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-12-11 10:50:00 | 322.75 | 2025-12-11 10:55:00 | 322.11 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-12-18 10:55:00 | 318.20 | 2025-12-18 11:55:00 | 318.69 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest1 | 2025-12-19 10:10:00 | 317.95 | 2025-12-19 11:55:00 | 318.41 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest1 | 2025-12-22 10:50:00 | 322.05 | 2025-12-22 11:25:00 | 322.76 | PARTIAL | 0.50 | 0.22% |
| BUY | retest1 | 2025-12-22 10:50:00 | 322.05 | 2025-12-22 12:30:00 | 322.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 11:05:00 | 324.05 | 2025-12-23 11:25:00 | 324.74 | PARTIAL | 0.50 | 0.21% |
| BUY | retest1 | 2025-12-23 11:05:00 | 324.05 | 2025-12-23 11:30:00 | 324.05 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-26 11:00:00 | 324.60 | 2025-12-26 11:15:00 | 324.15 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest1 | 2025-12-31 09:55:00 | 327.95 | 2025-12-31 10:20:00 | 329.23 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-12-31 09:55:00 | 327.95 | 2025-12-31 15:10:00 | 329.15 | TARGET_HIT | 0.50 | 0.37% |
| BUY | retest1 | 2026-01-01 11:10:00 | 334.40 | 2026-01-01 11:20:00 | 333.71 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-01-02 10:10:00 | 340.30 | 2026-01-02 10:20:00 | 341.64 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2026-01-02 10:10:00 | 340.30 | 2026-01-02 15:20:00 | 352.80 | TARGET_HIT | 0.50 | 3.67% |
| BUY | retest1 | 2026-01-05 10:00:00 | 354.50 | 2026-01-05 10:10:00 | 353.24 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2026-01-06 10:45:00 | 349.30 | 2026-01-06 11:15:00 | 348.16 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2026-01-06 10:45:00 | 349.30 | 2026-01-06 13:05:00 | 349.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-07 10:50:00 | 348.00 | 2026-01-07 11:10:00 | 347.12 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2026-01-07 10:50:00 | 348.00 | 2026-01-07 12:50:00 | 347.50 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2026-01-08 11:10:00 | 346.35 | 2026-01-08 11:25:00 | 345.37 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2026-01-08 11:10:00 | 346.35 | 2026-01-08 14:35:00 | 345.20 | TARGET_HIT | 0.50 | 0.33% |
| BUY | retest1 | 2026-01-14 10:05:00 | 343.15 | 2026-01-14 10:20:00 | 344.38 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2026-01-14 10:05:00 | 343.15 | 2026-01-14 15:20:00 | 349.30 | TARGET_HIT | 0.50 | 1.79% |
| BUY | retest1 | 2026-01-23 10:25:00 | 345.00 | 2026-01-23 10:50:00 | 344.21 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-01-27 09:55:00 | 342.00 | 2026-01-27 10:20:00 | 343.57 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2026-01-27 09:55:00 | 342.00 | 2026-01-27 10:50:00 | 342.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-06 10:25:00 | 363.55 | 2026-02-06 10:35:00 | 362.50 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2026-02-13 11:15:00 | 364.20 | 2026-02-13 11:25:00 | 364.89 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2026-02-20 10:30:00 | 369.05 | 2026-02-20 11:05:00 | 370.27 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-02-20 10:30:00 | 369.05 | 2026-02-20 15:20:00 | 373.15 | TARGET_HIT | 0.50 | 1.11% |
| BUY | retest1 | 2026-03-11 11:00:00 | 382.05 | 2026-03-11 11:15:00 | 381.09 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-03-12 10:10:00 | 383.25 | 2026-03-12 10:30:00 | 385.26 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-03-12 10:10:00 | 383.25 | 2026-03-12 15:20:00 | 390.40 | TARGET_HIT | 0.50 | 1.87% |
| SELL | retest1 | 2026-03-18 11:00:00 | 381.90 | 2026-03-18 11:30:00 | 380.73 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2026-03-18 11:00:00 | 381.90 | 2026-03-18 13:30:00 | 381.30 | TARGET_HIT | 0.50 | 0.16% |
| SELL | retest1 | 2026-03-23 10:55:00 | 371.50 | 2026-03-23 12:05:00 | 369.75 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2026-03-23 10:55:00 | 371.50 | 2026-03-23 12:35:00 | 371.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 11:00:00 | 371.90 | 2026-03-24 11:10:00 | 373.18 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-03-25 11:10:00 | 378.45 | 2026-03-25 12:10:00 | 379.71 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2026-03-25 11:10:00 | 378.45 | 2026-03-25 12:50:00 | 378.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-07 11:05:00 | 367.85 | 2026-04-07 11:40:00 | 366.58 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-04-09 10:50:00 | 381.65 | 2026-04-09 11:20:00 | 383.30 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2026-04-09 10:50:00 | 381.65 | 2026-04-09 13:10:00 | 381.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-13 09:40:00 | 380.90 | 2026-04-13 09:50:00 | 379.37 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2026-04-20 11:00:00 | 397.45 | 2026-04-20 11:05:00 | 398.82 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2026-04-20 11:00:00 | 397.45 | 2026-04-20 15:15:00 | 397.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-04-23 11:10:00 | 400.95 | 2026-04-23 11:45:00 | 401.82 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2026-04-24 11:10:00 | 399.80 | 2026-04-24 11:30:00 | 400.64 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2026-04-27 10:15:00 | 410.60 | 2026-04-27 10:25:00 | 409.43 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-05-05 11:05:00 | 396.10 | 2026-05-05 11:15:00 | 397.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2026-05-07 10:50:00 | 400.00 | 2026-05-07 12:35:00 | 402.17 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2026-05-07 10:50:00 | 400.00 | 2026-05-07 13:15:00 | 400.00 | STOP_HIT | 0.50 | 0.00% |
