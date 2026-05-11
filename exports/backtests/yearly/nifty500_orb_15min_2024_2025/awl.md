# AWL Agri Business Ltd. (AWL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 206.00
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
| ENTRY1 | 74 |
| ENTRY2 | 0 |
| PARTIAL | 24 |
| TARGET_HIT | 8 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 98 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 66
- **Target hits / Stop hits / Partials:** 8 / 66 / 24
- **Avg / median % per leg:** 0.12% / -0.20%
- **Sum % (uncompounded):** 12.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 14 | 25.0% | 3 | 42 | 11 | 0.05% | 2.7% |
| BUY @ 2nd Alert (retest1) | 56 | 14 | 25.0% | 3 | 42 | 11 | 0.05% | 2.7% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 18 | 42.9% | 5 | 24 | 13 | 0.22% | 9.4% |
| SELL @ 2nd Alert (retest1) | 42 | 18 | 42.9% | 5 | 24 | 13 | 0.22% | 9.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 98 | 32 | 32.7% | 8 | 66 | 24 | 0.12% | 12.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-14 11:15:00 | 335.25 | 333.82 | 0.00 | ORB-long ORB[332.95,335.00] vol=2.2x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-14 11:25:00 | 336.85 | 334.08 | 0.00 | T1 1.5R @ 336.85 |
| Stop hit — per-position SL triggered | 2024-05-14 11:30:00 | 335.25 | 334.19 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:20:00 | 337.80 | 336.61 | 0.00 | ORB-long ORB[335.00,336.75] vol=1.8x ATR=0.99 |
| Stop hit — per-position SL triggered | 2024-05-17 11:20:00 | 336.81 | 336.92 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-21 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:05:00 | 341.20 | 337.69 | 0.00 | ORB-long ORB[335.05,337.95] vol=4.3x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-05-21 10:10:00 | 340.10 | 338.33 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 11:15:00 | 348.45 | 342.68 | 0.00 | ORB-long ORB[339.55,342.80] vol=9.7x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-05-23 11:20:00 | 346.87 | 343.40 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-28 09:35:00 | 338.85 | 339.90 | 0.00 | ORB-short ORB[339.25,342.90] vol=3.3x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 10:10:00 | 337.22 | 339.35 | 0.00 | T1 1.5R @ 337.22 |
| Stop hit — per-position SL triggered | 2024-05-28 10:15:00 | 338.85 | 339.34 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-31 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-31 09:35:00 | 351.70 | 349.02 | 0.00 | ORB-long ORB[346.10,349.50] vol=2.5x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-05-31 09:40:00 | 350.16 | 349.49 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:35:00 | 344.85 | 345.81 | 0.00 | ORB-short ORB[345.00,347.95] vol=1.7x ATR=1.05 |
| Stop hit — per-position SL triggered | 2024-06-11 09:45:00 | 345.90 | 345.76 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 11:10:00 | 346.55 | 345.38 | 0.00 | ORB-long ORB[343.00,346.00] vol=6.1x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-06-14 11:20:00 | 345.70 | 345.46 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:10:00 | 342.30 | 339.30 | 0.00 | ORB-long ORB[337.25,339.80] vol=1.8x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-06-20 10:40:00 | 341.20 | 340.18 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-21 10:45:00 | 338.35 | 339.67 | 0.00 | ORB-short ORB[339.30,341.90] vol=2.4x ATR=0.73 |
| Stop hit — per-position SL triggered | 2024-06-21 10:50:00 | 339.08 | 339.64 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 10:00:00 | 336.50 | 337.72 | 0.00 | ORB-short ORB[337.60,338.85] vol=1.9x ATR=0.66 |
| Stop hit — per-position SL triggered | 2024-06-25 10:05:00 | 337.16 | 337.69 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-06-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-28 09:35:00 | 332.80 | 334.10 | 0.00 | ORB-short ORB[333.75,335.70] vol=1.8x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-06-28 09:40:00 | 333.75 | 333.98 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 331.90 | 332.73 | 0.00 | ORB-short ORB[332.10,333.80] vol=1.7x ATR=0.65 |
| Stop hit — per-position SL triggered | 2024-07-01 09:35:00 | 332.55 | 332.68 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 09:40:00 | 336.45 | 334.97 | 0.00 | ORB-long ORB[333.60,336.15] vol=2.5x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-02 09:50:00 | 337.86 | 335.85 | 0.00 | T1 1.5R @ 337.86 |
| Target hit | 2024-07-02 11:00:00 | 339.50 | 339.53 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2024-07-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 09:55:00 | 335.20 | 335.94 | 0.00 | ORB-short ORB[335.55,337.70] vol=1.7x ATR=0.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-04 10:30:00 | 334.04 | 335.62 | 0.00 | T1 1.5R @ 334.04 |
| Stop hit — per-position SL triggered | 2024-07-04 14:25:00 | 335.20 | 334.78 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-05 09:55:00 | 332.50 | 333.37 | 0.00 | ORB-short ORB[332.85,335.20] vol=1.9x ATR=0.81 |
| Stop hit — per-position SL triggered | 2024-07-05 10:30:00 | 333.31 | 333.20 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 09:30:00 | 343.55 | 342.34 | 0.00 | ORB-long ORB[338.90,342.90] vol=5.0x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-07-09 09:45:00 | 342.10 | 342.72 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 10:55:00 | 325.75 | 327.99 | 0.00 | ORB-short ORB[327.00,331.30] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-07-18 11:05:00 | 326.62 | 327.91 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:45:00 | 321.30 | 323.21 | 0.00 | ORB-short ORB[322.10,326.70] vol=1.5x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:10:00 | 319.83 | 322.20 | 0.00 | T1 1.5R @ 319.83 |
| Stop hit — per-position SL triggered | 2024-07-19 10:20:00 | 321.30 | 321.99 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-07-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-23 11:10:00 | 330.00 | 326.23 | 0.00 | ORB-long ORB[323.20,325.40] vol=8.3x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-07-23 11:15:00 | 328.77 | 326.39 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-07-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 09:45:00 | 327.20 | 326.13 | 0.00 | ORB-long ORB[323.30,326.95] vol=1.6x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-07-24 10:10:00 | 325.71 | 326.16 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-07-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:10:00 | 327.50 | 324.66 | 0.00 | ORB-long ORB[322.15,325.05] vol=4.8x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:15:00 | 329.21 | 325.68 | 0.00 | T1 1.5R @ 329.21 |
| Stop hit — per-position SL triggered | 2024-07-25 10:20:00 | 327.50 | 325.79 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-07-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 10:40:00 | 327.90 | 326.04 | 0.00 | ORB-long ORB[324.55,326.70] vol=2.0x ATR=0.92 |
| Stop hit — per-position SL triggered | 2024-07-26 10:45:00 | 326.98 | 326.11 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-07-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:30:00 | 330.70 | 328.89 | 0.00 | ORB-long ORB[327.10,329.35] vol=1.9x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-07-29 09:40:00 | 329.36 | 329.08 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:50:00 | 344.85 | 347.42 | 0.00 | ORB-short ORB[346.10,350.70] vol=1.6x ATR=1.17 |
| Stop hit — per-position SL triggered | 2024-08-01 11:25:00 | 346.02 | 347.25 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 368.35 | 364.34 | 0.00 | ORB-long ORB[361.55,365.85] vol=1.9x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-08-19 10:00:00 | 366.16 | 365.59 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-08-23 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 10:00:00 | 387.25 | 383.28 | 0.00 | ORB-long ORB[380.00,384.35] vol=1.7x ATR=2.36 |
| Stop hit — per-position SL triggered | 2024-08-23 10:10:00 | 384.89 | 383.65 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:30:00 | 381.95 | 378.82 | 0.00 | ORB-long ORB[376.00,378.90] vol=3.7x ATR=1.66 |
| Stop hit — per-position SL triggered | 2024-08-26 09:35:00 | 380.29 | 380.07 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-28 09:30:00 | 374.00 | 375.33 | 0.00 | ORB-short ORB[374.05,377.85] vol=1.5x ATR=1.48 |
| Stop hit — per-position SL triggered | 2024-08-28 09:45:00 | 375.48 | 375.16 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-29 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 10:10:00 | 365.35 | 368.00 | 0.00 | ORB-short ORB[367.55,370.30] vol=1.9x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 11:40:00 | 362.97 | 366.80 | 0.00 | T1 1.5R @ 362.97 |
| Stop hit — per-position SL triggered | 2024-08-29 12:40:00 | 365.35 | 366.22 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-09-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 09:50:00 | 369.00 | 365.96 | 0.00 | ORB-long ORB[362.10,365.95] vol=4.4x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 10:00:00 | 371.44 | 368.01 | 0.00 | T1 1.5R @ 371.44 |
| Stop hit — per-position SL triggered | 2024-09-02 10:20:00 | 369.00 | 368.84 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:30:00 | 372.35 | 368.97 | 0.00 | ORB-long ORB[365.10,369.90] vol=2.9x ATR=1.30 |
| Stop hit — per-position SL triggered | 2024-09-04 09:35:00 | 371.05 | 369.89 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:50:00 | 363.90 | 366.73 | 0.00 | ORB-short ORB[366.50,369.30] vol=2.2x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 10:05:00 | 361.98 | 365.79 | 0.00 | T1 1.5R @ 361.98 |
| Stop hit — per-position SL triggered | 2024-09-06 11:35:00 | 363.90 | 364.48 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-09-19 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:55:00 | 350.70 | 353.94 | 0.00 | ORB-short ORB[353.60,357.40] vol=2.1x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:05:00 | 348.98 | 352.72 | 0.00 | T1 1.5R @ 348.98 |
| Target hit | 2024-09-19 15:20:00 | 345.00 | 347.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-09-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 09:30:00 | 352.00 | 350.77 | 0.00 | ORB-long ORB[348.40,351.95] vol=2.3x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-23 09:45:00 | 354.06 | 351.90 | 0.00 | T1 1.5R @ 354.06 |
| Stop hit — per-position SL triggered | 2024-09-23 09:55:00 | 352.00 | 352.01 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-09-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 09:50:00 | 346.60 | 348.30 | 0.00 | ORB-short ORB[347.25,350.80] vol=2.1x ATR=0.93 |
| Stop hit — per-position SL triggered | 2024-09-25 09:55:00 | 347.53 | 348.26 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-26 09:40:00 | 343.70 | 344.92 | 0.00 | ORB-short ORB[344.55,347.50] vol=1.9x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-09-26 10:05:00 | 344.59 | 344.55 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-30 10:50:00 | 346.00 | 343.34 | 0.00 | ORB-long ORB[341.10,344.90] vol=1.7x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-09-30 11:00:00 | 344.68 | 343.44 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 09:40:00 | 354.95 | 352.85 | 0.00 | ORB-long ORB[349.90,353.00] vol=3.4x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-10-01 09:45:00 | 353.55 | 353.13 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 09:50:00 | 342.10 | 341.13 | 0.00 | ORB-long ORB[339.60,341.80] vol=1.9x ATR=1.13 |
| Stop hit — per-position SL triggered | 2024-10-09 10:25:00 | 340.97 | 341.35 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-10-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 10:25:00 | 330.60 | 326.00 | 0.00 | ORB-long ORB[322.30,327.20] vol=1.9x ATR=1.34 |
| Stop hit — per-position SL triggered | 2024-10-18 10:35:00 | 329.26 | 326.33 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-10-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:30:00 | 328.25 | 329.50 | 0.00 | ORB-short ORB[328.50,332.35] vol=1.5x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:55:00 | 326.36 | 328.56 | 0.00 | T1 1.5R @ 326.36 |
| Stop hit — per-position SL triggered | 2024-10-21 10:05:00 | 328.25 | 328.48 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 09:35:00 | 319.95 | 321.47 | 0.00 | ORB-short ORB[320.30,323.95] vol=2.2x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-24 10:00:00 | 317.57 | 320.72 | 0.00 | T1 1.5R @ 317.57 |
| Stop hit — per-position SL triggered | 2024-10-24 10:05:00 | 319.95 | 320.55 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-11-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:30:00 | 329.65 | 328.09 | 0.00 | ORB-long ORB[326.00,328.90] vol=1.9x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-11-19 09:40:00 | 328.61 | 328.31 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-12-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 11:05:00 | 314.10 | 312.71 | 0.00 | ORB-long ORB[311.00,313.65] vol=3.4x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 11:40:00 | 315.54 | 313.28 | 0.00 | T1 1.5R @ 315.54 |
| Stop hit — per-position SL triggered | 2024-12-03 12:20:00 | 314.10 | 313.86 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2024-12-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-04 10:10:00 | 312.85 | 314.11 | 0.00 | ORB-short ORB[312.95,315.95] vol=1.5x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-04 11:55:00 | 311.36 | 313.38 | 0.00 | T1 1.5R @ 311.36 |
| Stop hit — per-position SL triggered | 2024-12-04 14:15:00 | 312.85 | 313.06 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-12-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:45:00 | 312.90 | 311.73 | 0.00 | ORB-long ORB[309.10,312.70] vol=2.0x ATR=1.04 |
| Stop hit — per-position SL triggered | 2024-12-06 09:55:00 | 311.86 | 311.78 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-12-12 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 10:30:00 | 310.35 | 306.86 | 0.00 | ORB-long ORB[305.15,308.00] vol=5.0x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-12-12 10:35:00 | 309.40 | 307.38 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-12-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-18 09:55:00 | 301.35 | 302.38 | 0.00 | ORB-short ORB[301.50,304.45] vol=1.7x ATR=0.94 |
| Stop hit — per-position SL triggered | 2024-12-18 10:15:00 | 302.29 | 302.25 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2024-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 09:35:00 | 297.95 | 296.32 | 0.00 | ORB-long ORB[294.05,297.50] vol=1.9x ATR=1.05 |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 296.90 | 296.73 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2024-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:50:00 | 330.80 | 329.42 | 0.00 | ORB-long ORB[325.75,329.50] vol=2.8x ATR=1.87 |
| Stop hit — per-position SL triggered | 2024-12-30 11:10:00 | 328.93 | 329.46 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-01-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 10:40:00 | 266.75 | 268.38 | 0.00 | ORB-short ORB[267.05,269.90] vol=1.5x ATR=0.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 10:45:00 | 265.29 | 267.93 | 0.00 | T1 1.5R @ 265.29 |
| Target hit | 2025-01-20 15:20:00 | 264.10 | 264.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2025-01-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-21 09:50:00 | 270.00 | 267.67 | 0.00 | ORB-long ORB[264.45,267.40] vol=2.8x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-01-21 10:00:00 | 268.98 | 268.01 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-01-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-23 10:40:00 | 263.80 | 264.91 | 0.00 | ORB-short ORB[264.15,267.00] vol=3.4x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 11:45:00 | 262.51 | 264.42 | 0.00 | T1 1.5R @ 262.51 |
| Target hit | 2025-01-23 15:20:00 | 261.45 | 262.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2025-01-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:55:00 | 264.25 | 261.80 | 0.00 | ORB-long ORB[258.60,262.55] vol=1.6x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:10:00 | 266.33 | 262.82 | 0.00 | T1 1.5R @ 266.33 |
| Stop hit — per-position SL triggered | 2025-01-29 11:25:00 | 264.25 | 263.72 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-01-31 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 09:30:00 | 262.00 | 260.83 | 0.00 | ORB-long ORB[258.00,261.90] vol=1.7x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 09:35:00 | 263.37 | 261.23 | 0.00 | T1 1.5R @ 263.37 |
| Stop hit — per-position SL triggered | 2025-01-31 10:05:00 | 262.00 | 262.11 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2025-02-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 10:35:00 | 269.50 | 268.26 | 0.00 | ORB-long ORB[266.40,268.80] vol=3.4x ATR=0.78 |
| Stop hit — per-position SL triggered | 2025-02-01 10:40:00 | 268.72 | 268.29 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-02-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-04 09:35:00 | 268.30 | 267.60 | 0.00 | ORB-long ORB[265.10,268.10] vol=1.9x ATR=0.94 |
| Stop hit — per-position SL triggered | 2025-02-04 10:05:00 | 267.36 | 267.76 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-10 09:30:00 | 262.65 | 264.27 | 0.00 | ORB-short ORB[263.20,266.75] vol=1.6x ATR=1.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-10 09:50:00 | 260.98 | 262.65 | 0.00 | T1 1.5R @ 260.98 |
| Target hit | 2025-02-10 15:20:00 | 258.10 | 260.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2025-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:30:00 | 243.35 | 241.35 | 0.00 | ORB-long ORB[239.15,242.55] vol=2.5x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-02-13 09:35:00 | 242.24 | 241.49 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-02-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-18 09:55:00 | 238.80 | 237.35 | 0.00 | ORB-long ORB[235.75,238.25] vol=2.8x ATR=1.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 10:00:00 | 240.69 | 237.94 | 0.00 | T1 1.5R @ 240.69 |
| Target hit | 2025-02-18 15:20:00 | 256.80 | 249.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 62 — SELL (started 2025-03-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 09:50:00 | 247.01 | 248.61 | 0.00 | ORB-short ORB[247.07,250.00] vol=1.6x ATR=1.17 |
| Stop hit — per-position SL triggered | 2025-03-17 09:55:00 | 248.18 | 248.55 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-03-20 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 10:05:00 | 254.80 | 257.90 | 0.00 | ORB-short ORB[257.11,259.95] vol=1.6x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-03-20 10:20:00 | 256.02 | 257.73 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-04-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 11:05:00 | 275.55 | 272.84 | 0.00 | ORB-long ORB[270.90,274.90] vol=5.0x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-04-08 11:25:00 | 274.25 | 273.28 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-04-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-11 09:30:00 | 277.50 | 275.59 | 0.00 | ORB-long ORB[273.20,276.30] vol=2.2x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-11 09:35:00 | 278.94 | 276.64 | 0.00 | T1 1.5R @ 278.94 |
| Stop hit — per-position SL triggered | 2025-04-11 09:45:00 | 277.50 | 276.85 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2025-04-15 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 10:55:00 | 278.75 | 276.90 | 0.00 | ORB-long ORB[274.50,276.70] vol=9.2x ATR=0.87 |
| Stop hit — per-position SL triggered | 2025-04-15 11:05:00 | 277.88 | 277.36 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-04-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:40:00 | 289.80 | 286.56 | 0.00 | ORB-long ORB[281.75,285.00] vol=5.8x ATR=1.44 |
| Stop hit — per-position SL triggered | 2025-04-21 09:45:00 | 288.36 | 286.87 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:30:00 | 288.40 | 286.72 | 0.00 | ORB-long ORB[284.70,287.70] vol=2.5x ATR=0.89 |
| Stop hit — per-position SL triggered | 2025-04-22 09:50:00 | 287.51 | 287.20 | 0.00 | SL hit |

### Cycle 69 — SELL (started 2025-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-23 11:15:00 | 281.15 | 281.84 | 0.00 | ORB-short ORB[283.75,285.90] vol=1.7x ATR=0.81 |
| Stop hit — per-position SL triggered | 2025-04-23 12:45:00 | 281.96 | 281.72 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-04-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 09:30:00 | 286.00 | 284.66 | 0.00 | ORB-long ORB[282.10,285.10] vol=1.7x ATR=0.85 |
| Stop hit — per-position SL triggered | 2025-04-24 09:45:00 | 285.15 | 284.90 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2025-04-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:30:00 | 279.40 | 281.79 | 0.00 | ORB-short ORB[280.80,284.45] vol=2.2x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-04-25 09:35:00 | 280.19 | 280.94 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2025-04-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-28 09:30:00 | 276.00 | 273.30 | 0.00 | ORB-long ORB[271.00,274.30] vol=1.9x ATR=1.29 |
| Stop hit — per-position SL triggered | 2025-04-28 09:40:00 | 274.71 | 273.72 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-04-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:40:00 | 272.55 | 275.13 | 0.00 | ORB-short ORB[273.20,277.00] vol=2.0x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 10:10:00 | 270.64 | 274.04 | 0.00 | T1 1.5R @ 270.64 |
| Target hit | 2025-04-29 14:00:00 | 267.50 | 267.46 | 0.00 | Trail-exit close>VWAP |

### Cycle 74 — BUY (started 2025-05-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:10:00 | 272.35 | 270.59 | 0.00 | ORB-long ORB[268.50,271.45] vol=1.7x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-05 10:45:00 | 274.01 | 271.40 | 0.00 | T1 1.5R @ 274.01 |
| Target hit | 2025-05-05 13:25:00 | 275.35 | 275.49 | 0.00 | Trail-exit close<VWAP |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-14 11:15:00 | 335.25 | 2024-05-14 11:25:00 | 336.85 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-05-14 11:15:00 | 335.25 | 2024-05-14 11:30:00 | 335.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-17 10:20:00 | 337.80 | 2024-05-17 11:20:00 | 336.81 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-05-21 10:05:00 | 341.20 | 2024-05-21 10:10:00 | 340.10 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-05-23 11:15:00 | 348.45 | 2024-05-23 11:20:00 | 346.87 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-05-28 09:35:00 | 338.85 | 2024-05-28 10:10:00 | 337.22 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-05-28 09:35:00 | 338.85 | 2024-05-28 10:15:00 | 338.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-31 09:35:00 | 351.70 | 2024-05-31 09:40:00 | 350.16 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-06-11 09:35:00 | 344.85 | 2024-06-11 09:45:00 | 345.90 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-06-14 11:10:00 | 346.55 | 2024-06-14 11:20:00 | 345.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-06-20 10:10:00 | 342.30 | 2024-06-20 10:40:00 | 341.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-06-21 10:45:00 | 338.35 | 2024-06-21 10:50:00 | 339.08 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-06-25 10:00:00 | 336.50 | 2024-06-25 10:05:00 | 337.16 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-06-28 09:35:00 | 332.80 | 2024-06-28 09:40:00 | 333.75 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-01 09:30:00 | 331.90 | 2024-07-01 09:35:00 | 332.55 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-07-02 09:40:00 | 336.45 | 2024-07-02 09:50:00 | 337.86 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-07-02 09:40:00 | 336.45 | 2024-07-02 11:00:00 | 339.50 | TARGET_HIT | 0.50 | 0.91% |
| SELL | retest1 | 2024-07-04 09:55:00 | 335.20 | 2024-07-04 10:30:00 | 334.04 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-07-04 09:55:00 | 335.20 | 2024-07-04 14:25:00 | 335.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-05 09:55:00 | 332.50 | 2024-07-05 10:30:00 | 333.31 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-09 09:30:00 | 343.55 | 2024-07-09 09:45:00 | 342.10 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-07-18 10:55:00 | 325.75 | 2024-07-18 11:05:00 | 326.62 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-19 09:45:00 | 321.30 | 2024-07-19 10:10:00 | 319.83 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2024-07-19 09:45:00 | 321.30 | 2024-07-19 10:20:00 | 321.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-23 11:10:00 | 330.00 | 2024-07-23 11:15:00 | 328.77 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-07-24 09:45:00 | 327.20 | 2024-07-24 10:10:00 | 325.71 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-07-25 10:10:00 | 327.50 | 2024-07-25 10:15:00 | 329.21 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-07-25 10:10:00 | 327.50 | 2024-07-25 10:20:00 | 327.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 10:40:00 | 327.90 | 2024-07-26 10:45:00 | 326.98 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-29 09:30:00 | 330.70 | 2024-07-29 09:40:00 | 329.36 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-08-01 10:50:00 | 344.85 | 2024-08-01 11:25:00 | 346.02 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-19 09:35:00 | 368.35 | 2024-08-19 10:00:00 | 366.16 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest1 | 2024-08-23 10:00:00 | 387.25 | 2024-08-23 10:10:00 | 384.89 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2024-08-26 09:30:00 | 381.95 | 2024-08-26 09:35:00 | 380.29 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-08-28 09:30:00 | 374.00 | 2024-08-28 09:45:00 | 375.48 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-08-29 10:10:00 | 365.35 | 2024-08-29 11:40:00 | 362.97 | PARTIAL | 0.50 | 0.65% |
| SELL | retest1 | 2024-08-29 10:10:00 | 365.35 | 2024-08-29 12:40:00 | 365.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-02 09:50:00 | 369.00 | 2024-09-02 10:00:00 | 371.44 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-09-02 09:50:00 | 369.00 | 2024-09-02 10:20:00 | 369.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-04 09:30:00 | 372.35 | 2024-09-04 09:35:00 | 371.05 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-09-06 09:50:00 | 363.90 | 2024-09-06 10:05:00 | 361.98 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-09-06 09:50:00 | 363.90 | 2024-09-06 11:35:00 | 363.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-19 09:55:00 | 350.70 | 2024-09-19 10:05:00 | 348.98 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2024-09-19 09:55:00 | 350.70 | 2024-09-19 15:20:00 | 345.00 | TARGET_HIT | 0.50 | 1.63% |
| BUY | retest1 | 2024-09-23 09:30:00 | 352.00 | 2024-09-23 09:45:00 | 354.06 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-09-23 09:30:00 | 352.00 | 2024-09-23 09:55:00 | 352.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-25 09:50:00 | 346.60 | 2024-09-25 09:55:00 | 347.53 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-26 09:40:00 | 343.70 | 2024-09-26 10:05:00 | 344.59 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-09-30 10:50:00 | 346.00 | 2024-09-30 11:00:00 | 344.68 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-10-01 09:40:00 | 354.95 | 2024-10-01 09:45:00 | 353.55 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-10-09 09:50:00 | 342.10 | 2024-10-09 10:25:00 | 340.97 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-10-18 10:25:00 | 330.60 | 2024-10-18 10:35:00 | 329.26 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2024-10-21 09:30:00 | 328.25 | 2024-10-21 09:55:00 | 326.36 | PARTIAL | 0.50 | 0.58% |
| SELL | retest1 | 2024-10-21 09:30:00 | 328.25 | 2024-10-21 10:05:00 | 328.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-24 09:35:00 | 319.95 | 2024-10-24 10:00:00 | 317.57 | PARTIAL | 0.50 | 0.74% |
| SELL | retest1 | 2024-10-24 09:35:00 | 319.95 | 2024-10-24 10:05:00 | 319.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-19 09:30:00 | 329.65 | 2024-11-19 09:40:00 | 328.61 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-03 11:05:00 | 314.10 | 2024-12-03 11:40:00 | 315.54 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-12-03 11:05:00 | 314.10 | 2024-12-03 12:20:00 | 314.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-04 10:10:00 | 312.85 | 2024-12-04 11:55:00 | 311.36 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-12-04 10:10:00 | 312.85 | 2024-12-04 14:15:00 | 312.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-06 09:45:00 | 312.90 | 2024-12-06 09:55:00 | 311.86 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-12-12 10:30:00 | 310.35 | 2024-12-12 10:35:00 | 309.40 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-18 09:55:00 | 301.35 | 2024-12-18 10:15:00 | 302.29 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-19 09:35:00 | 297.95 | 2024-12-19 10:15:00 | 296.90 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-30 10:50:00 | 330.80 | 2024-12-30 11:10:00 | 328.93 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2025-01-20 10:40:00 | 266.75 | 2025-01-20 10:45:00 | 265.29 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2025-01-20 10:40:00 | 266.75 | 2025-01-20 15:20:00 | 264.10 | TARGET_HIT | 0.50 | 0.99% |
| BUY | retest1 | 2025-01-21 09:50:00 | 270.00 | 2025-01-21 10:00:00 | 268.98 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-23 10:40:00 | 263.80 | 2025-01-23 11:45:00 | 262.51 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-01-23 10:40:00 | 263.80 | 2025-01-23 15:20:00 | 261.45 | TARGET_HIT | 0.50 | 0.89% |
| BUY | retest1 | 2025-01-29 09:55:00 | 264.25 | 2025-01-29 10:10:00 | 266.33 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2025-01-29 09:55:00 | 264.25 | 2025-01-29 11:25:00 | 264.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-31 09:30:00 | 262.00 | 2025-01-31 09:35:00 | 263.37 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-01-31 09:30:00 | 262.00 | 2025-01-31 10:05:00 | 262.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-01 10:35:00 | 269.50 | 2025-02-01 10:40:00 | 268.72 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-02-04 09:35:00 | 268.30 | 2025-02-04 10:05:00 | 267.36 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-02-10 09:30:00 | 262.65 | 2025-02-10 09:50:00 | 260.98 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2025-02-10 09:30:00 | 262.65 | 2025-02-10 15:20:00 | 258.10 | TARGET_HIT | 0.50 | 1.73% |
| BUY | retest1 | 2025-02-13 09:30:00 | 243.35 | 2025-02-13 09:35:00 | 242.24 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-02-18 09:55:00 | 238.80 | 2025-02-18 10:00:00 | 240.69 | PARTIAL | 0.50 | 0.79% |
| BUY | retest1 | 2025-02-18 09:55:00 | 238.80 | 2025-02-18 15:20:00 | 256.80 | TARGET_HIT | 0.50 | 7.54% |
| SELL | retest1 | 2025-03-17 09:50:00 | 247.01 | 2025-03-17 09:55:00 | 248.18 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-03-20 10:05:00 | 254.80 | 2025-03-20 10:20:00 | 256.02 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-04-08 11:05:00 | 275.55 | 2025-04-08 11:25:00 | 274.25 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-04-11 09:30:00 | 277.50 | 2025-04-11 09:35:00 | 278.94 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2025-04-11 09:30:00 | 277.50 | 2025-04-11 09:45:00 | 277.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-15 10:55:00 | 278.75 | 2025-04-15 11:05:00 | 277.88 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-04-21 09:40:00 | 289.80 | 2025-04-21 09:45:00 | 288.36 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2025-04-22 09:30:00 | 288.40 | 2025-04-22 09:50:00 | 287.51 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-04-23 11:15:00 | 281.15 | 2025-04-23 12:45:00 | 281.96 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-04-24 09:30:00 | 286.00 | 2025-04-24 09:45:00 | 285.15 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-04-25 09:30:00 | 279.40 | 2025-04-25 09:35:00 | 280.19 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-04-28 09:30:00 | 276.00 | 2025-04-28 09:40:00 | 274.71 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-04-29 09:40:00 | 272.55 | 2025-04-29 10:10:00 | 270.64 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2025-04-29 09:40:00 | 272.55 | 2025-04-29 14:00:00 | 267.50 | TARGET_HIT | 0.50 | 1.85% |
| BUY | retest1 | 2025-05-05 10:10:00 | 272.35 | 2025-05-05 10:45:00 | 274.01 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-05-05 10:10:00 | 272.35 | 2025-05-05 13:25:00 | 275.35 | TARGET_HIT | 0.50 | 1.10% |
