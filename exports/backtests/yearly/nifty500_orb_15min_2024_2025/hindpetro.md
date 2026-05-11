# Hindustan Petroleum Corporation Ltd. (HINDPETRO)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 387.50
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
| ENTRY1 | 57 |
| ENTRY2 | 0 |
| PARTIAL | 23 |
| TARGET_HIT | 12 |
| STOP_HIT | 45 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 80 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 45
- **Target hits / Stop hits / Partials:** 12 / 45 / 23
- **Avg / median % per leg:** 0.25% / 0.00%
- **Sum % (uncompounded):** 20.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 48 | 21 | 43.8% | 7 | 27 | 14 | 0.23% | 10.9% |
| BUY @ 2nd Alert (retest1) | 48 | 21 | 43.8% | 7 | 27 | 14 | 0.23% | 10.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 32 | 14 | 43.8% | 5 | 18 | 9 | 0.30% | 9.4% |
| SELL @ 2nd Alert (retest1) | 32 | 14 | 43.8% | 5 | 18 | 9 | 0.30% | 9.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 80 | 35 | 43.8% | 12 | 45 | 23 | 0.25% | 20.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 09:40:00 | 341.03 | 338.38 | 0.00 | ORB-long ORB[334.47,339.47] vol=4.7x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-21 09:50:00 | 343.24 | 340.58 | 0.00 | T1 1.5R @ 343.24 |
| Target hit | 2024-05-21 15:20:00 | 351.17 | 347.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 2 — BUY (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:15:00 | 361.87 | 357.52 | 0.00 | ORB-long ORB[353.50,358.50] vol=2.6x ATR=1.61 |
| Stop hit — per-position SL triggered | 2024-05-23 10:25:00 | 360.26 | 357.97 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 09:40:00 | 352.53 | 355.02 | 0.00 | ORB-short ORB[354.47,358.67] vol=2.3x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-05-24 09:45:00 | 354.05 | 354.91 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:35:00 | 355.50 | 357.14 | 0.00 | ORB-short ORB[356.30,360.07] vol=2.1x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-05-30 09:40:00 | 356.72 | 357.10 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-31 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-31 10:00:00 | 357.13 | 359.75 | 0.00 | ORB-short ORB[358.23,362.53] vol=1.5x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-05-31 10:45:00 | 358.80 | 359.32 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 358.30 | 355.90 | 0.00 | ORB-long ORB[353.33,356.67] vol=3.2x ATR=1.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 09:35:00 | 359.86 | 357.89 | 0.00 | T1 1.5R @ 359.86 |
| Stop hit — per-position SL triggered | 2024-06-14 10:10:00 | 358.30 | 358.76 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-06-25 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:05:00 | 334.80 | 337.20 | 0.00 | ORB-short ORB[337.15,340.15] vol=1.9x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-25 11:15:00 | 333.41 | 336.94 | 0.00 | T1 1.5R @ 333.41 |
| Stop hit — per-position SL triggered | 2024-06-25 14:05:00 | 334.80 | 335.03 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-06-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:10:00 | 336.05 | 334.65 | 0.00 | ORB-long ORB[332.35,335.80] vol=2.1x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-06-26 12:20:00 | 335.10 | 335.02 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-06-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:10:00 | 332.00 | 333.17 | 0.00 | ORB-short ORB[332.10,334.65] vol=1.9x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-06-27 10:25:00 | 332.78 | 332.82 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:30:00 | 333.55 | 331.94 | 0.00 | ORB-long ORB[330.25,332.10] vol=1.9x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-07-01 10:40:00 | 332.68 | 332.02 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-03 11:00:00 | 327.00 | 327.78 | 0.00 | ORB-short ORB[327.45,329.85] vol=6.9x ATR=0.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 11:10:00 | 325.72 | 327.67 | 0.00 | T1 1.5R @ 325.72 |
| Stop hit — per-position SL triggered | 2024-07-03 11:15:00 | 327.00 | 327.62 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:00:00 | 329.75 | 329.10 | 0.00 | ORB-long ORB[327.50,329.70] vol=3.1x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-05 10:10:00 | 331.16 | 329.69 | 0.00 | T1 1.5R @ 331.16 |
| Target hit | 2024-07-05 10:50:00 | 330.80 | 330.81 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2024-07-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 10:30:00 | 331.10 | 333.09 | 0.00 | ORB-short ORB[331.35,335.95] vol=1.7x ATR=1.27 |
| Stop hit — per-position SL triggered | 2024-07-08 10:40:00 | 332.37 | 333.04 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-07-10 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:55:00 | 328.15 | 329.88 | 0.00 | ORB-short ORB[329.20,332.80] vol=2.0x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 10:05:00 | 326.27 | 329.31 | 0.00 | T1 1.5R @ 326.27 |
| Target hit | 2024-07-10 10:45:00 | 326.60 | 326.55 | 0.00 | Trail-exit close>VWAP |

### Cycle 15 — BUY (started 2024-07-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:45:00 | 345.80 | 343.30 | 0.00 | ORB-long ORB[341.40,345.00] vol=2.5x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:00:00 | 347.74 | 344.19 | 0.00 | T1 1.5R @ 347.74 |
| Stop hit — per-position SL triggered | 2024-07-15 10:20:00 | 345.80 | 344.87 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 361.55 | 360.50 | 0.00 | ORB-long ORB[357.30,361.45] vol=3.5x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-07-18 09:35:00 | 360.08 | 360.47 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:30:00 | 358.65 | 352.50 | 0.00 | ORB-long ORB[347.15,352.00] vol=2.6x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:40:00 | 361.12 | 353.93 | 0.00 | T1 1.5R @ 361.12 |
| Target hit | 2024-07-25 15:20:00 | 373.60 | 367.64 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — BUY (started 2024-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:40:00 | 380.95 | 378.06 | 0.00 | ORB-long ORB[375.10,378.00] vol=3.1x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-08-16 09:45:00 | 379.46 | 378.38 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-21 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:00:00 | 399.80 | 397.04 | 0.00 | ORB-long ORB[394.00,397.75] vol=1.6x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-21 11:05:00 | 401.42 | 397.55 | 0.00 | T1 1.5R @ 401.42 |
| Stop hit — per-position SL triggered | 2024-08-21 11:10:00 | 399.80 | 397.95 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:50:00 | 410.10 | 408.44 | 0.00 | ORB-long ORB[406.00,410.00] vol=1.7x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-08-23 10:00:00 | 408.87 | 408.49 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-26 11:15:00 | 401.15 | 403.44 | 0.00 | ORB-short ORB[403.20,408.80] vol=2.6x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-08-26 11:35:00 | 402.26 | 403.19 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 398.80 | 400.59 | 0.00 | ORB-short ORB[399.70,403.55] vol=2.0x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-08-27 09:45:00 | 400.20 | 400.17 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:45:00 | 409.70 | 407.97 | 0.00 | ORB-long ORB[405.10,408.95] vol=2.6x ATR=1.50 |
| Stop hit — per-position SL triggered | 2024-08-28 09:50:00 | 408.20 | 408.04 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-08-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 11:10:00 | 402.75 | 404.28 | 0.00 | ORB-short ORB[405.05,408.00] vol=6.6x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-08-29 11:25:00 | 403.71 | 404.15 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2024-09-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:40:00 | 430.40 | 425.48 | 0.00 | ORB-long ORB[422.00,426.00] vol=2.1x ATR=1.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 11:00:00 | 432.99 | 427.86 | 0.00 | T1 1.5R @ 432.99 |
| Stop hit — per-position SL triggered | 2024-09-02 13:05:00 | 430.40 | 431.77 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 09:30:00 | 434.60 | 431.92 | 0.00 | ORB-long ORB[428.15,434.40] vol=3.3x ATR=1.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:40:00 | 437.50 | 433.31 | 0.00 | T1 1.5R @ 437.50 |
| Target hit | 2024-09-04 10:25:00 | 436.30 | 437.24 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — BUY (started 2024-09-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 10:55:00 | 452.20 | 447.85 | 0.00 | ORB-long ORB[444.25,448.60] vol=2.2x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-05 11:05:00 | 454.79 | 448.38 | 0.00 | T1 1.5R @ 454.79 |
| Stop hit — per-position SL triggered | 2024-09-05 11:35:00 | 452.20 | 449.73 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-09-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 10:05:00 | 439.60 | 445.67 | 0.00 | ORB-short ORB[447.00,451.85] vol=1.6x ATR=1.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-06 11:15:00 | 436.65 | 443.32 | 0.00 | T1 1.5R @ 436.65 |
| Stop hit — per-position SL triggered | 2024-09-06 11:25:00 | 439.60 | 443.19 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 11:00:00 | 419.75 | 420.90 | 0.00 | ORB-short ORB[420.30,424.40] vol=2.1x ATR=1.21 |
| Stop hit — per-position SL triggered | 2024-09-10 11:35:00 | 420.96 | 420.64 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-09-13 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:50:00 | 415.25 | 413.74 | 0.00 | ORB-long ORB[410.15,414.85] vol=2.3x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 10:35:00 | 417.52 | 414.66 | 0.00 | T1 1.5R @ 417.52 |
| Stop hit — per-position SL triggered | 2024-09-13 10:50:00 | 415.25 | 414.75 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-09-17 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 10:30:00 | 407.25 | 408.06 | 0.00 | ORB-short ORB[408.10,411.75] vol=1.5x ATR=1.08 |
| Stop hit — per-position SL triggered | 2024-09-17 10:35:00 | 408.33 | 408.04 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 404.70 | 406.43 | 0.00 | ORB-short ORB[406.15,408.70] vol=2.4x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:40:00 | 402.98 | 405.16 | 0.00 | T1 1.5R @ 402.98 |
| Target hit | 2024-09-19 15:05:00 | 398.60 | 398.30 | 0.00 | Trail-exit close>VWAP |

### Cycle 33 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 11:15:00 | 408.00 | 403.93 | 0.00 | ORB-long ORB[396.65,399.50] vol=3.1x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-09-23 11:25:00 | 406.53 | 404.08 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:55:00 | 411.45 | 407.45 | 0.00 | ORB-long ORB[402.65,406.30] vol=2.0x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-09-24 10:10:00 | 410.03 | 408.71 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:50:00 | 430.60 | 425.74 | 0.00 | ORB-long ORB[421.20,425.80] vol=1.7x ATR=1.94 |
| Stop hit — per-position SL triggered | 2024-09-27 11:15:00 | 428.66 | 426.40 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-10-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:45:00 | 393.30 | 402.63 | 0.00 | ORB-short ORB[403.65,409.35] vol=1.8x ATR=1.99 |
| Stop hit — per-position SL triggered | 2024-10-07 11:10:00 | 395.29 | 399.69 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-10-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:35:00 | 401.00 | 398.69 | 0.00 | ORB-long ORB[396.05,399.00] vol=2.0x ATR=1.45 |
| Stop hit — per-position SL triggered | 2024-10-14 10:35:00 | 399.55 | 400.16 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:35:00 | 428.20 | 432.13 | 0.00 | ORB-short ORB[432.05,436.30] vol=3.5x ATR=1.69 |
| Stop hit — per-position SL triggered | 2024-10-21 09:55:00 | 429.89 | 429.70 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 09:30:00 | 395.50 | 396.92 | 0.00 | ORB-short ORB[395.85,400.00] vol=1.8x ATR=1.50 |
| Stop hit — per-position SL triggered | 2024-10-24 09:55:00 | 397.00 | 396.08 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 383.65 | 380.54 | 0.00 | ORB-long ORB[377.80,381.90] vol=2.0x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 10:15:00 | 385.73 | 382.45 | 0.00 | T1 1.5R @ 385.73 |
| Stop hit — per-position SL triggered | 2024-11-28 10:30:00 | 383.65 | 382.87 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2024-12-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-03 10:10:00 | 385.75 | 384.58 | 0.00 | ORB-long ORB[382.55,385.00] vol=1.5x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-12-03 10:20:00 | 384.66 | 384.65 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-12-06 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:50:00 | 399.95 | 396.97 | 0.00 | ORB-long ORB[392.05,397.25] vol=2.2x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-12-06 09:55:00 | 398.41 | 397.29 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-19 10:30:00 | 401.45 | 396.64 | 0.00 | ORB-long ORB[388.85,393.90] vol=1.9x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 10:55:00 | 403.87 | 398.23 | 0.00 | T1 1.5R @ 403.87 |
| Target hit | 2024-12-19 15:20:00 | 407.50 | 406.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — BUY (started 2024-12-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:40:00 | 412.85 | 409.98 | 0.00 | ORB-long ORB[407.70,411.45] vol=1.6x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-12-20 09:50:00 | 411.31 | 410.56 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2024-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:15:00 | 409.50 | 403.18 | 0.00 | ORB-long ORB[399.90,404.65] vol=2.2x ATR=1.52 |
| Stop hit — per-position SL triggered | 2024-12-23 12:15:00 | 407.98 | 405.40 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-12-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:05:00 | 410.00 | 406.73 | 0.00 | ORB-long ORB[401.60,405.70] vol=2.1x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-24 10:30:00 | 412.22 | 407.79 | 0.00 | T1 1.5R @ 412.22 |
| Target hit | 2024-12-24 13:10:00 | 412.25 | 412.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 47 — BUY (started 2024-12-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:35:00 | 417.70 | 415.82 | 0.00 | ORB-long ORB[412.05,416.50] vol=1.8x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-12-26 09:55:00 | 415.94 | 416.19 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:15:00 | 406.05 | 407.44 | 0.00 | ORB-short ORB[407.50,411.50] vol=1.5x ATR=1.30 |
| Stop hit — per-position SL triggered | 2025-01-01 10:50:00 | 407.35 | 407.30 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 10:15:00 | 391.45 | 385.58 | 0.00 | ORB-long ORB[381.70,387.00] vol=1.7x ATR=1.55 |
| Stop hit — per-position SL triggered | 2025-01-08 10:30:00 | 389.90 | 386.33 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-01-15 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 11:00:00 | 372.60 | 374.67 | 0.00 | ORB-short ORB[374.10,379.45] vol=1.7x ATR=1.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 11:30:00 | 370.48 | 374.19 | 0.00 | T1 1.5R @ 370.48 |
| Target hit | 2025-01-15 15:20:00 | 365.00 | 367.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2025-01-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:55:00 | 355.20 | 352.92 | 0.00 | ORB-long ORB[349.40,353.95] vol=2.3x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-31 13:10:00 | 356.80 | 354.02 | 0.00 | T1 1.5R @ 356.80 |
| Target hit | 2025-01-31 15:20:00 | 357.60 | 355.57 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 52 — SELL (started 2025-02-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-01 11:05:00 | 354.20 | 356.57 | 0.00 | ORB-short ORB[354.30,358.35] vol=3.9x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 11:45:00 | 352.60 | 355.36 | 0.00 | T1 1.5R @ 352.60 |
| Target hit | 2025-02-01 15:15:00 | 345.00 | 344.65 | 0.00 | Trail-exit close>VWAP |

### Cycle 53 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 10:15:00 | 325.00 | 327.37 | 0.00 | ORB-short ORB[328.10,331.00] vol=2.7x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 11:25:00 | 322.85 | 326.48 | 0.00 | T1 1.5R @ 322.85 |
| Stop hit — per-position SL triggered | 2025-02-21 11:35:00 | 325.00 | 326.38 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-02-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 10:00:00 | 324.60 | 321.81 | 0.00 | ORB-long ORB[318.40,321.75] vol=2.0x ATR=1.27 |
| Stop hit — per-position SL triggered | 2025-02-24 10:15:00 | 323.33 | 322.26 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-03-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-28 09:50:00 | 367.45 | 364.85 | 0.00 | ORB-long ORB[360.80,366.00] vol=1.6x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-03-28 10:20:00 | 365.79 | 365.77 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-04-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:35:00 | 391.50 | 387.59 | 0.00 | ORB-long ORB[384.50,389.50] vol=1.9x ATR=1.48 |
| Stop hit — per-position SL triggered | 2025-04-21 09:50:00 | 390.02 | 388.61 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-05-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 10:05:00 | 397.55 | 399.44 | 0.00 | ORB-short ORB[398.05,402.60] vol=1.5x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:35:00 | 395.08 | 398.79 | 0.00 | T1 1.5R @ 395.08 |
| Target hit | 2025-05-08 15:20:00 | 385.90 | 393.27 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-21 09:40:00 | 341.03 | 2024-05-21 09:50:00 | 343.24 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-05-21 09:40:00 | 341.03 | 2024-05-21 15:20:00 | 351.17 | TARGET_HIT | 0.50 | 2.97% |
| BUY | retest1 | 2024-05-23 10:15:00 | 361.87 | 2024-05-23 10:25:00 | 360.26 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-05-24 09:40:00 | 352.53 | 2024-05-24 09:45:00 | 354.05 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-05-30 09:35:00 | 355.50 | 2024-05-30 09:40:00 | 356.72 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-05-31 10:00:00 | 357.13 | 2024-05-31 10:45:00 | 358.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-06-14 09:30:00 | 358.30 | 2024-06-14 09:35:00 | 359.86 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-06-14 09:30:00 | 358.30 | 2024-06-14 10:10:00 | 358.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 11:05:00 | 334.80 | 2024-06-25 11:15:00 | 333.41 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2024-06-25 11:05:00 | 334.80 | 2024-06-25 14:05:00 | 334.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-26 11:10:00 | 336.05 | 2024-06-26 12:20:00 | 335.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-06-27 10:10:00 | 332.00 | 2024-06-27 10:25:00 | 332.78 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-07-01 10:30:00 | 333.55 | 2024-07-01 10:40:00 | 332.68 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-07-03 11:00:00 | 327.00 | 2024-07-03 11:10:00 | 325.72 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-03 11:00:00 | 327.00 | 2024-07-03 11:15:00 | 327.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-05 10:00:00 | 329.75 | 2024-07-05 10:10:00 | 331.16 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2024-07-05 10:00:00 | 329.75 | 2024-07-05 10:50:00 | 330.80 | TARGET_HIT | 0.50 | 0.32% |
| SELL | retest1 | 2024-07-08 10:30:00 | 331.10 | 2024-07-08 10:40:00 | 332.37 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-07-10 09:55:00 | 328.15 | 2024-07-10 10:05:00 | 326.27 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-07-10 09:55:00 | 328.15 | 2024-07-10 10:45:00 | 326.60 | TARGET_HIT | 0.50 | 0.47% |
| BUY | retest1 | 2024-07-15 09:45:00 | 345.80 | 2024-07-15 10:00:00 | 347.74 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-07-15 09:45:00 | 345.80 | 2024-07-15 10:20:00 | 345.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-18 09:30:00 | 361.55 | 2024-07-18 09:35:00 | 360.08 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-07-25 10:30:00 | 358.65 | 2024-07-25 10:40:00 | 361.12 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2024-07-25 10:30:00 | 358.65 | 2024-07-25 15:20:00 | 373.60 | TARGET_HIT | 0.50 | 4.17% |
| BUY | retest1 | 2024-08-16 09:40:00 | 380.95 | 2024-08-16 09:45:00 | 379.46 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-08-21 11:00:00 | 399.80 | 2024-08-21 11:05:00 | 401.42 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-21 11:00:00 | 399.80 | 2024-08-21 11:10:00 | 399.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-23 09:50:00 | 410.10 | 2024-08-23 10:00:00 | 408.87 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-08-26 11:15:00 | 401.15 | 2024-08-26 11:35:00 | 402.26 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-08-27 09:30:00 | 398.80 | 2024-08-27 09:45:00 | 400.20 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-08-28 09:45:00 | 409.70 | 2024-08-28 09:50:00 | 408.20 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-08-29 11:10:00 | 402.75 | 2024-08-29 11:25:00 | 403.71 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-09-02 10:40:00 | 430.40 | 2024-09-02 11:00:00 | 432.99 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-09-02 10:40:00 | 430.40 | 2024-09-02 13:05:00 | 430.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-04 09:30:00 | 434.60 | 2024-09-04 09:40:00 | 437.50 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2024-09-04 09:30:00 | 434.60 | 2024-09-04 10:25:00 | 436.30 | TARGET_HIT | 0.50 | 0.39% |
| BUY | retest1 | 2024-09-05 10:55:00 | 452.20 | 2024-09-05 11:05:00 | 454.79 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-09-05 10:55:00 | 452.20 | 2024-09-05 11:35:00 | 452.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-06 10:05:00 | 439.60 | 2024-09-06 11:15:00 | 436.65 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-09-06 10:05:00 | 439.60 | 2024-09-06 11:25:00 | 439.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-10 11:00:00 | 419.75 | 2024-09-10 11:35:00 | 420.96 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-09-13 09:50:00 | 415.25 | 2024-09-13 10:35:00 | 417.52 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-09-13 09:50:00 | 415.25 | 2024-09-13 10:50:00 | 415.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-17 10:30:00 | 407.25 | 2024-09-17 10:35:00 | 408.33 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-09-19 09:30:00 | 404.70 | 2024-09-19 09:40:00 | 402.98 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-09-19 09:30:00 | 404.70 | 2024-09-19 15:05:00 | 398.60 | TARGET_HIT | 0.50 | 1.51% |
| BUY | retest1 | 2024-09-23 11:15:00 | 408.00 | 2024-09-23 11:25:00 | 406.53 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-09-24 09:55:00 | 411.45 | 2024-09-24 10:10:00 | 410.03 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-09-27 10:50:00 | 430.60 | 2024-09-27 11:15:00 | 428.66 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-10-07 10:45:00 | 393.30 | 2024-10-07 11:10:00 | 395.29 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest1 | 2024-10-14 09:35:00 | 401.00 | 2024-10-14 10:35:00 | 399.55 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-21 09:35:00 | 428.20 | 2024-10-21 09:55:00 | 429.89 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-10-24 09:30:00 | 395.50 | 2024-10-24 09:55:00 | 397.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-11-28 09:30:00 | 383.65 | 2024-11-28 10:15:00 | 385.73 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-11-28 09:30:00 | 383.65 | 2024-11-28 10:30:00 | 383.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-03 10:10:00 | 385.75 | 2024-12-03 10:20:00 | 384.66 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-06 09:50:00 | 399.95 | 2024-12-06 09:55:00 | 398.41 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-19 10:30:00 | 401.45 | 2024-12-19 10:55:00 | 403.87 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-12-19 10:30:00 | 401.45 | 2024-12-19 15:20:00 | 407.50 | TARGET_HIT | 0.50 | 1.51% |
| BUY | retest1 | 2024-12-20 09:40:00 | 412.85 | 2024-12-20 09:50:00 | 411.31 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-23 11:15:00 | 409.50 | 2024-12-23 12:15:00 | 407.98 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-12-24 10:05:00 | 410.00 | 2024-12-24 10:30:00 | 412.22 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-12-24 10:05:00 | 410.00 | 2024-12-24 13:10:00 | 412.25 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2024-12-26 09:35:00 | 417.70 | 2024-12-26 09:55:00 | 415.94 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-01-01 10:15:00 | 406.05 | 2025-01-01 10:50:00 | 407.35 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-01-08 10:15:00 | 391.45 | 2025-01-08 10:30:00 | 389.90 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-01-15 11:00:00 | 372.60 | 2025-01-15 11:30:00 | 370.48 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2025-01-15 11:00:00 | 372.60 | 2025-01-15 15:20:00 | 365.00 | TARGET_HIT | 0.50 | 2.04% |
| BUY | retest1 | 2025-01-31 10:55:00 | 355.20 | 2025-01-31 13:10:00 | 356.80 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2025-01-31 10:55:00 | 355.20 | 2025-01-31 15:20:00 | 357.60 | TARGET_HIT | 0.50 | 0.68% |
| SELL | retest1 | 2025-02-01 11:05:00 | 354.20 | 2025-02-01 11:45:00 | 352.60 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-02-01 11:05:00 | 354.20 | 2025-02-01 15:15:00 | 345.00 | TARGET_HIT | 0.50 | 2.60% |
| SELL | retest1 | 2025-02-21 10:15:00 | 325.00 | 2025-02-21 11:25:00 | 322.85 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2025-02-21 10:15:00 | 325.00 | 2025-02-21 11:35:00 | 325.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-02-24 10:00:00 | 324.60 | 2025-02-24 10:15:00 | 323.33 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-03-28 09:50:00 | 367.45 | 2025-03-28 10:20:00 | 365.79 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-04-21 09:35:00 | 391.50 | 2025-04-21 09:50:00 | 390.02 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-05-08 10:05:00 | 397.55 | 2025-05-08 10:35:00 | 395.08 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2025-05-08 10:05:00 | 397.55 | 2025-05-08 15:20:00 | 385.90 | TARGET_HIT | 0.50 | 2.93% |
