# Aditya Birla Sun Life AMC Ltd. (ABSLAMC)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-10-27 15:25:00 (44104 bars)
- **Last close:** 817.50
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
| ENTRY1 | 113 |
| ENTRY2 | 0 |
| PARTIAL | 50 |
| TARGET_HIT | 25 |
| STOP_HIT | 88 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 163 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 75 / 88
- **Target hits / Stop hits / Partials:** 25 / 88 / 50
- **Avg / median % per leg:** 0.12% / 0.00%
- **Sum % (uncompounded):** 19.57%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 28 | 39.4% | 8 | 43 | 20 | 0.06% | 4.3% |
| BUY @ 2nd Alert (retest1) | 71 | 28 | 39.4% | 8 | 43 | 20 | 0.06% | 4.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 92 | 47 | 51.1% | 17 | 45 | 30 | 0.17% | 15.2% |
| SELL @ 2nd Alert (retest1) | 92 | 47 | 51.1% | 17 | 45 | 30 | 0.17% | 15.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 163 | 75 | 46.0% | 25 | 88 | 50 | 0.12% | 19.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:35:00 | 357.40 | 355.89 | 0.00 | ORB-long ORB[353.00,357.00] vol=3.5x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-15 09:45:00 | 359.23 | 356.99 | 0.00 | T1 1.5R @ 359.23 |
| Stop hit — per-position SL triggered | 2023-05-15 09:55:00 | 357.40 | 357.06 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2023-05-16 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-16 10:55:00 | 360.80 | 361.81 | 0.00 | ORB-short ORB[361.50,364.95] vol=2.5x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-16 11:30:00 | 359.61 | 361.17 | 0.00 | T1 1.5R @ 359.61 |
| Stop hit — per-position SL triggered | 2023-05-16 11:50:00 | 360.80 | 361.07 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 11:05:00 | 361.45 | 362.44 | 0.00 | ORB-short ORB[361.95,363.90] vol=2.5x ATR=0.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-17 15:15:00 | 359.97 | 360.50 | 0.00 | T1 1.5R @ 359.97 |
| Target hit | 2023-05-17 15:20:00 | 357.50 | 360.35 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 4 — SELL (started 2023-05-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 09:30:00 | 349.75 | 351.55 | 0.00 | ORB-short ORB[350.30,353.00] vol=2.3x ATR=1.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-19 09:40:00 | 347.95 | 350.79 | 0.00 | T1 1.5R @ 347.95 |
| Stop hit — per-position SL triggered | 2023-05-19 09:45:00 | 349.75 | 350.55 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 09:30:00 | 350.90 | 348.24 | 0.00 | ORB-long ORB[343.90,348.70] vol=3.6x ATR=1.33 |
| Stop hit — per-position SL triggered | 2023-05-23 09:35:00 | 349.57 | 348.43 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 09:35:00 | 347.45 | 345.41 | 0.00 | ORB-long ORB[342.65,345.50] vol=1.5x ATR=0.91 |
| Stop hit — per-position SL triggered | 2023-05-24 09:40:00 | 346.54 | 346.27 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2023-05-25 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 10:40:00 | 349.90 | 348.68 | 0.00 | ORB-long ORB[346.60,348.95] vol=3.0x ATR=0.71 |
| Stop hit — per-position SL triggered | 2023-05-25 10:45:00 | 349.19 | 348.68 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-05-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 10:55:00 | 353.45 | 352.62 | 0.00 | ORB-long ORB[350.45,352.90] vol=5.2x ATR=0.63 |
| Stop hit — per-position SL triggered | 2023-05-30 11:00:00 | 352.82 | 352.63 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-01 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-01 11:05:00 | 355.55 | 358.24 | 0.00 | ORB-short ORB[357.55,359.45] vol=3.2x ATR=0.90 |
| Stop hit — per-position SL triggered | 2023-06-01 11:10:00 | 356.45 | 358.16 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 09:30:00 | 362.35 | 361.80 | 0.00 | ORB-long ORB[359.80,362.20] vol=2.8x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-06-05 09:35:00 | 361.23 | 361.79 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-06 09:30:00 | 371.90 | 369.61 | 0.00 | ORB-long ORB[365.15,370.30] vol=4.7x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-06 10:15:00 | 373.87 | 371.69 | 0.00 | T1 1.5R @ 373.87 |
| Stop hit — per-position SL triggered | 2023-06-06 11:20:00 | 371.90 | 372.18 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-08 09:40:00 | 373.20 | 372.23 | 0.00 | ORB-long ORB[370.10,372.95] vol=3.1x ATR=1.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-08 10:30:00 | 374.82 | 373.08 | 0.00 | T1 1.5R @ 374.82 |
| Target hit | 2023-06-08 11:45:00 | 374.30 | 375.26 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2023-06-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-09 09:30:00 | 370.00 | 371.30 | 0.00 | ORB-short ORB[371.00,373.30] vol=1.9x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-09 09:55:00 | 368.47 | 370.05 | 0.00 | T1 1.5R @ 368.47 |
| Target hit | 2023-06-09 10:50:00 | 369.90 | 369.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 14 — SELL (started 2023-06-13 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-13 10:30:00 | 374.75 | 375.86 | 0.00 | ORB-short ORB[375.30,378.40] vol=1.8x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 10:55:00 | 373.48 | 375.61 | 0.00 | T1 1.5R @ 373.48 |
| Stop hit — per-position SL triggered | 2023-06-13 12:00:00 | 374.75 | 375.30 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-06-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-16 10:20:00 | 373.45 | 374.35 | 0.00 | ORB-short ORB[374.00,375.95] vol=1.7x ATR=0.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-16 12:35:00 | 372.47 | 373.72 | 0.00 | T1 1.5R @ 372.47 |
| Target hit | 2023-06-16 15:10:00 | 372.70 | 372.62 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — BUY (started 2023-06-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-20 10:25:00 | 373.40 | 371.54 | 0.00 | ORB-long ORB[369.35,372.95] vol=2.2x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-20 11:15:00 | 374.77 | 372.39 | 0.00 | T1 1.5R @ 374.77 |
| Stop hit — per-position SL triggered | 2023-06-20 11:20:00 | 373.40 | 372.48 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2023-06-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-21 09:50:00 | 377.50 | 375.70 | 0.00 | ORB-long ORB[373.05,376.00] vol=2.6x ATR=0.89 |
| Stop hit — per-position SL triggered | 2023-06-21 09:55:00 | 376.61 | 376.06 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-06-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-23 09:30:00 | 370.85 | 373.34 | 0.00 | ORB-short ORB[372.15,376.70] vol=4.9x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-23 09:35:00 | 369.13 | 372.31 | 0.00 | T1 1.5R @ 369.13 |
| Stop hit — per-position SL triggered | 2023-06-23 10:30:00 | 370.85 | 371.20 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-06-26 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-26 10:30:00 | 363.90 | 365.40 | 0.00 | ORB-short ORB[365.00,369.85] vol=3.3x ATR=1.08 |
| Stop hit — per-position SL triggered | 2023-06-26 10:50:00 | 364.98 | 365.20 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2023-06-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-28 10:35:00 | 372.70 | 370.73 | 0.00 | ORB-long ORB[368.90,372.00] vol=1.6x ATR=0.90 |
| Stop hit — per-position SL triggered | 2023-06-28 10:40:00 | 371.80 | 370.76 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-05 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-05 10:00:00 | 376.00 | 376.25 | 0.00 | ORB-short ORB[376.20,377.50] vol=2.3x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-05 10:20:00 | 374.74 | 375.63 | 0.00 | T1 1.5R @ 374.74 |
| Target hit | 2023-07-05 15:20:00 | 372.80 | 373.63 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 22 — BUY (started 2023-07-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-07 09:30:00 | 373.95 | 372.95 | 0.00 | ORB-long ORB[371.75,373.00] vol=6.0x ATR=0.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-07 09:35:00 | 374.81 | 373.13 | 0.00 | T1 1.5R @ 374.81 |
| Stop hit — per-position SL triggered | 2023-07-07 09:40:00 | 373.95 | 373.32 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2023-07-11 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-11 10:05:00 | 372.60 | 371.57 | 0.00 | ORB-long ORB[367.75,372.00] vol=1.7x ATR=0.84 |
| Stop hit — per-position SL triggered | 2023-07-11 10:35:00 | 371.76 | 371.78 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-07-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-12 09:40:00 | 380.50 | 378.53 | 0.00 | ORB-long ORB[375.30,378.50] vol=2.7x ATR=1.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-12 09:50:00 | 382.23 | 379.96 | 0.00 | T1 1.5R @ 382.23 |
| Stop hit — per-position SL triggered | 2023-07-12 09:55:00 | 380.50 | 380.02 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-07-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 10:45:00 | 378.55 | 379.46 | 0.00 | ORB-short ORB[379.55,381.70] vol=1.6x ATR=0.87 |
| Stop hit — per-position SL triggered | 2023-07-13 12:40:00 | 379.42 | 379.01 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 09:30:00 | 383.00 | 383.72 | 0.00 | ORB-short ORB[383.15,385.00] vol=2.2x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-18 11:45:00 | 381.69 | 382.88 | 0.00 | T1 1.5R @ 381.69 |
| Target hit | 2023-07-18 15:20:00 | 381.40 | 382.16 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2023-07-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-19 09:40:00 | 383.00 | 382.29 | 0.00 | ORB-long ORB[381.40,382.80] vol=2.9x ATR=0.60 |
| Stop hit — per-position SL triggered | 2023-07-19 10:05:00 | 382.40 | 382.53 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-20 11:15:00 | 387.70 | 385.36 | 0.00 | ORB-long ORB[383.20,386.70] vol=7.4x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-20 11:45:00 | 389.07 | 386.15 | 0.00 | T1 1.5R @ 389.07 |
| Target hit | 2023-07-20 12:25:00 | 392.30 | 392.85 | 0.00 | Trail-exit close<VWAP |

### Cycle 29 — BUY (started 2023-07-24 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-24 10:00:00 | 399.35 | 396.81 | 0.00 | ORB-long ORB[393.85,398.80] vol=2.7x ATR=1.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-24 10:15:00 | 401.45 | 398.03 | 0.00 | T1 1.5R @ 401.45 |
| Stop hit — per-position SL triggered | 2023-07-24 10:20:00 | 399.35 | 398.18 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2023-07-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 09:50:00 | 409.20 | 408.18 | 0.00 | ORB-long ORB[407.15,408.85] vol=2.4x ATR=0.90 |
| Stop hit — per-position SL triggered | 2023-07-28 10:45:00 | 408.30 | 408.55 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-08-01 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-01 10:10:00 | 414.40 | 412.68 | 0.00 | ORB-long ORB[410.50,413.00] vol=2.7x ATR=0.82 |
| Stop hit — per-position SL triggered | 2023-08-01 10:15:00 | 413.58 | 412.76 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-07 09:30:00 | 406.25 | 407.52 | 0.00 | ORB-short ORB[406.80,409.75] vol=3.1x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-07 11:10:00 | 404.81 | 406.66 | 0.00 | T1 1.5R @ 404.81 |
| Target hit | 2023-08-07 15:20:00 | 401.20 | 404.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — SELL (started 2023-08-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 09:30:00 | 401.55 | 402.55 | 0.00 | ORB-short ORB[402.20,405.10] vol=1.5x ATR=0.77 |
| Stop hit — per-position SL triggered | 2023-08-09 09:35:00 | 402.32 | 402.46 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-10 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 10:10:00 | 404.70 | 403.07 | 0.00 | ORB-long ORB[400.05,403.95] vol=3.4x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-08-10 10:25:00 | 403.61 | 403.29 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-08-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-16 09:40:00 | 395.00 | 396.27 | 0.00 | ORB-short ORB[396.60,398.85] vol=4.6x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-16 10:05:00 | 393.48 | 395.52 | 0.00 | T1 1.5R @ 393.48 |
| Target hit | 2023-08-16 15:20:00 | 392.50 | 393.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2023-08-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-18 09:45:00 | 388.95 | 389.65 | 0.00 | ORB-short ORB[389.00,391.90] vol=1.6x ATR=0.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-18 09:50:00 | 387.66 | 389.51 | 0.00 | T1 1.5R @ 387.66 |
| Stop hit — per-position SL triggered | 2023-08-18 10:15:00 | 388.95 | 389.23 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2023-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-22 09:35:00 | 390.05 | 388.98 | 0.00 | ORB-long ORB[385.20,389.00] vol=1.8x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-22 11:25:00 | 391.46 | 389.97 | 0.00 | T1 1.5R @ 391.46 |
| Target hit | 2023-08-22 14:55:00 | 390.90 | 391.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 38 — SELL (started 2023-08-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-25 10:30:00 | 388.10 | 389.53 | 0.00 | ORB-short ORB[388.30,390.35] vol=2.1x ATR=0.64 |
| Stop hit — per-position SL triggered | 2023-08-25 10:40:00 | 388.74 | 389.49 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2023-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-28 11:15:00 | 388.20 | 389.97 | 0.00 | ORB-short ORB[389.65,391.95] vol=1.5x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-28 11:35:00 | 387.17 | 389.45 | 0.00 | T1 1.5R @ 387.17 |
| Target hit | 2023-08-28 15:20:00 | 386.85 | 387.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2023-08-31 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-31 11:05:00 | 387.60 | 391.00 | 0.00 | ORB-short ORB[390.50,394.80] vol=1.5x ATR=0.99 |
| Stop hit — per-position SL triggered | 2023-08-31 11:30:00 | 388.59 | 390.78 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2023-09-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-05 09:55:00 | 403.85 | 401.41 | 0.00 | ORB-long ORB[398.00,402.00] vol=2.6x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-05 10:35:00 | 405.55 | 402.48 | 0.00 | T1 1.5R @ 405.55 |
| Target hit | 2023-09-05 15:20:00 | 408.35 | 404.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — SELL (started 2023-09-08 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-08 09:55:00 | 412.80 | 413.16 | 0.00 | ORB-short ORB[413.00,414.95] vol=2.4x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-08 10:15:00 | 411.13 | 412.75 | 0.00 | T1 1.5R @ 411.13 |
| Stop hit — per-position SL triggered | 2023-09-08 11:30:00 | 412.80 | 412.00 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2023-09-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-13 09:40:00 | 410.80 | 413.24 | 0.00 | ORB-short ORB[411.85,416.00] vol=1.6x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-13 10:10:00 | 408.46 | 411.83 | 0.00 | T1 1.5R @ 408.46 |
| Target hit | 2023-09-13 12:05:00 | 410.00 | 409.95 | 0.00 | Trail-exit close>VWAP |

### Cycle 44 — SELL (started 2023-09-15 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-15 10:00:00 | 415.20 | 416.47 | 0.00 | ORB-short ORB[415.75,421.95] vol=2.5x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-15 11:35:00 | 413.12 | 415.57 | 0.00 | T1 1.5R @ 413.12 |
| Stop hit — per-position SL triggered | 2023-09-15 11:55:00 | 415.20 | 415.41 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2023-09-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-18 09:40:00 | 416.00 | 414.81 | 0.00 | ORB-long ORB[411.55,415.40] vol=2.9x ATR=1.26 |
| Stop hit — per-position SL triggered | 2023-09-18 09:50:00 | 414.74 | 414.86 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2023-09-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-21 10:55:00 | 415.15 | 416.68 | 0.00 | ORB-short ORB[415.55,418.95] vol=1.7x ATR=0.86 |
| Stop hit — per-position SL triggered | 2023-09-21 11:20:00 | 416.01 | 416.97 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2023-09-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-25 09:35:00 | 415.55 | 416.43 | 0.00 | ORB-short ORB[416.00,419.40] vol=1.8x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-25 10:20:00 | 414.07 | 415.85 | 0.00 | T1 1.5R @ 414.07 |
| Stop hit — per-position SL triggered | 2023-09-25 12:20:00 | 415.55 | 414.84 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2023-09-27 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-27 10:50:00 | 419.75 | 420.44 | 0.00 | ORB-short ORB[420.25,425.00] vol=2.6x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-09-27 11:05:00 | 420.73 | 420.44 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-10-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-06 09:30:00 | 434.50 | 435.54 | 0.00 | ORB-short ORB[434.95,437.70] vol=2.2x ATR=0.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-06 10:35:00 | 433.33 | 434.71 | 0.00 | T1 1.5R @ 433.33 |
| Target hit | 2023-10-06 14:40:00 | 433.10 | 431.75 | 0.00 | Trail-exit close>VWAP |

### Cycle 50 — BUY (started 2023-10-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-12 11:10:00 | 437.25 | 435.10 | 0.00 | ORB-long ORB[432.10,434.75] vol=2.9x ATR=0.72 |
| Stop hit — per-position SL triggered | 2023-10-12 11:15:00 | 436.53 | 435.16 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2023-10-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 11:10:00 | 450.30 | 453.84 | 0.00 | ORB-short ORB[453.55,460.10] vol=2.2x ATR=1.09 |
| Stop hit — per-position SL triggered | 2023-10-16 11:15:00 | 451.39 | 453.73 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2023-10-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-23 10:10:00 | 435.20 | 440.55 | 0.00 | ORB-short ORB[441.85,445.00] vol=1.7x ATR=1.88 |
| Stop hit — per-position SL triggered | 2023-10-23 10:15:00 | 437.08 | 439.94 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2023-11-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-01 10:50:00 | 442.20 | 443.22 | 0.00 | ORB-short ORB[442.80,446.80] vol=2.3x ATR=0.92 |
| Stop hit — per-position SL triggered | 2023-11-01 11:00:00 | 443.12 | 443.22 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2023-11-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-02 10:40:00 | 444.40 | 447.51 | 0.00 | ORB-short ORB[447.15,449.95] vol=1.5x ATR=1.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-02 11:45:00 | 442.49 | 446.20 | 0.00 | T1 1.5R @ 442.49 |
| Target hit | 2023-11-02 15:20:00 | 436.50 | 443.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — SELL (started 2023-11-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-07 10:40:00 | 441.95 | 442.86 | 0.00 | ORB-short ORB[443.25,445.80] vol=1.5x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-07 11:40:00 | 440.59 | 442.27 | 0.00 | T1 1.5R @ 440.59 |
| Stop hit — per-position SL triggered | 2023-11-07 12:45:00 | 441.95 | 442.34 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2023-11-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-08 11:00:00 | 447.40 | 449.54 | 0.00 | ORB-short ORB[450.05,454.45] vol=5.1x ATR=1.30 |
| Target hit | 2023-11-08 15:20:00 | 445.40 | 448.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2023-11-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-09 10:45:00 | 445.00 | 446.47 | 0.00 | ORB-short ORB[445.35,447.95] vol=3.0x ATR=0.69 |
| Stop hit — per-position SL triggered | 2023-11-09 10:50:00 | 445.69 | 446.45 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2023-11-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-16 10:25:00 | 459.15 | 456.81 | 0.00 | ORB-long ORB[453.55,457.45] vol=2.6x ATR=1.53 |
| Stop hit — per-position SL triggered | 2023-11-16 10:30:00 | 457.62 | 456.96 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2023-11-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-21 09:55:00 | 450.10 | 452.22 | 0.00 | ORB-short ORB[451.65,454.00] vol=3.2x ATR=0.98 |
| Stop hit — per-position SL triggered | 2023-11-21 10:00:00 | 451.08 | 452.13 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2023-11-28 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 09:35:00 | 458.00 | 456.14 | 0.00 | ORB-long ORB[454.45,457.15] vol=2.3x ATR=1.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-28 09:40:00 | 460.16 | 457.19 | 0.00 | T1 1.5R @ 460.16 |
| Stop hit — per-position SL triggered | 2023-11-28 10:00:00 | 458.00 | 457.71 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2023-11-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-29 09:30:00 | 453.60 | 455.03 | 0.00 | ORB-short ORB[454.00,458.20] vol=1.8x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 13:50:00 | 451.74 | 453.64 | 0.00 | T1 1.5R @ 451.74 |
| Target hit | 2023-11-29 15:05:00 | 452.70 | 452.54 | 0.00 | Trail-exit close>VWAP |

### Cycle 62 — SELL (started 2023-11-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-30 09:50:00 | 450.50 | 451.78 | 0.00 | ORB-short ORB[450.85,454.75] vol=1.7x ATR=1.15 |
| Stop hit — per-position SL triggered | 2023-11-30 10:15:00 | 451.65 | 451.54 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2023-12-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-01 10:30:00 | 448.65 | 449.88 | 0.00 | ORB-short ORB[448.85,453.35] vol=1.6x ATR=0.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-01 10:40:00 | 447.38 | 449.48 | 0.00 | T1 1.5R @ 447.38 |
| Stop hit — per-position SL triggered | 2023-12-01 10:45:00 | 448.65 | 449.47 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2023-12-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 10:50:00 | 453.10 | 450.54 | 0.00 | ORB-long ORB[447.55,453.00] vol=3.0x ATR=1.41 |
| Stop hit — per-position SL triggered | 2023-12-04 11:05:00 | 451.69 | 450.69 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2023-12-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-11 09:40:00 | 457.20 | 458.81 | 0.00 | ORB-short ORB[457.70,460.40] vol=1.5x ATR=1.39 |
| Stop hit — per-position SL triggered | 2023-12-11 09:50:00 | 458.59 | 458.80 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-12 11:15:00 | 462.35 | 464.34 | 0.00 | ORB-short ORB[464.00,466.90] vol=3.1x ATR=0.88 |
| Stop hit — per-position SL triggered | 2023-12-12 11:20:00 | 463.23 | 464.29 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2023-12-15 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-15 10:30:00 | 459.70 | 461.64 | 0.00 | ORB-short ORB[461.70,463.90] vol=2.3x ATR=1.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-15 10:35:00 | 457.77 | 461.48 | 0.00 | T1 1.5R @ 457.77 |
| Stop hit — per-position SL triggered | 2023-12-15 12:00:00 | 459.70 | 459.94 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2023-12-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-19 09:35:00 | 473.25 | 475.41 | 0.00 | ORB-short ORB[474.75,477.60] vol=1.8x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-19 09:40:00 | 470.66 | 473.50 | 0.00 | T1 1.5R @ 470.66 |
| Target hit | 2023-12-19 12:30:00 | 473.05 | 471.90 | 0.00 | Trail-exit close>VWAP |

### Cycle 69 — BUY (started 2023-12-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 09:50:00 | 477.50 | 475.12 | 0.00 | ORB-long ORB[470.55,474.90] vol=4.1x ATR=2.41 |
| Stop hit — per-position SL triggered | 2023-12-22 10:05:00 | 475.09 | 475.31 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2023-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-26 09:30:00 | 475.40 | 476.50 | 0.00 | ORB-short ORB[476.10,479.40] vol=2.1x ATR=1.65 |
| Stop hit — per-position SL triggered | 2023-12-26 09:35:00 | 477.05 | 476.48 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2023-12-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-27 09:40:00 | 470.90 | 472.61 | 0.00 | ORB-short ORB[471.45,474.85] vol=5.1x ATR=1.31 |
| Stop hit — per-position SL triggered | 2023-12-27 09:50:00 | 472.21 | 472.55 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2023-12-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-28 10:40:00 | 473.15 | 468.64 | 0.00 | ORB-long ORB[465.00,469.00] vol=4.7x ATR=1.76 |
| Stop hit — per-position SL triggered | 2023-12-28 12:30:00 | 471.39 | 470.46 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2023-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-29 10:15:00 | 474.60 | 472.55 | 0.00 | ORB-long ORB[469.30,474.25] vol=1.6x ATR=1.50 |
| Stop hit — per-position SL triggered | 2023-12-29 10:55:00 | 473.10 | 472.91 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-01-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 10:30:00 | 472.90 | 470.80 | 0.00 | ORB-long ORB[468.35,472.00] vol=2.5x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-01-01 10:40:00 | 471.64 | 470.85 | 0.00 | SL hit |

### Cycle 75 — SELL (started 2024-01-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:50:00 | 471.20 | 472.57 | 0.00 | ORB-short ORB[471.90,475.50] vol=2.8x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 10:00:00 | 469.08 | 471.97 | 0.00 | T1 1.5R @ 469.08 |
| Stop hit — per-position SL triggered | 2024-01-02 10:10:00 | 471.20 | 471.58 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2024-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-03 09:30:00 | 474.30 | 472.78 | 0.00 | ORB-long ORB[469.00,473.95] vol=2.2x ATR=1.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-03 09:35:00 | 476.52 | 474.02 | 0.00 | T1 1.5R @ 476.52 |
| Stop hit — per-position SL triggered | 2024-01-03 09:40:00 | 474.30 | 474.26 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2024-01-04 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-04 10:10:00 | 473.50 | 470.69 | 0.00 | ORB-long ORB[469.00,473.10] vol=1.6x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-01-04 10:25:00 | 472.18 | 470.73 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2024-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-05 11:15:00 | 474.85 | 475.65 | 0.00 | ORB-short ORB[475.00,478.95] vol=3.7x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-01-05 11:20:00 | 475.73 | 476.44 | 0.00 | SL hit |

### Cycle 79 — SELL (started 2024-01-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 10:55:00 | 471.15 | 474.73 | 0.00 | ORB-short ORB[474.10,478.85] vol=2.1x ATR=1.13 |
| Stop hit — per-position SL triggered | 2024-01-10 12:10:00 | 472.28 | 473.27 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2024-01-11 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-11 10:35:00 | 470.75 | 473.50 | 0.00 | ORB-short ORB[472.60,477.70] vol=1.9x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-11 10:50:00 | 469.26 | 472.77 | 0.00 | T1 1.5R @ 469.26 |
| Stop hit — per-position SL triggered | 2024-01-11 10:55:00 | 470.75 | 472.74 | 0.00 | SL hit |

### Cycle 81 — SELL (started 2024-01-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-17 11:05:00 | 487.65 | 489.63 | 0.00 | ORB-short ORB[487.80,493.70] vol=2.5x ATR=1.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-17 11:40:00 | 485.34 | 489.19 | 0.00 | T1 1.5R @ 485.34 |
| Target hit | 2024-01-17 15:20:00 | 483.60 | 486.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 82 — SELL (started 2024-01-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-18 09:35:00 | 477.60 | 482.10 | 0.00 | ORB-short ORB[480.00,486.55] vol=1.6x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-18 09:50:00 | 474.61 | 478.77 | 0.00 | T1 1.5R @ 474.61 |
| Stop hit — per-position SL triggered | 2024-01-18 10:00:00 | 477.60 | 478.17 | 0.00 | SL hit |

### Cycle 83 — SELL (started 2024-01-20 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-20 10:10:00 | 479.90 | 481.05 | 0.00 | ORB-short ORB[480.25,485.70] vol=1.9x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-01-20 10:20:00 | 481.10 | 481.02 | 0.00 | SL hit |

### Cycle 84 — SELL (started 2024-01-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-23 09:55:00 | 482.55 | 486.52 | 0.00 | ORB-short ORB[485.00,492.05] vol=1.9x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-01-23 10:00:00 | 484.36 | 486.23 | 0.00 | SL hit |

### Cycle 85 — SELL (started 2024-01-25 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-25 10:20:00 | 465.00 | 467.93 | 0.00 | ORB-short ORB[468.05,472.20] vol=3.3x ATR=1.67 |
| Stop hit — per-position SL triggered | 2024-01-25 11:30:00 | 466.67 | 467.13 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2024-01-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-29 09:30:00 | 477.90 | 474.86 | 0.00 | ORB-long ORB[471.00,477.00] vol=2.1x ATR=2.02 |
| Stop hit — per-position SL triggered | 2024-01-29 10:20:00 | 475.88 | 476.46 | 0.00 | SL hit |

### Cycle 87 — SELL (started 2024-01-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-30 09:35:00 | 486.20 | 490.50 | 0.00 | ORB-short ORB[488.40,494.50] vol=1.7x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-30 10:30:00 | 482.05 | 488.81 | 0.00 | T1 1.5R @ 482.05 |
| Target hit | 2024-01-30 15:20:00 | 481.65 | 483.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 88 — BUY (started 2024-02-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-06 09:55:00 | 484.30 | 481.17 | 0.00 | ORB-long ORB[475.70,480.00] vol=2.8x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-06 11:10:00 | 487.37 | 484.73 | 0.00 | T1 1.5R @ 487.37 |
| Target hit | 2024-02-06 15:20:00 | 489.05 | 486.44 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 89 — SELL (started 2024-02-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 09:45:00 | 478.00 | 479.53 | 0.00 | ORB-short ORB[478.85,485.00] vol=2.0x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 10:20:00 | 475.94 | 478.00 | 0.00 | T1 1.5R @ 475.94 |
| Target hit | 2024-02-12 15:20:00 | 469.15 | 474.09 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 90 — BUY (started 2024-02-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-14 09:35:00 | 462.75 | 461.01 | 0.00 | ORB-long ORB[458.60,461.90] vol=2.6x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-14 09:50:00 | 464.89 | 461.65 | 0.00 | T1 1.5R @ 464.89 |
| Stop hit — per-position SL triggered | 2024-02-14 13:00:00 | 462.75 | 463.89 | 0.00 | SL hit |

### Cycle 91 — BUY (started 2024-02-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-15 09:50:00 | 470.65 | 470.04 | 0.00 | ORB-long ORB[468.00,470.05] vol=3.5x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-15 11:25:00 | 472.90 | 470.76 | 0.00 | T1 1.5R @ 472.90 |
| Stop hit — per-position SL triggered | 2024-02-15 11:55:00 | 470.65 | 470.80 | 0.00 | SL hit |

### Cycle 92 — BUY (started 2024-02-16 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 11:00:00 | 479.95 | 477.53 | 0.00 | ORB-long ORB[474.80,477.45] vol=8.1x ATR=1.05 |
| Stop hit — per-position SL triggered | 2024-02-16 11:05:00 | 478.90 | 477.58 | 0.00 | SL hit |

### Cycle 93 — SELL (started 2024-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-20 11:15:00 | 480.00 | 481.40 | 0.00 | ORB-short ORB[482.00,486.80] vol=2.3x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-02-20 11:45:00 | 480.85 | 481.30 | 0.00 | SL hit |

### Cycle 94 — BUY (started 2024-02-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-21 09:35:00 | 484.20 | 482.29 | 0.00 | ORB-long ORB[480.45,483.10] vol=1.6x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-21 09:45:00 | 485.94 | 483.39 | 0.00 | T1 1.5R @ 485.94 |
| Target hit | 2024-02-21 10:40:00 | 484.85 | 484.87 | 0.00 | Trail-exit close<VWAP |

### Cycle 95 — BUY (started 2024-02-26 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-26 10:45:00 | 500.05 | 496.81 | 0.00 | ORB-long ORB[491.15,496.90] vol=7.9x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-02-26 10:55:00 | 498.34 | 497.29 | 0.00 | SL hit |

### Cycle 96 — BUY (started 2024-02-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 09:30:00 | 514.65 | 510.59 | 0.00 | ORB-long ORB[506.00,513.00] vol=2.9x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-27 09:40:00 | 518.14 | 513.32 | 0.00 | T1 1.5R @ 518.14 |
| Stop hit — per-position SL triggered | 2024-02-27 10:05:00 | 514.65 | 515.80 | 0.00 | SL hit |

### Cycle 97 — SELL (started 2024-03-04 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-04 09:55:00 | 517.95 | 520.73 | 0.00 | ORB-short ORB[520.00,524.70] vol=1.8x ATR=1.85 |
| Stop hit — per-position SL triggered | 2024-03-04 10:20:00 | 519.80 | 520.39 | 0.00 | SL hit |

### Cycle 98 — SELL (started 2024-03-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-07 10:00:00 | 514.10 | 515.10 | 0.00 | ORB-short ORB[514.25,518.20] vol=2.5x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-03-07 10:10:00 | 515.41 | 515.15 | 0.00 | SL hit |

### Cycle 99 — SELL (started 2024-03-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-18 10:00:00 | 475.25 | 477.77 | 0.00 | ORB-short ORB[478.85,483.70] vol=1.5x ATR=1.96 |
| Stop hit — per-position SL triggered | 2024-03-18 10:10:00 | 477.21 | 477.64 | 0.00 | SL hit |

### Cycle 100 — SELL (started 2024-03-22 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-22 10:25:00 | 453.50 | 455.59 | 0.00 | ORB-short ORB[455.05,457.20] vol=1.8x ATR=1.12 |
| Stop hit — per-position SL triggered | 2024-03-22 10:35:00 | 454.62 | 455.49 | 0.00 | SL hit |

### Cycle 101 — BUY (started 2024-03-28 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-28 11:10:00 | 455.60 | 454.19 | 0.00 | ORB-long ORB[452.00,455.40] vol=1.8x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-28 11:30:00 | 457.00 | 454.72 | 0.00 | T1 1.5R @ 457.00 |
| Stop hit — per-position SL triggered | 2024-03-28 12:40:00 | 455.60 | 455.80 | 0.00 | SL hit |

### Cycle 102 — BUY (started 2024-04-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-03 09:30:00 | 474.45 | 472.49 | 0.00 | ORB-long ORB[469.65,472.90] vol=2.0x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-04-03 09:40:00 | 472.98 | 472.84 | 0.00 | SL hit |

### Cycle 103 — SELL (started 2024-04-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-04 09:35:00 | 477.00 | 479.17 | 0.00 | ORB-short ORB[478.10,485.00] vol=3.6x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-04-04 10:20:00 | 479.19 | 478.83 | 0.00 | SL hit |

### Cycle 104 — SELL (started 2024-04-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-04-05 09:45:00 | 477.95 | 479.18 | 0.00 | ORB-short ORB[478.00,483.20] vol=1.6x ATR=1.76 |
| Stop hit — per-position SL triggered | 2024-04-05 10:25:00 | 479.71 | 478.63 | 0.00 | SL hit |

### Cycle 105 — BUY (started 2024-04-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-09 10:20:00 | 480.55 | 477.20 | 0.00 | ORB-long ORB[474.30,477.60] vol=2.5x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-04-09 10:30:00 | 479.12 | 477.37 | 0.00 | SL hit |

### Cycle 106 — BUY (started 2024-04-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-12 09:35:00 | 499.90 | 496.41 | 0.00 | ORB-long ORB[491.05,498.00] vol=2.2x ATR=2.66 |
| Stop hit — per-position SL triggered | 2024-04-12 09:45:00 | 497.24 | 496.64 | 0.00 | SL hit |

### Cycle 107 — BUY (started 2024-04-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-18 09:35:00 | 520.90 | 516.32 | 0.00 | ORB-long ORB[511.05,517.50] vol=2.0x ATR=3.49 |
| Stop hit — per-position SL triggered | 2024-04-18 09:45:00 | 517.41 | 517.56 | 0.00 | SL hit |

### Cycle 108 — BUY (started 2024-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 09:30:00 | 511.50 | 508.13 | 0.00 | ORB-long ORB[503.95,510.90] vol=4.8x ATR=2.71 |
| Stop hit — per-position SL triggered | 2024-04-22 09:40:00 | 508.79 | 508.72 | 0.00 | SL hit |

### Cycle 109 — BUY (started 2024-04-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-23 10:10:00 | 520.50 | 515.29 | 0.00 | ORB-long ORB[510.50,512.35] vol=9.2x ATR=2.29 |
| Stop hit — per-position SL triggered | 2024-04-23 10:15:00 | 518.21 | 515.42 | 0.00 | SL hit |

### Cycle 110 — BUY (started 2024-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-25 11:15:00 | 555.00 | 549.47 | 0.00 | ORB-long ORB[545.60,550.55] vol=1.9x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-25 13:50:00 | 559.04 | 552.51 | 0.00 | T1 1.5R @ 559.04 |
| Target hit | 2024-04-25 15:20:00 | 560.05 | 555.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 111 — BUY (started 2024-04-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-30 09:30:00 | 553.35 | 548.45 | 0.00 | ORB-long ORB[542.35,547.80] vol=2.5x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-04-30 09:35:00 | 550.95 | 549.18 | 0.00 | SL hit |

### Cycle 112 — BUY (started 2024-05-02 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-02 10:35:00 | 550.50 | 547.02 | 0.00 | ORB-long ORB[543.00,549.10] vol=2.8x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-02 10:50:00 | 553.61 | 548.67 | 0.00 | T1 1.5R @ 553.61 |
| Target hit | 2024-05-02 12:30:00 | 551.00 | 551.08 | 0.00 | Trail-exit close<VWAP |

### Cycle 113 — SELL (started 2024-05-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-06 09:40:00 | 540.45 | 542.96 | 0.00 | ORB-short ORB[543.00,549.25] vol=3.7x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-05-06 09:45:00 | 542.85 | 542.83 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 09:35:00 | 357.40 | 2023-05-15 09:45:00 | 359.23 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-05-15 09:35:00 | 357.40 | 2023-05-15 09:55:00 | 357.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-16 10:55:00 | 360.80 | 2023-05-16 11:30:00 | 359.61 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-05-16 10:55:00 | 360.80 | 2023-05-16 11:50:00 | 360.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-17 11:05:00 | 361.45 | 2023-05-17 15:15:00 | 359.97 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-05-17 11:05:00 | 361.45 | 2023-05-17 15:20:00 | 357.50 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2023-05-19 09:30:00 | 349.75 | 2023-05-19 09:40:00 | 347.95 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2023-05-19 09:30:00 | 349.75 | 2023-05-19 09:45:00 | 349.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-05-23 09:30:00 | 350.90 | 2023-05-23 09:35:00 | 349.57 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2023-05-24 09:35:00 | 347.45 | 2023-05-24 09:40:00 | 346.54 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-05-25 10:40:00 | 349.90 | 2023-05-25 10:45:00 | 349.19 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-05-30 10:55:00 | 353.45 | 2023-05-30 11:00:00 | 352.82 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2023-06-01 11:05:00 | 355.55 | 2023-06-01 11:10:00 | 356.45 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-06-05 09:30:00 | 362.35 | 2023-06-05 09:35:00 | 361.23 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-06-06 09:30:00 | 371.90 | 2023-06-06 10:15:00 | 373.87 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-06-06 09:30:00 | 371.90 | 2023-06-06 11:20:00 | 371.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-08 09:40:00 | 373.20 | 2023-06-08 10:30:00 | 374.82 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2023-06-08 09:40:00 | 373.20 | 2023-06-08 11:45:00 | 374.30 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2023-06-09 09:30:00 | 370.00 | 2023-06-09 09:55:00 | 368.47 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-06-09 09:30:00 | 370.00 | 2023-06-09 10:50:00 | 369.90 | TARGET_HIT | 0.50 | 0.03% |
| SELL | retest1 | 2023-06-13 10:30:00 | 374.75 | 2023-06-13 10:55:00 | 373.48 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-06-13 10:30:00 | 374.75 | 2023-06-13 12:00:00 | 374.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-16 10:20:00 | 373.45 | 2023-06-16 12:35:00 | 372.47 | PARTIAL | 0.50 | 0.26% |
| SELL | retest1 | 2023-06-16 10:20:00 | 373.45 | 2023-06-16 15:10:00 | 372.70 | TARGET_HIT | 0.50 | 0.20% |
| BUY | retest1 | 2023-06-20 10:25:00 | 373.40 | 2023-06-20 11:15:00 | 374.77 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2023-06-20 10:25:00 | 373.40 | 2023-06-20 11:20:00 | 373.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-06-21 09:50:00 | 377.50 | 2023-06-21 09:55:00 | 376.61 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-06-23 09:30:00 | 370.85 | 2023-06-23 09:35:00 | 369.13 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-06-23 09:30:00 | 370.85 | 2023-06-23 10:30:00 | 370.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-26 10:30:00 | 363.90 | 2023-06-26 10:50:00 | 364.98 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-06-28 10:35:00 | 372.70 | 2023-06-28 10:40:00 | 371.80 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-05 10:00:00 | 376.00 | 2023-07-05 10:20:00 | 374.74 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-07-05 10:00:00 | 376.00 | 2023-07-05 15:20:00 | 372.80 | TARGET_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2023-07-07 09:30:00 | 373.95 | 2023-07-07 09:35:00 | 374.81 | PARTIAL | 0.50 | 0.23% |
| BUY | retest1 | 2023-07-07 09:30:00 | 373.95 | 2023-07-07 09:40:00 | 373.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-11 10:05:00 | 372.60 | 2023-07-11 10:35:00 | 371.76 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-07-12 09:40:00 | 380.50 | 2023-07-12 09:50:00 | 382.23 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2023-07-12 09:40:00 | 380.50 | 2023-07-12 09:55:00 | 380.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-13 10:45:00 | 378.55 | 2023-07-13 12:40:00 | 379.42 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-07-18 09:30:00 | 383.00 | 2023-07-18 11:45:00 | 381.69 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2023-07-18 09:30:00 | 383.00 | 2023-07-18 15:20:00 | 381.40 | TARGET_HIT | 0.50 | 0.42% |
| BUY | retest1 | 2023-07-19 09:40:00 | 383.00 | 2023-07-19 10:05:00 | 382.40 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-07-20 11:15:00 | 387.70 | 2023-07-20 11:45:00 | 389.07 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2023-07-20 11:15:00 | 387.70 | 2023-07-20 12:25:00 | 392.30 | TARGET_HIT | 0.50 | 1.19% |
| BUY | retest1 | 2023-07-24 10:00:00 | 399.35 | 2023-07-24 10:15:00 | 401.45 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2023-07-24 10:00:00 | 399.35 | 2023-07-24 10:20:00 | 399.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-07-28 09:50:00 | 409.20 | 2023-07-28 10:45:00 | 408.30 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-08-01 10:10:00 | 414.40 | 2023-08-01 10:15:00 | 413.58 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2023-08-07 09:30:00 | 406.25 | 2023-08-07 11:10:00 | 404.81 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2023-08-07 09:30:00 | 406.25 | 2023-08-07 15:20:00 | 401.20 | TARGET_HIT | 0.50 | 1.24% |
| SELL | retest1 | 2023-08-09 09:30:00 | 401.55 | 2023-08-09 09:35:00 | 402.32 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-08-10 10:10:00 | 404.70 | 2023-08-10 10:25:00 | 403.61 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-16 09:40:00 | 395.00 | 2023-08-16 10:05:00 | 393.48 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2023-08-16 09:40:00 | 395.00 | 2023-08-16 15:20:00 | 392.50 | TARGET_HIT | 0.50 | 0.63% |
| SELL | retest1 | 2023-08-18 09:45:00 | 388.95 | 2023-08-18 09:50:00 | 387.66 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2023-08-18 09:45:00 | 388.95 | 2023-08-18 10:15:00 | 388.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-22 09:35:00 | 390.05 | 2023-08-22 11:25:00 | 391.46 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2023-08-22 09:35:00 | 390.05 | 2023-08-22 14:55:00 | 390.90 | TARGET_HIT | 0.50 | 0.22% |
| SELL | retest1 | 2023-08-25 10:30:00 | 388.10 | 2023-08-25 10:40:00 | 388.74 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-08-28 11:15:00 | 388.20 | 2023-08-28 11:35:00 | 387.17 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-08-28 11:15:00 | 388.20 | 2023-08-28 15:20:00 | 386.85 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2023-08-31 11:05:00 | 387.60 | 2023-08-31 11:30:00 | 388.59 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2023-09-05 09:55:00 | 403.85 | 2023-09-05 10:35:00 | 405.55 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2023-09-05 09:55:00 | 403.85 | 2023-09-05 15:20:00 | 408.35 | TARGET_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2023-09-08 09:55:00 | 412.80 | 2023-09-08 10:15:00 | 411.13 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-09-08 09:55:00 | 412.80 | 2023-09-08 11:30:00 | 412.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-13 09:40:00 | 410.80 | 2023-09-13 10:10:00 | 408.46 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2023-09-13 09:40:00 | 410.80 | 2023-09-13 12:05:00 | 410.00 | TARGET_HIT | 0.50 | 0.19% |
| SELL | retest1 | 2023-09-15 10:00:00 | 415.20 | 2023-09-15 11:35:00 | 413.12 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2023-09-15 10:00:00 | 415.20 | 2023-09-15 11:55:00 | 415.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-09-18 09:40:00 | 416.00 | 2023-09-18 09:50:00 | 414.74 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-09-21 10:55:00 | 415.15 | 2023-09-21 11:20:00 | 416.01 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-09-25 09:35:00 | 415.55 | 2023-09-25 10:20:00 | 414.07 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2023-09-25 09:35:00 | 415.55 | 2023-09-25 12:20:00 | 415.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-27 10:50:00 | 419.75 | 2023-09-27 11:05:00 | 420.73 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2023-10-06 09:30:00 | 434.50 | 2023-10-06 10:35:00 | 433.33 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2023-10-06 09:30:00 | 434.50 | 2023-10-06 14:40:00 | 433.10 | TARGET_HIT | 0.50 | 0.32% |
| BUY | retest1 | 2023-10-12 11:10:00 | 437.25 | 2023-10-12 11:15:00 | 436.53 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2023-10-16 11:10:00 | 450.30 | 2023-10-16 11:15:00 | 451.39 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-10-23 10:10:00 | 435.20 | 2023-10-23 10:15:00 | 437.08 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-11-01 10:50:00 | 442.20 | 2023-11-01 11:00:00 | 443.12 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-11-02 10:40:00 | 444.40 | 2023-11-02 11:45:00 | 442.49 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2023-11-02 10:40:00 | 444.40 | 2023-11-02 15:20:00 | 436.50 | TARGET_HIT | 0.50 | 1.78% |
| SELL | retest1 | 2023-11-07 10:40:00 | 441.95 | 2023-11-07 11:40:00 | 440.59 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2023-11-07 10:40:00 | 441.95 | 2023-11-07 12:45:00 | 441.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-08 11:00:00 | 447.40 | 2023-11-08 15:20:00 | 445.40 | TARGET_HIT | 1.00 | 0.45% |
| SELL | retest1 | 2023-11-09 10:45:00 | 445.00 | 2023-11-09 10:50:00 | 445.69 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest1 | 2023-11-16 10:25:00 | 459.15 | 2023-11-16 10:30:00 | 457.62 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-11-21 09:55:00 | 450.10 | 2023-11-21 10:00:00 | 451.08 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2023-11-28 09:35:00 | 458.00 | 2023-11-28 09:40:00 | 460.16 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2023-11-28 09:35:00 | 458.00 | 2023-11-28 10:00:00 | 458.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-29 09:30:00 | 453.60 | 2023-11-29 13:50:00 | 451.74 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2023-11-29 09:30:00 | 453.60 | 2023-11-29 15:05:00 | 452.70 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2023-11-30 09:50:00 | 450.50 | 2023-11-30 10:15:00 | 451.65 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-12-01 10:30:00 | 448.65 | 2023-12-01 10:40:00 | 447.38 | PARTIAL | 0.50 | 0.28% |
| SELL | retest1 | 2023-12-01 10:30:00 | 448.65 | 2023-12-01 10:45:00 | 448.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-04 10:50:00 | 453.10 | 2023-12-04 11:05:00 | 451.69 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-12-11 09:40:00 | 457.20 | 2023-12-11 09:50:00 | 458.59 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-12-12 11:15:00 | 462.35 | 2023-12-12 11:20:00 | 463.23 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-12-15 10:30:00 | 459.70 | 2023-12-15 10:35:00 | 457.77 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2023-12-15 10:30:00 | 459.70 | 2023-12-15 12:00:00 | 459.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-12-19 09:35:00 | 473.25 | 2023-12-19 09:40:00 | 470.66 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2023-12-19 09:35:00 | 473.25 | 2023-12-19 12:30:00 | 473.05 | TARGET_HIT | 0.50 | 0.04% |
| BUY | retest1 | 2023-12-22 09:50:00 | 477.50 | 2023-12-22 10:05:00 | 475.09 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2023-12-26 09:30:00 | 475.40 | 2023-12-26 09:35:00 | 477.05 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-12-27 09:40:00 | 470.90 | 2023-12-27 09:50:00 | 472.21 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-12-28 10:40:00 | 473.15 | 2023-12-28 12:30:00 | 471.39 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-12-29 10:15:00 | 474.60 | 2023-12-29 10:55:00 | 473.10 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-01-01 10:30:00 | 472.90 | 2024-01-01 10:40:00 | 471.64 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-01-02 09:50:00 | 471.20 | 2024-01-02 10:00:00 | 469.08 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2024-01-02 09:50:00 | 471.20 | 2024-01-02 10:10:00 | 471.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-03 09:30:00 | 474.30 | 2024-01-03 09:35:00 | 476.52 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-01-03 09:30:00 | 474.30 | 2024-01-03 09:40:00 | 474.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-04 10:10:00 | 473.50 | 2024-01-04 10:25:00 | 472.18 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-01-05 11:15:00 | 474.85 | 2024-01-05 11:20:00 | 475.73 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2024-01-10 10:55:00 | 471.15 | 2024-01-10 12:10:00 | 472.28 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-01-11 10:35:00 | 470.75 | 2024-01-11 10:50:00 | 469.26 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2024-01-11 10:35:00 | 470.75 | 2024-01-11 10:55:00 | 470.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-17 11:05:00 | 487.65 | 2024-01-17 11:40:00 | 485.34 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-01-17 11:05:00 | 487.65 | 2024-01-17 15:20:00 | 483.60 | TARGET_HIT | 0.50 | 0.83% |
| SELL | retest1 | 2024-01-18 09:35:00 | 477.60 | 2024-01-18 09:50:00 | 474.61 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-01-18 09:35:00 | 477.60 | 2024-01-18 10:00:00 | 477.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-20 10:10:00 | 479.90 | 2024-01-20 10:20:00 | 481.10 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-01-23 09:55:00 | 482.55 | 2024-01-23 10:00:00 | 484.36 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-01-25 10:20:00 | 465.00 | 2024-01-25 11:30:00 | 466.67 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-01-29 09:30:00 | 477.90 | 2024-01-29 10:20:00 | 475.88 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-01-30 09:35:00 | 486.20 | 2024-01-30 10:30:00 | 482.05 | PARTIAL | 0.50 | 0.85% |
| SELL | retest1 | 2024-01-30 09:35:00 | 486.20 | 2024-01-30 15:20:00 | 481.65 | TARGET_HIT | 0.50 | 0.94% |
| BUY | retest1 | 2024-02-06 09:55:00 | 484.30 | 2024-02-06 11:10:00 | 487.37 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-02-06 09:55:00 | 484.30 | 2024-02-06 15:20:00 | 489.05 | TARGET_HIT | 0.50 | 0.98% |
| SELL | retest1 | 2024-02-12 09:45:00 | 478.00 | 2024-02-12 10:20:00 | 475.94 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-02-12 09:45:00 | 478.00 | 2024-02-12 15:20:00 | 469.15 | TARGET_HIT | 0.50 | 1.85% |
| BUY | retest1 | 2024-02-14 09:35:00 | 462.75 | 2024-02-14 09:50:00 | 464.89 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-02-14 09:35:00 | 462.75 | 2024-02-14 13:00:00 | 462.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-15 09:50:00 | 470.65 | 2024-02-15 11:25:00 | 472.90 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-02-15 09:50:00 | 470.65 | 2024-02-15 11:55:00 | 470.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-02-16 11:00:00 | 479.95 | 2024-02-16 11:05:00 | 478.90 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-02-20 11:15:00 | 480.00 | 2024-02-20 11:45:00 | 480.85 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-02-21 09:35:00 | 484.20 | 2024-02-21 09:45:00 | 485.94 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2024-02-21 09:35:00 | 484.20 | 2024-02-21 10:40:00 | 484.85 | TARGET_HIT | 0.50 | 0.13% |
| BUY | retest1 | 2024-02-26 10:45:00 | 500.05 | 2024-02-26 10:55:00 | 498.34 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-02-27 09:30:00 | 514.65 | 2024-02-27 09:40:00 | 518.14 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-02-27 09:30:00 | 514.65 | 2024-02-27 10:05:00 | 514.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-03-04 09:55:00 | 517.95 | 2024-03-04 10:20:00 | 519.80 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-03-07 10:00:00 | 514.10 | 2024-03-07 10:10:00 | 515.41 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-03-18 10:00:00 | 475.25 | 2024-03-18 10:10:00 | 477.21 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-03-22 10:25:00 | 453.50 | 2024-03-22 10:35:00 | 454.62 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-03-28 11:10:00 | 455.60 | 2024-03-28 11:30:00 | 457.00 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2024-03-28 11:10:00 | 455.60 | 2024-03-28 12:40:00 | 455.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-03 09:30:00 | 474.45 | 2024-04-03 09:40:00 | 472.98 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-04-04 09:35:00 | 477.00 | 2024-04-04 10:20:00 | 479.19 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-04-05 09:45:00 | 477.95 | 2024-04-05 10:25:00 | 479.71 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-04-09 10:20:00 | 480.55 | 2024-04-09 10:30:00 | 479.12 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2024-04-12 09:35:00 | 499.90 | 2024-04-12 09:45:00 | 497.24 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-04-18 09:35:00 | 520.90 | 2024-04-18 09:45:00 | 517.41 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest1 | 2024-04-22 09:30:00 | 511.50 | 2024-04-22 09:40:00 | 508.79 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-04-23 10:10:00 | 520.50 | 2024-04-23 10:15:00 | 518.21 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-04-25 11:15:00 | 555.00 | 2024-04-25 13:50:00 | 559.04 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-04-25 11:15:00 | 555.00 | 2024-04-25 15:20:00 | 560.05 | TARGET_HIT | 0.50 | 0.91% |
| BUY | retest1 | 2024-04-30 09:30:00 | 553.35 | 2024-04-30 09:35:00 | 550.95 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-05-02 10:35:00 | 550.50 | 2024-05-02 10:50:00 | 553.61 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-05-02 10:35:00 | 550.50 | 2024-05-02 12:30:00 | 551.00 | TARGET_HIT | 0.50 | 0.09% |
| SELL | retest1 | 2024-05-06 09:40:00 | 540.45 | 2024-05-06 09:45:00 | 542.85 | STOP_HIT | 1.00 | -0.44% |
