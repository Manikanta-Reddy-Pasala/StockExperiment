# Bharat Petroleum Corporation Ltd. (BPCL)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 303.20
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
| ENTRY1 | 70 |
| ENTRY2 | 0 |
| PARTIAL | 26 |
| TARGET_HIT | 12 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 96 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 58
- **Target hits / Stop hits / Partials:** 12 / 58 / 26
- **Avg / median % per leg:** 0.15% / 0.00%
- **Sum % (uncompounded):** 14.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 22 | 39.3% | 7 | 34 | 15 | 0.13% | 7.2% |
| BUY @ 2nd Alert (retest1) | 56 | 22 | 39.3% | 7 | 34 | 15 | 0.13% | 7.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 40 | 16 | 40.0% | 5 | 24 | 11 | 0.18% | 7.4% |
| SELL @ 2nd Alert (retest1) | 40 | 16 | 40.0% | 5 | 24 | 11 | 0.18% | 7.4% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 96 | 38 | 39.6% | 12 | 58 | 26 | 0.15% | 14.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 09:45:00 | 302.43 | 306.01 | 0.00 | ORB-short ORB[305.20,308.85] vol=1.5x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-05-14 13:50:00 | 303.75 | 303.95 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-23 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:45:00 | 324.50 | 322.15 | 0.00 | ORB-long ORB[320.30,323.50] vol=1.6x ATR=1.15 |
| Stop hit — per-position SL triggered | 2024-05-23 09:55:00 | 323.35 | 322.35 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-24 10:45:00 | 325.52 | 322.75 | 0.00 | ORB-long ORB[321.05,323.63] vol=2.9x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-24 11:10:00 | 327.05 | 323.89 | 0.00 | T1 1.5R @ 327.05 |
| Target hit | 2024-05-24 15:00:00 | 327.00 | 327.28 | 0.00 | Trail-exit close<VWAP |

### Cycle 4 — SELL (started 2024-05-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 09:55:00 | 323.88 | 328.02 | 0.00 | ORB-short ORB[326.10,329.98] vol=2.1x ATR=1.42 |
| Stop hit — per-position SL triggered | 2024-05-27 10:10:00 | 325.30 | 327.39 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-06-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 09:30:00 | 311.68 | 310.50 | 0.00 | ORB-long ORB[309.13,311.18] vol=1.9x ATR=0.72 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 09:35:00 | 312.77 | 311.54 | 0.00 | T1 1.5R @ 312.77 |
| Target hit | 2024-06-14 10:10:00 | 312.33 | 312.41 | 0.00 | Trail-exit close<VWAP |

### Cycle 6 — BUY (started 2024-06-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 10:45:00 | 314.23 | 310.22 | 0.00 | ORB-long ORB[306.40,310.00] vol=1.6x ATR=0.88 |
| Stop hit — per-position SL triggered | 2024-06-20 10:50:00 | 313.35 | 310.37 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 11:15:00 | 300.40 | 298.57 | 0.00 | ORB-long ORB[296.25,299.20] vol=3.4x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-06-26 11:20:00 | 299.65 | 298.60 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-01 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 10:30:00 | 306.75 | 304.92 | 0.00 | ORB-long ORB[304.00,305.70] vol=1.9x ATR=0.81 |
| Stop hit — per-position SL triggered | 2024-07-01 10:35:00 | 305.94 | 304.97 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:45:00 | 307.70 | 306.22 | 0.00 | ORB-long ORB[303.40,306.80] vol=1.6x ATR=0.90 |
| Stop hit — per-position SL triggered | 2024-07-02 10:55:00 | 306.80 | 306.26 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2024-07-05 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:45:00 | 306.00 | 305.19 | 0.00 | ORB-long ORB[302.00,304.95] vol=1.7x ATR=0.89 |
| Stop hit — per-position SL triggered | 2024-07-05 10:55:00 | 305.11 | 305.24 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 302.50 | 304.64 | 0.00 | ORB-short ORB[304.60,307.95] vol=1.5x ATR=0.75 |
| Stop hit — per-position SL triggered | 2024-07-08 11:20:00 | 303.25 | 304.58 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2024-07-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:35:00 | 294.05 | 297.37 | 0.00 | ORB-short ORB[298.65,301.80] vol=1.8x ATR=1.18 |
| Stop hit — per-position SL triggered | 2024-07-10 10:40:00 | 295.23 | 297.12 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-07-15 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:45:00 | 307.25 | 305.72 | 0.00 | ORB-long ORB[304.35,307.20] vol=1.8x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:05:00 | 308.66 | 306.43 | 0.00 | T1 1.5R @ 308.66 |
| Stop hit — per-position SL triggered | 2024-07-15 10:25:00 | 307.25 | 306.71 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-24 10:10:00 | 310.90 | 307.47 | 0.00 | ORB-long ORB[306.00,309.20] vol=2.2x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 10:55:00 | 312.81 | 309.04 | 0.00 | T1 1.5R @ 312.81 |
| Stop hit — per-position SL triggered | 2024-07-24 12:35:00 | 310.90 | 310.11 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-25 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 10:25:00 | 319.55 | 315.02 | 0.00 | ORB-long ORB[310.80,314.00] vol=2.1x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-25 10:40:00 | 321.55 | 316.83 | 0.00 | T1 1.5R @ 321.55 |
| Target hit | 2024-07-25 15:20:00 | 326.25 | 322.17 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2024-08-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 10:50:00 | 349.90 | 344.74 | 0.00 | ORB-long ORB[343.30,346.90] vol=2.0x ATR=1.48 |
| Stop hit — per-position SL triggered | 2024-08-02 11:00:00 | 348.42 | 345.20 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-08-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:50:00 | 334.85 | 338.68 | 0.00 | ORB-short ORB[339.05,343.95] vol=1.6x ATR=1.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 11:00:00 | 333.12 | 337.95 | 0.00 | T1 1.5R @ 333.12 |
| Stop hit — per-position SL triggered | 2024-08-08 11:10:00 | 334.85 | 337.74 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-08-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-16 09:30:00 | 330.20 | 328.77 | 0.00 | ORB-long ORB[326.05,329.70] vol=2.2x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-16 09:40:00 | 331.70 | 329.58 | 0.00 | T1 1.5R @ 331.70 |
| Stop hit — per-position SL triggered | 2024-08-16 09:50:00 | 330.20 | 329.89 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-08-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:35:00 | 342.00 | 340.25 | 0.00 | ORB-long ORB[336.35,341.20] vol=2.2x ATR=1.32 |
| Stop hit — per-position SL triggered | 2024-08-19 10:20:00 | 340.68 | 341.15 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 11:15:00 | 352.60 | 351.24 | 0.00 | ORB-long ORB[348.15,350.80] vol=2.7x ATR=0.79 |
| Stop hit — per-position SL triggered | 2024-08-21 11:25:00 | 351.81 | 351.34 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-22 09:30:00 | 349.85 | 352.28 | 0.00 | ORB-short ORB[350.55,355.10] vol=1.5x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-08-22 09:40:00 | 350.99 | 351.98 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 09:50:00 | 353.45 | 351.85 | 0.00 | ORB-long ORB[349.55,353.05] vol=2.3x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-08-23 09:55:00 | 352.49 | 351.90 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-08-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-27 09:30:00 | 344.35 | 346.43 | 0.00 | ORB-short ORB[346.10,349.10] vol=4.1x ATR=1.11 |
| Stop hit — per-position SL triggered | 2024-08-27 09:50:00 | 345.46 | 346.01 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-28 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-28 09:40:00 | 351.85 | 350.63 | 0.00 | ORB-long ORB[348.50,351.45] vol=2.1x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-28 09:45:00 | 353.25 | 351.77 | 0.00 | T1 1.5R @ 353.25 |
| Target hit | 2024-08-28 10:30:00 | 352.35 | 352.43 | 0.00 | Trail-exit close<VWAP |

### Cycle 25 — SELL (started 2024-08-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-29 09:30:00 | 347.70 | 348.75 | 0.00 | ORB-short ORB[347.85,350.10] vol=1.5x ATR=0.95 |
| Stop hit — per-position SL triggered | 2024-08-29 09:40:00 | 348.65 | 348.60 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-02 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:40:00 | 363.90 | 360.35 | 0.00 | ORB-long ORB[358.15,361.85] vol=1.8x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-09-02 10:45:00 | 362.50 | 360.56 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2024-09-03 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-03 10:20:00 | 353.85 | 356.56 | 0.00 | ORB-short ORB[357.00,359.70] vol=1.5x ATR=0.98 |
| Stop hit — per-position SL triggered | 2024-09-03 10:40:00 | 354.83 | 356.25 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-05 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-05 09:45:00 | 359.65 | 357.65 | 0.00 | ORB-long ORB[355.80,359.50] vol=2.0x ATR=1.19 |
| Stop hit — per-position SL triggered | 2024-09-05 10:05:00 | 358.46 | 358.05 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-09-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 10:30:00 | 347.85 | 349.50 | 0.00 | ORB-short ORB[348.50,351.10] vol=1.6x ATR=0.93 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 10:55:00 | 346.46 | 349.13 | 0.00 | T1 1.5R @ 346.46 |
| Stop hit — per-position SL triggered | 2024-09-10 13:25:00 | 347.85 | 348.05 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 347.35 | 349.91 | 0.00 | ORB-short ORB[348.05,352.80] vol=1.9x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 09:45:00 | 345.31 | 348.97 | 0.00 | T1 1.5R @ 345.31 |
| Target hit | 2024-09-11 15:20:00 | 340.60 | 343.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2024-09-13 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:45:00 | 346.85 | 346.61 | 0.00 | ORB-long ORB[342.90,346.20] vol=1.6x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-09-13 11:25:00 | 345.69 | 346.62 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:50:00 | 337.00 | 338.90 | 0.00 | ORB-short ORB[338.80,340.85] vol=2.4x ATR=0.87 |
| Stop hit — per-position SL triggered | 2024-09-17 09:55:00 | 337.87 | 338.68 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:30:00 | 333.85 | 335.22 | 0.00 | ORB-short ORB[334.55,337.95] vol=1.6x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 09:40:00 | 332.43 | 334.41 | 0.00 | T1 1.5R @ 332.43 |
| Target hit | 2024-09-19 15:20:00 | 324.40 | 326.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 34 — BUY (started 2024-09-23 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-23 10:50:00 | 338.00 | 335.76 | 0.00 | ORB-long ORB[331.30,333.90] vol=1.8x ATR=1.10 |
| Stop hit — per-position SL triggered | 2024-09-23 12:45:00 | 336.90 | 336.61 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:35:00 | 339.30 | 337.93 | 0.00 | ORB-long ORB[336.70,339.20] vol=2.1x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 09:40:00 | 340.48 | 338.56 | 0.00 | T1 1.5R @ 340.48 |
| Target hit | 2024-09-24 12:40:00 | 340.70 | 340.74 | 0.00 | Trail-exit close<VWAP |

### Cycle 36 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-27 10:15:00 | 354.25 | 350.16 | 0.00 | ORB-long ORB[346.35,350.65] vol=2.5x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-27 10:45:00 | 356.40 | 352.15 | 0.00 | T1 1.5R @ 356.40 |
| Target hit | 2024-09-27 15:20:00 | 367.10 | 363.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 37 — SELL (started 2024-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 11:15:00 | 365.90 | 369.84 | 0.00 | ORB-short ORB[369.10,373.00] vol=2.1x ATR=1.25 |
| Stop hit — per-position SL triggered | 2024-10-01 11:45:00 | 367.15 | 369.13 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-10-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 11:05:00 | 335.30 | 339.08 | 0.00 | ORB-short ORB[340.20,343.30] vol=1.6x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-10-07 11:10:00 | 336.70 | 338.93 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-10-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-08 09:50:00 | 336.40 | 331.92 | 0.00 | ORB-long ORB[328.25,332.70] vol=1.9x ATR=1.81 |
| Stop hit — per-position SL triggered | 2024-10-08 10:05:00 | 334.59 | 332.86 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-10 11:00:00 | 339.25 | 341.32 | 0.00 | ORB-short ORB[340.50,343.30] vol=1.5x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 11:30:00 | 337.95 | 340.87 | 0.00 | T1 1.5R @ 337.95 |
| Target hit | 2024-10-10 15:20:00 | 335.55 | 338.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2024-10-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 09:30:00 | 343.00 | 341.70 | 0.00 | ORB-long ORB[339.40,342.20] vol=3.4x ATR=0.87 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-14 09:40:00 | 344.31 | 342.56 | 0.00 | T1 1.5R @ 344.31 |
| Stop hit — per-position SL triggered | 2024-10-14 09:50:00 | 343.00 | 342.70 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-15 09:30:00 | 349.25 | 346.77 | 0.00 | ORB-long ORB[344.00,348.35] vol=2.4x ATR=1.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 09:50:00 | 351.50 | 347.83 | 0.00 | T1 1.5R @ 351.50 |
| Stop hit — per-position SL triggered | 2024-10-15 10:00:00 | 349.25 | 348.08 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 11:15:00 | 350.00 | 352.75 | 0.00 | ORB-short ORB[350.05,354.80] vol=1.6x ATR=0.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 11:45:00 | 348.58 | 352.43 | 0.00 | T1 1.5R @ 348.58 |
| Stop hit — per-position SL triggered | 2024-10-16 12:15:00 | 350.00 | 352.23 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2024-10-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-23 09:40:00 | 316.85 | 320.32 | 0.00 | ORB-short ORB[318.50,323.25] vol=1.5x ATR=1.65 |
| Stop hit — per-position SL triggered | 2024-10-23 09:45:00 | 318.50 | 320.04 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:45:00 | 303.25 | 307.61 | 0.00 | ORB-short ORB[308.85,312.95] vol=2.2x ATR=1.23 |
| Stop hit — per-position SL triggered | 2024-10-29 10:50:00 | 304.48 | 307.45 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2024-11-06 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:45:00 | 312.55 | 310.63 | 0.00 | ORB-long ORB[307.25,311.90] vol=1.6x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-11-06 09:50:00 | 311.15 | 310.69 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:30:00 | 296.30 | 294.45 | 0.00 | ORB-long ORB[293.40,295.00] vol=1.7x ATR=0.85 |
| Stop hit — per-position SL triggered | 2024-11-28 09:45:00 | 295.45 | 295.23 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-12-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-03 10:40:00 | 294.00 | 294.78 | 0.00 | ORB-short ORB[294.05,297.80] vol=1.9x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-03 10:55:00 | 292.81 | 294.55 | 0.00 | T1 1.5R @ 292.81 |
| Stop hit — per-position SL triggered | 2024-12-03 11:00:00 | 294.00 | 294.54 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2024-12-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:45:00 | 298.45 | 295.96 | 0.00 | ORB-long ORB[293.15,295.50] vol=2.7x ATR=0.78 |
| Stop hit — per-position SL triggered | 2024-12-04 09:50:00 | 297.67 | 296.21 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-12-16 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:40:00 | 296.80 | 298.86 | 0.00 | ORB-short ORB[298.60,301.70] vol=1.6x ATR=0.80 |
| Stop hit — per-position SL triggered | 2024-12-16 13:15:00 | 297.60 | 297.72 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-12-17 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:05:00 | 295.10 | 296.44 | 0.00 | ORB-short ORB[296.00,299.85] vol=1.7x ATR=0.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 10:20:00 | 294.06 | 296.15 | 0.00 | T1 1.5R @ 294.06 |
| Stop hit — per-position SL triggered | 2024-12-17 11:50:00 | 295.10 | 295.44 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-20 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 10:50:00 | 298.50 | 295.11 | 0.00 | ORB-long ORB[292.20,295.00] vol=2.2x ATR=1.03 |
| Stop hit — per-position SL triggered | 2024-12-20 11:05:00 | 297.47 | 295.74 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-23 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 11:00:00 | 293.10 | 290.36 | 0.00 | ORB-long ORB[288.75,292.90] vol=1.6x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:20:00 | 294.70 | 290.80 | 0.00 | T1 1.5R @ 294.70 |
| Stop hit — per-position SL triggered | 2024-12-23 11:55:00 | 293.10 | 291.33 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2024-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 11:15:00 | 295.10 | 291.66 | 0.00 | ORB-long ORB[287.35,289.95] vol=2.8x ATR=0.83 |
| Stop hit — per-position SL triggered | 2024-12-24 11:30:00 | 294.27 | 292.30 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2024-12-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-26 09:30:00 | 297.65 | 295.98 | 0.00 | ORB-long ORB[294.25,296.55] vol=3.2x ATR=1.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:40:00 | 299.26 | 296.64 | 0.00 | T1 1.5R @ 299.26 |
| Stop hit — per-position SL triggered | 2024-12-26 09:50:00 | 297.65 | 296.93 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:55:00 | 296.85 | 296.18 | 0.00 | ORB-long ORB[294.25,296.45] vol=1.5x ATR=0.70 |
| Stop hit — per-position SL triggered | 2024-12-27 11:15:00 | 296.15 | 296.22 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-01-01 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-01 10:05:00 | 291.10 | 292.05 | 0.00 | ORB-short ORB[291.50,293.20] vol=1.5x ATR=0.77 |
| Stop hit — per-position SL triggered | 2025-01-01 10:35:00 | 291.87 | 291.86 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2025-01-06 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-06 10:55:00 | 289.65 | 291.77 | 0.00 | ORB-short ORB[292.90,297.25] vol=1.8x ATR=0.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 11:05:00 | 288.29 | 291.33 | 0.00 | T1 1.5R @ 288.29 |
| Stop hit — per-position SL triggered | 2025-01-06 13:00:00 | 289.65 | 289.46 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2025-01-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 11:05:00 | 284.70 | 287.31 | 0.00 | ORB-short ORB[286.40,290.50] vol=3.9x ATR=1.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-07 11:20:00 | 283.20 | 286.28 | 0.00 | T1 1.5R @ 283.20 |
| Target hit | 2025-01-07 15:20:00 | 283.00 | 283.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 60 — BUY (started 2025-01-08 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 10:30:00 | 285.15 | 283.67 | 0.00 | ORB-long ORB[280.50,283.45] vol=1.6x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-01-08 11:20:00 | 284.15 | 283.90 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2025-01-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-14 11:10:00 | 270.70 | 268.87 | 0.00 | ORB-long ORB[266.00,270.05] vol=1.7x ATR=0.93 |
| Stop hit — per-position SL triggered | 2025-01-14 11:30:00 | 269.77 | 269.00 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:20:00 | 265.95 | 267.05 | 0.00 | ORB-short ORB[266.25,268.95] vol=2.1x ATR=0.95 |
| Stop hit — per-position SL triggered | 2025-01-16 10:45:00 | 266.90 | 266.90 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-01-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-17 09:45:00 | 272.10 | 269.85 | 0.00 | ORB-long ORB[267.25,270.35] vol=1.5x ATR=0.96 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-17 10:00:00 | 273.54 | 270.75 | 0.00 | T1 1.5R @ 273.54 |
| Stop hit — per-position SL triggered | 2025-01-17 10:45:00 | 272.10 | 271.56 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2025-01-28 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 10:35:00 | 256.10 | 258.10 | 0.00 | ORB-short ORB[260.80,263.90] vol=2.1x ATR=1.07 |
| Stop hit — per-position SL triggered | 2025-01-28 10:45:00 | 257.17 | 257.88 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-02-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-06 09:40:00 | 266.45 | 265.21 | 0.00 | ORB-long ORB[262.70,266.10] vol=1.9x ATR=1.02 |
| Stop hit — per-position SL triggered | 2025-02-06 10:20:00 | 265.43 | 265.55 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-03-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-07 10:50:00 | 262.50 | 262.63 | 0.00 | ORB-short ORB[262.80,265.79] vol=3.2x ATR=0.79 |
| Stop hit — per-position SL triggered | 2025-03-07 11:25:00 | 263.29 | 262.73 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-03-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-20 10:45:00 | 268.13 | 266.80 | 0.00 | ORB-long ORB[265.53,267.96] vol=2.5x ATR=0.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 11:10:00 | 269.31 | 267.36 | 0.00 | T1 1.5R @ 269.31 |
| Target hit | 2025-03-20 15:20:00 | 271.99 | 269.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 68 — BUY (started 2025-04-21 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:35:00 | 304.35 | 302.22 | 0.00 | ORB-long ORB[298.00,300.20] vol=4.9x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-04-21 12:05:00 | 303.30 | 303.25 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-22 09:30:00 | 307.15 | 305.00 | 0.00 | ORB-long ORB[302.65,306.00] vol=1.8x ATR=1.00 |
| Stop hit — per-position SL triggered | 2025-04-22 09:50:00 | 306.15 | 306.14 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-05-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-08 11:00:00 | 314.40 | 316.65 | 0.00 | ORB-short ORB[315.50,320.20] vol=1.6x ATR=0.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 11:45:00 | 313.15 | 316.12 | 0.00 | T1 1.5R @ 313.15 |
| Target hit | 2025-05-08 15:20:00 | 307.05 | 312.30 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 09:45:00 | 302.43 | 2024-05-14 13:50:00 | 303.75 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-05-23 09:45:00 | 324.50 | 2024-05-23 09:55:00 | 323.35 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-05-24 10:45:00 | 325.52 | 2024-05-24 11:10:00 | 327.05 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2024-05-24 10:45:00 | 325.52 | 2024-05-24 15:00:00 | 327.00 | TARGET_HIT | 0.50 | 0.45% |
| SELL | retest1 | 2024-05-27 09:55:00 | 323.88 | 2024-05-27 10:10:00 | 325.30 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-06-14 09:30:00 | 311.68 | 2024-06-14 09:35:00 | 312.77 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-06-14 09:30:00 | 311.68 | 2024-06-14 10:10:00 | 312.33 | TARGET_HIT | 0.50 | 0.21% |
| BUY | retest1 | 2024-06-20 10:45:00 | 314.23 | 2024-06-20 10:50:00 | 313.35 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-26 11:15:00 | 300.40 | 2024-06-26 11:20:00 | 299.65 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-07-01 10:30:00 | 306.75 | 2024-07-01 10:35:00 | 305.94 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-02 10:45:00 | 307.70 | 2024-07-02 10:55:00 | 306.80 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-05 10:45:00 | 306.00 | 2024-07-05 10:55:00 | 305.11 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-07-08 11:10:00 | 302.50 | 2024-07-08 11:20:00 | 303.25 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2024-07-10 10:35:00 | 294.05 | 2024-07-10 10:40:00 | 295.23 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-07-15 09:45:00 | 307.25 | 2024-07-15 10:05:00 | 308.66 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-15 09:45:00 | 307.25 | 2024-07-15 10:25:00 | 307.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-24 10:10:00 | 310.90 | 2024-07-24 10:55:00 | 312.81 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-07-24 10:10:00 | 310.90 | 2024-07-24 12:35:00 | 310.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-25 10:25:00 | 319.55 | 2024-07-25 10:40:00 | 321.55 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-07-25 10:25:00 | 319.55 | 2024-07-25 15:20:00 | 326.25 | TARGET_HIT | 0.50 | 2.10% |
| BUY | retest1 | 2024-08-02 10:50:00 | 349.90 | 2024-08-02 11:00:00 | 348.42 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-08-08 10:50:00 | 334.85 | 2024-08-08 11:00:00 | 333.12 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-08-08 10:50:00 | 334.85 | 2024-08-08 11:10:00 | 334.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-16 09:30:00 | 330.20 | 2024-08-16 09:40:00 | 331.70 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-08-16 09:30:00 | 330.20 | 2024-08-16 09:50:00 | 330.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-19 09:35:00 | 342.00 | 2024-08-19 10:20:00 | 340.68 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-08-21 11:15:00 | 352.60 | 2024-08-21 11:25:00 | 351.81 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-08-22 09:30:00 | 349.85 | 2024-08-22 09:40:00 | 350.99 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-23 09:50:00 | 353.45 | 2024-08-23 09:55:00 | 352.49 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-27 09:30:00 | 344.35 | 2024-08-27 09:50:00 | 345.46 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-08-28 09:40:00 | 351.85 | 2024-08-28 09:45:00 | 353.25 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-08-28 09:40:00 | 351.85 | 2024-08-28 10:30:00 | 352.35 | TARGET_HIT | 0.50 | 0.14% |
| SELL | retest1 | 2024-08-29 09:30:00 | 347.70 | 2024-08-29 09:40:00 | 348.65 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-09-02 10:40:00 | 363.90 | 2024-09-02 10:45:00 | 362.50 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-09-03 10:20:00 | 353.85 | 2024-09-03 10:40:00 | 354.83 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-09-05 09:45:00 | 359.65 | 2024-09-05 10:05:00 | 358.46 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-10 10:30:00 | 347.85 | 2024-09-10 10:55:00 | 346.46 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-09-10 10:30:00 | 347.85 | 2024-09-10 13:25:00 | 347.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-11 09:30:00 | 347.35 | 2024-09-11 09:45:00 | 345.31 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2024-09-11 09:30:00 | 347.35 | 2024-09-11 15:20:00 | 340.60 | TARGET_HIT | 0.50 | 1.94% |
| BUY | retest1 | 2024-09-13 10:45:00 | 346.85 | 2024-09-13 11:25:00 | 345.69 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-09-17 09:50:00 | 337.00 | 2024-09-17 09:55:00 | 337.87 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-09-19 09:30:00 | 333.85 | 2024-09-19 09:40:00 | 332.43 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-09-19 09:30:00 | 333.85 | 2024-09-19 15:20:00 | 324.40 | TARGET_HIT | 0.50 | 2.83% |
| BUY | retest1 | 2024-09-23 10:50:00 | 338.00 | 2024-09-23 12:45:00 | 336.90 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-09-24 09:35:00 | 339.30 | 2024-09-24 09:40:00 | 340.48 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-09-24 09:35:00 | 339.30 | 2024-09-24 12:40:00 | 340.70 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2024-09-27 10:15:00 | 354.25 | 2024-09-27 10:45:00 | 356.40 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2024-09-27 10:15:00 | 354.25 | 2024-09-27 15:20:00 | 367.10 | TARGET_HIT | 0.50 | 3.63% |
| SELL | retest1 | 2024-10-01 11:15:00 | 365.90 | 2024-10-01 11:45:00 | 367.15 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-10-07 11:05:00 | 335.30 | 2024-10-07 11:10:00 | 336.70 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-10-08 09:50:00 | 336.40 | 2024-10-08 10:05:00 | 334.59 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-10-10 11:00:00 | 339.25 | 2024-10-10 11:30:00 | 337.95 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-10-10 11:00:00 | 339.25 | 2024-10-10 15:20:00 | 335.55 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2024-10-14 09:30:00 | 343.00 | 2024-10-14 09:40:00 | 344.31 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2024-10-14 09:30:00 | 343.00 | 2024-10-14 09:50:00 | 343.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-15 09:30:00 | 349.25 | 2024-10-15 09:50:00 | 351.50 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-10-15 09:30:00 | 349.25 | 2024-10-15 10:00:00 | 349.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-16 11:15:00 | 350.00 | 2024-10-16 11:45:00 | 348.58 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-16 11:15:00 | 350.00 | 2024-10-16 12:15:00 | 350.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-23 09:40:00 | 316.85 | 2024-10-23 09:45:00 | 318.50 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-10-29 10:45:00 | 303.25 | 2024-10-29 10:50:00 | 304.48 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-11-06 09:45:00 | 312.55 | 2024-11-06 09:50:00 | 311.15 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-11-28 09:30:00 | 296.30 | 2024-11-28 09:45:00 | 295.45 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-12-03 10:40:00 | 294.00 | 2024-12-03 10:55:00 | 292.81 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-12-03 10:40:00 | 294.00 | 2024-12-03 11:00:00 | 294.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 09:45:00 | 298.45 | 2024-12-04 09:50:00 | 297.67 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-12-16 10:40:00 | 296.80 | 2024-12-16 13:15:00 | 297.60 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-12-17 10:05:00 | 295.10 | 2024-12-17 10:20:00 | 294.06 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-12-17 10:05:00 | 295.10 | 2024-12-17 11:50:00 | 295.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-20 10:50:00 | 298.50 | 2024-12-20 11:05:00 | 297.47 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-23 11:00:00 | 293.10 | 2024-12-23 11:20:00 | 294.70 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-12-23 11:00:00 | 293.10 | 2024-12-23 11:55:00 | 293.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 11:15:00 | 295.10 | 2024-12-24 11:30:00 | 294.27 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-12-26 09:30:00 | 297.65 | 2024-12-26 09:40:00 | 299.26 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-12-26 09:30:00 | 297.65 | 2024-12-26 09:50:00 | 297.65 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-27 10:55:00 | 296.85 | 2024-12-27 11:15:00 | 296.15 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-01-01 10:05:00 | 291.10 | 2025-01-01 10:35:00 | 291.87 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-01-06 10:55:00 | 289.65 | 2025-01-06 11:05:00 | 288.29 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2025-01-06 10:55:00 | 289.65 | 2025-01-06 13:00:00 | 289.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-07 11:05:00 | 284.70 | 2025-01-07 11:20:00 | 283.20 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2025-01-07 11:05:00 | 284.70 | 2025-01-07 15:20:00 | 283.00 | TARGET_HIT | 0.50 | 0.60% |
| BUY | retest1 | 2025-01-08 10:30:00 | 285.15 | 2025-01-08 11:20:00 | 284.15 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-01-14 11:10:00 | 270.70 | 2025-01-14 11:30:00 | 269.77 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-01-16 10:20:00 | 265.95 | 2025-01-16 10:45:00 | 266.90 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-01-17 09:45:00 | 272.10 | 2025-01-17 10:00:00 | 273.54 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-01-17 09:45:00 | 272.10 | 2025-01-17 10:45:00 | 272.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-28 10:35:00 | 256.10 | 2025-01-28 10:45:00 | 257.17 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-02-06 09:40:00 | 266.45 | 2025-02-06 10:20:00 | 265.43 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-03-07 10:50:00 | 262.50 | 2025-03-07 11:25:00 | 263.29 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-03-20 10:45:00 | 268.13 | 2025-03-20 11:10:00 | 269.31 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-03-20 10:45:00 | 268.13 | 2025-03-20 15:20:00 | 271.99 | TARGET_HIT | 0.50 | 1.44% |
| BUY | retest1 | 2025-04-21 10:35:00 | 304.35 | 2025-04-21 12:05:00 | 303.30 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-22 09:30:00 | 307.15 | 2025-04-22 09:50:00 | 306.15 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-05-08 11:00:00 | 314.40 | 2025-05-08 11:45:00 | 313.15 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-05-08 11:00:00 | 314.40 | 2025-05-08 15:20:00 | 307.05 | TARGET_HIT | 0.50 | 2.34% |
