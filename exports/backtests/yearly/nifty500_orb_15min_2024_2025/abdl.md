# Allied Blenders and Distillers Ltd. (ABDL)

## Backtest Summary

- **Window:** 2024-07-02 09:40:00 → 2026-05-08 15:25:00 (34295 bars)
- **Last close:** 594.00
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
| ENTRY1 | 56 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 7 |
| STOP_HIT | 49 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 78 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 29 / 49
- **Target hits / Stop hits / Partials:** 7 / 49 / 22
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 13.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 12 | 31.6% | 2 | 26 | 10 | 0.09% | 3.4% |
| BUY @ 2nd Alert (retest1) | 38 | 12 | 31.6% | 2 | 26 | 10 | 0.09% | 3.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 40 | 17 | 42.5% | 5 | 23 | 12 | 0.24% | 9.7% |
| SELL @ 2nd Alert (retest1) | 40 | 17 | 42.5% | 5 | 23 | 12 | 0.24% | 9.7% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 78 | 29 | 37.2% | 7 | 49 | 22 | 0.17% | 13.1% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-07-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:45:00 | 332.00 | 334.22 | 0.00 | ORB-short ORB[334.30,338.00] vol=2.8x ATR=1.25 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 09:55:00 | 330.12 | 331.39 | 0.00 | T1 1.5R @ 330.12 |
| Target hit | 2024-07-10 13:00:00 | 324.30 | 324.13 | 0.00 | Trail-exit close>VWAP |

### Cycle 2 — SELL (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 10:15:00 | 335.00 | 335.76 | 0.00 | ORB-short ORB[335.05,337.30] vol=1.7x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-12 11:00:00 | 332.74 | 335.30 | 0.00 | T1 1.5R @ 332.74 |
| Stop hit — per-position SL triggered | 2024-07-12 11:30:00 | 335.00 | 335.08 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-07-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:30:00 | 324.90 | 322.89 | 0.00 | ORB-long ORB[322.30,324.65] vol=2.5x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-07-16 10:45:00 | 323.43 | 322.94 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 317.05 | 318.95 | 0.00 | ORB-short ORB[317.25,321.00] vol=2.1x ATR=1.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:35:00 | 315.27 | 317.94 | 0.00 | T1 1.5R @ 315.27 |
| Target hit | 2024-07-18 15:20:00 | 309.65 | 313.10 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 5 — SELL (started 2024-07-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:50:00 | 307.35 | 310.18 | 0.00 | ORB-short ORB[310.00,314.55] vol=3.3x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:00:00 | 305.29 | 308.79 | 0.00 | T1 1.5R @ 305.29 |
| Target hit | 2024-07-19 11:35:00 | 305.35 | 305.16 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — BUY (started 2024-07-22 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-22 09:50:00 | 303.65 | 301.22 | 0.00 | ORB-long ORB[298.60,302.95] vol=2.1x ATR=1.75 |
| Stop hit — per-position SL triggered | 2024-07-22 10:00:00 | 301.90 | 301.52 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2024-07-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-29 11:05:00 | 314.25 | 315.43 | 0.00 | ORB-short ORB[315.00,318.00] vol=1.7x ATR=1.09 |
| Stop hit — per-position SL triggered | 2024-07-29 11:10:00 | 315.34 | 316.44 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2024-07-31 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-31 09:40:00 | 310.95 | 308.75 | 0.00 | ORB-long ORB[306.60,309.70] vol=2.4x ATR=1.43 |
| Stop hit — per-position SL triggered | 2024-07-31 09:45:00 | 309.52 | 308.81 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-01 10:15:00 | 305.00 | 306.76 | 0.00 | ORB-short ORB[306.65,309.65] vol=3.1x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-08-01 10:30:00 | 306.16 | 306.62 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-08-08 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 10:05:00 | 297.45 | 299.59 | 0.00 | ORB-short ORB[297.80,302.20] vol=2.7x ATR=1.29 |
| Stop hit — per-position SL triggered | 2024-08-08 10:10:00 | 298.74 | 299.50 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-08-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:50:00 | 335.00 | 330.59 | 0.00 | ORB-long ORB[324.55,329.50] vol=4.9x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 09:55:00 | 337.62 | 334.79 | 0.00 | T1 1.5R @ 337.62 |
| Target hit | 2024-08-27 15:20:00 | 348.95 | 346.28 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 12 — BUY (started 2024-09-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 09:35:00 | 356.25 | 353.29 | 0.00 | ORB-long ORB[351.00,354.75] vol=3.0x ATR=1.64 |
| Stop hit — per-position SL triggered | 2024-09-03 09:40:00 | 354.61 | 353.47 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2024-09-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:40:00 | 360.40 | 357.90 | 0.00 | ORB-long ORB[354.05,358.50] vol=2.1x ATR=1.94 |
| Stop hit — per-position SL triggered | 2024-09-13 09:50:00 | 358.46 | 358.25 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-17 09:35:00 | 353.60 | 356.73 | 0.00 | ORB-short ORB[356.15,361.30] vol=3.2x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 09:55:00 | 350.42 | 354.45 | 0.00 | T1 1.5R @ 350.42 |
| Target hit | 2024-09-17 15:20:00 | 345.65 | 350.96 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2024-09-25 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-25 10:05:00 | 350.60 | 353.80 | 0.00 | ORB-short ORB[354.00,358.30] vol=1.9x ATR=1.49 |
| Stop hit — per-position SL triggered | 2024-09-25 11:25:00 | 352.09 | 352.43 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-09-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-26 09:40:00 | 349.80 | 347.93 | 0.00 | ORB-long ORB[345.00,348.95] vol=3.1x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-09-26 09:50:00 | 347.97 | 348.36 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-27 10:15:00 | 344.55 | 346.58 | 0.00 | ORB-short ORB[345.05,349.55] vol=1.5x ATR=1.20 |
| Stop hit — per-position SL triggered | 2024-09-27 10:45:00 | 345.75 | 346.45 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-10-01 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:35:00 | 336.25 | 338.43 | 0.00 | ORB-short ORB[337.10,341.70] vol=1.9x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 11:35:00 | 334.40 | 337.59 | 0.00 | T1 1.5R @ 334.40 |
| Stop hit — per-position SL triggered | 2024-10-01 12:30:00 | 336.25 | 337.12 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-10-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:35:00 | 330.40 | 328.33 | 0.00 | ORB-long ORB[326.20,330.15] vol=2.1x ATR=1.55 |
| Stop hit — per-position SL triggered | 2024-10-09 10:40:00 | 328.85 | 328.46 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-10-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 11:00:00 | 330.75 | 328.78 | 0.00 | ORB-long ORB[326.30,329.80] vol=6.0x ATR=1.16 |
| Stop hit — per-position SL triggered | 2024-10-10 11:05:00 | 329.59 | 328.82 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-10-14 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 10:45:00 | 323.25 | 325.16 | 0.00 | ORB-short ORB[325.50,328.90] vol=4.1x ATR=1.14 |
| Stop hit — per-position SL triggered | 2024-10-14 11:00:00 | 324.39 | 324.98 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-16 11:15:00 | 336.30 | 335.07 | 0.00 | ORB-long ORB[330.10,335.15] vol=2.8x ATR=1.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 12:35:00 | 337.83 | 335.50 | 0.00 | T1 1.5R @ 337.83 |
| Stop hit — per-position SL triggered | 2024-10-16 14:20:00 | 336.30 | 336.41 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-10-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:45:00 | 334.00 | 336.01 | 0.00 | ORB-short ORB[336.20,338.25] vol=4.3x ATR=1.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 09:50:00 | 332.18 | 335.15 | 0.00 | T1 1.5R @ 332.18 |
| Stop hit — per-position SL triggered | 2024-10-17 10:25:00 | 334.00 | 334.06 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-10-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-24 09:45:00 | 311.30 | 312.44 | 0.00 | ORB-short ORB[311.75,315.65] vol=1.6x ATR=1.78 |
| Stop hit — per-position SL triggered | 2024-10-24 10:15:00 | 313.08 | 312.25 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-10-29 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 10:55:00 | 319.75 | 321.66 | 0.00 | ORB-short ORB[320.00,324.40] vol=1.6x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 11:00:00 | 317.63 | 321.41 | 0.00 | T1 1.5R @ 317.63 |
| Stop hit — per-position SL triggered | 2024-10-29 11:10:00 | 319.75 | 321.15 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-11-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 09:55:00 | 325.50 | 321.47 | 0.00 | ORB-long ORB[319.25,322.80] vol=2.5x ATR=1.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 10:00:00 | 327.79 | 322.77 | 0.00 | T1 1.5R @ 327.79 |
| Stop hit — per-position SL triggered | 2024-11-05 10:05:00 | 325.50 | 322.98 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-11-06 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 10:50:00 | 337.30 | 333.58 | 0.00 | ORB-long ORB[330.00,334.20] vol=4.5x ATR=1.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 10:55:00 | 339.49 | 336.36 | 0.00 | T1 1.5R @ 339.49 |
| Stop hit — per-position SL triggered | 2024-11-06 11:15:00 | 337.30 | 336.73 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-11-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 10:50:00 | 330.25 | 331.78 | 0.00 | ORB-short ORB[331.55,334.80] vol=3.7x ATR=1.26 |
| Stop hit — per-position SL triggered | 2024-11-07 11:00:00 | 331.51 | 331.72 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-11-12 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-12 11:05:00 | 331.60 | 329.63 | 0.00 | ORB-long ORB[326.80,331.30] vol=1.7x ATR=0.96 |
| Stop hit — per-position SL triggered | 2024-11-12 11:10:00 | 330.64 | 329.75 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-11-14 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-14 09:55:00 | 317.00 | 314.30 | 0.00 | ORB-long ORB[310.00,314.70] vol=2.2x ATR=2.04 |
| Stop hit — per-position SL triggered | 2024-11-14 10:05:00 | 314.96 | 314.51 | 0.00 | SL hit |

### Cycle 31 — SELL (started 2024-11-21 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 09:30:00 | 309.05 | 310.82 | 0.00 | ORB-short ORB[310.00,314.00] vol=3.4x ATR=1.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:35:00 | 307.48 | 310.33 | 0.00 | T1 1.5R @ 307.48 |
| Stop hit — per-position SL triggered | 2024-11-21 09:40:00 | 309.05 | 310.27 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-11-22 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-22 10:50:00 | 311.40 | 312.67 | 0.00 | ORB-short ORB[311.55,315.40] vol=4.7x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 13:05:00 | 309.50 | 311.93 | 0.00 | T1 1.5R @ 309.50 |
| Stop hit — per-position SL triggered | 2024-11-22 13:50:00 | 311.40 | 311.25 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-11-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-26 10:35:00 | 321.95 | 318.79 | 0.00 | ORB-long ORB[316.25,319.55] vol=3.9x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 10:40:00 | 323.79 | 319.90 | 0.00 | T1 1.5R @ 323.79 |
| Stop hit — per-position SL triggered | 2024-11-26 10:45:00 | 321.95 | 320.04 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2024-12-05 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:10:00 | 349.00 | 352.26 | 0.00 | ORB-short ORB[351.55,356.35] vol=3.7x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-12-05 12:05:00 | 350.90 | 350.41 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-12-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:30:00 | 354.10 | 352.51 | 0.00 | ORB-long ORB[348.90,353.20] vol=3.3x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 09:35:00 | 356.43 | 354.04 | 0.00 | T1 1.5R @ 356.43 |
| Stop hit — per-position SL triggered | 2024-12-06 10:00:00 | 354.10 | 355.42 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2024-12-12 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-12 10:10:00 | 370.85 | 372.84 | 0.00 | ORB-short ORB[372.20,377.00] vol=3.9x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-12-12 10:20:00 | 372.36 | 372.77 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-16 10:10:00 | 373.55 | 376.37 | 0.00 | ORB-short ORB[374.05,378.85] vol=1.8x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 12:10:00 | 370.68 | 374.95 | 0.00 | T1 1.5R @ 370.68 |
| Stop hit — per-position SL triggered | 2024-12-16 13:05:00 | 373.55 | 374.66 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-20 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 11:05:00 | 397.40 | 393.28 | 0.00 | ORB-long ORB[390.95,396.80] vol=3.4x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-12-20 11:15:00 | 395.60 | 393.70 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-23 09:50:00 | 394.50 | 391.39 | 0.00 | ORB-long ORB[387.20,393.10] vol=1.8x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 11:30:00 | 398.32 | 393.44 | 0.00 | T1 1.5R @ 398.32 |
| Stop hit — per-position SL triggered | 2024-12-23 12:00:00 | 394.50 | 393.85 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2024-12-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:30:00 | 416.45 | 421.88 | 0.00 | ORB-short ORB[420.85,426.50] vol=3.0x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-12-27 09:35:00 | 417.99 | 421.35 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-01-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:40:00 | 440.00 | 436.43 | 0.00 | ORB-long ORB[431.00,435.00] vol=6.3x ATR=1.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 09:45:00 | 442.99 | 437.35 | 0.00 | T1 1.5R @ 442.99 |
| Stop hit — per-position SL triggered | 2025-01-02 10:00:00 | 440.00 | 438.88 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-01-03 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 09:35:00 | 440.90 | 438.58 | 0.00 | ORB-long ORB[436.90,440.35] vol=1.8x ATR=1.58 |
| Stop hit — per-position SL triggered | 2025-01-03 09:50:00 | 439.32 | 438.85 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-01-07 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 10:00:00 | 424.35 | 426.67 | 0.00 | ORB-short ORB[424.50,429.75] vol=2.0x ATR=1.90 |
| Stop hit — per-position SL triggered | 2025-01-07 10:10:00 | 426.25 | 426.62 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2025-01-08 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-08 10:10:00 | 437.60 | 433.14 | 0.00 | ORB-long ORB[428.25,433.00] vol=2.6x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-01-08 10:20:00 | 435.69 | 433.44 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-09 10:30:00 | 435.35 | 431.37 | 0.00 | ORB-long ORB[427.05,433.00] vol=1.6x ATR=1.71 |
| Stop hit — per-position SL triggered | 2025-01-09 10:45:00 | 433.64 | 431.65 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-01-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-16 10:10:00 | 409.40 | 410.18 | 0.00 | ORB-short ORB[410.50,414.40] vol=1.8x ATR=2.04 |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 411.44 | 410.66 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-20 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 10:40:00 | 403.60 | 404.38 | 0.00 | ORB-short ORB[406.00,411.25] vol=2.9x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-01-20 10:50:00 | 405.26 | 404.40 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-01-21 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 10:55:00 | 400.55 | 406.80 | 0.00 | ORB-short ORB[408.50,413.65] vol=2.0x ATR=1.76 |
| Stop hit — per-position SL triggered | 2025-01-21 11:15:00 | 402.31 | 404.92 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2025-01-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-24 09:45:00 | 401.25 | 406.04 | 0.00 | ORB-short ORB[405.80,411.25] vol=2.8x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-01-24 09:50:00 | 403.23 | 405.28 | 0.00 | SL hit |

### Cycle 50 — BUY (started 2025-03-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-12 10:45:00 | 330.40 | 328.52 | 0.00 | ORB-long ORB[326.80,329.90] vol=1.9x ATR=1.39 |
| Stop hit — per-position SL triggered | 2025-03-12 10:55:00 | 329.01 | 328.57 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-03-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-21 09:45:00 | 321.80 | 320.72 | 0.00 | ORB-long ORB[318.05,320.60] vol=1.6x ATR=1.11 |
| Stop hit — per-position SL triggered | 2025-03-21 10:05:00 | 320.69 | 320.86 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-04-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-03 10:40:00 | 315.95 | 313.58 | 0.00 | ORB-long ORB[312.00,315.35] vol=3.7x ATR=1.32 |
| Stop hit — per-position SL triggered | 2025-04-03 11:00:00 | 314.63 | 313.86 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-04-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-15 09:30:00 | 323.60 | 322.07 | 0.00 | ORB-long ORB[319.00,322.95] vol=2.7x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-15 09:40:00 | 325.44 | 323.24 | 0.00 | T1 1.5R @ 325.44 |
| Target hit | 2025-04-15 11:15:00 | 325.30 | 327.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 54 — SELL (started 2025-04-24 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-24 11:05:00 | 331.20 | 332.55 | 0.00 | ORB-short ORB[331.80,335.70] vol=3.6x ATR=0.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 12:25:00 | 329.72 | 331.89 | 0.00 | T1 1.5R @ 329.72 |
| Target hit | 2025-04-24 15:20:00 | 326.00 | 328.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 55 — BUY (started 2025-05-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-05 10:50:00 | 312.60 | 309.99 | 0.00 | ORB-long ORB[306.00,309.95] vol=1.7x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-05-05 10:55:00 | 311.55 | 310.08 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-05-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:00:00 | 325.45 | 323.59 | 0.00 | ORB-long ORB[320.95,324.00] vol=2.2x ATR=1.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:45:00 | 327.43 | 324.35 | 0.00 | T1 1.5R @ 327.43 |
| Stop hit — per-position SL triggered | 2025-05-08 11:00:00 | 325.45 | 324.64 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-07-10 09:45:00 | 332.00 | 2024-07-10 09:55:00 | 330.12 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-07-10 09:45:00 | 332.00 | 2024-07-10 13:00:00 | 324.30 | TARGET_HIT | 0.50 | 2.32% |
| SELL | retest1 | 2024-07-12 10:15:00 | 335.00 | 2024-07-12 11:00:00 | 332.74 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-07-12 10:15:00 | 335.00 | 2024-07-12 11:30:00 | 335.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-16 10:30:00 | 324.90 | 2024-07-16 10:45:00 | 323.43 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-07-18 09:30:00 | 317.05 | 2024-07-18 09:35:00 | 315.27 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2024-07-18 09:30:00 | 317.05 | 2024-07-18 15:20:00 | 309.65 | TARGET_HIT | 0.50 | 2.33% |
| SELL | retest1 | 2024-07-19 09:50:00 | 307.35 | 2024-07-19 10:00:00 | 305.29 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-07-19 09:50:00 | 307.35 | 2024-07-19 11:35:00 | 305.35 | TARGET_HIT | 0.50 | 0.65% |
| BUY | retest1 | 2024-07-22 09:50:00 | 303.65 | 2024-07-22 10:00:00 | 301.90 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest1 | 2024-07-29 11:05:00 | 314.25 | 2024-07-29 11:10:00 | 315.34 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-31 09:40:00 | 310.95 | 2024-07-31 09:45:00 | 309.52 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-08-01 10:15:00 | 305.00 | 2024-08-01 10:30:00 | 306.16 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-08-08 10:05:00 | 297.45 | 2024-08-08 10:10:00 | 298.74 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-08-27 09:50:00 | 335.00 | 2024-08-27 09:55:00 | 337.62 | PARTIAL | 0.50 | 0.78% |
| BUY | retest1 | 2024-08-27 09:50:00 | 335.00 | 2024-08-27 15:20:00 | 348.95 | TARGET_HIT | 0.50 | 4.16% |
| BUY | retest1 | 2024-09-03 09:35:00 | 356.25 | 2024-09-03 09:40:00 | 354.61 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-09-13 09:40:00 | 360.40 | 2024-09-13 09:50:00 | 358.46 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest1 | 2024-09-17 09:35:00 | 353.60 | 2024-09-17 09:55:00 | 350.42 | PARTIAL | 0.50 | 0.90% |
| SELL | retest1 | 2024-09-17 09:35:00 | 353.60 | 2024-09-17 15:20:00 | 345.65 | TARGET_HIT | 0.50 | 2.25% |
| SELL | retest1 | 2024-09-25 10:05:00 | 350.60 | 2024-09-25 11:25:00 | 352.09 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-09-26 09:40:00 | 349.80 | 2024-09-26 09:50:00 | 347.97 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest1 | 2024-09-27 10:15:00 | 344.55 | 2024-09-27 10:45:00 | 345.75 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-01 10:35:00 | 336.25 | 2024-10-01 11:35:00 | 334.40 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-10-01 10:35:00 | 336.25 | 2024-10-01 12:30:00 | 336.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-09 10:35:00 | 330.40 | 2024-10-09 10:40:00 | 328.85 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-10-10 11:00:00 | 330.75 | 2024-10-10 11:05:00 | 329.59 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-14 10:45:00 | 323.25 | 2024-10-14 11:00:00 | 324.39 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-10-16 11:15:00 | 336.30 | 2024-10-16 12:35:00 | 337.83 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-10-16 11:15:00 | 336.30 | 2024-10-16 14:20:00 | 336.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-17 09:45:00 | 334.00 | 2024-10-17 09:50:00 | 332.18 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-10-17 09:45:00 | 334.00 | 2024-10-17 10:25:00 | 334.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-24 09:45:00 | 311.30 | 2024-10-24 10:15:00 | 313.08 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest1 | 2024-10-29 10:55:00 | 319.75 | 2024-10-29 11:00:00 | 317.63 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-10-29 10:55:00 | 319.75 | 2024-10-29 11:10:00 | 319.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-05 09:55:00 | 325.50 | 2024-11-05 10:00:00 | 327.79 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-11-05 09:55:00 | 325.50 | 2024-11-05 10:05:00 | 325.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-06 10:50:00 | 337.30 | 2024-11-06 10:55:00 | 339.49 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2024-11-06 10:50:00 | 337.30 | 2024-11-06 11:15:00 | 337.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-07 10:50:00 | 330.25 | 2024-11-07 11:00:00 | 331.51 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-11-12 11:05:00 | 331.60 | 2024-11-12 11:10:00 | 330.64 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-11-14 09:55:00 | 317.00 | 2024-11-14 10:05:00 | 314.96 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2024-11-21 09:30:00 | 309.05 | 2024-11-21 09:35:00 | 307.48 | PARTIAL | 0.50 | 0.51% |
| SELL | retest1 | 2024-11-21 09:30:00 | 309.05 | 2024-11-21 09:40:00 | 309.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-11-22 10:50:00 | 311.40 | 2024-11-22 13:05:00 | 309.50 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-11-22 10:50:00 | 311.40 | 2024-11-22 13:50:00 | 311.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-26 10:35:00 | 321.95 | 2024-11-26 10:40:00 | 323.79 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-11-26 10:35:00 | 321.95 | 2024-11-26 10:45:00 | 321.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-05 10:10:00 | 349.00 | 2024-12-05 12:05:00 | 350.90 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2024-12-06 09:30:00 | 354.10 | 2024-12-06 09:35:00 | 356.43 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-12-06 09:30:00 | 354.10 | 2024-12-06 10:00:00 | 354.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-12 10:10:00 | 370.85 | 2024-12-12 10:20:00 | 372.36 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-12-16 10:10:00 | 373.55 | 2024-12-16 12:10:00 | 370.68 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-12-16 10:10:00 | 373.55 | 2024-12-16 13:05:00 | 373.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-20 11:05:00 | 397.40 | 2024-12-20 11:15:00 | 395.60 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2024-12-23 09:50:00 | 394.50 | 2024-12-23 11:30:00 | 398.32 | PARTIAL | 0.50 | 0.97% |
| BUY | retest1 | 2024-12-23 09:50:00 | 394.50 | 2024-12-23 12:00:00 | 394.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-27 09:30:00 | 416.45 | 2024-12-27 09:35:00 | 417.99 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-01-02 09:40:00 | 440.00 | 2025-01-02 09:45:00 | 442.99 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-01-02 09:40:00 | 440.00 | 2025-01-02 10:00:00 | 440.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-03 09:35:00 | 440.90 | 2025-01-03 09:50:00 | 439.32 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-01-07 10:00:00 | 424.35 | 2025-01-07 10:10:00 | 426.25 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-01-08 10:10:00 | 437.60 | 2025-01-08 10:20:00 | 435.69 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-01-09 10:30:00 | 435.35 | 2025-01-09 10:45:00 | 433.64 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-01-16 10:10:00 | 409.40 | 2025-01-16 10:15:00 | 411.44 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-01-20 10:40:00 | 403.60 | 2025-01-20 10:50:00 | 405.26 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-01-21 10:55:00 | 400.55 | 2025-01-21 11:15:00 | 402.31 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2025-01-24 09:45:00 | 401.25 | 2025-01-24 09:50:00 | 403.23 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2025-03-12 10:45:00 | 330.40 | 2025-03-12 10:55:00 | 329.01 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-03-21 09:45:00 | 321.80 | 2025-03-21 10:05:00 | 320.69 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-04-03 10:40:00 | 315.95 | 2025-04-03 11:00:00 | 314.63 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-04-15 09:30:00 | 323.60 | 2025-04-15 09:40:00 | 325.44 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2025-04-15 09:30:00 | 323.60 | 2025-04-15 11:15:00 | 325.30 | TARGET_HIT | 0.50 | 0.53% |
| SELL | retest1 | 2025-04-24 11:05:00 | 331.20 | 2025-04-24 12:25:00 | 329.72 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-04-24 11:05:00 | 331.20 | 2025-04-24 15:20:00 | 326.00 | TARGET_HIT | 0.50 | 1.57% |
| BUY | retest1 | 2025-05-05 10:50:00 | 312.60 | 2025-05-05 10:55:00 | 311.55 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-05-08 10:00:00 | 325.45 | 2025-05-08 10:45:00 | 327.43 | PARTIAL | 0.50 | 0.61% |
| BUY | retest1 | 2025-05-08 10:00:00 | 325.45 | 2025-05-08 11:00:00 | 325.45 | STOP_HIT | 0.50 | 0.00% |
