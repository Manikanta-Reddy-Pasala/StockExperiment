# Cochin Shipyard Ltd. (COCHINSHIP)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2024-11-29 15:25:00 (25793 bars)
- **Last close:** 1577.00
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
| ENTRY1 | 35 |
| ENTRY2 | 0 |
| PARTIAL | 10 |
| TARGET_HIT | 7 |
| STOP_HIT | 28 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 28
- **Target hits / Stop hits / Partials:** 7 / 28 / 10
- **Avg / median % per leg:** 0.29% / -0.32%
- **Sum % (uncompounded):** 12.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 12 | 41.4% | 5 | 17 | 7 | 0.46% | 13.4% |
| BUY @ 2nd Alert (retest1) | 29 | 12 | 41.4% | 5 | 17 | 7 | 0.46% | 13.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 5 | 31.2% | 2 | 11 | 3 | -0.03% | -0.5% |
| SELL @ 2nd Alert (retest1) | 16 | 5 | 31.2% | 2 | 11 | 3 | -0.03% | -0.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 45 | 17 | 37.8% | 7 | 28 | 10 | 0.29% | 12.9% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-15 09:35:00 | 271.85 | 270.81 | 0.00 | ORB-long ORB[268.35,271.55] vol=3.3x ATR=1.22 |
| Stop hit — per-position SL triggered | 2023-05-15 10:15:00 | 270.63 | 271.30 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-16 09:50:00 | 272.50 | 271.20 | 0.00 | ORB-long ORB[268.85,271.95] vol=2.9x ATR=0.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-16 09:55:00 | 273.88 | 271.78 | 0.00 | T1 1.5R @ 273.88 |
| Target hit | 2023-05-16 14:40:00 | 278.50 | 278.71 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2023-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-23 09:35:00 | 247.75 | 245.26 | 0.00 | ORB-long ORB[243.00,245.75] vol=2.2x ATR=1.08 |
| Stop hit — per-position SL triggered | 2023-05-23 10:00:00 | 246.67 | 245.93 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-25 09:30:00 | 240.65 | 243.27 | 0.00 | ORB-short ORB[242.73,245.25] vol=1.7x ATR=0.99 |
| Stop hit — per-position SL triggered | 2023-05-25 09:40:00 | 241.64 | 242.72 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2023-05-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-26 09:30:00 | 236.35 | 238.16 | 0.00 | ORB-short ORB[237.05,239.75] vol=2.1x ATR=1.10 |
| Stop hit — per-position SL triggered | 2023-05-26 10:55:00 | 237.45 | 237.00 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-30 10:30:00 | 246.38 | 244.95 | 0.00 | ORB-long ORB[242.08,245.73] vol=2.1x ATR=0.94 |
| Stop hit — per-position SL triggered | 2023-05-30 10:40:00 | 245.44 | 245.02 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-31 11:15:00 | 245.05 | 247.81 | 0.00 | ORB-short ORB[247.90,251.00] vol=2.1x ATR=0.78 |
| Stop hit — per-position SL triggered | 2023-05-31 12:05:00 | 245.83 | 247.49 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2023-06-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-01 10:40:00 | 250.83 | 247.37 | 0.00 | ORB-long ORB[245.50,248.85] vol=2.8x ATR=0.82 |
| Stop hit — per-position SL triggered | 2023-06-01 10:45:00 | 250.01 | 247.50 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2023-06-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-02 09:30:00 | 251.88 | 250.72 | 0.00 | ORB-long ORB[249.50,251.33] vol=3.0x ATR=1.12 |
| Stop hit — per-position SL triggered | 2023-06-02 09:50:00 | 250.76 | 251.14 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2023-06-05 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-05 10:20:00 | 253.88 | 251.30 | 0.00 | ORB-long ORB[249.53,252.48] vol=4.8x ATR=1.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-05 10:30:00 | 255.59 | 252.44 | 0.00 | T1 1.5R @ 255.59 |
| Target hit | 2023-06-05 14:35:00 | 276.50 | 276.76 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — SELL (started 2023-06-14 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-14 11:00:00 | 275.18 | 276.14 | 0.00 | ORB-short ORB[275.50,278.00] vol=2.1x ATR=0.70 |
| Stop hit — per-position SL triggered | 2023-06-14 12:15:00 | 275.88 | 275.88 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2023-06-15 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-15 09:30:00 | 280.45 | 277.22 | 0.00 | ORB-long ORB[273.05,275.83] vol=8.2x ATR=1.21 |
| Stop hit — per-position SL triggered | 2023-06-15 09:35:00 | 279.24 | 278.34 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-28 10:55:00 | 286.95 | 289.76 | 0.00 | ORB-short ORB[288.77,291.35] vol=1.7x ATR=1.02 |
| Stop hit — per-position SL triggered | 2023-06-28 11:00:00 | 287.97 | 289.69 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-07-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-10 11:05:00 | 291.20 | 288.08 | 0.00 | ORB-long ORB[286.63,290.45] vol=4.9x ATR=1.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-10 11:20:00 | 293.04 | 289.47 | 0.00 | T1 1.5R @ 293.04 |
| Stop hit — per-position SL triggered | 2023-07-10 11:40:00 | 291.20 | 290.38 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2023-07-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-14 09:30:00 | 315.48 | 319.08 | 0.00 | ORB-short ORB[317.50,321.95] vol=1.5x ATR=2.11 |
| Target hit | 2023-07-14 15:20:00 | 313.80 | 316.41 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2023-07-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 09:35:00 | 339.85 | 341.46 | 0.00 | ORB-short ORB[340.65,344.45] vol=1.6x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 10:05:00 | 337.48 | 340.70 | 0.00 | T1 1.5R @ 337.48 |
| Target hit | 2023-07-27 15:20:00 | 337.98 | 337.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 17 — BUY (started 2023-07-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-07-28 09:45:00 | 340.85 | 337.81 | 0.00 | ORB-long ORB[335.50,338.48] vol=2.1x ATR=1.67 |
| Stop hit — per-position SL triggered | 2023-07-28 10:00:00 | 339.18 | 338.92 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2023-08-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-30 09:35:00 | 431.48 | 427.25 | 0.00 | ORB-long ORB[421.50,425.33] vol=8.6x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-30 09:40:00 | 433.99 | 429.40 | 0.00 | T1 1.5R @ 433.99 |
| Target hit | 2023-08-30 13:55:00 | 442.80 | 444.66 | 0.00 | Trail-exit close<VWAP |

### Cycle 19 — BUY (started 2023-09-04 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-04 10:25:00 | 458.90 | 455.61 | 0.00 | ORB-long ORB[453.00,458.80] vol=2.4x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-04 10:30:00 | 461.85 | 460.43 | 0.00 | T1 1.5R @ 461.85 |
| Target hit | 2023-09-04 11:05:00 | 461.38 | 461.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — BUY (started 2023-10-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-19 10:25:00 | 515.92 | 512.47 | 0.00 | ORB-long ORB[508.53,514.98] vol=1.7x ATR=1.89 |
| Stop hit — per-position SL triggered | 2023-10-19 10:30:00 | 514.03 | 512.58 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2023-11-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-01 09:30:00 | 478.20 | 475.64 | 0.00 | ORB-long ORB[471.58,477.90] vol=2.1x ATR=2.24 |
| Stop hit — per-position SL triggered | 2023-11-01 09:45:00 | 475.96 | 475.95 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2023-11-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-10 09:35:00 | 522.28 | 518.34 | 0.00 | ORB-long ORB[515.00,519.45] vol=2.3x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-10 09:55:00 | 526.44 | 520.82 | 0.00 | T1 1.5R @ 526.44 |
| Target hit | 2023-11-10 11:05:00 | 526.80 | 527.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 23 — SELL (started 2023-11-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-20 09:30:00 | 536.50 | 539.59 | 0.00 | ORB-short ORB[537.25,545.00] vol=2.1x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-20 09:35:00 | 533.33 | 538.93 | 0.00 | T1 1.5R @ 533.33 |
| Stop hit — per-position SL triggered | 2023-11-20 09:40:00 | 536.50 | 538.64 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2023-11-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-21 10:45:00 | 551.83 | 545.69 | 0.00 | ORB-long ORB[542.75,548.70] vol=4.7x ATR=2.23 |
| Stop hit — per-position SL triggered | 2023-11-21 10:50:00 | 549.60 | 547.13 | 0.00 | SL hit |

### Cycle 25 — BUY (started 2023-11-28 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-28 10:55:00 | 571.00 | 566.70 | 0.00 | ORB-long ORB[560.50,569.00] vol=5.3x ATR=2.40 |
| Stop hit — per-position SL triggered | 2023-11-28 11:00:00 | 568.60 | 566.84 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2023-12-06 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-06 10:35:00 | 602.50 | 592.85 | 0.00 | ORB-long ORB[589.23,595.20] vol=2.3x ATR=2.88 |
| Stop hit — per-position SL triggered | 2023-12-06 10:45:00 | 599.62 | 596.00 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-12-14 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-14 09:40:00 | 633.48 | 637.69 | 0.00 | ORB-short ORB[634.88,642.73] vol=1.5x ATR=2.44 |
| Stop hit — per-position SL triggered | 2023-12-14 09:45:00 | 635.92 | 637.08 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-01-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-01 09:55:00 | 689.90 | 684.99 | 0.00 | ORB-long ORB[679.25,686.83] vol=2.2x ATR=2.49 |
| Stop hit — per-position SL triggered | 2024-01-01 10:00:00 | 687.41 | 685.53 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-01-02 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 09:55:00 | 676.30 | 682.29 | 0.00 | ORB-short ORB[680.18,686.10] vol=2.6x ATR=2.79 |
| Stop hit — per-position SL triggered | 2024-01-02 10:05:00 | 679.09 | 680.41 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-01-03 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 09:40:00 | 670.05 | 675.91 | 0.00 | ORB-short ORB[673.03,679.98] vol=1.8x ATR=2.75 |
| Stop hit — per-position SL triggered | 2024-01-03 09:45:00 | 672.80 | 675.58 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-01-05 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-05 10:25:00 | 691.23 | 687.44 | 0.00 | ORB-long ORB[681.68,689.00] vol=3.9x ATR=2.65 |
| Stop hit — per-position SL triggered | 2024-01-05 10:30:00 | 688.58 | 687.50 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-02-07 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 11:05:00 | 879.35 | 888.35 | 0.00 | ORB-short ORB[887.00,897.00] vol=1.7x ATR=3.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-07 11:30:00 | 874.74 | 886.86 | 0.00 | T1 1.5R @ 874.74 |
| Stop hit — per-position SL triggered | 2024-02-07 11:40:00 | 879.35 | 886.23 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-02-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-08 11:00:00 | 866.80 | 871.43 | 0.00 | ORB-short ORB[867.10,878.55] vol=2.0x ATR=3.73 |
| Stop hit — per-position SL triggered | 2024-02-08 11:15:00 | 870.53 | 870.98 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-02-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 10:00:00 | 862.95 | 857.33 | 0.00 | ORB-long ORB[853.00,860.00] vol=2.2x ATR=3.14 |
| Stop hit — per-position SL triggered | 2024-02-27 10:10:00 | 859.81 | 858.30 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-03-07 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-07 09:55:00 | 866.00 | 861.03 | 0.00 | ORB-long ORB[853.85,865.00] vol=1.6x ATR=3.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-07 10:00:00 | 871.43 | 862.95 | 0.00 | T1 1.5R @ 871.43 |
| Stop hit — per-position SL triggered | 2024-03-07 10:40:00 | 866.00 | 865.79 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-15 09:35:00 | 271.85 | 2023-05-15 10:15:00 | 270.63 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2023-05-16 09:50:00 | 272.50 | 2023-05-16 09:55:00 | 273.88 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-05-16 09:50:00 | 272.50 | 2023-05-16 14:40:00 | 278.50 | TARGET_HIT | 0.50 | 2.20% |
| BUY | retest1 | 2023-05-23 09:35:00 | 247.75 | 2023-05-23 10:00:00 | 246.67 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2023-05-25 09:30:00 | 240.65 | 2023-05-25 09:40:00 | 241.64 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-05-26 09:30:00 | 236.35 | 2023-05-26 10:55:00 | 237.45 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2023-05-30 10:30:00 | 246.38 | 2023-05-30 10:40:00 | 245.44 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2023-05-31 11:15:00 | 245.05 | 2023-05-31 12:05:00 | 245.83 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-06-01 10:40:00 | 250.83 | 2023-06-01 10:45:00 | 250.01 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-06-02 09:30:00 | 251.88 | 2023-06-02 09:50:00 | 250.76 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-06-05 10:20:00 | 253.88 | 2023-06-05 10:30:00 | 255.59 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2023-06-05 10:20:00 | 253.88 | 2023-06-05 14:35:00 | 276.50 | TARGET_HIT | 0.50 | 8.91% |
| SELL | retest1 | 2023-06-14 11:00:00 | 275.18 | 2023-06-14 12:15:00 | 275.88 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-06-15 09:30:00 | 280.45 | 2023-06-15 09:35:00 | 279.24 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2023-06-28 10:55:00 | 286.95 | 2023-06-28 11:00:00 | 287.97 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2023-07-10 11:05:00 | 291.20 | 2023-07-10 11:20:00 | 293.04 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2023-07-10 11:05:00 | 291.20 | 2023-07-10 11:40:00 | 291.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-14 09:30:00 | 315.48 | 2023-07-14 15:20:00 | 313.80 | TARGET_HIT | 1.00 | 0.53% |
| SELL | retest1 | 2023-07-27 09:35:00 | 339.85 | 2023-07-27 10:05:00 | 337.48 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2023-07-27 09:35:00 | 339.85 | 2023-07-27 15:20:00 | 337.98 | TARGET_HIT | 0.50 | 0.55% |
| BUY | retest1 | 2023-07-28 09:45:00 | 340.85 | 2023-07-28 10:00:00 | 339.18 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2023-08-30 09:35:00 | 431.48 | 2023-08-30 09:40:00 | 433.99 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2023-08-30 09:35:00 | 431.48 | 2023-08-30 13:55:00 | 442.80 | TARGET_HIT | 0.50 | 2.62% |
| BUY | retest1 | 2023-09-04 10:25:00 | 458.90 | 2023-09-04 10:30:00 | 461.85 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2023-09-04 10:25:00 | 458.90 | 2023-09-04 11:05:00 | 461.38 | TARGET_HIT | 0.50 | 0.54% |
| BUY | retest1 | 2023-10-19 10:25:00 | 515.92 | 2023-10-19 10:30:00 | 514.03 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2023-11-01 09:30:00 | 478.20 | 2023-11-01 09:45:00 | 475.96 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2023-11-10 09:35:00 | 522.28 | 2023-11-10 09:55:00 | 526.44 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2023-11-10 09:35:00 | 522.28 | 2023-11-10 11:05:00 | 526.80 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2023-11-20 09:30:00 | 536.50 | 2023-11-20 09:35:00 | 533.33 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2023-11-20 09:30:00 | 536.50 | 2023-11-20 09:40:00 | 536.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-11-21 10:45:00 | 551.83 | 2023-11-21 10:50:00 | 549.60 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2023-11-28 10:55:00 | 571.00 | 2023-11-28 11:00:00 | 568.60 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2023-12-06 10:35:00 | 602.50 | 2023-12-06 10:45:00 | 599.62 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2023-12-14 09:40:00 | 633.48 | 2023-12-14 09:45:00 | 635.92 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-01-01 09:55:00 | 689.90 | 2024-01-01 10:00:00 | 687.41 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-01-02 09:55:00 | 676.30 | 2024-01-02 10:05:00 | 679.09 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-01-03 09:40:00 | 670.05 | 2024-01-03 09:45:00 | 672.80 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-01-05 10:25:00 | 691.23 | 2024-01-05 10:30:00 | 688.58 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-02-07 11:05:00 | 879.35 | 2024-02-07 11:30:00 | 874.74 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-02-07 11:05:00 | 879.35 | 2024-02-07 11:40:00 | 879.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-08 11:00:00 | 866.80 | 2024-02-08 11:15:00 | 870.53 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-02-27 10:00:00 | 862.95 | 2024-02-27 10:10:00 | 859.81 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-03-07 09:55:00 | 866.00 | 2024-03-07 10:00:00 | 871.43 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-03-07 09:55:00 | 866.00 | 2024-03-07 10:40:00 | 866.00 | STOP_HIT | 0.50 | 0.00% |
