# SHRIRAMFIN (SHRIRAMFIN)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1003.05
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
| PARTIAL | 23 |
| TARGET_HIT | 11 |
| STOP_HIT | 63 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 97 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 34 / 63
- **Target hits / Stop hits / Partials:** 11 / 63 / 23
- **Avg / median % per leg:** 0.12% / -0.19%
- **Sum % (uncompounded):** 11.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 21 | 41.2% | 7 | 30 | 14 | 0.18% | 9.3% |
| BUY @ 2nd Alert (retest1) | 51 | 21 | 41.2% | 7 | 30 | 14 | 0.18% | 9.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 46 | 13 | 28.3% | 4 | 33 | 9 | 0.04% | 2.0% |
| SELL @ 2nd Alert (retest1) | 46 | 13 | 28.3% | 4 | 33 | 9 | 0.04% | 2.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 97 | 34 | 35.1% | 11 | 63 | 23 | 0.12% | 11.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-16 11:15:00 | 662.65 | 667.94 | 0.00 | ORB-short ORB[669.40,674.80] vol=3.1x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-05-16 15:00:00 | 664.79 | 665.38 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-19 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-19 11:05:00 | 672.65 | 668.92 | 0.00 | ORB-long ORB[666.40,672.00] vol=3.0x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-05-19 11:10:00 | 670.77 | 668.97 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-05-26 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-26 11:05:00 | 655.60 | 657.91 | 0.00 | ORB-short ORB[658.45,665.60] vol=3.5x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-05-26 11:15:00 | 657.27 | 657.84 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-27 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-27 11:00:00 | 661.65 | 662.70 | 0.00 | ORB-short ORB[662.35,670.00] vol=2.2x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-27 11:15:00 | 658.88 | 662.45 | 0.00 | T1 1.5R @ 658.88 |
| Stop hit — per-position SL triggered | 2025-05-27 11:30:00 | 661.65 | 662.02 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2025-06-06 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-06 10:00:00 | 646.00 | 647.46 | 0.00 | ORB-short ORB[646.05,654.00] vol=3.4x ATR=1.87 |
| Stop hit — per-position SL triggered | 2025-06-06 10:05:00 | 647.87 | 651.47 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 10:45:00 | 678.55 | 675.08 | 0.00 | ORB-long ORB[668.60,677.00] vol=1.6x ATR=2.74 |
| Stop hit — per-position SL triggered | 2025-06-16 11:10:00 | 675.81 | 675.97 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-18 11:15:00 | 661.60 | 667.71 | 0.00 | ORB-short ORB[662.15,670.75] vol=1.6x ATR=1.80 |
| Stop hit — per-position SL triggered | 2025-06-18 12:05:00 | 663.40 | 665.81 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:45:00 | 665.90 | 659.49 | 0.00 | ORB-long ORB[650.50,658.00] vol=1.6x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 11:20:00 | 669.54 | 661.83 | 0.00 | T1 1.5R @ 669.54 |
| Stop hit — per-position SL triggered | 2025-06-20 12:00:00 | 665.90 | 663.45 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2025-06-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-25 10:30:00 | 673.30 | 675.55 | 0.00 | ORB-short ORB[674.00,678.00] vol=2.6x ATR=1.72 |
| Stop hit — per-position SL triggered | 2025-06-25 10:50:00 | 675.02 | 675.19 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-30 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 10:10:00 | 710.65 | 706.39 | 0.00 | ORB-long ORB[699.90,705.35] vol=2.2x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-06-30 10:20:00 | 708.48 | 707.64 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-07-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 10:55:00 | 679.25 | 675.76 | 0.00 | ORB-long ORB[671.15,678.35] vol=2.1x ATR=1.59 |
| Stop hit — per-position SL triggered | 2025-07-07 11:10:00 | 677.66 | 676.05 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-07-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-09 09:30:00 | 677.45 | 674.11 | 0.00 | ORB-long ORB[671.00,675.00] vol=1.7x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-07-09 09:35:00 | 675.66 | 674.35 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-07-10 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-10 11:00:00 | 677.35 | 683.15 | 0.00 | ORB-short ORB[682.00,687.35] vol=1.6x ATR=1.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 11:20:00 | 675.29 | 682.24 | 0.00 | T1 1.5R @ 675.29 |
| Target hit | 2025-07-10 15:20:00 | 671.60 | 676.58 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2025-07-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-15 09:35:00 | 678.20 | 675.40 | 0.00 | ORB-long ORB[670.70,676.35] vol=1.8x ATR=1.49 |
| Stop hit — per-position SL triggered | 2025-07-15 09:40:00 | 676.71 | 675.56 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2025-07-17 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:10:00 | 667.35 | 668.47 | 0.00 | ORB-short ORB[668.50,672.00] vol=1.6x ATR=1.05 |
| Stop hit — per-position SL triggered | 2025-07-17 11:20:00 | 668.40 | 668.45 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-18 09:45:00 | 661.95 | 664.25 | 0.00 | ORB-short ORB[665.05,670.50] vol=6.4x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-07-18 09:55:00 | 663.48 | 664.17 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-24 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 10:20:00 | 637.45 | 649.64 | 0.00 | ORB-short ORB[652.25,660.00] vol=3.6x ATR=3.07 |
| Stop hit — per-position SL triggered | 2025-07-24 10:45:00 | 640.52 | 646.05 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-07-30 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-30 10:30:00 | 631.60 | 634.09 | 0.00 | ORB-short ORB[633.20,638.35] vol=2.4x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-07-30 10:50:00 | 633.34 | 633.91 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-07-31 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 10:25:00 | 630.25 | 627.46 | 0.00 | ORB-long ORB[623.05,629.40] vol=1.7x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 10:55:00 | 633.29 | 628.57 | 0.00 | T1 1.5R @ 633.29 |
| Target hit | 2025-07-31 14:30:00 | 633.45 | 633.58 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — SELL (started 2025-08-06 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-06 10:05:00 | 629.90 | 632.31 | 0.00 | ORB-short ORB[630.55,639.80] vol=4.1x ATR=2.44 |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 632.34 | 631.99 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-08-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-07 11:00:00 | 619.40 | 625.33 | 0.00 | ORB-short ORB[624.20,628.65] vol=1.6x ATR=1.63 |
| Stop hit — per-position SL triggered | 2025-08-07 11:30:00 | 621.03 | 624.17 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-08-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-08 09:30:00 | 621.30 | 624.13 | 0.00 | ORB-short ORB[623.10,626.85] vol=1.5x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-08-08 09:35:00 | 623.00 | 623.61 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-08-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-20 10:55:00 | 614.70 | 617.44 | 0.00 | ORB-short ORB[616.45,624.95] vol=2.1x ATR=1.14 |
| Stop hit — per-position SL triggered | 2025-08-20 12:40:00 | 615.84 | 616.37 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2025-08-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:35:00 | 622.45 | 619.48 | 0.00 | ORB-long ORB[616.40,620.00] vol=2.2x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-08-25 10:00:00 | 621.02 | 620.65 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-26 09:30:00 | 611.45 | 615.39 | 0.00 | ORB-short ORB[613.50,620.45] vol=2.1x ATR=1.41 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:40:00 | 609.34 | 613.36 | 0.00 | T1 1.5R @ 609.34 |
| Target hit | 2025-08-26 15:20:00 | 594.95 | 600.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-09-01 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 10:45:00 | 583.90 | 582.10 | 0.00 | ORB-long ORB[578.60,583.05] vol=2.3x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 12:20:00 | 586.19 | 582.98 | 0.00 | T1 1.5R @ 586.19 |
| Target hit | 2025-09-01 15:20:00 | 587.95 | 584.93 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 27 — BUY (started 2025-09-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:40:00 | 593.35 | 590.09 | 0.00 | ORB-long ORB[587.45,590.35] vol=2.5x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-02 09:45:00 | 595.21 | 591.37 | 0.00 | T1 1.5R @ 595.21 |
| Stop hit — per-position SL triggered | 2025-09-02 09:55:00 | 593.35 | 591.82 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-09-03 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-03 10:00:00 | 587.80 | 583.14 | 0.00 | ORB-long ORB[580.35,586.00] vol=1.6x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 586.06 | 583.97 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-08 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 10:25:00 | 599.65 | 598.20 | 0.00 | ORB-long ORB[592.85,598.00] vol=1.9x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-09-08 11:00:00 | 598.04 | 598.27 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:30:00 | 605.35 | 601.63 | 0.00 | ORB-long ORB[596.70,600.45] vol=2.0x ATR=1.40 |
| Stop hit — per-position SL triggered | 2025-09-10 11:20:00 | 603.95 | 603.01 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-11 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 09:50:00 | 608.80 | 607.44 | 0.00 | ORB-long ORB[602.80,606.45] vol=2.3x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 10:10:00 | 610.79 | 608.15 | 0.00 | T1 1.5R @ 610.79 |
| Target hit | 2025-09-11 15:20:00 | 620.00 | 618.02 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 32 — BUY (started 2025-09-12 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:55:00 | 625.00 | 623.09 | 0.00 | ORB-long ORB[619.25,624.20] vol=2.2x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 10:00:00 | 626.90 | 623.60 | 0.00 | T1 1.5R @ 626.90 |
| Target hit | 2025-09-12 13:15:00 | 628.65 | 628.82 | 0.00 | Trail-exit close<VWAP |

### Cycle 33 — SELL (started 2025-09-15 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-15 11:10:00 | 624.75 | 627.06 | 0.00 | ORB-short ORB[625.70,631.60] vol=1.6x ATR=1.13 |
| Stop hit — per-position SL triggered | 2025-09-15 12:50:00 | 625.88 | 626.11 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:35:00 | 622.95 | 621.05 | 0.00 | ORB-long ORB[617.75,621.60] vol=1.7x ATR=1.47 |
| Stop hit — per-position SL triggered | 2025-09-17 09:40:00 | 621.48 | 621.22 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-18 09:40:00 | 627.55 | 623.83 | 0.00 | ORB-long ORB[620.00,625.65] vol=2.0x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-09-18 09:45:00 | 625.67 | 624.05 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-29 10:40:00 | 612.30 | 610.45 | 0.00 | ORB-long ORB[607.10,611.55] vol=4.2x ATR=1.43 |
| Stop hit — per-position SL triggered | 2025-09-29 10:50:00 | 610.87 | 610.51 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-10-03 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-03 11:10:00 | 640.80 | 642.98 | 0.00 | ORB-short ORB[643.00,650.00] vol=5.9x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-10-03 11:35:00 | 642.63 | 642.80 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-08 10:15:00 | 670.60 | 667.28 | 0.00 | ORB-long ORB[662.00,669.00] vol=1.9x ATR=1.94 |
| Stop hit — per-position SL triggered | 2025-10-08 10:40:00 | 668.66 | 668.52 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-10-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-13 09:45:00 | 668.60 | 664.70 | 0.00 | ORB-long ORB[658.75,666.45] vol=1.9x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-10-13 09:50:00 | 666.62 | 664.83 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2025-10-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 09:55:00 | 726.75 | 722.85 | 0.00 | ORB-long ORB[718.60,723.15] vol=2.1x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 11:25:00 | 729.41 | 726.28 | 0.00 | T1 1.5R @ 729.41 |
| Stop hit — per-position SL triggered | 2025-10-27 11:55:00 | 726.75 | 726.68 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 10:15:00 | 716.75 | 717.94 | 0.00 | ORB-short ORB[717.15,723.85] vol=2.0x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-28 10:25:00 | 713.26 | 717.52 | 0.00 | T1 1.5R @ 713.26 |
| Stop hit — per-position SL triggered | 2025-10-28 11:25:00 | 716.75 | 716.19 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-11-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 09:55:00 | 795.40 | 790.84 | 0.00 | ORB-long ORB[785.60,795.20] vol=1.7x ATR=3.28 |
| Stop hit — per-position SL triggered | 2025-11-06 10:35:00 | 792.12 | 791.46 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2025-11-10 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-10 11:10:00 | 825.90 | 821.07 | 0.00 | ORB-long ORB[816.65,825.40] vol=2.5x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 11:20:00 | 829.26 | 821.71 | 0.00 | T1 1.5R @ 829.26 |
| Stop hit — per-position SL triggered | 2025-11-10 11:40:00 | 825.90 | 822.27 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-11-13 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-13 10:00:00 | 813.80 | 819.34 | 0.00 | ORB-short ORB[815.05,826.35] vol=1.8x ATR=2.67 |
| Stop hit — per-position SL triggered | 2025-11-13 10:05:00 | 816.47 | 819.03 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-11-18 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-18 10:05:00 | 820.50 | 818.38 | 0.00 | ORB-long ORB[810.00,820.00] vol=1.8x ATR=2.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-18 10:30:00 | 823.68 | 819.03 | 0.00 | T1 1.5R @ 823.68 |
| Stop hit — per-position SL triggered | 2025-11-18 11:30:00 | 820.50 | 821.34 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-24 10:15:00 | 837.45 | 830.98 | 0.00 | ORB-long ORB[821.10,830.75] vol=2.8x ATR=2.59 |
| Stop hit — per-position SL triggered | 2025-11-24 10:50:00 | 834.86 | 833.22 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2025-11-27 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:35:00 | 864.90 | 861.17 | 0.00 | ORB-long ORB[857.25,862.80] vol=1.6x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-27 09:50:00 | 868.67 | 862.86 | 0.00 | T1 1.5R @ 868.67 |
| Stop hit — per-position SL triggered | 2025-11-27 10:00:00 | 864.90 | 863.24 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-11-28 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-28 11:00:00 | 862.80 | 866.18 | 0.00 | ORB-short ORB[864.50,872.00] vol=5.5x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-28 11:05:00 | 860.44 | 865.96 | 0.00 | T1 1.5R @ 860.44 |
| Target hit | 2025-11-28 15:20:00 | 850.15 | 855.32 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-12-01 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-01 09:40:00 | 851.60 | 854.91 | 0.00 | ORB-short ORB[853.65,861.95] vol=2.7x ATR=2.53 |
| Stop hit — per-position SL triggered | 2025-12-01 10:00:00 | 854.13 | 854.61 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-09 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 10:05:00 | 823.15 | 823.67 | 0.00 | ORB-short ORB[824.00,830.65] vol=2.2x ATR=2.89 |
| Stop hit — per-position SL triggered | 2025-12-09 10:10:00 | 826.04 | 824.02 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-12-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-10 10:55:00 | 844.00 | 850.26 | 0.00 | ORB-short ORB[845.25,853.45] vol=1.8x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-12-10 11:10:00 | 846.32 | 849.50 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:30:00 | 848.50 | 845.63 | 0.00 | ORB-long ORB[835.70,847.65] vol=1.6x ATR=2.95 |
| Stop hit — per-position SL triggered | 2025-12-11 11:00:00 | 845.55 | 846.13 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:15:00 | 872.80 | 869.74 | 0.00 | ORB-long ORB[864.20,872.50] vol=1.7x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-12-18 10:40:00 | 870.49 | 870.18 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-12-19 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-19 10:20:00 | 860.20 | 862.11 | 0.00 | ORB-short ORB[860.45,869.00] vol=3.6x ATR=2.81 |
| Stop hit — per-position SL triggered | 2025-12-19 11:25:00 | 863.01 | 861.75 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-12-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 11:05:00 | 940.75 | 927.82 | 0.00 | ORB-long ORB[915.00,929.00] vol=1.8x ATR=3.52 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-22 11:20:00 | 946.03 | 929.61 | 0.00 | T1 1.5R @ 946.03 |
| Stop hit — per-position SL triggered | 2025-12-22 13:10:00 | 940.75 | 934.01 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2025-12-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:35:00 | 945.70 | 940.04 | 0.00 | ORB-long ORB[931.70,943.20] vol=1.6x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-23 12:05:00 | 949.88 | 943.33 | 0.00 | T1 1.5R @ 949.88 |
| Target hit | 2025-12-23 15:20:00 | 957.60 | 948.89 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 57 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 10:15:00 | 950.70 | 958.78 | 0.00 | ORB-short ORB[959.15,966.00] vol=1.9x ATR=2.45 |
| Stop hit — per-position SL triggered | 2025-12-29 10:20:00 | 953.15 | 957.73 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:15:00 | 992.75 | 987.97 | 0.00 | ORB-long ORB[979.40,986.00] vol=1.7x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-12-31 11:25:00 | 990.47 | 988.32 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 11:10:00 | 1005.25 | 1008.15 | 0.00 | ORB-short ORB[1007.40,1014.80] vol=1.5x ATR=2.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 11:30:00 | 1001.66 | 1007.27 | 0.00 | T1 1.5R @ 1001.66 |
| Stop hit — per-position SL triggered | 2026-01-05 11:40:00 | 1005.25 | 1007.12 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:15:00 | 1000.10 | 1007.29 | 0.00 | ORB-short ORB[1002.50,1010.65] vol=1.6x ATR=2.54 |
| Stop hit — per-position SL triggered | 2026-01-06 11:45:00 | 1002.64 | 1005.84 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2026-01-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 11:00:00 | 992.15 | 997.22 | 0.00 | ORB-short ORB[996.25,1001.95] vol=1.5x ATR=2.40 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:10:00 | 988.55 | 996.62 | 0.00 | T1 1.5R @ 988.55 |
| Stop hit — per-position SL triggered | 2026-01-08 13:05:00 | 992.15 | 992.88 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2026-01-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-09 09:35:00 | 981.70 | 986.01 | 0.00 | ORB-short ORB[983.50,993.20] vol=1.9x ATR=3.08 |
| Stop hit — per-position SL triggered | 2026-01-09 09:45:00 | 984.78 | 985.29 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2026-01-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-13 10:50:00 | 982.00 | 976.93 | 0.00 | ORB-long ORB[971.00,979.95] vol=2.8x ATR=3.56 |
| Stop hit — per-position SL triggered | 2026-01-13 11:05:00 | 978.44 | 977.33 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-14 11:15:00 | 989.60 | 982.93 | 0.00 | ORB-long ORB[975.05,981.95] vol=2.0x ATR=2.53 |
| Stop hit — per-position SL triggered | 2026-01-14 11:25:00 | 987.07 | 983.36 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2026-02-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-06 11:05:00 | 975.40 | 986.50 | 0.00 | ORB-short ORB[990.50,998.50] vol=2.2x ATR=2.93 |
| Stop hit — per-position SL triggered | 2026-02-06 11:10:00 | 978.33 | 986.22 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-02-09 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:10:00 | 1019.70 | 1013.57 | 0.00 | ORB-long ORB[1003.40,1011.70] vol=1.7x ATR=2.83 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:15:00 | 1023.95 | 1014.04 | 0.00 | T1 1.5R @ 1023.95 |
| Target hit | 2026-02-09 15:20:00 | 1064.90 | 1040.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2026-02-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-17 10:45:00 | 1061.80 | 1066.36 | 0.00 | ORB-short ORB[1067.20,1081.90] vol=2.1x ATR=3.29 |
| Stop hit — per-position SL triggered | 2026-02-17 11:00:00 | 1065.09 | 1066.01 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2026-02-25 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-25 10:30:00 | 1077.70 | 1070.45 | 0.00 | ORB-long ORB[1055.80,1065.00] vol=1.8x ATR=3.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-25 11:15:00 | 1082.22 | 1072.75 | 0.00 | T1 1.5R @ 1082.22 |
| Target hit | 2026-02-25 15:20:00 | 1086.30 | 1080.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2026-03-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-20 09:35:00 | 945.40 | 957.38 | 0.00 | ORB-short ORB[954.50,966.90] vol=2.0x ATR=5.42 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-20 09:55:00 | 937.27 | 952.26 | 0.00 | T1 1.5R @ 937.27 |
| Stop hit — per-position SL triggered | 2026-03-20 10:45:00 | 945.40 | 943.63 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-03-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 09:45:00 | 902.40 | 908.35 | 0.00 | ORB-short ORB[903.00,916.00] vol=2.0x ATR=6.03 |
| Stop hit — per-position SL triggered | 2026-03-24 09:55:00 | 908.43 | 907.56 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-04-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-15 10:25:00 | 1028.70 | 1034.20 | 0.00 | ORB-short ORB[1030.00,1043.50] vol=2.1x ATR=3.98 |
| Stop hit — per-position SL triggered | 2026-04-15 11:40:00 | 1032.68 | 1032.91 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2026-04-17 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-17 09:50:00 | 1031.05 | 1024.85 | 0.00 | ORB-long ORB[1013.15,1025.60] vol=1.5x ATR=3.56 |
| Stop hit — per-position SL triggered | 2026-04-17 10:00:00 | 1027.49 | 1025.37 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2026-04-22 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 10:35:00 | 1030.25 | 1037.60 | 0.00 | ORB-short ORB[1036.35,1051.30] vol=2.3x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-04-22 10:40:00 | 1032.59 | 1037.13 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-04-28 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-28 11:05:00 | 967.70 | 973.36 | 0.00 | ORB-short ORB[968.00,976.25] vol=3.8x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:15:00 | 964.25 | 972.79 | 0.00 | T1 1.5R @ 964.25 |
| Target hit | 2026-04-28 15:20:00 | 952.10 | 959.62 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2025-05-16 11:15:00 | 662.65 | 2025-05-16 15:00:00 | 664.79 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-19 11:05:00 | 672.65 | 2025-05-19 11:10:00 | 670.77 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-05-26 11:05:00 | 655.60 | 2025-05-26 11:15:00 | 657.27 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-05-27 11:00:00 | 661.65 | 2025-05-27 11:15:00 | 658.88 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-05-27 11:00:00 | 661.65 | 2025-05-27 11:30:00 | 661.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-06 10:00:00 | 646.00 | 2025-06-06 10:05:00 | 647.87 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-06-16 10:45:00 | 678.55 | 2025-06-16 11:10:00 | 675.81 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-06-18 11:15:00 | 661.60 | 2025-06-18 12:05:00 | 663.40 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-20 10:45:00 | 665.90 | 2025-06-20 11:20:00 | 669.54 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-06-20 10:45:00 | 665.90 | 2025-06-20 12:00:00 | 665.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-25 10:30:00 | 673.30 | 2025-06-25 10:50:00 | 675.02 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-06-30 10:10:00 | 710.65 | 2025-06-30 10:20:00 | 708.48 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-07-07 10:55:00 | 679.25 | 2025-07-07 11:10:00 | 677.66 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-09 09:30:00 | 677.45 | 2025-07-09 09:35:00 | 675.66 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-07-10 11:00:00 | 677.35 | 2025-07-10 11:20:00 | 675.29 | PARTIAL | 0.50 | 0.30% |
| SELL | retest1 | 2025-07-10 11:00:00 | 677.35 | 2025-07-10 15:20:00 | 671.60 | TARGET_HIT | 0.50 | 0.85% |
| BUY | retest1 | 2025-07-15 09:35:00 | 678.20 | 2025-07-15 09:40:00 | 676.71 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-07-17 11:10:00 | 667.35 | 2025-07-17 11:20:00 | 668.40 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest1 | 2025-07-18 09:45:00 | 661.95 | 2025-07-18 09:55:00 | 663.48 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-07-24 10:20:00 | 637.45 | 2025-07-24 10:45:00 | 640.52 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2025-07-30 10:30:00 | 631.60 | 2025-07-30 10:50:00 | 633.34 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-31 10:25:00 | 630.25 | 2025-07-31 10:55:00 | 633.29 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-31 10:25:00 | 630.25 | 2025-07-31 14:30:00 | 633.45 | TARGET_HIT | 0.50 | 0.51% |
| SELL | retest1 | 2025-08-06 10:05:00 | 629.90 | 2025-08-06 10:15:00 | 632.34 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-08-07 11:00:00 | 619.40 | 2025-08-07 11:30:00 | 621.03 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-08-08 09:30:00 | 621.30 | 2025-08-08 09:35:00 | 623.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-08-20 10:55:00 | 614.70 | 2025-08-20 12:40:00 | 615.84 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-08-25 09:35:00 | 622.45 | 2025-08-25 10:00:00 | 621.02 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-08-26 09:30:00 | 611.45 | 2025-08-26 09:40:00 | 609.34 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-08-26 09:30:00 | 611.45 | 2025-08-26 15:20:00 | 594.95 | TARGET_HIT | 0.50 | 2.70% |
| BUY | retest1 | 2025-09-01 10:45:00 | 583.90 | 2025-09-01 12:20:00 | 586.19 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-09-01 10:45:00 | 583.90 | 2025-09-01 15:20:00 | 587.95 | TARGET_HIT | 0.50 | 0.69% |
| BUY | retest1 | 2025-09-02 09:40:00 | 593.35 | 2025-09-02 09:45:00 | 595.21 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2025-09-02 09:40:00 | 593.35 | 2025-09-02 09:55:00 | 593.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-03 10:00:00 | 587.80 | 2025-09-03 10:15:00 | 586.06 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-08 10:25:00 | 599.65 | 2025-09-08 11:00:00 | 598.04 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-09-10 10:30:00 | 605.35 | 2025-09-10 11:20:00 | 603.95 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-09-11 09:50:00 | 608.80 | 2025-09-11 10:10:00 | 610.79 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-09-11 09:50:00 | 608.80 | 2025-09-11 15:20:00 | 620.00 | TARGET_HIT | 0.50 | 1.84% |
| BUY | retest1 | 2025-09-12 09:55:00 | 625.00 | 2025-09-12 10:00:00 | 626.90 | PARTIAL | 0.50 | 0.30% |
| BUY | retest1 | 2025-09-12 09:55:00 | 625.00 | 2025-09-12 13:15:00 | 628.65 | TARGET_HIT | 0.50 | 0.58% |
| SELL | retest1 | 2025-09-15 11:10:00 | 624.75 | 2025-09-15 12:50:00 | 625.88 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-09-17 09:35:00 | 622.95 | 2025-09-17 09:40:00 | 621.48 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-09-18 09:40:00 | 627.55 | 2025-09-18 09:45:00 | 625.67 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-09-29 10:40:00 | 612.30 | 2025-09-29 10:50:00 | 610.87 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2025-10-03 11:10:00 | 640.80 | 2025-10-03 11:35:00 | 642.63 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-08 10:15:00 | 670.60 | 2025-10-08 10:40:00 | 668.66 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2025-10-13 09:45:00 | 668.60 | 2025-10-13 09:50:00 | 666.62 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2025-10-27 09:55:00 | 726.75 | 2025-10-27 11:25:00 | 729.41 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-10-27 09:55:00 | 726.75 | 2025-10-27 11:55:00 | 726.75 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-28 10:15:00 | 716.75 | 2025-10-28 10:25:00 | 713.26 | PARTIAL | 0.50 | 0.49% |
| SELL | retest1 | 2025-10-28 10:15:00 | 716.75 | 2025-10-28 11:25:00 | 716.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-06 09:55:00 | 795.40 | 2025-11-06 10:35:00 | 792.12 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2025-11-10 11:10:00 | 825.90 | 2025-11-10 11:20:00 | 829.26 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-11-10 11:10:00 | 825.90 | 2025-11-10 11:40:00 | 825.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-13 10:00:00 | 813.80 | 2025-11-13 10:05:00 | 816.47 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-11-18 10:05:00 | 820.50 | 2025-11-18 10:30:00 | 823.68 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-11-18 10:05:00 | 820.50 | 2025-11-18 11:30:00 | 820.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-11-24 10:15:00 | 837.45 | 2025-11-24 10:50:00 | 834.86 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-11-27 09:35:00 | 864.90 | 2025-11-27 09:50:00 | 868.67 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-11-27 09:35:00 | 864.90 | 2025-11-27 10:00:00 | 864.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-28 11:00:00 | 862.80 | 2025-11-28 11:05:00 | 860.44 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-11-28 11:00:00 | 862.80 | 2025-11-28 15:20:00 | 850.15 | TARGET_HIT | 0.50 | 1.47% |
| SELL | retest1 | 2025-12-01 09:40:00 | 851.60 | 2025-12-01 10:00:00 | 854.13 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-09 10:05:00 | 823.15 | 2025-12-09 10:10:00 | 826.04 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2025-12-10 10:55:00 | 844.00 | 2025-12-10 11:10:00 | 846.32 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-11 10:30:00 | 848.50 | 2025-12-11 11:00:00 | 845.55 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-12-18 10:15:00 | 872.80 | 2025-12-18 10:40:00 | 870.49 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-12-19 10:20:00 | 860.20 | 2025-12-19 11:25:00 | 863.01 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-12-22 11:05:00 | 940.75 | 2025-12-22 11:20:00 | 946.03 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-12-22 11:05:00 | 940.75 | 2025-12-22 13:10:00 | 940.75 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-23 10:35:00 | 945.70 | 2025-12-23 12:05:00 | 949.88 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2025-12-23 10:35:00 | 945.70 | 2025-12-23 15:20:00 | 957.60 | TARGET_HIT | 0.50 | 1.26% |
| SELL | retest1 | 2025-12-29 10:15:00 | 950.70 | 2025-12-29 10:20:00 | 953.15 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-12-31 11:15:00 | 992.75 | 2025-12-31 11:25:00 | 990.47 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-01-05 11:10:00 | 1005.25 | 2026-01-05 11:30:00 | 1001.66 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-05 11:10:00 | 1005.25 | 2026-01-05 11:40:00 | 1005.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-06 11:15:00 | 1000.10 | 2026-01-06 11:45:00 | 1002.64 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-08 11:00:00 | 992.15 | 2026-01-08 11:10:00 | 988.55 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-01-08 11:00:00 | 992.15 | 2026-01-08 13:05:00 | 992.15 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-09 09:35:00 | 981.70 | 2026-01-09 09:45:00 | 984.78 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-01-13 10:50:00 | 982.00 | 2026-01-13 11:05:00 | 978.44 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-01-14 11:15:00 | 989.60 | 2026-01-14 11:25:00 | 987.07 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2026-02-06 11:05:00 | 975.40 | 2026-02-06 11:10:00 | 978.33 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-02-09 10:10:00 | 1019.70 | 2026-02-09 10:15:00 | 1023.95 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-09 10:10:00 | 1019.70 | 2026-02-09 15:20:00 | 1064.90 | TARGET_HIT | 0.50 | 4.43% |
| SELL | retest1 | 2026-02-17 10:45:00 | 1061.80 | 2026-02-17 11:00:00 | 1065.09 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-25 10:30:00 | 1077.70 | 2026-02-25 11:15:00 | 1082.22 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2026-02-25 10:30:00 | 1077.70 | 2026-02-25 15:20:00 | 1086.30 | TARGET_HIT | 0.50 | 0.80% |
| SELL | retest1 | 2026-03-20 09:35:00 | 945.40 | 2026-03-20 09:55:00 | 937.27 | PARTIAL | 0.50 | 0.86% |
| SELL | retest1 | 2026-03-20 09:35:00 | 945.40 | 2026-03-20 10:45:00 | 945.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-24 09:45:00 | 902.40 | 2026-03-24 09:55:00 | 908.43 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest1 | 2026-04-15 10:25:00 | 1028.70 | 2026-04-15 11:40:00 | 1032.68 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-04-17 09:50:00 | 1031.05 | 2026-04-17 10:00:00 | 1027.49 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2026-04-22 10:35:00 | 1030.25 | 2026-04-22 10:40:00 | 1032.59 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-04-28 11:05:00 | 967.70 | 2026-04-28 11:15:00 | 964.25 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2026-04-28 11:05:00 | 967.70 | 2026-04-28 15:20:00 | 952.10 | TARGET_HIT | 0.50 | 1.61% |
