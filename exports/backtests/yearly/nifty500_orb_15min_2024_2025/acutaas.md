# Acutaas Chemicals Ltd. (ACUTAAS)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-04-30 15:25:00 (36496 bars)
- **Last close:** 2600.00
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
| ENTRY1 | 46 |
| ENTRY2 | 0 |
| PARTIAL | 22 |
| TARGET_HIT | 8 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 38
- **Target hits / Stop hits / Partials:** 8 / 38 / 22
- **Avg / median % per leg:** 0.19% / 0.00%
- **Sum % (uncompounded):** 12.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 19 | 50.0% | 7 | 19 | 12 | 0.23% | 8.9% |
| BUY @ 2nd Alert (retest1) | 38 | 19 | 50.0% | 7 | 19 | 12 | 0.23% | 8.9% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 30 | 11 | 36.7% | 1 | 19 | 10 | 0.13% | 3.9% |
| SELL @ 2nd Alert (retest1) | 30 | 11 | 36.7% | 1 | 19 | 10 | 0.13% | 3.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 68 | 30 | 44.1% | 8 | 38 | 22 | 0.19% | 12.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-15 09:55:00 | 607.83 | 608.17 | 0.00 | ORB-short ORB[608.00,612.00] vol=2.5x ATR=2.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-15 10:20:00 | 604.47 | 607.52 | 0.00 | T1 1.5R @ 604.47 |
| Stop hit — per-position SL triggered | 2024-05-15 10:35:00 | 607.83 | 607.32 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:50:00 | 609.15 | 610.82 | 0.00 | ORB-short ORB[610.58,617.48] vol=3.2x ATR=2.13 |
| Stop hit — per-position SL triggered | 2024-05-16 10:55:00 | 611.28 | 610.91 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:20:00 | 626.50 | 621.28 | 0.00 | ORB-long ORB[617.53,622.45] vol=2.5x ATR=2.84 |
| Stop hit — per-position SL triggered | 2024-05-17 10:30:00 | 623.66 | 622.79 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 09:35:00 | 610.20 | 615.38 | 0.00 | ORB-short ORB[614.00,620.73] vol=2.8x ATR=2.55 |
| Stop hit — per-position SL triggered | 2024-05-22 09:45:00 | 612.75 | 613.81 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:35:00 | 608.67 | 606.85 | 0.00 | ORB-long ORB[601.53,608.50] vol=2.9x ATR=2.10 |
| Stop hit — per-position SL triggered | 2024-05-23 09:45:00 | 606.57 | 606.91 | 0.00 | SL hit |

### Cycle 6 — SELL (started 2024-05-24 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-24 10:55:00 | 599.50 | 602.82 | 0.00 | ORB-short ORB[601.00,609.00] vol=2.4x ATR=1.98 |
| Stop hit — per-position SL triggered | 2024-05-24 11:05:00 | 601.48 | 602.23 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-05-29 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-29 09:50:00 | 607.53 | 605.19 | 0.00 | ORB-long ORB[600.50,607.23] vol=2.2x ATR=2.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 09:55:00 | 610.84 | 606.25 | 0.00 | T1 1.5R @ 610.84 |
| Target hit | 2024-05-29 15:20:00 | 627.95 | 623.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2024-06-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-06 09:35:00 | 557.70 | 561.83 | 0.00 | ORB-short ORB[560.40,568.33] vol=1.9x ATR=3.79 |
| Stop hit — per-position SL triggered | 2024-06-06 09:40:00 | 561.49 | 561.62 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:25:00 | 624.48 | 620.68 | 0.00 | ORB-long ORB[615.80,622.50] vol=5.3x ATR=2.59 |
| Stop hit — per-position SL triggered | 2024-06-11 10:30:00 | 621.89 | 620.86 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-14 11:10:00 | 660.65 | 663.98 | 0.00 | ORB-short ORB[661.33,670.60] vol=1.8x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-14 11:20:00 | 656.92 | 663.82 | 0.00 | T1 1.5R @ 656.92 |
| Stop hit — per-position SL triggered | 2024-06-14 11:30:00 | 660.65 | 664.04 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-06-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 09:40:00 | 663.70 | 671.42 | 0.00 | ORB-short ORB[669.98,676.65] vol=1.9x ATR=3.20 |
| Stop hit — per-position SL triggered | 2024-06-25 09:45:00 | 666.90 | 670.91 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-26 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-26 10:35:00 | 664.50 | 658.33 | 0.00 | ORB-long ORB[654.38,661.63] vol=2.1x ATR=2.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-26 10:50:00 | 668.62 | 660.11 | 0.00 | T1 1.5R @ 668.62 |
| Target hit | 2024-06-26 12:35:00 | 671.50 | 671.72 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — BUY (started 2024-06-28 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-28 10:45:00 | 663.28 | 662.64 | 0.00 | ORB-long ORB[657.55,662.95] vol=1.6x ATR=2.88 |
| Stop hit — per-position SL triggered | 2024-06-28 11:30:00 | 660.40 | 662.53 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:30:00 | 671.38 | 668.63 | 0.00 | ORB-long ORB[662.48,670.00] vol=1.9x ATR=3.46 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 09:40:00 | 676.57 | 670.38 | 0.00 | T1 1.5R @ 676.57 |
| Target hit | 2024-07-03 09:55:00 | 674.03 | 674.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — SELL (started 2024-07-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 09:40:00 | 683.20 | 684.41 | 0.00 | ORB-short ORB[683.50,691.53] vol=1.5x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-10 09:45:00 | 678.69 | 682.97 | 0.00 | T1 1.5R @ 678.69 |
| Target hit | 2024-07-10 11:00:00 | 671.98 | 671.92 | 0.00 | Trail-exit close>VWAP |

### Cycle 16 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:15:00 | 700.00 | 693.97 | 0.00 | ORB-long ORB[678.50,683.00] vol=13.1x ATR=4.45 |
| Stop hit — per-position SL triggered | 2024-07-12 11:30:00 | 695.55 | 695.70 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 10:45:00 | 683.50 | 684.58 | 0.00 | ORB-short ORB[684.53,689.68] vol=1.5x ATR=1.82 |
| Stop hit — per-position SL triggered | 2024-07-16 10:50:00 | 685.32 | 684.59 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-18 09:30:00 | 667.98 | 672.52 | 0.00 | ORB-short ORB[670.53,679.18] vol=2.0x ATR=2.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-18 09:35:00 | 664.48 | 670.16 | 0.00 | T1 1.5R @ 664.48 |
| Stop hit — per-position SL triggered | 2024-07-18 09:40:00 | 667.98 | 669.33 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2024-07-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:35:00 | 665.30 | 659.94 | 0.00 | ORB-long ORB[650.13,657.35] vol=4.2x ATR=2.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:45:00 | 669.40 | 665.91 | 0.00 | T1 1.5R @ 669.40 |
| Target hit | 2024-07-26 12:05:00 | 671.73 | 672.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 20 — SELL (started 2024-08-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-08 09:30:00 | 644.00 | 648.15 | 0.00 | ORB-short ORB[647.53,655.00] vol=2.6x ATR=2.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 10:00:00 | 640.04 | 644.26 | 0.00 | T1 1.5R @ 640.04 |
| Stop hit — per-position SL triggered | 2024-08-08 10:25:00 | 644.00 | 643.84 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2024-08-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-09 09:35:00 | 640.67 | 642.27 | 0.00 | ORB-short ORB[640.73,647.00] vol=1.8x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-08-09 09:45:00 | 643.50 | 642.27 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-16 10:15:00 | 610.13 | 613.48 | 0.00 | ORB-short ORB[615.30,620.00] vol=2.4x ATR=2.33 |
| Stop hit — per-position SL triggered | 2024-08-16 10:40:00 | 612.46 | 612.79 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-20 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 11:10:00 | 646.30 | 641.06 | 0.00 | ORB-long ORB[634.13,643.00] vol=2.1x ATR=2.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 13:00:00 | 649.43 | 642.89 | 0.00 | T1 1.5R @ 649.43 |
| Stop hit — per-position SL triggered | 2024-08-20 13:15:00 | 646.30 | 643.05 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:35:00 | 651.65 | 650.70 | 0.00 | ORB-long ORB[646.00,651.10] vol=4.0x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-08-21 09:45:00 | 649.59 | 650.79 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:55:00 | 654.03 | 656.63 | 0.00 | ORB-short ORB[657.00,662.93] vol=3.4x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 11:50:00 | 649.65 | 654.91 | 0.00 | T1 1.5R @ 649.65 |
| Stop hit — per-position SL triggered | 2024-08-23 13:45:00 | 654.03 | 653.77 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-09-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 10:20:00 | 693.20 | 686.68 | 0.00 | ORB-long ORB[680.10,689.85] vol=3.3x ATR=2.77 |
| Stop hit — per-position SL triggered | 2024-09-04 10:25:00 | 690.43 | 686.84 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-09-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:40:00 | 754.10 | 742.09 | 0.00 | ORB-long ORB[730.00,738.50] vol=8.8x ATR=4.92 |
| Stop hit — per-position SL triggered | 2024-09-13 09:55:00 | 749.18 | 746.63 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-09-18 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-18 10:35:00 | 775.00 | 767.38 | 0.00 | ORB-long ORB[753.10,764.70] vol=2.0x ATR=3.59 |
| Stop hit — per-position SL triggered | 2024-09-18 10:45:00 | 771.41 | 768.55 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-10-09 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-09 10:45:00 | 847.43 | 842.11 | 0.00 | ORB-long ORB[838.48,846.98] vol=3.8x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-10-09 11:00:00 | 843.74 | 842.44 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-10-10 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-10 09:50:00 | 849.75 | 845.27 | 0.00 | ORB-long ORB[838.98,849.50] vol=1.6x ATR=4.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-10 10:10:00 | 856.10 | 851.22 | 0.00 | T1 1.5R @ 856.10 |
| Target hit | 2024-10-10 12:50:00 | 854.10 | 854.44 | 0.00 | Trail-exit close<VWAP |

### Cycle 31 — BUY (started 2024-10-11 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 10:20:00 | 854.28 | 845.79 | 0.00 | ORB-long ORB[840.40,850.75] vol=1.6x ATR=3.85 |
| Stop hit — per-position SL triggered | 2024-10-11 11:00:00 | 850.43 | 847.28 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:45:00 | 808.70 | 815.69 | 0.00 | ORB-short ORB[818.58,824.48] vol=2.8x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 10:05:00 | 803.76 | 810.95 | 0.00 | T1 1.5R @ 803.76 |
| Stop hit — per-position SL triggered | 2024-10-17 10:10:00 | 808.70 | 810.59 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-21 10:15:00 | 1042.50 | 1026.19 | 0.00 | ORB-long ORB[1012.83,1027.20] vol=2.0x ATR=8.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 10:30:00 | 1054.77 | 1032.40 | 0.00 | T1 1.5R @ 1054.77 |
| Stop hit — per-position SL triggered | 2024-11-21 11:30:00 | 1042.50 | 1038.95 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-11-29 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-29 11:05:00 | 1075.00 | 1064.69 | 0.00 | ORB-long ORB[1051.00,1066.70] vol=6.1x ATR=4.47 |
| Stop hit — per-position SL triggered | 2024-11-29 11:40:00 | 1070.53 | 1066.18 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-02 09:30:00 | 1086.97 | 1080.70 | 0.00 | ORB-long ORB[1071.03,1084.00] vol=1.6x ATR=4.82 |
| Stop hit — per-position SL triggered | 2024-12-02 09:45:00 | 1082.15 | 1089.76 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:30:00 | 1081.28 | 1072.91 | 0.00 | ORB-long ORB[1060.28,1074.35] vol=2.7x ATR=5.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-16 09:40:00 | 1088.85 | 1076.83 | 0.00 | T1 1.5R @ 1088.85 |
| Stop hit — per-position SL triggered | 2024-12-16 09:55:00 | 1081.28 | 1081.29 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-12-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 09:30:00 | 1054.00 | 1062.15 | 0.00 | ORB-short ORB[1055.18,1069.90] vol=2.3x ATR=5.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-17 09:40:00 | 1046.44 | 1056.54 | 0.00 | T1 1.5R @ 1046.44 |
| Stop hit — per-position SL triggered | 2024-12-17 09:55:00 | 1054.00 | 1055.59 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-20 09:30:00 | 1069.50 | 1063.24 | 0.00 | ORB-long ORB[1053.95,1068.63] vol=1.9x ATR=4.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:55:00 | 1076.60 | 1068.64 | 0.00 | T1 1.5R @ 1076.60 |
| Target hit | 2024-12-20 13:05:00 | 1071.65 | 1075.18 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — SELL (started 2024-12-27 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-27 09:50:00 | 1124.18 | 1128.27 | 0.00 | ORB-short ORB[1127.85,1144.18] vol=1.6x ATR=5.18 |
| Stop hit — per-position SL triggered | 2024-12-27 09:55:00 | 1129.36 | 1128.08 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-12-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:30:00 | 1153.95 | 1142.24 | 0.00 | ORB-long ORB[1128.33,1142.20] vol=6.6x ATR=5.45 |
| Stop hit — per-position SL triggered | 2024-12-30 09:35:00 | 1148.50 | 1143.79 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-01-03 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 10:05:00 | 1058.65 | 1050.52 | 0.00 | ORB-long ORB[1038.78,1053.53] vol=1.6x ATR=4.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 12:35:00 | 1065.94 | 1057.87 | 0.00 | T1 1.5R @ 1065.94 |
| Target hit | 2025-01-03 15:20:00 | 1059.97 | 1059.74 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 42 — BUY (started 2025-01-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 10:10:00 | 1010.28 | 1001.94 | 0.00 | ORB-long ORB[988.10,1002.85] vol=3.6x ATR=4.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 10:15:00 | 1017.71 | 1005.10 | 0.00 | T1 1.5R @ 1017.71 |
| Stop hit — per-position SL triggered | 2025-01-23 10:40:00 | 1010.28 | 1006.00 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-12 11:15:00 | 1100.83 | 1117.73 | 0.00 | ORB-short ORB[1113.20,1129.70] vol=1.6x ATR=6.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-12 11:25:00 | 1091.48 | 1115.56 | 0.00 | T1 1.5R @ 1091.48 |
| Stop hit — per-position SL triggered | 2025-03-12 12:15:00 | 1100.83 | 1109.69 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-03-18 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-18 09:40:00 | 1092.65 | 1102.37 | 0.00 | ORB-short ORB[1099.50,1111.53] vol=3.8x ATR=6.00 |
| Stop hit — per-position SL triggered | 2025-03-18 09:45:00 | 1098.65 | 1101.31 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-03-20 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-20 09:50:00 | 1163.40 | 1179.81 | 0.00 | ORB-short ORB[1177.38,1190.00] vol=1.7x ATR=6.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-20 10:05:00 | 1153.94 | 1175.26 | 0.00 | T1 1.5R @ 1153.94 |
| Stop hit — per-position SL triggered | 2025-03-20 10:20:00 | 1163.40 | 1170.01 | 0.00 | SL hit |

### Cycle 46 — BUY (started 2025-05-08 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-08 10:00:00 | 1177.30 | 1161.70 | 0.00 | ORB-long ORB[1146.20,1163.60] vol=4.7x ATR=6.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-08 10:30:00 | 1187.41 | 1167.12 | 0.00 | T1 1.5R @ 1187.41 |
| Stop hit — per-position SL triggered | 2025-05-08 10:45:00 | 1177.30 | 1168.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-15 09:55:00 | 607.83 | 2024-05-15 10:20:00 | 604.47 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2024-05-15 09:55:00 | 607.83 | 2024-05-15 10:35:00 | 607.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-05-16 10:50:00 | 609.15 | 2024-05-16 10:55:00 | 611.28 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-05-17 10:20:00 | 626.50 | 2024-05-17 10:30:00 | 623.66 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-05-22 09:35:00 | 610.20 | 2024-05-22 09:45:00 | 612.75 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-05-23 09:35:00 | 608.67 | 2024-05-23 09:45:00 | 606.57 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-05-24 10:55:00 | 599.50 | 2024-05-24 11:05:00 | 601.48 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-05-29 09:50:00 | 607.53 | 2024-05-29 09:55:00 | 610.84 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2024-05-29 09:50:00 | 607.53 | 2024-05-29 15:20:00 | 627.95 | TARGET_HIT | 0.50 | 3.36% |
| SELL | retest1 | 2024-06-06 09:35:00 | 557.70 | 2024-06-06 09:40:00 | 561.49 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest1 | 2024-06-11 10:25:00 | 624.48 | 2024-06-11 10:30:00 | 621.89 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-06-14 11:10:00 | 660.65 | 2024-06-14 11:20:00 | 656.92 | PARTIAL | 0.50 | 0.57% |
| SELL | retest1 | 2024-06-14 11:10:00 | 660.65 | 2024-06-14 11:30:00 | 660.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-06-25 09:40:00 | 663.70 | 2024-06-25 09:45:00 | 666.90 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-06-26 10:35:00 | 664.50 | 2024-06-26 10:50:00 | 668.62 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-06-26 10:35:00 | 664.50 | 2024-06-26 12:35:00 | 671.50 | TARGET_HIT | 0.50 | 1.05% |
| BUY | retest1 | 2024-06-28 10:45:00 | 663.28 | 2024-06-28 11:30:00 | 660.40 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-07-03 09:30:00 | 671.38 | 2024-07-03 09:40:00 | 676.57 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2024-07-03 09:30:00 | 671.38 | 2024-07-03 09:55:00 | 674.03 | TARGET_HIT | 0.50 | 0.39% |
| SELL | retest1 | 2024-07-10 09:40:00 | 683.20 | 2024-07-10 09:45:00 | 678.69 | PARTIAL | 0.50 | 0.66% |
| SELL | retest1 | 2024-07-10 09:40:00 | 683.20 | 2024-07-10 11:00:00 | 671.98 | TARGET_HIT | 0.50 | 1.64% |
| BUY | retest1 | 2024-07-12 10:15:00 | 700.00 | 2024-07-12 11:30:00 | 695.55 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest1 | 2024-07-16 10:45:00 | 683.50 | 2024-07-16 10:50:00 | 685.32 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-07-18 09:30:00 | 667.98 | 2024-07-18 09:35:00 | 664.48 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-07-18 09:30:00 | 667.98 | 2024-07-18 09:40:00 | 667.98 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-07-26 09:35:00 | 665.30 | 2024-07-26 09:45:00 | 669.40 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-07-26 09:35:00 | 665.30 | 2024-07-26 12:05:00 | 671.73 | TARGET_HIT | 0.50 | 0.97% |
| SELL | retest1 | 2024-08-08 09:30:00 | 644.00 | 2024-08-08 10:00:00 | 640.04 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-08-08 09:30:00 | 644.00 | 2024-08-08 10:25:00 | 644.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-09 09:35:00 | 640.67 | 2024-08-09 09:45:00 | 643.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-08-16 10:15:00 | 610.13 | 2024-08-16 10:40:00 | 612.46 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-08-20 11:10:00 | 646.30 | 2024-08-20 13:00:00 | 649.43 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-08-20 11:10:00 | 646.30 | 2024-08-20 13:15:00 | 646.30 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-21 09:35:00 | 651.65 | 2024-08-21 09:45:00 | 649.59 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-08-23 09:55:00 | 654.03 | 2024-08-23 11:50:00 | 649.65 | PARTIAL | 0.50 | 0.67% |
| SELL | retest1 | 2024-08-23 09:55:00 | 654.03 | 2024-08-23 13:45:00 | 654.03 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-04 10:20:00 | 693.20 | 2024-09-04 10:25:00 | 690.43 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2024-09-13 09:40:00 | 754.10 | 2024-09-13 09:55:00 | 749.18 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest1 | 2024-09-18 10:35:00 | 775.00 | 2024-09-18 10:45:00 | 771.41 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-10-09 10:45:00 | 847.43 | 2024-10-09 11:00:00 | 843.74 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-10-10 09:50:00 | 849.75 | 2024-10-10 10:10:00 | 856.10 | PARTIAL | 0.50 | 0.75% |
| BUY | retest1 | 2024-10-10 09:50:00 | 849.75 | 2024-10-10 12:50:00 | 854.10 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2024-10-11 10:20:00 | 854.28 | 2024-10-11 11:00:00 | 850.43 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2024-10-17 09:45:00 | 808.70 | 2024-10-17 10:05:00 | 803.76 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-10-17 09:45:00 | 808.70 | 2024-10-17 10:10:00 | 808.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-21 10:15:00 | 1042.50 | 2024-11-21 10:30:00 | 1054.77 | PARTIAL | 0.50 | 1.18% |
| BUY | retest1 | 2024-11-21 10:15:00 | 1042.50 | 2024-11-21 11:30:00 | 1042.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-29 11:05:00 | 1075.00 | 2024-11-29 11:40:00 | 1070.53 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-12-02 09:30:00 | 1086.97 | 2024-12-02 09:45:00 | 1082.15 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-12-16 09:30:00 | 1081.28 | 2024-12-16 09:40:00 | 1088.85 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-12-16 09:30:00 | 1081.28 | 2024-12-16 09:55:00 | 1081.28 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-12-17 09:30:00 | 1054.00 | 2024-12-17 09:40:00 | 1046.44 | PARTIAL | 0.50 | 0.72% |
| SELL | retest1 | 2024-12-17 09:30:00 | 1054.00 | 2024-12-17 09:55:00 | 1054.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-20 09:30:00 | 1069.50 | 2024-12-20 09:55:00 | 1076.60 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-12-20 09:30:00 | 1069.50 | 2024-12-20 13:05:00 | 1071.65 | TARGET_HIT | 0.50 | 0.20% |
| SELL | retest1 | 2024-12-27 09:50:00 | 1124.18 | 2024-12-27 09:55:00 | 1129.36 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2024-12-30 09:30:00 | 1153.95 | 2024-12-30 09:35:00 | 1148.50 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-01-03 10:05:00 | 1058.65 | 2025-01-03 12:35:00 | 1065.94 | PARTIAL | 0.50 | 0.69% |
| BUY | retest1 | 2025-01-03 10:05:00 | 1058.65 | 2025-01-03 15:20:00 | 1059.97 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2025-01-23 10:10:00 | 1010.28 | 2025-01-23 10:15:00 | 1017.71 | PARTIAL | 0.50 | 0.74% |
| BUY | retest1 | 2025-01-23 10:10:00 | 1010.28 | 2025-01-23 10:40:00 | 1010.28 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-12 11:15:00 | 1100.83 | 2025-03-12 11:25:00 | 1091.48 | PARTIAL | 0.50 | 0.85% |
| SELL | retest1 | 2025-03-12 11:15:00 | 1100.83 | 2025-03-12 12:15:00 | 1100.83 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-18 09:40:00 | 1092.65 | 2025-03-18 09:45:00 | 1098.65 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2025-03-20 09:50:00 | 1163.40 | 2025-03-20 10:05:00 | 1153.94 | PARTIAL | 0.50 | 0.81% |
| SELL | retest1 | 2025-03-20 09:50:00 | 1163.40 | 2025-03-20 10:20:00 | 1163.40 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-05-08 10:00:00 | 1177.30 | 2025-05-08 10:30:00 | 1187.41 | PARTIAL | 0.50 | 0.86% |
| BUY | retest1 | 2025-05-08 10:00:00 | 1177.30 | 2025-05-08 10:45:00 | 1177.30 | STOP_HIT | 0.50 | 0.00% |
