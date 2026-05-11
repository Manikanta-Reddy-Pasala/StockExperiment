# Bajaj Finance Ltd. (BAJFINANCE)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 954.50
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
| ENTRY1 | 91 |
| ENTRY2 | 0 |
| PARTIAL | 36 |
| TARGET_HIT | 20 |
| STOP_HIT | 71 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 127 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 71
- **Target hits / Stop hits / Partials:** 20 / 71 / 36
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 23.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 32 | 51.6% | 12 | 30 | 20 | 0.25% | 15.3% |
| BUY @ 2nd Alert (retest1) | 62 | 32 | 51.6% | 12 | 30 | 20 | 0.25% | 15.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 65 | 24 | 36.9% | 8 | 41 | 16 | 0.12% | 7.9% |
| SELL @ 2nd Alert (retest1) | 65 | 24 | 36.9% | 8 | 41 | 16 | 0.12% | 7.9% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 127 | 56 | 44.1% | 20 | 71 | 36 | 0.18% | 23.2% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-14 10:55:00 | 666.32 | 669.87 | 0.00 | ORB-short ORB[669.25,674.19] vol=1.8x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-05-14 11:00:00 | 667.90 | 669.80 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 10:20:00 | 665.55 | 668.40 | 0.00 | ORB-short ORB[668.68,671.70] vol=1.8x ATR=1.31 |
| Stop hit — per-position SL triggered | 2024-05-16 10:30:00 | 666.86 | 668.21 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2024-05-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-17 10:55:00 | 676.80 | 672.94 | 0.00 | ORB-long ORB[671.10,674.92] vol=1.7x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-05-17 11:20:00 | 675.26 | 673.51 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-05-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-21 10:00:00 | 676.41 | 673.12 | 0.00 | ORB-long ORB[670.50,674.99] vol=2.7x ATR=1.57 |
| Stop hit — per-position SL triggered | 2024-05-21 10:55:00 | 674.84 | 674.56 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2024-05-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 09:40:00 | 676.69 | 675.46 | 0.00 | ORB-long ORB[673.01,676.55] vol=1.5x ATR=1.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 10:00:00 | 678.66 | 676.00 | 0.00 | T1 1.5R @ 678.66 |
| Stop hit — per-position SL triggered | 2024-05-23 10:05:00 | 676.69 | 676.03 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-05-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-27 10:55:00 | 688.94 | 685.40 | 0.00 | ORB-long ORB[683.80,687.50] vol=1.6x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-05-27 11:05:00 | 687.43 | 685.75 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-05 11:10:00 | 668.28 | 660.61 | 0.00 | ORB-long ORB[653.55,661.60] vol=2.4x ATR=3.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 11:50:00 | 673.65 | 663.76 | 0.00 | T1 1.5R @ 673.65 |
| Target hit | 2024-06-05 15:20:00 | 682.53 | 671.94 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-06-06 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-06 09:40:00 | 693.98 | 688.25 | 0.00 | ORB-long ORB[682.23,687.70] vol=2.2x ATR=2.98 |
| Stop hit — per-position SL triggered | 2024-06-06 09:45:00 | 691.00 | 688.61 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-07 09:35:00 | 704.11 | 698.61 | 0.00 | ORB-long ORB[694.10,701.20] vol=1.7x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-07 10:00:00 | 707.76 | 701.44 | 0.00 | T1 1.5R @ 707.76 |
| Target hit | 2024-06-07 14:20:00 | 715.00 | 715.39 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2024-06-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 10:05:00 | 724.50 | 718.27 | 0.00 | ORB-long ORB[711.50,714.80] vol=1.7x ATR=2.04 |
| Stop hit — per-position SL triggered | 2024-06-12 10:20:00 | 722.46 | 719.40 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-06-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-14 10:55:00 | 736.50 | 731.14 | 0.00 | ORB-long ORB[727.00,733.53] vol=3.3x ATR=1.70 |
| Stop hit — per-position SL triggered | 2024-06-14 11:05:00 | 734.80 | 731.47 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-06-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-18 10:10:00 | 740.50 | 737.67 | 0.00 | ORB-long ORB[733.10,740.00] vol=2.1x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-06-18 11:00:00 | 738.59 | 738.24 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-06-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-19 10:35:00 | 720.05 | 727.00 | 0.00 | ORB-short ORB[728.03,734.90] vol=1.9x ATR=2.06 |
| Stop hit — per-position SL triggered | 2024-06-19 10:45:00 | 722.11 | 725.99 | 0.00 | SL hit |

### Cycle 14 — SELL (started 2024-06-25 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-25 11:00:00 | 701.51 | 706.99 | 0.00 | ORB-short ORB[708.50,714.00] vol=2.1x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-06-25 12:25:00 | 703.04 | 704.48 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 11:15:00 | 714.70 | 711.44 | 0.00 | ORB-long ORB[707.50,713.50] vol=4.9x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-07-01 11:20:00 | 713.30 | 711.50 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2024-07-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:40:00 | 715.21 | 720.13 | 0.00 | ORB-short ORB[719.00,727.60] vol=1.5x ATR=2.37 |
| Stop hit — per-position SL triggered | 2024-07-02 11:05:00 | 717.58 | 717.05 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2024-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 09:30:00 | 722.93 | 727.32 | 0.00 | ORB-short ORB[723.48,732.50] vol=1.6x ATR=2.14 |
| Stop hit — per-position SL triggered | 2024-07-04 09:45:00 | 725.07 | 726.72 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2024-07-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 09:40:00 | 705.28 | 709.23 | 0.00 | ORB-short ORB[709.00,714.77] vol=2.3x ATR=1.71 |
| Stop hit — per-position SL triggered | 2024-07-08 09:50:00 | 706.99 | 708.67 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-07-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-09 09:40:00 | 709.00 | 710.05 | 0.00 | ORB-short ORB[709.48,714.24] vol=5.2x ATR=1.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 09:45:00 | 706.92 | 709.64 | 0.00 | T1 1.5R @ 706.92 |
| Stop hit — per-position SL triggered | 2024-07-09 11:05:00 | 709.00 | 708.38 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-11 10:15:00 | 701.48 | 704.08 | 0.00 | ORB-short ORB[703.00,708.79] vol=1.6x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-11 10:30:00 | 699.07 | 703.08 | 0.00 | T1 1.5R @ 699.07 |
| Target hit | 2024-07-11 15:20:00 | 695.50 | 697.20 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 21 — SELL (started 2024-07-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-12 09:50:00 | 693.70 | 695.87 | 0.00 | ORB-short ORB[695.68,699.87] vol=2.4x ATR=1.54 |
| Stop hit — per-position SL triggered | 2024-07-12 10:50:00 | 695.24 | 694.74 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2024-07-16 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-16 09:50:00 | 701.50 | 705.33 | 0.00 | ORB-short ORB[705.60,708.40] vol=1.5x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-07-16 10:05:00 | 703.03 | 704.67 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2024-07-19 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-19 09:50:00 | 701.17 | 705.39 | 0.00 | ORB-short ORB[705.60,711.00] vol=4.0x ATR=2.26 |
| Stop hit — per-position SL triggered | 2024-07-19 10:20:00 | 703.43 | 704.69 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2024-07-23 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 09:55:00 | 685.51 | 688.75 | 0.00 | ORB-short ORB[689.54,694.49] vol=1.8x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 10:15:00 | 682.59 | 687.81 | 0.00 | T1 1.5R @ 682.59 |
| Stop hit — per-position SL triggered | 2024-07-23 11:00:00 | 685.51 | 686.55 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-07-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-24 09:45:00 | 655.47 | 660.32 | 0.00 | ORB-short ORB[655.83,665.47] vol=1.6x ATR=3.18 |
| Stop hit — per-position SL triggered | 2024-07-24 11:35:00 | 658.65 | 658.08 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-07-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:35:00 | 673.68 | 670.63 | 0.00 | ORB-long ORB[666.71,671.50] vol=2.3x ATR=1.44 |
| Stop hit — per-position SL triggered | 2024-07-26 09:45:00 | 672.24 | 670.98 | 0.00 | SL hit |

### Cycle 27 — BUY (started 2024-07-30 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:05:00 | 684.21 | 682.55 | 0.00 | ORB-long ORB[678.30,683.00] vol=1.5x ATR=1.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-30 13:05:00 | 686.59 | 683.49 | 0.00 | T1 1.5R @ 686.59 |
| Stop hit — per-position SL triggered | 2024-07-30 13:15:00 | 684.21 | 683.67 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2024-08-05 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-05 11:10:00 | 656.27 | 659.13 | 0.00 | ORB-short ORB[656.51,665.00] vol=1.7x ATR=1.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 11:20:00 | 653.71 | 658.55 | 0.00 | T1 1.5R @ 653.71 |
| Stop hit — per-position SL triggered | 2024-08-05 11:25:00 | 656.27 | 658.47 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2024-08-06 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 10:45:00 | 659.70 | 663.52 | 0.00 | ORB-short ORB[662.60,669.70] vol=2.1x ATR=1.79 |
| Stop hit — per-position SL triggered | 2024-08-06 10:55:00 | 661.49 | 663.36 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-08-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-12 09:50:00 | 657.25 | 659.08 | 0.00 | ORB-short ORB[658.29,663.60] vol=1.5x ATR=1.13 |
| Stop hit — per-position SL triggered | 2024-08-12 10:15:00 | 658.38 | 658.60 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2024-08-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-19 09:40:00 | 664.96 | 662.77 | 0.00 | ORB-long ORB[659.54,663.91] vol=1.6x ATR=1.40 |
| Stop hit — per-position SL triggered | 2024-08-19 09:45:00 | 663.56 | 662.84 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2024-08-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:45:00 | 666.70 | 664.87 | 0.00 | ORB-long ORB[661.64,665.00] vol=2.6x ATR=1.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:25:00 | 668.53 | 666.34 | 0.00 | T1 1.5R @ 668.53 |
| Target hit | 2024-08-20 15:20:00 | 671.63 | 671.43 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 33 — BUY (started 2024-08-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:30:00 | 681.79 | 678.52 | 0.00 | ORB-long ORB[675.61,677.90] vol=3.5x ATR=1.22 |
| Stop hit — per-position SL triggered | 2024-08-26 09:35:00 | 680.57 | 679.02 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-08-29 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:45:00 | 696.13 | 691.97 | 0.00 | ORB-long ORB[688.16,690.72] vol=1.9x ATR=1.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-29 09:50:00 | 697.82 | 694.20 | 0.00 | T1 1.5R @ 697.82 |
| Target hit | 2024-08-29 13:45:00 | 706.56 | 707.20 | 0.00 | Trail-exit close<VWAP |

### Cycle 35 — BUY (started 2024-09-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 10:00:00 | 732.31 | 726.88 | 0.00 | ORB-long ORB[723.71,730.12] vol=1.8x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-02 10:15:00 | 735.70 | 729.08 | 0.00 | T1 1.5R @ 735.70 |
| Target hit | 2024-09-02 15:20:00 | 743.29 | 737.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 36 — SELL (started 2024-09-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-05 11:00:00 | 721.90 | 725.92 | 0.00 | ORB-short ORB[726.13,732.67] vol=2.5x ATR=1.51 |
| Stop hit — per-position SL triggered | 2024-09-05 11:10:00 | 723.41 | 725.70 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2024-09-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-10 09:45:00 | 726.60 | 729.63 | 0.00 | ORB-short ORB[728.52,737.00] vol=1.7x ATR=1.68 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-10 09:55:00 | 724.08 | 728.49 | 0.00 | T1 1.5R @ 724.08 |
| Stop hit — per-position SL triggered | 2024-09-10 11:00:00 | 726.60 | 726.50 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-11 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 10:40:00 | 730.43 | 726.53 | 0.00 | ORB-long ORB[720.63,725.73] vol=1.7x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 11:25:00 | 733.12 | 728.47 | 0.00 | T1 1.5R @ 733.12 |
| Target hit | 2024-09-11 14:05:00 | 734.10 | 734.56 | 0.00 | Trail-exit close<VWAP |

### Cycle 39 — BUY (started 2024-09-13 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:05:00 | 750.30 | 741.85 | 0.00 | ORB-long ORB[735.20,743.90] vol=1.8x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-13 10:15:00 | 753.94 | 745.26 | 0.00 | T1 1.5R @ 753.94 |
| Target hit | 2024-09-13 15:20:00 | 760.00 | 755.29 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 40 — SELL (started 2024-09-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-16 10:00:00 | 755.27 | 760.89 | 0.00 | ORB-short ORB[758.00,768.00] vol=4.1x ATR=3.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-16 10:05:00 | 749.93 | 759.96 | 0.00 | T1 1.5R @ 749.93 |
| Target hit | 2024-09-16 15:20:00 | 734.10 | 745.23 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 41 — BUY (started 2024-09-17 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:10:00 | 741.49 | 736.09 | 0.00 | ORB-long ORB[728.62,737.50] vol=1.9x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-17 10:15:00 | 745.28 | 737.24 | 0.00 | T1 1.5R @ 745.28 |
| Stop hit — per-position SL triggered | 2024-09-17 10:30:00 | 741.49 | 738.45 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2024-09-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:30:00 | 749.97 | 754.20 | 0.00 | ORB-short ORB[751.14,760.35] vol=2.5x ATR=1.91 |
| Stop hit — per-position SL triggered | 2024-09-24 09:45:00 | 751.88 | 752.93 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-09-25 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 09:40:00 | 756.57 | 754.57 | 0.00 | ORB-long ORB[749.81,755.00] vol=2.4x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-25 10:20:00 | 759.21 | 756.23 | 0.00 | T1 1.5R @ 759.21 |
| Target hit | 2024-09-25 11:50:00 | 757.32 | 758.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 44 — BUY (started 2024-10-01 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-01 10:40:00 | 779.25 | 775.39 | 0.00 | ORB-long ORB[765.10,771.00] vol=1.7x ATR=2.45 |
| Stop hit — per-position SL triggered | 2024-10-01 11:20:00 | 776.80 | 775.90 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-10-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-03 11:00:00 | 758.69 | 763.74 | 0.00 | ORB-short ORB[761.00,770.00] vol=2.1x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 11:25:00 | 755.69 | 762.96 | 0.00 | T1 1.5R @ 755.69 |
| Target hit | 2024-10-03 15:20:00 | 743.11 | 752.47 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — SELL (started 2024-10-07 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:55:00 | 720.91 | 724.70 | 0.00 | ORB-short ORB[721.14,729.23] vol=2.9x ATR=2.60 |
| Stop hit — per-position SL triggered | 2024-10-07 11:05:00 | 723.51 | 724.43 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2024-10-14 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-14 09:35:00 | 721.83 | 724.72 | 0.00 | ORB-short ORB[722.20,731.39] vol=1.6x ATR=1.53 |
| Stop hit — per-position SL triggered | 2024-10-14 09:40:00 | 723.36 | 724.58 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2024-10-15 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-15 09:50:00 | 718.61 | 722.41 | 0.00 | ORB-short ORB[722.26,725.62] vol=1.8x ATR=1.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-15 09:55:00 | 716.22 | 721.57 | 0.00 | T1 1.5R @ 716.22 |
| Target hit | 2024-10-15 15:20:00 | 702.25 | 706.13 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2024-10-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 11:05:00 | 691.20 | 693.89 | 0.00 | ORB-short ORB[692.50,699.88] vol=2.4x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-17 13:55:00 | 688.32 | 692.70 | 0.00 | T1 1.5R @ 688.32 |
| Stop hit — per-position SL triggered | 2024-10-17 15:10:00 | 691.20 | 691.73 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2024-10-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-21 09:35:00 | 685.00 | 689.72 | 0.00 | ORB-short ORB[688.57,695.08] vol=1.7x ATR=2.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 09:45:00 | 682.01 | 688.49 | 0.00 | T1 1.5R @ 682.01 |
| Stop hit — per-position SL triggered | 2024-10-21 10:05:00 | 685.00 | 686.45 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2024-10-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:35:00 | 698.44 | 700.49 | 0.00 | ORB-short ORB[700.11,704.99] vol=2.0x ATR=1.83 |
| Stop hit — per-position SL triggered | 2024-10-25 09:45:00 | 700.27 | 699.94 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2024-10-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-29 09:40:00 | 683.62 | 687.25 | 0.00 | ORB-short ORB[684.86,691.70] vol=1.8x ATR=2.44 |
| Stop hit — per-position SL triggered | 2024-10-29 09:45:00 | 686.06 | 686.95 | 0.00 | SL hit |

### Cycle 53 — SELL (started 2024-11-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 10:35:00 | 679.45 | 681.68 | 0.00 | ORB-short ORB[680.55,685.80] vol=1.9x ATR=1.97 |
| Stop hit — per-position SL triggered | 2024-11-05 10:50:00 | 681.42 | 681.41 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-07 11:15:00 | 688.50 | 692.75 | 0.00 | ORB-short ORB[694.70,703.90] vol=1.5x ATR=1.62 |
| Stop hit — per-position SL triggered | 2024-11-07 11:40:00 | 690.12 | 692.23 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-11-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 10:00:00 | 675.84 | 678.93 | 0.00 | ORB-short ORB[678.33,683.69] vol=1.5x ATR=2.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 10:40:00 | 672.68 | 677.57 | 0.00 | T1 1.5R @ 672.68 |
| Target hit | 2024-11-12 15:20:00 | 665.00 | 668.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2024-11-14 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 11:10:00 | 656.09 | 658.94 | 0.00 | ORB-short ORB[656.55,663.95] vol=3.2x ATR=1.80 |
| Stop hit — per-position SL triggered | 2024-11-14 11:30:00 | 657.89 | 658.73 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-11-18 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 10:50:00 | 657.36 | 660.22 | 0.00 | ORB-short ORB[658.00,665.70] vol=1.5x ATR=1.84 |
| Stop hit — per-position SL triggered | 2024-11-18 12:00:00 | 659.20 | 659.63 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-11-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-21 10:40:00 | 648.84 | 652.68 | 0.00 | ORB-short ORB[652.20,661.40] vol=1.5x ATR=2.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 12:30:00 | 645.57 | 650.46 | 0.00 | T1 1.5R @ 645.57 |
| Stop hit — per-position SL triggered | 2024-11-21 13:20:00 | 648.84 | 650.06 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-11-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-27 09:55:00 | 670.87 | 665.72 | 0.00 | ORB-long ORB[661.23,665.68] vol=1.5x ATR=1.78 |
| Stop hit — per-position SL triggered | 2024-11-27 10:00:00 | 669.09 | 665.91 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-11-28 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-28 10:40:00 | 664.63 | 668.29 | 0.00 | ORB-short ORB[665.62,672.63] vol=2.0x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-28 11:15:00 | 662.08 | 667.25 | 0.00 | T1 1.5R @ 662.08 |
| Stop hit — per-position SL triggered | 2024-11-28 11:45:00 | 664.63 | 666.52 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-12-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:45:00 | 673.80 | 671.13 | 0.00 | ORB-long ORB[667.55,671.20] vol=1.6x ATR=1.47 |
| Stop hit — per-position SL triggered | 2024-12-04 09:55:00 | 672.33 | 671.28 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 670.17 | 674.34 | 0.00 | ORB-short ORB[675.05,678.90] vol=2.1x ATR=1.63 |
| Stop hit — per-position SL triggered | 2024-12-05 11:25:00 | 671.80 | 673.86 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2024-12-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-10 09:35:00 | 694.46 | 692.00 | 0.00 | ORB-long ORB[687.60,692.74] vol=2.1x ATR=1.59 |
| Stop hit — per-position SL triggered | 2024-12-10 10:05:00 | 692.87 | 693.13 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-12-11 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-11 10:00:00 | 691.34 | 692.69 | 0.00 | ORB-short ORB[691.37,697.00] vol=2.4x ATR=1.58 |
| Stop hit — per-position SL triggered | 2024-12-11 10:10:00 | 692.92 | 692.83 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2024-12-16 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 09:40:00 | 722.30 | 719.28 | 0.00 | ORB-long ORB[715.61,721.03] vol=1.5x ATR=2.19 |
| Stop hit — per-position SL triggered | 2024-12-16 12:10:00 | 720.11 | 721.03 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-12-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-17 10:20:00 | 712.31 | 717.99 | 0.00 | ORB-short ORB[717.72,723.03] vol=1.7x ATR=1.72 |
| Stop hit — per-position SL triggered | 2024-12-17 10:45:00 | 714.03 | 716.64 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2024-12-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 10:00:00 | 694.20 | 690.77 | 0.00 | ORB-long ORB[683.04,693.38] vol=2.4x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:30:00 | 697.15 | 692.02 | 0.00 | T1 1.5R @ 697.15 |
| Stop hit — per-position SL triggered | 2024-12-27 11:05:00 | 694.20 | 692.68 | 0.00 | SL hit |

### Cycle 68 — BUY (started 2025-01-01 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-01 10:50:00 | 690.60 | 685.19 | 0.00 | ORB-long ORB[682.20,685.00] vol=1.6x ATR=1.72 |
| Stop hit — per-position SL triggered | 2025-01-01 11:35:00 | 688.88 | 686.62 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-03 11:15:00 | 744.00 | 740.87 | 0.00 | ORB-long ORB[736.44,743.30] vol=2.1x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-03 11:35:00 | 746.45 | 741.60 | 0.00 | T1 1.5R @ 746.45 |
| Stop hit — per-position SL triggered | 2025-01-03 12:45:00 | 744.00 | 742.95 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2025-01-09 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-09 10:30:00 | 728.15 | 729.18 | 0.00 | ORB-short ORB[729.31,735.54] vol=1.6x ATR=1.75 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 11:25:00 | 725.53 | 727.94 | 0.00 | T1 1.5R @ 725.53 |
| Target hit | 2025-01-09 12:15:00 | 727.34 | 727.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 71 — BUY (started 2025-01-10 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-10 11:05:00 | 733.60 | 728.86 | 0.00 | ORB-long ORB[726.63,731.79] vol=1.8x ATR=2.12 |
| Stop hit — per-position SL triggered | 2025-01-10 11:40:00 | 731.48 | 730.15 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-01-13 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:40:00 | 718.41 | 719.42 | 0.00 | ORB-short ORB[718.83,725.10] vol=4.4x ATR=2.27 |
| Stop hit — per-position SL triggered | 2025-01-13 10:00:00 | 720.68 | 719.22 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-01-15 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-15 11:05:00 | 718.15 | 718.60 | 0.00 | ORB-short ORB[724.41,735.00] vol=2.0x ATR=2.49 |
| Target hit | 2025-01-15 15:20:00 | 717.90 | 717.83 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 74 — BUY (started 2025-01-20 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-20 10:25:00 | 729.16 | 722.88 | 0.00 | ORB-long ORB[718.21,723.20] vol=1.9x ATR=1.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 10:35:00 | 731.82 | 724.49 | 0.00 | T1 1.5R @ 731.82 |
| Target hit | 2025-01-20 15:20:00 | 743.37 | 738.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 75 — BUY (started 2025-01-23 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-23 09:40:00 | 740.00 | 738.99 | 0.00 | ORB-long ORB[732.50,739.50] vol=1.9x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-23 09:45:00 | 743.46 | 739.40 | 0.00 | T1 1.5R @ 743.46 |
| Target hit | 2025-01-23 09:55:00 | 740.45 | 740.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 76 — SELL (started 2025-02-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-07 11:00:00 | 844.38 | 846.37 | 0.00 | ORB-short ORB[846.00,858.50] vol=2.8x ATR=3.51 |
| Stop hit — per-position SL triggered | 2025-02-07 11:10:00 | 847.89 | 846.55 | 0.00 | SL hit |

### Cycle 77 — SELL (started 2025-02-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-12 09:50:00 | 805.96 | 812.09 | 0.00 | ORB-short ORB[810.78,821.47] vol=1.6x ATR=2.60 |
| Stop hit — per-position SL triggered | 2025-02-12 10:30:00 | 808.56 | 809.61 | 0.00 | SL hit |

### Cycle 78 — BUY (started 2025-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-13 09:30:00 | 831.62 | 826.06 | 0.00 | ORB-long ORB[819.71,827.53] vol=1.7x ATR=3.01 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-13 09:50:00 | 836.13 | 830.61 | 0.00 | T1 1.5R @ 836.13 |
| Target hit | 2025-02-13 13:30:00 | 840.72 | 841.11 | 0.00 | Trail-exit close<VWAP |

### Cycle 79 — BUY (started 2025-02-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-18 09:45:00 | 845.74 | 841.44 | 0.00 | ORB-long ORB[836.76,845.00] vol=1.6x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-02-18 10:15:00 | 843.51 | 843.12 | 0.00 | SL hit |

### Cycle 80 — SELL (started 2025-02-20 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-20 09:45:00 | 835.37 | 840.63 | 0.00 | ORB-short ORB[835.59,845.56] vol=2.0x ATR=2.67 |
| Stop hit — per-position SL triggered | 2025-02-20 09:50:00 | 838.04 | 840.44 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2025-02-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-25 09:30:00 | 843.03 | 838.81 | 0.00 | ORB-long ORB[833.81,840.78] vol=3.2x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 09:35:00 | 845.92 | 840.32 | 0.00 | T1 1.5R @ 845.92 |
| Target hit | 2025-02-25 13:15:00 | 846.48 | 846.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 82 — BUY (started 2025-03-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-13 09:35:00 | 855.80 | 851.81 | 0.00 | ORB-long ORB[848.79,854.90] vol=1.6x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-03-13 09:40:00 | 853.39 | 851.96 | 0.00 | SL hit |

### Cycle 83 — BUY (started 2025-03-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-19 09:45:00 | 881.00 | 876.50 | 0.00 | ORB-long ORB[868.26,879.38] vol=2.4x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 09:55:00 | 885.12 | 877.93 | 0.00 | T1 1.5R @ 885.12 |
| Stop hit — per-position SL triggered | 2025-03-19 11:10:00 | 881.00 | 880.85 | 0.00 | SL hit |

### Cycle 84 — BUY (started 2025-03-25 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-25 09:30:00 | 917.00 | 912.40 | 0.00 | ORB-long ORB[907.97,913.89] vol=2.1x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-25 09:35:00 | 920.81 | 913.98 | 0.00 | T1 1.5R @ 920.81 |
| Stop hit — per-position SL triggered | 2025-03-25 10:05:00 | 917.00 | 917.21 | 0.00 | SL hit |

### Cycle 85 — BUY (started 2025-03-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-27 10:20:00 | 890.95 | 888.09 | 0.00 | ORB-long ORB[882.23,889.35] vol=1.7x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 10:30:00 | 894.35 | 888.43 | 0.00 | T1 1.5R @ 894.35 |
| Stop hit — per-position SL triggered | 2025-03-27 10:45:00 | 890.95 | 888.66 | 0.00 | SL hit |

### Cycle 86 — BUY (started 2025-04-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-04 09:35:00 | 871.27 | 865.99 | 0.00 | ORB-long ORB[860.28,870.22] vol=1.8x ATR=2.90 |
| Stop hit — per-position SL triggered | 2025-04-04 09:40:00 | 868.37 | 866.04 | 0.00 | SL hit |

### Cycle 87 — BUY (started 2025-04-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-08 09:30:00 | 879.50 | 874.48 | 0.00 | ORB-long ORB[865.00,878.00] vol=1.6x ATR=4.04 |
| Stop hit — per-position SL triggered | 2025-04-08 09:50:00 | 875.46 | 876.16 | 0.00 | SL hit |

### Cycle 88 — SELL (started 2025-04-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-16 11:05:00 | 905.05 | 911.30 | 0.00 | ORB-short ORB[910.20,917.15] vol=1.6x ATR=1.91 |
| Stop hit — per-position SL triggered | 2025-04-16 11:15:00 | 906.96 | 910.86 | 0.00 | SL hit |

### Cycle 89 — SELL (started 2025-04-25 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-25 09:35:00 | 916.05 | 925.32 | 0.00 | ORB-short ORB[923.20,934.45] vol=1.6x ATR=2.45 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:45:00 | 912.38 | 922.24 | 0.00 | T1 1.5R @ 912.38 |
| Target hit | 2025-04-25 13:05:00 | 912.85 | 911.80 | 0.00 | Trail-exit close>VWAP |

### Cycle 90 — SELL (started 2025-04-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-29 09:35:00 | 902.45 | 906.62 | 0.00 | ORB-short ORB[904.10,914.60] vol=1.6x ATR=2.44 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-29 09:40:00 | 898.79 | 905.22 | 0.00 | T1 1.5R @ 898.79 |
| Stop hit — per-position SL triggered | 2025-04-29 09:50:00 | 902.45 | 903.57 | 0.00 | SL hit |

### Cycle 91 — SELL (started 2025-05-06 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 10:10:00 | 878.35 | 887.36 | 0.00 | ORB-short ORB[887.25,897.40] vol=2.0x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-05-06 13:35:00 | 880.78 | 881.65 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-14 10:55:00 | 666.32 | 2024-05-14 11:00:00 | 667.90 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-05-16 10:20:00 | 665.55 | 2024-05-16 10:30:00 | 666.86 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-05-17 10:55:00 | 676.80 | 2024-05-17 11:20:00 | 675.26 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-05-21 10:00:00 | 676.41 | 2024-05-21 10:55:00 | 674.84 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-05-23 09:40:00 | 676.69 | 2024-05-23 10:00:00 | 678.66 | PARTIAL | 0.50 | 0.29% |
| BUY | retest1 | 2024-05-23 09:40:00 | 676.69 | 2024-05-23 10:05:00 | 676.69 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-05-27 10:55:00 | 688.94 | 2024-05-27 11:05:00 | 687.43 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-06-05 11:10:00 | 668.28 | 2024-06-05 11:50:00 | 673.65 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2024-06-05 11:10:00 | 668.28 | 2024-06-05 15:20:00 | 682.53 | TARGET_HIT | 0.50 | 2.13% |
| BUY | retest1 | 2024-06-06 09:40:00 | 693.98 | 2024-06-06 09:45:00 | 691.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2024-06-07 09:35:00 | 704.11 | 2024-06-07 10:00:00 | 707.76 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-06-07 09:35:00 | 704.11 | 2024-06-07 14:20:00 | 715.00 | TARGET_HIT | 0.50 | 1.55% |
| BUY | retest1 | 2024-06-12 10:05:00 | 724.50 | 2024-06-12 10:20:00 | 722.46 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-06-14 10:55:00 | 736.50 | 2024-06-14 11:05:00 | 734.80 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-06-18 10:10:00 | 740.50 | 2024-06-18 11:00:00 | 738.59 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-06-19 10:35:00 | 720.05 | 2024-06-19 10:45:00 | 722.11 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-06-25 11:00:00 | 701.51 | 2024-06-25 12:25:00 | 703.04 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2024-07-01 11:15:00 | 714.70 | 2024-07-01 11:20:00 | 713.30 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-07-02 09:40:00 | 715.21 | 2024-07-02 11:05:00 | 717.58 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2024-07-04 09:30:00 | 722.93 | 2024-07-04 09:45:00 | 725.07 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-07-08 09:40:00 | 705.28 | 2024-07-08 09:50:00 | 706.99 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-07-09 09:40:00 | 709.00 | 2024-07-09 09:45:00 | 706.92 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2024-07-09 09:40:00 | 709.00 | 2024-07-09 11:05:00 | 709.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-11 10:15:00 | 701.48 | 2024-07-11 10:30:00 | 699.07 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2024-07-11 10:15:00 | 701.48 | 2024-07-11 15:20:00 | 695.50 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2024-07-12 09:50:00 | 693.70 | 2024-07-12 10:50:00 | 695.24 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-16 09:50:00 | 701.50 | 2024-07-16 10:05:00 | 703.03 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-07-19 09:50:00 | 701.17 | 2024-07-19 10:20:00 | 703.43 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-07-23 09:55:00 | 685.51 | 2024-07-23 10:15:00 | 682.59 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-07-23 09:55:00 | 685.51 | 2024-07-23 11:00:00 | 685.51 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-24 09:45:00 | 655.47 | 2024-07-24 11:35:00 | 658.65 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-07-26 09:35:00 | 673.68 | 2024-07-26 09:45:00 | 672.24 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-07-30 11:05:00 | 684.21 | 2024-07-30 13:05:00 | 686.59 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-07-30 11:05:00 | 684.21 | 2024-07-30 13:15:00 | 684.21 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-05 11:10:00 | 656.27 | 2024-08-05 11:20:00 | 653.71 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2024-08-05 11:10:00 | 656.27 | 2024-08-05 11:25:00 | 656.27 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-06 10:45:00 | 659.70 | 2024-08-06 10:55:00 | 661.49 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2024-08-12 09:50:00 | 657.25 | 2024-08-12 10:15:00 | 658.38 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2024-08-19 09:40:00 | 664.96 | 2024-08-19 09:45:00 | 663.56 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2024-08-20 09:45:00 | 666.70 | 2024-08-20 10:25:00 | 668.53 | PARTIAL | 0.50 | 0.27% |
| BUY | retest1 | 2024-08-20 09:45:00 | 666.70 | 2024-08-20 15:20:00 | 671.63 | TARGET_HIT | 0.50 | 0.74% |
| BUY | retest1 | 2024-08-26 09:30:00 | 681.79 | 2024-08-26 09:35:00 | 680.57 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2024-08-29 09:45:00 | 696.13 | 2024-08-29 09:50:00 | 697.82 | PARTIAL | 0.50 | 0.24% |
| BUY | retest1 | 2024-08-29 09:45:00 | 696.13 | 2024-08-29 13:45:00 | 706.56 | TARGET_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2024-09-02 10:00:00 | 732.31 | 2024-09-02 10:15:00 | 735.70 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-09-02 10:00:00 | 732.31 | 2024-09-02 15:20:00 | 743.29 | TARGET_HIT | 0.50 | 1.50% |
| SELL | retest1 | 2024-09-05 11:00:00 | 721.90 | 2024-09-05 11:10:00 | 723.41 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-09-10 09:45:00 | 726.60 | 2024-09-10 09:55:00 | 724.08 | PARTIAL | 0.50 | 0.35% |
| SELL | retest1 | 2024-09-10 09:45:00 | 726.60 | 2024-09-10 11:00:00 | 726.60 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-11 10:40:00 | 730.43 | 2024-09-11 11:25:00 | 733.12 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2024-09-11 10:40:00 | 730.43 | 2024-09-11 14:05:00 | 734.10 | TARGET_HIT | 0.50 | 0.50% |
| BUY | retest1 | 2024-09-13 10:05:00 | 750.30 | 2024-09-13 10:15:00 | 753.94 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-09-13 10:05:00 | 750.30 | 2024-09-13 15:20:00 | 760.00 | TARGET_HIT | 0.50 | 1.29% |
| SELL | retest1 | 2024-09-16 10:00:00 | 755.27 | 2024-09-16 10:05:00 | 749.93 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2024-09-16 10:00:00 | 755.27 | 2024-09-16 15:20:00 | 734.10 | TARGET_HIT | 0.50 | 2.80% |
| BUY | retest1 | 2024-09-17 10:10:00 | 741.49 | 2024-09-17 10:15:00 | 745.28 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2024-09-17 10:10:00 | 741.49 | 2024-09-17 10:30:00 | 741.49 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-09-24 09:30:00 | 749.97 | 2024-09-24 09:45:00 | 751.88 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2024-09-25 09:40:00 | 756.57 | 2024-09-25 10:20:00 | 759.21 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2024-09-25 09:40:00 | 756.57 | 2024-09-25 11:50:00 | 757.32 | TARGET_HIT | 0.50 | 0.10% |
| BUY | retest1 | 2024-10-01 10:40:00 | 779.25 | 2024-10-01 11:20:00 | 776.80 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2024-10-03 11:00:00 | 758.69 | 2024-10-03 11:25:00 | 755.69 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2024-10-03 11:00:00 | 758.69 | 2024-10-03 15:20:00 | 743.11 | TARGET_HIT | 0.50 | 2.05% |
| SELL | retest1 | 2024-10-07 10:55:00 | 720.91 | 2024-10-07 11:05:00 | 723.51 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-10-14 09:35:00 | 721.83 | 2024-10-14 09:40:00 | 723.36 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2024-10-15 09:50:00 | 718.61 | 2024-10-15 09:55:00 | 716.22 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2024-10-15 09:50:00 | 718.61 | 2024-10-15 15:20:00 | 702.25 | TARGET_HIT | 0.50 | 2.28% |
| SELL | retest1 | 2024-10-17 11:05:00 | 691.20 | 2024-10-17 13:55:00 | 688.32 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2024-10-17 11:05:00 | 691.20 | 2024-10-17 15:10:00 | 691.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-21 09:35:00 | 685.00 | 2024-10-21 09:45:00 | 682.01 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-10-21 09:35:00 | 685.00 | 2024-10-21 10:05:00 | 685.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-25 09:35:00 | 698.44 | 2024-10-25 09:45:00 | 700.27 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-10-29 09:40:00 | 683.62 | 2024-10-29 09:45:00 | 686.06 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-11-05 10:35:00 | 679.45 | 2024-11-05 10:50:00 | 681.42 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-11-07 11:15:00 | 688.50 | 2024-11-07 11:40:00 | 690.12 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2024-11-12 10:00:00 | 675.84 | 2024-11-12 10:40:00 | 672.68 | PARTIAL | 0.50 | 0.47% |
| SELL | retest1 | 2024-11-12 10:00:00 | 675.84 | 2024-11-12 15:20:00 | 665.00 | TARGET_HIT | 0.50 | 1.60% |
| SELL | retest1 | 2024-11-14 11:10:00 | 656.09 | 2024-11-14 11:30:00 | 657.89 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-18 10:50:00 | 657.36 | 2024-11-18 12:00:00 | 659.20 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-11-21 10:40:00 | 648.84 | 2024-11-21 12:30:00 | 645.57 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-11-21 10:40:00 | 648.84 | 2024-11-21 13:20:00 | 648.84 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-27 09:55:00 | 670.87 | 2024-11-27 10:00:00 | 669.09 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2024-11-28 10:40:00 | 664.63 | 2024-11-28 11:15:00 | 662.08 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2024-11-28 10:40:00 | 664.63 | 2024-11-28 11:45:00 | 664.63 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-04 09:45:00 | 673.80 | 2024-12-04 09:55:00 | 672.33 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2024-12-05 10:55:00 | 670.17 | 2024-12-05 11:25:00 | 671.80 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-10 09:35:00 | 694.46 | 2024-12-10 10:05:00 | 692.87 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2024-12-11 10:00:00 | 691.34 | 2024-12-11 10:10:00 | 692.92 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2024-12-16 09:40:00 | 722.30 | 2024-12-16 12:10:00 | 720.11 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2024-12-17 10:20:00 | 712.31 | 2024-12-17 10:45:00 | 714.03 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2024-12-27 10:00:00 | 694.20 | 2024-12-27 10:30:00 | 697.15 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2024-12-27 10:00:00 | 694.20 | 2024-12-27 11:05:00 | 694.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-01 10:50:00 | 690.60 | 2025-01-01 11:35:00 | 688.88 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-01-03 11:15:00 | 744.00 | 2025-01-03 11:35:00 | 746.45 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-01-03 11:15:00 | 744.00 | 2025-01-03 12:45:00 | 744.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-09 10:30:00 | 728.15 | 2025-01-09 11:25:00 | 725.53 | PARTIAL | 0.50 | 0.36% |
| SELL | retest1 | 2025-01-09 10:30:00 | 728.15 | 2025-01-09 12:15:00 | 727.34 | TARGET_HIT | 0.50 | 0.11% |
| BUY | retest1 | 2025-01-10 11:05:00 | 733.60 | 2025-01-10 11:40:00 | 731.48 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-13 09:40:00 | 718.41 | 2025-01-13 10:00:00 | 720.68 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-01-15 11:05:00 | 718.15 | 2025-01-15 15:20:00 | 717.90 | TARGET_HIT | 1.00 | 0.03% |
| BUY | retest1 | 2025-01-20 10:25:00 | 729.16 | 2025-01-20 10:35:00 | 731.82 | PARTIAL | 0.50 | 0.37% |
| BUY | retest1 | 2025-01-20 10:25:00 | 729.16 | 2025-01-20 15:20:00 | 743.37 | TARGET_HIT | 0.50 | 1.95% |
| BUY | retest1 | 2025-01-23 09:40:00 | 740.00 | 2025-01-23 09:45:00 | 743.46 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-01-23 09:40:00 | 740.00 | 2025-01-23 09:55:00 | 740.45 | TARGET_HIT | 0.50 | 0.06% |
| SELL | retest1 | 2025-02-07 11:00:00 | 844.38 | 2025-02-07 11:10:00 | 847.89 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2025-02-12 09:50:00 | 805.96 | 2025-02-12 10:30:00 | 808.56 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-02-13 09:30:00 | 831.62 | 2025-02-13 09:50:00 | 836.13 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2025-02-13 09:30:00 | 831.62 | 2025-02-13 13:30:00 | 840.72 | TARGET_HIT | 0.50 | 1.09% |
| BUY | retest1 | 2025-02-18 09:45:00 | 845.74 | 2025-02-18 10:15:00 | 843.51 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-02-20 09:45:00 | 835.37 | 2025-02-20 09:50:00 | 838.04 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-02-25 09:30:00 | 843.03 | 2025-02-25 09:35:00 | 845.92 | PARTIAL | 0.50 | 0.34% |
| BUY | retest1 | 2025-02-25 09:30:00 | 843.03 | 2025-02-25 13:15:00 | 846.48 | TARGET_HIT | 0.50 | 0.41% |
| BUY | retest1 | 2025-03-13 09:35:00 | 855.80 | 2025-03-13 09:40:00 | 853.39 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-03-19 09:45:00 | 881.00 | 2025-03-19 09:55:00 | 885.12 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2025-03-19 09:45:00 | 881.00 | 2025-03-19 11:10:00 | 881.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-25 09:30:00 | 917.00 | 2025-03-25 09:35:00 | 920.81 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-03-25 09:30:00 | 917.00 | 2025-03-25 10:05:00 | 917.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-27 10:20:00 | 890.95 | 2025-03-27 10:30:00 | 894.35 | PARTIAL | 0.50 | 0.38% |
| BUY | retest1 | 2025-03-27 10:20:00 | 890.95 | 2025-03-27 10:45:00 | 890.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-04 09:35:00 | 871.27 | 2025-04-04 09:40:00 | 868.37 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-04-08 09:30:00 | 879.50 | 2025-04-08 09:50:00 | 875.46 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2025-04-16 11:05:00 | 905.05 | 2025-04-16 11:15:00 | 906.96 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-04-25 09:35:00 | 916.05 | 2025-04-25 09:45:00 | 912.38 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-04-25 09:35:00 | 916.05 | 2025-04-25 13:05:00 | 912.85 | TARGET_HIT | 0.50 | 0.35% |
| SELL | retest1 | 2025-04-29 09:35:00 | 902.45 | 2025-04-29 09:40:00 | 898.79 | PARTIAL | 0.50 | 0.41% |
| SELL | retest1 | 2025-04-29 09:35:00 | 902.45 | 2025-04-29 09:50:00 | 902.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-05-06 10:10:00 | 878.35 | 2025-05-06 13:35:00 | 880.78 | STOP_HIT | 1.00 | -0.28% |
