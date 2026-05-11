# CG Power and Industrial Solutions Ltd. (CGPOWER)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 875.10
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
| ENTRY1 | 85 |
| ENTRY2 | 0 |
| PARTIAL | 38 |
| TARGET_HIT | 19 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 123 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 57 / 66
- **Target hits / Stop hits / Partials:** 19 / 66 / 38
- **Avg / median % per leg:** 0.18% / 0.00%
- **Sum % (uncompounded):** 22.56%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 33 | 47.1% | 11 | 37 | 22 | 0.19% | 13.1% |
| BUY @ 2nd Alert (retest1) | 70 | 33 | 47.1% | 11 | 37 | 22 | 0.19% | 13.1% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 24 | 45.3% | 8 | 29 | 16 | 0.18% | 9.5% |
| SELL @ 2nd Alert (retest1) | 53 | 24 | 45.3% | 8 | 29 | 16 | 0.18% | 9.5% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 123 | 57 | 46.3% | 19 | 66 | 38 | 0.18% | 22.6% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 10:15:00 | 655.20 | 649.54 | 0.00 | ORB-long ORB[642.20,648.90] vol=2.9x ATR=2.95 |
| Stop hit — per-position SL triggered | 2025-05-13 10:50:00 | 652.25 | 650.45 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2025-05-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-19 09:35:00 | 694.60 | 700.16 | 0.00 | ORB-short ORB[697.00,704.35] vol=1.8x ATR=2.96 |
| Stop hit — per-position SL triggered | 2025-05-19 09:40:00 | 697.56 | 699.67 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 10:15:00 | 702.10 | 693.58 | 0.00 | ORB-long ORB[685.20,692.95] vol=2.0x ATR=2.69 |
| Stop hit — per-position SL triggered | 2025-05-21 11:15:00 | 699.41 | 696.25 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2025-05-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-26 09:40:00 | 706.10 | 701.34 | 0.00 | ORB-long ORB[697.05,700.95] vol=3.5x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-05-26 09:50:00 | 704.12 | 703.82 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 11:10:00 | 691.25 | 688.75 | 0.00 | ORB-long ORB[685.10,690.80] vol=12.2x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-29 11:15:00 | 694.60 | 689.13 | 0.00 | T1 1.5R @ 694.60 |
| Target hit | 2025-05-29 15:20:00 | 697.25 | 693.77 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 6 — SELL (started 2025-06-02 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-02 11:10:00 | 678.05 | 683.90 | 0.00 | ORB-short ORB[678.55,687.90] vol=2.0x ATR=2.22 |
| Stop hit — per-position SL triggered | 2025-06-02 14:25:00 | 680.27 | 680.28 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2025-06-03 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 11:00:00 | 689.85 | 685.53 | 0.00 | ORB-long ORB[678.95,687.60] vol=2.2x ATR=2.16 |
| Stop hit — per-position SL triggered | 2025-06-03 11:10:00 | 687.69 | 685.65 | 0.00 | SL hit |

### Cycle 8 — BUY (started 2025-06-04 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-04 10:20:00 | 689.70 | 683.83 | 0.00 | ORB-long ORB[676.00,684.00] vol=1.9x ATR=2.43 |
| Stop hit — per-position SL triggered | 2025-06-04 10:35:00 | 687.27 | 684.23 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-09 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-09 09:35:00 | 691.45 | 688.45 | 0.00 | ORB-long ORB[682.35,689.50] vol=2.8x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-09 10:00:00 | 694.37 | 689.33 | 0.00 | T1 1.5R @ 694.37 |
| Stop hit — per-position SL triggered | 2025-06-09 10:05:00 | 691.45 | 689.41 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-10 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:55:00 | 699.35 | 695.14 | 0.00 | ORB-long ORB[691.00,698.40] vol=2.6x ATR=1.85 |
| Stop hit — per-position SL triggered | 2025-06-10 11:05:00 | 697.50 | 695.51 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2025-06-11 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-11 09:40:00 | 687.50 | 691.04 | 0.00 | ORB-short ORB[690.10,695.00] vol=1.5x ATR=1.73 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-11 10:20:00 | 684.91 | 688.57 | 0.00 | T1 1.5R @ 684.91 |
| Stop hit — per-position SL triggered | 2025-06-11 11:25:00 | 687.50 | 687.54 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-06-16 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-16 11:10:00 | 679.50 | 673.29 | 0.00 | ORB-long ORB[670.60,677.00] vol=2.1x ATR=2.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-16 13:40:00 | 682.64 | 676.79 | 0.00 | T1 1.5R @ 682.64 |
| Target hit | 2025-06-16 15:20:00 | 683.00 | 678.73 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 13 — SELL (started 2025-06-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 11:05:00 | 682.80 | 685.02 | 0.00 | ORB-short ORB[684.40,691.60] vol=1.7x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-06-17 12:00:00 | 684.64 | 684.69 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:45:00 | 693.75 | 689.43 | 0.00 | ORB-long ORB[682.55,690.25] vol=2.1x ATR=2.32 |
| Stop hit — per-position SL triggered | 2025-06-19 09:55:00 | 691.43 | 689.96 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-06-20 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 10:45:00 | 681.25 | 673.84 | 0.00 | ORB-long ORB[666.55,675.45] vol=2.3x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-20 11:00:00 | 684.89 | 676.54 | 0.00 | T1 1.5R @ 684.89 |
| Target hit | 2025-06-20 15:20:00 | 687.00 | 686.56 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — BUY (started 2025-06-27 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 10:00:00 | 686.65 | 681.66 | 0.00 | ORB-long ORB[674.75,680.00] vol=3.3x ATR=2.34 |
| Stop hit — per-position SL triggered | 2025-06-27 10:15:00 | 684.31 | 683.82 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-07-03 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 11:05:00 | 674.65 | 670.37 | 0.00 | ORB-long ORB[665.00,672.00] vol=2.4x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-07-03 11:45:00 | 673.00 | 671.21 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2025-07-04 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:30:00 | 678.25 | 675.57 | 0.00 | ORB-long ORB[670.80,677.65] vol=3.4x ATR=2.12 |
| Stop hit — per-position SL triggered | 2025-07-04 09:35:00 | 676.13 | 675.61 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-08 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 09:50:00 | 668.30 | 673.74 | 0.00 | ORB-short ORB[670.15,679.35] vol=1.6x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 10:00:00 | 665.23 | 672.83 | 0.00 | T1 1.5R @ 665.23 |
| Stop hit — per-position SL triggered | 2025-07-08 10:05:00 | 668.30 | 672.65 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-09 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 10:35:00 | 669.35 | 672.95 | 0.00 | ORB-short ORB[672.15,676.95] vol=1.9x ATR=1.54 |
| Stop hit — per-position SL triggered | 2025-07-09 11:25:00 | 670.89 | 672.04 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-10 09:30:00 | 678.85 | 677.63 | 0.00 | ORB-long ORB[671.50,677.85] vol=5.1x ATR=1.75 |
| Stop hit — per-position SL triggered | 2025-07-10 09:45:00 | 677.10 | 677.98 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-14 10:40:00 | 677.80 | 671.55 | 0.00 | ORB-long ORB[666.20,672.00] vol=2.1x ATR=1.79 |
| Stop hit — per-position SL triggered | 2025-07-14 10:55:00 | 676.01 | 673.25 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-07-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:30:00 | 680.25 | 673.56 | 0.00 | ORB-long ORB[663.90,671.90] vol=2.9x ATR=2.07 |
| Stop hit — per-position SL triggered | 2025-07-21 10:40:00 | 678.18 | 674.29 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-07-22 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-22 10:55:00 | 680.60 | 682.43 | 0.00 | ORB-short ORB[680.85,687.55] vol=3.7x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-07-22 11:10:00 | 682.30 | 681.98 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-07-23 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-23 10:35:00 | 684.00 | 684.40 | 0.00 | ORB-short ORB[684.15,687.50] vol=1.5x ATR=1.56 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-23 10:40:00 | 681.66 | 684.22 | 0.00 | T1 1.5R @ 681.66 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 684.00 | 683.95 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2025-07-24 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:40:00 | 677.70 | 681.64 | 0.00 | ORB-short ORB[680.10,684.75] vol=2.1x ATR=1.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-24 09:55:00 | 675.55 | 680.63 | 0.00 | T1 1.5R @ 675.55 |
| Target hit | 2025-07-24 14:35:00 | 671.90 | 667.20 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — SELL (started 2025-07-29 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-29 10:25:00 | 646.55 | 654.22 | 0.00 | ORB-short ORB[653.00,658.75] vol=3.7x ATR=2.31 |
| Stop hit — per-position SL triggered | 2025-07-29 10:30:00 | 648.86 | 652.57 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2025-07-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-31 10:55:00 | 662.25 | 656.34 | 0.00 | ORB-long ORB[652.20,659.95] vol=4.2x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 11:00:00 | 665.40 | 657.28 | 0.00 | T1 1.5R @ 665.40 |
| Stop hit — per-position SL triggered | 2025-07-31 13:00:00 | 662.25 | 660.69 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-08-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-07 09:35:00 | 682.30 | 678.91 | 0.00 | ORB-long ORB[672.05,680.20] vol=1.7x ATR=1.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-07 09:55:00 | 685.07 | 681.13 | 0.00 | T1 1.5R @ 685.07 |
| Stop hit — per-position SL triggered | 2025-08-07 10:00:00 | 682.30 | 681.28 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2025-08-11 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-11 09:45:00 | 657.25 | 661.94 | 0.00 | ORB-short ORB[659.30,667.85] vol=3.3x ATR=2.41 |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 659.66 | 660.84 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-08-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 11:05:00 | 665.50 | 662.26 | 0.00 | ORB-long ORB[659.20,663.85] vol=1.7x ATR=1.61 |
| Stop hit — per-position SL triggered | 2025-08-13 11:20:00 | 663.89 | 662.45 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2025-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-22 09:35:00 | 687.40 | 683.54 | 0.00 | ORB-long ORB[677.60,684.90] vol=3.7x ATR=1.84 |
| Stop hit — per-position SL triggered | 2025-08-22 09:40:00 | 685.56 | 684.09 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-08-26 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-26 10:00:00 | 681.00 | 676.46 | 0.00 | ORB-long ORB[673.30,680.80] vol=1.8x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-08-26 10:45:00 | 679.12 | 677.86 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2025-09-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:35:00 | 716.80 | 709.49 | 0.00 | ORB-long ORB[703.50,711.90] vol=2.9x ATR=3.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 09:45:00 | 722.27 | 712.56 | 0.00 | T1 1.5R @ 722.27 |
| Stop hit — per-position SL triggered | 2025-09-01 10:00:00 | 716.80 | 713.82 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-09-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-02 09:40:00 | 728.20 | 722.99 | 0.00 | ORB-long ORB[718.30,727.00] vol=1.6x ATR=2.35 |
| Stop hit — per-position SL triggered | 2025-09-02 10:05:00 | 725.85 | 725.07 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2025-09-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-04 09:45:00 | 746.00 | 740.19 | 0.00 | ORB-long ORB[735.00,744.30] vol=1.5x ATR=2.62 |
| Stop hit — per-position SL triggered | 2025-09-04 09:55:00 | 743.38 | 741.95 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2025-09-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 10:00:00 | 755.70 | 749.80 | 0.00 | ORB-long ORB[746.10,753.50] vol=3.1x ATR=2.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 10:35:00 | 759.34 | 752.95 | 0.00 | T1 1.5R @ 759.34 |
| Target hit | 2025-09-10 15:20:00 | 768.35 | 761.66 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 38 — BUY (started 2025-09-11 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-11 10:30:00 | 782.00 | 776.68 | 0.00 | ORB-long ORB[764.70,775.40] vol=1.7x ATR=2.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-11 11:10:00 | 785.95 | 778.31 | 0.00 | T1 1.5R @ 785.95 |
| Stop hit — per-position SL triggered | 2025-09-11 11:50:00 | 782.00 | 779.75 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-15 11:15:00 | 791.65 | 785.94 | 0.00 | ORB-long ORB[784.80,789.00] vol=3.2x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-15 11:30:00 | 794.47 | 787.39 | 0.00 | T1 1.5R @ 794.47 |
| Stop hit — per-position SL triggered | 2025-09-15 12:25:00 | 791.65 | 789.41 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-09-16 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-16 10:10:00 | 789.85 | 793.13 | 0.00 | ORB-short ORB[790.00,796.25] vol=1.7x ATR=1.66 |
| Stop hit — per-position SL triggered | 2025-09-16 10:55:00 | 791.51 | 792.10 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2025-09-17 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-17 09:30:00 | 796.90 | 794.66 | 0.00 | ORB-long ORB[792.20,795.90] vol=1.9x ATR=1.78 |
| Stop hit — per-position SL triggered | 2025-09-17 09:35:00 | 795.12 | 794.82 | 0.00 | SL hit |

### Cycle 42 — SELL (started 2025-09-18 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-18 09:55:00 | 784.50 | 786.41 | 0.00 | ORB-short ORB[785.15,793.25] vol=1.9x ATR=1.88 |
| Stop hit — per-position SL triggered | 2025-09-18 10:05:00 | 786.38 | 786.29 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-09-19 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-19 10:55:00 | 778.05 | 782.72 | 0.00 | ORB-short ORB[782.00,787.60] vol=1.6x ATR=1.65 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-19 11:35:00 | 775.57 | 781.76 | 0.00 | T1 1.5R @ 775.57 |
| Stop hit — per-position SL triggered | 2025-09-19 15:00:00 | 778.05 | 778.74 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-09-23 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-23 11:10:00 | 770.25 | 774.06 | 0.00 | ORB-short ORB[771.00,777.70] vol=2.0x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-09-23 11:20:00 | 772.18 | 773.95 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2025-09-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-26 09:30:00 | 748.70 | 753.67 | 0.00 | ORB-short ORB[751.35,759.45] vol=2.0x ATR=2.23 |
| Stop hit — per-position SL triggered | 2025-09-26 09:40:00 | 750.93 | 752.53 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-09-30 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:05:00 | 742.60 | 744.02 | 0.00 | ORB-short ORB[744.10,751.75] vol=3.5x ATR=2.15 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 10:20:00 | 739.38 | 743.81 | 0.00 | T1 1.5R @ 739.38 |
| Stop hit — per-position SL triggered | 2025-09-30 11:05:00 | 742.60 | 743.22 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-10-08 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-08 10:20:00 | 742.40 | 744.83 | 0.00 | ORB-short ORB[742.50,749.60] vol=1.7x ATR=1.74 |
| Stop hit — per-position SL triggered | 2025-10-08 10:25:00 | 744.14 | 744.76 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-10-09 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 09:55:00 | 753.55 | 746.46 | 0.00 | ORB-long ORB[741.80,747.10] vol=1.8x ATR=2.31 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 10:25:00 | 757.01 | 750.23 | 0.00 | T1 1.5R @ 757.01 |
| Target hit | 2025-10-09 15:20:00 | 761.80 | 756.88 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 49 — SELL (started 2025-10-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-13 09:30:00 | 743.35 | 747.31 | 0.00 | ORB-short ORB[744.75,751.65] vol=1.9x ATR=2.30 |
| Stop hit — per-position SL triggered | 2025-10-13 09:35:00 | 745.65 | 747.00 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-10-14 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-14 10:40:00 | 744.15 | 747.50 | 0.00 | ORB-short ORB[746.10,751.65] vol=8.0x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-10-14 10:45:00 | 745.98 | 747.46 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-10-20 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-20 10:30:00 | 753.05 | 757.50 | 0.00 | ORB-short ORB[756.55,762.90] vol=3.7x ATR=2.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-20 10:40:00 | 749.56 | 754.51 | 0.00 | T1 1.5R @ 749.56 |
| Target hit | 2025-10-20 12:30:00 | 740.95 | 740.90 | 0.00 | Trail-exit close>VWAP |

### Cycle 52 — BUY (started 2025-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 11:15:00 | 727.20 | 726.18 | 0.00 | ORB-long ORB[724.55,727.00] vol=1.6x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-27 14:40:00 | 729.56 | 726.95 | 0.00 | T1 1.5R @ 729.56 |
| Target hit | 2025-10-27 15:20:00 | 729.35 | 727.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 53 — BUY (started 2025-10-29 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-29 10:40:00 | 732.35 | 727.63 | 0.00 | ORB-long ORB[721.15,727.00] vol=3.2x ATR=2.14 |
| Stop hit — per-position SL triggered | 2025-10-29 10:45:00 | 730.21 | 728.14 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2025-10-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 10:20:00 | 729.10 | 732.13 | 0.00 | ORB-short ORB[732.10,738.00] vol=2.6x ATR=1.83 |
| Stop hit — per-position SL triggered | 2025-10-31 10:45:00 | 730.93 | 731.61 | 0.00 | SL hit |

### Cycle 55 — BUY (started 2025-11-03 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-03 10:45:00 | 737.00 | 732.56 | 0.00 | ORB-long ORB[727.00,735.30] vol=1.8x ATR=2.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-03 11:05:00 | 740.20 | 733.65 | 0.00 | T1 1.5R @ 740.20 |
| Target hit | 2025-11-03 15:20:00 | 745.25 | 737.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — BUY (started 2025-11-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:00:00 | 752.00 | 745.81 | 0.00 | ORB-long ORB[740.35,745.30] vol=2.7x ATR=2.38 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:10:00 | 755.57 | 749.04 | 0.00 | T1 1.5R @ 755.57 |
| Stop hit — per-position SL triggered | 2025-11-04 10:20:00 | 752.00 | 749.60 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2025-11-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:35:00 | 734.05 | 735.22 | 0.00 | ORB-short ORB[734.25,738.65] vol=2.0x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-11-11 10:05:00 | 735.70 | 734.44 | 0.00 | SL hit |

### Cycle 58 — BUY (started 2025-11-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-13 10:20:00 | 742.90 | 740.23 | 0.00 | ORB-long ORB[735.00,741.55] vol=1.5x ATR=1.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 10:35:00 | 745.32 | 741.96 | 0.00 | T1 1.5R @ 745.32 |
| Target hit | 2025-11-13 12:50:00 | 745.05 | 745.50 | 0.00 | Trail-exit close<VWAP |

### Cycle 59 — SELL (started 2025-11-19 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-19 10:25:00 | 726.00 | 729.38 | 0.00 | ORB-short ORB[728.50,734.00] vol=1.7x ATR=1.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-19 11:00:00 | 723.57 | 728.17 | 0.00 | T1 1.5R @ 723.57 |
| Target hit | 2025-11-19 14:40:00 | 723.75 | 723.06 | 0.00 | Trail-exit close>VWAP |

### Cycle 60 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 11:15:00 | 715.55 | 719.10 | 0.00 | ORB-short ORB[719.10,724.60] vol=5.0x ATR=1.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 13:10:00 | 712.67 | 717.27 | 0.00 | T1 1.5R @ 712.67 |
| Target hit | 2025-11-21 15:20:00 | 710.40 | 714.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 61 — SELL (started 2025-11-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-27 09:40:00 | 678.20 | 682.49 | 0.00 | ORB-short ORB[681.30,691.15] vol=2.6x ATR=1.81 |
| Stop hit — per-position SL triggered | 2025-11-27 10:00:00 | 680.01 | 681.13 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-12-02 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 10:00:00 | 662.70 | 665.74 | 0.00 | ORB-short ORB[666.30,671.30] vol=1.7x ATR=1.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-02 10:15:00 | 660.50 | 664.85 | 0.00 | T1 1.5R @ 660.50 |
| Stop hit — per-position SL triggered | 2025-12-02 10:30:00 | 662.70 | 664.16 | 0.00 | SL hit |

### Cycle 63 — BUY (started 2025-12-10 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:45:00 | 674.90 | 674.33 | 0.00 | ORB-long ORB[668.75,674.50] vol=1.8x ATR=2.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 10:25:00 | 678.60 | 674.82 | 0.00 | T1 1.5R @ 678.60 |
| Stop hit — per-position SL triggered | 2025-12-10 10:40:00 | 674.90 | 675.00 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-12-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-22 11:00:00 | 676.95 | 674.43 | 0.00 | ORB-long ORB[671.30,676.50] vol=3.0x ATR=1.45 |
| Stop hit — per-position SL triggered | 2025-12-22 11:20:00 | 675.50 | 675.02 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2025-12-24 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 10:45:00 | 662.00 | 665.15 | 0.00 | ORB-short ORB[663.35,669.25] vol=1.6x ATR=1.36 |
| Stop hit — per-position SL triggered | 2025-12-24 11:20:00 | 663.36 | 663.11 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2025-12-29 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-29 11:00:00 | 652.80 | 655.26 | 0.00 | ORB-short ORB[653.10,657.75] vol=5.1x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-29 12:10:00 | 650.94 | 654.43 | 0.00 | T1 1.5R @ 650.94 |
| Target hit | 2025-12-29 15:20:00 | 647.45 | 650.67 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2025-12-30 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:50:00 | 642.45 | 644.96 | 0.00 | ORB-short ORB[643.75,648.00] vol=1.8x ATR=1.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 11:20:00 | 640.41 | 643.96 | 0.00 | T1 1.5R @ 640.41 |
| Stop hit — per-position SL triggered | 2025-12-30 11:25:00 | 642.45 | 643.89 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2026-01-01 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-01 09:55:00 | 645.05 | 646.56 | 0.00 | ORB-short ORB[645.70,650.95] vol=2.6x ATR=1.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-01 10:05:00 | 642.61 | 643.44 | 0.00 | T1 1.5R @ 642.61 |
| Target hit | 2026-01-01 15:20:00 | 638.65 | 640.95 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 69 — SELL (started 2026-01-05 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 11:00:00 | 643.20 | 646.19 | 0.00 | ORB-short ORB[645.75,653.30] vol=3.5x ATR=1.61 |
| Stop hit — per-position SL triggered | 2026-01-05 11:05:00 | 644.81 | 645.87 | 0.00 | SL hit |

### Cycle 70 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-06 11:15:00 | 639.85 | 642.51 | 0.00 | ORB-short ORB[641.35,646.65] vol=2.4x ATR=1.15 |
| Stop hit — per-position SL triggered | 2026-01-06 11:30:00 | 641.00 | 642.25 | 0.00 | SL hit |

### Cycle 71 — SELL (started 2026-01-16 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-16 11:05:00 | 572.35 | 576.27 | 0.00 | ORB-short ORB[574.65,581.40] vol=1.8x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 12:25:00 | 569.80 | 574.62 | 0.00 | T1 1.5R @ 569.80 |
| Target hit | 2026-01-16 15:20:00 | 560.85 | 566.59 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 72 — BUY (started 2026-02-06 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-06 09:30:00 | 672.00 | 670.04 | 0.00 | ORB-long ORB[666.95,670.50] vol=1.9x ATR=2.09 |
| Stop hit — per-position SL triggered | 2026-02-06 09:35:00 | 669.91 | 669.92 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2026-02-09 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-09 10:20:00 | 691.35 | 685.14 | 0.00 | ORB-long ORB[678.25,686.90] vol=2.4x ATR=2.28 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-09 10:55:00 | 694.77 | 687.04 | 0.00 | T1 1.5R @ 694.77 |
| Stop hit — per-position SL triggered | 2026-02-09 11:25:00 | 691.35 | 687.83 | 0.00 | SL hit |

### Cycle 74 — SELL (started 2026-02-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-10 09:30:00 | 687.20 | 690.01 | 0.00 | ORB-short ORB[688.15,694.95] vol=1.8x ATR=1.85 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-10 09:40:00 | 684.42 | 688.84 | 0.00 | T1 1.5R @ 684.42 |
| Stop hit — per-position SL triggered | 2026-02-10 10:25:00 | 687.20 | 687.75 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 691.05 | 688.44 | 0.00 | ORB-long ORB[684.85,690.00] vol=4.0x ATR=1.70 |
| Stop hit — per-position SL triggered | 2026-02-17 10:05:00 | 689.35 | 690.42 | 0.00 | SL hit |

### Cycle 76 — BUY (started 2026-02-18 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-18 10:55:00 | 694.00 | 691.02 | 0.00 | ORB-long ORB[687.40,693.00] vol=1.8x ATR=1.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 11:10:00 | 696.23 | 691.96 | 0.00 | T1 1.5R @ 696.23 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 694.00 | 692.07 | 0.00 | SL hit |

### Cycle 77 — BUY (started 2026-02-26 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 10:20:00 | 727.10 | 719.05 | 0.00 | ORB-long ORB[713.00,717.80] vol=4.3x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-26 10:25:00 | 730.90 | 721.62 | 0.00 | T1 1.5R @ 730.90 |
| Stop hit — per-position SL triggered | 2026-02-26 10:45:00 | 727.10 | 724.66 | 0.00 | SL hit |

### Cycle 78 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-27 10:15:00 | 721.10 | 722.93 | 0.00 | ORB-short ORB[722.30,728.70] vol=1.6x ATR=1.88 |
| Stop hit — per-position SL triggered | 2026-02-27 10:25:00 | 722.98 | 722.76 | 0.00 | SL hit |

### Cycle 79 — BUY (started 2026-03-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-10 09:40:00 | 718.20 | 711.61 | 0.00 | ORB-long ORB[703.30,714.00] vol=2.4x ATR=3.35 |
| Stop hit — per-position SL triggered | 2026-03-10 09:45:00 | 714.85 | 712.57 | 0.00 | SL hit |

### Cycle 80 — BUY (started 2026-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-11 09:30:00 | 737.35 | 733.64 | 0.00 | ORB-long ORB[725.65,733.90] vol=2.4x ATR=2.90 |
| Stop hit — per-position SL triggered | 2026-03-11 11:10:00 | 734.45 | 736.92 | 0.00 | SL hit |

### Cycle 81 — BUY (started 2026-03-17 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-17 11:05:00 | 710.60 | 703.06 | 0.00 | ORB-long ORB[697.00,704.55] vol=1.8x ATR=2.80 |
| Stop hit — per-position SL triggered | 2026-03-17 11:20:00 | 707.80 | 704.02 | 0.00 | SL hit |

### Cycle 82 — SELL (started 2026-03-23 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-23 09:35:00 | 667.00 | 670.12 | 0.00 | ORB-short ORB[667.30,675.50] vol=2.0x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:50:00 | 662.25 | 668.74 | 0.00 | T1 1.5R @ 662.25 |
| Target hit | 2026-03-23 13:45:00 | 657.00 | 655.03 | 0.00 | Trail-exit close>VWAP |

### Cycle 83 — BUY (started 2026-03-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-25 09:55:00 | 682.80 | 679.60 | 0.00 | ORB-long ORB[675.00,681.55] vol=1.8x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-25 12:35:00 | 687.57 | 681.62 | 0.00 | T1 1.5R @ 687.57 |
| Target hit | 2026-03-25 15:20:00 | 692.80 | 685.05 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 84 — BUY (started 2026-04-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:40:00 | 738.50 | 734.64 | 0.00 | ORB-long ORB[726.50,737.30] vol=3.1x ATR=3.20 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:05:00 | 743.30 | 738.48 | 0.00 | T1 1.5R @ 743.30 |
| Target hit | 2026-04-15 15:20:00 | 748.30 | 742.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 85 — BUY (started 2026-04-22 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-22 10:20:00 | 818.00 | 808.43 | 0.00 | ORB-long ORB[804.05,810.95] vol=3.0x ATR=3.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-22 10:25:00 | 823.04 | 810.73 | 0.00 | T1 1.5R @ 823.04 |
| Target hit | 2026-04-22 15:20:00 | 825.55 | 821.97 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 10:15:00 | 655.20 | 2025-05-13 10:50:00 | 652.25 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-05-19 09:35:00 | 694.60 | 2025-05-19 09:40:00 | 697.56 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-05-21 10:15:00 | 702.10 | 2025-05-21 11:15:00 | 699.41 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-05-26 09:40:00 | 706.10 | 2025-05-26 09:50:00 | 704.12 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-05-29 11:10:00 | 691.25 | 2025-05-29 11:15:00 | 694.60 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-05-29 11:10:00 | 691.25 | 2025-05-29 15:20:00 | 697.25 | TARGET_HIT | 0.50 | 0.87% |
| SELL | retest1 | 2025-06-02 11:10:00 | 678.05 | 2025-06-02 14:25:00 | 680.27 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-03 11:00:00 | 689.85 | 2025-06-03 11:10:00 | 687.69 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-04 10:20:00 | 689.70 | 2025-06-04 10:35:00 | 687.27 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-06-09 09:35:00 | 691.45 | 2025-06-09 10:00:00 | 694.37 | PARTIAL | 0.50 | 0.42% |
| BUY | retest1 | 2025-06-09 09:35:00 | 691.45 | 2025-06-09 10:05:00 | 691.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-10 10:55:00 | 699.35 | 2025-06-10 11:05:00 | 697.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2025-06-11 09:40:00 | 687.50 | 2025-06-11 10:20:00 | 684.91 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-06-11 09:40:00 | 687.50 | 2025-06-11 11:25:00 | 687.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-16 11:10:00 | 679.50 | 2025-06-16 13:40:00 | 682.64 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-06-16 11:10:00 | 679.50 | 2025-06-16 15:20:00 | 683.00 | TARGET_HIT | 0.50 | 0.52% |
| SELL | retest1 | 2025-06-17 11:05:00 | 682.80 | 2025-06-17 12:00:00 | 684.64 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-06-19 09:45:00 | 693.75 | 2025-06-19 09:55:00 | 691.43 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-06-20 10:45:00 | 681.25 | 2025-06-20 11:00:00 | 684.89 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2025-06-20 10:45:00 | 681.25 | 2025-06-20 15:20:00 | 687.00 | TARGET_HIT | 0.50 | 0.84% |
| BUY | retest1 | 2025-06-27 10:00:00 | 686.65 | 2025-06-27 10:15:00 | 684.31 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-07-03 11:05:00 | 674.65 | 2025-07-03 11:45:00 | 673.00 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-07-04 09:30:00 | 678.25 | 2025-07-04 09:35:00 | 676.13 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-07-08 09:50:00 | 668.30 | 2025-07-08 10:00:00 | 665.23 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-07-08 09:50:00 | 668.30 | 2025-07-08 10:05:00 | 668.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-09 10:35:00 | 669.35 | 2025-07-09 11:25:00 | 670.89 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-07-10 09:30:00 | 678.85 | 2025-07-10 09:45:00 | 677.10 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-07-14 10:40:00 | 677.80 | 2025-07-14 10:55:00 | 676.01 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-07-21 10:30:00 | 680.25 | 2025-07-21 10:40:00 | 678.18 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-07-22 10:55:00 | 680.60 | 2025-07-22 11:10:00 | 682.30 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-23 10:35:00 | 684.00 | 2025-07-23 10:40:00 | 681.66 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-07-23 10:35:00 | 684.00 | 2025-07-23 11:15:00 | 684.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-24 09:40:00 | 677.70 | 2025-07-24 09:55:00 | 675.55 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-07-24 09:40:00 | 677.70 | 2025-07-24 14:35:00 | 671.90 | TARGET_HIT | 0.50 | 0.86% |
| SELL | retest1 | 2025-07-29 10:25:00 | 646.55 | 2025-07-29 10:30:00 | 648.86 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2025-07-31 10:55:00 | 662.25 | 2025-07-31 11:00:00 | 665.40 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-07-31 10:55:00 | 662.25 | 2025-07-31 13:00:00 | 662.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-08-07 09:35:00 | 682.30 | 2025-08-07 09:55:00 | 685.07 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-08-07 09:35:00 | 682.30 | 2025-08-07 10:00:00 | 682.30 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-11 09:45:00 | 657.25 | 2025-08-11 10:15:00 | 659.66 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-08-13 11:05:00 | 665.50 | 2025-08-13 11:20:00 | 663.89 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2025-08-22 09:35:00 | 687.40 | 2025-08-22 09:40:00 | 685.56 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-08-26 10:00:00 | 681.00 | 2025-08-26 10:45:00 | 679.12 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-09-01 09:35:00 | 716.80 | 2025-09-01 09:45:00 | 722.27 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2025-09-01 09:35:00 | 716.80 | 2025-09-01 10:00:00 | 716.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-02 09:40:00 | 728.20 | 2025-09-02 10:05:00 | 725.85 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-09-04 09:45:00 | 746.00 | 2025-09-04 09:55:00 | 743.38 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-09-10 10:00:00 | 755.70 | 2025-09-10 10:35:00 | 759.34 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-09-10 10:00:00 | 755.70 | 2025-09-10 15:20:00 | 768.35 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2025-09-11 10:30:00 | 782.00 | 2025-09-11 11:10:00 | 785.95 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2025-09-11 10:30:00 | 782.00 | 2025-09-11 11:50:00 | 782.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-15 11:15:00 | 791.65 | 2025-09-15 11:30:00 | 794.47 | PARTIAL | 0.50 | 0.36% |
| BUY | retest1 | 2025-09-15 11:15:00 | 791.65 | 2025-09-15 12:25:00 | 791.65 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-16 10:10:00 | 789.85 | 2025-09-16 10:55:00 | 791.51 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest1 | 2025-09-17 09:30:00 | 796.90 | 2025-09-17 09:35:00 | 795.12 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2025-09-18 09:55:00 | 784.50 | 2025-09-18 10:05:00 | 786.38 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-09-19 10:55:00 | 778.05 | 2025-09-19 11:35:00 | 775.57 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-09-19 10:55:00 | 778.05 | 2025-09-19 15:00:00 | 778.05 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-23 11:10:00 | 770.25 | 2025-09-23 11:20:00 | 772.18 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-09-26 09:30:00 | 748.70 | 2025-09-26 09:40:00 | 750.93 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-30 10:05:00 | 742.60 | 2025-09-30 10:20:00 | 739.38 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2025-09-30 10:05:00 | 742.60 | 2025-09-30 11:05:00 | 742.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-08 10:20:00 | 742.40 | 2025-10-08 10:25:00 | 744.14 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest1 | 2025-10-09 09:55:00 | 753.55 | 2025-10-09 10:25:00 | 757.01 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2025-10-09 09:55:00 | 753.55 | 2025-10-09 15:20:00 | 761.80 | TARGET_HIT | 0.50 | 1.09% |
| SELL | retest1 | 2025-10-13 09:30:00 | 743.35 | 2025-10-13 09:35:00 | 745.65 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-10-14 10:40:00 | 744.15 | 2025-10-14 10:45:00 | 745.98 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-20 10:30:00 | 753.05 | 2025-10-20 10:40:00 | 749.56 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2025-10-20 10:30:00 | 753.05 | 2025-10-20 12:30:00 | 740.95 | TARGET_HIT | 0.50 | 1.61% |
| BUY | retest1 | 2025-10-27 11:15:00 | 727.20 | 2025-10-27 14:40:00 | 729.56 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2025-10-27 11:15:00 | 727.20 | 2025-10-27 15:20:00 | 729.35 | TARGET_HIT | 0.50 | 0.30% |
| BUY | retest1 | 2025-10-29 10:40:00 | 732.35 | 2025-10-29 10:45:00 | 730.21 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-10-31 10:20:00 | 729.10 | 2025-10-31 10:45:00 | 730.93 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2025-11-03 10:45:00 | 737.00 | 2025-11-03 11:05:00 | 740.20 | PARTIAL | 0.50 | 0.43% |
| BUY | retest1 | 2025-11-03 10:45:00 | 737.00 | 2025-11-03 15:20:00 | 745.25 | TARGET_HIT | 0.50 | 1.12% |
| BUY | retest1 | 2025-11-04 10:00:00 | 752.00 | 2025-11-04 10:10:00 | 755.57 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-11-04 10:00:00 | 752.00 | 2025-11-04 10:20:00 | 752.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 09:35:00 | 734.05 | 2025-11-11 10:05:00 | 735.70 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-11-13 10:20:00 | 742.90 | 2025-11-13 10:35:00 | 745.32 | PARTIAL | 0.50 | 0.33% |
| BUY | retest1 | 2025-11-13 10:20:00 | 742.90 | 2025-11-13 12:50:00 | 745.05 | TARGET_HIT | 0.50 | 0.29% |
| SELL | retest1 | 2025-11-19 10:25:00 | 726.00 | 2025-11-19 11:00:00 | 723.57 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-19 10:25:00 | 726.00 | 2025-11-19 14:40:00 | 723.75 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-11-21 11:15:00 | 715.55 | 2025-11-21 13:10:00 | 712.67 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2025-11-21 11:15:00 | 715.55 | 2025-11-21 15:20:00 | 710.40 | TARGET_HIT | 0.50 | 0.72% |
| SELL | retest1 | 2025-11-27 09:40:00 | 678.20 | 2025-11-27 10:00:00 | 680.01 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2025-12-02 10:00:00 | 662.70 | 2025-12-02 10:15:00 | 660.50 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-02 10:00:00 | 662.70 | 2025-12-02 10:30:00 | 662.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-10 09:45:00 | 674.90 | 2025-12-10 10:25:00 | 678.60 | PARTIAL | 0.50 | 0.55% |
| BUY | retest1 | 2025-12-10 09:45:00 | 674.90 | 2025-12-10 10:40:00 | 674.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-22 11:00:00 | 676.95 | 2025-12-22 11:20:00 | 675.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-24 10:45:00 | 662.00 | 2025-12-24 11:20:00 | 663.36 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2025-12-29 11:00:00 | 652.80 | 2025-12-29 12:10:00 | 650.94 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-12-29 11:00:00 | 652.80 | 2025-12-29 15:20:00 | 647.45 | TARGET_HIT | 0.50 | 0.82% |
| SELL | retest1 | 2025-12-30 10:50:00 | 642.45 | 2025-12-30 11:20:00 | 640.41 | PARTIAL | 0.50 | 0.32% |
| SELL | retest1 | 2025-12-30 10:50:00 | 642.45 | 2025-12-30 11:25:00 | 642.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-01-01 09:55:00 | 645.05 | 2026-01-01 10:05:00 | 642.61 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2026-01-01 09:55:00 | 645.05 | 2026-01-01 15:20:00 | 638.65 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2026-01-05 11:00:00 | 643.20 | 2026-01-05 11:05:00 | 644.81 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2026-01-06 11:15:00 | 639.85 | 2026-01-06 11:30:00 | 641.00 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2026-01-16 11:05:00 | 572.35 | 2026-01-16 12:25:00 | 569.80 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-01-16 11:05:00 | 572.35 | 2026-01-16 15:20:00 | 560.85 | TARGET_HIT | 0.50 | 2.01% |
| BUY | retest1 | 2026-02-06 09:30:00 | 672.00 | 2026-02-06 09:35:00 | 669.91 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2026-02-09 10:20:00 | 691.35 | 2026-02-09 10:55:00 | 694.77 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2026-02-09 10:20:00 | 691.35 | 2026-02-09 11:25:00 | 691.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-10 09:30:00 | 687.20 | 2026-02-10 09:40:00 | 684.42 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2026-02-10 09:30:00 | 687.20 | 2026-02-10 10:25:00 | 687.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-17 09:40:00 | 691.05 | 2026-02-17 10:05:00 | 689.35 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2026-02-18 10:55:00 | 694.00 | 2026-02-18 11:10:00 | 696.23 | PARTIAL | 0.50 | 0.32% |
| BUY | retest1 | 2026-02-18 10:55:00 | 694.00 | 2026-02-18 11:15:00 | 694.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-26 10:20:00 | 727.10 | 2026-02-26 10:25:00 | 730.90 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2026-02-26 10:20:00 | 727.10 | 2026-02-26 10:45:00 | 727.10 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-27 10:15:00 | 721.10 | 2026-02-27 10:25:00 | 722.98 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2026-03-10 09:40:00 | 718.20 | 2026-03-10 09:45:00 | 714.85 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-03-11 09:30:00 | 737.35 | 2026-03-11 11:10:00 | 734.45 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2026-03-17 11:05:00 | 710.60 | 2026-03-17 11:20:00 | 707.80 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2026-03-23 09:35:00 | 667.00 | 2026-03-23 09:50:00 | 662.25 | PARTIAL | 0.50 | 0.71% |
| SELL | retest1 | 2026-03-23 09:35:00 | 667.00 | 2026-03-23 13:45:00 | 657.00 | TARGET_HIT | 0.50 | 1.50% |
| BUY | retest1 | 2026-03-25 09:55:00 | 682.80 | 2026-03-25 12:35:00 | 687.57 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2026-03-25 09:55:00 | 682.80 | 2026-03-25 15:20:00 | 692.80 | TARGET_HIT | 0.50 | 1.46% |
| BUY | retest1 | 2026-04-15 09:40:00 | 738.50 | 2026-04-15 10:05:00 | 743.30 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2026-04-15 09:40:00 | 738.50 | 2026-04-15 15:20:00 | 748.30 | TARGET_HIT | 0.50 | 1.33% |
| BUY | retest1 | 2026-04-22 10:20:00 | 818.00 | 2026-04-22 10:25:00 | 823.04 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-22 10:20:00 | 818.00 | 2026-04-22 15:20:00 | 825.55 | TARGET_HIT | 0.50 | 0.92% |
