# Tata Investment Corporation Ltd. (TATAINVEST)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 719.00
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
| ENTRY1 | 67 |
| ENTRY2 | 0 |
| PARTIAL | 31 |
| TARGET_HIT | 7 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 98 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 38 / 60
- **Target hits / Stop hits / Partials:** 7 / 60 / 31
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 7.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 45 | 13 | 28.9% | 0 | 32 | 13 | 0.01% | 0.2% |
| BUY @ 2nd Alert (retest1) | 45 | 13 | 28.9% | 0 | 32 | 13 | 0.01% | 0.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 53 | 25 | 47.2% | 7 | 28 | 18 | 0.14% | 7.3% |
| SELL @ 2nd Alert (retest1) | 53 | 25 | 47.2% | 7 | 28 | 18 | 0.14% | 7.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 98 | 38 | 38.8% | 7 | 60 | 31 | 0.08% | 7.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-13 10:40:00 | 610.20 | 605.78 | 0.00 | ORB-long ORB[599.50,605.95] vol=4.4x ATR=2.26 |
| Stop hit — per-position SL triggered | 2025-05-13 11:10:00 | 607.94 | 606.65 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-05-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:30:00 | 613.05 | 610.87 | 0.00 | ORB-long ORB[606.30,612.55] vol=4.3x ATR=2.01 |
| Stop hit — per-position SL triggered | 2025-05-14 10:00:00 | 611.04 | 611.37 | 0.00 | SL hit |

### Cycle 3 — BUY (started 2025-05-15 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:40:00 | 618.75 | 614.15 | 0.00 | ORB-long ORB[610.30,615.00] vol=3.7x ATR=2.00 |
| Stop hit — per-position SL triggered | 2025-05-15 09:45:00 | 616.75 | 614.60 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-05-22 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-22 09:40:00 | 616.30 | 618.31 | 0.00 | ORB-short ORB[617.00,621.15] vol=1.9x ATR=1.67 |
| Stop hit — per-position SL triggered | 2025-05-22 10:20:00 | 617.97 | 617.67 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-05-23 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-23 09:50:00 | 624.35 | 620.31 | 0.00 | ORB-long ORB[614.00,622.60] vol=3.5x ATR=2.00 |
| Stop hit — per-position SL triggered | 2025-05-23 09:55:00 | 622.35 | 620.52 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-05-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-28 10:00:00 | 635.70 | 633.06 | 0.00 | ORB-long ORB[630.60,634.45] vol=3.5x ATR=2.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-28 10:05:00 | 638.76 | 634.52 | 0.00 | T1 1.5R @ 638.76 |
| Stop hit — per-position SL triggered | 2025-05-28 10:10:00 | 635.70 | 634.60 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-03 11:15:00 | 672.50 | 677.35 | 0.00 | ORB-short ORB[675.00,684.55] vol=2.5x ATR=1.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 12:20:00 | 670.24 | 676.17 | 0.00 | T1 1.5R @ 670.24 |
| Target hit | 2025-06-03 15:20:00 | 664.90 | 672.30 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — SELL (started 2025-06-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:40:00 | 665.35 | 669.54 | 0.00 | ORB-short ORB[665.90,673.15] vol=1.6x ATR=2.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-04 09:50:00 | 662.00 | 668.02 | 0.00 | T1 1.5R @ 662.00 |
| Stop hit — per-position SL triggered | 2025-06-04 10:55:00 | 665.35 | 666.87 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2025-06-10 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 10:35:00 | 697.70 | 690.27 | 0.00 | ORB-long ORB[685.90,691.50] vol=4.1x ATR=1.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-10 10:40:00 | 700.57 | 696.15 | 0.00 | T1 1.5R @ 700.57 |
| Stop hit — per-position SL triggered | 2025-06-10 10:45:00 | 697.70 | 696.61 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-06-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:30:00 | 664.95 | 660.14 | 0.00 | ORB-long ORB[655.40,664.30] vol=2.5x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-06-18 09:35:00 | 662.92 | 660.77 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2025-06-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-24 09:30:00 | 676.40 | 671.49 | 0.00 | ORB-long ORB[667.50,672.60] vol=3.0x ATR=2.54 |
| Stop hit — per-position SL triggered | 2025-06-24 09:35:00 | 673.86 | 673.13 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2025-06-26 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-26 10:40:00 | 677.95 | 679.05 | 0.00 | ORB-short ORB[680.00,684.00] vol=1.7x ATR=1.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-26 10:50:00 | 675.86 | 678.93 | 0.00 | T1 1.5R @ 675.86 |
| Stop hit — per-position SL triggered | 2025-06-26 13:45:00 | 677.95 | 678.19 | 0.00 | SL hit |

### Cycle 13 — BUY (started 2025-06-27 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-27 09:30:00 | 684.20 | 681.23 | 0.00 | ORB-long ORB[678.90,682.50] vol=2.4x ATR=1.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-27 09:35:00 | 686.86 | 682.49 | 0.00 | T1 1.5R @ 686.86 |
| Stop hit — per-position SL triggered | 2025-06-27 09:40:00 | 684.20 | 682.49 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2025-06-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 11:10:00 | 686.80 | 685.57 | 0.00 | ORB-long ORB[681.25,686.00] vol=9.0x ATR=1.93 |
| Stop hit — per-position SL triggered | 2025-06-30 11:15:00 | 684.87 | 685.58 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-07-01 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-01 09:45:00 | 692.50 | 689.64 | 0.00 | ORB-long ORB[686.00,689.90] vol=4.9x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-01 09:55:00 | 695.18 | 691.16 | 0.00 | T1 1.5R @ 695.18 |
| Stop hit — per-position SL triggered | 2025-07-01 10:00:00 | 692.50 | 691.22 | 0.00 | SL hit |

### Cycle 16 — SELL (started 2025-07-02 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-02 10:10:00 | 679.90 | 685.25 | 0.00 | ORB-short ORB[683.00,689.85] vol=1.5x ATR=2.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-02 10:50:00 | 676.85 | 683.80 | 0.00 | T1 1.5R @ 676.85 |
| Stop hit — per-position SL triggered | 2025-07-02 11:05:00 | 679.90 | 683.44 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2025-07-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-07 10:45:00 | 670.85 | 676.29 | 0.00 | ORB-short ORB[671.85,681.70] vol=5.6x ATR=1.88 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-07 11:50:00 | 668.03 | 674.86 | 0.00 | T1 1.5R @ 668.03 |
| Target hit | 2025-07-07 15:20:00 | 666.85 | 672.26 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 18 — SELL (started 2025-07-08 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-08 11:05:00 | 665.45 | 669.48 | 0.00 | ORB-short ORB[668.05,673.50] vol=1.9x ATR=1.19 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 12:20:00 | 663.66 | 668.56 | 0.00 | T1 1.5R @ 663.66 |
| Stop hit — per-position SL triggered | 2025-07-08 12:45:00 | 665.45 | 668.39 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2025-07-09 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-09 09:40:00 | 662.65 | 666.68 | 0.00 | ORB-short ORB[666.00,671.10] vol=1.9x ATR=1.65 |
| Stop hit — per-position SL triggered | 2025-07-09 09:45:00 | 664.30 | 666.43 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-07-11 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-11 11:10:00 | 659.40 | 663.36 | 0.00 | ORB-short ORB[665.10,668.60] vol=2.9x ATR=1.22 |
| Stop hit — per-position SL triggered | 2025-07-11 11:30:00 | 660.62 | 663.13 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-17 11:15:00 | 659.35 | 661.23 | 0.00 | ORB-short ORB[659.40,663.75] vol=3.5x ATR=1.11 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-17 12:25:00 | 657.68 | 660.75 | 0.00 | T1 1.5R @ 657.68 |
| Stop hit — per-position SL triggered | 2025-07-17 14:40:00 | 659.35 | 660.35 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2025-07-18 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-18 10:00:00 | 662.50 | 660.25 | 0.00 | ORB-long ORB[658.05,661.95] vol=1.6x ATR=1.21 |
| Stop hit — per-position SL triggered | 2025-07-18 10:05:00 | 661.29 | 660.53 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2025-07-21 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-21 10:40:00 | 655.00 | 651.81 | 0.00 | ORB-long ORB[647.60,654.10] vol=2.1x ATR=1.24 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-21 10:45:00 | 656.86 | 652.01 | 0.00 | T1 1.5R @ 656.86 |
| Stop hit — per-position SL triggered | 2025-07-21 10:50:00 | 655.00 | 652.05 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-13 11:15:00 | 690.45 | 692.32 | 0.00 | ORB-short ORB[690.55,694.90] vol=2.0x ATR=1.34 |
| Stop hit — per-position SL triggered | 2025-08-13 11:20:00 | 691.79 | 692.29 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2025-08-18 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-18 10:25:00 | 690.80 | 693.17 | 0.00 | ORB-short ORB[691.10,700.00] vol=1.5x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-18 12:05:00 | 688.16 | 692.29 | 0.00 | T1 1.5R @ 688.16 |
| Target hit | 2025-08-18 15:20:00 | 689.65 | 690.79 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 26 — BUY (started 2025-08-25 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-25 09:50:00 | 701.05 | 697.88 | 0.00 | ORB-long ORB[692.05,699.40] vol=3.0x ATR=1.68 |
| Stop hit — per-position SL triggered | 2025-08-25 09:55:00 | 699.37 | 698.07 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-08-28 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-28 09:30:00 | 680.85 | 683.02 | 0.00 | ORB-short ORB[682.00,690.05] vol=4.0x ATR=2.06 |
| Stop hit — per-position SL triggered | 2025-08-28 09:35:00 | 682.91 | 682.87 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-09-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 10:45:00 | 677.80 | 681.21 | 0.00 | ORB-short ORB[680.30,686.00] vol=2.4x ATR=1.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-04 12:20:00 | 675.81 | 679.78 | 0.00 | T1 1.5R @ 675.81 |
| Stop hit — per-position SL triggered | 2025-09-04 12:45:00 | 677.80 | 679.63 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-09-05 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-05 09:55:00 | 678.00 | 675.54 | 0.00 | ORB-long ORB[671.15,677.85] vol=2.9x ATR=1.57 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:00:00 | 680.36 | 676.49 | 0.00 | T1 1.5R @ 680.36 |
| Stop hit — per-position SL triggered | 2025-09-05 10:10:00 | 678.00 | 676.73 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-09-08 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-08 09:40:00 | 679.15 | 676.70 | 0.00 | ORB-long ORB[674.10,678.65] vol=1.6x ATR=1.51 |
| Stop hit — per-position SL triggered | 2025-09-08 10:50:00 | 677.64 | 678.48 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2025-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-10 09:35:00 | 678.70 | 675.25 | 0.00 | ORB-long ORB[669.95,676.30] vol=2.5x ATR=1.76 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-10 09:40:00 | 681.33 | 676.22 | 0.00 | T1 1.5R @ 681.33 |
| Stop hit — per-position SL triggered | 2025-09-10 09:45:00 | 678.70 | 676.25 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2025-09-12 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-12 11:10:00 | 688.00 | 693.72 | 0.00 | ORB-short ORB[690.10,698.00] vol=2.3x ATR=1.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-12 13:05:00 | 685.32 | 692.72 | 0.00 | T1 1.5R @ 685.32 |
| Stop hit — per-position SL triggered | 2025-09-12 15:15:00 | 688.00 | 691.20 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2025-09-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-16 09:30:00 | 692.40 | 690.22 | 0.00 | ORB-long ORB[687.55,692.00] vol=1.5x ATR=1.70 |
| Stop hit — per-position SL triggered | 2025-09-16 09:40:00 | 690.70 | 690.59 | 0.00 | SL hit |

### Cycle 34 — SELL (started 2025-10-30 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-30 10:55:00 | 804.00 | 809.21 | 0.00 | ORB-short ORB[808.00,818.00] vol=2.3x ATR=2.13 |
| Stop hit — per-position SL triggered | 2025-10-30 11:25:00 | 806.13 | 808.89 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2025-10-31 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-31 09:45:00 | 806.00 | 799.58 | 0.00 | ORB-long ORB[795.00,805.00] vol=1.8x ATR=2.93 |
| Stop hit — per-position SL triggered | 2025-10-31 09:55:00 | 803.07 | 801.66 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2025-11-04 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-04 09:45:00 | 789.00 | 793.41 | 0.00 | ORB-short ORB[793.50,799.00] vol=1.8x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 10:00:00 | 785.97 | 792.25 | 0.00 | T1 1.5R @ 785.97 |
| Stop hit — per-position SL triggered | 2025-11-04 15:05:00 | 789.00 | 787.37 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2025-11-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:30:00 | 770.35 | 774.72 | 0.00 | ORB-short ORB[771.40,780.00] vol=1.7x ATR=3.06 |
| Stop hit — per-position SL triggered | 2025-11-11 10:05:00 | 773.41 | 772.08 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2025-11-14 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-14 10:10:00 | 772.15 | 777.31 | 0.00 | ORB-short ORB[773.60,780.85] vol=1.7x ATR=2.65 |
| Stop hit — per-position SL triggered | 2025-11-14 10:20:00 | 774.80 | 776.97 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2025-11-17 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-17 10:35:00 | 777.45 | 779.19 | 0.00 | ORB-short ORB[778.00,783.50] vol=1.5x ATR=1.53 |
| Stop hit — per-position SL triggered | 2025-11-17 11:20:00 | 778.98 | 779.00 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2025-11-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 10:00:00 | 753.05 | 754.68 | 0.00 | ORB-short ORB[754.00,762.00] vol=1.9x ATR=1.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 10:15:00 | 751.14 | 753.94 | 0.00 | T1 1.5R @ 751.14 |
| Target hit | 2025-11-21 14:15:00 | 752.55 | 752.39 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — SELL (started 2025-11-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-24 09:30:00 | 743.50 | 746.67 | 0.00 | ORB-short ORB[744.40,753.10] vol=1.7x ATR=1.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 09:35:00 | 740.95 | 744.91 | 0.00 | T1 1.5R @ 740.95 |
| Stop hit — per-position SL triggered | 2025-11-24 10:20:00 | 743.50 | 742.40 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2025-12-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-01 09:35:00 | 758.20 | 748.86 | 0.00 | ORB-long ORB[742.20,748.70] vol=4.0x ATR=3.38 |
| Stop hit — per-position SL triggered | 2025-12-01 09:45:00 | 754.82 | 751.21 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2025-12-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 09:30:00 | 735.05 | 738.06 | 0.00 | ORB-short ORB[738.00,744.80] vol=5.8x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-12-02 10:30:00 | 737.22 | 736.99 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2025-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:30:00 | 707.05 | 709.98 | 0.00 | ORB-short ORB[708.50,713.65] vol=1.5x ATR=1.98 |
| Stop hit — per-position SL triggered | 2025-12-05 10:00:00 | 709.03 | 708.86 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-12-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-10 09:30:00 | 724.55 | 719.93 | 0.00 | ORB-long ORB[715.00,722.45] vol=3.0x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-10 09:35:00 | 728.74 | 722.04 | 0.00 | T1 1.5R @ 728.74 |
| Stop hit — per-position SL triggered | 2025-12-10 09:40:00 | 724.55 | 722.57 | 0.00 | SL hit |

### Cycle 46 — SELL (started 2025-12-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 09:30:00 | 715.85 | 719.17 | 0.00 | ORB-short ORB[717.00,727.60] vol=1.8x ATR=2.17 |
| Stop hit — per-position SL triggered | 2025-12-16 09:40:00 | 718.02 | 718.86 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-12-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-17 10:45:00 | 708.95 | 711.07 | 0.00 | ORB-short ORB[710.00,716.15] vol=1.9x ATR=1.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-17 11:35:00 | 706.63 | 710.22 | 0.00 | T1 1.5R @ 706.63 |
| Stop hit — per-position SL triggered | 2025-12-17 15:05:00 | 708.95 | 708.51 | 0.00 | SL hit |

### Cycle 48 — SELL (started 2025-12-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:35:00 | 700.00 | 703.27 | 0.00 | ORB-short ORB[702.15,709.00] vol=1.6x ATR=2.40 |
| Stop hit — per-position SL triggered | 2025-12-18 11:35:00 | 702.40 | 701.19 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-12-23 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-23 10:10:00 | 724.95 | 721.38 | 0.00 | ORB-long ORB[715.85,721.60] vol=3.1x ATR=2.03 |
| Stop hit — per-position SL triggered | 2025-12-23 10:15:00 | 722.92 | 721.43 | 0.00 | SL hit |

### Cycle 50 — SELL (started 2025-12-26 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-26 11:10:00 | 707.00 | 709.47 | 0.00 | ORB-short ORB[707.35,712.85] vol=3.8x ATR=1.41 |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 708.41 | 709.41 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2025-12-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-29 09:30:00 | 712.90 | 710.00 | 0.00 | ORB-long ORB[704.90,712.80] vol=1.8x ATR=2.21 |
| Stop hit — per-position SL triggered | 2025-12-29 09:40:00 | 710.69 | 710.61 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2025-12-31 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 10:35:00 | 709.15 | 700.30 | 0.00 | ORB-long ORB[691.00,700.00] vol=5.9x ATR=2.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-31 10:40:00 | 712.61 | 702.87 | 0.00 | T1 1.5R @ 712.61 |
| Stop hit — per-position SL triggered | 2025-12-31 10:45:00 | 709.15 | 703.42 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2026-01-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:55:00 | 701.40 | 696.87 | 0.00 | ORB-long ORB[693.55,699.00] vol=2.0x ATR=1.60 |
| Stop hit — per-position SL triggered | 2026-01-02 11:05:00 | 699.80 | 697.29 | 0.00 | SL hit |

### Cycle 54 — SELL (started 2026-01-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-05 09:30:00 | 697.20 | 701.71 | 0.00 | ORB-short ORB[700.20,705.90] vol=1.5x ATR=1.87 |
| Stop hit — per-position SL triggered | 2026-01-05 09:50:00 | 699.07 | 699.71 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2026-01-08 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-08 10:50:00 | 693.05 | 699.55 | 0.00 | ORB-short ORB[698.95,707.45] vol=1.6x ATR=2.02 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 11:30:00 | 690.02 | 698.09 | 0.00 | T1 1.5R @ 690.02 |
| Target hit | 2026-01-08 15:20:00 | 685.45 | 693.48 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 56 — SELL (started 2026-01-14 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-14 09:45:00 | 664.95 | 668.98 | 0.00 | ORB-short ORB[666.50,671.75] vol=1.5x ATR=2.35 |
| Stop hit — per-position SL triggered | 2026-01-14 09:50:00 | 667.30 | 668.70 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2026-01-16 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-16 10:30:00 | 671.60 | 668.08 | 0.00 | ORB-long ORB[663.50,670.00] vol=1.9x ATR=1.89 |
| Stop hit — per-position SL triggered | 2026-01-16 10:50:00 | 669.71 | 668.54 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2026-01-20 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-20 09:35:00 | 643.00 | 646.07 | 0.00 | ORB-short ORB[645.00,653.10] vol=2.3x ATR=2.13 |
| Stop hit — per-position SL triggered | 2026-01-20 09:40:00 | 645.13 | 645.47 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2026-01-29 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 09:55:00 | 611.90 | 616.84 | 0.00 | ORB-short ORB[617.95,626.45] vol=2.1x ATR=2.59 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 10:05:00 | 608.01 | 615.65 | 0.00 | T1 1.5R @ 608.01 |
| Stop hit — per-position SL triggered | 2026-01-29 11:40:00 | 611.90 | 611.82 | 0.00 | SL hit |

### Cycle 60 — BUY (started 2026-01-30 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-30 09:30:00 | 625.70 | 614.19 | 0.00 | ORB-long ORB[607.00,615.15] vol=3.3x ATR=3.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:35:00 | 630.73 | 619.38 | 0.00 | T1 1.5R @ 630.73 |
| Stop hit — per-position SL triggered | 2026-01-30 09:40:00 | 625.70 | 620.06 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2026-02-16 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-16 09:30:00 | 634.95 | 629.68 | 0.00 | ORB-long ORB[622.65,631.15] vol=3.0x ATR=2.79 |
| Stop hit — per-position SL triggered | 2026-02-16 09:40:00 | 632.16 | 630.75 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 634.20 | 630.86 | 0.00 | ORB-long ORB[628.00,633.00] vol=1.7x ATR=1.97 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 10:05:00 | 637.16 | 632.49 | 0.00 | T1 1.5R @ 637.16 |
| Stop hit — per-position SL triggered | 2026-02-17 10:40:00 | 634.20 | 633.19 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2026-03-06 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-06 11:05:00 | 646.20 | 652.48 | 0.00 | ORB-short ORB[650.05,658.00] vol=3.6x ATR=2.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-06 15:05:00 | 642.41 | 649.95 | 0.00 | T1 1.5R @ 642.41 |
| Target hit | 2026-03-06 15:20:00 | 640.55 | 649.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 64 — BUY (started 2026-04-27 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:40:00 | 713.80 | 709.83 | 0.00 | ORB-long ORB[705.45,713.25] vol=1.8x ATR=2.77 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-27 09:50:00 | 717.96 | 711.34 | 0.00 | T1 1.5R @ 717.96 |
| Stop hit — per-position SL triggered | 2026-04-27 10:00:00 | 713.80 | 711.74 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2026-04-28 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-28 10:50:00 | 724.50 | 717.82 | 0.00 | ORB-long ORB[711.05,717.65] vol=9.0x ATR=3.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:55:00 | 729.09 | 720.04 | 0.00 | T1 1.5R @ 729.09 |
| Stop hit — per-position SL triggered | 2026-04-28 11:00:00 | 724.50 | 722.52 | 0.00 | SL hit |

### Cycle 66 — BUY (started 2026-05-05 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-05 09:40:00 | 727.75 | 719.31 | 0.00 | ORB-long ORB[710.05,716.70] vol=7.4x ATR=3.30 |
| Stop hit — per-position SL triggered | 2026-05-05 09:45:00 | 724.45 | 721.28 | 0.00 | SL hit |

### Cycle 67 — SELL (started 2026-05-08 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-05-08 09:45:00 | 724.55 | 728.60 | 0.00 | ORB-short ORB[728.40,735.25] vol=2.2x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-05-08 10:10:00 | 720.55 | 726.49 | 0.00 | T1 1.5R @ 720.55 |
| Target hit | 2026-05-08 15:20:00 | 717.15 | 722.59 | 0.00 | EOD square-off @ 15:20:00 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-13 10:40:00 | 610.20 | 2025-05-13 11:10:00 | 607.94 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-05-14 09:30:00 | 613.05 | 2025-05-14 10:00:00 | 611.04 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2025-05-15 09:40:00 | 618.75 | 2025-05-15 09:45:00 | 616.75 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2025-05-22 09:40:00 | 616.30 | 2025-05-22 10:20:00 | 617.97 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2025-05-23 09:50:00 | 624.35 | 2025-05-23 09:55:00 | 622.35 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-05-28 10:00:00 | 635.70 | 2025-05-28 10:05:00 | 638.76 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2025-05-28 10:00:00 | 635.70 | 2025-05-28 10:10:00 | 635.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-03 11:15:00 | 672.50 | 2025-06-03 12:20:00 | 670.24 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-06-03 11:15:00 | 672.50 | 2025-06-03 15:20:00 | 664.90 | TARGET_HIT | 0.50 | 1.13% |
| SELL | retest1 | 2025-06-04 09:40:00 | 665.35 | 2025-06-04 09:50:00 | 662.00 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2025-06-04 09:40:00 | 665.35 | 2025-06-04 10:55:00 | 665.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-10 10:35:00 | 697.70 | 2025-06-10 10:40:00 | 700.57 | PARTIAL | 0.50 | 0.41% |
| BUY | retest1 | 2025-06-10 10:35:00 | 697.70 | 2025-06-10 10:45:00 | 697.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-18 09:30:00 | 664.95 | 2025-06-18 09:35:00 | 662.92 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-06-24 09:30:00 | 676.40 | 2025-06-24 09:35:00 | 673.86 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-06-26 10:40:00 | 677.95 | 2025-06-26 10:50:00 | 675.86 | PARTIAL | 0.50 | 0.31% |
| SELL | retest1 | 2025-06-26 10:40:00 | 677.95 | 2025-06-26 13:45:00 | 677.95 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-27 09:30:00 | 684.20 | 2025-06-27 09:35:00 | 686.86 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-06-27 09:30:00 | 684.20 | 2025-06-27 09:40:00 | 684.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-06-30 11:10:00 | 686.80 | 2025-06-30 11:15:00 | 684.87 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-07-01 09:45:00 | 692.50 | 2025-07-01 09:55:00 | 695.18 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-07-01 09:45:00 | 692.50 | 2025-07-01 10:00:00 | 692.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-02 10:10:00 | 679.90 | 2025-07-02 10:50:00 | 676.85 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2025-07-02 10:10:00 | 679.90 | 2025-07-02 11:05:00 | 679.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-07 10:45:00 | 670.85 | 2025-07-07 11:50:00 | 668.03 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-07-07 10:45:00 | 670.85 | 2025-07-07 15:20:00 | 666.85 | TARGET_HIT | 0.50 | 0.60% |
| SELL | retest1 | 2025-07-08 11:05:00 | 665.45 | 2025-07-08 12:20:00 | 663.66 | PARTIAL | 0.50 | 0.27% |
| SELL | retest1 | 2025-07-08 11:05:00 | 665.45 | 2025-07-08 12:45:00 | 665.45 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-07-09 09:40:00 | 662.65 | 2025-07-09 09:45:00 | 664.30 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-07-11 11:10:00 | 659.40 | 2025-07-11 11:30:00 | 660.62 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest1 | 2025-07-17 11:15:00 | 659.35 | 2025-07-17 12:25:00 | 657.68 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-07-17 11:15:00 | 659.35 | 2025-07-17 14:40:00 | 659.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-07-18 10:00:00 | 662.50 | 2025-07-18 10:05:00 | 661.29 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest1 | 2025-07-21 10:40:00 | 655.00 | 2025-07-21 10:45:00 | 656.86 | PARTIAL | 0.50 | 0.28% |
| BUY | retest1 | 2025-07-21 10:40:00 | 655.00 | 2025-07-21 10:50:00 | 655.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-08-13 11:15:00 | 690.45 | 2025-08-13 11:20:00 | 691.79 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2025-08-18 10:25:00 | 690.80 | 2025-08-18 12:05:00 | 688.16 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-08-18 10:25:00 | 690.80 | 2025-08-18 15:20:00 | 689.65 | TARGET_HIT | 0.50 | 0.17% |
| BUY | retest1 | 2025-08-25 09:50:00 | 701.05 | 2025-08-25 09:55:00 | 699.37 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2025-08-28 09:30:00 | 680.85 | 2025-08-28 09:35:00 | 682.91 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-09-04 10:45:00 | 677.80 | 2025-09-04 12:20:00 | 675.81 | PARTIAL | 0.50 | 0.29% |
| SELL | retest1 | 2025-09-04 10:45:00 | 677.80 | 2025-09-04 12:45:00 | 677.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-05 09:55:00 | 678.00 | 2025-09-05 10:00:00 | 680.36 | PARTIAL | 0.50 | 0.35% |
| BUY | retest1 | 2025-09-05 09:55:00 | 678.00 | 2025-09-05 10:10:00 | 678.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-08 09:40:00 | 679.15 | 2025-09-08 10:50:00 | 677.64 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-09-10 09:35:00 | 678.70 | 2025-09-10 09:40:00 | 681.33 | PARTIAL | 0.50 | 0.39% |
| BUY | retest1 | 2025-09-10 09:35:00 | 678.70 | 2025-09-10 09:45:00 | 678.70 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-09-12 11:10:00 | 688.00 | 2025-09-12 13:05:00 | 685.32 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-09-12 11:10:00 | 688.00 | 2025-09-12 15:15:00 | 688.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-09-16 09:30:00 | 692.40 | 2025-09-16 09:40:00 | 690.70 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2025-10-30 10:55:00 | 804.00 | 2025-10-30 11:25:00 | 806.13 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-10-31 09:45:00 | 806.00 | 2025-10-31 09:55:00 | 803.07 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2025-11-04 09:45:00 | 789.00 | 2025-11-04 10:00:00 | 785.97 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2025-11-04 09:45:00 | 789.00 | 2025-11-04 15:05:00 | 789.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-11-11 09:30:00 | 770.35 | 2025-11-11 10:05:00 | 773.41 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-11-14 10:10:00 | 772.15 | 2025-11-14 10:20:00 | 774.80 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2025-11-17 10:35:00 | 777.45 | 2025-11-17 11:20:00 | 778.98 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2025-11-21 10:00:00 | 753.05 | 2025-11-21 10:15:00 | 751.14 | PARTIAL | 0.50 | 0.25% |
| SELL | retest1 | 2025-11-21 10:00:00 | 753.05 | 2025-11-21 14:15:00 | 752.55 | TARGET_HIT | 0.50 | 0.07% |
| SELL | retest1 | 2025-11-24 09:30:00 | 743.50 | 2025-11-24 09:35:00 | 740.95 | PARTIAL | 0.50 | 0.34% |
| SELL | retest1 | 2025-11-24 09:30:00 | 743.50 | 2025-11-24 10:20:00 | 743.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-01 09:35:00 | 758.20 | 2025-12-01 09:45:00 | 754.82 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-12-02 09:30:00 | 735.05 | 2025-12-02 10:30:00 | 737.22 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-05 09:30:00 | 707.05 | 2025-12-05 10:00:00 | 709.03 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2025-12-10 09:30:00 | 724.55 | 2025-12-10 09:35:00 | 728.74 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-12-10 09:30:00 | 724.55 | 2025-12-10 09:40:00 | 724.55 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-16 09:30:00 | 715.85 | 2025-12-16 09:40:00 | 718.02 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2025-12-17 10:45:00 | 708.95 | 2025-12-17 11:35:00 | 706.63 | PARTIAL | 0.50 | 0.33% |
| SELL | retest1 | 2025-12-17 10:45:00 | 708.95 | 2025-12-17 15:05:00 | 708.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-18 09:35:00 | 700.00 | 2025-12-18 11:35:00 | 702.40 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2025-12-23 10:10:00 | 724.95 | 2025-12-23 10:15:00 | 722.92 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2025-12-26 11:10:00 | 707.00 | 2025-12-26 11:15:00 | 708.41 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2025-12-29 09:30:00 | 712.90 | 2025-12-29 09:40:00 | 710.69 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2025-12-31 10:35:00 | 709.15 | 2025-12-31 10:40:00 | 712.61 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2025-12-31 10:35:00 | 709.15 | 2025-12-31 10:45:00 | 709.15 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-02 10:55:00 | 701.40 | 2026-01-02 11:05:00 | 699.80 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest1 | 2026-01-05 09:30:00 | 697.20 | 2026-01-05 09:50:00 | 699.07 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2026-01-08 10:50:00 | 693.05 | 2026-01-08 11:30:00 | 690.02 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2026-01-08 10:50:00 | 693.05 | 2026-01-08 15:20:00 | 685.45 | TARGET_HIT | 0.50 | 1.10% |
| SELL | retest1 | 2026-01-14 09:45:00 | 664.95 | 2026-01-14 09:50:00 | 667.30 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2026-01-16 10:30:00 | 671.60 | 2026-01-16 10:50:00 | 669.71 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-01-20 09:35:00 | 643.00 | 2026-01-20 09:40:00 | 645.13 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2026-01-29 09:55:00 | 611.90 | 2026-01-29 10:05:00 | 608.01 | PARTIAL | 0.50 | 0.64% |
| SELL | retest1 | 2026-01-29 09:55:00 | 611.90 | 2026-01-29 11:40:00 | 611.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-30 09:30:00 | 625.70 | 2026-01-30 09:35:00 | 630.73 | PARTIAL | 0.50 | 0.80% |
| BUY | retest1 | 2026-01-30 09:30:00 | 625.70 | 2026-01-30 09:40:00 | 625.70 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-02-16 09:30:00 | 634.95 | 2026-02-16 09:40:00 | 632.16 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2026-02-17 09:40:00 | 634.20 | 2026-02-17 10:05:00 | 637.16 | PARTIAL | 0.50 | 0.47% |
| BUY | retest1 | 2026-02-17 09:40:00 | 634.20 | 2026-02-17 10:40:00 | 634.20 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-03-06 11:05:00 | 646.20 | 2026-03-06 15:05:00 | 642.41 | PARTIAL | 0.50 | 0.59% |
| SELL | retest1 | 2026-03-06 11:05:00 | 646.20 | 2026-03-06 15:20:00 | 640.55 | TARGET_HIT | 0.50 | 0.87% |
| BUY | retest1 | 2026-04-27 09:40:00 | 713.80 | 2026-04-27 09:50:00 | 717.96 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2026-04-27 09:40:00 | 713.80 | 2026-04-27 10:00:00 | 713.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-28 10:50:00 | 724.50 | 2026-04-28 10:55:00 | 729.09 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2026-04-28 10:50:00 | 724.50 | 2026-04-28 11:00:00 | 724.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-05-05 09:40:00 | 727.75 | 2026-05-05 09:45:00 | 724.45 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2026-05-08 09:45:00 | 724.55 | 2026-05-08 10:10:00 | 720.55 | PARTIAL | 0.50 | 0.55% |
| SELL | retest1 | 2026-05-08 09:45:00 | 724.55 | 2026-05-08 15:20:00 | 717.15 | TARGET_HIT | 0.50 | 1.02% |
