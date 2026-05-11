# Syrma SGS Technology Ltd. (SYRMA)

## Backtest Summary

- **Window:** 2025-05-12 09:15:00 → 2026-05-08 15:25:00 (18463 bars)
- **Last close:** 1100.55
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
| ENTRY1 | 42 |
| ENTRY2 | 0 |
| PARTIAL | 15 |
| TARGET_HIT | 8 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 23 / 34
- **Target hits / Stop hits / Partials:** 8 / 34 / 15
- **Avg / median % per leg:** 0.20% / 0.00%
- **Sum % (uncompounded):** 11.51%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 14 | 41.2% | 5 | 20 | 9 | 0.07% | 2.3% |
| BUY @ 2nd Alert (retest1) | 34 | 14 | 41.2% | 5 | 20 | 9 | 0.07% | 2.3% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 23 | 9 | 39.1% | 3 | 14 | 6 | 0.40% | 9.2% |
| SELL @ 2nd Alert (retest1) | 23 | 9 | 39.1% | 3 | 14 | 6 | 0.40% | 9.2% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 57 | 23 | 40.4% | 8 | 34 | 15 | 0.20% | 11.5% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-21 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-21 10:00:00 | 535.30 | 528.71 | 0.00 | ORB-long ORB[520.40,528.00] vol=6.9x ATR=2.84 |
| Stop hit — per-position SL triggered | 2025-05-21 10:05:00 | 532.46 | 529.51 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2025-06-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-03 09:30:00 | 545.90 | 543.24 | 0.00 | ORB-long ORB[537.40,544.90] vol=1.7x ATR=2.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-03 09:35:00 | 549.05 | 544.26 | 0.00 | T1 1.5R @ 549.05 |
| Stop hit — per-position SL triggered | 2025-06-03 09:45:00 | 545.90 | 544.50 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2025-06-04 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-04 09:35:00 | 534.65 | 538.83 | 0.00 | ORB-short ORB[536.55,544.20] vol=2.0x ATR=2.42 |
| Stop hit — per-position SL triggered | 2025-06-04 11:05:00 | 537.07 | 536.26 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2025-06-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-12 10:20:00 | 529.85 | 534.33 | 0.00 | ORB-short ORB[535.80,542.00] vol=4.2x ATR=2.08 |
| Stop hit — per-position SL triggered | 2025-06-12 10:25:00 | 531.93 | 534.22 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2025-06-13 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-13 11:05:00 | 527.40 | 517.77 | 0.00 | ORB-long ORB[511.25,519.00] vol=1.5x ATR=2.38 |
| Stop hit — per-position SL triggered | 2025-06-13 11:10:00 | 525.02 | 518.55 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2025-06-30 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-30 11:10:00 | 566.35 | 561.76 | 0.00 | ORB-long ORB[555.00,563.45] vol=4.5x ATR=2.24 |
| Stop hit — per-position SL triggered | 2025-06-30 11:15:00 | 564.11 | 561.99 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2025-08-12 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-08-12 10:20:00 | 703.55 | 712.39 | 0.00 | ORB-short ORB[715.50,724.85] vol=3.3x ATR=3.23 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-12 11:10:00 | 698.70 | 709.55 | 0.00 | T1 1.5R @ 698.70 |
| Target hit | 2025-08-12 15:20:00 | 675.25 | 678.46 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2025-08-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-19 09:40:00 | 719.80 | 716.96 | 0.00 | ORB-long ORB[713.00,719.00] vol=2.0x ATR=2.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-19 09:50:00 | 723.83 | 718.65 | 0.00 | T1 1.5R @ 723.83 |
| Target hit | 2025-08-19 11:30:00 | 723.10 | 723.21 | 0.00 | Trail-exit close<VWAP |

### Cycle 9 — BUY (started 2025-08-21 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-21 09:55:00 | 735.80 | 729.00 | 0.00 | ORB-long ORB[720.90,728.60] vol=2.7x ATR=2.88 |
| Stop hit — per-position SL triggered | 2025-08-21 10:05:00 | 732.92 | 730.58 | 0.00 | SL hit |

### Cycle 10 — BUY (started 2025-08-29 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-29 10:45:00 | 751.55 | 743.74 | 0.00 | ORB-long ORB[736.30,745.75] vol=2.5x ATR=3.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 10:50:00 | 756.39 | 746.02 | 0.00 | T1 1.5R @ 756.39 |
| Target hit | 2025-08-29 14:40:00 | 753.00 | 753.49 | 0.00 | Trail-exit close<VWAP |

### Cycle 11 — BUY (started 2025-09-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-01 09:35:00 | 769.65 | 761.66 | 0.00 | ORB-long ORB[751.65,762.20] vol=3.5x ATR=3.65 |
| Stop hit — per-position SL triggered | 2025-09-01 09:55:00 | 766.00 | 763.92 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2025-09-12 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:05:00 | 849.25 | 838.48 | 0.00 | ORB-long ORB[830.60,843.05] vol=1.9x ATR=3.98 |
| Stop hit — per-position SL triggered | 2025-09-12 10:10:00 | 845.27 | 839.62 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2025-09-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-17 09:45:00 | 812.45 | 820.00 | 0.00 | ORB-short ORB[818.30,829.00] vol=3.2x ATR=4.09 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-17 10:00:00 | 806.31 | 816.99 | 0.00 | T1 1.5R @ 806.31 |
| Target hit | 2025-09-17 15:20:00 | 798.85 | 805.19 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 14 — BUY (started 2025-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:35:00 | 815.45 | 810.08 | 0.00 | ORB-long ORB[803.45,812.00] vol=1.5x ATR=3.80 |
| Stop hit — per-position SL triggered | 2025-09-19 09:40:00 | 811.65 | 810.40 | 0.00 | SL hit |

### Cycle 15 — BUY (started 2025-09-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-24 09:45:00 | 830.10 | 823.89 | 0.00 | ORB-long ORB[816.00,823.50] vol=3.4x ATR=3.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-24 09:50:00 | 835.60 | 831.63 | 0.00 | T1 1.5R @ 835.60 |
| Target hit | 2025-09-24 10:25:00 | 835.00 | 836.98 | 0.00 | Trail-exit close<VWAP |

### Cycle 16 — SELL (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-30 10:15:00 | 780.15 | 791.23 | 0.00 | ORB-short ORB[792.50,800.95] vol=4.5x ATR=4.24 |
| Stop hit — per-position SL triggered | 2025-09-30 10:30:00 | 784.39 | 788.89 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2025-10-07 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-07 09:40:00 | 834.85 | 831.34 | 0.00 | ORB-long ORB[818.00,830.30] vol=7.3x ATR=3.81 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-07 09:45:00 | 840.56 | 832.60 | 0.00 | T1 1.5R @ 840.56 |
| Stop hit — per-position SL triggered | 2025-10-07 10:10:00 | 834.85 | 835.37 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2025-10-15 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-15 10:25:00 | 794.90 | 800.78 | 0.00 | ORB-short ORB[800.10,807.95] vol=3.2x ATR=3.53 |
| Stop hit — per-position SL triggered | 2025-10-15 10:40:00 | 798.43 | 800.40 | 0.00 | SL hit |

### Cycle 19 — BUY (started 2025-10-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-17 09:40:00 | 804.00 | 797.30 | 0.00 | ORB-long ORB[788.40,799.90] vol=1.8x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-10-17 10:00:00 | 801.02 | 800.39 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2025-11-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-03 09:30:00 | 806.50 | 813.16 | 0.00 | ORB-short ORB[810.00,821.35] vol=1.6x ATR=3.63 |
| Stop hit — per-position SL triggered | 2025-11-03 09:40:00 | 810.13 | 811.24 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2025-11-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-04 10:35:00 | 797.45 | 793.63 | 0.00 | ORB-long ORB[788.45,797.30] vol=2.4x ATR=3.75 |
| Stop hit — per-position SL triggered | 2025-11-04 10:40:00 | 793.70 | 793.74 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2025-11-20 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-20 10:55:00 | 867.60 | 872.92 | 0.00 | ORB-short ORB[872.10,880.00] vol=4.3x ATR=2.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-20 11:05:00 | 864.21 | 870.90 | 0.00 | T1 1.5R @ 864.21 |
| Stop hit — per-position SL triggered | 2025-11-20 11:20:00 | 867.60 | 870.17 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2025-12-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-05 09:30:00 | 781.55 | 784.64 | 0.00 | ORB-short ORB[782.30,788.90] vol=1.9x ATR=2.51 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:35:00 | 777.79 | 781.76 | 0.00 | T1 1.5R @ 777.79 |
| Target hit | 2025-12-05 15:20:00 | 743.80 | 757.49 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 24 — BUY (started 2025-12-09 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-09 11:10:00 | 745.35 | 728.62 | 0.00 | ORB-long ORB[717.80,728.85] vol=2.6x ATR=4.43 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 11:35:00 | 751.99 | 731.90 | 0.00 | T1 1.5R @ 751.99 |
| Target hit | 2025-12-09 15:20:00 | 756.40 | 743.82 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 25 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-11 10:15:00 | 741.45 | 736.71 | 0.00 | ORB-long ORB[729.85,739.45] vol=2.1x ATR=3.46 |
| Stop hit — per-position SL triggered | 2025-12-11 10:20:00 | 737.99 | 736.80 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2025-12-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:30:00 | 750.65 | 744.58 | 0.00 | ORB-long ORB[736.90,748.00] vol=3.3x ATR=3.05 |
| Stop hit — per-position SL triggered | 2025-12-12 09:35:00 | 747.60 | 745.30 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2025-12-16 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-16 10:20:00 | 723.00 | 731.15 | 0.00 | ORB-short ORB[730.80,739.15] vol=2.0x ATR=2.39 |
| Stop hit — per-position SL triggered | 2025-12-16 10:50:00 | 725.39 | 729.76 | 0.00 | SL hit |

### Cycle 28 — SELL (started 2025-12-24 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-24 09:50:00 | 730.00 | 737.12 | 0.00 | ORB-short ORB[730.05,740.50] vol=2.4x ATR=3.19 |
| Stop hit — per-position SL triggered | 2025-12-24 10:20:00 | 733.19 | 734.31 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2025-12-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-30 09:35:00 | 731.50 | 722.27 | 0.00 | ORB-long ORB[711.00,718.90] vol=3.6x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 09:40:00 | 736.24 | 728.23 | 0.00 | T1 1.5R @ 736.24 |
| Stop hit — per-position SL triggered | 2025-12-30 09:45:00 | 731.50 | 729.13 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-31 11:15:00 | 732.80 | 728.31 | 0.00 | ORB-long ORB[722.25,732.50] vol=5.2x ATR=2.19 |
| Stop hit — per-position SL triggered | 2025-12-31 11:20:00 | 730.61 | 728.46 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2026-01-02 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-02 10:05:00 | 729.45 | 723.09 | 0.00 | ORB-long ORB[718.00,724.20] vol=2.8x ATR=2.45 |
| Stop hit — per-position SL triggered | 2026-01-02 10:10:00 | 727.00 | 723.60 | 0.00 | SL hit |

### Cycle 32 — BUY (started 2026-01-07 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-07 10:50:00 | 755.70 | 749.98 | 0.00 | ORB-long ORB[744.75,752.50] vol=1.5x ATR=2.34 |
| Stop hit — per-position SL triggered | 2026-01-07 11:15:00 | 753.36 | 750.97 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2026-01-19 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-19 10:35:00 | 697.35 | 702.86 | 0.00 | ORB-short ORB[700.05,708.40] vol=1.7x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 11:15:00 | 693.17 | 701.52 | 0.00 | T1 1.5R @ 693.17 |
| Stop hit — per-position SL triggered | 2026-01-19 12:00:00 | 697.35 | 700.16 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2026-01-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-23 09:30:00 | 674.50 | 669.42 | 0.00 | ORB-long ORB[660.20,669.70] vol=3.1x ATR=3.21 |
| Stop hit — per-position SL triggered | 2026-01-23 09:40:00 | 671.29 | 670.75 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2026-02-12 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-12 10:55:00 | 876.65 | 882.44 | 0.00 | ORB-short ORB[879.00,890.00] vol=1.5x ATR=2.45 |
| Stop hit — per-position SL triggered | 2026-02-12 11:00:00 | 879.10 | 882.26 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2026-02-13 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-13 09:30:00 | 853.75 | 859.72 | 0.00 | ORB-short ORB[856.00,867.55] vol=2.0x ATR=3.59 |
| Stop hit — per-position SL triggered | 2026-02-13 09:40:00 | 857.34 | 858.43 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2026-02-17 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-17 09:40:00 | 867.40 | 862.23 | 0.00 | ORB-long ORB[856.05,864.50] vol=1.6x ATR=3.14 |
| Stop hit — per-position SL triggered | 2026-02-17 10:00:00 | 864.26 | 863.46 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2026-02-18 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-18 09:50:00 | 858.25 | 863.41 | 0.00 | ORB-short ORB[859.10,869.90] vol=2.7x ATR=2.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-18 10:40:00 | 854.38 | 861.05 | 0.00 | T1 1.5R @ 854.38 |
| Stop hit — per-position SL triggered | 2026-02-18 11:15:00 | 858.25 | 860.54 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2026-02-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:30:00 | 826.70 | 832.56 | 0.00 | ORB-short ORB[830.95,843.15] vol=1.9x ATR=3.86 |
| Stop hit — per-position SL triggered | 2026-02-24 10:20:00 | 830.56 | 830.10 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2026-04-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-07 10:30:00 | 811.20 | 806.71 | 0.00 | ORB-long ORB[800.00,810.50] vol=1.9x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-07 11:55:00 | 816.19 | 808.68 | 0.00 | T1 1.5R @ 816.19 |
| Stop hit — per-position SL triggered | 2026-04-07 14:10:00 | 811.20 | 810.01 | 0.00 | SL hit |

### Cycle 41 — BUY (started 2026-04-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-15 09:55:00 | 887.25 | 881.07 | 0.00 | ORB-long ORB[875.10,884.00] vol=1.9x ATR=4.04 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-15 10:55:00 | 893.32 | 884.89 | 0.00 | T1 1.5R @ 893.32 |
| Target hit | 2026-04-15 12:35:00 | 890.00 | 892.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 42 — SELL (started 2026-04-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-24 09:35:00 | 984.10 | 990.69 | 0.00 | ORB-short ORB[986.00,999.00] vol=1.9x ATR=4.84 |
| Stop hit — per-position SL triggered | 2026-04-24 09:40:00 | 988.94 | 990.44 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-21 10:00:00 | 535.30 | 2025-05-21 10:05:00 | 532.46 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2025-06-03 09:30:00 | 545.90 | 2025-06-03 09:35:00 | 549.05 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2025-06-03 09:30:00 | 545.90 | 2025-06-03 09:45:00 | 545.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-06-04 09:35:00 | 534.65 | 2025-06-04 11:05:00 | 537.07 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest1 | 2025-06-12 10:20:00 | 529.85 | 2025-06-12 10:25:00 | 531.93 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-06-13 11:05:00 | 527.40 | 2025-06-13 11:10:00 | 525.02 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-06-30 11:10:00 | 566.35 | 2025-06-30 11:15:00 | 564.11 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-08-12 10:20:00 | 703.55 | 2025-08-12 11:10:00 | 698.70 | PARTIAL | 0.50 | 0.69% |
| SELL | retest1 | 2025-08-12 10:20:00 | 703.55 | 2025-08-12 15:20:00 | 675.25 | TARGET_HIT | 0.50 | 4.02% |
| BUY | retest1 | 2025-08-19 09:40:00 | 719.80 | 2025-08-19 09:50:00 | 723.83 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2025-08-19 09:40:00 | 719.80 | 2025-08-19 11:30:00 | 723.10 | TARGET_HIT | 0.50 | 0.46% |
| BUY | retest1 | 2025-08-21 09:55:00 | 735.80 | 2025-08-21 10:05:00 | 732.92 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-08-29 10:45:00 | 751.55 | 2025-08-29 10:50:00 | 756.39 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2025-08-29 10:45:00 | 751.55 | 2025-08-29 14:40:00 | 753.00 | TARGET_HIT | 0.50 | 0.19% |
| BUY | retest1 | 2025-09-01 09:35:00 | 769.65 | 2025-09-01 09:55:00 | 766.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-09-12 10:05:00 | 849.25 | 2025-09-12 10:10:00 | 845.27 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-09-17 09:45:00 | 812.45 | 2025-09-17 10:00:00 | 806.31 | PARTIAL | 0.50 | 0.76% |
| SELL | retest1 | 2025-09-17 09:45:00 | 812.45 | 2025-09-17 15:20:00 | 798.85 | TARGET_HIT | 0.50 | 1.67% |
| BUY | retest1 | 2025-09-19 09:35:00 | 815.45 | 2025-09-19 09:40:00 | 811.65 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-09-24 09:45:00 | 830.10 | 2025-09-24 09:50:00 | 835.60 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2025-09-24 09:45:00 | 830.10 | 2025-09-24 10:25:00 | 835.00 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2025-09-30 10:15:00 | 780.15 | 2025-09-30 10:30:00 | 784.39 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest1 | 2025-10-07 09:40:00 | 834.85 | 2025-10-07 09:45:00 | 840.56 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2025-10-07 09:40:00 | 834.85 | 2025-10-07 10:10:00 | 834.85 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-10-15 10:25:00 | 794.90 | 2025-10-15 10:40:00 | 798.43 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-10-17 09:40:00 | 804.00 | 2025-10-17 10:00:00 | 801.02 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2025-11-03 09:30:00 | 806.50 | 2025-11-03 09:40:00 | 810.13 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest1 | 2025-11-04 10:35:00 | 797.45 | 2025-11-04 10:40:00 | 793.70 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest1 | 2025-11-20 10:55:00 | 867.60 | 2025-11-20 11:05:00 | 864.21 | PARTIAL | 0.50 | 0.39% |
| SELL | retest1 | 2025-11-20 10:55:00 | 867.60 | 2025-11-20 11:20:00 | 867.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-12-05 09:30:00 | 781.55 | 2025-12-05 09:35:00 | 777.79 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2025-12-05 09:30:00 | 781.55 | 2025-12-05 15:20:00 | 743.80 | TARGET_HIT | 0.50 | 4.83% |
| BUY | retest1 | 2025-12-09 11:10:00 | 745.35 | 2025-12-09 11:35:00 | 751.99 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2025-12-09 11:10:00 | 745.35 | 2025-12-09 15:20:00 | 756.40 | TARGET_HIT | 0.50 | 1.48% |
| BUY | retest1 | 2025-12-11 10:15:00 | 741.45 | 2025-12-11 10:20:00 | 737.99 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2025-12-12 09:30:00 | 750.65 | 2025-12-12 09:35:00 | 747.60 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2025-12-16 10:20:00 | 723.00 | 2025-12-16 10:50:00 | 725.39 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2025-12-24 09:50:00 | 730.00 | 2025-12-24 10:20:00 | 733.19 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2025-12-30 09:35:00 | 731.50 | 2025-12-30 09:40:00 | 736.24 | PARTIAL | 0.50 | 0.65% |
| BUY | retest1 | 2025-12-30 09:35:00 | 731.50 | 2025-12-30 09:45:00 | 731.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-12-31 11:15:00 | 732.80 | 2025-12-31 11:20:00 | 730.61 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2026-01-02 10:05:00 | 729.45 | 2026-01-02 10:10:00 | 727.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2026-01-07 10:50:00 | 755.70 | 2026-01-07 11:15:00 | 753.36 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2026-01-19 10:35:00 | 697.35 | 2026-01-19 11:15:00 | 693.17 | PARTIAL | 0.50 | 0.60% |
| SELL | retest1 | 2026-01-19 10:35:00 | 697.35 | 2026-01-19 12:00:00 | 697.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-01-23 09:30:00 | 674.50 | 2026-01-23 09:40:00 | 671.29 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2026-02-12 10:55:00 | 876.65 | 2026-02-12 11:00:00 | 879.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2026-02-13 09:30:00 | 853.75 | 2026-02-13 09:40:00 | 857.34 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2026-02-17 09:40:00 | 867.40 | 2026-02-17 10:00:00 | 864.26 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2026-02-18 09:50:00 | 858.25 | 2026-02-18 10:40:00 | 854.38 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2026-02-18 09:50:00 | 858.25 | 2026-02-18 11:15:00 | 858.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2026-02-24 09:30:00 | 826.70 | 2026-02-24 10:20:00 | 830.56 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2026-04-07 10:30:00 | 811.20 | 2026-04-07 11:55:00 | 816.19 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2026-04-07 10:30:00 | 811.20 | 2026-04-07 14:10:00 | 811.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2026-04-15 09:55:00 | 887.25 | 2026-04-15 10:55:00 | 893.32 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2026-04-15 09:55:00 | 887.25 | 2026-04-15 12:35:00 | 890.00 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2026-04-24 09:35:00 | 984.10 | 2026-04-24 09:40:00 | 988.94 | STOP_HIT | 1.00 | -0.49% |
