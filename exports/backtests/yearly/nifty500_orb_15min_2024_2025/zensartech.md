# Zensar Technolgies Ltd. (ZENSARTECH)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2026-05-08 15:25:00 (36871 bars)
- **Last close:** 525.00
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
| ENTRY1 | 54 |
| ENTRY2 | 0 |
| PARTIAL | 28 |
| TARGET_HIT | 15 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 82 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 40
- **Target hits / Stop hits / Partials:** 15 / 39 / 28
- **Avg / median % per leg:** 0.27% / 0.12%
- **Sum % (uncompounded):** 22.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 27 | 50.9% | 11 | 25 | 17 | 0.28% | 15.0% |
| BUY @ 2nd Alert (retest1) | 53 | 27 | 50.9% | 11 | 25 | 17 | 0.28% | 15.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 29 | 15 | 51.7% | 4 | 14 | 11 | 0.25% | 7.3% |
| SELL @ 2nd Alert (retest1) | 29 | 15 | 51.7% | 4 | 14 | 11 | 0.25% | 7.3% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 82 | 42 | 51.2% | 15 | 39 | 28 | 0.27% | 22.3% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-22 10:15:00 | 604.05 | 607.46 | 0.00 | ORB-short ORB[606.40,614.80] vol=7.8x ATR=3.01 |
| Stop hit — per-position SL triggered | 2024-05-22 10:20:00 | 607.06 | 607.13 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2024-05-23 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-23 10:05:00 | 625.05 | 623.85 | 0.00 | ORB-long ORB[619.90,623.65] vol=1.5x ATR=2.34 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-23 10:50:00 | 628.56 | 625.37 | 0.00 | T1 1.5R @ 628.56 |
| Target hit | 2024-05-23 12:30:00 | 625.80 | 625.86 | 0.00 | Trail-exit close<VWAP |

### Cycle 3 — BUY (started 2024-05-28 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-28 09:45:00 | 642.45 | 633.45 | 0.00 | ORB-long ORB[621.65,628.40] vol=9.9x ATR=2.74 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-28 09:50:00 | 646.56 | 641.16 | 0.00 | T1 1.5R @ 646.56 |
| Stop hit — per-position SL triggered | 2024-05-28 09:55:00 | 642.45 | 641.71 | 0.00 | SL hit |

### Cycle 4 — BUY (started 2024-06-11 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:10:00 | 690.45 | 682.91 | 0.00 | ORB-long ORB[679.35,689.25] vol=1.7x ATR=3.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-11 10:20:00 | 695.67 | 688.58 | 0.00 | T1 1.5R @ 695.67 |
| Target hit | 2024-06-11 14:00:00 | 701.85 | 704.79 | 0.00 | Trail-exit close<VWAP |

### Cycle 5 — BUY (started 2024-06-13 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:35:00 | 709.80 | 703.80 | 0.00 | ORB-long ORB[698.10,707.10] vol=2.5x ATR=4.01 |
| Stop hit — per-position SL triggered | 2024-06-13 09:45:00 | 705.79 | 704.86 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2024-06-20 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-20 11:00:00 | 721.05 | 715.58 | 0.00 | ORB-long ORB[705.00,715.55] vol=2.1x ATR=2.46 |
| Target hit | 2024-06-20 15:20:00 | 721.10 | 718.78 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 7 — BUY (started 2024-06-25 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-25 09:45:00 | 758.90 | 755.17 | 0.00 | ORB-long ORB[751.60,757.70] vol=1.5x ATR=2.68 |
| Stop hit — per-position SL triggered | 2024-06-25 10:10:00 | 756.22 | 756.44 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2024-06-27 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:55:00 | 747.50 | 751.22 | 0.00 | ORB-short ORB[750.50,754.60] vol=7.2x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-06-27 11:20:00 | 749.67 | 750.94 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-07-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-01 09:30:00 | 748.30 | 744.94 | 0.00 | ORB-long ORB[738.00,746.50] vol=3.1x ATR=3.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-01 09:35:00 | 753.38 | 746.36 | 0.00 | T1 1.5R @ 753.38 |
| Target hit | 2024-07-01 14:30:00 | 754.30 | 755.33 | 0.00 | Trail-exit close<VWAP |

### Cycle 10 — BUY (started 2024-07-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-02 10:20:00 | 763.60 | 755.35 | 0.00 | ORB-long ORB[751.25,762.00] vol=2.5x ATR=3.69 |
| Stop hit — per-position SL triggered | 2024-07-02 10:45:00 | 759.91 | 758.50 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2024-07-04 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-04 09:40:00 | 755.75 | 749.46 | 0.00 | ORB-long ORB[743.00,753.00] vol=2.2x ATR=2.65 |
| Stop hit — per-position SL triggered | 2024-07-04 09:45:00 | 753.10 | 750.06 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-09 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-09 10:00:00 | 743.60 | 737.06 | 0.00 | ORB-long ORB[734.00,737.30] vol=1.7x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-09 10:10:00 | 747.00 | 739.56 | 0.00 | T1 1.5R @ 747.00 |
| Stop hit — per-position SL triggered | 2024-07-09 10:15:00 | 743.60 | 739.69 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2024-07-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-10 10:00:00 | 718.65 | 723.25 | 0.00 | ORB-short ORB[722.05,729.95] vol=1.9x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-07-10 10:25:00 | 721.17 | 721.56 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-26 09:30:00 | 797.80 | 789.98 | 0.00 | ORB-long ORB[773.60,782.90] vol=10.5x ATR=3.86 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-26 09:35:00 | 803.59 | 796.30 | 0.00 | T1 1.5R @ 803.59 |
| Target hit | 2024-07-26 11:10:00 | 812.95 | 813.13 | 0.00 | Trail-exit close<VWAP |

### Cycle 15 — BUY (started 2024-08-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-02 09:45:00 | 767.50 | 762.61 | 0.00 | ORB-long ORB[756.55,766.40] vol=9.9x ATR=5.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-02 11:20:00 | 775.45 | 764.56 | 0.00 | T1 1.5R @ 775.45 |
| Stop hit — per-position SL triggered | 2024-08-02 12:40:00 | 767.50 | 765.68 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-08-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:30:00 | 766.50 | 761.70 | 0.00 | ORB-long ORB[757.55,765.65] vol=2.1x ATR=3.63 |
| Stop hit — per-position SL triggered | 2024-08-09 09:55:00 | 762.87 | 762.52 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 10:15:00 | 793.55 | 787.38 | 0.00 | ORB-long ORB[783.30,790.00] vol=4.0x ATR=3.13 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 10:25:00 | 798.25 | 792.65 | 0.00 | T1 1.5R @ 798.25 |
| Target hit | 2024-08-20 11:00:00 | 800.95 | 801.80 | 0.00 | Trail-exit close<VWAP |

### Cycle 18 — BUY (started 2024-08-21 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-21 09:45:00 | 802.20 | 795.97 | 0.00 | ORB-long ORB[788.50,796.75] vol=2.1x ATR=2.86 |
| Stop hit — per-position SL triggered | 2024-08-21 10:05:00 | 799.34 | 798.50 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2024-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-23 09:30:00 | 778.50 | 782.58 | 0.00 | ORB-short ORB[780.90,787.50] vol=1.5x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-23 10:20:00 | 774.30 | 779.12 | 0.00 | T1 1.5R @ 774.30 |
| Stop hit — per-position SL triggered | 2024-08-23 10:35:00 | 778.50 | 778.91 | 0.00 | SL hit |

### Cycle 20 — BUY (started 2024-08-27 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-27 09:45:00 | 776.55 | 770.74 | 0.00 | ORB-long ORB[766.25,774.40] vol=1.7x ATR=2.70 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-27 09:50:00 | 780.60 | 774.33 | 0.00 | T1 1.5R @ 780.60 |
| Stop hit — per-position SL triggered | 2024-08-27 10:20:00 | 776.55 | 776.05 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-09-03 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-03 10:10:00 | 796.90 | 790.62 | 0.00 | ORB-long ORB[783.95,793.85] vol=3.7x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-03 10:15:00 | 801.11 | 794.17 | 0.00 | T1 1.5R @ 801.11 |
| Stop hit — per-position SL triggered | 2024-09-03 10:25:00 | 796.90 | 794.74 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-09-04 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-04 10:00:00 | 795.20 | 788.45 | 0.00 | ORB-long ORB[781.00,787.25] vol=1.8x ATR=2.90 |
| Stop hit — per-position SL triggered | 2024-09-04 10:20:00 | 792.30 | 789.23 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 802.00 | 797.90 | 0.00 | ORB-long ORB[790.10,800.60] vol=4.5x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 10:00:00 | 806.76 | 801.46 | 0.00 | T1 1.5R @ 806.76 |
| Target hit | 2024-09-11 10:00:00 | 799.35 | 801.46 | 0.00 | Trail-exit close<VWAP |

### Cycle 24 — SELL (started 2024-09-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-18 09:35:00 | 760.90 | 763.90 | 0.00 | ORB-short ORB[762.00,772.15] vol=2.1x ATR=2.62 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-18 09:40:00 | 756.97 | 762.54 | 0.00 | T1 1.5R @ 756.97 |
| Target hit | 2024-09-18 12:35:00 | 750.65 | 747.71 | 0.00 | Trail-exit close>VWAP |

### Cycle 25 — SELL (started 2024-09-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 09:35:00 | 746.85 | 753.67 | 0.00 | ORB-short ORB[752.40,761.20] vol=1.9x ATR=3.25 |
| Stop hit — per-position SL triggered | 2024-09-19 09:40:00 | 750.10 | 753.31 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2024-09-24 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-24 09:45:00 | 714.40 | 720.67 | 0.00 | ORB-short ORB[720.00,726.40] vol=3.2x ATR=2.27 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-24 09:55:00 | 710.99 | 718.26 | 0.00 | T1 1.5R @ 710.99 |
| Target hit | 2024-09-24 12:40:00 | 702.50 | 701.28 | 0.00 | Trail-exit close>VWAP |

### Cycle 27 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-01 10:15:00 | 669.40 | 675.40 | 0.00 | ORB-short ORB[675.25,682.45] vol=2.0x ATR=2.79 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-01 10:40:00 | 665.21 | 671.99 | 0.00 | T1 1.5R @ 665.21 |
| Target hit | 2024-10-01 12:30:00 | 668.50 | 665.56 | 0.00 | Trail-exit close>VWAP |

### Cycle 28 — SELL (started 2024-10-07 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:30:00 | 670.00 | 673.34 | 0.00 | ORB-short ORB[676.15,684.15] vol=1.8x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:40:00 | 665.31 | 672.39 | 0.00 | T1 1.5R @ 665.31 |
| Stop hit — per-position SL triggered | 2024-10-07 11:15:00 | 670.00 | 670.76 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-10-14 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-14 10:55:00 | 706.40 | 696.98 | 0.00 | ORB-long ORB[690.75,700.00] vol=3.2x ATR=2.71 |
| Stop hit — per-position SL triggered | 2024-10-14 11:05:00 | 703.69 | 697.68 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2024-10-16 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-16 10:45:00 | 701.65 | 705.41 | 0.00 | ORB-short ORB[702.55,709.30] vol=1.6x ATR=2.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-16 10:55:00 | 698.55 | 704.91 | 0.00 | T1 1.5R @ 698.55 |
| Target hit | 2024-10-16 14:00:00 | 700.00 | 699.98 | 0.00 | Trail-exit close>VWAP |

### Cycle 31 — BUY (started 2024-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-24 10:15:00 | 690.55 | 688.54 | 0.00 | ORB-long ORB[681.00,689.95] vol=2.9x ATR=2.91 |
| Stop hit — per-position SL triggered | 2024-10-24 11:15:00 | 687.64 | 688.60 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-10-25 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:10:00 | 676.55 | 679.73 | 0.00 | ORB-short ORB[677.05,683.00] vol=3.1x ATR=2.96 |
| Stop hit — per-position SL triggered | 2024-10-25 10:20:00 | 679.51 | 679.69 | 0.00 | SL hit |

### Cycle 33 — BUY (started 2024-10-30 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-30 09:50:00 | 705.20 | 700.11 | 0.00 | ORB-long ORB[695.00,700.90] vol=4.5x ATR=2.27 |
| Stop hit — per-position SL triggered | 2024-10-30 10:00:00 | 702.93 | 700.58 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-11-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-06 09:35:00 | 712.80 | 707.21 | 0.00 | ORB-long ORB[702.00,709.60] vol=1.6x ATR=3.12 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 09:50:00 | 717.48 | 711.07 | 0.00 | T1 1.5R @ 717.48 |
| Target hit | 2024-11-06 15:20:00 | 748.90 | 731.69 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 35 — BUY (started 2024-11-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 11:10:00 | 743.75 | 738.57 | 0.00 | ORB-long ORB[733.55,741.40] vol=10.5x ATR=2.52 |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 741.23 | 738.96 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-11-19 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 10:40:00 | 713.35 | 708.45 | 0.00 | ORB-long ORB[698.80,709.40] vol=1.8x ATR=2.44 |
| Stop hit — per-position SL triggered | 2024-11-19 10:45:00 | 710.91 | 708.56 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-11-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-22 09:35:00 | 727.00 | 722.41 | 0.00 | ORB-long ORB[714.10,722.95] vol=4.0x ATR=2.83 |
| Stop hit — per-position SL triggered | 2024-11-22 09:45:00 | 724.17 | 723.90 | 0.00 | SL hit |

### Cycle 38 — SELL (started 2024-12-05 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-05 10:55:00 | 765.35 | 770.51 | 0.00 | ORB-short ORB[769.05,777.75] vol=2.0x ATR=2.34 |
| Stop hit — per-position SL triggered | 2024-12-05 11:00:00 | 767.69 | 770.49 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2024-12-06 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 09:35:00 | 783.50 | 780.60 | 0.00 | ORB-long ORB[775.00,783.00] vol=2.8x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 10:00:00 | 788.44 | 783.09 | 0.00 | T1 1.5R @ 788.44 |
| Target hit | 2024-12-06 11:50:00 | 785.55 | 787.06 | 0.00 | Trail-exit close<VWAP |

### Cycle 40 — BUY (started 2024-12-16 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-16 10:00:00 | 813.20 | 808.46 | 0.00 | ORB-long ORB[800.40,807.50] vol=4.9x ATR=3.92 |
| Stop hit — per-position SL triggered | 2024-12-16 10:05:00 | 809.28 | 808.65 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-12-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-20 09:30:00 | 782.55 | 787.95 | 0.00 | ORB-short ORB[785.40,793.20] vol=2.1x ATR=3.18 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:45:00 | 777.77 | 785.74 | 0.00 | T1 1.5R @ 777.77 |
| Stop hit — per-position SL triggered | 2024-12-20 11:55:00 | 782.55 | 780.34 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-12-24 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-24 10:10:00 | 748.10 | 743.28 | 0.00 | ORB-long ORB[738.10,746.75] vol=1.8x ATR=3.52 |
| Stop hit — per-position SL triggered | 2024-12-24 10:35:00 | 744.58 | 743.80 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2024-12-27 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-27 09:55:00 | 759.55 | 754.85 | 0.00 | ORB-long ORB[731.00,742.25] vol=9.4x ATR=4.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-27 10:00:00 | 765.80 | 756.07 | 0.00 | T1 1.5R @ 765.80 |
| Stop hit — per-position SL triggered | 2024-12-27 10:05:00 | 759.55 | 755.45 | 0.00 | SL hit |

### Cycle 44 — BUY (started 2024-12-30 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 10:45:00 | 746.40 | 740.24 | 0.00 | ORB-long ORB[736.85,745.35] vol=2.4x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-12-30 10:50:00 | 742.95 | 740.53 | 0.00 | SL hit |

### Cycle 45 — BUY (started 2025-01-02 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-02 09:40:00 | 761.95 | 757.34 | 0.00 | ORB-long ORB[748.60,757.40] vol=2.4x ATR=2.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-02 10:10:00 | 766.42 | 760.64 | 0.00 | T1 1.5R @ 766.42 |
| Target hit | 2025-01-02 10:25:00 | 764.30 | 769.10 | 0.00 | Trail-exit close<VWAP |

### Cycle 46 — SELL (started 2025-01-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-20 09:30:00 | 748.40 | 752.16 | 0.00 | ORB-short ORB[751.00,759.75] vol=2.1x ATR=2.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 09:50:00 | 745.29 | 747.38 | 0.00 | T1 1.5R @ 745.29 |
| Stop hit — per-position SL triggered | 2025-01-20 10:05:00 | 748.40 | 747.13 | 0.00 | SL hit |

### Cycle 47 — SELL (started 2025-01-21 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-21 09:35:00 | 753.95 | 760.26 | 0.00 | ORB-short ORB[754.55,765.45] vol=1.9x ATR=2.98 |
| Stop hit — per-position SL triggered | 2025-01-21 09:40:00 | 756.93 | 759.45 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2025-02-07 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 10:40:00 | 923.65 | 918.00 | 0.00 | ORB-long ORB[906.25,917.00] vol=2.3x ATR=5.30 |
| Stop hit — per-position SL triggered | 2025-02-07 12:10:00 | 918.35 | 919.36 | 0.00 | SL hit |

### Cycle 49 — BUY (started 2025-02-19 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-19 09:35:00 | 817.25 | 808.77 | 0.00 | ORB-long ORB[800.15,809.50] vol=1.6x ATR=4.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 09:50:00 | 823.58 | 813.84 | 0.00 | T1 1.5R @ 823.58 |
| Target hit | 2025-02-19 11:35:00 | 822.05 | 830.57 | 0.00 | Trail-exit close<VWAP |

### Cycle 50 — SELL (started 2025-02-21 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-21 09:40:00 | 802.35 | 811.48 | 0.00 | ORB-short ORB[808.00,820.05] vol=2.2x ATR=4.39 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-21 10:05:00 | 795.76 | 806.39 | 0.00 | T1 1.5R @ 795.76 |
| Stop hit — per-position SL triggered | 2025-02-21 12:20:00 | 802.35 | 801.66 | 0.00 | SL hit |

### Cycle 51 — SELL (started 2025-03-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-11 09:30:00 | 704.80 | 710.56 | 0.00 | ORB-short ORB[707.50,716.50] vol=2.8x ATR=4.07 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-11 09:35:00 | 698.69 | 708.04 | 0.00 | T1 1.5R @ 698.69 |
| Stop hit — per-position SL triggered | 2025-03-11 10:15:00 | 704.80 | 702.37 | 0.00 | SL hit |

### Cycle 52 — SELL (started 2025-03-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-17 09:35:00 | 643.35 | 649.23 | 0.00 | ORB-short ORB[648.30,657.35] vol=2.4x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-17 09:40:00 | 639.45 | 647.84 | 0.00 | T1 1.5R @ 639.45 |
| Stop hit — per-position SL triggered | 2025-03-17 09:45:00 | 643.35 | 646.76 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2025-04-17 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 09:45:00 | 661.85 | 655.58 | 0.00 | ORB-long ORB[650.50,656.40] vol=2.5x ATR=2.28 |
| Stop hit — per-position SL triggered | 2025-04-17 10:00:00 | 659.57 | 657.64 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2025-04-24 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-24 10:35:00 | 700.10 | 693.90 | 0.00 | ORB-long ORB[688.65,698.45] vol=1.6x ATR=2.80 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-24 11:50:00 | 704.30 | 696.99 | 0.00 | T1 1.5R @ 704.30 |
| Stop hit — per-position SL triggered | 2025-04-24 12:30:00 | 700.10 | 697.64 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2024-05-22 10:15:00 | 604.05 | 2024-05-22 10:20:00 | 607.06 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest1 | 2024-05-23 10:05:00 | 625.05 | 2024-05-23 10:50:00 | 628.56 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-05-23 10:05:00 | 625.05 | 2024-05-23 12:30:00 | 625.80 | TARGET_HIT | 0.50 | 0.12% |
| BUY | retest1 | 2024-05-28 09:45:00 | 642.45 | 2024-05-28 09:50:00 | 646.56 | PARTIAL | 0.50 | 0.64% |
| BUY | retest1 | 2024-05-28 09:45:00 | 642.45 | 2024-05-28 09:55:00 | 642.45 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-11 10:10:00 | 690.45 | 2024-06-11 10:20:00 | 695.67 | PARTIAL | 0.50 | 0.76% |
| BUY | retest1 | 2024-06-11 10:10:00 | 690.45 | 2024-06-11 14:00:00 | 701.85 | TARGET_HIT | 0.50 | 1.65% |
| BUY | retest1 | 2024-06-13 09:35:00 | 709.80 | 2024-06-13 09:45:00 | 705.79 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2024-06-20 11:00:00 | 721.05 | 2024-06-20 15:20:00 | 721.10 | TARGET_HIT | 1.00 | 0.01% |
| BUY | retest1 | 2024-06-25 09:45:00 | 758.90 | 2024-06-25 10:10:00 | 756.22 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-06-27 10:55:00 | 747.50 | 2024-06-27 11:20:00 | 749.67 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-07-01 09:30:00 | 748.30 | 2024-07-01 09:35:00 | 753.38 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-07-01 09:30:00 | 748.30 | 2024-07-01 14:30:00 | 754.30 | TARGET_HIT | 0.50 | 0.80% |
| BUY | retest1 | 2024-07-02 10:20:00 | 763.60 | 2024-07-02 10:45:00 | 759.91 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2024-07-04 09:40:00 | 755.75 | 2024-07-04 09:45:00 | 753.10 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-09 10:00:00 | 743.60 | 2024-07-09 10:10:00 | 747.00 | PARTIAL | 0.50 | 0.46% |
| BUY | retest1 | 2024-07-09 10:00:00 | 743.60 | 2024-07-09 10:15:00 | 743.60 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-10 10:00:00 | 718.65 | 2024-07-10 10:25:00 | 721.17 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-07-26 09:30:00 | 797.80 | 2024-07-26 09:35:00 | 803.59 | PARTIAL | 0.50 | 0.73% |
| BUY | retest1 | 2024-07-26 09:30:00 | 797.80 | 2024-07-26 11:10:00 | 812.95 | TARGET_HIT | 0.50 | 1.90% |
| BUY | retest1 | 2024-08-02 09:45:00 | 767.50 | 2024-08-02 11:20:00 | 775.45 | PARTIAL | 0.50 | 1.04% |
| BUY | retest1 | 2024-08-02 09:45:00 | 767.50 | 2024-08-02 12:40:00 | 767.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-09 09:30:00 | 766.50 | 2024-08-09 09:55:00 | 762.87 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-08-20 10:15:00 | 793.55 | 2024-08-20 10:25:00 | 798.25 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-08-20 10:15:00 | 793.55 | 2024-08-20 11:00:00 | 800.95 | TARGET_HIT | 0.50 | 0.93% |
| BUY | retest1 | 2024-08-21 09:45:00 | 802.20 | 2024-08-21 10:05:00 | 799.34 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-08-23 09:30:00 | 778.50 | 2024-08-23 10:20:00 | 774.30 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2024-08-23 09:30:00 | 778.50 | 2024-08-23 10:35:00 | 778.50 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-27 09:45:00 | 776.55 | 2024-08-27 09:50:00 | 780.60 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-08-27 09:45:00 | 776.55 | 2024-08-27 10:20:00 | 776.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-03 10:10:00 | 796.90 | 2024-09-03 10:15:00 | 801.11 | PARTIAL | 0.50 | 0.53% |
| BUY | retest1 | 2024-09-03 10:10:00 | 796.90 | 2024-09-03 10:25:00 | 796.90 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-04 10:00:00 | 795.20 | 2024-09-04 10:20:00 | 792.30 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2024-09-11 09:30:00 | 802.00 | 2024-09-11 10:00:00 | 806.76 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2024-09-11 09:30:00 | 802.00 | 2024-09-11 10:00:00 | 799.35 | TARGET_HIT | 0.50 | -0.33% |
| SELL | retest1 | 2024-09-18 09:35:00 | 760.90 | 2024-09-18 09:40:00 | 756.97 | PARTIAL | 0.50 | 0.52% |
| SELL | retest1 | 2024-09-18 09:35:00 | 760.90 | 2024-09-18 12:35:00 | 750.65 | TARGET_HIT | 0.50 | 1.35% |
| SELL | retest1 | 2024-09-19 09:35:00 | 746.85 | 2024-09-19 09:40:00 | 750.10 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest1 | 2024-09-24 09:45:00 | 714.40 | 2024-09-24 09:55:00 | 710.99 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-09-24 09:45:00 | 714.40 | 2024-09-24 12:40:00 | 702.50 | TARGET_HIT | 0.50 | 1.67% |
| SELL | retest1 | 2024-10-01 10:15:00 | 669.40 | 2024-10-01 10:40:00 | 665.21 | PARTIAL | 0.50 | 0.63% |
| SELL | retest1 | 2024-10-01 10:15:00 | 669.40 | 2024-10-01 12:30:00 | 668.50 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2024-10-07 10:30:00 | 670.00 | 2024-10-07 10:40:00 | 665.31 | PARTIAL | 0.50 | 0.70% |
| SELL | retest1 | 2024-10-07 10:30:00 | 670.00 | 2024-10-07 11:15:00 | 670.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-10-14 10:55:00 | 706.40 | 2024-10-14 11:05:00 | 703.69 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-10-16 10:45:00 | 701.65 | 2024-10-16 10:55:00 | 698.55 | PARTIAL | 0.50 | 0.44% |
| SELL | retest1 | 2024-10-16 10:45:00 | 701.65 | 2024-10-16 14:00:00 | 700.00 | TARGET_HIT | 0.50 | 0.24% |
| BUY | retest1 | 2024-10-24 10:15:00 | 690.55 | 2024-10-24 11:15:00 | 687.64 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest1 | 2024-10-25 10:10:00 | 676.55 | 2024-10-25 10:20:00 | 679.51 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2024-10-30 09:50:00 | 705.20 | 2024-10-30 10:00:00 | 702.93 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-11-06 09:35:00 | 712.80 | 2024-11-06 09:50:00 | 717.48 | PARTIAL | 0.50 | 0.66% |
| BUY | retest1 | 2024-11-06 09:35:00 | 712.80 | 2024-11-06 15:20:00 | 748.90 | TARGET_HIT | 0.50 | 5.06% |
| BUY | retest1 | 2024-11-08 11:10:00 | 743.75 | 2024-11-08 11:15:00 | 741.23 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-19 10:40:00 | 713.35 | 2024-11-19 10:45:00 | 710.91 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-11-22 09:35:00 | 727.00 | 2024-11-22 09:45:00 | 724.17 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2024-12-05 10:55:00 | 765.35 | 2024-12-05 11:00:00 | 767.69 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-12-06 09:35:00 | 783.50 | 2024-12-06 10:00:00 | 788.44 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-12-06 09:35:00 | 783.50 | 2024-12-06 11:50:00 | 785.55 | TARGET_HIT | 0.50 | 0.26% |
| BUY | retest1 | 2024-12-16 10:00:00 | 813.20 | 2024-12-16 10:05:00 | 809.28 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest1 | 2024-12-20 09:30:00 | 782.55 | 2024-12-20 09:45:00 | 777.77 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2024-12-20 09:30:00 | 782.55 | 2024-12-20 11:55:00 | 782.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-24 10:10:00 | 748.10 | 2024-12-24 10:35:00 | 744.58 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-12-27 09:55:00 | 759.55 | 2024-12-27 10:00:00 | 765.80 | PARTIAL | 0.50 | 0.82% |
| BUY | retest1 | 2024-12-27 09:55:00 | 759.55 | 2024-12-27 10:05:00 | 759.55 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-30 10:45:00 | 746.40 | 2024-12-30 10:50:00 | 742.95 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest1 | 2025-01-02 09:40:00 | 761.95 | 2025-01-02 10:10:00 | 766.42 | PARTIAL | 0.50 | 0.59% |
| BUY | retest1 | 2025-01-02 09:40:00 | 761.95 | 2025-01-02 10:25:00 | 764.30 | TARGET_HIT | 0.50 | 0.31% |
| SELL | retest1 | 2025-01-20 09:30:00 | 748.40 | 2025-01-20 09:50:00 | 745.29 | PARTIAL | 0.50 | 0.42% |
| SELL | retest1 | 2025-01-20 09:30:00 | 748.40 | 2025-01-20 10:05:00 | 748.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-01-21 09:35:00 | 753.95 | 2025-01-21 09:40:00 | 756.93 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest1 | 2025-02-07 10:40:00 | 923.65 | 2025-02-07 12:10:00 | 918.35 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest1 | 2025-02-19 09:35:00 | 817.25 | 2025-02-19 09:50:00 | 823.58 | PARTIAL | 0.50 | 0.77% |
| BUY | retest1 | 2025-02-19 09:35:00 | 817.25 | 2025-02-19 11:35:00 | 822.05 | TARGET_HIT | 0.50 | 0.59% |
| SELL | retest1 | 2025-02-21 09:40:00 | 802.35 | 2025-02-21 10:05:00 | 795.76 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2025-02-21 09:40:00 | 802.35 | 2025-02-21 12:20:00 | 802.35 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-11 09:30:00 | 704.80 | 2025-03-11 09:35:00 | 698.69 | PARTIAL | 0.50 | 0.87% |
| SELL | retest1 | 2025-03-11 09:30:00 | 704.80 | 2025-03-11 10:15:00 | 704.80 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-17 09:35:00 | 643.35 | 2025-03-17 09:40:00 | 639.45 | PARTIAL | 0.50 | 0.61% |
| SELL | retest1 | 2025-03-17 09:35:00 | 643.35 | 2025-03-17 09:45:00 | 643.35 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-04-17 09:45:00 | 661.85 | 2025-04-17 10:00:00 | 659.57 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2025-04-24 10:35:00 | 700.10 | 2025-04-24 11:50:00 | 704.30 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2025-04-24 10:35:00 | 700.10 | 2025-04-24 12:30:00 | 700.10 | STOP_HIT | 0.50 | 0.00% |
