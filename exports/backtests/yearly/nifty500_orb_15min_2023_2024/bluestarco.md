# Blue Star Ltd. (BLUESTARCO)

## Backtest Summary

- **Window:** 2023-05-12 09:15:00 → 2025-05-30 15:25:00 (38016 bars)
- **Last close:** 1527.50
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
| ENTRY1 | 75 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 10 |
| STOP_HIT | 65 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 100 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 35 / 65
- **Target hits / Stop hits / Partials:** 10 / 65 / 25
- **Avg / median % per leg:** 0.08% / 0.00%
- **Sum % (uncompounded):** 7.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 41 | 17 | 41.5% | 4 | 24 | 13 | 0.21% | 8.4% |
| BUY @ 2nd Alert (retest1) | 41 | 17 | 41.5% | 4 | 24 | 13 | 0.21% | 8.4% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 59 | 18 | 30.5% | 6 | 41 | 12 | -0.01% | -0.6% |
| SELL @ 2nd Alert (retest1) | 59 | 18 | 30.5% | 6 | 41 | 12 | -0.01% | -0.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 100 | 35 | 35.0% | 10 | 65 | 25 | 0.08% | 7.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-17 10:20:00 | 704.85 | 710.13 | 0.00 | ORB-short ORB[710.65,715.98] vol=1.7x ATR=2.00 |
| Stop hit — per-position SL triggered | 2023-05-17 10:30:00 | 706.85 | 709.35 | 0.00 | SL hit |

### Cycle 2 — BUY (started 2023-05-18 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 10:40:00 | 720.50 | 714.51 | 0.00 | ORB-long ORB[705.48,715.13] vol=5.3x ATR=2.37 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-05-18 10:45:00 | 724.06 | 716.04 | 0.00 | T1 1.5R @ 724.06 |
| Stop hit — per-position SL triggered | 2023-05-18 10:50:00 | 720.50 | 716.29 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2023-05-19 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-19 10:30:00 | 706.60 | 708.40 | 0.00 | ORB-short ORB[710.63,715.00] vol=1.7x ATR=1.48 |
| Stop hit — per-position SL triggered | 2023-05-19 11:15:00 | 708.08 | 708.15 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2023-05-22 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-22 09:55:00 | 700.00 | 703.58 | 0.00 | ORB-short ORB[701.65,706.75] vol=1.7x ATR=2.08 |
| Stop hit — per-position SL triggered | 2023-05-22 10:00:00 | 702.08 | 703.17 | 0.00 | SL hit |

### Cycle 5 — BUY (started 2023-05-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-24 09:30:00 | 703.50 | 700.26 | 0.00 | ORB-long ORB[695.50,700.58] vol=1.6x ATR=1.77 |
| Stop hit — per-position SL triggered | 2023-05-24 09:50:00 | 701.73 | 702.82 | 0.00 | SL hit |

### Cycle 6 — BUY (started 2023-05-25 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-25 09:55:00 | 725.98 | 721.35 | 0.00 | ORB-long ORB[712.40,718.50] vol=4.7x ATR=2.55 |
| Stop hit — per-position SL triggered | 2023-05-25 11:40:00 | 723.43 | 724.30 | 0.00 | SL hit |

### Cycle 7 — SELL (started 2023-05-29 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-29 09:30:00 | 728.73 | 733.64 | 0.00 | ORB-short ORB[732.38,737.48] vol=2.5x ATR=1.78 |
| Stop hit — per-position SL triggered | 2023-05-29 09:35:00 | 730.51 | 733.01 | 0.00 | SL hit |

### Cycle 8 — SELL (started 2023-05-31 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-05-31 10:55:00 | 724.55 | 729.05 | 0.00 | ORB-short ORB[728.05,732.50] vol=2.4x ATR=1.37 |
| Stop hit — per-position SL triggered | 2023-05-31 11:10:00 | 725.92 | 728.80 | 0.00 | SL hit |

### Cycle 9 — SELL (started 2023-06-01 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-01 09:35:00 | 713.48 | 717.34 | 0.00 | ORB-short ORB[718.00,721.25] vol=2.1x ATR=2.30 |
| Stop hit — per-position SL triggered | 2023-06-01 10:00:00 | 715.78 | 714.85 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2023-06-02 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-02 10:55:00 | 718.50 | 721.10 | 0.00 | ORB-short ORB[721.08,729.78] vol=5.6x ATR=1.40 |
| Stop hit — per-position SL triggered | 2023-06-02 11:10:00 | 719.90 | 720.60 | 0.00 | SL hit |

### Cycle 11 — BUY (started 2023-06-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-07 09:45:00 | 729.25 | 727.59 | 0.00 | ORB-long ORB[725.10,728.40] vol=2.6x ATR=1.53 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-07 09:50:00 | 731.54 | 728.31 | 0.00 | T1 1.5R @ 731.54 |
| Stop hit — per-position SL triggered | 2023-06-07 10:10:00 | 729.25 | 729.04 | 0.00 | SL hit |

### Cycle 12 — SELL (started 2023-06-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-08 09:30:00 | 728.78 | 731.70 | 0.00 | ORB-short ORB[730.80,736.55] vol=1.9x ATR=1.62 |
| Stop hit — per-position SL triggered | 2023-06-08 10:00:00 | 730.40 | 730.96 | 0.00 | SL hit |

### Cycle 13 — SELL (started 2023-06-12 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-12 09:45:00 | 728.03 | 731.02 | 0.00 | ORB-short ORB[730.48,733.50] vol=2.2x ATR=2.09 |
| Stop hit — per-position SL triggered | 2023-06-12 12:25:00 | 730.12 | 729.55 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2023-06-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-06-13 09:45:00 | 748.03 | 747.37 | 0.00 | ORB-long ORB[743.00,747.50] vol=6.6x ATR=2.54 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-13 12:40:00 | 751.84 | 748.41 | 0.00 | T1 1.5R @ 751.84 |
| Target hit | 2023-06-13 15:20:00 | 755.43 | 754.25 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 15 — SELL (started 2023-06-19 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-19 11:00:00 | 766.45 | 770.88 | 0.00 | ORB-short ORB[770.00,780.00] vol=2.1x ATR=1.95 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-19 11:35:00 | 763.52 | 769.57 | 0.00 | T1 1.5R @ 763.52 |
| Target hit | 2023-06-19 15:20:00 | 759.93 | 765.84 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 16 — SELL (started 2023-06-30 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-06-30 09:40:00 | 763.95 | 764.87 | 0.00 | ORB-short ORB[766.00,772.15] vol=2.6x ATR=2.65 |
| Stop hit — per-position SL triggered | 2023-06-30 09:45:00 | 766.60 | 764.97 | 0.00 | SL hit |

### Cycle 17 — SELL (started 2023-07-04 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-04 10:45:00 | 768.50 | 771.16 | 0.00 | ORB-short ORB[769.05,773.90] vol=2.4x ATR=1.94 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-04 11:25:00 | 765.59 | 769.99 | 0.00 | T1 1.5R @ 765.59 |
| Stop hit — per-position SL triggered | 2023-07-04 11:35:00 | 768.50 | 769.90 | 0.00 | SL hit |

### Cycle 18 — SELL (started 2023-07-10 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-10 09:30:00 | 785.00 | 787.69 | 0.00 | ORB-short ORB[786.55,792.90] vol=2.9x ATR=2.54 |
| Stop hit — per-position SL triggered | 2023-07-10 09:35:00 | 787.54 | 787.42 | 0.00 | SL hit |

### Cycle 19 — SELL (started 2023-07-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-12 09:35:00 | 787.75 | 789.39 | 0.00 | ORB-short ORB[789.95,799.00] vol=2.5x ATR=2.10 |
| Stop hit — per-position SL triggered | 2023-07-12 09:55:00 | 789.85 | 789.23 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2023-07-13 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-13 10:50:00 | 783.00 | 785.79 | 0.00 | ORB-short ORB[785.50,790.00] vol=5.2x ATR=1.73 |
| Stop hit — per-position SL triggered | 2023-07-13 11:00:00 | 784.73 | 785.75 | 0.00 | SL hit |

### Cycle 21 — SELL (started 2023-07-18 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-18 09:35:00 | 788.95 | 791.31 | 0.00 | ORB-short ORB[793.00,799.00] vol=3.4x ATR=2.14 |
| Stop hit — per-position SL triggered | 2023-07-18 09:50:00 | 791.09 | 791.15 | 0.00 | SL hit |

### Cycle 22 — SELL (started 2023-07-19 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-19 09:30:00 | 789.30 | 791.44 | 0.00 | ORB-short ORB[790.00,795.00] vol=2.0x ATR=2.03 |
| Stop hit — per-position SL triggered | 2023-07-19 09:40:00 | 791.33 | 791.07 | 0.00 | SL hit |

### Cycle 23 — SELL (started 2023-07-21 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-21 10:20:00 | 775.45 | 781.34 | 0.00 | ORB-short ORB[778.30,786.10] vol=1.9x ATR=2.42 |
| Stop hit — per-position SL triggered | 2023-07-21 10:35:00 | 777.87 | 781.08 | 0.00 | SL hit |

### Cycle 24 — SELL (started 2023-07-24 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-24 10:05:00 | 781.55 | 782.98 | 0.00 | ORB-short ORB[782.55,789.90] vol=1.7x ATR=1.88 |
| Stop hit — per-position SL triggered | 2023-07-24 10:55:00 | 783.43 | 782.83 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2023-07-27 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-07-27 10:10:00 | 771.40 | 774.12 | 0.00 | ORB-short ORB[772.60,782.95] vol=1.6x ATR=2.05 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-27 11:15:00 | 768.33 | 772.66 | 0.00 | T1 1.5R @ 768.33 |
| Stop hit — per-position SL triggered | 2023-07-27 12:55:00 | 771.40 | 770.96 | 0.00 | SL hit |

### Cycle 26 — SELL (started 2023-08-02 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-02 10:50:00 | 777.20 | 782.44 | 0.00 | ORB-short ORB[781.05,787.50] vol=1.5x ATR=1.96 |
| Stop hit — per-position SL triggered | 2023-08-02 10:55:00 | 779.16 | 782.02 | 0.00 | SL hit |

### Cycle 27 — SELL (started 2023-08-09 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-09 09:45:00 | 734.85 | 736.96 | 0.00 | ORB-short ORB[736.00,741.60] vol=1.7x ATR=2.22 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-08-09 10:00:00 | 731.52 | 735.98 | 0.00 | T1 1.5R @ 731.52 |
| Stop hit — per-position SL triggered | 2023-08-09 11:15:00 | 734.85 | 733.74 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2023-08-10 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-10 09:40:00 | 754.40 | 751.08 | 0.00 | ORB-long ORB[742.70,749.35] vol=1.6x ATR=2.46 |
| Stop hit — per-position SL triggered | 2023-08-10 10:05:00 | 751.94 | 752.19 | 0.00 | SL hit |

### Cycle 29 — SELL (started 2023-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-14 10:15:00 | 735.30 | 737.29 | 0.00 | ORB-short ORB[735.40,745.40] vol=3.3x ATR=2.56 |
| Stop hit — per-position SL triggered | 2023-08-14 11:35:00 | 737.86 | 737.37 | 0.00 | SL hit |

### Cycle 30 — SELL (started 2023-08-17 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-17 09:35:00 | 741.20 | 742.80 | 0.00 | ORB-short ORB[741.95,745.00] vol=1.7x ATR=1.40 |
| Stop hit — per-position SL triggered | 2023-08-17 09:45:00 | 742.60 | 742.75 | 0.00 | SL hit |

### Cycle 31 — BUY (started 2023-08-18 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-18 09:45:00 | 746.60 | 744.21 | 0.00 | ORB-long ORB[740.55,744.45] vol=2.1x ATR=2.27 |
| Stop hit — per-position SL triggered | 2023-08-18 09:55:00 | 744.33 | 744.31 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2023-08-21 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-21 09:50:00 | 743.70 | 746.35 | 0.00 | ORB-short ORB[745.25,751.35] vol=2.1x ATR=1.82 |
| Stop hit — per-position SL triggered | 2023-08-21 14:10:00 | 745.52 | 744.59 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2023-08-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-22 09:35:00 | 738.05 | 740.53 | 0.00 | ORB-short ORB[740.00,747.45] vol=2.2x ATR=2.31 |
| Stop hit — per-position SL triggered | 2023-08-22 09:40:00 | 740.36 | 740.51 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2023-08-23 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-08-23 09:30:00 | 742.30 | 740.79 | 0.00 | ORB-long ORB[737.05,742.05] vol=1.9x ATR=2.04 |
| Stop hit — per-position SL triggered | 2023-08-23 09:55:00 | 740.26 | 741.28 | 0.00 | SL hit |

### Cycle 35 — SELL (started 2023-08-24 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-08-24 09:55:00 | 732.90 | 737.10 | 0.00 | ORB-short ORB[737.65,741.50] vol=4.3x ATR=2.15 |
| Stop hit — per-position SL triggered | 2023-08-24 10:00:00 | 735.05 | 736.65 | 0.00 | SL hit |

### Cycle 36 — SELL (started 2023-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-01 10:15:00 | 729.00 | 732.88 | 0.00 | ORB-short ORB[730.10,739.75] vol=1.6x ATR=2.15 |
| Stop hit — per-position SL triggered | 2023-09-01 10:55:00 | 731.15 | 732.13 | 0.00 | SL hit |

### Cycle 37 — SELL (started 2023-09-05 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-05 09:35:00 | 740.00 | 743.06 | 0.00 | ORB-short ORB[740.95,745.95] vol=1.6x ATR=2.45 |
| Stop hit — per-position SL triggered | 2023-09-05 09:50:00 | 742.45 | 742.22 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2023-09-06 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-06 10:30:00 | 752.30 | 747.65 | 0.00 | ORB-long ORB[741.55,749.10] vol=7.7x ATR=1.81 |
| Stop hit — per-position SL triggered | 2023-09-06 10:45:00 | 750.49 | 749.30 | 0.00 | SL hit |

### Cycle 39 — BUY (started 2023-09-07 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-07 09:30:00 | 759.50 | 755.88 | 0.00 | ORB-long ORB[750.00,758.05] vol=3.8x ATR=2.48 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-07 09:35:00 | 763.22 | 757.74 | 0.00 | T1 1.5R @ 763.22 |
| Stop hit — per-position SL triggered | 2023-09-07 09:40:00 | 759.50 | 758.16 | 0.00 | SL hit |

### Cycle 40 — SELL (started 2023-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-11 09:30:00 | 807.40 | 808.38 | 0.00 | ORB-short ORB[809.20,816.50] vol=6.5x ATR=2.50 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-11 09:35:00 | 803.66 | 807.24 | 0.00 | T1 1.5R @ 803.66 |
| Target hit | 2023-09-11 11:50:00 | 802.95 | 802.93 | 0.00 | Trail-exit close>VWAP |

### Cycle 41 — SELL (started 2023-09-12 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-09-12 09:30:00 | 796.05 | 800.03 | 0.00 | ORB-short ORB[800.00,810.45] vol=3.0x ATR=2.84 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-12 09:35:00 | 791.79 | 798.44 | 0.00 | T1 1.5R @ 791.79 |
| Target hit | 2023-09-12 10:45:00 | 792.00 | 790.91 | 0.00 | Trail-exit close>VWAP |

### Cycle 42 — BUY (started 2023-09-14 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-14 09:30:00 | 806.95 | 802.95 | 0.00 | ORB-long ORB[797.50,803.85] vol=3.5x ATR=3.94 |
| Stop hit — per-position SL triggered | 2023-09-14 10:40:00 | 803.01 | 804.29 | 0.00 | SL hit |

### Cycle 43 — BUY (started 2023-09-28 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-09-28 10:00:00 | 897.45 | 890.35 | 0.00 | ORB-long ORB[886.00,895.70] vol=3.2x ATR=3.67 |
| Stop hit — per-position SL triggered | 2023-09-28 10:15:00 | 893.78 | 892.21 | 0.00 | SL hit |

### Cycle 44 — SELL (started 2023-10-05 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-05 10:40:00 | 881.00 | 885.17 | 0.00 | ORB-short ORB[882.80,889.60] vol=2.0x ATR=1.88 |
| Stop hit — per-position SL triggered | 2023-10-05 10:45:00 | 882.88 | 884.92 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2023-10-13 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-13 10:20:00 | 875.15 | 881.75 | 0.00 | ORB-short ORB[877.60,888.35] vol=1.5x ATR=2.47 |
| Target hit | 2023-10-13 15:20:00 | 874.05 | 875.06 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 46 — BUY (started 2023-10-27 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-10-27 10:45:00 | 876.00 | 870.00 | 0.00 | ORB-long ORB[858.25,867.50] vol=3.3x ATR=2.87 |
| Stop hit — per-position SL triggered | 2023-10-27 10:50:00 | 873.13 | 870.20 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2023-11-03 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-03 10:40:00 | 939.45 | 931.07 | 0.00 | ORB-long ORB[921.55,929.90] vol=5.6x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-03 10:45:00 | 944.20 | 933.27 | 0.00 | T1 1.5R @ 944.20 |
| Target hit | 2023-11-03 15:20:00 | 972.90 | 957.81 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 48 — BUY (started 2023-11-15 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-11-15 10:35:00 | 973.90 | 967.57 | 0.00 | ORB-long ORB[957.15,970.90] vol=2.0x ATR=2.89 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-15 10:40:00 | 978.23 | 969.67 | 0.00 | T1 1.5R @ 978.23 |
| Stop hit — per-position SL triggered | 2023-11-15 11:05:00 | 973.90 | 972.32 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2023-11-29 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-29 11:10:00 | 989.95 | 995.78 | 0.00 | ORB-short ORB[998.00,1010.40] vol=1.8x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-11-29 14:50:00 | 985.45 | 991.79 | 0.00 | T1 1.5R @ 985.45 |
| Target hit | 2023-11-29 15:20:00 | 983.95 | 990.99 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — SELL (started 2023-12-08 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-08 11:00:00 | 1003.20 | 1007.48 | 0.00 | ORB-short ORB[1005.55,1016.50] vol=4.3x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-08 13:00:00 | 999.38 | 1005.42 | 0.00 | T1 1.5R @ 999.38 |
| Stop hit — per-position SL triggered | 2023-12-08 14:40:00 | 1003.20 | 1003.40 | 0.00 | SL hit |

### Cycle 51 — BUY (started 2023-12-12 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-12 09:50:00 | 1007.50 | 1001.87 | 0.00 | ORB-long ORB[990.00,1004.75] vol=2.0x ATR=3.22 |
| Stop hit — per-position SL triggered | 2023-12-12 10:10:00 | 1004.28 | 1003.17 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2023-12-18 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-18 11:05:00 | 977.00 | 971.63 | 0.00 | ORB-long ORB[964.10,976.10] vol=2.7x ATR=3.33 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-18 11:30:00 | 982.00 | 973.66 | 0.00 | T1 1.5R @ 982.00 |
| Stop hit — per-position SL triggered | 2023-12-18 12:35:00 | 977.00 | 977.97 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2023-12-22 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-22 11:00:00 | 946.85 | 936.26 | 0.00 | ORB-long ORB[931.20,939.25] vol=6.0x ATR=4.14 |
| Stop hit — per-position SL triggered | 2023-12-22 11:35:00 | 942.71 | 942.34 | 0.00 | SL hit |

### Cycle 54 — BUY (started 2023-12-27 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-27 10:20:00 | 950.00 | 946.15 | 0.00 | ORB-long ORB[938.15,948.95] vol=1.9x ATR=3.00 |
| Stop hit — per-position SL triggered | 2023-12-27 12:35:00 | 947.00 | 948.22 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2023-12-29 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-12-29 10:05:00 | 938.65 | 945.00 | 0.00 | ORB-short ORB[944.45,953.80] vol=2.1x ATR=2.99 |
| Stop hit — per-position SL triggered | 2023-12-29 10:15:00 | 941.64 | 944.50 | 0.00 | SL hit |

### Cycle 56 — SELL (started 2024-01-01 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-01 10:00:00 | 941.45 | 944.83 | 0.00 | ORB-short ORB[945.20,956.50] vol=2.2x ATR=3.35 |
| Stop hit — per-position SL triggered | 2024-01-01 11:10:00 | 944.80 | 943.53 | 0.00 | SL hit |

### Cycle 57 — SELL (started 2024-01-02 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-02 10:20:00 | 940.95 | 944.64 | 0.00 | ORB-short ORB[942.05,955.00] vol=1.5x ATR=2.71 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-02 10:40:00 | 936.89 | 943.12 | 0.00 | T1 1.5R @ 936.89 |
| Stop hit — per-position SL triggered | 2024-01-02 10:45:00 | 940.95 | 943.03 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-01-03 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-03 10:55:00 | 943.40 | 946.11 | 0.00 | ORB-short ORB[945.45,954.50] vol=1.6x ATR=1.84 |
| Stop hit — per-position SL triggered | 2024-01-03 12:00:00 | 945.24 | 945.32 | 0.00 | SL hit |

### Cycle 59 — SELL (started 2024-01-04 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-04 10:40:00 | 936.00 | 940.96 | 0.00 | ORB-short ORB[939.05,947.45] vol=3.0x ATR=1.90 |
| Stop hit — per-position SL triggered | 2024-01-04 10:45:00 | 937.90 | 940.66 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2024-01-10 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-01-10 10:00:00 | 960.10 | 968.27 | 0.00 | ORB-short ORB[966.90,977.75] vol=1.7x ATR=4.36 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-10 11:15:00 | 953.55 | 964.92 | 0.00 | T1 1.5R @ 953.55 |
| Stop hit — per-position SL triggered | 2024-01-10 11:45:00 | 960.10 | 964.64 | 0.00 | SL hit |

### Cycle 61 — BUY (started 2024-01-16 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-16 10:25:00 | 1081.00 | 1076.39 | 0.00 | ORB-long ORB[1064.95,1075.00] vol=9.4x ATR=4.47 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-16 10:30:00 | 1087.70 | 1079.03 | 0.00 | T1 1.5R @ 1087.70 |
| Stop hit — per-position SL triggered | 2024-01-16 10:35:00 | 1081.00 | 1079.42 | 0.00 | SL hit |

### Cycle 62 — BUY (started 2024-01-19 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-01-19 09:45:00 | 1072.00 | 1064.78 | 0.00 | ORB-long ORB[1054.45,1069.35] vol=2.7x ATR=6.78 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-01-19 10:00:00 | 1082.16 | 1072.25 | 0.00 | T1 1.5R @ 1082.16 |
| Target hit | 2024-01-19 13:40:00 | 1094.00 | 1096.45 | 0.00 | Trail-exit close<VWAP |

### Cycle 63 — BUY (started 2024-02-02 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-02 10:45:00 | 1206.00 | 1199.54 | 0.00 | ORB-long ORB[1187.30,1200.95] vol=2.0x ATR=5.61 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-02 10:55:00 | 1214.42 | 1201.12 | 0.00 | T1 1.5R @ 1214.42 |
| Stop hit — per-position SL triggered | 2024-02-02 11:50:00 | 1206.00 | 1203.01 | 0.00 | SL hit |

### Cycle 64 — SELL (started 2024-02-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-07 09:45:00 | 1148.05 | 1161.08 | 0.00 | ORB-short ORB[1162.05,1171.45] vol=1.9x ATR=5.61 |
| Stop hit — per-position SL triggered | 2024-02-07 09:55:00 | 1153.66 | 1159.03 | 0.00 | SL hit |

### Cycle 65 — SELL (started 2024-02-09 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-09 09:50:00 | 1149.75 | 1163.26 | 0.00 | ORB-short ORB[1158.30,1174.65] vol=1.6x ATR=5.76 |
| Stop hit — per-position SL triggered | 2024-02-09 10:05:00 | 1155.51 | 1160.69 | 0.00 | SL hit |

### Cycle 66 — SELL (started 2024-02-12 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-12 10:00:00 | 1165.30 | 1180.32 | 0.00 | ORB-short ORB[1175.00,1189.85] vol=2.0x ATR=5.99 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-12 10:20:00 | 1156.31 | 1174.71 | 0.00 | T1 1.5R @ 1156.31 |
| Target hit | 2024-02-12 15:20:00 | 1152.75 | 1162.53 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 67 — SELL (started 2024-02-13 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-13 09:45:00 | 1144.00 | 1157.87 | 0.00 | ORB-short ORB[1150.00,1167.00] vol=1.9x ATR=6.44 |
| Stop hit — per-position SL triggered | 2024-02-13 11:00:00 | 1150.44 | 1151.96 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2024-02-22 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-22 09:35:00 | 1268.40 | 1270.72 | 0.00 | ORB-short ORB[1270.00,1281.40] vol=4.6x ATR=6.25 |
| Stop hit — per-position SL triggered | 2024-02-22 09:45:00 | 1274.65 | 1270.90 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2024-02-27 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-27 10:40:00 | 1298.00 | 1281.26 | 0.00 | ORB-long ORB[1265.10,1277.00] vol=2.1x ATR=6.09 |
| Stop hit — per-position SL triggered | 2024-02-27 11:45:00 | 1291.91 | 1288.79 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2024-03-05 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-05 10:50:00 | 1326.85 | 1316.10 | 0.00 | ORB-long ORB[1305.05,1321.95] vol=2.5x ATR=4.63 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-05 11:20:00 | 1333.79 | 1324.56 | 0.00 | T1 1.5R @ 1333.79 |
| Target hit | 2024-03-05 12:00:00 | 1327.00 | 1331.01 | 0.00 | Trail-exit close<VWAP |

### Cycle 71 — SELL (started 2024-03-19 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-19 11:10:00 | 1265.25 | 1273.66 | 0.00 | ORB-short ORB[1277.20,1295.00] vol=2.1x ATR=3.64 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-19 11:20:00 | 1259.78 | 1270.82 | 0.00 | T1 1.5R @ 1259.78 |
| Stop hit — per-position SL triggered | 2024-03-19 11:55:00 | 1265.25 | 1269.71 | 0.00 | SL hit |

### Cycle 72 — BUY (started 2024-03-21 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-21 10:45:00 | 1257.40 | 1250.49 | 0.00 | ORB-long ORB[1238.00,1252.95] vol=1.6x ATR=4.10 |
| Stop hit — per-position SL triggered | 2024-03-21 11:05:00 | 1253.30 | 1251.36 | 0.00 | SL hit |

### Cycle 73 — BUY (started 2024-04-22 11:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-22 11:05:00 | 1450.00 | 1446.79 | 0.00 | ORB-long ORB[1433.25,1449.00] vol=12.4x ATR=3.91 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-22 12:35:00 | 1455.86 | 1448.76 | 0.00 | T1 1.5R @ 1455.86 |
| Stop hit — per-position SL triggered | 2024-04-22 12:40:00 | 1450.00 | 1448.78 | 0.00 | SL hit |

### Cycle 74 — BUY (started 2024-04-26 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-04-26 09:30:00 | 1470.05 | 1464.54 | 0.00 | ORB-long ORB[1452.00,1469.45] vol=1.5x ATR=7.23 |
| Stop hit — per-position SL triggered | 2024-04-26 09:35:00 | 1462.82 | 1464.33 | 0.00 | SL hit |

### Cycle 75 — BUY (started 2024-05-07 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-07 09:35:00 | 1460.50 | 1451.17 | 0.00 | ORB-long ORB[1435.20,1454.95] vol=3.2x ATR=6.58 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-07 09:45:00 | 1470.37 | 1456.10 | 0.00 | T1 1.5R @ 1470.37 |
| Stop hit — per-position SL triggered | 2024-05-07 11:35:00 | 1460.50 | 1464.96 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest1 | 2023-05-17 10:20:00 | 704.85 | 2023-05-17 10:30:00 | 706.85 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2023-05-18 10:40:00 | 720.50 | 2023-05-18 10:45:00 | 724.06 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-05-18 10:40:00 | 720.50 | 2023-05-18 10:50:00 | 720.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-05-19 10:30:00 | 706.60 | 2023-05-19 11:15:00 | 708.08 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-05-22 09:55:00 | 700.00 | 2023-05-22 10:00:00 | 702.08 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest1 | 2023-05-24 09:30:00 | 703.50 | 2023-05-24 09:50:00 | 701.73 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest1 | 2023-05-25 09:55:00 | 725.98 | 2023-05-25 11:40:00 | 723.43 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-05-29 09:30:00 | 728.73 | 2023-05-29 09:35:00 | 730.51 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-05-31 10:55:00 | 724.55 | 2023-05-31 11:10:00 | 725.92 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest1 | 2023-06-01 09:35:00 | 713.48 | 2023-06-01 10:00:00 | 715.78 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-06-02 10:55:00 | 718.50 | 2023-06-02 11:10:00 | 719.90 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2023-06-07 09:45:00 | 729.25 | 2023-06-07 09:50:00 | 731.54 | PARTIAL | 0.50 | 0.31% |
| BUY | retest1 | 2023-06-07 09:45:00 | 729.25 | 2023-06-07 10:10:00 | 729.25 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-06-08 09:30:00 | 728.78 | 2023-06-08 10:00:00 | 730.40 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-06-12 09:45:00 | 728.03 | 2023-06-12 12:25:00 | 730.12 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2023-06-13 09:45:00 | 748.03 | 2023-06-13 12:40:00 | 751.84 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-06-13 09:45:00 | 748.03 | 2023-06-13 15:20:00 | 755.43 | TARGET_HIT | 0.50 | 0.99% |
| SELL | retest1 | 2023-06-19 11:00:00 | 766.45 | 2023-06-19 11:35:00 | 763.52 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-06-19 11:00:00 | 766.45 | 2023-06-19 15:20:00 | 759.93 | TARGET_HIT | 0.50 | 0.85% |
| SELL | retest1 | 2023-06-30 09:40:00 | 763.95 | 2023-06-30 09:45:00 | 766.60 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-07-04 10:45:00 | 768.50 | 2023-07-04 11:25:00 | 765.59 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-07-04 10:45:00 | 768.50 | 2023-07-04 11:35:00 | 768.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-07-10 09:30:00 | 785.00 | 2023-07-10 09:35:00 | 787.54 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-07-12 09:35:00 | 787.75 | 2023-07-12 09:55:00 | 789.85 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-07-13 10:50:00 | 783.00 | 2023-07-13 11:00:00 | 784.73 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest1 | 2023-07-18 09:35:00 | 788.95 | 2023-07-18 09:50:00 | 791.09 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-07-19 09:30:00 | 789.30 | 2023-07-19 09:40:00 | 791.33 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest1 | 2023-07-21 10:20:00 | 775.45 | 2023-07-21 10:35:00 | 777.87 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2023-07-24 10:05:00 | 781.55 | 2023-07-24 10:55:00 | 783.43 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-07-27 10:10:00 | 771.40 | 2023-07-27 11:15:00 | 768.33 | PARTIAL | 0.50 | 0.40% |
| SELL | retest1 | 2023-07-27 10:10:00 | 771.40 | 2023-07-27 12:55:00 | 771.40 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-08-02 10:50:00 | 777.20 | 2023-08-02 10:55:00 | 779.16 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest1 | 2023-08-09 09:45:00 | 734.85 | 2023-08-09 10:00:00 | 731.52 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-08-09 09:45:00 | 734.85 | 2023-08-09 11:15:00 | 734.85 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-08-10 09:40:00 | 754.40 | 2023-08-10 10:05:00 | 751.94 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest1 | 2023-08-14 10:15:00 | 735.30 | 2023-08-14 11:35:00 | 737.86 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2023-08-17 09:35:00 | 741.20 | 2023-08-17 09:45:00 | 742.60 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2023-08-18 09:45:00 | 746.60 | 2023-08-18 09:55:00 | 744.33 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-08-21 09:50:00 | 743.70 | 2023-08-21 14:10:00 | 745.52 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest1 | 2023-08-22 09:35:00 | 738.05 | 2023-08-22 09:40:00 | 740.36 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2023-08-23 09:30:00 | 742.30 | 2023-08-23 09:55:00 | 740.26 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest1 | 2023-08-24 09:55:00 | 732.90 | 2023-08-24 10:00:00 | 735.05 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2023-09-01 10:15:00 | 729.00 | 2023-09-01 10:55:00 | 731.15 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest1 | 2023-09-05 09:35:00 | 740.00 | 2023-09-05 09:50:00 | 742.45 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-09-06 10:30:00 | 752.30 | 2023-09-06 10:45:00 | 750.49 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest1 | 2023-09-07 09:30:00 | 759.50 | 2023-09-07 09:35:00 | 763.22 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2023-09-07 09:30:00 | 759.50 | 2023-09-07 09:40:00 | 759.50 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-09-11 09:30:00 | 807.40 | 2023-09-11 09:35:00 | 803.66 | PARTIAL | 0.50 | 0.46% |
| SELL | retest1 | 2023-09-11 09:30:00 | 807.40 | 2023-09-11 11:50:00 | 802.95 | TARGET_HIT | 0.50 | 0.55% |
| SELL | retest1 | 2023-09-12 09:30:00 | 796.05 | 2023-09-12 09:35:00 | 791.79 | PARTIAL | 0.50 | 0.54% |
| SELL | retest1 | 2023-09-12 09:30:00 | 796.05 | 2023-09-12 10:45:00 | 792.00 | TARGET_HIT | 0.50 | 0.51% |
| BUY | retest1 | 2023-09-14 09:30:00 | 806.95 | 2023-09-14 10:40:00 | 803.01 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2023-09-28 10:00:00 | 897.45 | 2023-09-28 10:15:00 | 893.78 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2023-10-05 10:40:00 | 881.00 | 2023-10-05 10:45:00 | 882.88 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest1 | 2023-10-13 10:20:00 | 875.15 | 2023-10-13 15:20:00 | 874.05 | TARGET_HIT | 1.00 | 0.13% |
| BUY | retest1 | 2023-10-27 10:45:00 | 876.00 | 2023-10-27 10:50:00 | 873.13 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2023-11-03 10:40:00 | 939.45 | 2023-11-03 10:45:00 | 944.20 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-11-03 10:40:00 | 939.45 | 2023-11-03 15:20:00 | 972.90 | TARGET_HIT | 0.50 | 3.56% |
| BUY | retest1 | 2023-11-15 10:35:00 | 973.90 | 2023-11-15 10:40:00 | 978.23 | PARTIAL | 0.50 | 0.44% |
| BUY | retest1 | 2023-11-15 10:35:00 | 973.90 | 2023-11-15 11:05:00 | 973.90 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2023-11-29 11:10:00 | 989.95 | 2023-11-29 14:50:00 | 985.45 | PARTIAL | 0.50 | 0.45% |
| SELL | retest1 | 2023-11-29 11:10:00 | 989.95 | 2023-11-29 15:20:00 | 983.95 | TARGET_HIT | 0.50 | 0.61% |
| SELL | retest1 | 2023-12-08 11:00:00 | 1003.20 | 2023-12-08 13:00:00 | 999.38 | PARTIAL | 0.50 | 0.38% |
| SELL | retest1 | 2023-12-08 11:00:00 | 1003.20 | 2023-12-08 14:40:00 | 1003.20 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-12 09:50:00 | 1007.50 | 2023-12-12 10:10:00 | 1004.28 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2023-12-18 11:05:00 | 977.00 | 2023-12-18 11:30:00 | 982.00 | PARTIAL | 0.50 | 0.51% |
| BUY | retest1 | 2023-12-18 11:05:00 | 977.00 | 2023-12-18 12:35:00 | 977.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2023-12-22 11:00:00 | 946.85 | 2023-12-22 11:35:00 | 942.71 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest1 | 2023-12-27 10:20:00 | 950.00 | 2023-12-27 12:35:00 | 947.00 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2023-12-29 10:05:00 | 938.65 | 2023-12-29 10:15:00 | 941.64 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-01-01 10:00:00 | 941.45 | 2024-01-01 11:10:00 | 944.80 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-01-02 10:20:00 | 940.95 | 2024-01-02 10:40:00 | 936.89 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-01-02 10:20:00 | 940.95 | 2024-01-02 10:45:00 | 940.95 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-01-03 10:55:00 | 943.40 | 2024-01-03 12:00:00 | 945.24 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-01-04 10:40:00 | 936.00 | 2024-01-04 10:45:00 | 937.90 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest1 | 2024-01-10 10:00:00 | 960.10 | 2024-01-10 11:15:00 | 953.55 | PARTIAL | 0.50 | 0.68% |
| SELL | retest1 | 2024-01-10 10:00:00 | 960.10 | 2024-01-10 11:45:00 | 960.10 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-16 10:25:00 | 1081.00 | 2024-01-16 10:30:00 | 1087.70 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-01-16 10:25:00 | 1081.00 | 2024-01-16 10:35:00 | 1081.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-01-19 09:45:00 | 1072.00 | 2024-01-19 10:00:00 | 1082.16 | PARTIAL | 0.50 | 0.95% |
| BUY | retest1 | 2024-01-19 09:45:00 | 1072.00 | 2024-01-19 13:40:00 | 1094.00 | TARGET_HIT | 0.50 | 2.05% |
| BUY | retest1 | 2024-02-02 10:45:00 | 1206.00 | 2024-02-02 10:55:00 | 1214.42 | PARTIAL | 0.50 | 0.70% |
| BUY | retest1 | 2024-02-02 10:45:00 | 1206.00 | 2024-02-02 11:50:00 | 1206.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-02-07 09:45:00 | 1148.05 | 2024-02-07 09:55:00 | 1153.66 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest1 | 2024-02-09 09:50:00 | 1149.75 | 2024-02-09 10:05:00 | 1155.51 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2024-02-12 10:00:00 | 1165.30 | 2024-02-12 10:20:00 | 1156.31 | PARTIAL | 0.50 | 0.77% |
| SELL | retest1 | 2024-02-12 10:00:00 | 1165.30 | 2024-02-12 15:20:00 | 1152.75 | TARGET_HIT | 0.50 | 1.08% |
| SELL | retest1 | 2024-02-13 09:45:00 | 1144.00 | 2024-02-13 11:00:00 | 1150.44 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest1 | 2024-02-22 09:35:00 | 1268.40 | 2024-02-22 09:45:00 | 1274.65 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-02-27 10:40:00 | 1298.00 | 2024-02-27 11:45:00 | 1291.91 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-03-05 10:50:00 | 1326.85 | 2024-03-05 11:20:00 | 1333.79 | PARTIAL | 0.50 | 0.52% |
| BUY | retest1 | 2024-03-05 10:50:00 | 1326.85 | 2024-03-05 12:00:00 | 1327.00 | TARGET_HIT | 0.50 | 0.01% |
| SELL | retest1 | 2024-03-19 11:10:00 | 1265.25 | 2024-03-19 11:20:00 | 1259.78 | PARTIAL | 0.50 | 0.43% |
| SELL | retest1 | 2024-03-19 11:10:00 | 1265.25 | 2024-03-19 11:55:00 | 1265.25 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-03-21 10:45:00 | 1257.40 | 2024-03-21 11:05:00 | 1253.30 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-04-22 11:05:00 | 1450.00 | 2024-04-22 12:35:00 | 1455.86 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-04-22 11:05:00 | 1450.00 | 2024-04-22 12:40:00 | 1450.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-04-26 09:30:00 | 1470.05 | 2024-04-26 09:35:00 | 1462.82 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest1 | 2024-05-07 09:35:00 | 1460.50 | 2024-05-07 09:45:00 | 1470.37 | PARTIAL | 0.50 | 0.68% |
| BUY | retest1 | 2024-05-07 09:35:00 | 1460.50 | 2024-05-07 11:35:00 | 1460.50 | STOP_HIT | 0.50 | 0.00% |
