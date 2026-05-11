# Multi Commodity Exchange of India Ltd. (MCX)

## Backtest Summary

- **Window:** 2024-05-13 09:15:00 → 2025-04-04 15:25:00 (16833 bars)
- **Last close:** 1011.80
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
| ENTRY1 | 73 |
| ENTRY2 | 0 |
| PARTIAL | 25 |
| TARGET_HIT | 12 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 98 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 61
- **Target hits / Stop hits / Partials:** 12 / 61 / 25
- **Avg / median % per leg:** 0.17% / 0.00%
- **Sum % (uncompounded):** 16.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 62 | 26 | 41.9% | 8 | 36 | 18 | 0.21% | 13.2% |
| BUY @ 2nd Alert (retest1) | 62 | 26 | 41.9% | 8 | 36 | 18 | 0.21% | 13.2% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 36 | 11 | 30.6% | 4 | 25 | 7 | 0.10% | 3.6% |
| SELL @ 2nd Alert (retest1) | 36 | 11 | 30.6% | 4 | 25 | 7 | 0.10% | 3.6% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 98 | 37 | 37.8% | 12 | 61 | 25 | 0.17% | 16.8% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-15 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-15 09:35:00 | 790.00 | 783.03 | 0.00 | ORB-long ORB[774.40,785.95] vol=3.8x ATR=3.24 |
| Stop hit — per-position SL triggered | 2024-05-15 09:45:00 | 786.76 | 786.40 | 0.00 | SL hit |

### Cycle 2 — SELL (started 2024-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-16 11:15:00 | 778.24 | 785.53 | 0.00 | ORB-short ORB[782.75,790.88] vol=2.9x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-05-16 11:20:00 | 780.71 | 785.37 | 0.00 | SL hit |

### Cycle 3 — SELL (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-17 11:15:00 | 785.60 | 785.61 | 0.00 | ORB-short ORB[786.02,792.20] vol=1.9x ATR=2.17 |
| Stop hit — per-position SL triggered | 2024-05-17 12:35:00 | 787.77 | 785.56 | 0.00 | SL hit |

### Cycle 4 — SELL (started 2024-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-27 11:15:00 | 749.00 | 750.67 | 0.00 | ORB-short ORB[749.84,758.20] vol=3.2x ATR=2.18 |
| Stop hit — per-position SL triggered | 2024-05-27 11:50:00 | 751.18 | 750.38 | 0.00 | SL hit |

### Cycle 5 — SELL (started 2024-05-30 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-30 09:35:00 | 738.54 | 742.85 | 0.00 | ORB-short ORB[740.80,747.80] vol=1.9x ATR=2.35 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 09:45:00 | 735.02 | 740.77 | 0.00 | T1 1.5R @ 735.02 |
| Target hit | 2024-05-30 10:55:00 | 736.77 | 734.76 | 0.00 | Trail-exit close>VWAP |

### Cycle 6 — SELL (started 2024-06-11 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-11 09:35:00 | 728.76 | 732.91 | 0.00 | ORB-short ORB[728.80,737.70] vol=7.4x ATR=2.99 |
| Stop hit — per-position SL triggered | 2024-06-11 09:40:00 | 731.75 | 732.71 | 0.00 | SL hit |

### Cycle 7 — BUY (started 2024-06-12 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:00:00 | 752.30 | 747.50 | 0.00 | ORB-long ORB[741.72,749.80] vol=2.8x ATR=2.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 11:45:00 | 756.04 | 749.41 | 0.00 | T1 1.5R @ 756.04 |
| Target hit | 2024-06-12 15:20:00 | 759.04 | 757.62 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 8 — BUY (started 2024-06-13 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-13 09:55:00 | 770.47 | 765.85 | 0.00 | ORB-long ORB[759.99,769.54] vol=2.6x ATR=3.16 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-13 10:05:00 | 775.21 | 769.33 | 0.00 | T1 1.5R @ 775.21 |
| Stop hit — per-position SL triggered | 2024-06-13 11:10:00 | 770.47 | 771.50 | 0.00 | SL hit |

### Cycle 9 — BUY (started 2024-06-21 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-21 10:30:00 | 778.00 | 769.84 | 0.00 | ORB-long ORB[765.20,774.20] vol=2.0x ATR=2.49 |
| Stop hit — per-position SL triggered | 2024-06-21 10:40:00 | 775.51 | 770.83 | 0.00 | SL hit |

### Cycle 10 — SELL (started 2024-06-27 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-06-27 10:05:00 | 752.29 | 756.47 | 0.00 | ORB-short ORB[754.20,763.80] vol=2.0x ATR=2.67 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-27 10:15:00 | 748.29 | 754.95 | 0.00 | T1 1.5R @ 748.29 |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 752.29 | 751.29 | 0.00 | SL hit |

### Cycle 11 — SELL (started 2024-07-02 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-02 09:30:00 | 772.00 | 775.92 | 0.00 | ORB-short ORB[772.52,781.80] vol=1.8x ATR=2.40 |
| Stop hit — per-position SL triggered | 2024-07-02 09:35:00 | 774.40 | 775.56 | 0.00 | SL hit |

### Cycle 12 — BUY (started 2024-07-03 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-03 09:55:00 | 781.57 | 779.38 | 0.00 | ORB-long ORB[774.23,779.55] vol=1.9x ATR=2.60 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-03 10:00:00 | 785.47 | 780.72 | 0.00 | T1 1.5R @ 785.47 |
| Target hit | 2024-07-03 11:10:00 | 782.61 | 782.95 | 0.00 | Trail-exit close<VWAP |

### Cycle 13 — SELL (started 2024-07-04 10:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-04 10:50:00 | 775.38 | 782.70 | 0.00 | ORB-short ORB[781.21,788.77] vol=2.1x ATR=3.23 |
| Stop hit — per-position SL triggered | 2024-07-04 10:55:00 | 778.61 | 782.60 | 0.00 | SL hit |

### Cycle 14 — BUY (started 2024-07-05 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-05 10:35:00 | 794.00 | 786.36 | 0.00 | ORB-long ORB[780.67,789.92] vol=4.6x ATR=2.94 |
| Stop hit — per-position SL triggered | 2024-07-05 11:05:00 | 791.06 | 788.78 | 0.00 | SL hit |

### Cycle 15 — SELL (started 2024-07-08 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-08 11:10:00 | 788.10 | 792.04 | 0.00 | ORB-short ORB[790.38,800.00] vol=3.2x ATR=2.20 |
| Stop hit — per-position SL triggered | 2024-07-08 11:20:00 | 790.30 | 791.82 | 0.00 | SL hit |

### Cycle 16 — BUY (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-11 11:15:00 | 752.15 | 749.47 | 0.00 | ORB-long ORB[740.12,748.40] vol=3.3x ATR=2.04 |
| Stop hit — per-position SL triggered | 2024-07-11 12:45:00 | 750.11 | 749.71 | 0.00 | SL hit |

### Cycle 17 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 10:15:00 | 758.25 | 754.41 | 0.00 | ORB-long ORB[746.00,756.00] vol=7.3x ATR=2.47 |
| Stop hit — per-position SL triggered | 2024-07-12 10:20:00 | 755.78 | 754.57 | 0.00 | SL hit |

### Cycle 18 — BUY (started 2024-07-15 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-15 09:55:00 | 765.88 | 761.76 | 0.00 | ORB-long ORB[753.55,763.55] vol=1.7x ATR=2.92 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-15 10:15:00 | 770.26 | 763.65 | 0.00 | T1 1.5R @ 770.26 |
| Target hit | 2024-07-15 15:20:00 | 780.29 | 774.98 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 19 — BUY (started 2024-07-16 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-16 10:35:00 | 790.73 | 785.70 | 0.00 | ORB-long ORB[775.60,785.97] vol=2.5x ATR=2.55 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-16 12:10:00 | 794.56 | 788.25 | 0.00 | T1 1.5R @ 794.56 |
| Stop hit — per-position SL triggered | 2024-07-16 12:30:00 | 790.73 | 788.71 | 0.00 | SL hit |

### Cycle 20 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-07-23 11:15:00 | 750.45 | 753.78 | 0.00 | ORB-short ORB[752.66,762.02] vol=2.9x ATR=2.74 |
| Stop hit — per-position SL triggered | 2024-07-23 11:20:00 | 753.19 | 753.74 | 0.00 | SL hit |

### Cycle 21 — BUY (started 2024-08-01 11:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-01 11:10:00 | 869.40 | 861.28 | 0.00 | ORB-long ORB[855.60,866.00] vol=5.3x ATR=3.38 |
| Stop hit — per-position SL triggered | 2024-08-01 11:15:00 | 866.02 | 861.54 | 0.00 | SL hit |

### Cycle 22 — BUY (started 2024-08-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-07 10:45:00 | 848.48 | 842.94 | 0.00 | ORB-long ORB[834.01,845.00] vol=2.4x ATR=3.57 |
| Stop hit — per-position SL triggered | 2024-08-07 11:10:00 | 844.91 | 843.53 | 0.00 | SL hit |

### Cycle 23 — BUY (started 2024-08-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-08 09:30:00 | 862.11 | 855.76 | 0.00 | ORB-long ORB[848.76,858.00] vol=1.8x ATR=3.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-08 09:45:00 | 866.75 | 860.57 | 0.00 | T1 1.5R @ 866.75 |
| Stop hit — per-position SL triggered | 2024-08-08 10:05:00 | 862.11 | 861.55 | 0.00 | SL hit |

### Cycle 24 — BUY (started 2024-08-12 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-12 10:45:00 | 883.57 | 876.08 | 0.00 | ORB-long ORB[869.00,878.00] vol=1.9x ATR=3.29 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 11:30:00 | 888.51 | 878.92 | 0.00 | T1 1.5R @ 888.51 |
| Stop hit — per-position SL triggered | 2024-08-12 14:50:00 | 883.57 | 882.89 | 0.00 | SL hit |

### Cycle 25 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-13 10:15:00 | 874.88 | 878.93 | 0.00 | ORB-short ORB[876.01,885.40] vol=1.5x ATR=3.00 |
| Stop hit — per-position SL triggered | 2024-08-13 10:45:00 | 877.88 | 877.97 | 0.00 | SL hit |

### Cycle 26 — BUY (started 2024-08-20 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:40:00 | 968.53 | 960.68 | 0.00 | ORB-long ORB[953.57,961.60] vol=1.5x ATR=3.17 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-20 09:45:00 | 973.28 | 965.70 | 0.00 | T1 1.5R @ 973.28 |
| Target hit | 2024-08-20 11:05:00 | 970.00 | 974.05 | 0.00 | Trail-exit close<VWAP |

### Cycle 27 — BUY (started 2024-08-23 10:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-23 10:40:00 | 964.92 | 956.91 | 0.00 | ORB-long ORB[956.00,961.96] vol=2.0x ATR=2.84 |
| Stop hit — per-position SL triggered | 2024-08-23 10:50:00 | 962.08 | 957.94 | 0.00 | SL hit |

### Cycle 28 — BUY (started 2024-08-26 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-26 09:35:00 | 982.13 | 978.23 | 0.00 | ORB-long ORB[972.10,980.98] vol=2.7x ATR=3.30 |
| Stop hit — per-position SL triggered | 2024-08-26 10:25:00 | 978.83 | 980.24 | 0.00 | SL hit |

### Cycle 29 — BUY (started 2024-08-29 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-29 09:40:00 | 997.41 | 996.45 | 0.00 | ORB-long ORB[990.35,997.34] vol=1.5x ATR=3.31 |
| Stop hit — per-position SL triggered | 2024-08-29 10:35:00 | 994.10 | 997.09 | 0.00 | SL hit |

### Cycle 30 — BUY (started 2024-08-30 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-30 10:00:00 | 1009.26 | 1003.71 | 0.00 | ORB-long ORB[998.00,1008.30] vol=2.2x ATR=3.00 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-30 10:05:00 | 1013.76 | 1006.29 | 0.00 | T1 1.5R @ 1013.76 |
| Target hit | 2024-08-30 15:20:00 | 1035.79 | 1032.31 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 31 — BUY (started 2024-09-02 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-02 09:45:00 | 1050.59 | 1041.65 | 0.00 | ORB-long ORB[1033.26,1045.79] vol=1.7x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-09-02 09:55:00 | 1046.57 | 1043.95 | 0.00 | SL hit |

### Cycle 32 — SELL (started 2024-09-06 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-06 09:55:00 | 1066.58 | 1074.48 | 0.00 | ORB-short ORB[1072.53,1083.80] vol=1.6x ATR=3.87 |
| Stop hit — per-position SL triggered | 2024-09-06 10:20:00 | 1070.45 | 1072.68 | 0.00 | SL hit |

### Cycle 33 — SELL (started 2024-09-09 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-09 09:30:00 | 1064.99 | 1068.99 | 0.00 | ORB-short ORB[1065.14,1076.42] vol=1.6x ATR=3.56 |
| Stop hit — per-position SL triggered | 2024-09-09 09:35:00 | 1068.55 | 1068.90 | 0.00 | SL hit |

### Cycle 34 — BUY (started 2024-09-10 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-10 09:35:00 | 1060.79 | 1055.88 | 0.00 | ORB-long ORB[1047.53,1058.10] vol=1.9x ATR=3.45 |
| Stop hit — per-position SL triggered | 2024-09-10 09:45:00 | 1057.34 | 1056.45 | 0.00 | SL hit |

### Cycle 35 — BUY (started 2024-09-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-11 09:30:00 | 1058.47 | 1054.02 | 0.00 | ORB-long ORB[1045.01,1057.90] vol=2.3x ATR=2.82 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-11 09:40:00 | 1062.70 | 1055.61 | 0.00 | T1 1.5R @ 1062.70 |
| Stop hit — per-position SL triggered | 2024-09-11 10:00:00 | 1058.47 | 1058.02 | 0.00 | SL hit |

### Cycle 36 — BUY (started 2024-09-17 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-17 10:45:00 | 1129.57 | 1121.92 | 0.00 | ORB-long ORB[1113.85,1126.51] vol=1.5x ATR=5.36 |
| Stop hit — per-position SL triggered | 2024-09-17 11:05:00 | 1124.21 | 1122.37 | 0.00 | SL hit |

### Cycle 37 — BUY (started 2024-09-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-20 09:30:00 | 1165.24 | 1158.30 | 0.00 | ORB-long ORB[1149.13,1164.10] vol=1.6x ATR=6.71 |
| Stop hit — per-position SL triggered | 2024-09-20 09:35:00 | 1158.53 | 1158.91 | 0.00 | SL hit |

### Cycle 38 — BUY (started 2024-09-25 10:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-25 10:00:00 | 1201.79 | 1192.34 | 0.00 | ORB-long ORB[1184.64,1200.80] vol=3.1x ATR=5.55 |
| Stop hit — per-position SL triggered | 2024-09-25 10:20:00 | 1196.24 | 1194.11 | 0.00 | SL hit |

### Cycle 39 — SELL (started 2024-10-07 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-07 10:45:00 | 1143.06 | 1162.71 | 0.00 | ORB-short ORB[1164.67,1176.17] vol=1.6x ATR=5.94 |
| Stop hit — per-position SL triggered | 2024-10-07 11:05:00 | 1149.00 | 1156.73 | 0.00 | SL hit |

### Cycle 40 — BUY (started 2024-10-11 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-11 09:30:00 | 1262.00 | 1251.66 | 0.00 | ORB-long ORB[1240.98,1257.60] vol=2.4x ATR=4.38 |
| Stop hit — per-position SL triggered | 2024-10-11 09:35:00 | 1257.62 | 1253.21 | 0.00 | SL hit |

### Cycle 41 — SELL (started 2024-10-17 09:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-17 09:55:00 | 1272.00 | 1279.61 | 0.00 | ORB-short ORB[1275.00,1289.80] vol=2.2x ATR=4.94 |
| Stop hit — per-position SL triggered | 2024-10-17 10:15:00 | 1276.94 | 1278.62 | 0.00 | SL hit |

### Cycle 42 — BUY (started 2024-10-18 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-18 09:30:00 | 1284.00 | 1266.69 | 0.00 | ORB-long ORB[1256.11,1272.54] vol=2.5x ATR=6.14 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 09:55:00 | 1293.21 | 1280.27 | 0.00 | T1 1.5R @ 1293.21 |
| Stop hit — per-position SL triggered | 2024-10-18 10:00:00 | 1284.00 | 1281.02 | 0.00 | SL hit |

### Cycle 43 — SELL (started 2024-10-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 09:30:00 | 1319.57 | 1329.34 | 0.00 | ORB-short ORB[1321.04,1339.68] vol=3.0x ATR=8.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:50:00 | 1307.45 | 1323.83 | 0.00 | T1 1.5R @ 1307.45 |
| Target hit | 2024-10-22 15:20:00 | 1290.65 | 1302.71 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 44 — SELL (started 2024-10-25 10:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 10:45:00 | 1269.04 | 1306.67 | 0.00 | ORB-short ORB[1319.49,1335.87] vol=1.5x ATR=7.89 |
| Stop hit — per-position SL triggered | 2024-10-25 10:50:00 | 1276.93 | 1305.14 | 0.00 | SL hit |

### Cycle 45 — SELL (started 2024-11-05 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-05 09:30:00 | 1279.45 | 1287.07 | 0.00 | ORB-short ORB[1280.00,1297.11] vol=2.8x ATR=6.98 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:35:00 | 1268.98 | 1282.95 | 0.00 | T1 1.5R @ 1268.98 |
| Target hit | 2024-11-05 13:25:00 | 1262.28 | 1256.29 | 0.00 | Trail-exit close>VWAP |

### Cycle 46 — SELL (started 2024-11-06 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-06 10:25:00 | 1289.01 | 1306.31 | 0.00 | ORB-short ORB[1299.61,1313.96] vol=1.8x ATR=7.27 |
| Stop hit — per-position SL triggered | 2024-11-06 10:45:00 | 1296.28 | 1304.34 | 0.00 | SL hit |

### Cycle 47 — BUY (started 2024-11-08 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-08 09:30:00 | 1290.72 | 1285.41 | 0.00 | ORB-long ORB[1275.34,1290.00] vol=1.6x ATR=5.03 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-08 09:45:00 | 1298.27 | 1288.42 | 0.00 | T1 1.5R @ 1298.27 |
| Stop hit — per-position SL triggered | 2024-11-08 09:50:00 | 1290.72 | 1288.60 | 0.00 | SL hit |

### Cycle 48 — BUY (started 2024-11-11 10:25:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-11 10:25:00 | 1293.04 | 1281.52 | 0.00 | ORB-long ORB[1266.00,1279.63] vol=2.4x ATR=4.45 |
| Stop hit — per-position SL triggered | 2024-11-11 10:35:00 | 1288.59 | 1282.56 | 0.00 | SL hit |

### Cycle 49 — SELL (started 2024-11-12 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-12 09:40:00 | 1271.43 | 1272.72 | 0.00 | ORB-short ORB[1272.22,1289.49] vol=5.1x ATR=4.26 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 09:50:00 | 1265.04 | 1270.82 | 0.00 | T1 1.5R @ 1265.04 |
| Target hit | 2024-11-12 15:20:00 | 1220.74 | 1244.08 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 50 — BUY (started 2024-11-19 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-19 09:40:00 | 1207.43 | 1201.20 | 0.00 | ORB-long ORB[1186.03,1200.00] vol=1.9x ATR=5.06 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 10:35:00 | 1215.02 | 1205.15 | 0.00 | T1 1.5R @ 1215.02 |
| Target hit | 2024-11-19 15:20:00 | 1220.31 | 1217.11 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 51 — BUY (started 2024-11-28 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-28 09:50:00 | 1240.22 | 1230.77 | 0.00 | ORB-long ORB[1219.53,1233.00] vol=2.3x ATR=4.29 |
| Stop hit — per-position SL triggered | 2024-11-28 10:05:00 | 1235.93 | 1233.86 | 0.00 | SL hit |

### Cycle 52 — BUY (started 2024-12-05 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-05 10:30:00 | 1287.18 | 1271.20 | 0.00 | ORB-long ORB[1258.28,1273.80] vol=4.0x ATR=4.88 |
| Stop hit — per-position SL triggered | 2024-12-05 10:55:00 | 1282.30 | 1275.96 | 0.00 | SL hit |

### Cycle 53 — BUY (started 2024-12-06 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-06 11:00:00 | 1318.46 | 1300.81 | 0.00 | ORB-long ORB[1290.00,1309.34] vol=1.6x ATR=5.30 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-06 11:10:00 | 1326.42 | 1304.21 | 0.00 | T1 1.5R @ 1326.42 |
| Target hit | 2024-12-06 15:20:00 | 1382.60 | 1374.14 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 54 — BUY (started 2024-12-09 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-09 11:00:00 | 1399.40 | 1391.91 | 0.00 | ORB-long ORB[1383.01,1397.59] vol=1.9x ATR=5.96 |
| Stop hit — per-position SL triggered | 2024-12-09 11:05:00 | 1393.44 | 1391.99 | 0.00 | SL hit |

### Cycle 55 — SELL (started 2024-12-10 10:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-10 10:30:00 | 1337.80 | 1354.84 | 0.00 | ORB-short ORB[1358.00,1373.80] vol=1.5x ATR=5.49 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 11:00:00 | 1329.57 | 1349.72 | 0.00 | T1 1.5R @ 1329.57 |
| Stop hit — per-position SL triggered | 2024-12-10 11:30:00 | 1337.80 | 1347.34 | 0.00 | SL hit |

### Cycle 56 — BUY (started 2024-12-12 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-12 09:35:00 | 1362.00 | 1356.11 | 0.00 | ORB-long ORB[1342.22,1360.00] vol=2.0x ATR=4.08 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-12 09:50:00 | 1368.12 | 1359.31 | 0.00 | T1 1.5R @ 1368.12 |
| Stop hit — per-position SL triggered | 2024-12-12 10:00:00 | 1362.00 | 1359.59 | 0.00 | SL hit |

### Cycle 57 — BUY (started 2024-12-17 10:55:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-17 10:55:00 | 1355.71 | 1343.64 | 0.00 | ORB-long ORB[1337.64,1350.18] vol=2.7x ATR=4.38 |
| Stop hit — per-position SL triggered | 2024-12-17 11:00:00 | 1351.33 | 1344.17 | 0.00 | SL hit |

### Cycle 58 — SELL (started 2024-12-24 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-24 09:30:00 | 1263.49 | 1268.16 | 0.00 | ORB-short ORB[1264.02,1279.79] vol=1.9x ATR=4.02 |
| Stop hit — per-position SL triggered | 2024-12-24 09:35:00 | 1267.51 | 1268.05 | 0.00 | SL hit |

### Cycle 59 — BUY (started 2024-12-30 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-30 09:45:00 | 1282.00 | 1272.36 | 0.00 | ORB-long ORB[1262.76,1274.06] vol=2.9x ATR=4.85 |
| Stop hit — per-position SL triggered | 2024-12-30 10:10:00 | 1277.15 | 1275.73 | 0.00 | SL hit |

### Cycle 60 — SELL (started 2025-01-02 09:50:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-02 09:50:00 | 1249.31 | 1253.63 | 0.00 | ORB-short ORB[1252.01,1265.00] vol=2.4x ATR=3.59 |
| Stop hit — per-position SL triggered | 2025-01-02 10:00:00 | 1252.90 | 1253.09 | 0.00 | SL hit |

### Cycle 61 — SELL (started 2025-01-03 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-03 09:30:00 | 1265.60 | 1273.19 | 0.00 | ORB-short ORB[1267.20,1282.68] vol=2.2x ATR=3.96 |
| Stop hit — per-position SL triggered | 2025-01-03 09:40:00 | 1269.56 | 1272.02 | 0.00 | SL hit |

### Cycle 62 — SELL (started 2025-01-07 09:45:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-07 09:45:00 | 1155.54 | 1167.43 | 0.00 | ORB-short ORB[1162.67,1176.92] vol=1.6x ATR=5.95 |
| Stop hit — per-position SL triggered | 2025-01-07 10:10:00 | 1161.49 | 1161.97 | 0.00 | SL hit |

### Cycle 63 — SELL (started 2025-01-10 10:05:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-10 10:05:00 | 1156.00 | 1160.13 | 0.00 | ORB-short ORB[1163.01,1172.80] vol=2.2x ATR=4.32 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 10:30:00 | 1149.52 | 1158.82 | 0.00 | T1 1.5R @ 1149.52 |
| Stop hit — per-position SL triggered | 2025-01-10 11:10:00 | 1156.00 | 1156.92 | 0.00 | SL hit |

### Cycle 64 — BUY (started 2025-01-16 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-16 09:35:00 | 1224.43 | 1214.71 | 0.00 | ORB-long ORB[1202.15,1217.55] vol=2.2x ATR=5.33 |
| Stop hit — per-position SL triggered | 2025-01-16 09:40:00 | 1219.10 | 1215.13 | 0.00 | SL hit |

### Cycle 65 — BUY (started 2025-01-29 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-29 09:35:00 | 1127.60 | 1114.25 | 0.00 | ORB-long ORB[1103.10,1116.59] vol=1.9x ATR=6.69 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-29 10:05:00 | 1137.63 | 1121.05 | 0.00 | T1 1.5R @ 1137.63 |
| Target hit | 2025-01-29 15:20:00 | 1140.33 | 1134.42 | 0.00 | EOD square-off @ 15:20:00 |

### Cycle 66 — BUY (started 2025-01-31 10:20:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-01-31 10:20:00 | 1140.86 | 1128.95 | 0.00 | ORB-long ORB[1117.51,1127.80] vol=1.9x ATR=4.45 |
| Stop hit — per-position SL triggered | 2025-01-31 10:50:00 | 1136.41 | 1130.76 | 0.00 | SL hit |

### Cycle 67 — BUY (started 2025-02-01 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 09:30:00 | 1160.06 | 1150.43 | 0.00 | ORB-long ORB[1140.07,1151.99] vol=2.4x ATR=4.47 |
| Stop hit — per-position SL triggered | 2025-02-01 09:40:00 | 1155.59 | 1152.50 | 0.00 | SL hit |

### Cycle 68 — SELL (started 2025-02-04 10:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-02-04 10:35:00 | 1133.12 | 1144.89 | 0.00 | ORB-short ORB[1143.89,1155.26] vol=3.4x ATR=4.22 |
| Stop hit — per-position SL triggered | 2025-02-04 11:15:00 | 1137.34 | 1140.80 | 0.00 | SL hit |

### Cycle 69 — BUY (started 2025-02-07 11:00:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-07 11:00:00 | 1215.66 | 1203.88 | 0.00 | ORB-long ORB[1196.24,1210.18] vol=1.6x ATR=5.16 |
| Stop hit — per-position SL triggered | 2025-02-07 12:15:00 | 1210.50 | 1210.90 | 0.00 | SL hit |

### Cycle 70 — BUY (started 2025-02-20 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-20 09:30:00 | 1140.00 | 1133.44 | 0.00 | ORB-long ORB[1120.30,1136.83] vol=3.5x ATR=5.10 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-20 09:40:00 | 1147.64 | 1137.97 | 0.00 | T1 1.5R @ 1147.64 |
| Stop hit — per-position SL triggered | 2025-02-20 09:45:00 | 1140.00 | 1138.23 | 0.00 | SL hit |

### Cycle 71 — BUY (started 2025-03-18 10:10:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 10:10:00 | 1018.44 | 1007.36 | 0.00 | ORB-long ORB[995.00,1008.72] vol=1.8x ATR=4.21 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-18 11:25:00 | 1024.75 | 1012.13 | 0.00 | T1 1.5R @ 1024.75 |
| Stop hit — per-position SL triggered | 2025-03-18 12:30:00 | 1018.44 | 1013.91 | 0.00 | SL hit |

### Cycle 72 — SELL (started 2025-03-24 09:35:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-24 09:35:00 | 1086.44 | 1100.39 | 0.00 | ORB-short ORB[1095.00,1107.80] vol=1.5x ATR=4.62 |
| Stop hit — per-position SL triggered | 2025-03-24 09:50:00 | 1091.06 | 1096.48 | 0.00 | SL hit |

### Cycle 73 — SELL (started 2025-03-26 09:40:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-26 09:40:00 | 1016.84 | 1031.11 | 0.00 | ORB-short ORB[1026.20,1040.97] vol=1.9x ATR=5.12 |
| Stop hit — per-position SL triggered | 2025-03-26 09:45:00 | 1021.96 | 1030.01 | 0.00 | SL hit |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2024-05-15 09:35:00 | 790.00 | 2024-05-15 09:45:00 | 786.76 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest1 | 2024-05-16 11:15:00 | 778.24 | 2024-05-16 11:20:00 | 780.71 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-05-17 11:15:00 | 785.60 | 2024-05-17 12:35:00 | 787.77 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest1 | 2024-05-27 11:15:00 | 749.00 | 2024-05-27 11:50:00 | 751.18 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2024-05-30 09:35:00 | 738.54 | 2024-05-30 09:45:00 | 735.02 | PARTIAL | 0.50 | 0.48% |
| SELL | retest1 | 2024-05-30 09:35:00 | 738.54 | 2024-05-30 10:55:00 | 736.77 | TARGET_HIT | 0.50 | 0.24% |
| SELL | retest1 | 2024-06-11 09:35:00 | 728.76 | 2024-06-11 09:40:00 | 731.75 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest1 | 2024-06-12 11:00:00 | 752.30 | 2024-06-12 11:45:00 | 756.04 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-06-12 11:00:00 | 752.30 | 2024-06-12 15:20:00 | 759.04 | TARGET_HIT | 0.50 | 0.90% |
| BUY | retest1 | 2024-06-13 09:55:00 | 770.47 | 2024-06-13 10:05:00 | 775.21 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2024-06-13 09:55:00 | 770.47 | 2024-06-13 11:10:00 | 770.47 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-06-21 10:30:00 | 778.00 | 2024-06-21 10:40:00 | 775.51 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-06-27 10:05:00 | 752.29 | 2024-06-27 10:15:00 | 748.29 | PARTIAL | 0.50 | 0.53% |
| SELL | retest1 | 2024-06-27 10:05:00 | 752.29 | 2024-06-27 11:15:00 | 752.29 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-02 09:30:00 | 772.00 | 2024-07-02 09:35:00 | 774.40 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-03 09:55:00 | 781.57 | 2024-07-03 10:00:00 | 785.47 | PARTIAL | 0.50 | 0.50% |
| BUY | retest1 | 2024-07-03 09:55:00 | 781.57 | 2024-07-03 11:10:00 | 782.61 | TARGET_HIT | 0.50 | 0.13% |
| SELL | retest1 | 2024-07-04 10:50:00 | 775.38 | 2024-07-04 10:55:00 | 778.61 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-07-05 10:35:00 | 794.00 | 2024-07-05 11:05:00 | 791.06 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest1 | 2024-07-08 11:10:00 | 788.10 | 2024-07-08 11:20:00 | 790.30 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest1 | 2024-07-11 11:15:00 | 752.15 | 2024-07-11 12:45:00 | 750.11 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest1 | 2024-07-12 10:15:00 | 758.25 | 2024-07-12 10:20:00 | 755.78 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-07-15 09:55:00 | 765.88 | 2024-07-15 10:15:00 | 770.26 | PARTIAL | 0.50 | 0.57% |
| BUY | retest1 | 2024-07-15 09:55:00 | 765.88 | 2024-07-15 15:20:00 | 780.29 | TARGET_HIT | 0.50 | 1.88% |
| BUY | retest1 | 2024-07-16 10:35:00 | 790.73 | 2024-07-16 12:10:00 | 794.56 | PARTIAL | 0.50 | 0.48% |
| BUY | retest1 | 2024-07-16 10:35:00 | 790.73 | 2024-07-16 12:30:00 | 790.73 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-07-23 11:15:00 | 750.45 | 2024-07-23 11:20:00 | 753.19 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2024-08-01 11:10:00 | 869.40 | 2024-08-01 11:15:00 | 866.02 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-08-07 10:45:00 | 848.48 | 2024-08-07 11:10:00 | 844.91 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2024-08-08 09:30:00 | 862.11 | 2024-08-08 09:45:00 | 866.75 | PARTIAL | 0.50 | 0.54% |
| BUY | retest1 | 2024-08-08 09:30:00 | 862.11 | 2024-08-08 10:05:00 | 862.11 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-08-12 10:45:00 | 883.57 | 2024-08-12 11:30:00 | 888.51 | PARTIAL | 0.50 | 0.56% |
| BUY | retest1 | 2024-08-12 10:45:00 | 883.57 | 2024-08-12 14:50:00 | 883.57 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-08-13 10:15:00 | 874.88 | 2024-08-13 10:45:00 | 877.88 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-20 09:40:00 | 968.53 | 2024-08-20 09:45:00 | 973.28 | PARTIAL | 0.50 | 0.49% |
| BUY | retest1 | 2024-08-20 09:40:00 | 968.53 | 2024-08-20 11:05:00 | 970.00 | TARGET_HIT | 0.50 | 0.15% |
| BUY | retest1 | 2024-08-23 10:40:00 | 964.92 | 2024-08-23 10:50:00 | 962.08 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest1 | 2024-08-26 09:35:00 | 982.13 | 2024-08-26 10:25:00 | 978.83 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest1 | 2024-08-29 09:40:00 | 997.41 | 2024-08-29 10:35:00 | 994.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-08-30 10:00:00 | 1009.26 | 2024-08-30 10:05:00 | 1013.76 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-08-30 10:00:00 | 1009.26 | 2024-08-30 15:20:00 | 1035.79 | TARGET_HIT | 0.50 | 2.63% |
| BUY | retest1 | 2024-09-02 09:45:00 | 1050.59 | 2024-09-02 09:55:00 | 1046.57 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2024-09-06 09:55:00 | 1066.58 | 2024-09-06 10:20:00 | 1070.45 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest1 | 2024-09-09 09:30:00 | 1064.99 | 2024-09-09 09:35:00 | 1068.55 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-10 09:35:00 | 1060.79 | 2024-09-10 09:45:00 | 1057.34 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest1 | 2024-09-11 09:30:00 | 1058.47 | 2024-09-11 09:40:00 | 1062.70 | PARTIAL | 0.50 | 0.40% |
| BUY | retest1 | 2024-09-11 09:30:00 | 1058.47 | 2024-09-11 10:00:00 | 1058.47 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-09-17 10:45:00 | 1129.57 | 2024-09-17 11:05:00 | 1124.21 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest1 | 2024-09-20 09:30:00 | 1165.24 | 2024-09-20 09:35:00 | 1158.53 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2024-09-25 10:00:00 | 1201.79 | 2024-09-25 10:20:00 | 1196.24 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest1 | 2024-10-07 10:45:00 | 1143.06 | 2024-10-07 11:05:00 | 1149.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest1 | 2024-10-11 09:30:00 | 1262.00 | 2024-10-11 09:35:00 | 1257.62 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest1 | 2024-10-17 09:55:00 | 1272.00 | 2024-10-17 10:15:00 | 1276.94 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2024-10-18 09:30:00 | 1284.00 | 2024-10-18 09:55:00 | 1293.21 | PARTIAL | 0.50 | 0.72% |
| BUY | retest1 | 2024-10-18 09:30:00 | 1284.00 | 2024-10-18 10:00:00 | 1284.00 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2024-10-22 09:30:00 | 1319.57 | 2024-10-22 09:50:00 | 1307.45 | PARTIAL | 0.50 | 0.92% |
| SELL | retest1 | 2024-10-22 09:30:00 | 1319.57 | 2024-10-22 15:20:00 | 1290.65 | TARGET_HIT | 0.50 | 2.19% |
| SELL | retest1 | 2024-10-25 10:45:00 | 1269.04 | 2024-10-25 10:50:00 | 1276.93 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2024-11-05 09:30:00 | 1279.45 | 2024-11-05 09:35:00 | 1268.98 | PARTIAL | 0.50 | 0.82% |
| SELL | retest1 | 2024-11-05 09:30:00 | 1279.45 | 2024-11-05 13:25:00 | 1262.28 | TARGET_HIT | 0.50 | 1.34% |
| SELL | retest1 | 2024-11-06 10:25:00 | 1289.01 | 2024-11-06 10:45:00 | 1296.28 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest1 | 2024-11-08 09:30:00 | 1290.72 | 2024-11-08 09:45:00 | 1298.27 | PARTIAL | 0.50 | 0.58% |
| BUY | retest1 | 2024-11-08 09:30:00 | 1290.72 | 2024-11-08 09:50:00 | 1290.72 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-11-11 10:25:00 | 1293.04 | 2024-11-11 10:35:00 | 1288.59 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest1 | 2024-11-12 09:40:00 | 1271.43 | 2024-11-12 09:50:00 | 1265.04 | PARTIAL | 0.50 | 0.50% |
| SELL | retest1 | 2024-11-12 09:40:00 | 1271.43 | 2024-11-12 15:20:00 | 1220.74 | TARGET_HIT | 0.50 | 3.99% |
| BUY | retest1 | 2024-11-19 09:40:00 | 1207.43 | 2024-11-19 10:35:00 | 1215.02 | PARTIAL | 0.50 | 0.63% |
| BUY | retest1 | 2024-11-19 09:40:00 | 1207.43 | 2024-11-19 15:20:00 | 1220.31 | TARGET_HIT | 0.50 | 1.07% |
| BUY | retest1 | 2024-11-28 09:50:00 | 1240.22 | 2024-11-28 10:05:00 | 1235.93 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest1 | 2024-12-05 10:30:00 | 1287.18 | 2024-12-05 10:55:00 | 1282.30 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2024-12-06 11:00:00 | 1318.46 | 2024-12-06 11:10:00 | 1326.42 | PARTIAL | 0.50 | 0.60% |
| BUY | retest1 | 2024-12-06 11:00:00 | 1318.46 | 2024-12-06 15:20:00 | 1382.60 | TARGET_HIT | 0.50 | 4.86% |
| BUY | retest1 | 2024-12-09 11:00:00 | 1399.40 | 2024-12-09 11:05:00 | 1393.44 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2024-12-10 10:30:00 | 1337.80 | 2024-12-10 11:00:00 | 1329.57 | PARTIAL | 0.50 | 0.62% |
| SELL | retest1 | 2024-12-10 10:30:00 | 1337.80 | 2024-12-10 11:30:00 | 1337.80 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-12 09:35:00 | 1362.00 | 2024-12-12 09:50:00 | 1368.12 | PARTIAL | 0.50 | 0.45% |
| BUY | retest1 | 2024-12-12 09:35:00 | 1362.00 | 2024-12-12 10:00:00 | 1362.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2024-12-17 10:55:00 | 1355.71 | 2024-12-17 11:00:00 | 1351.33 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest1 | 2024-12-24 09:30:00 | 1263.49 | 2024-12-24 09:35:00 | 1267.51 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2024-12-30 09:45:00 | 1282.00 | 2024-12-30 10:10:00 | 1277.15 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest1 | 2025-01-02 09:50:00 | 1249.31 | 2025-01-02 10:00:00 | 1252.90 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest1 | 2025-01-03 09:30:00 | 1265.60 | 2025-01-03 09:40:00 | 1269.56 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest1 | 2025-01-07 09:45:00 | 1155.54 | 2025-01-07 10:10:00 | 1161.49 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest1 | 2025-01-10 10:05:00 | 1156.00 | 2025-01-10 10:30:00 | 1149.52 | PARTIAL | 0.50 | 0.56% |
| SELL | retest1 | 2025-01-10 10:05:00 | 1156.00 | 2025-01-10 11:10:00 | 1156.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-01-16 09:35:00 | 1224.43 | 2025-01-16 09:40:00 | 1219.10 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest1 | 2025-01-29 09:35:00 | 1127.60 | 2025-01-29 10:05:00 | 1137.63 | PARTIAL | 0.50 | 0.89% |
| BUY | retest1 | 2025-01-29 09:35:00 | 1127.60 | 2025-01-29 15:20:00 | 1140.33 | TARGET_HIT | 0.50 | 1.13% |
| BUY | retest1 | 2025-01-31 10:20:00 | 1140.86 | 2025-01-31 10:50:00 | 1136.41 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-02-01 09:30:00 | 1160.06 | 2025-02-01 09:40:00 | 1155.59 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest1 | 2025-02-04 10:35:00 | 1133.12 | 2025-02-04 11:15:00 | 1137.34 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest1 | 2025-02-07 11:00:00 | 1215.66 | 2025-02-07 12:15:00 | 1210.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest1 | 2025-02-20 09:30:00 | 1140.00 | 2025-02-20 09:40:00 | 1147.64 | PARTIAL | 0.50 | 0.67% |
| BUY | retest1 | 2025-02-20 09:30:00 | 1140.00 | 2025-02-20 09:45:00 | 1140.00 | STOP_HIT | 0.50 | 0.00% |
| BUY | retest1 | 2025-03-18 10:10:00 | 1018.44 | 2025-03-18 11:25:00 | 1024.75 | PARTIAL | 0.50 | 0.62% |
| BUY | retest1 | 2025-03-18 10:10:00 | 1018.44 | 2025-03-18 12:30:00 | 1018.44 | STOP_HIT | 0.50 | 0.00% |
| SELL | retest1 | 2025-03-24 09:35:00 | 1086.44 | 2025-03-24 09:50:00 | 1091.06 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest1 | 2025-03-26 09:40:00 | 1016.84 | 2025-03-26 09:45:00 | 1021.96 | STOP_HIT | 1.00 | -0.50% |
