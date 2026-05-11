# United Spirits Ltd. (UNITDSPR)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1284.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 243 |
| ALERT1 | 168 |
| ALERT2 | 168 |
| ALERT2_SKIP | 106 |
| ALERT3 | 392 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 149 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 153 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 155 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 33 / 122
- **Target hits / Stop hits / Partials:** 0 / 153 / 2
- **Avg / median % per leg:** -0.60% / -0.83%
- **Sum % (uncompounded):** -93.41%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 14 | 18.4% | 0 | 76 | 0 | -0.73% | -55.5% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.75% | -0.7% |
| BUY @ 3rd Alert (retest2) | 75 | 14 | 18.7% | 0 | 75 | 0 | -0.73% | -54.7% |
| SELL (all) | 79 | 19 | 24.1% | 0 | 77 | 2 | -0.48% | -37.9% |
| SELL @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 6 | 0 | -0.53% | -3.2% |
| SELL @ 3rd Alert (retest2) | 73 | 18 | 24.7% | 0 | 71 | 2 | -0.48% | -34.7% |
| retest1 (combined) | 7 | 1 | 14.3% | 0 | 7 | 0 | -0.56% | -3.9% |
| retest2 (combined) | 148 | 32 | 21.6% | 0 | 146 | 2 | -0.60% | -89.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 11:15:00 | 793.90 | 800.28 | 800.83 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 10:15:00 | 816.35 | 802.43 | 800.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 10:15:00 | 820.20 | 814.13 | 808.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 12:15:00 | 821.60 | 821.95 | 817.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 09:15:00 | 830.15 | 823.17 | 819.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 830.15 | 823.17 | 819.10 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-05 12:15:00 | 871.95 | 873.15 | 873.24 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-05 13:15:00 | 876.65 | 873.85 | 873.55 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-06 10:15:00 | 870.35 | 873.58 | 873.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-06 11:15:00 | 868.45 | 872.55 | 873.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-06 15:15:00 | 869.75 | 869.60 | 871.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-07 09:15:00 | 874.65 | 870.61 | 871.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-07 09:15:00 | 874.65 | 870.61 | 871.62 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-07 14:15:00 | 875.75 | 872.04 | 872.01 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 11:15:00 | 867.90 | 871.91 | 872.05 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 09:15:00 | 874.05 | 872.07 | 872.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-09 13:15:00 | 883.35 | 875.25 | 873.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-12 09:15:00 | 876.00 | 878.76 | 875.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 876.00 | 878.76 | 875.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 876.00 | 878.76 | 875.91 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 10:15:00 | 894.00 | 899.72 | 900.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-20 14:15:00 | 891.15 | 895.38 | 897.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 09:15:00 | 905.50 | 897.29 | 898.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 905.50 | 897.29 | 898.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 905.50 | 897.29 | 898.18 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 12:15:00 | 901.00 | 898.67 | 898.62 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 14:15:00 | 896.40 | 898.38 | 898.51 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-22 09:15:00 | 901.55 | 898.79 | 898.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-22 10:15:00 | 904.25 | 899.89 | 899.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 12:15:00 | 897.25 | 899.57 | 899.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 12:15:00 | 897.25 | 899.57 | 899.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 897.25 | 899.57 | 899.16 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 14:15:00 | 894.20 | 898.47 | 898.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 890.75 | 896.09 | 897.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 11:15:00 | 900.10 | 896.72 | 897.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 11:15:00 | 900.10 | 896.72 | 897.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 11:15:00 | 900.10 | 896.72 | 897.57 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 12:15:00 | 905.40 | 898.45 | 898.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-23 14:15:00 | 907.15 | 901.28 | 899.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-26 12:15:00 | 900.00 | 902.77 | 901.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-26 12:15:00 | 900.00 | 902.77 | 901.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-26 12:15:00 | 900.00 | 902.77 | 901.21 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 13:15:00 | 915.55 | 929.69 | 929.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 15:15:00 | 914.35 | 924.33 | 927.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 10:15:00 | 917.10 | 916.11 | 920.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 11:15:00 | 920.05 | 916.90 | 920.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 11:15:00 | 920.05 | 916.90 | 920.07 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 09:15:00 | 937.70 | 924.23 | 922.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 13:15:00 | 942.45 | 933.22 | 929.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 10:15:00 | 957.50 | 962.56 | 952.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 09:15:00 | 963.95 | 968.31 | 963.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 963.95 | 968.31 | 963.21 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 15:15:00 | 977.60 | 989.22 | 990.34 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 11:15:00 | 992.80 | 987.32 | 986.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 12:15:00 | 995.75 | 989.00 | 987.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-28 09:15:00 | 992.90 | 992.93 | 990.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 1006.60 | 1016.10 | 1012.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 1006.60 | 1016.10 | 1012.02 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 12:15:00 | 998.50 | 1008.08 | 1008.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 987.95 | 1004.05 | 1007.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 09:15:00 | 1007.95 | 1004.64 | 1006.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 09:15:00 | 1007.95 | 1004.64 | 1006.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 09:15:00 | 1007.95 | 1004.64 | 1006.55 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2023-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 09:15:00 | 1006.95 | 1003.80 | 1003.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 10:15:00 | 1017.50 | 1006.54 | 1004.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-09 09:15:00 | 1012.20 | 1013.42 | 1009.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 12:15:00 | 1011.40 | 1013.82 | 1010.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 12:15:00 | 1011.40 | 1013.82 | 1010.88 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 09:15:00 | 1003.40 | 1010.64 | 1011.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 10:15:00 | 994.65 | 1007.44 | 1009.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-14 11:15:00 | 994.40 | 993.80 | 999.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 14:15:00 | 995.15 | 987.61 | 991.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 14:15:00 | 995.15 | 987.61 | 991.71 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 11:15:00 | 998.95 | 993.48 | 993.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 14:15:00 | 1002.30 | 997.01 | 995.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 12:15:00 | 998.60 | 999.36 | 997.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 13:15:00 | 999.05 | 999.30 | 997.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 13:15:00 | 999.05 | 999.30 | 997.49 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 09:15:00 | 991.05 | 996.01 | 996.30 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2023-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 12:15:00 | 1004.55 | 997.80 | 996.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 09:15:00 | 1013.50 | 1002.83 | 999.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 12:15:00 | 1019.10 | 1025.42 | 1017.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 13:15:00 | 1024.15 | 1025.17 | 1018.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 1024.15 | 1025.17 | 1018.35 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 1005.00 | 1016.40 | 1017.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 10:15:00 | 1002.80 | 1008.10 | 1011.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 1011.20 | 1004.46 | 1007.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 1011.20 | 1004.46 | 1007.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 1011.20 | 1004.46 | 1007.79 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 10:15:00 | 1013.00 | 1009.12 | 1008.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 11:15:00 | 1019.20 | 1011.13 | 1009.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 10:15:00 | 1011.40 | 1012.64 | 1011.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 10:15:00 | 1011.40 | 1012.64 | 1011.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 1011.40 | 1012.64 | 1011.29 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2023-08-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 13:15:00 | 1002.15 | 1008.89 | 1009.79 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-01 09:15:00 | 1019.30 | 1010.66 | 1010.33 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 09:15:00 | 1005.35 | 1010.98 | 1011.02 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2023-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 12:15:00 | 1022.05 | 1012.64 | 1011.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 13:15:00 | 1029.80 | 1016.07 | 1013.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 09:15:00 | 1028.15 | 1028.65 | 1023.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 09:15:00 | 1028.15 | 1028.65 | 1023.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 1028.15 | 1028.65 | 1023.41 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 1041.90 | 1053.80 | 1053.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 14:15:00 | 1037.30 | 1047.77 | 1050.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 1038.00 | 1034.63 | 1040.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 1038.00 | 1034.63 | 1040.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 1038.00 | 1034.63 | 1040.38 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 1055.05 | 1044.34 | 1043.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 14:15:00 | 1066.85 | 1051.00 | 1047.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 10:15:00 | 1053.45 | 1055.08 | 1050.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 10:15:00 | 1053.45 | 1055.08 | 1050.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 10:15:00 | 1053.45 | 1055.08 | 1050.28 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 09:15:00 | 1039.90 | 1048.94 | 1048.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 13:15:00 | 1033.60 | 1039.21 | 1042.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 14:15:00 | 1039.60 | 1039.29 | 1042.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 1016.60 | 1019.21 | 1024.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 1016.60 | 1019.21 | 1024.00 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-10-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-05 14:15:00 | 1012.60 | 1000.86 | 999.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 09:15:00 | 1017.10 | 1005.73 | 1002.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 1017.00 | 1017.49 | 1010.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 10:15:00 | 1014.30 | 1016.85 | 1011.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 10:15:00 | 1014.30 | 1016.85 | 1011.24 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 12:15:00 | 1059.90 | 1070.20 | 1071.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 14:15:00 | 1054.00 | 1065.31 | 1068.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 10:15:00 | 1028.65 | 1023.20 | 1032.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1030.55 | 1014.25 | 1017.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1030.55 | 1014.25 | 1017.76 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-10-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 11:15:00 | 1034.10 | 1021.05 | 1020.42 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2023-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-30 09:15:00 | 1013.50 | 1020.76 | 1020.82 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 11:15:00 | 1023.50 | 1021.24 | 1021.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 12:15:00 | 1026.00 | 1022.19 | 1021.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-31 09:15:00 | 1019.50 | 1023.55 | 1022.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-31 09:15:00 | 1019.50 | 1023.55 | 1022.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-31 09:15:00 | 1019.50 | 1023.55 | 1022.53 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 1032.35 | 1082.87 | 1083.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 10:15:00 | 1026.35 | 1071.56 | 1077.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 1047.75 | 1046.77 | 1053.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 1047.75 | 1046.77 | 1053.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 1047.75 | 1046.77 | 1053.87 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-17 11:15:00 | 1056.50 | 1048.87 | 1048.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-17 15:15:00 | 1058.00 | 1053.41 | 1050.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-21 09:15:00 | 1056.00 | 1057.81 | 1055.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 1056.00 | 1057.81 | 1055.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 1056.00 | 1057.81 | 1055.11 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-21 14:15:00 | 1053.00 | 1053.79 | 1053.83 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-22 09:15:00 | 1055.25 | 1054.12 | 1053.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-22 11:15:00 | 1058.00 | 1055.22 | 1054.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-22 12:15:00 | 1054.00 | 1054.98 | 1054.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-22 12:15:00 | 1054.00 | 1054.98 | 1054.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 12:15:00 | 1054.00 | 1054.98 | 1054.47 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 1049.00 | 1056.73 | 1056.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 13:15:00 | 1046.85 | 1054.76 | 1055.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 15:15:00 | 1042.50 | 1042.08 | 1046.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 1042.95 | 1042.25 | 1046.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 1042.95 | 1042.25 | 1046.28 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 09:15:00 | 1057.85 | 1048.87 | 1047.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-05 09:15:00 | 1073.15 | 1055.44 | 1052.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-06 09:15:00 | 1065.80 | 1065.82 | 1060.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 11:15:00 | 1059.60 | 1064.22 | 1060.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 11:15:00 | 1059.60 | 1064.22 | 1060.75 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 11:15:00 | 1050.75 | 1061.16 | 1061.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 1036.30 | 1056.19 | 1059.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-12 09:15:00 | 1049.45 | 1048.19 | 1051.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 1049.45 | 1048.19 | 1051.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 1049.45 | 1048.19 | 1051.37 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 11:15:00 | 1055.80 | 1052.32 | 1051.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 12:15:00 | 1060.00 | 1053.85 | 1052.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 10:15:00 | 1079.80 | 1081.97 | 1073.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 1069.20 | 1080.61 | 1076.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 1069.20 | 1080.61 | 1076.63 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 13:15:00 | 1066.00 | 1084.21 | 1084.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 1057.35 | 1078.83 | 1082.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 1081.05 | 1066.09 | 1070.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 1081.05 | 1066.09 | 1070.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 1081.05 | 1066.09 | 1070.98 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 1091.40 | 1076.50 | 1074.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-29 14:15:00 | 1117.00 | 1098.42 | 1091.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-01 14:15:00 | 1107.55 | 1109.40 | 1101.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 15:15:00 | 1104.00 | 1108.32 | 1102.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 15:15:00 | 1104.00 | 1108.32 | 1102.12 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 11:15:00 | 1080.50 | 1097.58 | 1098.35 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 13:15:00 | 1103.10 | 1097.17 | 1097.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-04 09:15:00 | 1109.55 | 1100.08 | 1098.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-05 10:15:00 | 1101.60 | 1106.16 | 1103.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-05 10:15:00 | 1101.60 | 1106.16 | 1103.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 1101.60 | 1106.16 | 1103.43 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 14:15:00 | 1097.50 | 1103.50 | 1103.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-09 09:15:00 | 1086.65 | 1099.24 | 1101.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 11:15:00 | 1104.45 | 1098.87 | 1100.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 11:15:00 | 1104.45 | 1098.87 | 1100.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 11:15:00 | 1104.45 | 1098.87 | 1100.90 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-01-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-10 11:15:00 | 1118.85 | 1104.32 | 1102.60 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-15 09:15:00 | 1091.70 | 1106.52 | 1108.17 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-16 10:15:00 | 1119.20 | 1108.85 | 1108.19 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 10:15:00 | 1102.05 | 1106.98 | 1107.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 14:15:00 | 1097.60 | 1104.40 | 1106.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 13:15:00 | 1072.90 | 1072.81 | 1081.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 14:15:00 | 1086.50 | 1075.55 | 1082.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 14:15:00 | 1086.50 | 1075.55 | 1082.40 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 12:15:00 | 1095.30 | 1086.24 | 1085.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 13:15:00 | 1111.50 | 1091.30 | 1087.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-23 13:15:00 | 1108.20 | 1108.33 | 1100.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 15:15:00 | 1096.60 | 1106.19 | 1100.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 15:15:00 | 1096.60 | 1106.19 | 1100.50 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 11:15:00 | 1082.95 | 1100.15 | 1102.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 12:15:00 | 1081.35 | 1096.39 | 1100.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 14:15:00 | 1097.80 | 1095.59 | 1099.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-29 09:15:00 | 1106.05 | 1098.07 | 1099.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 1106.05 | 1098.07 | 1099.82 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 1093.30 | 1088.27 | 1088.22 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-02 12:15:00 | 1083.20 | 1087.26 | 1087.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 11:15:00 | 1076.25 | 1085.31 | 1086.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 10:15:00 | 1072.40 | 1072.01 | 1078.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 15:15:00 | 1076.45 | 1071.48 | 1075.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 15:15:00 | 1076.45 | 1071.48 | 1075.35 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 14:15:00 | 1080.80 | 1077.61 | 1077.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 15:15:00 | 1083.45 | 1078.78 | 1077.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 10:15:00 | 1076.90 | 1078.73 | 1078.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 10:15:00 | 1076.90 | 1078.73 | 1078.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 1076.90 | 1078.73 | 1078.00 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 14:15:00 | 1146.20 | 1154.46 | 1154.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 09:15:00 | 1144.60 | 1151.51 | 1153.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 14:15:00 | 1154.65 | 1150.40 | 1151.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-27 14:15:00 | 1154.65 | 1150.40 | 1151.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-27 14:15:00 | 1154.65 | 1150.40 | 1151.77 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 09:15:00 | 1160.20 | 1152.75 | 1152.62 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 1144.00 | 1151.29 | 1152.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 1135.20 | 1148.07 | 1150.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 11:15:00 | 1148.65 | 1142.22 | 1145.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 11:15:00 | 1148.65 | 1142.22 | 1145.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 1148.65 | 1142.22 | 1145.74 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 14:15:00 | 1167.95 | 1149.58 | 1148.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-29 15:15:00 | 1170.00 | 1153.67 | 1150.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-01 14:15:00 | 1164.00 | 1164.03 | 1158.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-02 11:15:00 | 1166.25 | 1165.06 | 1160.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-02 11:15:00 | 1166.25 | 1165.06 | 1160.17 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 14:15:00 | 1154.20 | 1161.00 | 1161.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 15:15:00 | 1152.00 | 1159.20 | 1160.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 14:15:00 | 1153.55 | 1148.62 | 1153.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 14:15:00 | 1153.55 | 1148.62 | 1153.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 1153.55 | 1148.62 | 1153.52 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 1110.50 | 1101.93 | 1101.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 14:15:00 | 1111.65 | 1104.83 | 1102.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 10:15:00 | 1128.35 | 1128.40 | 1119.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-27 09:15:00 | 1134.50 | 1128.76 | 1123.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 1134.50 | 1128.76 | 1123.49 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 12:15:00 | 1137.00 | 1145.40 | 1146.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-03 13:15:00 | 1132.10 | 1142.74 | 1144.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-05 09:15:00 | 1129.35 | 1124.72 | 1131.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 14:15:00 | 1127.25 | 1124.26 | 1128.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 14:15:00 | 1127.25 | 1124.26 | 1128.39 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 12:15:00 | 1144.50 | 1130.56 | 1129.95 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 14:15:00 | 1127.50 | 1131.47 | 1131.96 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 09:15:00 | 1160.35 | 1137.01 | 1134.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-10 11:15:00 | 1167.75 | 1146.52 | 1139.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 13:15:00 | 1183.10 | 1183.31 | 1168.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-12 14:00:00 | 1183.10 | 1183.31 | 1168.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 1163.80 | 1179.18 | 1170.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 09:30:00 | 1157.45 | 1179.18 | 1170.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 10:15:00 | 1155.00 | 1174.34 | 1168.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-15 11:00:00 | 1155.00 | 1174.34 | 1168.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 13:15:00 | 1151.70 | 1165.49 | 1165.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 14:15:00 | 1149.00 | 1162.19 | 1164.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 1159.80 | 1159.60 | 1162.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 10:00:00 | 1159.80 | 1159.60 | 1162.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 1161.10 | 1159.90 | 1162.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:45:00 | 1161.55 | 1159.90 | 1162.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 1152.15 | 1158.35 | 1161.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 12:15:00 | 1150.40 | 1158.35 | 1161.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-16 14:15:00 | 1149.15 | 1156.22 | 1159.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 09:15:00 | 1150.90 | 1154.42 | 1158.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 14:30:00 | 1150.00 | 1154.91 | 1157.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 11:15:00 | 1150.50 | 1133.89 | 1139.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 12:00:00 | 1150.50 | 1133.89 | 1139.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 12:15:00 | 1158.00 | 1138.71 | 1141.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 13:00:00 | 1158.00 | 1138.71 | 1141.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-04-22 14:15:00 | 1159.15 | 1145.93 | 1144.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 1159.15 | 1145.93 | 1144.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 12:15:00 | 1161.95 | 1153.86 | 1149.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-26 14:15:00 | 1198.00 | 1199.33 | 1189.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-26 15:00:00 | 1198.00 | 1199.33 | 1189.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 1183.00 | 1196.13 | 1189.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:00:00 | 1183.00 | 1196.13 | 1189.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 1179.30 | 1192.76 | 1188.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 11:00:00 | 1179.30 | 1192.76 | 1188.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 12:15:00 | 1170.75 | 1185.45 | 1185.90 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-30 12:15:00 | 1199.50 | 1184.74 | 1184.43 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 1175.85 | 1183.87 | 1184.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 15:15:00 | 1174.00 | 1181.90 | 1183.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 09:15:00 | 1184.30 | 1182.38 | 1183.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 09:15:00 | 1184.30 | 1182.38 | 1183.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 1184.30 | 1182.38 | 1183.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 10:15:00 | 1187.15 | 1182.38 | 1183.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2024-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 10:15:00 | 1193.10 | 1184.52 | 1184.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 11:15:00 | 1196.80 | 1186.98 | 1185.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 13:15:00 | 1186.10 | 1187.69 | 1186.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 13:15:00 | 1186.10 | 1187.69 | 1186.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 13:15:00 | 1186.10 | 1187.69 | 1186.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-02 13:45:00 | 1184.05 | 1187.69 | 1186.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 1194.95 | 1189.14 | 1186.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-02 15:15:00 | 1196.00 | 1189.14 | 1186.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 09:45:00 | 1197.30 | 1190.85 | 1188.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1182.60 | 1189.56 | 1187.99 | SL hit (close<static) qty=1.00 sl=1184.50 alert=retest2 |

### Cycle 77 — SELL (started 2024-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 12:15:00 | 1201.70 | 1215.77 | 1215.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 13:15:00 | 1199.50 | 1212.52 | 1214.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 1209.60 | 1206.95 | 1210.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 09:15:00 | 1209.60 | 1206.95 | 1210.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 1209.60 | 1206.95 | 1210.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:00:00 | 1209.60 | 1206.95 | 1210.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 1215.25 | 1208.61 | 1211.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:00:00 | 1215.25 | 1208.61 | 1211.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 1218.00 | 1210.49 | 1211.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:30:00 | 1218.35 | 1210.49 | 1211.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 09:15:00 | 1181.05 | 1199.46 | 1205.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 15:00:00 | 1175.30 | 1182.04 | 1190.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 13:15:00 | 1176.65 | 1176.03 | 1183.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 11:00:00 | 1175.10 | 1175.96 | 1180.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 12:45:00 | 1171.30 | 1176.24 | 1179.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1179.90 | 1175.05 | 1178.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 1179.90 | 1175.05 | 1178.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 1179.00 | 1175.84 | 1178.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 09:15:00 | 1170.55 | 1175.84 | 1178.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 12:15:00 | 1176.95 | 1176.96 | 1178.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-17 13:00:00 | 1173.90 | 1176.35 | 1178.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 14:15:00 | 1183.00 | 1178.50 | 1178.83 | SL hit (close>static) qty=1.00 sl=1182.10 alert=retest2 |

### Cycle 78 — BUY (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 15:15:00 | 1183.00 | 1179.40 | 1179.21 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 1164.55 | 1177.35 | 1178.60 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 14:15:00 | 1187.50 | 1178.29 | 1177.76 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 12:15:00 | 1172.50 | 1177.62 | 1177.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 13:15:00 | 1170.25 | 1176.14 | 1177.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 1179.10 | 1175.95 | 1176.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 1179.10 | 1175.95 | 1176.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 1179.10 | 1175.95 | 1176.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 1177.00 | 1175.95 | 1176.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 1179.15 | 1176.59 | 1176.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:30:00 | 1182.90 | 1176.59 | 1176.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 1177.65 | 1176.80 | 1177.03 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 12:15:00 | 1180.25 | 1177.49 | 1177.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 13:15:00 | 1185.30 | 1179.06 | 1178.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 1177.70 | 1178.78 | 1178.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 14:15:00 | 1177.70 | 1178.78 | 1178.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 1177.70 | 1178.78 | 1178.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 1177.70 | 1178.78 | 1178.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 1179.85 | 1179.00 | 1178.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 1185.25 | 1179.00 | 1178.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 10:15:00 | 1169.60 | 1176.92 | 1177.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 1169.60 | 1176.92 | 1177.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 11:15:00 | 1166.25 | 1174.78 | 1176.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 09:15:00 | 1173.50 | 1168.18 | 1172.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 09:15:00 | 1173.50 | 1168.18 | 1172.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1173.50 | 1168.18 | 1172.06 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 13:15:00 | 1182.00 | 1175.04 | 1174.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 14:15:00 | 1190.00 | 1178.03 | 1175.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 09:15:00 | 1174.50 | 1178.28 | 1176.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 1174.50 | 1178.28 | 1176.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 1174.50 | 1178.28 | 1176.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 1174.50 | 1178.28 | 1176.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 1175.80 | 1177.79 | 1176.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:15:00 | 1172.85 | 1177.79 | 1176.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 1172.60 | 1176.75 | 1176.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:30:00 | 1174.00 | 1176.75 | 1176.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 1175.40 | 1176.48 | 1175.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:30:00 | 1174.65 | 1176.48 | 1175.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 1182.10 | 1177.60 | 1176.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:30:00 | 1176.05 | 1177.60 | 1176.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 1180.85 | 1178.25 | 1176.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:45:00 | 1178.50 | 1178.25 | 1176.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 15:15:00 | 1177.80 | 1178.16 | 1177.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 09:15:00 | 1176.45 | 1178.16 | 1177.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 1165.30 | 1175.59 | 1175.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 11:15:00 | 1164.65 | 1172.63 | 1174.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 09:15:00 | 1172.80 | 1170.89 | 1172.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 09:15:00 | 1172.80 | 1170.89 | 1172.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1172.80 | 1170.89 | 1172.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:30:00 | 1176.65 | 1170.89 | 1172.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 1174.00 | 1171.51 | 1172.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:15:00 | 1175.90 | 1171.51 | 1172.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 1176.35 | 1172.48 | 1173.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:30:00 | 1177.85 | 1172.48 | 1173.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 12:15:00 | 1173.70 | 1172.72 | 1173.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 13:15:00 | 1175.90 | 1172.72 | 1173.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 13:15:00 | 1177.00 | 1173.58 | 1173.54 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 14:15:00 | 1155.20 | 1169.90 | 1171.87 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 1178.85 | 1173.51 | 1172.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 14:15:00 | 1182.45 | 1175.69 | 1174.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 15:15:00 | 1175.45 | 1175.64 | 1174.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-04 09:15:00 | 1162.10 | 1175.64 | 1174.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1183.95 | 1177.30 | 1175.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1170.60 | 1177.30 | 1175.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1162.95 | 1174.43 | 1173.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1162.95 | 1174.43 | 1173.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1156.40 | 1170.83 | 1172.37 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 12:15:00 | 1193.50 | 1175.36 | 1174.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 13:15:00 | 1206.25 | 1181.54 | 1177.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 14:15:00 | 1316.35 | 1316.63 | 1307.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 15:00:00 | 1316.35 | 1316.63 | 1307.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 1306.90 | 1314.91 | 1307.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 09:30:00 | 1302.55 | 1314.91 | 1307.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 1300.50 | 1312.03 | 1307.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 11:00:00 | 1300.50 | 1312.03 | 1307.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 1299.05 | 1309.44 | 1306.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:00:00 | 1299.05 | 1309.44 | 1306.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 13:15:00 | 1292.00 | 1304.06 | 1304.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 09:15:00 | 1281.80 | 1295.99 | 1300.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 15:15:00 | 1254.00 | 1253.91 | 1265.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:15:00 | 1249.70 | 1253.91 | 1265.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1265.20 | 1255.59 | 1263.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 11:00:00 | 1265.20 | 1255.59 | 1263.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 1276.95 | 1259.86 | 1265.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 12:00:00 | 1276.95 | 1259.86 | 1265.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 1267.75 | 1266.82 | 1267.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 1265.55 | 1266.82 | 1267.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1266.45 | 1266.75 | 1267.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 11:00:00 | 1262.45 | 1265.89 | 1266.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 12:00:00 | 1262.10 | 1265.13 | 1266.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:30:00 | 1260.15 | 1263.88 | 1265.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 09:45:00 | 1258.90 | 1263.71 | 1265.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 13:15:00 | 1265.75 | 1263.21 | 1264.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 13:30:00 | 1266.75 | 1263.21 | 1264.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 1259.45 | 1262.46 | 1263.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 15:15:00 | 1257.00 | 1262.46 | 1263.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 10:15:00 | 1270.95 | 1263.95 | 1264.15 | SL hit (close>static) qty=1.00 sl=1270.80 alert=retest2 |

### Cycle 92 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 11:15:00 | 1273.55 | 1265.87 | 1265.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 12:15:00 | 1279.60 | 1268.62 | 1266.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 13:15:00 | 1280.50 | 1284.51 | 1278.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 14:00:00 | 1280.50 | 1284.51 | 1278.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 1283.00 | 1284.21 | 1278.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:45:00 | 1275.55 | 1284.21 | 1278.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1282.55 | 1283.52 | 1279.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 12:15:00 | 1290.05 | 1281.63 | 1279.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 13:30:00 | 1286.90 | 1281.97 | 1279.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 1276.40 | 1278.49 | 1278.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 11:15:00 | 1276.40 | 1278.49 | 1278.58 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 1288.80 | 1280.54 | 1279.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 1293.45 | 1284.64 | 1281.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 11:15:00 | 1281.55 | 1284.81 | 1282.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 11:15:00 | 1281.55 | 1284.81 | 1282.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 1281.55 | 1284.81 | 1282.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:45:00 | 1280.55 | 1284.81 | 1282.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 1276.00 | 1283.05 | 1281.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:00:00 | 1276.00 | 1283.05 | 1281.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 1279.50 | 1282.29 | 1281.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 1279.50 | 1282.29 | 1281.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 1281.00 | 1282.03 | 1281.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 1270.05 | 1282.03 | 1281.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 09:15:00 | 1266.60 | 1278.95 | 1280.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 11:15:00 | 1259.75 | 1273.23 | 1277.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 14:15:00 | 1270.70 | 1269.92 | 1274.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-01 15:00:00 | 1270.70 | 1269.92 | 1274.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 1268.95 | 1269.09 | 1273.25 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 1281.55 | 1276.32 | 1275.65 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-03 12:15:00 | 1271.55 | 1275.18 | 1275.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 09:15:00 | 1266.75 | 1271.90 | 1273.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 09:15:00 | 1272.45 | 1260.97 | 1263.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 1272.45 | 1260.97 | 1263.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1272.45 | 1260.97 | 1263.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:45:00 | 1271.95 | 1260.97 | 1263.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1265.85 | 1261.95 | 1264.00 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 13:15:00 | 1269.95 | 1265.86 | 1265.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-08 14:15:00 | 1271.45 | 1266.98 | 1266.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 1278.45 | 1285.48 | 1279.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 1278.45 | 1285.48 | 1279.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1278.45 | 1285.48 | 1279.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 1278.45 | 1285.48 | 1279.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1271.30 | 1282.64 | 1278.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 11:00:00 | 1271.30 | 1282.64 | 1278.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1273.10 | 1280.73 | 1278.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 13:15:00 | 1282.50 | 1279.59 | 1278.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 12:15:00 | 1278.15 | 1279.19 | 1278.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 13:15:00 | 1277.90 | 1278.61 | 1278.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 09:30:00 | 1280.00 | 1281.01 | 1279.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 1278.80 | 1280.56 | 1279.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:45:00 | 1278.00 | 1280.56 | 1279.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 1278.95 | 1280.24 | 1279.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:30:00 | 1278.00 | 1280.24 | 1279.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 1285.00 | 1281.03 | 1280.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 15:00:00 | 1290.65 | 1282.96 | 1281.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 12:00:00 | 1289.40 | 1285.70 | 1283.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 12:00:00 | 1290.00 | 1296.32 | 1294.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:45:00 | 1290.95 | 1294.89 | 1293.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1296.10 | 1296.61 | 1294.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:45:00 | 1290.60 | 1296.61 | 1294.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 1304.30 | 1298.15 | 1295.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-19 12:00:00 | 1311.25 | 1300.77 | 1297.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 12:30:00 | 1310.40 | 1303.63 | 1300.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 13:00:00 | 1311.00 | 1303.63 | 1300.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 13:45:00 | 1310.90 | 1304.76 | 1301.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 1303.00 | 1305.17 | 1302.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 1288.10 | 1305.17 | 1302.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1282.15 | 1300.57 | 1300.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-23 09:15:00 | 1282.15 | 1300.57 | 1300.23 | SL hit (close<static) qty=1.00 sl=1295.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 10:15:00 | 1290.05 | 1298.46 | 1299.30 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 12:15:00 | 1308.20 | 1299.84 | 1299.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 14:15:00 | 1331.45 | 1308.82 | 1304.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 11:15:00 | 1420.70 | 1423.01 | 1398.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 11:45:00 | 1424.20 | 1423.01 | 1398.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1422.00 | 1420.23 | 1412.77 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 1410.75 | 1413.51 | 1413.63 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 15:15:00 | 1417.70 | 1414.12 | 1413.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-02 09:15:00 | 1427.65 | 1416.83 | 1415.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 11:15:00 | 1414.90 | 1417.66 | 1415.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 11:15:00 | 1414.90 | 1417.66 | 1415.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 1414.90 | 1417.66 | 1415.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 12:00:00 | 1414.90 | 1417.66 | 1415.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 1414.10 | 1416.95 | 1415.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 12:45:00 | 1414.15 | 1416.95 | 1415.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 1421.60 | 1417.88 | 1416.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 14:30:00 | 1423.25 | 1418.19 | 1416.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 15:15:00 | 1427.95 | 1418.19 | 1416.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 09:30:00 | 1424.40 | 1422.61 | 1418.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 1410.20 | 1420.12 | 1418.11 | SL hit (close<static) qty=1.00 sl=1413.50 alert=retest2 |

### Cycle 103 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 1390.90 | 1414.28 | 1415.64 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 12:15:00 | 1425.00 | 1414.63 | 1413.74 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 1409.35 | 1413.20 | 1413.22 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 09:15:00 | 1446.05 | 1419.58 | 1416.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 10:15:00 | 1461.30 | 1427.92 | 1420.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 15:15:00 | 1456.00 | 1458.14 | 1447.93 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-09 09:15:00 | 1467.60 | 1458.14 | 1447.93 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 1456.60 | 1464.63 | 1456.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-09 14:15:00 | 1456.60 | 1464.63 | 1456.72 | SL hit (close<ema400) qty=1.00 sl=1456.72 alert=retest1 |

### Cycle 107 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 1429.80 | 1449.70 | 1451.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 11:15:00 | 1427.30 | 1445.22 | 1449.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 14:15:00 | 1404.95 | 1403.57 | 1413.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 15:00:00 | 1404.95 | 1403.57 | 1413.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1401.85 | 1403.13 | 1411.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 11:00:00 | 1399.45 | 1402.40 | 1410.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 11:45:00 | 1399.00 | 1401.47 | 1409.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:00:00 | 1398.50 | 1400.87 | 1408.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:45:00 | 1398.35 | 1400.18 | 1407.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 1406.80 | 1401.50 | 1407.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:45:00 | 1404.10 | 1401.50 | 1407.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 1406.50 | 1402.50 | 1407.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 1406.65 | 1402.50 | 1407.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1410.95 | 1404.19 | 1407.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-19 10:15:00 | 1418.10 | 1406.97 | 1408.66 | SL hit (close>static) qty=1.00 sl=1417.45 alert=retest2 |

### Cycle 108 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 1413.10 | 1410.19 | 1409.95 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 13:15:00 | 1406.60 | 1409.47 | 1409.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-20 09:15:00 | 1404.10 | 1407.31 | 1408.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 12:15:00 | 1405.60 | 1404.90 | 1406.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-20 13:00:00 | 1405.60 | 1404.90 | 1406.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 1406.40 | 1405.20 | 1406.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:00:00 | 1406.40 | 1405.20 | 1406.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 1406.85 | 1405.53 | 1406.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 1406.85 | 1405.53 | 1406.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 1407.00 | 1405.82 | 1406.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-21 09:15:00 | 1403.80 | 1405.82 | 1406.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1409.10 | 1406.48 | 1407.09 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2024-08-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 12:15:00 | 1414.00 | 1408.17 | 1407.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 13:15:00 | 1417.85 | 1410.11 | 1408.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 1441.60 | 1445.58 | 1433.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 09:30:00 | 1441.50 | 1445.58 | 1433.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1436.00 | 1443.67 | 1434.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:30:00 | 1435.25 | 1443.67 | 1434.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 1430.95 | 1441.12 | 1433.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 1430.95 | 1441.12 | 1433.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 1427.95 | 1438.49 | 1433.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:00:00 | 1427.95 | 1438.49 | 1433.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 1428.20 | 1435.56 | 1432.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 1428.20 | 1435.56 | 1432.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1426.05 | 1433.66 | 1432.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:30:00 | 1430.70 | 1433.32 | 1432.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1441.95 | 1435.04 | 1433.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:15:00 | 1443.75 | 1435.04 | 1433.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 12:45:00 | 1443.95 | 1438.56 | 1435.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 13:15:00 | 1446.65 | 1438.56 | 1435.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 13:15:00 | 1429.10 | 1438.29 | 1438.22 | SL hit (close<static) qty=1.00 sl=1431.50 alert=retest2 |

### Cycle 111 — SELL (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 14:15:00 | 1427.85 | 1436.20 | 1437.28 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 10:15:00 | 1445.10 | 1438.79 | 1438.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 14:15:00 | 1449.10 | 1442.90 | 1440.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 09:15:00 | 1440.90 | 1442.84 | 1440.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 1440.90 | 1442.84 | 1440.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 1440.90 | 1442.84 | 1440.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 1440.90 | 1442.84 | 1440.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1434.40 | 1441.16 | 1440.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:00:00 | 1434.40 | 1441.16 | 1440.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1437.00 | 1440.32 | 1439.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:15:00 | 1445.05 | 1440.32 | 1439.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 13:15:00 | 1435.75 | 1439.50 | 1439.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 13:15:00 | 1435.75 | 1439.50 | 1439.67 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 14:15:00 | 1448.95 | 1441.39 | 1440.51 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 10:15:00 | 1435.20 | 1439.97 | 1440.09 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 1444.50 | 1440.87 | 1440.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 12:15:00 | 1463.10 | 1445.32 | 1442.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 13:15:00 | 1486.05 | 1486.38 | 1475.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-03 14:00:00 | 1486.05 | 1486.38 | 1475.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 1476.65 | 1484.43 | 1475.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 1476.65 | 1484.43 | 1475.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 1477.00 | 1482.95 | 1476.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 1476.00 | 1482.95 | 1476.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 1487.60 | 1483.88 | 1477.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 12:30:00 | 1493.50 | 1487.54 | 1480.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 12:15:00 | 1493.50 | 1494.31 | 1487.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 13:45:00 | 1495.15 | 1494.26 | 1489.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 1471.95 | 1484.00 | 1485.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 1471.95 | 1484.00 | 1485.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 1460.95 | 1476.00 | 1480.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 1485.50 | 1475.64 | 1479.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 10:15:00 | 1485.50 | 1475.64 | 1479.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 1485.50 | 1475.64 | 1479.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:00:00 | 1485.50 | 1475.64 | 1479.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 11:15:00 | 1490.10 | 1478.53 | 1480.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 11:45:00 | 1493.60 | 1478.53 | 1480.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 12:15:00 | 1499.15 | 1482.65 | 1482.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 13:15:00 | 1500.85 | 1486.29 | 1483.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 10:15:00 | 1521.90 | 1526.17 | 1520.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 11:00:00 | 1521.90 | 1526.17 | 1520.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 1521.90 | 1525.32 | 1520.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 13:30:00 | 1530.55 | 1525.43 | 1521.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 15:15:00 | 1523.90 | 1524.41 | 1521.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 15:15:00 | 1534.05 | 1539.79 | 1539.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 1534.05 | 1539.79 | 1539.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 1528.75 | 1537.31 | 1538.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 1534.65 | 1533.41 | 1536.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 1534.65 | 1533.41 | 1536.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 1538.00 | 1534.33 | 1536.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 1548.95 | 1534.33 | 1536.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 1543.85 | 1536.23 | 1537.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:30:00 | 1546.95 | 1536.23 | 1537.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-09-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 10:15:00 | 1554.90 | 1539.96 | 1538.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 11:15:00 | 1566.30 | 1545.23 | 1541.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 10:15:00 | 1561.95 | 1562.72 | 1553.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 11:00:00 | 1561.95 | 1562.72 | 1553.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 1611.90 | 1617.69 | 1607.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 14:15:00 | 1620.95 | 1617.69 | 1607.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 11:45:00 | 1616.15 | 1620.30 | 1613.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 12:30:00 | 1621.00 | 1619.92 | 1613.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 1582.45 | 1620.74 | 1617.07 | SL hit (close<static) qty=1.00 sl=1607.65 alert=retest2 |

### Cycle 121 — SELL (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 10:15:00 | 1585.00 | 1613.59 | 1614.16 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 14:15:00 | 1613.90 | 1600.54 | 1600.30 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2024-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 09:15:00 | 1573.65 | 1596.37 | 1598.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 10:15:00 | 1567.65 | 1590.62 | 1595.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 1520.95 | 1516.04 | 1533.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:45:00 | 1520.15 | 1516.04 | 1533.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1530.05 | 1518.34 | 1527.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1523.35 | 1518.34 | 1527.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1522.75 | 1519.22 | 1527.46 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 1543.85 | 1530.01 | 1529.90 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 1520.70 | 1528.15 | 1529.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 1508.40 | 1524.20 | 1527.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 10:15:00 | 1514.40 | 1513.33 | 1519.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-11 10:30:00 | 1516.80 | 1513.33 | 1519.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 1517.45 | 1514.01 | 1518.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:45:00 | 1518.70 | 1514.01 | 1518.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 1517.05 | 1514.62 | 1518.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 13:30:00 | 1516.70 | 1514.62 | 1518.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 1518.50 | 1515.39 | 1518.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 15:00:00 | 1518.50 | 1515.39 | 1518.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 1522.00 | 1516.71 | 1518.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 09:15:00 | 1520.85 | 1516.71 | 1518.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1528.85 | 1519.14 | 1519.62 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 10:15:00 | 1531.00 | 1521.51 | 1520.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 11:15:00 | 1537.70 | 1524.75 | 1522.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 1523.70 | 1530.64 | 1526.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 1523.70 | 1530.64 | 1526.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1523.70 | 1530.64 | 1526.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 1523.70 | 1530.64 | 1526.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1533.45 | 1531.20 | 1527.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 14:45:00 | 1550.30 | 1537.50 | 1531.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 09:30:00 | 1550.50 | 1541.47 | 1534.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 12:15:00 | 1549.00 | 1543.81 | 1537.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 11:15:00 | 1522.55 | 1538.01 | 1538.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 1522.55 | 1538.01 | 1538.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 1509.80 | 1527.15 | 1532.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 1521.10 | 1520.51 | 1526.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 15:00:00 | 1521.10 | 1520.51 | 1526.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1512.90 | 1518.40 | 1524.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 1502.85 | 1511.55 | 1519.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:30:00 | 1502.05 | 1508.32 | 1516.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 1479.10 | 1473.26 | 1472.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 10:15:00 | 1479.10 | 1473.26 | 1472.61 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 12:15:00 | 1464.00 | 1471.45 | 1471.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 09:15:00 | 1439.20 | 1463.22 | 1467.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 15:15:00 | 1452.15 | 1448.81 | 1456.79 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-30 09:15:00 | 1442.80 | 1448.81 | 1456.79 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-30 10:15:00 | 1442.25 | 1448.24 | 1455.80 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 1450.00 | 1445.13 | 1450.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 1444.40 | 1445.13 | 1450.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 1437.90 | 1443.69 | 1449.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 10:15:00 | 1432.35 | 1443.69 | 1449.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-31 11:30:00 | 1433.40 | 1440.55 | 1446.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-31 14:15:00 | 1451.70 | 1440.75 | 1445.17 | SL hit (close>ema400) qty=1.00 sl=1445.17 alert=retest1 |

### Cycle 130 — BUY (started 2024-11-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 14:15:00 | 1447.90 | 1442.44 | 1441.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 1452.70 | 1445.09 | 1443.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1437.65 | 1449.91 | 1447.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 1437.65 | 1449.91 | 1447.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1437.65 | 1449.91 | 1447.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 1437.65 | 1449.91 | 1447.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 1425.75 | 1445.08 | 1445.52 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 10:15:00 | 1450.80 | 1444.33 | 1444.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-11 09:15:00 | 1461.40 | 1449.58 | 1446.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-12 10:15:00 | 1462.85 | 1469.90 | 1462.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 10:15:00 | 1462.85 | 1469.90 | 1462.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 1462.85 | 1469.90 | 1462.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:00:00 | 1462.85 | 1469.90 | 1462.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 1469.25 | 1469.77 | 1462.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 11:30:00 | 1457.95 | 1469.77 | 1462.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 1463.85 | 1468.59 | 1462.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:45:00 | 1463.00 | 1468.59 | 1462.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 1458.45 | 1466.56 | 1462.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:30:00 | 1457.45 | 1466.56 | 1462.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 14:15:00 | 1443.25 | 1461.90 | 1460.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 15:00:00 | 1443.25 | 1461.90 | 1460.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 1448.50 | 1459.22 | 1459.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 1434.30 | 1454.23 | 1457.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1446.85 | 1438.53 | 1445.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1446.85 | 1438.53 | 1445.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1446.85 | 1438.53 | 1445.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1446.85 | 1438.53 | 1445.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1447.95 | 1440.41 | 1446.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:45:00 | 1452.10 | 1440.41 | 1446.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1448.10 | 1441.95 | 1446.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:15:00 | 1450.60 | 1441.95 | 1446.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1449.55 | 1443.47 | 1446.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:45:00 | 1449.95 | 1443.47 | 1446.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1441.90 | 1443.16 | 1446.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:30:00 | 1444.95 | 1443.16 | 1446.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 1443.00 | 1443.42 | 1445.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:15:00 | 1447.20 | 1443.42 | 1445.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1457.95 | 1446.33 | 1446.86 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 10:15:00 | 1462.20 | 1449.50 | 1448.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 11:15:00 | 1474.70 | 1454.54 | 1450.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 1485.70 | 1486.34 | 1473.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 1485.70 | 1486.34 | 1473.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 1483.55 | 1507.78 | 1501.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 1483.55 | 1507.78 | 1501.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 1484.90 | 1503.20 | 1500.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:15:00 | 1477.15 | 1503.20 | 1500.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 1504.30 | 1509.08 | 1505.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 1503.50 | 1509.08 | 1505.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 1507.75 | 1508.82 | 1505.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:45:00 | 1506.75 | 1508.82 | 1505.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 1506.35 | 1508.32 | 1505.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 1505.00 | 1508.32 | 1505.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 1506.40 | 1507.94 | 1505.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:45:00 | 1505.00 | 1507.94 | 1505.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 1505.55 | 1507.46 | 1505.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 14:45:00 | 1519.90 | 1508.69 | 1506.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 15:15:00 | 1522.35 | 1508.69 | 1506.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:45:00 | 1517.00 | 1511.88 | 1508.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 1501.70 | 1509.84 | 1507.85 | SL hit (close<static) qty=1.00 sl=1502.75 alert=retest2 |

### Cycle 135 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 1499.50 | 1505.73 | 1506.29 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 09:15:00 | 1525.55 | 1509.14 | 1507.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 10:15:00 | 1537.35 | 1514.78 | 1510.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 14:15:00 | 1531.70 | 1533.44 | 1526.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 15:00:00 | 1531.70 | 1533.44 | 1526.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1538.25 | 1534.23 | 1527.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:15:00 | 1540.45 | 1534.23 | 1527.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:45:00 | 1540.00 | 1534.47 | 1528.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 12:45:00 | 1542.05 | 1536.32 | 1530.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 14:00:00 | 1542.20 | 1537.49 | 1531.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1530.00 | 1537.70 | 1533.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:00:00 | 1530.00 | 1537.70 | 1533.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 1527.15 | 1535.59 | 1532.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:45:00 | 1524.15 | 1535.59 | 1532.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1527.95 | 1534.06 | 1532.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 1523.95 | 1534.06 | 1532.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 1528.30 | 1532.51 | 1531.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 13:45:00 | 1527.50 | 1532.51 | 1531.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-04 14:15:00 | 1525.10 | 1531.02 | 1531.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2024-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 14:15:00 | 1525.10 | 1531.02 | 1531.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 10:15:00 | 1522.90 | 1528.30 | 1529.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 12:15:00 | 1528.85 | 1528.11 | 1529.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 12:15:00 | 1528.85 | 1528.11 | 1529.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 12:15:00 | 1528.85 | 1528.11 | 1529.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 11:45:00 | 1524.70 | 1527.89 | 1528.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 15:15:00 | 1521.00 | 1514.80 | 1514.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2024-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 15:15:00 | 1521.00 | 1514.80 | 1514.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 13:15:00 | 1522.75 | 1517.76 | 1515.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 1500.80 | 1516.71 | 1516.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 1500.80 | 1516.71 | 1516.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1500.80 | 1516.71 | 1516.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 1500.80 | 1516.71 | 1516.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 1493.60 | 1512.09 | 1514.10 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 09:15:00 | 1539.55 | 1515.60 | 1512.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 12:15:00 | 1563.25 | 1536.77 | 1523.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 13:15:00 | 1576.25 | 1577.27 | 1565.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-19 13:45:00 | 1576.95 | 1577.27 | 1565.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 1565.40 | 1573.73 | 1568.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 1565.40 | 1573.73 | 1568.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 1560.75 | 1571.14 | 1567.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:45:00 | 1565.05 | 1571.14 | 1567.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 1543.75 | 1565.66 | 1565.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 1543.75 | 1565.66 | 1565.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 1554.60 | 1563.45 | 1564.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 12:15:00 | 1537.90 | 1552.45 | 1558.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1549.15 | 1548.48 | 1554.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 1549.15 | 1548.48 | 1554.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1549.15 | 1548.48 | 1554.26 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 15:15:00 | 1560.00 | 1556.72 | 1556.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 1573.85 | 1563.38 | 1560.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 1673.15 | 1674.59 | 1655.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 10:30:00 | 1675.15 | 1674.59 | 1655.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1658.75 | 1673.21 | 1663.56 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-01-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 13:15:00 | 1642.65 | 1657.60 | 1658.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 1638.95 | 1653.87 | 1656.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 1594.70 | 1589.84 | 1609.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-09 10:00:00 | 1594.70 | 1589.84 | 1609.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1417.20 | 1414.29 | 1440.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:45:00 | 1409.25 | 1414.11 | 1437.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 12:00:00 | 1410.70 | 1413.43 | 1435.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 12:15:00 | 1455.70 | 1428.23 | 1427.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 1455.70 | 1428.23 | 1427.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 09:15:00 | 1489.25 | 1459.61 | 1450.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 10:15:00 | 1482.75 | 1488.52 | 1474.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 11:00:00 | 1482.75 | 1488.52 | 1474.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 1474.45 | 1485.62 | 1476.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:00:00 | 1474.45 | 1485.62 | 1476.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 1469.00 | 1482.30 | 1475.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:45:00 | 1470.55 | 1482.30 | 1475.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 15:15:00 | 1469.55 | 1479.75 | 1475.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:15:00 | 1440.40 | 1479.75 | 1475.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 1438.50 | 1471.50 | 1471.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 1419.00 | 1461.00 | 1467.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 13:15:00 | 1401.00 | 1399.29 | 1422.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:45:00 | 1401.95 | 1399.29 | 1422.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 1425.55 | 1405.61 | 1413.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 1425.55 | 1405.61 | 1413.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 1428.95 | 1410.27 | 1415.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 1428.95 | 1410.27 | 1415.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 1427.95 | 1418.12 | 1417.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 1429.35 | 1425.50 | 1422.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 12:15:00 | 1425.20 | 1425.44 | 1422.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 13:00:00 | 1425.20 | 1425.44 | 1422.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 1421.75 | 1424.70 | 1422.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 13:45:00 | 1424.70 | 1424.70 | 1422.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 1424.00 | 1424.56 | 1422.70 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2025-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 10:15:00 | 1396.10 | 1416.66 | 1419.43 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 1486.30 | 1428.86 | 1424.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 14:15:00 | 1500.00 | 1452.68 | 1436.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 14:15:00 | 1473.25 | 1475.25 | 1459.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 14:45:00 | 1471.50 | 1475.25 | 1459.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 12:15:00 | 1485.50 | 1483.12 | 1469.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:30:00 | 1484.35 | 1483.12 | 1469.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1458.55 | 1477.36 | 1471.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 1458.55 | 1477.36 | 1471.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1447.55 | 1471.40 | 1469.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 1447.55 | 1471.40 | 1469.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 11:15:00 | 1450.70 | 1467.26 | 1467.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 14:15:00 | 1441.90 | 1457.11 | 1462.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 1428.90 | 1421.97 | 1432.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 15:00:00 | 1428.90 | 1421.97 | 1432.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1434.00 | 1424.86 | 1431.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 1434.00 | 1424.86 | 1431.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1436.35 | 1427.16 | 1432.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:45:00 | 1440.10 | 1427.16 | 1432.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 1422.65 | 1426.26 | 1431.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:30:00 | 1437.10 | 1426.26 | 1431.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1411.25 | 1399.52 | 1406.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 1411.25 | 1399.52 | 1406.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 1396.15 | 1398.85 | 1405.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:15:00 | 1389.00 | 1398.14 | 1402.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:15:00 | 1390.10 | 1396.25 | 1400.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 09:15:00 | 1319.55 | 1335.64 | 1349.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-19 09:15:00 | 1320.59 | 1335.64 | 1349.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-19 15:15:00 | 1331.00 | 1330.48 | 1340.07 | SL hit (close>ema200) qty=0.50 sl=1330.48 alert=retest2 |

### Cycle 150 — BUY (started 2025-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 14:15:00 | 1343.95 | 1330.63 | 1329.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 11:15:00 | 1347.90 | 1338.42 | 1333.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-27 11:15:00 | 1343.65 | 1345.41 | 1340.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-27 12:00:00 | 1343.65 | 1345.41 | 1340.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 1336.75 | 1343.68 | 1339.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:00:00 | 1336.75 | 1343.68 | 1339.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 1330.30 | 1341.00 | 1339.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:00:00 | 1330.30 | 1341.00 | 1339.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 1332.75 | 1339.35 | 1338.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:45:00 | 1330.05 | 1339.35 | 1338.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 1333.20 | 1338.12 | 1338.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 09:15:00 | 1320.90 | 1338.12 | 1338.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 1311.80 | 1332.86 | 1335.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 10:15:00 | 1305.85 | 1327.46 | 1332.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 10:15:00 | 1300.65 | 1297.11 | 1311.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 11:00:00 | 1300.65 | 1297.11 | 1311.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 1313.20 | 1302.72 | 1310.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:45:00 | 1310.00 | 1302.72 | 1310.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 1307.05 | 1303.59 | 1310.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 1291.00 | 1305.03 | 1310.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 1315.85 | 1300.47 | 1301.54 | SL hit (close>static) qty=1.00 sl=1315.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 1320.20 | 1304.41 | 1303.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 1324.45 | 1308.42 | 1305.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 1323.00 | 1327.40 | 1320.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 1323.00 | 1327.40 | 1320.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1344.50 | 1331.00 | 1325.03 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 1312.05 | 1322.81 | 1324.19 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 11:15:00 | 1326.00 | 1323.31 | 1323.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 14:15:00 | 1337.30 | 1327.20 | 1324.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 09:15:00 | 1347.00 | 1349.20 | 1340.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-17 09:45:00 | 1344.50 | 1349.20 | 1340.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1348.60 | 1349.08 | 1341.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 1340.70 | 1349.08 | 1341.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 09:15:00 | 1337.45 | 1358.15 | 1354.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 10:00:00 | 1337.45 | 1358.15 | 1354.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 1329.40 | 1352.40 | 1351.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 11:00:00 | 1329.40 | 1352.40 | 1351.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 11:15:00 | 1335.65 | 1349.05 | 1350.30 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 10:15:00 | 1369.90 | 1353.39 | 1351.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 11:15:00 | 1376.60 | 1358.03 | 1353.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 12:15:00 | 1387.35 | 1390.25 | 1382.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 13:00:00 | 1387.35 | 1390.25 | 1382.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1386.40 | 1390.70 | 1384.95 | EMA400 retest candle locked (from upside) |

### Cycle 157 — SELL (started 2025-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 13:15:00 | 1370.30 | 1379.75 | 1380.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 1365.55 | 1376.91 | 1379.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 13:15:00 | 1368.25 | 1367.92 | 1372.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-26 13:30:00 | 1366.85 | 1367.92 | 1372.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 1389.65 | 1370.53 | 1372.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:00:00 | 1389.65 | 1370.53 | 1372.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 1385.45 | 1373.52 | 1373.89 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 1384.45 | 1375.70 | 1374.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 12:15:00 | 1390.45 | 1378.65 | 1376.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 09:15:00 | 1398.10 | 1398.85 | 1392.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-01 10:00:00 | 1398.10 | 1398.85 | 1392.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 1391.10 | 1397.30 | 1392.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 11:00:00 | 1391.10 | 1397.30 | 1392.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 11:15:00 | 1397.50 | 1397.34 | 1392.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 12:15:00 | 1399.90 | 1397.34 | 1392.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 13:30:00 | 1400.30 | 1398.42 | 1393.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1382.90 | 1419.69 | 1420.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1382.90 | 1419.69 | 1420.16 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 1435.10 | 1418.11 | 1416.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 10:15:00 | 1440.00 | 1428.25 | 1422.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 11:15:00 | 1494.60 | 1495.13 | 1479.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:45:00 | 1493.60 | 1495.13 | 1479.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 1561.00 | 1573.66 | 1561.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 15:00:00 | 1561.00 | 1573.66 | 1561.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 1562.60 | 1571.45 | 1561.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 1566.40 | 1571.45 | 1561.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1550.10 | 1567.18 | 1560.76 | SL hit (close<static) qty=1.00 sl=1559.60 alert=retest2 |

### Cycle 161 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 1550.20 | 1556.40 | 1556.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 09:15:00 | 1539.10 | 1549.19 | 1552.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 13:15:00 | 1552.60 | 1545.42 | 1549.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 13:15:00 | 1552.60 | 1545.42 | 1549.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 1552.60 | 1545.42 | 1549.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 14:00:00 | 1552.60 | 1545.42 | 1549.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 14:15:00 | 1555.30 | 1547.40 | 1549.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 15:00:00 | 1555.30 | 1547.40 | 1549.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 1547.00 | 1547.32 | 1549.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 1544.00 | 1547.32 | 1549.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1546.40 | 1547.13 | 1549.37 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 1556.50 | 1549.44 | 1549.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 12:15:00 | 1563.80 | 1553.23 | 1550.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 15:15:00 | 1555.30 | 1557.25 | 1553.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 09:15:00 | 1555.10 | 1557.25 | 1553.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 1558.40 | 1557.48 | 1554.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 10:15:00 | 1569.80 | 1557.48 | 1554.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-02 11:15:00 | 1571.60 | 1559.54 | 1555.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 09:30:00 | 1578.20 | 1562.63 | 1559.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 09:15:00 | 1545.00 | 1569.17 | 1566.42 | SL hit (close<static) qty=1.00 sl=1551.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 1545.40 | 1568.41 | 1570.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 1540.60 | 1562.85 | 1567.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 1562.00 | 1538.31 | 1546.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 1562.00 | 1538.31 | 1546.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1562.00 | 1538.31 | 1546.46 | EMA400 retest candle locked (from downside) |

### Cycle 164 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 1565.60 | 1553.59 | 1551.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1571.00 | 1559.82 | 1555.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 1557.70 | 1560.92 | 1557.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 12:15:00 | 1557.70 | 1560.92 | 1557.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 1557.70 | 1560.92 | 1557.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 1557.70 | 1560.92 | 1557.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 1554.10 | 1559.56 | 1556.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:45:00 | 1553.80 | 1559.56 | 1556.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 14:15:00 | 1553.30 | 1558.31 | 1556.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 15:00:00 | 1553.30 | 1558.31 | 1556.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 1553.20 | 1557.29 | 1556.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:15:00 | 1560.30 | 1557.29 | 1556.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — SELL (started 2025-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 09:15:00 | 1546.70 | 1555.17 | 1555.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 10:15:00 | 1538.00 | 1551.74 | 1553.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 11:15:00 | 1543.70 | 1538.05 | 1543.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 11:15:00 | 1543.70 | 1538.05 | 1543.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1543.70 | 1538.05 | 1543.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 1543.70 | 1538.05 | 1543.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 1542.20 | 1538.88 | 1543.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 13:15:00 | 1537.70 | 1538.88 | 1543.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 15:00:00 | 1535.90 | 1538.75 | 1542.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 11:15:00 | 1550.30 | 1542.08 | 1542.85 | SL hit (close>static) qty=1.00 sl=1546.00 alert=retest2 |

### Cycle 166 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 1557.80 | 1545.08 | 1543.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 15:15:00 | 1565.00 | 1556.12 | 1552.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-21 09:15:00 | 1544.50 | 1553.79 | 1551.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1544.50 | 1553.79 | 1551.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1544.50 | 1553.79 | 1551.52 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1530.50 | 1549.02 | 1549.74 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 15:15:00 | 1553.00 | 1549.92 | 1549.71 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 1547.20 | 1549.37 | 1549.49 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 1554.00 | 1550.30 | 1549.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 15:15:00 | 1569.20 | 1558.45 | 1554.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 15:15:00 | 1571.70 | 1576.36 | 1567.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 09:15:00 | 1580.30 | 1576.36 | 1567.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1575.90 | 1576.27 | 1568.53 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 1555.60 | 1565.68 | 1565.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 1542.20 | 1559.12 | 1562.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 15:15:00 | 1527.60 | 1527.52 | 1537.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 09:15:00 | 1524.70 | 1527.52 | 1537.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1515.80 | 1520.27 | 1527.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 1504.30 | 1515.73 | 1523.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 13:15:00 | 1540.60 | 1527.50 | 1525.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 1540.60 | 1527.50 | 1525.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 1550.20 | 1532.04 | 1528.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 15:15:00 | 1581.00 | 1581.82 | 1570.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 09:15:00 | 1588.60 | 1581.82 | 1570.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1612.70 | 1588.00 | 1574.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 10:15:00 | 1624.70 | 1588.00 | 1574.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 09:30:00 | 1613.30 | 1611.66 | 1596.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 10:00:00 | 1613.00 | 1611.66 | 1596.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 12:15:00 | 1586.80 | 1593.62 | 1593.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-06-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 12:15:00 | 1586.80 | 1593.62 | 1593.99 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 1617.20 | 1597.71 | 1595.63 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-06-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 09:15:00 | 1503.80 | 1589.01 | 1595.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 10:15:00 | 1488.60 | 1568.93 | 1585.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 1462.40 | 1459.80 | 1480.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:45:00 | 1463.00 | 1459.80 | 1480.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1483.60 | 1465.46 | 1479.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:45:00 | 1485.80 | 1465.46 | 1479.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1487.80 | 1469.93 | 1480.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 1487.80 | 1469.93 | 1480.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1477.90 | 1474.92 | 1480.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 1470.80 | 1474.95 | 1480.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 1471.40 | 1474.98 | 1479.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 13:45:00 | 1472.20 | 1474.39 | 1478.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 15:15:00 | 1472.60 | 1474.51 | 1478.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1469.90 | 1467.23 | 1471.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 1462.20 | 1466.78 | 1470.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 11:15:00 | 1464.80 | 1466.78 | 1470.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 12:30:00 | 1463.90 | 1466.04 | 1469.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 14:30:00 | 1461.10 | 1465.17 | 1468.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1456.60 | 1462.95 | 1467.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 10:00:00 | 1443.60 | 1455.35 | 1461.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 11:15:00 | 1457.90 | 1449.15 | 1448.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 11:15:00 | 1457.90 | 1449.15 | 1448.01 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 1445.30 | 1449.00 | 1449.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 12:15:00 | 1440.00 | 1447.20 | 1448.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-26 15:15:00 | 1445.60 | 1445.59 | 1447.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 15:15:00 | 1445.60 | 1445.59 | 1447.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 1445.60 | 1445.59 | 1447.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 1456.50 | 1445.59 | 1447.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1457.20 | 1447.91 | 1448.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:00:00 | 1457.20 | 1447.91 | 1448.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 1452.40 | 1448.81 | 1448.43 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 15:15:00 | 1442.10 | 1447.40 | 1448.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-30 09:15:00 | 1427.40 | 1443.40 | 1446.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 1385.90 | 1381.93 | 1387.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 1385.90 | 1381.93 | 1387.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1385.90 | 1381.93 | 1387.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1388.90 | 1381.93 | 1387.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1384.00 | 1382.34 | 1387.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 1381.00 | 1382.34 | 1387.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:15:00 | 1376.40 | 1381.29 | 1384.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-10 13:15:00 | 1379.90 | 1373.47 | 1373.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 180 — BUY (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 13:15:00 | 1379.90 | 1373.47 | 1373.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 14:15:00 | 1383.10 | 1375.39 | 1374.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 1371.90 | 1376.71 | 1375.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 1371.90 | 1376.71 | 1375.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1371.90 | 1376.71 | 1375.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 1371.90 | 1376.71 | 1375.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1370.50 | 1375.47 | 1374.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 1370.50 | 1375.47 | 1374.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1366.00 | 1373.58 | 1374.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 1357.20 | 1369.14 | 1371.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1367.10 | 1366.79 | 1370.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 1367.10 | 1366.79 | 1370.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1367.10 | 1366.79 | 1370.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:15:00 | 1368.00 | 1366.79 | 1370.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1367.30 | 1366.89 | 1370.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:30:00 | 1363.40 | 1366.23 | 1369.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:45:00 | 1363.20 | 1365.97 | 1368.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:45:00 | 1364.60 | 1365.54 | 1368.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 10:15:00 | 1371.60 | 1368.99 | 1368.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 10:15:00 | 1371.60 | 1368.99 | 1368.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 11:15:00 | 1373.40 | 1369.88 | 1369.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 13:15:00 | 1379.60 | 1381.89 | 1377.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 14:00:00 | 1379.60 | 1381.89 | 1377.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 1378.20 | 1380.88 | 1377.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 1376.60 | 1380.88 | 1377.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1373.10 | 1379.33 | 1377.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 1373.10 | 1379.33 | 1377.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1366.80 | 1376.82 | 1376.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 1366.80 | 1376.82 | 1376.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 11:15:00 | 1367.10 | 1374.88 | 1375.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 1364.40 | 1372.78 | 1374.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 1369.80 | 1368.42 | 1371.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 1369.80 | 1368.42 | 1371.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1369.80 | 1368.42 | 1371.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 1373.60 | 1368.42 | 1371.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1366.00 | 1367.94 | 1370.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 1369.70 | 1367.94 | 1370.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 15:15:00 | 1368.90 | 1366.79 | 1369.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:15:00 | 1363.10 | 1366.79 | 1369.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1357.20 | 1364.87 | 1367.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:30:00 | 1353.00 | 1361.80 | 1366.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 1330.00 | 1319.82 | 1319.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 09:15:00 | 1330.00 | 1319.82 | 1319.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 10:15:00 | 1334.10 | 1322.67 | 1321.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 1320.00 | 1332.48 | 1328.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 1320.00 | 1332.48 | 1328.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1320.00 | 1332.48 | 1328.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 1320.00 | 1332.48 | 1328.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1325.00 | 1330.98 | 1328.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:45:00 | 1323.10 | 1330.98 | 1328.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2025-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 13:15:00 | 1317.90 | 1325.53 | 1326.02 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 11:15:00 | 1330.80 | 1325.65 | 1325.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 14:15:00 | 1337.70 | 1328.75 | 1327.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 11:15:00 | 1329.50 | 1330.98 | 1328.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 11:15:00 | 1329.50 | 1330.98 | 1328.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1329.50 | 1330.98 | 1328.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 1329.50 | 1330.98 | 1328.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 1327.20 | 1330.22 | 1328.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:15:00 | 1324.90 | 1330.22 | 1328.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 1320.90 | 1328.36 | 1328.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 14:00:00 | 1320.90 | 1328.36 | 1328.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 14:15:00 | 1324.00 | 1327.49 | 1327.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 1316.90 | 1324.65 | 1326.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 09:15:00 | 1302.60 | 1298.13 | 1306.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 1302.60 | 1298.13 | 1306.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1302.60 | 1298.13 | 1306.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:15:00 | 1294.80 | 1300.58 | 1305.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:45:00 | 1293.20 | 1298.83 | 1303.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 11:00:00 | 1294.60 | 1299.58 | 1301.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 11:45:00 | 1294.80 | 1298.67 | 1300.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 1297.90 | 1297.84 | 1299.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:30:00 | 1303.40 | 1297.84 | 1299.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 1298.00 | 1297.87 | 1299.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:15:00 | 1298.90 | 1297.87 | 1299.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1303.10 | 1298.92 | 1299.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 12:15:00 | 1295.90 | 1299.18 | 1299.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:00:00 | 1295.40 | 1298.43 | 1299.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 14:15:00 | 1305.40 | 1300.60 | 1300.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 14:15:00 | 1305.40 | 1300.60 | 1300.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 15:15:00 | 1307.00 | 1301.88 | 1300.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 1316.00 | 1316.46 | 1310.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 09:15:00 | 1309.60 | 1316.46 | 1310.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1321.60 | 1317.49 | 1311.50 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 1306.60 | 1309.95 | 1310.10 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 13:15:00 | 1314.30 | 1310.94 | 1310.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 14:15:00 | 1327.70 | 1314.29 | 1312.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 12:15:00 | 1330.70 | 1332.47 | 1327.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 12:15:00 | 1330.70 | 1332.47 | 1327.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1330.70 | 1332.47 | 1327.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 1327.00 | 1332.47 | 1327.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1329.80 | 1333.33 | 1329.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:45:00 | 1329.40 | 1333.33 | 1329.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1328.60 | 1332.39 | 1329.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:45:00 | 1328.80 | 1332.39 | 1329.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1332.70 | 1332.45 | 1329.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 13:00:00 | 1335.20 | 1333.00 | 1330.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 09:15:00 | 1319.90 | 1329.73 | 1329.52 | SL hit (close<static) qty=1.00 sl=1328.40 alert=retest2 |

### Cycle 191 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 1320.60 | 1327.90 | 1328.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 15:15:00 | 1310.00 | 1318.73 | 1323.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 09:15:00 | 1292.30 | 1289.68 | 1299.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 1299.80 | 1291.71 | 1299.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1299.80 | 1291.71 | 1299.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 1299.80 | 1291.71 | 1299.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1308.30 | 1295.03 | 1300.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 1307.30 | 1295.03 | 1300.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 1307.50 | 1297.52 | 1300.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:45:00 | 1309.60 | 1297.52 | 1300.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 192 — BUY (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 15:15:00 | 1310.90 | 1304.36 | 1303.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1317.60 | 1307.24 | 1304.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 10:15:00 | 1331.50 | 1338.05 | 1331.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 10:15:00 | 1331.50 | 1338.05 | 1331.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1331.50 | 1338.05 | 1331.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:00:00 | 1331.50 | 1338.05 | 1331.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 11:15:00 | 1329.70 | 1336.38 | 1331.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 11:45:00 | 1331.40 | 1336.38 | 1331.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 1326.00 | 1334.30 | 1331.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 13:00:00 | 1326.00 | 1334.30 | 1331.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 1321.70 | 1327.76 | 1328.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 1315.20 | 1325.25 | 1327.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 11:15:00 | 1302.80 | 1302.75 | 1309.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 11:45:00 | 1302.80 | 1302.75 | 1309.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1308.90 | 1304.26 | 1307.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 1310.10 | 1304.26 | 1307.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1312.30 | 1305.87 | 1308.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 1312.30 | 1305.87 | 1308.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1311.50 | 1306.99 | 1308.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 1309.10 | 1307.42 | 1308.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:30:00 | 1307.60 | 1307.31 | 1308.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 1319.10 | 1310.34 | 1309.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 14:15:00 | 1319.10 | 1310.34 | 1309.23 | EMA200 above EMA400 |

### Cycle 195 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 1299.00 | 1307.90 | 1308.74 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2025-09-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 11:15:00 | 1318.00 | 1309.95 | 1309.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 09:15:00 | 1327.60 | 1315.97 | 1312.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 1330.10 | 1333.28 | 1327.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 1330.10 | 1333.28 | 1327.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1330.10 | 1333.28 | 1327.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 1330.10 | 1333.28 | 1327.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1325.50 | 1331.28 | 1327.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 1325.50 | 1331.28 | 1327.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1321.60 | 1329.34 | 1327.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:45:00 | 1323.50 | 1329.34 | 1327.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 1327.00 | 1328.50 | 1327.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 1331.20 | 1328.40 | 1327.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:00:00 | 1332.30 | 1329.18 | 1327.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 14:45:00 | 1329.30 | 1328.14 | 1327.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 1331.00 | 1328.14 | 1327.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1331.00 | 1328.71 | 1327.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 1355.30 | 1328.71 | 1327.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 1327.10 | 1336.09 | 1334.81 | SL hit (close<static) qty=1.00 sl=1327.40 alert=retest2 |

### Cycle 197 — SELL (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 13:15:00 | 1325.50 | 1341.74 | 1342.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 1315.40 | 1336.47 | 1339.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 1309.80 | 1307.87 | 1318.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 11:00:00 | 1309.80 | 1307.87 | 1318.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1303.30 | 1306.86 | 1313.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:15:00 | 1303.10 | 1306.86 | 1313.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:15:00 | 1302.60 | 1305.87 | 1311.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 1326.00 | 1307.08 | 1310.39 | SL hit (close>static) qty=1.00 sl=1319.80 alert=retest2 |

### Cycle 198 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 1335.20 | 1312.71 | 1312.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 1348.70 | 1328.93 | 1321.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1350.80 | 1353.53 | 1342.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 12:15:00 | 1343.50 | 1350.98 | 1344.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 1343.50 | 1350.98 | 1344.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:45:00 | 1344.90 | 1350.98 | 1344.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 1352.00 | 1351.19 | 1344.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:30:00 | 1362.80 | 1352.67 | 1346.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 1337.90 | 1348.54 | 1347.83 | SL hit (close<static) qty=1.00 sl=1342.10 alert=retest2 |

### Cycle 199 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 1334.80 | 1345.79 | 1346.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 11:15:00 | 1331.60 | 1342.95 | 1345.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 1339.80 | 1336.80 | 1339.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 12:15:00 | 1339.80 | 1336.80 | 1339.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1339.80 | 1336.80 | 1339.92 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 1343.60 | 1340.95 | 1340.77 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 12:15:00 | 1339.30 | 1340.62 | 1340.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 13:15:00 | 1331.60 | 1338.81 | 1339.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1320.90 | 1312.14 | 1319.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1320.90 | 1312.14 | 1319.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1320.90 | 1312.14 | 1319.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 1320.90 | 1312.14 | 1319.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1331.00 | 1315.91 | 1320.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 1331.00 | 1315.91 | 1320.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1321.90 | 1317.60 | 1320.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 1322.10 | 1317.60 | 1320.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1328.00 | 1319.68 | 1320.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 1328.00 | 1319.68 | 1320.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 1331.70 | 1322.09 | 1321.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 1347.50 | 1328.92 | 1325.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 1364.50 | 1365.33 | 1357.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 1364.50 | 1365.33 | 1357.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 1359.00 | 1363.07 | 1358.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 1359.00 | 1363.07 | 1358.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1372.60 | 1364.98 | 1359.42 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 1354.40 | 1358.11 | 1358.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 1339.40 | 1347.76 | 1351.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 13:15:00 | 1347.80 | 1346.99 | 1350.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 13:15:00 | 1347.80 | 1346.99 | 1350.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1347.80 | 1346.99 | 1350.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 1347.80 | 1346.99 | 1350.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1349.80 | 1347.91 | 1350.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1370.00 | 1347.91 | 1350.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 1370.90 | 1352.50 | 1352.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1384.70 | 1358.94 | 1355.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 09:15:00 | 1429.50 | 1445.67 | 1437.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 1429.50 | 1445.67 | 1437.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1429.50 | 1445.67 | 1437.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 1429.50 | 1445.67 | 1437.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1425.00 | 1441.53 | 1436.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:30:00 | 1425.60 | 1441.53 | 1436.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 1424.10 | 1433.36 | 1433.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 14:15:00 | 1415.10 | 1429.71 | 1431.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 1427.70 | 1426.90 | 1429.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:00:00 | 1427.70 | 1426.90 | 1429.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1427.80 | 1427.08 | 1429.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:45:00 | 1429.00 | 1427.08 | 1429.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1429.70 | 1427.83 | 1429.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 1429.10 | 1427.83 | 1429.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1427.30 | 1427.72 | 1429.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:30:00 | 1431.50 | 1427.72 | 1429.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1429.10 | 1428.00 | 1429.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1422.80 | 1428.00 | 1429.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:00:00 | 1427.00 | 1419.69 | 1421.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 12:30:00 | 1426.10 | 1421.09 | 1422.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 13:45:00 | 1422.10 | 1421.05 | 1422.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1414.80 | 1419.80 | 1421.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:30:00 | 1418.30 | 1419.80 | 1421.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1415.90 | 1418.25 | 1420.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 1421.90 | 1418.25 | 1420.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 1435.00 | 1421.64 | 1421.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-12 11:15:00 | 1435.00 | 1421.64 | 1421.68 | SL hit (close>static) qty=1.00 sl=1431.40 alert=retest2 |

### Cycle 206 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 1433.70 | 1424.05 | 1422.77 | EMA200 above EMA400 |

### Cycle 207 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1420.40 | 1424.95 | 1425.42 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 1429.70 | 1425.41 | 1425.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 13:15:00 | 1431.10 | 1428.35 | 1427.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 1417.90 | 1428.48 | 1427.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 1417.90 | 1428.48 | 1427.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1417.90 | 1428.48 | 1427.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1417.90 | 1428.48 | 1427.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1424.30 | 1427.64 | 1427.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:15:00 | 1426.20 | 1427.64 | 1427.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:30:00 | 1426.80 | 1429.02 | 1428.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:15:00 | 1426.10 | 1429.02 | 1428.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 10:15:00 | 1421.90 | 1427.59 | 1428.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 209 — SELL (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 10:15:00 | 1421.90 | 1427.59 | 1428.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 12:15:00 | 1417.70 | 1425.42 | 1426.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 1430.10 | 1421.08 | 1423.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 10:15:00 | 1430.10 | 1421.08 | 1423.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1430.10 | 1421.08 | 1423.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 1430.10 | 1421.08 | 1423.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 1426.40 | 1422.14 | 1423.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:15:00 | 1432.20 | 1422.14 | 1423.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 1420.80 | 1416.66 | 1419.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:00:00 | 1420.80 | 1416.66 | 1419.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1427.60 | 1418.85 | 1420.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 1427.60 | 1418.85 | 1420.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 1425.00 | 1420.08 | 1420.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 1431.00 | 1420.08 | 1420.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 09:15:00 | 1434.10 | 1422.88 | 1421.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 1448.00 | 1433.21 | 1429.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 1448.50 | 1451.02 | 1442.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:45:00 | 1449.90 | 1451.02 | 1442.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 13:15:00 | 1446.70 | 1449.51 | 1443.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 13:30:00 | 1442.90 | 1449.51 | 1443.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1446.50 | 1448.91 | 1443.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 14:45:00 | 1443.00 | 1448.91 | 1443.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 15:15:00 | 1444.00 | 1447.93 | 1443.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 09:15:00 | 1441.80 | 1447.93 | 1443.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1449.60 | 1448.26 | 1444.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:15:00 | 1453.00 | 1448.26 | 1444.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 12:15:00 | 1444.10 | 1446.16 | 1446.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 1444.10 | 1446.16 | 1446.33 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 1449.90 | 1446.78 | 1446.52 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1438.10 | 1445.04 | 1445.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 1435.20 | 1442.08 | 1444.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 15:15:00 | 1442.30 | 1441.40 | 1443.12 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-03 09:15:00 | 1436.80 | 1441.40 | 1443.12 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1433.40 | 1428.70 | 1432.27 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1433.40 | 1428.70 | 1432.27 | SL hit (close>ema400) qty=1.00 sl=1432.27 alert=retest1 |

### Cycle 214 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 1442.00 | 1433.28 | 1433.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 1444.80 | 1435.58 | 1434.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 10:15:00 | 1445.30 | 1445.93 | 1441.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 11:00:00 | 1445.30 | 1445.93 | 1441.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 1436.70 | 1444.08 | 1440.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 1436.70 | 1444.08 | 1440.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1429.30 | 1441.13 | 1439.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 1429.30 | 1441.13 | 1439.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 1427.60 | 1438.42 | 1438.62 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 1440.80 | 1436.66 | 1436.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 1448.90 | 1441.60 | 1439.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1435.80 | 1440.78 | 1439.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1435.80 | 1440.78 | 1439.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1435.80 | 1440.78 | 1439.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 1433.30 | 1440.78 | 1439.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1443.20 | 1441.26 | 1439.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 09:30:00 | 1446.80 | 1443.05 | 1441.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:15:00 | 1451.40 | 1443.05 | 1441.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 09:30:00 | 1446.50 | 1448.21 | 1445.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 12:15:00 | 1435.50 | 1444.07 | 1444.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 217 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 1435.50 | 1444.07 | 1444.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 1427.90 | 1440.83 | 1442.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 1397.20 | 1396.85 | 1409.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 1397.20 | 1396.85 | 1409.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1407.40 | 1400.88 | 1408.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 1411.00 | 1400.88 | 1408.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1409.90 | 1402.68 | 1408.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 1414.90 | 1402.68 | 1408.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1411.00 | 1404.35 | 1408.88 | EMA400 retest candle locked (from downside) |

### Cycle 218 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 1424.20 | 1412.58 | 1411.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 15:15:00 | 1430.00 | 1417.94 | 1414.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 1431.30 | 1432.60 | 1426.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 1431.30 | 1432.60 | 1426.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 1431.30 | 1432.60 | 1426.03 | EMA400 retest candle locked (from upside) |

### Cycle 219 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 1418.10 | 1425.77 | 1426.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 12:15:00 | 1410.80 | 1419.67 | 1422.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 1423.20 | 1419.61 | 1421.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1423.20 | 1419.61 | 1421.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1423.20 | 1419.61 | 1421.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1423.20 | 1419.61 | 1421.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1430.00 | 1421.69 | 1422.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:15:00 | 1418.90 | 1421.69 | 1422.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 1429.80 | 1423.82 | 1423.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 220 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1429.80 | 1423.82 | 1423.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 1438.80 | 1426.81 | 1424.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 1423.70 | 1431.22 | 1428.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 10:15:00 | 1423.70 | 1431.22 | 1428.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1423.70 | 1431.22 | 1428.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:00:00 | 1423.70 | 1431.22 | 1428.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — SELL (started 2026-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 11:15:00 | 1395.30 | 1424.03 | 1425.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 11:15:00 | 1383.80 | 1402.57 | 1412.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 12:15:00 | 1379.60 | 1376.77 | 1385.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 13:00:00 | 1379.60 | 1376.77 | 1385.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1379.40 | 1374.45 | 1378.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 15:00:00 | 1379.40 | 1374.45 | 1378.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1376.70 | 1374.90 | 1378.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 1366.10 | 1374.90 | 1378.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 1337.00 | 1332.14 | 1331.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 222 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 1337.00 | 1332.14 | 1331.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1344.10 | 1334.53 | 1332.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 1340.30 | 1343.67 | 1339.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 1340.30 | 1343.67 | 1339.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1340.30 | 1343.67 | 1339.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 1340.30 | 1343.67 | 1339.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1338.00 | 1342.54 | 1339.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 1336.10 | 1342.54 | 1339.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 1335.90 | 1341.21 | 1339.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:00:00 | 1335.90 | 1341.21 | 1339.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 1333.50 | 1339.67 | 1338.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:45:00 | 1333.00 | 1339.67 | 1338.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — SELL (started 2026-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 13:15:00 | 1324.60 | 1336.65 | 1337.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 1321.50 | 1330.28 | 1333.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1316.30 | 1309.41 | 1318.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 1316.30 | 1309.41 | 1318.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 1316.30 | 1309.41 | 1318.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 1316.30 | 1309.41 | 1318.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 1313.90 | 1310.31 | 1318.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:30:00 | 1319.60 | 1310.31 | 1318.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 1320.90 | 1312.43 | 1318.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 1320.90 | 1312.43 | 1318.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 1318.00 | 1313.54 | 1318.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1327.30 | 1316.33 | 1319.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1335.10 | 1320.09 | 1320.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 1335.10 | 1320.09 | 1320.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 1335.60 | 1323.19 | 1321.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 1339.50 | 1326.45 | 1323.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 1338.80 | 1342.58 | 1335.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 13:15:00 | 1338.80 | 1342.58 | 1335.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1338.80 | 1342.58 | 1335.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 1337.00 | 1342.58 | 1335.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1333.20 | 1340.70 | 1335.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 1333.20 | 1340.70 | 1335.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1342.00 | 1340.96 | 1335.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 1321.00 | 1340.96 | 1335.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 1323.70 | 1337.51 | 1334.59 | EMA400 retest candle locked (from upside) |

### Cycle 225 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 1316.60 | 1329.71 | 1331.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 1311.30 | 1326.03 | 1329.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 10:15:00 | 1325.00 | 1319.64 | 1324.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 10:15:00 | 1325.00 | 1319.64 | 1324.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 1325.00 | 1319.64 | 1324.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 1325.00 | 1319.64 | 1324.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 11:15:00 | 1318.40 | 1319.39 | 1323.62 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 10:15:00 | 1328.00 | 1325.04 | 1324.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 1334.20 | 1329.00 | 1326.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1338.50 | 1349.91 | 1342.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 1338.50 | 1349.91 | 1342.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1338.50 | 1349.91 | 1342.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 1338.50 | 1349.91 | 1342.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1338.70 | 1347.67 | 1341.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:00:00 | 1338.70 | 1347.67 | 1341.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1323.50 | 1340.97 | 1339.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:45:00 | 1315.90 | 1340.97 | 1339.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 1337.30 | 1339.54 | 1339.21 | EMA400 retest candle locked (from upside) |

### Cycle 227 — SELL (started 2026-02-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 09:15:00 | 1330.30 | 1337.69 | 1338.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 10:15:00 | 1319.30 | 1334.02 | 1336.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 1334.00 | 1333.29 | 1335.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 1334.00 | 1333.29 | 1335.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1334.00 | 1333.29 | 1335.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 1334.00 | 1333.29 | 1335.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1341.10 | 1334.85 | 1336.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 1342.50 | 1334.85 | 1336.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 228 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1347.40 | 1337.36 | 1337.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1351.70 | 1341.68 | 1339.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 11:15:00 | 1360.50 | 1361.42 | 1353.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:45:00 | 1356.00 | 1361.42 | 1353.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1354.70 | 1360.07 | 1353.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:45:00 | 1354.40 | 1360.07 | 1353.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1353.60 | 1358.78 | 1353.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:00:00 | 1353.60 | 1358.78 | 1353.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 1359.60 | 1358.94 | 1354.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 1362.30 | 1358.95 | 1354.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 10:00:00 | 1360.00 | 1359.16 | 1355.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 10:45:00 | 1360.60 | 1360.33 | 1356.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:45:00 | 1360.40 | 1359.56 | 1356.76 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 1357.00 | 1359.12 | 1357.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:15:00 | 1364.50 | 1359.12 | 1357.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 1354.80 | 1357.90 | 1356.84 | SL hit (close<static) qty=1.00 sl=1355.90 alert=retest2 |

### Cycle 229 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1400.20 | 1409.54 | 1409.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 12:15:00 | 1399.80 | 1404.71 | 1407.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 1416.30 | 1404.87 | 1406.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 1416.30 | 1404.87 | 1406.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1416.30 | 1404.87 | 1406.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 1417.00 | 1404.87 | 1406.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1406.70 | 1405.48 | 1406.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 11:45:00 | 1407.50 | 1405.48 | 1406.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 230 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 1416.60 | 1407.71 | 1407.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 13:15:00 | 1420.30 | 1410.22 | 1408.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 1420.60 | 1421.70 | 1417.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 1420.60 | 1421.70 | 1417.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1420.60 | 1421.70 | 1417.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 1419.40 | 1421.70 | 1417.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1413.00 | 1419.96 | 1417.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1413.00 | 1419.96 | 1417.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1406.00 | 1417.17 | 1416.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1406.00 | 1417.17 | 1416.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 231 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 1400.30 | 1412.94 | 1414.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 1391.50 | 1408.65 | 1412.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 09:15:00 | 1392.80 | 1389.46 | 1397.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 10:15:00 | 1394.60 | 1389.46 | 1397.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 1404.50 | 1392.34 | 1397.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:00:00 | 1404.50 | 1392.34 | 1397.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 1405.50 | 1394.98 | 1398.26 | EMA400 retest candle locked (from downside) |

### Cycle 232 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 1417.70 | 1403.33 | 1401.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 1420.10 | 1413.11 | 1408.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 10:15:00 | 1413.10 | 1415.01 | 1410.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:45:00 | 1411.60 | 1415.01 | 1410.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1410.00 | 1414.09 | 1411.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:30:00 | 1409.60 | 1414.09 | 1411.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 1412.50 | 1413.77 | 1411.19 | EMA400 retest candle locked (from upside) |

### Cycle 233 — SELL (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 09:15:00 | 1392.70 | 1409.28 | 1409.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-26 10:15:00 | 1386.40 | 1404.70 | 1407.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 12:15:00 | 1385.20 | 1384.16 | 1392.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-27 13:00:00 | 1385.20 | 1384.16 | 1392.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1380.10 | 1383.77 | 1390.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 14:45:00 | 1391.00 | 1383.77 | 1390.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1392.20 | 1339.56 | 1340.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:00:00 | 1392.20 | 1339.56 | 1340.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 234 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 1380.30 | 1347.71 | 1344.03 | EMA200 above EMA400 |

### Cycle 235 — SELL (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 10:15:00 | 1360.30 | 1375.63 | 1376.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1345.50 | 1362.38 | 1368.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 1323.70 | 1318.40 | 1330.21 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 11:30:00 | 1311.90 | 1317.66 | 1328.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-17 12:00:00 | 1314.70 | 1317.66 | 1328.80 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-18 11:15:00 | 1310.80 | 1310.93 | 1319.86 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 1321.90 | 1312.61 | 1319.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-18 12:15:00 | 1321.90 | 1312.61 | 1319.04 | SL hit (close>ema400) qty=1.00 sl=1319.04 alert=retest1 |

### Cycle 236 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 1321.60 | 1299.48 | 1297.36 | EMA200 above EMA400 |

### Cycle 237 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1277.50 | 1305.18 | 1305.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 10:15:00 | 1268.20 | 1297.78 | 1302.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1239.90 | 1236.43 | 1255.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1239.90 | 1236.43 | 1255.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1239.90 | 1236.43 | 1255.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 1223.80 | 1245.67 | 1252.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 1224.30 | 1231.09 | 1234.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 13:45:00 | 1226.40 | 1226.40 | 1230.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1262.10 | 1236.82 | 1234.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 238 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1262.10 | 1236.82 | 1234.25 | EMA200 above EMA400 |

### Cycle 239 — SELL (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 12:15:00 | 1236.10 | 1247.54 | 1249.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 1231.00 | 1242.19 | 1246.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 1250.80 | 1242.16 | 1245.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 1250.80 | 1242.16 | 1245.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1250.80 | 1242.16 | 1245.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:00:00 | 1250.80 | 1242.16 | 1245.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 1247.90 | 1243.30 | 1245.62 | EMA400 retest candle locked (from downside) |

### Cycle 240 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 1253.20 | 1247.48 | 1247.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 1253.50 | 1248.68 | 1247.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 1244.70 | 1248.18 | 1247.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 1244.70 | 1248.18 | 1247.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 1244.70 | 1248.18 | 1247.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:00:00 | 1244.70 | 1248.18 | 1247.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 1246.20 | 1247.78 | 1247.54 | EMA400 retest candle locked (from upside) |

### Cycle 241 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 1240.80 | 1246.36 | 1246.94 | EMA200 below EMA400 |

### Cycle 242 — BUY (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 14:15:00 | 1253.80 | 1248.14 | 1247.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 1298.00 | 1259.21 | 1252.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 13:15:00 | 1385.70 | 1385.73 | 1366.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 14:00:00 | 1385.70 | 1385.73 | 1366.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 1380.60 | 1383.88 | 1370.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 1375.00 | 1383.88 | 1370.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 1387.30 | 1393.46 | 1385.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 1387.30 | 1393.46 | 1385.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 1388.00 | 1392.37 | 1386.01 | EMA400 retest candle locked (from upside) |

### Cycle 243 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 1374.00 | 1383.69 | 1384.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 15:15:00 | 1372.50 | 1379.88 | 1382.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-05 14:15:00 | 1316.40 | 1316.25 | 1327.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-05 15:00:00 | 1316.40 | 1316.25 | 1327.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1279.80 | 1308.60 | 1321.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:30:00 | 1277.00 | 1295.08 | 1311.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 13:45:00 | 1274.70 | 1290.52 | 1308.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:30:00 | 1274.50 | 1284.10 | 1293.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-16 12:15:00 | 1150.40 | 2024-04-22 14:15:00 | 1159.15 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-04-16 14:15:00 | 1149.15 | 2024-04-22 14:15:00 | 1159.15 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-04-18 09:15:00 | 1150.90 | 2024-04-22 14:15:00 | 1159.15 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2024-04-18 14:30:00 | 1150.00 | 2024-04-22 14:15:00 | 1159.15 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-05-02 15:15:00 | 1196.00 | 2024-05-03 11:15:00 | 1182.60 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-05-03 09:45:00 | 1197.30 | 2024-05-03 11:15:00 | 1182.60 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-05-03 12:30:00 | 1198.00 | 2024-05-09 12:15:00 | 1201.70 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2024-05-03 14:00:00 | 1196.30 | 2024-05-09 12:15:00 | 1201.70 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2024-05-14 15:00:00 | 1175.30 | 2024-05-17 14:15:00 | 1183.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-05-15 13:15:00 | 1176.65 | 2024-05-17 14:15:00 | 1183.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-05-16 11:00:00 | 1175.10 | 2024-05-17 14:15:00 | 1183.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-05-16 12:45:00 | 1171.30 | 2024-05-17 15:15:00 | 1183.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-05-17 09:15:00 | 1170.55 | 2024-05-17 15:15:00 | 1183.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-05-17 12:15:00 | 1176.95 | 2024-05-17 15:15:00 | 1183.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-05-17 13:00:00 | 1173.90 | 2024-05-17 15:15:00 | 1183.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-05-27 09:15:00 | 1185.25 | 2024-05-27 10:15:00 | 1169.60 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-06-20 11:00:00 | 1262.45 | 2024-06-24 10:15:00 | 1270.95 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-06-20 12:00:00 | 1262.10 | 2024-06-24 11:15:00 | 1273.55 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-06-20 13:30:00 | 1260.15 | 2024-06-24 11:15:00 | 1273.55 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2024-06-21 09:45:00 | 1258.90 | 2024-06-24 11:15:00 | 1273.55 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-06-21 15:15:00 | 1257.00 | 2024-06-24 11:15:00 | 1273.55 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-06-26 12:15:00 | 1290.05 | 2024-06-27 11:15:00 | 1276.40 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-06-26 13:30:00 | 1286.90 | 2024-06-27 11:15:00 | 1276.40 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-07-10 13:15:00 | 1282.50 | 2024-07-23 09:15:00 | 1282.15 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-07-11 12:15:00 | 1278.15 | 2024-07-23 09:15:00 | 1282.15 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2024-07-11 13:15:00 | 1277.90 | 2024-07-23 09:15:00 | 1282.15 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2024-07-12 09:30:00 | 1280.00 | 2024-07-23 09:15:00 | 1282.15 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2024-07-12 15:00:00 | 1290.65 | 2024-07-23 10:15:00 | 1290.05 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2024-07-15 12:00:00 | 1289.40 | 2024-07-23 10:15:00 | 1290.05 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2024-07-18 12:00:00 | 1290.00 | 2024-07-23 10:15:00 | 1290.05 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-07-18 13:45:00 | 1290.95 | 2024-07-23 10:15:00 | 1290.05 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-07-19 12:00:00 | 1311.25 | 2024-07-23 10:15:00 | 1290.05 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-07-22 12:30:00 | 1310.40 | 2024-07-23 10:15:00 | 1290.05 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-07-22 13:00:00 | 1311.00 | 2024-07-23 10:15:00 | 1290.05 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2024-07-22 13:45:00 | 1310.90 | 2024-07-23 10:15:00 | 1290.05 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-08-02 14:30:00 | 1423.25 | 2024-08-05 10:15:00 | 1410.20 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-08-02 15:15:00 | 1427.95 | 2024-08-05 10:15:00 | 1410.20 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-08-05 09:30:00 | 1424.40 | 2024-08-05 10:15:00 | 1410.20 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest1 | 2024-08-09 09:15:00 | 1467.60 | 2024-08-09 14:15:00 | 1456.60 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2024-08-16 11:00:00 | 1399.45 | 2024-08-19 10:15:00 | 1418.10 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-08-16 11:45:00 | 1399.00 | 2024-08-19 10:15:00 | 1418.10 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-08-16 13:00:00 | 1398.50 | 2024-08-19 10:15:00 | 1418.10 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-08-16 13:45:00 | 1398.35 | 2024-08-19 10:15:00 | 1418.10 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-08-26 11:15:00 | 1443.75 | 2024-08-27 13:15:00 | 1429.10 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-08-26 12:45:00 | 1443.95 | 2024-08-27 13:15:00 | 1429.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-08-26 13:15:00 | 1446.65 | 2024-08-27 13:15:00 | 1429.10 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-08-29 12:15:00 | 1445.05 | 2024-08-29 13:15:00 | 1435.75 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-09-04 12:30:00 | 1493.50 | 2024-09-06 10:15:00 | 1471.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-09-05 12:15:00 | 1493.50 | 2024-09-06 10:15:00 | 1471.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-09-05 13:45:00 | 1495.15 | 2024-09-06 10:15:00 | 1471.95 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-09-13 13:30:00 | 1530.55 | 2024-09-18 15:15:00 | 1534.05 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-09-13 15:15:00 | 1523.90 | 2024-09-18 15:15:00 | 1534.05 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2024-09-25 14:15:00 | 1620.95 | 2024-09-27 09:15:00 | 1582.45 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-09-26 11:45:00 | 1616.15 | 2024-09-27 09:15:00 | 1582.45 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-09-26 12:30:00 | 1621.00 | 2024-09-27 09:15:00 | 1582.45 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-10-15 14:45:00 | 1550.30 | 2024-10-17 11:15:00 | 1522.55 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-10-16 09:30:00 | 1550.50 | 2024-10-17 11:15:00 | 1522.55 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-10-16 12:15:00 | 1549.00 | 2024-10-17 11:15:00 | 1522.55 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-10-21 14:00:00 | 1502.85 | 2024-10-28 10:15:00 | 1479.10 | STOP_HIT | 1.00 | 1.58% |
| SELL | retest2 | 2024-10-21 14:30:00 | 1502.05 | 2024-10-28 10:15:00 | 1479.10 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest1 | 2024-10-30 09:15:00 | 1442.80 | 2024-10-31 14:15:00 | 1451.70 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest1 | 2024-10-30 10:15:00 | 1442.25 | 2024-10-31 14:15:00 | 1451.70 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-10-31 10:15:00 | 1432.35 | 2024-11-05 14:15:00 | 1447.90 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-10-31 11:30:00 | 1433.40 | 2024-11-05 14:15:00 | 1447.90 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-11-04 10:15:00 | 1434.35 | 2024-11-05 14:15:00 | 1447.90 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-11-04 10:45:00 | 1422.85 | 2024-11-05 14:15:00 | 1447.90 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-11-27 14:45:00 | 1519.90 | 2024-11-28 10:15:00 | 1501.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-11-27 15:15:00 | 1522.35 | 2024-11-28 10:15:00 | 1501.70 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-11-28 09:45:00 | 1517.00 | 2024-11-28 10:15:00 | 1501.70 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-12-03 10:15:00 | 1540.45 | 2024-12-04 14:15:00 | 1525.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-12-03 10:45:00 | 1540.00 | 2024-12-04 14:15:00 | 1525.10 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-12-03 12:45:00 | 1542.05 | 2024-12-04 14:15:00 | 1525.10 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-12-03 14:00:00 | 1542.20 | 2024-12-04 14:15:00 | 1525.10 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-12-06 11:45:00 | 1524.70 | 2024-12-11 15:15:00 | 1521.00 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-01-16 10:45:00 | 1409.25 | 2025-01-20 12:15:00 | 1455.70 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2025-01-16 12:00:00 | 1410.70 | 2025-01-20 12:15:00 | 1455.70 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-02-13 15:15:00 | 1389.00 | 2025-02-19 09:15:00 | 1319.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:15:00 | 1390.10 | 2025-02-19 09:15:00 | 1320.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:15:00 | 1389.00 | 2025-02-19 15:15:00 | 1331.00 | STOP_HIT | 0.50 | 4.18% |
| SELL | retest2 | 2025-02-14 10:15:00 | 1390.10 | 2025-02-19 15:15:00 | 1331.00 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2025-03-04 09:15:00 | 1291.00 | 2025-03-05 11:15:00 | 1315.85 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-04-01 12:15:00 | 1399.90 | 2025-04-07 09:15:00 | 1382.90 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-04-01 13:30:00 | 1400.30 | 2025-04-07 09:15:00 | 1382.90 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-04-25 09:15:00 | 1566.40 | 2025-04-25 09:15:00 | 1550.10 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-05-02 10:15:00 | 1569.80 | 2025-05-07 09:15:00 | 1545.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-05-02 11:15:00 | 1571.60 | 2025-05-07 09:15:00 | 1545.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-05-06 09:30:00 | 1578.20 | 2025-05-07 09:15:00 | 1545.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-05-07 10:30:00 | 1569.50 | 2025-05-08 13:15:00 | 1545.40 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-05-15 13:15:00 | 1537.70 | 2025-05-16 11:15:00 | 1550.30 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-05-15 15:00:00 | 1535.90 | 2025-05-16 11:15:00 | 1550.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-05-16 14:15:00 | 1534.50 | 2025-05-19 09:15:00 | 1557.80 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-05-30 13:00:00 | 1504.30 | 2025-06-02 13:15:00 | 1540.60 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-06-05 10:15:00 | 1624.70 | 2025-06-09 12:15:00 | 1586.80 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-06-06 09:30:00 | 1613.30 | 2025-06-09 12:15:00 | 1586.80 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-06-06 10:00:00 | 1613.00 | 2025-06-09 12:15:00 | 1586.80 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-06-17 11:45:00 | 1470.80 | 2025-06-25 11:15:00 | 1457.90 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-06-17 12:30:00 | 1471.40 | 2025-06-25 11:15:00 | 1457.90 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2025-06-17 13:45:00 | 1472.20 | 2025-06-25 11:15:00 | 1457.90 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-06-17 15:15:00 | 1472.60 | 2025-06-25 11:15:00 | 1457.90 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2025-06-19 10:30:00 | 1462.20 | 2025-06-25 11:15:00 | 1457.90 | STOP_HIT | 1.00 | 0.29% |
| SELL | retest2 | 2025-06-19 11:15:00 | 1464.80 | 2025-06-25 11:15:00 | 1457.90 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-06-19 12:30:00 | 1463.90 | 2025-06-25 11:15:00 | 1457.90 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-06-19 14:30:00 | 1461.10 | 2025-06-25 11:15:00 | 1457.90 | STOP_HIT | 1.00 | 0.22% |
| SELL | retest2 | 2025-06-23 10:00:00 | 1443.60 | 2025-06-25 11:15:00 | 1457.90 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-07-07 11:15:00 | 1381.00 | 2025-07-10 13:15:00 | 1379.90 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-07-08 09:15:00 | 1376.40 | 2025-07-10 13:15:00 | 1379.90 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-07-14 11:30:00 | 1363.40 | 2025-07-16 10:15:00 | 1371.60 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-07-14 13:45:00 | 1363.20 | 2025-07-16 10:15:00 | 1371.60 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-14 14:45:00 | 1364.60 | 2025-07-16 10:15:00 | 1371.60 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-07-22 10:30:00 | 1353.00 | 2025-07-31 09:15:00 | 1330.00 | STOP_HIT | 1.00 | 1.70% |
| SELL | retest2 | 2025-08-08 14:15:00 | 1294.80 | 2025-08-13 14:15:00 | 1305.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-08 14:45:00 | 1293.20 | 2025-08-13 14:15:00 | 1305.40 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-08-12 11:00:00 | 1294.60 | 2025-08-13 14:15:00 | 1305.40 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-08-12 11:45:00 | 1294.80 | 2025-08-13 14:15:00 | 1305.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-13 12:15:00 | 1295.90 | 2025-08-13 14:15:00 | 1305.40 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-08-13 13:00:00 | 1295.40 | 2025-08-13 14:15:00 | 1305.40 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-08-22 13:00:00 | 1335.20 | 2025-08-25 09:15:00 | 1319.90 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-10 13:00:00 | 1309.10 | 2025-09-11 14:15:00 | 1319.10 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-11 10:30:00 | 1307.60 | 2025-09-11 14:15:00 | 1319.10 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-19 09:15:00 | 1331.20 | 2025-09-23 10:15:00 | 1327.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-09-19 10:00:00 | 1332.30 | 2025-09-25 13:15:00 | 1325.50 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-09-19 14:45:00 | 1329.30 | 2025-09-25 13:15:00 | 1325.50 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-09-19 15:15:00 | 1331.00 | 2025-09-25 13:15:00 | 1325.50 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-09-22 09:15:00 | 1355.30 | 2025-09-25 13:15:00 | 1325.50 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-09-23 11:45:00 | 1331.40 | 2025-09-25 13:15:00 | 1325.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-30 10:15:00 | 1303.10 | 2025-09-30 14:15:00 | 1326.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-09-30 12:15:00 | 1302.60 | 2025-09-30 14:15:00 | 1326.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-10-06 14:30:00 | 1362.80 | 2025-10-08 09:15:00 | 1337.90 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1422.80 | 2025-11-12 11:15:00 | 1435.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-11-11 12:00:00 | 1427.00 | 2025-11-12 11:15:00 | 1435.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-11-11 12:30:00 | 1426.10 | 2025-11-12 11:15:00 | 1435.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-11-11 13:45:00 | 1422.10 | 2025-11-12 11:15:00 | 1435.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-18 11:15:00 | 1426.20 | 2025-11-19 10:15:00 | 1421.90 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-11-19 09:30:00 | 1426.80 | 2025-11-19 10:15:00 | 1421.90 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-11-19 10:15:00 | 1426.10 | 2025-11-19 10:15:00 | 1421.90 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2025-11-28 10:15:00 | 1453.00 | 2025-12-01 12:15:00 | 1444.10 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-12-03 09:15:00 | 1436.80 | 2025-12-04 12:15:00 | 1433.40 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2025-12-16 09:30:00 | 1446.80 | 2025-12-17 12:15:00 | 1435.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-12-16 10:15:00 | 1451.40 | 2025-12-17 12:15:00 | 1435.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-17 09:30:00 | 1446.50 | 2025-12-17 12:15:00 | 1435.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-12-31 09:15:00 | 1418.90 | 2025-12-31 12:15:00 | 1429.80 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1366.10 | 2026-01-14 15:15:00 | 1337.00 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2026-02-05 09:15:00 | 1362.30 | 2026-02-06 10:15:00 | 1354.80 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2026-02-05 10:00:00 | 1360.00 | 2026-02-13 15:15:00 | 1400.20 | STOP_HIT | 1.00 | 2.96% |
| BUY | retest2 | 2026-02-05 10:45:00 | 1360.60 | 2026-02-13 15:15:00 | 1400.20 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2026-02-05 13:45:00 | 1360.40 | 2026-02-13 15:15:00 | 1400.20 | STOP_HIT | 1.00 | 2.93% |
| BUY | retest2 | 2026-02-06 09:15:00 | 1364.50 | 2026-02-13 15:15:00 | 1400.20 | STOP_HIT | 1.00 | 2.62% |
| BUY | retest2 | 2026-02-06 12:15:00 | 1364.20 | 2026-02-13 15:15:00 | 1400.20 | STOP_HIT | 1.00 | 2.64% |
| SELL | retest1 | 2026-03-17 11:30:00 | 1311.90 | 2026-03-18 12:15:00 | 1321.90 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest1 | 2026-03-17 12:00:00 | 1314.70 | 2026-03-18 12:15:00 | 1321.90 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest1 | 2026-03-18 11:15:00 | 1310.80 | 2026-03-18 12:15:00 | 1321.90 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1298.80 | 2026-03-24 12:15:00 | 1321.60 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-04-02 09:15:00 | 1223.80 | 2026-04-08 09:15:00 | 1262.10 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2026-04-07 09:15:00 | 1224.30 | 2026-04-08 09:15:00 | 1262.10 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2026-04-07 13:45:00 | 1226.40 | 2026-04-08 09:15:00 | 1262.10 | STOP_HIT | 1.00 | -2.91% |
