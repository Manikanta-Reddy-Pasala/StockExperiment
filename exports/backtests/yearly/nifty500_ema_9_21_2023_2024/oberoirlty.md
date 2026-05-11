# Oberoi Realty Ltd. (OBEROIRLTY)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1710.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 232 |
| ALERT1 | 156 |
| ALERT2 | 155 |
| ALERT2_SKIP | 115 |
| ALERT3 | 355 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 126 |
| PARTIAL | 16 |
| TARGET_HIT | 4 |
| STOP_HIT | 127 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 147 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 45 / 102
- **Target hits / Stop hits / Partials:** 4 / 127 / 16
- **Avg / median % per leg:** 0.57% / -0.91%
- **Sum % (uncompounded):** 84.25%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 60 | 11 | 18.3% | 1 | 59 | 0 | -0.41% | -24.8% |
| BUY @ 2nd Alert (retest1) | 3 | 1 | 33.3% | 0 | 3 | 0 | -0.09% | -0.3% |
| BUY @ 3rd Alert (retest2) | 57 | 10 | 17.5% | 1 | 56 | 0 | -0.43% | -24.5% |
| SELL (all) | 87 | 34 | 39.1% | 3 | 68 | 16 | 1.25% | 109.1% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.93% | 8.8% |
| SELL @ 3rd Alert (retest2) | 84 | 32 | 38.1% | 3 | 66 | 15 | 1.19% | 100.3% |
| retest1 (combined) | 6 | 3 | 50.0% | 0 | 5 | 1 | 1.42% | 8.5% |
| retest2 (combined) | 141 | 42 | 29.8% | 4 | 122 | 15 | 0.54% | 75.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-17 10:15:00 | 920.40 | 959.13 | 962.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-18 09:15:00 | 910.45 | 927.18 | 942.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 10:15:00 | 906.65 | 902.41 | 918.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-19 14:15:00 | 922.50 | 908.28 | 916.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 922.50 | 908.28 | 916.25 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 13:15:00 | 923.90 | 919.31 | 919.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 14:15:00 | 927.05 | 920.86 | 919.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 09:15:00 | 921.00 | 921.51 | 920.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 09:15:00 | 921.00 | 921.51 | 920.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 09:15:00 | 921.00 | 921.51 | 920.26 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 15:15:00 | 919.00 | 923.09 | 923.21 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-25 09:15:00 | 928.50 | 924.17 | 923.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 14:15:00 | 939.25 | 929.20 | 926.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 11:15:00 | 929.80 | 933.81 | 929.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-26 11:15:00 | 929.80 | 933.81 | 929.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 11:15:00 | 929.80 | 933.81 | 929.96 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 10:15:00 | 927.85 | 930.48 | 930.60 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-30 14:15:00 | 932.80 | 931.02 | 930.79 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 11:15:00 | 924.70 | 929.85 | 930.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 12:15:00 | 921.75 | 928.23 | 929.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 14:15:00 | 931.00 | 927.95 | 929.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 14:15:00 | 931.00 | 927.95 | 929.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 931.00 | 927.95 | 929.16 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 949.25 | 932.62 | 931.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 10:15:00 | 959.00 | 944.90 | 938.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 12:15:00 | 956.00 | 956.97 | 950.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-08 09:15:00 | 977.85 | 980.27 | 974.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 977.85 | 980.27 | 974.35 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 15:15:00 | 966.85 | 971.53 | 971.91 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 10:15:00 | 979.80 | 973.42 | 972.72 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 13:15:00 | 968.55 | 972.33 | 972.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 14:15:00 | 965.90 | 971.04 | 971.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-13 10:15:00 | 969.20 | 965.85 | 967.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 10:15:00 | 969.20 | 965.85 | 967.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 969.20 | 965.85 | 967.59 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 14:15:00 | 983.00 | 971.43 | 969.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 09:15:00 | 986.35 | 976.61 | 972.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 12:15:00 | 1012.70 | 1014.04 | 999.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-15 14:15:00 | 995.20 | 1009.52 | 999.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 14:15:00 | 995.20 | 1009.52 | 999.54 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 12:15:00 | 1005.00 | 1010.50 | 1010.57 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 13:15:00 | 1011.70 | 1010.74 | 1010.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 14:15:00 | 1015.40 | 1011.67 | 1011.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 10:15:00 | 1006.25 | 1011.06 | 1011.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 10:15:00 | 1006.25 | 1011.06 | 1011.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 1006.25 | 1011.06 | 1011.00 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-06-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 11:15:00 | 1007.50 | 1010.35 | 1010.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 12:15:00 | 994.00 | 1007.08 | 1009.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 991.70 | 985.31 | 989.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 991.70 | 985.31 | 989.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 991.70 | 985.31 | 989.82 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 11:15:00 | 990.50 | 985.96 | 985.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-03 15:15:00 | 1001.90 | 991.97 | 988.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-04 14:15:00 | 989.35 | 995.57 | 992.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-04 14:15:00 | 989.35 | 995.57 | 992.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 989.35 | 995.57 | 992.60 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 13:15:00 | 1016.30 | 1019.64 | 1020.05 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 1029.10 | 1021.80 | 1020.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 11:15:00 | 1037.35 | 1024.91 | 1022.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 09:15:00 | 1028.85 | 1032.66 | 1027.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-12 09:15:00 | 1028.85 | 1032.66 | 1027.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 1028.85 | 1032.66 | 1027.94 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-21 11:15:00 | 1055.65 | 1064.97 | 1065.07 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 15:15:00 | 1065.00 | 1062.75 | 1062.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 09:15:00 | 1085.05 | 1067.21 | 1064.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-25 12:15:00 | 1066.10 | 1071.00 | 1067.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 12:15:00 | 1066.10 | 1071.00 | 1067.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 12:15:00 | 1066.10 | 1071.00 | 1067.44 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 12:15:00 | 1099.40 | 1104.66 | 1105.04 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-01 14:15:00 | 1113.30 | 1105.43 | 1105.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-02 09:15:00 | 1146.00 | 1114.43 | 1109.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-02 10:15:00 | 1106.60 | 1112.87 | 1109.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 10:15:00 | 1106.60 | 1112.87 | 1109.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 10:15:00 | 1106.60 | 1112.87 | 1109.18 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 1093.25 | 1105.59 | 1106.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 09:15:00 | 1082.80 | 1102.03 | 1104.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 14:15:00 | 1095.65 | 1093.54 | 1098.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-03 15:15:00 | 1100.75 | 1094.98 | 1099.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 15:15:00 | 1100.75 | 1094.98 | 1099.11 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 12:15:00 | 1115.15 | 1101.43 | 1101.06 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-07 11:15:00 | 1095.60 | 1100.83 | 1101.21 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2023-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 15:15:00 | 1107.00 | 1100.20 | 1099.88 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 09:15:00 | 1087.10 | 1097.58 | 1098.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 10:15:00 | 1078.60 | 1093.78 | 1096.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-09 14:15:00 | 1089.95 | 1089.14 | 1093.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-10 09:15:00 | 1092.40 | 1089.29 | 1092.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-10 09:15:00 | 1092.40 | 1089.29 | 1092.58 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 09:15:00 | 1098.60 | 1092.24 | 1092.14 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 10:15:00 | 1087.75 | 1091.35 | 1091.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 14:15:00 | 1079.30 | 1087.91 | 1089.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 09:15:00 | 1074.25 | 1072.42 | 1078.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 10:15:00 | 1070.40 | 1072.02 | 1077.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 1070.40 | 1072.02 | 1077.65 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-08-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-17 10:15:00 | 1083.40 | 1079.72 | 1079.54 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-08-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 11:15:00 | 1076.00 | 1078.97 | 1079.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 12:15:00 | 1071.55 | 1077.49 | 1078.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 14:15:00 | 1076.85 | 1076.14 | 1077.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 14:15:00 | 1076.85 | 1076.14 | 1077.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 14:15:00 | 1076.85 | 1076.14 | 1077.65 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 1089.00 | 1076.04 | 1074.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 10:15:00 | 1097.20 | 1080.27 | 1076.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 15:15:00 | 1093.25 | 1094.10 | 1089.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-24 15:15:00 | 1099.35 | 1099.64 | 1095.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 1099.35 | 1099.64 | 1095.12 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-08-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 11:15:00 | 1083.50 | 1091.29 | 1092.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 15:15:00 | 1078.15 | 1086.20 | 1088.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 1096.20 | 1088.20 | 1088.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 1096.20 | 1088.20 | 1088.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 1096.20 | 1088.20 | 1088.84 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 10:15:00 | 1096.60 | 1089.88 | 1089.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 13:15:00 | 1097.65 | 1092.86 | 1091.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 14:15:00 | 1119.55 | 1122.17 | 1114.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 14:15:00 | 1119.55 | 1122.17 | 1114.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 1119.55 | 1122.17 | 1114.02 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 1136.60 | 1169.32 | 1169.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 1132.00 | 1147.09 | 1157.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 1154.30 | 1140.22 | 1147.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 1154.30 | 1140.22 | 1147.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 1154.30 | 1140.22 | 1147.22 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-09-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 14:15:00 | 1165.60 | 1153.20 | 1151.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 11:15:00 | 1172.45 | 1161.99 | 1156.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-18 10:15:00 | 1167.35 | 1168.43 | 1162.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-18 12:15:00 | 1152.00 | 1164.51 | 1161.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 12:15:00 | 1152.00 | 1164.51 | 1161.99 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 1146.35 | 1159.16 | 1160.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 10:15:00 | 1134.00 | 1154.13 | 1157.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 14:15:00 | 1134.95 | 1125.92 | 1130.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 14:15:00 | 1134.95 | 1125.92 | 1130.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 14:15:00 | 1134.95 | 1125.92 | 1130.98 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 13:15:00 | 1135.15 | 1131.40 | 1131.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 15:15:00 | 1138.00 | 1133.54 | 1132.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 11:15:00 | 1133.05 | 1135.97 | 1133.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 11:15:00 | 1133.05 | 1135.97 | 1133.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 1133.05 | 1135.97 | 1133.93 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 1130.00 | 1132.46 | 1132.62 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-09-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 09:15:00 | 1141.50 | 1133.87 | 1133.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 13:15:00 | 1157.90 | 1144.31 | 1138.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 10:15:00 | 1148.10 | 1148.17 | 1142.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 11:15:00 | 1149.65 | 1148.46 | 1143.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 11:15:00 | 1149.65 | 1148.46 | 1143.27 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-10-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 11:15:00 | 1117.65 | 1137.90 | 1140.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 12:15:00 | 1107.00 | 1131.72 | 1137.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 1146.40 | 1128.90 | 1133.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 1146.40 | 1128.90 | 1133.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 1146.40 | 1128.90 | 1133.72 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 1140.40 | 1116.14 | 1113.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 10:15:00 | 1168.00 | 1126.51 | 1118.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 1149.10 | 1150.68 | 1140.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-12 09:15:00 | 1143.90 | 1148.79 | 1142.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 1143.90 | 1148.79 | 1142.09 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 14:15:00 | 1128.80 | 1141.13 | 1142.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 15:15:00 | 1127.05 | 1138.31 | 1141.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 13:15:00 | 1135.80 | 1135.69 | 1138.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 1132.75 | 1133.13 | 1136.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 1132.75 | 1133.13 | 1136.52 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-27 13:15:00 | 1098.95 | 1082.61 | 1081.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 09:15:00 | 1113.65 | 1093.20 | 1087.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 09:15:00 | 1229.15 | 1237.69 | 1220.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 11:15:00 | 1221.40 | 1232.76 | 1221.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 1221.40 | 1232.76 | 1221.48 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2023-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 11:15:00 | 1384.65 | 1389.21 | 1389.52 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 12:15:00 | 1396.00 | 1390.57 | 1390.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-28 13:15:00 | 1402.80 | 1393.01 | 1391.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 14:15:00 | 1401.80 | 1404.88 | 1399.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 15:15:00 | 1398.00 | 1403.50 | 1399.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 15:15:00 | 1398.00 | 1403.50 | 1399.79 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-11-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-30 11:15:00 | 1384.60 | 1397.81 | 1397.99 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 14:15:00 | 1401.50 | 1397.99 | 1397.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 15:15:00 | 1419.95 | 1402.38 | 1399.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 10:15:00 | 1429.30 | 1434.55 | 1425.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 11:15:00 | 1416.60 | 1430.96 | 1424.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 11:15:00 | 1416.60 | 1430.96 | 1424.24 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 09:15:00 | 1437.05 | 1449.22 | 1449.91 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2023-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 13:15:00 | 1461.15 | 1451.40 | 1450.55 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2023-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 09:15:00 | 1430.50 | 1447.07 | 1448.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 10:15:00 | 1426.35 | 1442.92 | 1446.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 12:15:00 | 1431.80 | 1426.70 | 1433.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 12:15:00 | 1431.80 | 1426.70 | 1433.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 1431.80 | 1426.70 | 1433.92 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-12-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 10:15:00 | 1449.10 | 1436.61 | 1436.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 1469.50 | 1443.18 | 1439.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 13:15:00 | 1475.85 | 1478.61 | 1466.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 1464.70 | 1475.83 | 1467.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 1464.70 | 1475.83 | 1467.85 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2023-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-18 12:15:00 | 1435.00 | 1460.49 | 1462.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-19 09:15:00 | 1427.15 | 1447.52 | 1454.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 1390.75 | 1389.81 | 1406.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-22 09:15:00 | 1396.00 | 1391.56 | 1404.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 1396.00 | 1391.56 | 1404.11 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2023-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-27 09:15:00 | 1412.25 | 1403.52 | 1403.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-28 10:15:00 | 1426.00 | 1414.80 | 1409.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 15:15:00 | 1427.00 | 1434.92 | 1428.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 15:15:00 | 1427.00 | 1434.92 | 1428.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 15:15:00 | 1427.00 | 1434.92 | 1428.35 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 15:15:00 | 1489.50 | 1502.40 | 1503.32 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 10:15:00 | 1515.00 | 1505.15 | 1504.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 11:15:00 | 1519.40 | 1508.00 | 1505.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 1511.55 | 1520.91 | 1514.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 1511.55 | 1520.91 | 1514.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 1511.55 | 1520.91 | 1514.38 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 1526.80 | 1551.14 | 1551.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 1521.00 | 1532.53 | 1540.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 13:15:00 | 1534.70 | 1532.81 | 1539.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 1520.20 | 1513.45 | 1521.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 1520.20 | 1513.45 | 1521.17 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-01-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 14:15:00 | 1322.95 | 1317.04 | 1316.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 15:15:00 | 1326.00 | 1318.83 | 1317.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 10:15:00 | 1314.30 | 1318.23 | 1317.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 10:15:00 | 1314.30 | 1318.23 | 1317.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 1314.30 | 1318.23 | 1317.67 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 11:15:00 | 1301.30 | 1314.84 | 1316.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 14:15:00 | 1287.70 | 1302.52 | 1308.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-05 10:15:00 | 1300.75 | 1298.88 | 1304.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-05 12:15:00 | 1301.35 | 1299.39 | 1303.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-05 12:15:00 | 1301.35 | 1299.39 | 1303.91 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 10:15:00 | 1324.20 | 1305.42 | 1305.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 09:15:00 | 1337.55 | 1321.00 | 1314.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-07 10:15:00 | 1318.60 | 1320.52 | 1314.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-07 11:15:00 | 1310.45 | 1318.50 | 1314.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-07 11:15:00 | 1310.45 | 1318.50 | 1314.27 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 09:15:00 | 1312.75 | 1326.63 | 1328.38 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2024-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 09:15:00 | 1318.90 | 1315.07 | 1314.98 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 13:15:00 | 1309.75 | 1314.55 | 1314.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 14:15:00 | 1298.00 | 1311.24 | 1313.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 09:15:00 | 1309.80 | 1309.46 | 1312.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 1309.80 | 1309.46 | 1312.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 1309.80 | 1309.46 | 1312.05 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 09:15:00 | 1348.00 | 1317.79 | 1314.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 10:15:00 | 1367.20 | 1337.87 | 1327.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 11:15:00 | 1366.55 | 1368.79 | 1352.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 09:15:00 | 1370.95 | 1381.22 | 1375.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 09:15:00 | 1370.95 | 1381.22 | 1375.95 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 1346.35 | 1369.14 | 1371.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 1333.00 | 1361.92 | 1367.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 1345.20 | 1341.47 | 1350.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 1345.20 | 1341.47 | 1350.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 1345.20 | 1341.47 | 1350.34 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 13:15:00 | 1366.35 | 1354.24 | 1353.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 14:15:00 | 1368.15 | 1357.02 | 1354.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-05 09:15:00 | 1381.90 | 1382.02 | 1373.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 12:15:00 | 1373.05 | 1379.69 | 1374.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 1373.05 | 1379.69 | 1374.59 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 10:15:00 | 1357.45 | 1369.91 | 1371.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 1343.50 | 1364.63 | 1368.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 15:15:00 | 1359.20 | 1359.00 | 1364.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-07 09:15:00 | 1360.05 | 1359.21 | 1363.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 1360.05 | 1359.21 | 1363.97 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-03-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 11:15:00 | 1381.40 | 1366.98 | 1366.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 09:15:00 | 1392.00 | 1376.85 | 1372.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 11:15:00 | 1370.10 | 1376.64 | 1373.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 11:15:00 | 1370.10 | 1376.64 | 1373.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 1370.10 | 1376.64 | 1373.03 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-11 15:15:00 | 1360.00 | 1369.64 | 1370.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 09:15:00 | 1340.50 | 1363.81 | 1367.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-13 09:15:00 | 1341.25 | 1338.15 | 1349.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 09:15:00 | 1341.25 | 1338.15 | 1349.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 1341.25 | 1338.15 | 1349.83 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 13:15:00 | 1349.30 | 1328.98 | 1326.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-19 11:15:00 | 1358.95 | 1344.48 | 1335.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 09:15:00 | 1501.10 | 1508.13 | 1483.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 14:15:00 | 1475.90 | 1495.28 | 1486.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 14:15:00 | 1475.90 | 1495.28 | 1486.14 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 09:15:00 | 1541.25 | 1545.54 | 1546.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 13:15:00 | 1520.95 | 1535.97 | 1540.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-12 11:15:00 | 1525.10 | 1519.06 | 1525.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 11:15:00 | 1525.10 | 1519.06 | 1525.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 11:15:00 | 1525.10 | 1519.06 | 1525.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-12 12:00:00 | 1525.10 | 1519.06 | 1525.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 12:15:00 | 1517.80 | 1518.80 | 1524.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 15:00:00 | 1503.05 | 1515.30 | 1521.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 13:15:00 | 1427.90 | 1444.92 | 1463.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-22 09:15:00 | 1404.55 | 1401.17 | 1421.55 | SL hit (close>ema200) qty=0.50 sl=1401.17 alert=retest2 |

### Cycle 72 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 1464.05 | 1428.67 | 1427.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-23 13:15:00 | 1477.95 | 1454.29 | 1441.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-24 12:15:00 | 1459.55 | 1468.16 | 1455.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-24 13:00:00 | 1459.55 | 1468.16 | 1455.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 1450.40 | 1461.88 | 1456.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:15:00 | 1455.80 | 1461.88 | 1456.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 1450.00 | 1459.51 | 1455.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 11:00:00 | 1450.00 | 1459.51 | 1455.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 1445.55 | 1456.71 | 1454.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 11:45:00 | 1445.70 | 1456.71 | 1454.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-25 13:15:00 | 1444.95 | 1452.27 | 1453.06 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 14:15:00 | 1477.90 | 1457.40 | 1455.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 10:15:00 | 1494.45 | 1469.46 | 1461.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 09:15:00 | 1475.85 | 1487.95 | 1476.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-29 09:15:00 | 1475.85 | 1487.95 | 1476.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 09:15:00 | 1475.85 | 1487.95 | 1476.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 10:00:00 | 1475.85 | 1487.95 | 1476.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 10:15:00 | 1476.90 | 1485.74 | 1476.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-29 11:00:00 | 1476.90 | 1485.74 | 1476.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 11:15:00 | 1474.40 | 1483.47 | 1476.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 14:15:00 | 1480.65 | 1479.67 | 1475.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 15:00:00 | 1480.40 | 1479.82 | 1476.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 1467.50 | 1482.47 | 1481.00 | SL hit (close<static) qty=1.00 sl=1470.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 15:15:00 | 1475.25 | 1487.82 | 1489.25 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 11:15:00 | 1504.25 | 1491.43 | 1490.46 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 11:15:00 | 1458.50 | 1487.22 | 1489.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 1452.05 | 1480.19 | 1486.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 1494.85 | 1478.03 | 1482.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 1494.85 | 1478.03 | 1482.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 1494.85 | 1478.03 | 1482.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 10:00:00 | 1494.85 | 1478.03 | 1482.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 1496.55 | 1481.74 | 1484.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 1496.55 | 1481.74 | 1484.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2024-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-08 11:15:00 | 1501.60 | 1485.71 | 1485.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-08 12:15:00 | 1509.00 | 1490.37 | 1487.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-08 13:15:00 | 1489.25 | 1490.14 | 1487.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-08 14:00:00 | 1489.25 | 1490.14 | 1487.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 14:15:00 | 1498.45 | 1491.80 | 1488.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-08 14:30:00 | 1492.00 | 1491.80 | 1488.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 10:15:00 | 1486.05 | 1493.08 | 1490.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-09 11:00:00 | 1486.05 | 1493.08 | 1490.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 11:15:00 | 1489.95 | 1492.45 | 1490.39 | EMA400 retest candle locked (from upside) |

### Cycle 79 — SELL (started 2024-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-09 13:15:00 | 1472.15 | 1486.14 | 1487.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 14:15:00 | 1453.25 | 1479.56 | 1484.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 15:15:00 | 1479.00 | 1468.21 | 1474.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 15:15:00 | 1479.00 | 1468.21 | 1474.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 15:15:00 | 1479.00 | 1468.21 | 1474.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 1462.80 | 1468.21 | 1474.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 13:15:00 | 1480.95 | 1470.64 | 1472.61 | SL hit (close>static) qty=1.00 sl=1479.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 1480.00 | 1474.37 | 1474.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 1493.25 | 1479.13 | 1476.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 10:15:00 | 1733.30 | 1736.10 | 1702.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 10:45:00 | 1730.95 | 1736.10 | 1702.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1735.05 | 1731.16 | 1713.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 1714.10 | 1731.16 | 1713.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 1720.10 | 1727.81 | 1716.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:45:00 | 1716.40 | 1727.81 | 1716.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 1725.40 | 1727.33 | 1717.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 14:30:00 | 1742.70 | 1735.67 | 1722.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 12:15:00 | 1796.95 | 1814.26 | 1814.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 12:15:00 | 1796.95 | 1814.26 | 1814.91 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 15:15:00 | 1848.70 | 1816.67 | 1815.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 1874.10 | 1828.45 | 1821.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1862.85 | 1868.21 | 1850.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1862.85 | 1868.21 | 1850.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1862.85 | 1868.21 | 1850.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1825.50 | 1868.21 | 1850.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 1766.95 | 1847.96 | 1842.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 1766.95 | 1847.96 | 1842.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 1695.55 | 1817.48 | 1829.42 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 15:15:00 | 1844.00 | 1814.68 | 1811.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 1901.40 | 1832.02 | 1819.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 14:15:00 | 1890.00 | 1899.91 | 1877.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 15:00:00 | 1890.00 | 1899.91 | 1877.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 1896.15 | 1898.38 | 1880.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:30:00 | 1893.60 | 1898.38 | 1880.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 1901.30 | 1900.03 | 1888.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:30:00 | 1899.65 | 1900.03 | 1888.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 1914.55 | 1922.16 | 1911.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:45:00 | 1913.00 | 1922.16 | 1911.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 1911.90 | 1920.11 | 1911.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:00:00 | 1911.90 | 1920.11 | 1911.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 1905.35 | 1917.15 | 1910.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 13:00:00 | 1905.35 | 1917.15 | 1910.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 1909.30 | 1915.58 | 1910.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 1935.55 | 1916.78 | 1911.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 11:15:00 | 1895.30 | 1918.70 | 1918.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 11:15:00 | 1895.30 | 1918.70 | 1918.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 1861.25 | 1900.55 | 1907.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 09:15:00 | 1882.80 | 1874.81 | 1887.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 09:15:00 | 1882.80 | 1874.81 | 1887.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 1882.80 | 1874.81 | 1887.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:45:00 | 1884.95 | 1874.81 | 1887.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 10:15:00 | 1876.60 | 1875.17 | 1886.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:15:00 | 1884.50 | 1875.17 | 1886.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 1892.75 | 1878.69 | 1887.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 11:45:00 | 1890.25 | 1878.69 | 1887.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 1879.70 | 1878.89 | 1886.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:30:00 | 1879.20 | 1878.89 | 1886.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 1888.70 | 1881.29 | 1886.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 15:00:00 | 1888.70 | 1881.29 | 1886.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 1890.05 | 1883.04 | 1886.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 09:15:00 | 1899.20 | 1883.04 | 1886.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 1902.75 | 1886.99 | 1888.08 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 1897.95 | 1889.18 | 1888.98 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 1878.05 | 1887.27 | 1888.35 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 1899.20 | 1889.93 | 1889.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 12:15:00 | 1908.00 | 1896.03 | 1892.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 09:15:00 | 1867.15 | 1894.63 | 1893.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 1867.15 | 1894.63 | 1893.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 1867.15 | 1894.63 | 1893.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:00:00 | 1867.15 | 1894.63 | 1893.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 10:15:00 | 1838.30 | 1883.36 | 1888.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 11:15:00 | 1831.00 | 1872.89 | 1883.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 1775.75 | 1774.67 | 1798.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-28 10:45:00 | 1776.70 | 1774.67 | 1798.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1753.95 | 1765.96 | 1783.45 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 1810.00 | 1783.43 | 1783.08 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 1776.35 | 1782.01 | 1782.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 1763.50 | 1778.31 | 1780.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 13:15:00 | 1787.20 | 1780.09 | 1781.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 13:15:00 | 1787.20 | 1780.09 | 1781.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 1787.20 | 1780.09 | 1781.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 14:00:00 | 1787.20 | 1780.09 | 1781.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2024-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 14:15:00 | 1798.75 | 1783.82 | 1782.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 1814.50 | 1791.91 | 1786.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 13:15:00 | 1796.00 | 1799.75 | 1792.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 14:00:00 | 1796.00 | 1799.75 | 1792.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 1798.20 | 1799.44 | 1793.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:30:00 | 1795.00 | 1799.44 | 1793.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 1798.00 | 1799.15 | 1793.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 1806.90 | 1799.15 | 1793.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 10:15:00 | 1786.25 | 1796.16 | 1793.38 | SL hit (close<static) qty=1.00 sl=1793.05 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 1771.35 | 1789.31 | 1791.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 09:15:00 | 1735.40 | 1774.56 | 1783.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 11:15:00 | 1721.85 | 1718.22 | 1732.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 12:00:00 | 1721.85 | 1718.22 | 1732.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 1721.55 | 1719.91 | 1727.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:45:00 | 1712.25 | 1719.20 | 1726.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:00:00 | 1707.40 | 1716.84 | 1725.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-15 13:15:00 | 1727.45 | 1713.36 | 1713.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-07-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 13:15:00 | 1727.45 | 1713.36 | 1713.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 09:15:00 | 1738.70 | 1719.21 | 1715.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 13:15:00 | 1722.00 | 1726.32 | 1721.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 13:15:00 | 1722.00 | 1726.32 | 1721.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1722.00 | 1726.32 | 1721.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:00:00 | 1722.00 | 1726.32 | 1721.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1729.85 | 1727.03 | 1721.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:45:00 | 1716.10 | 1727.03 | 1721.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1706.85 | 1722.75 | 1720.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1707.05 | 1722.75 | 1720.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 1691.80 | 1716.56 | 1718.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 1666.35 | 1705.07 | 1711.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1743.30 | 1690.69 | 1696.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 1743.30 | 1690.69 | 1696.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1743.30 | 1690.69 | 1696.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 1750.85 | 1690.69 | 1696.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 10:15:00 | 1746.60 | 1701.87 | 1701.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 09:15:00 | 1792.80 | 1742.86 | 1724.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 13:15:00 | 1682.35 | 1747.80 | 1734.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 13:15:00 | 1682.35 | 1747.80 | 1734.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 1682.35 | 1747.80 | 1734.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 1682.35 | 1747.80 | 1734.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 1683.75 | 1734.99 | 1729.94 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 15:15:00 | 1686.75 | 1725.34 | 1726.02 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 1757.30 | 1732.29 | 1729.09 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 13:15:00 | 1726.70 | 1732.37 | 1732.40 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 1783.10 | 1741.57 | 1736.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 1797.45 | 1752.75 | 1742.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 1795.45 | 1798.46 | 1780.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 15:00:00 | 1795.45 | 1798.46 | 1780.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 1843.75 | 1852.59 | 1834.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:45:00 | 1839.45 | 1852.59 | 1834.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 1822.95 | 1846.66 | 1833.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 1822.95 | 1846.66 | 1833.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 1830.00 | 1843.33 | 1833.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:30:00 | 1821.20 | 1843.33 | 1833.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 1840.60 | 1841.95 | 1834.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 1832.80 | 1841.95 | 1834.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1807.65 | 1835.10 | 1832.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 10:00:00 | 1807.65 | 1835.10 | 1832.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 11:15:00 | 1808.40 | 1826.76 | 1828.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 12:15:00 | 1803.90 | 1822.19 | 1826.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1749.85 | 1730.27 | 1761.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1749.85 | 1730.27 | 1761.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1749.85 | 1730.27 | 1761.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:30:00 | 1755.30 | 1730.27 | 1761.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1756.25 | 1735.06 | 1748.98 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 1770.05 | 1756.67 | 1755.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 1775.00 | 1762.44 | 1758.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 1763.10 | 1766.10 | 1761.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 12:15:00 | 1763.10 | 1766.10 | 1761.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 1763.10 | 1766.10 | 1761.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:00:00 | 1763.10 | 1766.10 | 1761.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 1756.50 | 1764.18 | 1761.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:00:00 | 1756.50 | 1764.18 | 1761.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 1742.80 | 1759.90 | 1759.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 15:00:00 | 1742.80 | 1759.90 | 1759.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 15:15:00 | 1749.70 | 1757.86 | 1758.51 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 1801.95 | 1766.68 | 1762.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 13:15:00 | 1810.00 | 1790.18 | 1780.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 1750.05 | 1788.78 | 1783.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 1750.05 | 1788.78 | 1783.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1750.05 | 1788.78 | 1783.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 1750.05 | 1788.78 | 1783.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 1760.30 | 1783.08 | 1781.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:30:00 | 1742.80 | 1783.08 | 1781.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 1751.80 | 1776.83 | 1778.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 14:15:00 | 1742.95 | 1762.82 | 1770.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1739.95 | 1729.70 | 1744.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1739.95 | 1729.70 | 1744.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1739.95 | 1729.70 | 1744.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:30:00 | 1748.90 | 1729.70 | 1744.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 1743.20 | 1732.40 | 1744.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:30:00 | 1744.20 | 1732.40 | 1744.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 1744.30 | 1734.78 | 1744.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:30:00 | 1742.95 | 1734.78 | 1744.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 1757.00 | 1739.22 | 1745.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 12:45:00 | 1758.50 | 1739.22 | 1745.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 1767.10 | 1744.80 | 1747.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 14:00:00 | 1767.10 | 1744.80 | 1747.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 1766.85 | 1749.21 | 1749.14 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 10:15:00 | 1744.75 | 1753.86 | 1755.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 11:15:00 | 1738.30 | 1750.75 | 1753.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 1719.80 | 1713.14 | 1724.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 09:45:00 | 1719.50 | 1713.14 | 1724.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 1731.30 | 1716.77 | 1724.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 11:00:00 | 1731.30 | 1716.77 | 1724.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 11:15:00 | 1729.45 | 1719.31 | 1725.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 12:00:00 | 1729.45 | 1719.31 | 1725.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 1732.45 | 1721.94 | 1725.83 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 15:15:00 | 1737.75 | 1729.82 | 1728.88 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 1723.25 | 1728.13 | 1728.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 11:15:00 | 1721.10 | 1724.74 | 1726.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 12:15:00 | 1713.00 | 1711.22 | 1717.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 12:30:00 | 1715.50 | 1711.22 | 1717.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 1721.75 | 1712.64 | 1716.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 15:00:00 | 1721.75 | 1712.64 | 1716.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1728.00 | 1715.71 | 1717.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 1777.85 | 1715.71 | 1717.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 1768.25 | 1726.22 | 1722.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 11:15:00 | 1779.50 | 1744.04 | 1731.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 11:15:00 | 1758.60 | 1764.45 | 1751.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 12:00:00 | 1758.60 | 1764.45 | 1751.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1758.80 | 1763.32 | 1751.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 1755.10 | 1763.32 | 1751.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 1758.65 | 1762.31 | 1753.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:45:00 | 1753.65 | 1762.31 | 1753.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 1771.80 | 1763.89 | 1755.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 15:00:00 | 1788.00 | 1775.84 | 1769.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 10:00:00 | 1791.25 | 1780.07 | 1772.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 10:45:00 | 1793.10 | 1781.79 | 1773.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 11:45:00 | 1789.85 | 1783.83 | 1775.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1749.05 | 1779.46 | 1777.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 1749.05 | 1779.46 | 1777.11 | SL hit (close<static) qty=1.00 sl=1752.20 alert=retest2 |

### Cycle 111 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 1726.00 | 1768.77 | 1772.46 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-11 09:15:00 | 1769.70 | 1758.02 | 1756.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 1777.65 | 1761.94 | 1758.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 1760.50 | 1765.74 | 1761.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 13:15:00 | 1760.50 | 1765.74 | 1761.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 1760.50 | 1765.74 | 1761.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 14:00:00 | 1760.50 | 1765.74 | 1761.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1749.80 | 1762.55 | 1760.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 1749.80 | 1762.55 | 1760.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1754.80 | 1761.00 | 1759.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 1761.40 | 1761.00 | 1759.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 10:15:00 | 1752.95 | 1758.65 | 1758.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 1752.95 | 1758.65 | 1758.90 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 1767.65 | 1759.16 | 1758.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 1787.90 | 1765.84 | 1762.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 1801.70 | 1802.74 | 1787.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 11:45:00 | 1795.95 | 1802.74 | 1787.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 1789.80 | 1805.29 | 1801.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:00:00 | 1789.80 | 1805.29 | 1801.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 1795.00 | 1803.23 | 1800.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 1795.00 | 1803.23 | 1800.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 1802.00 | 1802.08 | 1800.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 1829.10 | 1802.08 | 1800.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 11:15:00 | 1795.85 | 1801.67 | 1800.85 | SL hit (close<static) qty=1.00 sl=1796.10 alert=retest2 |

### Cycle 115 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 1896.65 | 1914.48 | 1915.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 1875.60 | 1898.25 | 1906.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 12:15:00 | 1882.90 | 1882.71 | 1893.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 13:00:00 | 1882.90 | 1882.71 | 1893.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 1885.10 | 1880.91 | 1891.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 1885.10 | 1880.91 | 1891.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 1892.00 | 1883.13 | 1891.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 1869.85 | 1883.13 | 1891.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 1776.36 | 1806.41 | 1833.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 13:15:00 | 1754.00 | 1752.47 | 1780.01 | SL hit (close>ema200) qty=0.50 sl=1752.47 alert=retest2 |

### Cycle 116 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 1831.50 | 1793.83 | 1789.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1873.60 | 1821.70 | 1804.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 09:15:00 | 2004.75 | 2006.11 | 1972.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 10:00:00 | 2004.75 | 2006.11 | 1972.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 1959.60 | 2006.51 | 1990.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 1959.60 | 2006.51 | 1990.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 1930.85 | 1991.38 | 1985.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 11:00:00 | 1930.85 | 1991.38 | 1985.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 1932.90 | 1979.68 | 1980.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 1901.80 | 1954.53 | 1968.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 14:15:00 | 1928.70 | 1915.59 | 1934.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 14:15:00 | 1928.70 | 1915.59 | 1934.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 1928.70 | 1915.59 | 1934.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 1941.25 | 1915.59 | 1934.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 1930.95 | 1918.66 | 1934.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 1989.10 | 1918.66 | 1934.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 1973.00 | 1929.53 | 1937.99 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 11:15:00 | 1982.20 | 1948.23 | 1945.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-21 13:15:00 | 2008.60 | 1967.30 | 1955.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 10:15:00 | 1968.50 | 1974.95 | 1963.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-22 10:15:00 | 1968.50 | 1974.95 | 1963.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 1968.50 | 1974.95 | 1963.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 1968.95 | 1974.95 | 1963.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 11:15:00 | 1958.90 | 1971.74 | 1963.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 10:15:00 | 1978.15 | 1962.59 | 1960.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 12:45:00 | 1970.00 | 1965.59 | 1962.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 13:45:00 | 1970.00 | 1965.67 | 1963.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 15:15:00 | 1975.00 | 1963.49 | 1962.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 1975.00 | 1965.79 | 1963.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:15:00 | 1997.45 | 1965.79 | 1963.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 1984.00 | 1969.44 | 1965.42 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 1936.20 | 1970.50 | 1969.52 | SL hit (close<static) qty=1.00 sl=1951.25 alert=retest2 |

### Cycle 119 — SELL (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 10:15:00 | 1933.30 | 1963.06 | 1966.23 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 1990.00 | 1962.26 | 1961.43 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 11:15:00 | 1951.00 | 1963.12 | 1963.97 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 1975.40 | 1966.70 | 1965.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 1985.00 | 1970.36 | 1967.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 12:15:00 | 1963.35 | 1973.47 | 1970.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-30 12:15:00 | 1963.35 | 1973.47 | 1970.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 12:15:00 | 1963.35 | 1973.47 | 1970.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 13:00:00 | 1963.35 | 1973.47 | 1970.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2024-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 13:15:00 | 1938.90 | 1966.55 | 1967.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 14:15:00 | 1926.25 | 1958.49 | 1963.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 10:15:00 | 1958.60 | 1949.09 | 1957.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 10:15:00 | 1958.60 | 1949.09 | 1957.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 1958.60 | 1949.09 | 1957.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 1958.60 | 1949.09 | 1957.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1967.35 | 1952.74 | 1958.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:00:00 | 1967.35 | 1952.74 | 1958.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1961.45 | 1954.48 | 1958.68 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 15:15:00 | 1980.00 | 1960.60 | 1960.46 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 09:15:00 | 1931.80 | 1958.72 | 1959.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 1923.60 | 1951.69 | 1956.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 1951.55 | 1942.23 | 1948.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 1951.55 | 1942.23 | 1948.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1951.55 | 1942.23 | 1948.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:00:00 | 1951.55 | 1942.23 | 1948.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 1960.55 | 1945.89 | 1949.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 10:45:00 | 1959.00 | 1945.89 | 1949.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 1952.00 | 1947.11 | 1949.50 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 1977.00 | 1954.62 | 1952.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 14:15:00 | 1999.05 | 1963.51 | 1956.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 2030.90 | 2040.40 | 2017.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 14:45:00 | 2030.85 | 2040.40 | 2017.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 11:15:00 | 2026.35 | 2036.92 | 2023.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 12:00:00 | 2026.35 | 2036.92 | 2023.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 12:15:00 | 2031.05 | 2035.75 | 2024.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 09:45:00 | 2035.85 | 2026.65 | 2022.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:15:00 | 2034.90 | 2026.65 | 2022.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-11 11:15:00 | 2015.95 | 2025.25 | 2022.80 | SL hit (close<static) qty=1.00 sl=2018.80 alert=retest2 |

### Cycle 127 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 2011.95 | 2020.17 | 2020.90 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 11:15:00 | 2032.75 | 2022.46 | 2021.56 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 13:15:00 | 2009.30 | 2019.09 | 2020.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 1989.30 | 2013.13 | 2017.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1978.00 | 1955.06 | 1976.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1978.00 | 1955.06 | 1976.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1978.00 | 1955.06 | 1976.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1978.00 | 1955.06 | 1976.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1971.25 | 1958.30 | 1975.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 1990.25 | 1958.30 | 1975.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 2001.05 | 1966.85 | 1977.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 12:00:00 | 2001.05 | 1966.85 | 1977.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 2000.00 | 1973.48 | 1979.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:45:00 | 1981.00 | 1974.81 | 1980.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 2000.50 | 1947.07 | 1943.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 2000.50 | 1947.07 | 1943.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 11:15:00 | 2047.00 | 2015.95 | 2001.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 10:15:00 | 2040.05 | 2043.58 | 2023.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 11:00:00 | 2040.05 | 2043.58 | 2023.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 2137.70 | 2140.84 | 2133.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 2132.55 | 2140.84 | 2133.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 2127.30 | 2138.13 | 2133.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 2127.30 | 2138.13 | 2133.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 2130.45 | 2136.60 | 2133.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 11:00:00 | 2130.45 | 2136.60 | 2133.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 2132.25 | 2135.73 | 2132.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:15:00 | 2127.65 | 2135.73 | 2132.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 2129.45 | 2134.47 | 2132.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:30:00 | 2128.65 | 2134.47 | 2132.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 2128.10 | 2133.20 | 2132.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 13:45:00 | 2129.80 | 2133.20 | 2132.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 09:15:00 | 2122.00 | 2130.06 | 2130.97 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 2135.95 | 2131.87 | 2131.67 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 2128.10 | 2131.81 | 2131.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 2122.60 | 2129.97 | 2131.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 11:15:00 | 2130.70 | 2130.12 | 2131.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 11:15:00 | 2130.70 | 2130.12 | 2131.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 2130.70 | 2130.12 | 2131.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:45:00 | 2137.05 | 2130.12 | 2131.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 2134.75 | 2131.04 | 2131.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 14:45:00 | 2127.80 | 2130.64 | 2131.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 09:15:00 | 2178.75 | 2127.14 | 2125.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 2178.75 | 2127.14 | 2125.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 10:15:00 | 2193.80 | 2140.47 | 2131.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 11:15:00 | 2298.35 | 2300.04 | 2260.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-18 11:45:00 | 2298.45 | 2300.04 | 2260.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 2287.10 | 2304.93 | 2278.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 2325.00 | 2295.34 | 2284.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 10:30:00 | 2317.45 | 2301.69 | 2288.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 12:15:00 | 2272.75 | 2296.19 | 2288.39 | SL hit (close<static) qty=1.00 sl=2273.20 alert=retest2 |

### Cycle 135 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 2242.65 | 2279.37 | 2281.72 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 10:15:00 | 2314.95 | 2276.58 | 2275.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 2322.70 | 2296.18 | 2288.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 10:15:00 | 2292.50 | 2300.70 | 2293.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 10:15:00 | 2292.50 | 2300.70 | 2293.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 2292.50 | 2300.70 | 2293.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 2292.50 | 2300.70 | 2293.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 2290.55 | 2298.67 | 2292.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 2289.65 | 2298.67 | 2292.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 12:15:00 | 2310.95 | 2301.12 | 2294.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 13:15:00 | 2314.25 | 2301.12 | 2294.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-27 15:00:00 | 2320.05 | 2307.49 | 2298.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 11:15:00 | 2313.30 | 2309.48 | 2302.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 2277.90 | 2299.52 | 2299.06 | SL hit (close<static) qty=1.00 sl=2288.00 alert=retest2 |

### Cycle 137 — SELL (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 14:15:00 | 2257.65 | 2291.15 | 2295.30 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 2313.90 | 2296.74 | 2294.90 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 11:15:00 | 2281.00 | 2292.29 | 2293.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 12:15:00 | 2271.30 | 2288.10 | 2291.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 15:15:00 | 2283.00 | 2281.09 | 2286.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-02 09:15:00 | 2250.05 | 2281.09 | 2286.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 2229.80 | 2270.83 | 2281.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-06 11:00:00 | 2216.60 | 2252.77 | 2262.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-07 14:15:00 | 2268.65 | 2248.29 | 2247.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 2268.65 | 2248.29 | 2247.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 14:15:00 | 2288.15 | 2263.54 | 2256.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 09:15:00 | 2237.30 | 2261.57 | 2256.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 2237.30 | 2261.57 | 2256.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 2237.30 | 2261.57 | 2256.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:30:00 | 2229.70 | 2261.57 | 2256.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 2255.70 | 2260.39 | 2256.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 11:30:00 | 2263.45 | 2262.03 | 2257.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 15:15:00 | 2240.00 | 2253.75 | 2254.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 2240.00 | 2253.75 | 2254.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 2199.80 | 2242.96 | 2249.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 10:15:00 | 2003.95 | 1996.06 | 2043.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 10:45:00 | 2013.05 | 1996.06 | 2043.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 2042.15 | 2011.58 | 2031.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:00:00 | 2042.15 | 2011.58 | 2031.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 2000.05 | 2009.27 | 2028.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 11:15:00 | 1992.00 | 2009.27 | 2028.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 1896.30 | 1994.91 | 1995.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-21 09:15:00 | 1892.40 | 1970.06 | 1984.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-22 09:15:00 | 1792.80 | 1862.37 | 1913.38 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 142 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 1738.20 | 1724.41 | 1723.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 1746.90 | 1728.91 | 1725.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 1803.15 | 1803.79 | 1782.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 1803.15 | 1803.79 | 1782.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1803.15 | 1803.79 | 1782.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1784.60 | 1803.79 | 1782.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 1813.85 | 1822.15 | 1801.15 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 1800.20 | 1815.01 | 1815.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 14:15:00 | 1794.40 | 1810.89 | 1813.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 1811.50 | 1809.27 | 1812.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 1811.50 | 1809.27 | 1812.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1811.50 | 1809.27 | 1812.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 1788.80 | 1809.27 | 1812.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1823.05 | 1812.02 | 1813.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 1823.05 | 1812.02 | 1813.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1806.65 | 1810.95 | 1812.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:30:00 | 1840.45 | 1810.95 | 1812.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 1813.95 | 1811.55 | 1812.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:30:00 | 1814.70 | 1811.55 | 1812.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 1812.25 | 1811.69 | 1812.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:30:00 | 1833.65 | 1811.69 | 1812.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 14:15:00 | 1826.05 | 1814.56 | 1814.00 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1796.25 | 1810.71 | 1812.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 13:15:00 | 1791.00 | 1802.39 | 1807.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 1810.20 | 1803.95 | 1807.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-10 14:15:00 | 1810.20 | 1803.95 | 1807.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 1810.20 | 1803.95 | 1807.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 15:00:00 | 1810.20 | 1803.95 | 1807.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 1800.00 | 1803.16 | 1807.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:15:00 | 1747.00 | 1803.16 | 1807.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 1726.65 | 1787.86 | 1799.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:30:00 | 1724.05 | 1772.69 | 1791.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 11:00:00 | 1712.00 | 1772.69 | 1791.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1637.85 | 1707.88 | 1747.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 1626.40 | 1707.88 | 1747.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-14 13:15:00 | 1551.64 | 1597.25 | 1634.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 146 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 1628.45 | 1594.57 | 1594.08 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 12:15:00 | 1591.45 | 1596.98 | 1597.61 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 1618.35 | 1601.26 | 1599.50 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 14:15:00 | 1585.05 | 1601.81 | 1602.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 15:15:00 | 1578.00 | 1597.05 | 1600.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 10:15:00 | 1480.25 | 1476.77 | 1501.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 11:00:00 | 1480.25 | 1476.77 | 1501.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 1500.35 | 1482.22 | 1499.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 1500.35 | 1482.22 | 1499.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 1505.35 | 1486.85 | 1499.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:00:00 | 1505.35 | 1486.85 | 1499.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 1508.10 | 1491.10 | 1500.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:45:00 | 1511.65 | 1491.10 | 1500.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1508.65 | 1501.08 | 1503.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 11:15:00 | 1507.55 | 1501.08 | 1503.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:00:00 | 1507.25 | 1502.31 | 1503.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:30:00 | 1508.10 | 1503.18 | 1504.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-04 13:15:00 | 1516.90 | 1505.92 | 1505.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 13:15:00 | 1516.90 | 1505.92 | 1505.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 15:15:00 | 1522.00 | 1510.58 | 1507.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 11:15:00 | 1555.45 | 1557.32 | 1541.28 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-07 09:15:00 | 1565.35 | 1560.06 | 1547.73 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 1537.55 | 1555.05 | 1548.47 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-07 11:15:00 | 1537.55 | 1555.05 | 1548.47 | SL hit (close<ema400) qty=1.00 sl=1548.47 alert=retest1 |

### Cycle 151 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 1529.50 | 1543.93 | 1545.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 1523.05 | 1539.76 | 1543.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 09:15:00 | 1561.45 | 1540.93 | 1542.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 1561.45 | 1540.93 | 1542.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1561.45 | 1540.93 | 1542.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:00:00 | 1561.45 | 1540.93 | 1542.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 10:15:00 | 1569.90 | 1546.73 | 1545.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 13:15:00 | 1577.75 | 1558.65 | 1551.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 1557.40 | 1566.78 | 1559.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 11:15:00 | 1557.40 | 1566.78 | 1559.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 1557.40 | 1566.78 | 1559.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 1557.40 | 1566.78 | 1559.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 1559.10 | 1565.24 | 1559.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 13:15:00 | 1570.40 | 1565.24 | 1559.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 13:15:00 | 1541.40 | 1559.46 | 1560.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 13:15:00 | 1541.40 | 1559.46 | 1560.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 1533.55 | 1554.28 | 1558.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 1545.30 | 1532.85 | 1541.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 09:15:00 | 1545.30 | 1532.85 | 1541.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1545.30 | 1532.85 | 1541.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:00:00 | 1545.30 | 1532.85 | 1541.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 1535.00 | 1533.28 | 1540.85 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 13:15:00 | 1563.85 | 1545.63 | 1545.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 1566.10 | 1549.73 | 1547.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1663.40 | 1671.45 | 1655.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 1663.40 | 1671.45 | 1655.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1659.50 | 1669.06 | 1655.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 1658.70 | 1669.06 | 1655.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1643.10 | 1663.87 | 1654.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:00:00 | 1643.10 | 1663.87 | 1654.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1631.55 | 1657.41 | 1652.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 1631.55 | 1657.41 | 1652.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 1624.45 | 1645.09 | 1647.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 1617.95 | 1628.37 | 1636.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 13:15:00 | 1628.35 | 1623.35 | 1630.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 13:15:00 | 1628.35 | 1623.35 | 1630.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 1628.35 | 1623.35 | 1630.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:45:00 | 1633.95 | 1623.35 | 1630.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1646.05 | 1627.89 | 1631.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1646.05 | 1627.89 | 1631.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1626.50 | 1627.61 | 1631.34 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 1648.65 | 1636.47 | 1634.95 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 09:15:00 | 1575.75 | 1623.39 | 1629.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 1567.80 | 1612.28 | 1623.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 1595.20 | 1586.03 | 1601.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 1595.20 | 1586.03 | 1601.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 1595.20 | 1586.03 | 1601.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 1595.20 | 1586.03 | 1601.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 1601.90 | 1590.40 | 1600.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 13:00:00 | 1601.90 | 1590.40 | 1600.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 13:15:00 | 1611.10 | 1594.54 | 1601.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 14:00:00 | 1611.10 | 1594.54 | 1601.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1619.05 | 1599.44 | 1603.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 1619.05 | 1599.44 | 1603.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 1602.00 | 1603.55 | 1604.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:30:00 | 1612.40 | 1603.55 | 1604.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 1606.35 | 1604.11 | 1604.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:00:00 | 1606.35 | 1604.11 | 1604.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 1594.60 | 1602.21 | 1603.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 14:30:00 | 1592.00 | 1599.41 | 1602.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1512.40 | 1547.36 | 1570.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 09:15:00 | 1503.05 | 1498.47 | 1528.13 | SL hit (close>ema200) qty=0.50 sl=1498.47 alert=retest2 |

### Cycle 158 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 1563.25 | 1517.96 | 1517.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1593.80 | 1550.94 | 1536.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 1622.70 | 1628.72 | 1604.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 1622.70 | 1628.72 | 1604.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 1616.70 | 1632.05 | 1619.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:30:00 | 1630.00 | 1632.22 | 1620.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1619.50 | 1670.02 | 1675.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1619.50 | 1670.02 | 1675.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-29 12:15:00 | 1586.90 | 1620.99 | 1634.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 15:15:00 | 1624.70 | 1620.42 | 1630.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 15:15:00 | 1624.70 | 1620.42 | 1630.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 1624.70 | 1620.42 | 1630.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 1623.30 | 1620.42 | 1630.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1625.10 | 1621.36 | 1629.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:30:00 | 1619.40 | 1621.36 | 1629.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 10:15:00 | 1660.60 | 1629.21 | 1632.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 11:00:00 | 1660.60 | 1629.21 | 1632.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 11:15:00 | 1651.50 | 1633.67 | 1634.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 12:30:00 | 1647.90 | 1634.49 | 1634.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 14:15:00 | 1641.70 | 1635.87 | 1635.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 14:15:00 | 1641.70 | 1635.87 | 1635.28 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 09:15:00 | 1629.50 | 1634.87 | 1634.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 12:15:00 | 1597.60 | 1623.93 | 1629.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 10:15:00 | 1621.90 | 1616.98 | 1623.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 10:15:00 | 1621.90 | 1616.98 | 1623.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1621.90 | 1616.98 | 1623.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 1620.90 | 1616.98 | 1623.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1623.00 | 1618.18 | 1623.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:30:00 | 1625.10 | 1618.18 | 1623.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1625.90 | 1619.73 | 1623.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 13:00:00 | 1625.90 | 1619.73 | 1623.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 13:15:00 | 1624.60 | 1620.70 | 1623.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:00:00 | 1624.60 | 1620.70 | 1623.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1626.60 | 1621.88 | 1623.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 14:30:00 | 1625.90 | 1621.88 | 1623.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 1627.00 | 1622.90 | 1624.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 1612.70 | 1622.90 | 1624.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1603.90 | 1619.10 | 1622.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:30:00 | 1593.90 | 1612.54 | 1618.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:15:00 | 1597.10 | 1612.54 | 1618.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1514.20 | 1561.28 | 1575.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 1517.24 | 1561.28 | 1575.10 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 1540.00 | 1538.51 | 1555.17 | SL hit (close>ema200) qty=0.50 sl=1538.51 alert=retest2 |

### Cycle 162 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1597.70 | 1568.05 | 1566.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 1615.10 | 1586.90 | 1576.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 1580.00 | 1591.11 | 1581.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 10:15:00 | 1580.00 | 1591.11 | 1581.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 1580.00 | 1591.11 | 1581.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:00:00 | 1580.00 | 1591.11 | 1581.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 1589.00 | 1590.69 | 1581.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1597.30 | 1587.51 | 1582.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-20 10:15:00 | 1757.03 | 1723.60 | 1692.02 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 12:15:00 | 1721.60 | 1741.45 | 1741.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 1704.40 | 1728.09 | 1734.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 11:15:00 | 1734.30 | 1726.97 | 1732.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 11:15:00 | 1734.30 | 1726.97 | 1732.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 1734.30 | 1726.97 | 1732.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 1734.30 | 1726.97 | 1732.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1747.40 | 1731.05 | 1734.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 1747.40 | 1731.05 | 1734.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1746.30 | 1734.10 | 1735.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:00:00 | 1746.30 | 1734.10 | 1735.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1735.70 | 1735.49 | 1735.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 1731.70 | 1735.49 | 1735.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:30:00 | 1730.80 | 1734.31 | 1735.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:15:00 | 1734.00 | 1734.31 | 1735.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 1757.80 | 1739.02 | 1737.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 14:15:00 | 1757.80 | 1739.02 | 1737.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 11:15:00 | 1772.70 | 1749.99 | 1742.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 1776.40 | 1785.60 | 1772.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 1776.40 | 1785.60 | 1772.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 1776.40 | 1785.60 | 1772.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 1769.20 | 1785.60 | 1772.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 1773.40 | 1783.16 | 1772.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 1773.40 | 1783.16 | 1772.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 1786.70 | 1783.87 | 1774.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 1789.00 | 1780.95 | 1775.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 10:30:00 | 1795.50 | 1783.99 | 1778.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1883.90 | 1894.97 | 1896.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 1883.90 | 1894.97 | 1896.29 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 15:15:00 | 1905.10 | 1895.71 | 1895.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 1915.90 | 1903.42 | 1899.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 1922.60 | 1925.39 | 1914.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 12:45:00 | 1919.20 | 1925.39 | 1914.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 1913.60 | 1923.82 | 1916.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 1913.60 | 1923.82 | 1916.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 1908.00 | 1920.65 | 1915.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 09:15:00 | 1924.50 | 1920.65 | 1915.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 10:15:00 | 1898.90 | 1915.99 | 1914.09 | SL hit (close<static) qty=1.00 sl=1905.10 alert=retest2 |

### Cycle 167 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 1898.00 | 1912.39 | 1912.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 13:15:00 | 1886.60 | 1898.48 | 1903.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 09:15:00 | 1899.00 | 1893.71 | 1899.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 09:15:00 | 1899.00 | 1893.71 | 1899.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1899.00 | 1893.71 | 1899.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:45:00 | 1891.90 | 1893.71 | 1899.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1911.50 | 1897.26 | 1900.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:00:00 | 1911.50 | 1897.26 | 1900.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1908.80 | 1899.57 | 1901.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 1898.20 | 1899.57 | 1901.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:45:00 | 1898.00 | 1899.46 | 1901.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 1910.30 | 1902.26 | 1902.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 1910.30 | 1902.26 | 1902.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 09:15:00 | 1915.50 | 1905.66 | 1903.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 11:15:00 | 1972.30 | 1977.21 | 1956.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 12:00:00 | 1972.30 | 1977.21 | 1956.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1964.80 | 1979.80 | 1966.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 1966.80 | 1979.80 | 1966.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 1959.30 | 1975.70 | 1965.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:00:00 | 1959.30 | 1975.70 | 1965.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 1959.00 | 1972.36 | 1965.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 11:30:00 | 1953.60 | 1972.36 | 1965.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 13:15:00 | 1978.20 | 1972.53 | 1966.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 13:30:00 | 1963.00 | 1972.53 | 1966.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 1974.80 | 1974.27 | 1968.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 1952.20 | 1974.27 | 1968.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1948.60 | 1969.14 | 1966.51 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 10:15:00 | 1943.00 | 1963.91 | 1964.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 11:15:00 | 1930.80 | 1957.29 | 1961.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 09:15:00 | 1918.90 | 1917.63 | 1930.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 1918.90 | 1917.63 | 1930.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 1918.90 | 1917.63 | 1930.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 1927.80 | 1917.63 | 1930.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1933.50 | 1920.81 | 1930.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 1932.10 | 1920.81 | 1930.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1927.00 | 1922.05 | 1930.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-01 13:00:00 | 1916.10 | 1920.86 | 1929.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-08 09:15:00 | 1820.29 | 1843.75 | 1861.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-08 12:15:00 | 1840.60 | 1838.87 | 1854.26 | SL hit (close>ema200) qty=0.50 sl=1838.87 alert=retest2 |

### Cycle 170 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 1842.00 | 1823.58 | 1822.77 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 1812.60 | 1830.25 | 1831.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 1760.90 | 1814.06 | 1822.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 1784.00 | 1778.89 | 1797.63 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:45:00 | 1760.20 | 1773.91 | 1793.67 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 09:15:00 | 1672.19 | 1705.38 | 1733.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 1666.80 | 1655.88 | 1689.14 | SL hit (close>ema200) qty=0.50 sl=1655.88 alert=retest1 |

### Cycle 172 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 1611.00 | 1597.19 | 1595.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 1612.20 | 1602.01 | 1598.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 15:15:00 | 1610.30 | 1611.02 | 1604.57 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:15:00 | 1634.50 | 1611.02 | 1604.57 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1627.60 | 1626.47 | 1618.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:30:00 | 1620.00 | 1626.47 | 1618.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 1619.70 | 1623.73 | 1618.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 1619.70 | 1623.73 | 1618.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1618.10 | 1622.60 | 1618.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 1618.10 | 1622.60 | 1618.71 | SL hit (close<ema400) qty=1.00 sl=1618.71 alert=retest1 |

### Cycle 173 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1650.00 | 1660.30 | 1660.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 1640.40 | 1654.78 | 1657.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1634.70 | 1626.68 | 1636.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 1634.70 | 1626.68 | 1636.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 1634.70 | 1626.68 | 1636.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 1634.70 | 1626.68 | 1636.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 1631.80 | 1627.70 | 1635.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:15:00 | 1619.90 | 1627.70 | 1635.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:00:00 | 1627.10 | 1620.48 | 1627.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 11:45:00 | 1623.20 | 1622.24 | 1627.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1658.60 | 1635.40 | 1632.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 174 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1658.60 | 1635.40 | 1632.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 1669.80 | 1642.28 | 1635.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 1651.00 | 1651.59 | 1645.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 10:30:00 | 1647.00 | 1651.59 | 1645.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1657.30 | 1662.84 | 1656.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1657.30 | 1662.84 | 1656.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1657.00 | 1661.67 | 1656.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1660.50 | 1661.67 | 1656.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1659.40 | 1661.22 | 1656.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:15:00 | 1649.10 | 1661.22 | 1656.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1636.20 | 1656.22 | 1655.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:45:00 | 1639.20 | 1656.22 | 1655.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1618.90 | 1648.75 | 1651.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 13:15:00 | 1612.90 | 1620.68 | 1629.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1621.40 | 1619.60 | 1626.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 1621.40 | 1619.60 | 1626.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1621.40 | 1619.60 | 1626.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:45:00 | 1619.30 | 1620.87 | 1625.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 1616.00 | 1619.89 | 1625.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:00:00 | 1615.50 | 1617.99 | 1622.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:00:00 | 1619.90 | 1607.96 | 1609.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 12:15:00 | 1633.00 | 1612.97 | 1611.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 1633.00 | 1612.97 | 1611.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 14:15:00 | 1637.70 | 1621.54 | 1616.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-16 11:15:00 | 1626.80 | 1628.05 | 1621.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-16 12:00:00 | 1626.80 | 1628.05 | 1621.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1663.10 | 1660.47 | 1651.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 1675.00 | 1660.46 | 1654.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 1673.10 | 1662.87 | 1655.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 1673.20 | 1667.35 | 1659.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 14:45:00 | 1672.80 | 1679.59 | 1669.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 1668.60 | 1676.31 | 1669.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1638.40 | 1662.33 | 1665.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 1638.40 | 1662.33 | 1665.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 10:15:00 | 1629.40 | 1655.75 | 1661.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1596.50 | 1593.53 | 1606.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 09:30:00 | 1594.20 | 1593.53 | 1606.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 1608.30 | 1596.22 | 1604.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:00:00 | 1608.30 | 1596.22 | 1604.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 1604.60 | 1597.90 | 1604.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:30:00 | 1605.10 | 1597.90 | 1604.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1600.30 | 1598.38 | 1604.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 1607.40 | 1598.38 | 1604.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1600.00 | 1598.70 | 1603.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1587.50 | 1598.70 | 1603.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1585.10 | 1595.98 | 1601.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 1580.90 | 1595.38 | 1601.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:30:00 | 1580.40 | 1589.55 | 1596.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 1599.10 | 1594.65 | 1594.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 1599.10 | 1594.65 | 1594.16 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 13:15:00 | 1589.40 | 1593.15 | 1593.53 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 1604.10 | 1595.34 | 1594.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 1606.00 | 1597.47 | 1595.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1618.10 | 1627.39 | 1618.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 1618.10 | 1627.39 | 1618.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1618.10 | 1627.39 | 1618.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 1617.30 | 1627.39 | 1618.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1611.50 | 1624.21 | 1618.01 | EMA400 retest candle locked (from upside) |

### Cycle 181 — SELL (started 2025-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 13:15:00 | 1590.00 | 1610.30 | 1612.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 1575.20 | 1603.28 | 1609.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 1596.90 | 1591.88 | 1600.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 12:15:00 | 1596.90 | 1591.88 | 1600.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1596.90 | 1591.88 | 1600.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 1596.90 | 1591.88 | 1600.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1591.30 | 1591.77 | 1599.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:15:00 | 1586.80 | 1591.77 | 1599.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:45:00 | 1586.70 | 1591.47 | 1598.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 1587.00 | 1591.47 | 1598.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1606.40 | 1593.74 | 1598.20 | SL hit (close>static) qty=1.00 sl=1601.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1613.10 | 1603.16 | 1601.85 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1588.10 | 1598.77 | 1600.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 11:15:00 | 1565.20 | 1582.29 | 1589.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 1599.00 | 1579.53 | 1584.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 1599.00 | 1579.53 | 1584.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1599.00 | 1579.53 | 1584.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 1599.00 | 1579.53 | 1584.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1594.30 | 1582.48 | 1585.56 | EMA400 retest candle locked (from downside) |

### Cycle 184 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 1603.70 | 1590.07 | 1588.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 1609.20 | 1596.97 | 1592.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 1691.90 | 1695.76 | 1672.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:45:00 | 1687.60 | 1695.76 | 1672.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 1690.80 | 1694.55 | 1684.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 1730.90 | 1695.71 | 1690.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 14:15:00 | 1780.40 | 1783.27 | 1783.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 1780.40 | 1783.27 | 1783.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 15:15:00 | 1776.00 | 1781.82 | 1782.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1784.30 | 1780.57 | 1781.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 11:15:00 | 1784.30 | 1780.57 | 1781.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 1784.30 | 1780.57 | 1781.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 1784.30 | 1780.57 | 1781.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1787.80 | 1782.02 | 1782.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 1789.50 | 1782.02 | 1782.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 186 — BUY (started 2025-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 13:15:00 | 1791.50 | 1783.91 | 1783.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 1797.90 | 1788.36 | 1785.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 12:15:00 | 1788.70 | 1790.44 | 1787.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 12:15:00 | 1788.70 | 1790.44 | 1787.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 1788.70 | 1790.44 | 1787.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:45:00 | 1789.70 | 1790.44 | 1787.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 1788.20 | 1789.99 | 1787.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:45:00 | 1786.00 | 1789.99 | 1787.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1788.00 | 1789.59 | 1787.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 14:45:00 | 1785.50 | 1789.59 | 1787.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 1788.40 | 1789.35 | 1787.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 1786.80 | 1789.35 | 1787.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 1774.40 | 1786.36 | 1786.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-11 10:15:00 | 1762.50 | 1781.59 | 1784.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 09:15:00 | 1756.50 | 1750.27 | 1760.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1756.50 | 1750.27 | 1760.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1756.50 | 1750.27 | 1760.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 09:45:00 | 1758.10 | 1750.27 | 1760.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1756.60 | 1751.53 | 1760.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1756.60 | 1751.53 | 1760.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1755.80 | 1752.39 | 1759.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:45:00 | 1755.90 | 1752.39 | 1759.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1743.90 | 1747.67 | 1754.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:30:00 | 1758.70 | 1747.67 | 1754.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1753.20 | 1747.65 | 1751.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 1753.20 | 1747.65 | 1751.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1755.00 | 1749.12 | 1752.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1765.00 | 1749.12 | 1752.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1751.70 | 1749.64 | 1752.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:15:00 | 1747.50 | 1749.64 | 1752.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:15:00 | 1750.90 | 1748.90 | 1750.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 1728.70 | 1748.85 | 1750.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 13:15:00 | 1663.36 | 1685.93 | 1700.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 14:15:00 | 1660.12 | 1680.02 | 1696.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 11:15:00 | 1642.26 | 1657.52 | 1679.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 1627.30 | 1625.89 | 1643.54 | SL hit (close>ema200) qty=0.50 sl=1625.89 alert=retest2 |

### Cycle 188 — BUY (started 2025-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 13:15:00 | 1663.00 | 1648.52 | 1648.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 14:15:00 | 1665.10 | 1651.84 | 1649.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 15:15:00 | 1659.00 | 1659.08 | 1655.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 09:15:00 | 1646.60 | 1659.08 | 1655.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1650.90 | 1657.45 | 1655.24 | EMA400 retest candle locked (from upside) |

### Cycle 189 — SELL (started 2025-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 12:15:00 | 1648.50 | 1653.32 | 1653.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 1639.90 | 1647.65 | 1650.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 10:15:00 | 1639.00 | 1630.78 | 1638.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 1639.00 | 1630.78 | 1638.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1639.00 | 1630.78 | 1638.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 1639.00 | 1630.78 | 1638.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 1632.10 | 1631.05 | 1637.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 12:15:00 | 1626.50 | 1631.05 | 1637.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:00:00 | 1625.90 | 1628.22 | 1633.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 13:30:00 | 1630.30 | 1632.17 | 1633.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 14:15:00 | 1642.00 | 1634.14 | 1634.47 | SL hit (close>static) qty=1.00 sl=1639.00 alert=retest2 |

### Cycle 190 — BUY (started 2025-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 15:15:00 | 1638.00 | 1634.91 | 1634.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1647.80 | 1637.49 | 1635.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 1653.30 | 1668.17 | 1658.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 13:15:00 | 1653.30 | 1668.17 | 1658.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1653.30 | 1668.17 | 1658.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1653.30 | 1668.17 | 1658.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1658.40 | 1666.22 | 1658.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 1645.50 | 1666.22 | 1658.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1659.00 | 1664.78 | 1658.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1659.80 | 1664.78 | 1658.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 1646.60 | 1661.14 | 1657.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:45:00 | 1646.80 | 1661.14 | 1657.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 1642.30 | 1657.37 | 1656.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 1642.30 | 1657.37 | 1656.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 1623.20 | 1650.54 | 1653.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 1602.00 | 1640.83 | 1648.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 09:15:00 | 1630.10 | 1626.85 | 1638.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 10:00:00 | 1630.10 | 1626.85 | 1638.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 1649.10 | 1631.30 | 1639.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 11:00:00 | 1649.10 | 1631.30 | 1639.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1639.80 | 1633.00 | 1639.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 12:30:00 | 1632.00 | 1631.52 | 1638.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 1625.30 | 1633.19 | 1636.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 12:15:00 | 1644.70 | 1637.04 | 1636.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1644.70 | 1637.04 | 1636.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 1665.90 | 1643.35 | 1639.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1640.00 | 1651.87 | 1647.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1640.00 | 1651.87 | 1647.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1640.00 | 1651.87 | 1647.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 1641.00 | 1651.87 | 1647.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1639.10 | 1649.31 | 1646.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 1639.10 | 1649.31 | 1646.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1638.70 | 1648.64 | 1647.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 1638.70 | 1648.64 | 1647.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 1648.30 | 1648.57 | 1647.59 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 1633.60 | 1644.33 | 1645.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 1632.70 | 1642.01 | 1644.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 1638.90 | 1623.51 | 1629.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 1638.90 | 1623.51 | 1629.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1638.90 | 1623.51 | 1629.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 1637.50 | 1623.51 | 1629.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1656.20 | 1630.04 | 1632.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1656.20 | 1630.04 | 1632.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — BUY (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 12:15:00 | 1661.80 | 1636.40 | 1634.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1675.80 | 1659.91 | 1650.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 12:15:00 | 1662.30 | 1668.94 | 1659.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 13:00:00 | 1662.30 | 1668.94 | 1659.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 1662.30 | 1667.25 | 1660.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 14:45:00 | 1663.00 | 1667.25 | 1660.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 1670.00 | 1667.80 | 1661.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 1671.00 | 1667.80 | 1661.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 10:00:00 | 1670.40 | 1668.32 | 1662.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 15:15:00 | 1658.50 | 1663.84 | 1662.28 | SL hit (close<static) qty=1.00 sl=1660.30 alert=retest2 |

### Cycle 195 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 1667.60 | 1674.86 | 1675.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1657.60 | 1671.41 | 1673.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 1665.00 | 1657.76 | 1664.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 1665.00 | 1657.76 | 1664.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1665.00 | 1657.76 | 1664.45 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 1668.60 | 1665.89 | 1665.84 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 09:15:00 | 1656.00 | 1663.92 | 1664.95 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 1672.20 | 1666.03 | 1665.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 1685.00 | 1669.83 | 1667.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 1731.00 | 1734.25 | 1716.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 1731.00 | 1734.25 | 1716.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1722.70 | 1732.21 | 1724.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 1722.70 | 1732.21 | 1724.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1725.40 | 1730.85 | 1724.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:30:00 | 1721.10 | 1730.85 | 1724.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 15:15:00 | 1720.60 | 1728.80 | 1724.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:15:00 | 1715.20 | 1728.80 | 1724.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 1715.00 | 1726.04 | 1723.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:45:00 | 1714.60 | 1726.04 | 1723.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 1721.70 | 1725.17 | 1723.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 1715.50 | 1725.17 | 1723.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 1714.50 | 1723.04 | 1722.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 1714.50 | 1723.04 | 1722.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 1704.60 | 1719.35 | 1720.80 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 11:15:00 | 1724.90 | 1720.41 | 1720.18 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1695.00 | 1717.02 | 1718.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 1675.80 | 1703.46 | 1712.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 1695.80 | 1692.58 | 1703.20 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 09:15:00 | 1653.00 | 1692.58 | 1703.20 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1654.20 | 1661.58 | 1677.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:00:00 | 1641.90 | 1657.64 | 1674.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 1640.00 | 1653.61 | 1671.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 1637.50 | 1653.61 | 1671.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:45:00 | 1634.60 | 1650.25 | 1667.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1641.20 | 1648.38 | 1661.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:30:00 | 1653.60 | 1648.38 | 1661.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1648.00 | 1649.78 | 1659.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:45:00 | 1643.40 | 1648.71 | 1657.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1678.10 | 1654.67 | 1658.08 | SL hit (close>ema400) qty=1.00 sl=1658.08 alert=retest1 |

### Cycle 202 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 1678.20 | 1663.53 | 1661.77 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 1645.00 | 1659.91 | 1660.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 1568.80 | 1636.59 | 1647.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1469.40 | 1458.57 | 1474.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-27 14:45:00 | 1468.80 | 1458.57 | 1474.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 1468.00 | 1460.46 | 1474.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 1490.10 | 1460.46 | 1474.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1473.20 | 1463.01 | 1474.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 12:15:00 | 1464.20 | 1466.03 | 1473.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 12:15:00 | 1490.30 | 1476.38 | 1475.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 1490.30 | 1476.38 | 1475.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 1501.10 | 1483.77 | 1479.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 1485.10 | 1487.45 | 1481.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 1485.10 | 1487.45 | 1481.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1485.10 | 1487.45 | 1481.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 1485.10 | 1487.45 | 1481.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1482.80 | 1486.52 | 1481.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 1482.20 | 1486.52 | 1481.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 1478.90 | 1485.00 | 1481.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:45:00 | 1476.20 | 1485.00 | 1481.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1478.70 | 1483.74 | 1481.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:15:00 | 1482.20 | 1483.74 | 1481.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 09:45:00 | 1485.10 | 1486.13 | 1483.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 1481.10 | 1486.38 | 1484.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 1462.40 | 1481.58 | 1482.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1462.40 | 1481.58 | 1482.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1457.60 | 1476.79 | 1480.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 1462.90 | 1461.36 | 1469.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:00:00 | 1462.90 | 1461.36 | 1469.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1476.70 | 1464.28 | 1469.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 1476.70 | 1464.28 | 1469.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1474.40 | 1466.30 | 1470.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1533.00 | 1466.30 | 1470.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 1536.30 | 1480.30 | 1476.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 11:15:00 | 1551.10 | 1535.51 | 1522.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1505.90 | 1536.39 | 1528.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 1505.90 | 1536.39 | 1528.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1505.90 | 1536.39 | 1528.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1505.90 | 1536.39 | 1528.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1515.50 | 1532.21 | 1527.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 15:00:00 | 1532.00 | 1526.89 | 1526.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 1518.20 | 1556.44 | 1558.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 1518.20 | 1556.44 | 1558.79 | EMA200 below EMA400 |

### Cycle 208 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 1561.70 | 1555.37 | 1555.04 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 15:15:00 | 1542.80 | 1554.12 | 1555.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 09:15:00 | 1540.10 | 1551.31 | 1554.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 1543.10 | 1540.90 | 1546.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 1543.10 | 1540.90 | 1546.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1543.10 | 1540.90 | 1546.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 1543.10 | 1540.90 | 1546.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1518.60 | 1513.29 | 1522.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:30:00 | 1522.00 | 1513.29 | 1522.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1520.10 | 1514.66 | 1522.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 1520.10 | 1514.66 | 1522.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1519.50 | 1515.62 | 1521.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 1521.80 | 1515.38 | 1521.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 1516.10 | 1515.38 | 1519.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:00:00 | 1516.10 | 1515.38 | 1519.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 1520.50 | 1516.41 | 1519.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:00:00 | 1520.50 | 1516.41 | 1519.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 1520.60 | 1517.24 | 1519.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 1520.60 | 1517.24 | 1519.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1525.00 | 1518.80 | 1520.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1510.40 | 1518.80 | 1520.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:15:00 | 1518.30 | 1519.62 | 1520.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:00:00 | 1513.90 | 1519.06 | 1520.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 09:30:00 | 1512.70 | 1514.62 | 1516.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1523.10 | 1516.32 | 1517.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 11:45:00 | 1516.80 | 1516.08 | 1517.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 10:00:00 | 1519.40 | 1515.23 | 1515.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 10:15:00 | 1524.00 | 1516.99 | 1516.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 210 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 1524.00 | 1516.99 | 1516.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 1545.30 | 1523.54 | 1519.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 10:15:00 | 1524.30 | 1526.55 | 1522.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 10:15:00 | 1524.30 | 1526.55 | 1522.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 1524.30 | 1526.55 | 1522.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 10:45:00 | 1522.00 | 1526.55 | 1522.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 1525.40 | 1526.68 | 1523.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 12:45:00 | 1523.10 | 1526.68 | 1523.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 1530.90 | 1527.52 | 1523.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:30:00 | 1524.50 | 1527.52 | 1523.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 1520.90 | 1526.20 | 1523.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 1520.90 | 1526.20 | 1523.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 1525.00 | 1525.96 | 1523.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 1484.40 | 1525.96 | 1523.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 211 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 1493.20 | 1519.41 | 1520.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1474.70 | 1501.21 | 1511.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 1506.00 | 1497.40 | 1506.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 15:15:00 | 1506.00 | 1497.40 | 1506.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1506.00 | 1497.40 | 1506.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 1445.70 | 1497.40 | 1506.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:45:00 | 1482.00 | 1475.14 | 1477.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1482.90 | 1474.61 | 1476.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 1483.50 | 1477.17 | 1477.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 12:15:00 | 1485.60 | 1478.85 | 1478.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 212 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 1485.60 | 1478.85 | 1478.53 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 1475.00 | 1477.92 | 1478.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 1448.80 | 1472.10 | 1475.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 1471.70 | 1461.85 | 1467.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 14:15:00 | 1471.70 | 1461.85 | 1467.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 1471.70 | 1461.85 | 1467.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 15:00:00 | 1471.70 | 1461.85 | 1467.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 1475.00 | 1464.48 | 1468.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 1478.10 | 1464.48 | 1468.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1471.80 | 1465.95 | 1468.84 | EMA400 retest candle locked (from downside) |

### Cycle 214 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 1483.90 | 1471.42 | 1470.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 15:15:00 | 1506.00 | 1483.98 | 1477.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 1484.10 | 1487.55 | 1481.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 11:15:00 | 1484.10 | 1487.55 | 1481.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 1484.10 | 1487.55 | 1481.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:00:00 | 1484.10 | 1487.55 | 1481.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1481.80 | 1486.40 | 1481.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 1481.80 | 1486.40 | 1481.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1478.70 | 1484.86 | 1480.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 1478.70 | 1484.86 | 1480.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1483.20 | 1484.53 | 1481.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:15:00 | 1470.10 | 1484.53 | 1481.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1470.10 | 1481.64 | 1480.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1459.30 | 1481.64 | 1480.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 215 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1467.00 | 1478.71 | 1478.93 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 1486.10 | 1480.19 | 1479.58 | EMA200 above EMA400 |

### Cycle 217 — SELL (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 13:15:00 | 1475.10 | 1479.12 | 1479.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 1469.40 | 1477.18 | 1478.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 15:15:00 | 1484.90 | 1478.72 | 1478.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 15:15:00 | 1484.90 | 1478.72 | 1478.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 1484.90 | 1478.72 | 1478.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 1465.70 | 1478.72 | 1478.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 1464.30 | 1449.41 | 1447.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 218 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 1464.30 | 1449.41 | 1447.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 1475.60 | 1456.47 | 1451.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 1431.00 | 1460.14 | 1456.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 1431.00 | 1460.14 | 1456.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 1431.00 | 1460.14 | 1456.32 | EMA400 retest candle locked (from upside) |

### Cycle 219 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 1438.60 | 1451.91 | 1453.00 | EMA200 below EMA400 |

### Cycle 220 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 1455.00 | 1451.29 | 1451.25 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 1448.00 | 1450.63 | 1450.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 1443.10 | 1449.13 | 1450.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1428.00 | 1416.84 | 1427.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 1428.00 | 1416.84 | 1427.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1428.00 | 1416.84 | 1427.93 | EMA400 retest candle locked (from downside) |

### Cycle 222 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 1444.70 | 1432.54 | 1432.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1504.00 | 1448.36 | 1439.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1475.80 | 1482.55 | 1465.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1475.80 | 1482.55 | 1465.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1475.80 | 1482.55 | 1465.32 | EMA400 retest candle locked (from upside) |

### Cycle 223 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 1435.60 | 1460.71 | 1462.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 1432.30 | 1449.29 | 1456.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1449.60 | 1439.29 | 1448.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1449.60 | 1439.29 | 1448.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1449.60 | 1439.29 | 1448.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1449.60 | 1439.29 | 1448.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1440.00 | 1439.43 | 1448.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 1448.00 | 1439.43 | 1448.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 1445.30 | 1440.60 | 1447.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:30:00 | 1444.30 | 1440.60 | 1447.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 1442.80 | 1441.04 | 1447.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:45:00 | 1448.00 | 1441.04 | 1447.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1448.90 | 1442.62 | 1447.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 14:00:00 | 1448.90 | 1442.62 | 1447.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 14:15:00 | 1465.70 | 1447.23 | 1449.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 15:00:00 | 1465.70 | 1447.23 | 1449.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 224 — BUY (started 2026-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 15:15:00 | 1474.00 | 1452.59 | 1451.46 | EMA200 above EMA400 |

### Cycle 225 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1430.60 | 1448.19 | 1449.57 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 11:15:00 | 1464.30 | 1451.89 | 1451.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 1477.60 | 1457.03 | 1453.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 15:15:00 | 1641.50 | 1644.32 | 1613.99 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1659.20 | 1644.32 | 1613.99 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1658.30 | 1669.25 | 1647.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 1667.60 | 1668.06 | 1649.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 1700.80 | 1708.33 | 1703.29 | SL hit (close<ema400) qty=1.00 sl=1703.29 alert=retest1 |

### Cycle 227 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 1695.00 | 1700.29 | 1700.93 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 1731.10 | 1706.45 | 1703.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 10:15:00 | 1746.00 | 1714.36 | 1707.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 14:15:00 | 1733.80 | 1734.80 | 1725.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-22 15:00:00 | 1733.80 | 1734.80 | 1725.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 1730.00 | 1733.84 | 1726.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 1722.70 | 1733.84 | 1726.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1722.40 | 1731.55 | 1725.68 | EMA400 retest candle locked (from upside) |

### Cycle 229 — SELL (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 13:15:00 | 1711.40 | 1721.07 | 1722.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 1708.50 | 1718.56 | 1720.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1717.80 | 1699.93 | 1706.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1717.80 | 1699.93 | 1706.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1717.80 | 1699.93 | 1706.89 | EMA400 retest candle locked (from downside) |

### Cycle 230 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1725.70 | 1711.38 | 1710.99 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 14:15:00 | 1701.80 | 1713.06 | 1714.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 1672.00 | 1703.24 | 1709.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 1684.60 | 1678.42 | 1690.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 1684.60 | 1678.42 | 1690.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1684.60 | 1678.42 | 1690.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:15:00 | 1661.60 | 1685.23 | 1689.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 13:45:00 | 1665.20 | 1675.30 | 1683.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 14:45:00 | 1663.00 | 1673.28 | 1681.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 10:45:00 | 1664.20 | 1668.89 | 1677.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1670.60 | 1667.49 | 1673.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:30:00 | 1676.50 | 1667.49 | 1673.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1678.00 | 1669.59 | 1674.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 1670.40 | 1669.59 | 1674.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1665.70 | 1668.81 | 1673.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 1694.00 | 1675.71 | 1674.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 232 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 1694.00 | 1675.71 | 1674.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 13:15:00 | 1696.30 | 1686.50 | 1680.71 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 15:00:00 | 1503.05 | 2024-04-18 13:15:00 | 1427.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 15:00:00 | 1503.05 | 2024-04-22 09:15:00 | 1404.55 | STOP_HIT | 0.50 | 6.55% |
| BUY | retest2 | 2024-04-29 14:15:00 | 1480.65 | 2024-05-02 09:15:00 | 1467.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2024-04-29 15:00:00 | 1480.40 | 2024-05-02 09:15:00 | 1467.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-05-02 12:15:00 | 1483.90 | 2024-05-03 15:15:00 | 1475.25 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2024-05-13 09:15:00 | 1462.80 | 2024-05-13 13:15:00 | 1480.95 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-05-22 14:30:00 | 1742.70 | 2024-05-30 12:15:00 | 1796.95 | STOP_HIT | 1.00 | 3.11% |
| BUY | retest2 | 2024-06-13 09:15:00 | 1935.55 | 2024-06-14 11:15:00 | 1895.30 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-07-04 09:15:00 | 1806.90 | 2024-07-04 10:15:00 | 1786.25 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-07-04 11:45:00 | 1799.45 | 2024-07-05 09:15:00 | 1781.05 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-07-04 13:15:00 | 1800.75 | 2024-07-05 09:15:00 | 1781.05 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-07-11 10:45:00 | 1712.25 | 2024-07-15 13:15:00 | 1727.45 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-07-11 12:00:00 | 1707.40 | 2024-07-15 13:15:00 | 1727.45 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-09-04 15:00:00 | 1788.00 | 2024-09-06 09:15:00 | 1749.05 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-09-05 10:00:00 | 1791.25 | 2024-09-06 09:15:00 | 1749.05 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2024-09-05 10:45:00 | 1793.10 | 2024-09-06 09:15:00 | 1749.05 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2024-09-05 11:45:00 | 1789.85 | 2024-09-06 09:15:00 | 1749.05 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-09-12 09:15:00 | 1761.40 | 2024-09-12 10:15:00 | 1752.95 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-09-19 09:15:00 | 1829.10 | 2024-09-19 11:15:00 | 1795.85 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-09-19 12:30:00 | 1816.25 | 2024-09-27 14:15:00 | 1896.65 | STOP_HIT | 1.00 | 4.43% |
| BUY | retest2 | 2024-09-19 13:15:00 | 1814.05 | 2024-09-27 14:15:00 | 1896.65 | STOP_HIT | 1.00 | 4.55% |
| BUY | retest2 | 2024-09-19 14:30:00 | 1812.85 | 2024-09-27 14:15:00 | 1896.65 | STOP_HIT | 1.00 | 4.62% |
| BUY | retest2 | 2024-09-25 10:15:00 | 1915.20 | 2024-09-27 14:15:00 | 1896.65 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-09-25 12:00:00 | 1914.40 | 2024-09-27 14:15:00 | 1896.65 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1869.85 | 2024-10-07 10:15:00 | 1776.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 1869.85 | 2024-10-08 13:15:00 | 1754.00 | STOP_HIT | 0.50 | 6.20% |
| BUY | retest2 | 2024-10-23 10:15:00 | 1978.15 | 2024-10-25 09:15:00 | 1936.20 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-10-23 12:45:00 | 1970.00 | 2024-10-25 09:15:00 | 1936.20 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-10-23 13:45:00 | 1970.00 | 2024-10-25 09:15:00 | 1936.20 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-10-23 15:15:00 | 1975.00 | 2024-10-25 09:15:00 | 1936.20 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-11-11 09:45:00 | 2035.85 | 2024-11-11 11:15:00 | 2015.95 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-11-11 10:15:00 | 2034.90 | 2024-11-11 11:15:00 | 2015.95 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-11-14 13:45:00 | 1981.00 | 2024-11-25 09:15:00 | 2000.50 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-12-12 14:45:00 | 2127.80 | 2024-12-16 09:15:00 | 2178.75 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2024-12-20 09:30:00 | 2325.00 | 2024-12-20 12:15:00 | 2272.75 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2024-12-20 10:30:00 | 2317.45 | 2024-12-20 12:15:00 | 2272.75 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-12-27 13:15:00 | 2314.25 | 2024-12-30 13:15:00 | 2277.90 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2024-12-27 15:00:00 | 2320.05 | 2024-12-30 13:15:00 | 2277.90 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2024-12-30 11:15:00 | 2313.30 | 2024-12-30 13:15:00 | 2277.90 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-01-06 11:00:00 | 2216.60 | 2025-01-07 14:15:00 | 2268.65 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-01-09 11:30:00 | 2263.45 | 2025-01-09 15:15:00 | 2240.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-01-16 11:15:00 | 1992.00 | 2025-01-21 09:15:00 | 1892.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 11:15:00 | 1992.00 | 2025-01-22 09:15:00 | 1792.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-21 09:15:00 | 1896.30 | 2025-01-22 09:15:00 | 1801.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 09:15:00 | 1896.30 | 2025-01-23 09:15:00 | 1824.15 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2025-02-11 10:30:00 | 1724.05 | 2025-02-12 09:15:00 | 1637.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 11:00:00 | 1712.00 | 2025-02-12 09:15:00 | 1626.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 10:30:00 | 1724.05 | 2025-02-14 13:15:00 | 1551.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-11 11:00:00 | 1712.00 | 2025-02-17 09:15:00 | 1540.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-04 11:15:00 | 1507.55 | 2025-03-04 13:15:00 | 1516.90 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-03-04 12:00:00 | 1507.25 | 2025-03-04 13:15:00 | 1516.90 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-03-04 12:30:00 | 1508.10 | 2025-03-04 13:15:00 | 1516.90 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest1 | 2025-03-07 09:15:00 | 1565.35 | 2025-03-07 11:15:00 | 1537.55 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-03-07 14:30:00 | 1547.75 | 2025-03-10 13:15:00 | 1529.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-03-10 12:30:00 | 1543.60 | 2025-03-10 13:15:00 | 1529.50 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-03-12 13:15:00 | 1570.40 | 2025-03-13 13:15:00 | 1541.40 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-04-03 14:30:00 | 1592.00 | 2025-04-07 09:15:00 | 1512.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 14:30:00 | 1592.00 | 2025-04-08 09:15:00 | 1503.05 | STOP_HIT | 0.50 | 5.59% |
| BUY | retest2 | 2025-04-21 10:30:00 | 1630.00 | 2025-04-25 09:15:00 | 1619.50 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-04-30 12:30:00 | 1647.90 | 2025-04-30 14:15:00 | 1641.70 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-05-06 11:30:00 | 1593.90 | 2025-05-09 09:15:00 | 1514.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 12:15:00 | 1597.10 | 2025-05-09 09:15:00 | 1517.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:30:00 | 1593.90 | 2025-05-09 15:15:00 | 1540.00 | STOP_HIT | 0.50 | 3.38% |
| SELL | retest2 | 2025-05-06 12:15:00 | 1597.10 | 2025-05-09 15:15:00 | 1540.00 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-05-12 09:30:00 | 1581.40 | 2025-05-12 11:15:00 | 1597.70 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-05-14 09:15:00 | 1597.30 | 2025-05-20 10:15:00 | 1757.03 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-30 10:15:00 | 1731.70 | 2025-05-30 14:15:00 | 1757.80 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-05-30 12:30:00 | 1730.80 | 2025-05-30 14:15:00 | 1757.80 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-05-30 13:15:00 | 1734.00 | 2025-05-30 14:15:00 | 1757.80 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-06-05 09:15:00 | 1789.00 | 2025-06-13 09:15:00 | 1883.90 | STOP_HIT | 1.00 | 5.30% |
| BUY | retest2 | 2025-06-05 10:30:00 | 1795.50 | 2025-06-13 09:15:00 | 1883.90 | STOP_HIT | 1.00 | 4.92% |
| BUY | retest2 | 2025-06-18 09:15:00 | 1924.50 | 2025-06-18 10:15:00 | 1898.90 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-06-20 12:15:00 | 1898.20 | 2025-06-20 14:15:00 | 1910.30 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-06-20 12:45:00 | 1898.00 | 2025-06-20 14:15:00 | 1910.30 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-07-01 13:00:00 | 1916.10 | 2025-07-08 09:15:00 | 1820.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-01 13:00:00 | 1916.10 | 2025-07-08 12:15:00 | 1840.60 | STOP_HIT | 0.50 | 3.94% |
| SELL | retest1 | 2025-07-24 09:45:00 | 1760.20 | 2025-07-28 09:15:00 | 1672.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-07-24 09:45:00 | 1760.20 | 2025-07-29 09:15:00 | 1666.80 | STOP_HIT | 0.50 | 5.31% |
| SELL | retest2 | 2025-07-29 15:00:00 | 1664.50 | 2025-08-01 15:15:00 | 1585.45 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2025-07-29 15:00:00 | 1664.50 | 2025-08-04 12:15:00 | 1604.30 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2025-07-30 11:45:00 | 1668.90 | 2025-08-08 10:15:00 | 1581.27 | PARTIAL | 0.50 | 5.25% |
| SELL | retest2 | 2025-07-30 11:45:00 | 1668.90 | 2025-08-11 10:15:00 | 1591.40 | STOP_HIT | 0.50 | 4.64% |
| BUY | retest1 | 2025-08-13 09:15:00 | 1634.50 | 2025-08-14 13:15:00 | 1618.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-08-18 09:15:00 | 1633.50 | 2025-08-26 11:15:00 | 1650.00 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2025-08-29 12:15:00 | 1619.90 | 2025-09-02 09:15:00 | 1658.60 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-09-01 11:00:00 | 1627.10 | 2025-09-02 09:15:00 | 1658.60 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-09-01 11:45:00 | 1623.20 | 2025-09-02 09:15:00 | 1658.60 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-09-10 11:45:00 | 1619.30 | 2025-09-15 12:15:00 | 1633.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-09-10 13:00:00 | 1616.00 | 2025-09-15 12:15:00 | 1633.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-11 10:00:00 | 1615.50 | 2025-09-15 12:15:00 | 1633.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-09-15 12:00:00 | 1619.90 | 2025-09-15 12:15:00 | 1633.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-19 14:00:00 | 1675.00 | 2025-09-24 09:15:00 | 1638.40 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-09-19 14:30:00 | 1673.10 | 2025-09-24 09:15:00 | 1638.40 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-09-22 09:30:00 | 1673.20 | 2025-09-24 09:15:00 | 1638.40 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-09-22 14:45:00 | 1672.80 | 2025-09-24 09:15:00 | 1638.40 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-09-30 10:30:00 | 1580.90 | 2025-10-03 11:15:00 | 1599.10 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-30 13:30:00 | 1580.40 | 2025-10-03 11:15:00 | 1599.10 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-10-09 14:15:00 | 1586.80 | 2025-10-10 09:15:00 | 1606.40 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-09 14:45:00 | 1586.70 | 2025-10-10 09:15:00 | 1606.40 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-10-09 15:15:00 | 1587.00 | 2025-10-10 09:15:00 | 1606.40 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-10-27 09:15:00 | 1730.90 | 2025-11-06 14:15:00 | 1780.40 | STOP_HIT | 1.00 | 2.86% |
| SELL | retest2 | 2025-11-17 10:15:00 | 1747.50 | 2025-11-21 13:15:00 | 1663.36 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2025-11-17 14:15:00 | 1750.90 | 2025-11-21 14:15:00 | 1660.12 | PARTIAL | 0.50 | 5.18% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1728.70 | 2025-11-24 11:15:00 | 1642.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-17 10:15:00 | 1747.50 | 2025-11-25 14:15:00 | 1627.30 | STOP_HIT | 0.50 | 6.88% |
| SELL | retest2 | 2025-11-17 14:15:00 | 1750.90 | 2025-11-25 14:15:00 | 1627.30 | STOP_HIT | 0.50 | 7.06% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1728.70 | 2025-11-25 14:15:00 | 1627.30 | STOP_HIT | 0.50 | 5.87% |
| SELL | retest2 | 2025-12-02 12:15:00 | 1626.50 | 2025-12-03 14:15:00 | 1642.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-03 10:00:00 | 1625.90 | 2025-12-03 14:15:00 | 1642.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-03 13:30:00 | 1630.30 | 2025-12-03 14:15:00 | 1642.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-09 12:30:00 | 1632.00 | 2025-12-11 12:15:00 | 1644.70 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1625.30 | 2025-12-11 12:15:00 | 1644.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-12-23 09:15:00 | 1671.00 | 2025-12-23 15:15:00 | 1658.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-12-23 10:00:00 | 1670.40 | 2025-12-23 15:15:00 | 1658.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1686.30 | 2025-12-29 15:15:00 | 1667.60 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-24 14:45:00 | 1671.90 | 2025-12-29 15:15:00 | 1667.60 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-12-26 10:45:00 | 1690.00 | 2025-12-29 15:15:00 | 1667.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-12-26 13:30:00 | 1692.20 | 2025-12-29 15:15:00 | 1667.60 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-29 09:15:00 | 1690.40 | 2025-12-29 15:15:00 | 1667.60 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest1 | 2026-01-12 09:15:00 | 1653.00 | 2026-01-16 09:15:00 | 1678.10 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-01-13 11:00:00 | 1641.90 | 2026-01-16 09:15:00 | 1678.10 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-01-13 11:30:00 | 1640.00 | 2026-01-16 11:15:00 | 1678.20 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1637.50 | 2026-01-16 11:15:00 | 1678.20 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2026-01-13 12:45:00 | 1634.60 | 2026-01-16 11:15:00 | 1678.20 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-01-14 13:45:00 | 1643.40 | 2026-01-16 11:15:00 | 1678.20 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-01-28 12:15:00 | 1464.20 | 2026-01-29 12:15:00 | 1490.30 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-01-30 13:15:00 | 1482.20 | 2026-02-01 13:15:00 | 1462.40 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2026-02-01 09:45:00 | 1485.10 | 2026-02-01 13:15:00 | 1462.40 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-02-01 12:30:00 | 1481.10 | 2026-02-01 13:15:00 | 1462.40 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-02-06 15:00:00 | 1532.00 | 2026-02-13 09:15:00 | 1518.20 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1510.40 | 2026-02-26 10:15:00 | 1524.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2026-02-24 10:15:00 | 1518.30 | 2026-02-26 10:15:00 | 1524.00 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2026-02-24 12:00:00 | 1513.90 | 2026-02-26 10:15:00 | 1524.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-02-25 09:30:00 | 1512.70 | 2026-02-26 10:15:00 | 1524.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-02-25 11:45:00 | 1516.80 | 2026-02-26 10:15:00 | 1524.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-02-26 10:00:00 | 1519.40 | 2026-02-26 10:15:00 | 1524.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2026-03-04 09:15:00 | 1445.70 | 2026-03-06 12:15:00 | 1485.60 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2026-03-06 09:45:00 | 1482.00 | 2026-03-06 12:15:00 | 1485.60 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2026-03-06 10:45:00 | 1482.90 | 2026-03-06 12:15:00 | 1485.60 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-03-06 12:15:00 | 1483.50 | 2026-03-06 12:15:00 | 1485.60 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2026-03-13 09:15:00 | 1465.70 | 2026-03-18 09:15:00 | 1464.30 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest1 | 2026-04-10 09:15:00 | 1659.20 | 2026-04-20 09:15:00 | 1700.80 | STOP_HIT | 1.00 | 2.51% |
| BUY | retest2 | 2026-04-13 11:15:00 | 1667.60 | 2026-04-20 15:15:00 | 1695.00 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2026-05-05 10:15:00 | 1661.60 | 2026-05-08 09:15:00 | 1694.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-05-05 13:45:00 | 1665.20 | 2026-05-08 09:15:00 | 1694.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-05-05 14:45:00 | 1663.00 | 2026-05-08 09:15:00 | 1694.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-05-06 10:45:00 | 1664.20 | 2026-05-08 09:15:00 | 1694.00 | STOP_HIT | 1.00 | -1.79% |
