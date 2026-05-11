# KPIT Technologies Ltd. (KPITTECH)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 725.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 211 |
| ALERT1 | 144 |
| ALERT2 | 142 |
| ALERT2_SKIP | 86 |
| ALERT3 | 327 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 142 |
| PARTIAL | 28 |
| TARGET_HIT | 19 |
| STOP_HIT | 128 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 175 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 68 / 107
- **Target hits / Stop hits / Partials:** 19 / 128 / 28
- **Avg / median % per leg:** 1.31% / -0.55%
- **Sum % (uncompounded):** 228.45%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 12 | 16.9% | 6 | 65 | 0 | -0.09% | -6.6% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.49% | -1.0% |
| BUY @ 3rd Alert (retest2) | 69 | 12 | 17.4% | 6 | 63 | 0 | -0.08% | -5.6% |
| SELL (all) | 104 | 56 | 53.8% | 13 | 63 | 28 | 2.26% | 235.1% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | -1.04% | -3.1% |
| SELL @ 3rd Alert (retest2) | 101 | 54 | 53.5% | 13 | 60 | 28 | 2.36% | 238.2% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 5 | 0 | -0.82% | -4.1% |
| retest2 (combined) | 170 | 66 | 38.8% | 19 | 123 | 28 | 1.37% | 232.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 15:15:00 | 926.40 | 935.64 | 936.71 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 11:15:00 | 942.05 | 938.08 | 937.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 13:15:00 | 965.40 | 944.54 | 940.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-26 11:15:00 | 1002.00 | 1003.15 | 985.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-29 12:15:00 | 994.80 | 1001.79 | 994.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 12:15:00 | 994.80 | 1001.79 | 994.38 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-13 11:15:00 | 1128.65 | 1134.17 | 1134.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-13 14:15:00 | 1093.00 | 1124.33 | 1129.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-15 09:15:00 | 1065.05 | 1057.39 | 1082.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 09:15:00 | 1083.20 | 1066.16 | 1074.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 1083.20 | 1066.16 | 1074.93 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2023-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 15:15:00 | 1086.55 | 1078.80 | 1078.37 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 13:15:00 | 1072.65 | 1078.78 | 1078.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 15:15:00 | 1070.00 | 1076.56 | 1077.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 15:15:00 | 1064.10 | 1063.98 | 1069.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 09:15:00 | 1087.00 | 1068.59 | 1071.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 1087.00 | 1068.59 | 1071.24 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2023-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 12:15:00 | 1079.60 | 1073.82 | 1073.23 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2023-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 15:15:00 | 1063.00 | 1071.04 | 1072.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-23 09:15:00 | 1051.25 | 1063.33 | 1067.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-23 11:15:00 | 1065.15 | 1063.29 | 1066.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 12:15:00 | 1082.15 | 1067.06 | 1067.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 12:15:00 | 1082.15 | 1067.06 | 1067.97 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 15:15:00 | 1076.80 | 1067.21 | 1066.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 11:15:00 | 1079.10 | 1072.81 | 1069.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-30 10:15:00 | 1086.45 | 1088.35 | 1082.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 14:15:00 | 1083.00 | 1088.13 | 1084.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 14:15:00 | 1083.00 | 1088.13 | 1084.39 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-03 13:15:00 | 1077.00 | 1083.32 | 1083.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 09:15:00 | 1065.85 | 1077.86 | 1080.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-06 09:15:00 | 1070.60 | 1060.98 | 1065.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-06 09:15:00 | 1070.60 | 1060.98 | 1065.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 1070.60 | 1060.98 | 1065.63 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-07-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 12:15:00 | 1076.95 | 1070.05 | 1069.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 14:15:00 | 1085.00 | 1074.33 | 1071.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 10:15:00 | 1077.85 | 1078.16 | 1074.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 09:15:00 | 1069.00 | 1076.64 | 1075.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 1069.00 | 1076.64 | 1075.19 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 11:15:00 | 1067.90 | 1073.42 | 1073.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 13:15:00 | 1057.00 | 1069.03 | 1071.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 11:15:00 | 1064.40 | 1061.15 | 1066.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 12:15:00 | 1064.20 | 1061.76 | 1066.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 12:15:00 | 1064.20 | 1061.76 | 1066.17 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-13 09:15:00 | 1076.25 | 1064.24 | 1063.82 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-07-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 13:15:00 | 1048.75 | 1061.20 | 1062.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 10:15:00 | 1036.05 | 1051.80 | 1057.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 12:15:00 | 1049.40 | 1048.56 | 1054.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 13:15:00 | 1062.20 | 1051.29 | 1055.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 1062.20 | 1051.29 | 1055.46 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 1093.55 | 1062.76 | 1059.94 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-20 11:15:00 | 1065.00 | 1071.72 | 1072.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 14:15:00 | 1061.55 | 1067.50 | 1070.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-24 10:15:00 | 1018.35 | 1009.94 | 1030.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-24 13:15:00 | 1026.30 | 1010.19 | 1025.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 13:15:00 | 1026.30 | 1010.19 | 1025.32 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 09:15:00 | 1067.35 | 1038.47 | 1035.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-01 10:15:00 | 1097.80 | 1083.30 | 1077.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 14:15:00 | 1091.85 | 1094.29 | 1085.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-02 09:15:00 | 1119.50 | 1099.01 | 1089.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 1119.50 | 1099.01 | 1089.38 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-03 12:15:00 | 1079.15 | 1090.33 | 1091.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-03 13:15:00 | 1077.70 | 1087.81 | 1090.10 | Break + close below crossover candle low |

### Cycle 18 — BUY (started 2023-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-04 09:15:00 | 1128.00 | 1094.25 | 1092.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-04 10:15:00 | 1136.50 | 1102.70 | 1096.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-08 10:15:00 | 1144.80 | 1145.26 | 1133.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-09 09:15:00 | 1154.50 | 1147.76 | 1139.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-09 09:15:00 | 1154.50 | 1147.76 | 1139.86 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-08-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 10:15:00 | 1126.65 | 1138.28 | 1139.26 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 11:15:00 | 1149.80 | 1137.25 | 1136.44 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 10:15:00 | 1121.90 | 1134.33 | 1135.87 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 10:15:00 | 1145.45 | 1135.02 | 1134.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-17 09:15:00 | 1156.00 | 1140.47 | 1137.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 15:15:00 | 1152.00 | 1153.21 | 1149.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 09:15:00 | 1129.00 | 1148.36 | 1147.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 1129.00 | 1148.36 | 1147.24 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2023-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 11:15:00 | 1134.65 | 1144.15 | 1145.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-21 15:15:00 | 1131.90 | 1138.01 | 1141.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-22 15:15:00 | 1134.90 | 1134.29 | 1137.67 | EMA200 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-23 09:15:00 | 1169.75 | 1141.38 | 1140.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-23 10:15:00 | 1192.85 | 1151.68 | 1145.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 15:15:00 | 1173.00 | 1173.56 | 1166.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-25 09:15:00 | 1158.00 | 1170.45 | 1165.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 1158.00 | 1170.45 | 1165.29 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 12:15:00 | 1152.15 | 1161.90 | 1162.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 14:15:00 | 1148.75 | 1157.92 | 1160.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 11:15:00 | 1154.55 | 1152.60 | 1156.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 1157.00 | 1152.66 | 1154.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 1157.00 | 1152.66 | 1154.97 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 12:15:00 | 1162.20 | 1157.22 | 1156.71 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 11:15:00 | 1155.00 | 1156.50 | 1156.67 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-08-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 09:15:00 | 1164.20 | 1158.03 | 1157.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-31 14:15:00 | 1178.80 | 1164.91 | 1161.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 09:15:00 | 1193.65 | 1193.67 | 1181.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 13:15:00 | 1182.15 | 1190.17 | 1183.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 1182.15 | 1190.17 | 1183.78 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-05 10:15:00 | 1167.85 | 1180.48 | 1180.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-05 13:15:00 | 1165.00 | 1175.47 | 1178.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-06 09:15:00 | 1175.00 | 1172.64 | 1176.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 09:15:00 | 1175.00 | 1172.64 | 1176.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 1175.00 | 1172.64 | 1176.06 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 10:15:00 | 1179.75 | 1171.60 | 1170.51 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 09:15:00 | 1160.25 | 1168.52 | 1169.58 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2023-09-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 12:15:00 | 1163.80 | 1155.87 | 1155.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-14 14:15:00 | 1168.00 | 1159.68 | 1156.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 1166.85 | 1169.65 | 1164.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-15 14:15:00 | 1166.85 | 1169.65 | 1164.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 14:15:00 | 1166.85 | 1169.65 | 1164.55 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 12:15:00 | 1154.85 | 1161.91 | 1162.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 13:15:00 | 1149.75 | 1159.48 | 1161.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-22 14:15:00 | 1077.00 | 1074.64 | 1093.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 11:15:00 | 1062.00 | 1057.28 | 1070.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 11:15:00 | 1062.00 | 1057.28 | 1070.14 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 09:15:00 | 1098.15 | 1078.89 | 1077.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 10:15:00 | 1103.55 | 1083.82 | 1079.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 14:15:00 | 1151.10 | 1151.84 | 1134.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 1128.25 | 1146.51 | 1135.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 1128.25 | 1146.51 | 1135.05 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 10:15:00 | 1127.95 | 1140.60 | 1141.10 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 09:15:00 | 1170.80 | 1141.76 | 1140.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 1210.05 | 1165.95 | 1153.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 09:15:00 | 1200.75 | 1207.19 | 1195.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 1200.75 | 1207.19 | 1195.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 1200.75 | 1207.19 | 1195.65 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 14:15:00 | 1198.25 | 1206.01 | 1206.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 09:15:00 | 1195.00 | 1202.85 | 1205.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-18 13:15:00 | 1195.00 | 1192.17 | 1198.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-18 14:15:00 | 1205.45 | 1194.83 | 1199.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 14:15:00 | 1205.45 | 1194.83 | 1199.04 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-10-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-19 11:15:00 | 1210.10 | 1202.60 | 1201.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-19 14:15:00 | 1214.00 | 1207.34 | 1204.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 11:15:00 | 1205.85 | 1209.06 | 1206.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 11:15:00 | 1205.85 | 1209.06 | 1206.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 11:15:00 | 1205.85 | 1209.06 | 1206.33 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 14:15:00 | 1197.15 | 1203.34 | 1204.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 09:15:00 | 1185.45 | 1198.75 | 1201.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 1136.45 | 1127.50 | 1142.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1136.45 | 1127.50 | 1142.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1136.45 | 1127.50 | 1142.97 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2023-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 12:15:00 | 1176.80 | 1140.94 | 1140.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 09:15:00 | 1246.40 | 1179.60 | 1160.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 12:15:00 | 1210.95 | 1215.68 | 1197.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 15:15:00 | 1220.00 | 1225.54 | 1215.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 15:15:00 | 1220.00 | 1225.54 | 1215.44 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 09:15:00 | 1454.70 | 1516.24 | 1518.31 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-28 09:15:00 | 1492.80 | 1464.17 | 1463.56 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 12:15:00 | 1482.75 | 1488.43 | 1488.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 15:15:00 | 1476.00 | 1483.37 | 1486.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-07 09:15:00 | 1476.00 | 1468.45 | 1474.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-07 09:15:00 | 1476.00 | 1468.45 | 1474.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-07 09:15:00 | 1476.00 | 1468.45 | 1474.55 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2023-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-08 13:15:00 | 1479.65 | 1472.36 | 1472.16 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 10:15:00 | 1466.35 | 1471.79 | 1472.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 10:15:00 | 1448.00 | 1462.97 | 1467.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 09:15:00 | 1500.45 | 1456.17 | 1459.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 09:15:00 | 1500.45 | 1456.17 | 1459.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 09:15:00 | 1500.45 | 1456.17 | 1459.95 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 1469.40 | 1461.74 | 1461.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 09:15:00 | 1497.10 | 1471.58 | 1466.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 09:15:00 | 1494.60 | 1511.32 | 1498.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-18 09:15:00 | 1494.60 | 1511.32 | 1498.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 1494.60 | 1511.32 | 1498.25 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 1466.60 | 1501.91 | 1503.40 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2023-12-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 14:15:00 | 1519.45 | 1503.69 | 1502.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 09:15:00 | 1542.20 | 1513.46 | 1507.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 09:15:00 | 1528.90 | 1529.97 | 1520.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-26 12:15:00 | 1519.65 | 1527.21 | 1521.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 12:15:00 | 1519.65 | 1527.21 | 1521.68 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2023-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 12:15:00 | 1500.00 | 1518.14 | 1519.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-29 13:15:00 | 1495.00 | 1506.68 | 1511.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-29 14:15:00 | 1510.00 | 1507.34 | 1511.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 14:15:00 | 1510.00 | 1507.34 | 1511.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 14:15:00 | 1510.00 | 1507.34 | 1511.02 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 10:15:00 | 1490.85 | 1476.44 | 1475.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 12:15:00 | 1497.50 | 1483.15 | 1478.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-12 13:15:00 | 1545.45 | 1545.80 | 1537.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-15 09:15:00 | 1538.55 | 1545.64 | 1539.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 1538.55 | 1545.64 | 1539.39 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-01-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-18 09:15:00 | 1520.50 | 1539.04 | 1541.08 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 15:15:00 | 1549.00 | 1537.97 | 1537.27 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 11:15:00 | 1533.55 | 1536.94 | 1536.97 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-20 12:15:00 | 1537.45 | 1537.04 | 1537.01 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2024-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-20 13:15:00 | 1534.20 | 1536.47 | 1536.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-20 14:15:00 | 1528.50 | 1534.88 | 1536.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-23 09:15:00 | 1543.70 | 1535.06 | 1535.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-23 09:15:00 | 1543.70 | 1535.06 | 1535.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 1543.70 | 1535.06 | 1535.80 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2024-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-30 14:15:00 | 1461.30 | 1433.20 | 1432.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 09:15:00 | 1497.15 | 1449.48 | 1439.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 14:15:00 | 1524.50 | 1528.47 | 1505.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-12 09:15:00 | 1751.05 | 1721.58 | 1696.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-12 09:15:00 | 1751.05 | 1721.58 | 1696.76 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2024-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-13 09:15:00 | 1613.50 | 1693.40 | 1695.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 11:15:00 | 1600.05 | 1662.59 | 1680.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 11:15:00 | 1642.20 | 1619.98 | 1643.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-14 11:15:00 | 1642.20 | 1619.98 | 1643.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 11:15:00 | 1642.20 | 1619.98 | 1643.63 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 1655.45 | 1644.95 | 1644.17 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 1634.70 | 1650.65 | 1651.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 15:15:00 | 1627.85 | 1646.09 | 1649.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 1588.05 | 1579.28 | 1594.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 1588.05 | 1579.28 | 1594.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 1588.05 | 1579.28 | 1594.85 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2024-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 09:15:00 | 1596.90 | 1591.41 | 1590.84 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 13:15:00 | 1586.65 | 1590.40 | 1590.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 14:15:00 | 1578.20 | 1587.96 | 1589.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 15:15:00 | 1580.20 | 1574.72 | 1579.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 15:15:00 | 1580.20 | 1574.72 | 1579.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 15:15:00 | 1580.20 | 1574.72 | 1579.85 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-03-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 11:15:00 | 1587.00 | 1576.63 | 1576.15 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-04 10:15:00 | 1570.15 | 1576.88 | 1577.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-04 12:15:00 | 1558.30 | 1572.55 | 1575.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 1517.70 | 1497.57 | 1516.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-06 13:15:00 | 1517.70 | 1497.57 | 1516.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 13:15:00 | 1517.70 | 1497.57 | 1516.72 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 12:15:00 | 1454.90 | 1435.64 | 1433.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 13:15:00 | 1461.30 | 1440.77 | 1436.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 14:15:00 | 1424.65 | 1437.55 | 1435.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-15 14:15:00 | 1424.65 | 1437.55 | 1435.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 1424.65 | 1437.55 | 1435.28 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-18 09:15:00 | 1386.70 | 1427.29 | 1431.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 09:15:00 | 1346.20 | 1384.03 | 1403.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 15:15:00 | 1352.95 | 1352.59 | 1366.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-21 09:15:00 | 1357.20 | 1353.51 | 1365.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 1357.20 | 1353.51 | 1365.37 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-03-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-22 13:15:00 | 1366.00 | 1362.94 | 1362.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 15:15:00 | 1393.00 | 1371.70 | 1366.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 14:15:00 | 1510.30 | 1520.00 | 1499.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 09:15:00 | 1503.95 | 1515.78 | 1501.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 1503.95 | 1515.78 | 1501.02 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 12:15:00 | 1497.80 | 1508.14 | 1509.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 13:15:00 | 1494.05 | 1505.32 | 1507.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 09:15:00 | 1516.65 | 1505.13 | 1506.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 09:15:00 | 1516.65 | 1505.13 | 1506.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 09:15:00 | 1516.65 | 1505.13 | 1506.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 1466.10 | 1496.21 | 1500.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-18 15:15:00 | 1392.79 | 1405.90 | 1422.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-19 12:15:00 | 1407.95 | 1403.29 | 1415.30 | SL hit (close>ema200) qty=0.50 sl=1403.29 alert=retest2 |

### Cycle 68 — BUY (started 2024-04-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 09:15:00 | 1408.50 | 1380.27 | 1380.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 10:15:00 | 1421.25 | 1388.46 | 1383.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-29 12:15:00 | 1413.50 | 1415.71 | 1405.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-29 13:00:00 | 1413.50 | 1415.71 | 1405.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 13:15:00 | 1510.70 | 1434.71 | 1414.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 09:15:00 | 1526.90 | 1506.27 | 1485.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 10:15:00 | 1522.45 | 1526.86 | 1510.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 10:45:00 | 1523.00 | 1525.89 | 1511.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-06 13:45:00 | 1523.80 | 1521.75 | 1513.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 14:15:00 | 1513.25 | 1520.05 | 1513.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-06 15:00:00 | 1513.25 | 1520.05 | 1513.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 15:15:00 | 1516.80 | 1519.40 | 1513.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 09:15:00 | 1506.10 | 1519.40 | 1513.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 1508.00 | 1517.12 | 1513.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-07 11:15:00 | 1497.10 | 1510.46 | 1510.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 11:15:00 | 1497.10 | 1510.46 | 1510.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 1490.40 | 1506.44 | 1508.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 14:15:00 | 1507.15 | 1505.61 | 1507.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 14:15:00 | 1507.15 | 1505.61 | 1507.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 14:15:00 | 1507.15 | 1505.61 | 1507.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-07 15:00:00 | 1507.15 | 1505.61 | 1507.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 15:15:00 | 1507.60 | 1506.01 | 1507.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 09:15:00 | 1504.35 | 1506.01 | 1507.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:00:00 | 1502.00 | 1500.27 | 1503.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 09:15:00 | 1429.13 | 1462.51 | 1476.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-13 10:15:00 | 1426.90 | 1456.95 | 1472.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-14 09:15:00 | 1447.00 | 1446.41 | 1459.31 | SL hit (close>ema200) qty=0.50 sl=1446.41 alert=retest2 |

### Cycle 70 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 1490.55 | 1465.23 | 1464.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 1536.00 | 1494.99 | 1482.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 09:15:00 | 1513.55 | 1514.80 | 1500.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-17 09:30:00 | 1515.35 | 1514.80 | 1500.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 11:15:00 | 1511.45 | 1516.56 | 1509.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 12:00:00 | 1511.45 | 1516.56 | 1509.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 1510.00 | 1515.25 | 1509.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 1485.40 | 1515.25 | 1509.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1484.50 | 1509.10 | 1507.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 1481.80 | 1509.10 | 1507.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 1490.40 | 1505.36 | 1505.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 1478.60 | 1492.92 | 1498.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 13:15:00 | 1490.10 | 1489.30 | 1494.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 14:00:00 | 1490.10 | 1489.30 | 1494.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 1504.85 | 1492.41 | 1495.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 1504.85 | 1492.41 | 1495.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 1499.95 | 1493.92 | 1495.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 1491.70 | 1493.92 | 1495.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:00:00 | 1494.95 | 1494.13 | 1495.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 12:15:00 | 1500.55 | 1497.10 | 1496.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 1500.55 | 1497.10 | 1496.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 14:15:00 | 1505.15 | 1499.50 | 1498.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 14:15:00 | 1552.00 | 1560.49 | 1543.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 14:15:00 | 1552.00 | 1560.49 | 1543.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 1552.00 | 1560.49 | 1543.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 14:45:00 | 1549.80 | 1560.49 | 1543.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 1552.00 | 1558.79 | 1544.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 09:30:00 | 1533.00 | 1552.48 | 1542.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 10:15:00 | 1528.25 | 1547.64 | 1541.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:45:00 | 1530.85 | 1547.64 | 1541.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 1518.65 | 1537.11 | 1537.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 1504.60 | 1527.41 | 1532.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 15:15:00 | 1478.00 | 1470.49 | 1486.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 15:15:00 | 1478.00 | 1470.49 | 1486.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 1478.00 | 1470.49 | 1486.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 09:15:00 | 1480.70 | 1470.49 | 1486.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 1454.20 | 1467.24 | 1483.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:15:00 | 1447.55 | 1467.24 | 1483.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:30:00 | 1450.00 | 1456.31 | 1467.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:30:00 | 1450.55 | 1455.57 | 1464.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:00:00 | 1451.80 | 1455.57 | 1464.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1409.30 | 1443.39 | 1456.23 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1378.02 | 1443.39 | 1456.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 1379.21 | 1443.39 | 1456.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 1375.17 | 1429.29 | 1448.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 10:15:00 | 1377.50 | 1429.29 | 1448.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:00:00 | 1372.90 | 1429.29 | 1448.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:30:00 | 1376.00 | 1414.04 | 1439.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-04 12:15:00 | 1302.80 | 1410.58 | 1436.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 74 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1450.35 | 1432.39 | 1430.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 11:15:00 | 1470.90 | 1445.63 | 1436.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 1479.55 | 1498.59 | 1480.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-10 09:15:00 | 1479.55 | 1498.59 | 1480.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 09:15:00 | 1479.55 | 1498.59 | 1480.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:30:00 | 1480.55 | 1498.59 | 1480.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 1472.05 | 1493.28 | 1479.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 11:00:00 | 1472.05 | 1493.28 | 1479.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 11:15:00 | 1476.35 | 1489.90 | 1479.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 12:30:00 | 1479.85 | 1487.63 | 1479.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 13:15:00 | 1479.10 | 1487.63 | 1479.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 15:00:00 | 1485.20 | 1485.25 | 1479.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:45:00 | 1482.45 | 1484.71 | 1480.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 1476.90 | 1483.09 | 1480.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:00:00 | 1476.90 | 1483.09 | 1480.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 1480.40 | 1482.55 | 1480.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-12 10:15:00 | 1477.00 | 1479.01 | 1479.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 10:15:00 | 1477.00 | 1479.01 | 1479.03 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 14:15:00 | 1484.75 | 1479.90 | 1479.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 09:15:00 | 1515.35 | 1486.65 | 1482.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 13:15:00 | 1495.40 | 1497.33 | 1489.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 14:00:00 | 1495.40 | 1497.33 | 1489.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 1485.85 | 1495.03 | 1489.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 15:00:00 | 1485.85 | 1495.03 | 1489.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 15:15:00 | 1490.00 | 1494.03 | 1489.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 09:15:00 | 1493.70 | 1494.03 | 1489.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 1483.25 | 1491.87 | 1488.91 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 12:15:00 | 1483.10 | 1486.54 | 1486.90 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 1502.70 | 1489.62 | 1488.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 12:15:00 | 1510.00 | 1496.63 | 1491.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 11:15:00 | 1593.00 | 1598.34 | 1582.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 12:15:00 | 1583.70 | 1598.34 | 1582.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 1585.85 | 1595.85 | 1582.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 1595.55 | 1588.72 | 1582.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:45:00 | 1593.35 | 1592.63 | 1584.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 15:15:00 | 1595.00 | 1600.18 | 1592.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 11:30:00 | 1596.35 | 1593.71 | 1591.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 12:15:00 | 1575.35 | 1590.04 | 1590.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 1575.35 | 1590.04 | 1590.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 13:15:00 | 1564.75 | 1584.98 | 1587.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 09:15:00 | 1605.70 | 1582.74 | 1585.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 09:15:00 | 1605.70 | 1582.74 | 1585.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1605.70 | 1582.74 | 1585.47 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 1628.00 | 1591.79 | 1589.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 11:15:00 | 1644.05 | 1602.24 | 1594.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 14:15:00 | 1659.75 | 1664.04 | 1640.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 15:00:00 | 1659.75 | 1664.04 | 1640.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 1641.85 | 1662.11 | 1649.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:45:00 | 1637.95 | 1662.11 | 1649.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 1650.05 | 1659.70 | 1649.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:45:00 | 1662.35 | 1660.84 | 1650.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 1710.35 | 1658.87 | 1650.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 15:15:00 | 1695.00 | 1705.54 | 1706.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 15:15:00 | 1695.00 | 1705.54 | 1706.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 09:15:00 | 1677.50 | 1699.93 | 1703.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 15:15:00 | 1693.00 | 1685.01 | 1692.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 15:15:00 | 1693.00 | 1685.01 | 1692.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 1693.00 | 1685.01 | 1692.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 1712.00 | 1685.01 | 1692.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 1717.60 | 1691.53 | 1694.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 11:00:00 | 1705.00 | 1694.22 | 1695.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 11:30:00 | 1698.60 | 1694.59 | 1695.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 13:15:00 | 1710.25 | 1699.05 | 1697.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 13:15:00 | 1710.25 | 1699.05 | 1697.56 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 1695.75 | 1698.33 | 1698.54 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 1701.50 | 1698.96 | 1698.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 09:15:00 | 1716.70 | 1702.83 | 1700.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 12:15:00 | 1705.20 | 1705.83 | 1702.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 12:15:00 | 1705.20 | 1705.83 | 1702.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 1705.20 | 1705.83 | 1702.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:00:00 | 1705.20 | 1705.83 | 1702.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 1708.00 | 1706.27 | 1703.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 14:15:00 | 1710.05 | 1706.27 | 1703.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 15:15:00 | 1720.00 | 1706.70 | 1703.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 10:30:00 | 1736.75 | 1722.25 | 1711.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-12 14:15:00 | 1881.06 | 1793.82 | 1752.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-07-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 15:15:00 | 1819.50 | 1837.14 | 1838.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 1790.55 | 1822.62 | 1830.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 15:15:00 | 1800.00 | 1797.72 | 1812.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 09:15:00 | 1824.30 | 1797.72 | 1812.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1836.75 | 1805.53 | 1814.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 1836.75 | 1805.53 | 1814.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1832.05 | 1810.83 | 1815.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 11:00:00 | 1832.05 | 1810.83 | 1815.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 1853.50 | 1822.92 | 1820.78 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 10:15:00 | 1778.60 | 1818.50 | 1821.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 13:15:00 | 1774.30 | 1797.13 | 1809.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 10:15:00 | 1798.45 | 1792.11 | 1802.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 10:15:00 | 1798.45 | 1792.11 | 1802.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 1798.45 | 1792.11 | 1802.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:30:00 | 1799.50 | 1792.11 | 1802.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 11:15:00 | 1803.10 | 1794.31 | 1802.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:00:00 | 1803.10 | 1794.31 | 1802.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 1795.00 | 1794.45 | 1801.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-26 15:00:00 | 1786.05 | 1794.46 | 1800.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 12:00:00 | 1776.35 | 1791.74 | 1797.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 15:15:00 | 1806.05 | 1794.12 | 1796.56 | SL hit (close>static) qty=1.00 sl=1804.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 11:15:00 | 1825.00 | 1802.38 | 1799.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 13:15:00 | 1835.85 | 1812.73 | 1805.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 14:15:00 | 1834.10 | 1850.50 | 1834.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-31 14:15:00 | 1834.10 | 1850.50 | 1834.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 1834.10 | 1850.50 | 1834.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 1834.10 | 1850.50 | 1834.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 1849.00 | 1850.20 | 1835.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 1890.10 | 1850.20 | 1835.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 11:15:00 | 1852.15 | 1852.30 | 1839.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 13:15:00 | 1850.05 | 1849.64 | 1840.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 14:00:00 | 1850.00 | 1849.71 | 1841.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 1837.50 | 1847.27 | 1840.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 15:00:00 | 1837.50 | 1847.27 | 1840.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1839.00 | 1845.61 | 1840.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 1846.00 | 1845.61 | 1840.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1869.70 | 1850.43 | 1843.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-02 14:15:00 | 1823.60 | 1847.60 | 1845.29 | SL hit (close<static) qty=1.00 sl=1830.10 alert=retest2 |

### Cycle 89 — SELL (started 2024-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 15:15:00 | 1818.65 | 1841.81 | 1842.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1764.55 | 1826.35 | 1835.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 1762.40 | 1757.43 | 1788.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 1762.40 | 1757.43 | 1788.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1762.40 | 1757.43 | 1788.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:45:00 | 1744.65 | 1754.29 | 1783.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:15:00 | 1745.00 | 1730.06 | 1733.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 13:15:00 | 1752.75 | 1737.34 | 1736.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-08-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 13:15:00 | 1752.75 | 1737.34 | 1736.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 14:15:00 | 1758.00 | 1741.47 | 1738.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 10:15:00 | 1780.40 | 1791.21 | 1773.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 11:00:00 | 1780.40 | 1791.21 | 1773.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 1766.15 | 1786.20 | 1773.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:00:00 | 1766.15 | 1786.20 | 1773.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1758.60 | 1780.68 | 1771.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 1758.60 | 1780.68 | 1771.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 1751.15 | 1766.79 | 1766.94 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 14:15:00 | 1785.70 | 1768.81 | 1767.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 1806.30 | 1779.22 | 1772.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 13:15:00 | 1829.15 | 1844.58 | 1828.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 13:15:00 | 1829.15 | 1844.58 | 1828.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 13:15:00 | 1829.15 | 1844.58 | 1828.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 14:00:00 | 1829.15 | 1844.58 | 1828.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 1826.75 | 1841.02 | 1827.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 15:00:00 | 1826.75 | 1841.02 | 1827.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 15:15:00 | 1833.00 | 1839.41 | 1828.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 1843.95 | 1839.41 | 1828.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:45:00 | 1841.90 | 1837.56 | 1828.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 10:45:00 | 1835.65 | 1837.33 | 1829.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 11:45:00 | 1837.00 | 1838.50 | 1830.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 1832.05 | 1836.81 | 1831.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:00:00 | 1832.05 | 1836.81 | 1831.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 1829.00 | 1835.25 | 1830.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:45:00 | 1828.50 | 1835.25 | 1830.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 1835.00 | 1835.20 | 1831.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:30:00 | 1836.90 | 1835.06 | 1831.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:30:00 | 1843.55 | 1839.21 | 1833.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 10:15:00 | 1828.05 | 1832.50 | 1832.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 1828.05 | 1832.50 | 1832.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 11:15:00 | 1824.65 | 1830.93 | 1831.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 15:15:00 | 1834.45 | 1825.88 | 1828.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 15:15:00 | 1834.45 | 1825.88 | 1828.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 1834.45 | 1825.88 | 1828.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 1852.00 | 1825.88 | 1828.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 1855.70 | 1831.85 | 1831.06 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 13:15:00 | 1827.95 | 1830.36 | 1830.59 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 1850.55 | 1834.40 | 1832.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 12:15:00 | 1861.30 | 1843.23 | 1837.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 12:15:00 | 1864.40 | 1864.52 | 1852.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 13:00:00 | 1864.40 | 1864.52 | 1852.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 1853.00 | 1862.21 | 1852.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:00:00 | 1853.00 | 1862.21 | 1852.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 1846.30 | 1859.03 | 1852.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 15:00:00 | 1846.30 | 1859.03 | 1852.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 15:15:00 | 1844.50 | 1856.12 | 1851.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 09:15:00 | 1852.50 | 1856.12 | 1851.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1864.35 | 1857.44 | 1852.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 1857.85 | 1857.44 | 1852.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1852.60 | 1856.47 | 1852.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:45:00 | 1852.05 | 1856.47 | 1852.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 1849.40 | 1855.06 | 1852.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 13:00:00 | 1849.40 | 1855.06 | 1852.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 13:15:00 | 1846.95 | 1853.43 | 1852.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:00:00 | 1846.95 | 1853.43 | 1852.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 09:15:00 | 1820.00 | 1848.44 | 1850.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-30 14:15:00 | 1811.05 | 1828.08 | 1838.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 14:15:00 | 1770.45 | 1769.83 | 1788.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-03 15:00:00 | 1770.45 | 1769.83 | 1788.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 1729.00 | 1731.17 | 1746.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 09:30:00 | 1737.50 | 1731.17 | 1746.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1727.55 | 1722.03 | 1730.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:00:00 | 1727.55 | 1722.03 | 1730.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1749.70 | 1727.56 | 1732.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 1749.70 | 1727.56 | 1732.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1756.00 | 1733.25 | 1734.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 1754.25 | 1733.25 | 1734.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 1755.85 | 1737.77 | 1736.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 1775.40 | 1755.36 | 1746.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 11:15:00 | 1820.60 | 1826.59 | 1806.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 12:00:00 | 1820.60 | 1826.59 | 1806.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1791.30 | 1818.82 | 1810.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 10:00:00 | 1791.30 | 1818.82 | 1810.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 1784.85 | 1812.03 | 1807.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 1784.85 | 1812.03 | 1807.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 11:15:00 | 1774.05 | 1804.43 | 1804.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 14:15:00 | 1759.30 | 1788.11 | 1796.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 14:15:00 | 1666.90 | 1665.24 | 1690.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 15:00:00 | 1666.90 | 1665.24 | 1690.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 09:15:00 | 1685.35 | 1669.22 | 1688.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:00:00 | 1685.35 | 1669.22 | 1688.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 1683.60 | 1674.36 | 1684.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:00:00 | 1683.60 | 1674.36 | 1684.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 1692.95 | 1678.08 | 1685.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 15:00:00 | 1692.95 | 1678.08 | 1685.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 15:15:00 | 1692.00 | 1680.86 | 1686.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:15:00 | 1650.00 | 1680.86 | 1686.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 1669.90 | 1656.23 | 1666.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 1664.35 | 1656.23 | 1666.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 1674.00 | 1659.78 | 1667.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 1674.00 | 1659.78 | 1667.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1675.95 | 1663.02 | 1668.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 1675.95 | 1663.02 | 1668.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1657.75 | 1662.42 | 1667.13 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 1676.00 | 1668.48 | 1668.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 13:15:00 | 1679.95 | 1672.08 | 1669.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 10:15:00 | 1675.50 | 1675.63 | 1672.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 10:15:00 | 1675.50 | 1675.63 | 1672.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 1675.50 | 1675.63 | 1672.57 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 1665.05 | 1670.77 | 1671.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 1635.45 | 1663.26 | 1667.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 1643.90 | 1639.96 | 1650.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 1643.90 | 1639.96 | 1650.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 1643.90 | 1639.96 | 1650.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:15:00 | 1634.45 | 1639.96 | 1650.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:45:00 | 1630.00 | 1638.94 | 1649.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 13:15:00 | 1655.45 | 1644.22 | 1649.35 | SL hit (close>static) qty=1.00 sl=1654.60 alert=retest2 |

### Cycle 102 — BUY (started 2024-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 15:15:00 | 1673.00 | 1654.28 | 1653.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-03 09:15:00 | 1695.00 | 1662.42 | 1657.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 14:15:00 | 1680.00 | 1680.48 | 1669.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-03 15:00:00 | 1680.00 | 1680.48 | 1669.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 15:15:00 | 1672.00 | 1678.78 | 1669.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:15:00 | 1655.15 | 1678.78 | 1669.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 1671.95 | 1677.41 | 1670.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 1665.05 | 1677.41 | 1670.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 1684.35 | 1678.80 | 1671.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 11:45:00 | 1690.00 | 1680.32 | 1672.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 13:15:00 | 1691.00 | 1681.45 | 1673.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 15:00:00 | 1691.65 | 1685.08 | 1676.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 10:15:00 | 1659.80 | 1682.40 | 1677.97 | SL hit (close<static) qty=1.00 sl=1667.40 alert=retest2 |

### Cycle 103 — SELL (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 09:15:00 | 1769.40 | 1779.88 | 1780.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 1737.30 | 1768.20 | 1774.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1723.80 | 1708.08 | 1727.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 1723.80 | 1708.08 | 1727.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1723.80 | 1708.08 | 1727.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 1723.80 | 1708.08 | 1727.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1732.00 | 1712.87 | 1727.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:15:00 | 1737.90 | 1712.87 | 1727.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1719.20 | 1714.13 | 1727.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:45:00 | 1730.00 | 1714.13 | 1727.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 1695.00 | 1710.31 | 1724.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:30:00 | 1708.10 | 1710.31 | 1724.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1375.05 | 1360.03 | 1388.48 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2024-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 13:15:00 | 1404.65 | 1390.95 | 1390.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 1423.80 | 1397.78 | 1393.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1402.30 | 1403.13 | 1397.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1402.30 | 1403.13 | 1397.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1402.30 | 1403.13 | 1397.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 13:00:00 | 1433.30 | 1414.94 | 1404.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 12:15:00 | 1380.55 | 1402.16 | 1402.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 12:15:00 | 1380.55 | 1402.16 | 1402.86 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 1429.55 | 1402.85 | 1402.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 1465.90 | 1417.28 | 1409.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 1455.80 | 1457.53 | 1438.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 1455.80 | 1457.53 | 1438.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 1444.55 | 1455.68 | 1442.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 1444.55 | 1455.68 | 1442.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 1440.00 | 1452.55 | 1442.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 1448.90 | 1452.55 | 1442.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1448.95 | 1451.83 | 1443.08 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-11-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 15:15:00 | 1422.00 | 1437.08 | 1438.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1412.40 | 1432.14 | 1436.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 11:15:00 | 1414.85 | 1412.41 | 1420.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-12 12:00:00 | 1414.85 | 1412.41 | 1420.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 1413.60 | 1412.65 | 1420.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:30:00 | 1412.40 | 1412.65 | 1420.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 1383.15 | 1400.66 | 1411.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 13:00:00 | 1371.15 | 1390.01 | 1403.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 14:00:00 | 1372.40 | 1386.49 | 1400.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 14:30:00 | 1367.90 | 1379.64 | 1396.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 15:15:00 | 1302.59 | 1316.85 | 1334.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 15:15:00 | 1303.78 | 1316.85 | 1334.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-21 09:15:00 | 1299.51 | 1311.08 | 1330.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 09:15:00 | 1304.60 | 1299.08 | 1313.26 | SL hit (close>ema200) qty=0.50 sl=1299.08 alert=retest2 |

### Cycle 108 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 1357.25 | 1314.04 | 1313.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 1365.95 | 1324.42 | 1318.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 12:15:00 | 1405.75 | 1408.01 | 1394.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 13:00:00 | 1405.75 | 1408.01 | 1394.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1371.55 | 1401.73 | 1396.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 1371.55 | 1401.73 | 1396.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1372.40 | 1395.86 | 1394.13 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 11:15:00 | 1368.30 | 1390.35 | 1391.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 12:15:00 | 1362.75 | 1384.83 | 1389.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 15:15:00 | 1380.00 | 1378.91 | 1384.92 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-02 09:15:00 | 1356.00 | 1378.91 | 1384.92 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1422.50 | 1367.25 | 1372.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-03 09:15:00 | 1422.50 | 1367.25 | 1372.04 | SL hit (close>ema400) qty=1.00 sl=1372.04 alert=retest1 |

### Cycle 110 — BUY (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 10:15:00 | 1430.00 | 1379.80 | 1377.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 12:15:00 | 1436.40 | 1398.83 | 1386.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 13:15:00 | 1476.70 | 1484.08 | 1466.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-06 13:45:00 | 1474.50 | 1484.08 | 1466.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1534.85 | 1541.36 | 1532.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:45:00 | 1537.25 | 1541.36 | 1532.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 1546.40 | 1542.36 | 1533.54 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 09:15:00 | 1526.55 | 1533.03 | 1533.23 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 10:15:00 | 1543.30 | 1535.09 | 1534.14 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 1524.95 | 1534.38 | 1534.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 1520.00 | 1531.51 | 1533.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 09:15:00 | 1526.55 | 1523.22 | 1528.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 1526.55 | 1523.22 | 1528.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1526.55 | 1523.22 | 1528.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:45:00 | 1528.45 | 1523.22 | 1528.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 1522.55 | 1523.08 | 1527.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 12:30:00 | 1516.55 | 1521.92 | 1526.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:45:00 | 1516.75 | 1520.97 | 1525.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 1496.30 | 1521.38 | 1524.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 1513.30 | 1507.41 | 1514.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 1440.72 | 1472.05 | 1491.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 15:15:00 | 1440.91 | 1472.05 | 1491.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 1437.63 | 1466.67 | 1487.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 12:15:00 | 1421.48 | 1446.27 | 1472.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 1438.65 | 1433.70 | 1457.03 | SL hit (close>ema200) qty=0.50 sl=1433.70 alert=retest2 |

### Cycle 114 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 1469.45 | 1449.29 | 1447.67 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 1441.85 | 1449.40 | 1450.33 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1471.75 | 1453.87 | 1452.28 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 1429.40 | 1451.24 | 1451.50 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 10:15:00 | 1453.85 | 1451.76 | 1451.71 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 11:15:00 | 1448.95 | 1451.20 | 1451.46 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 1461.15 | 1453.46 | 1452.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 14:15:00 | 1465.05 | 1455.78 | 1453.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 1478.80 | 1482.63 | 1472.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 1464.50 | 1479.89 | 1476.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 1464.50 | 1479.89 | 1476.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:00:00 | 1464.50 | 1479.89 | 1476.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 1470.30 | 1477.97 | 1475.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 10:30:00 | 1466.45 | 1477.97 | 1475.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 1469.10 | 1475.77 | 1475.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 13:00:00 | 1469.10 | 1475.77 | 1475.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 1452.85 | 1471.19 | 1473.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 1450.80 | 1467.11 | 1471.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1454.90 | 1453.31 | 1459.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 1454.90 | 1453.31 | 1459.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1454.90 | 1453.31 | 1459.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:30:00 | 1461.00 | 1453.31 | 1459.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1466.95 | 1456.03 | 1460.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 1466.95 | 1456.03 | 1460.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1461.75 | 1457.18 | 1460.55 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 13:15:00 | 1478.20 | 1465.03 | 1463.76 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 1442.00 | 1462.61 | 1463.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 1420.85 | 1446.57 | 1455.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 1453.05 | 1440.88 | 1448.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 1453.05 | 1440.88 | 1448.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1453.05 | 1440.88 | 1448.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 1453.05 | 1440.88 | 1448.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1434.90 | 1439.68 | 1447.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 11:15:00 | 1432.75 | 1439.68 | 1447.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 13:00:00 | 1427.05 | 1435.38 | 1444.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 09:15:00 | 1361.11 | 1421.58 | 1434.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1355.70 | 1380.91 | 1404.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 1333.15 | 1331.20 | 1360.05 | SL hit (close>ema200) qty=0.50 sl=1331.20 alert=retest2 |

### Cycle 124 — BUY (started 2025-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 13:15:00 | 1338.00 | 1329.63 | 1328.76 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 1319.40 | 1328.29 | 1328.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 1315.55 | 1323.19 | 1325.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1325.40 | 1302.87 | 1309.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1325.40 | 1302.87 | 1309.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1325.40 | 1302.87 | 1309.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1325.40 | 1302.87 | 1309.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1329.95 | 1308.29 | 1311.30 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 11:15:00 | 1349.95 | 1316.62 | 1314.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 09:15:00 | 1362.65 | 1339.63 | 1328.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 12:15:00 | 1336.35 | 1344.02 | 1333.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 12:15:00 | 1336.35 | 1344.02 | 1333.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 1336.35 | 1344.02 | 1333.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 13:00:00 | 1336.35 | 1344.02 | 1333.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 13:15:00 | 1326.00 | 1340.42 | 1332.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:00:00 | 1326.00 | 1340.42 | 1332.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 1322.80 | 1336.89 | 1331.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:30:00 | 1321.65 | 1336.89 | 1331.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 1298.80 | 1326.25 | 1327.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1260.35 | 1298.95 | 1311.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1283.35 | 1278.16 | 1292.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:00:00 | 1283.35 | 1278.16 | 1292.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 1338.50 | 1287.34 | 1292.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:45:00 | 1362.00 | 1287.34 | 1292.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 1369.55 | 1303.78 | 1299.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1466.55 | 1356.45 | 1326.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 1398.40 | 1414.39 | 1378.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 09:45:00 | 1398.10 | 1414.39 | 1378.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1416.35 | 1413.43 | 1397.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1407.10 | 1413.43 | 1397.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1392.95 | 1409.33 | 1397.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1392.95 | 1409.33 | 1397.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1412.65 | 1409.99 | 1398.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:15:00 | 1420.00 | 1411.10 | 1400.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 1444.00 | 1408.97 | 1404.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 10:00:00 | 1419.05 | 1422.67 | 1417.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 11:15:00 | 1420.00 | 1420.95 | 1416.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 1422.80 | 1421.32 | 1417.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 12:30:00 | 1429.25 | 1424.17 | 1419.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 10:30:00 | 1425.90 | 1435.71 | 1433.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 11:00:00 | 1429.05 | 1435.71 | 1433.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-11 09:15:00 | 1397.00 | 1427.68 | 1430.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 1397.00 | 1427.68 | 1430.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 1377.50 | 1414.12 | 1423.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 12:15:00 | 1359.60 | 1355.21 | 1370.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 13:00:00 | 1359.60 | 1355.21 | 1370.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 14:15:00 | 1344.05 | 1331.17 | 1346.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:45:00 | 1345.35 | 1331.17 | 1346.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 1332.10 | 1331.35 | 1345.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:15:00 | 1334.05 | 1331.35 | 1345.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 1318.35 | 1328.75 | 1342.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:15:00 | 1311.00 | 1328.75 | 1342.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 11:00:00 | 1305.95 | 1324.19 | 1339.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 11:30:00 | 1310.95 | 1322.35 | 1337.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 13:00:00 | 1312.00 | 1320.28 | 1334.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1297.30 | 1312.35 | 1326.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 1296.95 | 1312.35 | 1326.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 14:15:00 | 1329.95 | 1314.10 | 1321.18 | SL hit (close>static) qty=1.00 sl=1328.50 alert=retest2 |

### Cycle 130 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 1347.85 | 1328.06 | 1326.33 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 1314.90 | 1330.59 | 1332.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 1272.25 | 1307.81 | 1319.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 1285.90 | 1285.13 | 1298.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 15:15:00 | 1295.00 | 1285.53 | 1292.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 1295.00 | 1285.53 | 1292.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:15:00 | 1272.00 | 1285.53 | 1292.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 1269.30 | 1282.28 | 1290.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 11:45:00 | 1260.80 | 1275.58 | 1285.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 14:15:00 | 1197.76 | 1222.95 | 1247.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 1201.40 | 1191.07 | 1219.85 | SL hit (close>ema200) qty=0.50 sl=1191.07 alert=retest2 |

### Cycle 132 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1290.00 | 1231.42 | 1223.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1306.95 | 1273.04 | 1251.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 13:15:00 | 1313.30 | 1313.81 | 1294.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 13:45:00 | 1311.80 | 1313.81 | 1294.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1300.95 | 1310.35 | 1296.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 1298.85 | 1310.35 | 1296.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1291.85 | 1306.65 | 1295.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 1291.85 | 1306.65 | 1295.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 1293.90 | 1304.10 | 1295.69 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 1265.00 | 1289.95 | 1290.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 1253.50 | 1282.66 | 1287.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 1266.95 | 1263.18 | 1272.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 15:00:00 | 1266.95 | 1263.18 | 1272.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 1278.00 | 1266.15 | 1273.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 1260.45 | 1266.15 | 1273.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 1242.80 | 1261.48 | 1270.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:45:00 | 1234.30 | 1253.79 | 1265.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 15:15:00 | 1246.00 | 1242.45 | 1242.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 1246.00 | 1242.45 | 1242.16 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 09:15:00 | 1239.00 | 1241.76 | 1241.88 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 1257.60 | 1243.63 | 1241.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 1271.65 | 1249.24 | 1244.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1362.15 | 1365.31 | 1335.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 1362.15 | 1365.31 | 1335.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1340.55 | 1351.85 | 1340.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 1356.25 | 1351.85 | 1340.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 1356.10 | 1354.68 | 1343.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 09:15:00 | 1335.00 | 1356.58 | 1350.91 | SL hit (close<static) qty=1.00 sl=1337.95 alert=retest2 |

### Cycle 137 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 1333.15 | 1345.47 | 1346.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 09:15:00 | 1326.40 | 1337.61 | 1342.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 1288.00 | 1282.52 | 1298.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-02 11:00:00 | 1288.00 | 1282.52 | 1298.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 1311.80 | 1289.05 | 1296.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 1311.80 | 1289.05 | 1296.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 1313.55 | 1293.95 | 1298.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 09:15:00 | 1263.10 | 1293.95 | 1298.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 09:15:00 | 1199.94 | 1220.74 | 1252.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-04 11:15:00 | 1136.79 | 1191.44 | 1233.24 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 138 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 1111.30 | 1106.95 | 1106.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1157.20 | 1121.14 | 1113.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 13:15:00 | 1150.00 | 1150.57 | 1139.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:45:00 | 1149.00 | 1150.57 | 1139.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 1125.20 | 1144.82 | 1139.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 13:00:00 | 1142.40 | 1140.58 | 1138.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 13:30:00 | 1145.00 | 1141.70 | 1139.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 1142.90 | 1140.29 | 1138.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-28 13:15:00 | 1256.64 | 1227.21 | 1215.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 15:15:00 | 1232.50 | 1245.57 | 1246.77 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 09:15:00 | 1260.90 | 1248.64 | 1248.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 10:15:00 | 1261.80 | 1256.11 | 1252.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 11:15:00 | 1248.20 | 1254.53 | 1252.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 1248.20 | 1254.53 | 1252.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 1248.20 | 1254.53 | 1252.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:45:00 | 1249.20 | 1254.53 | 1252.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 1242.00 | 1252.03 | 1251.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 1242.00 | 1252.03 | 1251.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 1240.00 | 1249.62 | 1250.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 1237.00 | 1244.41 | 1247.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 1257.40 | 1247.01 | 1248.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 1257.40 | 1247.01 | 1248.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 1257.40 | 1247.01 | 1248.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 1257.40 | 1247.01 | 1248.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1256.10 | 1248.83 | 1249.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:15:00 | 1263.40 | 1248.83 | 1249.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — BUY (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 12:15:00 | 1259.20 | 1250.90 | 1250.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 13:15:00 | 1275.00 | 1255.72 | 1252.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 1284.00 | 1285.72 | 1272.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 14:00:00 | 1284.00 | 1285.72 | 1272.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1273.00 | 1283.17 | 1272.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:45:00 | 1272.60 | 1283.17 | 1272.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1265.10 | 1279.56 | 1271.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1250.00 | 1279.56 | 1271.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1248.60 | 1273.37 | 1269.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 1248.60 | 1273.37 | 1269.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 11:15:00 | 1254.70 | 1265.78 | 1266.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 14:15:00 | 1241.60 | 1255.37 | 1261.04 | Break + close below crossover candle low |

### Cycle 144 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1310.50 | 1264.74 | 1264.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1332.60 | 1296.50 | 1280.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 1354.00 | 1354.91 | 1334.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 1354.00 | 1354.91 | 1334.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1369.70 | 1377.25 | 1365.32 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 1344.70 | 1359.61 | 1361.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 13:15:00 | 1342.30 | 1354.18 | 1358.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 1336.90 | 1331.50 | 1341.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 11:00:00 | 1336.90 | 1331.50 | 1341.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1343.00 | 1326.69 | 1330.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 1345.10 | 1326.69 | 1330.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 1336.00 | 1328.56 | 1330.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 1332.50 | 1330.14 | 1331.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 1337.00 | 1332.48 | 1332.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 1337.00 | 1332.48 | 1332.15 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 1326.00 | 1330.79 | 1331.42 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1337.90 | 1332.21 | 1332.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 1348.40 | 1337.21 | 1334.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 12:15:00 | 1332.70 | 1337.07 | 1335.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 12:15:00 | 1332.70 | 1337.07 | 1335.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 1332.70 | 1337.07 | 1335.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 1332.70 | 1337.07 | 1335.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 1325.50 | 1334.76 | 1334.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 1325.50 | 1334.76 | 1334.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 14:15:00 | 1330.80 | 1333.97 | 1334.11 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 09:15:00 | 1340.50 | 1333.79 | 1333.43 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 12:15:00 | 1332.00 | 1336.09 | 1336.57 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 1340.20 | 1336.91 | 1336.90 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 1334.90 | 1336.54 | 1336.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 1329.00 | 1335.03 | 1336.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 13:15:00 | 1331.50 | 1331.45 | 1333.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-02 13:15:00 | 1331.50 | 1331.45 | 1333.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 1331.50 | 1331.45 | 1333.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 14:00:00 | 1331.50 | 1331.45 | 1333.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 1332.80 | 1331.72 | 1333.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 15:00:00 | 1332.80 | 1331.72 | 1333.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 1341.40 | 1333.73 | 1334.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:30:00 | 1330.10 | 1333.19 | 1333.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 14:15:00 | 1328.10 | 1332.93 | 1333.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 1356.20 | 1325.19 | 1322.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 1356.20 | 1325.19 | 1322.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 15:15:00 | 1357.90 | 1346.78 | 1336.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 12:15:00 | 1359.50 | 1361.61 | 1347.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 12:45:00 | 1362.00 | 1361.61 | 1347.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1375.50 | 1376.11 | 1366.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 1367.80 | 1376.11 | 1366.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1360.40 | 1374.49 | 1368.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1360.40 | 1374.49 | 1368.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1363.00 | 1372.20 | 1368.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:15:00 | 1359.00 | 1372.20 | 1368.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 15:15:00 | 1359.00 | 1369.56 | 1367.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:15:00 | 1343.50 | 1369.56 | 1367.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 1346.90 | 1365.03 | 1365.49 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 1389.60 | 1366.48 | 1363.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 1393.50 | 1371.88 | 1366.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 12:15:00 | 1396.60 | 1398.91 | 1386.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-17 12:45:00 | 1393.90 | 1398.91 | 1386.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1409.10 | 1400.56 | 1390.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 12:45:00 | 1413.90 | 1404.73 | 1395.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 14:30:00 | 1418.20 | 1410.32 | 1399.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 1386.80 | 1407.00 | 1400.10 | SL hit (close<static) qty=1.00 sl=1390.80 alert=retest2 |

### Cycle 157 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 1390.50 | 1395.61 | 1396.07 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2025-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 09:15:00 | 1401.90 | 1395.76 | 1395.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 10:15:00 | 1410.10 | 1398.63 | 1396.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 13:15:00 | 1396.10 | 1400.10 | 1397.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 13:15:00 | 1396.10 | 1400.10 | 1397.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1396.10 | 1400.10 | 1397.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:00:00 | 1396.10 | 1400.10 | 1397.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1389.30 | 1397.94 | 1397.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 1389.30 | 1397.94 | 1397.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 1391.70 | 1396.69 | 1396.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 1333.80 | 1396.69 | 1396.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 09:15:00 | 1324.40 | 1382.23 | 1390.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 11:15:00 | 1312.50 | 1357.62 | 1376.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 1287.40 | 1281.32 | 1298.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-27 14:45:00 | 1268.80 | 1279.74 | 1291.76 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-30 10:15:00 | 1273.30 | 1277.86 | 1288.69 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1254.30 | 1249.83 | 1260.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:30:00 | 1256.70 | 1249.83 | 1260.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 1251.60 | 1251.59 | 1258.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:30:00 | 1258.60 | 1251.59 | 1258.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1254.40 | 1249.40 | 1255.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:45:00 | 1256.50 | 1249.40 | 1255.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1259.70 | 1251.46 | 1255.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1259.70 | 1251.46 | 1255.51 | SL hit (close>ema400) qty=1.00 sl=1255.51 alert=retest1 |

### Cycle 160 — BUY (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 14:15:00 | 1260.20 | 1257.86 | 1257.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 15:15:00 | 1263.90 | 1259.07 | 1258.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 12:15:00 | 1260.00 | 1260.46 | 1259.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 12:15:00 | 1260.00 | 1260.46 | 1259.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1260.00 | 1260.46 | 1259.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 13:45:00 | 1262.80 | 1260.73 | 1259.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 14:15:00 | 1262.80 | 1260.73 | 1259.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 09:45:00 | 1263.30 | 1262.57 | 1260.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 13:15:00 | 1262.80 | 1267.62 | 1266.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 13:15:00 | 1261.60 | 1266.42 | 1266.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 13:15:00 | 1261.60 | 1266.42 | 1266.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 15:15:00 | 1258.00 | 1263.88 | 1265.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 15:15:00 | 1256.00 | 1254.34 | 1258.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 09:15:00 | 1260.30 | 1254.34 | 1258.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1264.60 | 1256.39 | 1259.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 1264.60 | 1256.39 | 1259.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1268.60 | 1258.83 | 1259.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 1267.20 | 1258.83 | 1259.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 1272.00 | 1261.47 | 1261.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 1280.30 | 1268.45 | 1265.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 1291.60 | 1292.12 | 1284.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 14:00:00 | 1291.60 | 1292.12 | 1284.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1286.40 | 1290.12 | 1284.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 1294.50 | 1290.12 | 1284.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1288.30 | 1289.76 | 1284.81 | EMA400 retest candle locked (from upside) |

### Cycle 163 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 1271.00 | 1280.61 | 1281.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 1260.60 | 1272.06 | 1276.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 15:15:00 | 1266.80 | 1265.79 | 1270.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:15:00 | 1267.70 | 1265.79 | 1270.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1259.80 | 1264.60 | 1269.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:00:00 | 1254.00 | 1261.84 | 1267.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1255.70 | 1260.55 | 1266.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 1278.90 | 1264.68 | 1265.90 | SL hit (close>static) qty=1.00 sl=1273.70 alert=retest2 |

### Cycle 164 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 1272.80 | 1267.86 | 1267.23 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1259.60 | 1267.31 | 1267.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 1253.00 | 1261.14 | 1264.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 11:15:00 | 1203.20 | 1201.04 | 1216.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 12:00:00 | 1203.20 | 1201.04 | 1216.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1220.00 | 1206.10 | 1214.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:45:00 | 1221.50 | 1206.10 | 1214.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 1218.00 | 1208.48 | 1215.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1207.90 | 1208.48 | 1215.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1215.30 | 1209.84 | 1215.10 | EMA400 retest candle locked (from downside) |

### Cycle 166 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 1273.50 | 1225.48 | 1220.67 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 10:15:00 | 1213.00 | 1227.17 | 1227.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 1205.60 | 1220.74 | 1224.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1216.90 | 1214.59 | 1219.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 14:00:00 | 1216.90 | 1214.59 | 1219.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1218.70 | 1215.41 | 1218.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 1218.00 | 1215.41 | 1218.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1218.00 | 1215.93 | 1218.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1213.90 | 1215.93 | 1218.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1205.50 | 1213.85 | 1217.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 1196.90 | 1207.01 | 1211.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:30:00 | 1200.10 | 1206.17 | 1210.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 1198.20 | 1205.08 | 1208.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:00:00 | 1200.20 | 1204.78 | 1206.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1191.00 | 1199.25 | 1202.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:30:00 | 1197.50 | 1199.25 | 1202.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1197.80 | 1197.03 | 1200.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:30:00 | 1203.00 | 1197.03 | 1200.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1200.80 | 1197.79 | 1200.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 1200.80 | 1197.79 | 1200.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 1202.20 | 1198.67 | 1200.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:45:00 | 1205.00 | 1198.67 | 1200.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 1202.70 | 1199.47 | 1201.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:00:00 | 1202.70 | 1199.47 | 1201.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1198.50 | 1199.28 | 1200.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:30:00 | 1198.60 | 1199.28 | 1200.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 1214.90 | 1202.25 | 1201.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 1214.90 | 1202.25 | 1201.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 1224.50 | 1212.09 | 1207.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 1209.40 | 1214.16 | 1209.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 1209.40 | 1214.16 | 1209.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1209.40 | 1214.16 | 1209.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 1209.40 | 1214.16 | 1209.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 1219.30 | 1215.18 | 1210.35 | EMA400 retest candle locked (from upside) |

### Cycle 169 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 1202.00 | 1212.83 | 1212.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 09:15:00 | 1195.30 | 1207.90 | 1210.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 11:15:00 | 1212.00 | 1208.23 | 1210.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 11:15:00 | 1212.00 | 1208.23 | 1210.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 1212.00 | 1208.23 | 1210.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:45:00 | 1211.60 | 1208.23 | 1210.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 12:15:00 | 1201.70 | 1206.92 | 1209.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 13:15:00 | 1200.60 | 1206.92 | 1209.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 09:15:00 | 1197.70 | 1203.67 | 1207.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 11:15:00 | 1218.00 | 1207.24 | 1207.87 | SL hit (close>static) qty=1.00 sl=1212.60 alert=retest2 |

### Cycle 170 — BUY (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 12:15:00 | 1217.30 | 1209.25 | 1208.73 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 1199.80 | 1209.76 | 1210.87 | EMA200 below EMA400 |

### Cycle 172 — BUY (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 10:15:00 | 1227.80 | 1211.46 | 1210.07 | EMA200 above EMA400 |

### Cycle 173 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1204.80 | 1210.10 | 1210.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 1200.60 | 1208.08 | 1209.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 12:15:00 | 1191.60 | 1190.67 | 1196.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 12:45:00 | 1192.80 | 1190.67 | 1196.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1194.40 | 1191.42 | 1196.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:45:00 | 1190.90 | 1191.42 | 1196.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1202.90 | 1193.27 | 1195.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1204.50 | 1193.27 | 1195.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1206.50 | 1195.92 | 1196.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 1206.50 | 1195.92 | 1196.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 1215.80 | 1199.90 | 1198.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 1222.70 | 1204.46 | 1200.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 1207.90 | 1214.32 | 1209.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 1207.90 | 1214.32 | 1209.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1207.90 | 1214.32 | 1209.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1207.90 | 1214.32 | 1209.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1207.30 | 1212.91 | 1209.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 1207.30 | 1212.91 | 1209.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1207.00 | 1211.73 | 1209.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1211.90 | 1211.73 | 1209.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:00:00 | 1211.20 | 1211.84 | 1209.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 1251.90 | 1259.71 | 1260.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 1251.90 | 1259.71 | 1260.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 15:15:00 | 1250.00 | 1255.93 | 1258.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 14:15:00 | 1251.00 | 1250.99 | 1254.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 14:15:00 | 1251.00 | 1250.99 | 1254.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 1251.00 | 1250.99 | 1254.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:30:00 | 1254.80 | 1250.99 | 1254.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1253.00 | 1251.39 | 1253.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1249.30 | 1251.39 | 1253.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1251.60 | 1251.43 | 1253.78 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 1273.60 | 1253.96 | 1253.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 11:15:00 | 1276.40 | 1261.06 | 1256.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 1294.80 | 1299.46 | 1288.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 10:30:00 | 1295.70 | 1299.46 | 1288.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1286.90 | 1296.95 | 1288.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 1286.90 | 1296.95 | 1288.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1283.40 | 1294.24 | 1287.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 1283.40 | 1294.24 | 1287.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1284.80 | 1291.26 | 1287.49 | EMA400 retest candle locked (from upside) |

### Cycle 177 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1262.70 | 1284.72 | 1285.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 1248.30 | 1257.19 | 1265.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1265.70 | 1248.21 | 1253.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1265.70 | 1248.21 | 1253.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1265.70 | 1248.21 | 1253.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:45:00 | 1270.90 | 1248.21 | 1253.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1267.60 | 1252.08 | 1255.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:45:00 | 1271.00 | 1252.08 | 1255.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 1254.40 | 1253.49 | 1255.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:45:00 | 1253.30 | 1253.49 | 1255.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1255.60 | 1253.91 | 1255.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:15:00 | 1260.70 | 1253.91 | 1255.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 1260.70 | 1255.27 | 1255.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1251.00 | 1255.27 | 1255.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 12:15:00 | 1188.45 | 1201.21 | 1216.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-30 13:15:00 | 1125.90 | 1186.22 | 1208.54 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 14:15:00 | 1172.00 | 1159.51 | 1158.81 | EMA200 above EMA400 |

### Cycle 179 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 1152.10 | 1159.37 | 1159.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 11:15:00 | 1146.80 | 1155.90 | 1158.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 14:15:00 | 1161.60 | 1155.82 | 1157.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 14:15:00 | 1161.60 | 1155.82 | 1157.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1161.60 | 1155.82 | 1157.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1161.60 | 1155.82 | 1157.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1161.90 | 1157.04 | 1157.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1164.10 | 1157.04 | 1157.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 09:15:00 | 1166.30 | 1158.89 | 1158.64 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 1154.60 | 1158.92 | 1159.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 1145.40 | 1155.12 | 1157.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 13:15:00 | 1153.50 | 1150.07 | 1153.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 13:15:00 | 1153.50 | 1150.07 | 1153.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1153.50 | 1150.07 | 1153.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 1153.50 | 1150.07 | 1153.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1154.60 | 1150.97 | 1153.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:30:00 | 1155.40 | 1150.97 | 1153.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 15:15:00 | 1155.00 | 1151.78 | 1153.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:15:00 | 1160.00 | 1151.78 | 1153.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 1153.70 | 1152.16 | 1153.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:15:00 | 1145.00 | 1151.29 | 1153.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:15:00 | 1144.80 | 1150.11 | 1152.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 13:00:00 | 1142.60 | 1148.61 | 1151.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 1164.90 | 1154.54 | 1153.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 1164.90 | 1154.54 | 1153.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 1166.50 | 1160.28 | 1157.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 09:15:00 | 1159.10 | 1160.49 | 1157.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 10:15:00 | 1150.50 | 1160.49 | 1157.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1153.20 | 1159.03 | 1157.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:30:00 | 1153.60 | 1159.03 | 1157.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1154.80 | 1158.19 | 1157.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 1150.90 | 1158.19 | 1157.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1156.00 | 1158.22 | 1157.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 1153.60 | 1158.22 | 1157.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1155.80 | 1157.73 | 1157.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 1155.40 | 1157.73 | 1157.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 1158.40 | 1157.87 | 1157.47 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 12:15:00 | 1145.10 | 1155.31 | 1156.34 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 1168.50 | 1153.62 | 1152.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 15:15:00 | 1179.00 | 1171.20 | 1165.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 1190.90 | 1198.98 | 1188.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 12:00:00 | 1190.90 | 1198.98 | 1188.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1191.40 | 1197.46 | 1188.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:30:00 | 1190.40 | 1197.46 | 1188.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1186.40 | 1195.25 | 1188.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:00:00 | 1186.40 | 1195.25 | 1188.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1191.40 | 1194.48 | 1188.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 1181.20 | 1194.48 | 1188.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1192.00 | 1193.98 | 1188.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1186.00 | 1193.98 | 1188.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1182.10 | 1191.61 | 1188.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 14:15:00 | 1197.30 | 1191.50 | 1189.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 1170.00 | 1187.65 | 1188.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1170.00 | 1187.65 | 1188.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 11:15:00 | 1168.10 | 1173.01 | 1178.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 1173.10 | 1172.06 | 1176.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:00:00 | 1173.10 | 1172.06 | 1176.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1168.90 | 1162.29 | 1167.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:45:00 | 1160.70 | 1162.30 | 1166.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1172.20 | 1158.46 | 1156.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 1172.20 | 1158.46 | 1156.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 1192.00 | 1167.12 | 1161.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 15:15:00 | 1234.10 | 1237.39 | 1224.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 09:15:00 | 1239.10 | 1237.39 | 1224.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1223.50 | 1234.21 | 1227.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 1223.50 | 1234.21 | 1227.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1225.10 | 1232.39 | 1226.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 1222.80 | 1232.39 | 1226.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1219.30 | 1228.88 | 1226.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1221.00 | 1228.88 | 1226.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 1221.30 | 1227.36 | 1225.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:15:00 | 1216.20 | 1227.36 | 1225.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-11-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 11:15:00 | 1220.00 | 1223.75 | 1224.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1196.30 | 1215.26 | 1219.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 11:15:00 | 1212.00 | 1211.51 | 1217.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 11:15:00 | 1212.00 | 1211.51 | 1217.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 1212.00 | 1211.51 | 1217.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 1212.00 | 1211.51 | 1217.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1215.00 | 1205.97 | 1211.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 1215.00 | 1205.97 | 1211.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 10:15:00 | 1206.20 | 1206.02 | 1211.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:45:00 | 1210.30 | 1206.02 | 1211.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 11:15:00 | 1214.30 | 1207.67 | 1211.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 12:00:00 | 1214.30 | 1207.67 | 1211.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1208.00 | 1207.74 | 1211.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 15:00:00 | 1204.20 | 1207.44 | 1210.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 1204.20 | 1207.90 | 1209.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 11:45:00 | 1194.20 | 1187.33 | 1192.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1205.60 | 1191.02 | 1189.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 1205.60 | 1191.02 | 1189.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 1214.30 | 1195.67 | 1191.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 1249.00 | 1249.27 | 1234.61 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-03 11:15:00 | 1263.50 | 1254.57 | 1245.05 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1268.00 | 1259.38 | 1251.57 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1264.40 | 1267.99 | 1262.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 1262.90 | 1267.99 | 1262.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1265.20 | 1267.43 | 1262.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 1267.70 | 1267.43 | 1262.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1259.50 | 1265.84 | 1262.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-05 11:15:00 | 1259.50 | 1265.84 | 1262.12 | SL hit (close<ema400) qty=1.00 sl=1262.12 alert=retest1 |

### Cycle 189 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1245.60 | 1259.24 | 1260.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1234.60 | 1254.31 | 1258.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 1211.70 | 1205.48 | 1215.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 1211.70 | 1205.48 | 1215.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1216.10 | 1207.60 | 1215.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 1216.10 | 1207.60 | 1215.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1228.70 | 1211.82 | 1216.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 1228.70 | 1211.82 | 1216.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1228.90 | 1215.24 | 1217.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 1230.40 | 1215.24 | 1217.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 1231.00 | 1220.72 | 1219.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 1237.00 | 1227.34 | 1223.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1221.90 | 1230.78 | 1227.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1221.90 | 1230.78 | 1227.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1221.90 | 1230.78 | 1227.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 1221.90 | 1230.78 | 1227.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1233.50 | 1231.32 | 1227.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:45:00 | 1225.80 | 1231.32 | 1227.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 1226.50 | 1230.36 | 1227.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:30:00 | 1227.80 | 1230.36 | 1227.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 1219.30 | 1228.15 | 1226.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 1219.30 | 1228.15 | 1226.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-12-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 13:15:00 | 1211.30 | 1224.78 | 1225.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 14:15:00 | 1207.20 | 1221.26 | 1223.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 1163.80 | 1163.50 | 1172.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 1163.80 | 1163.50 | 1172.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 1163.80 | 1163.50 | 1172.84 | EMA400 retest candle locked (from downside) |

### Cycle 192 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 1214.50 | 1179.30 | 1177.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 1225.00 | 1188.44 | 1181.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 1228.80 | 1230.70 | 1222.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 12:45:00 | 1230.10 | 1230.70 | 1222.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1217.60 | 1227.66 | 1222.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:45:00 | 1218.20 | 1227.66 | 1222.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 1218.90 | 1225.91 | 1222.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 1234.30 | 1225.91 | 1222.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1221.10 | 1224.79 | 1222.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 1221.10 | 1224.79 | 1222.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 1223.40 | 1224.51 | 1222.66 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 14:15:00 | 1206.90 | 1218.68 | 1220.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 10:15:00 | 1199.30 | 1212.49 | 1216.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 14:15:00 | 1172.60 | 1171.10 | 1180.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 14:30:00 | 1174.30 | 1171.10 | 1180.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1168.90 | 1170.98 | 1178.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:30:00 | 1162.00 | 1168.18 | 1175.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1161.30 | 1166.58 | 1172.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 10:30:00 | 1157.00 | 1165.09 | 1170.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1157.60 | 1165.08 | 1168.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1156.50 | 1163.36 | 1167.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 10:30:00 | 1151.60 | 1161.55 | 1166.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 14:15:00 | 1151.20 | 1159.30 | 1164.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 15:00:00 | 1151.50 | 1157.74 | 1162.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 1193.60 | 1154.69 | 1156.11 | SL hit (close>static) qty=1.00 sl=1184.00 alert=retest2 |

### Cycle 194 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 1191.30 | 1162.01 | 1159.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 13:15:00 | 1200.40 | 1177.77 | 1167.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 1191.50 | 1193.17 | 1179.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 1191.50 | 1193.17 | 1179.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1187.50 | 1192.52 | 1181.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:45:00 | 1186.10 | 1192.52 | 1181.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1175.00 | 1189.02 | 1181.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 1175.00 | 1189.02 | 1181.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 1181.40 | 1187.49 | 1181.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 09:45:00 | 1186.90 | 1183.86 | 1180.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 10:15:00 | 1189.20 | 1183.86 | 1180.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 13:45:00 | 1187.10 | 1181.32 | 1179.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 14:30:00 | 1185.80 | 1181.72 | 1180.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-12 09:15:00 | 1163.40 | 1177.99 | 1178.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 1163.40 | 1177.99 | 1178.82 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 1188.20 | 1174.66 | 1173.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 1216.00 | 1189.94 | 1182.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1193.80 | 1196.65 | 1187.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 1193.80 | 1196.65 | 1187.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1187.00 | 1193.42 | 1187.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 1167.10 | 1193.42 | 1187.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1167.80 | 1188.29 | 1186.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 1167.80 | 1188.29 | 1186.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 197 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1161.50 | 1182.93 | 1183.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 1144.70 | 1163.93 | 1172.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1131.50 | 1118.94 | 1132.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1131.50 | 1118.94 | 1132.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1131.50 | 1118.94 | 1132.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1134.90 | 1118.94 | 1132.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1119.00 | 1118.95 | 1130.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1116.50 | 1118.26 | 1129.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 1117.40 | 1118.05 | 1128.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 14:00:00 | 1110.10 | 1116.46 | 1126.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:30:00 | 1116.70 | 1115.32 | 1123.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1113.10 | 1111.27 | 1118.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:45:00 | 1121.80 | 1111.27 | 1118.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1099.20 | 1103.76 | 1109.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:15:00 | 1095.90 | 1103.76 | 1109.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 1097.70 | 1100.85 | 1104.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 12:15:00 | 1060.67 | 1089.72 | 1097.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 12:15:00 | 1061.53 | 1089.72 | 1097.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 12:15:00 | 1060.87 | 1089.72 | 1097.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 13:15:00 | 1054.59 | 1083.46 | 1094.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 14:15:00 | 1041.11 | 1074.17 | 1088.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-29 14:15:00 | 1042.82 | 1074.17 | 1088.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-01 11:15:00 | 1004.85 | 1030.16 | 1051.47 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 198 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 980.70 | 968.68 | 967.51 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 960.00 | 967.37 | 968.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 15:15:00 | 957.50 | 962.65 | 965.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 875.50 | 870.29 | 893.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 10:00:00 | 875.50 | 870.29 | 893.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 890.90 | 879.63 | 887.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 890.90 | 879.63 | 887.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 880.00 | 879.70 | 886.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 13:45:00 | 879.00 | 880.11 | 885.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-20 09:15:00 | 835.05 | 851.55 | 861.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-24 09:15:00 | 791.10 | 811.34 | 827.54 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 200 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 681.50 | 655.51 | 655.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 691.15 | 662.64 | 658.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 661.95 | 677.68 | 669.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 661.95 | 677.68 | 669.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 661.95 | 677.68 | 669.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 661.95 | 677.68 | 669.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 661.25 | 674.39 | 669.16 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2026-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 14:15:00 | 657.10 | 665.07 | 665.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 654.00 | 662.00 | 663.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 658.80 | 657.95 | 660.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 14:45:00 | 659.45 | 657.95 | 660.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 656.80 | 657.72 | 660.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:15:00 | 660.95 | 657.72 | 660.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 653.20 | 656.81 | 659.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 652.95 | 656.81 | 659.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 666.05 | 661.66 | 661.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 666.05 | 661.66 | 661.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 680.70 | 665.69 | 663.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 666.85 | 672.17 | 668.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 666.85 | 672.17 | 668.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 666.85 | 672.17 | 668.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 666.85 | 672.17 | 668.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 668.20 | 671.38 | 668.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:45:00 | 666.15 | 671.38 | 668.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 666.05 | 670.31 | 668.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:15:00 | 665.95 | 670.31 | 668.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 663.25 | 668.90 | 668.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 663.25 | 668.90 | 668.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 660.70 | 667.26 | 667.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 648.00 | 661.32 | 664.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 679.30 | 651.91 | 656.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 679.30 | 651.91 | 656.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 679.30 | 651.91 | 656.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 679.30 | 651.91 | 656.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 675.40 | 656.61 | 658.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 678.50 | 656.61 | 658.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 678.90 | 661.06 | 659.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 12:15:00 | 680.20 | 672.60 | 667.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 710.00 | 713.64 | 705.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 710.00 | 713.64 | 705.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 712.00 | 715.85 | 711.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:45:00 | 713.40 | 715.85 | 711.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 711.75 | 715.03 | 711.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 711.75 | 715.03 | 711.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 719.35 | 715.89 | 712.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:45:00 | 712.80 | 715.89 | 712.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 14:15:00 | 716.55 | 716.08 | 713.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 14:30:00 | 713.50 | 716.08 | 713.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 709.10 | 714.91 | 713.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 713.60 | 714.91 | 713.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 718.00 | 715.40 | 713.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 710.35 | 712.45 | 712.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 710.35 | 712.45 | 712.55 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 731.55 | 715.81 | 714.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 739.15 | 729.02 | 722.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 12:15:00 | 742.05 | 742.26 | 735.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 12:30:00 | 741.70 | 742.26 | 735.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 728.85 | 741.59 | 737.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:00:00 | 728.85 | 741.59 | 737.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 734.10 | 740.10 | 736.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 11:15:00 | 738.05 | 740.10 | 736.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 13:15:00 | 738.90 | 739.28 | 737.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 13:45:00 | 738.15 | 739.30 | 737.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:30:00 | 739.85 | 738.25 | 737.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 727.75 | 740.00 | 739.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 727.75 | 740.00 | 739.37 | SL hit (close<static) qty=1.00 sl=728.20 alert=retest2 |

### Cycle 207 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 726.65 | 737.33 | 738.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 722.45 | 734.35 | 736.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 736.30 | 733.73 | 736.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 13:15:00 | 736.30 | 733.73 | 736.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 736.30 | 733.73 | 736.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 13:45:00 | 740.00 | 733.73 | 736.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 735.40 | 734.06 | 735.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:45:00 | 737.10 | 734.06 | 735.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 15:15:00 | 735.80 | 734.41 | 735.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:15:00 | 735.70 | 734.41 | 735.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 739.60 | 735.45 | 736.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:45:00 | 741.85 | 735.45 | 736.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 740.20 | 736.40 | 736.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 740.20 | 736.40 | 736.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 11:15:00 | 740.35 | 737.19 | 736.97 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 732.00 | 736.32 | 736.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 719.50 | 732.96 | 735.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 726.05 | 715.25 | 722.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 726.05 | 715.25 | 722.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 726.05 | 715.25 | 722.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 728.20 | 715.25 | 722.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 729.15 | 718.03 | 722.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 729.60 | 718.03 | 722.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 739.80 | 728.20 | 726.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 750.85 | 737.65 | 733.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 740.55 | 740.95 | 736.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 15:00:00 | 740.55 | 740.95 | 736.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 747.25 | 741.90 | 738.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:30:00 | 754.30 | 744.84 | 740.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:45:00 | 756.60 | 747.87 | 742.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:15:00 | 788.50 | 773.38 | 768.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 14:15:00 | 748.65 | 762.30 | 763.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 14:15:00 | 748.65 | 762.30 | 763.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 09:15:00 | 720.50 | 751.89 | 758.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 10:15:00 | 728.85 | 727.64 | 738.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:45:00 | 730.85 | 727.64 | 738.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 727.10 | 728.12 | 735.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 734.45 | 728.12 | 735.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-04-12 09:15:00 | 1466.10 | 2024-04-18 15:15:00 | 1392.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 09:15:00 | 1466.10 | 2024-04-19 12:15:00 | 1407.95 | STOP_HIT | 0.50 | 3.97% |
| BUY | retest2 | 2024-05-03 09:15:00 | 1526.90 | 2024-05-07 11:15:00 | 1497.10 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-05-06 10:15:00 | 1522.45 | 2024-05-07 11:15:00 | 1497.10 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-05-06 10:45:00 | 1523.00 | 2024-05-07 11:15:00 | 1497.10 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-05-06 13:45:00 | 1523.80 | 2024-05-07 11:15:00 | 1497.10 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-05-08 09:15:00 | 1504.35 | 2024-05-13 09:15:00 | 1429.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-09 10:00:00 | 1502.00 | 2024-05-13 10:15:00 | 1426.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-08 09:15:00 | 1504.35 | 2024-05-14 09:15:00 | 1447.00 | STOP_HIT | 0.50 | 3.81% |
| SELL | retest2 | 2024-05-09 10:00:00 | 1502.00 | 2024-05-14 09:15:00 | 1447.00 | STOP_HIT | 0.50 | 3.66% |
| SELL | retest2 | 2024-05-23 09:15:00 | 1491.70 | 2024-05-23 12:15:00 | 1500.55 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2024-05-23 10:00:00 | 1494.95 | 2024-05-23 12:15:00 | 1500.55 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2024-05-31 10:15:00 | 1447.55 | 2024-06-04 09:15:00 | 1378.02 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2024-06-03 10:30:00 | 1450.00 | 2024-06-04 09:15:00 | 1379.21 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2024-06-03 13:30:00 | 1450.55 | 2024-06-04 10:15:00 | 1375.17 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2024-06-03 14:00:00 | 1451.80 | 2024-06-04 10:15:00 | 1377.50 | PARTIAL | 0.50 | 5.12% |
| SELL | retest2 | 2024-05-31 10:15:00 | 1447.55 | 2024-06-04 12:15:00 | 1302.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 10:30:00 | 1450.00 | 2024-06-04 12:15:00 | 1305.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 13:30:00 | 1450.55 | 2024-06-04 12:15:00 | 1305.49 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-03 14:00:00 | 1451.80 | 2024-06-04 12:15:00 | 1306.62 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-04 11:00:00 | 1372.90 | 2024-06-04 12:15:00 | 1304.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 11:30:00 | 1376.00 | 2024-06-04 12:15:00 | 1307.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 11:00:00 | 1372.90 | 2024-06-05 10:15:00 | 1439.95 | STOP_HIT | 0.50 | -4.88% |
| SELL | retest2 | 2024-06-04 11:30:00 | 1376.00 | 2024-06-05 10:15:00 | 1439.95 | STOP_HIT | 0.50 | -4.65% |
| SELL | retest2 | 2024-06-04 14:15:00 | 1377.20 | 2024-06-06 09:15:00 | 1450.35 | STOP_HIT | 1.00 | -5.31% |
| SELL | retest2 | 2024-06-05 09:15:00 | 1360.00 | 2024-06-06 09:15:00 | 1450.35 | STOP_HIT | 1.00 | -6.64% |
| BUY | retest2 | 2024-06-10 12:30:00 | 1479.85 | 2024-06-12 10:15:00 | 1477.00 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-06-10 13:15:00 | 1479.10 | 2024-06-12 10:15:00 | 1477.00 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-06-10 15:00:00 | 1485.20 | 2024-06-12 10:15:00 | 1477.00 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-06-11 09:45:00 | 1482.45 | 2024-06-12 10:15:00 | 1477.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-06-26 09:15:00 | 1595.55 | 2024-06-27 12:15:00 | 1575.35 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-06-26 09:45:00 | 1593.35 | 2024-06-27 12:15:00 | 1575.35 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-06-26 15:15:00 | 1595.00 | 2024-06-27 12:15:00 | 1575.35 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-06-27 11:30:00 | 1596.35 | 2024-06-27 12:15:00 | 1575.35 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-07-02 14:45:00 | 1662.35 | 2024-07-05 15:15:00 | 1695.00 | STOP_HIT | 1.00 | 1.96% |
| BUY | retest2 | 2024-07-03 09:15:00 | 1710.35 | 2024-07-05 15:15:00 | 1695.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-07-09 11:00:00 | 1705.00 | 2024-07-09 13:15:00 | 1710.25 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-07-09 11:30:00 | 1698.60 | 2024-07-09 13:15:00 | 1710.25 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-07-11 14:15:00 | 1710.05 | 2024-07-12 14:15:00 | 1881.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-11 15:15:00 | 1720.00 | 2024-07-12 14:15:00 | 1892.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-12 10:30:00 | 1736.75 | 2024-07-12 14:15:00 | 1910.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-26 15:00:00 | 1786.05 | 2024-07-29 15:15:00 | 1806.05 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-07-29 12:00:00 | 1776.35 | 2024-07-29 15:15:00 | 1806.05 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-07-30 10:15:00 | 1791.20 | 2024-07-30 10:15:00 | 1804.85 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-08-01 09:15:00 | 1890.10 | 2024-08-02 14:15:00 | 1823.60 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2024-08-01 11:15:00 | 1852.15 | 2024-08-02 14:15:00 | 1823.60 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-08-01 13:15:00 | 1850.05 | 2024-08-02 14:15:00 | 1823.60 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-08-01 14:00:00 | 1850.00 | 2024-08-02 14:15:00 | 1823.60 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-08-06 10:45:00 | 1744.65 | 2024-08-09 13:15:00 | 1752.75 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-08-09 11:15:00 | 1745.00 | 2024-08-09 13:15:00 | 1752.75 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-08-21 09:15:00 | 1843.95 | 2024-08-23 10:15:00 | 1828.05 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-08-21 09:45:00 | 1841.90 | 2024-08-23 10:15:00 | 1828.05 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-08-21 10:45:00 | 1835.65 | 2024-08-23 10:15:00 | 1828.05 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-08-21 11:45:00 | 1837.00 | 2024-08-23 10:15:00 | 1828.05 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-08-22 09:30:00 | 1836.90 | 2024-08-23 10:15:00 | 1828.05 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-08-22 10:30:00 | 1843.55 | 2024-08-23 10:15:00 | 1828.05 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-10-01 10:15:00 | 1634.45 | 2024-10-01 13:15:00 | 1655.45 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-10-01 10:45:00 | 1630.00 | 2024-10-01 13:15:00 | 1655.45 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-10-04 11:45:00 | 1690.00 | 2024-10-07 10:15:00 | 1659.80 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-10-04 13:15:00 | 1691.00 | 2024-10-07 10:15:00 | 1659.80 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-10-04 15:00:00 | 1691.65 | 2024-10-07 10:15:00 | 1659.80 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2024-10-07 12:15:00 | 1692.75 | 2024-10-21 09:15:00 | 1769.40 | STOP_HIT | 1.00 | 4.53% |
| BUY | retest2 | 2024-10-08 14:30:00 | 1725.80 | 2024-10-21 09:15:00 | 1769.40 | STOP_HIT | 1.00 | 2.53% |
| BUY | retest2 | 2024-10-09 09:15:00 | 1730.75 | 2024-10-21 09:15:00 | 1769.40 | STOP_HIT | 1.00 | 2.23% |
| BUY | retest2 | 2024-11-04 13:00:00 | 1433.30 | 2024-11-05 12:15:00 | 1380.55 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2024-11-13 13:00:00 | 1371.15 | 2024-11-19 15:15:00 | 1302.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 14:00:00 | 1372.40 | 2024-11-19 15:15:00 | 1303.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 14:30:00 | 1367.90 | 2024-11-21 09:15:00 | 1299.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-13 13:00:00 | 1371.15 | 2024-11-22 09:15:00 | 1304.60 | STOP_HIT | 0.50 | 4.85% |
| SELL | retest2 | 2024-11-13 14:00:00 | 1372.40 | 2024-11-22 09:15:00 | 1304.60 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2024-11-13 14:30:00 | 1367.90 | 2024-11-22 09:15:00 | 1304.60 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest1 | 2024-12-02 09:15:00 | 1356.00 | 2024-12-03 09:15:00 | 1422.50 | STOP_HIT | 1.00 | -4.90% |
| SELL | retest2 | 2024-12-18 12:30:00 | 1516.55 | 2024-12-20 15:15:00 | 1440.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-18 14:45:00 | 1516.75 | 2024-12-20 15:15:00 | 1440.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-19 09:15:00 | 1496.30 | 2024-12-23 09:15:00 | 1437.63 | PARTIAL | 0.50 | 3.92% |
| SELL | retest2 | 2024-12-20 09:30:00 | 1513.30 | 2024-12-23 12:15:00 | 1421.48 | PARTIAL | 0.50 | 6.07% |
| SELL | retest2 | 2024-12-18 12:30:00 | 1516.55 | 2024-12-24 09:15:00 | 1438.65 | STOP_HIT | 0.50 | 5.14% |
| SELL | retest2 | 2024-12-18 14:45:00 | 1516.75 | 2024-12-24 09:15:00 | 1438.65 | STOP_HIT | 0.50 | 5.15% |
| SELL | retest2 | 2024-12-19 09:15:00 | 1496.30 | 2024-12-24 09:15:00 | 1438.65 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2024-12-20 09:30:00 | 1513.30 | 2024-12-24 09:15:00 | 1438.65 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest2 | 2025-01-09 11:15:00 | 1432.75 | 2025-01-10 09:15:00 | 1361.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 13:00:00 | 1427.05 | 2025-01-13 09:15:00 | 1355.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 11:15:00 | 1432.75 | 2025-01-14 10:15:00 | 1333.15 | STOP_HIT | 0.50 | 6.95% |
| SELL | retest2 | 2025-01-09 13:00:00 | 1427.05 | 2025-01-14 10:15:00 | 1333.15 | STOP_HIT | 0.50 | 6.58% |
| BUY | retest2 | 2025-02-01 15:15:00 | 1420.00 | 2025-02-11 09:15:00 | 1397.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-02-04 09:15:00 | 1444.00 | 2025-02-11 09:15:00 | 1397.00 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2025-02-05 10:00:00 | 1419.05 | 2025-02-11 09:15:00 | 1397.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-02-05 11:15:00 | 1420.00 | 2025-02-11 09:15:00 | 1397.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-02-05 12:30:00 | 1429.25 | 2025-02-11 09:15:00 | 1397.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-02-10 10:30:00 | 1425.90 | 2025-02-11 09:15:00 | 1397.00 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-02-10 11:00:00 | 1429.05 | 2025-02-11 09:15:00 | 1397.00 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-02-17 10:15:00 | 1311.00 | 2025-02-18 14:15:00 | 1329.95 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-02-17 11:00:00 | 1305.95 | 2025-02-19 10:15:00 | 1347.85 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2025-02-17 11:30:00 | 1310.95 | 2025-02-19 10:15:00 | 1347.85 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-02-17 13:00:00 | 1312.00 | 2025-02-19 10:15:00 | 1347.85 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-02-18 10:15:00 | 1296.95 | 2025-02-19 10:15:00 | 1347.85 | STOP_HIT | 1.00 | -3.92% |
| SELL | retest2 | 2025-02-27 11:45:00 | 1260.80 | 2025-02-28 14:15:00 | 1197.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 11:45:00 | 1260.80 | 2025-03-03 12:15:00 | 1201.40 | STOP_HIT | 0.50 | 4.71% |
| SELL | retest2 | 2025-03-12 11:45:00 | 1234.30 | 2025-03-17 15:15:00 | 1246.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-03-26 09:15:00 | 1356.25 | 2025-03-27 09:15:00 | 1335.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-03-26 09:45:00 | 1356.10 | 2025-03-27 09:15:00 | 1335.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-04-03 09:15:00 | 1263.10 | 2025-04-04 09:15:00 | 1199.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 09:15:00 | 1263.10 | 2025-04-04 11:15:00 | 1136.79 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-17 13:00:00 | 1142.40 | 2025-04-28 13:15:00 | 1256.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 13:30:00 | 1145.00 | 2025-04-28 13:15:00 | 1259.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-21 09:15:00 | 1142.90 | 2025-04-28 13:15:00 | 1257.19 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-23 11:30:00 | 1332.50 | 2025-05-23 13:15:00 | 1337.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-03 11:30:00 | 1330.10 | 2025-06-09 09:15:00 | 1356.20 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-06-03 14:15:00 | 1328.10 | 2025-06-09 09:15:00 | 1356.20 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-06-18 12:45:00 | 1413.90 | 2025-06-19 09:15:00 | 1386.80 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-06-18 14:30:00 | 1418.20 | 2025-06-19 09:15:00 | 1386.80 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest1 | 2025-06-27 14:45:00 | 1268.80 | 2025-07-03 10:15:00 | 1259.70 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest1 | 2025-06-30 10:15:00 | 1273.30 | 2025-07-03 10:15:00 | 1259.70 | STOP_HIT | 1.00 | 1.07% |
| BUY | retest2 | 2025-07-04 13:45:00 | 1262.80 | 2025-07-09 13:15:00 | 1261.60 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-07-04 14:15:00 | 1262.80 | 2025-07-09 13:15:00 | 1261.60 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-07-07 09:45:00 | 1263.30 | 2025-07-09 13:15:00 | 1261.60 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-07-09 13:15:00 | 1262.80 | 2025-07-09 13:15:00 | 1261.60 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-07-22 12:00:00 | 1254.00 | 2025-07-23 10:15:00 | 1278.90 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2025-07-22 12:45:00 | 1255.70 | 2025-07-23 10:15:00 | 1278.90 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-08-06 10:00:00 | 1196.90 | 2025-08-12 09:15:00 | 1214.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-08-06 11:30:00 | 1200.10 | 2025-08-12 09:15:00 | 1214.90 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-08-07 09:30:00 | 1198.20 | 2025-08-12 09:15:00 | 1214.90 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-08-08 10:00:00 | 1200.20 | 2025-08-12 09:15:00 | 1214.90 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-08-19 13:15:00 | 1200.60 | 2025-08-20 11:15:00 | 1218.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-08-20 09:15:00 | 1197.70 | 2025-08-20 11:15:00 | 1218.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1211.90 | 2025-09-12 12:15:00 | 1251.90 | STOP_HIT | 1.00 | 3.30% |
| BUY | retest2 | 2025-09-03 11:00:00 | 1211.20 | 2025-09-12 12:15:00 | 1251.90 | STOP_HIT | 1.00 | 3.36% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1251.00 | 2025-09-30 12:15:00 | 1188.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1251.00 | 2025-09-30 13:15:00 | 1125.90 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-14 11:15:00 | 1145.00 | 2025-10-15 09:15:00 | 1164.90 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-10-14 12:15:00 | 1144.80 | 2025-10-15 09:15:00 | 1164.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-10-14 13:00:00 | 1142.60 | 2025-10-15 09:15:00 | 1164.90 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-10-29 14:15:00 | 1197.30 | 2025-10-30 09:15:00 | 1170.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-11-04 09:45:00 | 1160.70 | 2025-11-10 10:15:00 | 1172.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-11-19 15:00:00 | 1204.20 | 2025-11-27 09:15:00 | 1205.60 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-11-20 14:15:00 | 1204.20 | 2025-11-27 09:15:00 | 1205.60 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-11-24 11:45:00 | 1194.20 | 2025-11-27 09:15:00 | 1205.60 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest1 | 2025-12-03 11:15:00 | 1263.50 | 2025-12-05 11:15:00 | 1259.50 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest1 | 2025-12-04 09:15:00 | 1268.00 | 2025-12-05 11:15:00 | 1259.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-12-05 13:45:00 | 1269.20 | 2025-12-08 09:15:00 | 1249.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-01-01 12:30:00 | 1162.00 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1161.30 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2026-01-02 10:30:00 | 1157.00 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2026-01-05 09:15:00 | 1157.60 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2026-01-05 10:30:00 | 1151.60 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -3.65% |
| SELL | retest2 | 2026-01-05 14:15:00 | 1151.20 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -3.68% |
| SELL | retest2 | 2026-01-05 15:00:00 | 1151.50 | 2026-01-07 09:15:00 | 1193.60 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2026-01-09 09:45:00 | 1186.90 | 2026-01-12 09:15:00 | 1163.40 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2026-01-09 10:15:00 | 1189.20 | 2026-01-12 09:15:00 | 1163.40 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-01-09 13:45:00 | 1187.10 | 2026-01-12 09:15:00 | 1163.40 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-01-09 14:30:00 | 1185.80 | 2026-01-12 09:15:00 | 1163.40 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1116.50 | 2026-01-29 12:15:00 | 1060.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 12:45:00 | 1117.40 | 2026-01-29 12:15:00 | 1061.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 14:00:00 | 1110.10 | 2026-01-29 12:15:00 | 1060.87 | PARTIAL | 0.50 | 4.44% |
| SELL | retest2 | 2026-01-23 09:30:00 | 1116.70 | 2026-01-29 13:15:00 | 1054.59 | PARTIAL | 0.50 | 5.56% |
| SELL | retest2 | 2026-01-28 10:15:00 | 1095.90 | 2026-01-29 14:15:00 | 1041.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1097.70 | 2026-01-29 14:15:00 | 1042.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 11:30:00 | 1116.50 | 2026-02-01 11:15:00 | 1004.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 12:45:00 | 1117.40 | 2026-02-01 11:15:00 | 1005.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 14:00:00 | 1110.10 | 2026-02-01 11:15:00 | 999.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-23 09:30:00 | 1116.70 | 2026-02-01 11:15:00 | 1005.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-28 10:15:00 | 1095.90 | 2026-02-01 12:15:00 | 986.31 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-29 09:15:00 | 1097.70 | 2026-02-01 12:15:00 | 987.93 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-17 13:45:00 | 879.00 | 2026-02-20 09:15:00 | 835.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 13:45:00 | 879.00 | 2026-02-24 09:15:00 | 791.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 10:15:00 | 652.95 | 2026-03-24 14:15:00 | 666.05 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-04-13 10:15:00 | 713.60 | 2026-04-13 14:15:00 | 710.35 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-04-13 10:45:00 | 718.00 | 2026-04-13 14:15:00 | 710.35 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-04-20 11:15:00 | 738.05 | 2026-04-22 09:15:00 | 727.75 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-20 13:15:00 | 738.90 | 2026-04-22 09:15:00 | 727.75 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2026-04-20 13:45:00 | 738.15 | 2026-04-22 09:15:00 | 727.75 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-04-21 09:30:00 | 739.85 | 2026-04-22 09:15:00 | 727.75 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-04-30 12:30:00 | 754.30 | 2026-05-06 14:15:00 | 748.65 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2026-04-30 13:45:00 | 756.60 | 2026-05-06 14:15:00 | 748.65 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-05-06 13:15:00 | 788.50 | 2026-05-06 14:15:00 | 748.65 | STOP_HIT | 1.00 | -5.05% |
