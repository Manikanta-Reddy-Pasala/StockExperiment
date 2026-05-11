# Lupin Ltd. (LUPIN)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2373.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 237 |
| ALERT1 | 144 |
| ALERT2 | 143 |
| ALERT2_SKIP | 96 |
| ALERT3 | 310 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 118 |
| PARTIAL | 5 |
| TARGET_HIT | 4 |
| STOP_HIT | 116 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 125 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 92
- **Target hits / Stop hits / Partials:** 4 / 116 / 5
- **Avg / median % per leg:** -0.21% / -0.91%
- **Sum % (uncompounded):** -26.58%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 15 | 23.4% | 4 | 59 | 1 | 0.06% | 3.6% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.44% | 8.9% |
| BUY @ 3rd Alert (retest2) | 62 | 13 | 21.0% | 4 | 58 | 0 | -0.09% | -5.3% |
| SELL (all) | 61 | 18 | 29.5% | 0 | 57 | 4 | -0.49% | -30.1% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.55% | -1.5% |
| SELL @ 3rd Alert (retest2) | 60 | 18 | 30.0% | 0 | 56 | 4 | -0.48% | -28.6% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 2.44% | 7.3% |
| retest2 (combined) | 122 | 31 | 25.4% | 4 | 114 | 4 | -0.28% | -33.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 10:15:00 | 778.70 | 780.92 | 781.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 15:15:00 | 772.40 | 777.54 | 779.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 09:15:00 | 780.70 | 778.17 | 779.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 780.70 | 778.17 | 779.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 780.70 | 778.17 | 779.17 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 09:15:00 | 782.00 | 777.63 | 777.48 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 13:15:00 | 775.15 | 777.27 | 777.39 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 15:15:00 | 778.60 | 777.68 | 777.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-25 09:15:00 | 780.80 | 778.30 | 777.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 12:15:00 | 778.00 | 779.34 | 778.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 12:15:00 | 778.00 | 779.34 | 778.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 12:15:00 | 778.00 | 779.34 | 778.53 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2023-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-26 11:15:00 | 774.80 | 778.17 | 778.29 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 14:15:00 | 780.00 | 778.53 | 778.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 15:15:00 | 780.55 | 778.93 | 778.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 10:15:00 | 799.50 | 799.63 | 791.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-02 11:15:00 | 806.15 | 809.58 | 807.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 11:15:00 | 806.15 | 809.58 | 807.08 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 12:15:00 | 808.90 | 815.74 | 816.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 14:15:00 | 806.00 | 810.05 | 812.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 816.50 | 811.33 | 812.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 816.50 | 811.33 | 812.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 816.50 | 811.33 | 812.73 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-12 12:15:00 | 819.05 | 814.23 | 813.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 15:15:00 | 820.85 | 816.51 | 815.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 13:15:00 | 816.15 | 817.73 | 816.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 13:15:00 | 816.15 | 817.73 | 816.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 13:15:00 | 816.15 | 817.73 | 816.36 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 09:15:00 | 826.55 | 828.04 | 828.23 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-20 13:15:00 | 829.05 | 828.33 | 828.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-20 15:15:00 | 829.90 | 828.77 | 828.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 09:15:00 | 859.15 | 860.59 | 848.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 10:15:00 | 846.45 | 857.76 | 848.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 846.45 | 857.76 | 848.15 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2023-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 12:15:00 | 1078.00 | 1083.61 | 1083.65 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 09:15:00 | 1096.95 | 1083.56 | 1083.28 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-08-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 15:15:00 | 1080.25 | 1084.26 | 1084.38 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 09:15:00 | 1087.65 | 1084.94 | 1084.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-16 10:15:00 | 1092.75 | 1086.50 | 1085.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 11:15:00 | 1092.00 | 1100.53 | 1095.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-17 11:15:00 | 1092.00 | 1100.53 | 1095.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 1092.00 | 1100.53 | 1095.33 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 10:15:00 | 1072.55 | 1089.73 | 1092.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 11:15:00 | 1068.90 | 1085.56 | 1089.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 1075.00 | 1074.06 | 1081.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 11:15:00 | 1080.60 | 1075.36 | 1081.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 1080.60 | 1075.36 | 1081.00 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 10:15:00 | 1089.65 | 1084.16 | 1083.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 12:15:00 | 1090.95 | 1086.45 | 1084.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 12:15:00 | 1091.05 | 1091.41 | 1088.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-23 12:15:00 | 1091.05 | 1091.41 | 1088.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 1091.05 | 1091.41 | 1088.58 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 10:15:00 | 1081.20 | 1089.11 | 1090.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 11:15:00 | 1080.00 | 1087.29 | 1089.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 1090.30 | 1083.72 | 1086.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 1090.30 | 1083.72 | 1086.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 1090.30 | 1083.72 | 1086.23 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2023-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 12:15:00 | 1094.45 | 1088.95 | 1088.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-28 13:15:00 | 1097.05 | 1090.57 | 1089.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-29 11:15:00 | 1096.35 | 1097.23 | 1093.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 11:15:00 | 1096.35 | 1097.23 | 1093.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 11:15:00 | 1096.35 | 1097.23 | 1093.67 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2023-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 09:15:00 | 1090.00 | 1098.48 | 1099.33 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 1104.60 | 1098.54 | 1098.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 13:15:00 | 1123.00 | 1105.44 | 1101.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 10:15:00 | 1131.35 | 1131.69 | 1121.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 10:15:00 | 1121.95 | 1131.58 | 1127.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 10:15:00 | 1121.95 | 1131.58 | 1127.11 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2023-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 11:15:00 | 1132.00 | 1144.92 | 1146.59 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 13:15:00 | 1150.35 | 1144.87 | 1144.77 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2023-09-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 09:15:00 | 1137.25 | 1144.46 | 1144.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 14:15:00 | 1131.65 | 1138.89 | 1141.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 10:15:00 | 1107.60 | 1107.14 | 1118.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 14:15:00 | 1110.00 | 1110.33 | 1113.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 1110.00 | 1110.33 | 1113.41 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 10:15:00 | 1124.00 | 1115.53 | 1115.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 11:15:00 | 1133.00 | 1119.02 | 1116.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 1128.60 | 1130.38 | 1125.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 14:15:00 | 1130.00 | 1130.39 | 1126.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 1130.00 | 1130.39 | 1126.13 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 14:15:00 | 1149.60 | 1155.94 | 1155.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 10:15:00 | 1145.65 | 1151.92 | 1153.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 11:15:00 | 1156.00 | 1152.74 | 1154.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 11:15:00 | 1156.00 | 1152.74 | 1154.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 11:15:00 | 1156.00 | 1152.74 | 1154.16 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 12:15:00 | 1156.00 | 1154.23 | 1154.03 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 11:15:00 | 1149.15 | 1153.85 | 1154.08 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 11:15:00 | 1156.60 | 1153.84 | 1153.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-12 09:15:00 | 1163.10 | 1159.11 | 1157.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 15:15:00 | 1169.65 | 1171.44 | 1165.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 09:15:00 | 1181.70 | 1173.49 | 1167.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 09:15:00 | 1181.70 | 1173.49 | 1167.12 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 10:15:00 | 1190.70 | 1195.06 | 1195.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 1181.95 | 1192.44 | 1194.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-26 14:15:00 | 1141.05 | 1138.26 | 1150.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1148.00 | 1140.42 | 1149.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1148.00 | 1140.42 | 1149.28 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-11-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 09:15:00 | 1166.80 | 1142.50 | 1139.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 10:15:00 | 1180.45 | 1150.09 | 1143.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 12:15:00 | 1194.95 | 1197.92 | 1188.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 13:15:00 | 1190.30 | 1196.39 | 1188.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 13:15:00 | 1190.30 | 1196.39 | 1188.95 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2023-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 10:15:00 | 1195.25 | 1202.46 | 1202.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-12 18:15:00 | 1177.00 | 1188.09 | 1194.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-15 09:15:00 | 1181.20 | 1174.85 | 1182.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-15 09:15:00 | 1181.20 | 1174.85 | 1182.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 1181.20 | 1174.85 | 1182.24 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-11-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 12:15:00 | 1189.75 | 1181.48 | 1180.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 13:15:00 | 1195.85 | 1184.36 | 1182.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 12:15:00 | 1193.30 | 1194.54 | 1189.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-20 12:15:00 | 1198.50 | 1200.52 | 1195.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 12:15:00 | 1198.50 | 1200.52 | 1195.35 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-11-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 10:15:00 | 1193.20 | 1210.87 | 1210.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 11:15:00 | 1189.45 | 1206.58 | 1209.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 09:15:00 | 1220.80 | 1205.26 | 1206.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 09:15:00 | 1220.80 | 1205.26 | 1206.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 09:15:00 | 1220.80 | 1205.26 | 1206.84 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 10:15:00 | 1232.00 | 1210.60 | 1209.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 12:15:00 | 1240.45 | 1220.94 | 1214.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 09:15:00 | 1269.45 | 1286.79 | 1275.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 09:15:00 | 1269.45 | 1286.79 | 1275.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 09:15:00 | 1269.45 | 1286.79 | 1275.03 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-04 14:15:00 | 1260.15 | 1269.27 | 1269.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 13:15:00 | 1257.25 | 1263.70 | 1266.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-05 15:15:00 | 1263.45 | 1262.64 | 1265.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 09:15:00 | 1260.75 | 1262.26 | 1265.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 09:15:00 | 1260.75 | 1262.26 | 1265.09 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2023-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 09:15:00 | 1251.00 | 1243.07 | 1242.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 14:15:00 | 1252.10 | 1247.25 | 1244.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 09:15:00 | 1247.30 | 1247.97 | 1245.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 09:15:00 | 1247.30 | 1247.97 | 1245.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 1247.30 | 1247.97 | 1245.71 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 1228.00 | 1256.80 | 1260.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 1225.50 | 1250.54 | 1256.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 11:15:00 | 1251.00 | 1246.32 | 1252.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 14:15:00 | 1258.00 | 1249.01 | 1252.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 14:15:00 | 1258.00 | 1249.01 | 1252.59 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 09:15:00 | 1280.50 | 1256.11 | 1255.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-26 15:15:00 | 1289.00 | 1275.31 | 1268.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-27 14:15:00 | 1286.75 | 1287.64 | 1279.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-01 10:15:00 | 1312.15 | 1320.03 | 1312.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 1312.15 | 1320.03 | 1312.70 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 11:15:00 | 1380.65 | 1387.99 | 1388.54 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 11:15:00 | 1390.70 | 1388.07 | 1388.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 12:15:00 | 1393.20 | 1389.09 | 1388.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 13:15:00 | 1402.45 | 1405.66 | 1400.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-11 14:15:00 | 1400.55 | 1404.63 | 1400.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 14:15:00 | 1400.55 | 1404.63 | 1400.79 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-01-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-12 15:15:00 | 1396.45 | 1399.47 | 1399.54 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-15 09:15:00 | 1402.80 | 1400.14 | 1399.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 12:15:00 | 1422.40 | 1408.82 | 1404.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-16 09:15:00 | 1398.90 | 1409.62 | 1406.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-16 09:15:00 | 1398.90 | 1409.62 | 1406.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 1398.90 | 1409.62 | 1406.44 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 11:15:00 | 1390.30 | 1404.05 | 1404.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 1381.00 | 1399.44 | 1402.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-17 09:15:00 | 1398.00 | 1393.61 | 1398.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-17 09:15:00 | 1398.00 | 1393.61 | 1398.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 1398.00 | 1393.61 | 1398.03 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-01-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 11:15:00 | 1415.60 | 1401.77 | 1400.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 13:15:00 | 1419.50 | 1407.28 | 1403.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 09:15:00 | 1423.25 | 1425.28 | 1418.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 13:15:00 | 1416.10 | 1423.56 | 1419.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 13:15:00 | 1416.10 | 1423.56 | 1419.66 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 13:15:00 | 1597.70 | 1602.98 | 1603.66 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2024-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-13 15:15:00 | 1612.50 | 1603.28 | 1602.53 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 10:15:00 | 1594.55 | 1600.54 | 1601.35 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 14:15:00 | 1612.15 | 1601.48 | 1601.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-14 15:15:00 | 1618.00 | 1604.79 | 1602.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 10:15:00 | 1599.85 | 1604.47 | 1603.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-15 10:15:00 | 1599.85 | 1604.47 | 1603.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 1599.85 | 1604.47 | 1603.06 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-02-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-15 13:15:00 | 1598.90 | 1602.22 | 1602.28 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-02-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-15 14:15:00 | 1605.00 | 1602.78 | 1602.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 11:15:00 | 1623.85 | 1607.43 | 1604.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 1602.00 | 1613.09 | 1609.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 1602.00 | 1613.09 | 1609.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 1602.00 | 1613.09 | 1609.60 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 12:15:00 | 1592.05 | 1604.48 | 1606.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-19 13:15:00 | 1588.40 | 1601.26 | 1604.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-20 14:15:00 | 1585.05 | 1579.94 | 1589.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 15:15:00 | 1596.95 | 1583.34 | 1589.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 1596.95 | 1583.34 | 1589.91 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2024-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-21 12:15:00 | 1600.00 | 1593.18 | 1593.11 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-02-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 09:15:00 | 1579.75 | 1592.67 | 1593.22 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-23 09:15:00 | 1613.10 | 1594.26 | 1592.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 12:15:00 | 1619.75 | 1605.54 | 1598.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-26 09:15:00 | 1598.50 | 1608.20 | 1602.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-26 09:15:00 | 1598.50 | 1608.20 | 1602.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 1598.50 | 1608.20 | 1602.76 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 12:15:00 | 1583.85 | 1596.78 | 1598.34 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-26 14:15:00 | 1604.55 | 1599.33 | 1599.29 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 15:15:00 | 1598.80 | 1599.23 | 1599.25 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 09:15:00 | 1615.85 | 1602.55 | 1600.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-27 11:15:00 | 1631.00 | 1611.78 | 1605.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 10:15:00 | 1617.55 | 1620.20 | 1613.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 11:15:00 | 1616.35 | 1619.43 | 1613.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 1616.35 | 1619.43 | 1613.56 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 10:15:00 | 1600.45 | 1610.68 | 1611.35 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 14:15:00 | 1622.15 | 1612.15 | 1611.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 09:15:00 | 1628.50 | 1617.12 | 1614.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-01 14:15:00 | 1624.05 | 1625.98 | 1620.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 14:15:00 | 1624.05 | 1625.98 | 1620.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 14:15:00 | 1624.05 | 1625.98 | 1620.44 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2024-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 11:15:00 | 1648.00 | 1667.36 | 1669.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 1629.25 | 1650.51 | 1659.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 1623.60 | 1618.10 | 1635.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 10:15:00 | 1633.55 | 1621.19 | 1635.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 1633.55 | 1621.19 | 1635.12 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 13:15:00 | 1604.35 | 1591.01 | 1590.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 1615.85 | 1602.14 | 1596.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 12:15:00 | 1601.05 | 1603.58 | 1598.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-26 09:15:00 | 1595.60 | 1603.48 | 1600.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1595.60 | 1603.48 | 1600.23 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 1596.95 | 1604.12 | 1604.79 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 12:15:00 | 1615.60 | 1605.99 | 1605.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 15:15:00 | 1624.00 | 1612.22 | 1608.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-01 12:15:00 | 1618.20 | 1618.81 | 1613.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 13:15:00 | 1611.00 | 1617.25 | 1613.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 13:15:00 | 1611.00 | 1617.25 | 1613.30 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 12:15:00 | 1606.00 | 1612.59 | 1612.76 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 10:15:00 | 1616.00 | 1612.80 | 1612.62 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 11:15:00 | 1607.20 | 1611.68 | 1612.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-03 14:15:00 | 1605.25 | 1609.31 | 1610.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 09:15:00 | 1608.00 | 1607.90 | 1609.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 09:15:00 | 1608.00 | 1607.90 | 1609.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 1608.00 | 1607.90 | 1609.86 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2024-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 11:15:00 | 1617.80 | 1602.43 | 1602.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-09 09:15:00 | 1624.00 | 1611.71 | 1607.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 11:15:00 | 1607.80 | 1611.43 | 1608.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 11:15:00 | 1607.80 | 1611.43 | 1608.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 11:15:00 | 1607.80 | 1611.43 | 1608.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:15:00 | 1615.40 | 1608.94 | 1608.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 1636.50 | 1625.74 | 1619.80 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2024-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 14:15:00 | 1609.65 | 1616.94 | 1617.43 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 10:15:00 | 1635.65 | 1620.32 | 1618.79 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-16 14:15:00 | 1611.65 | 1617.42 | 1617.89 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2024-04-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 10:15:00 | 1633.15 | 1620.48 | 1619.15 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-04-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 14:15:00 | 1591.70 | 1613.91 | 1616.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 09:15:00 | 1578.45 | 1604.10 | 1611.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 09:15:00 | 1598.55 | 1571.72 | 1586.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 09:15:00 | 1598.55 | 1571.72 | 1586.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 1598.55 | 1571.72 | 1586.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 09:30:00 | 1599.60 | 1571.72 | 1586.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 10:15:00 | 1601.05 | 1577.59 | 1588.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 10:45:00 | 1600.10 | 1577.59 | 1588.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 15:15:00 | 1608.70 | 1593.90 | 1593.08 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 13:15:00 | 1586.90 | 1592.34 | 1592.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-23 14:15:00 | 1580.00 | 1589.87 | 1591.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 10:15:00 | 1588.05 | 1587.34 | 1589.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-24 11:00:00 | 1588.05 | 1587.34 | 1589.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 11:15:00 | 1583.70 | 1586.62 | 1589.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 11:30:00 | 1587.60 | 1586.62 | 1589.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 12:15:00 | 1574.65 | 1584.22 | 1587.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 12:30:00 | 1590.70 | 1584.22 | 1587.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 13:15:00 | 1585.85 | 1584.55 | 1587.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 13:45:00 | 1587.60 | 1584.55 | 1587.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 1584.00 | 1584.44 | 1587.40 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 12:15:00 | 1598.95 | 1590.34 | 1589.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-26 09:15:00 | 1609.00 | 1595.44 | 1592.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 14:15:00 | 1646.25 | 1647.10 | 1634.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-30 15:00:00 | 1646.25 | 1647.10 | 1634.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 15:15:00 | 1654.00 | 1649.82 | 1642.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 09:15:00 | 1665.00 | 1649.82 | 1642.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 13:00:00 | 1656.90 | 1659.17 | 1650.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 13:30:00 | 1655.00 | 1657.25 | 1650.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 14:15:00 | 1654.90 | 1657.25 | 1650.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 14:15:00 | 1655.00 | 1656.80 | 1650.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 15:15:00 | 1662.00 | 1656.80 | 1650.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-06 09:15:00 | 1646.95 | 1655.66 | 1651.42 | SL hit (close<static) qty=1.00 sl=1648.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 10:15:00 | 1608.05 | 1648.18 | 1650.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 11:15:00 | 1592.45 | 1637.04 | 1645.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 1615.30 | 1615.00 | 1628.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 09:15:00 | 1615.30 | 1615.00 | 1628.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 09:15:00 | 1615.30 | 1615.00 | 1628.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 10:45:00 | 1599.15 | 1610.86 | 1625.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:00:00 | 1601.70 | 1613.59 | 1621.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:30:00 | 1602.95 | 1612.87 | 1620.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 12:15:00 | 1604.05 | 1612.42 | 1619.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 1613.00 | 1600.55 | 1608.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 11:00:00 | 1613.00 | 1600.55 | 1608.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 1604.95 | 1601.43 | 1608.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 12:15:00 | 1602.10 | 1601.43 | 1608.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 14:15:00 | 1601.40 | 1600.64 | 1606.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 09:15:00 | 1617.80 | 1606.45 | 1608.14 | SL hit (close>static) qty=1.00 sl=1617.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-05-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 10:15:00 | 1625.50 | 1610.26 | 1609.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 12:15:00 | 1651.70 | 1620.98 | 1614.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 10:15:00 | 1637.00 | 1648.34 | 1633.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 10:15:00 | 1637.00 | 1648.34 | 1633.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 1637.00 | 1648.34 | 1633.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:00:00 | 1637.00 | 1648.34 | 1633.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 1638.50 | 1644.45 | 1636.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 15:00:00 | 1638.50 | 1644.45 | 1636.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 1634.00 | 1642.36 | 1636.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 09:15:00 | 1645.75 | 1642.36 | 1636.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 10:30:00 | 1639.70 | 1641.64 | 1636.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-15 12:15:00 | 1630.40 | 1637.89 | 1635.85 | SL hit (close<static) qty=1.00 sl=1630.90 alert=retest2 |

### Cycle 79 — SELL (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 11:15:00 | 1635.20 | 1668.73 | 1671.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 1613.65 | 1628.69 | 1643.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 09:15:00 | 1632.00 | 1626.84 | 1639.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 1632.00 | 1626.84 | 1639.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 1632.00 | 1626.84 | 1639.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 09:45:00 | 1634.60 | 1626.84 | 1639.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 1639.10 | 1630.77 | 1638.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:00:00 | 1639.10 | 1630.77 | 1638.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 13:15:00 | 1637.55 | 1632.12 | 1638.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:30:00 | 1642.75 | 1632.12 | 1638.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 1618.00 | 1629.30 | 1636.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:15:00 | 1612.75 | 1623.69 | 1632.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 1532.11 | 1572.28 | 1581.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 13:15:00 | 1572.55 | 1572.34 | 1580.59 | SL hit (close>ema200) qty=0.50 sl=1572.34 alert=retest2 |

### Cycle 80 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 1620.00 | 1589.56 | 1586.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 12:15:00 | 1629.90 | 1601.70 | 1592.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 11:15:00 | 1614.00 | 1618.97 | 1607.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 11:45:00 | 1621.05 | 1618.97 | 1607.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 1607.10 | 1616.60 | 1607.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:00:00 | 1607.10 | 1616.60 | 1607.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 1606.35 | 1614.55 | 1607.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:00:00 | 1606.35 | 1614.55 | 1607.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 1607.35 | 1613.11 | 1607.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 14:30:00 | 1604.55 | 1613.11 | 1607.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 1610.00 | 1612.49 | 1607.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:15:00 | 1627.95 | 1612.49 | 1607.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-11 15:15:00 | 1613.75 | 1620.91 | 1621.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2024-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 15:15:00 | 1613.75 | 1620.91 | 1621.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 10:15:00 | 1606.95 | 1616.89 | 1619.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 09:15:00 | 1614.10 | 1610.11 | 1614.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 09:15:00 | 1614.10 | 1610.11 | 1614.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 09:15:00 | 1614.10 | 1610.11 | 1614.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 09:30:00 | 1602.00 | 1607.78 | 1610.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 12:15:00 | 1601.70 | 1606.22 | 1609.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-14 12:45:00 | 1601.50 | 1605.07 | 1608.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 1591.15 | 1604.19 | 1607.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 1589.10 | 1601.17 | 1605.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-18 10:15:00 | 1584.40 | 1601.17 | 1605.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-24 09:45:00 | 1576.90 | 1563.45 | 1566.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 11:15:00 | 1578.40 | 1569.15 | 1568.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 11:15:00 | 1578.40 | 1569.15 | 1568.70 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 1566.60 | 1568.25 | 1568.38 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 1571.40 | 1568.82 | 1568.62 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 11:15:00 | 1560.60 | 1566.99 | 1567.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 12:15:00 | 1557.00 | 1564.99 | 1566.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 1571.95 | 1563.04 | 1564.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 10:15:00 | 1571.95 | 1563.04 | 1564.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1571.95 | 1563.04 | 1564.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:00:00 | 1571.95 | 1563.04 | 1564.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 11:15:00 | 1582.50 | 1566.93 | 1566.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 14:15:00 | 1587.40 | 1575.55 | 1570.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 1582.30 | 1583.33 | 1577.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 13:00:00 | 1582.30 | 1583.33 | 1577.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 1581.00 | 1582.87 | 1577.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:00:00 | 1581.00 | 1582.87 | 1577.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1582.10 | 1582.71 | 1577.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 1582.10 | 1582.71 | 1577.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1608.95 | 1588.81 | 1581.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-28 10:15:00 | 1613.60 | 1588.81 | 1581.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 14:45:00 | 1612.20 | 1611.45 | 1604.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-05 09:15:00 | 1774.96 | 1731.54 | 1689.49 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 1797.95 | 1819.43 | 1822.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 1785.00 | 1812.54 | 1818.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1817.00 | 1801.88 | 1809.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 1817.00 | 1801.88 | 1809.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1817.00 | 1801.88 | 1809.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 1817.00 | 1801.88 | 1809.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1836.90 | 1808.88 | 1811.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:00:00 | 1836.90 | 1808.88 | 1811.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 1836.00 | 1814.30 | 1814.07 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 14:15:00 | 1800.95 | 1813.61 | 1814.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 14:15:00 | 1799.20 | 1804.50 | 1808.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 15:15:00 | 1804.90 | 1804.58 | 1808.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-24 09:15:00 | 1801.60 | 1804.58 | 1808.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 1821.40 | 1807.94 | 1809.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 1821.40 | 1807.94 | 1809.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 1812.30 | 1808.81 | 1809.65 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 1816.90 | 1810.43 | 1810.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 15:15:00 | 1821.00 | 1814.52 | 1812.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1804.50 | 1812.51 | 1811.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 1804.50 | 1812.51 | 1811.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1804.50 | 1812.51 | 1811.70 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 13:15:00 | 1804.00 | 1810.94 | 1811.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 14:15:00 | 1800.90 | 1808.93 | 1810.42 | Break + close below crossover candle low |

### Cycle 92 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 1836.80 | 1812.75 | 1811.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 1846.00 | 1833.72 | 1824.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 1864.00 | 1867.65 | 1855.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 15:00:00 | 1864.00 | 1867.65 | 1855.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 1858.75 | 1865.87 | 1856.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 1887.70 | 1865.87 | 1856.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-06 13:15:00 | 1918.25 | 1937.97 | 1938.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 1918.25 | 1937.97 | 1938.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 1909.85 | 1932.34 | 1935.51 | Break + close below crossover candle low |

### Cycle 94 — BUY (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 09:15:00 | 1982.00 | 1938.75 | 1937.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 2036.80 | 1991.50 | 1969.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 2092.90 | 2096.18 | 2060.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 10:00:00 | 2092.90 | 2096.18 | 2060.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 2094.65 | 2099.46 | 2089.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 09:30:00 | 2087.00 | 2099.46 | 2089.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 2080.60 | 2095.69 | 2089.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:00:00 | 2080.60 | 2095.69 | 2089.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 2076.35 | 2091.82 | 2087.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 11:30:00 | 2077.05 | 2091.82 | 2087.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 14:15:00 | 2072.95 | 2084.29 | 2085.16 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 10:15:00 | 2097.45 | 2087.00 | 2086.08 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 09:15:00 | 2070.45 | 2084.96 | 2086.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 13:15:00 | 2066.35 | 2077.54 | 2081.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-20 09:15:00 | 2087.60 | 2078.06 | 2080.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 09:15:00 | 2087.60 | 2078.06 | 2080.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 2087.60 | 2078.06 | 2080.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 09:30:00 | 2082.70 | 2078.06 | 2080.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 2089.30 | 2080.31 | 2081.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 2089.30 | 2080.31 | 2081.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 11:15:00 | 2094.50 | 2083.14 | 2082.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 13:15:00 | 2098.65 | 2088.14 | 2085.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 2103.30 | 2107.66 | 2099.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 2103.30 | 2107.66 | 2099.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 2103.30 | 2107.66 | 2099.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 10:45:00 | 2128.00 | 2111.02 | 2101.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 09:15:00 | 2092.35 | 2106.31 | 2103.65 | SL hit (close<static) qty=1.00 sl=2096.25 alert=retest2 |

### Cycle 99 — SELL (started 2024-08-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 14:15:00 | 2097.20 | 2101.79 | 2102.13 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2024-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 09:15:00 | 2119.05 | 2103.69 | 2102.84 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-08-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 12:15:00 | 2094.80 | 2101.75 | 2102.16 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 14:15:00 | 2117.60 | 2104.97 | 2103.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-26 15:15:00 | 2123.95 | 2108.77 | 2105.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 12:15:00 | 2188.45 | 2195.72 | 2175.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 13:00:00 | 2188.45 | 2195.72 | 2175.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 14:15:00 | 2192.45 | 2193.50 | 2178.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 14:30:00 | 2181.80 | 2193.50 | 2178.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 2239.90 | 2244.45 | 2236.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 10:45:00 | 2262.65 | 2247.00 | 2238.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-09 09:15:00 | 2223.65 | 2260.82 | 2265.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 2223.65 | 2260.82 | 2265.73 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 2247.90 | 2238.34 | 2237.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 2259.95 | 2246.17 | 2241.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 15:15:00 | 2250.00 | 2254.40 | 2248.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 09:15:00 | 2239.00 | 2254.40 | 2248.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 2258.50 | 2255.22 | 2249.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:30:00 | 2259.45 | 2255.22 | 2249.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 2262.85 | 2258.84 | 2252.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 12:30:00 | 2255.05 | 2258.84 | 2252.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 2253.60 | 2257.55 | 2253.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:30:00 | 2247.25 | 2257.55 | 2253.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 2253.00 | 2256.64 | 2253.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 2264.25 | 2256.64 | 2253.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 2244.90 | 2264.32 | 2260.88 | SL hit (close<static) qty=1.00 sl=2250.75 alert=retest2 |

### Cycle 105 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 2229.05 | 2254.42 | 2256.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 2218.10 | 2247.16 | 2253.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 09:15:00 | 2164.55 | 2158.68 | 2178.96 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-23 10:30:00 | 2148.05 | 2157.18 | 2176.43 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 2181.30 | 2165.19 | 2174.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-23 14:15:00 | 2181.30 | 2165.19 | 2174.17 | SL hit (close>ema400) qty=1.00 sl=2174.17 alert=retest1 |

### Cycle 106 — BUY (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 10:15:00 | 2201.70 | 2179.52 | 2179.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 2217.45 | 2196.13 | 2187.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 11:15:00 | 2207.95 | 2216.87 | 2207.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 11:15:00 | 2207.95 | 2216.87 | 2207.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 2207.95 | 2216.87 | 2207.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 2207.95 | 2216.87 | 2207.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 2163.50 | 2206.20 | 2203.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 2163.50 | 2206.20 | 2203.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2024-09-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 13:15:00 | 2177.30 | 2200.42 | 2201.08 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 14:15:00 | 2218.55 | 2201.54 | 2199.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 15:15:00 | 2222.20 | 2205.68 | 2201.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 2201.60 | 2204.86 | 2201.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 2201.60 | 2204.86 | 2201.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 2201.60 | 2204.86 | 2201.82 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 2196.05 | 2199.57 | 2199.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 2190.05 | 2197.67 | 2199.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 2201.60 | 2197.85 | 2198.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 2201.60 | 2197.85 | 2198.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 2201.60 | 2197.85 | 2198.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 10:30:00 | 2178.80 | 2194.77 | 2197.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 2183.70 | 2193.34 | 2195.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 10:00:00 | 2186.00 | 2191.87 | 2194.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 11:15:00 | 2186.85 | 2192.12 | 2194.56 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 2181.35 | 2189.96 | 2193.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 12:15:00 | 2176.00 | 2189.96 | 2193.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 13:45:00 | 2165.50 | 2183.37 | 2189.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 09:15:00 | 2167.25 | 2183.77 | 2188.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 09:15:00 | 2205.70 | 2188.16 | 2190.30 | SL hit (close>static) qty=1.00 sl=2194.45 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 2221.50 | 2194.83 | 2193.13 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 2155.60 | 2195.00 | 2196.52 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 2203.25 | 2192.15 | 2191.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 13:15:00 | 2208.15 | 2195.35 | 2192.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 10:15:00 | 2237.10 | 2256.06 | 2236.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 10:15:00 | 2237.10 | 2256.06 | 2236.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 2237.10 | 2256.06 | 2236.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 2237.10 | 2256.06 | 2236.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 2196.25 | 2244.10 | 2233.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 2196.25 | 2244.10 | 2233.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 2153.55 | 2225.99 | 2225.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:00:00 | 2153.55 | 2225.99 | 2225.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 2134.45 | 2207.68 | 2217.58 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 13:15:00 | 2239.60 | 2214.20 | 2211.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 2252.15 | 2230.44 | 2220.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 10:15:00 | 2237.00 | 2243.89 | 2235.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 10:15:00 | 2237.00 | 2243.89 | 2235.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 2237.00 | 2243.89 | 2235.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:00:00 | 2237.00 | 2243.89 | 2235.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 2211.20 | 2237.35 | 2232.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:00:00 | 2211.20 | 2237.35 | 2232.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 2205.45 | 2230.97 | 2230.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 13:00:00 | 2205.45 | 2230.97 | 2230.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2024-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 13:15:00 | 2197.90 | 2224.36 | 2227.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 2167.00 | 2203.76 | 2216.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 2185.00 | 2182.08 | 2196.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 10:15:00 | 2186.50 | 2182.08 | 2196.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 2191.20 | 2179.67 | 2190.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:45:00 | 2193.95 | 2179.67 | 2190.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 2181.70 | 2180.08 | 2189.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:30:00 | 2185.00 | 2180.08 | 2189.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 2182.00 | 2180.46 | 2188.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 2189.35 | 2180.46 | 2188.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 2191.05 | 2182.58 | 2189.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:00:00 | 2169.75 | 2182.13 | 2187.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 13:15:00 | 2138.20 | 2125.64 | 2125.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 13:15:00 | 2138.20 | 2125.64 | 2125.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 09:15:00 | 2172.05 | 2143.38 | 2134.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 2184.95 | 2185.62 | 2164.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-29 09:45:00 | 2176.65 | 2185.62 | 2164.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 2192.45 | 2195.41 | 2180.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-30 10:15:00 | 2205.65 | 2195.41 | 2180.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 13:15:00 | 2164.40 | 2188.79 | 2182.07 | SL hit (close<static) qty=1.00 sl=2167.05 alert=retest2 |

### Cycle 117 — SELL (started 2024-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-31 09:15:00 | 2162.20 | 2175.64 | 2177.03 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-10-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 15:15:00 | 2194.95 | 2177.69 | 2176.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 18:15:00 | 2201.00 | 2185.11 | 2180.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 10:15:00 | 2187.10 | 2187.98 | 2182.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 10:15:00 | 2187.10 | 2187.98 | 2182.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 2187.10 | 2187.98 | 2182.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:45:00 | 2184.40 | 2187.98 | 2182.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 2181.05 | 2186.60 | 2182.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:00:00 | 2181.05 | 2186.60 | 2182.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 2188.60 | 2187.00 | 2183.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 2207.20 | 2187.62 | 2184.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-05 09:15:00 | 2162.65 | 2182.63 | 2182.36 | SL hit (close<static) qty=1.00 sl=2181.05 alert=retest2 |

### Cycle 119 — SELL (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 10:15:00 | 2125.60 | 2171.22 | 2177.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 2086.00 | 2109.71 | 2126.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 2119.55 | 2097.76 | 2110.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 2119.55 | 2097.76 | 2110.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 2119.55 | 2097.76 | 2110.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:30:00 | 2118.35 | 2097.76 | 2110.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 2119.70 | 2102.15 | 2111.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 2118.95 | 2102.15 | 2111.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 2097.00 | 2103.02 | 2110.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:30:00 | 2109.20 | 2103.02 | 2110.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 2032.95 | 2030.85 | 2042.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 13:30:00 | 2040.60 | 2030.85 | 2042.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 2031.45 | 2032.67 | 2040.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 09:30:00 | 2036.55 | 2032.67 | 2040.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 2038.15 | 2033.77 | 2040.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 2038.15 | 2033.77 | 2040.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 2047.10 | 2036.44 | 2041.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:00:00 | 2047.10 | 2036.44 | 2041.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 2050.25 | 2039.20 | 2042.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 15:00:00 | 2036.50 | 2040.64 | 2042.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 11:15:00 | 2051.20 | 2042.64 | 2042.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 11:15:00 | 2051.20 | 2042.64 | 2042.46 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 12:15:00 | 2037.90 | 2041.69 | 2042.04 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2024-11-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 09:15:00 | 2055.95 | 2044.30 | 2043.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 12:15:00 | 2066.55 | 2052.12 | 2047.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 2066.40 | 2089.64 | 2076.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 2066.40 | 2089.64 | 2076.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 2066.40 | 2089.64 | 2076.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:00:00 | 2066.40 | 2089.64 | 2076.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 2074.55 | 2086.62 | 2076.22 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2024-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 14:15:00 | 2032.15 | 2064.55 | 2068.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 09:15:00 | 2025.00 | 2052.55 | 2061.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 09:15:00 | 2019.70 | 2006.01 | 2019.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 09:15:00 | 2019.70 | 2006.01 | 2019.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 2019.70 | 2006.01 | 2019.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:45:00 | 2032.95 | 2006.01 | 2019.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 2049.05 | 2014.62 | 2021.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 2049.05 | 2014.62 | 2021.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 2043.10 | 2020.31 | 2023.69 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 12:15:00 | 2049.95 | 2026.24 | 2026.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 13:15:00 | 2055.60 | 2032.11 | 2028.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 09:15:00 | 2065.70 | 2075.42 | 2066.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 2065.70 | 2075.42 | 2066.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 2065.70 | 2075.42 | 2066.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:45:00 | 2064.15 | 2075.42 | 2066.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 2085.15 | 2077.37 | 2067.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 11:15:00 | 2087.20 | 2077.37 | 2067.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 12:00:00 | 2092.55 | 2080.40 | 2070.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:00:00 | 2093.80 | 2089.04 | 2080.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 14:15:00 | 2122.20 | 2134.75 | 2135.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 2122.20 | 2134.75 | 2135.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 2112.05 | 2130.21 | 2133.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 09:15:00 | 2077.85 | 2058.25 | 2068.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 2077.85 | 2058.25 | 2068.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 2077.85 | 2058.25 | 2068.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:45:00 | 2077.60 | 2058.25 | 2068.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 2102.25 | 2067.05 | 2071.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:00:00 | 2102.25 | 2067.05 | 2071.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — BUY (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 11:15:00 | 2108.85 | 2075.41 | 2075.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 10:15:00 | 2113.00 | 2096.67 | 2087.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 2146.35 | 2154.19 | 2134.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 15:00:00 | 2146.35 | 2154.19 | 2134.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 2156.55 | 2152.39 | 2136.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:30:00 | 2127.75 | 2152.39 | 2136.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 2168.85 | 2163.75 | 2151.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-24 10:15:00 | 2175.50 | 2163.75 | 2151.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 10:30:00 | 2174.40 | 2174.03 | 2164.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-02 14:15:00 | 2393.05 | 2367.96 | 2344.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 2318.80 | 2356.26 | 2358.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 10:15:00 | 2294.80 | 2343.97 | 2352.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 12:15:00 | 2162.10 | 2159.98 | 2187.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 12:45:00 | 2156.55 | 2159.98 | 2187.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 2110.75 | 2106.23 | 2117.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 09:15:00 | 2090.10 | 2108.37 | 2117.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 14:15:00 | 2130.95 | 2114.02 | 2115.17 | SL hit (close>static) qty=1.00 sl=2126.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 15:15:00 | 2132.90 | 2117.80 | 2116.78 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 2103.00 | 2115.49 | 2115.95 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-01-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 11:15:00 | 2119.80 | 2116.36 | 2116.30 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2025-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 14:15:00 | 2102.30 | 2115.76 | 2116.27 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 13:15:00 | 2120.40 | 2115.76 | 2115.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-22 14:15:00 | 2136.65 | 2119.94 | 2117.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 2124.55 | 2141.21 | 2133.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 2124.55 | 2141.21 | 2133.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 2124.55 | 2141.21 | 2133.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 10:00:00 | 2124.55 | 2141.21 | 2133.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 2154.75 | 2143.92 | 2135.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:30:00 | 2160.55 | 2145.05 | 2136.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 2085.70 | 2128.69 | 2131.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 2085.70 | 2128.69 | 2131.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 11:15:00 | 2080.95 | 2112.09 | 2123.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 13:15:00 | 2049.25 | 2048.97 | 2076.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 14:00:00 | 2049.25 | 2048.97 | 2076.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 13:15:00 | 2058.25 | 2043.31 | 2058.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:00:00 | 2058.25 | 2043.31 | 2058.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 2069.90 | 2048.63 | 2059.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 2069.90 | 2048.63 | 2059.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 2070.00 | 2052.90 | 2060.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 2073.15 | 2052.90 | 2060.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 2065.15 | 2063.54 | 2063.68 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 14:15:00 | 2068.25 | 2064.49 | 2064.09 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 15:15:00 | 2053.20 | 2062.23 | 2063.10 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 2074.55 | 2064.69 | 2064.14 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 2043.00 | 2071.45 | 2071.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 2006.55 | 2053.11 | 2062.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 14:15:00 | 2033.60 | 2032.16 | 2046.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-03 15:00:00 | 2033.60 | 2032.16 | 2046.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 2079.05 | 2041.33 | 2048.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 2086.55 | 2041.33 | 2048.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 2084.45 | 2049.95 | 2051.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:45:00 | 2082.70 | 2049.95 | 2051.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 2093.05 | 2058.57 | 2055.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 2109.05 | 2077.45 | 2065.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 14:15:00 | 2184.00 | 2184.95 | 2155.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 15:00:00 | 2184.00 | 2184.95 | 2155.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 2171.65 | 2192.38 | 2178.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:30:00 | 2169.50 | 2192.38 | 2178.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 2165.10 | 2186.92 | 2177.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:45:00 | 2164.20 | 2186.92 | 2177.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 15:15:00 | 2163.10 | 2170.62 | 2171.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 2097.35 | 2155.97 | 2164.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 2113.00 | 2060.63 | 2084.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 2113.00 | 2060.63 | 2084.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 2113.00 | 2060.63 | 2084.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:45:00 | 2133.55 | 2060.63 | 2084.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 2103.35 | 2069.17 | 2086.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:30:00 | 2114.30 | 2069.17 | 2086.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 1985.40 | 2047.70 | 2068.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:30:00 | 1971.95 | 2031.19 | 2059.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 09:15:00 | 1929.45 | 2009.19 | 2011.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:00:00 | 1981.85 | 1987.91 | 1999.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 11:15:00 | 1981.45 | 1983.45 | 1992.15 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1948.70 | 1972.09 | 1982.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 10:45:00 | 1940.20 | 1963.11 | 1977.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 1882.76 | 1932.38 | 1952.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 1941.15 | 1932.38 | 1952.83 | SL hit (close>static) qty=0.50 sl=1932.38 alert=retest2 |

### Cycle 140 — BUY (started 2025-03-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 11:15:00 | 1934.35 | 1908.68 | 1905.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-03 12:15:00 | 1951.85 | 1917.32 | 1909.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 2015.40 | 2020.22 | 2000.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 2015.40 | 2020.22 | 2000.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 2018.60 | 2023.18 | 2012.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:30:00 | 2011.90 | 2023.18 | 2012.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 13:15:00 | 2018.20 | 2022.18 | 2013.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 13:45:00 | 2016.20 | 2022.18 | 2013.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 2011.05 | 2019.96 | 2013.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 2002.80 | 2019.96 | 2013.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 2003.00 | 2016.56 | 2012.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1980.60 | 2016.56 | 2012.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 10:15:00 | 1995.75 | 2008.14 | 2008.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 1969.50 | 1996.29 | 2003.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 1969.35 | 1965.01 | 1979.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 15:00:00 | 1969.35 | 1965.01 | 1979.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1971.80 | 1966.64 | 1977.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:00:00 | 1971.80 | 1966.64 | 1977.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 1974.00 | 1970.39 | 1976.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:45:00 | 1978.25 | 1970.39 | 1976.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 13:15:00 | 1978.00 | 1971.91 | 1976.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 14:00:00 | 1978.00 | 1971.91 | 1976.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 14:15:00 | 1969.25 | 1971.38 | 1976.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 15:15:00 | 1960.10 | 1971.38 | 1976.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 1998.80 | 1975.06 | 1976.87 | SL hit (close>static) qty=1.00 sl=1978.05 alert=retest2 |

### Cycle 142 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 2004.10 | 1980.87 | 1979.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 11:15:00 | 2004.60 | 1985.61 | 1981.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-17 15:15:00 | 1988.10 | 1990.14 | 1985.54 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 09:15:00 | 2007.25 | 1990.14 | 1985.54 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-21 09:15:00 | 2107.61 | 2083.40 | 2060.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-21 12:15:00 | 2085.15 | 2088.60 | 2068.98 | SL hit (close<ema200) qty=0.50 sl=2088.60 alert=retest1 |

### Cycle 143 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 2067.00 | 2090.02 | 2090.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 2066.00 | 2084.71 | 2087.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 14:15:00 | 2031.60 | 2016.80 | 2031.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 14:15:00 | 2031.60 | 2016.80 | 2031.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 2031.60 | 2016.80 | 2031.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 2031.60 | 2016.80 | 2031.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 2024.00 | 2018.24 | 2030.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 2007.95 | 2018.24 | 2030.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 14:30:00 | 2008.40 | 1993.04 | 1997.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 2100.00 | 2017.51 | 2008.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 2100.00 | 2017.51 | 2008.16 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 1980.00 | 2018.50 | 2023.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1909.25 | 1974.22 | 1998.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 1952.40 | 1942.26 | 1964.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 1952.40 | 1942.26 | 1964.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 1978.30 | 1949.47 | 1965.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 1978.30 | 1949.47 | 1965.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1982.30 | 1956.03 | 1967.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:45:00 | 1982.30 | 1956.03 | 1967.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1980.80 | 1965.01 | 1969.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1929.75 | 1969.00 | 1970.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 09:30:00 | 1970.45 | 1937.52 | 1948.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 12:15:00 | 1978.75 | 1955.48 | 1954.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 1978.75 | 1955.48 | 1954.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 2018.60 | 1974.30 | 1964.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 11:15:00 | 1983.90 | 2000.38 | 1987.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 11:15:00 | 1983.90 | 2000.38 | 1987.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 11:15:00 | 1983.90 | 2000.38 | 1987.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 11:30:00 | 1968.20 | 2000.38 | 1987.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 12:15:00 | 1951.60 | 1990.63 | 1984.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:00:00 | 1951.60 | 1990.63 | 1984.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 1943.00 | 1981.10 | 1980.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:45:00 | 1945.80 | 1981.10 | 1980.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 14:15:00 | 1932.70 | 1971.42 | 1976.34 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2025-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 11:15:00 | 1998.50 | 1962.68 | 1961.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 12:15:00 | 2006.40 | 1971.42 | 1965.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 2062.50 | 2092.39 | 2075.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 2062.50 | 2092.39 | 2075.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2062.50 | 2092.39 | 2075.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 2062.50 | 2092.39 | 2075.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 2036.90 | 2081.29 | 2072.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 2036.90 | 2081.29 | 2072.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-04-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 13:15:00 | 2034.40 | 2061.21 | 2064.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 14:15:00 | 2018.50 | 2052.67 | 2060.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 2096.00 | 2056.43 | 2060.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 2096.00 | 2056.43 | 2060.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 2096.00 | 2056.43 | 2060.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 2096.00 | 2056.43 | 2060.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 2103.70 | 2065.88 | 2064.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 2111.60 | 2075.03 | 2068.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 09:15:00 | 2071.00 | 2088.08 | 2079.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 09:15:00 | 2071.00 | 2088.08 | 2079.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 2071.00 | 2088.08 | 2079.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:45:00 | 2072.90 | 2088.08 | 2079.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 2083.70 | 2087.20 | 2079.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 10:30:00 | 2071.20 | 2087.20 | 2079.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 11:15:00 | 2091.40 | 2088.04 | 2080.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 2095.10 | 2079.92 | 2078.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 11:15:00 | 2067.00 | 2089.55 | 2088.08 | SL hit (close<static) qty=1.00 sl=2078.50 alert=retest2 |

### Cycle 151 — SELL (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 12:15:00 | 2057.90 | 2083.22 | 2085.34 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 13:15:00 | 2090.80 | 2082.97 | 2082.89 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 09:15:00 | 2047.00 | 2075.96 | 2079.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 2008.00 | 2035.49 | 2049.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 11:15:00 | 2022.30 | 2020.87 | 2038.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 12:00:00 | 2022.30 | 2020.87 | 2038.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 2042.60 | 2023.94 | 2036.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:00:00 | 2042.60 | 2023.94 | 2036.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 2041.50 | 2027.45 | 2036.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 1979.50 | 2028.96 | 2036.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 10:45:00 | 2019.80 | 2024.91 | 2033.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 09:15:00 | 2088.70 | 2042.69 | 2037.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — BUY (started 2025-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 09:15:00 | 2088.70 | 2042.69 | 2037.99 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 13:15:00 | 2051.00 | 2064.53 | 2064.55 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 2069.40 | 2065.51 | 2064.99 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 10:15:00 | 2055.10 | 2063.42 | 2064.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 11:15:00 | 2050.20 | 2060.78 | 2062.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 2013.40 | 1999.00 | 2020.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 09:30:00 | 2006.00 | 1999.00 | 2020.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1995.60 | 1978.57 | 1983.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:30:00 | 1992.40 | 1978.57 | 1983.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 1994.60 | 1981.78 | 1984.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:45:00 | 2000.20 | 1981.78 | 1984.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1972.00 | 1980.26 | 1983.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:30:00 | 1982.50 | 1980.26 | 1983.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1985.00 | 1981.21 | 1983.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 1985.00 | 1981.21 | 1983.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 1987.90 | 1982.54 | 1983.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 11:30:00 | 1973.90 | 1981.04 | 1982.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 13:30:00 | 1973.50 | 1977.04 | 1980.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 10:15:00 | 1976.10 | 1956.18 | 1953.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — BUY (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 10:15:00 | 1976.10 | 1956.18 | 1953.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 1995.00 | 1968.05 | 1959.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 1994.00 | 1995.00 | 1984.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 1994.00 | 1995.00 | 1984.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 1986.20 | 1991.60 | 1985.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:00:00 | 1986.20 | 1991.60 | 1985.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 14:15:00 | 1995.10 | 1992.30 | 1986.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 14:45:00 | 1987.80 | 1992.30 | 1986.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2004.60 | 1995.22 | 1988.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 1993.80 | 1995.22 | 1988.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 2016.00 | 2028.68 | 2018.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 2016.00 | 2028.68 | 2018.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 2022.10 | 2027.37 | 2018.57 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 2000.40 | 2015.06 | 2015.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 12:15:00 | 1995.50 | 2011.14 | 2013.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 2011.90 | 2003.91 | 2007.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 11:15:00 | 2011.90 | 2003.91 | 2007.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 2011.90 | 2003.91 | 2007.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:00:00 | 2011.90 | 2003.91 | 2007.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 2008.20 | 2004.77 | 2007.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 2013.00 | 2004.77 | 2007.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 2009.60 | 2005.74 | 2008.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 2008.90 | 2005.74 | 2008.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 2009.50 | 2006.49 | 2008.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 2009.50 | 2006.49 | 2008.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 2013.00 | 2007.79 | 2008.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 1994.30 | 2007.79 | 2008.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1974.70 | 2001.17 | 2005.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 1966.00 | 1993.54 | 2001.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 11:15:00 | 1957.40 | 1942.67 | 1942.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 160 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 1957.40 | 1942.67 | 1942.22 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 14:15:00 | 1929.60 | 1940.57 | 1941.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 11:15:00 | 1924.00 | 1934.81 | 1938.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 14:15:00 | 1934.90 | 1933.58 | 1936.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-25 15:15:00 | 1933.60 | 1933.58 | 1936.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 1933.60 | 1933.59 | 1936.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 1935.10 | 1933.59 | 1936.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 1930.30 | 1932.93 | 1935.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 10:30:00 | 1924.70 | 1931.12 | 1934.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 10:15:00 | 1948.20 | 1932.71 | 1932.72 | SL hit (close>static) qty=1.00 sl=1941.10 alert=retest2 |

### Cycle 162 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 1940.80 | 1934.33 | 1933.45 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 1931.70 | 1934.00 | 1934.06 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 1941.00 | 1935.40 | 1934.69 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 1922.70 | 1933.51 | 1934.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 1918.90 | 1930.59 | 1932.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 13:15:00 | 1949.10 | 1931.38 | 1932.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 13:15:00 | 1949.10 | 1931.38 | 1932.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1949.10 | 1931.38 | 1932.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:00:00 | 1949.10 | 1931.38 | 1932.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 14:15:00 | 1959.50 | 1937.01 | 1934.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 15:15:00 | 1966.70 | 1942.94 | 1937.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 13:15:00 | 1951.20 | 1956.44 | 1947.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 13:15:00 | 1951.20 | 1956.44 | 1947.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1951.20 | 1956.44 | 1947.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:30:00 | 1946.90 | 1956.44 | 1947.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1969.70 | 1959.10 | 1949.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:45:00 | 1971.60 | 1959.61 | 1954.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1924.10 | 1966.06 | 1967.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 1924.10 | 1966.06 | 1967.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 1920.00 | 1956.85 | 1963.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 1943.20 | 1934.91 | 1946.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 1943.20 | 1934.91 | 1946.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1943.20 | 1934.91 | 1946.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 1961.00 | 1934.91 | 1946.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1933.40 | 1934.61 | 1945.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 1938.00 | 1934.61 | 1945.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1917.60 | 1898.72 | 1905.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 1914.30 | 1898.72 | 1905.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1925.10 | 1904.00 | 1907.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 1929.10 | 1904.00 | 1907.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — BUY (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 12:15:00 | 1925.00 | 1911.86 | 1910.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 1951.50 | 1928.60 | 1920.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 1945.00 | 1945.27 | 1934.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 12:15:00 | 1941.10 | 1945.27 | 1934.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1932.50 | 1942.72 | 1934.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 1932.50 | 1942.72 | 1934.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 1932.40 | 1940.66 | 1934.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1948.70 | 1936.96 | 1933.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 10:15:00 | 1929.00 | 1944.20 | 1941.50 | SL hit (close<static) qty=1.00 sl=1929.30 alert=retest2 |

### Cycle 169 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1933.90 | 1939.42 | 1939.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 1931.10 | 1937.76 | 1938.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 12:15:00 | 1912.00 | 1907.97 | 1919.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 13:00:00 | 1912.00 | 1907.97 | 1919.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1918.30 | 1908.50 | 1916.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1918.30 | 1908.50 | 1916.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1909.10 | 1908.62 | 1915.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:45:00 | 1908.40 | 1909.09 | 1914.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 14:15:00 | 1919.90 | 1911.25 | 1914.64 | SL hit (close>static) qty=1.00 sl=1918.50 alert=retest2 |

### Cycle 170 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 1927.80 | 1916.94 | 1916.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 11:15:00 | 1948.90 | 1923.33 | 1919.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 12:15:00 | 1951.50 | 1957.64 | 1947.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 13:00:00 | 1951.50 | 1957.64 | 1947.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1960.20 | 1958.82 | 1951.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 10:30:00 | 1965.60 | 1960.09 | 1952.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 11:30:00 | 1967.00 | 1962.10 | 1953.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 1928.40 | 1972.10 | 1969.63 | SL hit (close<static) qty=1.00 sl=1947.70 alert=retest2 |

### Cycle 171 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 1933.70 | 1964.42 | 1966.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 1925.00 | 1942.24 | 1953.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 15:15:00 | 1883.00 | 1881.96 | 1898.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 09:15:00 | 1872.50 | 1881.96 | 1898.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1858.30 | 1866.36 | 1879.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 14:00:00 | 1842.00 | 1855.50 | 1869.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-07 09:15:00 | 1912.80 | 1865.85 | 1870.95 | SL hit (close>static) qty=1.00 sl=1908.00 alert=retest2 |

### Cycle 172 — BUY (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 11:15:00 | 1889.20 | 1876.16 | 1875.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 12:15:00 | 1927.80 | 1886.49 | 1879.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 13:15:00 | 1922.50 | 1922.70 | 1907.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 14:00:00 | 1922.50 | 1922.70 | 1907.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1918.80 | 1920.73 | 1910.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 10:30:00 | 1933.30 | 1923.79 | 1912.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 10:15:00 | 1958.80 | 1964.98 | 1965.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 10:15:00 | 1958.80 | 1964.98 | 1965.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 11:15:00 | 1941.60 | 1960.31 | 1963.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1957.70 | 1949.14 | 1954.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 1957.70 | 1949.14 | 1954.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1957.70 | 1949.14 | 1954.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 1961.20 | 1949.14 | 1954.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1963.60 | 1952.03 | 1955.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 1963.60 | 1952.03 | 1955.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1960.00 | 1953.62 | 1956.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 1963.40 | 1953.62 | 1956.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 14:15:00 | 1963.50 | 1957.50 | 1957.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-22 09:15:00 | 1970.90 | 1961.81 | 1959.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 13:15:00 | 1970.50 | 1973.68 | 1969.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 13:15:00 | 1970.50 | 1973.68 | 1969.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1970.50 | 1973.68 | 1969.31 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1926.70 | 1961.51 | 1964.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 1913.90 | 1937.82 | 1951.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1909.40 | 1905.38 | 1920.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1909.40 | 1905.38 | 1920.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1920.30 | 1906.02 | 1913.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:00:00 | 1920.30 | 1906.02 | 1913.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 1924.40 | 1909.69 | 1914.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 1923.90 | 1909.69 | 1914.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1909.00 | 1912.47 | 1915.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 1916.00 | 1912.47 | 1915.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1910.90 | 1899.36 | 1904.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:00:00 | 1910.90 | 1899.36 | 1904.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1925.00 | 1904.49 | 1906.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 1925.00 | 1904.49 | 1906.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 1930.30 | 1909.65 | 1908.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 12:15:00 | 1933.00 | 1914.32 | 1910.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 10:15:00 | 1938.70 | 1940.85 | 1932.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 11:00:00 | 1938.70 | 1940.85 | 1932.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1938.10 | 1939.97 | 1933.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 1938.10 | 1939.97 | 1933.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1951.90 | 1944.61 | 1938.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 12:45:00 | 1955.90 | 1948.53 | 1941.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 15:00:00 | 1956.40 | 1950.37 | 1946.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 2023.00 | 2038.43 | 2040.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 2023.00 | 2038.43 | 2040.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 2010.70 | 2032.88 | 2037.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 2007.10 | 2006.78 | 2015.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 12:00:00 | 2007.10 | 2006.78 | 2015.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2011.70 | 2002.20 | 2009.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:15:00 | 1998.40 | 2004.25 | 2009.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 1983.00 | 1950.52 | 1946.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 1983.00 | 1950.52 | 1946.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 1985.80 | 1957.57 | 1950.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 1969.60 | 1976.82 | 1967.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 14:00:00 | 1969.60 | 1976.82 | 1967.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1933.80 | 1967.47 | 1965.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:00:00 | 1933.80 | 1967.47 | 1965.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 1938.10 | 1961.60 | 1962.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 09:15:00 | 1923.10 | 1940.54 | 1950.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1983.00 | 1927.49 | 1930.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1983.00 | 1927.49 | 1930.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1983.00 | 1927.49 | 1930.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:00:00 | 1983.00 | 1927.49 | 1930.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 180 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 1970.00 | 1935.99 | 1934.29 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 1935.10 | 1952.44 | 1954.17 | EMA200 below EMA400 |

### Cycle 182 — BUY (started 2025-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 14:15:00 | 1953.20 | 1948.00 | 1947.83 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 09:15:00 | 1933.00 | 1945.42 | 1946.71 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 1949.00 | 1944.78 | 1944.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 12:15:00 | 1955.00 | 1947.40 | 1945.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 1937.30 | 1945.73 | 1945.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 1937.30 | 1945.73 | 1945.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1937.30 | 1945.73 | 1945.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1937.30 | 1945.73 | 1945.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1945.00 | 1945.59 | 1945.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1935.90 | 1945.59 | 1945.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 1938.00 | 1944.07 | 1944.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 1926.70 | 1937.58 | 1941.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 1919.20 | 1915.22 | 1923.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 1919.20 | 1915.22 | 1923.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1919.20 | 1915.22 | 1923.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 1919.20 | 1915.22 | 1923.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1927.30 | 1917.63 | 1923.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 1927.60 | 1917.63 | 1923.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1930.70 | 1920.25 | 1924.12 | EMA400 retest candle locked (from downside) |

### Cycle 186 — BUY (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 11:15:00 | 1949.00 | 1929.73 | 1927.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 1956.20 | 1944.31 | 1936.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1934.70 | 1942.39 | 1936.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1934.70 | 1942.39 | 1936.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1934.70 | 1942.39 | 1936.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 1934.70 | 1942.39 | 1936.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1935.50 | 1941.01 | 1936.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 1931.70 | 1941.01 | 1936.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1935.40 | 1939.89 | 1936.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:30:00 | 1934.00 | 1939.89 | 1936.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 1944.00 | 1940.71 | 1936.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 1949.10 | 1942.45 | 1938.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 10:15:00 | 1955.00 | 1942.14 | 1938.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 13:15:00 | 1955.90 | 1973.91 | 1974.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 13:15:00 | 1955.90 | 1973.91 | 1974.20 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 09:15:00 | 1996.00 | 1975.92 | 1974.77 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 14:15:00 | 1970.00 | 1974.45 | 1974.76 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 2009.50 | 1980.59 | 1977.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 11:15:00 | 2022.50 | 1991.28 | 1982.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 14:15:00 | 1985.90 | 1998.35 | 1989.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 14:15:00 | 1985.90 | 1998.35 | 1989.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1985.90 | 1998.35 | 1989.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 1985.90 | 1998.35 | 1989.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 1985.00 | 1995.68 | 1988.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 1976.90 | 1995.68 | 1988.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1980.30 | 1992.60 | 1987.90 | EMA400 retest candle locked (from upside) |

### Cycle 191 — SELL (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 13:15:00 | 1979.90 | 1984.89 | 1985.23 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1990.00 | 1985.56 | 1985.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 12:15:00 | 2014.10 | 1993.85 | 1989.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 11:15:00 | 2046.60 | 2048.70 | 2033.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 11:45:00 | 2046.60 | 2048.70 | 2033.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 2050.70 | 2052.28 | 2045.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:45:00 | 2046.50 | 2052.28 | 2045.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 2048.50 | 2051.85 | 2046.22 | EMA400 retest candle locked (from upside) |

### Cycle 193 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 2040.30 | 2044.77 | 2045.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 2030.00 | 2041.06 | 2043.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 10:15:00 | 2036.80 | 2035.53 | 2039.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:30:00 | 2036.00 | 2035.53 | 2039.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 2035.00 | 2035.42 | 2039.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 2035.00 | 2035.42 | 2039.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 2034.70 | 2035.37 | 2038.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:30:00 | 2036.10 | 2035.37 | 2038.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 13:15:00 | 2035.60 | 2032.51 | 2035.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 13:30:00 | 2036.60 | 2032.51 | 2035.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 2028.90 | 2031.79 | 2034.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 2002.30 | 2031.63 | 2034.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 2037.90 | 2008.26 | 2016.40 | SL hit (close>static) qty=1.00 sl=2035.60 alert=retest2 |

### Cycle 194 — BUY (started 2025-11-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 12:15:00 | 2043.00 | 2024.75 | 2022.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 2056.90 | 2040.56 | 2032.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 13:15:00 | 2062.20 | 2065.80 | 2054.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 14:00:00 | 2062.20 | 2065.80 | 2054.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 2075.00 | 2078.26 | 2070.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:00:00 | 2075.00 | 2078.26 | 2070.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 2071.50 | 2076.91 | 2070.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:45:00 | 2070.10 | 2076.91 | 2070.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 2081.00 | 2077.73 | 2071.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 2070.10 | 2077.73 | 2071.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 2076.10 | 2077.40 | 2071.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:45:00 | 2076.90 | 2077.40 | 2071.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 2085.10 | 2080.00 | 2075.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:15:00 | 2088.20 | 2081.33 | 2076.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:15:00 | 2089.50 | 2082.96 | 2078.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 2069.30 | 2081.28 | 2078.39 | SL hit (close<static) qty=1.00 sl=2074.50 alert=retest2 |

### Cycle 195 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 2071.90 | 2086.13 | 2086.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 2067.20 | 2082.34 | 2085.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 09:15:00 | 2070.20 | 2062.60 | 2069.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 09:15:00 | 2070.20 | 2062.60 | 2069.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 2070.20 | 2062.60 | 2069.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 2070.20 | 2062.60 | 2069.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 2059.70 | 2062.02 | 2068.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 2067.60 | 2062.02 | 2068.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 2056.40 | 2060.46 | 2066.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:30:00 | 2065.00 | 2060.46 | 2066.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 2070.20 | 2060.77 | 2064.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 2071.20 | 2060.77 | 2064.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 2077.60 | 2064.13 | 2065.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 2077.60 | 2064.13 | 2065.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 196 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 2077.80 | 2066.87 | 2066.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 2097.10 | 2078.10 | 2072.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 2086.40 | 2097.91 | 2087.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 2086.40 | 2097.91 | 2087.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2086.40 | 2097.91 | 2087.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 2086.40 | 2097.91 | 2087.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 2091.50 | 2096.62 | 2088.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 12:00:00 | 2096.40 | 2096.58 | 2088.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 2095.70 | 2092.32 | 2089.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 2085.90 | 2090.74 | 2089.93 | SL hit (close<static) qty=1.00 sl=2086.40 alert=retest2 |

### Cycle 197 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 2111.30 | 2118.33 | 2118.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 15:15:00 | 2108.50 | 2115.53 | 2117.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 13:15:00 | 2115.00 | 2107.05 | 2109.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 13:15:00 | 2115.00 | 2107.05 | 2109.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 2115.00 | 2107.05 | 2109.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:00:00 | 2115.00 | 2107.05 | 2109.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 2114.00 | 2108.44 | 2110.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 2112.00 | 2108.44 | 2110.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 2100.60 | 2106.87 | 2109.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 10:15:00 | 2099.00 | 2105.80 | 2108.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 2093.80 | 2103.40 | 2107.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 2110.00 | 2092.26 | 2091.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 2110.00 | 2092.26 | 2091.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 2120.00 | 2105.85 | 2100.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 2100.40 | 2105.20 | 2101.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 12:15:00 | 2100.40 | 2105.20 | 2101.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 2100.40 | 2105.20 | 2101.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:45:00 | 2096.50 | 2105.20 | 2101.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 2106.30 | 2105.42 | 2101.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 14:15:00 | 2109.70 | 2105.42 | 2101.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 2098.90 | 2104.40 | 2102.21 | SL hit (close<static) qty=1.00 sl=2099.50 alert=retest2 |

### Cycle 199 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 2076.70 | 2096.85 | 2099.27 | EMA200 below EMA400 |

### Cycle 200 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 2122.50 | 2100.93 | 2099.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 13:15:00 | 2134.00 | 2107.55 | 2102.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 2191.30 | 2192.85 | 2165.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 2191.30 | 2192.85 | 2165.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 2178.30 | 2190.48 | 2176.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 2164.90 | 2190.48 | 2176.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 2184.80 | 2189.34 | 2177.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 2177.40 | 2189.34 | 2177.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 2179.90 | 2184.92 | 2178.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 13:30:00 | 2179.70 | 2184.92 | 2178.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 2178.00 | 2183.54 | 2178.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:30:00 | 2183.40 | 2183.54 | 2178.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 15:15:00 | 2178.00 | 2182.43 | 2178.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:15:00 | 2144.60 | 2182.43 | 2178.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 2157.40 | 2177.42 | 2176.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 2156.00 | 2177.42 | 2176.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 2166.10 | 2175.16 | 2175.27 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2026-01-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-12 13:15:00 | 2182.50 | 2176.63 | 2175.83 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 11:15:00 | 2159.40 | 2173.06 | 2174.84 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 2189.40 | 2175.86 | 2175.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 11:15:00 | 2197.80 | 2180.25 | 2177.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 12:15:00 | 2194.30 | 2198.76 | 2190.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 13:00:00 | 2194.30 | 2198.76 | 2190.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 13:15:00 | 2180.00 | 2195.01 | 2189.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:00:00 | 2180.00 | 2195.01 | 2189.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 2177.60 | 2191.53 | 2188.63 | EMA400 retest candle locked (from upside) |

### Cycle 205 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 2168.40 | 2184.11 | 2185.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 13:15:00 | 2135.70 | 2160.63 | 2168.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2187.40 | 2159.92 | 2165.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2187.40 | 2159.92 | 2165.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2187.40 | 2159.92 | 2165.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 2187.40 | 2159.92 | 2165.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2181.40 | 2164.21 | 2167.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 2171.10 | 2168.88 | 2169.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 2142.00 | 2133.19 | 2133.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 206 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 2142.00 | 2133.19 | 2133.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 2147.10 | 2135.97 | 2134.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 2130.00 | 2138.25 | 2136.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 2130.00 | 2138.25 | 2136.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2130.00 | 2138.25 | 2136.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 2128.80 | 2138.25 | 2136.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 2144.00 | 2139.40 | 2136.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:15:00 | 2131.30 | 2139.40 | 2136.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 207 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 2114.50 | 2134.42 | 2134.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 13:15:00 | 2108.40 | 2126.68 | 2131.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 14:15:00 | 2138.60 | 2129.07 | 2131.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 14:15:00 | 2138.60 | 2129.07 | 2131.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 2138.60 | 2129.07 | 2131.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 2138.60 | 2129.07 | 2131.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 2131.00 | 2129.45 | 2131.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 2118.00 | 2129.45 | 2131.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2112.60 | 2126.08 | 2129.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:15:00 | 2097.10 | 2126.08 | 2129.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 2168.70 | 2125.08 | 2124.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2168.70 | 2125.08 | 2124.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 10:15:00 | 2229.50 | 2200.56 | 2180.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 2173.80 | 2203.36 | 2191.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 2173.80 | 2203.36 | 2191.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 2173.80 | 2203.36 | 2191.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 2173.80 | 2203.36 | 2191.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 2178.00 | 2198.29 | 2190.47 | EMA400 retest candle locked (from upside) |

### Cycle 209 — SELL (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 14:15:00 | 2172.80 | 2185.08 | 2186.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 15:15:00 | 2162.10 | 2180.48 | 2183.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2190.00 | 2182.39 | 2184.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 2190.00 | 2182.39 | 2184.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2190.00 | 2182.39 | 2184.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 2190.00 | 2182.39 | 2184.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 2199.60 | 2185.83 | 2185.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 2205.30 | 2189.72 | 2187.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-09 12:15:00 | 2185.00 | 2188.78 | 2187.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 12:15:00 | 2185.00 | 2188.78 | 2187.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 2185.00 | 2188.78 | 2187.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:45:00 | 2186.20 | 2188.78 | 2187.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 2190.30 | 2189.08 | 2187.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 15:15:00 | 2208.70 | 2189.15 | 2187.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:45:00 | 2201.90 | 2205.38 | 2198.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:30:00 | 2204.60 | 2210.16 | 2207.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:30:00 | 2205.40 | 2205.33 | 2205.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 2210.30 | 2206.33 | 2205.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:45:00 | 2203.50 | 2206.33 | 2205.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 2179.90 | 2202.43 | 2204.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 2179.90 | 2202.43 | 2204.15 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 2220.90 | 2205.52 | 2203.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 2254.00 | 2223.16 | 2214.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 10:15:00 | 2239.50 | 2240.84 | 2230.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:30:00 | 2242.60 | 2240.84 | 2230.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 2242.50 | 2241.17 | 2231.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:30:00 | 2231.80 | 2241.17 | 2231.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2227.70 | 2238.76 | 2232.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:45:00 | 2232.70 | 2238.76 | 2232.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 2230.80 | 2237.17 | 2232.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 2226.30 | 2237.17 | 2232.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2222.20 | 2234.18 | 2231.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 2240.20 | 2233.20 | 2231.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:15:00 | 2242.40 | 2233.20 | 2231.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 2218.40 | 2230.13 | 2231.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 2218.40 | 2230.13 | 2231.24 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 2248.80 | 2232.06 | 2230.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 10:15:00 | 2252.30 | 2239.97 | 2235.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 2305.00 | 2308.01 | 2290.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:00:00 | 2305.00 | 2308.01 | 2290.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 12:15:00 | 2322.00 | 2309.91 | 2295.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:45:00 | 2329.10 | 2312.33 | 2298.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 2274.80 | 2301.34 | 2296.31 | SL hit (close<static) qty=1.00 sl=2291.70 alert=retest2 |

### Cycle 215 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 2278.30 | 2293.31 | 2294.51 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-03-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 15:15:00 | 2307.00 | 2295.68 | 2294.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 2344.70 | 2305.48 | 2299.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 15:15:00 | 2331.80 | 2338.56 | 2327.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 09:15:00 | 2288.00 | 2338.56 | 2327.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2303.20 | 2331.49 | 2325.17 | EMA400 retest candle locked (from upside) |

### Cycle 217 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 2300.00 | 2317.74 | 2319.73 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 2338.30 | 2320.22 | 2319.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 2367.10 | 2339.13 | 2329.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 2339.90 | 2348.92 | 2339.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 14:15:00 | 2339.90 | 2348.92 | 2339.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 2339.90 | 2348.92 | 2339.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 2339.90 | 2348.92 | 2339.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 2337.00 | 2346.54 | 2339.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 2312.80 | 2346.54 | 2339.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 2317.10 | 2340.65 | 2337.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 2319.90 | 2340.65 | 2337.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 2334.50 | 2339.42 | 2336.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 2340.00 | 2340.26 | 2337.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 10:30:00 | 2342.90 | 2344.07 | 2341.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 2345.20 | 2344.07 | 2341.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 2317.00 | 2336.64 | 2338.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 219 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 2317.00 | 2336.64 | 2338.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 15:15:00 | 2306.00 | 2325.81 | 2332.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 2304.10 | 2296.15 | 2308.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 2304.10 | 2296.15 | 2308.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 2304.10 | 2296.15 | 2308.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 2304.80 | 2296.15 | 2308.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 2296.50 | 2296.22 | 2307.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 2288.70 | 2296.22 | 2307.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:45:00 | 2294.80 | 2295.99 | 2305.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:15:00 | 2293.40 | 2295.99 | 2305.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 13:15:00 | 2311.50 | 2304.45 | 2305.33 | SL hit (close>static) qty=1.00 sl=2310.00 alert=retest2 |

### Cycle 220 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 2319.00 | 2292.68 | 2291.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 2327.50 | 2299.65 | 2294.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 2285.90 | 2303.68 | 2298.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 09:15:00 | 2285.90 | 2303.68 | 2298.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 2285.90 | 2303.68 | 2298.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 2285.90 | 2303.68 | 2298.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 2293.80 | 2301.70 | 2298.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 13:15:00 | 2300.60 | 2299.03 | 2297.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 2317.80 | 2298.95 | 2298.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 2291.00 | 2296.28 | 2296.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 221 — SELL (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 10:15:00 | 2291.00 | 2296.28 | 2296.99 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 2312.70 | 2299.57 | 2298.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 12:15:00 | 2335.10 | 2306.67 | 2301.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 11:15:00 | 2344.80 | 2344.90 | 2333.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 12:00:00 | 2344.80 | 2344.90 | 2333.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 2334.30 | 2342.70 | 2335.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 2334.30 | 2342.70 | 2335.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 2331.90 | 2340.54 | 2334.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 2316.50 | 2340.54 | 2334.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 2307.00 | 2333.83 | 2332.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 2307.00 | 2333.83 | 2332.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 2304.30 | 2327.93 | 2329.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 10:15:00 | 2289.00 | 2314.23 | 2321.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 2271.70 | 2261.65 | 2282.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 2271.70 | 2261.65 | 2282.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 2265.00 | 2264.65 | 2279.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 2251.10 | 2264.65 | 2279.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 2250.60 | 2261.84 | 2277.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 09:15:00 | 2287.00 | 2268.17 | 2271.61 | SL hit (close>static) qty=1.00 sl=2286.00 alert=retest2 |

### Cycle 224 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 2298.70 | 2274.28 | 2274.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 2312.90 | 2297.51 | 2291.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 11:15:00 | 2320.00 | 2322.60 | 2311.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 11:45:00 | 2317.40 | 2322.60 | 2311.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 2313.60 | 2321.69 | 2313.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 15:00:00 | 2313.60 | 2321.69 | 2313.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 2310.90 | 2319.54 | 2313.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 2334.00 | 2319.54 | 2313.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 10:00:00 | 2319.40 | 2332.65 | 2326.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 11:00:00 | 2319.00 | 2329.92 | 2325.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 11:15:00 | 2298.40 | 2323.62 | 2322.98 | SL hit (close<static) qty=1.00 sl=2310.20 alert=retest2 |

### Cycle 225 — SELL (started 2026-04-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 12:15:00 | 2304.20 | 2319.73 | 2321.27 | EMA200 below EMA400 |

### Cycle 226 — BUY (started 2026-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 09:15:00 | 2333.00 | 2323.03 | 2322.22 | EMA200 above EMA400 |

### Cycle 227 — SELL (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 11:15:00 | 2304.00 | 2318.77 | 2320.40 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-04-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 09:15:00 | 2332.80 | 2321.59 | 2320.77 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-04-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 13:15:00 | 2314.70 | 2323.27 | 2323.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 2296.30 | 2314.81 | 2319.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 12:15:00 | 2315.00 | 2313.66 | 2317.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 12:45:00 | 2315.20 | 2313.66 | 2317.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 2303.00 | 2311.52 | 2316.16 | EMA400 retest candle locked (from downside) |

### Cycle 230 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 2378.50 | 2324.35 | 2320.77 | EMA200 above EMA400 |

### Cycle 231 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 2292.20 | 2326.91 | 2327.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 2269.40 | 2315.41 | 2322.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 2331.60 | 2306.34 | 2313.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 2331.60 | 2306.34 | 2313.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 2331.60 | 2306.34 | 2313.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 2331.60 | 2306.34 | 2313.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 2318.50 | 2308.77 | 2313.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:15:00 | 2308.00 | 2308.77 | 2313.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 11:45:00 | 2315.80 | 2311.18 | 2314.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 12:45:00 | 2315.00 | 2312.30 | 2314.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 14:15:00 | 2323.20 | 2316.58 | 2316.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 232 — BUY (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 14:15:00 | 2323.20 | 2316.58 | 2316.30 | EMA200 above EMA400 |

### Cycle 233 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 10:15:00 | 2278.50 | 2308.76 | 2312.84 | EMA200 below EMA400 |

### Cycle 234 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 2320.40 | 2311.61 | 2310.93 | EMA200 above EMA400 |

### Cycle 235 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 2303.00 | 2310.02 | 2310.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 2292.40 | 2306.49 | 2308.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 12:15:00 | 2304.70 | 2301.29 | 2305.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 12:15:00 | 2304.70 | 2301.29 | 2305.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 2304.70 | 2301.29 | 2305.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:00:00 | 2304.70 | 2301.29 | 2305.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 2304.20 | 2301.87 | 2305.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:15:00 | 2305.70 | 2301.87 | 2305.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 2304.10 | 2302.32 | 2305.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 15:00:00 | 2304.10 | 2302.32 | 2305.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 2312.30 | 2304.32 | 2305.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 2312.50 | 2304.32 | 2305.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 236 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 2338.40 | 2311.13 | 2308.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 11:15:00 | 2349.70 | 2323.99 | 2315.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 2337.90 | 2338.11 | 2326.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 09:30:00 | 2350.70 | 2338.11 | 2326.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 2336.00 | 2338.74 | 2329.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 12:00:00 | 2336.00 | 2338.74 | 2329.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 2434.60 | 2449.61 | 2422.49 | EMA400 retest candle locked (from upside) |

### Cycle 237 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 2380.10 | 2407.08 | 2409.49 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-03 09:15:00 | 1665.00 | 2024-05-06 09:15:00 | 1646.95 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-05-03 13:00:00 | 1656.90 | 2024-05-07 09:15:00 | 1634.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-05-03 13:30:00 | 1655.00 | 2024-05-07 09:15:00 | 1634.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-05-03 14:15:00 | 1654.90 | 2024-05-07 09:15:00 | 1634.50 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-05-03 15:15:00 | 1662.00 | 2024-05-07 09:15:00 | 1634.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-05-06 10:30:00 | 1659.95 | 2024-05-07 09:15:00 | 1634.50 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-05-06 13:30:00 | 1657.80 | 2024-05-07 09:15:00 | 1634.50 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-05-06 14:45:00 | 1678.60 | 2024-05-07 09:15:00 | 1634.50 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2024-05-08 10:45:00 | 1599.15 | 2024-05-13 09:15:00 | 1617.80 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2024-05-09 10:00:00 | 1601.70 | 2024-05-13 09:15:00 | 1617.80 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-05-09 10:30:00 | 1602.95 | 2024-05-13 10:15:00 | 1625.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-05-09 12:15:00 | 1604.05 | 2024-05-13 10:15:00 | 1625.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-05-10 12:15:00 | 1602.10 | 2024-05-13 10:15:00 | 1625.50 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-05-10 14:15:00 | 1601.40 | 2024-05-13 10:15:00 | 1625.50 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-05-15 09:15:00 | 1645.75 | 2024-05-15 12:15:00 | 1630.40 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-05-15 10:30:00 | 1639.70 | 2024-05-15 12:15:00 | 1630.40 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-05-15 15:15:00 | 1640.00 | 2024-05-23 11:15:00 | 1635.20 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-05-21 10:00:00 | 1645.50 | 2024-05-23 11:15:00 | 1635.20 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-05-28 11:15:00 | 1612.75 | 2024-06-04 12:15:00 | 1532.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 11:15:00 | 1612.75 | 2024-06-04 13:15:00 | 1572.55 | STOP_HIT | 0.50 | 2.49% |
| BUY | retest2 | 2024-06-07 09:15:00 | 1627.95 | 2024-06-11 15:15:00 | 1613.75 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-06-14 09:30:00 | 1602.00 | 2024-06-24 11:15:00 | 1578.40 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2024-06-14 12:15:00 | 1601.70 | 2024-06-24 11:15:00 | 1578.40 | STOP_HIT | 1.00 | 1.45% |
| SELL | retest2 | 2024-06-14 12:45:00 | 1601.50 | 2024-06-24 11:15:00 | 1578.40 | STOP_HIT | 1.00 | 1.44% |
| SELL | retest2 | 2024-06-18 09:15:00 | 1591.15 | 2024-06-24 11:15:00 | 1578.40 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2024-06-18 10:15:00 | 1584.40 | 2024-06-24 11:15:00 | 1578.40 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2024-06-24 09:45:00 | 1576.90 | 2024-06-24 11:15:00 | 1578.40 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-06-28 10:15:00 | 1613.60 | 2024-07-05 09:15:00 | 1774.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-01 14:45:00 | 1612.20 | 2024-07-05 09:15:00 | 1773.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-31 09:15:00 | 1887.70 | 2024-08-06 13:15:00 | 1918.25 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2024-08-22 10:45:00 | 2128.00 | 2024-08-23 09:15:00 | 2092.35 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-09-04 10:45:00 | 2262.65 | 2024-09-09 09:15:00 | 2223.65 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-09-17 09:15:00 | 2264.25 | 2024-09-18 09:15:00 | 2244.90 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-09-18 10:30:00 | 2258.70 | 2024-09-18 12:15:00 | 2229.05 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest1 | 2024-09-23 10:30:00 | 2148.05 | 2024-09-23 14:15:00 | 2181.30 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2024-10-01 10:30:00 | 2178.80 | 2024-10-04 09:15:00 | 2205.70 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-10-03 09:15:00 | 2183.70 | 2024-10-04 09:15:00 | 2205.70 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-10-03 10:00:00 | 2186.00 | 2024-10-04 09:15:00 | 2205.70 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-10-03 11:15:00 | 2186.85 | 2024-10-04 10:15:00 | 2221.50 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-10-03 12:15:00 | 2176.00 | 2024-10-04 10:15:00 | 2221.50 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2024-10-03 13:45:00 | 2165.50 | 2024-10-04 10:15:00 | 2221.50 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-10-04 09:15:00 | 2167.25 | 2024-10-04 10:15:00 | 2221.50 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-10-21 13:00:00 | 2169.75 | 2024-10-25 13:15:00 | 2138.20 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2024-10-30 10:15:00 | 2205.65 | 2024-10-30 13:15:00 | 2164.40 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-11-05 09:15:00 | 2207.20 | 2024-11-05 09:15:00 | 2162.65 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2024-11-19 15:00:00 | 2036.50 | 2024-11-21 11:15:00 | 2051.20 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-12-04 11:15:00 | 2087.20 | 2024-12-12 14:15:00 | 2122.20 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2024-12-04 12:00:00 | 2092.55 | 2024-12-12 14:15:00 | 2122.20 | STOP_HIT | 1.00 | 1.42% |
| BUY | retest2 | 2024-12-05 12:00:00 | 2093.80 | 2024-12-12 14:15:00 | 2122.20 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2024-12-24 10:15:00 | 2175.50 | 2025-01-02 14:15:00 | 2393.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-26 10:30:00 | 2174.40 | 2025-01-02 14:15:00 | 2391.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-20 09:15:00 | 2090.10 | 2025-01-20 14:15:00 | 2130.95 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-01-24 11:30:00 | 2160.55 | 2025-01-27 09:15:00 | 2085.70 | STOP_HIT | 1.00 | -3.46% |
| SELL | retest2 | 2025-02-14 10:30:00 | 1971.95 | 2025-02-24 09:15:00 | 1882.76 | PARTIAL | 0.50 | 4.52% |
| SELL | retest2 | 2025-02-14 10:30:00 | 1971.95 | 2025-02-24 09:15:00 | 1941.15 | STOP_HIT | 0.50 | 1.56% |
| SELL | retest2 | 2025-02-19 09:15:00 | 1929.45 | 2025-02-24 09:15:00 | 1882.38 | PARTIAL | 0.50 | 2.44% |
| SELL | retest2 | 2025-02-19 09:15:00 | 1929.45 | 2025-02-24 09:15:00 | 1941.15 | STOP_HIT | 0.50 | -0.61% |
| SELL | retest2 | 2025-02-19 13:00:00 | 1981.85 | 2025-02-28 09:15:00 | 1873.35 | PARTIAL | 0.50 | 5.47% |
| SELL | retest2 | 2025-02-19 13:00:00 | 1981.85 | 2025-02-28 11:15:00 | 1888.30 | STOP_HIT | 0.50 | 4.72% |
| SELL | retest2 | 2025-02-20 11:15:00 | 1981.45 | 2025-03-03 11:15:00 | 1934.35 | STOP_HIT | 1.00 | 2.38% |
| SELL | retest2 | 2025-02-21 10:45:00 | 1940.20 | 2025-03-03 11:15:00 | 1934.35 | STOP_HIT | 1.00 | 0.30% |
| SELL | retest2 | 2025-03-13 15:15:00 | 1960.10 | 2025-03-17 09:15:00 | 1998.80 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest1 | 2025-03-18 09:15:00 | 2007.25 | 2025-03-21 09:15:00 | 2107.61 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-18 09:15:00 | 2007.25 | 2025-03-21 12:15:00 | 2085.15 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2025-04-01 09:15:00 | 2007.95 | 2025-04-03 09:15:00 | 2100.00 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2025-04-02 14:30:00 | 2008.40 | 2025-04-03 09:15:00 | 2100.00 | STOP_HIT | 1.00 | -4.56% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1929.75 | 2025-04-11 12:15:00 | 1978.75 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-04-11 09:30:00 | 1970.45 | 2025-04-11 12:15:00 | 1978.75 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-04-30 09:15:00 | 2095.10 | 2025-05-02 11:15:00 | 2067.00 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-05-12 09:15:00 | 1979.50 | 2025-05-13 09:15:00 | 2088.70 | STOP_HIT | 1.00 | -5.52% |
| SELL | retest2 | 2025-05-12 10:45:00 | 2019.80 | 2025-05-13 09:15:00 | 2088.70 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2025-05-27 11:30:00 | 1973.90 | 2025-06-05 10:15:00 | 1976.10 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-05-27 13:30:00 | 1973.50 | 2025-06-05 10:15:00 | 1976.10 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-06-17 10:45:00 | 1966.00 | 2025-06-24 11:15:00 | 1957.40 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-06-26 10:30:00 | 1924.70 | 2025-06-27 10:15:00 | 1948.20 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-04 09:45:00 | 1971.60 | 2025-07-08 09:15:00 | 1924.10 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1948.70 | 2025-07-18 10:15:00 | 1929.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-07-18 12:15:00 | 1943.50 | 2025-07-18 13:15:00 | 1933.90 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-07-23 13:45:00 | 1908.40 | 2025-07-23 14:15:00 | 1919.90 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-07-29 10:30:00 | 1965.60 | 2025-07-31 09:15:00 | 1928.40 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-07-29 11:30:00 | 1967.00 | 2025-07-31 09:15:00 | 1928.40 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-08-06 14:00:00 | 1842.00 | 2025-08-07 09:15:00 | 1912.80 | STOP_HIT | 1.00 | -3.84% |
| BUY | retest2 | 2025-08-11 10:30:00 | 1933.30 | 2025-08-20 10:15:00 | 1958.80 | STOP_HIT | 1.00 | 1.32% |
| BUY | retest2 | 2025-09-08 12:45:00 | 1955.90 | 2025-09-22 12:15:00 | 2023.00 | STOP_HIT | 1.00 | 3.43% |
| BUY | retest2 | 2025-09-09 15:00:00 | 1956.40 | 2025-09-22 12:15:00 | 2023.00 | STOP_HIT | 1.00 | 3.40% |
| SELL | retest2 | 2025-09-25 12:15:00 | 1998.40 | 2025-10-01 11:15:00 | 1983.00 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2025-10-31 09:15:00 | 1949.10 | 2025-11-06 13:15:00 | 1955.90 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2025-10-31 10:15:00 | 1955.00 | 2025-11-06 13:15:00 | 1955.90 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-11-24 09:15:00 | 2002.30 | 2025-11-25 09:15:00 | 2037.90 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-12-02 13:15:00 | 2088.20 | 2025-12-03 09:15:00 | 2069.30 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-02 15:15:00 | 2089.50 | 2025-12-03 09:15:00 | 2069.30 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-12-04 13:00:00 | 2091.70 | 2025-12-08 12:15:00 | 2071.90 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-12-05 10:00:00 | 2087.60 | 2025-12-08 12:15:00 | 2071.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-12-15 12:00:00 | 2096.40 | 2025-12-16 15:15:00 | 2085.90 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-12-16 12:15:00 | 2095.70 | 2025-12-16 15:15:00 | 2085.90 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-12-17 09:15:00 | 2117.80 | 2025-12-23 13:15:00 | 2111.30 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-12-29 10:15:00 | 2099.00 | 2025-12-31 12:15:00 | 2110.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-29 11:00:00 | 2093.80 | 2025-12-31 12:15:00 | 2110.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-01-02 14:15:00 | 2109.70 | 2026-01-05 09:15:00 | 2098.90 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-01-22 13:15:00 | 2171.10 | 2026-01-30 12:15:00 | 2142.00 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2026-02-02 10:15:00 | 2097.10 | 2026-02-03 09:15:00 | 2168.70 | STOP_HIT | 1.00 | -3.41% |
| BUY | retest2 | 2026-02-09 15:15:00 | 2208.70 | 2026-02-13 09:15:00 | 2179.90 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2026-02-10 14:45:00 | 2201.90 | 2026-02-13 09:15:00 | 2179.90 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-02-12 11:30:00 | 2204.60 | 2026-02-13 09:15:00 | 2179.90 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-02-12 13:30:00 | 2205.40 | 2026-02-13 09:15:00 | 2179.90 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-02-19 11:45:00 | 2240.20 | 2026-02-20 09:15:00 | 2218.40 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-02-19 12:15:00 | 2242.40 | 2026-02-20 09:15:00 | 2218.40 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2026-02-27 13:45:00 | 2329.10 | 2026-03-02 09:15:00 | 2274.80 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-03-12 11:30:00 | 2340.00 | 2026-03-13 12:15:00 | 2317.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-03-13 10:30:00 | 2342.90 | 2026-03-13 12:15:00 | 2317.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-03-13 11:15:00 | 2345.20 | 2026-03-13 12:15:00 | 2317.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-03-17 11:15:00 | 2288.70 | 2026-03-18 13:15:00 | 2311.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-03-17 12:45:00 | 2294.80 | 2026-03-18 13:15:00 | 2311.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-03-17 13:15:00 | 2293.40 | 2026-03-18 13:15:00 | 2311.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-03-19 09:15:00 | 2280.00 | 2026-03-20 11:15:00 | 2311.90 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-03-23 13:15:00 | 2300.60 | 2026-03-24 10:15:00 | 2291.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2026-03-24 09:15:00 | 2317.80 | 2026-03-24 10:15:00 | 2291.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-06 09:15:00 | 2251.10 | 2026-04-07 09:15:00 | 2287.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-04-06 10:00:00 | 2250.60 | 2026-04-07 09:15:00 | 2287.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2026-04-15 09:15:00 | 2334.00 | 2026-04-16 11:15:00 | 2298.40 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-04-16 10:00:00 | 2319.40 | 2026-04-16 11:15:00 | 2298.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-04-16 11:00:00 | 2319.00 | 2026-04-16 11:15:00 | 2298.40 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-04-27 11:15:00 | 2308.00 | 2026-04-27 14:15:00 | 2323.20 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-04-27 11:45:00 | 2315.80 | 2026-04-27 14:15:00 | 2323.20 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2026-04-27 12:45:00 | 2315.00 | 2026-04-27 14:15:00 | 2323.20 | STOP_HIT | 1.00 | -0.35% |
