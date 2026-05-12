# Bharti Hexacom Ltd. (BHARTIHEXA)

## Backtest Summary

- **Window:** 2024-04-12 09:15:00 → 2026-05-11 15:15:00 (3584 bars)
- **Last close:** 1467.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 171 |
| ALERT1 | 99 |
| ALERT2 | 99 |
| ALERT2_SKIP | 45 |
| ALERT3 | 280 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 128 |
| PARTIAL | 6 |
| TARGET_HIT | 8 |
| STOP_HIT | 122 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 40 / 96
- **Target hits / Stop hits / Partials:** 8 / 122 / 6
- **Avg / median % per leg:** 0.02% / -0.95%
- **Sum % (uncompounded):** 2.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 21 | 29.2% | 6 | 66 | 0 | 0.20% | 14.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 72 | 21 | 29.2% | 6 | 66 | 0 | 0.20% | 14.1% |
| SELL (all) | 64 | 19 | 29.7% | 2 | 56 | 6 | -0.18% | -11.3% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.33% | 0.7% |
| SELL @ 3rd Alert (retest2) | 62 | 18 | 29.0% | 2 | 54 | 6 | -0.19% | -12.0% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | 0.33% | 0.7% |
| retest2 (combined) | 134 | 39 | 29.1% | 8 | 120 | 6 | 0.02% | 2.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 942.00 | 949.29 | 949.34 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 955.00 | 950.43 | 949.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 11:15:00 | 958.00 | 952.45 | 950.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 11:15:00 | 1016.50 | 1017.83 | 1002.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-27 12:00:00 | 1016.50 | 1017.83 | 1002.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 09:15:00 | 1019.85 | 1023.68 | 1011.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 10:00:00 | 1019.85 | 1023.68 | 1011.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 11:15:00 | 1011.00 | 1019.88 | 1012.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 11:45:00 | 1007.50 | 1019.88 | 1012.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 1004.10 | 1016.72 | 1011.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 1001.80 | 1016.72 | 1011.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 992.95 | 1010.29 | 1009.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 992.95 | 1010.29 | 1009.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 15:15:00 | 992.00 | 1006.63 | 1007.74 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 11:15:00 | 1011.65 | 1008.60 | 1008.45 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 1002.75 | 1012.72 | 1012.78 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 1026.00 | 1012.84 | 1012.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 1056.50 | 1021.57 | 1016.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 1035.85 | 1052.14 | 1038.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 1035.85 | 1052.14 | 1038.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1035.85 | 1052.14 | 1038.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 1043.30 | 1052.14 | 1038.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 989.35 | 1039.58 | 1034.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 989.35 | 1039.58 | 1034.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 985.80 | 1028.82 | 1029.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 14:15:00 | 962.05 | 1001.07 | 1015.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 11:15:00 | 990.50 | 987.65 | 1003.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 11:15:00 | 990.50 | 987.65 | 1003.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 990.50 | 987.65 | 1003.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:45:00 | 1018.35 | 987.65 | 1003.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 12:15:00 | 993.85 | 988.89 | 1002.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:00:00 | 993.85 | 988.89 | 1002.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1026.10 | 996.33 | 1004.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:45:00 | 1032.60 | 996.33 | 1004.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1014.25 | 999.91 | 1005.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:15:00 | 1032.00 | 999.91 | 1005.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 1032.00 | 1006.33 | 1007.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 1032.90 | 1006.33 | 1007.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1030.35 | 1011.14 | 1009.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 1055.50 | 1029.76 | 1020.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 1078.75 | 1084.82 | 1070.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 09:30:00 | 1077.15 | 1084.82 | 1070.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 1060.60 | 1077.85 | 1069.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:00:00 | 1060.60 | 1077.85 | 1069.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 12:15:00 | 1084.00 | 1079.08 | 1070.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:15:00 | 1090.35 | 1079.08 | 1070.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 14:15:00 | 1091.65 | 1080.28 | 1071.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1086.40 | 1079.18 | 1072.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:45:00 | 1089.20 | 1093.33 | 1090.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 10:15:00 | 1075.00 | 1089.66 | 1089.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-14 11:00:00 | 1075.00 | 1089.66 | 1089.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-14 11:15:00 | 1066.35 | 1085.00 | 1087.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 11:15:00 | 1066.35 | 1085.00 | 1087.18 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 1103.00 | 1088.05 | 1086.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 12:15:00 | 1123.15 | 1098.93 | 1092.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 10:15:00 | 1111.15 | 1119.79 | 1107.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-19 10:15:00 | 1111.15 | 1119.79 | 1107.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 10:15:00 | 1111.15 | 1119.79 | 1107.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-19 10:45:00 | 1112.10 | 1119.79 | 1107.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 11:15:00 | 1141.65 | 1124.16 | 1110.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 14:30:00 | 1151.05 | 1134.35 | 1118.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 1149.55 | 1136.63 | 1121.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:45:00 | 1148.00 | 1137.92 | 1123.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 1151.95 | 1137.92 | 1123.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 1129.00 | 1137.08 | 1128.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-21 14:15:00 | 1120.00 | 1125.98 | 1126.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 1120.00 | 1125.98 | 1126.36 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 11:15:00 | 1127.10 | 1126.37 | 1126.34 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 12:15:00 | 1125.20 | 1126.13 | 1126.24 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 1131.45 | 1127.20 | 1126.71 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 14:15:00 | 1118.95 | 1125.55 | 1126.00 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 1155.50 | 1131.77 | 1128.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 13:15:00 | 1168.00 | 1148.24 | 1138.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 09:15:00 | 1180.00 | 1208.42 | 1182.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 1180.00 | 1208.42 | 1182.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 1180.00 | 1208.42 | 1182.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:00:00 | 1180.00 | 1208.42 | 1182.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 1194.55 | 1205.64 | 1183.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:45:00 | 1199.90 | 1205.64 | 1183.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 1189.10 | 1202.34 | 1183.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 12:30:00 | 1195.90 | 1199.85 | 1184.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 13:15:00 | 1172.90 | 1194.46 | 1183.34 | SL hit (close<static) qty=1.00 sl=1179.35 alert=retest2 |

### Cycle 17 — SELL (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 09:15:00 | 1117.15 | 1176.53 | 1177.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 11:15:00 | 1100.05 | 1153.92 | 1166.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 14:15:00 | 1100.00 | 1099.42 | 1122.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-01 14:45:00 | 1100.10 | 1099.42 | 1122.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 1093.25 | 1090.08 | 1102.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 11:00:00 | 1093.25 | 1090.08 | 1102.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 1062.50 | 1082.74 | 1093.91 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2024-07-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 15:15:00 | 1092.25 | 1087.54 | 1087.13 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 1082.85 | 1086.60 | 1086.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 1074.90 | 1083.09 | 1085.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 12:15:00 | 1070.85 | 1070.73 | 1076.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 13:00:00 | 1070.85 | 1070.73 | 1076.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 1070.20 | 1070.51 | 1075.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:30:00 | 1072.20 | 1070.51 | 1075.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1056.10 | 1067.55 | 1073.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 1045.00 | 1067.55 | 1073.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 12:45:00 | 1055.00 | 1057.45 | 1062.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 1084.05 | 1063.58 | 1062.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-07-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 10:15:00 | 1084.05 | 1063.58 | 1062.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 12:15:00 | 1085.60 | 1070.71 | 1066.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-15 09:15:00 | 1071.90 | 1078.84 | 1072.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 09:15:00 | 1071.90 | 1078.84 | 1072.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 09:15:00 | 1071.90 | 1078.84 | 1072.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 09:15:00 | 1142.55 | 1072.61 | 1072.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 13:15:00 | 1108.70 | 1121.92 | 1123.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 13:15:00 | 1108.70 | 1121.92 | 1123.47 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 11:15:00 | 1136.50 | 1125.39 | 1124.12 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 1112.25 | 1122.76 | 1123.04 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 13:15:00 | 1126.10 | 1123.43 | 1123.32 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 14:15:00 | 1115.25 | 1121.79 | 1122.58 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 1158.35 | 1128.10 | 1125.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 1173.00 | 1158.46 | 1151.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 12:15:00 | 1161.05 | 1163.97 | 1156.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 13:15:00 | 1157.35 | 1163.97 | 1156.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 1155.05 | 1162.19 | 1156.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:00:00 | 1155.05 | 1162.19 | 1156.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 1168.95 | 1163.54 | 1157.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 1192.90 | 1164.25 | 1158.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 14:15:00 | 1136.05 | 1162.19 | 1164.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 14:15:00 | 1136.05 | 1162.19 | 1164.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 1132.90 | 1149.30 | 1154.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 1112.15 | 1100.24 | 1115.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:00:00 | 1112.15 | 1100.24 | 1115.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 1115.00 | 1103.19 | 1115.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:00:00 | 1115.00 | 1103.19 | 1115.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 1125.25 | 1107.60 | 1116.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 12:00:00 | 1125.25 | 1107.60 | 1116.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 1125.05 | 1111.09 | 1116.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:00:00 | 1125.05 | 1111.09 | 1116.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 1126.05 | 1114.08 | 1117.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 1125.55 | 1114.08 | 1117.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 1138.00 | 1122.49 | 1120.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 14:15:00 | 1147.45 | 1138.85 | 1133.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 12:15:00 | 1138.10 | 1145.16 | 1139.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 12:15:00 | 1138.10 | 1145.16 | 1139.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 1138.10 | 1145.16 | 1139.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:45:00 | 1139.25 | 1145.16 | 1139.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 1137.15 | 1143.56 | 1139.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:45:00 | 1139.65 | 1143.56 | 1139.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 1140.00 | 1142.85 | 1139.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:15:00 | 1127.00 | 1142.85 | 1139.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 1127.00 | 1139.68 | 1138.32 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 1118.40 | 1135.42 | 1136.51 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 1141.80 | 1132.83 | 1132.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 1149.15 | 1138.49 | 1135.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 09:15:00 | 1139.60 | 1139.69 | 1136.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 09:45:00 | 1139.00 | 1139.69 | 1136.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 1142.15 | 1140.18 | 1136.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:30:00 | 1142.40 | 1140.18 | 1136.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 1131.65 | 1138.47 | 1136.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 12:00:00 | 1131.65 | 1138.47 | 1136.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 1131.50 | 1137.08 | 1135.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 12:30:00 | 1130.00 | 1137.08 | 1135.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 14:15:00 | 1129.15 | 1134.27 | 1134.78 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 13:15:00 | 1148.45 | 1137.27 | 1135.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 14:15:00 | 1155.10 | 1140.84 | 1137.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 12:15:00 | 1141.70 | 1147.65 | 1142.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-21 12:15:00 | 1141.70 | 1147.65 | 1142.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 1141.70 | 1147.65 | 1142.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:00:00 | 1141.70 | 1147.65 | 1142.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 13:15:00 | 1139.60 | 1146.04 | 1142.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 13:30:00 | 1137.70 | 1146.04 | 1142.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 1139.25 | 1144.68 | 1142.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 15:15:00 | 1137.10 | 1144.68 | 1142.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2024-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 10:15:00 | 1125.95 | 1138.43 | 1139.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 12:15:00 | 1119.00 | 1132.40 | 1136.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 09:15:00 | 1131.25 | 1127.62 | 1132.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 09:15:00 | 1131.25 | 1127.62 | 1132.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 1131.25 | 1127.62 | 1132.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 1131.25 | 1127.62 | 1132.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1138.30 | 1129.76 | 1133.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:30:00 | 1140.65 | 1129.76 | 1133.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 1138.95 | 1131.60 | 1133.65 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2024-08-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-23 13:15:00 | 1143.05 | 1135.68 | 1135.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 15:15:00 | 1145.50 | 1138.98 | 1136.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 14:15:00 | 1178.35 | 1179.31 | 1172.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 15:00:00 | 1178.35 | 1179.31 | 1172.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1174.50 | 1178.48 | 1173.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:45:00 | 1170.55 | 1178.48 | 1173.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1169.15 | 1176.62 | 1173.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 1169.15 | 1176.62 | 1173.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 1172.30 | 1175.75 | 1173.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 15:15:00 | 1185.95 | 1174.63 | 1172.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 13:15:00 | 1215.10 | 1221.02 | 1221.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 13:15:00 | 1215.10 | 1221.02 | 1221.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 14:15:00 | 1205.80 | 1217.97 | 1219.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 14:15:00 | 1200.00 | 1194.92 | 1201.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-06 14:15:00 | 1200.00 | 1194.92 | 1201.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 1200.00 | 1194.92 | 1201.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:45:00 | 1200.00 | 1194.92 | 1201.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 13:15:00 | 1188.00 | 1190.08 | 1196.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 13:45:00 | 1192.75 | 1190.08 | 1196.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1195.40 | 1191.14 | 1196.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 14:45:00 | 1190.85 | 1191.14 | 1196.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1200.00 | 1192.91 | 1196.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 1215.25 | 1192.91 | 1196.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 1225.10 | 1199.35 | 1199.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 10:15:00 | 1232.80 | 1206.04 | 1202.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 09:15:00 | 1341.50 | 1349.09 | 1314.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 10:00:00 | 1341.50 | 1349.09 | 1314.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1345.40 | 1355.50 | 1341.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:30:00 | 1342.00 | 1355.50 | 1341.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1353.80 | 1365.60 | 1354.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:30:00 | 1354.40 | 1365.60 | 1354.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1354.40 | 1363.36 | 1354.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 12:15:00 | 1373.70 | 1361.40 | 1354.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 14:45:00 | 1357.90 | 1364.21 | 1357.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 1363.60 | 1364.21 | 1357.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:30:00 | 1359.55 | 1364.00 | 1358.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1381.55 | 1367.51 | 1360.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 12:45:00 | 1399.20 | 1375.64 | 1365.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 13:15:00 | 1382.10 | 1404.71 | 1405.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 13:15:00 | 1382.10 | 1404.71 | 1405.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 14:15:00 | 1364.55 | 1396.68 | 1402.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 11:15:00 | 1393.95 | 1388.20 | 1395.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 11:15:00 | 1393.95 | 1388.20 | 1395.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 1393.95 | 1388.20 | 1395.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:30:00 | 1390.00 | 1388.20 | 1395.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 1420.10 | 1394.58 | 1397.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:00:00 | 1420.10 | 1394.58 | 1397.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 13:15:00 | 1424.50 | 1400.57 | 1400.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 09:15:00 | 1446.75 | 1415.46 | 1407.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 1446.95 | 1458.33 | 1442.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 09:15:00 | 1446.95 | 1458.33 | 1442.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 1446.95 | 1458.33 | 1442.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:30:00 | 1442.85 | 1458.33 | 1442.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 1435.90 | 1453.84 | 1441.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:30:00 | 1436.75 | 1453.84 | 1441.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 1425.35 | 1448.14 | 1439.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 12:00:00 | 1425.35 | 1448.14 | 1439.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 1441.00 | 1446.72 | 1440.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 13:45:00 | 1444.95 | 1443.80 | 1439.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 14:45:00 | 1445.00 | 1442.97 | 1439.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:45:00 | 1448.60 | 1441.35 | 1439.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 10:15:00 | 1412.00 | 1435.48 | 1436.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-10-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 10:15:00 | 1412.00 | 1435.48 | 1436.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 11:15:00 | 1399.90 | 1428.37 | 1433.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 10:15:00 | 1418.90 | 1412.28 | 1421.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-03 11:00:00 | 1418.90 | 1412.28 | 1421.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 1450.25 | 1407.79 | 1412.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 11:00:00 | 1450.25 | 1407.79 | 1412.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 11:15:00 | 1428.30 | 1411.89 | 1414.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 12:15:00 | 1424.80 | 1411.89 | 1414.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:00:00 | 1426.30 | 1397.11 | 1401.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 10:30:00 | 1425.50 | 1403.45 | 1403.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 11:15:00 | 1424.25 | 1403.45 | 1403.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 11:15:00 | 1418.65 | 1406.49 | 1405.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 11:15:00 | 1418.65 | 1406.49 | 1405.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 12:15:00 | 1438.65 | 1412.92 | 1408.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 09:15:00 | 1430.45 | 1438.46 | 1424.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 1430.45 | 1438.46 | 1424.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1430.45 | 1438.46 | 1424.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 1422.95 | 1438.46 | 1424.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 1430.00 | 1435.77 | 1429.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 1442.55 | 1435.77 | 1429.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 15:00:00 | 1440.25 | 1439.93 | 1434.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 10:30:00 | 1440.00 | 1439.52 | 1435.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 11:15:00 | 1440.00 | 1439.52 | 1435.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1426.25 | 1438.35 | 1437.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 10:45:00 | 1467.60 | 1444.68 | 1440.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 11:15:00 | 1485.30 | 1498.17 | 1498.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-10-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 11:15:00 | 1485.30 | 1498.17 | 1498.92 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 15:15:00 | 1515.50 | 1500.76 | 1499.57 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-10-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 09:15:00 | 1484.60 | 1497.53 | 1498.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 1460.00 | 1490.02 | 1494.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 10:15:00 | 1470.85 | 1460.61 | 1473.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 11:00:00 | 1470.85 | 1460.61 | 1473.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1463.85 | 1461.26 | 1473.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:30:00 | 1469.00 | 1461.26 | 1473.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 1472.25 | 1463.46 | 1472.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:45:00 | 1471.70 | 1463.46 | 1472.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 1485.10 | 1467.79 | 1474.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:00:00 | 1485.10 | 1467.79 | 1474.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 1451.65 | 1464.56 | 1472.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 09:45:00 | 1445.70 | 1459.77 | 1468.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:15:00 | 1450.00 | 1458.25 | 1466.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 13:00:00 | 1450.55 | 1456.42 | 1464.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 1450.70 | 1459.53 | 1464.09 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1427.15 | 1453.05 | 1460.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 14:15:00 | 1424.55 | 1443.86 | 1453.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 14:45:00 | 1421.45 | 1440.54 | 1451.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 14:15:00 | 1475.40 | 1456.57 | 1454.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 1475.40 | 1456.57 | 1454.84 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 1424.15 | 1450.56 | 1452.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 11:15:00 | 1390.25 | 1411.65 | 1422.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 1411.95 | 1405.63 | 1416.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-31 15:00:00 | 1411.95 | 1405.63 | 1416.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 1402.00 | 1404.90 | 1415.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1423.95 | 1408.71 | 1416.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1390.85 | 1406.96 | 1414.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:45:00 | 1384.15 | 1402.60 | 1411.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 13:15:00 | 1385.85 | 1396.85 | 1407.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 1425.00 | 1405.70 | 1403.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 09:15:00 | 1425.00 | 1405.70 | 1403.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 1451.50 | 1421.15 | 1411.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 10:15:00 | 1416.15 | 1436.05 | 1424.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 10:15:00 | 1416.15 | 1436.05 | 1424.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1416.15 | 1436.05 | 1424.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 1416.15 | 1436.05 | 1424.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1420.80 | 1433.00 | 1424.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:30:00 | 1417.00 | 1433.00 | 1424.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2024-11-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 10:15:00 | 1409.60 | 1418.97 | 1420.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 1399.85 | 1415.14 | 1418.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 15:15:00 | 1380.00 | 1377.82 | 1390.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 15:15:00 | 1380.00 | 1377.82 | 1390.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 1380.00 | 1377.82 | 1390.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 09:30:00 | 1369.70 | 1381.57 | 1391.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-13 10:15:00 | 1415.80 | 1390.86 | 1390.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2024-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-13 10:15:00 | 1415.80 | 1390.86 | 1390.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-13 11:15:00 | 1425.00 | 1397.69 | 1393.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-14 10:15:00 | 1411.00 | 1413.16 | 1405.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:45:00 | 1414.35 | 1413.16 | 1405.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 1389.05 | 1415.52 | 1410.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:30:00 | 1380.05 | 1415.52 | 1410.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 1395.85 | 1411.58 | 1409.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-18 12:30:00 | 1402.90 | 1407.75 | 1407.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-18 13:15:00 | 1403.05 | 1407.75 | 1407.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 13:15:00 | 1402.10 | 1406.62 | 1407.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 13:15:00 | 1402.10 | 1406.62 | 1407.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 14:15:00 | 1392.10 | 1403.72 | 1405.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 1407.25 | 1403.03 | 1405.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 1407.25 | 1403.03 | 1405.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 1407.25 | 1403.03 | 1405.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 1407.25 | 1403.03 | 1405.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1406.90 | 1403.80 | 1405.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 1407.25 | 1403.80 | 1405.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 1404.00 | 1403.84 | 1405.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:45:00 | 1406.60 | 1403.84 | 1405.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 1407.55 | 1404.58 | 1405.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:30:00 | 1408.85 | 1404.58 | 1405.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 13:15:00 | 1403.50 | 1404.37 | 1405.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 1396.75 | 1403.27 | 1404.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-22 09:15:00 | 1326.91 | 1363.74 | 1379.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-22 15:15:00 | 1340.00 | 1335.29 | 1355.43 | SL hit (close>ema200) qty=0.50 sl=1335.29 alert=retest2 |

### Cycle 50 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 1332.00 | 1317.03 | 1316.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 11:15:00 | 1340.70 | 1321.76 | 1318.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 13:15:00 | 1388.90 | 1396.85 | 1379.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 14:00:00 | 1388.90 | 1396.85 | 1379.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1377.10 | 1390.05 | 1380.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:45:00 | 1377.60 | 1390.05 | 1380.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 1367.50 | 1385.54 | 1379.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:45:00 | 1374.00 | 1385.54 | 1379.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 1366.15 | 1381.66 | 1378.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 12:00:00 | 1366.15 | 1381.66 | 1378.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 1403.90 | 1382.69 | 1379.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 1412.55 | 1397.97 | 1389.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 13:45:00 | 1407.00 | 1427.61 | 1425.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 14:45:00 | 1406.45 | 1424.53 | 1424.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 15:15:00 | 1405.55 | 1420.73 | 1422.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 15:15:00 | 1405.55 | 1420.73 | 1422.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 09:15:00 | 1396.55 | 1415.90 | 1420.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 13:15:00 | 1409.10 | 1408.20 | 1414.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-11 14:00:00 | 1409.10 | 1408.20 | 1414.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 1422.35 | 1410.23 | 1413.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:30:00 | 1426.05 | 1410.23 | 1413.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1420.40 | 1412.27 | 1414.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:30:00 | 1426.20 | 1412.27 | 1414.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 1419.85 | 1414.05 | 1414.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:45:00 | 1420.60 | 1414.05 | 1414.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 13:15:00 | 1420.20 | 1415.28 | 1415.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 13:45:00 | 1422.40 | 1415.28 | 1415.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 14:15:00 | 1420.55 | 1416.33 | 1415.87 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 1408.15 | 1414.21 | 1415.03 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 14:15:00 | 1482.35 | 1427.31 | 1420.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 15:15:00 | 1510.00 | 1443.85 | 1428.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 13:15:00 | 1455.60 | 1457.66 | 1442.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-16 13:30:00 | 1456.75 | 1457.66 | 1442.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 15:15:00 | 1441.55 | 1453.85 | 1443.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 09:15:00 | 1449.50 | 1453.85 | 1443.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 1453.50 | 1453.78 | 1444.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 1469.00 | 1451.61 | 1447.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 09:30:00 | 1466.80 | 1501.93 | 1498.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 10:15:00 | 1466.00 | 1494.74 | 1495.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 10:15:00 | 1466.00 | 1494.74 | 1495.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 13:15:00 | 1456.55 | 1470.14 | 1477.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 15:15:00 | 1475.00 | 1468.59 | 1475.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 15:15:00 | 1475.00 | 1468.59 | 1475.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1475.00 | 1468.59 | 1475.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 1476.75 | 1468.59 | 1475.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1478.00 | 1470.48 | 1475.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 1448.85 | 1465.91 | 1471.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:15:00 | 1456.25 | 1466.28 | 1469.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 14:15:00 | 1514.30 | 1473.13 | 1472.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1514.30 | 1473.13 | 1472.20 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 12:15:00 | 1458.35 | 1471.13 | 1472.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 14:15:00 | 1452.60 | 1468.22 | 1470.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 09:15:00 | 1468.75 | 1466.21 | 1469.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 09:15:00 | 1468.75 | 1466.21 | 1469.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 09:15:00 | 1468.75 | 1466.21 | 1469.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:00:00 | 1468.75 | 1466.21 | 1469.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 1472.00 | 1467.37 | 1469.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 1471.95 | 1467.37 | 1469.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 1475.50 | 1468.99 | 1470.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 12:00:00 | 1475.50 | 1468.99 | 1470.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 12:15:00 | 1481.10 | 1471.41 | 1471.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 1500.35 | 1480.43 | 1476.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 15:15:00 | 1488.55 | 1494.88 | 1487.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 15:15:00 | 1488.55 | 1494.88 | 1487.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 1488.55 | 1494.88 | 1487.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 1483.20 | 1494.88 | 1487.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1479.45 | 1491.80 | 1486.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 10:45:00 | 1500.15 | 1490.03 | 1487.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 12:15:00 | 1479.55 | 1488.16 | 1488.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 1479.55 | 1488.16 | 1488.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 1448.35 | 1472.39 | 1478.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-13 14:15:00 | 1417.05 | 1416.62 | 1439.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-13 15:00:00 | 1417.05 | 1416.62 | 1439.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1426.50 | 1419.37 | 1433.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 1422.20 | 1419.37 | 1433.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 13:15:00 | 1419.15 | 1420.56 | 1432.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 15:15:00 | 1351.09 | 1369.54 | 1392.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 1370.50 | 1369.48 | 1388.68 | SL hit (close>ema200) qty=0.50 sl=1369.48 alert=retest2 |

### Cycle 60 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 1316.35 | 1293.45 | 1290.45 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 10:15:00 | 1274.90 | 1304.54 | 1306.49 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 1332.60 | 1303.72 | 1303.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 10:15:00 | 1348.30 | 1312.63 | 1307.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 14:15:00 | 1355.95 | 1359.69 | 1343.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 15:00:00 | 1355.95 | 1359.69 | 1343.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 1324.05 | 1352.29 | 1342.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 10:00:00 | 1324.05 | 1352.29 | 1342.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 1365.15 | 1354.86 | 1344.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 12:45:00 | 1369.55 | 1361.81 | 1355.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:15:00 | 1368.90 | 1361.81 | 1355.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1371.15 | 1362.92 | 1356.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 15:15:00 | 1408.40 | 1358.86 | 1357.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 1408.40 | 1368.77 | 1361.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-05 12:15:00 | 1357.60 | 1368.42 | 1368.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 12:15:00 | 1357.60 | 1368.42 | 1368.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 13:15:00 | 1340.05 | 1362.75 | 1366.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 1389.05 | 1340.05 | 1347.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 09:15:00 | 1389.05 | 1340.05 | 1347.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1389.05 | 1340.05 | 1347.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 1390.10 | 1340.05 | 1347.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1386.55 | 1349.35 | 1351.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 1402.80 | 1349.35 | 1351.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 11:15:00 | 1399.70 | 1359.42 | 1355.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-07 13:15:00 | 1418.75 | 1379.54 | 1366.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-10 10:15:00 | 1403.70 | 1413.30 | 1389.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-10 11:00:00 | 1403.70 | 1413.30 | 1389.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 1389.95 | 1408.63 | 1389.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 1389.95 | 1408.63 | 1389.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 1391.55 | 1405.21 | 1389.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 13:00:00 | 1391.55 | 1405.21 | 1389.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 1385.20 | 1401.21 | 1389.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:00:00 | 1385.20 | 1401.21 | 1389.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 14:15:00 | 1372.00 | 1395.37 | 1387.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:30:00 | 1374.95 | 1395.37 | 1387.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 1366.50 | 1389.59 | 1385.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:15:00 | 1338.50 | 1389.59 | 1385.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 1312.35 | 1374.14 | 1379.30 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 1378.40 | 1360.70 | 1360.06 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 1345.00 | 1359.38 | 1361.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 1327.00 | 1350.12 | 1355.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 12:15:00 | 1328.30 | 1318.70 | 1332.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 13:00:00 | 1328.30 | 1318.70 | 1332.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 1326.55 | 1320.27 | 1331.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 13:45:00 | 1328.65 | 1320.27 | 1331.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 1333.35 | 1322.89 | 1331.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 1333.35 | 1322.89 | 1331.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 1327.00 | 1323.71 | 1331.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 1346.55 | 1328.28 | 1332.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1349.40 | 1332.50 | 1334.18 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 1358.30 | 1337.66 | 1336.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 1372.45 | 1347.67 | 1341.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 1353.65 | 1356.07 | 1348.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 12:00:00 | 1353.65 | 1356.07 | 1348.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 1329.00 | 1354.47 | 1351.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 1329.00 | 1354.47 | 1351.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 1324.75 | 1348.53 | 1348.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 1319.95 | 1335.32 | 1342.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 09:15:00 | 1272.40 | 1257.58 | 1275.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-27 09:15:00 | 1272.40 | 1257.58 | 1275.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 1272.40 | 1257.58 | 1275.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:30:00 | 1274.05 | 1257.58 | 1275.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 12:15:00 | 1308.00 | 1270.20 | 1277.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:00:00 | 1308.00 | 1270.20 | 1277.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 1318.05 | 1279.77 | 1281.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:30:00 | 1324.50 | 1279.77 | 1281.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 14:15:00 | 1301.00 | 1284.01 | 1282.86 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 1258.00 | 1282.37 | 1282.53 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 10:15:00 | 1290.20 | 1283.94 | 1283.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-28 14:15:00 | 1315.00 | 1290.85 | 1286.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-28 15:15:00 | 1277.00 | 1288.08 | 1285.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 15:15:00 | 1277.00 | 1288.08 | 1285.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 1277.00 | 1288.08 | 1285.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 1267.00 | 1288.08 | 1285.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 1283.65 | 1287.19 | 1285.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:45:00 | 1276.85 | 1287.19 | 1285.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 1300.00 | 1289.75 | 1286.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 09:30:00 | 1324.40 | 1294.51 | 1291.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 11:00:00 | 1306.95 | 1297.00 | 1293.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 12:15:00 | 1305.95 | 1298.03 | 1293.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-05 14:15:00 | 1456.84 | 1318.54 | 1304.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-03-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 14:15:00 | 1353.05 | 1359.72 | 1360.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 09:15:00 | 1337.70 | 1353.61 | 1357.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 14:15:00 | 1356.80 | 1349.97 | 1353.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 14:15:00 | 1356.80 | 1349.97 | 1353.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 1356.80 | 1349.97 | 1353.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 1325.00 | 1349.62 | 1353.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-24 09:15:00 | 1350.05 | 1330.23 | 1328.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-03-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 09:15:00 | 1350.05 | 1330.23 | 1328.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 10:15:00 | 1382.45 | 1340.67 | 1333.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-27 14:15:00 | 1455.50 | 1460.44 | 1440.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-27 14:30:00 | 1457.90 | 1460.44 | 1440.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1428.60 | 1453.76 | 1441.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 1428.60 | 1453.76 | 1441.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 1471.70 | 1457.35 | 1443.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 15:15:00 | 1476.00 | 1443.73 | 1441.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 1477.50 | 1458.57 | 1449.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 14:45:00 | 1473.60 | 1467.97 | 1457.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 14:15:00 | 1445.65 | 1453.71 | 1454.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 14:15:00 | 1445.65 | 1453.71 | 1454.50 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 1462.25 | 1455.43 | 1455.06 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 1444.70 | 1454.85 | 1455.52 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 10:15:00 | 1460.65 | 1456.01 | 1455.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 11:15:00 | 1465.35 | 1457.88 | 1456.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 15:15:00 | 1458.10 | 1460.72 | 1458.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 15:15:00 | 1458.10 | 1460.72 | 1458.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 1458.10 | 1460.72 | 1458.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:15:00 | 1396.75 | 1460.72 | 1458.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1403.55 | 1449.28 | 1453.67 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-04-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 10:15:00 | 1455.80 | 1433.75 | 1432.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 1499.00 | 1448.55 | 1440.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 12:15:00 | 1493.30 | 1499.70 | 1480.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 13:00:00 | 1493.30 | 1499.70 | 1480.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 13:15:00 | 1523.30 | 1504.42 | 1484.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 14:15:00 | 1525.20 | 1504.42 | 1484.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-16 14:30:00 | 1525.70 | 1508.52 | 1497.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:00:00 | 1530.10 | 1514.61 | 1503.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-23 09:15:00 | 1677.72 | 1640.68 | 1617.38 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1596.90 | 1634.99 | 1635.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 1575.20 | 1598.88 | 1614.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 14:15:00 | 1592.00 | 1588.04 | 1601.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 15:00:00 | 1592.00 | 1588.04 | 1601.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1588.20 | 1587.91 | 1599.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 1581.70 | 1587.91 | 1599.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 1655.20 | 1606.35 | 1602.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 1655.20 | 1606.35 | 1602.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 13:15:00 | 1667.70 | 1628.29 | 1615.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 12:15:00 | 1654.10 | 1666.94 | 1644.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 13:00:00 | 1654.10 | 1666.94 | 1644.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 1695.90 | 1688.11 | 1675.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 09:45:00 | 1725.00 | 1702.35 | 1692.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 13:45:00 | 1720.80 | 1716.15 | 1703.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 14:30:00 | 1721.60 | 1720.78 | 1706.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:45:00 | 1721.90 | 1713.98 | 1706.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 13:15:00 | 1714.40 | 1713.65 | 1707.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 14:15:00 | 1719.90 | 1713.65 | 1707.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:15:00 | 1730.00 | 1714.03 | 1708.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 14:45:00 | 1745.20 | 1727.55 | 1718.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-13 11:15:00 | 1698.90 | 1719.80 | 1717.99 | SL hit (close<static) qty=1.00 sl=1706.30 alert=retest2 |

### Cycle 83 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 1701.80 | 1716.20 | 1716.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 09:15:00 | 1687.70 | 1699.19 | 1705.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 10:15:00 | 1695.70 | 1689.90 | 1695.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-16 11:00:00 | 1695.70 | 1689.90 | 1695.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1663.30 | 1684.58 | 1692.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-20 14:45:00 | 1660.00 | 1676.83 | 1680.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-21 12:15:00 | 1686.90 | 1683.21 | 1682.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 12:15:00 | 1686.90 | 1683.21 | 1682.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 15:15:00 | 1689.00 | 1685.61 | 1684.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 12:15:00 | 1700.00 | 1702.32 | 1696.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 13:00:00 | 1700.00 | 1702.32 | 1696.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 1698.50 | 1701.56 | 1697.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:00:00 | 1698.50 | 1701.56 | 1697.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 1670.30 | 1695.31 | 1694.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:00:00 | 1670.30 | 1695.31 | 1694.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 15:15:00 | 1669.50 | 1690.15 | 1692.36 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1713.40 | 1694.80 | 1694.27 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 1688.50 | 1694.97 | 1695.00 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 1712.50 | 1697.72 | 1696.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 1742.00 | 1706.58 | 1700.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 09:15:00 | 1759.70 | 1776.01 | 1753.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 09:30:00 | 1768.80 | 1776.01 | 1753.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1794.90 | 1773.33 | 1759.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 14:30:00 | 1764.70 | 1773.33 | 1759.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 1876.90 | 1885.58 | 1876.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:00:00 | 1876.90 | 1885.58 | 1876.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1878.40 | 1884.15 | 1876.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:30:00 | 1879.50 | 1884.15 | 1876.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1879.70 | 1883.26 | 1876.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:30:00 | 1875.20 | 1883.26 | 1876.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 1877.70 | 1882.15 | 1876.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 1840.00 | 1882.15 | 1876.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 1834.90 | 1872.70 | 1873.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 10:15:00 | 1804.30 | 1859.02 | 1866.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1799.90 | 1781.75 | 1798.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1799.90 | 1781.75 | 1798.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1799.90 | 1781.75 | 1798.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 1799.90 | 1781.75 | 1798.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1798.00 | 1785.00 | 1798.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:15:00 | 1802.30 | 1785.00 | 1798.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 1800.80 | 1788.16 | 1798.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:30:00 | 1800.70 | 1788.16 | 1798.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 1815.50 | 1793.63 | 1800.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:45:00 | 1815.90 | 1793.63 | 1800.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1812.00 | 1797.30 | 1801.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:30:00 | 1809.90 | 1797.30 | 1801.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1816.90 | 1801.81 | 1802.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 1844.30 | 1801.81 | 1802.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — BUY (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 09:15:00 | 1848.10 | 1811.07 | 1806.81 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1783.10 | 1801.45 | 1803.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 12:15:00 | 1769.90 | 1786.63 | 1791.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 1775.20 | 1754.17 | 1765.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 1775.20 | 1754.17 | 1765.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1775.20 | 1754.17 | 1765.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 1770.40 | 1754.17 | 1765.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1775.50 | 1758.44 | 1766.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 1779.00 | 1758.44 | 1766.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1764.00 | 1759.55 | 1766.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:30:00 | 1753.10 | 1757.68 | 1764.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:00:00 | 1753.90 | 1755.78 | 1762.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 1738.90 | 1755.43 | 1761.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 1784.20 | 1756.86 | 1758.21 | SL hit (close>static) qty=1.00 sl=1776.00 alert=retest2 |

### Cycle 92 — BUY (started 2025-06-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 10:15:00 | 1795.00 | 1764.49 | 1761.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 1828.60 | 1789.19 | 1775.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 12:15:00 | 1790.20 | 1801.51 | 1788.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 12:15:00 | 1790.20 | 1801.51 | 1788.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 1790.20 | 1801.51 | 1788.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:45:00 | 1791.00 | 1801.51 | 1788.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1797.70 | 1800.75 | 1789.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 1829.70 | 1798.90 | 1790.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 13:15:00 | 1811.70 | 1835.62 | 1835.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-06-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 13:15:00 | 1811.70 | 1835.62 | 1835.90 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 1888.10 | 1840.07 | 1836.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 12:15:00 | 1891.00 | 1850.26 | 1841.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 1910.50 | 1930.88 | 1904.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:30:00 | 1913.60 | 1930.88 | 1904.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1899.00 | 1924.51 | 1903.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1899.00 | 1924.51 | 1903.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 1899.90 | 1919.58 | 1903.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 1951.50 | 1920.08 | 1907.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 1913.80 | 1934.78 | 1937.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 1913.80 | 1934.78 | 1937.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 1910.00 | 1929.83 | 1934.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 13:15:00 | 1786.40 | 1785.63 | 1803.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 14:00:00 | 1786.40 | 1785.63 | 1803.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 1792.40 | 1786.99 | 1802.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:45:00 | 1794.70 | 1786.99 | 1802.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1803.50 | 1791.57 | 1801.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 1787.00 | 1791.57 | 1801.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:00:00 | 1786.70 | 1792.02 | 1799.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:15:00 | 1790.40 | 1792.61 | 1799.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 15:00:00 | 1790.00 | 1792.09 | 1798.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1743.90 | 1781.16 | 1792.17 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-16 09:15:00 | 1805.00 | 1779.80 | 1779.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 1805.00 | 1779.80 | 1779.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 10:15:00 | 1850.00 | 1793.84 | 1785.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 1801.50 | 1807.24 | 1797.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 1801.50 | 1807.24 | 1797.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1801.50 | 1807.24 | 1797.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:30:00 | 1792.00 | 1807.24 | 1797.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1800.00 | 1805.80 | 1798.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 1798.10 | 1805.80 | 1798.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1797.50 | 1803.45 | 1798.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:45:00 | 1796.00 | 1803.45 | 1798.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 1797.90 | 1802.34 | 1798.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 1810.60 | 1801.12 | 1798.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 11:15:00 | 1801.40 | 1800.77 | 1798.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 14:15:00 | 1792.70 | 1800.07 | 1799.16 | SL hit (close<static) qty=1.00 sl=1793.30 alert=retest2 |

### Cycle 97 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 1786.20 | 1797.29 | 1797.98 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 1810.80 | 1799.99 | 1799.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 1822.30 | 1804.46 | 1801.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 1812.20 | 1816.42 | 1810.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1812.20 | 1816.42 | 1810.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1812.20 | 1816.42 | 1810.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 1812.20 | 1816.42 | 1810.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1813.00 | 1815.74 | 1810.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:30:00 | 1816.60 | 1815.74 | 1810.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1814.70 | 1815.53 | 1810.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 1817.00 | 1815.53 | 1810.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 1822.50 | 1815.80 | 1811.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 14:00:00 | 1819.40 | 1816.52 | 1812.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 1806.70 | 1810.83 | 1810.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 1806.70 | 1810.83 | 1810.93 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 14:15:00 | 1816.00 | 1811.86 | 1811.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 10:15:00 | 1828.30 | 1814.63 | 1812.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 1801.30 | 1817.54 | 1816.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 09:15:00 | 1801.30 | 1817.54 | 1816.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 1801.30 | 1817.54 | 1816.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:00:00 | 1801.30 | 1817.54 | 1816.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 1799.90 | 1814.01 | 1814.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 1787.00 | 1808.61 | 1812.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 1796.90 | 1794.73 | 1802.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 1796.90 | 1794.73 | 1802.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1790.70 | 1793.93 | 1801.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:45:00 | 1787.00 | 1793.26 | 1799.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 13:15:00 | 1783.00 | 1793.26 | 1799.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 14:15:00 | 1782.50 | 1792.49 | 1798.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 14:45:00 | 1787.00 | 1785.49 | 1795.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 1785.10 | 1768.85 | 1779.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 1785.10 | 1768.85 | 1779.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 1796.00 | 1774.28 | 1781.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 1796.00 | 1774.28 | 1781.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 1774.10 | 1777.28 | 1781.34 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-30 15:15:00 | 1799.90 | 1785.62 | 1783.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 1799.90 | 1785.62 | 1783.86 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 11:15:00 | 1781.80 | 1782.67 | 1782.73 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 1800.50 | 1786.24 | 1784.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 13:15:00 | 1812.50 | 1791.49 | 1786.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 15:15:00 | 1829.90 | 1832.90 | 1817.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-04 09:15:00 | 1831.80 | 1832.90 | 1817.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1823.60 | 1831.04 | 1817.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 1808.10 | 1831.04 | 1817.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1839.80 | 1846.66 | 1833.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:00:00 | 1839.80 | 1846.66 | 1833.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 15:15:00 | 1855.40 | 1853.04 | 1843.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:15:00 | 1800.00 | 1853.04 | 1843.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1801.10 | 1842.65 | 1839.89 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 1794.90 | 1833.10 | 1835.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 1791.00 | 1824.68 | 1831.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 13:15:00 | 1744.40 | 1733.95 | 1758.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 14:00:00 | 1744.40 | 1733.95 | 1758.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1758.00 | 1738.76 | 1758.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:30:00 | 1769.50 | 1738.76 | 1758.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1753.10 | 1738.32 | 1752.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:00:00 | 1753.10 | 1738.32 | 1752.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 1738.50 | 1738.36 | 1750.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:15:00 | 1763.70 | 1738.36 | 1750.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 1751.30 | 1740.94 | 1750.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:30:00 | 1755.00 | 1740.94 | 1750.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 1756.00 | 1743.96 | 1751.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 15:00:00 | 1756.00 | 1743.96 | 1751.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 1757.00 | 1746.56 | 1751.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 1752.90 | 1746.56 | 1751.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 1756.00 | 1744.50 | 1748.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 1756.00 | 1744.50 | 1748.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 1750.00 | 1745.60 | 1748.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:30:00 | 1753.10 | 1745.60 | 1748.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 1728.60 | 1741.46 | 1745.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 10:30:00 | 1727.00 | 1738.07 | 1744.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:30:00 | 1726.60 | 1735.40 | 1742.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 09:15:00 | 1720.30 | 1714.98 | 1714.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 09:15:00 | 1720.30 | 1714.98 | 1714.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 11:15:00 | 1767.90 | 1728.08 | 1720.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 1782.90 | 1788.40 | 1774.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:00:00 | 1782.90 | 1788.40 | 1774.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1847.70 | 1844.51 | 1824.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 1842.20 | 1844.51 | 1824.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 1824.20 | 1843.56 | 1832.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 1824.20 | 1843.56 | 1832.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1820.10 | 1838.87 | 1831.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 1803.20 | 1838.87 | 1831.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1799.90 | 1831.08 | 1828.34 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 1791.50 | 1823.16 | 1824.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 11:15:00 | 1776.00 | 1813.73 | 1820.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 12:15:00 | 1764.60 | 1763.25 | 1777.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 13:00:00 | 1764.60 | 1763.25 | 1777.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 1771.00 | 1764.80 | 1777.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 1774.00 | 1764.80 | 1777.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 1770.00 | 1767.63 | 1776.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 1783.60 | 1767.63 | 1776.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1774.40 | 1768.98 | 1776.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 1776.00 | 1768.98 | 1776.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1780.50 | 1771.29 | 1776.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 1782.10 | 1771.29 | 1776.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 1770.90 | 1771.21 | 1776.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:30:00 | 1777.90 | 1771.21 | 1776.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 1769.90 | 1770.95 | 1775.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:45:00 | 1765.70 | 1769.34 | 1774.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 1778.70 | 1772.12 | 1774.41 | SL hit (close>static) qty=1.00 sl=1777.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 1786.00 | 1777.94 | 1776.85 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 1773.00 | 1777.85 | 1777.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 1757.00 | 1772.45 | 1775.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 1753.90 | 1751.34 | 1758.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:45:00 | 1752.80 | 1751.34 | 1758.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 1759.20 | 1748.90 | 1752.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 1759.20 | 1748.90 | 1752.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1760.90 | 1751.30 | 1753.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:30:00 | 1763.60 | 1751.30 | 1753.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1703.70 | 1726.14 | 1735.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 1695.00 | 1718.28 | 1730.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:45:00 | 1698.40 | 1708.65 | 1719.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 1747.00 | 1715.84 | 1715.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 1747.00 | 1715.84 | 1715.47 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-17 13:15:00 | 1713.00 | 1719.51 | 1719.63 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 14:15:00 | 1726.90 | 1720.99 | 1720.29 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 1707.60 | 1717.59 | 1718.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 1704.10 | 1712.78 | 1716.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 1709.50 | 1706.25 | 1711.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 1709.50 | 1706.25 | 1711.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1709.50 | 1706.25 | 1711.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1709.50 | 1706.25 | 1711.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1712.90 | 1707.58 | 1711.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 09:15:00 | 1693.00 | 1707.58 | 1711.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:00:00 | 1696.10 | 1694.26 | 1700.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 15:15:00 | 1709.90 | 1703.33 | 1702.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 15:15:00 | 1709.90 | 1703.33 | 1702.67 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 1691.80 | 1701.02 | 1701.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 1685.00 | 1694.50 | 1698.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1702.20 | 1693.21 | 1696.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1702.20 | 1693.21 | 1696.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1702.20 | 1693.21 | 1696.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:00:00 | 1681.00 | 1690.77 | 1694.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 14:15:00 | 1724.60 | 1695.29 | 1695.30 | SL hit (close>static) qty=1.00 sl=1708.70 alert=retest2 |

### Cycle 116 — BUY (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 15:15:00 | 1720.00 | 1700.23 | 1697.55 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 1684.00 | 1694.95 | 1695.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 09:15:00 | 1665.20 | 1680.82 | 1687.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 1657.00 | 1650.75 | 1665.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 09:45:00 | 1654.10 | 1650.75 | 1665.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1652.90 | 1648.95 | 1659.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 1652.90 | 1648.95 | 1659.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 1650.00 | 1645.06 | 1652.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:30:00 | 1651.30 | 1645.06 | 1652.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 1657.50 | 1647.55 | 1652.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 1657.50 | 1647.55 | 1652.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 1657.00 | 1649.44 | 1653.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 1663.20 | 1649.44 | 1653.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1654.00 | 1650.35 | 1653.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:30:00 | 1666.30 | 1650.35 | 1653.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 1649.00 | 1650.08 | 1652.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 11:15:00 | 1644.20 | 1650.08 | 1652.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:45:00 | 1648.20 | 1644.40 | 1647.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 1647.10 | 1644.40 | 1647.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:45:00 | 1640.40 | 1646.40 | 1648.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 1661.10 | 1649.34 | 1649.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-06 11:15:00 | 1661.10 | 1649.34 | 1649.61 | SL hit (close>static) qty=1.00 sl=1659.20 alert=retest2 |

### Cycle 118 — BUY (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 12:15:00 | 1660.30 | 1651.53 | 1650.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 1669.10 | 1655.05 | 1652.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 11:15:00 | 1736.40 | 1737.75 | 1721.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:45:00 | 1736.70 | 1737.75 | 1721.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1763.00 | 1748.70 | 1739.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:30:00 | 1773.80 | 1757.96 | 1744.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 10:30:00 | 1775.00 | 1777.86 | 1764.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 1770.00 | 1770.51 | 1764.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 15:15:00 | 1735.20 | 1763.44 | 1762.19 | SL hit (close<static) qty=1.00 sl=1737.50 alert=retest2 |

### Cycle 119 — SELL (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 09:15:00 | 1732.80 | 1757.32 | 1759.51 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 1772.30 | 1754.47 | 1754.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 1807.00 | 1764.98 | 1759.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 13:15:00 | 1765.20 | 1768.55 | 1762.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 13:15:00 | 1765.20 | 1768.55 | 1762.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1765.20 | 1768.55 | 1762.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:45:00 | 1792.30 | 1774.56 | 1766.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 12:00:00 | 1787.50 | 1798.79 | 1794.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 1781.20 | 1792.61 | 1792.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 1781.20 | 1792.61 | 1792.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 1769.20 | 1784.13 | 1788.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 1783.90 | 1782.99 | 1787.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 1783.90 | 1782.99 | 1787.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1783.90 | 1782.99 | 1787.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 1783.90 | 1782.99 | 1787.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1782.90 | 1782.97 | 1786.73 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 1826.60 | 1792.76 | 1790.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 1848.50 | 1811.00 | 1799.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 12:15:00 | 1853.60 | 1856.70 | 1835.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:00:00 | 1853.60 | 1856.70 | 1835.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1830.30 | 1847.73 | 1837.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 1830.30 | 1847.73 | 1837.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1826.00 | 1843.39 | 1836.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 1826.00 | 1843.39 | 1836.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1855.00 | 1861.19 | 1852.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:30:00 | 1853.70 | 1861.19 | 1852.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1862.00 | 1863.67 | 1857.17 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 11:15:00 | 1840.70 | 1854.95 | 1856.40 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 1867.80 | 1857.88 | 1857.38 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1830.40 | 1853.47 | 1855.68 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 1880.40 | 1859.26 | 1857.95 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 1836.70 | 1855.46 | 1857.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 1831.70 | 1850.71 | 1855.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 13:15:00 | 1768.00 | 1758.01 | 1782.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 14:00:00 | 1768.00 | 1758.01 | 1782.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1803.20 | 1771.43 | 1782.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 1803.20 | 1771.43 | 1782.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1794.90 | 1776.12 | 1783.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:30:00 | 1803.00 | 1776.12 | 1783.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 1798.10 | 1788.10 | 1787.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 1800.20 | 1793.28 | 1790.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 15:15:00 | 1790.00 | 1804.59 | 1800.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 15:15:00 | 1790.00 | 1804.59 | 1800.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1790.00 | 1804.59 | 1800.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1784.10 | 1804.59 | 1800.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1767.20 | 1797.12 | 1797.67 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 10:15:00 | 1806.10 | 1791.97 | 1791.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 11:15:00 | 1812.80 | 1796.14 | 1793.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 10:15:00 | 1817.30 | 1818.00 | 1807.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 10:45:00 | 1817.70 | 1818.00 | 1807.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1839.80 | 1826.71 | 1816.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 1819.60 | 1826.71 | 1816.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 1826.40 | 1838.13 | 1828.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 1818.10 | 1838.13 | 1828.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1831.20 | 1836.74 | 1828.94 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 1784.60 | 1818.79 | 1821.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 13:15:00 | 1775.30 | 1804.46 | 1814.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 12:15:00 | 1746.10 | 1739.00 | 1761.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 13:00:00 | 1746.10 | 1739.00 | 1761.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 1761.50 | 1743.50 | 1761.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 13:45:00 | 1763.70 | 1743.50 | 1761.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 1765.60 | 1747.92 | 1761.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 1765.60 | 1747.92 | 1761.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 15:15:00 | 1763.60 | 1751.06 | 1761.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 09:30:00 | 1746.70 | 1751.83 | 1761.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 12:15:00 | 1780.80 | 1765.98 | 1765.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 1780.80 | 1765.98 | 1765.88 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 1751.00 | 1766.48 | 1767.20 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 1776.90 | 1766.31 | 1765.91 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2025-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 15:15:00 | 1756.70 | 1765.28 | 1765.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 1738.40 | 1759.11 | 1762.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 09:15:00 | 1751.30 | 1750.84 | 1756.38 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-02 11:45:00 | 1732.20 | 1747.31 | 1753.84 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1747.00 | 1735.18 | 1741.94 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-03 12:15:00 | 1747.00 | 1735.18 | 1741.94 | SL hit (close>ema400) qty=1.00 sl=1741.94 alert=retest1 |

### Cycle 136 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 1750.40 | 1746.50 | 1746.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 10:15:00 | 1779.30 | 1753.06 | 1749.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 12:15:00 | 1755.70 | 1757.77 | 1752.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-04 13:00:00 | 1755.70 | 1757.77 | 1752.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1754.80 | 1758.54 | 1754.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:30:00 | 1752.40 | 1758.54 | 1754.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1753.20 | 1757.47 | 1754.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 1747.80 | 1757.47 | 1754.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1755.40 | 1757.06 | 1754.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 14:45:00 | 1759.80 | 1757.14 | 1755.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 09:15:00 | 1736.00 | 1753.05 | 1753.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1736.00 | 1753.05 | 1753.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1722.00 | 1743.45 | 1748.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 1732.80 | 1732.55 | 1739.80 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-09 12:00:00 | 1721.60 | 1730.36 | 1738.15 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1695.50 | 1682.41 | 1693.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 1695.50 | 1682.41 | 1693.05 | SL hit (close>ema400) qty=1.00 sl=1693.05 alert=retest1 |

### Cycle 138 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 1698.60 | 1691.66 | 1691.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 09:15:00 | 1745.00 | 1703.84 | 1696.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 1728.80 | 1731.48 | 1717.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 11:15:00 | 1728.60 | 1730.56 | 1719.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1728.60 | 1730.56 | 1719.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:30:00 | 1724.00 | 1730.56 | 1719.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1745.00 | 1746.01 | 1732.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 1734.00 | 1746.01 | 1732.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1738.70 | 1744.55 | 1733.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 1736.00 | 1744.55 | 1733.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1729.00 | 1740.10 | 1733.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 14:00:00 | 1729.00 | 1740.10 | 1733.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 1759.80 | 1744.04 | 1736.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 1766.10 | 1744.04 | 1736.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 1769.20 | 1817.51 | 1817.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 1769.20 | 1817.51 | 1817.76 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1808.80 | 1803.01 | 1802.55 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 10:15:00 | 1777.70 | 1797.94 | 1800.29 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1826.70 | 1803.56 | 1801.41 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 1795.40 | 1803.82 | 1804.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 1785.00 | 1800.06 | 1802.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 13:15:00 | 1809.80 | 1797.68 | 1800.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 13:15:00 | 1809.80 | 1797.68 | 1800.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 1809.80 | 1797.68 | 1800.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 1809.80 | 1797.68 | 1800.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 1787.10 | 1795.56 | 1799.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 1759.30 | 1794.03 | 1798.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 15:00:00 | 1782.10 | 1780.81 | 1787.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 1692.99 | 1732.61 | 1750.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-13 11:15:00 | 1671.33 | 1697.00 | 1721.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-16 09:15:00 | 1603.89 | 1638.42 | 1667.80 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 144 — BUY (started 2026-01-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 14:15:00 | 1647.10 | 1621.77 | 1618.71 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 11:15:00 | 1598.20 | 1617.21 | 1617.94 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 1625.30 | 1618.91 | 1618.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 15:15:00 | 1627.00 | 1620.53 | 1619.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 1615.00 | 1619.42 | 1618.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 1615.00 | 1619.42 | 1618.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 1615.00 | 1619.42 | 1618.89 | EMA400 retest candle locked (from upside) |

### Cycle 147 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 1612.30 | 1618.00 | 1618.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 1599.90 | 1612.30 | 1615.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 14:15:00 | 1617.00 | 1613.24 | 1615.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 14:15:00 | 1617.00 | 1613.24 | 1615.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1617.00 | 1613.24 | 1615.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 1617.00 | 1613.24 | 1615.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1615.00 | 1613.59 | 1615.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 1594.30 | 1613.59 | 1615.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 1602.00 | 1574.11 | 1570.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 1602.00 | 1574.11 | 1570.33 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 1553.50 | 1567.04 | 1568.85 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 1582.10 | 1567.12 | 1565.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 1635.00 | 1583.55 | 1573.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 1612.70 | 1624.22 | 1604.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 1602.80 | 1619.94 | 1604.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 1602.80 | 1619.94 | 1604.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:00:00 | 1602.80 | 1619.94 | 1604.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 1616.90 | 1619.33 | 1605.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:45:00 | 1603.80 | 1619.33 | 1605.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1608.30 | 1617.12 | 1605.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 1608.30 | 1617.12 | 1605.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1614.00 | 1616.50 | 1606.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:00:00 | 1614.00 | 1616.50 | 1606.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 1604.30 | 1614.06 | 1606.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:45:00 | 1601.80 | 1614.06 | 1606.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 1610.00 | 1613.25 | 1606.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 1591.20 | 1613.25 | 1606.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 1600.10 | 1610.62 | 1606.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 12:00:00 | 1622.70 | 1612.14 | 1607.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 10:15:00 | 1678.10 | 1691.48 | 1692.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2026-02-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 10:15:00 | 1678.10 | 1691.48 | 1692.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 12:15:00 | 1665.50 | 1678.94 | 1684.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 13:15:00 | 1682.20 | 1679.59 | 1684.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 13:15:00 | 1682.20 | 1679.59 | 1684.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 1682.20 | 1679.59 | 1684.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:00:00 | 1682.20 | 1679.59 | 1684.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 1682.30 | 1680.13 | 1683.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:15:00 | 1687.00 | 1680.13 | 1683.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 1687.00 | 1681.51 | 1684.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1668.70 | 1681.51 | 1684.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 15:15:00 | 1671.90 | 1660.43 | 1667.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 14:15:00 | 1705.10 | 1674.27 | 1671.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1705.10 | 1674.27 | 1671.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 11:15:00 | 1708.70 | 1689.77 | 1680.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 1696.30 | 1697.69 | 1688.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:15:00 | 1691.40 | 1697.69 | 1688.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1688.00 | 1695.75 | 1688.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 1688.00 | 1695.75 | 1688.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1679.20 | 1692.44 | 1687.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1679.20 | 1692.44 | 1687.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1679.50 | 1689.85 | 1686.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 1682.30 | 1689.85 | 1686.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 1680.00 | 1684.53 | 1684.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 09:15:00 | 1673.70 | 1682.37 | 1683.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 11:15:00 | 1686.70 | 1682.52 | 1683.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 11:15:00 | 1686.70 | 1682.52 | 1683.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 1686.70 | 1682.52 | 1683.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:30:00 | 1685.00 | 1682.52 | 1683.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1684.40 | 1682.89 | 1683.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 12:30:00 | 1685.30 | 1682.89 | 1683.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1685.80 | 1683.48 | 1683.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:45:00 | 1687.00 | 1683.48 | 1683.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1686.10 | 1684.00 | 1684.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 1690.10 | 1684.00 | 1684.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — BUY (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 15:15:00 | 1692.40 | 1685.68 | 1684.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1700.60 | 1688.66 | 1686.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 15:15:00 | 1694.00 | 1695.54 | 1691.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 15:15:00 | 1694.00 | 1695.54 | 1691.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1694.00 | 1695.54 | 1691.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:15:00 | 1681.00 | 1695.54 | 1691.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1678.30 | 1692.09 | 1690.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 1672.80 | 1692.09 | 1690.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 155 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 1676.00 | 1688.87 | 1689.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 11:15:00 | 1672.40 | 1685.58 | 1687.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 10:15:00 | 1660.50 | 1657.79 | 1665.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:00:00 | 1660.50 | 1657.79 | 1665.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 1657.60 | 1657.75 | 1664.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 1654.20 | 1659.45 | 1663.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 1571.49 | 1622.08 | 1638.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1575.90 | 1574.53 | 1592.63 | SL hit (close>ema200) qty=0.50 sl=1574.53 alert=retest2 |

### Cycle 156 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 1606.50 | 1596.24 | 1594.91 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1568.70 | 1596.58 | 1596.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 10:15:00 | 1557.30 | 1577.27 | 1583.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1487.80 | 1475.03 | 1494.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 1487.80 | 1475.03 | 1494.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 1488.00 | 1477.63 | 1494.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 1517.60 | 1477.63 | 1494.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1513.40 | 1484.78 | 1496.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 1517.80 | 1484.78 | 1496.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1532.60 | 1494.35 | 1499.39 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 1528.00 | 1506.46 | 1504.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 1538.90 | 1520.13 | 1511.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 14:15:00 | 1590.00 | 1594.10 | 1571.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 15:00:00 | 1590.00 | 1594.10 | 1571.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 1579.60 | 1591.68 | 1580.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 1579.60 | 1591.68 | 1580.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1575.00 | 1588.34 | 1580.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 1551.40 | 1588.34 | 1580.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 1540.20 | 1578.72 | 1576.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:00:00 | 1540.20 | 1578.72 | 1576.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 1543.90 | 1571.75 | 1573.79 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 1580.50 | 1566.83 | 1566.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1596.60 | 1572.79 | 1569.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 1579.00 | 1585.20 | 1577.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 15:00:00 | 1579.00 | 1585.20 | 1577.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 1589.90 | 1586.14 | 1579.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 1566.60 | 1586.14 | 1579.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1557.80 | 1580.47 | 1577.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 1560.30 | 1580.47 | 1577.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 1544.40 | 1573.26 | 1574.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 1537.00 | 1559.95 | 1567.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 12:15:00 | 1525.00 | 1524.65 | 1542.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-30 12:45:00 | 1521.10 | 1524.65 | 1542.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1496.00 | 1510.32 | 1529.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 1485.30 | 1510.32 | 1529.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 10:15:00 | 1515.10 | 1504.07 | 1503.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 1515.10 | 1504.07 | 1503.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 1533.30 | 1515.70 | 1509.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 14:15:00 | 1520.00 | 1522.58 | 1516.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 15:00:00 | 1520.00 | 1522.58 | 1516.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 1520.00 | 1522.06 | 1516.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1551.50 | 1522.06 | 1516.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 14:15:00 | 1524.00 | 1528.93 | 1529.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2026-04-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 14:15:00 | 1524.00 | 1528.93 | 1529.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 1486.10 | 1520.18 | 1525.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 15:15:00 | 1517.00 | 1516.82 | 1520.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-15 09:15:00 | 1541.60 | 1516.82 | 1520.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1548.50 | 1523.16 | 1523.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:30:00 | 1564.50 | 1523.16 | 1523.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 1550.00 | 1528.53 | 1525.85 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2026-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 13:15:00 | 1524.80 | 1530.40 | 1530.47 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 1534.30 | 1530.33 | 1530.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 1559.50 | 1537.72 | 1534.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 1538.20 | 1541.16 | 1536.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 13:15:00 | 1538.20 | 1541.16 | 1536.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 1538.20 | 1541.16 | 1536.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:00:00 | 1538.20 | 1541.16 | 1536.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 1537.60 | 1540.45 | 1536.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:45:00 | 1532.10 | 1540.45 | 1536.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 1534.00 | 1539.16 | 1536.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:15:00 | 1543.80 | 1539.16 | 1536.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1565.80 | 1544.49 | 1539.28 | EMA400 retest candle locked (from upside) |

### Cycle 167 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 1529.90 | 1551.43 | 1553.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 15:15:00 | 1519.90 | 1538.42 | 1546.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 14:15:00 | 1507.20 | 1501.75 | 1514.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-28 15:00:00 | 1507.20 | 1501.75 | 1514.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 1518.00 | 1505.00 | 1514.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:15:00 | 1539.90 | 1505.00 | 1514.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 1527.10 | 1509.42 | 1515.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 12:45:00 | 1511.70 | 1514.83 | 1516.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 15:15:00 | 1521.90 | 1518.65 | 1518.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 15:15:00 | 1521.90 | 1518.65 | 1518.28 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1497.70 | 1514.46 | 1516.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 12:15:00 | 1490.00 | 1499.55 | 1504.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1492.00 | 1491.62 | 1498.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 1492.00 | 1491.62 | 1498.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 1492.00 | 1491.62 | 1498.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:15:00 | 1492.00 | 1491.62 | 1498.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 1490.50 | 1491.40 | 1497.72 | EMA400 retest candle locked (from downside) |

### Cycle 170 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1513.20 | 1501.12 | 1500.66 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 1494.80 | 1505.31 | 1506.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-11 09:15:00 | 1468.90 | 1495.01 | 1500.60 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-14 13:30:00 | 911.70 | 2024-05-21 15:15:00 | 942.00 | STOP_HIT | 1.00 | 3.32% |
| BUY | retest2 | 2024-05-15 09:15:00 | 933.15 | 2024-05-21 15:15:00 | 942.00 | STOP_HIT | 1.00 | 0.95% |
| BUY | retest2 | 2024-05-15 10:45:00 | 905.90 | 2024-05-21 15:15:00 | 942.00 | STOP_HIT | 1.00 | 3.98% |
| BUY | retest2 | 2024-06-11 13:15:00 | 1090.35 | 2024-06-14 11:15:00 | 1066.35 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2024-06-11 14:15:00 | 1091.65 | 2024-06-14 11:15:00 | 1066.35 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1086.40 | 2024-06-14 11:15:00 | 1066.35 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-06-14 09:45:00 | 1089.20 | 2024-06-14 11:15:00 | 1066.35 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-06-19 14:30:00 | 1151.05 | 2024-06-21 14:15:00 | 1120.00 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-06-20 09:15:00 | 1149.55 | 2024-06-21 14:15:00 | 1120.00 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-06-20 09:45:00 | 1148.00 | 2024-06-21 14:15:00 | 1120.00 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2024-06-20 10:15:00 | 1151.95 | 2024-06-21 14:15:00 | 1120.00 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-06-27 12:30:00 | 1195.90 | 2024-06-27 13:15:00 | 1172.90 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-07-10 10:15:00 | 1045.00 | 2024-07-12 10:15:00 | 1084.05 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2024-07-11 12:45:00 | 1055.00 | 2024-07-12 10:15:00 | 1084.05 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2024-07-16 09:15:00 | 1142.55 | 2024-07-22 13:15:00 | 1108.70 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2024-07-30 09:15:00 | 1192.90 | 2024-07-31 14:15:00 | 1136.05 | STOP_HIT | 1.00 | -4.77% |
| BUY | retest2 | 2024-08-29 15:15:00 | 1185.95 | 2024-09-04 13:15:00 | 1215.10 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2024-09-19 12:15:00 | 1373.70 | 2024-09-24 13:15:00 | 1382.10 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2024-09-19 14:45:00 | 1357.90 | 2024-09-24 13:15:00 | 1382.10 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2024-09-19 15:15:00 | 1363.60 | 2024-09-24 13:15:00 | 1382.10 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2024-09-20 09:30:00 | 1359.55 | 2024-09-24 13:15:00 | 1382.10 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2024-09-20 12:45:00 | 1399.20 | 2024-09-24 13:15:00 | 1382.10 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-09-30 13:45:00 | 1444.95 | 2024-10-01 10:15:00 | 1412.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-09-30 14:45:00 | 1445.00 | 2024-10-01 10:15:00 | 1412.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-10-01 09:45:00 | 1448.60 | 2024-10-01 10:15:00 | 1412.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2024-10-04 12:15:00 | 1424.80 | 2024-10-08 11:15:00 | 1418.65 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2024-10-08 10:00:00 | 1426.30 | 2024-10-08 11:15:00 | 1418.65 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2024-10-08 10:30:00 | 1425.50 | 2024-10-08 11:15:00 | 1418.65 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-10-08 11:15:00 | 1424.25 | 2024-10-08 11:15:00 | 1418.65 | STOP_HIT | 1.00 | 0.39% |
| BUY | retest2 | 2024-10-10 09:15:00 | 1442.55 | 2024-10-21 11:15:00 | 1485.30 | STOP_HIT | 1.00 | 2.96% |
| BUY | retest2 | 2024-10-10 15:00:00 | 1440.25 | 2024-10-21 11:15:00 | 1485.30 | STOP_HIT | 1.00 | 3.13% |
| BUY | retest2 | 2024-10-11 10:30:00 | 1440.00 | 2024-10-21 11:15:00 | 1485.30 | STOP_HIT | 1.00 | 3.15% |
| BUY | retest2 | 2024-10-11 11:15:00 | 1440.00 | 2024-10-21 11:15:00 | 1485.30 | STOP_HIT | 1.00 | 3.15% |
| BUY | retest2 | 2024-10-14 10:45:00 | 1467.60 | 2024-10-21 11:15:00 | 1485.30 | STOP_HIT | 1.00 | 1.21% |
| SELL | retest2 | 2024-10-24 09:45:00 | 1445.70 | 2024-10-28 14:15:00 | 1475.40 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-10-24 11:15:00 | 1450.00 | 2024-10-28 14:15:00 | 1475.40 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-10-24 13:00:00 | 1450.55 | 2024-10-28 14:15:00 | 1475.40 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-10-25 09:15:00 | 1450.70 | 2024-10-28 14:15:00 | 1475.40 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-10-25 14:15:00 | 1424.55 | 2024-10-28 14:15:00 | 1475.40 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2024-10-25 14:45:00 | 1421.45 | 2024-10-28 14:15:00 | 1475.40 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2024-11-04 10:45:00 | 1384.15 | 2024-11-06 09:15:00 | 1425.00 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-11-04 13:15:00 | 1385.85 | 2024-11-06 09:15:00 | 1425.00 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2024-11-12 09:30:00 | 1369.70 | 2024-11-13 10:15:00 | 1415.80 | STOP_HIT | 1.00 | -3.37% |
| BUY | retest2 | 2024-11-18 12:30:00 | 1402.90 | 2024-11-18 13:15:00 | 1402.10 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2024-11-18 13:15:00 | 1403.05 | 2024-11-18 13:15:00 | 1402.10 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-11-19 14:45:00 | 1396.75 | 2024-11-22 09:15:00 | 1326.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 14:45:00 | 1396.75 | 2024-11-22 15:15:00 | 1340.00 | STOP_HIT | 0.50 | 4.06% |
| BUY | retest2 | 2024-12-05 15:00:00 | 1412.55 | 2024-12-10 15:15:00 | 1405.55 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-12-10 13:45:00 | 1407.00 | 2024-12-10 15:15:00 | 1405.55 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2024-12-10 14:45:00 | 1406.45 | 2024-12-10 15:15:00 | 1405.55 | STOP_HIT | 1.00 | -0.06% |
| BUY | retest2 | 2024-12-18 09:15:00 | 1469.00 | 2024-12-23 10:15:00 | 1466.00 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-12-23 09:30:00 | 1466.80 | 2024-12-23 10:15:00 | 1466.00 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2024-12-30 09:15:00 | 1448.85 | 2024-12-30 14:15:00 | 1514.30 | STOP_HIT | 1.00 | -4.52% |
| SELL | retest2 | 2024-12-30 13:15:00 | 1456.25 | 2024-12-30 14:15:00 | 1514.30 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2025-01-07 10:45:00 | 1500.15 | 2025-01-08 12:15:00 | 1479.55 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-01-14 12:15:00 | 1422.20 | 2025-01-15 15:15:00 | 1351.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-14 12:15:00 | 1422.20 | 2025-01-16 10:15:00 | 1370.50 | STOP_HIT | 0.50 | 3.64% |
| SELL | retest2 | 2025-01-14 13:15:00 | 1419.15 | 2025-01-17 09:15:00 | 1348.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-14 13:15:00 | 1419.15 | 2025-01-21 09:15:00 | 1277.24 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-01 12:45:00 | 1369.55 | 2025-02-05 12:15:00 | 1357.60 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-02-01 13:15:00 | 1368.90 | 2025-02-05 12:15:00 | 1357.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1371.15 | 2025-02-05 12:15:00 | 1357.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-02-03 15:15:00 | 1408.40 | 2025-02-05 12:15:00 | 1357.60 | STOP_HIT | 1.00 | -3.61% |
| BUY | retest2 | 2025-03-05 09:30:00 | 1324.40 | 2025-03-05 14:15:00 | 1456.84 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-05 11:00:00 | 1306.95 | 2025-03-05 14:15:00 | 1437.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-05 12:15:00 | 1305.95 | 2025-03-05 14:15:00 | 1436.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-18 09:15:00 | 1325.00 | 2025-03-24 09:15:00 | 1350.05 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-03-28 15:15:00 | 1476.00 | 2025-04-02 14:15:00 | 1445.65 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-04-01 11:15:00 | 1477.50 | 2025-04-02 14:15:00 | 1445.65 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-04-01 14:45:00 | 1473.60 | 2025-04-02 14:15:00 | 1445.65 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-04-15 14:15:00 | 1525.20 | 2025-04-23 09:15:00 | 1677.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-16 14:30:00 | 1525.70 | 2025-04-23 09:15:00 | 1678.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-17 11:00:00 | 1530.10 | 2025-04-23 09:15:00 | 1683.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-29 10:15:00 | 1581.70 | 2025-04-30 09:15:00 | 1655.20 | STOP_HIT | 1.00 | -4.65% |
| BUY | retest2 | 2025-05-08 09:45:00 | 1725.00 | 2025-05-13 11:15:00 | 1698.90 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-05-08 13:45:00 | 1720.80 | 2025-05-13 11:15:00 | 1698.90 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-05-08 14:30:00 | 1721.60 | 2025-05-13 11:15:00 | 1698.90 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-05-09 11:45:00 | 1721.90 | 2025-05-13 12:15:00 | 1701.80 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-05-09 14:15:00 | 1719.90 | 2025-05-13 12:15:00 | 1701.80 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-05-12 09:15:00 | 1730.00 | 2025-05-13 12:15:00 | 1701.80 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-05-12 14:45:00 | 1745.20 | 2025-05-13 12:15:00 | 1701.80 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-05-20 14:45:00 | 1660.00 | 2025-05-21 12:15:00 | 1686.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-06-18 12:30:00 | 1753.10 | 2025-06-20 09:15:00 | 1784.20 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-06-18 15:00:00 | 1753.90 | 2025-06-20 09:15:00 | 1784.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-06-19 10:15:00 | 1738.90 | 2025-06-20 09:15:00 | 1784.20 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1829.70 | 2025-06-26 13:15:00 | 1811.70 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-07-01 15:00:00 | 1951.50 | 2025-07-04 10:15:00 | 1913.80 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-07-11 10:15:00 | 1787.00 | 2025-07-16 09:15:00 | 1805.00 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-11 13:00:00 | 1786.70 | 2025-07-16 09:15:00 | 1805.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-07-11 14:15:00 | 1790.40 | 2025-07-16 09:15:00 | 1805.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-11 15:00:00 | 1790.00 | 2025-07-16 09:15:00 | 1805.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-07-18 09:15:00 | 1810.60 | 2025-07-18 14:15:00 | 1792.70 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-18 11:15:00 | 1801.40 | 2025-07-18 14:15:00 | 1792.70 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-07-22 12:15:00 | 1817.00 | 2025-07-23 13:15:00 | 1806.70 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-07-22 12:45:00 | 1822.50 | 2025-07-23 13:15:00 | 1806.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-22 14:00:00 | 1819.40 | 2025-07-23 13:15:00 | 1806.70 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-07-28 12:45:00 | 1787.00 | 2025-07-30 15:15:00 | 1799.90 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-07-28 13:15:00 | 1783.00 | 2025-07-30 15:15:00 | 1799.90 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-07-28 14:15:00 | 1782.50 | 2025-07-30 15:15:00 | 1799.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-07-28 14:45:00 | 1787.00 | 2025-07-30 15:15:00 | 1799.90 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-08-13 10:30:00 | 1727.00 | 2025-08-19 09:15:00 | 1720.30 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2025-08-13 11:30:00 | 1726.60 | 2025-08-19 09:15:00 | 1720.30 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2025-09-02 13:45:00 | 1765.70 | 2025-09-03 09:15:00 | 1778.70 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-12 12:15:00 | 1695.00 | 2025-09-16 11:15:00 | 1747.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-09-15 10:45:00 | 1698.40 | 2025-09-16 11:15:00 | 1747.00 | STOP_HIT | 1.00 | -2.86% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1693.00 | 2025-09-23 15:15:00 | 1709.90 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-09-23 10:00:00 | 1696.10 | 2025-09-23 15:15:00 | 1709.90 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-09-25 11:00:00 | 1681.00 | 2025-09-25 14:15:00 | 1724.60 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-10-03 11:15:00 | 1644.20 | 2025-10-06 11:15:00 | 1661.10 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-10-06 09:45:00 | 1648.20 | 2025-10-06 11:15:00 | 1661.10 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-10-06 10:15:00 | 1647.10 | 2025-10-06 11:15:00 | 1661.10 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-10-06 10:45:00 | 1640.40 | 2025-10-06 11:15:00 | 1661.10 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-10-13 10:30:00 | 1773.80 | 2025-10-14 15:15:00 | 1735.20 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-10-14 10:30:00 | 1775.00 | 2025-10-14 15:15:00 | 1735.20 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-10-14 14:30:00 | 1770.00 | 2025-10-14 15:15:00 | 1735.20 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-10-17 11:45:00 | 1792.30 | 2025-10-23 14:15:00 | 1781.20 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-10-23 12:00:00 | 1787.50 | 2025-10-23 14:15:00 | 1781.20 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-11-26 09:30:00 | 1746.70 | 2025-11-26 12:15:00 | 1780.80 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest1 | 2025-12-02 11:45:00 | 1732.20 | 2025-12-03 12:15:00 | 1747.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-12-05 14:45:00 | 1759.80 | 2025-12-08 09:15:00 | 1736.00 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest1 | 2025-12-09 12:00:00 | 1721.60 | 2025-12-11 13:15:00 | 1695.50 | STOP_HIT | 1.00 | 1.52% |
| SELL | retest2 | 2025-12-12 09:15:00 | 1684.60 | 2025-12-15 12:15:00 | 1698.60 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-18 15:15:00 | 1766.10 | 2025-12-30 10:15:00 | 1769.20 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2026-01-07 09:15:00 | 1759.30 | 2026-01-12 11:15:00 | 1692.99 | PARTIAL | 0.50 | 3.77% |
| SELL | retest2 | 2026-01-07 15:00:00 | 1782.10 | 2026-01-13 11:15:00 | 1671.33 | PARTIAL | 0.50 | 6.22% |
| SELL | retest2 | 2026-01-07 09:15:00 | 1759.30 | 2026-01-16 09:15:00 | 1603.89 | TARGET_HIT | 0.50 | 8.83% |
| SELL | retest2 | 2026-01-07 15:00:00 | 1782.10 | 2026-01-19 11:15:00 | 1616.00 | STOP_HIT | 0.50 | 9.32% |
| SELL | retest2 | 2026-01-27 09:15:00 | 1594.30 | 2026-01-30 11:15:00 | 1602.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2026-02-05 12:00:00 | 1622.70 | 2026-02-11 10:15:00 | 1678.10 | STOP_HIT | 1.00 | 3.41% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1668.70 | 2026-02-17 14:15:00 | 1705.10 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-02-16 15:15:00 | 1671.90 | 2026-02-17 14:15:00 | 1705.10 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-02-26 15:15:00 | 1654.20 | 2026-03-02 09:15:00 | 1571.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 1654.20 | 2026-03-05 09:15:00 | 1575.90 | STOP_HIT | 0.50 | 4.73% |
| SELL | retest2 | 2026-04-01 10:15:00 | 1485.30 | 2026-04-06 10:15:00 | 1515.10 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1551.50 | 2026-04-10 14:15:00 | 1524.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-04-29 12:45:00 | 1511.70 | 2026-04-29 15:15:00 | 1521.90 | STOP_HIT | 1.00 | -0.67% |
