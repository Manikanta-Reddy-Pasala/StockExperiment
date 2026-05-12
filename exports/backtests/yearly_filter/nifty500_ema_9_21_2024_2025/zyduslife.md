# Zydus Lifesciences Ltd. (ZYDUSLIFE)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 939.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 159 |
| ALERT1 | 105 |
| ALERT2 | 105 |
| ALERT2_SKIP | 50 |
| ALERT3 | 265 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 130 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 135 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 137 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 100
- **Target hits / Stop hits / Partials:** 0 / 132 / 5
- **Avg / median % per leg:** -0.43% / -0.73%
- **Sum % (uncompounded):** -59.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 63 | 20 | 31.7% | 0 | 63 | 0 | -0.53% | -33.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.46% | -1.5% |
| BUY @ 3rd Alert (retest2) | 62 | 20 | 32.3% | 0 | 62 | 0 | -0.51% | -31.7% |
| SELL (all) | 74 | 17 | 23.0% | 0 | 69 | 5 | -0.36% | -26.3% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.44% | 1.4% |
| SELL @ 3rd Alert (retest2) | 73 | 16 | 21.9% | 0 | 68 | 5 | -0.38% | -27.7% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.01% | -0.0% |
| retest2 (combined) | 135 | 36 | 26.7% | 0 | 130 | 5 | -0.44% | -59.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 13:15:00 | 993.75 | 987.32 | 986.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 09:15:00 | 1000.60 | 991.04 | 988.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 10:15:00 | 990.70 | 990.98 | 988.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-14 11:00:00 | 990.70 | 990.98 | 988.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 11:15:00 | 987.70 | 990.32 | 988.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 11:45:00 | 986.35 | 990.32 | 988.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 12:15:00 | 989.00 | 990.06 | 988.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 12:30:00 | 985.55 | 990.06 | 988.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 984.45 | 988.94 | 988.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:00:00 | 984.45 | 988.94 | 988.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 14:15:00 | 982.50 | 987.65 | 987.94 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 09:15:00 | 999.90 | 990.04 | 988.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 12:15:00 | 1008.00 | 1000.38 | 996.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 14:15:00 | 1085.65 | 1095.29 | 1069.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 15:00:00 | 1085.65 | 1095.29 | 1069.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1072.55 | 1089.48 | 1071.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 1065.85 | 1089.48 | 1071.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1071.05 | 1085.80 | 1071.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:45:00 | 1072.35 | 1085.80 | 1071.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 1067.80 | 1082.20 | 1070.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:45:00 | 1064.25 | 1082.20 | 1070.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 1069.35 | 1079.63 | 1070.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:30:00 | 1060.25 | 1079.63 | 1070.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 1067.20 | 1077.14 | 1070.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 13:30:00 | 1066.20 | 1077.14 | 1070.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 1079.15 | 1077.54 | 1071.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 15:15:00 | 1087.00 | 1077.54 | 1071.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 12:45:00 | 1083.45 | 1082.90 | 1080.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 1066.70 | 1077.26 | 1078.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 09:15:00 | 1066.70 | 1077.26 | 1078.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 11:15:00 | 1062.10 | 1072.63 | 1075.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 15:15:00 | 1054.00 | 1053.41 | 1060.97 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-05-29 09:15:00 | 1042.25 | 1053.41 | 1060.97 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 1027.25 | 1012.93 | 1023.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 1027.25 | 1012.93 | 1023.33 | SL hit (close>ema400) qty=1.00 sl=1023.33 alert=retest1 |

### Cycle 5 — BUY (started 2024-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 15:15:00 | 1038.50 | 1027.73 | 1027.35 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 977.35 | 1017.44 | 1022.73 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 1049.60 | 1021.31 | 1019.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 11:15:00 | 1056.60 | 1028.37 | 1023.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 11:15:00 | 1038.15 | 1044.31 | 1036.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 12:00:00 | 1038.15 | 1044.31 | 1036.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 1046.15 | 1044.61 | 1039.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 10:45:00 | 1054.35 | 1047.42 | 1040.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 13:30:00 | 1055.70 | 1051.39 | 1044.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 14:30:00 | 1056.05 | 1052.56 | 1045.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-18 12:15:00 | 1077.05 | 1089.73 | 1090.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 12:15:00 | 1077.05 | 1089.73 | 1090.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 12:15:00 | 1075.70 | 1080.19 | 1084.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 10:15:00 | 1077.50 | 1075.88 | 1080.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 11:00:00 | 1077.50 | 1075.88 | 1080.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 11:15:00 | 1080.55 | 1076.82 | 1080.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 12:00:00 | 1080.55 | 1076.82 | 1080.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 12:15:00 | 1080.15 | 1077.48 | 1080.49 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 1097.30 | 1085.01 | 1083.35 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 13:15:00 | 1079.00 | 1084.28 | 1084.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 09:15:00 | 1078.65 | 1082.47 | 1083.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 1075.90 | 1073.68 | 1077.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 10:45:00 | 1074.50 | 1073.68 | 1077.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 1083.45 | 1075.64 | 1077.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 12:00:00 | 1083.45 | 1075.64 | 1077.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 1074.95 | 1075.50 | 1077.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 13:45:00 | 1072.85 | 1074.71 | 1076.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:00:00 | 1071.95 | 1072.43 | 1075.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 13:15:00 | 1079.95 | 1072.59 | 1071.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2024-06-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 13:15:00 | 1079.95 | 1072.59 | 1071.65 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 10:15:00 | 1065.70 | 1071.22 | 1071.41 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 15:15:00 | 1073.00 | 1070.34 | 1070.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 09:15:00 | 1112.20 | 1078.71 | 1073.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 14:15:00 | 1158.70 | 1162.59 | 1146.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-08 15:00:00 | 1158.70 | 1162.59 | 1146.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 1148.50 | 1162.34 | 1156.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 1148.50 | 1162.34 | 1156.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1149.10 | 1159.69 | 1155.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 1140.15 | 1159.69 | 1155.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 1158.25 | 1159.47 | 1156.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:30:00 | 1167.45 | 1161.44 | 1157.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 11:30:00 | 1163.00 | 1166.36 | 1161.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 13:00:00 | 1162.45 | 1165.58 | 1161.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 14:15:00 | 1162.35 | 1164.49 | 1161.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 1163.20 | 1164.23 | 1161.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 15:00:00 | 1163.20 | 1164.23 | 1161.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 1173.40 | 1166.07 | 1162.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 09:15:00 | 1176.60 | 1166.07 | 1162.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 10:00:00 | 1178.55 | 1168.56 | 1164.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 13:30:00 | 1176.90 | 1171.37 | 1167.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 14:30:00 | 1176.65 | 1172.14 | 1167.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 1182.15 | 1186.96 | 1181.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:30:00 | 1180.00 | 1186.96 | 1181.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1181.75 | 1185.92 | 1181.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 14:30:00 | 1184.85 | 1185.42 | 1181.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 15:15:00 | 1184.95 | 1185.42 | 1181.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-18 09:15:00 | 1173.50 | 1182.96 | 1181.39 | SL hit (close<static) qty=1.00 sl=1178.20 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 1169.05 | 1180.18 | 1180.27 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 14:15:00 | 1186.65 | 1181.22 | 1180.58 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 1154.85 | 1176.15 | 1178.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 12:15:00 | 1149.70 | 1166.55 | 1173.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1162.55 | 1156.01 | 1165.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 1162.55 | 1156.01 | 1165.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1162.55 | 1156.01 | 1165.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 1164.00 | 1156.01 | 1165.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1156.35 | 1156.08 | 1164.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 1163.30 | 1156.08 | 1164.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1139.25 | 1150.91 | 1157.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 1132.25 | 1148.41 | 1155.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 1171.70 | 1153.59 | 1155.23 | SL hit (close>static) qty=1.00 sl=1159.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 11:15:00 | 1170.55 | 1159.13 | 1157.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 1179.80 | 1164.51 | 1160.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 1230.50 | 1230.65 | 1218.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 12:00:00 | 1230.50 | 1230.65 | 1218.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 1218.50 | 1228.22 | 1218.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:00:00 | 1218.50 | 1228.22 | 1218.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 1218.95 | 1226.37 | 1218.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:15:00 | 1218.00 | 1226.37 | 1218.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 1219.75 | 1225.04 | 1218.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 1219.75 | 1225.04 | 1218.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 1222.00 | 1224.44 | 1218.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:15:00 | 1243.05 | 1224.44 | 1218.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 11:15:00 | 1226.50 | 1240.41 | 1241.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 11:15:00 | 1226.50 | 1240.41 | 1241.67 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 1251.65 | 1242.61 | 1241.42 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 1227.35 | 1239.48 | 1240.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 15:15:00 | 1223.75 | 1236.33 | 1238.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 1257.10 | 1240.49 | 1240.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 1257.10 | 1240.49 | 1240.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1257.10 | 1240.49 | 1240.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:00:00 | 1257.10 | 1240.49 | 1240.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 10:15:00 | 1268.40 | 1246.07 | 1243.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 15:15:00 | 1275.00 | 1262.30 | 1253.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 1277.05 | 1280.19 | 1268.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 1277.05 | 1280.19 | 1268.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1262.65 | 1290.64 | 1282.92 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 11:15:00 | 1250.20 | 1278.01 | 1278.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 12:15:00 | 1240.95 | 1270.60 | 1274.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1176.30 | 1176.02 | 1197.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-16 09:30:00 | 1181.35 | 1176.02 | 1197.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 1189.00 | 1182.39 | 1191.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 1190.15 | 1182.39 | 1191.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 09:15:00 | 1192.00 | 1184.31 | 1191.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 13:00:00 | 1183.90 | 1185.38 | 1190.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 15:15:00 | 1181.00 | 1184.62 | 1189.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 09:30:00 | 1180.55 | 1183.44 | 1187.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 09:15:00 | 1210.65 | 1191.53 | 1189.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 1210.65 | 1191.53 | 1189.52 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 1183.95 | 1196.77 | 1198.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 1180.65 | 1192.03 | 1195.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 1132.10 | 1119.74 | 1134.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 1132.10 | 1119.74 | 1134.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1132.10 | 1119.74 | 1134.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 13:45:00 | 1122.40 | 1130.03 | 1132.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-29 14:30:00 | 1125.20 | 1130.51 | 1132.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-30 09:15:00 | 1146.10 | 1133.86 | 1133.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 1146.10 | 1133.86 | 1133.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 10:15:00 | 1154.25 | 1137.94 | 1135.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 14:15:00 | 1122.60 | 1139.80 | 1137.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 14:15:00 | 1122.60 | 1139.80 | 1137.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 1122.60 | 1139.80 | 1137.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 1122.60 | 1139.80 | 1137.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 15:15:00 | 1116.00 | 1135.04 | 1135.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 13:15:00 | 1113.05 | 1122.63 | 1128.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 11:15:00 | 1118.40 | 1116.10 | 1119.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-04 12:00:00 | 1118.40 | 1116.10 | 1119.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 1122.05 | 1117.29 | 1119.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:00:00 | 1122.05 | 1117.29 | 1119.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 1125.30 | 1118.89 | 1120.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 14:00:00 | 1125.30 | 1118.89 | 1120.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 15:15:00 | 1127.55 | 1122.14 | 1121.63 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 14:15:00 | 1115.60 | 1121.74 | 1122.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 1109.00 | 1118.25 | 1120.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 11:15:00 | 1107.30 | 1105.61 | 1110.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 11:30:00 | 1107.60 | 1105.61 | 1110.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 1114.45 | 1107.33 | 1109.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 14:00:00 | 1108.40 | 1109.75 | 1110.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 14:15:00 | 1115.35 | 1110.87 | 1110.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 1115.35 | 1110.87 | 1110.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 1118.65 | 1112.62 | 1111.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 1116.90 | 1117.93 | 1114.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 1116.90 | 1117.93 | 1114.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1112.30 | 1116.81 | 1114.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 1112.30 | 1116.81 | 1114.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 1115.00 | 1116.45 | 1114.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 1126.80 | 1116.45 | 1114.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 12:15:00 | 1115.90 | 1118.69 | 1117.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 13:45:00 | 1118.55 | 1117.88 | 1117.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-16 15:15:00 | 1116.00 | 1117.49 | 1117.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 15:15:00 | 1116.00 | 1117.49 | 1117.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 09:15:00 | 1110.75 | 1116.14 | 1117.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-17 13:15:00 | 1114.40 | 1114.22 | 1115.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 13:15:00 | 1114.40 | 1114.22 | 1115.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 13:15:00 | 1114.40 | 1114.22 | 1115.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 13:30:00 | 1115.55 | 1114.22 | 1115.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 1111.60 | 1113.70 | 1115.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 15:15:00 | 1108.65 | 1113.70 | 1115.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-19 10:15:00 | 1053.22 | 1077.69 | 1092.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-20 11:15:00 | 1056.00 | 1055.97 | 1070.32 | SL hit (close>ema200) qty=0.50 sl=1055.97 alert=retest2 |

### Cycle 31 — BUY (started 2024-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 15:15:00 | 1064.95 | 1056.42 | 1055.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 1069.00 | 1062.51 | 1059.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 09:15:00 | 1052.15 | 1061.62 | 1059.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 1052.15 | 1061.62 | 1059.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 1052.15 | 1061.62 | 1059.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 11:30:00 | 1059.85 | 1060.08 | 1059.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 12:15:00 | 1059.55 | 1060.08 | 1059.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 14:00:00 | 1065.60 | 1071.19 | 1070.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 15:15:00 | 1066.00 | 1070.02 | 1070.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 15:15:00 | 1066.00 | 1070.02 | 1070.14 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 1074.30 | 1070.34 | 1070.23 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 12:15:00 | 1062.95 | 1069.32 | 1069.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 13:15:00 | 1058.45 | 1067.15 | 1068.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 1054.90 | 1052.98 | 1058.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:30:00 | 1053.00 | 1052.98 | 1058.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1055.00 | 1053.77 | 1056.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1055.55 | 1053.77 | 1056.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1055.60 | 1054.14 | 1056.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 1058.10 | 1054.14 | 1056.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1058.40 | 1054.99 | 1056.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 1060.35 | 1054.99 | 1056.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 1060.10 | 1056.01 | 1056.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:30:00 | 1058.00 | 1056.01 | 1056.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 1061.00 | 1057.01 | 1057.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:45:00 | 1061.55 | 1057.01 | 1057.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 1068.70 | 1059.35 | 1058.38 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 1049.30 | 1057.66 | 1058.43 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 11:15:00 | 1062.45 | 1058.60 | 1058.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 12:15:00 | 1069.80 | 1060.84 | 1059.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 1060.00 | 1062.49 | 1060.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 1060.00 | 1062.49 | 1060.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 1060.00 | 1062.49 | 1060.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 1060.00 | 1062.49 | 1060.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 1057.95 | 1061.58 | 1060.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:45:00 | 1057.10 | 1061.58 | 1060.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 11:15:00 | 1059.50 | 1061.17 | 1060.59 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 1056.00 | 1060.13 | 1060.17 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 13:15:00 | 1062.80 | 1060.67 | 1060.41 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 11:15:00 | 1058.65 | 1060.27 | 1060.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 13:15:00 | 1057.50 | 1059.73 | 1060.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 14:15:00 | 1020.40 | 1019.64 | 1029.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 15:00:00 | 1020.40 | 1019.64 | 1029.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1027.25 | 1021.46 | 1028.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:45:00 | 1013.60 | 1018.22 | 1022.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 14:00:00 | 1009.30 | 1002.60 | 1005.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 14:15:00 | 1006.20 | 999.70 | 999.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2024-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 14:15:00 | 1006.20 | 999.70 | 999.16 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 09:15:00 | 989.10 | 998.85 | 999.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 13:15:00 | 983.80 | 992.65 | 995.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 09:15:00 | 990.80 | 990.32 | 993.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 990.80 | 990.32 | 993.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 990.80 | 990.32 | 993.89 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 14:15:00 | 1001.55 | 995.12 | 995.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 1010.75 | 999.65 | 997.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 11:15:00 | 1002.45 | 1002.71 | 999.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 11:45:00 | 1002.50 | 1002.71 | 999.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 1000.65 | 1002.30 | 999.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:45:00 | 997.60 | 1002.30 | 999.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 13:15:00 | 1002.00 | 1002.24 | 999.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 13:30:00 | 999.85 | 1002.24 | 999.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-11-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-05 09:15:00 | 978.05 | 997.64 | 998.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 10:15:00 | 977.55 | 993.62 | 996.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 987.20 | 986.43 | 991.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 15:00:00 | 987.20 | 986.43 | 991.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 997.25 | 988.84 | 991.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 993.95 | 988.84 | 991.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 993.40 | 989.76 | 991.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 11:15:00 | 991.90 | 989.76 | 991.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 12:15:00 | 999.60 | 992.72 | 992.95 | SL hit (close>static) qty=1.00 sl=998.45 alert=retest2 |

### Cycle 45 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 997.20 | 993.62 | 993.34 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 979.75 | 991.19 | 992.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 12:15:00 | 974.55 | 984.72 | 988.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 980.25 | 979.43 | 984.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 09:45:00 | 979.35 | 979.43 | 984.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 966.65 | 970.21 | 974.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 14:30:00 | 973.60 | 970.21 | 974.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 980.15 | 971.86 | 974.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:45:00 | 979.05 | 971.86 | 974.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 985.00 | 974.49 | 975.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 11:15:00 | 979.95 | 974.49 | 975.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 11:15:00 | 985.90 | 976.77 | 976.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 11:15:00 | 985.90 | 976.77 | 976.49 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 12:15:00 | 942.65 | 969.95 | 973.41 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 957.25 | 949.38 | 948.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 14:15:00 | 964.80 | 955.62 | 952.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 958.85 | 959.46 | 955.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 958.85 | 959.46 | 955.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 12:15:00 | 956.70 | 958.91 | 955.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 12:45:00 | 954.60 | 958.91 | 955.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 958.85 | 958.89 | 955.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:30:00 | 956.55 | 958.89 | 955.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 953.45 | 957.81 | 955.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 953.45 | 957.81 | 955.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 950.85 | 956.41 | 955.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 953.50 | 956.41 | 955.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 952.65 | 955.64 | 955.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:00:00 | 952.65 | 955.64 | 955.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 11:15:00 | 950.00 | 954.51 | 954.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 948.45 | 951.69 | 953.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 15:15:00 | 951.85 | 950.16 | 951.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 15:15:00 | 951.85 | 950.16 | 951.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 951.85 | 950.16 | 951.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:15:00 | 960.00 | 950.16 | 951.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 960.30 | 952.19 | 952.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 960.30 | 952.19 | 952.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 968.80 | 955.51 | 953.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 14:15:00 | 976.10 | 968.80 | 963.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 09:15:00 | 975.30 | 979.04 | 973.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 09:15:00 | 975.30 | 979.04 | 973.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 975.30 | 979.04 | 973.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:00:00 | 975.30 | 979.04 | 973.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 969.05 | 977.04 | 973.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 10:30:00 | 970.45 | 977.04 | 973.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 973.20 | 976.27 | 973.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 976.55 | 972.71 | 972.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 10:15:00 | 966.00 | 971.03 | 971.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 966.00 | 971.03 | 971.65 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 13:15:00 | 976.70 | 972.82 | 972.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 14:15:00 | 978.70 | 974.00 | 972.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 13:15:00 | 985.90 | 989.72 | 984.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-09 14:00:00 | 985.90 | 989.72 | 984.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 14:15:00 | 981.55 | 988.09 | 984.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 15:00:00 | 981.55 | 988.09 | 984.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 985.00 | 987.47 | 984.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 980.70 | 987.47 | 984.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 09:15:00 | 978.55 | 985.69 | 983.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 10:00:00 | 978.55 | 985.69 | 983.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 10:15:00 | 982.60 | 985.07 | 983.83 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 971.80 | 982.09 | 982.67 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 10:15:00 | 991.65 | 981.90 | 981.87 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 973.05 | 981.39 | 982.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 962.85 | 975.98 | 979.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 974.55 | 972.94 | 976.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 12:15:00 | 974.55 | 972.94 | 976.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 974.55 | 972.94 | 976.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 974.55 | 972.94 | 976.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 976.00 | 973.55 | 976.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 977.75 | 973.55 | 976.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 977.60 | 974.36 | 976.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 977.60 | 974.36 | 976.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 977.00 | 974.89 | 976.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 981.15 | 974.89 | 976.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 981.95 | 976.30 | 977.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:45:00 | 983.55 | 976.30 | 977.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 979.60 | 976.96 | 977.55 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 988.15 | 979.98 | 978.87 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2024-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 12:15:00 | 974.00 | 979.47 | 979.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 970.00 | 977.58 | 978.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 09:15:00 | 977.90 | 974.98 | 977.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 977.90 | 974.98 | 977.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 977.90 | 974.98 | 977.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:45:00 | 979.85 | 974.98 | 977.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 980.65 | 976.11 | 977.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:30:00 | 980.05 | 976.11 | 977.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 977.05 | 976.30 | 977.39 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2024-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 13:15:00 | 981.95 | 978.29 | 978.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 11:15:00 | 984.05 | 980.33 | 979.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 983.00 | 984.08 | 981.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 983.00 | 984.08 | 981.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 983.00 | 984.08 | 981.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 983.00 | 984.08 | 981.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 989.60 | 985.19 | 982.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 11:45:00 | 990.25 | 986.15 | 983.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 13:15:00 | 978.65 | 984.83 | 983.14 | SL hit (close<static) qty=1.00 sl=981.40 alert=retest2 |

### Cycle 60 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 971.55 | 980.61 | 981.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 970.25 | 978.53 | 980.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 970.95 | 962.26 | 965.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 970.95 | 962.26 | 965.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 970.95 | 962.26 | 965.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:45:00 | 973.40 | 962.26 | 965.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 972.75 | 964.36 | 966.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 972.75 | 964.36 | 966.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 972.95 | 966.08 | 966.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 975.35 | 966.08 | 966.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2024-12-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 12:15:00 | 976.25 | 968.11 | 967.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 13:15:00 | 982.55 | 971.00 | 969.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 968.90 | 971.24 | 969.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 968.90 | 971.24 | 969.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 968.90 | 971.24 | 969.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:30:00 | 970.00 | 971.24 | 969.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 966.75 | 970.34 | 969.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 966.75 | 970.34 | 969.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 963.95 | 969.06 | 969.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 12:00:00 | 963.95 | 969.06 | 969.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 12:15:00 | 962.75 | 967.80 | 968.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 13:15:00 | 957.70 | 965.78 | 967.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 971.35 | 964.90 | 966.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 10:15:00 | 971.35 | 964.90 | 966.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 971.35 | 964.90 | 966.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:00:00 | 971.35 | 964.90 | 966.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 970.65 | 966.05 | 966.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 11:30:00 | 972.90 | 966.05 | 966.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 12:15:00 | 973.80 | 967.60 | 967.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 13:15:00 | 975.95 | 969.27 | 968.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 15:15:00 | 972.20 | 972.54 | 970.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 09:15:00 | 971.00 | 972.54 | 970.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 968.20 | 971.67 | 970.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:45:00 | 969.05 | 971.67 | 970.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 10:15:00 | 970.45 | 971.43 | 970.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 11:45:00 | 977.45 | 972.50 | 971.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 09:15:00 | 958.60 | 974.51 | 975.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 958.60 | 974.51 | 975.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 956.40 | 970.89 | 974.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 989.40 | 969.60 | 971.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 989.40 | 969.60 | 971.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 989.40 | 969.60 | 971.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 989.40 | 969.60 | 971.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 1009.60 | 977.60 | 974.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 09:15:00 | 1012.95 | 990.65 | 986.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 11:15:00 | 1005.05 | 1009.30 | 1002.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-10 12:00:00 | 1005.05 | 1009.30 | 1002.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 14:15:00 | 1004.00 | 1007.39 | 1003.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 14:30:00 | 1005.05 | 1007.39 | 1003.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 15:15:00 | 1004.00 | 1006.71 | 1003.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:15:00 | 988.50 | 1006.71 | 1003.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 985.65 | 1002.50 | 1001.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:00:00 | 985.65 | 1002.50 | 1001.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 10:15:00 | 976.85 | 997.37 | 999.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 13:15:00 | 970.50 | 987.87 | 994.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 988.40 | 985.10 | 990.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 988.40 | 985.10 | 990.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 988.50 | 985.78 | 990.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:45:00 | 989.20 | 985.78 | 990.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 12:15:00 | 996.20 | 987.86 | 990.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 13:00:00 | 996.20 | 987.86 | 990.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 13:15:00 | 995.40 | 989.37 | 991.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:15:00 | 995.90 | 989.37 | 991.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 15:15:00 | 1001.50 | 993.65 | 992.99 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 985.15 | 991.95 | 992.27 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 10:15:00 | 991.05 | 988.70 | 988.69 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 985.85 | 990.65 | 991.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 976.00 | 987.72 | 989.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 994.60 | 981.51 | 984.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 994.60 | 981.51 | 984.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 994.60 | 981.51 | 984.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 994.60 | 981.51 | 984.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 988.60 | 982.93 | 984.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 990.10 | 982.93 | 984.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 989.45 | 985.21 | 985.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:45:00 | 988.50 | 985.21 | 985.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 990.05 | 986.18 | 985.83 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 962.90 | 982.52 | 984.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 941.50 | 962.08 | 972.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 13:15:00 | 918.85 | 918.36 | 936.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 14:00:00 | 918.85 | 918.36 | 936.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 931.55 | 919.23 | 931.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 931.55 | 919.23 | 931.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 936.55 | 922.69 | 932.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 936.55 | 922.69 | 932.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 930.35 | 924.22 | 932.15 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 947.95 | 936.35 | 935.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 14:15:00 | 955.05 | 944.86 | 940.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 957.15 | 964.10 | 957.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 957.15 | 964.10 | 957.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 957.15 | 964.10 | 957.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 957.15 | 964.10 | 957.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 967.05 | 964.69 | 958.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 970.90 | 964.69 | 958.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:45:00 | 967.90 | 968.19 | 962.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 11:15:00 | 968.90 | 964.47 | 963.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 12:00:00 | 970.60 | 971.17 | 967.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 949.40 | 966.82 | 966.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-05 12:15:00 | 949.40 | 966.82 | 966.29 | SL hit (close<static) qty=1.00 sl=953.35 alert=retest2 |

### Cycle 74 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 975.00 | 989.56 | 989.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 963.65 | 984.37 | 987.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 09:15:00 | 950.55 | 940.13 | 948.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 09:15:00 | 950.55 | 940.13 | 948.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 950.55 | 940.13 | 948.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 950.55 | 940.13 | 948.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 951.00 | 942.30 | 949.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:00:00 | 945.05 | 945.06 | 948.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 14:30:00 | 944.65 | 944.59 | 948.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 15:15:00 | 897.80 | 913.44 | 927.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-14 15:15:00 | 897.42 | 913.44 | 927.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 09:15:00 | 915.00 | 913.75 | 926.78 | SL hit (close>ema200) qty=0.50 sl=913.75 alert=retest2 |

### Cycle 75 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 881.65 | 875.63 | 874.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 887.30 | 879.55 | 877.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 901.85 | 905.28 | 896.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 901.85 | 905.28 | 896.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 898.50 | 903.03 | 897.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 13:45:00 | 897.60 | 903.03 | 897.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 908.30 | 903.81 | 899.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 10:30:00 | 910.00 | 903.68 | 899.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 890.65 | 900.94 | 899.49 | SL hit (close<static) qty=1.00 sl=894.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 876.75 | 895.66 | 897.65 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 892.90 | 891.04 | 890.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 904.80 | 894.59 | 892.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 14:15:00 | 918.90 | 920.79 | 915.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 14:45:00 | 920.55 | 920.79 | 915.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 926.00 | 927.15 | 922.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 09:15:00 | 907.00 | 927.15 | 922.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 914.25 | 924.57 | 921.95 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 908.95 | 919.12 | 919.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 15:15:00 | 905.00 | 909.62 | 912.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 10:15:00 | 899.30 | 894.94 | 900.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-28 11:00:00 | 899.30 | 894.94 | 900.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 892.95 | 890.25 | 895.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 899.15 | 890.25 | 895.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 890.45 | 890.29 | 895.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 885.15 | 890.29 | 895.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 912.70 | 890.84 | 889.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 912.70 | 890.84 | 889.09 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 835.40 | 888.13 | 890.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 13:15:00 | 832.90 | 851.32 | 865.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 847.25 | 846.35 | 858.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 847.25 | 846.35 | 858.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 857.60 | 848.60 | 858.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 857.60 | 848.60 | 858.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 865.00 | 851.88 | 858.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:00:00 | 865.00 | 851.88 | 858.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 859.60 | 853.42 | 858.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:30:00 | 861.65 | 853.42 | 858.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 859.10 | 854.56 | 858.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 837.70 | 855.65 | 858.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 10:15:00 | 862.35 | 851.60 | 853.13 | SL hit (close>static) qty=1.00 sl=861.85 alert=retest2 |

### Cycle 81 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 868.30 | 854.94 | 854.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 14:15:00 | 869.70 | 861.51 | 857.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 11:15:00 | 870.05 | 876.02 | 870.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-16 11:15:00 | 870.05 | 876.02 | 870.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 11:15:00 | 870.05 | 876.02 | 870.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 11:30:00 | 870.35 | 876.02 | 870.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 12:15:00 | 839.80 | 868.77 | 867.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:00:00 | 839.80 | 868.77 | 867.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 13:15:00 | 827.00 | 860.42 | 863.87 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 11:15:00 | 847.85 | 843.74 | 843.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-22 12:15:00 | 849.80 | 844.95 | 844.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 865.30 | 874.10 | 867.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 865.30 | 874.10 | 867.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 865.30 | 874.10 | 867.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 865.30 | 874.10 | 867.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 863.80 | 872.04 | 867.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:30:00 | 862.40 | 872.04 | 867.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 13:15:00 | 860.10 | 867.90 | 866.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:00:00 | 860.10 | 867.90 | 866.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 859.95 | 866.31 | 865.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 15:00:00 | 859.95 | 866.31 | 865.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 859.80 | 865.01 | 865.14 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 877.00 | 867.41 | 866.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 882.05 | 871.68 | 868.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 12:15:00 | 889.35 | 890.94 | 885.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 13:00:00 | 889.35 | 890.94 | 885.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 15:15:00 | 885.40 | 889.25 | 885.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-02 09:15:00 | 885.15 | 889.25 | 885.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 890.70 | 889.54 | 886.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 14:30:00 | 893.70 | 889.62 | 887.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 09:15:00 | 880.00 | 888.19 | 887.27 | SL hit (close<static) qty=1.00 sl=882.75 alert=retest2 |

### Cycle 86 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 880.95 | 885.75 | 886.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 876.00 | 881.16 | 883.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 884.45 | 881.82 | 883.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 884.45 | 881.82 | 883.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 884.45 | 881.82 | 883.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 10:30:00 | 881.40 | 881.82 | 883.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 886.30 | 882.72 | 883.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 886.30 | 882.72 | 883.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 885.10 | 883.19 | 884.00 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 887.95 | 884.85 | 884.62 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 882.35 | 884.46 | 884.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 877.40 | 883.05 | 883.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 877.35 | 876.76 | 879.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 14:00:00 | 877.35 | 876.76 | 879.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 877.10 | 876.83 | 879.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 14:45:00 | 878.50 | 876.83 | 879.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 880.90 | 877.64 | 879.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 869.40 | 877.64 | 879.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 880.50 | 878.22 | 879.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:15:00 | 885.70 | 878.22 | 879.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 884.10 | 879.39 | 879.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:30:00 | 886.95 | 879.39 | 879.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 885.00 | 881.09 | 880.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 898.00 | 886.06 | 883.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 892.10 | 892.33 | 887.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 892.10 | 892.33 | 887.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 900.35 | 896.51 | 890.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 13:45:00 | 905.40 | 900.10 | 894.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 14:00:00 | 906.65 | 902.82 | 898.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 12:15:00 | 905.25 | 902.97 | 900.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 13:15:00 | 905.40 | 903.33 | 900.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 13:15:00 | 902.25 | 903.11 | 900.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 13:45:00 | 901.60 | 903.11 | 900.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 903.00 | 903.22 | 901.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:15:00 | 906.00 | 903.22 | 901.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 12:15:00 | 909.15 | 908.35 | 906.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 899.05 | 906.49 | 906.07 | SL hit (close<static) qty=1.00 sl=899.95 alert=retest2 |

### Cycle 90 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 882.40 | 901.67 | 903.92 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 12:15:00 | 903.50 | 898.22 | 898.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 13:15:00 | 909.60 | 900.50 | 899.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 10:15:00 | 900.85 | 901.66 | 900.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 10:15:00 | 900.85 | 901.66 | 900.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 900.85 | 901.66 | 900.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:15:00 | 899.25 | 901.66 | 900.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 903.75 | 902.08 | 900.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 12:15:00 | 907.95 | 902.08 | 900.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 10:15:00 | 965.30 | 974.49 | 975.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 965.30 | 974.49 | 975.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 10:15:00 | 961.20 | 969.00 | 971.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 12:15:00 | 960.00 | 958.98 | 963.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 13:00:00 | 960.00 | 958.98 | 963.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 961.30 | 959.45 | 963.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:00:00 | 956.50 | 958.86 | 962.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 10:00:00 | 959.35 | 954.49 | 954.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 10:15:00 | 959.50 | 955.50 | 955.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 959.50 | 955.50 | 955.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 11:15:00 | 967.80 | 957.96 | 956.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 09:15:00 | 982.90 | 983.06 | 978.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:00:00 | 982.90 | 983.06 | 978.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 979.60 | 982.37 | 978.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 13:00:00 | 983.65 | 982.36 | 979.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 13:15:00 | 995.70 | 997.74 | 997.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — SELL (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 13:15:00 | 995.70 | 997.74 | 997.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 981.65 | 994.35 | 996.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 14:15:00 | 978.25 | 977.05 | 982.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 15:00:00 | 978.25 | 977.05 | 982.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 977.65 | 973.97 | 977.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 981.00 | 973.97 | 977.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 974.20 | 974.01 | 976.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 11:45:00 | 971.25 | 974.26 | 976.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 15:15:00 | 979.95 | 976.46 | 976.99 | SL hit (close>static) qty=1.00 sl=978.75 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 981.00 | 977.79 | 977.53 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 14:15:00 | 966.10 | 975.60 | 976.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 09:15:00 | 965.85 | 970.22 | 972.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 13:15:00 | 969.85 | 968.79 | 971.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-16 13:30:00 | 969.35 | 968.79 | 971.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 965.00 | 968.11 | 970.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 980.10 | 968.11 | 970.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 977.95 | 970.08 | 971.14 | EMA400 retest candle locked (from downside) |

### Cycle 97 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 978.95 | 971.85 | 971.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 12:15:00 | 981.15 | 974.71 | 973.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 974.75 | 977.43 | 975.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 974.75 | 977.43 | 975.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 974.75 | 977.43 | 975.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 974.75 | 977.43 | 975.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 972.85 | 976.51 | 975.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:45:00 | 972.35 | 976.51 | 975.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 977.05 | 976.99 | 975.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:30:00 | 976.95 | 976.99 | 975.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 974.60 | 976.51 | 975.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:00:00 | 974.60 | 976.51 | 975.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 977.80 | 976.77 | 975.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 972.30 | 976.77 | 975.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 976.15 | 976.65 | 975.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 973.15 | 976.65 | 975.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 976.00 | 976.52 | 975.83 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 969.80 | 975.08 | 975.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 955.00 | 970.17 | 973.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 962.15 | 960.81 | 964.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 14:45:00 | 962.65 | 960.81 | 964.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 961.75 | 961.35 | 963.76 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 967.70 | 964.80 | 964.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 10:15:00 | 973.00 | 968.92 | 966.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 15:15:00 | 973.50 | 973.87 | 970.53 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-28 09:15:00 | 980.80 | 973.87 | 970.53 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 978.75 | 979.16 | 974.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 978.75 | 979.16 | 974.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 966.50 | 988.39 | 987.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-31 09:15:00 | 966.50 | 988.39 | 987.24 | SL hit (close<ema400) qty=1.00 sl=987.24 alert=retest1 |

### Cycle 100 — SELL (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 10:15:00 | 974.60 | 985.64 | 986.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 964.00 | 975.20 | 980.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 958.65 | 955.74 | 964.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:00:00 | 958.65 | 955.74 | 964.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 950.40 | 956.38 | 961.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 940.10 | 958.45 | 960.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 955.20 | 942.87 | 942.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 955.20 | 942.87 | 942.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 956.25 | 951.69 | 947.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 12:15:00 | 951.60 | 953.47 | 949.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 12:15:00 | 951.60 | 953.47 | 949.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 951.60 | 953.47 | 949.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 951.60 | 953.47 | 949.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 954.55 | 953.69 | 950.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 11:00:00 | 966.30 | 957.09 | 952.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-20 15:15:00 | 981.75 | 984.13 | 984.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 15:15:00 | 981.75 | 984.13 | 984.34 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 991.65 | 985.42 | 984.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-25 09:15:00 | 1000.70 | 991.38 | 989.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 989.00 | 1006.97 | 1000.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 989.00 | 1006.97 | 1000.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 989.00 | 1006.97 | 1000.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:45:00 | 989.25 | 1006.97 | 1000.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 992.00 | 1003.98 | 999.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 992.00 | 1003.98 | 999.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-08-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 13:15:00 | 986.55 | 996.90 | 997.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 968.85 | 988.14 | 992.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 985.60 | 979.77 | 984.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 985.60 | 979.77 | 984.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 985.60 | 979.77 | 984.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 985.60 | 979.77 | 984.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 987.30 | 981.28 | 984.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:45:00 | 987.30 | 981.28 | 984.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 981.00 | 981.22 | 984.60 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 993.10 | 986.52 | 985.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 996.00 | 988.42 | 986.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 15:15:00 | 992.10 | 995.45 | 992.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 15:15:00 | 992.10 | 995.45 | 992.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 992.10 | 995.45 | 992.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1006.80 | 995.45 | 992.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 13:15:00 | 1036.10 | 1039.94 | 1040.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 1036.10 | 1039.94 | 1040.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 12:15:00 | 1034.40 | 1037.35 | 1038.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 11:15:00 | 1036.00 | 1035.07 | 1036.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 11:30:00 | 1036.00 | 1035.07 | 1036.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 1037.95 | 1035.65 | 1036.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 13:00:00 | 1037.95 | 1035.65 | 1036.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 1035.10 | 1035.54 | 1036.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 09:45:00 | 1031.90 | 1034.63 | 1035.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:45:00 | 1031.70 | 1033.95 | 1035.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:30:00 | 1031.55 | 1033.46 | 1035.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 1045.05 | 1035.23 | 1035.61 | SL hit (close>static) qty=1.00 sl=1038.50 alert=retest2 |

### Cycle 107 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 1046.60 | 1037.50 | 1036.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 15:15:00 | 1049.95 | 1039.99 | 1037.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 11:15:00 | 1039.90 | 1042.16 | 1039.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 12:00:00 | 1039.90 | 1042.16 | 1039.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1039.95 | 1041.72 | 1039.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:30:00 | 1039.50 | 1041.72 | 1039.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1038.80 | 1041.14 | 1039.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 1038.80 | 1041.14 | 1039.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1035.60 | 1040.03 | 1039.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:30:00 | 1041.70 | 1039.56 | 1039.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 1041.20 | 1041.24 | 1040.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 15:15:00 | 1033.85 | 1038.87 | 1039.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-09-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 15:15:00 | 1033.85 | 1038.87 | 1039.18 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-09-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 13:15:00 | 1043.50 | 1039.79 | 1039.33 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 15:15:00 | 1034.60 | 1038.93 | 1039.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 1032.35 | 1037.62 | 1038.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 10:15:00 | 990.65 | 989.98 | 1004.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 11:00:00 | 990.65 | 989.98 | 1004.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 991.85 | 987.65 | 992.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 993.55 | 987.65 | 992.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 989.65 | 988.05 | 992.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 988.35 | 988.05 | 992.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 990.95 | 989.07 | 991.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 990.95 | 989.07 | 991.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 994.00 | 990.06 | 991.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 994.10 | 990.06 | 991.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 994.05 | 990.85 | 992.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:00:00 | 987.80 | 990.16 | 991.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 15:15:00 | 988.55 | 988.53 | 990.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 15:15:00 | 993.00 | 989.72 | 989.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 993.00 | 989.72 | 989.69 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 987.00 | 989.60 | 989.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 983.25 | 988.33 | 989.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 12:15:00 | 990.45 | 988.76 | 989.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 12:15:00 | 990.45 | 988.76 | 989.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 990.45 | 988.76 | 989.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:00:00 | 990.45 | 988.76 | 989.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 989.80 | 988.97 | 989.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:45:00 | 988.65 | 988.97 | 989.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 987.30 | 988.63 | 989.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 985.65 | 988.34 | 988.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:00:00 | 986.00 | 987.87 | 988.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 13:15:00 | 990.20 | 988.65 | 988.78 | SL hit (close>static) qty=1.00 sl=989.95 alert=retest2 |

### Cycle 113 — BUY (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 09:15:00 | 1001.20 | 989.58 | 989.04 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 984.55 | 993.04 | 993.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 977.05 | 989.84 | 992.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 989.30 | 984.30 | 987.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 989.30 | 984.30 | 987.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 989.30 | 984.30 | 987.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 989.30 | 984.30 | 987.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 989.55 | 985.35 | 987.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 989.55 | 985.35 | 987.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 987.55 | 985.79 | 987.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 989.30 | 985.79 | 987.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 981.15 | 984.86 | 987.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:15:00 | 980.00 | 984.09 | 986.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:15:00 | 979.75 | 983.25 | 984.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 987.20 | 984.87 | 984.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 14:15:00 | 987.20 | 984.87 | 984.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 995.00 | 986.96 | 985.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 1001.10 | 1004.78 | 1000.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 09:15:00 | 1001.10 | 1004.78 | 1000.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1001.10 | 1004.78 | 1000.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:30:00 | 1009.00 | 1002.86 | 1000.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 11:00:00 | 1010.30 | 1004.34 | 1001.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 12:30:00 | 1010.00 | 1006.64 | 1003.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 14:15:00 | 1001.25 | 1004.08 | 1004.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 1001.25 | 1004.08 | 1004.41 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 1007.35 | 1004.66 | 1004.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1013.35 | 1006.40 | 1005.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 11:15:00 | 1004.85 | 1006.09 | 1005.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 11:15:00 | 1004.85 | 1006.09 | 1005.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1004.85 | 1006.09 | 1005.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 1004.85 | 1006.09 | 1005.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1005.10 | 1005.89 | 1005.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:30:00 | 1005.10 | 1005.89 | 1005.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 1001.05 | 1004.92 | 1004.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 987.30 | 1000.49 | 1002.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 981.00 | 978.65 | 983.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 15:00:00 | 981.00 | 978.65 | 983.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 983.05 | 979.53 | 983.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 981.95 | 979.53 | 983.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 981.10 | 979.84 | 983.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:30:00 | 977.25 | 978.94 | 982.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 12:00:00 | 976.95 | 978.94 | 982.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 928.39 | 946.05 | 959.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 928.10 | 946.05 | 959.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 15:15:00 | 945.00 | 943.82 | 953.09 | SL hit (close>ema200) qty=0.50 sl=943.82 alert=retest2 |

### Cycle 119 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 951.45 | 948.09 | 948.00 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-11-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 10:15:00 | 945.85 | 948.05 | 948.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 13:15:00 | 943.55 | 946.42 | 947.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 09:15:00 | 947.45 | 946.14 | 946.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 947.45 | 946.14 | 946.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 947.45 | 946.14 | 946.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:45:00 | 949.40 | 946.14 | 946.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 947.70 | 946.45 | 947.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:15:00 | 948.85 | 946.45 | 947.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 946.05 | 946.37 | 946.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:30:00 | 947.30 | 946.37 | 946.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 943.75 | 945.85 | 946.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 939.95 | 945.04 | 946.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 942.25 | 944.94 | 945.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 11:15:00 | 932.25 | 929.33 | 929.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 932.25 | 929.33 | 929.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 937.00 | 930.84 | 929.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 11:15:00 | 934.10 | 938.88 | 935.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 11:15:00 | 934.10 | 938.88 | 935.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 934.10 | 938.88 | 935.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:30:00 | 935.20 | 938.88 | 935.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 935.35 | 938.18 | 935.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 13:45:00 | 937.25 | 937.91 | 935.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 14:30:00 | 937.35 | 937.88 | 936.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 937.75 | 937.88 | 936.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 13:15:00 | 937.00 | 937.62 | 937.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 13:15:00 | 937.00 | 937.62 | 937.69 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 10:15:00 | 940.05 | 938.10 | 937.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 11:15:00 | 942.45 | 938.97 | 938.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 935.10 | 939.70 | 939.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 935.10 | 939.70 | 939.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 935.10 | 939.70 | 939.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 935.10 | 939.70 | 939.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 933.75 | 938.51 | 938.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 930.20 | 935.42 | 936.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 09:15:00 | 934.60 | 933.72 | 935.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 934.60 | 933.72 | 935.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 934.60 | 933.72 | 935.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:45:00 | 936.40 | 933.72 | 935.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 932.95 | 933.57 | 934.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:30:00 | 935.25 | 933.57 | 934.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 934.55 | 924.73 | 926.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:00:00 | 934.55 | 924.73 | 926.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 929.35 | 925.66 | 927.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 11:15:00 | 927.55 | 925.66 | 927.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:45:00 | 928.50 | 926.76 | 927.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 12:15:00 | 927.65 | 926.14 | 925.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 927.65 | 926.14 | 925.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 930.60 | 927.03 | 926.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 15:15:00 | 927.45 | 927.57 | 926.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:15:00 | 918.55 | 927.57 | 926.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 126 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 919.90 | 926.03 | 926.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 14:15:00 | 915.50 | 919.80 | 921.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 11:15:00 | 919.30 | 918.31 | 920.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 12:00:00 | 919.30 | 918.31 | 920.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 917.75 | 918.20 | 920.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 14:15:00 | 916.95 | 918.07 | 919.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 915.10 | 918.20 | 919.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 916.45 | 915.41 | 916.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 09:15:00 | 921.40 | 917.31 | 917.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 921.40 | 917.31 | 917.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 15:15:00 | 929.45 | 925.11 | 922.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 09:15:00 | 920.20 | 924.13 | 922.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 09:15:00 | 920.20 | 924.13 | 922.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 920.20 | 924.13 | 922.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 920.20 | 924.13 | 922.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 920.50 | 923.40 | 921.95 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 917.65 | 920.52 | 920.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 912.40 | 918.33 | 919.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 908.40 | 903.66 | 906.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 908.40 | 903.66 | 906.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 908.40 | 903.66 | 906.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:45:00 | 910.65 | 903.66 | 906.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 910.90 | 905.11 | 906.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:45:00 | 911.30 | 905.11 | 906.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 912.85 | 908.15 | 907.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 914.60 | 910.31 | 909.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 909.05 | 910.65 | 909.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 909.05 | 910.65 | 909.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 909.05 | 910.65 | 909.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 909.25 | 910.65 | 909.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 911.25 | 910.77 | 909.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 11:45:00 | 913.90 | 911.18 | 909.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 12:15:00 | 913.00 | 911.18 | 909.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 13:30:00 | 913.15 | 911.74 | 910.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:30:00 | 914.65 | 913.24 | 911.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 917.25 | 915.35 | 913.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:45:00 | 915.00 | 915.35 | 913.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 914.50 | 915.52 | 913.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 913.00 | 915.52 | 913.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 921.90 | 916.80 | 914.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:45:00 | 923.25 | 918.72 | 915.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 14:45:00 | 923.50 | 919.41 | 916.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:45:00 | 925.65 | 921.01 | 917.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 13:15:00 | 915.20 | 924.31 | 924.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 915.20 | 924.31 | 924.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 907.85 | 921.02 | 923.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 895.05 | 894.20 | 901.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 897.80 | 894.20 | 901.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 891.65 | 894.14 | 900.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 882.30 | 893.49 | 897.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 884.70 | 891.98 | 896.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:45:00 | 883.25 | 890.14 | 894.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:30:00 | 885.30 | 886.11 | 890.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 882.40 | 873.65 | 878.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 882.40 | 873.65 | 878.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 878.60 | 874.64 | 878.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 880.50 | 874.64 | 878.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 883.10 | 876.33 | 878.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:15:00 | 876.05 | 877.06 | 878.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:30:00 | 876.65 | 876.83 | 878.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 873.40 | 876.68 | 877.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 12:15:00 | 876.50 | 877.65 | 878.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 877.15 | 877.55 | 878.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 13:45:00 | 873.20 | 876.35 | 877.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 886.45 | 878.42 | 878.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 886.45 | 878.42 | 878.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 891.00 | 884.62 | 881.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 13:15:00 | 882.10 | 885.24 | 883.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 13:15:00 | 882.10 | 885.24 | 883.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 882.10 | 885.24 | 883.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:45:00 | 881.15 | 885.24 | 883.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 881.75 | 884.55 | 883.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:15:00 | 886.00 | 884.55 | 883.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 886.00 | 884.84 | 883.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 885.00 | 884.84 | 883.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 891.55 | 886.18 | 884.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-28 14:45:00 | 894.65 | 891.70 | 889.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 10:15:00 | 885.15 | 888.42 | 888.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 885.15 | 888.42 | 888.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 883.55 | 887.07 | 887.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 13:15:00 | 887.20 | 887.09 | 887.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 13:15:00 | 887.20 | 887.09 | 887.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 887.20 | 887.09 | 887.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:45:00 | 886.70 | 887.09 | 887.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 884.50 | 886.58 | 887.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:45:00 | 888.65 | 886.58 | 887.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 884.00 | 886.06 | 887.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 09:15:00 | 880.05 | 886.06 | 887.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 10:15:00 | 889.70 | 886.11 | 886.99 | SL hit (close>static) qty=1.00 sl=889.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 905.80 | 885.84 | 883.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 910.55 | 899.64 | 892.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 902.25 | 903.44 | 898.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 902.25 | 903.44 | 898.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 886.75 | 901.26 | 900.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 886.75 | 901.26 | 900.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 888.45 | 898.70 | 898.97 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 905.10 | 896.88 | 896.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 13:15:00 | 917.95 | 901.10 | 898.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 898.50 | 906.65 | 902.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 09:15:00 | 898.50 | 906.65 | 902.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 898.50 | 906.65 | 902.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 898.50 | 906.65 | 902.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 891.50 | 903.62 | 901.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:00:00 | 891.50 | 903.62 | 901.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 888.80 | 900.65 | 900.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:30:00 | 888.65 | 900.65 | 900.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 12:15:00 | 889.70 | 898.46 | 899.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 09:15:00 | 885.55 | 891.64 | 895.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 892.00 | 891.71 | 894.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-11 11:00:00 | 892.00 | 891.71 | 894.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 895.60 | 892.49 | 895.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:00:00 | 895.60 | 892.49 | 895.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 894.65 | 892.92 | 895.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:15:00 | 896.00 | 892.92 | 895.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 895.95 | 893.53 | 895.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 13:30:00 | 895.60 | 893.53 | 895.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 898.45 | 894.51 | 895.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 898.45 | 894.51 | 895.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 899.00 | 895.41 | 895.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:15:00 | 901.80 | 895.41 | 895.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 09:15:00 | 919.50 | 900.23 | 897.89 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 14:15:00 | 907.15 | 907.63 | 907.67 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 910.25 | 907.27 | 907.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 15:15:00 | 912.00 | 908.22 | 907.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 908.25 | 909.44 | 908.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 908.25 | 909.44 | 908.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 908.25 | 909.44 | 908.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:45:00 | 912.55 | 909.44 | 908.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 909.25 | 909.40 | 908.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 909.25 | 909.40 | 908.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 907.70 | 909.06 | 908.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 907.70 | 909.06 | 908.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 901.75 | 907.60 | 907.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 901.35 | 906.35 | 907.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 903.25 | 902.72 | 904.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 903.25 | 902.72 | 904.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 908.75 | 903.59 | 904.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:45:00 | 903.70 | 903.54 | 904.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:45:00 | 904.00 | 903.13 | 904.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 902.35 | 904.37 | 904.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 10:15:00 | 903.35 | 904.55 | 904.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 905.55 | 904.75 | 904.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 905.55 | 904.75 | 904.66 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 13:15:00 | 903.35 | 904.39 | 904.53 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 15:15:00 | 907.00 | 904.79 | 904.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 907.85 | 905.40 | 904.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 924.00 | 930.20 | 923.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:00:00 | 924.00 | 930.20 | 923.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 926.85 | 929.53 | 923.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 14:45:00 | 929.65 | 925.50 | 923.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 902.05 | 920.39 | 921.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 902.05 | 920.39 | 921.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 891.95 | 903.43 | 910.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 909.10 | 899.65 | 904.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 909.10 | 899.65 | 904.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 909.10 | 899.65 | 904.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 911.40 | 899.65 | 904.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 908.10 | 901.34 | 904.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:30:00 | 904.70 | 902.54 | 905.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 13:15:00 | 910.80 | 905.31 | 906.05 | SL hit (close>static) qty=1.00 sl=910.70 alert=retest2 |

### Cycle 145 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 914.05 | 907.06 | 906.78 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 893.20 | 908.69 | 908.83 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 917.55 | 907.65 | 907.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 930.70 | 919.60 | 914.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 14:15:00 | 920.00 | 923.85 | 918.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 15:00:00 | 920.00 | 923.85 | 918.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 921.25 | 923.33 | 919.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 914.15 | 923.33 | 919.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 911.95 | 921.06 | 918.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 10:15:00 | 920.00 | 921.06 | 918.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:00:00 | 919.25 | 921.23 | 919.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 907.05 | 916.81 | 917.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 907.05 | 916.81 | 917.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 905.20 | 914.49 | 916.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 898.60 | 891.86 | 899.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 898.60 | 891.86 | 899.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 898.00 | 893.08 | 899.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 893.80 | 893.08 | 899.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 903.20 | 894.69 | 897.26 | SL hit (close>static) qty=1.00 sl=902.70 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 13:15:00 | 901.40 | 898.87 | 898.72 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 883.85 | 896.45 | 897.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 880.55 | 888.24 | 892.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 885.90 | 884.59 | 889.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 885.90 | 884.59 | 889.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 885.90 | 884.59 | 889.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 888.10 | 884.59 | 889.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 888.50 | 885.37 | 889.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 888.50 | 885.37 | 889.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 888.90 | 886.08 | 889.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:45:00 | 888.75 | 886.08 | 889.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 12:15:00 | 893.45 | 887.55 | 889.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:00:00 | 893.45 | 887.55 | 889.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 891.35 | 888.31 | 890.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 13:30:00 | 894.25 | 888.31 | 890.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 890.45 | 889.09 | 890.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 09:15:00 | 884.20 | 889.09 | 890.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 09:15:00 | 877.35 | 886.74 | 888.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 12:15:00 | 876.45 | 883.55 | 887.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 12:00:00 | 876.05 | 872.16 | 878.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 901.65 | 884.45 | 882.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 901.65 | 884.45 | 882.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 905.10 | 888.58 | 884.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 12:15:00 | 901.95 | 903.40 | 896.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 13:00:00 | 901.95 | 903.40 | 896.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 896.30 | 901.87 | 897.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 896.30 | 901.87 | 897.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 896.20 | 900.74 | 897.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 886.10 | 900.74 | 897.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 884.35 | 897.46 | 896.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:15:00 | 882.85 | 897.46 | 896.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 882.30 | 894.43 | 894.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 880.90 | 889.51 | 892.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 886.85 | 884.95 | 889.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 886.85 | 884.95 | 889.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 886.85 | 884.95 | 889.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:45:00 | 877.05 | 882.75 | 887.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 12:15:00 | 871.00 | 867.38 | 867.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 12:15:00 | 871.00 | 867.38 | 867.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 15:15:00 | 874.40 | 870.04 | 868.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 910.65 | 910.77 | 902.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 910.65 | 910.77 | 902.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 910.65 | 910.77 | 902.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 915.10 | 910.77 | 902.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-21 10:15:00 | 931.50 | 936.04 | 936.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 931.50 | 936.04 | 936.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 11:15:00 | 928.00 | 934.43 | 935.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 10:15:00 | 930.50 | 930.40 | 932.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 11:00:00 | 930.50 | 930.40 | 932.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 930.65 | 930.45 | 932.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 12:30:00 | 929.85 | 930.33 | 932.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 15:00:00 | 929.60 | 930.24 | 931.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 957.45 | 935.33 | 933.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 957.45 | 935.33 | 933.94 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 930.00 | 937.53 | 937.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 927.10 | 934.22 | 935.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 949.55 | 935.21 | 935.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 949.55 | 935.21 | 935.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 949.55 | 935.21 | 935.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 949.55 | 935.21 | 935.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 949.50 | 938.07 | 937.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 951.90 | 943.70 | 940.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 15:15:00 | 932.50 | 941.98 | 940.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 15:15:00 | 932.50 | 941.98 | 940.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 932.50 | 941.98 | 940.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 919.25 | 941.98 | 940.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 913.15 | 936.22 | 937.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 901.65 | 912.69 | 919.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 900.60 | 899.42 | 907.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:45:00 | 902.00 | 899.42 | 907.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 905.35 | 902.16 | 905.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:00:00 | 905.35 | 902.16 | 905.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 909.45 | 903.62 | 905.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:00:00 | 909.45 | 903.62 | 905.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 907.50 | 904.39 | 905.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:45:00 | 908.80 | 904.39 | 905.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 909.45 | 906.25 | 906.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:45:00 | 910.80 | 906.25 | 906.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 909.00 | 906.80 | 906.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 15:15:00 | 915.05 | 908.45 | 907.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 11:15:00 | 940.75 | 941.83 | 934.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 12:00:00 | 940.75 | 941.83 | 934.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:30:00 | 978.75 | 2024-05-13 12:15:00 | 995.40 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2024-05-22 15:15:00 | 1087.00 | 2024-05-27 09:15:00 | 1066.70 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-05-24 12:45:00 | 1083.45 | 2024-05-27 09:15:00 | 1066.70 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest1 | 2024-05-29 09:15:00 | 1042.25 | 2024-06-03 09:15:00 | 1027.25 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2024-06-07 10:45:00 | 1054.35 | 2024-06-18 12:15:00 | 1077.05 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2024-06-07 13:30:00 | 1055.70 | 2024-06-18 12:15:00 | 1077.05 | STOP_HIT | 1.00 | 2.02% |
| BUY | retest2 | 2024-06-07 14:30:00 | 1056.05 | 2024-06-18 12:15:00 | 1077.05 | STOP_HIT | 1.00 | 1.99% |
| SELL | retest2 | 2024-06-26 13:45:00 | 1072.85 | 2024-06-28 13:15:00 | 1079.95 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-06-27 10:00:00 | 1071.95 | 2024-06-28 13:15:00 | 1079.95 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-07-10 14:30:00 | 1167.45 | 2024-07-18 09:15:00 | 1173.50 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2024-07-11 11:30:00 | 1163.00 | 2024-07-18 09:15:00 | 1173.50 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2024-07-11 13:00:00 | 1162.45 | 2024-07-18 10:15:00 | 1169.05 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2024-07-11 14:15:00 | 1162.35 | 2024-07-18 10:15:00 | 1169.05 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2024-07-12 09:15:00 | 1176.60 | 2024-07-18 10:15:00 | 1169.05 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-07-12 10:00:00 | 1178.55 | 2024-07-18 10:15:00 | 1169.05 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-07-12 13:30:00 | 1176.90 | 2024-07-18 10:15:00 | 1169.05 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2024-07-12 14:30:00 | 1176.65 | 2024-07-18 10:15:00 | 1169.05 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2024-07-16 14:30:00 | 1184.85 | 2024-07-18 10:15:00 | 1169.05 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-07-16 15:15:00 | 1184.95 | 2024-07-18 10:15:00 | 1169.05 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-07-23 12:15:00 | 1132.25 | 2024-07-24 09:15:00 | 1171.70 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2024-07-31 09:15:00 | 1243.05 | 2024-08-05 11:15:00 | 1226.50 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-08-19 13:00:00 | 1183.90 | 2024-08-21 09:15:00 | 1210.65 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-08-19 15:15:00 | 1181.00 | 2024-08-21 09:15:00 | 1210.65 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-08-20 09:30:00 | 1180.55 | 2024-08-21 09:15:00 | 1210.65 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2024-08-29 13:45:00 | 1122.40 | 2024-08-30 09:15:00 | 1146.10 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2024-08-29 14:30:00 | 1125.20 | 2024-08-30 09:15:00 | 1146.10 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2024-09-10 14:00:00 | 1108.40 | 2024-09-10 14:15:00 | 1115.35 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-09-12 09:15:00 | 1126.80 | 2024-09-16 15:15:00 | 1116.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-09-13 12:15:00 | 1115.90 | 2024-09-16 15:15:00 | 1116.00 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2024-09-13 13:45:00 | 1118.55 | 2024-09-16 15:15:00 | 1116.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-09-17 15:15:00 | 1108.65 | 2024-09-19 10:15:00 | 1053.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-17 15:15:00 | 1108.65 | 2024-09-20 11:15:00 | 1056.00 | STOP_HIT | 0.50 | 4.75% |
| BUY | retest2 | 2024-09-27 11:30:00 | 1059.85 | 2024-10-03 15:15:00 | 1066.00 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2024-09-27 12:15:00 | 1059.55 | 2024-10-03 15:15:00 | 1066.00 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2024-10-03 14:00:00 | 1065.60 | 2024-10-03 15:15:00 | 1066.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-10-21 11:45:00 | 1013.60 | 2024-10-28 14:15:00 | 1006.20 | STOP_HIT | 1.00 | 0.73% |
| SELL | retest2 | 2024-10-23 14:00:00 | 1009.30 | 2024-10-28 14:15:00 | 1006.20 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2024-11-06 11:15:00 | 991.90 | 2024-11-06 12:15:00 | 999.60 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-11-12 11:15:00 | 979.95 | 2024-11-12 11:15:00 | 985.90 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-12-05 09:15:00 | 976.55 | 2024-12-05 10:15:00 | 966.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-12-20 11:45:00 | 990.25 | 2024-12-20 13:15:00 | 978.65 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-01-02 11:45:00 | 977.45 | 2025-01-06 09:15:00 | 958.60 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-02-01 14:15:00 | 970.90 | 2025-02-05 12:15:00 | 949.40 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-02-03 10:45:00 | 967.90 | 2025-02-05 12:15:00 | 949.40 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-02-04 11:15:00 | 968.90 | 2025-02-05 12:15:00 | 949.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-02-05 12:00:00 | 970.60 | 2025-02-05 12:15:00 | 949.40 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-02-06 09:15:00 | 1002.15 | 2025-02-10 10:15:00 | 975.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-02-13 14:00:00 | 945.05 | 2025-02-14 15:15:00 | 897.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:30:00 | 944.65 | 2025-02-14 15:15:00 | 897.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 14:00:00 | 945.05 | 2025-02-17 09:15:00 | 915.00 | STOP_HIT | 0.50 | 3.18% |
| SELL | retest2 | 2025-02-13 14:30:00 | 944.65 | 2025-02-17 09:15:00 | 915.00 | STOP_HIT | 0.50 | 3.14% |
| BUY | retest2 | 2025-03-10 10:30:00 | 910.00 | 2025-03-10 14:15:00 | 890.65 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2025-04-01 11:15:00 | 885.15 | 2025-04-03 09:15:00 | 912.70 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-04-09 09:15:00 | 837.70 | 2025-04-11 10:15:00 | 862.35 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-05-05 14:30:00 | 893.70 | 2025-05-06 09:15:00 | 880.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-05-14 13:45:00 | 905.40 | 2025-05-20 12:15:00 | 899.05 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-05-15 14:00:00 | 906.65 | 2025-05-20 12:15:00 | 899.05 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-05-16 12:15:00 | 905.25 | 2025-05-20 13:15:00 | 882.40 | STOP_HIT | 1.00 | -2.52% |
| BUY | retest2 | 2025-05-16 13:15:00 | 905.40 | 2025-05-20 13:15:00 | 882.40 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-05-19 09:15:00 | 906.00 | 2025-05-20 13:15:00 | 882.40 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-05-20 12:15:00 | 909.15 | 2025-05-20 13:15:00 | 882.40 | STOP_HIT | 1.00 | -2.94% |
| BUY | retest2 | 2025-05-23 12:15:00 | 907.95 | 2025-06-13 10:15:00 | 965.30 | STOP_HIT | 1.00 | 6.32% |
| SELL | retest2 | 2025-06-18 15:00:00 | 956.50 | 2025-06-24 10:15:00 | 959.50 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-06-24 10:00:00 | 959.35 | 2025-06-24 10:15:00 | 959.50 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-06-30 13:00:00 | 983.65 | 2025-07-07 13:15:00 | 995.70 | STOP_HIT | 1.00 | 1.23% |
| SELL | retest2 | 2025-07-11 11:45:00 | 971.25 | 2025-07-11 15:15:00 | 979.95 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest1 | 2025-07-28 09:15:00 | 980.80 | 2025-07-31 09:15:00 | 966.50 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-08-06 09:15:00 | 940.10 | 2025-08-11 10:15:00 | 955.20 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-08-13 11:00:00 | 966.30 | 2025-08-20 15:15:00 | 981.75 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1006.80 | 2025-09-15 13:15:00 | 1036.10 | STOP_HIT | 1.00 | 2.91% |
| SELL | retest2 | 2025-09-18 09:45:00 | 1031.90 | 2025-09-18 13:15:00 | 1045.05 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-18 10:45:00 | 1031.70 | 2025-09-18 13:15:00 | 1045.05 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-18 11:30:00 | 1031.55 | 2025-09-18 13:15:00 | 1045.05 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-22 10:30:00 | 1041.70 | 2025-09-22 15:15:00 | 1033.85 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-22 13:45:00 | 1041.20 | 2025-09-22 15:15:00 | 1033.85 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-10-03 12:00:00 | 987.80 | 2025-10-06 15:15:00 | 993.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-10-03 15:15:00 | 988.55 | 2025-10-06 15:15:00 | 993.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-10-08 09:15:00 | 985.65 | 2025-10-08 13:15:00 | 990.20 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-10-08 10:00:00 | 986.00 | 2025-10-08 13:15:00 | 990.20 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-08 14:45:00 | 985.80 | 2025-10-09 09:15:00 | 1001.20 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-15 14:15:00 | 980.00 | 2025-10-17 14:15:00 | 987.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-10-17 10:15:00 | 979.75 | 2025-10-17 14:15:00 | 987.20 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-27 09:30:00 | 1009.00 | 2025-10-28 14:15:00 | 1001.25 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-27 11:00:00 | 1010.30 | 2025-10-28 14:15:00 | 1001.25 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-10-27 12:30:00 | 1010.00 | 2025-10-28 14:15:00 | 1001.25 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-04 11:30:00 | 977.25 | 2025-11-07 10:15:00 | 928.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 12:00:00 | 976.95 | 2025-11-07 10:15:00 | 928.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:30:00 | 977.25 | 2025-11-07 15:15:00 | 945.00 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2025-11-04 12:00:00 | 976.95 | 2025-11-07 15:15:00 | 945.00 | STOP_HIT | 0.50 | 3.27% |
| SELL | retest2 | 2025-11-14 14:45:00 | 939.95 | 2025-11-25 11:15:00 | 932.25 | STOP_HIT | 1.00 | 0.82% |
| SELL | retest2 | 2025-11-17 09:15:00 | 942.25 | 2025-11-25 11:15:00 | 932.25 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2025-11-27 13:45:00 | 937.25 | 2025-12-01 13:15:00 | 937.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-11-27 14:30:00 | 937.35 | 2025-12-01 13:15:00 | 937.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-11-27 15:00:00 | 937.75 | 2025-12-01 13:15:00 | 937.00 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-12-10 11:15:00 | 927.55 | 2025-12-12 12:15:00 | 927.65 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-12-10 13:45:00 | 928.50 | 2025-12-12 12:15:00 | 927.65 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-12-17 14:15:00 | 916.95 | 2025-12-22 09:15:00 | 921.40 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-12-18 09:15:00 | 915.10 | 2025-12-22 09:15:00 | 921.40 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-12-19 10:45:00 | 916.45 | 2025-12-22 09:15:00 | 921.40 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-01-01 11:45:00 | 913.90 | 2026-01-08 13:15:00 | 915.20 | STOP_HIT | 1.00 | 0.14% |
| BUY | retest2 | 2026-01-01 12:15:00 | 913.00 | 2026-01-08 13:15:00 | 915.20 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2026-01-01 13:30:00 | 913.15 | 2026-01-08 13:15:00 | 915.20 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2026-01-02 09:30:00 | 914.65 | 2026-01-08 13:15:00 | 915.20 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2026-01-05 12:45:00 | 923.25 | 2026-01-08 13:15:00 | 915.20 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-01-05 14:45:00 | 923.50 | 2026-01-08 13:15:00 | 915.20 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2026-01-06 09:45:00 | 925.65 | 2026-01-08 13:15:00 | 915.20 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2026-01-14 10:00:00 | 882.30 | 2026-01-22 09:15:00 | 886.45 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-01-14 10:30:00 | 884.70 | 2026-01-22 09:15:00 | 886.45 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2026-01-14 12:45:00 | 883.25 | 2026-01-22 09:15:00 | 886.45 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2026-01-16 10:30:00 | 885.30 | 2026-01-22 09:15:00 | 886.45 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2026-01-20 12:15:00 | 876.05 | 2026-01-22 09:15:00 | 886.45 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-01-20 13:30:00 | 876.65 | 2026-01-22 09:15:00 | 886.45 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-01-21 10:30:00 | 873.40 | 2026-01-22 09:15:00 | 886.45 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-01-21 12:15:00 | 876.50 | 2026-01-22 09:15:00 | 886.45 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-01-21 13:45:00 | 873.20 | 2026-01-22 09:15:00 | 886.45 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-01-28 14:45:00 | 894.65 | 2026-01-30 10:15:00 | 885.15 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-01 09:15:00 | 880.05 | 2026-02-01 10:15:00 | 889.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-02-01 12:15:00 | 882.85 | 2026-02-03 09:15:00 | 901.10 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-02-01 13:15:00 | 883.90 | 2026-02-03 09:15:00 | 901.10 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-02-01 13:45:00 | 882.95 | 2026-02-03 09:15:00 | 901.10 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-02-02 09:15:00 | 876.50 | 2026-02-03 09:15:00 | 901.10 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-02-02 09:45:00 | 877.20 | 2026-02-03 09:15:00 | 901.10 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-02-23 10:45:00 | 903.70 | 2026-02-24 10:15:00 | 905.55 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2026-02-23 13:45:00 | 904.00 | 2026-02-24 10:15:00 | 905.55 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2026-02-24 09:15:00 | 902.35 | 2026-02-24 10:15:00 | 905.55 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-02-24 10:15:00 | 903.35 | 2026-02-24 10:15:00 | 905.55 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2026-02-27 14:45:00 | 929.65 | 2026-03-02 09:15:00 | 902.05 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2026-03-05 11:30:00 | 904.70 | 2026-03-05 13:15:00 | 910.80 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-03-12 10:15:00 | 920.00 | 2026-03-13 09:15:00 | 907.05 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-03-12 13:00:00 | 919.25 | 2026-03-13 09:15:00 | 907.05 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-03-17 11:15:00 | 893.80 | 2026-03-18 09:15:00 | 903.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-03-23 12:15:00 | 876.45 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2026-03-24 12:00:00 | 876.05 | 2026-03-25 09:15:00 | 901.65 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2026-04-01 10:45:00 | 877.05 | 2026-04-07 12:15:00 | 871.00 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2026-04-13 10:15:00 | 915.10 | 2026-04-21 10:15:00 | 931.50 | STOP_HIT | 1.00 | 1.79% |
| SELL | retest2 | 2026-04-22 12:30:00 | 929.85 | 2026-04-23 09:15:00 | 957.45 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2026-04-22 15:00:00 | 929.60 | 2026-04-23 09:15:00 | 957.45 | STOP_HIT | 1.00 | -3.00% |
