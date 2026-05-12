# Intellect Design Arena Ltd. (INTELLECT)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 808.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 84 |
| ALERT1 | 49 |
| ALERT2 | 47 |
| ALERT2_SKIP | 29 |
| ALERT3 | 130 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 53 |
| PARTIAL | 6 |
| TARGET_HIT | 9 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 60 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 41
- **Target hits / Stop hits / Partials:** 7 / 47 / 6
- **Avg / median % per leg:** 0.28% / -0.93%
- **Sum % (uncompounded):** 16.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 7 | 46.7% | 6 | 9 | 0 | 2.53% | 38.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.05% | -2.1% |
| BUY @ 3rd Alert (retest2) | 14 | 7 | 50.0% | 6 | 8 | 0 | 2.86% | 40.1% |
| SELL (all) | 45 | 12 | 26.7% | 1 | 38 | 6 | -0.47% | -21.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 45 | 12 | 26.7% | 1 | 38 | 6 | -0.47% | -21.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.05% | -2.1% |
| retest2 (combined) | 59 | 19 | 32.2% | 7 | 46 | 6 | 0.32% | 18.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 10:15:00 | 1172.00 | 1181.95 | 1182.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 12:15:00 | 1169.60 | 1177.90 | 1180.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 1167.70 | 1156.13 | 1163.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 1167.70 | 1156.13 | 1163.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 1167.70 | 1156.13 | 1163.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 1172.60 | 1156.13 | 1163.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 1170.10 | 1158.93 | 1164.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:30:00 | 1172.60 | 1158.93 | 1164.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1173.50 | 1164.49 | 1165.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:00:00 | 1173.50 | 1164.49 | 1165.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 14:15:00 | 1174.70 | 1166.53 | 1166.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 09:15:00 | 1226.50 | 1179.56 | 1172.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 10:15:00 | 1203.60 | 1204.32 | 1192.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-12 12:30:00 | 1211.90 | 1204.71 | 1194.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1187.00 | 1201.17 | 1193.78 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1187.00 | 1201.17 | 1193.78 | SL hit (close<ema400) qty=1.00 sl=1193.78 alert=retest1 |

### Cycle 3 — SELL (started 2025-06-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 14:15:00 | 1187.60 | 1193.30 | 1193.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1164.20 | 1186.81 | 1190.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 1190.20 | 1185.73 | 1188.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 12:15:00 | 1190.20 | 1185.73 | 1188.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1190.20 | 1185.73 | 1188.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 1183.50 | 1185.73 | 1188.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1199.90 | 1188.56 | 1189.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:45:00 | 1199.80 | 1188.56 | 1189.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 1213.90 | 1193.63 | 1191.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1238.20 | 1204.86 | 1197.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 1220.40 | 1221.52 | 1212.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 10:15:00 | 1220.40 | 1221.52 | 1212.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1220.40 | 1221.52 | 1212.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:45:00 | 1216.10 | 1221.52 | 1212.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1212.70 | 1219.76 | 1212.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:00:00 | 1212.70 | 1219.76 | 1212.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 1204.90 | 1216.79 | 1212.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:30:00 | 1204.00 | 1216.79 | 1212.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 1199.80 | 1213.39 | 1211.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:15:00 | 1194.10 | 1213.39 | 1211.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 15:15:00 | 1198.00 | 1207.85 | 1208.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1170.80 | 1200.44 | 1205.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 13:15:00 | 1071.70 | 1069.72 | 1092.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-24 13:45:00 | 1071.60 | 1069.72 | 1092.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1088.30 | 1075.78 | 1089.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:15:00 | 1090.50 | 1075.78 | 1089.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 1087.30 | 1078.09 | 1089.61 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 14:15:00 | 1116.80 | 1096.28 | 1095.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 1135.70 | 1107.01 | 1100.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 1119.10 | 1119.31 | 1110.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 1119.10 | 1119.31 | 1110.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1111.50 | 1118.75 | 1112.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 1111.50 | 1118.75 | 1112.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 1118.80 | 1118.76 | 1113.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:45:00 | 1134.30 | 1127.58 | 1118.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 11:45:00 | 1140.30 | 1131.33 | 1123.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 11:15:00 | 1139.00 | 1140.21 | 1137.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 10:15:00 | 1131.90 | 1139.71 | 1140.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 10:15:00 | 1131.90 | 1139.71 | 1140.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 1125.60 | 1136.89 | 1138.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 1140.90 | 1133.18 | 1135.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 1140.90 | 1133.18 | 1135.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1140.90 | 1133.18 | 1135.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 1144.30 | 1133.18 | 1135.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1139.10 | 1134.36 | 1135.97 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 1149.40 | 1139.34 | 1138.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 13:15:00 | 1161.80 | 1148.70 | 1143.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1160.50 | 1173.13 | 1164.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 1160.50 | 1173.13 | 1164.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1160.50 | 1173.13 | 1164.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 1163.00 | 1173.13 | 1164.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1154.10 | 1169.32 | 1163.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:30:00 | 1151.00 | 1169.32 | 1163.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 1147.40 | 1159.82 | 1159.90 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 1173.30 | 1158.87 | 1157.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 1177.50 | 1166.31 | 1161.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 14:15:00 | 1169.90 | 1171.53 | 1165.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 14:15:00 | 1169.90 | 1171.53 | 1165.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1169.90 | 1171.53 | 1165.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:45:00 | 1168.60 | 1171.53 | 1165.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1177.40 | 1181.36 | 1176.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:45:00 | 1177.30 | 1181.36 | 1176.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1175.10 | 1180.11 | 1176.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 1174.00 | 1180.11 | 1176.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1188.10 | 1181.71 | 1177.14 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 1171.70 | 1178.42 | 1178.90 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 09:15:00 | 1199.10 | 1179.18 | 1177.38 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 13:15:00 | 1147.80 | 1179.38 | 1181.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 1139.60 | 1171.43 | 1177.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 10:15:00 | 1081.50 | 1079.75 | 1114.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:30:00 | 1089.60 | 1079.75 | 1114.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 1045.50 | 1041.56 | 1061.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 1041.00 | 1041.56 | 1061.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 13:15:00 | 1063.30 | 1050.86 | 1059.97 | SL hit (close>static) qty=1.00 sl=1061.90 alert=retest2 |

### Cycle 14 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 941.20 | 932.36 | 931.74 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 909.10 | 928.61 | 930.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 12:15:00 | 905.70 | 918.62 | 924.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 925.00 | 914.94 | 920.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 925.00 | 914.94 | 920.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 925.00 | 914.94 | 920.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 925.00 | 914.94 | 920.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 937.20 | 919.40 | 922.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 11:00:00 | 937.20 | 919.40 | 922.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 11:15:00 | 953.90 | 926.30 | 924.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 963.00 | 933.64 | 928.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 966.10 | 968.36 | 954.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 966.10 | 968.36 | 954.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 969.20 | 966.87 | 956.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 955.40 | 966.87 | 956.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 982.40 | 972.31 | 965.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:45:00 | 997.40 | 978.80 | 969.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 14:30:00 | 988.50 | 980.84 | 977.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 15:15:00 | 965.50 | 975.56 | 976.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 965.50 | 975.56 | 976.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 964.00 | 973.25 | 975.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 979.30 | 974.46 | 975.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-29 10:15:00 | 979.30 | 974.46 | 975.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 979.30 | 974.46 | 975.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 11:00:00 | 979.30 | 974.46 | 975.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 975.00 | 974.57 | 975.76 | EMA400 retest candle locked (from downside) |

### Cycle 18 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 980.90 | 976.82 | 976.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 990.05 | 981.13 | 978.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 984.85 | 987.05 | 983.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 984.85 | 987.05 | 983.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 984.85 | 987.05 | 983.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:30:00 | 981.10 | 987.05 | 983.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 988.45 | 987.33 | 983.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:15:00 | 982.05 | 987.33 | 983.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 982.05 | 986.27 | 983.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 975.85 | 986.27 | 983.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 971.80 | 983.38 | 982.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 969.40 | 983.38 | 982.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 967.35 | 980.17 | 981.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 11:15:00 | 965.65 | 977.27 | 979.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 933.70 | 913.96 | 921.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 933.70 | 913.96 | 921.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 933.70 | 913.96 | 921.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 933.70 | 913.96 | 921.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 941.00 | 919.37 | 923.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 943.95 | 919.37 | 923.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 12:15:00 | 949.50 | 929.49 | 927.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 13:15:00 | 968.00 | 937.20 | 931.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 1020.75 | 1020.98 | 1005.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 1020.75 | 1020.98 | 1005.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1019.80 | 1021.56 | 1015.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:30:00 | 1016.10 | 1021.56 | 1015.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 1030.15 | 1032.98 | 1026.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 1030.20 | 1032.98 | 1026.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 1025.65 | 1031.52 | 1026.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 1025.65 | 1031.52 | 1026.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1027.05 | 1030.62 | 1026.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 1023.45 | 1030.62 | 1026.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1027.25 | 1029.95 | 1026.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 1052.00 | 1029.95 | 1026.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1052.05 | 1058.81 | 1059.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-09-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 09:15:00 | 1052.05 | 1058.81 | 1059.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 1038.00 | 1052.90 | 1056.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 1075.15 | 1046.46 | 1050.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 1075.15 | 1046.46 | 1050.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1075.15 | 1046.46 | 1050.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 1075.15 | 1046.46 | 1050.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1063.85 | 1049.94 | 1051.80 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 11:15:00 | 1068.35 | 1053.62 | 1053.31 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 1009.75 | 1049.56 | 1052.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 1005.00 | 1040.65 | 1048.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 976.00 | 974.16 | 995.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 10:00:00 | 976.00 | 974.16 | 995.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 975.15 | 974.38 | 980.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:30:00 | 976.90 | 974.38 | 980.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 976.65 | 974.76 | 979.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:30:00 | 984.65 | 974.76 | 979.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 14:15:00 | 984.00 | 976.78 | 978.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 15:00:00 | 984.00 | 976.78 | 978.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 983.75 | 978.17 | 979.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:15:00 | 979.00 | 978.17 | 979.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 10:15:00 | 988.15 | 981.36 | 980.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 988.15 | 981.36 | 980.56 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 10:15:00 | 973.75 | 980.27 | 980.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 11:15:00 | 971.50 | 978.52 | 979.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 12:15:00 | 985.85 | 979.98 | 980.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 12:15:00 | 985.85 | 979.98 | 980.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 985.85 | 979.98 | 980.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:30:00 | 982.95 | 979.98 | 980.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 985.75 | 981.14 | 980.86 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-10-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 11:15:00 | 975.70 | 980.94 | 981.02 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 15:15:00 | 988.80 | 981.20 | 981.01 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-09 09:15:00 | 969.25 | 978.81 | 979.94 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 10:15:00 | 993.20 | 980.91 | 979.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 11:15:00 | 1009.10 | 986.55 | 982.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 992.65 | 998.41 | 990.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 992.65 | 998.41 | 990.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 992.65 | 998.41 | 990.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:45:00 | 990.30 | 998.41 | 990.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 990.00 | 996.73 | 990.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 990.00 | 996.73 | 990.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 988.40 | 995.06 | 990.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 988.55 | 995.06 | 990.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 988.05 | 993.66 | 990.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 988.05 | 993.66 | 990.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 977.00 | 986.90 | 987.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 962.85 | 982.09 | 985.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 997.05 | 971.10 | 976.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 997.05 | 971.10 | 976.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 997.05 | 971.10 | 976.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 997.05 | 971.10 | 976.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 992.25 | 975.33 | 977.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 995.00 | 975.33 | 977.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1000.15 | 980.29 | 979.95 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 11:15:00 | 971.10 | 981.39 | 981.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 09:15:00 | 966.60 | 974.24 | 977.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 973.90 | 957.19 | 961.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 973.90 | 957.19 | 961.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 973.90 | 957.19 | 961.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 973.90 | 957.19 | 961.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 969.60 | 959.67 | 962.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 959.40 | 959.67 | 962.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 964.10 | 960.60 | 962.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 964.10 | 960.60 | 962.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 971.15 | 962.71 | 963.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 971.15 | 962.71 | 963.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 12:15:00 | 975.15 | 965.20 | 964.26 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 956.40 | 962.55 | 963.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 954.50 | 960.94 | 962.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 963.35 | 961.00 | 962.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 963.35 | 961.00 | 962.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 963.35 | 961.00 | 962.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:00:00 | 963.35 | 961.00 | 962.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 13:15:00 | 992.55 | 967.31 | 964.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-24 14:15:00 | 998.10 | 973.47 | 968.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 1026.00 | 1026.21 | 1007.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 10:30:00 | 1025.90 | 1026.21 | 1007.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1017.80 | 1021.95 | 1013.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 10:45:00 | 1043.80 | 1025.26 | 1015.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 1054.35 | 1027.38 | 1020.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 1050.30 | 1033.56 | 1024.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 1045.55 | 1037.57 | 1028.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-03 09:15:00 | 1148.18 | 1122.88 | 1085.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 1131.30 | 1148.74 | 1149.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 1123.30 | 1143.65 | 1146.81 | Break + close below crossover candle low |

### Cycle 38 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 1156.00 | 1110.80 | 1110.70 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 1087.40 | 1115.12 | 1117.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1079.10 | 1093.50 | 1099.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 11:15:00 | 1105.90 | 1095.82 | 1099.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 11:15:00 | 1105.90 | 1095.82 | 1099.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 1105.90 | 1095.82 | 1099.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 1105.90 | 1095.82 | 1099.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 1100.80 | 1096.81 | 1099.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 1108.80 | 1096.81 | 1099.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 1099.70 | 1097.39 | 1099.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 1099.70 | 1097.39 | 1099.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 1087.50 | 1095.41 | 1098.47 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 1137.90 | 1102.72 | 1101.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 10:15:00 | 1158.50 | 1113.88 | 1106.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 1109.20 | 1131.35 | 1121.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1109.20 | 1131.35 | 1121.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1109.20 | 1131.35 | 1121.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 1099.60 | 1131.35 | 1121.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1091.30 | 1123.34 | 1118.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 11:00:00 | 1091.30 | 1123.34 | 1118.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 1095.90 | 1113.62 | 1114.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1078.20 | 1099.70 | 1107.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 11:15:00 | 1093.80 | 1092.18 | 1098.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 12:00:00 | 1093.80 | 1092.18 | 1098.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1098.30 | 1093.40 | 1098.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:45:00 | 1098.40 | 1093.40 | 1098.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1099.50 | 1094.62 | 1098.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 1099.50 | 1094.62 | 1098.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1095.50 | 1094.80 | 1097.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:15:00 | 1090.30 | 1094.80 | 1097.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 1092.80 | 1092.43 | 1095.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:15:00 | 1092.70 | 1093.19 | 1095.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1101.00 | 1093.36 | 1094.83 | SL hit (close>static) qty=1.00 sl=1100.10 alert=retest2 |

### Cycle 42 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1103.30 | 1095.77 | 1095.69 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 1092.70 | 1095.83 | 1096.04 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 1110.00 | 1098.66 | 1097.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 1132.30 | 1112.91 | 1106.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 12:15:00 | 1113.90 | 1114.24 | 1108.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 12:45:00 | 1114.60 | 1114.24 | 1108.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 1109.70 | 1113.33 | 1108.63 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 1095.00 | 1105.72 | 1106.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 1091.50 | 1102.88 | 1105.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 09:15:00 | 1100.00 | 1099.46 | 1102.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 1100.00 | 1099.46 | 1102.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1100.00 | 1099.46 | 1102.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 1094.20 | 1098.53 | 1101.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:30:00 | 1090.20 | 1096.32 | 1100.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 1109.30 | 1100.91 | 1100.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 1109.30 | 1100.91 | 1100.85 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 1091.80 | 1099.60 | 1100.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 09:15:00 | 1080.00 | 1094.11 | 1097.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 1031.00 | 1030.18 | 1046.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:15:00 | 1035.40 | 1030.18 | 1046.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1044.20 | 1037.34 | 1045.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 1028.90 | 1036.17 | 1044.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 13:30:00 | 1033.00 | 1028.48 | 1033.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:30:00 | 1029.00 | 1029.92 | 1033.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 1029.50 | 1031.94 | 1034.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1032.90 | 1032.30 | 1034.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 1031.70 | 1032.30 | 1034.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1030.30 | 1031.90 | 1033.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:30:00 | 1027.00 | 1030.82 | 1033.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1025.50 | 1030.18 | 1032.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 11:45:00 | 1027.10 | 1029.32 | 1031.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1025.00 | 1029.13 | 1030.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1020.00 | 1027.30 | 1029.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:00:00 | 1012.10 | 1021.85 | 1026.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 13:45:00 | 1015.90 | 1020.68 | 1025.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 1013.30 | 1021.92 | 1023.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 1054.50 | 1029.74 | 1026.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1054.50 | 1029.74 | 1026.67 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 15:15:00 | 1027.00 | 1032.68 | 1033.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 10:15:00 | 1024.20 | 1030.48 | 1032.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 09:15:00 | 1001.50 | 1000.45 | 1008.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1001.50 | 1000.45 | 1008.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1001.50 | 1000.45 | 1008.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 1007.00 | 1000.45 | 1008.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 995.30 | 999.42 | 1007.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 994.70 | 999.42 | 1007.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:45:00 | 994.70 | 998.54 | 1006.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 12:15:00 | 993.10 | 998.54 | 1006.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 15:15:00 | 990.40 | 997.30 | 1003.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 14:15:00 | 944.97 | 965.47 | 982.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 14:15:00 | 944.97 | 965.47 | 982.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 14:15:00 | 943.44 | 965.47 | 982.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-30 14:15:00 | 940.88 | 965.47 | 982.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 970.10 | 962.97 | 974.16 | SL hit (close>ema200) qty=0.50 sl=962.97 alert=retest2 |

### Cycle 50 — BUY (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 13:15:00 | 967.60 | 962.49 | 962.31 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 958.00 | 962.60 | 962.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 948.80 | 958.24 | 960.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 972.00 | 957.47 | 959.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 972.00 | 957.47 | 959.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 972.00 | 957.47 | 959.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 972.00 | 957.47 | 959.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 976.85 | 961.35 | 960.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 13:15:00 | 987.05 | 970.39 | 965.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 962.85 | 975.72 | 970.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 962.85 | 975.72 | 970.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 962.85 | 975.72 | 970.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 962.85 | 975.72 | 970.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 969.65 | 974.51 | 970.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:30:00 | 966.10 | 974.51 | 970.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 951.80 | 967.44 | 967.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 945.35 | 963.02 | 965.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 919.75 | 916.03 | 928.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 09:15:00 | 925.35 | 916.03 | 928.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 936.60 | 920.14 | 929.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 936.60 | 920.14 | 929.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 940.90 | 924.29 | 930.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 940.90 | 924.29 | 930.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 930.65 | 929.21 | 931.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 930.65 | 929.21 | 931.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 929.00 | 929.17 | 931.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 930.10 | 929.17 | 931.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 917.05 | 926.75 | 929.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 904.35 | 923.58 | 925.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 911.75 | 901.53 | 903.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 11:15:00 | 913.40 | 905.32 | 905.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 913.40 | 905.32 | 905.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 917.10 | 907.68 | 906.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 905.85 | 913.15 | 909.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 905.85 | 913.15 | 909.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 905.85 | 913.15 | 909.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 902.60 | 913.15 | 909.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 908.55 | 912.23 | 909.69 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 892.65 | 905.50 | 906.89 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 907.50 | 901.56 | 900.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 908.40 | 903.70 | 902.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 888.30 | 901.63 | 901.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 888.30 | 901.63 | 901.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 888.30 | 901.63 | 901.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 888.00 | 901.63 | 901.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 889.00 | 899.10 | 900.35 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 919.15 | 903.15 | 901.40 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 09:15:00 | 834.00 | 896.61 | 900.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 777.25 | 796.01 | 811.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 766.20 | 764.41 | 775.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 10:30:00 | 767.10 | 764.41 | 775.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 775.95 | 766.72 | 775.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:00:00 | 775.95 | 766.72 | 775.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 12:15:00 | 773.85 | 768.15 | 775.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 12:30:00 | 774.60 | 768.15 | 775.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 773.60 | 769.24 | 775.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 14:00:00 | 773.60 | 769.24 | 775.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 775.90 | 771.05 | 774.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 11:00:00 | 767.80 | 770.40 | 774.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:15:00 | 729.41 | 740.81 | 753.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-13 09:15:00 | 691.02 | 712.24 | 730.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 60 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 736.30 | 704.71 | 701.68 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 692.00 | 701.96 | 702.87 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 704.50 | 703.35 | 703.29 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-02-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 15:15:00 | 702.50 | 703.18 | 703.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 09:15:00 | 700.20 | 702.58 | 702.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 10:15:00 | 704.00 | 702.87 | 703.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 10:15:00 | 704.00 | 702.87 | 703.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 704.00 | 702.87 | 703.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 10:30:00 | 702.70 | 702.87 | 703.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 700.50 | 702.39 | 702.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 12:15:00 | 697.00 | 702.39 | 702.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 15:15:00 | 704.60 | 701.91 | 702.34 | SL hit (close>static) qty=1.00 sl=704.20 alert=retest2 |

### Cycle 64 — BUY (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 13:15:00 | 700.15 | 697.29 | 697.10 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 692.35 | 697.13 | 697.15 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-27 10:15:00 | 697.55 | 697.22 | 697.19 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 690.80 | 695.93 | 696.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 684.00 | 693.40 | 695.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 686.00 | 685.05 | 689.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 686.00 | 685.05 | 689.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 679.00 | 664.99 | 670.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 679.00 | 664.99 | 670.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 684.45 | 668.88 | 671.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 684.45 | 668.88 | 671.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2026-03-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 12:15:00 | 688.60 | 675.73 | 674.60 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-03-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 12:15:00 | 665.30 | 673.46 | 674.48 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 682.00 | 674.76 | 674.60 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 11:15:00 | 669.35 | 674.06 | 674.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 666.80 | 672.61 | 673.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 668.25 | 664.90 | 668.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 668.25 | 664.90 | 668.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 669.35 | 665.79 | 668.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:00:00 | 669.35 | 665.79 | 668.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 671.00 | 666.83 | 668.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 671.95 | 666.83 | 668.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 667.15 | 666.89 | 668.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 14:45:00 | 667.70 | 666.89 | 668.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 664.35 | 665.05 | 667.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:30:00 | 667.20 | 665.05 | 667.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 660.65 | 663.46 | 665.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:30:00 | 664.05 | 663.46 | 665.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 655.55 | 661.48 | 664.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:15:00 | 647.45 | 661.48 | 664.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 11:00:00 | 647.45 | 658.68 | 662.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 11:45:00 | 647.95 | 656.75 | 661.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:30:00 | 646.35 | 654.69 | 660.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 646.70 | 649.48 | 655.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:00:00 | 642.65 | 647.66 | 652.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 676.90 | 652.66 | 653.88 | SL hit (close>static) qty=1.00 sl=666.75 alert=retest2 |

### Cycle 72 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 692.40 | 660.61 | 657.38 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2026-03-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 11:15:00 | 658.75 | 662.89 | 663.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 657.50 | 661.34 | 662.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 647.50 | 641.54 | 647.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 647.50 | 641.54 | 647.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 647.50 | 641.54 | 647.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 647.50 | 641.54 | 647.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 648.00 | 642.83 | 647.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 647.05 | 642.83 | 647.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 646.00 | 643.46 | 647.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 648.05 | 643.46 | 647.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 643.35 | 643.44 | 647.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:45:00 | 644.25 | 643.44 | 647.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 657.50 | 646.82 | 647.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 657.50 | 646.82 | 647.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 662.25 | 649.91 | 649.27 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 637.80 | 649.47 | 650.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 634.45 | 644.19 | 647.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 639.85 | 616.33 | 626.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 639.85 | 616.33 | 626.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 639.85 | 616.33 | 626.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 639.85 | 616.33 | 626.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 635.20 | 620.10 | 626.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 639.20 | 620.10 | 626.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 634.70 | 630.66 | 630.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 15:15:00 | 638.70 | 632.27 | 631.36 | Break + close above crossover candle high |

### Cycle 77 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 621.10 | 630.03 | 630.42 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 646.20 | 633.13 | 631.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 658.55 | 638.21 | 634.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 675.50 | 679.03 | 670.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 675.50 | 679.03 | 670.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 673.75 | 678.08 | 671.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:00:00 | 673.75 | 678.08 | 671.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 669.85 | 676.43 | 671.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 669.85 | 676.43 | 671.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 671.95 | 675.53 | 671.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:45:00 | 669.00 | 675.53 | 671.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 673.40 | 675.11 | 671.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:15:00 | 672.90 | 675.11 | 671.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 680.60 | 676.21 | 672.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:30:00 | 685.65 | 678.24 | 673.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 684.70 | 682.33 | 677.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 646.20 | 675.48 | 675.36 | SL hit (close<static) qty=1.00 sl=670.00 alert=retest2 |

### Cycle 79 — SELL (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 10:15:00 | 657.30 | 671.85 | 673.71 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 681.85 | 669.55 | 668.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 690.00 | 673.64 | 670.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 13:15:00 | 694.10 | 695.79 | 689.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-17 13:30:00 | 695.35 | 695.79 | 689.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 684.30 | 693.17 | 690.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 679.05 | 693.17 | 690.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 685.00 | 691.54 | 689.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 11:00:00 | 685.00 | 691.54 | 689.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 681.05 | 688.79 | 688.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:00:00 | 681.05 | 688.79 | 688.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 678.75 | 686.78 | 687.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 09:15:00 | 665.70 | 677.52 | 681.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 695.00 | 677.87 | 680.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-22 13:15:00 | 695.00 | 677.87 | 680.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 695.00 | 677.87 | 680.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 14:00:00 | 695.00 | 677.87 | 680.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 14:15:00 | 694.65 | 681.23 | 681.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 15:15:00 | 697.50 | 681.23 | 681.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 697.50 | 684.48 | 682.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 703.70 | 688.32 | 684.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 09:15:00 | 700.00 | 700.17 | 694.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 09:15:00 | 700.00 | 700.17 | 694.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 700.00 | 700.17 | 694.17 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 683.65 | 691.47 | 691.52 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 712.25 | 694.50 | 692.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 725.05 | 708.94 | 702.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 738.95 | 742.51 | 730.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 14:00:00 | 738.95 | 742.51 | 730.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 740.70 | 741.73 | 733.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 14:15:00 | 755.00 | 742.65 | 740.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 14:15:00 | 830.50 | 795.53 | 784.79 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 866.80 | 2025-05-12 12:15:00 | 940.67 | TARGET_HIT | 1.00 | 8.52% |
| BUY | retest1 | 2025-06-12 12:30:00 | 1211.90 | 2025-06-12 13:15:00 | 1187.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-06-27 14:45:00 | 1134.30 | 2025-07-04 10:15:00 | 1131.90 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-06-30 11:45:00 | 1140.30 | 2025-07-04 10:15:00 | 1131.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-02 11:15:00 | 1139.00 | 2025-07-04 10:15:00 | 1131.90 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-30 10:15:00 | 1041.00 | 2025-07-30 13:15:00 | 1063.30 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1038.70 | 2025-08-01 14:15:00 | 986.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-31 09:15:00 | 1038.70 | 2025-08-04 11:15:00 | 997.70 | STOP_HIT | 0.50 | 3.95% |
| BUY | retest2 | 2025-08-25 11:45:00 | 997.40 | 2025-08-28 15:15:00 | 965.50 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2025-08-26 14:30:00 | 988.50 | 2025-08-28 15:15:00 | 965.50 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-09-18 09:15:00 | 1052.00 | 2025-09-24 09:15:00 | 1052.05 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-10-06 09:15:00 | 979.00 | 2025-10-06 10:15:00 | 988.15 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-10-29 10:45:00 | 1043.80 | 2025-11-03 09:15:00 | 1148.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-30 09:15:00 | 1054.35 | 2025-11-03 09:15:00 | 1159.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-30 11:15:00 | 1050.30 | 2025-11-03 09:15:00 | 1155.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-30 12:30:00 | 1045.55 | 2025-11-03 09:15:00 | 1150.11 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-24 15:15:00 | 1090.30 | 2025-11-26 09:15:00 | 1101.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-11-25 12:15:00 | 1092.80 | 2025-11-26 09:15:00 | 1101.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-11-25 14:15:00 | 1092.70 | 2025-11-26 09:15:00 | 1101.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-26 09:45:00 | 1093.00 | 2025-11-26 11:15:00 | 1103.30 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-03 10:30:00 | 1094.20 | 2025-12-04 10:15:00 | 1109.30 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-12-03 12:30:00 | 1090.20 | 2025-12-04 10:15:00 | 1109.30 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-12-10 10:45:00 | 1028.90 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-12-11 13:30:00 | 1033.00 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-12-11 14:30:00 | 1029.00 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-12-12 09:15:00 | 1029.50 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-12-12 12:30:00 | 1027.00 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1025.50 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-12-15 11:45:00 | 1027.10 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1025.00 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -2.88% |
| SELL | retest2 | 2025-12-16 13:00:00 | 1012.10 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-12-16 13:45:00 | 1015.90 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2025-12-18 09:30:00 | 1013.30 | 2025-12-19 09:15:00 | 1054.50 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2025-12-29 11:15:00 | 994.70 | 2025-12-30 14:15:00 | 944.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 11:45:00 | 994.70 | 2025-12-30 14:15:00 | 944.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 12:15:00 | 993.10 | 2025-12-30 14:15:00 | 943.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 15:15:00 | 990.40 | 2025-12-30 14:15:00 | 940.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-29 11:15:00 | 994.70 | 2025-12-31 12:15:00 | 970.10 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2025-12-29 11:45:00 | 994.70 | 2025-12-31 12:15:00 | 970.10 | STOP_HIT | 0.50 | 2.47% |
| SELL | retest2 | 2025-12-29 12:15:00 | 993.10 | 2025-12-31 12:15:00 | 970.10 | STOP_HIT | 0.50 | 2.32% |
| SELL | retest2 | 2025-12-29 15:15:00 | 990.40 | 2025-12-31 12:15:00 | 970.10 | STOP_HIT | 0.50 | 2.05% |
| SELL | retest2 | 2026-01-01 10:30:00 | 962.45 | 2026-01-05 13:15:00 | 967.60 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-01 12:00:00 | 962.80 | 2026-01-05 13:15:00 | 967.60 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2026-01-19 09:15:00 | 904.35 | 2026-01-22 11:15:00 | 913.40 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-22 10:15:00 | 911.75 | 2026-01-22 11:15:00 | 913.40 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-02-10 11:00:00 | 767.80 | 2026-02-12 09:15:00 | 729.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 11:00:00 | 767.80 | 2026-02-13 09:15:00 | 691.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-23 12:15:00 | 697.00 | 2026-02-23 15:15:00 | 704.60 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-02-24 09:15:00 | 687.80 | 2026-02-26 13:15:00 | 700.15 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-02-24 15:15:00 | 694.35 | 2026-02-26 13:15:00 | 700.15 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-25 10:30:00 | 687.50 | 2026-02-26 13:15:00 | 700.15 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2026-02-26 12:15:00 | 694.50 | 2026-02-26 13:15:00 | 700.15 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2026-03-16 10:15:00 | 647.45 | 2026-03-18 09:15:00 | 676.90 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2026-03-16 11:00:00 | 647.45 | 2026-03-18 09:15:00 | 676.90 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest2 | 2026-03-16 11:45:00 | 647.95 | 2026-03-18 09:15:00 | 676.90 | STOP_HIT | 1.00 | -4.47% |
| SELL | retest2 | 2026-03-16 12:30:00 | 646.35 | 2026-03-18 09:15:00 | 676.90 | STOP_HIT | 1.00 | -4.73% |
| SELL | retest2 | 2026-03-17 15:00:00 | 642.65 | 2026-03-18 09:15:00 | 676.90 | STOP_HIT | 1.00 | -5.33% |
| BUY | retest2 | 2026-04-10 10:30:00 | 685.65 | 2026-04-13 09:15:00 | 646.20 | STOP_HIT | 1.00 | -5.75% |
| BUY | retest2 | 2026-04-10 15:15:00 | 684.70 | 2026-04-13 09:15:00 | 646.20 | STOP_HIT | 1.00 | -5.62% |
| BUY | retest2 | 2026-05-05 14:15:00 | 755.00 | 2026-05-08 14:15:00 | 830.50 | TARGET_HIT | 1.00 | 10.00% |
