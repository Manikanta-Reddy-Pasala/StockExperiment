# Kalpataru Projects International Ltd. (KPIL)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1277.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 96 |
| ALERT1 | 60 |
| ALERT2 | 59 |
| ALERT2_SKIP | 36 |
| ALERT3 | 176 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 97 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 96 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 100 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 9 / 91
- **Target hits / Stop hits / Partials:** 1 / 96 / 3
- **Avg / median % per leg:** -1.15% / -1.10%
- **Sum % (uncompounded):** -114.89%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 58 | 1 | 1.7% | 1 | 57 | 0 | -1.24% | -71.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 58 | 1 | 1.7% | 1 | 57 | 0 | -1.24% | -71.8% |
| SELL (all) | 42 | 8 | 19.0% | 0 | 39 | 3 | -1.03% | -43.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 8 | 19.0% | 0 | 39 | 3 | -1.03% | -43.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 100 | 9 | 9.0% | 1 | 96 | 3 | -1.15% | -114.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 986.70 | 951.46 | 949.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1001.75 | 977.11 | 964.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 14:15:00 | 1109.15 | 1118.26 | 1102.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-20 15:00:00 | 1109.15 | 1118.26 | 1102.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1102.15 | 1113.39 | 1102.63 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 13:15:00 | 1076.20 | 1095.22 | 1096.58 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 1105.50 | 1098.10 | 1097.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 1118.90 | 1105.86 | 1102.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 1107.45 | 1113.26 | 1108.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 1107.45 | 1113.26 | 1108.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1107.45 | 1113.26 | 1108.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:00:00 | 1132.15 | 1116.99 | 1111.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 14:30:00 | 1128.35 | 1117.00 | 1114.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 1127.80 | 1118.40 | 1115.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:45:00 | 1129.10 | 1120.26 | 1116.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1114.00 | 1119.57 | 1116.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 1114.00 | 1119.57 | 1116.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 1113.50 | 1118.35 | 1116.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 09:15:00 | 1115.80 | 1117.23 | 1116.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 1109.15 | 1115.61 | 1115.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 1109.15 | 1115.61 | 1115.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 1109.15 | 1115.61 | 1115.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 1109.15 | 1115.61 | 1115.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 09:15:00 | 1109.15 | 1115.61 | 1115.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 1109.15 | 1115.61 | 1115.75 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 1121.50 | 1116.79 | 1116.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 11:15:00 | 1123.60 | 1118.15 | 1116.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1138.70 | 1148.95 | 1140.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 1138.70 | 1148.95 | 1140.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1138.70 | 1148.95 | 1140.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 1138.70 | 1148.95 | 1140.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 1171.60 | 1153.48 | 1143.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:15:00 | 1179.50 | 1153.48 | 1143.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 14:30:00 | 1173.90 | 1163.97 | 1155.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 15:15:00 | 1151.00 | 1154.91 | 1155.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-05 15:15:00 | 1151.00 | 1154.91 | 1155.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 1151.00 | 1154.91 | 1155.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 1143.60 | 1152.65 | 1154.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 11:15:00 | 1152.00 | 1151.81 | 1153.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 11:15:00 | 1152.00 | 1151.81 | 1153.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 1152.00 | 1151.81 | 1153.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:30:00 | 1153.10 | 1151.81 | 1153.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 1171.90 | 1155.82 | 1155.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 1189.40 | 1162.52 | 1158.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 1183.60 | 1185.52 | 1174.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 09:15:00 | 1181.80 | 1185.52 | 1174.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 1172.10 | 1181.52 | 1174.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 1171.80 | 1181.52 | 1174.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1171.30 | 1179.47 | 1174.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 12:30:00 | 1172.80 | 1177.60 | 1173.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 14:15:00 | 1167.10 | 1174.12 | 1172.75 | SL hit (close<static) qty=1.00 sl=1168.60 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:30:00 | 1174.00 | 1173.70 | 1172.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 10:30:00 | 1174.50 | 1174.24 | 1173.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:00:00 | 1172.60 | 1173.87 | 1173.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 1181.00 | 1176.53 | 1174.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 1189.00 | 1176.53 | 1174.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1161.30 | 1174.63 | 1174.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1161.30 | 1174.63 | 1174.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1161.30 | 1174.63 | 1174.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-12 13:15:00 | 1161.30 | 1174.63 | 1174.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 1161.30 | 1174.63 | 1174.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1144.20 | 1158.98 | 1164.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 12:15:00 | 1163.50 | 1157.43 | 1162.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 12:15:00 | 1163.50 | 1157.43 | 1162.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1163.50 | 1157.43 | 1162.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:00:00 | 1163.50 | 1157.43 | 1162.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1176.50 | 1161.24 | 1163.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1176.50 | 1161.24 | 1163.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1176.90 | 1164.37 | 1164.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 1178.00 | 1164.37 | 1164.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 1173.20 | 1166.14 | 1165.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1200.50 | 1173.01 | 1168.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 10:15:00 | 1184.90 | 1191.13 | 1183.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 10:45:00 | 1184.10 | 1191.13 | 1183.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1195.70 | 1192.04 | 1184.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 1197.10 | 1192.04 | 1184.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1199.80 | 1198.66 | 1191.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:30:00 | 1192.10 | 1198.66 | 1191.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 1192.90 | 1197.51 | 1191.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 1192.90 | 1197.51 | 1191.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1189.70 | 1195.95 | 1191.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 1184.40 | 1195.95 | 1191.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 1192.20 | 1195.20 | 1191.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 1185.10 | 1195.20 | 1191.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1192.40 | 1194.64 | 1191.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:15:00 | 1190.00 | 1194.64 | 1191.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 1190.00 | 1193.71 | 1191.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 1176.20 | 1193.71 | 1191.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 1168.40 | 1188.65 | 1189.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 1161.20 | 1177.32 | 1183.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 1179.50 | 1171.29 | 1177.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1179.50 | 1171.29 | 1177.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1179.50 | 1171.29 | 1177.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1179.50 | 1171.29 | 1177.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1184.30 | 1173.89 | 1178.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 1184.30 | 1173.89 | 1178.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 1186.10 | 1176.34 | 1179.23 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 14:15:00 | 1188.60 | 1182.02 | 1181.39 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 12:15:00 | 1180.00 | 1181.21 | 1181.31 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 15:15:00 | 1182.00 | 1181.46 | 1181.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1214.30 | 1188.03 | 1184.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 14:15:00 | 1225.70 | 1227.82 | 1216.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 15:00:00 | 1225.70 | 1227.82 | 1216.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1219.10 | 1225.48 | 1217.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 15:00:00 | 1240.80 | 1225.97 | 1222.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 1211.80 | 1221.04 | 1220.85 | SL hit (close<static) qty=1.00 sl=1212.10 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 1210.10 | 1218.85 | 1219.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 1193.10 | 1208.16 | 1213.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 1199.00 | 1198.03 | 1205.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-02 15:00:00 | 1199.00 | 1198.03 | 1205.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1208.70 | 1198.64 | 1204.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1208.70 | 1198.64 | 1204.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 1203.90 | 1199.69 | 1204.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 1209.20 | 1199.69 | 1204.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 1203.90 | 1200.53 | 1204.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:00:00 | 1203.90 | 1200.53 | 1204.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 1204.60 | 1201.35 | 1204.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 12:30:00 | 1204.50 | 1201.35 | 1204.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 13:15:00 | 1203.30 | 1201.74 | 1204.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:30:00 | 1198.50 | 1200.93 | 1203.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 09:30:00 | 1198.90 | 1198.55 | 1202.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 1196.00 | 1188.69 | 1188.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 1196.00 | 1188.69 | 1188.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 1196.00 | 1188.69 | 1188.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 1208.30 | 1194.88 | 1191.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 09:15:00 | 1197.50 | 1198.81 | 1195.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:45:00 | 1198.80 | 1198.81 | 1195.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1196.00 | 1198.25 | 1195.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 1196.10 | 1198.25 | 1195.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1202.70 | 1199.14 | 1195.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:15:00 | 1197.80 | 1199.14 | 1195.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1199.60 | 1199.23 | 1196.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:30:00 | 1196.00 | 1199.23 | 1196.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1197.90 | 1198.96 | 1196.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 14:00:00 | 1197.90 | 1198.96 | 1196.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1201.00 | 1199.54 | 1197.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:15:00 | 1186.90 | 1199.54 | 1197.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1185.80 | 1196.79 | 1196.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 1185.80 | 1196.79 | 1196.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1193.90 | 1196.21 | 1196.07 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 1194.10 | 1195.79 | 1195.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 14:15:00 | 1191.10 | 1194.26 | 1195.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1199.40 | 1194.61 | 1195.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 1199.40 | 1194.61 | 1195.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1199.40 | 1194.61 | 1195.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 1199.40 | 1194.61 | 1195.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1192.10 | 1194.11 | 1194.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:15:00 | 1188.80 | 1194.11 | 1194.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:00:00 | 1190.20 | 1192.65 | 1194.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 13:45:00 | 1189.30 | 1192.68 | 1193.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 1197.00 | 1194.72 | 1194.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 1197.00 | 1194.72 | 1194.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 15:15:00 | 1197.00 | 1194.72 | 1194.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 15:15:00 | 1197.00 | 1194.72 | 1194.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 1202.10 | 1196.66 | 1195.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1203.10 | 1205.03 | 1200.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:30:00 | 1205.00 | 1205.03 | 1200.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1197.60 | 1203.54 | 1200.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 1197.60 | 1203.54 | 1200.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 1197.90 | 1202.41 | 1200.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:30:00 | 1195.40 | 1202.41 | 1200.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1192.10 | 1200.35 | 1199.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 1192.10 | 1200.35 | 1199.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1198.80 | 1199.06 | 1198.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1224.00 | 1199.06 | 1198.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 11:15:00 | 1185.00 | 1196.43 | 1197.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 11:15:00 | 1185.00 | 1196.43 | 1197.85 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 1205.70 | 1195.67 | 1195.11 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 1189.90 | 1194.77 | 1194.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 14:15:00 | 1184.20 | 1190.92 | 1192.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1195.30 | 1190.70 | 1192.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1195.30 | 1190.70 | 1192.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1195.30 | 1190.70 | 1192.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 10:00:00 | 1195.30 | 1190.70 | 1192.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 1193.80 | 1191.32 | 1192.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:15:00 | 1188.60 | 1191.30 | 1192.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:00:00 | 1189.00 | 1188.41 | 1190.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 1199.40 | 1192.10 | 1191.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 13:15:00 | 1199.40 | 1192.10 | 1191.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 13:15:00 | 1199.40 | 1192.10 | 1191.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 15:15:00 | 1204.00 | 1194.75 | 1193.17 | Break + close above crossover candle high |

### Cycle 22 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1181.00 | 1192.00 | 1192.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 1165.00 | 1186.60 | 1189.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 09:15:00 | 1132.50 | 1127.29 | 1134.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 1132.50 | 1127.29 | 1134.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1132.50 | 1127.29 | 1134.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:45:00 | 1125.10 | 1127.29 | 1134.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 1135.50 | 1128.94 | 1134.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:00:00 | 1135.50 | 1128.94 | 1134.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 1140.00 | 1131.15 | 1134.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:30:00 | 1141.60 | 1131.15 | 1134.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1146.10 | 1134.14 | 1135.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 1145.30 | 1134.14 | 1135.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 14:15:00 | 1145.10 | 1138.16 | 1137.59 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 11:15:00 | 1131.90 | 1136.67 | 1137.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 1125.70 | 1134.00 | 1135.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 14:15:00 | 1122.20 | 1121.98 | 1127.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 14:15:00 | 1122.20 | 1121.98 | 1127.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1122.20 | 1121.98 | 1127.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 1123.70 | 1121.98 | 1127.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 15:15:00 | 1125.00 | 1122.58 | 1127.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 09:15:00 | 1130.00 | 1122.58 | 1127.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1121.80 | 1122.43 | 1126.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:00:00 | 1117.00 | 1125.86 | 1127.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 1116.60 | 1123.40 | 1125.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 12:15:00 | 1115.90 | 1122.36 | 1125.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:00:00 | 1116.10 | 1119.07 | 1122.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 1116.20 | 1117.20 | 1121.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 1129.80 | 1117.20 | 1121.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 1118.40 | 1117.44 | 1121.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 11:15:00 | 1111.30 | 1117.44 | 1121.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 1196.00 | 1129.13 | 1123.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 1196.00 | 1129.13 | 1123.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 1196.00 | 1129.13 | 1123.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 1196.00 | 1129.13 | 1123.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 1196.00 | 1129.13 | 1123.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 09:15:00 | 1196.00 | 1129.13 | 1123.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 1233.80 | 1182.68 | 1158.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 11:15:00 | 1220.10 | 1223.65 | 1199.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:30:00 | 1223.10 | 1223.65 | 1199.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1222.80 | 1228.31 | 1220.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 1222.80 | 1228.31 | 1220.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 1227.00 | 1227.39 | 1221.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:45:00 | 1227.20 | 1227.39 | 1221.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1231.70 | 1228.26 | 1222.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 1230.10 | 1228.26 | 1222.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1271.50 | 1272.75 | 1261.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:45:00 | 1276.90 | 1273.58 | 1262.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 11:30:00 | 1276.10 | 1274.44 | 1264.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 13:30:00 | 1277.80 | 1275.48 | 1266.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 1292.80 | 1275.77 | 1268.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1298.90 | 1292.02 | 1282.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 1287.40 | 1292.02 | 1282.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1292.80 | 1297.81 | 1291.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 1292.80 | 1297.81 | 1291.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1292.60 | 1296.77 | 1291.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:45:00 | 1292.30 | 1296.77 | 1291.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1292.60 | 1295.93 | 1291.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:15:00 | 1288.00 | 1295.93 | 1291.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1289.00 | 1294.55 | 1291.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:30:00 | 1289.00 | 1294.55 | 1291.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 14:15:00 | 1288.00 | 1293.24 | 1291.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 14:45:00 | 1284.30 | 1293.24 | 1291.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1261.40 | 1285.55 | 1287.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1261.40 | 1285.55 | 1287.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1261.40 | 1285.55 | 1287.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1261.40 | 1285.55 | 1287.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 09:15:00 | 1261.40 | 1285.55 | 1287.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 10:15:00 | 1258.70 | 1280.18 | 1285.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 1280.60 | 1270.86 | 1277.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 1280.60 | 1270.86 | 1277.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 1280.60 | 1270.86 | 1277.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 1280.60 | 1270.86 | 1277.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 1285.30 | 1273.75 | 1277.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:45:00 | 1285.70 | 1273.75 | 1277.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 13:15:00 | 1276.50 | 1275.43 | 1277.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 13:45:00 | 1277.70 | 1275.43 | 1277.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 14:15:00 | 1270.00 | 1274.34 | 1276.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 15:15:00 | 1264.10 | 1274.34 | 1276.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 09:45:00 | 1263.20 | 1270.01 | 1274.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:45:00 | 1263.50 | 1269.13 | 1273.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 12:30:00 | 1265.00 | 1268.44 | 1272.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1247.30 | 1257.44 | 1265.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:30:00 | 1244.90 | 1253.45 | 1262.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 10:00:00 | 1241.90 | 1243.00 | 1252.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 09:30:00 | 1245.90 | 1235.80 | 1243.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:45:00 | 1240.00 | 1237.58 | 1243.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 1250.50 | 1240.16 | 1243.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:00:00 | 1250.50 | 1240.16 | 1243.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 1257.00 | 1243.53 | 1245.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 1257.00 | 1243.53 | 1245.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 1266.10 | 1248.04 | 1247.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 1266.10 | 1248.04 | 1247.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 1266.10 | 1248.04 | 1247.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 1266.10 | 1248.04 | 1247.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 1266.10 | 1248.04 | 1247.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 1266.10 | 1248.04 | 1247.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 1266.10 | 1248.04 | 1247.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 13:15:00 | 1266.10 | 1248.04 | 1247.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 13:15:00 | 1266.10 | 1248.04 | 1247.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 14:15:00 | 1274.30 | 1253.30 | 1249.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1260.10 | 1261.73 | 1256.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 1260.10 | 1261.73 | 1256.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 10:15:00 | 1267.30 | 1262.53 | 1258.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 11:30:00 | 1270.30 | 1264.64 | 1259.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 15:15:00 | 1276.30 | 1268.02 | 1262.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1273.10 | 1267.74 | 1265.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 12:45:00 | 1272.40 | 1271.88 | 1268.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 1271.30 | 1272.15 | 1269.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:30:00 | 1287.00 | 1273.46 | 1270.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 10:45:00 | 1276.80 | 1274.54 | 1271.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 1258.40 | 1269.30 | 1269.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 1258.40 | 1269.30 | 1269.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 1258.40 | 1269.30 | 1269.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 1258.40 | 1269.30 | 1269.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 1258.40 | 1269.30 | 1269.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 14:15:00 | 1258.40 | 1269.30 | 1269.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 14:15:00 | 1258.40 | 1269.30 | 1269.57 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 12:15:00 | 1295.10 | 1273.30 | 1270.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 13:15:00 | 1302.20 | 1292.28 | 1283.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 1290.10 | 1291.85 | 1284.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 1290.10 | 1291.85 | 1284.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 1291.00 | 1292.34 | 1286.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:15:00 | 1300.80 | 1293.28 | 1287.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 14:00:00 | 1303.00 | 1296.25 | 1289.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 1301.10 | 1300.04 | 1294.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 12:15:00 | 1282.70 | 1296.57 | 1293.52 | SL hit (close<static) qty=1.00 sl=1285.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 12:15:00 | 1282.70 | 1296.57 | 1293.52 | SL hit (close<static) qty=1.00 sl=1285.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 12:15:00 | 1282.70 | 1296.57 | 1293.52 | SL hit (close<static) qty=1.00 sl=1285.50 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 1281.10 | 1291.45 | 1291.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 10:15:00 | 1278.90 | 1287.72 | 1289.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-19 14:15:00 | 1266.20 | 1265.85 | 1270.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-19 14:15:00 | 1266.20 | 1265.85 | 1270.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1266.20 | 1265.85 | 1270.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1266.20 | 1265.85 | 1270.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1260.00 | 1264.68 | 1269.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 1258.60 | 1264.68 | 1269.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1266.70 | 1265.09 | 1269.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:00:00 | 1254.60 | 1261.36 | 1266.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:45:00 | 1252.80 | 1257.77 | 1264.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:45:00 | 1252.70 | 1251.13 | 1256.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:30:00 | 1253.60 | 1256.06 | 1256.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 1260.60 | 1256.96 | 1256.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 1260.60 | 1256.96 | 1256.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 1260.60 | 1256.96 | 1256.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 14:15:00 | 1260.60 | 1256.96 | 1256.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 14:15:00 | 1260.60 | 1256.96 | 1256.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 09:15:00 | 1270.50 | 1260.16 | 1258.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 14:15:00 | 1253.90 | 1262.35 | 1260.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 14:15:00 | 1253.90 | 1262.35 | 1260.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1253.90 | 1262.35 | 1260.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:30:00 | 1257.10 | 1262.35 | 1260.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1246.00 | 1259.08 | 1259.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1245.10 | 1256.28 | 1258.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1255.90 | 1245.61 | 1250.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1255.90 | 1245.61 | 1250.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1255.90 | 1245.61 | 1250.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 1253.20 | 1245.61 | 1250.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1247.60 | 1246.01 | 1250.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 1244.60 | 1246.01 | 1250.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:30:00 | 1245.50 | 1241.16 | 1245.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 10:30:00 | 1244.70 | 1241.97 | 1245.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:45:00 | 1245.60 | 1243.04 | 1245.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1247.00 | 1243.83 | 1245.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 1247.00 | 1243.83 | 1245.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1254.50 | 1245.97 | 1246.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 1254.50 | 1245.97 | 1246.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 1253.00 | 1247.37 | 1246.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 1253.00 | 1247.37 | 1246.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 1253.00 | 1247.37 | 1246.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 1253.00 | 1247.37 | 1246.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 1253.00 | 1247.37 | 1246.81 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 11:15:00 | 1239.50 | 1245.26 | 1245.96 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1249.70 | 1246.36 | 1246.35 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 14:15:00 | 1241.80 | 1245.45 | 1245.94 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 1253.00 | 1246.32 | 1245.62 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 10:15:00 | 1238.10 | 1244.67 | 1244.94 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 12:15:00 | 1247.90 | 1245.44 | 1245.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 1253.80 | 1247.04 | 1246.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1257.20 | 1262.35 | 1256.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 1257.20 | 1262.35 | 1256.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1257.20 | 1262.35 | 1256.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 1257.90 | 1262.35 | 1256.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1253.80 | 1260.64 | 1255.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1253.80 | 1260.64 | 1255.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1258.10 | 1260.13 | 1256.17 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1243.80 | 1253.42 | 1253.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 1242.20 | 1251.18 | 1252.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 1254.10 | 1249.93 | 1251.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 11:15:00 | 1254.10 | 1249.93 | 1251.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1254.10 | 1249.93 | 1251.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1254.10 | 1249.93 | 1251.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1257.30 | 1251.40 | 1252.14 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 1258.00 | 1253.57 | 1253.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 1259.60 | 1254.77 | 1253.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 1252.00 | 1254.51 | 1253.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 13:15:00 | 1252.00 | 1254.51 | 1253.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 1252.00 | 1254.51 | 1253.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 1252.00 | 1254.51 | 1253.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 1249.60 | 1253.53 | 1253.52 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 15:15:00 | 1250.60 | 1252.94 | 1253.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 1242.00 | 1250.76 | 1252.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 13:15:00 | 1253.30 | 1248.70 | 1250.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 13:15:00 | 1253.30 | 1248.70 | 1250.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 1253.30 | 1248.70 | 1250.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:45:00 | 1251.80 | 1248.70 | 1250.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1261.40 | 1251.24 | 1251.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 1260.80 | 1251.24 | 1251.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-10-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 15:15:00 | 1260.00 | 1252.99 | 1252.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 1276.80 | 1257.76 | 1254.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 1265.00 | 1265.09 | 1260.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 09:15:00 | 1264.50 | 1265.09 | 1260.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 1262.50 | 1264.57 | 1260.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 1259.00 | 1264.57 | 1260.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1263.30 | 1264.32 | 1260.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 1262.50 | 1264.32 | 1260.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1253.00 | 1262.05 | 1259.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:00:00 | 1253.00 | 1262.05 | 1259.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1247.20 | 1259.08 | 1258.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 1248.50 | 1259.08 | 1258.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 13:15:00 | 1250.60 | 1257.39 | 1258.06 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1277.50 | 1259.69 | 1258.77 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1253.70 | 1264.52 | 1264.74 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 14:15:00 | 1271.40 | 1265.90 | 1265.35 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1249.90 | 1262.09 | 1263.67 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1279.90 | 1265.32 | 1264.68 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 15:15:00 | 1260.10 | 1263.97 | 1264.23 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1276.00 | 1266.35 | 1265.26 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 1257.20 | 1264.58 | 1265.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 1250.20 | 1261.47 | 1263.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 1245.90 | 1238.24 | 1245.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 1245.90 | 1238.24 | 1245.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 1245.90 | 1238.24 | 1245.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 1264.80 | 1238.24 | 1245.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1246.50 | 1239.89 | 1245.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 1247.00 | 1239.89 | 1245.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 1249.90 | 1241.89 | 1246.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:45:00 | 1252.90 | 1241.89 | 1246.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1267.50 | 1247.01 | 1248.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 1267.50 | 1247.01 | 1248.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 13:15:00 | 1261.40 | 1249.89 | 1249.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 15:15:00 | 1274.00 | 1257.91 | 1253.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 12:15:00 | 1258.50 | 1258.69 | 1255.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 12:45:00 | 1259.10 | 1258.69 | 1255.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 1259.40 | 1258.83 | 1255.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 14:45:00 | 1264.90 | 1257.85 | 1255.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 15:15:00 | 1265.00 | 1257.85 | 1255.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 10:15:00 | 1262.40 | 1259.42 | 1256.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 13:00:00 | 1261.10 | 1260.41 | 1257.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1263.00 | 1264.29 | 1261.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:30:00 | 1263.00 | 1264.29 | 1261.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 1260.90 | 1263.61 | 1261.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:45:00 | 1262.20 | 1263.61 | 1261.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 1253.60 | 1261.61 | 1260.44 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 1253.60 | 1261.61 | 1260.44 | SL hit (close<static) qty=1.00 sl=1254.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 1253.60 | 1261.61 | 1260.44 | SL hit (close<static) qty=1.00 sl=1254.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 1253.60 | 1261.61 | 1260.44 | SL hit (close<static) qty=1.00 sl=1254.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-31 13:15:00 | 1253.60 | 1261.61 | 1260.44 | SL hit (close<static) qty=1.00 sl=1254.30 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-10-31 14:00:00 | 1253.60 | 1261.61 | 1260.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 1258.30 | 1260.95 | 1260.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 1258.30 | 1260.95 | 1260.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1256.20 | 1260.00 | 1259.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 1291.00 | 1260.00 | 1259.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 1284.60 | 1294.65 | 1294.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 1284.60 | 1294.65 | 1294.84 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 1298.40 | 1295.40 | 1295.16 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 11:15:00 | 1287.10 | 1293.74 | 1294.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 1280.90 | 1291.17 | 1293.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 10:15:00 | 1262.00 | 1260.72 | 1270.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 10:45:00 | 1262.50 | 1260.72 | 1270.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 1261.40 | 1261.89 | 1268.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 15:15:00 | 1260.10 | 1261.89 | 1268.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:45:00 | 1260.20 | 1260.89 | 1266.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 1255.40 | 1252.48 | 1255.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 1197.09 | 1218.35 | 1223.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 10:15:00 | 1197.19 | 1218.35 | 1223.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-24 14:15:00 | 1192.63 | 1208.16 | 1216.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1220.60 | 1209.52 | 1215.47 | SL hit (close>ema200) qty=0.50 sl=1209.52 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1220.60 | 1209.52 | 1215.47 | SL hit (close>ema200) qty=0.50 sl=1209.52 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 1220.60 | 1209.52 | 1215.47 | SL hit (close>ema200) qty=0.50 sl=1209.52 alert=retest2 |

### Cycle 57 — BUY (started 2025-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 14:15:00 | 1225.50 | 1218.67 | 1218.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1229.60 | 1222.01 | 1219.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 12:15:00 | 1223.30 | 1225.54 | 1222.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 12:15:00 | 1223.30 | 1225.54 | 1222.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 1223.30 | 1225.54 | 1222.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 13:00:00 | 1223.30 | 1225.54 | 1222.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 13:15:00 | 1226.50 | 1225.73 | 1222.70 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 1211.70 | 1221.12 | 1221.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 14:15:00 | 1208.00 | 1217.02 | 1219.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 1186.70 | 1186.54 | 1193.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:45:00 | 1189.10 | 1186.54 | 1193.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1190.30 | 1188.33 | 1193.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 1171.90 | 1188.33 | 1193.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 1194.30 | 1190.38 | 1190.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 1194.30 | 1190.38 | 1190.27 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-12-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 12:15:00 | 1170.20 | 1186.34 | 1188.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 13:15:00 | 1163.50 | 1181.77 | 1186.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 1163.10 | 1162.71 | 1171.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 15:00:00 | 1163.10 | 1162.71 | 1171.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1166.90 | 1163.54 | 1171.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 1161.70 | 1162.00 | 1169.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 10:45:00 | 1155.50 | 1160.14 | 1168.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1171.90 | 1163.71 | 1162.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 1171.90 | 1163.71 | 1162.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1171.90 | 1163.71 | 1162.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 1197.80 | 1171.52 | 1166.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 1173.50 | 1174.39 | 1168.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 12:00:00 | 1173.50 | 1174.39 | 1168.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 1162.30 | 1171.97 | 1168.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:00:00 | 1162.30 | 1171.97 | 1168.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 1168.50 | 1171.28 | 1168.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 1173.30 | 1168.85 | 1167.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:45:00 | 1173.80 | 1169.38 | 1168.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:15:00 | 1177.90 | 1169.38 | 1168.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:15:00 | 1173.20 | 1171.87 | 1170.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1172.40 | 1172.22 | 1170.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 1172.40 | 1172.22 | 1170.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 1170.00 | 1171.78 | 1170.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 1170.00 | 1171.78 | 1170.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 1171.00 | 1171.62 | 1170.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:45:00 | 1171.30 | 1171.62 | 1170.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1175.80 | 1172.46 | 1171.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:30:00 | 1164.90 | 1172.46 | 1171.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1173.00 | 1172.57 | 1171.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:00:00 | 1173.00 | 1172.57 | 1171.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 1162.00 | 1170.45 | 1170.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 1162.00 | 1170.45 | 1170.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 1162.00 | 1170.45 | 1170.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 1162.00 | 1170.45 | 1170.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 15:15:00 | 1162.00 | 1170.45 | 1170.54 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 1176.60 | 1171.68 | 1171.09 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 1162.70 | 1171.19 | 1172.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 1159.80 | 1168.92 | 1170.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 1167.20 | 1165.04 | 1168.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 1167.20 | 1165.04 | 1168.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1167.20 | 1165.04 | 1168.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 1168.80 | 1165.04 | 1168.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1166.50 | 1165.33 | 1167.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1152.90 | 1164.04 | 1166.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 1167.00 | 1152.69 | 1151.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 1167.00 | 1152.69 | 1151.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 09:15:00 | 1180.00 | 1164.49 | 1158.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 15:15:00 | 1191.30 | 1195.44 | 1188.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:15:00 | 1193.40 | 1195.44 | 1188.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1192.80 | 1194.91 | 1188.92 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 1179.10 | 1186.30 | 1187.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 13:15:00 | 1171.00 | 1181.59 | 1184.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 1182.60 | 1177.02 | 1181.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 1182.60 | 1177.02 | 1181.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1182.60 | 1177.02 | 1181.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 1181.30 | 1177.02 | 1181.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1184.80 | 1178.58 | 1181.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 1184.30 | 1178.58 | 1181.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 1187.80 | 1180.42 | 1181.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 1189.00 | 1180.42 | 1181.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1196.90 | 1183.72 | 1183.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 1202.40 | 1187.45 | 1185.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 11:15:00 | 1188.70 | 1190.15 | 1187.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 11:15:00 | 1188.70 | 1190.15 | 1187.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1188.70 | 1190.15 | 1187.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:00:00 | 1188.70 | 1190.15 | 1187.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1200.00 | 1192.12 | 1188.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:30:00 | 1187.70 | 1192.12 | 1188.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 1199.60 | 1200.87 | 1196.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:00:00 | 1199.60 | 1200.87 | 1196.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1196.30 | 1199.96 | 1196.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:30:00 | 1194.30 | 1199.96 | 1196.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1199.20 | 1199.81 | 1196.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:30:00 | 1203.10 | 1200.48 | 1197.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 14:15:00 | 1201.00 | 1204.19 | 1200.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 1192.50 | 1198.74 | 1198.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 1192.50 | 1198.74 | 1198.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 1192.50 | 1198.74 | 1198.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 09:15:00 | 1188.50 | 1194.74 | 1196.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1115.00 | 1114.78 | 1132.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:30:00 | 1116.50 | 1114.78 | 1132.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1124.90 | 1118.15 | 1128.42 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 11:15:00 | 1145.00 | 1132.44 | 1130.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 13:15:00 | 1147.90 | 1137.40 | 1133.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 14:15:00 | 1135.70 | 1137.06 | 1133.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 15:00:00 | 1135.70 | 1137.06 | 1133.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1147.10 | 1139.25 | 1135.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 1151.20 | 1139.25 | 1135.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 11:30:00 | 1147.80 | 1141.57 | 1137.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 12:45:00 | 1147.90 | 1141.97 | 1137.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 13:45:00 | 1149.20 | 1142.60 | 1138.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1130.10 | 1141.89 | 1139.23 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1130.10 | 1141.89 | 1139.23 | SL hit (close<static) qty=1.00 sl=1133.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1130.10 | 1141.89 | 1139.23 | SL hit (close<static) qty=1.00 sl=1133.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1130.10 | 1141.89 | 1139.23 | SL hit (close<static) qty=1.00 sl=1133.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1130.10 | 1141.89 | 1139.23 | SL hit (close<static) qty=1.00 sl=1133.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 11:15:00 | 1116.90 | 1133.62 | 1135.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 1101.90 | 1118.72 | 1125.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1094.10 | 1083.31 | 1097.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1094.10 | 1083.31 | 1097.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1094.10 | 1083.31 | 1097.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 1093.70 | 1083.31 | 1097.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1098.80 | 1088.80 | 1095.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 1098.80 | 1088.80 | 1095.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1104.30 | 1091.90 | 1096.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:45:00 | 1107.40 | 1091.90 | 1096.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 1094.70 | 1094.98 | 1097.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 1097.30 | 1094.98 | 1097.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 12:15:00 | 1096.70 | 1095.17 | 1096.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 13:00:00 | 1096.70 | 1095.17 | 1096.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1099.00 | 1095.93 | 1096.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 1099.00 | 1095.93 | 1096.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 14:15:00 | 1107.80 | 1098.31 | 1097.96 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 09:15:00 | 1088.00 | 1096.53 | 1097.23 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 1113.90 | 1097.31 | 1095.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 15:15:00 | 1120.00 | 1106.75 | 1100.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 1097.80 | 1104.96 | 1100.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 1097.80 | 1104.96 | 1100.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 1097.80 | 1104.96 | 1100.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 1097.80 | 1104.96 | 1100.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 1092.00 | 1102.36 | 1099.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 1092.30 | 1102.36 | 1099.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1115.80 | 1105.54 | 1102.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 1118.60 | 1107.79 | 1103.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:00:00 | 1116.80 | 1107.79 | 1103.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 11:45:00 | 1118.80 | 1109.95 | 1104.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:30:00 | 1118.10 | 1135.98 | 1130.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1127.30 | 1130.17 | 1128.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1109.60 | 1130.17 | 1128.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1115.10 | 1127.16 | 1127.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1115.10 | 1127.16 | 1127.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1115.10 | 1127.16 | 1127.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 1115.10 | 1127.16 | 1127.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 09:15:00 | 1115.10 | 1127.16 | 1127.63 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-02-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 15:15:00 | 1134.50 | 1126.71 | 1126.28 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 1118.00 | 1124.96 | 1125.53 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2026-02-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 11:15:00 | 1131.40 | 1111.41 | 1111.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 14:15:00 | 1137.20 | 1127.94 | 1121.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 1118.60 | 1126.72 | 1122.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 1118.60 | 1126.72 | 1122.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1118.60 | 1126.72 | 1122.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 1116.30 | 1126.72 | 1122.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1122.50 | 1125.88 | 1122.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:45:00 | 1127.90 | 1126.22 | 1122.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1110.70 | 1126.00 | 1124.37 | SL hit (close<static) qty=1.00 sl=1118.60 alert=retest2 |

### Cycle 78 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1107.20 | 1122.24 | 1122.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 14:15:00 | 1104.30 | 1115.28 | 1119.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 12:15:00 | 1098.50 | 1097.34 | 1102.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:00:00 | 1098.50 | 1097.34 | 1102.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 13:15:00 | 1107.00 | 1099.27 | 1103.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 14:00:00 | 1107.00 | 1099.27 | 1103.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 1100.00 | 1099.41 | 1102.82 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 1116.40 | 1104.63 | 1104.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 13:15:00 | 1124.20 | 1111.92 | 1108.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 1113.20 | 1118.80 | 1115.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 1113.20 | 1118.80 | 1115.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 1113.20 | 1118.80 | 1115.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 1109.90 | 1118.80 | 1115.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 1108.20 | 1116.68 | 1114.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:00:00 | 1108.20 | 1116.68 | 1114.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2026-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 11:15:00 | 1099.90 | 1113.33 | 1113.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1095.00 | 1107.66 | 1110.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 1117.30 | 1108.24 | 1109.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 12:15:00 | 1117.30 | 1108.24 | 1109.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 1117.30 | 1108.24 | 1109.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:00:00 | 1117.30 | 1108.24 | 1109.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1120.10 | 1110.61 | 1110.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 1120.10 | 1110.61 | 1110.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 1133.20 | 1115.13 | 1112.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 1143.90 | 1124.70 | 1117.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 1189.20 | 1194.04 | 1177.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 13:00:00 | 1189.20 | 1194.04 | 1177.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 1195.00 | 1203.01 | 1192.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 13:45:00 | 1194.70 | 1203.01 | 1192.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 1206.10 | 1201.33 | 1194.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 10:45:00 | 1212.20 | 1203.56 | 1195.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 09:45:00 | 1221.80 | 1221.79 | 1210.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 1171.20 | 1211.67 | 1206.97 | SL hit (close<static) qty=1.00 sl=1186.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 1171.20 | 1211.67 | 1206.97 | SL hit (close<static) qty=1.00 sl=1186.50 alert=retest2 |

### Cycle 82 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 1168.90 | 1203.12 | 1203.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1136.70 | 1178.49 | 1190.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1132.40 | 1125.23 | 1142.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 1132.40 | 1125.23 | 1142.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 1127.70 | 1126.54 | 1140.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:30:00 | 1124.00 | 1124.49 | 1138.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 15:15:00 | 1125.00 | 1130.96 | 1137.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 1123.10 | 1114.76 | 1115.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 1145.00 | 1120.81 | 1117.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 1145.00 | 1120.81 | 1117.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 1145.00 | 1120.81 | 1117.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 1145.00 | 1120.81 | 1117.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 1148.30 | 1126.31 | 1120.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 1119.20 | 1134.49 | 1128.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 1119.20 | 1134.49 | 1128.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 1119.20 | 1134.49 | 1128.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:15:00 | 1117.60 | 1134.49 | 1128.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 1127.00 | 1132.99 | 1127.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:30:00 | 1117.90 | 1132.99 | 1127.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 1142.80 | 1134.96 | 1129.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 1147.50 | 1134.96 | 1129.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 13:15:00 | 1145.90 | 1135.60 | 1130.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 1147.00 | 1138.91 | 1132.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1121.20 | 1138.10 | 1133.52 | SL hit (close<static) qty=1.00 sl=1125.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1121.20 | 1138.10 | 1133.52 | SL hit (close<static) qty=1.00 sl=1125.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 09:15:00 | 1121.20 | 1138.10 | 1133.52 | SL hit (close<static) qty=1.00 sl=1125.00 alert=retest2 |

### Cycle 84 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 1111.50 | 1128.86 | 1129.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 1101.10 | 1123.31 | 1127.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1073.80 | 1063.95 | 1078.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 1073.80 | 1063.95 | 1078.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1073.80 | 1063.95 | 1078.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1045.30 | 1073.20 | 1073.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 1085.60 | 1051.49 | 1050.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1085.60 | 1051.49 | 1050.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 1088.10 | 1058.81 | 1053.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1071.00 | 1087.87 | 1073.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 1071.00 | 1087.87 | 1073.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 1071.00 | 1087.87 | 1073.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 1072.20 | 1087.87 | 1073.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1069.20 | 1084.13 | 1073.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1069.20 | 1084.13 | 1073.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1072.70 | 1081.85 | 1073.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:00:00 | 1076.00 | 1078.27 | 1073.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:30:00 | 1075.40 | 1077.68 | 1073.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 15:00:00 | 1075.30 | 1077.68 | 1073.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 15:15:00 | 1065.90 | 1075.32 | 1072.55 | SL hit (close<static) qty=1.00 sl=1066.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-27 15:15:00 | 1065.90 | 1075.32 | 1072.55 | SL hit (close<static) qty=1.00 sl=1066.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-27 15:15:00 | 1065.90 | 1075.32 | 1072.55 | SL hit (close<static) qty=1.00 sl=1066.70 alert=retest2 |

### Cycle 86 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1047.60 | 1069.78 | 1070.28 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 1087.80 | 1071.10 | 1069.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 1090.50 | 1074.98 | 1071.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 15:15:00 | 1084.00 | 1085.44 | 1078.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-02 09:15:00 | 1054.50 | 1085.44 | 1078.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 1058.10 | 1079.97 | 1076.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 09:30:00 | 1051.90 | 1079.97 | 1076.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 1060.10 | 1076.00 | 1074.93 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2026-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 11:15:00 | 1064.80 | 1073.76 | 1074.01 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 1094.10 | 1077.83 | 1075.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 1102.40 | 1082.74 | 1078.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 1082.40 | 1088.09 | 1082.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 1082.40 | 1088.09 | 1082.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1082.40 | 1088.09 | 1082.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1103.20 | 1088.09 | 1085.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 09:15:00 | 1213.52 | 1178.90 | 1165.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 1241.80 | 1252.11 | 1252.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 15:15:00 | 1236.70 | 1246.69 | 1249.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1251.50 | 1247.65 | 1249.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1251.50 | 1247.65 | 1249.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1251.50 | 1247.65 | 1249.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 1258.20 | 1247.65 | 1249.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1260.70 | 1250.26 | 1250.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1260.70 | 1250.26 | 1250.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 1255.90 | 1251.39 | 1251.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 1269.10 | 1254.93 | 1252.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 10:15:00 | 1257.20 | 1259.02 | 1256.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 10:15:00 | 1257.20 | 1259.02 | 1256.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1257.20 | 1259.02 | 1256.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:30:00 | 1254.00 | 1259.02 | 1256.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 1252.40 | 1257.70 | 1255.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:30:00 | 1255.60 | 1257.70 | 1255.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 1254.50 | 1257.06 | 1255.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:45:00 | 1254.60 | 1257.06 | 1255.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 1254.00 | 1256.45 | 1255.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 1254.00 | 1256.45 | 1255.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 1254.30 | 1256.02 | 1255.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:15:00 | 1253.00 | 1256.02 | 1255.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 1253.00 | 1255.41 | 1255.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 1267.70 | 1255.41 | 1255.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 12:45:00 | 1271.80 | 1258.71 | 1256.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1244.80 | 1257.46 | 1257.13 | SL hit (close<static) qty=1.00 sl=1250.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1244.80 | 1257.46 | 1257.13 | SL hit (close<static) qty=1.00 sl=1250.00 alert=retest2 |

### Cycle 92 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1246.60 | 1255.29 | 1256.17 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1281.60 | 1257.04 | 1255.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 10:15:00 | 1305.20 | 1266.67 | 1260.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 13:15:00 | 1267.50 | 1288.10 | 1279.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 13:15:00 | 1267.50 | 1288.10 | 1279.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 1267.50 | 1288.10 | 1279.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 14:00:00 | 1267.50 | 1288.10 | 1279.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 1264.70 | 1283.42 | 1278.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 15:00:00 | 1264.70 | 1283.42 | 1278.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 09:15:00 | 1249.00 | 1274.37 | 1274.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-06 10:15:00 | 1243.00 | 1268.10 | 1272.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 11:15:00 | 1283.30 | 1271.14 | 1273.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 11:15:00 | 1283.30 | 1271.14 | 1273.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1283.30 | 1271.14 | 1273.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:00:00 | 1283.30 | 1271.14 | 1273.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1280.00 | 1272.91 | 1273.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:30:00 | 1286.40 | 1272.91 | 1273.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2026-05-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 14:15:00 | 1284.40 | 1276.17 | 1275.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 09:15:00 | 1292.80 | 1279.96 | 1277.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 10:15:00 | 1291.80 | 1294.78 | 1287.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 10:15:00 | 1291.80 | 1294.78 | 1287.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1291.80 | 1294.78 | 1287.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 1291.80 | 1294.78 | 1287.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1281.20 | 1291.30 | 1287.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 1281.20 | 1291.30 | 1287.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1273.60 | 1287.76 | 1286.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:00:00 | 1273.60 | 1287.76 | 1286.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 1273.10 | 1284.83 | 1284.89 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-27 12:00:00 | 1132.15 | 2025-05-30 09:15:00 | 1109.15 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-05-28 14:30:00 | 1128.35 | 2025-05-30 09:15:00 | 1109.15 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1127.80 | 2025-05-30 09:15:00 | 1109.15 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-29 09:45:00 | 1129.10 | 2025-05-30 09:15:00 | 1109.15 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-05-30 09:15:00 | 1115.80 | 2025-05-30 09:15:00 | 1109.15 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-03 14:15:00 | 1179.50 | 2025-06-05 15:15:00 | 1151.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-06-04 14:30:00 | 1173.90 | 2025-06-05 15:15:00 | 1151.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-10 12:30:00 | 1172.80 | 2025-06-10 14:15:00 | 1167.10 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-06-11 09:30:00 | 1174.00 | 2025-06-12 13:15:00 | 1161.30 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-06-11 10:30:00 | 1174.50 | 2025-06-12 13:15:00 | 1161.30 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-11 14:00:00 | 1172.60 | 2025-06-12 13:15:00 | 1161.30 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-06-12 09:15:00 | 1189.00 | 2025-06-12 13:15:00 | 1161.30 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-06-30 15:00:00 | 1240.80 | 2025-07-01 10:15:00 | 1211.80 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-07-03 14:30:00 | 1198.50 | 2025-07-09 09:15:00 | 1196.00 | STOP_HIT | 1.00 | 0.21% |
| SELL | retest2 | 2025-07-04 09:30:00 | 1198.90 | 2025-07-09 09:15:00 | 1196.00 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-07-14 11:15:00 | 1188.80 | 2025-07-14 15:15:00 | 1197.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-14 13:00:00 | 1190.20 | 2025-07-14 15:15:00 | 1197.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-14 13:45:00 | 1189.30 | 2025-07-14 15:15:00 | 1197.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1224.00 | 2025-07-17 11:15:00 | 1185.00 | STOP_HIT | 1.00 | -3.19% |
| SELL | retest2 | 2025-07-22 12:15:00 | 1188.60 | 2025-07-23 13:15:00 | 1199.40 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-23 10:00:00 | 1189.00 | 2025-07-23 13:15:00 | 1199.40 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-08-06 10:00:00 | 1117.00 | 2025-08-08 09:15:00 | 1196.00 | STOP_HIT | 1.00 | -7.07% |
| SELL | retest2 | 2025-08-06 10:30:00 | 1116.60 | 2025-08-08 09:15:00 | 1196.00 | STOP_HIT | 1.00 | -7.11% |
| SELL | retest2 | 2025-08-06 12:15:00 | 1115.90 | 2025-08-08 09:15:00 | 1196.00 | STOP_HIT | 1.00 | -7.18% |
| SELL | retest2 | 2025-08-06 15:00:00 | 1116.10 | 2025-08-08 09:15:00 | 1196.00 | STOP_HIT | 1.00 | -7.16% |
| SELL | retest2 | 2025-08-07 11:15:00 | 1111.30 | 2025-08-08 09:15:00 | 1196.00 | STOP_HIT | 1.00 | -7.62% |
| BUY | retest2 | 2025-08-20 10:45:00 | 1276.90 | 2025-08-26 09:15:00 | 1261.40 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-08-20 11:30:00 | 1276.10 | 2025-08-26 09:15:00 | 1261.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-08-20 13:30:00 | 1277.80 | 2025-08-26 09:15:00 | 1261.40 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-08-21 09:15:00 | 1292.80 | 2025-08-26 09:15:00 | 1261.40 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2025-08-28 15:15:00 | 1264.10 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-29 09:45:00 | 1263.20 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-08-29 10:45:00 | 1263.50 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-08-29 12:30:00 | 1265.00 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-09-01 10:30:00 | 1244.90 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-09-02 10:00:00 | 1241.90 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-09-03 09:30:00 | 1245.90 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-09-03 10:45:00 | 1240.00 | 2025-09-03 13:15:00 | 1266.10 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-09-05 11:30:00 | 1270.30 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-05 15:15:00 | 1276.30 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-09-09 09:15:00 | 1273.10 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-09-09 12:45:00 | 1272.40 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-09-10 09:30:00 | 1287.00 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-09-10 10:45:00 | 1276.80 | 2025-09-10 14:15:00 | 1258.40 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-09-15 12:15:00 | 1300.80 | 2025-09-16 12:15:00 | 1282.70 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-09-15 14:00:00 | 1303.00 | 2025-09-16 12:15:00 | 1282.70 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-09-16 11:45:00 | 1301.10 | 2025-09-16 12:15:00 | 1282.70 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-09-22 13:00:00 | 1254.60 | 2025-09-24 14:15:00 | 1260.60 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-09-22 13:45:00 | 1252.80 | 2025-09-24 14:15:00 | 1260.60 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-23 14:45:00 | 1252.70 | 2025-09-24 14:15:00 | 1260.60 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-09-24 13:30:00 | 1253.60 | 2025-09-24 14:15:00 | 1260.60 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-09-29 11:15:00 | 1244.60 | 2025-09-30 15:15:00 | 1253.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-30 09:30:00 | 1245.50 | 2025-09-30 15:15:00 | 1253.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-09-30 10:30:00 | 1244.70 | 2025-09-30 15:15:00 | 1253.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-09-30 12:45:00 | 1245.60 | 2025-09-30 15:15:00 | 1253.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-29 14:45:00 | 1264.90 | 2025-10-31 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-29 15:15:00 | 1265.00 | 2025-10-31 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-10-30 10:15:00 | 1262.40 | 2025-10-31 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-30 13:00:00 | 1261.10 | 2025-10-31 13:15:00 | 1253.60 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-11-03 09:15:00 | 1291.00 | 2025-11-10 09:15:00 | 1284.60 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2025-11-12 15:15:00 | 1260.10 | 2025-11-24 10:15:00 | 1197.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 09:45:00 | 1260.20 | 2025-11-24 10:15:00 | 1197.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-14 14:45:00 | 1255.40 | 2025-11-24 14:15:00 | 1192.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 15:15:00 | 1260.10 | 2025-11-25 09:15:00 | 1220.60 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-11-13 09:45:00 | 1260.20 | 2025-11-25 09:15:00 | 1220.60 | STOP_HIT | 0.50 | 3.14% |
| SELL | retest2 | 2025-11-14 14:45:00 | 1255.40 | 2025-11-25 09:15:00 | 1220.60 | STOP_HIT | 0.50 | 2.77% |
| SELL | retest2 | 2025-12-03 10:15:00 | 1171.90 | 2025-12-04 11:15:00 | 1194.30 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-12-08 09:45:00 | 1161.70 | 2025-12-09 14:15:00 | 1171.90 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-08 10:45:00 | 1155.50 | 2025-12-09 14:15:00 | 1171.90 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-12-11 09:15:00 | 1173.30 | 2025-12-12 15:15:00 | 1162.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-12-11 09:45:00 | 1173.80 | 2025-12-12 15:15:00 | 1162.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-11 10:15:00 | 1177.90 | 2025-12-12 15:15:00 | 1162.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-12 09:15:00 | 1173.20 | 2025-12-12 15:15:00 | 1162.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1152.90 | 2025-12-22 10:15:00 | 1167.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2026-01-05 09:30:00 | 1203.10 | 2026-01-06 11:15:00 | 1192.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-05 14:15:00 | 1201.00 | 2026-01-06 11:15:00 | 1192.50 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2026-01-16 10:15:00 | 1151.20 | 2026-01-19 09:15:00 | 1130.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-01-16 11:30:00 | 1147.80 | 2026-01-19 09:15:00 | 1130.10 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-01-16 12:45:00 | 1147.90 | 2026-01-19 09:15:00 | 1130.10 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2026-01-16 13:45:00 | 1149.20 | 2026-01-19 09:15:00 | 1130.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-01-30 10:30:00 | 1118.60 | 2026-02-03 09:15:00 | 1115.10 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-01-30 11:00:00 | 1116.80 | 2026-02-03 09:15:00 | 1115.10 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-01-30 11:45:00 | 1118.80 | 2026-02-03 09:15:00 | 1115.10 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-02-02 11:30:00 | 1118.10 | 2026-02-03 09:15:00 | 1115.10 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2026-02-11 11:45:00 | 1127.90 | 2026-02-12 09:15:00 | 1110.70 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-02-27 10:45:00 | 1212.20 | 2026-03-02 10:15:00 | 1171.20 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2026-03-02 09:45:00 | 1221.80 | 2026-03-02 10:15:00 | 1171.20 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2026-03-06 10:30:00 | 1124.00 | 2026-03-11 10:15:00 | 1145.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-03-06 15:15:00 | 1125.00 | 2026-03-11 10:15:00 | 1145.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-11 10:15:00 | 1123.10 | 2026-03-11 10:15:00 | 1145.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-03-12 12:15:00 | 1147.50 | 2026-03-13 09:15:00 | 1121.20 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-03-12 13:15:00 | 1145.90 | 2026-03-13 09:15:00 | 1121.20 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-03-12 15:00:00 | 1147.00 | 2026-03-13 09:15:00 | 1121.20 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1045.30 | 2026-03-25 09:15:00 | 1085.60 | STOP_HIT | 1.00 | -3.86% |
| BUY | retest2 | 2026-03-27 14:00:00 | 1076.00 | 2026-03-27 15:15:00 | 1065.90 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-03-27 14:30:00 | 1075.40 | 2026-03-27 15:15:00 | 1065.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-03-27 15:00:00 | 1075.30 | 2026-03-27 15:15:00 | 1065.90 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1103.20 | 2026-04-15 09:15:00 | 1213.52 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-29 09:15:00 | 1267.70 | 2026-04-30 09:15:00 | 1244.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-04-29 12:45:00 | 1271.80 | 2026-04-30 09:15:00 | 1244.80 | STOP_HIT | 1.00 | -2.12% |
