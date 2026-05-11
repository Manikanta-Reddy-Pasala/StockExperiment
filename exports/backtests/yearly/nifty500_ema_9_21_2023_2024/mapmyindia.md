# C.E. Info Systems Ltd. (MAPMYINDIA)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 957.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 233 |
| ALERT1 | 148 |
| ALERT2 | 148 |
| ALERT2_SKIP | 91 |
| ALERT3 | 363 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 16 |
| ENTRY2 | 166 |
| PARTIAL | 17 |
| TARGET_HIT | 16 |
| STOP_HIT | 166 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 199 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 64 / 135
- **Target hits / Stop hits / Partials:** 16 / 166 / 17
- **Avg / median % per leg:** 0.54% / -1.04%
- **Sum % (uncompounded):** 108.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 91 | 28 | 30.8% | 10 | 80 | 1 | 0.54% | 49.4% |
| BUY @ 2nd Alert (retest1) | 7 | 2 | 28.6% | 0 | 6 | 1 | 0.49% | 3.4% |
| BUY @ 3rd Alert (retest2) | 84 | 26 | 31.0% | 10 | 74 | 0 | 0.55% | 46.0% |
| SELL (all) | 108 | 36 | 33.3% | 6 | 86 | 16 | 0.54% | 58.7% |
| SELL @ 2nd Alert (retest1) | 13 | 6 | 46.2% | 3 | 7 | 3 | 2.77% | 36.0% |
| SELL @ 3rd Alert (retest2) | 95 | 30 | 31.6% | 3 | 79 | 13 | 0.24% | 22.7% |
| retest1 (combined) | 20 | 8 | 40.0% | 3 | 13 | 4 | 1.97% | 39.4% |
| retest2 (combined) | 179 | 56 | 31.3% | 13 | 153 | 13 | 0.38% | 68.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-18 11:15:00 | 1076.10 | 1068.10 | 1067.78 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-19 09:15:00 | 1066.65 | 1067.90 | 1068.00 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 10:15:00 | 1080.55 | 1070.43 | 1069.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-19 11:15:00 | 1088.55 | 1074.05 | 1070.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-19 13:15:00 | 1074.80 | 1075.95 | 1072.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-19 14:00:00 | 1074.80 | 1075.95 | 1072.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 1070.70 | 1074.90 | 1072.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-19 14:45:00 | 1071.10 | 1074.90 | 1072.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 15:15:00 | 1073.00 | 1074.52 | 1072.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-22 09:15:00 | 1088.40 | 1074.52 | 1072.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 15:15:00 | 1066.95 | 1076.37 | 1075.37 | SL hit (close<static) qty=1.00 sl=1070.50 alert=retest2 |

### Cycle 4 — SELL (started 2023-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 12:15:00 | 1072.00 | 1075.10 | 1075.15 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 13:15:00 | 1076.50 | 1075.38 | 1075.28 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 14:15:00 | 1068.10 | 1073.93 | 1074.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 12:15:00 | 1058.20 | 1071.14 | 1073.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-25 09:15:00 | 1067.30 | 1067.25 | 1070.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-25 09:15:00 | 1067.30 | 1067.25 | 1070.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 1067.30 | 1067.25 | 1070.28 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 1093.25 | 1073.11 | 1071.72 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2023-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 12:15:00 | 1074.25 | 1081.98 | 1082.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 10:15:00 | 1070.65 | 1076.22 | 1079.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 14:15:00 | 1088.00 | 1075.77 | 1077.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 14:15:00 | 1088.00 | 1075.77 | 1077.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 1088.00 | 1075.77 | 1077.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 1088.00 | 1075.77 | 1077.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 1073.00 | 1075.21 | 1077.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:15:00 | 1084.35 | 1075.21 | 1077.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 1085.90 | 1077.35 | 1078.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 11:00:00 | 1079.05 | 1077.69 | 1078.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-01 11:15:00 | 1090.90 | 1080.33 | 1079.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 11:15:00 | 1090.90 | 1080.33 | 1079.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 12:15:00 | 1105.15 | 1085.30 | 1081.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-02 14:15:00 | 1102.60 | 1105.86 | 1097.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-02 15:00:00 | 1102.60 | 1105.86 | 1097.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-02 15:15:00 | 1091.80 | 1103.05 | 1096.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 09:15:00 | 1107.90 | 1103.05 | 1096.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-05 10:30:00 | 1106.00 | 1105.35 | 1098.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-16 12:15:00 | 1182.50 | 1190.04 | 1190.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2023-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-16 12:15:00 | 1182.50 | 1190.04 | 1190.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-16 13:15:00 | 1180.60 | 1188.15 | 1189.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-20 09:15:00 | 1177.05 | 1172.14 | 1178.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 1177.05 | 1172.14 | 1178.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 1177.05 | 1172.14 | 1178.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:00:00 | 1177.05 | 1172.14 | 1178.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 1176.00 | 1172.91 | 1178.07 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2023-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-21 09:15:00 | 1198.35 | 1181.36 | 1180.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-21 11:15:00 | 1201.00 | 1187.79 | 1183.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-22 11:15:00 | 1197.25 | 1205.48 | 1196.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-22 11:15:00 | 1197.25 | 1205.48 | 1196.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 1197.25 | 1205.48 | 1196.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:15:00 | 1188.50 | 1205.48 | 1196.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 1178.25 | 1200.04 | 1194.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:00:00 | 1178.25 | 1200.04 | 1194.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 1188.00 | 1197.63 | 1194.27 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2023-06-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 09:15:00 | 1176.95 | 1189.74 | 1191.19 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-23 11:15:00 | 1216.00 | 1195.67 | 1193.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-23 12:15:00 | 1228.00 | 1202.14 | 1196.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-26 15:15:00 | 1236.00 | 1237.58 | 1224.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-27 09:15:00 | 1237.95 | 1237.58 | 1224.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 12:15:00 | 1214.20 | 1231.24 | 1225.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 12:30:00 | 1215.50 | 1231.24 | 1225.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 13:15:00 | 1208.50 | 1226.69 | 1223.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 13:45:00 | 1205.00 | 1226.69 | 1223.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2023-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 15:15:00 | 1207.65 | 1219.73 | 1220.99 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-06-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 12:15:00 | 1229.90 | 1222.76 | 1221.93 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-06-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 10:15:00 | 1210.50 | 1221.55 | 1221.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 11:15:00 | 1204.50 | 1218.14 | 1220.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 11:15:00 | 1192.95 | 1190.39 | 1198.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-04 11:45:00 | 1188.05 | 1190.39 | 1198.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 1191.55 | 1189.50 | 1196.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:00:00 | 1191.55 | 1189.50 | 1196.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 1195.20 | 1190.64 | 1196.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 14:45:00 | 1195.65 | 1190.64 | 1196.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 15:15:00 | 1184.00 | 1189.31 | 1195.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-05 10:00:00 | 1196.00 | 1190.65 | 1195.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 10:15:00 | 1192.95 | 1191.11 | 1195.21 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2023-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 12:15:00 | 1212.45 | 1197.60 | 1197.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-05 13:15:00 | 1220.60 | 1202.20 | 1199.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 09:15:00 | 1230.00 | 1232.35 | 1222.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 09:15:00 | 1230.00 | 1232.35 | 1222.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 09:15:00 | 1230.00 | 1232.35 | 1222.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 09:45:00 | 1225.05 | 1232.35 | 1222.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 1220.35 | 1229.95 | 1222.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 1218.80 | 1229.95 | 1222.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 11:15:00 | 1218.30 | 1227.62 | 1221.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 11:45:00 | 1218.30 | 1227.62 | 1221.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 1213.00 | 1224.70 | 1220.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 13:00:00 | 1213.00 | 1224.70 | 1220.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 13:15:00 | 1217.45 | 1223.25 | 1220.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 13:30:00 | 1214.00 | 1223.25 | 1220.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 1223.50 | 1222.60 | 1220.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:15:00 | 1212.90 | 1222.60 | 1220.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 09:15:00 | 1207.55 | 1219.59 | 1219.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:45:00 | 1208.75 | 1219.59 | 1219.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 1230.15 | 1221.70 | 1220.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 11:45:00 | 1234.05 | 1223.36 | 1221.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 12:30:00 | 1233.60 | 1226.17 | 1222.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2023-07-13 09:15:00 | 1357.46 | 1327.23 | 1295.79 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2023-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 14:15:00 | 1481.05 | 1487.49 | 1488.14 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-31 09:15:00 | 1520.00 | 1493.59 | 1490.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-03 09:15:00 | 1534.55 | 1512.54 | 1509.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 14:15:00 | 1540.00 | 1541.02 | 1532.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-04 15:00:00 | 1540.00 | 1541.02 | 1532.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 15:15:00 | 1539.00 | 1540.62 | 1532.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-07 09:15:00 | 1611.75 | 1540.62 | 1532.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-10 15:15:00 | 1583.00 | 1587.91 | 1588.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2023-08-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-10 15:15:00 | 1583.00 | 1587.91 | 1588.18 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-11 09:15:00 | 1633.05 | 1596.94 | 1592.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-11 10:15:00 | 1670.75 | 1611.70 | 1599.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-14 10:15:00 | 1665.80 | 1666.57 | 1639.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-14 11:00:00 | 1665.80 | 1666.57 | 1639.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 09:15:00 | 1658.95 | 1659.83 | 1647.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-16 09:30:00 | 1629.95 | 1659.83 | 1647.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 1660.00 | 1659.86 | 1648.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 09:30:00 | 1670.15 | 1660.67 | 1653.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-17 10:00:00 | 1670.00 | 1660.67 | 1653.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-18 11:00:00 | 1670.70 | 1660.77 | 1657.00 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 14:15:00 | 1652.90 | 1656.12 | 1656.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-21 14:15:00 | 1652.90 | 1656.12 | 1656.44 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2023-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 10:15:00 | 1663.85 | 1657.65 | 1657.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 11:15:00 | 1666.45 | 1659.41 | 1657.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 14:15:00 | 1659.00 | 1660.63 | 1658.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-22 14:15:00 | 1659.00 | 1660.63 | 1658.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 14:15:00 | 1659.00 | 1660.63 | 1658.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-22 14:30:00 | 1660.00 | 1660.63 | 1658.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 15:15:00 | 1650.60 | 1658.63 | 1658.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 09:15:00 | 1662.35 | 1658.63 | 1658.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 10:15:00 | 1659.95 | 1658.57 | 1658.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-23 13:30:00 | 1661.65 | 1660.72 | 1659.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:15:00 | 1717.40 | 1659.29 | 1659.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 14:15:00 | 1738.25 | 1757.51 | 1750.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-29 15:00:00 | 1738.25 | 1757.51 | 1750.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 15:15:00 | 1738.00 | 1753.60 | 1749.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 09:15:00 | 1752.70 | 1753.60 | 1749.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-30 10:15:00 | 1749.50 | 1751.48 | 1748.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-30 12:15:00 | 1734.60 | 1744.88 | 1746.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 12:15:00 | 1734.60 | 1744.88 | 1746.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 14:15:00 | 1725.00 | 1739.27 | 1743.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-31 09:15:00 | 1753.45 | 1738.54 | 1741.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-31 09:15:00 | 1753.45 | 1738.54 | 1741.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 1753.45 | 1738.54 | 1741.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:00:00 | 1753.45 | 1738.54 | 1741.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 1758.60 | 1742.55 | 1743.48 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 11:15:00 | 1758.20 | 1745.68 | 1744.82 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2023-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 15:15:00 | 1720.30 | 1742.62 | 1744.59 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 10:15:00 | 1754.95 | 1745.76 | 1745.72 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2023-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 12:15:00 | 1744.25 | 1745.60 | 1745.66 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2023-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 13:15:00 | 1749.50 | 1746.38 | 1746.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 14:15:00 | 1750.05 | 1747.11 | 1746.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-05 12:15:00 | 1757.40 | 1759.77 | 1754.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 12:15:00 | 1757.40 | 1759.77 | 1754.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 12:15:00 | 1757.40 | 1759.77 | 1754.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-05 12:45:00 | 1756.00 | 1759.77 | 1754.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 09:15:00 | 1762.60 | 1762.22 | 1757.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 10:15:00 | 1750.00 | 1762.22 | 1757.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 10:15:00 | 1744.10 | 1758.60 | 1756.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:00:00 | 1744.10 | 1758.60 | 1756.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 11:15:00 | 1749.95 | 1756.87 | 1755.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 11:30:00 | 1744.15 | 1756.87 | 1755.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 12:15:00 | 1753.15 | 1756.13 | 1755.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-06 12:30:00 | 1751.25 | 1756.13 | 1755.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 13:15:00 | 1762.00 | 1757.30 | 1755.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 15:15:00 | 1768.60 | 1758.11 | 1756.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-07 09:15:00 | 1748.40 | 1757.85 | 1756.64 | SL hit (close<static) qty=1.00 sl=1752.85 alert=retest2 |

### Cycle 30 — SELL (started 2023-09-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-07 10:15:00 | 1746.40 | 1755.56 | 1755.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-07 12:15:00 | 1743.00 | 1752.16 | 1754.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-11 13:15:00 | 1735.10 | 1730.08 | 1736.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-11 13:15:00 | 1735.10 | 1730.08 | 1736.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 13:15:00 | 1735.10 | 1730.08 | 1736.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 14:00:00 | 1735.10 | 1730.08 | 1736.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 14:15:00 | 1738.20 | 1731.70 | 1736.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-11 14:30:00 | 1748.15 | 1731.70 | 1736.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 15:15:00 | 1753.00 | 1735.96 | 1738.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:15:00 | 1756.35 | 1735.96 | 1738.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 1673.75 | 1723.52 | 1732.44 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 1728.95 | 1698.57 | 1696.55 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2023-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 12:15:00 | 1699.95 | 1712.00 | 1712.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 14:15:00 | 1689.95 | 1705.42 | 1709.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 09:15:00 | 1715.95 | 1705.93 | 1708.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 09:15:00 | 1715.95 | 1705.93 | 1708.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 09:15:00 | 1715.95 | 1705.93 | 1708.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:00:00 | 1715.95 | 1705.93 | 1708.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 1717.05 | 1708.15 | 1709.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:45:00 | 1717.35 | 1708.15 | 1709.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 11:15:00 | 1722.55 | 1711.03 | 1710.63 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-09-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 14:15:00 | 1688.60 | 1708.81 | 1711.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 14:15:00 | 1684.15 | 1697.90 | 1703.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-25 10:15:00 | 1711.75 | 1697.31 | 1701.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 10:15:00 | 1711.75 | 1697.31 | 1701.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 10:15:00 | 1711.75 | 1697.31 | 1701.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 10:30:00 | 1712.45 | 1697.31 | 1701.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 11:15:00 | 1725.00 | 1702.85 | 1703.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-25 11:45:00 | 1718.00 | 1702.85 | 1703.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2023-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 12:15:00 | 1723.20 | 1706.92 | 1705.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-26 09:15:00 | 1753.55 | 1718.36 | 1711.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 11:15:00 | 1997.80 | 1998.40 | 1935.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 1965.00 | 1991.33 | 1956.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 1965.00 | 1991.33 | 1956.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 10:30:00 | 2024.85 | 2009.62 | 1967.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-10-04 13:15:00 | 2227.34 | 2157.66 | 2087.16 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 11:15:00 | 2128.80 | 2174.28 | 2177.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-10 12:15:00 | 2090.65 | 2157.56 | 2169.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-16 09:15:00 | 2041.00 | 2040.85 | 2066.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 11:00:00 | 2020.00 | 2036.68 | 2061.85 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-16 15:15:00 | 2014.00 | 2030.69 | 2050.77 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 2015.80 | 2025.04 | 2044.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:00:00 | 2015.80 | 2025.04 | 2044.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-18 09:15:00 | 2046.45 | 2011.62 | 2025.35 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-18 09:15:00 | 2046.45 | 2011.62 | 2025.35 | SL hit (close>ema400) qty=1.00 sl=2025.35 alert=retest1 |

### Cycle 37 — BUY (started 2023-10-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-26 14:15:00 | 1993.35 | 1952.88 | 1952.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 09:15:00 | 2003.15 | 1969.92 | 1960.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-27 15:15:00 | 2000.00 | 2001.70 | 1983.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-30 09:15:00 | 1995.20 | 2001.70 | 1983.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 2045.00 | 2010.36 | 1989.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-30 10:30:00 | 2052.05 | 2023.53 | 1997.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-06 11:15:00 | 2100.45 | 2107.63 | 2108.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-06 11:15:00 | 2100.45 | 2107.63 | 2108.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-07 11:15:00 | 2085.00 | 2099.57 | 2103.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-08 09:15:00 | 2092.90 | 2076.36 | 2088.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-08 09:15:00 | 2092.90 | 2076.36 | 2088.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 09:15:00 | 2092.90 | 2076.36 | 2088.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-08 09:30:00 | 2099.95 | 2076.36 | 2088.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 10:15:00 | 2092.45 | 2079.58 | 2088.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-08 15:15:00 | 2085.00 | 2088.05 | 2090.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-09 11:15:00 | 2097.65 | 2092.86 | 2092.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2023-11-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-09 11:15:00 | 2097.65 | 2092.86 | 2092.25 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2023-11-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 12:15:00 | 2077.30 | 2089.75 | 2090.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 15:15:00 | 2061.00 | 2080.84 | 2086.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-10 09:15:00 | 2086.05 | 2081.88 | 2086.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-10 09:15:00 | 2086.05 | 2081.88 | 2086.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 2086.05 | 2081.88 | 2086.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:45:00 | 2073.45 | 2081.88 | 2086.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 10:15:00 | 2091.90 | 2083.88 | 2086.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-10 10:30:00 | 2096.00 | 2083.88 | 2086.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 11:15:00 | 2075.40 | 2082.19 | 2085.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-10 14:00:00 | 2067.40 | 2077.83 | 2083.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-12 18:15:00 | 2106.80 | 2078.45 | 2081.58 | SL hit (close>static) qty=1.00 sl=2091.95 alert=retest2 |

### Cycle 41 — BUY (started 2023-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-13 09:15:00 | 2146.50 | 2092.06 | 2087.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 09:15:00 | 2230.00 | 2144.23 | 2124.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-17 12:15:00 | 2247.50 | 2249.52 | 2208.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-17 13:00:00 | 2247.50 | 2249.52 | 2208.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 09:15:00 | 2216.50 | 2241.01 | 2217.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 09:30:00 | 2212.15 | 2241.01 | 2217.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 10:15:00 | 2205.75 | 2233.96 | 2216.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-20 11:00:00 | 2205.75 | 2233.96 | 2216.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-20 11:15:00 | 2200.80 | 2227.33 | 2215.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-20 14:15:00 | 2226.00 | 2216.15 | 2211.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-22 11:15:00 | 2192.00 | 2212.77 | 2215.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2023-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-22 11:15:00 | 2192.00 | 2212.77 | 2215.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-22 12:15:00 | 2164.30 | 2203.08 | 2210.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-23 09:15:00 | 2254.35 | 2203.00 | 2206.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-23 09:15:00 | 2254.35 | 2203.00 | 2206.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-23 09:15:00 | 2254.35 | 2203.00 | 2206.91 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2023-11-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 10:15:00 | 2255.55 | 2213.51 | 2211.34 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2023-11-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 11:15:00 | 2163.60 | 2203.53 | 2207.00 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2023-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 11:15:00 | 2217.45 | 2205.88 | 2205.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 13:15:00 | 2220.60 | 2210.76 | 2208.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-24 14:15:00 | 2199.50 | 2208.51 | 2207.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 14:15:00 | 2199.50 | 2208.51 | 2207.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 14:15:00 | 2199.50 | 2208.51 | 2207.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-24 15:00:00 | 2199.50 | 2208.51 | 2207.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 15:15:00 | 2206.05 | 2208.02 | 2207.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 09:15:00 | 2221.85 | 2208.02 | 2207.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-28 10:00:00 | 2212.70 | 2208.96 | 2207.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-28 12:15:00 | 2187.50 | 2204.64 | 2206.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2023-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 12:15:00 | 2187.50 | 2204.64 | 2206.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 13:15:00 | 2179.35 | 2199.58 | 2203.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 09:15:00 | 2195.45 | 2195.45 | 2200.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 2195.45 | 2195.45 | 2200.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 2195.45 | 2195.45 | 2200.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:30:00 | 2196.00 | 2195.45 | 2200.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 2188.00 | 2193.96 | 2199.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 11:30:00 | 2177.00 | 2188.77 | 2196.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-30 14:15:00 | 2222.50 | 2195.15 | 2191.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2023-11-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 14:15:00 | 2222.50 | 2195.15 | 2191.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 2249.95 | 2211.13 | 2199.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-01 14:15:00 | 2206.10 | 2227.98 | 2214.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-01 14:15:00 | 2206.10 | 2227.98 | 2214.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 14:15:00 | 2206.10 | 2227.98 | 2214.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-01 15:00:00 | 2206.10 | 2227.98 | 2214.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 15:15:00 | 2206.00 | 2223.58 | 2213.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 2226.55 | 2223.58 | 2213.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 11:00:00 | 2234.95 | 2225.72 | 2216.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 09:45:00 | 2219.75 | 2221.13 | 2217.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-05 11:15:00 | 2194.65 | 2215.65 | 2215.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2023-12-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-05 11:15:00 | 2194.65 | 2215.65 | 2215.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-05 13:15:00 | 2179.95 | 2203.76 | 2210.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-06 11:15:00 | 2208.75 | 2195.54 | 2202.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-06 11:15:00 | 2208.75 | 2195.54 | 2202.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 11:15:00 | 2208.75 | 2195.54 | 2202.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-06 12:00:00 | 2208.75 | 2195.54 | 2202.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-06 12:15:00 | 2183.60 | 2193.15 | 2200.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-06 13:15:00 | 2179.30 | 2193.15 | 2200.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 14:15:00 | 2215.55 | 2197.46 | 2201.23 | SL hit (close>static) qty=1.00 sl=2209.65 alert=retest2 |

### Cycle 49 — BUY (started 2023-12-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 09:15:00 | 2224.30 | 2206.12 | 2204.72 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2023-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 12:15:00 | 2191.90 | 2208.48 | 2208.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 13:15:00 | 2162.40 | 2199.27 | 2204.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-11 14:15:00 | 2187.90 | 2180.25 | 2189.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-11 14:15:00 | 2187.90 | 2180.25 | 2189.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 14:15:00 | 2187.90 | 2180.25 | 2189.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 15:00:00 | 2187.90 | 2180.25 | 2189.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 15:15:00 | 2198.00 | 2183.80 | 2189.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:15:00 | 2198.25 | 2183.80 | 2189.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 2177.10 | 2182.46 | 2188.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 10:15:00 | 2173.45 | 2182.46 | 2188.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-12 12:45:00 | 2173.65 | 2181.72 | 2186.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 13:15:00 | 2064.78 | 2091.86 | 2107.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2023-12-20 13:15:00 | 2064.97 | 2091.86 | 2107.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-12-21 14:15:00 | 2034.05 | 2021.79 | 2054.05 | SL hit (close>ema200) qty=0.50 sl=2021.79 alert=retest2 |

### Cycle 51 — BUY (started 2024-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 14:15:00 | 1970.75 | 1950.69 | 1948.66 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 12:15:00 | 1937.95 | 1947.85 | 1948.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 14:15:00 | 1921.60 | 1942.15 | 1945.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 1953.85 | 1941.75 | 1944.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 1953.85 | 1941.75 | 1944.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 1953.85 | 1941.75 | 1944.86 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 11:15:00 | 1962.95 | 1947.47 | 1947.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 12:15:00 | 1972.70 | 1952.52 | 1949.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 1951.95 | 1961.51 | 1955.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-10 09:15:00 | 1951.95 | 1961.51 | 1955.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 09:15:00 | 1951.95 | 1961.51 | 1955.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 09:30:00 | 1954.70 | 1961.51 | 1955.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 1954.90 | 1960.18 | 1955.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 10:45:00 | 1946.20 | 1960.18 | 1955.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 11:15:00 | 1966.75 | 1961.50 | 1956.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 11:30:00 | 1968.75 | 1961.50 | 1956.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-11 09:15:00 | 2006.10 | 1974.16 | 1964.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 10:30:00 | 2017.35 | 1980.73 | 1968.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 12:30:00 | 2011.05 | 1988.91 | 1974.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-11 15:15:00 | 2039.35 | 1994.73 | 1979.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 14:45:00 | 2040.00 | 2050.26 | 2040.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 15:15:00 | 2039.00 | 2048.01 | 2040.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 09:15:00 | 2002.95 | 2048.01 | 2040.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 1996.85 | 2037.78 | 2036.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-17 10:00:00 | 1996.85 | 2037.78 | 2036.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-01-17 10:15:00 | 2000.05 | 2030.23 | 2032.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2024-01-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 10:15:00 | 2000.05 | 2030.23 | 2032.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 11:15:00 | 1989.80 | 2022.15 | 2028.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 1995.00 | 1979.87 | 1993.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-19 09:15:00 | 1995.00 | 1979.87 | 1993.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-19 09:15:00 | 1995.00 | 1979.87 | 1993.89 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2024-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 14:15:00 | 2010.05 | 2000.96 | 2000.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-20 09:15:00 | 2025.00 | 2006.49 | 2003.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-24 09:15:00 | 2040.00 | 2043.68 | 2033.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 09:15:00 | 2040.00 | 2043.68 | 2033.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 2040.00 | 2043.68 | 2033.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 10:30:00 | 2055.00 | 2044.09 | 2035.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 12:30:00 | 2050.00 | 2046.05 | 2037.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-25 10:15:00 | 2003.80 | 2037.64 | 2037.36 | SL hit (close<static) qty=1.00 sl=2010.50 alert=retest2 |

### Cycle 56 — SELL (started 2024-01-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-25 11:15:00 | 2004.15 | 2030.94 | 2034.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-25 12:15:00 | 1995.30 | 2023.82 | 2030.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 15:15:00 | 2023.00 | 2019.73 | 2026.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 15:15:00 | 2023.00 | 2019.73 | 2026.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 15:15:00 | 2023.00 | 2019.73 | 2026.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-29 09:15:00 | 2017.75 | 2019.73 | 2026.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 09:15:00 | 1994.00 | 2014.59 | 2023.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-29 10:15:00 | 1991.20 | 2014.59 | 2023.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-02-01 09:15:00 | 1891.64 | 1928.54 | 1948.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-02-02 09:15:00 | 1792.08 | 1855.39 | 1895.00 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2024-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-06 15:15:00 | 1860.00 | 1852.26 | 1851.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 09:15:00 | 1887.35 | 1859.28 | 1855.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-08 09:15:00 | 1881.65 | 1891.31 | 1877.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-08 09:15:00 | 1881.65 | 1891.31 | 1877.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 09:15:00 | 1881.65 | 1891.31 | 1877.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-08 10:00:00 | 1881.65 | 1891.31 | 1877.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 10:15:00 | 1900.00 | 1893.05 | 1879.28 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 09:15:00 | 1835.70 | 1875.55 | 1880.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 15:15:00 | 1825.00 | 1847.71 | 1862.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 11:15:00 | 1826.05 | 1826.04 | 1838.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-14 12:00:00 | 1826.05 | 1826.04 | 1838.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 12:15:00 | 1850.60 | 1830.96 | 1839.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 13:00:00 | 1850.60 | 1830.96 | 1839.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-14 13:15:00 | 1854.55 | 1835.67 | 1840.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 14:00:00 | 1854.55 | 1835.67 | 1840.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2024-02-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 15:15:00 | 1864.00 | 1846.68 | 1845.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 09:15:00 | 1900.10 | 1857.36 | 1850.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 15:15:00 | 1875.50 | 1875.52 | 1864.03 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-16 09:15:00 | 1900.75 | 1875.52 | 1864.03 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 15:15:00 | 1866.00 | 1873.37 | 1868.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-16 15:15:00 | 1866.00 | 1873.37 | 1868.69 | SL hit (close<ema400) qty=1.00 sl=1868.69 alert=retest1 |

### Cycle 60 — SELL (started 2024-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-19 14:15:00 | 1856.20 | 1864.64 | 1865.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-20 12:15:00 | 1846.90 | 1858.33 | 1862.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 09:15:00 | 1852.20 | 1851.66 | 1857.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 09:15:00 | 1852.20 | 1851.66 | 1857.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 1852.20 | 1851.66 | 1857.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 09:30:00 | 1865.00 | 1851.66 | 1857.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 1825.00 | 1817.98 | 1828.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-23 09:15:00 | 1836.00 | 1817.98 | 1828.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 1850.00 | 1824.38 | 1830.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 09:45:00 | 1816.95 | 1822.95 | 1827.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 13:15:00 | 1816.80 | 1807.68 | 1813.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-27 15:00:00 | 1819.65 | 1813.24 | 1815.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 09:15:00 | 1837.25 | 1818.33 | 1817.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-28 09:15:00 | 1837.25 | 1818.33 | 1817.09 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2024-02-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 10:15:00 | 1805.30 | 1815.72 | 1816.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 1802.60 | 1812.84 | 1814.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 1817.00 | 1807.30 | 1810.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 10:15:00 | 1817.00 | 1807.30 | 1810.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 10:15:00 | 1817.00 | 1807.30 | 1810.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 11:00:00 | 1817.00 | 1807.30 | 1810.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 1824.35 | 1810.71 | 1811.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 11:45:00 | 1822.00 | 1810.71 | 1811.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 1816.00 | 1811.77 | 1812.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 12:45:00 | 1816.95 | 1811.77 | 1812.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 13:15:00 | 1811.60 | 1811.73 | 1812.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 13:30:00 | 1817.00 | 1811.73 | 1812.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 1810.05 | 1811.40 | 1811.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:15:00 | 1819.00 | 1811.40 | 1811.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 15:15:00 | 1819.00 | 1812.92 | 1812.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 09:15:00 | 1861.55 | 1822.64 | 1817.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-01 14:15:00 | 1824.00 | 1829.37 | 1823.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-01 14:15:00 | 1824.00 | 1829.37 | 1823.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 14:15:00 | 1824.00 | 1829.37 | 1823.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-01 15:00:00 | 1824.00 | 1829.37 | 1823.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 15:15:00 | 1835.00 | 1830.49 | 1824.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-02 09:15:00 | 1850.00 | 1830.49 | 1824.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 10:45:00 | 1845.00 | 1833.57 | 1828.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-05 09:15:00 | 1809.00 | 1826.23 | 1827.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2024-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 09:15:00 | 1809.00 | 1826.23 | 1827.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 14:15:00 | 1799.95 | 1814.53 | 1820.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 15:15:00 | 1815.00 | 1814.63 | 1820.06 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-06 09:30:00 | 1789.80 | 1802.10 | 1813.87 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-11 09:15:00 | 1700.31 | 1740.26 | 1762.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-03-13 15:15:00 | 1610.82 | 1653.79 | 1680.49 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 65 — BUY (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 12:15:00 | 1739.00 | 1682.29 | 1680.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 15:15:00 | 1744.95 | 1724.18 | 1709.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 15:15:00 | 1730.00 | 1732.52 | 1721.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-03-20 09:15:00 | 1747.80 | 1732.52 | 1721.56 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-22 09:15:00 | 1835.19 | 1804.94 | 1780.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 1851.80 | 1843.62 | 1816.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-26 09:30:00 | 1820.75 | 1843.62 | 1816.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-03-26 10:15:00 | 1841.20 | 1843.13 | 1818.82 | SL hit (close<ema200) qty=0.50 sl=1843.13 alert=retest1 |

### Cycle 66 — SELL (started 2024-04-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 13:15:00 | 1865.80 | 1892.57 | 1895.39 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2024-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 15:15:00 | 1910.00 | 1898.29 | 1897.63 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-09 11:15:00 | 1885.50 | 1895.23 | 1896.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 1876.05 | 1891.40 | 1894.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-15 13:15:00 | 1835.45 | 1828.34 | 1842.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-15 14:00:00 | 1835.45 | 1828.34 | 1842.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 1840.00 | 1829.18 | 1839.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 09:45:00 | 1840.00 | 1829.18 | 1839.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 1847.10 | 1832.76 | 1840.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:00:00 | 1847.10 | 1832.76 | 1840.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 11:15:00 | 1853.05 | 1836.82 | 1841.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 11:45:00 | 1859.00 | 1836.82 | 1841.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 1849.30 | 1841.77 | 1842.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 15:15:00 | 1859.95 | 1841.77 | 1842.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 15:15:00 | 1859.95 | 1845.40 | 1844.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 09:15:00 | 1894.05 | 1855.13 | 1848.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-19 10:15:00 | 1914.30 | 1918.43 | 1893.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-19 10:45:00 | 1918.85 | 1918.43 | 1893.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 09:15:00 | 1900.25 | 1914.00 | 1902.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-22 10:00:00 | 1900.25 | 1914.00 | 1902.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 10:15:00 | 1904.35 | 1912.07 | 1902.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 13:30:00 | 1917.15 | 1914.56 | 1905.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 14:45:00 | 1915.85 | 1911.82 | 1905.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 15:15:00 | 1926.00 | 1911.82 | 1905.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-02 09:15:00 | 1993.45 | 2007.88 | 2009.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2024-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 09:15:00 | 1993.45 | 2007.88 | 2009.73 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 11:15:00 | 2046.00 | 2013.56 | 2011.87 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2024-05-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 12:15:00 | 1980.90 | 2012.31 | 2015.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 1936.40 | 1988.12 | 2002.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-07 13:15:00 | 1926.05 | 1923.52 | 1950.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-07 14:00:00 | 1926.05 | 1923.52 | 1950.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 14:15:00 | 1901.00 | 1809.57 | 1824.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-13 14:45:00 | 1928.30 | 1809.57 | 1824.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-13 15:15:00 | 1900.00 | 1827.66 | 1831.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:15:00 | 1995.75 | 1827.66 | 1831.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 09:15:00 | 2009.00 | 1863.93 | 1847.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-18 11:15:00 | 2037.00 | 2006.99 | 1993.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 1996.40 | 2008.75 | 1996.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 1996.40 | 2008.75 | 1996.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1996.40 | 2008.75 | 1996.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 12:30:00 | 2020.00 | 2006.10 | 2001.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 10:15:00 | 1975.00 | 1995.57 | 1997.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 1975.00 | 1995.57 | 1997.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 12:15:00 | 1964.50 | 1976.85 | 1984.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 1979.85 | 1975.12 | 1980.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 10:15:00 | 1979.85 | 1975.12 | 1980.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 1979.85 | 1975.12 | 1980.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:00:00 | 1979.85 | 1975.12 | 1980.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 1974.10 | 1974.92 | 1980.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 1965.00 | 1975.65 | 1978.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 12:15:00 | 1940.25 | 1928.88 | 1927.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 1940.25 | 1928.88 | 1927.73 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-03 13:15:00 | 1918.85 | 1926.88 | 1926.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 09:15:00 | 1883.60 | 1917.91 | 1922.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 1900.00 | 1897.97 | 1910.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 14:00:00 | 1900.00 | 1897.97 | 1910.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 14:15:00 | 1870.95 | 1892.57 | 1906.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 14:30:00 | 1885.00 | 1892.57 | 1906.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 1910.00 | 1896.05 | 1906.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 09:30:00 | 1865.00 | 1886.52 | 1901.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 09:15:00 | 1949.95 | 1911.55 | 1907.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 1949.95 | 1911.55 | 1907.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 1955.60 | 1920.36 | 1912.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 14:15:00 | 1980.00 | 1988.76 | 1964.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 15:00:00 | 1980.00 | 1988.76 | 1964.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 1978.45 | 1987.44 | 1976.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 15:00:00 | 1978.45 | 1987.44 | 1976.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 1974.00 | 1984.75 | 1976.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 1980.00 | 1984.75 | 1976.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:00:00 | 1980.00 | 1983.61 | 1977.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:30:00 | 1978.55 | 1981.83 | 1977.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 13:15:00 | 1978.95 | 1981.30 | 1979.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 1977.70 | 1980.58 | 1979.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:00:00 | 1977.70 | 1980.58 | 1979.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 1980.00 | 1980.46 | 1979.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 15:15:00 | 1989.00 | 1980.46 | 1979.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 15:15:00 | 1989.00 | 1982.17 | 1980.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 1995.65 | 1982.17 | 1980.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 11:15:00 | 1974.35 | 1979.12 | 1979.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 11:15:00 | 1974.35 | 1979.12 | 1979.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 09:15:00 | 1968.25 | 1974.21 | 1976.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 13:15:00 | 1988.25 | 1964.77 | 1967.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 13:15:00 | 1988.25 | 1964.77 | 1967.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 1988.25 | 1964.77 | 1967.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:00:00 | 1988.25 | 1964.77 | 1967.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 1982.20 | 1968.26 | 1968.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:45:00 | 1993.15 | 1968.26 | 1968.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2024-06-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 15:15:00 | 1982.80 | 1971.17 | 1970.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 1991.00 | 1975.13 | 1972.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 09:15:00 | 2467.65 | 2475.41 | 2359.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 2445.00 | 2433.54 | 2389.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 2445.00 | 2433.54 | 2389.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 11:00:00 | 2463.00 | 2439.43 | 2395.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 09:15:00 | 2324.95 | 2408.31 | 2399.67 | SL hit (close<static) qty=1.00 sl=2380.55 alert=retest2 |

### Cycle 80 — SELL (started 2024-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 10:15:00 | 2307.40 | 2388.13 | 2391.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 12:15:00 | 2262.95 | 2291.47 | 2326.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 2275.15 | 2263.53 | 2297.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 10:15:00 | 2275.15 | 2263.53 | 2297.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 2275.15 | 2263.53 | 2297.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:00:00 | 2275.15 | 2263.53 | 2297.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 2295.00 | 2269.83 | 2297.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:00:00 | 2295.00 | 2269.83 | 2297.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 2304.50 | 2276.76 | 2297.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:00:00 | 2304.50 | 2276.76 | 2297.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 2317.05 | 2284.82 | 2299.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 13:45:00 | 2319.85 | 2284.82 | 2299.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 2307.00 | 2290.89 | 2299.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 2330.40 | 2290.89 | 2299.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 2341.70 | 2301.05 | 2303.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:00:00 | 2341.70 | 2301.05 | 2303.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 2328.85 | 2306.61 | 2305.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 10:15:00 | 2398.00 | 2340.69 | 2331.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-08 09:15:00 | 2402.75 | 2431.87 | 2408.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-08 09:15:00 | 2402.75 | 2431.87 | 2408.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 2402.75 | 2431.87 | 2408.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:00:00 | 2402.75 | 2431.87 | 2408.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 2386.95 | 2422.88 | 2406.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 2386.95 | 2422.88 | 2406.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 2379.00 | 2414.11 | 2403.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 2379.00 | 2414.11 | 2403.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 12:15:00 | 2429.80 | 2417.24 | 2406.16 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 12:15:00 | 2396.00 | 2404.71 | 2404.76 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 13:15:00 | 2407.40 | 2405.25 | 2405.00 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 14:15:00 | 2389.85 | 2402.17 | 2403.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 15:15:00 | 2381.00 | 2397.93 | 2401.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 13:15:00 | 2375.95 | 2374.55 | 2386.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 13:30:00 | 2371.35 | 2374.55 | 2386.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 2395.50 | 2378.74 | 2387.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 2395.50 | 2378.74 | 2387.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 2395.00 | 2381.99 | 2387.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 2456.40 | 2381.99 | 2387.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 2430.80 | 2391.75 | 2391.78 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 10:15:00 | 2430.20 | 2399.44 | 2395.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 12:15:00 | 2441.85 | 2412.75 | 2402.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 15:15:00 | 2410.00 | 2416.90 | 2407.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 15:15:00 | 2410.00 | 2416.90 | 2407.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 2410.00 | 2416.90 | 2407.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:45:00 | 2402.15 | 2412.64 | 2406.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 2423.25 | 2414.76 | 2407.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 12:00:00 | 2442.00 | 2420.21 | 2410.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 12:30:00 | 2426.85 | 2421.68 | 2412.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 14:30:00 | 2427.95 | 2424.21 | 2415.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 2549.60 | 2421.37 | 2414.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-15 09:15:00 | 2686.20 | 2447.65 | 2427.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 2344.90 | 2452.43 | 2460.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 2298.00 | 2421.55 | 2445.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 14:15:00 | 2375.00 | 2371.94 | 2409.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 15:00:00 | 2375.00 | 2371.94 | 2409.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 2300.25 | 2278.05 | 2305.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 2300.25 | 2278.05 | 2305.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 2451.05 | 2312.65 | 2318.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:00:00 | 2451.05 | 2312.65 | 2318.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 12:15:00 | 2410.75 | 2332.27 | 2327.11 | EMA200 above EMA400 |

### Cycle 88 — SELL (started 2024-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 10:15:00 | 2373.65 | 2401.34 | 2403.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 14:15:00 | 2361.30 | 2382.02 | 2392.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 09:15:00 | 2395.45 | 2380.87 | 2390.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 2395.45 | 2380.87 | 2390.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 2395.45 | 2380.87 | 2390.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:30:00 | 2460.25 | 2380.87 | 2390.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 2400.95 | 2384.88 | 2391.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:00:00 | 2400.95 | 2384.88 | 2391.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 2383.45 | 2386.07 | 2390.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:30:00 | 2397.30 | 2386.07 | 2390.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 2393.15 | 2387.49 | 2390.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:00:00 | 2393.15 | 2387.49 | 2390.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 2396.00 | 2389.19 | 2391.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 14:45:00 | 2408.65 | 2389.19 | 2391.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 15:15:00 | 2386.00 | 2388.55 | 2390.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 10:45:00 | 2380.20 | 2386.89 | 2389.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 11:15:00 | 2380.55 | 2386.89 | 2389.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 12:15:00 | 2419.20 | 2393.28 | 2392.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 12:15:00 | 2419.20 | 2393.28 | 2392.07 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 13:15:00 | 2381.50 | 2390.92 | 2391.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 14:15:00 | 2380.00 | 2388.74 | 2390.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 2218.75 | 2199.17 | 2244.89 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 10:30:00 | 2193.35 | 2199.87 | 2241.05 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 11:30:00 | 2186.65 | 2199.82 | 2237.28 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 13:00:00 | 2193.00 | 2198.46 | 2233.26 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 13:30:00 | 2187.95 | 2189.66 | 2226.10 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 2183.85 | 2174.48 | 2208.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:15:00 | 2238.30 | 2174.48 | 2208.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 2216.65 | 2182.92 | 2209.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 10:15:00 | 2216.65 | 2182.92 | 2209.45 | SL hit (close>ema400) qty=1.00 sl=2209.45 alert=retest1 |

### Cycle 91 — BUY (started 2024-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 14:15:00 | 2226.10 | 2206.61 | 2204.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 11:15:00 | 2314.30 | 2231.55 | 2216.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 09:15:00 | 2281.60 | 2292.92 | 2259.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 10:15:00 | 2260.00 | 2286.34 | 2259.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 2260.00 | 2286.34 | 2259.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:00:00 | 2260.00 | 2286.34 | 2259.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 2254.95 | 2280.06 | 2259.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:30:00 | 2251.45 | 2280.06 | 2259.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 2228.70 | 2269.79 | 2256.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 13:00:00 | 2228.70 | 2269.79 | 2256.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 2196.65 | 2247.19 | 2247.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 2134.25 | 2216.52 | 2233.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 2232.35 | 2163.17 | 2189.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 2232.35 | 2163.17 | 2189.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 2232.35 | 2163.17 | 2189.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:00:00 | 2232.35 | 2163.17 | 2189.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 2207.05 | 2171.94 | 2190.72 | EMA400 retest candle locked (from downside) |

### Cycle 93 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 2249.00 | 2207.31 | 2202.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 12:15:00 | 2260.00 | 2236.70 | 2220.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 2248.00 | 2248.68 | 2232.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:00:00 | 2248.00 | 2248.68 | 2232.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 2201.95 | 2239.34 | 2229.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:45:00 | 2202.25 | 2239.34 | 2229.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 2200.00 | 2231.47 | 2227.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:30:00 | 2195.65 | 2231.47 | 2227.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2024-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 12:15:00 | 2189.05 | 2222.98 | 2223.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 10:15:00 | 2175.00 | 2202.70 | 2212.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-22 09:15:00 | 2183.45 | 2180.25 | 2194.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 09:15:00 | 2183.45 | 2180.25 | 2194.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 2183.45 | 2180.25 | 2194.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 12:00:00 | 2175.30 | 2179.79 | 2191.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-22 14:30:00 | 2175.05 | 2176.67 | 2187.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 12:00:00 | 2163.00 | 2103.48 | 2109.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-28 12:15:00 | 2163.90 | 2115.57 | 2114.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 12:15:00 | 2163.90 | 2115.57 | 2114.56 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2024-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 12:15:00 | 2093.00 | 2113.55 | 2116.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 13:15:00 | 2087.10 | 2108.26 | 2113.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 10:15:00 | 2105.75 | 2101.25 | 2107.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 10:15:00 | 2105.75 | 2101.25 | 2107.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 2105.75 | 2101.25 | 2107.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 2105.75 | 2101.25 | 2107.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 2102.60 | 2101.52 | 2107.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 12:15:00 | 2097.05 | 2101.52 | 2107.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 13:15:00 | 2098.55 | 2092.49 | 2098.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 2097.10 | 2089.87 | 2095.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 1992.20 | 2016.95 | 2031.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 1993.62 | 2016.95 | 2031.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-09 09:15:00 | 1992.24 | 2016.95 | 2031.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-10 09:15:00 | 2013.20 | 2008.36 | 2018.48 | SL hit (close>ema200) qty=0.50 sl=2008.36 alert=retest2 |

### Cycle 97 — BUY (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 15:15:00 | 2039.85 | 2023.46 | 2021.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 2047.80 | 2028.33 | 2024.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 15:15:00 | 2071.40 | 2079.71 | 2064.60 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 09:45:00 | 2090.40 | 2080.37 | 2066.27 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 10:30:00 | 2089.30 | 2082.68 | 2068.60 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 12:00:00 | 2086.05 | 2083.36 | 2070.19 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-13 13:45:00 | 2092.30 | 2084.69 | 2073.07 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 2095.05 | 2089.28 | 2078.30 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-18 09:15:00 | 2062.80 | 2086.67 | 2085.92 | SL hit (close<ema400) qty=1.00 sl=2085.92 alert=retest1 |

### Cycle 98 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 2055.00 | 2080.34 | 2083.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 10:15:00 | 2046.90 | 2069.80 | 2076.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 10:15:00 | 2073.60 | 2049.08 | 2058.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 10:15:00 | 2073.60 | 2049.08 | 2058.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 2073.60 | 2049.08 | 2058.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 2070.95 | 2049.08 | 2058.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 2075.25 | 2054.31 | 2060.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 12:45:00 | 2060.20 | 2056.02 | 2060.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 2118.55 | 2073.92 | 2067.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 09:15:00 | 2118.55 | 2073.92 | 2067.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 10:15:00 | 2126.10 | 2084.36 | 2073.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 14:15:00 | 2081.00 | 2087.62 | 2078.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 14:30:00 | 2090.30 | 2087.62 | 2078.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 2086.10 | 2088.18 | 2080.58 | EMA400 retest candle locked (from upside) |

### Cycle 100 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 2064.00 | 2076.37 | 2077.95 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 10:15:00 | 2124.30 | 2085.96 | 2082.16 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 2052.15 | 2079.16 | 2079.73 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-09-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 14:15:00 | 2087.10 | 2074.32 | 2074.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 2145.90 | 2089.86 | 2081.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 09:15:00 | 2132.75 | 2139.84 | 2117.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 09:45:00 | 2135.00 | 2139.84 | 2117.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 2116.10 | 2135.09 | 2116.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 2116.10 | 2135.09 | 2116.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 2125.85 | 2133.24 | 2117.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 12:15:00 | 2138.85 | 2133.24 | 2117.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 09:30:00 | 2132.80 | 2169.63 | 2166.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-07 09:15:00 | 2097.05 | 2160.63 | 2166.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 09:15:00 | 2097.05 | 2160.63 | 2166.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 10:15:00 | 2052.15 | 2138.94 | 2156.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 2104.50 | 2086.98 | 2116.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:00:00 | 2104.50 | 2086.98 | 2116.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 2112.75 | 2092.13 | 2116.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 2112.75 | 2092.13 | 2116.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 2117.55 | 2097.22 | 2116.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 2117.55 | 2097.22 | 2116.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 2108.20 | 2099.41 | 2115.42 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 2132.55 | 2122.57 | 2122.27 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2024-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 12:15:00 | 2115.20 | 2120.84 | 2121.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-09 14:15:00 | 2097.85 | 2114.91 | 2118.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-10 12:15:00 | 2109.70 | 2105.20 | 2111.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-10 13:00:00 | 2109.70 | 2105.20 | 2111.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 2090.00 | 2102.16 | 2109.66 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2024-10-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 10:15:00 | 2117.60 | 2113.56 | 2113.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-11 13:15:00 | 2139.70 | 2122.50 | 2117.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 09:15:00 | 2110.10 | 2121.75 | 2118.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 2110.10 | 2121.75 | 2118.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 2110.10 | 2121.75 | 2118.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-14 10:00:00 | 2110.10 | 2121.75 | 2118.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 10:15:00 | 2148.75 | 2127.15 | 2121.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:15:00 | 2159.00 | 2127.15 | 2121.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 15:00:00 | 2151.70 | 2144.48 | 2132.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 2174.40 | 2144.38 | 2133.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 10:15:00 | 2176.00 | 2156.37 | 2155.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 2151.70 | 2154.10 | 2154.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 11:15:00 | 2151.70 | 2154.10 | 2154.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 09:15:00 | 2122.05 | 2143.33 | 2148.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 15:15:00 | 2129.55 | 2128.80 | 2138.02 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 10:00:00 | 2083.60 | 2119.76 | 2133.07 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-22 11:15:00 | 2081.05 | 2114.38 | 2129.41 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 12:15:00 | 1979.42 | 2010.94 | 2037.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 12:15:00 | 1977.00 | 2010.94 | 2037.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-10-29 14:15:00 | 1875.24 | 1903.52 | 1938.86 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 109 — BUY (started 2024-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 12:15:00 | 1953.80 | 1935.13 | 1932.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 1976.10 | 1949.09 | 1940.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-01 18:15:00 | 1948.00 | 1948.88 | 1941.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-01 18:45:00 | 1948.00 | 1948.88 | 1941.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1918.85 | 1942.87 | 1939.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 1918.80 | 1942.87 | 1939.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1929.75 | 1940.25 | 1938.51 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 1927.15 | 1935.77 | 1936.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 1911.40 | 1926.83 | 1931.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 09:15:00 | 1944.95 | 1914.40 | 1920.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 09:15:00 | 1944.95 | 1914.40 | 1920.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 1944.95 | 1914.40 | 1920.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:00:00 | 1944.95 | 1914.40 | 1920.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 2008.70 | 1933.26 | 1928.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 12:15:00 | 2029.40 | 1996.48 | 1970.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 09:15:00 | 1902.15 | 2022.54 | 2013.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 09:15:00 | 1902.15 | 2022.54 | 2013.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 09:15:00 | 1902.15 | 2022.54 | 2013.44 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2024-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 10:15:00 | 1913.85 | 2000.81 | 2004.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 1896.00 | 1950.24 | 1977.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 12:15:00 | 1743.25 | 1739.09 | 1775.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 13:00:00 | 1743.25 | 1739.09 | 1775.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 1787.15 | 1748.70 | 1776.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 1787.15 | 1748.70 | 1776.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 1746.90 | 1748.34 | 1774.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 1734.65 | 1748.47 | 1771.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-19 14:15:00 | 1647.92 | 1707.20 | 1739.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-22 11:15:00 | 1561.19 | 1593.13 | 1641.82 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 113 — BUY (started 2024-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-26 09:15:00 | 1703.00 | 1649.23 | 1642.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 10:15:00 | 1720.60 | 1663.50 | 1649.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 15:15:00 | 1735.00 | 1740.23 | 1714.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 09:15:00 | 1750.85 | 1740.23 | 1714.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 1736.75 | 1743.50 | 1733.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 15:00:00 | 1754.55 | 1743.27 | 1735.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 1692.95 | 1734.57 | 1733.04 | SL hit (close<static) qty=1.00 sl=1720.05 alert=retest2 |

### Cycle 114 — SELL (started 2024-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 10:15:00 | 1696.80 | 1727.01 | 1729.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 15:15:00 | 1690.00 | 1703.49 | 1715.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-04 09:15:00 | 1585.95 | 1582.11 | 1631.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-04 10:00:00 | 1585.95 | 1582.11 | 1631.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1617.05 | 1581.45 | 1605.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:45:00 | 1625.00 | 1581.45 | 1605.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1634.30 | 1592.02 | 1607.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:00:00 | 1634.30 | 1592.02 | 1607.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 15:15:00 | 1625.40 | 1615.12 | 1614.96 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 1610.80 | 1614.26 | 1614.58 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 1621.95 | 1615.80 | 1615.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 11:15:00 | 1651.00 | 1622.84 | 1618.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 09:15:00 | 1775.55 | 1792.82 | 1749.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-11 09:45:00 | 1778.10 | 1792.82 | 1749.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 1770.00 | 1784.75 | 1769.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 1770.00 | 1784.75 | 1769.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 1775.00 | 1782.80 | 1770.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 13:45:00 | 1778.75 | 1780.82 | 1770.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 14:45:00 | 1780.00 | 1779.09 | 1770.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 09:15:00 | 1742.60 | 1770.00 | 1767.80 | SL hit (close<static) qty=1.00 sl=1764.25 alert=retest2 |

### Cycle 118 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 1724.10 | 1760.82 | 1763.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 09:15:00 | 1695.65 | 1715.57 | 1723.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 11:15:00 | 1609.35 | 1607.52 | 1622.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 12:00:00 | 1609.35 | 1607.52 | 1622.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 1587.15 | 1602.64 | 1614.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 09:15:00 | 1576.30 | 1592.93 | 1603.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 11:15:00 | 1630.30 | 1601.88 | 1600.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 11:15:00 | 1630.30 | 1601.88 | 1600.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 1639.50 | 1620.80 | 1611.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 10:15:00 | 1622.55 | 1631.41 | 1623.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 10:15:00 | 1622.55 | 1631.41 | 1623.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 10:15:00 | 1622.55 | 1631.41 | 1623.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:00:00 | 1622.55 | 1631.41 | 1623.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 1647.45 | 1634.62 | 1625.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:30:00 | 1623.80 | 1634.62 | 1625.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 1629.00 | 1633.62 | 1628.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 09:15:00 | 1643.45 | 1633.62 | 1628.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 10:15:00 | 1637.65 | 1651.44 | 1652.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 10:15:00 | 1637.65 | 1651.44 | 1652.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 14:15:00 | 1626.25 | 1639.73 | 1644.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 1644.50 | 1638.33 | 1642.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 1644.50 | 1638.33 | 1642.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 1644.50 | 1638.33 | 1642.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:45:00 | 1649.15 | 1638.33 | 1642.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 1644.00 | 1639.46 | 1642.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:45:00 | 1645.95 | 1639.46 | 1642.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 1643.55 | 1640.28 | 1642.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:15:00 | 1641.60 | 1640.28 | 1642.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-14 14:15:00 | 1651.75 | 1643.80 | 1644.02 | SL hit (close>static) qty=1.00 sl=1646.45 alert=retest2 |

### Cycle 121 — BUY (started 2025-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 15:15:00 | 1650.00 | 1645.04 | 1644.56 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-01-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 09:15:00 | 1631.90 | 1642.41 | 1643.41 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 14:15:00 | 1650.65 | 1641.62 | 1640.42 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 09:15:00 | 1640.90 | 1656.52 | 1657.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 10:15:00 | 1626.35 | 1650.48 | 1654.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1630.70 | 1630.46 | 1641.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 1630.70 | 1630.46 | 1641.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 1630.70 | 1630.46 | 1641.47 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 10:15:00 | 1651.05 | 1642.19 | 1641.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 1666.85 | 1651.63 | 1647.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 15:15:00 | 1747.00 | 1748.21 | 1732.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 15:15:00 | 1747.00 | 1748.21 | 1732.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 1747.00 | 1748.21 | 1732.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 1723.40 | 1743.25 | 1731.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1701.85 | 1734.97 | 1728.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:45:00 | 1701.45 | 1734.97 | 1728.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 126 — SELL (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 12:15:00 | 1703.45 | 1723.84 | 1724.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 1697.00 | 1709.63 | 1716.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 09:15:00 | 1699.50 | 1696.44 | 1705.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 12:15:00 | 1700.40 | 1698.40 | 1703.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 1700.40 | 1698.40 | 1703.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:45:00 | 1704.65 | 1698.40 | 1703.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 1700.80 | 1698.88 | 1703.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:30:00 | 1700.00 | 1698.88 | 1703.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 1700.95 | 1699.30 | 1703.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:45:00 | 1713.25 | 1699.30 | 1703.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 1696.00 | 1698.64 | 1702.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 1689.20 | 1698.64 | 1702.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 1656.05 | 1690.12 | 1698.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:45:00 | 1654.00 | 1682.36 | 1694.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:30:00 | 1641.85 | 1647.97 | 1668.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 12:15:00 | 1651.60 | 1649.52 | 1667.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 14:15:00 | 1654.60 | 1651.21 | 1664.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 1688.80 | 1658.73 | 1667.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 15:00:00 | 1688.80 | 1658.73 | 1667.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 1670.00 | 1660.98 | 1667.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 09:15:00 | 1647.45 | 1660.98 | 1667.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 1645.60 | 1657.91 | 1665.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-13 10:15:00 | 1680.20 | 1663.98 | 1663.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 1680.20 | 1663.98 | 1663.89 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 1648.00 | 1662.25 | 1663.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 15:15:00 | 1640.25 | 1652.56 | 1657.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 09:15:00 | 1653.25 | 1652.70 | 1657.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 09:15:00 | 1653.25 | 1652.70 | 1657.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 1653.25 | 1652.70 | 1657.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:45:00 | 1654.80 | 1652.70 | 1657.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 1652.45 | 1652.65 | 1656.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:30:00 | 1657.20 | 1652.65 | 1656.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 1649.55 | 1651.98 | 1655.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:30:00 | 1650.05 | 1651.98 | 1655.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 1654.20 | 1652.43 | 1655.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 1661.95 | 1652.43 | 1655.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1657.25 | 1653.39 | 1655.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1657.25 | 1653.39 | 1655.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1655.60 | 1653.83 | 1655.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 1657.00 | 1653.83 | 1655.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1667.00 | 1656.47 | 1656.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:00:00 | 1667.00 | 1656.47 | 1656.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 10:15:00 | 1672.30 | 1659.63 | 1658.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 11:15:00 | 1678.90 | 1663.49 | 1660.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 15:15:00 | 1651.20 | 1663.82 | 1661.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 15:15:00 | 1651.20 | 1663.82 | 1661.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 1651.20 | 1663.82 | 1661.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-19 09:15:00 | 1679.45 | 1663.82 | 1661.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 1653.30 | 1683.94 | 1686.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 1653.30 | 1683.94 | 1686.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 10:15:00 | 1633.30 | 1673.81 | 1681.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 14:15:00 | 1699.45 | 1648.53 | 1656.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 14:15:00 | 1699.45 | 1648.53 | 1656.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 1699.45 | 1648.53 | 1656.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 1699.45 | 1648.53 | 1656.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 1686.00 | 1656.02 | 1658.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 1680.15 | 1656.02 | 1658.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-27 14:15:00 | 1734.55 | 1659.09 | 1657.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-27 14:15:00 | 1734.55 | 1659.09 | 1657.53 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 1614.70 | 1656.38 | 1661.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 10:15:00 | 1586.35 | 1642.37 | 1654.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 14:15:00 | 1688.85 | 1627.77 | 1640.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 14:15:00 | 1688.85 | 1627.77 | 1640.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 1688.85 | 1627.77 | 1640.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 1688.85 | 1627.77 | 1640.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 1658.00 | 1633.82 | 1642.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 1643.20 | 1633.82 | 1642.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 10:30:00 | 1648.45 | 1636.59 | 1642.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:00:00 | 1639.05 | 1635.76 | 1639.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 14:15:00 | 1680.40 | 1647.58 | 1643.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 14:15:00 | 1680.40 | 1647.58 | 1643.16 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-03-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 15:15:00 | 1639.00 | 1653.98 | 1654.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 09:15:00 | 1610.10 | 1645.20 | 1650.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 14:15:00 | 1627.50 | 1614.77 | 1630.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 14:15:00 | 1627.50 | 1614.77 | 1630.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1627.50 | 1614.77 | 1630.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 1627.50 | 1614.77 | 1630.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1645.00 | 1620.81 | 1632.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 09:15:00 | 1591.20 | 1620.81 | 1632.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 13:15:00 | 1633.60 | 1625.40 | 1625.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 13:15:00 | 1633.60 | 1625.40 | 1625.13 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 09:15:00 | 1616.80 | 1623.92 | 1624.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 14:15:00 | 1590.25 | 1608.67 | 1616.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 14:15:00 | 1594.85 | 1577.58 | 1593.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 14:15:00 | 1594.85 | 1577.58 | 1593.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 1594.85 | 1577.58 | 1593.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 15:00:00 | 1594.85 | 1577.58 | 1593.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 1601.85 | 1582.43 | 1593.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 09:30:00 | 1588.20 | 1582.75 | 1593.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-18 15:00:00 | 1588.30 | 1587.27 | 1591.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 11:15:00 | 1632.90 | 1600.36 | 1596.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 11:15:00 | 1632.90 | 1600.36 | 1596.54 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 11:15:00 | 1605.20 | 1608.77 | 1609.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 12:15:00 | 1596.70 | 1606.35 | 1607.89 | Break + close below crossover candle low |

### Cycle 139 — BUY (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 14:15:00 | 1623.60 | 1608.79 | 1608.66 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 1604.20 | 1607.74 | 1608.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 10:15:00 | 1591.55 | 1604.50 | 1606.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 14:15:00 | 1620.15 | 1599.74 | 1602.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 14:15:00 | 1620.15 | 1599.74 | 1602.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 1620.15 | 1599.74 | 1602.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 1620.15 | 1599.74 | 1602.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 1600.15 | 1599.82 | 1602.65 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 09:15:00 | 1624.55 | 1604.77 | 1604.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 10:15:00 | 1635.00 | 1610.81 | 1607.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-01 15:15:00 | 1691.00 | 1701.39 | 1678.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-02 09:15:00 | 1680.85 | 1701.39 | 1678.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 1686.40 | 1698.39 | 1679.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 10:45:00 | 1703.30 | 1697.91 | 1680.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 14:45:00 | 1738.80 | 1703.21 | 1688.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 14:30:00 | 1699.60 | 1714.50 | 1710.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1597.95 | 1693.83 | 1701.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1597.95 | 1693.83 | 1701.86 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 1711.00 | 1686.79 | 1686.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 1716.95 | 1692.82 | 1689.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 14:15:00 | 1715.80 | 1721.03 | 1709.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-09 15:00:00 | 1715.80 | 1721.03 | 1709.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 15:15:00 | 1691.10 | 1715.04 | 1708.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 1721.20 | 1715.04 | 1708.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 1749.60 | 1793.07 | 1794.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 1749.60 | 1793.07 | 1794.78 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 1840.00 | 1791.28 | 1785.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 15:15:00 | 1853.00 | 1803.63 | 1791.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 14:15:00 | 1831.10 | 1832.60 | 1814.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 14:45:00 | 1836.60 | 1832.60 | 1814.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 1813.00 | 1828.68 | 1814.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:15:00 | 1798.40 | 1828.68 | 1814.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 1805.80 | 1824.11 | 1813.57 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 1776.00 | 1807.28 | 1809.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 13:15:00 | 1772.70 | 1790.87 | 1800.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 14:15:00 | 1838.70 | 1800.44 | 1803.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 14:15:00 | 1838.70 | 1800.44 | 1803.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 1838.70 | 1800.44 | 1803.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 15:00:00 | 1838.70 | 1800.44 | 1803.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2025-05-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 15:15:00 | 1828.00 | 1805.95 | 1805.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 10:15:00 | 1852.00 | 1819.33 | 1812.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 11:15:00 | 1840.90 | 1846.68 | 1834.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 11:30:00 | 1835.50 | 1846.68 | 1834.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 1832.00 | 1842.34 | 1834.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:30:00 | 1839.70 | 1842.34 | 1834.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 1856.20 | 1845.11 | 1836.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 15:15:00 | 1865.00 | 1839.28 | 1835.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 11:00:00 | 1888.50 | 1856.04 | 1844.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 14:15:00 | 1815.10 | 1853.80 | 1848.45 | SL hit (close<static) qty=1.00 sl=1827.00 alert=retest2 |

### Cycle 148 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 1804.00 | 1843.84 | 1844.41 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 1941.30 | 1853.92 | 1844.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 2034.60 | 1910.09 | 1874.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 13:15:00 | 2049.60 | 2050.46 | 2004.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 14:00:00 | 2049.60 | 2050.46 | 2004.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 2076.90 | 2095.53 | 2076.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 2067.10 | 2095.53 | 2076.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 2074.10 | 2091.24 | 2076.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:30:00 | 2072.40 | 2091.24 | 2076.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 2078.10 | 2088.62 | 2076.53 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2025-05-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 15:15:00 | 2048.90 | 2070.42 | 2070.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 10:15:00 | 2044.50 | 2061.78 | 2066.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 1999.80 | 1999.06 | 2012.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 14:45:00 | 2000.00 | 1999.06 | 2012.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 1997.40 | 1996.81 | 2004.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:00:00 | 1997.40 | 1996.81 | 2004.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1982.50 | 1992.38 | 2000.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 1976.10 | 1989.12 | 1998.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-02 09:15:00 | 1877.29 | 1929.67 | 1932.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 1938.90 | 1929.67 | 1932.83 | SL hit (close>static) qty=0.50 sl=1929.67 alert=retest2 |

### Cycle 151 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 1941.10 | 1935.32 | 1934.91 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 1915.90 | 1932.73 | 1934.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 11:15:00 | 1910.00 | 1925.35 | 1930.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 11:15:00 | 1921.40 | 1916.00 | 1921.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 11:15:00 | 1921.40 | 1916.00 | 1921.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 1921.40 | 1916.00 | 1921.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 1920.00 | 1916.00 | 1921.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 12:15:00 | 1926.00 | 1918.00 | 1922.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 12:30:00 | 1923.60 | 1918.00 | 1922.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1921.00 | 1918.60 | 1922.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 09:15:00 | 1908.40 | 1917.73 | 1919.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 11:00:00 | 1909.90 | 1909.26 | 1912.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 12:00:00 | 1911.90 | 1909.79 | 1912.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 14:15:00 | 1929.00 | 1914.83 | 1914.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — BUY (started 2025-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 14:15:00 | 1929.00 | 1914.83 | 1914.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 10:15:00 | 1949.80 | 1926.47 | 1921.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 1929.50 | 1933.99 | 1926.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 1929.50 | 1933.99 | 1926.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1929.50 | 1933.99 | 1926.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 1934.00 | 1933.99 | 1926.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1949.00 | 1936.99 | 1928.88 | EMA400 retest candle locked (from upside) |

### Cycle 154 — SELL (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 09:15:00 | 1796.00 | 1909.27 | 1917.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 1784.90 | 1884.40 | 1905.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-17 14:15:00 | 1745.00 | 1740.51 | 1759.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-17 14:45:00 | 1748.80 | 1740.51 | 1759.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 1759.30 | 1745.03 | 1758.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 1763.50 | 1745.03 | 1758.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 1749.20 | 1745.87 | 1757.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 10:30:00 | 1758.40 | 1745.87 | 1757.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 1753.70 | 1747.43 | 1757.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:45:00 | 1749.90 | 1747.43 | 1757.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1737.70 | 1743.79 | 1751.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 1733.40 | 1743.79 | 1751.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 1731.70 | 1741.52 | 1747.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 1730.50 | 1737.95 | 1742.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 1760.00 | 1742.33 | 1741.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 1760.00 | 1742.33 | 1741.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 11:15:00 | 1778.30 | 1760.05 | 1752.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 15:15:00 | 1763.10 | 1767.90 | 1763.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 15:15:00 | 1763.10 | 1767.90 | 1763.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 15:15:00 | 1763.10 | 1767.90 | 1763.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:15:00 | 1757.00 | 1767.90 | 1763.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1755.30 | 1765.38 | 1762.80 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 11:15:00 | 1753.10 | 1760.86 | 1761.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 12:15:00 | 1750.20 | 1758.73 | 1760.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 14:15:00 | 1763.00 | 1748.88 | 1751.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-30 14:15:00 | 1763.00 | 1748.88 | 1751.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1763.00 | 1748.88 | 1751.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:45:00 | 1764.00 | 1748.88 | 1751.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1753.60 | 1749.83 | 1752.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 1759.80 | 1749.83 | 1752.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 157 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 1765.80 | 1755.37 | 1754.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-01 13:15:00 | 1773.10 | 1761.61 | 1757.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 1766.20 | 1768.80 | 1762.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:00:00 | 1766.20 | 1768.80 | 1762.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1758.40 | 1766.72 | 1762.14 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 15:15:00 | 1746.60 | 1758.41 | 1759.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 1736.90 | 1750.19 | 1754.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 10:15:00 | 1744.20 | 1739.00 | 1744.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 10:15:00 | 1744.20 | 1739.00 | 1744.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 1744.20 | 1739.00 | 1744.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 11:00:00 | 1744.20 | 1739.00 | 1744.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1740.70 | 1739.34 | 1743.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 13:00:00 | 1732.00 | 1737.87 | 1742.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-07 14:15:00 | 1780.10 | 1746.98 | 1746.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2025-07-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 14:15:00 | 1780.10 | 1746.98 | 1746.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 1785.00 | 1765.48 | 1757.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 15:15:00 | 1775.60 | 1775.72 | 1768.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:15:00 | 1768.80 | 1775.72 | 1768.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1769.90 | 1774.56 | 1768.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:30:00 | 1767.90 | 1774.56 | 1768.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1768.20 | 1773.29 | 1768.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1768.20 | 1773.29 | 1768.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1787.90 | 1776.21 | 1770.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 14:45:00 | 1792.30 | 1782.14 | 1774.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-11 10:15:00 | 1767.20 | 1778.50 | 1774.78 | SL hit (close<static) qty=1.00 sl=1768.00 alert=retest2 |

### Cycle 160 — SELL (started 2025-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 09:15:00 | 1780.10 | 1796.34 | 1797.02 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 15:15:00 | 1825.10 | 1799.38 | 1797.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 11:15:00 | 1850.00 | 1820.76 | 1809.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 10:15:00 | 1838.60 | 1839.39 | 1825.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-23 11:00:00 | 1838.60 | 1839.39 | 1825.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 1818.50 | 1832.68 | 1825.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 1818.50 | 1832.68 | 1825.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 1821.70 | 1830.49 | 1825.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:15:00 | 1808.10 | 1830.49 | 1825.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 1808.10 | 1826.01 | 1823.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 1824.30 | 1826.01 | 1823.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 1812.10 | 1821.80 | 1822.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1812.10 | 1821.80 | 1822.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 1809.70 | 1819.38 | 1821.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 14:15:00 | 1842.30 | 1821.00 | 1821.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 14:15:00 | 1842.30 | 1821.00 | 1821.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 1842.30 | 1821.00 | 1821.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 1842.30 | 1821.00 | 1821.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 15:15:00 | 1823.70 | 1821.54 | 1821.37 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1812.90 | 1819.81 | 1820.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 1801.00 | 1816.05 | 1818.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 12:15:00 | 1819.30 | 1813.91 | 1817.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 12:15:00 | 1819.30 | 1813.91 | 1817.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1819.30 | 1813.91 | 1817.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:00:00 | 1819.30 | 1813.91 | 1817.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1825.00 | 1816.13 | 1817.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:30:00 | 1825.50 | 1816.13 | 1817.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 165 — BUY (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-25 14:15:00 | 1835.00 | 1819.90 | 1819.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-28 09:15:00 | 1842.10 | 1826.92 | 1822.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-29 09:15:00 | 1817.40 | 1836.70 | 1831.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 1817.40 | 1836.70 | 1831.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1817.40 | 1836.70 | 1831.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:15:00 | 1815.70 | 1836.70 | 1831.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1803.70 | 1830.10 | 1829.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 1803.70 | 1830.10 | 1829.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — SELL (started 2025-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 11:15:00 | 1813.60 | 1826.80 | 1827.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 1799.10 | 1810.70 | 1815.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 12:15:00 | 1816.20 | 1809.29 | 1813.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 12:15:00 | 1816.20 | 1809.29 | 1813.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 1816.20 | 1809.29 | 1813.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 1816.20 | 1809.29 | 1813.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1813.00 | 1810.03 | 1813.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:15:00 | 1792.10 | 1809.63 | 1813.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:30:00 | 1805.80 | 1804.96 | 1810.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 10:15:00 | 1782.40 | 1764.21 | 1763.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2025-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 10:15:00 | 1782.40 | 1764.21 | 1763.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 14:15:00 | 1811.10 | 1778.91 | 1772.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 1788.00 | 1793.35 | 1784.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 14:00:00 | 1788.00 | 1793.35 | 1784.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 1785.00 | 1792.01 | 1785.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:45:00 | 1797.20 | 1785.43 | 1784.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 15:15:00 | 1776.40 | 1783.62 | 1783.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — SELL (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 15:15:00 | 1776.40 | 1783.62 | 1783.86 | EMA200 below EMA400 |

### Cycle 169 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 1789.60 | 1784.30 | 1784.10 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 13:15:00 | 1780.00 | 1783.29 | 1783.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 15:15:00 | 1776.00 | 1781.92 | 1783.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 11:15:00 | 1785.00 | 1781.57 | 1782.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 11:15:00 | 1785.00 | 1781.57 | 1782.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 1785.00 | 1781.57 | 1782.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 1785.00 | 1781.57 | 1782.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 1785.90 | 1782.43 | 1782.79 | EMA400 retest candle locked (from downside) |

### Cycle 171 — BUY (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 13:15:00 | 1787.40 | 1783.43 | 1783.21 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 14:15:00 | 1780.00 | 1782.74 | 1782.92 | EMA200 below EMA400 |

### Cycle 173 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 1785.00 | 1783.19 | 1783.11 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 13:15:00 | 1780.10 | 1782.83 | 1783.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 15:15:00 | 1779.10 | 1781.87 | 1782.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 13:15:00 | 1642.30 | 1639.55 | 1653.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 14:00:00 | 1642.30 | 1639.55 | 1653.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 1638.50 | 1639.34 | 1651.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:45:00 | 1654.70 | 1639.34 | 1651.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1647.90 | 1641.96 | 1651.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:30:00 | 1649.90 | 1641.96 | 1651.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 1650.40 | 1643.65 | 1650.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:30:00 | 1650.10 | 1643.65 | 1650.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 1650.00 | 1644.92 | 1650.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:00:00 | 1644.10 | 1645.57 | 1650.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 10:45:00 | 1646.00 | 1646.83 | 1649.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 14:15:00 | 1643.40 | 1646.40 | 1648.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:30:00 | 1642.60 | 1647.04 | 1648.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1648.00 | 1647.24 | 1648.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1648.00 | 1647.24 | 1648.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1647.00 | 1647.19 | 1647.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-08 09:15:00 | 1649.20 | 1648.18 | 1648.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 1649.20 | 1648.18 | 1648.08 | EMA200 above EMA400 |

### Cycle 176 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 1647.00 | 1647.98 | 1648.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 1640.60 | 1646.51 | 1647.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 1647.60 | 1645.49 | 1646.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1647.60 | 1645.49 | 1646.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1647.60 | 1645.49 | 1646.68 | EMA400 retest candle locked (from downside) |

### Cycle 177 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 1647.50 | 1646.74 | 1646.69 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-09-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 11:15:00 | 1642.80 | 1646.41 | 1646.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 12:15:00 | 1640.00 | 1645.13 | 1646.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 1644.70 | 1644.43 | 1645.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 14:15:00 | 1644.70 | 1644.43 | 1645.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 1644.70 | 1644.43 | 1645.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:45:00 | 1646.50 | 1644.43 | 1645.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 1644.10 | 1644.37 | 1645.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 1640.00 | 1644.37 | 1645.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1635.60 | 1642.61 | 1644.61 | EMA400 retest candle locked (from downside) |

### Cycle 179 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 1647.90 | 1643.15 | 1642.72 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 1637.00 | 1642.63 | 1642.91 | EMA200 below EMA400 |

### Cycle 181 — BUY (started 2025-09-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 10:15:00 | 1646.60 | 1642.89 | 1642.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 15:15:00 | 1649.90 | 1645.83 | 1644.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 12:15:00 | 1640.90 | 1646.09 | 1645.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 12:15:00 | 1640.90 | 1646.09 | 1645.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 1640.90 | 1646.09 | 1645.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:00:00 | 1640.90 | 1646.09 | 1645.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 1643.90 | 1645.65 | 1645.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 1642.40 | 1645.65 | 1645.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1654.20 | 1648.77 | 1646.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 1647.90 | 1648.77 | 1646.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1647.90 | 1649.45 | 1647.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:30:00 | 1644.10 | 1649.45 | 1647.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 1647.50 | 1649.06 | 1647.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 1638.70 | 1649.06 | 1647.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 182 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1638.60 | 1646.97 | 1647.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 1630.00 | 1637.60 | 1641.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 1634.70 | 1632.95 | 1636.80 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1623.40 | 1632.95 | 1636.80 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1644.30 | 1635.22 | 1637.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 1644.30 | 1635.22 | 1637.49 | SL hit (close>ema400) qty=1.00 sl=1637.49 alert=retest1 |

### Cycle 183 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 1649.90 | 1622.70 | 1621.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 1652.80 | 1636.49 | 1630.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 1666.90 | 1669.08 | 1655.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 13:00:00 | 1666.90 | 1669.08 | 1655.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1670.20 | 1680.01 | 1675.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 14:45:00 | 1699.90 | 1690.92 | 1683.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 09:15:00 | 1703.80 | 1692.12 | 1684.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 11:30:00 | 1700.10 | 1697.52 | 1689.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 14:00:00 | 1706.80 | 1700.14 | 1691.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-13 09:15:00 | 1869.89 | 1726.24 | 1705.87 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 184 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 1798.60 | 1833.59 | 1837.96 | EMA200 below EMA400 |

### Cycle 185 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1839.50 | 1831.80 | 1831.47 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 09:15:00 | 1817.30 | 1828.90 | 1830.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 15:15:00 | 1805.20 | 1818.42 | 1823.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 12:15:00 | 1821.60 | 1818.45 | 1822.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 12:15:00 | 1821.60 | 1818.45 | 1822.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 1821.60 | 1818.45 | 1822.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 12:45:00 | 1823.40 | 1818.45 | 1822.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 1816.70 | 1818.10 | 1821.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:00:00 | 1813.00 | 1817.24 | 1820.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:00:00 | 1810.00 | 1814.25 | 1818.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:00:00 | 1811.50 | 1805.68 | 1810.90 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 1810.00 | 1806.44 | 1810.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 1810.00 | 1807.15 | 1810.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:45:00 | 1793.00 | 1804.92 | 1808.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-30 14:15:00 | 1829.90 | 1792.40 | 1795.04 | SL hit (close>static) qty=1.00 sl=1822.70 alert=retest2 |

### Cycle 187 — BUY (started 2025-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 15:15:00 | 1831.50 | 1800.22 | 1798.35 | EMA200 above EMA400 |

### Cycle 188 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 1788.90 | 1796.03 | 1796.65 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 14:15:00 | 1821.10 | 1797.56 | 1796.69 | EMA200 above EMA400 |

### Cycle 190 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1791.40 | 1799.03 | 1799.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 1787.60 | 1796.74 | 1798.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1768.60 | 1765.56 | 1775.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:45:00 | 1767.80 | 1765.56 | 1775.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 1780.20 | 1768.49 | 1775.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:45:00 | 1777.70 | 1768.49 | 1775.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1803.90 | 1775.57 | 1778.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 1802.00 | 1775.57 | 1778.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — BUY (started 2025-11-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 14:15:00 | 1816.00 | 1783.65 | 1781.91 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 1748.80 | 1784.67 | 1788.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 1724.00 | 1748.38 | 1765.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 10:15:00 | 1697.80 | 1696.06 | 1708.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:45:00 | 1696.00 | 1696.06 | 1708.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1699.80 | 1697.42 | 1705.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:30:00 | 1713.90 | 1697.42 | 1705.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1693.70 | 1697.09 | 1703.60 | EMA400 retest candle locked (from downside) |

### Cycle 193 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1702.60 | 1699.06 | 1698.60 | EMA200 above EMA400 |

### Cycle 194 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 1679.00 | 1696.19 | 1698.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 15:15:00 | 1657.20 | 1668.35 | 1679.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1671.00 | 1668.88 | 1678.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 1671.00 | 1668.88 | 1678.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1671.00 | 1668.88 | 1678.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:30:00 | 1673.20 | 1668.88 | 1678.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 1668.50 | 1664.60 | 1671.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 10:15:00 | 1664.10 | 1664.60 | 1671.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:15:00 | 1662.00 | 1665.42 | 1670.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 1664.10 | 1665.53 | 1670.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:00:00 | 1661.40 | 1664.71 | 1669.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 14:15:00 | 1656.60 | 1661.53 | 1667.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-27 15:00:00 | 1656.60 | 1661.53 | 1667.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1673.90 | 1664.56 | 1667.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 10:00:00 | 1673.90 | 1664.56 | 1667.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1669.20 | 1665.49 | 1667.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-28 11:15:00 | 1687.70 | 1669.93 | 1669.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — BUY (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 11:15:00 | 1687.70 | 1669.93 | 1669.61 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 1659.90 | 1669.04 | 1670.20 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2025-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 13:15:00 | 1697.60 | 1671.61 | 1669.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 1729.40 | 1683.17 | 1675.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 1685.30 | 1705.36 | 1696.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 1685.30 | 1705.36 | 1696.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1685.30 | 1705.36 | 1696.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 1685.30 | 1705.36 | 1696.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1685.70 | 1701.43 | 1695.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1684.40 | 1701.43 | 1695.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1686.50 | 1696.44 | 1693.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 1686.50 | 1696.44 | 1693.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1688.00 | 1694.75 | 1693.32 | EMA400 retest candle locked (from upside) |

### Cycle 198 — SELL (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 14:15:00 | 1680.40 | 1691.88 | 1692.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 15:15:00 | 1676.90 | 1688.88 | 1690.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1654.80 | 1645.33 | 1658.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 1654.80 | 1645.33 | 1658.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1654.80 | 1645.33 | 1658.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 1654.80 | 1645.33 | 1658.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1651.70 | 1646.60 | 1657.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 1641.90 | 1649.15 | 1656.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 15:15:00 | 1644.00 | 1635.74 | 1634.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 199 — BUY (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 15:15:00 | 1644.00 | 1635.74 | 1634.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 09:15:00 | 1666.10 | 1641.81 | 1637.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 15:15:00 | 1657.00 | 1657.75 | 1649.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 09:15:00 | 1645.20 | 1657.75 | 1649.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1646.90 | 1655.58 | 1649.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 13:15:00 | 1650.20 | 1651.09 | 1648.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 1633.20 | 1647.05 | 1647.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 1633.20 | 1647.05 | 1647.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 1628.70 | 1637.16 | 1641.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 1648.30 | 1638.43 | 1641.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 1648.30 | 1638.43 | 1641.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1648.30 | 1638.43 | 1641.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 1648.30 | 1638.43 | 1641.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1659.40 | 1642.63 | 1642.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 1659.40 | 1642.63 | 1642.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 201 — BUY (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 13:15:00 | 1656.90 | 1645.48 | 1644.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 1675.00 | 1658.07 | 1651.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 15:15:00 | 1651.30 | 1658.75 | 1653.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 15:15:00 | 1651.30 | 1658.75 | 1653.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 1651.30 | 1658.75 | 1653.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:30:00 | 1674.10 | 1661.88 | 1655.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 1692.00 | 1701.71 | 1702.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 202 — SELL (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 11:15:00 | 1692.00 | 1701.71 | 1702.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 14:15:00 | 1677.00 | 1692.50 | 1697.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 1719.00 | 1674.47 | 1683.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 1719.00 | 1674.47 | 1683.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 1719.00 | 1674.47 | 1683.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 1719.00 | 1674.47 | 1683.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1720.00 | 1683.58 | 1687.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 1721.20 | 1683.58 | 1687.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 1717.00 | 1690.26 | 1689.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 09:15:00 | 1723.00 | 1696.81 | 1692.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 14:15:00 | 1698.00 | 1719.13 | 1707.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 14:15:00 | 1698.00 | 1719.13 | 1707.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1698.00 | 1719.13 | 1707.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 1698.00 | 1719.13 | 1707.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1687.00 | 1712.71 | 1705.76 | EMA400 retest candle locked (from upside) |

### Cycle 204 — SELL (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 12:15:00 | 1697.00 | 1701.58 | 1701.82 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 1712.00 | 1703.66 | 1702.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 1728.40 | 1708.61 | 1705.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 10:15:00 | 1709.10 | 1709.79 | 1706.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 11:00:00 | 1709.10 | 1709.79 | 1706.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 11:15:00 | 1708.30 | 1709.49 | 1706.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 11:30:00 | 1705.50 | 1709.49 | 1706.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 1715.20 | 1710.63 | 1707.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 12:30:00 | 1710.10 | 1710.63 | 1707.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 1710.70 | 1710.65 | 1707.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:30:00 | 1707.10 | 1710.65 | 1707.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1740.00 | 1716.52 | 1710.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:15:00 | 1715.40 | 1716.52 | 1710.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1715.40 | 1716.29 | 1711.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:30:00 | 1715.60 | 1717.38 | 1712.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 1712.30 | 1716.36 | 1712.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:00:00 | 1712.30 | 1716.36 | 1712.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 1697.20 | 1712.53 | 1710.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:45:00 | 1698.00 | 1712.53 | 1710.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 206 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 1695.40 | 1709.10 | 1709.39 | EMA200 below EMA400 |

### Cycle 207 — BUY (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 12:15:00 | 1722.00 | 1708.48 | 1707.96 | EMA200 above EMA400 |

### Cycle 208 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 1701.00 | 1708.57 | 1708.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1697.10 | 1704.89 | 1707.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 13:15:00 | 1702.10 | 1699.52 | 1702.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 13:15:00 | 1702.10 | 1699.52 | 1702.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1702.10 | 1699.52 | 1702.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 14:00:00 | 1702.10 | 1699.52 | 1702.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 1697.60 | 1699.14 | 1701.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 1682.80 | 1698.51 | 1701.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 12:15:00 | 1598.66 | 1632.94 | 1658.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-13 14:15:00 | 1563.90 | 1563.43 | 1584.74 | SL hit (close>ema200) qty=0.50 sl=1563.43 alert=retest2 |

### Cycle 209 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 1331.10 | 1310.39 | 1307.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 1334.80 | 1319.69 | 1313.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 1315.60 | 1322.70 | 1316.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 1315.60 | 1322.70 | 1316.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1315.60 | 1322.70 | 1316.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1315.60 | 1322.70 | 1316.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1285.00 | 1315.16 | 1313.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 1287.70 | 1315.16 | 1313.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1283.90 | 1308.91 | 1310.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1278.50 | 1302.83 | 1307.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1272.10 | 1259.80 | 1276.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1272.10 | 1259.80 | 1276.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1272.10 | 1259.80 | 1276.40 | EMA400 retest candle locked (from downside) |

### Cycle 211 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 1320.20 | 1289.76 | 1285.68 | EMA200 above EMA400 |

### Cycle 212 — SELL (started 2026-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 13:15:00 | 1272.90 | 1282.22 | 1283.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 1242.70 | 1263.36 | 1271.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 13:15:00 | 1254.30 | 1254.19 | 1263.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 13:45:00 | 1254.50 | 1254.19 | 1263.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 15:15:00 | 1256.00 | 1255.29 | 1262.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:15:00 | 1286.80 | 1255.29 | 1262.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1267.80 | 1257.79 | 1263.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:30:00 | 1269.30 | 1257.79 | 1263.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 213 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 1281.80 | 1268.93 | 1267.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 1313.30 | 1283.42 | 1275.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 13:15:00 | 1292.50 | 1293.65 | 1283.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 13:45:00 | 1291.10 | 1293.65 | 1283.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1282.90 | 1292.09 | 1285.45 | EMA400 retest candle locked (from upside) |

### Cycle 214 — SELL (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 14:15:00 | 1274.10 | 1281.77 | 1282.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 1248.00 | 1273.93 | 1278.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 14:15:00 | 1284.70 | 1263.64 | 1270.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 14:15:00 | 1284.70 | 1263.64 | 1270.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 1284.70 | 1263.64 | 1270.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-12 15:00:00 | 1284.70 | 1263.64 | 1270.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 1270.20 | 1264.96 | 1270.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 1243.20 | 1264.96 | 1270.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-16 09:15:00 | 1181.04 | 1229.33 | 1247.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 15:15:00 | 1163.00 | 1161.83 | 1184.08 | SL hit (close>ema200) qty=0.50 sl=1161.83 alert=retest2 |

### Cycle 215 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 1005.90 | 999.37 | 998.89 | EMA200 above EMA400 |

### Cycle 216 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 994.80 | 998.49 | 998.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 988.30 | 996.45 | 997.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 988.20 | 985.92 | 991.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 988.20 | 985.92 | 991.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 979.40 | 984.61 | 990.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 13:45:00 | 976.50 | 983.05 | 989.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 09:15:00 | 927.67 | 937.25 | 957.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-17 09:15:00 | 878.85 | 900.25 | 924.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 217 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 918.20 | 889.10 | 886.03 | EMA200 above EMA400 |

### Cycle 218 — SELL (started 2026-03-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 11:15:00 | 871.30 | 886.57 | 886.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 12:15:00 | 864.40 | 882.13 | 884.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 13:15:00 | 883.80 | 882.47 | 884.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 13:15:00 | 883.80 | 882.47 | 884.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 883.80 | 882.47 | 884.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:45:00 | 884.80 | 882.47 | 884.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 864.00 | 878.77 | 882.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:00:00 | 856.00 | 871.45 | 878.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 852.00 | 867.60 | 876.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 891.70 | 873.64 | 875.36 | SL hit (close>static) qty=1.00 sl=885.00 alert=retest2 |

### Cycle 219 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 895.00 | 877.91 | 877.14 | EMA200 above EMA400 |

### Cycle 220 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 859.10 | 876.90 | 878.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 853.40 | 866.85 | 872.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 862.70 | 830.79 | 844.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 862.70 | 830.79 | 844.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 862.70 | 830.79 | 844.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 862.70 | 830.79 | 844.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 860.80 | 836.79 | 845.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 870.10 | 836.79 | 845.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 221 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 864.10 | 851.25 | 850.85 | EMA200 above EMA400 |

### Cycle 222 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 840.60 | 849.86 | 850.65 | EMA200 below EMA400 |

### Cycle 223 — BUY (started 2026-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 12:15:00 | 865.70 | 853.53 | 852.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 13:15:00 | 879.10 | 858.64 | 854.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 855.55 | 862.50 | 857.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 855.55 | 862.50 | 857.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 855.55 | 862.50 | 857.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 857.70 | 862.50 | 857.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 865.95 | 863.19 | 858.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:45:00 | 878.05 | 867.54 | 861.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 15:00:00 | 878.70 | 870.52 | 863.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 10:15:00 | 923.60 | 931.36 | 931.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 224 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 923.60 | 931.36 | 931.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 11:15:00 | 914.90 | 928.07 | 929.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-16 14:15:00 | 923.20 | 922.83 | 926.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-16 14:45:00 | 921.85 | 922.83 | 926.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 933.85 | 925.22 | 927.13 | EMA400 retest candle locked (from downside) |

### Cycle 225 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 945.65 | 930.94 | 929.50 | EMA200 above EMA400 |

### Cycle 226 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 922.90 | 932.55 | 933.68 | EMA200 below EMA400 |

### Cycle 227 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 937.70 | 934.47 | 934.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 11:15:00 | 943.25 | 936.23 | 935.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 13:15:00 | 930.90 | 935.78 | 935.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 13:15:00 | 930.90 | 935.78 | 935.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 930.90 | 935.78 | 935.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 930.90 | 935.78 | 935.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 932.15 | 935.06 | 934.95 | EMA400 retest candle locked (from upside) |

### Cycle 228 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 932.05 | 934.53 | 934.73 | EMA200 below EMA400 |

### Cycle 229 — BUY (started 2026-04-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 11:15:00 | 937.20 | 935.22 | 935.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 956.35 | 939.45 | 936.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 948.95 | 954.03 | 948.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 948.95 | 954.03 | 948.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 948.95 | 954.03 | 948.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 14:45:00 | 948.40 | 954.03 | 948.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 952.00 | 953.63 | 949.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 935.00 | 953.63 | 949.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 933.45 | 949.59 | 947.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:45:00 | 926.45 | 949.59 | 947.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 230 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 922.00 | 944.07 | 945.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 12:15:00 | 918.15 | 935.81 | 941.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 953.95 | 933.85 | 937.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 953.95 | 933.85 | 937.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 953.95 | 933.85 | 937.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 951.10 | 933.85 | 937.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 954.15 | 937.91 | 939.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 958.35 | 937.91 | 939.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 231 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 948.25 | 941.66 | 940.93 | EMA200 above EMA400 |

### Cycle 232 — SELL (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 13:15:00 | 927.00 | 938.51 | 940.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 915.50 | 931.13 | 934.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 937.60 | 927.51 | 930.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 937.60 | 927.51 | 930.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 937.60 | 927.51 | 930.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 937.60 | 927.51 | 930.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 945.80 | 931.17 | 931.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 945.80 | 931.17 | 931.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 233 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 935.40 | 932.01 | 931.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 961.00 | 942.36 | 938.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 12:15:00 | 947.30 | 947.40 | 941.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 13:00:00 | 947.30 | 947.40 | 941.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 959.45 | 963.05 | 955.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:45:00 | 960.85 | 963.05 | 955.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 958.05 | 961.27 | 957.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 958.05 | 961.27 | 957.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 957.00 | 960.42 | 957.58 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-15 15:15:00 | 1072.00 | 2023-05-18 11:15:00 | 1076.10 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2023-05-16 13:00:00 | 1070.95 | 2023-05-18 11:15:00 | 1076.10 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-05-22 09:15:00 | 1088.40 | 2023-05-22 15:15:00 | 1066.95 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2023-05-23 09:15:00 | 1083.85 | 2023-05-23 11:15:00 | 1069.65 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2023-06-01 11:00:00 | 1079.05 | 2023-06-01 11:15:00 | 1090.90 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2023-06-05 09:15:00 | 1107.90 | 2023-06-16 12:15:00 | 1182.50 | STOP_HIT | 1.00 | 6.73% |
| BUY | retest2 | 2023-06-05 10:30:00 | 1106.00 | 2023-06-16 12:15:00 | 1182.50 | STOP_HIT | 1.00 | 6.92% |
| BUY | retest2 | 2023-07-10 11:45:00 | 1234.05 | 2023-07-13 09:15:00 | 1357.46 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-07-10 12:30:00 | 1233.60 | 2023-07-13 09:15:00 | 1356.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-08-07 09:15:00 | 1611.75 | 2023-08-10 15:15:00 | 1583.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2023-08-17 09:30:00 | 1670.15 | 2023-08-21 14:15:00 | 1652.90 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-08-17 10:00:00 | 1670.00 | 2023-08-21 14:15:00 | 1652.90 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2023-08-18 11:00:00 | 1670.70 | 2023-08-21 14:15:00 | 1652.90 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2023-08-23 09:15:00 | 1662.35 | 2023-08-30 12:15:00 | 1734.60 | STOP_HIT | 1.00 | 4.35% |
| BUY | retest2 | 2023-08-23 10:15:00 | 1659.95 | 2023-08-30 12:15:00 | 1734.60 | STOP_HIT | 1.00 | 4.50% |
| BUY | retest2 | 2023-08-23 13:30:00 | 1661.65 | 2023-08-30 12:15:00 | 1734.60 | STOP_HIT | 1.00 | 4.39% |
| BUY | retest2 | 2023-08-24 09:15:00 | 1717.40 | 2023-08-30 12:15:00 | 1734.60 | STOP_HIT | 1.00 | 1.00% |
| BUY | retest2 | 2023-08-30 09:15:00 | 1752.70 | 2023-08-30 12:15:00 | 1734.60 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2023-08-30 10:15:00 | 1749.50 | 2023-08-30 12:15:00 | 1734.60 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2023-09-06 15:15:00 | 1768.60 | 2023-09-07 09:15:00 | 1748.40 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2023-10-03 10:30:00 | 2024.85 | 2023-10-04 13:15:00 | 2227.34 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2023-10-16 11:00:00 | 2020.00 | 2023-10-18 09:15:00 | 2046.45 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest1 | 2023-10-16 15:15:00 | 2014.00 | 2023-10-18 09:15:00 | 2046.45 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2023-10-20 12:45:00 | 1987.40 | 2023-10-23 12:15:00 | 1888.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 13:30:00 | 1970.35 | 2023-10-23 15:15:00 | 1871.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 12:45:00 | 1987.40 | 2023-10-25 09:15:00 | 1976.60 | STOP_HIT | 0.50 | 0.54% |
| SELL | retest2 | 2023-10-20 13:30:00 | 1970.35 | 2023-10-25 09:15:00 | 1976.60 | STOP_HIT | 0.50 | -0.32% |
| BUY | retest2 | 2023-10-30 10:30:00 | 2052.05 | 2023-11-06 11:15:00 | 2100.45 | STOP_HIT | 1.00 | 2.36% |
| SELL | retest2 | 2023-11-08 15:15:00 | 2085.00 | 2023-11-09 11:15:00 | 2097.65 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2023-11-10 14:00:00 | 2067.40 | 2023-11-12 18:15:00 | 2106.80 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2023-11-20 14:15:00 | 2226.00 | 2023-11-22 11:15:00 | 2192.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2023-11-28 09:15:00 | 2221.85 | 2023-11-28 12:15:00 | 2187.50 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2023-11-28 10:00:00 | 2212.70 | 2023-11-28 12:15:00 | 2187.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2023-11-29 11:30:00 | 2177.00 | 2023-11-30 14:15:00 | 2222.50 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2023-12-04 09:15:00 | 2226.55 | 2023-12-05 11:15:00 | 2194.65 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2023-12-04 11:00:00 | 2234.95 | 2023-12-05 11:15:00 | 2194.65 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2023-12-05 09:45:00 | 2219.75 | 2023-12-05 11:15:00 | 2194.65 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-12-06 13:15:00 | 2179.30 | 2023-12-06 14:15:00 | 2215.55 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2023-12-12 10:15:00 | 2173.45 | 2023-12-20 13:15:00 | 2064.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-12 12:45:00 | 2173.65 | 2023-12-20 13:15:00 | 2064.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-12-12 10:15:00 | 2173.45 | 2023-12-21 14:15:00 | 2034.05 | STOP_HIT | 0.50 | 6.41% |
| SELL | retest2 | 2023-12-12 12:45:00 | 2173.65 | 2023-12-21 14:15:00 | 2034.05 | STOP_HIT | 0.50 | 6.42% |
| BUY | retest2 | 2024-01-11 10:30:00 | 2017.35 | 2024-01-17 10:15:00 | 2000.05 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-01-11 12:30:00 | 2011.05 | 2024-01-17 10:15:00 | 2000.05 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-01-11 15:15:00 | 2039.35 | 2024-01-17 10:15:00 | 2000.05 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-01-16 14:45:00 | 2040.00 | 2024-01-17 10:15:00 | 2000.05 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-01-24 10:30:00 | 2055.00 | 2024-01-25 10:15:00 | 2003.80 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-01-24 12:30:00 | 2050.00 | 2024-01-25 10:15:00 | 2003.80 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-01-29 10:15:00 | 1991.20 | 2024-02-01 09:15:00 | 1891.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-01-29 10:15:00 | 1991.20 | 2024-02-02 09:15:00 | 1792.08 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2024-02-16 09:15:00 | 1900.75 | 2024-02-16 15:15:00 | 1866.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-02-26 09:45:00 | 1816.95 | 2024-02-28 09:15:00 | 1837.25 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-02-27 13:15:00 | 1816.80 | 2024-02-28 09:15:00 | 1837.25 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-02-27 15:00:00 | 1819.65 | 2024-02-28 09:15:00 | 1837.25 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-03-02 09:15:00 | 1850.00 | 2024-03-05 09:15:00 | 1809.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-03-04 10:45:00 | 1845.00 | 2024-03-05 09:15:00 | 1809.00 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest1 | 2024-03-06 09:30:00 | 1789.80 | 2024-03-11 09:15:00 | 1700.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-03-06 09:30:00 | 1789.80 | 2024-03-13 15:15:00 | 1610.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-03-14 13:30:00 | 1670.40 | 2024-03-15 12:15:00 | 1739.00 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2024-03-14 15:00:00 | 1655.00 | 2024-03-15 12:15:00 | 1739.00 | STOP_HIT | 1.00 | -5.08% |
| SELL | retest2 | 2024-03-15 09:30:00 | 1666.05 | 2024-03-15 12:15:00 | 1739.00 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2024-03-15 10:30:00 | 1667.60 | 2024-03-15 12:15:00 | 1739.00 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest1 | 2024-03-20 09:15:00 | 1747.80 | 2024-03-22 09:15:00 | 1835.19 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-03-20 09:15:00 | 1747.80 | 2024-03-26 10:15:00 | 1841.20 | STOP_HIT | 0.50 | 5.34% |
| BUY | retest2 | 2024-03-28 14:00:00 | 1853.00 | 2024-04-08 13:15:00 | 1865.80 | STOP_HIT | 1.00 | 0.69% |
| BUY | retest2 | 2024-03-28 14:30:00 | 1853.55 | 2024-04-08 13:15:00 | 1865.80 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2024-04-22 13:30:00 | 1917.15 | 2024-05-02 09:15:00 | 1993.45 | STOP_HIT | 1.00 | 3.98% |
| BUY | retest2 | 2024-04-22 14:45:00 | 1915.85 | 2024-05-02 09:15:00 | 1993.45 | STOP_HIT | 1.00 | 4.05% |
| BUY | retest2 | 2024-04-22 15:15:00 | 1926.00 | 2024-05-02 09:15:00 | 1993.45 | STOP_HIT | 1.00 | 3.50% |
| BUY | retest2 | 2024-05-22 12:30:00 | 2020.00 | 2024-05-23 10:15:00 | 1975.00 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2024-05-28 09:15:00 | 1965.00 | 2024-06-03 12:15:00 | 1940.25 | STOP_HIT | 1.00 | 1.26% |
| SELL | retest2 | 2024-06-05 09:30:00 | 1865.00 | 2024-06-06 09:15:00 | 1949.95 | STOP_HIT | 1.00 | -4.55% |
| BUY | retest2 | 2024-06-11 09:15:00 | 1980.00 | 2024-06-13 11:15:00 | 1974.35 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-06-11 12:00:00 | 1980.00 | 2024-06-13 11:15:00 | 1974.35 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-06-11 13:30:00 | 1978.55 | 2024-06-13 11:15:00 | 1974.35 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2024-06-12 13:15:00 | 1978.95 | 2024-06-13 11:15:00 | 1974.35 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-06-13 09:15:00 | 1995.65 | 2024-06-13 11:15:00 | 1974.35 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-06-25 11:00:00 | 2463.00 | 2024-06-26 09:15:00 | 2324.95 | STOP_HIT | 1.00 | -5.60% |
| BUY | retest2 | 2024-07-12 12:00:00 | 2442.00 | 2024-07-15 09:15:00 | 2686.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-12 12:30:00 | 2426.85 | 2024-07-15 09:15:00 | 2669.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-12 14:30:00 | 2427.95 | 2024-07-15 09:15:00 | 2670.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-15 09:15:00 | 2549.60 | 2024-07-18 09:15:00 | 2344.90 | STOP_HIT | 1.00 | -8.03% |
| SELL | retest2 | 2024-07-31 10:45:00 | 2380.20 | 2024-07-31 12:15:00 | 2419.20 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-07-31 11:15:00 | 2380.55 | 2024-07-31 12:15:00 | 2419.20 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest1 | 2024-08-06 10:30:00 | 2193.35 | 2024-08-07 10:15:00 | 2216.65 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest1 | 2024-08-06 11:30:00 | 2186.65 | 2024-08-07 10:15:00 | 2216.65 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest1 | 2024-08-06 13:00:00 | 2193.00 | 2024-08-07 10:15:00 | 2216.65 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest1 | 2024-08-06 13:30:00 | 2187.95 | 2024-08-07 10:15:00 | 2216.65 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-08-08 10:00:00 | 2188.20 | 2024-08-09 13:15:00 | 2231.45 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2024-08-08 10:45:00 | 2186.95 | 2024-08-09 14:15:00 | 2226.10 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-08-08 13:15:00 | 2193.70 | 2024-08-09 14:15:00 | 2226.10 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-08-08 13:45:00 | 2181.65 | 2024-08-09 14:15:00 | 2226.10 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-08-09 10:30:00 | 2187.05 | 2024-08-09 14:15:00 | 2226.10 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-08-22 12:00:00 | 2175.30 | 2024-08-28 12:15:00 | 2163.90 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2024-08-22 14:30:00 | 2175.05 | 2024-08-28 12:15:00 | 2163.90 | STOP_HIT | 1.00 | 0.51% |
| SELL | retest2 | 2024-08-28 12:00:00 | 2163.00 | 2024-08-28 12:15:00 | 2163.90 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2024-08-30 12:15:00 | 2097.05 | 2024-09-09 09:15:00 | 1992.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-02 13:15:00 | 2098.55 | 2024-09-09 09:15:00 | 1993.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-03 09:30:00 | 2097.10 | 2024-09-09 09:15:00 | 1992.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-30 12:15:00 | 2097.05 | 2024-09-10 09:15:00 | 2013.20 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2024-09-02 13:15:00 | 2098.55 | 2024-09-10 09:15:00 | 2013.20 | STOP_HIT | 0.50 | 4.07% |
| SELL | retest2 | 2024-09-03 09:30:00 | 2097.10 | 2024-09-10 09:15:00 | 2013.20 | STOP_HIT | 0.50 | 4.00% |
| BUY | retest1 | 2024-09-13 09:45:00 | 2090.40 | 2024-09-18 09:15:00 | 2062.80 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest1 | 2024-09-13 10:30:00 | 2089.30 | 2024-09-18 09:15:00 | 2062.80 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest1 | 2024-09-13 12:00:00 | 2086.05 | 2024-09-18 09:15:00 | 2062.80 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest1 | 2024-09-13 13:45:00 | 2092.30 | 2024-09-18 09:15:00 | 2062.80 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-09-20 12:45:00 | 2060.20 | 2024-09-23 09:15:00 | 2118.55 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2024-09-30 12:15:00 | 2138.85 | 2024-10-07 09:15:00 | 2097.05 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-10-04 09:30:00 | 2132.80 | 2024-10-07 09:15:00 | 2097.05 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-10-14 11:15:00 | 2159.00 | 2024-10-18 11:15:00 | 2151.70 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-10-14 15:00:00 | 2151.70 | 2024-10-18 11:15:00 | 2151.70 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2024-10-15 09:15:00 | 2174.40 | 2024-10-18 11:15:00 | 2151.70 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-10-18 10:15:00 | 2176.00 | 2024-10-18 11:15:00 | 2151.70 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest1 | 2024-10-22 10:00:00 | 2083.60 | 2024-10-25 12:15:00 | 1979.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-22 11:15:00 | 2081.05 | 2024-10-25 12:15:00 | 1977.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-10-22 10:00:00 | 2083.60 | 2024-10-29 14:15:00 | 1875.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-10-22 11:15:00 | 2081.05 | 2024-10-29 14:15:00 | 1872.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-31 09:15:00 | 1920.30 | 2024-10-31 12:15:00 | 1953.80 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2024-11-19 09:15:00 | 1734.65 | 2024-11-19 14:15:00 | 1647.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-19 09:15:00 | 1734.65 | 2024-11-22 11:15:00 | 1561.19 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-29 15:00:00 | 1754.55 | 2024-12-02 09:15:00 | 1692.95 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2024-12-12 13:45:00 | 1778.75 | 2024-12-13 09:15:00 | 1742.60 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-12-12 14:45:00 | 1780.00 | 2024-12-13 09:15:00 | 1742.60 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-12-31 09:15:00 | 1576.30 | 2025-01-01 11:15:00 | 1630.30 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2025-01-06 09:15:00 | 1643.45 | 2025-01-10 10:15:00 | 1637.65 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-01-14 12:15:00 | 1641.60 | 2025-01-14 14:15:00 | 1651.75 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-02-10 10:45:00 | 1654.00 | 2025-02-13 10:15:00 | 1680.20 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-02-11 10:30:00 | 1641.85 | 2025-02-13 10:15:00 | 1680.20 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-02-11 12:15:00 | 1651.60 | 2025-02-13 10:15:00 | 1680.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-02-11 14:15:00 | 1654.60 | 2025-02-13 10:15:00 | 1680.20 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-02-19 09:15:00 | 1679.45 | 2025-02-24 09:15:00 | 1653.30 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-02-27 09:15:00 | 1680.15 | 2025-02-27 14:15:00 | 1734.55 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2025-03-04 09:15:00 | 1643.20 | 2025-03-05 14:15:00 | 1680.40 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2025-03-04 10:30:00 | 1648.45 | 2025-03-05 14:15:00 | 1680.40 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2025-03-04 15:00:00 | 1639.05 | 2025-03-05 14:15:00 | 1680.40 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-03-11 09:15:00 | 1591.20 | 2025-03-12 13:15:00 | 1633.60 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-03-18 09:30:00 | 1588.20 | 2025-03-19 11:15:00 | 1632.90 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-03-18 15:00:00 | 1588.30 | 2025-03-19 11:15:00 | 1632.90 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-04-02 10:45:00 | 1703.30 | 2025-04-07 09:15:00 | 1597.95 | STOP_HIT | 1.00 | -6.19% |
| BUY | retest2 | 2025-04-02 14:45:00 | 1738.80 | 2025-04-07 09:15:00 | 1597.95 | STOP_HIT | 1.00 | -8.10% |
| BUY | retest2 | 2025-04-04 14:30:00 | 1699.60 | 2025-04-07 09:15:00 | 1597.95 | STOP_HIT | 1.00 | -5.98% |
| BUY | retest2 | 2025-04-11 09:15:00 | 1721.20 | 2025-04-25 09:15:00 | 1749.60 | STOP_HIT | 1.00 | 1.65% |
| BUY | retest2 | 2025-05-07 15:15:00 | 1865.00 | 2025-05-08 14:15:00 | 1815.10 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-05-08 11:00:00 | 1888.50 | 2025-05-08 14:15:00 | 1815.10 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2025-05-26 11:00:00 | 1976.10 | 2025-06-02 09:15:00 | 1877.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-26 11:00:00 | 1976.10 | 2025-06-02 09:15:00 | 1938.90 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2025-06-06 09:15:00 | 1908.40 | 2025-06-09 14:15:00 | 1929.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-06-09 11:00:00 | 1909.90 | 2025-06-09 14:15:00 | 1929.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-06-09 12:00:00 | 1911.90 | 2025-06-09 14:15:00 | 1929.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-06-19 10:15:00 | 1733.40 | 2025-06-24 09:15:00 | 1760.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-06-20 09:15:00 | 1731.70 | 2025-06-24 09:15:00 | 1760.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-23 09:15:00 | 1730.50 | 2025-06-24 09:15:00 | 1760.00 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-07-07 13:00:00 | 1732.00 | 2025-07-07 14:15:00 | 1780.10 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-07-10 14:45:00 | 1792.30 | 2025-07-11 10:15:00 | 1767.20 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-07-11 15:00:00 | 1798.00 | 2025-07-18 09:15:00 | 1780.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-07-14 14:45:00 | 1799.70 | 2025-07-18 09:15:00 | 1780.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-07-15 10:15:00 | 1796.00 | 2025-07-18 09:15:00 | 1780.10 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-07-15 15:00:00 | 1807.00 | 2025-07-18 09:15:00 | 1780.10 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-24 09:15:00 | 1824.30 | 2025-07-24 10:15:00 | 1812.10 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-31 15:15:00 | 1792.10 | 2025-08-08 10:15:00 | 1782.40 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-08-01 09:30:00 | 1805.80 | 2025-08-08 10:15:00 | 1782.40 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-08-13 14:45:00 | 1797.20 | 2025-08-13 15:15:00 | 1776.40 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-09-02 14:00:00 | 1644.10 | 2025-09-08 09:15:00 | 1649.20 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-09-03 10:45:00 | 1646.00 | 2025-09-08 09:15:00 | 1649.20 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2025-09-03 14:15:00 | 1643.40 | 2025-09-08 09:15:00 | 1649.20 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-09-04 13:30:00 | 1642.60 | 2025-09-08 09:15:00 | 1649.20 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest1 | 2025-09-24 09:15:00 | 1623.40 | 2025-09-24 09:15:00 | 1644.30 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-09-24 11:30:00 | 1638.00 | 2025-09-30 14:15:00 | 1649.90 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-09-25 10:45:00 | 1637.30 | 2025-09-30 14:15:00 | 1649.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-29 15:15:00 | 1615.00 | 2025-09-30 14:15:00 | 1649.90 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-10-09 14:45:00 | 1699.90 | 2025-10-13 09:15:00 | 1869.89 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-10 09:15:00 | 1703.80 | 2025-10-13 09:15:00 | 1874.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-10 11:30:00 | 1700.10 | 2025-10-13 09:15:00 | 1870.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-10 14:00:00 | 1706.80 | 2025-10-13 09:15:00 | 1877.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-16 12:45:00 | 1845.80 | 2025-10-17 09:15:00 | 1804.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-10-16 14:15:00 | 1845.30 | 2025-10-17 09:15:00 | 1804.00 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-10-16 14:45:00 | 1849.60 | 2025-10-17 09:15:00 | 1804.00 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-10-27 11:00:00 | 1813.00 | 2025-10-30 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-10-27 13:00:00 | 1810.00 | 2025-10-30 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-10-28 13:00:00 | 1811.50 | 2025-10-30 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-28 15:15:00 | 1810.00 | 2025-10-30 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-10-29 09:45:00 | 1793.00 | 2025-10-30 14:15:00 | 1829.90 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-11-27 10:15:00 | 1664.10 | 2025-11-28 11:15:00 | 1687.70 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-27 11:15:00 | 1662.00 | 2025-11-28 11:15:00 | 1687.70 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-27 12:15:00 | 1664.10 | 2025-11-28 11:15:00 | 1687.70 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-27 13:00:00 | 1661.40 | 2025-11-28 11:15:00 | 1687.70 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-12-10 09:15:00 | 1641.90 | 2025-12-12 15:15:00 | 1644.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2025-12-16 13:15:00 | 1650.20 | 2025-12-16 15:15:00 | 1633.20 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-12-22 09:30:00 | 1674.10 | 2025-12-26 11:15:00 | 1692.00 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1682.80 | 2026-01-09 12:15:00 | 1598.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1682.80 | 2026-01-13 14:15:00 | 1563.90 | STOP_HIT | 0.50 | 7.07% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1243.20 | 2026-02-16 09:15:00 | 1181.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 1243.20 | 2026-02-17 15:15:00 | 1163.00 | STOP_HIT | 0.50 | 6.45% |
| SELL | retest2 | 2026-03-12 13:45:00 | 976.50 | 2026-03-16 09:15:00 | 927.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 13:45:00 | 976.50 | 2026-03-17 09:15:00 | 878.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-24 10:00:00 | 856.00 | 2026-03-25 09:15:00 | 891.70 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2026-03-24 10:30:00 | 852.00 | 2026-03-25 09:15:00 | 891.70 | STOP_HIT | 1.00 | -4.66% |
| BUY | retest2 | 2026-04-06 12:45:00 | 878.05 | 2026-04-16 10:15:00 | 923.60 | STOP_HIT | 1.00 | 5.19% |
| BUY | retest2 | 2026-04-06 15:00:00 | 878.70 | 2026-04-16 10:15:00 | 923.60 | STOP_HIT | 1.00 | 5.11% |
