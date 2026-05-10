# Kajaria Ceramics Ltd. (KAJARIACER)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1105.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 63 |
| ALERT1 | 39 |
| ALERT2 | 39 |
| ALERT2_SKIP | 19 |
| ALERT3 | 131 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 59 |
| PARTIAL | 8 |
| TARGET_HIT | 3 |
| STOP_HIT | 57 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 36 / 32
- **Target hits / Stop hits / Partials:** 3 / 57 / 8
- **Avg / median % per leg:** 1.29% / 0.66%
- **Sum % (uncompounded):** 87.84%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 12 | 42.9% | 3 | 25 | 0 | 1.21% | 33.8% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -2.08% | -4.2% |
| BUY @ 3rd Alert (retest2) | 26 | 12 | 46.2% | 3 | 23 | 0 | 1.46% | 38.0% |
| SELL (all) | 40 | 24 | 60.0% | 0 | 32 | 8 | 1.35% | 54.0% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | -0.53% | -1.6% |
| SELL @ 3rd Alert (retest2) | 37 | 22 | 59.5% | 0 | 29 | 8 | 1.50% | 55.6% |
| retest1 (combined) | 5 | 2 | 40.0% | 0 | 5 | 0 | -1.15% | -5.7% |
| retest2 (combined) | 63 | 34 | 54.0% | 3 | 52 | 8 | 1.49% | 93.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 14:15:00 | 1031.50 | 1043.35 | 1044.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 09:15:00 | 1027.10 | 1038.53 | 1041.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 1018.50 | 1017.22 | 1024.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 1025.50 | 1017.00 | 1020.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1025.50 | 1017.00 | 1020.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 1025.50 | 1017.00 | 1020.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 1026.50 | 1018.90 | 1021.21 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 13:15:00 | 1027.80 | 1023.15 | 1022.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 15:15:00 | 1039.50 | 1027.66 | 1024.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 11:15:00 | 1055.30 | 1055.39 | 1044.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 11:45:00 | 1055.00 | 1055.39 | 1044.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 1045.70 | 1050.83 | 1048.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 1045.70 | 1050.83 | 1048.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 1050.50 | 1050.77 | 1048.54 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 11:15:00 | 1040.00 | 1046.52 | 1047.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 15:15:00 | 1035.00 | 1041.72 | 1044.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 14:15:00 | 1035.40 | 1034.84 | 1038.96 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1028.80 | 1034.65 | 1038.50 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-06-17 10:00:00 | 1029.90 | 1033.70 | 1037.72 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1023.10 | 1021.86 | 1025.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 1026.80 | 1021.86 | 1025.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 1018.00 | 1020.34 | 1023.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 1018.00 | 1020.34 | 1023.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1021.00 | 1019.94 | 1022.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 1025.10 | 1021.20 | 1022.16 | SL hit (close>ema400) qty=1.00 sl=1022.16 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 1025.10 | 1021.20 | 1022.16 | SL hit (close>ema400) qty=1.00 sl=1022.16 alert=retest1 |

### Cycle 4 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1035.40 | 1023.58 | 1022.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 13:15:00 | 1050.00 | 1030.79 | 1026.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 15:15:00 | 1090.10 | 1102.95 | 1083.93 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 13:45:00 | 1116.60 | 1101.49 | 1089.95 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-26 14:30:00 | 1121.00 | 1104.73 | 1092.48 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 1098.40 | 1104.58 | 1095.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:45:00 | 1097.90 | 1104.58 | 1095.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 1095.50 | 1102.77 | 1095.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-27 11:15:00 | 1095.50 | 1102.77 | 1095.63 | SL hit (close<ema400) qty=1.00 sl=1095.63 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-06-27 11:15:00 | 1095.50 | 1102.77 | 1095.63 | SL hit (close<ema400) qty=1.00 sl=1095.63 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-06-27 11:30:00 | 1096.10 | 1102.77 | 1095.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 1095.10 | 1101.23 | 1095.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:30:00 | 1095.40 | 1101.23 | 1095.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1094.20 | 1099.83 | 1095.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:00:00 | 1094.20 | 1099.83 | 1095.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 1087.00 | 1097.26 | 1094.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 1087.00 | 1097.26 | 1094.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1085.00 | 1094.81 | 1093.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1083.40 | 1094.81 | 1093.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1073.10 | 1090.47 | 1091.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 1069.90 | 1079.89 | 1085.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 13:15:00 | 1075.50 | 1074.47 | 1080.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 14:00:00 | 1075.50 | 1074.47 | 1080.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1075.00 | 1074.88 | 1079.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1086.00 | 1074.88 | 1079.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1107.80 | 1081.47 | 1082.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 1107.80 | 1081.47 | 1082.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 1110.90 | 1087.35 | 1084.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 10:15:00 | 1146.60 | 1119.66 | 1104.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 09:15:00 | 1151.10 | 1156.85 | 1146.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 10:00:00 | 1151.10 | 1156.85 | 1146.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1154.50 | 1161.74 | 1154.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 1152.30 | 1161.74 | 1154.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1169.70 | 1163.33 | 1155.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 12:45:00 | 1171.80 | 1165.53 | 1158.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 13:15:00 | 1172.90 | 1165.53 | 1158.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 10:00:00 | 1186.10 | 1175.76 | 1166.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 1208.70 | 1179.96 | 1177.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1193.00 | 1182.57 | 1179.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:00:00 | 1218.70 | 1193.50 | 1189.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 1224.40 | 1198.20 | 1191.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:15:00 | 1222.80 | 1198.20 | 1191.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 12:00:00 | 1219.90 | 1202.54 | 1194.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1214.30 | 1214.04 | 1204.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 12:15:00 | 1218.90 | 1214.73 | 1206.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:30:00 | 1223.50 | 1215.84 | 1208.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 11:15:00 | 1216.70 | 1230.72 | 1231.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1200.00 | 1216.94 | 1223.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 1187.20 | 1174.01 | 1186.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 1187.20 | 1174.01 | 1186.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1187.20 | 1174.01 | 1186.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:00:00 | 1164.20 | 1172.05 | 1184.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 15:00:00 | 1161.80 | 1165.85 | 1177.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:30:00 | 1170.10 | 1167.43 | 1176.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 10:00:00 | 1169.50 | 1167.43 | 1176.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1170.10 | 1167.97 | 1175.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:00:00 | 1163.60 | 1167.09 | 1174.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1189.70 | 1170.62 | 1173.02 | SL hit (close>static) qty=1.00 sl=1179.70 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:15:00 | 1169.00 | 1172.74 | 1173.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:45:00 | 1164.10 | 1170.59 | 1172.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 11:15:00 | 1180.70 | 1171.08 | 1172.15 | SL hit (close>static) qty=1.00 sl=1179.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 11:15:00 | 1180.70 | 1171.08 | 1172.15 | SL hit (close>static) qty=1.00 sl=1179.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1184.80 | 1173.82 | 1173.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1184.80 | 1173.82 | 1173.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1184.80 | 1173.82 | 1173.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1184.80 | 1173.82 | 1173.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 1184.80 | 1173.82 | 1173.30 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 1160.00 | 1174.82 | 1176.70 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 1191.50 | 1178.22 | 1177.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 12:15:00 | 1195.10 | 1187.24 | 1184.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 1285.00 | 1291.61 | 1262.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 11:15:00 | 1279.90 | 1286.59 | 1265.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1279.90 | 1286.59 | 1265.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:30:00 | 1268.00 | 1286.59 | 1265.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 1268.00 | 1288.49 | 1279.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 13:00:00 | 1268.00 | 1288.49 | 1279.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 1271.40 | 1285.07 | 1278.65 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 15:15:00 | 1245.00 | 1272.97 | 1274.01 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 1296.00 | 1274.52 | 1273.08 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2025-08-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 14:15:00 | 1275.00 | 1281.62 | 1282.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 15:15:00 | 1272.80 | 1279.86 | 1281.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 1283.20 | 1280.52 | 1281.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 1283.20 | 1280.52 | 1281.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1283.20 | 1280.52 | 1281.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:30:00 | 1274.20 | 1279.23 | 1280.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:00:00 | 1274.30 | 1277.81 | 1279.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:45:00 | 1274.60 | 1276.34 | 1278.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 1271.10 | 1275.95 | 1277.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1283.70 | 1276.73 | 1277.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 1283.90 | 1276.73 | 1277.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1276.70 | 1276.72 | 1277.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 1275.30 | 1276.72 | 1277.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1210.49 | 1226.90 | 1240.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1210.58 | 1226.90 | 1240.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1210.87 | 1226.90 | 1240.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 1211.53 | 1226.90 | 1240.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 15:15:00 | 1207.54 | 1215.84 | 1228.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1229.10 | 1218.49 | 1228.63 | SL hit (close>ema200) qty=0.50 sl=1218.49 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1229.10 | 1218.49 | 1228.63 | SL hit (close>ema200) qty=0.50 sl=1218.49 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1229.10 | 1218.49 | 1228.63 | SL hit (close>ema200) qty=0.50 sl=1218.49 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1229.10 | 1218.49 | 1228.63 | SL hit (close>ema200) qty=0.50 sl=1218.49 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1229.10 | 1218.49 | 1228.63 | SL hit (close>ema200) qty=0.50 sl=1218.49 alert=retest2 |

### Cycle 14 — BUY (started 2025-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 13:15:00 | 1238.60 | 1225.92 | 1225.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 10:15:00 | 1242.80 | 1232.35 | 1228.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 15:15:00 | 1236.70 | 1238.27 | 1233.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 15:15:00 | 1236.70 | 1238.27 | 1233.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1236.70 | 1238.27 | 1233.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1236.50 | 1238.27 | 1233.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1235.50 | 1237.71 | 1233.80 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 1218.70 | 1230.11 | 1231.13 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 13:15:00 | 1241.80 | 1231.80 | 1231.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 09:15:00 | 1245.00 | 1235.13 | 1232.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 12:15:00 | 1234.70 | 1236.85 | 1234.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-09 13:00:00 | 1234.70 | 1236.85 | 1234.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 1243.00 | 1238.08 | 1235.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 14:30:00 | 1243.80 | 1238.62 | 1235.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 1249.40 | 1238.90 | 1236.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:15:00 | 1244.00 | 1250.74 | 1244.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 14:45:00 | 1243.60 | 1249.29 | 1244.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1244.20 | 1248.27 | 1244.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1256.10 | 1248.27 | 1244.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 10:45:00 | 1250.90 | 1247.65 | 1244.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1242.10 | 1245.76 | 1244.29 | SL hit (close<static) qty=1.00 sl=1242.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 1242.10 | 1245.76 | 1244.29 | SL hit (close<static) qty=1.00 sl=1242.20 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 1233.00 | 1241.41 | 1242.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 1233.00 | 1241.41 | 1242.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 1233.00 | 1241.41 | 1242.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 15:15:00 | 1233.00 | 1241.41 | 1242.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 15:15:00 | 1233.00 | 1241.41 | 1242.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 09:15:00 | 1227.60 | 1238.65 | 1241.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 13:15:00 | 1228.50 | 1225.27 | 1230.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 13:15:00 | 1228.50 | 1225.27 | 1230.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 1228.50 | 1225.27 | 1230.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 1231.20 | 1225.27 | 1230.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1225.10 | 1224.04 | 1228.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 1224.90 | 1224.04 | 1228.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1229.10 | 1225.05 | 1228.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 11:00:00 | 1229.10 | 1225.05 | 1228.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 1233.70 | 1226.78 | 1228.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 1233.70 | 1226.78 | 1228.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 12:15:00 | 1229.00 | 1227.22 | 1228.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:30:00 | 1232.70 | 1227.22 | 1228.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 13:15:00 | 1229.80 | 1227.74 | 1228.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 13:45:00 | 1229.30 | 1227.74 | 1228.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 14:15:00 | 1233.30 | 1228.85 | 1229.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 14:45:00 | 1232.60 | 1228.85 | 1229.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 1235.00 | 1230.08 | 1229.88 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 09:15:00 | 1225.10 | 1230.05 | 1230.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 13:15:00 | 1221.20 | 1226.23 | 1228.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 12:15:00 | 1215.70 | 1213.58 | 1218.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 12:45:00 | 1214.70 | 1213.58 | 1218.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 1199.60 | 1210.78 | 1216.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 1186.90 | 1206.81 | 1213.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:30:00 | 1197.00 | 1203.96 | 1210.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 12:45:00 | 1193.80 | 1199.91 | 1207.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 1219.10 | 1206.01 | 1206.25 | SL hit (close>static) qty=1.00 sl=1217.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 1219.10 | 1206.01 | 1206.25 | SL hit (close>static) qty=1.00 sl=1217.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 12:15:00 | 1219.10 | 1206.01 | 1206.25 | SL hit (close>static) qty=1.00 sl=1217.90 alert=retest2 |

### Cycle 20 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 1223.00 | 1209.41 | 1207.77 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 12:15:00 | 1198.40 | 1207.90 | 1208.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 1192.50 | 1203.43 | 1205.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 1197.60 | 1189.22 | 1194.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 1197.60 | 1189.22 | 1194.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1197.60 | 1189.22 | 1194.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 1194.50 | 1189.22 | 1194.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1187.90 | 1188.96 | 1194.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1184.00 | 1186.57 | 1192.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 1194.10 | 1180.17 | 1179.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 1194.10 | 1180.17 | 1179.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 13:15:00 | 1206.00 | 1191.70 | 1185.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 1193.30 | 1203.34 | 1197.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 15:15:00 | 1193.30 | 1203.34 | 1197.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 1193.30 | 1203.34 | 1197.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 09:15:00 | 1210.70 | 1203.34 | 1197.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:00:00 | 1210.30 | 1204.73 | 1198.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 12:45:00 | 1209.40 | 1207.10 | 1201.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 13:45:00 | 1209.20 | 1207.58 | 1202.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 1196.00 | 1205.26 | 1201.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 1196.00 | 1205.26 | 1201.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1208.00 | 1205.81 | 1202.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:15:00 | 1197.40 | 1205.81 | 1202.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1194.10 | 1203.47 | 1201.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:15:00 | 1190.20 | 1203.47 | 1201.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1199.00 | 1202.57 | 1201.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:30:00 | 1191.30 | 1202.57 | 1201.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1199.40 | 1201.94 | 1201.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:15:00 | 1199.10 | 1201.94 | 1201.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 1199.60 | 1201.47 | 1200.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:45:00 | 1197.60 | 1201.47 | 1200.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 1201.00 | 1202.42 | 1201.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 1201.00 | 1202.42 | 1201.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1205.20 | 1202.98 | 1201.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1199.20 | 1202.98 | 1201.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1207.60 | 1203.90 | 1202.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1204.10 | 1203.90 | 1202.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1231.50 | 1212.66 | 1207.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 12:30:00 | 1236.70 | 1223.32 | 1214.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1224.00 | 1235.53 | 1237.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1224.00 | 1235.53 | 1237.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1224.00 | 1235.53 | 1237.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1224.00 | 1235.53 | 1237.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 10:15:00 | 1224.00 | 1235.53 | 1237.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 10:15:00 | 1224.00 | 1235.53 | 1237.05 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 1251.40 | 1237.28 | 1236.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 1268.00 | 1252.07 | 1245.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 1252.60 | 1256.37 | 1249.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 12:15:00 | 1252.60 | 1256.37 | 1249.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 1252.60 | 1256.37 | 1249.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:30:00 | 1255.30 | 1256.37 | 1249.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1246.20 | 1254.33 | 1249.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 1248.60 | 1254.33 | 1249.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1243.20 | 1252.11 | 1248.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 1241.10 | 1252.11 | 1248.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 1248.00 | 1251.29 | 1248.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 1233.40 | 1251.29 | 1248.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 1228.00 | 1246.63 | 1246.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 10:15:00 | 1214.40 | 1240.18 | 1243.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-23 10:15:00 | 1229.70 | 1226.14 | 1232.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 10:15:00 | 1229.70 | 1226.14 | 1232.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 1229.70 | 1226.14 | 1232.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:45:00 | 1233.00 | 1226.14 | 1232.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 1233.70 | 1227.65 | 1232.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:00:00 | 1233.70 | 1227.65 | 1232.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 1234.00 | 1228.92 | 1232.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 1233.90 | 1228.92 | 1232.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 1236.70 | 1230.48 | 1232.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 1235.00 | 1230.48 | 1232.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1229.00 | 1230.18 | 1232.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:30:00 | 1239.20 | 1230.18 | 1232.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 1218.40 | 1212.45 | 1217.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:30:00 | 1216.70 | 1212.45 | 1217.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1217.10 | 1213.38 | 1217.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 1217.10 | 1213.38 | 1217.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1215.30 | 1213.76 | 1217.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 10:00:00 | 1212.60 | 1213.41 | 1216.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 1211.70 | 1213.54 | 1215.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 15:15:00 | 1219.30 | 1214.93 | 1215.84 | SL hit (close>static) qty=1.00 sl=1217.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-28 15:15:00 | 1219.30 | 1214.93 | 1215.84 | SL hit (close>static) qty=1.00 sl=1217.70 alert=retest2 |

### Cycle 26 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 1229.70 | 1217.88 | 1217.10 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1210.40 | 1217.92 | 1218.21 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 1227.50 | 1218.26 | 1217.92 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 12:15:00 | 1214.90 | 1217.30 | 1217.55 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 1219.40 | 1217.72 | 1217.72 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 1210.10 | 1216.20 | 1217.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 1175.50 | 1205.79 | 1212.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 15:15:00 | 1189.80 | 1186.87 | 1197.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 1189.80 | 1186.87 | 1197.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1186.70 | 1186.84 | 1196.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 1185.80 | 1186.84 | 1196.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1195.40 | 1188.55 | 1196.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:30:00 | 1195.80 | 1188.55 | 1196.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1120.70 | 1112.45 | 1117.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 1120.70 | 1112.45 | 1117.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1133.80 | 1116.72 | 1119.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1133.80 | 1116.72 | 1119.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 11:15:00 | 1137.60 | 1120.90 | 1120.74 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 13:15:00 | 1119.00 | 1122.76 | 1122.89 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-14 14:15:00 | 1124.20 | 1123.05 | 1123.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 15:15:00 | 1125.00 | 1123.44 | 1123.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-17 11:15:00 | 1119.00 | 1124.40 | 1123.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 11:15:00 | 1119.00 | 1124.40 | 1123.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 1119.00 | 1124.40 | 1123.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 1119.00 | 1124.40 | 1123.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 1122.50 | 1124.02 | 1123.72 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 13:15:00 | 1118.00 | 1122.82 | 1123.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1113.70 | 1119.95 | 1121.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 1084.40 | 1083.64 | 1089.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 13:45:00 | 1083.00 | 1083.64 | 1089.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 1072.10 | 1077.06 | 1084.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 1072.10 | 1077.06 | 1084.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 1087.80 | 1079.81 | 1084.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:45:00 | 1086.80 | 1079.81 | 1084.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1092.60 | 1082.36 | 1085.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:30:00 | 1102.30 | 1082.36 | 1085.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 09:15:00 | 1100.00 | 1088.42 | 1087.44 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 1088.90 | 1090.77 | 1090.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 15:15:00 | 1080.30 | 1087.67 | 1089.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 09:15:00 | 1083.00 | 1074.86 | 1079.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 1083.00 | 1074.86 | 1079.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 1083.00 | 1074.86 | 1079.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:45:00 | 1083.00 | 1074.86 | 1079.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 1081.40 | 1076.17 | 1079.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 10:30:00 | 1085.10 | 1076.17 | 1079.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1074.40 | 1075.82 | 1079.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:15:00 | 1069.50 | 1075.82 | 1079.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 13:15:00 | 1071.60 | 1075.73 | 1079.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:00:00 | 1071.90 | 1074.97 | 1078.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:45:00 | 1072.00 | 1074.95 | 1078.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1072.30 | 1069.61 | 1073.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 1072.30 | 1069.61 | 1073.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 1075.70 | 1070.83 | 1073.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 09:15:00 | 1069.80 | 1070.83 | 1073.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 1069.90 | 1069.98 | 1072.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 1053.90 | 1041.74 | 1040.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 1053.90 | 1041.74 | 1040.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 1053.90 | 1041.74 | 1040.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 1053.90 | 1041.74 | 1040.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 1053.90 | 1041.74 | 1040.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 1053.90 | 1041.74 | 1040.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 1053.90 | 1041.74 | 1040.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 1055.00 | 1044.39 | 1041.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 1075.20 | 1083.27 | 1068.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 10:00:00 | 1075.20 | 1083.27 | 1068.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1085.00 | 1087.11 | 1082.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 1079.30 | 1087.11 | 1082.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1079.00 | 1085.49 | 1082.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 1077.40 | 1085.49 | 1082.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1073.10 | 1083.01 | 1081.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:00:00 | 1073.10 | 1083.01 | 1081.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1074.90 | 1081.39 | 1080.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:30:00 | 1074.40 | 1081.39 | 1080.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2025-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 12:15:00 | 1072.90 | 1079.69 | 1080.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 14:15:00 | 1068.30 | 1076.47 | 1078.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 1065.60 | 1061.85 | 1067.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 09:15:00 | 1065.60 | 1061.85 | 1067.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1065.60 | 1061.85 | 1067.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 1030.00 | 1052.80 | 1060.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 10:15:00 | 1029.10 | 1049.34 | 1057.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 11:15:00 | 978.50 | 1008.82 | 1028.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-24 12:15:00 | 977.64 | 1003.52 | 1024.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 988.70 | 979.27 | 992.99 | SL hit (close>ema200) qty=0.50 sl=979.27 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 09:15:00 | 988.70 | 979.27 | 992.99 | SL hit (close>ema200) qty=0.50 sl=979.27 alert=retest2 |

### Cycle 40 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 979.85 | 969.82 | 968.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 10:15:00 | 983.70 | 976.26 | 972.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 15:15:00 | 1006.80 | 1008.73 | 1003.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 09:15:00 | 1000.40 | 1008.73 | 1003.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 1008.15 | 1008.61 | 1003.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 999.40 | 1008.61 | 1003.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 1005.00 | 1007.89 | 1003.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 1005.00 | 1007.89 | 1003.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 995.60 | 1005.43 | 1003.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:45:00 | 995.15 | 1005.43 | 1003.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 995.00 | 1003.34 | 1002.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:30:00 | 995.00 | 1003.34 | 1002.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 994.85 | 1001.65 | 1001.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 975.80 | 995.31 | 998.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 990.85 | 990.50 | 995.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 990.85 | 990.50 | 995.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 992.85 | 990.97 | 995.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 994.80 | 990.97 | 995.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 995.00 | 991.78 | 995.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 995.00 | 991.78 | 995.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 995.00 | 992.42 | 995.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 15:00:00 | 995.00 | 992.42 | 995.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 995.00 | 992.94 | 995.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 994.25 | 992.94 | 995.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 987.55 | 991.86 | 994.59 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 998.00 | 993.32 | 993.22 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 991.20 | 992.90 | 993.03 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 996.00 | 993.52 | 993.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 1001.60 | 995.13 | 994.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 994.00 | 996.95 | 995.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 15:15:00 | 994.00 | 996.95 | 995.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 994.00 | 996.95 | 995.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 985.90 | 996.95 | 995.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 989.00 | 995.36 | 994.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:45:00 | 986.70 | 995.36 | 994.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 987.15 | 993.72 | 994.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 980.60 | 990.47 | 992.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 985.30 | 970.76 | 975.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 985.30 | 970.76 | 975.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 985.30 | 970.76 | 975.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 985.30 | 970.76 | 975.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 988.75 | 974.36 | 976.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 988.75 | 974.36 | 976.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 991.70 | 980.53 | 979.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 994.00 | 983.22 | 980.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 968.30 | 983.46 | 981.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 968.30 | 983.46 | 981.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 968.30 | 983.46 | 981.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 968.30 | 983.46 | 981.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 956.85 | 978.14 | 979.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 948.80 | 972.27 | 976.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 12:15:00 | 889.00 | 886.37 | 901.73 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-29 15:15:00 | 882.40 | 887.85 | 899.75 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 894.60 | 888.33 | 897.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 891.90 | 888.33 | 897.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 903.65 | 892.51 | 896.39 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-30 14:15:00 | 903.65 | 892.51 | 896.39 | SL hit (close>ema400) qty=1.00 sl=896.39 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 903.65 | 892.51 | 896.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 894.90 | 892.98 | 896.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 915.50 | 892.98 | 896.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 915.55 | 897.50 | 898.01 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 913.35 | 900.67 | 899.41 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 891.45 | 897.56 | 898.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 878.00 | 893.65 | 896.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 895.80 | 888.58 | 891.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 14:15:00 | 895.80 | 888.58 | 891.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 895.80 | 888.58 | 891.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 895.80 | 888.58 | 891.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 894.65 | 889.80 | 892.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 905.45 | 889.80 | 892.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 906.10 | 893.06 | 893.38 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 909.80 | 896.40 | 894.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 912.40 | 899.60 | 896.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 912.65 | 913.37 | 906.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 10:45:00 | 911.75 | 913.37 | 906.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 908.95 | 911.71 | 907.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 908.65 | 911.71 | 907.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 907.95 | 910.96 | 907.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 14:45:00 | 907.00 | 910.96 | 907.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 912.50 | 911.27 | 907.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 907.60 | 911.27 | 907.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 900.85 | 909.18 | 906.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 900.85 | 909.18 | 906.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 904.35 | 908.22 | 906.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 11:15:00 | 905.00 | 908.22 | 906.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 939.95 | 947.70 | 948.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 939.95 | 947.70 | 948.48 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 953.00 | 949.44 | 949.17 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 13:15:00 | 946.55 | 948.86 | 948.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 14:15:00 | 941.20 | 947.33 | 948.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 944.85 | 936.64 | 940.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 944.85 | 936.64 | 940.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 944.85 | 936.64 | 940.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 944.85 | 936.64 | 940.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 941.40 | 937.59 | 940.20 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 963.10 | 944.24 | 942.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 14:15:00 | 968.10 | 949.01 | 945.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 990.00 | 990.59 | 976.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:30:00 | 990.65 | 990.59 | 976.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 983.15 | 989.00 | 981.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 15:00:00 | 995.00 | 987.76 | 983.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 11:15:00 | 972.30 | 986.70 | 987.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2026-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 11:15:00 | 972.30 | 986.70 | 987.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 963.80 | 982.12 | 985.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-26 09:15:00 | 962.50 | 957.88 | 965.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-26 10:00:00 | 962.50 | 957.88 | 965.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 965.15 | 959.34 | 965.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:00:00 | 965.15 | 959.34 | 965.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 966.70 | 960.81 | 965.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:00:00 | 960.40 | 962.40 | 965.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 10:15:00 | 961.95 | 962.68 | 964.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 969.25 | 963.99 | 965.32 | SL hit (close>static) qty=1.00 sl=968.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 969.25 | 963.99 | 965.32 | SL hit (close>static) qty=1.00 sl=968.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 13:30:00 | 960.70 | 962.69 | 964.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 912.66 | 957.30 | 961.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-04 12:15:00 | 941.70 | 937.52 | 945.04 | SL hit (close>ema200) qty=0.50 sl=937.52 alert=retest2 |

### Cycle 56 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 951.55 | 935.00 | 934.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 974.40 | 950.05 | 942.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 960.20 | 960.48 | 950.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:00:00 | 960.20 | 960.48 | 950.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 959.30 | 959.94 | 952.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 934.55 | 959.94 | 952.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 934.10 | 954.78 | 950.62 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 935.15 | 947.45 | 947.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 916.75 | 938.38 | 943.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 897.20 | 890.58 | 904.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-17 10:00:00 | 897.20 | 890.58 | 904.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 900.00 | 892.46 | 904.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 900.00 | 892.46 | 904.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 910.15 | 896.39 | 904.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 910.15 | 896.39 | 904.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 918.30 | 900.77 | 905.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:45:00 | 917.00 | 900.77 | 905.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 949.35 | 914.82 | 911.06 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 891.65 | 926.27 | 927.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 891.00 | 919.22 | 924.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 902.00 | 900.12 | 909.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 902.00 | 900.12 | 909.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 906.45 | 901.38 | 909.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 11:45:00 | 907.60 | 901.38 | 909.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 907.10 | 902.53 | 909.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 907.50 | 902.53 | 909.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 905.70 | 903.62 | 908.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 905.05 | 903.62 | 908.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 950.05 | 913.03 | 912.01 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 913.25 | 932.07 | 932.99 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 969.20 | 939.50 | 936.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 14:15:00 | 991.00 | 966.23 | 951.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 955.80 | 964.79 | 953.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 955.80 | 964.79 | 953.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 955.80 | 964.79 | 953.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 955.80 | 964.79 | 953.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 956.10 | 963.05 | 953.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 960.05 | 963.05 | 953.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:00:00 | 962.85 | 963.01 | 954.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-08 09:15:00 | 1056.06 | 1007.00 | 990.18 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-08 09:15:00 | 1059.14 | 1007.00 | 990.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2026-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 14:15:00 | 1181.95 | 1218.04 | 1220.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 09:15:00 | 1138.70 | 1196.08 | 1210.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 09:15:00 | 1107.60 | 1086.14 | 1117.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 09:30:00 | 1112.90 | 1086.14 | 1117.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1109.00 | 1104.73 | 1113.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 13:15:00 | 1104.50 | 1107.22 | 1112.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 13:45:00 | 1104.40 | 1106.66 | 1112.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 15:00:00 | 1098.10 | 1104.94 | 1110.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:45:00 | 1103.00 | 1103.84 | 1109.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1107.60 | 1103.96 | 1107.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 1107.60 | 1103.96 | 1107.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 1102.70 | 1103.70 | 1107.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:30:00 | 1105.00 | 1103.70 | 1107.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 1103.30 | 1103.62 | 1107.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 14:30:00 | 1108.30 | 1103.62 | 1107.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 1105.00 | 1103.90 | 1106.88 | EMA400 retest candle locked (from downside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 861.05 | 2025-05-16 09:15:00 | 947.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-06-17 09:15:00 | 1028.80 | 2025-06-20 15:15:00 | 1025.10 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest1 | 2025-06-17 10:00:00 | 1029.90 | 2025-06-20 15:15:00 | 1025.10 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest1 | 2025-06-26 13:45:00 | 1116.60 | 2025-06-27 11:15:00 | 1095.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest1 | 2025-06-26 14:30:00 | 1121.00 | 2025-06-27 11:15:00 | 1095.50 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-07-09 12:45:00 | 1171.80 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | 3.83% |
| BUY | retest2 | 2025-07-09 13:15:00 | 1172.90 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | 3.73% |
| BUY | retest2 | 2025-07-10 10:00:00 | 1186.10 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | 2.58% |
| BUY | retest2 | 2025-07-14 09:15:00 | 1208.70 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2025-07-17 10:00:00 | 1218.70 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-07-17 10:30:00 | 1224.40 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-07-17 11:15:00 | 1222.80 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-07-17 12:00:00 | 1219.90 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-07-18 12:15:00 | 1218.90 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2025-07-18 13:30:00 | 1223.50 | 2025-07-23 11:15:00 | 1216.70 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-07-28 11:00:00 | 1164.20 | 2025-07-30 09:15:00 | 1189.70 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-07-28 15:00:00 | 1161.80 | 2025-07-31 11:15:00 | 1180.70 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-07-29 09:30:00 | 1170.10 | 2025-07-31 11:15:00 | 1180.70 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-07-29 10:00:00 | 1169.50 | 2025-07-31 12:15:00 | 1184.80 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-07-29 12:00:00 | 1163.60 | 2025-07-31 12:15:00 | 1184.80 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-07-30 15:15:00 | 1169.00 | 2025-07-31 12:15:00 | 1184.80 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-07-31 09:45:00 | 1164.10 | 2025-07-31 12:15:00 | 1184.80 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-21 11:30:00 | 1274.20 | 2025-08-29 09:15:00 | 1210.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-21 15:00:00 | 1274.30 | 2025-08-29 09:15:00 | 1210.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 12:45:00 | 1274.60 | 2025-08-29 09:15:00 | 1210.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-22 15:15:00 | 1271.10 | 2025-08-29 09:15:00 | 1211.53 | PARTIAL | 0.50 | 4.69% |
| SELL | retest2 | 2025-08-25 11:15:00 | 1275.30 | 2025-08-29 15:15:00 | 1207.54 | PARTIAL | 0.50 | 5.31% |
| SELL | retest2 | 2025-08-21 11:30:00 | 1274.20 | 2025-09-01 09:15:00 | 1229.10 | STOP_HIT | 0.50 | 3.54% |
| SELL | retest2 | 2025-08-21 15:00:00 | 1274.30 | 2025-09-01 09:15:00 | 1229.10 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2025-08-22 12:45:00 | 1274.60 | 2025-09-01 09:15:00 | 1229.10 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2025-08-22 15:15:00 | 1271.10 | 2025-09-01 09:15:00 | 1229.10 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2025-08-25 11:15:00 | 1275.30 | 2025-09-01 09:15:00 | 1229.10 | STOP_HIT | 0.50 | 3.62% |
| BUY | retest2 | 2025-09-09 14:30:00 | 1243.80 | 2025-09-11 12:15:00 | 1242.10 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-09-10 09:15:00 | 1249.40 | 2025-09-11 12:15:00 | 1242.10 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-10 14:15:00 | 1244.00 | 2025-09-11 15:15:00 | 1233.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-10 14:45:00 | 1243.60 | 2025-09-11 15:15:00 | 1233.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-09-11 09:15:00 | 1256.10 | 2025-09-11 15:15:00 | 1233.00 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-09-11 10:45:00 | 1250.90 | 2025-09-11 15:15:00 | 1233.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-09-23 09:15:00 | 1186.90 | 2025-09-24 12:15:00 | 1219.10 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-09-23 10:30:00 | 1197.00 | 2025-09-24 12:15:00 | 1219.10 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-09-23 12:45:00 | 1193.80 | 2025-09-24 12:15:00 | 1219.10 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1184.00 | 2025-10-01 13:15:00 | 1194.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-10-07 09:15:00 | 1210.70 | 2025-10-15 10:15:00 | 1224.00 | STOP_HIT | 1.00 | 1.10% |
| BUY | retest2 | 2025-10-07 10:00:00 | 1210.30 | 2025-10-15 10:15:00 | 1224.00 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2025-10-07 12:45:00 | 1209.40 | 2025-10-15 10:15:00 | 1224.00 | STOP_HIT | 1.00 | 1.21% |
| BUY | retest2 | 2025-10-07 13:45:00 | 1209.20 | 2025-10-15 10:15:00 | 1224.00 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2025-10-10 12:30:00 | 1236.70 | 2025-10-15 10:15:00 | 1224.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-10-28 10:00:00 | 1212.60 | 2025-10-28 15:15:00 | 1219.30 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-10-28 14:00:00 | 1211.70 | 2025-10-28 15:15:00 | 1219.30 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-12-01 12:15:00 | 1069.50 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2025-12-01 13:15:00 | 1071.60 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2025-12-01 14:00:00 | 1071.90 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-12-01 14:45:00 | 1072.00 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.69% |
| SELL | retest2 | 2025-12-03 09:15:00 | 1069.80 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.49% |
| SELL | retest2 | 2025-12-03 10:30:00 | 1069.90 | 2025-12-12 13:15:00 | 1053.90 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1030.00 | 2025-12-24 11:15:00 | 978.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1029.10 | 2025-12-24 12:15:00 | 977.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-23 09:15:00 | 1030.00 | 2025-12-29 09:15:00 | 988.70 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2025-12-23 10:15:00 | 1029.10 | 2025-12-29 09:15:00 | 988.70 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest1 | 2026-01-29 15:15:00 | 882.40 | 2026-01-30 14:15:00 | 903.65 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-02-05 11:15:00 | 905.00 | 2026-02-13 10:15:00 | 939.95 | STOP_HIT | 1.00 | 3.86% |
| BUY | retest2 | 2026-02-20 15:00:00 | 995.00 | 2026-02-24 11:15:00 | 972.30 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-02-26 15:00:00 | 960.40 | 2026-02-27 10:15:00 | 969.25 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-02-27 10:15:00 | 961.95 | 2026-02-27 10:15:00 | 969.25 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-02-27 13:30:00 | 960.70 | 2026-03-02 09:15:00 | 912.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 13:30:00 | 960.70 | 2026-03-04 12:15:00 | 941.70 | STOP_HIT | 0.50 | 1.98% |
| BUY | retest2 | 2026-04-02 11:15:00 | 960.05 | 2026-04-08 09:15:00 | 1056.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 12:00:00 | 962.85 | 2026-04-08 09:15:00 | 1059.14 | TARGET_HIT | 1.00 | 10.00% |
