# Premier Energies Ltd. (PREMIERENE)

## Backtest Summary

- **Window:** 2024-09-03 09:15:00 → 2026-05-11 15:15:00 (2909 bars)
- **Last close:** 1002.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 129 |
| ALERT1 | 86 |
| ALERT2 | 83 |
| ALERT2_SKIP | 44 |
| ALERT3 | 223 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 104 |
| PARTIAL | 13 |
| TARGET_HIT | 12 |
| STOP_HIT | 93 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 118 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 41 / 77
- **Target hits / Stop hits / Partials:** 12 / 93 / 13
- **Avg / median % per leg:** 0.32% / -1.39%
- **Sum % (uncompounded):** 37.88%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 6 | 15.8% | 4 | 34 | 0 | -0.92% | -34.9% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.65% | -2.7% |
| BUY @ 3rd Alert (retest2) | 37 | 6 | 16.2% | 4 | 33 | 0 | -0.87% | -32.3% |
| SELL (all) | 80 | 35 | 43.8% | 8 | 59 | 13 | 0.91% | 72.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 80 | 35 | 43.8% | 8 | 59 | 13 | 0.91% | 72.8% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.65% | -2.7% |
| retest2 (combined) | 117 | 41 | 35.0% | 12 | 92 | 13 | 0.35% | 40.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 15:15:00 | 1096.00 | 1122.68 | 1123.09 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 09:15:00 | 1135.30 | 1125.20 | 1124.20 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 12:15:00 | 1117.05 | 1122.69 | 1123.18 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 1128.85 | 1123.73 | 1123.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 15:15:00 | 1137.95 | 1126.58 | 1124.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 09:15:00 | 1120.10 | 1125.28 | 1124.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 09:15:00 | 1120.10 | 1125.28 | 1124.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1120.10 | 1125.28 | 1124.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:30:00 | 1116.45 | 1125.28 | 1124.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 10:15:00 | 1117.05 | 1123.64 | 1123.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-13 12:15:00 | 1098.80 | 1116.61 | 1120.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 14:15:00 | 1121.90 | 1086.48 | 1098.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 14:15:00 | 1121.90 | 1086.48 | 1098.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 1121.90 | 1086.48 | 1098.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 15:00:00 | 1121.90 | 1086.48 | 1098.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 1102.00 | 1089.58 | 1099.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-17 09:30:00 | 1095.00 | 1091.05 | 1098.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 14:15:00 | 1115.80 | 1103.01 | 1102.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-09-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 14:15:00 | 1115.80 | 1103.01 | 1102.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-18 14:15:00 | 1136.60 | 1111.16 | 1107.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 1110.65 | 1115.25 | 1109.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 10:00:00 | 1110.65 | 1115.25 | 1109.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1098.55 | 1111.91 | 1108.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:00:00 | 1098.55 | 1111.91 | 1108.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1102.50 | 1110.03 | 1108.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 1089.00 | 1110.03 | 1108.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 1095.75 | 1107.17 | 1107.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 15:15:00 | 1095.00 | 1104.12 | 1105.73 | Break + close below crossover candle low |

### Cycle 8 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 1117.65 | 1106.83 | 1106.81 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 09:15:00 | 1092.25 | 1105.11 | 1106.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-23 15:15:00 | 1081.95 | 1093.83 | 1099.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-24 14:15:00 | 1077.70 | 1077.37 | 1087.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 14:15:00 | 1077.70 | 1077.37 | 1087.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 1077.70 | 1077.37 | 1087.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:00:00 | 1077.70 | 1077.37 | 1087.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 1077.40 | 1048.16 | 1060.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:00:00 | 1077.40 | 1048.16 | 1060.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 1059.00 | 1050.33 | 1060.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:45:00 | 1052.25 | 1052.16 | 1059.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-27 14:45:00 | 1051.85 | 1047.54 | 1052.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 15:15:00 | 1079.50 | 1051.01 | 1049.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 15:15:00 | 1079.50 | 1051.01 | 1049.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 10:15:00 | 1093.30 | 1065.60 | 1056.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1080.20 | 1092.02 | 1077.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 1080.20 | 1092.02 | 1077.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1080.20 | 1092.02 | 1077.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-04 11:00:00 | 1094.00 | 1083.15 | 1078.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-04 14:15:00 | 1049.40 | 1076.02 | 1076.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 14:15:00 | 1049.40 | 1076.02 | 1076.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 15:15:00 | 1039.90 | 1068.79 | 1073.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 1006.35 | 1006.20 | 1029.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:45:00 | 1004.05 | 1006.20 | 1029.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 1021.00 | 1008.70 | 1021.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 1029.00 | 1008.70 | 1021.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1029.20 | 1012.80 | 1022.37 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 1061.45 | 1030.28 | 1029.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 1093.55 | 1063.17 | 1047.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 14:15:00 | 1137.65 | 1141.91 | 1119.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 15:00:00 | 1137.65 | 1141.91 | 1119.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1124.55 | 1135.49 | 1121.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:15:00 | 1123.65 | 1135.49 | 1121.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 1130.40 | 1134.47 | 1122.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 12:15:00 | 1136.80 | 1134.47 | 1122.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 12:45:00 | 1139.60 | 1134.27 | 1123.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 13:15:00 | 1133.10 | 1136.65 | 1131.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:15:00 | 1139.35 | 1135.73 | 1131.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 1129.80 | 1134.51 | 1131.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-17 09:15:00 | 1107.95 | 1129.19 | 1129.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 09:15:00 | 1107.95 | 1129.19 | 1129.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 1097.05 | 1118.63 | 1124.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 13:15:00 | 1014.70 | 1012.63 | 1032.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 13:15:00 | 1014.70 | 1012.63 | 1032.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 1014.70 | 1012.63 | 1032.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:45:00 | 1018.30 | 1012.63 | 1032.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 11:15:00 | 972.00 | 954.23 | 970.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:45:00 | 969.70 | 954.23 | 970.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 12:15:00 | 968.35 | 957.05 | 970.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 15:00:00 | 961.05 | 960.40 | 970.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 980.25 | 959.95 | 963.30 | SL hit (close>static) qty=1.00 sl=972.10 alert=retest2 |

### Cycle 14 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 1013.95 | 970.75 | 967.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 09:15:00 | 1022.15 | 997.74 | 984.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 1150.80 | 1174.41 | 1143.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 1150.80 | 1174.41 | 1143.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 1150.80 | 1174.41 | 1143.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:15:00 | 1131.80 | 1174.41 | 1143.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 1130.75 | 1165.67 | 1142.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 1129.50 | 1165.67 | 1142.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1122.90 | 1157.12 | 1140.92 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 15:15:00 | 1095.00 | 1126.31 | 1130.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 1083.85 | 1115.45 | 1124.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 12:15:00 | 1115.85 | 1115.53 | 1123.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 13:00:00 | 1115.85 | 1115.53 | 1123.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 1112.25 | 1114.88 | 1122.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:00:00 | 1112.25 | 1114.88 | 1122.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 1132.00 | 1118.30 | 1123.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 15:00:00 | 1132.00 | 1118.30 | 1123.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 1125.00 | 1119.64 | 1123.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 09:15:00 | 1063.70 | 1119.64 | 1123.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-11 09:15:00 | 1010.51 | 1103.72 | 1115.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 1065.45 | 1052.67 | 1076.71 | SL hit (close>ema200) qty=0.50 sl=1052.67 alert=retest2 |

### Cycle 16 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 1080.40 | 1019.88 | 1016.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 13:15:00 | 1089.30 | 1033.77 | 1023.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 1086.90 | 1087.27 | 1064.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 09:15:00 | 1083.15 | 1087.27 | 1064.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 1073.70 | 1085.37 | 1077.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:45:00 | 1075.30 | 1085.37 | 1077.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 1063.10 | 1080.92 | 1075.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 11:30:00 | 1058.15 | 1080.92 | 1075.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 1066.85 | 1075.17 | 1073.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:30:00 | 1064.15 | 1075.17 | 1073.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 14:15:00 | 1062.80 | 1072.69 | 1072.86 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 1093.00 | 1072.07 | 1071.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-26 09:15:00 | 1109.90 | 1084.42 | 1077.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 14:15:00 | 1109.80 | 1110.31 | 1100.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 15:15:00 | 1114.00 | 1110.31 | 1100.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1108.10 | 1110.46 | 1102.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 11:00:00 | 1142.00 | 1116.77 | 1105.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-02 11:15:00 | 1256.20 | 1198.95 | 1165.54 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 13:15:00 | 1263.00 | 1276.25 | 1277.03 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 15:15:00 | 1281.00 | 1277.89 | 1277.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 1303.60 | 1283.04 | 1280.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 1307.35 | 1309.61 | 1297.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 09:45:00 | 1314.90 | 1309.61 | 1297.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 1301.00 | 1313.22 | 1305.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 1300.35 | 1313.22 | 1305.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 1282.05 | 1306.98 | 1303.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 1284.25 | 1306.98 | 1303.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2024-12-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 10:15:00 | 1278.65 | 1301.32 | 1301.40 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 1338.70 | 1303.98 | 1301.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 10:15:00 | 1342.80 | 1311.74 | 1305.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 1353.10 | 1355.16 | 1338.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 13:00:00 | 1353.10 | 1355.16 | 1338.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 1329.95 | 1350.12 | 1337.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 1329.95 | 1350.12 | 1337.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 1324.80 | 1345.05 | 1336.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 1327.90 | 1345.05 | 1336.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 09:15:00 | 1286.40 | 1329.95 | 1331.03 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 1345.20 | 1314.38 | 1312.05 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 14:15:00 | 1293.00 | 1308.35 | 1310.27 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-23 13:15:00 | 1320.20 | 1309.97 | 1309.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-23 14:15:00 | 1322.90 | 1312.56 | 1310.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-24 15:15:00 | 1315.00 | 1318.55 | 1315.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 15:15:00 | 1315.00 | 1318.55 | 1315.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 1315.00 | 1318.55 | 1315.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-26 09:15:00 | 1326.70 | 1318.55 | 1315.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 09:15:00 | 1303.75 | 1315.59 | 1314.79 | SL hit (close<static) qty=1.00 sl=1310.35 alert=retest2 |

### Cycle 27 — SELL (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 10:15:00 | 1293.00 | 1311.07 | 1312.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 11:15:00 | 1288.80 | 1306.62 | 1310.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 09:15:00 | 1312.40 | 1299.84 | 1304.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 09:15:00 | 1312.40 | 1299.84 | 1304.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1312.40 | 1299.84 | 1304.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:30:00 | 1319.75 | 1299.84 | 1304.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1303.70 | 1300.61 | 1304.82 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2024-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 14:15:00 | 1310.60 | 1307.03 | 1306.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 09:15:00 | 1339.05 | 1313.93 | 1310.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 1301.60 | 1325.91 | 1321.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 1301.60 | 1325.91 | 1321.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1301.60 | 1325.91 | 1321.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 1301.60 | 1325.91 | 1321.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 1315.70 | 1323.87 | 1320.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:15:00 | 1322.50 | 1320.28 | 1319.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 09:45:00 | 1323.00 | 1335.50 | 1332.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 11:15:00 | 1323.30 | 1330.03 | 1330.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 11:15:00 | 1323.30 | 1330.03 | 1330.58 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 1369.05 | 1336.83 | 1333.17 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1323.35 | 1333.11 | 1333.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 1305.90 | 1327.67 | 1331.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 1292.20 | 1291.83 | 1308.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 1289.75 | 1291.83 | 1308.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1301.05 | 1294.77 | 1307.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 1303.75 | 1294.77 | 1307.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1303.20 | 1298.76 | 1306.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:15:00 | 1307.00 | 1298.76 | 1306.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 1307.00 | 1300.41 | 1306.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 1288.55 | 1300.41 | 1306.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-08 13:15:00 | 1224.12 | 1267.39 | 1286.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-10 09:15:00 | 1159.69 | 1212.03 | 1241.24 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 32 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 1172.50 | 1147.97 | 1145.76 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 12:15:00 | 1142.10 | 1153.01 | 1153.14 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 09:15:00 | 1171.20 | 1152.84 | 1152.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 10:15:00 | 1186.45 | 1159.56 | 1155.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 12:15:00 | 1182.00 | 1183.65 | 1174.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-21 13:15:00 | 1184.00 | 1183.65 | 1174.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 1177.20 | 1181.88 | 1175.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 1177.20 | 1181.88 | 1175.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 15:15:00 | 1175.75 | 1180.65 | 1175.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:15:00 | 1127.95 | 1180.65 | 1175.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 1116.80 | 1167.88 | 1169.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 1095.10 | 1153.33 | 1162.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 893.00 | 891.59 | 931.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 09:30:00 | 902.95 | 891.59 | 931.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 938.80 | 906.02 | 923.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 15:00:00 | 938.80 | 906.02 | 923.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 15:15:00 | 940.00 | 912.81 | 924.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 09:15:00 | 929.70 | 912.81 | 924.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 950.75 | 928.29 | 930.20 | EMA400 retest candle locked (from downside) |

### Cycle 36 — BUY (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 11:15:00 | 948.30 | 932.29 | 931.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 993.00 | 953.11 | 942.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 1034.40 | 1064.04 | 1032.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 10:00:00 | 1034.40 | 1064.04 | 1032.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 1033.50 | 1057.94 | 1032.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 13:30:00 | 1048.35 | 1053.64 | 1036.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-02-04 09:15:00 | 1153.18 | 1064.32 | 1046.01 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 14:15:00 | 1019.80 | 1038.72 | 1038.80 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-02-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-07 12:15:00 | 1040.90 | 1031.26 | 1030.63 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 1017.70 | 1028.67 | 1029.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 996.70 | 1022.27 | 1026.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 11:15:00 | 960.95 | 941.55 | 963.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 11:15:00 | 960.95 | 941.55 | 963.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 960.95 | 941.55 | 963.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 960.95 | 941.55 | 963.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 977.65 | 948.77 | 964.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:00:00 | 977.65 | 948.77 | 964.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 970.75 | 953.17 | 965.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 970.55 | 953.17 | 965.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 14:15:00 | 970.85 | 956.70 | 965.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 15:00:00 | 970.85 | 956.70 | 965.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 969.00 | 959.16 | 966.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 09:15:00 | 977.25 | 959.16 | 966.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 1015.80 | 977.68 | 973.79 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-02-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 13:15:00 | 957.75 | 977.64 | 979.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-17 09:15:00 | 952.95 | 967.65 | 974.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 967.30 | 965.78 | 971.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 12:15:00 | 967.30 | 965.78 | 971.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 967.30 | 965.78 | 971.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:45:00 | 970.00 | 965.78 | 971.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 965.45 | 965.71 | 970.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 972.40 | 965.71 | 970.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 967.10 | 965.99 | 970.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 14:45:00 | 970.00 | 965.99 | 970.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 978.00 | 968.39 | 971.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 961.90 | 968.39 | 971.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 969.00 | 968.51 | 971.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:45:00 | 950.25 | 963.77 | 968.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 15:15:00 | 976.00 | 970.62 | 970.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2025-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 15:15:00 | 976.00 | 970.62 | 970.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 998.10 | 976.12 | 973.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 997.15 | 999.10 | 989.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 11:30:00 | 999.70 | 999.10 | 989.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 989.00 | 996.00 | 990.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 15:00:00 | 989.00 | 996.00 | 990.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 982.70 | 993.34 | 989.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 09:15:00 | 993.70 | 993.34 | 989.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 965.40 | 987.75 | 987.73 | SL hit (close<static) qty=1.00 sl=980.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 969.50 | 984.10 | 986.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 963.75 | 975.98 | 981.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 09:15:00 | 956.70 | 952.01 | 962.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 09:15:00 | 956.70 | 952.01 | 962.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 956.70 | 952.01 | 962.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:45:00 | 957.10 | 952.01 | 962.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 923.85 | 943.35 | 952.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 11:15:00 | 915.85 | 938.43 | 949.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 882.10 | 924.53 | 937.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 870.06 | 918.18 | 933.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 10:15:00 | 838.00 | 871.76 | 897.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 870.00 | 867.32 | 890.60 | SL hit (close>ema200) qty=0.50 sl=867.32 alert=retest2 |

### Cycle 44 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 913.50 | 882.34 | 879.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-07 09:15:00 | 956.70 | 921.47 | 906.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 920.70 | 933.96 | 922.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 920.70 | 933.96 | 922.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 920.70 | 933.96 | 922.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 920.70 | 933.96 | 922.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 916.40 | 930.45 | 921.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 916.40 | 930.45 | 921.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 913.40 | 927.04 | 921.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 913.40 | 927.04 | 921.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 896.75 | 916.37 | 917.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 15:15:00 | 896.00 | 912.30 | 915.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 914.00 | 908.27 | 911.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 14:15:00 | 914.00 | 908.27 | 911.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 914.00 | 908.27 | 911.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:00:00 | 914.00 | 908.27 | 911.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 914.90 | 909.59 | 911.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 903.40 | 909.59 | 911.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 903.55 | 908.38 | 910.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 11:30:00 | 896.75 | 904.90 | 908.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 13:30:00 | 897.65 | 902.35 | 906.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 15:15:00 | 893.50 | 901.54 | 906.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:30:00 | 891.80 | 898.09 | 903.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 915.05 | 896.19 | 899.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 916.90 | 896.19 | 899.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 907.45 | 898.44 | 900.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 11:45:00 | 903.55 | 899.85 | 900.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:30:00 | 897.35 | 899.65 | 900.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-19 09:15:00 | 913.00 | 897.86 | 897.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 09:15:00 | 913.00 | 897.86 | 897.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 10:15:00 | 928.55 | 903.99 | 900.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 09:15:00 | 923.70 | 935.44 | 926.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 09:15:00 | 923.70 | 935.44 | 926.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 923.70 | 935.44 | 926.18 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2025-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 15:15:00 | 915.00 | 922.50 | 922.94 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 10:15:00 | 936.05 | 922.55 | 922.16 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 911.25 | 921.13 | 921.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 909.00 | 918.71 | 920.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 10:15:00 | 912.10 | 905.46 | 910.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 10:15:00 | 912.10 | 905.46 | 910.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 912.10 | 905.46 | 910.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 10:45:00 | 910.10 | 905.46 | 910.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 11:15:00 | 912.05 | 906.77 | 911.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:30:00 | 913.65 | 906.77 | 911.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 918.80 | 909.18 | 911.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 918.80 | 909.18 | 911.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 922.00 | 911.74 | 912.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 14:00:00 | 922.00 | 911.74 | 912.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 14:15:00 | 951.90 | 919.77 | 916.26 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 13:15:00 | 914.55 | 924.65 | 925.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 910.00 | 921.72 | 924.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 10:15:00 | 904.15 | 897.09 | 905.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 10:15:00 | 904.15 | 897.09 | 905.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 10:15:00 | 904.15 | 897.09 | 905.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 11:00:00 | 904.15 | 897.09 | 905.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 880.95 | 868.99 | 882.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-04 15:00:00 | 880.95 | 868.99 | 882.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 879.00 | 870.99 | 882.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 830.00 | 870.99 | 882.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 788.50 | 858.16 | 875.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-07 15:15:00 | 860.00 | 837.25 | 854.19 | SL hit (close>ema200) qty=0.50 sl=837.25 alert=retest2 |

### Cycle 52 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 872.05 | 852.07 | 851.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 885.00 | 862.34 | 856.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 13:15:00 | 919.15 | 922.14 | 906.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:45:00 | 918.95 | 922.14 | 906.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 1037.70 | 1048.55 | 1037.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:45:00 | 1035.20 | 1048.55 | 1037.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 15:15:00 | 1035.70 | 1045.98 | 1037.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 09:15:00 | 1045.40 | 1045.98 | 1037.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1005.00 | 1037.79 | 1034.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 1005.00 | 1037.79 | 1034.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 1000.00 | 1030.23 | 1031.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 15:15:00 | 985.00 | 1004.26 | 1016.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 1012.00 | 1005.81 | 1015.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 1012.00 | 1005.81 | 1015.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 1009.40 | 1006.53 | 1015.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:15:00 | 1005.70 | 1008.53 | 1014.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:30:00 | 1005.85 | 1008.75 | 1012.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 1005.90 | 1009.04 | 1012.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:45:00 | 1005.50 | 1008.24 | 1011.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 994.85 | 1003.94 | 1008.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 14:15:00 | 985.10 | 996.79 | 1003.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 10:45:00 | 986.00 | 989.40 | 997.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 11:15:00 | 1005.00 | 998.19 | 997.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 1005.00 | 998.19 | 997.27 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 15:15:00 | 987.00 | 995.39 | 996.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 09:15:00 | 976.10 | 991.54 | 994.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 965.55 | 961.92 | 973.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 12:00:00 | 965.55 | 961.92 | 973.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 969.65 | 960.11 | 967.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:30:00 | 965.90 | 960.11 | 967.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 975.00 | 963.09 | 968.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:15:00 | 978.30 | 963.09 | 968.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 978.35 | 966.14 | 969.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 11:30:00 | 977.40 | 966.14 | 969.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 994.00 | 953.76 | 955.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 994.00 | 953.76 | 955.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 984.25 | 959.86 | 957.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1005.95 | 984.26 | 972.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 1015.00 | 1015.80 | 1004.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 09:15:00 | 1010.50 | 1015.80 | 1004.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 1089.80 | 1099.94 | 1081.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 1086.00 | 1099.94 | 1081.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 1096.50 | 1099.26 | 1082.76 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 1058.00 | 1073.15 | 1074.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 1051.50 | 1066.62 | 1071.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 1089.20 | 1068.04 | 1070.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 14:15:00 | 1089.20 | 1068.04 | 1070.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 1089.20 | 1068.04 | 1070.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 14:45:00 | 1086.00 | 1068.04 | 1070.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 1077.00 | 1069.84 | 1071.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 1070.35 | 1069.84 | 1071.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 10:15:00 | 1073.95 | 1071.92 | 1071.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 1073.95 | 1071.92 | 1071.90 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 09:15:00 | 1031.15 | 1067.01 | 1070.13 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 10:15:00 | 1079.50 | 1054.78 | 1052.11 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 12:15:00 | 1057.90 | 1059.85 | 1059.91 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 1070.00 | 1061.88 | 1060.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 1077.70 | 1065.04 | 1062.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 12:15:00 | 1063.30 | 1068.79 | 1065.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 12:15:00 | 1063.30 | 1068.79 | 1065.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 1063.30 | 1068.79 | 1065.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 13:00:00 | 1063.30 | 1068.79 | 1065.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 13:15:00 | 1095.00 | 1074.03 | 1068.43 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 1055.20 | 1066.76 | 1068.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 13:15:00 | 1051.50 | 1058.23 | 1062.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 09:15:00 | 1058.00 | 1055.11 | 1060.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 1058.00 | 1055.11 | 1060.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1058.00 | 1055.11 | 1060.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:15:00 | 1064.80 | 1055.11 | 1060.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1058.40 | 1055.77 | 1059.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 12:30:00 | 1054.50 | 1055.83 | 1059.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 09:15:00 | 1071.70 | 1058.06 | 1059.05 | SL hit (close>static) qty=1.00 sl=1066.00 alert=retest2 |

### Cycle 64 — BUY (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 10:15:00 | 1066.70 | 1059.79 | 1059.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 15:15:00 | 1080.00 | 1066.64 | 1063.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 14:15:00 | 1087.00 | 1088.55 | 1077.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 14:30:00 | 1087.40 | 1088.55 | 1077.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1065.80 | 1083.03 | 1077.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 1065.50 | 1083.03 | 1077.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1069.20 | 1080.27 | 1076.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 1068.30 | 1080.27 | 1076.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 1075.00 | 1079.21 | 1076.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:30:00 | 1072.00 | 1079.21 | 1076.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 1081.60 | 1079.69 | 1076.81 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 1059.00 | 1072.72 | 1073.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 1026.40 | 1061.02 | 1068.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 1017.40 | 1014.28 | 1027.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 1017.40 | 1014.28 | 1027.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 1027.10 | 1018.43 | 1027.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:45:00 | 1028.50 | 1018.43 | 1027.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1028.30 | 1020.41 | 1027.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1019.40 | 1024.04 | 1028.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 968.43 | 984.87 | 997.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 1007.50 | 982.62 | 991.89 | SL hit (close>ema200) qty=0.50 sl=982.62 alert=retest2 |

### Cycle 66 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 1018.20 | 1000.92 | 998.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 1034.00 | 1021.11 | 1014.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 1023.90 | 1026.70 | 1019.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 1023.90 | 1026.70 | 1019.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 1023.90 | 1026.70 | 1019.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 14:00:00 | 1023.90 | 1026.70 | 1019.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1022.90 | 1025.94 | 1019.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 1022.90 | 1025.94 | 1019.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1021.90 | 1025.13 | 1020.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1036.10 | 1025.13 | 1020.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 1017.10 | 1023.33 | 1020.13 | SL hit (close<static) qty=1.00 sl=1020.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 13:15:00 | 1010.00 | 1017.36 | 1017.93 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 09:15:00 | 1021.50 | 1018.82 | 1018.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 1034.10 | 1023.74 | 1020.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 12:15:00 | 1037.10 | 1040.61 | 1033.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 13:00:00 | 1037.10 | 1040.61 | 1033.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 1058.10 | 1061.70 | 1053.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 1058.10 | 1061.70 | 1053.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 1063.50 | 1062.06 | 1054.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 1067.10 | 1055.89 | 1055.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 10:15:00 | 1069.00 | 1057.65 | 1056.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 12:15:00 | 1048.80 | 1054.85 | 1055.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 1048.80 | 1054.85 | 1055.13 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 14:15:00 | 1059.80 | 1055.30 | 1055.25 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 1048.60 | 1054.07 | 1054.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 10:15:00 | 1045.30 | 1052.32 | 1053.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 1048.80 | 1044.77 | 1048.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 1048.80 | 1044.77 | 1048.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1048.80 | 1044.77 | 1048.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:30:00 | 1054.00 | 1044.77 | 1048.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 1044.80 | 1044.77 | 1048.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:15:00 | 1051.50 | 1044.77 | 1048.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 1045.90 | 1045.00 | 1047.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 12:15:00 | 1042.10 | 1045.00 | 1047.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 15:15:00 | 1056.60 | 1050.57 | 1049.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2025-07-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 15:15:00 | 1056.60 | 1050.57 | 1049.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 09:15:00 | 1072.10 | 1054.87 | 1051.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 1092.30 | 1100.62 | 1086.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 1092.30 | 1100.62 | 1086.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1080.50 | 1096.59 | 1085.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:45:00 | 1080.00 | 1096.59 | 1085.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1073.20 | 1091.92 | 1084.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 1073.20 | 1091.92 | 1084.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1100.20 | 1104.58 | 1100.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 1100.20 | 1104.58 | 1100.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 1081.70 | 1100.00 | 1098.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:00:00 | 1081.70 | 1100.00 | 1098.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 14:15:00 | 1076.40 | 1095.28 | 1096.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 13:15:00 | 1067.90 | 1079.40 | 1086.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 10:15:00 | 1080.40 | 1078.70 | 1083.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 10:15:00 | 1080.40 | 1078.70 | 1083.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1080.40 | 1078.70 | 1083.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:45:00 | 1078.10 | 1078.70 | 1083.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 1087.90 | 1080.54 | 1084.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:00:00 | 1087.90 | 1080.54 | 1084.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1082.00 | 1080.83 | 1084.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 14:15:00 | 1080.10 | 1081.26 | 1084.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:30:00 | 1078.20 | 1079.77 | 1082.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 1074.90 | 1079.77 | 1082.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 12:15:00 | 1073.70 | 1057.85 | 1057.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 1073.70 | 1057.85 | 1057.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 13:15:00 | 1089.90 | 1064.26 | 1060.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 10:15:00 | 1085.70 | 1091.52 | 1082.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 11:00:00 | 1085.70 | 1091.52 | 1082.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 1085.60 | 1090.33 | 1082.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 1085.60 | 1090.33 | 1082.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1083.40 | 1088.95 | 1082.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:15:00 | 1083.50 | 1088.95 | 1082.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1090.60 | 1089.28 | 1083.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:30:00 | 1087.00 | 1089.28 | 1083.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 1084.70 | 1089.68 | 1084.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 1069.50 | 1089.68 | 1084.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1070.10 | 1085.76 | 1083.24 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1058.90 | 1080.39 | 1081.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 1048.00 | 1067.19 | 1074.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 10:15:00 | 1061.30 | 1061.16 | 1068.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 10:15:00 | 1061.30 | 1061.16 | 1068.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 1061.30 | 1061.16 | 1068.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:00:00 | 1052.20 | 1059.37 | 1067.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 14:45:00 | 1055.70 | 1055.81 | 1063.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1081.80 | 1060.33 | 1064.11 | SL hit (close>static) qty=1.00 sl=1076.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 1083.00 | 1068.21 | 1067.24 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1042.80 | 1062.57 | 1065.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 1022.50 | 1037.24 | 1047.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 1024.70 | 1022.16 | 1032.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 11:45:00 | 1024.90 | 1022.16 | 1032.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 1013.30 | 1020.63 | 1028.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 1011.00 | 1020.63 | 1028.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 1003.30 | 994.24 | 993.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 1003.30 | 994.24 | 993.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 1014.30 | 998.25 | 995.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 994.50 | 1007.72 | 1003.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 994.50 | 1007.72 | 1003.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 994.50 | 1007.72 | 1003.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:00:00 | 994.50 | 1007.72 | 1003.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 999.50 | 1006.07 | 1003.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:30:00 | 993.00 | 1006.07 | 1003.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 999.00 | 1004.66 | 1002.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 997.30 | 1004.66 | 1002.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 1002.20 | 1004.17 | 1002.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:45:00 | 1004.00 | 1004.17 | 1002.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1001.20 | 1003.57 | 1002.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:30:00 | 999.50 | 1003.57 | 1002.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 999.00 | 1002.66 | 1002.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 999.00 | 1002.66 | 1002.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 1001.00 | 1002.33 | 1002.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 992.40 | 1002.33 | 1002.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 994.60 | 1000.78 | 1001.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 984.50 | 994.06 | 997.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1025.00 | 998.54 | 998.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1025.00 | 998.54 | 998.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1025.00 | 998.54 | 998.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1025.00 | 998.54 | 998.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 1020.10 | 1002.85 | 1000.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 1026.60 | 1012.68 | 1006.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 10:15:00 | 1033.10 | 1034.53 | 1026.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 11:00:00 | 1033.10 | 1034.53 | 1026.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 1023.00 | 1032.74 | 1027.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:00:00 | 1023.00 | 1032.74 | 1027.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1023.30 | 1030.85 | 1026.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 1023.30 | 1030.85 | 1026.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 1026.00 | 1029.88 | 1026.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:15:00 | 1028.60 | 1029.88 | 1026.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 1018.20 | 1025.41 | 1025.40 | SL hit (close<static) qty=1.00 sl=1020.70 alert=retest2 |

### Cycle 81 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 1014.60 | 1023.25 | 1024.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 1012.60 | 1021.12 | 1023.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 14:15:00 | 1003.80 | 1002.31 | 1010.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 15:00:00 | 1003.80 | 1002.31 | 1010.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1003.00 | 1001.36 | 1007.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 1003.00 | 1001.36 | 1007.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1006.70 | 1002.43 | 1007.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 13:00:00 | 999.00 | 1001.75 | 1006.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1017.90 | 1006.39 | 1007.49 | SL hit (close>static) qty=1.00 sl=1017.80 alert=retest2 |

### Cycle 82 — BUY (started 2025-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 11:15:00 | 1013.00 | 1008.66 | 1008.38 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 14:15:00 | 1003.30 | 1008.35 | 1008.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 13:15:00 | 999.10 | 1004.73 | 1006.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 1019.90 | 1004.30 | 1005.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 1019.90 | 1004.30 | 1005.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1019.90 | 1004.30 | 1005.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1013.00 | 1004.30 | 1005.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 1016.40 | 1006.72 | 1006.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 1031.20 | 1013.42 | 1009.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 1033.15 | 1033.72 | 1024.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 13:00:00 | 1033.15 | 1033.72 | 1024.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 1025.20 | 1032.02 | 1024.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 1025.20 | 1032.02 | 1024.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 1020.00 | 1029.61 | 1023.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 1020.00 | 1029.61 | 1023.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 1021.00 | 1027.89 | 1023.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 1030.50 | 1027.89 | 1023.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1026.85 | 1027.79 | 1024.22 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 1020.70 | 1023.22 | 1023.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 12:15:00 | 1014.55 | 1021.48 | 1022.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 1011.65 | 1005.84 | 1010.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1011.65 | 1005.84 | 1010.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1011.65 | 1005.84 | 1010.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:30:00 | 1015.95 | 1005.84 | 1010.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1000.00 | 1004.67 | 1009.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:30:00 | 998.00 | 1000.93 | 1006.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:45:00 | 996.05 | 999.20 | 1004.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:00:00 | 997.50 | 999.03 | 1003.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:30:00 | 998.00 | 998.53 | 1002.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1014.65 | 1001.08 | 1002.56 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 1014.65 | 1001.08 | 1002.56 | SL hit (close>static) qty=1.00 sl=1013.90 alert=retest2 |

### Cycle 86 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 1011.05 | 1004.60 | 1003.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 1017.75 | 1007.23 | 1005.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 12:15:00 | 1050.20 | 1051.83 | 1039.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 13:00:00 | 1050.20 | 1051.83 | 1039.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 1052.60 | 1051.08 | 1041.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:30:00 | 1042.00 | 1051.08 | 1041.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1075.80 | 1058.53 | 1051.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:45:00 | 1079.80 | 1068.21 | 1059.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 1084.30 | 1070.27 | 1060.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 10:30:00 | 1084.45 | 1073.74 | 1064.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 12:45:00 | 1082.20 | 1075.19 | 1066.60 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1069.90 | 1074.58 | 1069.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 1069.90 | 1074.58 | 1069.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 1066.10 | 1072.89 | 1068.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:00:00 | 1066.10 | 1072.89 | 1068.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 1055.95 | 1069.50 | 1067.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:45:00 | 1054.80 | 1069.50 | 1067.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 1053.80 | 1064.23 | 1065.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 1053.80 | 1064.23 | 1065.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 1051.45 | 1058.46 | 1062.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 1030.50 | 1020.20 | 1029.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 1030.50 | 1020.20 | 1029.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1030.50 | 1020.20 | 1029.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1030.50 | 1020.20 | 1029.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1035.00 | 1023.16 | 1030.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 1035.00 | 1023.16 | 1030.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1064.35 | 1031.40 | 1033.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 1064.35 | 1031.40 | 1033.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 13:15:00 | 1042.00 | 1035.66 | 1035.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 14:15:00 | 1049.00 | 1041.05 | 1038.90 | Break + close above crossover candle high |

### Cycle 89 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 1016.50 | 1037.51 | 1037.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 1012.30 | 1029.18 | 1033.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 14:15:00 | 1027.85 | 1023.51 | 1029.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 14:15:00 | 1027.85 | 1023.51 | 1029.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 1027.85 | 1023.51 | 1029.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 1027.85 | 1023.51 | 1029.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 1022.20 | 1023.24 | 1028.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:15:00 | 1019.70 | 1023.24 | 1028.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1019.80 | 1022.56 | 1028.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1011.50 | 1019.30 | 1025.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:00:00 | 1011.55 | 1019.30 | 1025.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:30:00 | 1010.00 | 1018.64 | 1024.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:15:00 | 1011.80 | 1017.87 | 1022.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 1018.40 | 1017.98 | 1021.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 1018.40 | 1017.98 | 1021.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1015.00 | 1017.38 | 1021.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:30:00 | 1012.85 | 1016.36 | 1020.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 14:15:00 | 1023.50 | 1017.79 | 1020.61 | SL hit (close>static) qty=1.00 sl=1022.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 1032.30 | 1021.30 | 1020.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 1044.70 | 1025.98 | 1022.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 1020.00 | 1030.27 | 1027.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 1020.00 | 1030.27 | 1027.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1020.00 | 1030.27 | 1027.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 1020.00 | 1030.27 | 1027.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 10:15:00 | 1020.00 | 1028.22 | 1026.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 10:30:00 | 1018.30 | 1028.22 | 1026.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 1018.50 | 1024.69 | 1025.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 1017.00 | 1020.96 | 1022.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 14:15:00 | 1018.40 | 1016.67 | 1018.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 14:15:00 | 1018.40 | 1016.67 | 1018.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 1018.40 | 1016.67 | 1018.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 15:00:00 | 1018.40 | 1016.67 | 1018.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 1018.80 | 1017.09 | 1018.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 1020.50 | 1017.09 | 1018.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1017.10 | 1017.10 | 1018.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1019.00 | 1017.10 | 1018.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1016.30 | 1016.94 | 1018.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 1017.10 | 1016.94 | 1018.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1017.70 | 1017.09 | 1018.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1017.70 | 1017.09 | 1018.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1017.80 | 1017.23 | 1018.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 1019.70 | 1017.23 | 1018.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1013.90 | 1016.56 | 1017.83 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 14:15:00 | 1027.90 | 1018.83 | 1018.74 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 1020.50 | 1020.73 | 1020.73 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 14:15:00 | 1030.10 | 1022.53 | 1021.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 1047.40 | 1029.02 | 1024.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-15 14:15:00 | 1059.50 | 1061.64 | 1049.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-15 15:00:00 | 1059.50 | 1061.64 | 1049.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1051.00 | 1058.90 | 1052.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 1051.00 | 1058.90 | 1052.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 1053.20 | 1057.76 | 1052.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 14:45:00 | 1066.80 | 1059.00 | 1053.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 1058.40 | 1060.75 | 1055.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:15:00 | 1069.70 | 1057.39 | 1055.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 11:15:00 | 1058.00 | 1058.24 | 1056.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 1056.50 | 1057.90 | 1056.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-20 13:15:00 | 1038.80 | 1053.77 | 1055.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 13:15:00 | 1038.80 | 1053.77 | 1055.03 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 1064.30 | 1056.72 | 1055.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 10:15:00 | 1077.00 | 1060.78 | 1057.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 1066.50 | 1066.65 | 1061.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 14:00:00 | 1066.50 | 1066.65 | 1061.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1055.10 | 1064.34 | 1061.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1055.10 | 1064.34 | 1061.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1069.00 | 1065.27 | 1061.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 1100.40 | 1066.99 | 1064.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 1078.20 | 1084.45 | 1080.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-29 09:30:00 | 1087.30 | 1085.95 | 1082.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1077.00 | 1086.84 | 1088.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1077.00 | 1086.84 | 1088.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 10:15:00 | 1068.20 | 1079.57 | 1083.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1011.90 | 1009.60 | 1027.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:00:00 | 1011.90 | 1009.60 | 1027.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1023.70 | 1012.79 | 1024.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1023.70 | 1012.79 | 1024.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1031.80 | 1016.59 | 1025.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 1014.50 | 1016.59 | 1025.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:00:00 | 1016.80 | 1012.02 | 1018.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:15:00 | 1019.00 | 1014.64 | 1016.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1041.90 | 1020.79 | 1019.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 1041.90 | 1020.79 | 1019.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 11:15:00 | 1052.00 | 1030.91 | 1024.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 10:15:00 | 1049.70 | 1049.96 | 1038.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 10:45:00 | 1047.80 | 1049.96 | 1038.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1032.50 | 1047.81 | 1042.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 1032.50 | 1047.81 | 1042.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1017.00 | 1041.65 | 1040.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 1017.00 | 1041.65 | 1040.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 1009.10 | 1035.14 | 1037.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 13:15:00 | 1004.10 | 1025.16 | 1032.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 1011.40 | 1009.81 | 1018.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 14:15:00 | 1011.40 | 1009.81 | 1018.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 1011.40 | 1009.81 | 1018.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 1011.40 | 1009.81 | 1018.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 997.60 | 994.13 | 999.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:00:00 | 995.00 | 994.70 | 998.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 15:15:00 | 994.90 | 995.87 | 998.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 12:15:00 | 985.30 | 981.71 | 981.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 985.30 | 981.71 | 981.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 14:15:00 | 987.60 | 983.50 | 982.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 980.60 | 983.56 | 982.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 980.60 | 983.56 | 982.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 980.60 | 983.56 | 982.64 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 978.50 | 981.64 | 981.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 975.60 | 979.46 | 980.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 15:15:00 | 972.80 | 971.69 | 975.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 09:15:00 | 972.50 | 971.69 | 975.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 968.35 | 971.03 | 974.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 964.70 | 968.93 | 972.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-02 15:15:00 | 963.00 | 968.84 | 972.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:15:00 | 965.00 | 967.47 | 970.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:15:00 | 964.50 | 966.22 | 969.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 964.30 | 965.83 | 969.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:45:00 | 962.00 | 966.38 | 968.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 916.47 | 945.90 | 956.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 09:15:00 | 916.75 | 945.90 | 956.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:15:00 | 914.85 | 939.40 | 952.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:15:00 | 916.27 | 939.40 | 952.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-05 10:15:00 | 913.90 | 939.40 | 952.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 13:15:00 | 868.23 | 888.41 | 911.72 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 102 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 914.45 | 875.31 | 872.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 918.80 | 884.01 | 876.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 12:15:00 | 908.90 | 909.38 | 898.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 13:00:00 | 908.90 | 909.38 | 898.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 909.75 | 918.49 | 911.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 909.75 | 918.49 | 911.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 906.40 | 916.07 | 911.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 903.00 | 916.07 | 911.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 901.70 | 912.38 | 910.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 901.70 | 912.38 | 910.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 901.60 | 910.23 | 909.50 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 904.05 | 908.99 | 909.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 10:15:00 | 900.50 | 905.01 | 906.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 900.00 | 898.41 | 902.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 900.00 | 898.41 | 902.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 900.00 | 898.41 | 902.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:15:00 | 895.80 | 898.41 | 902.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 891.40 | 896.53 | 900.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 900.00 | 888.99 | 888.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 10:15:00 | 900.00 | 888.99 | 888.68 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 885.90 | 888.22 | 888.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 12:15:00 | 881.00 | 885.57 | 887.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 14:15:00 | 852.30 | 851.89 | 861.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 15:00:00 | 852.30 | 851.89 | 861.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 855.70 | 852.19 | 859.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:00:00 | 852.55 | 852.26 | 859.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 13:15:00 | 852.45 | 853.06 | 858.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 809.92 | 837.32 | 844.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-05 09:15:00 | 809.83 | 837.32 | 844.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-06 09:15:00 | 767.29 | 797.55 | 817.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 106 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 741.50 | 738.93 | 738.89 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 12:15:00 | 731.15 | 738.34 | 739.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 13:15:00 | 725.65 | 735.80 | 738.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-16 09:15:00 | 749.50 | 735.98 | 737.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 749.50 | 735.98 | 737.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 749.50 | 735.98 | 737.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 749.50 | 735.98 | 737.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 749.75 | 738.74 | 738.44 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 732.55 | 739.09 | 739.83 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-20 11:15:00 | 742.00 | 740.20 | 739.96 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 738.15 | 739.79 | 739.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 726.60 | 737.15 | 738.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 728.60 | 721.10 | 727.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 728.60 | 721.10 | 727.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 728.60 | 721.10 | 727.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:30:00 | 724.50 | 720.94 | 726.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 13:15:00 | 722.80 | 722.78 | 726.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 747.00 | 730.92 | 729.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 747.00 | 730.92 | 729.52 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 709.05 | 726.55 | 727.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 681.95 | 708.76 | 717.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 716.70 | 705.74 | 714.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 716.70 | 705.74 | 714.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 716.70 | 705.74 | 714.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:30:00 | 707.90 | 705.74 | 714.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 712.40 | 707.08 | 714.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:15:00 | 703.20 | 708.61 | 712.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 14:00:00 | 705.75 | 707.72 | 710.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 719.95 | 711.25 | 711.80 | SL hit (close>static) qty=1.00 sl=716.85 alert=retest2 |

### Cycle 114 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 713.30 | 711.56 | 711.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 725.80 | 714.78 | 712.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 14:15:00 | 717.95 | 718.90 | 716.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 15:00:00 | 717.95 | 718.90 | 716.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 711.45 | 719.48 | 717.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 711.45 | 719.48 | 717.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 703.75 | 716.33 | 716.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 699.45 | 716.33 | 716.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 698.80 | 712.82 | 714.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 690.00 | 708.26 | 712.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 712.90 | 706.86 | 710.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 712.90 | 706.86 | 710.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 712.90 | 706.86 | 710.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:45:00 | 715.45 | 706.86 | 710.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 710.05 | 707.50 | 710.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 10:45:00 | 710.65 | 707.50 | 710.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 717.05 | 709.41 | 711.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:30:00 | 716.05 | 709.41 | 711.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 722.20 | 711.97 | 712.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 722.20 | 711.97 | 712.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 722.00 | 713.97 | 713.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 734.70 | 718.12 | 715.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 774.85 | 778.64 | 757.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 09:30:00 | 775.90 | 778.64 | 757.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 779.50 | 781.82 | 769.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 770.90 | 781.82 | 769.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 772.55 | 778.33 | 771.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 772.55 | 778.33 | 771.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 791.00 | 780.86 | 773.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 793.70 | 780.86 | 773.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:00:00 | 791.50 | 785.04 | 776.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:00:00 | 791.50 | 786.33 | 778.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 791.60 | 787.85 | 781.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 789.00 | 810.83 | 804.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:30:00 | 787.55 | 810.83 | 804.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 786.30 | 805.92 | 802.71 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-11 12:15:00 | 782.75 | 797.72 | 799.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 782.75 | 797.72 | 799.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 771.40 | 786.43 | 792.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 762.50 | 761.30 | 769.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 765.70 | 763.03 | 766.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 765.70 | 763.03 | 766.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:45:00 | 765.60 | 763.03 | 766.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 754.30 | 759.74 | 763.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 759.90 | 759.74 | 763.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 759.30 | 759.65 | 762.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:00:00 | 759.30 | 759.65 | 762.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 760.75 | 759.87 | 762.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 11:45:00 | 761.25 | 759.87 | 762.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 764.00 | 760.73 | 762.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 758.65 | 761.84 | 762.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:00:00 | 759.65 | 756.89 | 759.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 09:15:00 | 771.00 | 758.08 | 759.19 | SL hit (close>static) qty=1.00 sl=768.40 alert=retest2 |

### Cycle 118 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 10:15:00 | 769.15 | 760.30 | 760.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 13:15:00 | 774.20 | 765.89 | 762.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 15:15:00 | 765.25 | 766.42 | 763.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 09:15:00 | 771.00 | 766.42 | 763.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 762.60 | 765.56 | 763.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 11:00:00 | 762.60 | 765.56 | 763.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 11:15:00 | 766.00 | 765.65 | 764.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 13:00:00 | 767.70 | 766.06 | 764.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 13:45:00 | 769.60 | 767.02 | 764.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 13:00:00 | 769.20 | 768.68 | 767.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 14:30:00 | 768.00 | 768.91 | 767.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 735.20 | 764.15 | 765.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-25 09:15:00 | 735.20 | 764.15 | 765.62 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 725.75 | 714.99 | 713.90 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 706.55 | 715.19 | 715.65 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-03-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 15:15:00 | 719.05 | 715.93 | 715.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 728.65 | 718.47 | 716.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 776.75 | 782.18 | 769.63 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-13 10:45:00 | 796.65 | 785.33 | 772.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 776.70 | 785.11 | 775.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:00:00 | 776.70 | 785.11 | 775.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 782.40 | 784.57 | 776.21 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-16 10:15:00 | 775.50 | 781.88 | 776.96 | SL hit (close<ema400) qty=1.00 sl=776.96 alert=retest1 |

### Cycle 123 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 884.35 | 892.31 | 892.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 882.00 | 890.24 | 891.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 14:15:00 | 890.95 | 890.39 | 891.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 14:15:00 | 890.95 | 890.39 | 891.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 890.95 | 890.39 | 891.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 890.95 | 890.39 | 891.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — BUY (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 15:15:00 | 898.75 | 892.06 | 892.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 906.10 | 894.87 | 893.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 911.70 | 918.81 | 908.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 911.70 | 918.81 | 908.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 911.70 | 918.81 | 908.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 911.70 | 918.81 | 908.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 901.15 | 915.28 | 908.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 901.00 | 915.28 | 908.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 901.80 | 912.58 | 907.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 12:15:00 | 902.30 | 912.58 | 907.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 904.00 | 910.86 | 907.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:30:00 | 909.25 | 911.36 | 907.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-15 10:15:00 | 1000.18 | 983.17 | 974.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 1000.00 | 1008.53 | 1008.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 13:15:00 | 996.00 | 1004.51 | 1006.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 1006.00 | 996.16 | 999.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 1006.00 | 996.16 | 999.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 1006.00 | 996.16 | 999.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 1006.00 | 996.16 | 999.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 1001.25 | 997.18 | 999.29 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2026-04-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 13:15:00 | 1005.65 | 1001.43 | 1000.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-24 09:15:00 | 1014.45 | 1003.89 | 1002.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 14:15:00 | 1005.80 | 1008.10 | 1005.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 14:15:00 | 1005.80 | 1008.10 | 1005.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 1005.80 | 1008.10 | 1005.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 1005.80 | 1008.10 | 1005.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 1012.00 | 1008.88 | 1005.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-27 09:15:00 | 1022.20 | 1008.88 | 1005.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1003.75 | 1028.73 | 1026.99 | SL hit (close<static) qty=1.00 sl=1004.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 1000.60 | 1023.11 | 1024.59 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 1033.70 | 1022.52 | 1021.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 1044.20 | 1035.77 | 1030.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 1035.00 | 1037.58 | 1033.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-06 11:00:00 | 1035.00 | 1037.58 | 1033.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 1033.00 | 1036.67 | 1033.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:30:00 | 1033.60 | 1036.67 | 1033.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 1032.70 | 1035.87 | 1033.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:00:00 | 1032.70 | 1035.87 | 1033.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 1029.60 | 1034.62 | 1032.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:30:00 | 1029.50 | 1034.62 | 1032.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 1033.50 | 1034.39 | 1032.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 14:30:00 | 1029.80 | 1034.39 | 1032.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 1030.90 | 1033.70 | 1032.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 1029.20 | 1033.70 | 1032.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 09:15:00 | 1015.10 | 1029.98 | 1031.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 1009.90 | 1018.59 | 1022.63 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-09-17 09:30:00 | 1095.00 | 2024-09-17 14:15:00 | 1115.80 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-09-26 13:45:00 | 1052.25 | 2024-09-30 15:15:00 | 1079.50 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-09-27 14:45:00 | 1051.85 | 2024-09-30 15:15:00 | 1079.50 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-10-04 11:00:00 | 1094.00 | 2024-10-04 14:15:00 | 1049.40 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2024-10-15 12:15:00 | 1136.80 | 2024-10-17 09:15:00 | 1107.95 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2024-10-15 12:45:00 | 1139.60 | 2024-10-17 09:15:00 | 1107.95 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2024-10-16 13:15:00 | 1133.10 | 2024-10-17 09:15:00 | 1107.95 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-10-16 14:15:00 | 1139.35 | 2024-10-17 09:15:00 | 1107.95 | STOP_HIT | 1.00 | -2.76% |
| SELL | retest2 | 2024-10-28 15:00:00 | 961.05 | 2024-10-30 09:15:00 | 980.25 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-11-11 09:15:00 | 1063.70 | 2024-11-11 09:15:00 | 1010.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-11 09:15:00 | 1063.70 | 2024-11-12 09:15:00 | 1065.45 | STOP_HIT | 0.50 | -0.16% |
| BUY | retest2 | 2024-11-28 11:00:00 | 1142.00 | 2024-12-02 11:15:00 | 1256.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-26 09:15:00 | 1326.70 | 2024-12-26 09:15:00 | 1303.75 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-12-31 13:15:00 | 1322.50 | 2025-01-02 11:15:00 | 1323.30 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-01-02 09:45:00 | 1323.00 | 2025-01-02 11:15:00 | 1323.30 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1288.55 | 2025-01-08 13:15:00 | 1224.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 1288.55 | 2025-01-10 09:15:00 | 1159.69 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-03 13:30:00 | 1048.35 | 2025-02-04 09:15:00 | 1153.18 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-18 11:45:00 | 950.25 | 2025-02-18 15:15:00 | 976.00 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-02-21 09:15:00 | 993.70 | 2025-02-21 09:15:00 | 965.40 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-02-27 11:15:00 | 915.85 | 2025-02-28 09:15:00 | 870.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 882.10 | 2025-03-03 10:15:00 | 838.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 11:15:00 | 915.85 | 2025-03-03 12:15:00 | 870.00 | STOP_HIT | 0.50 | 5.01% |
| SELL | retest2 | 2025-02-28 09:15:00 | 882.10 | 2025-03-03 12:15:00 | 870.00 | STOP_HIT | 0.50 | 1.37% |
| SELL | retest2 | 2025-03-12 11:30:00 | 896.75 | 2025-03-19 09:15:00 | 913.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-03-12 13:30:00 | 897.65 | 2025-03-19 09:15:00 | 913.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-03-12 15:15:00 | 893.50 | 2025-03-19 09:15:00 | 913.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-03-13 10:30:00 | 891.80 | 2025-03-19 09:15:00 | 913.00 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-03-17 11:45:00 | 903.55 | 2025-03-19 09:15:00 | 913.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-03-17 12:30:00 | 897.35 | 2025-03-19 09:15:00 | 913.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-04-07 09:15:00 | 830.00 | 2025-04-07 09:15:00 | 788.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 830.00 | 2025-04-07 15:15:00 | 860.00 | STOP_HIT | 0.50 | -3.61% |
| SELL | retest2 | 2025-04-28 14:15:00 | 1005.70 | 2025-05-05 11:15:00 | 1005.00 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-04-29 09:30:00 | 1005.85 | 2025-05-05 11:15:00 | 1005.00 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2025-04-29 11:45:00 | 1005.90 | 2025-05-05 11:15:00 | 1005.00 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-04-29 13:45:00 | 1005.50 | 2025-05-05 11:15:00 | 1005.00 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-04-30 14:15:00 | 985.10 | 2025-05-05 11:15:00 | 1005.00 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-05-02 10:45:00 | 986.00 | 2025-05-05 11:15:00 | 1005.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1070.35 | 2025-05-22 10:15:00 | 1073.95 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-06 12:30:00 | 1054.50 | 2025-06-09 09:15:00 | 1071.70 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1019.40 | 2025-06-19 12:15:00 | 968.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1019.40 | 2025-06-20 09:15:00 | 1007.50 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2025-06-20 12:00:00 | 1021.20 | 2025-06-20 12:15:00 | 1018.20 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1036.10 | 2025-06-25 10:15:00 | 1017.10 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-07-04 09:15:00 | 1067.10 | 2025-07-04 12:15:00 | 1048.80 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-07-04 10:15:00 | 1069.00 | 2025-07-04 12:15:00 | 1048.80 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-07-08 12:15:00 | 1042.10 | 2025-07-08 15:15:00 | 1056.60 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-07-17 14:15:00 | 1080.10 | 2025-07-23 12:15:00 | 1073.70 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2025-07-18 09:30:00 | 1078.20 | 2025-07-23 12:15:00 | 1073.70 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-07-18 10:00:00 | 1074.90 | 2025-07-23 12:15:00 | 1073.70 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2025-07-29 12:00:00 | 1052.20 | 2025-07-30 09:15:00 | 1081.80 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-07-29 14:45:00 | 1055.70 | 2025-07-30 09:15:00 | 1081.80 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-08-05 10:15:00 | 1011.00 | 2025-08-11 15:15:00 | 1003.30 | STOP_HIT | 1.00 | 0.76% |
| BUY | retest2 | 2025-08-22 09:15:00 | 1028.60 | 2025-08-22 11:15:00 | 1018.20 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-26 13:00:00 | 999.00 | 2025-08-28 09:15:00 | 1017.90 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-08-28 09:30:00 | 1000.90 | 2025-08-28 11:15:00 | 1013.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-09-08 14:30:00 | 998.00 | 2025-09-10 09:15:00 | 1014.65 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-09 09:45:00 | 996.05 | 2025-09-10 09:15:00 | 1014.65 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-09-09 12:00:00 | 997.50 | 2025-09-10 09:15:00 | 1014.65 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-09-09 12:30:00 | 998.00 | 2025-09-10 09:15:00 | 1014.65 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-09-10 11:15:00 | 1006.80 | 2025-09-10 12:15:00 | 1011.05 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-09-16 14:45:00 | 1079.80 | 2025-09-18 13:15:00 | 1053.80 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-09-17 09:15:00 | 1084.30 | 2025-09-18 13:15:00 | 1053.80 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2025-09-17 10:30:00 | 1084.45 | 2025-09-18 13:15:00 | 1053.80 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-09-17 12:45:00 | 1082.20 | 2025-09-18 13:15:00 | 1053.80 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1011.50 | 2025-09-30 14:15:00 | 1023.50 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-09-29 12:00:00 | 1011.55 | 2025-10-03 09:15:00 | 1032.30 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-09-29 12:30:00 | 1010.00 | 2025-10-03 09:15:00 | 1032.30 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-09-30 11:15:00 | 1011.80 | 2025-10-03 09:15:00 | 1032.30 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-09-30 13:30:00 | 1012.85 | 2025-10-03 09:15:00 | 1032.30 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-10-16 14:45:00 | 1066.80 | 2025-10-20 13:15:00 | 1038.80 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-10-17 11:00:00 | 1058.40 | 2025-10-20 13:15:00 | 1038.80 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-17 14:15:00 | 1069.70 | 2025-10-20 13:15:00 | 1038.80 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2025-10-20 11:15:00 | 1058.00 | 2025-10-20 13:15:00 | 1038.80 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-10-27 09:15:00 | 1100.40 | 2025-10-31 11:15:00 | 1077.00 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2025-10-28 12:45:00 | 1078.20 | 2025-10-31 11:15:00 | 1077.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-10-29 09:30:00 | 1087.30 | 2025-10-31 11:15:00 | 1077.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-11-10 09:15:00 | 1014.50 | 2025-11-12 09:15:00 | 1041.90 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2025-11-10 15:00:00 | 1016.80 | 2025-11-12 09:15:00 | 1041.90 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-11-11 15:15:00 | 1019.00 | 2025-11-12 09:15:00 | 1041.90 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2025-11-20 12:00:00 | 995.00 | 2025-11-27 12:15:00 | 985.30 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-11-20 15:15:00 | 994.90 | 2025-11-27 12:15:00 | 985.30 | STOP_HIT | 1.00 | 0.96% |
| SELL | retest2 | 2025-12-02 13:45:00 | 964.70 | 2025-12-05 09:15:00 | 916.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 15:15:00 | 963.00 | 2025-12-05 09:15:00 | 916.75 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-12-03 10:15:00 | 965.00 | 2025-12-05 10:15:00 | 914.85 | PARTIAL | 0.50 | 5.20% |
| SELL | retest2 | 2025-12-03 12:15:00 | 964.50 | 2025-12-05 10:15:00 | 916.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 09:45:00 | 962.00 | 2025-12-05 10:15:00 | 913.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-02 13:45:00 | 964.70 | 2025-12-08 13:15:00 | 868.23 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-02 15:15:00 | 963.00 | 2025-12-08 13:15:00 | 866.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-03 10:15:00 | 965.00 | 2025-12-08 13:15:00 | 868.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-03 12:15:00 | 964.50 | 2025-12-08 13:15:00 | 868.05 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-04 09:45:00 | 962.00 | 2025-12-08 14:15:00 | 865.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-19 10:15:00 | 895.80 | 2025-12-24 10:15:00 | 900.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-12-19 10:45:00 | 891.40 | 2025-12-24 10:15:00 | 900.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-12-31 11:00:00 | 852.55 | 2026-01-05 09:15:00 | 809.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 13:15:00 | 852.45 | 2026-01-05 09:15:00 | 809.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-31 11:00:00 | 852.55 | 2026-01-06 09:15:00 | 767.29 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-12-31 13:15:00 | 852.45 | 2026-01-06 09:15:00 | 767.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 10:30:00 | 724.50 | 2026-01-22 15:15:00 | 747.00 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2026-01-22 13:15:00 | 722.80 | 2026-01-22 15:15:00 | 747.00 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2026-01-28 10:15:00 | 703.20 | 2026-01-28 15:15:00 | 719.95 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-01-28 14:00:00 | 705.75 | 2026-01-28 15:15:00 | 719.95 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-01-29 10:15:00 | 704.30 | 2026-01-29 14:15:00 | 713.30 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-01-29 11:00:00 | 705.10 | 2026-01-29 14:15:00 | 713.30 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-02-05 15:15:00 | 793.70 | 2026-02-11 12:15:00 | 782.75 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2026-02-06 10:00:00 | 791.50 | 2026-02-11 12:15:00 | 782.75 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-06 11:00:00 | 791.50 | 2026-02-11 12:15:00 | 782.75 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2026-02-06 14:30:00 | 791.60 | 2026-02-11 12:15:00 | 782.75 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-19 09:15:00 | 758.65 | 2026-02-20 09:15:00 | 771.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-02-19 13:00:00 | 759.65 | 2026-02-20 09:15:00 | 771.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-02-23 13:00:00 | 767.70 | 2026-02-25 09:15:00 | 735.20 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2026-02-23 13:45:00 | 769.60 | 2026-02-25 09:15:00 | 735.20 | STOP_HIT | 1.00 | -4.47% |
| BUY | retest2 | 2026-02-24 13:00:00 | 769.20 | 2026-02-25 09:15:00 | 735.20 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2026-02-24 14:30:00 | 768.00 | 2026-02-25 09:15:00 | 735.20 | STOP_HIT | 1.00 | -4.27% |
| BUY | retest1 | 2026-03-13 10:45:00 | 796.65 | 2026-03-16 10:15:00 | 775.50 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2026-03-16 14:15:00 | 789.40 | 2026-03-18 13:15:00 | 868.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 13:30:00 | 909.25 | 2026-04-15 10:15:00 | 1000.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-27 09:15:00 | 1022.20 | 2026-04-30 09:15:00 | 1003.75 | STOP_HIT | 1.00 | -1.80% |
