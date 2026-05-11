# Tata Consumer Products Ltd. (TATACONSUM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1176.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 2 |
| ALERT3 | 53 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 39 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 35
- **Target hits / Stop hits / Partials:** 0 / 35 / 4
- **Avg / median % per leg:** -1.61% / -1.46%
- **Sum % (uncompounded):** -62.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.26% | -16.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 0 | 0.0% | 0 | 13 | 0 | -1.26% | -16.3% |
| SELL (all) | 26 | 4 | 15.4% | 0 | 22 | 4 | -1.79% | -46.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 26 | 4 | 15.4% | 0 | 22 | 4 | -1.79% | -46.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 39 | 4 | 10.3% | 0 | 35 | 4 | -1.61% | -62.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 11:15:00 | 1132.42 | 1097.41 | 1097.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 10:15:00 | 1156.13 | 1109.10 | 1103.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-12 09:15:00 | 1166.95 | 1169.95 | 1145.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-12 10:00:00 | 1166.95 | 1169.95 | 1145.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 1187.60 | 1201.55 | 1183.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:30:00 | 1187.20 | 1201.55 | 1183.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 1190.25 | 1200.98 | 1183.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:30:00 | 1184.80 | 1200.98 | 1183.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1175.55 | 1200.84 | 1185.98 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2024-10-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 11:15:00 | 1108.60 | 1173.69 | 1174.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 1097.55 | 1158.32 | 1165.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 10:15:00 | 934.30 | 932.79 | 975.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 11:00:00 | 934.30 | 932.79 | 975.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 981.15 | 937.73 | 972.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 11:00:00 | 981.15 | 937.73 | 972.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 969.85 | 938.05 | 972.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 961.95 | 939.33 | 972.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 961.10 | 941.52 | 972.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 11:00:00 | 966.25 | 941.97 | 972.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 13:00:00 | 964.25 | 942.42 | 972.05 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 966.45 | 943.35 | 971.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 14:15:00 | 960.10 | 944.39 | 971.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 948.30 | 944.76 | 971.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 13:15:00 | 961.05 | 945.42 | 968.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-20 14:15:00 | 960.70 | 945.59 | 968.93 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 969.70 | 946.11 | 968.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:45:00 | 974.85 | 946.11 | 968.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 969.10 | 946.34 | 968.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:45:00 | 969.95 | 946.34 | 968.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 973.05 | 946.61 | 968.87 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-21 11:15:00 | 973.05 | 946.61 | 968.87 | SL hit (close>static) qty=1.00 sl=973.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 11:15:00 | 1016.80 | 980.56 | 980.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 12:15:00 | 1020.65 | 980.96 | 980.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1000.50 | 1002.45 | 993.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 1000.50 | 1002.45 | 993.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 13:15:00 | 997.00 | 1002.46 | 993.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 13:45:00 | 993.60 | 1002.46 | 993.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1005.45 | 1002.53 | 993.88 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-03-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 14:15:00 | 962.05 | 988.42 | 988.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 957.75 | 986.70 | 987.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-24 11:15:00 | 973.60 | 971.81 | 978.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-24 12:00:00 | 973.60 | 971.81 | 978.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 969.40 | 971.84 | 978.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:45:00 | 965.60 | 971.76 | 978.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 10:15:00 | 960.20 | 971.51 | 978.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 09:15:00 | 1009.10 | 971.33 | 977.57 | SL hit (close>static) qty=1.00 sl=979.90 alert=retest2 |

### Cycle 5 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 1069.40 | 983.33 | 983.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 12:15:00 | 1071.00 | 984.20 | 983.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1093.70 | 1101.70 | 1061.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-09 10:00:00 | 1093.70 | 1101.70 | 1061.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1091.20 | 1118.66 | 1097.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 1091.20 | 1118.66 | 1097.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1089.20 | 1118.36 | 1097.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1089.20 | 1118.36 | 1097.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1093.20 | 1104.42 | 1093.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 1089.80 | 1104.42 | 1093.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1093.40 | 1104.31 | 1093.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 1093.80 | 1104.31 | 1093.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1099.40 | 1104.26 | 1093.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:00:00 | 1101.50 | 1103.95 | 1093.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 10:30:00 | 1101.70 | 1109.50 | 1098.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 10:15:00 | 1086.80 | 1108.53 | 1098.24 | SL hit (close<static) qty=1.00 sl=1093.40 alert=retest2 |

### Cycle 6 — SELL (started 2025-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 15:15:00 | 1064.00 | 1093.61 | 1093.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 1061.20 | 1092.08 | 1092.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1081.30 | 1070.87 | 1079.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1081.30 | 1070.87 | 1079.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1081.30 | 1070.87 | 1079.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 10:00:00 | 1056.60 | 1075.64 | 1080.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1090.10 | 1074.79 | 1079.45 | SL hit (close>static) qty=1.00 sl=1090.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-09-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 14:15:00 | 1102.90 | 1082.45 | 1082.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 1120.20 | 1083.89 | 1083.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 10:15:00 | 1114.90 | 1114.92 | 1104.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:45:00 | 1113.70 | 1114.92 | 1104.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1149.30 | 1162.92 | 1147.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 1149.30 | 1162.92 | 1147.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 1144.20 | 1162.73 | 1147.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 1143.80 | 1162.73 | 1147.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 1139.10 | 1162.49 | 1147.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:45:00 | 1135.80 | 1162.49 | 1147.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1142.80 | 1161.18 | 1146.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:00:00 | 1142.80 | 1161.18 | 1146.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1141.80 | 1160.98 | 1146.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 1141.80 | 1160.98 | 1146.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1148.20 | 1160.52 | 1146.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 10:45:00 | 1151.60 | 1160.14 | 1146.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1137.60 | 1159.42 | 1147.29 | SL hit (close<static) qty=1.00 sl=1143.10 alert=retest2 |

### Cycle 8 — SELL (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 14:15:00 | 1124.50 | 1161.68 | 1161.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 1115.20 | 1156.69 | 1158.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-09 11:15:00 | 1082.30 | 1076.96 | 1104.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-09 11:45:00 | 1081.90 | 1076.96 | 1104.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1102.40 | 1078.71 | 1102.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 1092.00 | 1079.78 | 1102.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 1124.50 | 1081.55 | 1102.40 | SL hit (close>static) qty=1.00 sl=1108.30 alert=retest2 |

### Cycle 9 — BUY (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 11:15:00 | 1148.60 | 1117.10 | 1116.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 1163.00 | 1118.63 | 1117.75 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-24 09:30:00 | 1089.36 | 2024-06-04 09:15:00 | 1034.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 10:00:00 | 1089.51 | 2024-06-04 09:15:00 | 1035.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 11:00:00 | 1088.96 | 2024-06-04 09:15:00 | 1034.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 12:45:00 | 1088.67 | 2024-06-04 09:15:00 | 1034.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 09:30:00 | 1089.36 | 2024-06-05 09:15:00 | 1136.87 | STOP_HIT | 0.50 | -4.36% |
| SELL | retest2 | 2024-05-24 10:00:00 | 1089.51 | 2024-06-05 09:15:00 | 1136.87 | STOP_HIT | 0.50 | -4.35% |
| SELL | retest2 | 2024-05-24 11:00:00 | 1088.96 | 2024-06-05 09:15:00 | 1136.87 | STOP_HIT | 0.50 | -4.40% |
| SELL | retest2 | 2024-05-24 12:45:00 | 1088.67 | 2024-06-05 09:15:00 | 1136.87 | STOP_HIT | 0.50 | -4.43% |
| SELL | retest2 | 2024-06-13 12:30:00 | 1094.79 | 2024-06-13 13:15:00 | 1102.50 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-06-19 14:45:00 | 1094.40 | 2024-07-03 09:15:00 | 1111.98 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-01-10 09:15:00 | 961.95 | 2025-01-21 11:15:00 | 973.05 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-01-13 09:15:00 | 961.10 | 2025-01-21 11:15:00 | 973.05 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-01-13 11:00:00 | 966.25 | 2025-01-21 11:15:00 | 973.05 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-01-13 13:00:00 | 964.25 | 2025-01-21 11:15:00 | 973.05 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-01-14 14:15:00 | 960.10 | 2025-01-23 09:15:00 | 983.85 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-01-15 09:15:00 | 948.30 | 2025-01-23 09:15:00 | 983.85 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-01-20 13:15:00 | 961.05 | 2025-01-23 09:15:00 | 983.85 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-01-20 14:15:00 | 960.70 | 2025-01-23 09:15:00 | 983.85 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-01-27 14:30:00 | 961.90 | 2025-01-28 11:15:00 | 975.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-01-28 14:00:00 | 961.85 | 2025-01-31 09:15:00 | 1006.35 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2025-01-28 14:45:00 | 961.15 | 2025-01-31 09:15:00 | 1006.35 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2025-01-30 12:00:00 | 958.20 | 2025-01-31 09:15:00 | 1006.35 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2025-03-25 10:45:00 | 965.60 | 2025-03-28 09:15:00 | 1009.10 | STOP_HIT | 1.00 | -4.50% |
| SELL | retest2 | 2025-03-26 10:15:00 | 960.20 | 2025-03-28 09:15:00 | 1009.10 | STOP_HIT | 1.00 | -5.09% |
| BUY | retest2 | 2025-06-23 11:00:00 | 1101.50 | 2025-07-01 10:15:00 | 1086.80 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-06-30 10:30:00 | 1101.70 | 2025-07-01 10:15:00 | 1086.80 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-07-03 10:00:00 | 1101.80 | 2025-07-03 11:15:00 | 1093.20 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-07 11:30:00 | 1103.20 | 2025-07-09 15:15:00 | 1095.60 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-07-09 11:15:00 | 1105.50 | 2025-07-09 15:15:00 | 1095.60 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-07-09 12:00:00 | 1105.80 | 2025-07-10 14:15:00 | 1089.30 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-07-17 13:45:00 | 1107.80 | 2025-07-18 12:15:00 | 1095.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-07-18 09:30:00 | 1104.80 | 2025-07-18 12:15:00 | 1095.80 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-08-29 10:00:00 | 1056.60 | 2025-09-02 09:15:00 | 1090.10 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-12-05 10:45:00 | 1151.60 | 2025-12-09 09:15:00 | 1137.60 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-10 10:00:00 | 1155.00 | 2025-12-10 12:15:00 | 1141.70 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-12-11 11:15:00 | 1151.60 | 2025-12-11 14:15:00 | 1141.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-12 15:00:00 | 1150.70 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-12-15 10:30:00 | 1160.60 | 2026-01-28 09:15:00 | 1129.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2026-04-15 15:15:00 | 1092.00 | 2026-04-17 09:15:00 | 1124.50 | STOP_HIT | 1.00 | -2.98% |
