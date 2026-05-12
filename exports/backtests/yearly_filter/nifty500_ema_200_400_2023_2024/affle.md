# Affle 3i Ltd. (AFFLE)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1510.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 1 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 7 |
| TARGET_HIT | 11 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 52 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 34
- **Target hits / Stop hits / Partials:** 11 / 34 / 7
- **Avg / median % per leg:** 1.35% / -1.36%
- **Sum % (uncompounded):** 70.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 9 | 34.6% | 9 | 17 | 0 | 1.93% | 50.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 26 | 9 | 34.6% | 9 | 17 | 0 | 1.93% | 50.1% |
| SELL (all) | 26 | 9 | 34.6% | 2 | 17 | 7 | 0.78% | 20.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 26 | 9 | 34.6% | 2 | 17 | 7 | 0.78% | 20.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 52 | 18 | 34.6% | 11 | 34 | 7 | 1.35% | 70.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-06-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 10:15:00 | 1027.95 | 963.69 | 963.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-15 11:15:00 | 1038.30 | 964.43 | 963.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-10 12:15:00 | 1039.00 | 1040.43 | 1012.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-10 13:00:00 | 1039.00 | 1040.43 | 1012.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 12:15:00 | 1086.20 | 1102.62 | 1084.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-21 12:45:00 | 1086.25 | 1102.62 | 1084.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 1087.00 | 1102.25 | 1084.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:30:00 | 1089.60 | 1102.06 | 1084.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 10:15:00 | 1090.10 | 1101.94 | 1084.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 11:45:00 | 1094.15 | 1097.17 | 1083.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 13:00:00 | 1092.45 | 1097.13 | 1083.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 14:15:00 | 1093.00 | 1097.07 | 1083.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 15:15:00 | 1093.80 | 1097.02 | 1083.73 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 10:15:00 | 1087.25 | 1096.80 | 1083.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 10:30:00 | 1089.10 | 1096.80 | 1083.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 1101.35 | 1096.64 | 1083.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 13:45:00 | 1110.00 | 1096.62 | 1084.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:15:00 | 1106.75 | 1098.25 | 1086.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 09:15:00 | 1074.30 | 1097.46 | 1086.62 | SL hit (close<static) qty=1.00 sl=1080.20 alert=retest2 |

### Cycle 2 — SELL (started 2023-10-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-25 12:15:00 | 1023.10 | 1079.95 | 1080.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 09:15:00 | 1008.50 | 1077.64 | 1078.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 14:15:00 | 1068.85 | 1067.80 | 1073.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-02 15:00:00 | 1068.85 | 1067.80 | 1073.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 09:15:00 | 1067.70 | 1067.81 | 1073.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-03 10:30:00 | 1064.45 | 1067.79 | 1073.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-03 11:00:00 | 1065.40 | 1067.79 | 1073.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-06 10:15:00 | 1079.60 | 1067.76 | 1072.85 | SL hit (close>static) qty=1.00 sl=1077.90 alert=retest2 |

### Cycle 3 — BUY (started 2023-11-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 15:15:00 | 1115.00 | 1070.32 | 1070.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-01 09:15:00 | 1126.70 | 1070.88 | 1070.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 1249.45 | 1249.85 | 1197.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-18 09:45:00 | 1236.50 | 1249.85 | 1197.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 1202.00 | 1250.10 | 1203.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 1202.00 | 1250.10 | 1203.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 1197.25 | 1249.58 | 1203.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:00:00 | 1197.25 | 1249.58 | 1203.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 13:15:00 | 1188.45 | 1248.97 | 1202.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 13:45:00 | 1189.05 | 1248.97 | 1202.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 10:15:00 | 1200.00 | 1243.77 | 1204.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-29 10:30:00 | 1199.75 | 1243.77 | 1204.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-29 11:15:00 | 1201.00 | 1243.34 | 1204.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-30 10:15:00 | 1214.00 | 1241.33 | 1204.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-30 12:15:00 | 1196.80 | 1240.16 | 1204.07 | SL hit (close<static) qty=1.00 sl=1198.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 15:15:00 | 1127.00 | 1187.78 | 1187.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-27 13:15:00 | 1117.30 | 1174.96 | 1181.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-02 09:15:00 | 1096.45 | 1085.65 | 1118.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-02 10:00:00 | 1096.45 | 1085.65 | 1118.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 1113.75 | 1086.54 | 1117.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:30:00 | 1115.30 | 1086.54 | 1117.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 11:15:00 | 1115.00 | 1086.83 | 1117.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 11:45:00 | 1116.55 | 1086.83 | 1117.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 12:15:00 | 1108.85 | 1087.04 | 1117.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 12:45:00 | 1116.55 | 1087.04 | 1117.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 1093.05 | 1087.66 | 1117.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 09:30:00 | 1090.55 | 1088.20 | 1116.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-08 12:30:00 | 1091.00 | 1089.15 | 1115.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-09 10:45:00 | 1088.00 | 1089.02 | 1114.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 10:45:00 | 1090.30 | 1088.90 | 1113.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 1036.02 | 1088.27 | 1111.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 1036.45 | 1088.27 | 1111.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 1033.60 | 1088.27 | 1111.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 1035.78 | 1088.27 | 1111.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-18 09:15:00 | 1095.30 | 1086.54 | 1109.45 | SL hit (close>ema200) qty=0.50 sl=1086.54 alert=retest2 |

### Cycle 5 — BUY (started 2024-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 12:15:00 | 1203.00 | 1109.28 | 1108.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 09:15:00 | 1225.00 | 1118.30 | 1113.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-31 13:15:00 | 1142.90 | 1143.75 | 1128.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 1142.90 | 1143.75 | 1128.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 1119.00 | 1143.81 | 1129.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:45:00 | 1154.25 | 1138.62 | 1127.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 11:00:00 | 1162.80 | 1138.86 | 1127.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 09:45:00 | 1154.65 | 1139.92 | 1128.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 11:45:00 | 1153.65 | 1140.17 | 1129.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-12 09:15:00 | 1269.68 | 1154.68 | 1137.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-23 11:15:00 | 1558.25 | 1652.13 | 1652.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 09:15:00 | 1537.50 | 1647.52 | 1650.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 1588.80 | 1578.70 | 1609.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 10:00:00 | 1588.80 | 1578.70 | 1609.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1596.95 | 1578.89 | 1609.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:30:00 | 1595.00 | 1578.89 | 1609.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 1621.30 | 1579.31 | 1609.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:00:00 | 1621.30 | 1579.31 | 1609.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 1624.00 | 1579.75 | 1609.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:45:00 | 1621.85 | 1579.75 | 1609.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1634.80 | 1582.30 | 1610.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 1634.80 | 1582.30 | 1610.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 1671.40 | 1583.19 | 1610.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:45:00 | 1674.35 | 1583.19 | 1610.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1637.00 | 1591.76 | 1613.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 1637.00 | 1591.76 | 1613.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 13:15:00 | 1643.30 | 1593.02 | 1613.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 14:00:00 | 1643.30 | 1593.02 | 1613.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 1618.80 | 1594.46 | 1613.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 12:30:00 | 1590.20 | 1594.44 | 1613.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 10:15:00 | 1510.69 | 1592.12 | 1611.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-14 10:15:00 | 1611.85 | 1586.83 | 1607.82 | SL hit (close>ema200) qty=0.50 sl=1586.83 alert=retest2 |

### Cycle 7 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 1616.80 | 1542.48 | 1542.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 13:15:00 | 1622.00 | 1543.27 | 1542.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 10:15:00 | 1558.00 | 1558.16 | 1550.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 11:00:00 | 1558.00 | 1558.16 | 1550.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 1545.20 | 1558.03 | 1550.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:00:00 | 1545.20 | 1558.03 | 1550.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 1539.10 | 1557.84 | 1550.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 1539.10 | 1557.84 | 1550.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 10:15:00 | 1542.00 | 1554.80 | 1549.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 10:30:00 | 1535.00 | 1554.80 | 1549.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1550.20 | 1550.19 | 1547.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:30:00 | 1540.30 | 1550.19 | 1547.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 1566.90 | 1550.36 | 1547.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 11:45:00 | 1593.00 | 1550.79 | 1547.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-23 11:15:00 | 1752.30 | 1610.44 | 1582.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 15:15:00 | 1918.50 | 1949.77 | 1949.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 09:15:00 | 1890.00 | 1949.18 | 1949.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 1954.70 | 1945.37 | 1947.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 1954.70 | 1945.37 | 1947.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1954.70 | 1945.37 | 1947.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 1954.70 | 1945.37 | 1947.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1928.00 | 1945.20 | 1947.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:15:00 | 1925.00 | 1944.85 | 1947.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 09:15:00 | 1828.75 | 1940.73 | 1945.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-10 09:15:00 | 1732.50 | 1909.68 | 1928.33 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-09-27 11:45:00 | 1094.15 | 2023-10-09 09:15:00 | 1074.30 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2023-09-27 13:00:00 | 1092.45 | 2023-10-09 09:15:00 | 1074.30 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2023-09-27 14:15:00 | 1093.00 | 2023-10-09 09:15:00 | 1074.30 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2023-09-27 15:15:00 | 1093.80 | 2023-10-09 09:15:00 | 1074.30 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2023-09-29 13:45:00 | 1110.00 | 2023-10-09 09:15:00 | 1074.30 | STOP_HIT | 1.00 | -3.22% |
| BUY | retest2 | 2023-10-05 09:15:00 | 1106.75 | 2023-10-09 09:15:00 | 1074.30 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2023-11-03 10:30:00 | 1064.45 | 2023-11-06 10:15:00 | 1079.60 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2023-11-03 11:00:00 | 1065.40 | 2023-11-06 10:15:00 | 1079.60 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2023-11-07 09:15:00 | 1063.20 | 2023-11-23 12:15:00 | 1095.05 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2023-11-07 11:45:00 | 1065.55 | 2023-11-23 12:15:00 | 1095.05 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2023-11-22 10:15:00 | 1056.95 | 2023-11-23 12:15:00 | 1095.05 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2024-01-30 10:15:00 | 1214.00 | 2024-01-30 12:15:00 | 1196.80 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-01-31 09:30:00 | 1212.70 | 2024-02-05 13:15:00 | 1197.85 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-01-31 10:00:00 | 1214.40 | 2024-02-05 13:15:00 | 1197.85 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-04-05 09:30:00 | 1090.55 | 2024-04-15 09:15:00 | 1036.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-08 12:30:00 | 1091.00 | 2024-04-15 09:15:00 | 1036.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-09 10:45:00 | 1088.00 | 2024-04-15 09:15:00 | 1033.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-10 10:45:00 | 1090.30 | 2024-04-15 09:15:00 | 1035.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-05 09:30:00 | 1090.55 | 2024-04-18 09:15:00 | 1095.30 | STOP_HIT | 0.50 | -0.44% |
| SELL | retest2 | 2024-04-08 12:30:00 | 1091.00 | 2024-04-18 09:15:00 | 1095.30 | STOP_HIT | 0.50 | -0.39% |
| SELL | retest2 | 2024-04-09 10:45:00 | 1088.00 | 2024-04-18 09:15:00 | 1095.30 | STOP_HIT | 0.50 | -0.67% |
| SELL | retest2 | 2024-04-10 10:45:00 | 1090.30 | 2024-04-18 09:15:00 | 1095.30 | STOP_HIT | 0.50 | -0.46% |
| SELL | retest2 | 2024-05-06 09:45:00 | 1092.75 | 2024-05-17 09:15:00 | 1126.05 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2024-05-06 12:00:00 | 1093.80 | 2024-05-17 09:15:00 | 1126.05 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2024-05-06 14:15:00 | 1093.40 | 2024-05-17 09:15:00 | 1126.05 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2024-05-06 14:45:00 | 1092.00 | 2024-05-17 09:15:00 | 1126.05 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2024-06-06 09:45:00 | 1154.25 | 2024-06-12 09:15:00 | 1269.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 11:00:00 | 1162.80 | 2024-06-12 09:15:00 | 1270.12 | TARGET_HIT | 1.00 | 9.23% |
| BUY | retest2 | 2024-06-07 09:45:00 | 1154.65 | 2024-06-12 09:15:00 | 1269.02 | TARGET_HIT | 1.00 | 9.90% |
| BUY | retest2 | 2024-06-07 11:45:00 | 1153.65 | 2024-06-21 09:15:00 | 1279.08 | TARGET_HIT | 1.00 | 10.87% |
| BUY | retest2 | 2024-09-19 12:30:00 | 1522.35 | 2024-10-14 09:15:00 | 1659.24 | TARGET_HIT | 1.00 | 8.99% |
| BUY | retest2 | 2024-10-07 12:00:00 | 1508.40 | 2024-10-14 09:15:00 | 1658.09 | TARGET_HIT | 1.00 | 9.92% |
| BUY | retest2 | 2024-10-08 09:45:00 | 1507.35 | 2024-10-22 09:15:00 | 1481.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-10-08 11:00:00 | 1517.90 | 2024-10-22 09:15:00 | 1481.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-11-06 09:15:00 | 1583.00 | 2024-11-13 09:15:00 | 1511.10 | STOP_HIT | 1.00 | -4.54% |
| BUY | retest2 | 2024-11-14 11:45:00 | 1576.10 | 2024-11-21 15:15:00 | 1530.50 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-11-19 12:30:00 | 1578.70 | 2024-11-21 15:15:00 | 1530.50 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2024-11-19 13:15:00 | 1585.65 | 2024-11-21 15:15:00 | 1530.50 | STOP_HIT | 1.00 | -3.48% |
| BUY | retest2 | 2024-11-21 14:30:00 | 1560.00 | 2024-11-21 15:15:00 | 1530.50 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2024-11-22 09:15:00 | 1562.85 | 2024-12-02 13:15:00 | 1719.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-22 12:45:00 | 1564.35 | 2024-12-02 13:15:00 | 1720.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-13 10:45:00 | 1557.70 | 2025-01-13 12:15:00 | 1532.15 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-02-11 12:30:00 | 1590.20 | 2025-02-12 10:15:00 | 1510.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 12:30:00 | 1590.20 | 2025-02-14 10:15:00 | 1611.85 | STOP_HIT | 0.50 | -1.36% |
| SELL | retest2 | 2025-02-14 11:15:00 | 1602.25 | 2025-02-14 13:15:00 | 1522.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 11:15:00 | 1602.25 | 2025-02-20 09:15:00 | 1442.03 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-26 13:00:00 | 1607.80 | 2025-03-27 15:15:00 | 1640.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-03-26 13:30:00 | 1607.80 | 2025-03-27 15:15:00 | 1640.00 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-04-15 11:30:00 | 1514.40 | 2025-04-16 10:15:00 | 1563.80 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-05-12 11:45:00 | 1593.00 | 2025-05-23 11:15:00 | 1752.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-31 15:15:00 | 1925.00 | 2025-11-04 09:15:00 | 1828.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 15:15:00 | 1925.00 | 2025-11-10 09:15:00 | 1732.50 | TARGET_HIT | 0.50 | 10.00% |
