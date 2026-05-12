# Kalpataru Projects International Ltd. (KPIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1277.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 5 |
| ALERT2_SKIP | 2 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 16 |
| TARGET_HIT | 13 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 19
- **Target hits / Stop hits / Partials:** 13 / 22 / 16
- **Avg / median % per leg:** 3.45% / 5.00%
- **Sum % (uncompounded):** 176.06%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 1 | 100.0% | 1 | 0 | 0 | 10.00% | 10.0% |
| SELL (all) | 50 | 31 | 62.0% | 12 | 22 | 16 | 3.32% | 166.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 50 | 31 | 62.0% | 12 | 22 | 16 | 3.32% | 166.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 51 | 32 | 62.7% | 13 | 22 | 16 | 3.45% | 176.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-24 15:15:00 | 1219.00 | 1304.25 | 1304.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 09:15:00 | 1194.75 | 1303.16 | 1303.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-01 17:15:00 | 1316.95 | 1281.99 | 1291.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-01 17:15:00 | 1316.95 | 1281.99 | 1291.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1316.95 | 1281.99 | 1291.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1316.95 | 1281.99 | 1291.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 1295.00 | 1276.42 | 1287.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 1295.00 | 1276.42 | 1287.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 1291.40 | 1276.57 | 1287.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:30:00 | 1293.10 | 1276.57 | 1287.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 13:15:00 | 1274.95 | 1276.55 | 1287.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-07 15:00:00 | 1267.80 | 1276.46 | 1287.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 1262.00 | 1276.45 | 1287.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 10:15:00 | 1204.41 | 1270.36 | 1283.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 10:15:00 | 1198.90 | 1270.36 | 1283.13 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-21 14:15:00 | 1141.02 | 1248.61 | 1269.58 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 1288.75 | 1255.52 | 1255.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 1312.65 | 1257.75 | 1256.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 09:15:00 | 1258.05 | 1268.77 | 1262.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 09:15:00 | 1258.05 | 1268.77 | 1262.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 1258.05 | 1268.77 | 1262.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 1258.05 | 1268.77 | 1262.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 1228.25 | 1268.37 | 1262.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 1228.25 | 1268.37 | 1262.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1257.00 | 1266.62 | 1261.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:45:00 | 1258.55 | 1266.62 | 1261.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 1259.50 | 1266.55 | 1261.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:30:00 | 1258.90 | 1266.55 | 1261.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 1260.00 | 1266.49 | 1261.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 1257.35 | 1266.49 | 1261.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 1260.00 | 1266.42 | 1261.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:15:00 | 1256.50 | 1266.42 | 1261.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 1241.95 | 1265.88 | 1261.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 10:30:00 | 1241.35 | 1265.88 | 1261.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 1170.10 | 1258.10 | 1258.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 1166.40 | 1257.19 | 1257.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 14:15:00 | 937.60 | 934.55 | 1007.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-20 15:00:00 | 937.60 | 934.55 | 1007.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1001.00 | 939.97 | 1004.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:00:00 | 1001.00 | 939.97 | 1004.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1004.75 | 940.61 | 1004.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 1007.20 | 940.61 | 1004.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 1005.75 | 941.26 | 1004.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:30:00 | 1003.85 | 941.26 | 1004.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1004.25 | 941.89 | 1004.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 13:15:00 | 1000.20 | 941.89 | 1004.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 13:15:00 | 1021.30 | 942.68 | 1004.81 | SL hit (close>static) qty=1.00 sl=1007.90 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 10:15:00 | 1126.45 | 988.03 | 987.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 1132.15 | 1023.90 | 1007.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 1181.00 | 1183.96 | 1148.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 10:15:00 | 1173.50 | 1183.96 | 1148.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 1144.00 | 1182.31 | 1148.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 1146.20 | 1182.31 | 1148.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1140.50 | 1181.89 | 1148.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 1175.00 | 1155.72 | 1142.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-21 09:15:00 | 1292.50 | 1194.78 | 1167.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 1200.60 | 1243.05 | 1243.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 15:15:00 | 1199.40 | 1242.19 | 1242.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 11:15:00 | 1205.00 | 1189.03 | 1208.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:45:00 | 1205.00 | 1189.03 | 1208.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 1198.80 | 1189.60 | 1208.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:00:00 | 1190.50 | 1190.04 | 1207.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1183.00 | 1190.02 | 1207.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:15:00 | 1190.30 | 1189.05 | 1205.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:45:00 | 1189.20 | 1189.05 | 1205.31 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 13:15:00 | 1203.10 | 1189.30 | 1205.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 14:00:00 | 1203.10 | 1189.30 | 1205.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 14:15:00 | 1202.00 | 1189.42 | 1205.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 15:00:00 | 1202.00 | 1189.42 | 1205.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 1207.10 | 1189.71 | 1205.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:15:00 | 1203.60 | 1189.88 | 1205.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 12:15:00 | 1204.40 | 1190.04 | 1205.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 13:00:00 | 1200.00 | 1190.14 | 1205.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 11:15:00 | 1216.30 | 1190.95 | 1205.19 | SL hit (close>static) qty=1.00 sl=1213.90 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 1228.10 | 1133.11 | 1132.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 13:15:00 | 1243.00 | 1136.30 | 1134.50 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-11-07 15:00:00 | 1267.80 | 2024-11-13 10:15:00 | 1204.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 1262.00 | 2024-11-13 10:15:00 | 1198.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-07 15:00:00 | 1267.80 | 2024-11-21 14:15:00 | 1141.02 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-08 09:15:00 | 1262.00 | 2024-11-22 09:15:00 | 1135.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-06 15:00:00 | 1263.40 | 2024-12-10 10:15:00 | 1297.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2024-12-13 10:30:00 | 1266.30 | 2024-12-13 14:15:00 | 1300.00 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2024-12-24 11:30:00 | 1266.70 | 2024-12-26 12:15:00 | 1285.55 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-12-24 12:00:00 | 1270.00 | 2024-12-26 12:15:00 | 1285.55 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-03-25 13:15:00 | 1000.20 | 2025-03-25 13:15:00 | 1021.30 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-03-26 11:45:00 | 999.00 | 2025-03-26 14:15:00 | 1009.75 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-03-26 12:15:00 | 999.70 | 2025-03-26 14:15:00 | 1009.75 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-03-26 13:30:00 | 1000.00 | 2025-03-26 14:15:00 | 1009.75 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-03-27 15:00:00 | 985.00 | 2025-04-04 10:15:00 | 935.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 10:30:00 | 976.35 | 2025-04-04 14:15:00 | 927.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-27 15:00:00 | 985.00 | 2025-04-07 09:15:00 | 886.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-28 10:30:00 | 976.35 | 2025-04-07 09:15:00 | 878.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-23 10:00:00 | 978.85 | 2025-04-28 14:15:00 | 935.75 | PARTIAL | 0.50 | 4.40% |
| SELL | retest2 | 2025-04-23 10:00:00 | 978.85 | 2025-04-28 15:15:00 | 957.95 | STOP_HIT | 0.50 | 2.14% |
| SELL | retest2 | 2025-04-23 14:15:00 | 985.00 | 2025-05-06 14:15:00 | 929.91 | PARTIAL | 0.50 | 5.59% |
| SELL | retest2 | 2025-05-06 10:00:00 | 949.45 | 2025-05-07 09:15:00 | 901.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-23 14:15:00 | 985.00 | 2025-05-07 14:15:00 | 973.65 | STOP_HIT | 0.50 | 1.15% |
| SELL | retest2 | 2025-05-06 10:00:00 | 949.45 | 2025-05-07 14:15:00 | 973.65 | STOP_HIT | 0.50 | -2.55% |
| SELL | retest2 | 2025-05-07 14:45:00 | 951.40 | 2025-05-12 10:15:00 | 986.70 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-05-08 14:45:00 | 949.80 | 2025-05-12 10:15:00 | 986.70 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-08-08 09:15:00 | 1175.00 | 2025-08-21 09:15:00 | 1292.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-29 11:00:00 | 1190.50 | 2026-01-05 11:15:00 | 1216.30 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-12-30 09:15:00 | 1183.00 | 2026-01-05 11:15:00 | 1216.30 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-01-01 11:15:00 | 1190.30 | 2026-01-05 11:15:00 | 1216.30 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2026-01-01 11:45:00 | 1189.20 | 2026-01-05 11:15:00 | 1216.30 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2026-01-02 11:15:00 | 1203.60 | 2026-01-05 11:15:00 | 1216.30 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-02 12:15:00 | 1204.40 | 2026-01-05 11:15:00 | 1216.30 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-01-02 13:00:00 | 1200.00 | 2026-01-05 11:15:00 | 1216.30 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-01-05 13:15:00 | 1204.00 | 2026-01-08 14:15:00 | 1143.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 09:15:00 | 1195.10 | 2026-01-08 14:15:00 | 1135.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 10:00:00 | 1195.40 | 2026-01-08 14:15:00 | 1135.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 15:15:00 | 1193.50 | 2026-01-08 15:15:00 | 1133.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-05 13:15:00 | 1204.00 | 2026-01-21 09:15:00 | 1083.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-06 09:15:00 | 1195.10 | 2026-01-21 09:15:00 | 1075.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-06 10:00:00 | 1195.40 | 2026-01-21 09:15:00 | 1075.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-06 15:15:00 | 1193.50 | 2026-01-21 10:15:00 | 1074.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-24 12:15:00 | 1195.80 | 2026-02-25 14:15:00 | 1213.20 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-03-11 14:30:00 | 1141.90 | 2026-03-16 09:15:00 | 1084.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 09:15:00 | 1119.50 | 2026-03-16 09:15:00 | 1079.39 | PARTIAL | 0.50 | 3.58% |
| SELL | retest2 | 2026-03-12 12:45:00 | 1136.20 | 2026-03-16 09:15:00 | 1084.99 | PARTIAL | 0.50 | 4.51% |
| SELL | retest2 | 2026-03-12 13:30:00 | 1142.10 | 2026-03-16 10:15:00 | 1063.52 | PARTIAL | 0.50 | 6.88% |
| SELL | retest2 | 2026-03-13 10:30:00 | 1116.10 | 2026-03-17 10:15:00 | 1060.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 14:30:00 | 1141.90 | 2026-03-23 09:15:00 | 1027.71 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-12 09:15:00 | 1119.50 | 2026-03-23 09:15:00 | 1022.58 | TARGET_HIT | 0.50 | 8.66% |
| SELL | retest2 | 2026-03-12 12:45:00 | 1136.20 | 2026-03-23 09:15:00 | 1027.89 | TARGET_HIT | 0.50 | 9.53% |
| SELL | retest2 | 2026-03-12 13:30:00 | 1142.10 | 2026-03-23 12:15:00 | 1007.55 | TARGET_HIT | 0.50 | 11.78% |
| SELL | retest2 | 2026-03-13 10:30:00 | 1116.10 | 2026-03-25 13:15:00 | 1110.70 | STOP_HIT | 0.50 | 0.48% |
