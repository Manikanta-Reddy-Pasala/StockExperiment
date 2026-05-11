# Nuvama Wealth Management Ltd. (NUVAMA)

## Backtest Summary

- **Window:** 2023-09-26 09:15:00 → 2026-05-08 15:15:00 (4512 bars)
- **Last close:** 1631.10
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
| ALERT2_SKIP | 1 |
| ALERT3 | 22 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 0 |
| TARGET_HIT | 8 |
| STOP_HIT | 13 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 11
- **Target hits / Stop hits / Partials:** 8 / 13 / 0
- **Avg / median % per leg:** 2.72% / -0.25%
- **Sum % (uncompounded):** 57.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 8 | 42.1% | 6 | 13 | 0 | 1.95% | 37.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 8 | 42.1% | 6 | 13 | 0 | 1.95% | 37.1% |
| SELL (all) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 10 | 47.6% | 8 | 13 | 0 | 2.72% | 57.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 10:15:00 | 1229.22 | 1344.87 | 1345.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 11:15:00 | 1228.20 | 1343.71 | 1344.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 14:15:00 | 1198.85 | 1189.52 | 1248.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 14:45:00 | 1197.37 | 1189.52 | 1248.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 1137.88 | 1091.78 | 1143.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:00:00 | 1137.88 | 1091.78 | 1143.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 1153.66 | 1093.47 | 1142.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-04 14:00:00 | 1125.46 | 1151.80 | 1162.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 09:15:00 | 1078.59 | 1151.39 | 1162.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 1012.91 | 1150.72 | 1162.19 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 12:15:00 | 1257.10 | 1164.40 | 1164.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 1267.90 | 1168.08 | 1166.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 09:15:00 | 1174.10 | 1183.08 | 1174.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 1174.10 | 1183.08 | 1174.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1174.10 | 1183.08 | 1174.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:00:00 | 1190.30 | 1182.69 | 1174.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 10:45:00 | 1191.40 | 1182.73 | 1174.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:30:00 | 1197.40 | 1181.46 | 1174.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 15:00:00 | 1192.20 | 1181.46 | 1174.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 1175.10 | 1182.42 | 1175.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 1175.10 | 1182.42 | 1175.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 1172.00 | 1182.32 | 1175.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 1174.00 | 1182.32 | 1175.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1179.90 | 1182.30 | 1175.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 10:45:00 | 1184.90 | 1182.36 | 1175.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 15:15:00 | 1190.00 | 1182.58 | 1175.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-14 09:15:00 | 1309.33 | 1191.13 | 1180.51 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 1382.00 | 1421.65 | 1421.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 12:15:00 | 1374.70 | 1418.24 | 1419.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 1294.00 | 1290.89 | 1331.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 14:00:00 | 1294.00 | 1290.89 | 1331.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1355.80 | 1291.68 | 1331.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:45:00 | 1356.60 | 1291.68 | 1331.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 1329.60 | 1292.06 | 1331.19 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1422.40 | 1357.21 | 1357.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 14:15:00 | 1450.00 | 1364.88 | 1361.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 11:15:00 | 1438.90 | 1440.45 | 1414.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 12:00:00 | 1438.90 | 1440.45 | 1414.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1412.10 | 1439.76 | 1414.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:45:00 | 1407.70 | 1439.76 | 1414.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1410.00 | 1439.46 | 1414.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 1448.90 | 1436.67 | 1413.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 11:45:00 | 1413.20 | 1446.11 | 1423.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 14:00:00 | 1415.30 | 1445.51 | 1423.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 14:45:00 | 1415.60 | 1445.20 | 1423.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1422.90 | 1443.81 | 1423.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-05 13:30:00 | 1427.00 | 1443.79 | 1423.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 15:00:00 | 1427.00 | 1443.36 | 1424.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1409.70 | 1442.86 | 1423.93 | SL hit (close<static) qty=1.00 sl=1419.50 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1270.00 | 1434.29 | 1434.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 1241.40 | 1429.18 | 1432.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 14:15:00 | 1392.00 | 1387.16 | 1407.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 15:00:00 | 1392.00 | 1387.16 | 1407.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1378.20 | 1384.15 | 1403.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 1378.20 | 1384.15 | 1403.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 1256.80 | 1200.95 | 1257.07 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 1360.60 | 1290.46 | 1290.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 15:15:00 | 1362.70 | 1291.84 | 1291.12 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-04-04 14:00:00 | 1125.46 | 2025-04-07 09:15:00 | 1012.91 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 1078.59 | 2025-04-07 09:15:00 | 970.73 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-06 10:00:00 | 1190.30 | 2025-05-14 09:15:00 | 1309.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-06 10:45:00 | 1191.40 | 2025-05-14 09:15:00 | 1310.54 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-07 14:30:00 | 1197.40 | 2025-05-14 09:15:00 | 1317.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-07 15:00:00 | 1192.20 | 2025-05-14 09:15:00 | 1311.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 10:45:00 | 1184.90 | 2025-05-14 09:15:00 | 1303.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-09 15:15:00 | 1190.00 | 2025-05-14 09:15:00 | 1309.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-26 09:15:00 | 1448.90 | 2025-12-09 09:15:00 | 1409.70 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-12-04 11:45:00 | 1413.20 | 2025-12-09 09:15:00 | 1409.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-12-04 14:00:00 | 1415.30 | 2025-12-19 11:15:00 | 1418.40 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-12-04 14:45:00 | 1415.60 | 2026-01-12 09:15:00 | 1416.00 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-12-05 13:30:00 | 1427.00 | 2026-01-12 09:15:00 | 1416.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-08 15:00:00 | 1427.00 | 2026-01-12 09:15:00 | 1416.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-11 09:15:00 | 1431.30 | 2026-01-12 09:15:00 | 1416.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-19 14:30:00 | 1426.20 | 2026-01-12 09:15:00 | 1416.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-12-23 11:00:00 | 1442.20 | 2026-01-20 10:15:00 | 1433.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-12-23 12:30:00 | 1441.70 | 2026-01-21 11:15:00 | 1396.00 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2025-12-29 10:30:00 | 1452.50 | 2026-01-21 11:15:00 | 1396.00 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2026-01-09 09:30:00 | 1440.00 | 2026-01-21 11:15:00 | 1396.00 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2026-01-16 09:15:00 | 1487.20 | 2026-01-21 11:15:00 | 1396.00 | STOP_HIT | 1.00 | -6.13% |
