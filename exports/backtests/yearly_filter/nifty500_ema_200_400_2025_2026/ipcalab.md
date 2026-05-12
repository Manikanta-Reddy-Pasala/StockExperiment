# Ipca Laboratories Ltd. (IPCALAB)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1554.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 40 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 5 |
| TARGET_HIT | 11 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 44 (incl. partial bookings)
- **Trades open at end:** 6
- **Winners / losers:** 21 / 23
- **Target hits / Stop hits / Partials:** 11 / 28 / 5
- **Avg / median % per leg:** 1.85% / -1.10%
- **Sum % (uncompounded):** 81.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 11 | 47.8% | 11 | 12 | 0 | 3.56% | 81.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 11 | 47.8% | 11 | 12 | 0 | 3.56% | 81.8% |
| SELL (all) | 21 | 10 | 47.6% | 0 | 16 | 5 | -0.02% | -0.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 10 | 47.6% | 0 | 16 | 5 | -0.02% | -0.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 44 | 21 | 47.7% | 11 | 28 | 5 | 1.85% | 81.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 1482.20 | 1419.69 | 1419.39 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 10:15:00 | 1376.50 | 1419.35 | 1419.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 15:15:00 | 1368.80 | 1412.30 | 1415.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1416.60 | 1407.52 | 1413.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1416.60 | 1407.52 | 1413.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1416.60 | 1407.52 | 1413.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 1415.00 | 1407.52 | 1413.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1407.90 | 1407.52 | 1413.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 11:30:00 | 1401.90 | 1407.48 | 1413.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 12:45:00 | 1403.30 | 1407.44 | 1412.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1331.81 | 1393.86 | 1404.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-19 12:15:00 | 1333.13 | 1393.86 | 1404.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-30 13:15:00 | 1379.30 | 1375.44 | 1391.97 | SL hit (close>ema200) qty=0.50 sl=1375.44 alert=retest2 |

### Cycle 3 — BUY (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 10:15:00 | 1444.70 | 1402.50 | 1402.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 1450.60 | 1402.98 | 1402.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 1444.20 | 1454.87 | 1434.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:45:00 | 1442.00 | 1454.87 | 1434.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 1434.20 | 1454.66 | 1434.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 1423.70 | 1454.66 | 1434.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 1432.10 | 1454.44 | 1434.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 12:15:00 | 1444.50 | 1454.21 | 1434.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:00:00 | 1440.20 | 1454.07 | 1434.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 14:45:00 | 1442.10 | 1453.85 | 1434.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-05 12:15:00 | 1416.00 | 1452.65 | 1434.59 | SL hit (close<static) qty=1.00 sl=1427.20 alert=retest2 |

### Cycle 4 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 1371.70 | 1420.89 | 1421.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 1357.20 | 1411.30 | 1415.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 1417.80 | 1408.85 | 1414.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 12:15:00 | 1417.80 | 1408.85 | 1414.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 1417.80 | 1408.85 | 1414.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 13:00:00 | 1417.80 | 1408.85 | 1414.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 13:15:00 | 1403.60 | 1408.80 | 1414.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 1392.30 | 1408.62 | 1414.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1380.70 | 1408.21 | 1413.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 1435.00 | 1407.12 | 1413.24 | SL hit (close>static) qty=1.00 sl=1418.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1441.10 | 1348.96 | 1348.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 10:15:00 | 1448.00 | 1352.50 | 1350.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 12:15:00 | 1399.00 | 1414.53 | 1391.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 13:00:00 | 1399.00 | 1414.53 | 1391.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1391.70 | 1414.30 | 1391.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:45:00 | 1392.20 | 1414.30 | 1391.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1393.70 | 1414.10 | 1391.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 15:00:00 | 1399.70 | 1412.47 | 1391.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:30:00 | 1406.10 | 1412.35 | 1391.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 13:15:00 | 1400.00 | 1415.34 | 1396.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 10:15:00 | 1413.40 | 1414.21 | 1397.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1424.60 | 1414.31 | 1397.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-31 11:30:00 | 1429.50 | 1414.55 | 1397.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:00:00 | 1425.70 | 1415.42 | 1398.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 10:45:00 | 1425.70 | 1415.53 | 1398.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1432.50 | 1414.29 | 1399.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-08 12:15:00 | 1539.67 | 1426.73 | 1407.10 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-15 11:30:00 | 1400.40 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-05-15 12:30:00 | 1396.70 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-05-15 13:15:00 | 1399.30 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -3.81% |
| SELL | retest2 | 2025-05-16 10:00:00 | 1397.20 | 2025-05-19 09:15:00 | 1452.60 | STOP_HIT | 1.00 | -3.97% |
| SELL | retest2 | 2025-06-12 11:30:00 | 1401.90 | 2025-06-19 12:15:00 | 1331.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 12:45:00 | 1403.30 | 2025-06-19 12:15:00 | 1333.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-12 11:30:00 | 1401.90 | 2025-06-30 13:15:00 | 1379.30 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-06-12 12:45:00 | 1403.30 | 2025-06-30 13:15:00 | 1379.30 | STOP_HIT | 0.50 | 1.71% |
| BUY | retest2 | 2025-08-04 12:15:00 | 1444.50 | 2025-08-05 12:15:00 | 1416.00 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-08-04 13:00:00 | 1440.20 | 2025-08-05 12:15:00 | 1416.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-08-04 14:45:00 | 1442.10 | 2025-08-05 12:15:00 | 1416.00 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-08-25 11:30:00 | 1392.30 | 2025-08-26 14:15:00 | 1435.00 | STOP_HIT | 1.00 | -3.07% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1380.70 | 2025-08-26 14:15:00 | 1435.00 | STOP_HIT | 1.00 | -3.93% |
| SELL | retest2 | 2025-08-29 10:15:00 | 1388.50 | 2025-09-08 13:15:00 | 1319.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-29 10:15:00 | 1388.50 | 2025-09-19 13:15:00 | 1361.30 | STOP_HIT | 0.50 | 1.96% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1390.30 | 2025-09-26 09:15:00 | 1320.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1390.30 | 2025-09-26 12:15:00 | 1374.50 | STOP_HIT | 0.50 | 1.14% |
| SELL | retest2 | 2025-09-29 09:15:00 | 1351.00 | 2025-10-07 12:15:00 | 1388.40 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-09-29 10:15:00 | 1347.20 | 2025-10-07 12:15:00 | 1388.40 | STOP_HIT | 1.00 | -3.06% |
| SELL | retest2 | 2025-09-30 12:15:00 | 1350.00 | 2025-10-07 12:15:00 | 1388.40 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2025-10-08 09:15:00 | 1351.20 | 2025-10-16 12:15:00 | 1283.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-08 09:15:00 | 1351.20 | 2025-11-03 12:15:00 | 1312.40 | STOP_HIT | 0.50 | 2.87% |
| SELL | retest2 | 2025-11-13 13:15:00 | 1323.10 | 2025-11-14 09:15:00 | 1349.10 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2025-11-13 13:45:00 | 1327.50 | 2025-11-14 09:15:00 | 1349.10 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-12-18 15:00:00 | 1399.70 | 2026-01-08 12:15:00 | 1539.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-19 13:30:00 | 1406.10 | 2026-01-08 12:15:00 | 1546.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-26 13:15:00 | 1400.00 | 2026-01-08 12:15:00 | 1540.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-31 10:15:00 | 1413.40 | 2026-01-09 13:15:00 | 1554.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-31 11:30:00 | 1429.50 | 2026-01-09 13:15:00 | 1572.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-01 10:00:00 | 1425.70 | 2026-01-09 13:15:00 | 1568.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-01 10:45:00 | 1425.70 | 2026-01-09 13:15:00 | 1568.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1432.50 | 2026-01-09 13:15:00 | 1575.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-27 10:15:00 | 1474.50 | 2026-02-01 11:15:00 | 1424.50 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2026-01-27 11:15:00 | 1468.60 | 2026-02-01 11:15:00 | 1424.50 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2026-01-28 10:30:00 | 1472.60 | 2026-02-01 11:15:00 | 1424.50 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2026-01-29 11:45:00 | 1472.70 | 2026-02-01 11:15:00 | 1424.50 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2026-02-11 13:00:00 | 1469.80 | 2026-02-11 13:15:00 | 1436.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-02-12 14:45:00 | 1468.00 | 2026-02-13 09:15:00 | 1433.70 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2026-02-13 12:30:00 | 1474.00 | 2026-03-17 09:15:00 | 1621.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-13 13:00:00 | 1475.70 | 2026-03-17 09:15:00 | 1623.27 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-23 09:15:00 | 1479.10 | 2026-03-17 09:15:00 | 1608.20 | TARGET_HIT | 1.00 | 8.73% |
| BUY | retest2 | 2026-03-09 10:45:00 | 1462.00 | 2026-04-07 14:15:00 | 1441.60 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-02 11:45:00 | 1461.50 | 2026-04-07 14:15:00 | 1441.60 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2026-04-06 09:30:00 | 1457.60 | 2026-04-07 14:15:00 | 1441.60 | STOP_HIT | 1.00 | -1.10% |
