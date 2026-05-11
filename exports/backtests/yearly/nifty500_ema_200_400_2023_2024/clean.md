# Clean Science and Technology Ltd. (CLEAN)

## Backtest Summary

- **Window:** 2022-04-07 14:15:00 → 2026-05-08 15:15:00 (7049 bars)
- **Last close:** 891.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 1 |
| ALERT3 | 79 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 61 |
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 60 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 55
- **Target hits / Stop hits / Partials:** 1 / 60 / 7
- **Avg / median % per leg:** -1.92% / -2.08%
- **Sum % (uncompounded):** -130.38%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 0 | 0.0% | 0 | 30 | 0 | -2.45% | -73.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 0 | 0.0% | 0 | 30 | 0 | -2.45% | -73.5% |
| SELL (all) | 38 | 13 | 34.2% | 1 | 30 | 7 | -1.50% | -56.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 38 | 13 | 34.2% | 1 | 30 | 7 | -1.50% | -56.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 68 | 13 | 19.1% | 1 | 60 | 7 | -1.92% | -130.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-06-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-09 14:15:00 | 1390.00 | 1436.91 | 1437.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 15:15:00 | 1368.00 | 1436.23 | 1436.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 12:15:00 | 1397.65 | 1396.76 | 1411.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-30 12:45:00 | 1398.20 | 1396.76 | 1411.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-03 09:15:00 | 1406.95 | 1396.87 | 1411.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-03 15:00:00 | 1402.00 | 1397.39 | 1411.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-07-11 14:15:00 | 1331.90 | 1387.29 | 1403.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-08-07 09:15:00 | 1338.00 | 1329.63 | 1358.40 | SL hit (close>ema200) qty=0.50 sl=1329.63 alert=retest2 |

### Cycle 2 — BUY (started 2023-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 11:15:00 | 1409.10 | 1374.34 | 1374.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 1426.65 | 1376.04 | 1375.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-21 13:15:00 | 1424.85 | 1427.41 | 1407.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-21 14:00:00 | 1424.85 | 1427.41 | 1407.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 14:15:00 | 1407.95 | 1426.63 | 1408.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-22 15:00:00 | 1407.95 | 1426.63 | 1408.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 15:15:00 | 1402.50 | 1426.39 | 1408.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 09:15:00 | 1410.50 | 1426.39 | 1408.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 1412.35 | 1426.25 | 1408.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-26 13:15:00 | 1422.40 | 1424.88 | 1408.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-26 14:45:00 | 1421.50 | 1424.77 | 1408.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-26 15:15:00 | 1424.50 | 1424.77 | 1408.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 11:30:00 | 1421.55 | 1424.48 | 1408.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 12:15:00 | 1411.75 | 1424.13 | 1409.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 12:30:00 | 1412.20 | 1424.13 | 1409.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 1407.95 | 1423.96 | 1409.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:00:00 | 1407.95 | 1423.96 | 1409.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 1415.00 | 1423.88 | 1409.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 1415.00 | 1423.88 | 1409.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 14:15:00 | 1406.90 | 1423.30 | 1409.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 15:00:00 | 1406.90 | 1423.30 | 1409.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 15:15:00 | 1411.50 | 1423.18 | 1409.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:15:00 | 1415.00 | 1423.18 | 1409.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 1410.00 | 1423.05 | 1409.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-03 10:15:00 | 1415.95 | 1423.05 | 1409.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 09:30:00 | 1418.90 | 1422.22 | 1409.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-04 10:30:00 | 1415.60 | 1422.13 | 1409.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:15:00 | 1421.00 | 1421.63 | 1409.47 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 11:15:00 | 1411.40 | 1421.41 | 1409.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 11:45:00 | 1410.60 | 1421.41 | 1409.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 12:15:00 | 1408.60 | 1421.28 | 1409.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 12:30:00 | 1409.35 | 1421.28 | 1409.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 13:15:00 | 1412.15 | 1421.19 | 1409.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 13:30:00 | 1410.40 | 1421.19 | 1409.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 14:15:00 | 1410.00 | 1421.08 | 1409.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-05 15:00:00 | 1410.00 | 1421.08 | 1409.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 15:15:00 | 1410.95 | 1420.98 | 1409.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-06 09:15:00 | 1418.55 | 1420.98 | 1409.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 15:15:00 | 1406.00 | 1420.74 | 1409.83 | SL hit (close<static) qty=1.00 sl=1408.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-20 10:15:00 | 1370.00 | 1402.01 | 1402.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-20 11:15:00 | 1364.10 | 1401.64 | 1401.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 11:15:00 | 1370.00 | 1369.23 | 1383.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-02 12:00:00 | 1370.00 | 1369.23 | 1383.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 12:15:00 | 1360.00 | 1369.14 | 1383.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 12:45:00 | 1380.00 | 1369.14 | 1383.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 13:15:00 | 1368.20 | 1369.13 | 1383.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-03 09:15:00 | 1332.10 | 1369.22 | 1383.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-03 12:30:00 | 1355.00 | 1368.89 | 1382.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-08 09:15:00 | 1353.00 | 1369.12 | 1381.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 10:00:00 | 1358.40 | 1370.49 | 1379.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 10:15:00 | 1409.15 | 1361.50 | 1372.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-30 10:15:00 | 1409.15 | 1361.50 | 1372.58 | SL hit (close>static) qty=1.00 sl=1394.90 alert=retest2 |

### Cycle 4 — BUY (started 2023-12-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-07 15:15:00 | 1431.95 | 1381.53 | 1381.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 09:15:00 | 1453.95 | 1386.38 | 1383.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 1496.15 | 1500.65 | 1459.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-10 09:30:00 | 1496.50 | 1500.65 | 1459.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 1465.00 | 1501.61 | 1467.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-18 09:45:00 | 1445.75 | 1501.61 | 1467.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 1487.85 | 1501.47 | 1467.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-18 11:45:00 | 1491.00 | 1501.34 | 1468.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:15:00 | 1501.70 | 1500.56 | 1468.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 12:30:00 | 1491.00 | 1500.23 | 1468.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-20 10:45:00 | 1491.25 | 1499.87 | 1469.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 1471.05 | 1498.85 | 1469.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 10:00:00 | 1471.05 | 1498.85 | 1469.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 10:15:00 | 1471.15 | 1498.57 | 1469.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 11:15:00 | 1467.00 | 1498.57 | 1469.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 11:15:00 | 1459.45 | 1498.18 | 1469.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 12:00:00 | 1459.45 | 1498.18 | 1469.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 12:15:00 | 1456.00 | 1497.76 | 1469.61 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-01-23 12:15:00 | 1456.00 | 1497.76 | 1469.61 | SL hit (close<static) qty=1.00 sl=1457.75 alert=retest2 |

### Cycle 5 — SELL (started 2024-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 13:15:00 | 1395.40 | 1454.19 | 1454.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-14 09:15:00 | 1384.45 | 1448.33 | 1451.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-21 10:15:00 | 1446.90 | 1434.10 | 1442.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-21 10:15:00 | 1446.90 | 1434.10 | 1442.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 1446.90 | 1434.10 | 1442.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 11:00:00 | 1446.90 | 1434.10 | 1442.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 11:15:00 | 1456.15 | 1434.32 | 1442.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 11:30:00 | 1460.85 | 1434.32 | 1442.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 12:15:00 | 1473.00 | 1434.70 | 1443.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-21 12:45:00 | 1472.00 | 1434.70 | 1443.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 1442.15 | 1435.88 | 1443.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:30:00 | 1443.95 | 1435.88 | 1443.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 12:15:00 | 1448.75 | 1436.01 | 1443.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 13:00:00 | 1448.75 | 1436.01 | 1443.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 1456.00 | 1436.21 | 1443.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 13:45:00 | 1455.05 | 1436.21 | 1443.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 1370.40 | 1350.73 | 1378.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 15:00:00 | 1356.55 | 1351.38 | 1377.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-15 09:15:00 | 1335.90 | 1351.54 | 1377.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-09 14:15:00 | 1288.72 | 1322.44 | 1347.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-15 14:15:00 | 1325.80 | 1315.54 | 1340.88 | SL hit (close>ema200) qty=0.50 sl=1315.54 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 13:15:00 | 1457.05 | 1347.56 | 1347.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 13:15:00 | 1472.35 | 1361.24 | 1354.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 1437.75 | 1442.36 | 1409.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 10:00:00 | 1437.75 | 1442.36 | 1409.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 1428.00 | 1443.36 | 1412.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 1425.25 | 1443.36 | 1412.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 10:15:00 | 1483.70 | 1545.78 | 1505.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 11:00:00 | 1483.70 | 1545.78 | 1505.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1472.30 | 1545.04 | 1504.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 1472.30 | 1545.04 | 1504.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1484.60 | 1541.25 | 1504.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 1484.60 | 1541.25 | 1504.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 1481.75 | 1537.54 | 1503.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:00:00 | 1481.75 | 1537.54 | 1503.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 1508.00 | 1529.80 | 1502.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 13:15:00 | 1513.50 | 1529.80 | 1502.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 13:45:00 | 1514.25 | 1529.62 | 1502.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 1519.85 | 1529.21 | 1502.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 10:15:00 | 1501.00 | 1528.76 | 1502.34 | SL hit (close<static) qty=1.00 sl=1502.05 alert=retest2 |

### Cycle 7 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 1465.00 | 1532.78 | 1532.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 1435.95 | 1529.54 | 1531.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 11:15:00 | 1378.50 | 1359.73 | 1419.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-10 12:00:00 | 1378.50 | 1359.73 | 1419.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 1407.55 | 1361.24 | 1418.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 14:00:00 | 1373.10 | 1427.38 | 1432.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 15:15:00 | 1368.05 | 1426.92 | 1432.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 09:45:00 | 1374.80 | 1425.80 | 1431.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-14 12:00:00 | 1375.80 | 1424.87 | 1431.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 1425.60 | 1417.66 | 1426.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:45:00 | 1428.25 | 1417.66 | 1426.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 14:15:00 | 1425.00 | 1417.74 | 1426.49 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-21 14:15:00 | 1436.45 | 1419.32 | 1426.72 | SL hit (close>static) qty=1.00 sl=1436.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 1425.20 | 1255.50 | 1254.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 10:15:00 | 1441.50 | 1271.93 | 1263.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1407.20 | 1412.68 | 1360.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 1407.20 | 1412.68 | 1360.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1350.60 | 1444.01 | 1407.19 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 10:15:00 | 1233.90 | 1379.04 | 1379.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 13:15:00 | 1224.40 | 1357.26 | 1367.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 09:15:00 | 1195.70 | 1192.79 | 1239.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-17 09:30:00 | 1200.90 | 1192.79 | 1239.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 747.30 | 721.12 | 757.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:30:00 | 751.40 | 721.12 | 757.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 752.00 | 722.94 | 757.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 11:45:00 | 746.30 | 723.45 | 757.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 747.85 | 724.15 | 757.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:30:00 | 742.40 | 727.02 | 757.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 13:15:00 | 745.15 | 727.72 | 757.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 755.85 | 728.69 | 757.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 758.25 | 728.69 | 757.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 753.90 | 729.44 | 757.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 757.35 | 729.44 | 757.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 11:15:00 | 759.00 | 730.58 | 757.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:00:00 | 759.00 | 730.58 | 757.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 12:15:00 | 766.45 | 730.93 | 757.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-22 12:30:00 | 770.05 | 730.93 | 757.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 13:15:00 | 764.05 | 731.26 | 757.10 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 831.95 | 732.94 | 757.56 | SL hit (close>static) qty=1.00 sl=770.00 alert=retest2 |

### Cycle 10 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 875.60 | 775.43 | 775.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 15:15:00 | 879.00 | 779.40 | 777.05 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-25 15:15:00 | 1454.00 | 2023-05-26 13:15:00 | 1433.05 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2023-07-03 15:00:00 | 1402.00 | 2023-07-11 14:15:00 | 1331.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-07-03 15:00:00 | 1402.00 | 2023-08-07 09:15:00 | 1338.00 | STOP_HIT | 0.50 | 4.56% |
| SELL | retest2 | 2023-08-09 10:00:00 | 1398.15 | 2023-08-10 09:15:00 | 1419.65 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2023-08-09 11:00:00 | 1401.70 | 2023-08-10 09:15:00 | 1419.65 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2023-08-09 14:00:00 | 1402.25 | 2023-08-10 09:15:00 | 1419.65 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2023-09-26 13:15:00 | 1422.40 | 2023-10-06 15:15:00 | 1406.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2023-09-26 14:45:00 | 1421.50 | 2023-10-09 09:15:00 | 1392.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2023-09-26 15:15:00 | 1424.50 | 2023-10-09 09:15:00 | 1392.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2023-09-27 11:30:00 | 1421.55 | 2023-10-09 09:15:00 | 1392.00 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2023-10-03 10:15:00 | 1415.95 | 2023-10-09 09:15:00 | 1392.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2023-10-04 09:30:00 | 1418.90 | 2023-10-09 09:15:00 | 1392.00 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2023-10-04 10:30:00 | 1415.60 | 2023-10-09 09:15:00 | 1392.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2023-10-05 09:15:00 | 1421.00 | 2023-10-09 09:15:00 | 1392.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2023-10-06 09:15:00 | 1418.55 | 2023-10-09 09:15:00 | 1392.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2023-11-03 09:15:00 | 1332.10 | 2023-11-30 10:15:00 | 1409.15 | STOP_HIT | 1.00 | -5.78% |
| SELL | retest2 | 2023-11-03 12:30:00 | 1355.00 | 2023-11-30 10:15:00 | 1409.15 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2023-11-08 09:15:00 | 1353.00 | 2023-11-30 10:15:00 | 1409.15 | STOP_HIT | 1.00 | -4.15% |
| SELL | retest2 | 2023-11-20 10:00:00 | 1358.40 | 2023-11-30 10:15:00 | 1409.15 | STOP_HIT | 1.00 | -3.74% |
| SELL | retest2 | 2023-11-30 12:30:00 | 1376.00 | 2023-12-01 13:15:00 | 1443.95 | STOP_HIT | 1.00 | -4.94% |
| BUY | retest2 | 2024-01-18 11:45:00 | 1491.00 | 2024-01-23 12:15:00 | 1456.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-01-19 09:15:00 | 1501.70 | 2024-01-23 12:15:00 | 1456.00 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2024-01-19 12:30:00 | 1491.00 | 2024-01-23 12:15:00 | 1456.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-01-20 10:45:00 | 1491.25 | 2024-01-23 12:15:00 | 1456.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2024-04-12 15:00:00 | 1356.55 | 2024-05-09 14:15:00 | 1288.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-12 15:00:00 | 1356.55 | 2024-05-15 14:15:00 | 1325.80 | STOP_HIT | 0.50 | 2.27% |
| SELL | retest2 | 2024-04-15 09:15:00 | 1335.90 | 2024-05-22 14:15:00 | 1394.75 | STOP_HIT | 1.00 | -4.41% |
| SELL | retest2 | 2024-05-27 09:30:00 | 1356.15 | 2024-06-04 09:15:00 | 1288.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 15:15:00 | 1355.00 | 2024-06-04 09:15:00 | 1287.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 15:00:00 | 1341.40 | 2024-06-04 11:15:00 | 1274.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-27 09:30:00 | 1356.15 | 2024-06-10 09:15:00 | 1343.15 | STOP_HIT | 0.50 | 0.96% |
| SELL | retest2 | 2024-05-28 15:15:00 | 1355.00 | 2024-06-10 09:15:00 | 1343.15 | STOP_HIT | 0.50 | 0.87% |
| SELL | retest2 | 2024-05-29 15:00:00 | 1341.40 | 2024-06-10 09:15:00 | 1343.15 | STOP_HIT | 0.50 | -0.13% |
| SELL | retest2 | 2024-06-10 10:15:00 | 1337.20 | 2024-06-11 10:15:00 | 1378.55 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2024-06-10 12:00:00 | 1341.90 | 2024-06-11 10:15:00 | 1378.55 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2024-06-10 12:45:00 | 1341.30 | 2024-06-11 10:15:00 | 1378.55 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2024-09-04 13:15:00 | 1513.50 | 2024-09-05 10:15:00 | 1501.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-09-04 13:45:00 | 1514.25 | 2024-09-05 10:15:00 | 1501.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2024-09-05 09:15:00 | 1519.85 | 2024-09-05 10:15:00 | 1501.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-09-06 09:15:00 | 1514.90 | 2024-09-06 09:15:00 | 1488.35 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-09-09 11:15:00 | 1529.00 | 2024-10-07 10:15:00 | 1493.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-09-09 12:30:00 | 1524.55 | 2024-10-07 10:15:00 | 1493.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-09-19 11:30:00 | 1525.90 | 2024-10-07 10:15:00 | 1493.00 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2024-09-24 14:30:00 | 1524.90 | 2024-10-07 14:15:00 | 1506.95 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-09-25 09:15:00 | 1526.10 | 2024-10-25 10:15:00 | 1467.80 | STOP_HIT | 1.00 | -3.82% |
| BUY | retest2 | 2024-09-25 14:45:00 | 1532.25 | 2024-10-25 10:15:00 | 1467.80 | STOP_HIT | 1.00 | -4.21% |
| BUY | retest2 | 2024-09-26 11:00:00 | 1529.25 | 2024-10-25 10:15:00 | 1467.80 | STOP_HIT | 1.00 | -4.02% |
| BUY | retest2 | 2024-10-07 12:30:00 | 1532.00 | 2024-10-25 10:15:00 | 1467.80 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-11-06 11:45:00 | 1542.95 | 2024-11-07 13:15:00 | 1478.95 | STOP_HIT | 1.00 | -4.15% |
| BUY | retest2 | 2024-11-06 14:15:00 | 1542.70 | 2024-11-07 13:15:00 | 1478.95 | STOP_HIT | 1.00 | -4.13% |
| BUY | retest2 | 2024-11-06 14:45:00 | 1541.95 | 2024-11-07 13:15:00 | 1478.95 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2024-11-07 10:15:00 | 1542.30 | 2024-11-07 13:15:00 | 1478.95 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2025-01-13 14:00:00 | 1373.10 | 2025-01-21 14:15:00 | 1436.45 | STOP_HIT | 1.00 | -4.61% |
| SELL | retest2 | 2025-01-13 15:15:00 | 1368.05 | 2025-01-21 14:15:00 | 1436.45 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2025-01-14 09:45:00 | 1374.80 | 2025-01-21 14:15:00 | 1436.45 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2025-01-14 12:00:00 | 1375.80 | 2025-01-21 14:15:00 | 1436.45 | STOP_HIT | 1.00 | -4.41% |
| SELL | retest2 | 2025-01-22 09:15:00 | 1419.20 | 2025-01-23 09:15:00 | 1439.80 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-01-22 11:15:00 | 1422.85 | 2025-01-23 09:15:00 | 1439.80 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1416.15 | 2025-01-27 09:15:00 | 1345.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1416.15 | 2025-01-31 09:15:00 | 1404.75 | STOP_HIT | 0.50 | 0.80% |
| SELL | retest2 | 2025-02-01 11:30:00 | 1423.10 | 2025-02-01 14:15:00 | 1438.65 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-02-01 13:15:00 | 1397.90 | 2025-02-01 14:15:00 | 1438.65 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1393.75 | 2025-02-12 09:15:00 | 1324.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1393.75 | 2025-02-27 13:15:00 | 1254.38 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-16 11:45:00 | 746.30 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -11.48% |
| SELL | retest2 | 2026-04-16 14:30:00 | 747.85 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -11.25% |
| SELL | retest2 | 2026-04-20 09:30:00 | 742.40 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -12.06% |
| SELL | retest2 | 2026-04-20 13:15:00 | 745.15 | 2026-04-23 09:15:00 | 831.95 | STOP_HIT | 1.00 | -11.65% |
