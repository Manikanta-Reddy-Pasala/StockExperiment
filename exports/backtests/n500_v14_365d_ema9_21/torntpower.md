# Torrent Power Ltd. (TORNTPOWER)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1717.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 51 |
| ALERT2 | 50 |
| ALERT2_SKIP | 26 |
| ALERT3 | 135 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 62 |
| PARTIAL | 10 |
| TARGET_HIT | 3 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 77 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 52
- **Target hits / Stop hits / Partials:** 3 / 64 / 10
- **Avg / median % per leg:** 0.76% / -0.89%
- **Sum % (uncompounded):** 58.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 3 | 11.5% | 2 | 24 | 0 | -0.03% | -0.7% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.57% | -2.3% |
| BUY @ 3rd Alert (retest2) | 22 | 3 | 13.6% | 2 | 20 | 0 | 0.07% | 1.6% |
| SELL (all) | 51 | 22 | 43.1% | 1 | 40 | 10 | 1.16% | 59.0% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.36% | -1.4% |
| SELL @ 3rd Alert (retest2) | 50 | 22 | 44.0% | 1 | 39 | 10 | 1.21% | 60.4% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.73% | -3.7% |
| retest2 (combined) | 72 | 25 | 34.7% | 3 | 59 | 10 | 0.86% | 62.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 1466.60 | 1442.05 | 1439.57 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 1433.60 | 1439.95 | 1439.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 14:15:00 | 1425.50 | 1435.61 | 1437.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-14 09:15:00 | 1444.20 | 1436.16 | 1437.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 1444.20 | 1436.16 | 1437.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 1444.20 | 1436.16 | 1437.68 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 1451.60 | 1439.25 | 1438.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 1457.70 | 1446.82 | 1442.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 1430.40 | 1443.73 | 1442.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 1430.40 | 1443.73 | 1442.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 1430.40 | 1443.73 | 1442.19 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 1425.80 | 1440.14 | 1440.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 11:15:00 | 1412.50 | 1434.61 | 1438.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 14:15:00 | 1430.80 | 1429.37 | 1434.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 14:15:00 | 1430.80 | 1429.37 | 1434.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 1430.80 | 1429.37 | 1434.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:30:00 | 1430.20 | 1429.37 | 1434.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 1435.00 | 1430.50 | 1434.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 1459.20 | 1430.50 | 1434.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 1438.90 | 1432.18 | 1434.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:30:00 | 1443.20 | 1432.18 | 1434.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1443.00 | 1434.34 | 1435.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 12:45:00 | 1434.50 | 1435.69 | 1436.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 13:15:00 | 1441.00 | 1436.76 | 1436.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 13:15:00 | 1441.00 | 1436.76 | 1436.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 1465.10 | 1444.23 | 1440.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 1433.60 | 1443.27 | 1441.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 13:15:00 | 1433.60 | 1443.27 | 1441.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 1433.60 | 1443.27 | 1441.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 1433.60 | 1443.27 | 1441.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 14:15:00 | 1424.50 | 1439.52 | 1439.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 10:15:00 | 1419.20 | 1433.42 | 1436.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1412.00 | 1411.98 | 1422.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 10:00:00 | 1412.00 | 1411.98 | 1422.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 1400.30 | 1397.19 | 1403.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:45:00 | 1399.40 | 1397.19 | 1403.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 1400.20 | 1397.80 | 1402.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 14:45:00 | 1392.50 | 1398.13 | 1402.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 1429.50 | 1407.09 | 1405.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1429.50 | 1407.09 | 1405.66 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 12:15:00 | 1405.20 | 1414.67 | 1415.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 1401.20 | 1411.97 | 1414.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 1398.20 | 1396.98 | 1404.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 12:15:00 | 1398.20 | 1396.98 | 1404.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 1398.20 | 1396.98 | 1404.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 1398.20 | 1396.98 | 1404.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 1402.10 | 1397.80 | 1403.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 1402.10 | 1397.80 | 1403.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 1383.30 | 1395.09 | 1401.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:15:00 | 1382.00 | 1395.09 | 1401.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:00:00 | 1381.30 | 1378.50 | 1387.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:00:00 | 1381.50 | 1379.10 | 1387.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 1416.80 | 1394.32 | 1391.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 1416.80 | 1394.32 | 1391.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 09:15:00 | 1416.80 | 1394.32 | 1391.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 1416.80 | 1394.32 | 1391.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 09:15:00 | 1425.40 | 1408.75 | 1401.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 09:15:00 | 1416.10 | 1416.57 | 1409.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 10:00:00 | 1416.10 | 1416.57 | 1409.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1413.00 | 1416.95 | 1412.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:00:00 | 1413.00 | 1416.95 | 1412.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1410.00 | 1415.56 | 1412.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:45:00 | 1408.10 | 1415.56 | 1412.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 1412.00 | 1414.85 | 1412.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 1414.60 | 1414.85 | 1412.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1418.90 | 1415.66 | 1412.64 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 15:15:00 | 1409.80 | 1411.24 | 1411.42 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 1418.00 | 1412.59 | 1412.02 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 10:15:00 | 1402.90 | 1410.65 | 1411.19 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 1421.40 | 1412.54 | 1411.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 11:15:00 | 1427.00 | 1415.44 | 1412.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 13:15:00 | 1424.50 | 1442.84 | 1433.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 13:15:00 | 1424.50 | 1442.84 | 1433.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 1424.50 | 1442.84 | 1433.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 14:00:00 | 1424.50 | 1442.84 | 1433.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 1432.90 | 1440.86 | 1433.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 15:15:00 | 1434.00 | 1440.86 | 1433.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:45:00 | 1434.40 | 1438.27 | 1433.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1421.70 | 1434.95 | 1432.08 | SL hit (close<static) qty=1.00 sl=1422.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 1421.70 | 1434.95 | 1432.08 | SL hit (close<static) qty=1.00 sl=1422.50 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 1419.70 | 1429.27 | 1429.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 1403.00 | 1424.02 | 1427.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 1416.00 | 1412.15 | 1417.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 1416.00 | 1412.15 | 1417.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1408.40 | 1411.54 | 1416.32 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 15:15:00 | 1423.10 | 1417.99 | 1417.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 1435.10 | 1421.41 | 1419.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 10:15:00 | 1415.30 | 1420.19 | 1418.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 10:15:00 | 1415.30 | 1420.19 | 1418.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1415.30 | 1420.19 | 1418.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 1415.30 | 1420.19 | 1418.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 1418.70 | 1419.89 | 1418.94 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 13:15:00 | 1412.60 | 1417.57 | 1417.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 1408.20 | 1415.70 | 1417.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1383.70 | 1377.09 | 1387.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 10:15:00 | 1383.70 | 1377.09 | 1387.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 1383.70 | 1377.09 | 1387.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 1383.70 | 1377.09 | 1387.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 1383.60 | 1378.39 | 1386.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:45:00 | 1385.20 | 1378.39 | 1386.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 1388.00 | 1380.32 | 1386.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:00:00 | 1388.00 | 1380.32 | 1386.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1385.50 | 1381.35 | 1386.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 13:45:00 | 1386.00 | 1381.35 | 1386.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1389.80 | 1383.04 | 1387.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:30:00 | 1391.70 | 1383.04 | 1387.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1383.00 | 1383.03 | 1386.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 1395.90 | 1386.53 | 1387.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 1407.10 | 1390.64 | 1389.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 1407.60 | 1394.03 | 1391.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 13:15:00 | 1465.00 | 1465.47 | 1453.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 13:30:00 | 1458.80 | 1465.47 | 1453.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1468.30 | 1475.66 | 1467.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 1453.30 | 1475.66 | 1467.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1465.80 | 1473.69 | 1467.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 10:30:00 | 1465.80 | 1473.69 | 1467.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 1460.30 | 1471.01 | 1466.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:45:00 | 1459.90 | 1471.01 | 1466.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1466.50 | 1466.07 | 1465.25 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 1452.40 | 1463.34 | 1464.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 1448.00 | 1460.27 | 1462.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-01 14:15:00 | 1457.20 | 1457.15 | 1460.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-01 15:00:00 | 1457.20 | 1457.15 | 1460.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1450.00 | 1455.72 | 1459.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1459.20 | 1455.72 | 1459.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1448.70 | 1454.32 | 1458.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 10:30:00 | 1448.00 | 1453.45 | 1457.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 12:15:00 | 1463.50 | 1455.12 | 1457.55 | SL hit (close>static) qty=1.00 sl=1463.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 15:15:00 | 1467.20 | 1460.54 | 1459.66 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 1451.30 | 1459.58 | 1459.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 1444.50 | 1456.56 | 1458.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 15:15:00 | 1442.30 | 1441.24 | 1447.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 1443.50 | 1441.69 | 1447.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1443.50 | 1441.69 | 1447.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 1443.50 | 1441.69 | 1447.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 1439.70 | 1441.66 | 1446.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 13:00:00 | 1435.90 | 1440.51 | 1445.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 14:45:00 | 1436.10 | 1440.30 | 1444.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:15:00 | 1438.10 | 1440.77 | 1443.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 10:45:00 | 1436.00 | 1429.96 | 1430.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 13:15:00 | 1429.20 | 1430.49 | 1431.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:45:00 | 1430.40 | 1430.49 | 1431.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 1416.20 | 1426.88 | 1429.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:15:00 | 1411.60 | 1426.88 | 1429.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:30:00 | 1410.30 | 1418.54 | 1423.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 15:15:00 | 1399.00 | 1418.54 | 1423.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 10:15:00 | 1412.70 | 1414.67 | 1420.71 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 1406.10 | 1398.24 | 1403.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:00:00 | 1406.10 | 1398.24 | 1403.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1399.30 | 1398.45 | 1403.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 1397.80 | 1398.45 | 1403.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1364.11 | 1383.11 | 1390.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1364.29 | 1383.11 | 1390.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1366.19 | 1383.11 | 1390.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-18 10:15:00 | 1364.20 | 1383.11 | 1390.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 12:15:00 | 1371.40 | 1371.06 | 1378.27 | SL hit (close>ema200) qty=0.50 sl=1371.06 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 12:15:00 | 1371.40 | 1371.06 | 1378.27 | SL hit (close>ema200) qty=0.50 sl=1371.06 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 12:15:00 | 1371.40 | 1371.06 | 1378.27 | SL hit (close>ema200) qty=0.50 sl=1371.06 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 12:15:00 | 1371.40 | 1371.06 | 1378.27 | SL hit (close>ema200) qty=0.50 sl=1371.06 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 13:15:00 | 1341.02 | 1356.72 | 1366.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 13:15:00 | 1342.07 | 1356.72 | 1366.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-22 14:15:00 | 1339.78 | 1354.32 | 1364.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1356.90 | 1351.66 | 1359.61 | SL hit (close>ema200) qty=0.50 sl=1351.66 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1356.90 | 1351.66 | 1359.61 | SL hit (close>ema200) qty=0.50 sl=1351.66 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-23 11:15:00 | 1356.90 | 1351.66 | 1359.61 | SL hit (close>ema200) qty=0.50 sl=1351.66 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 1329.05 | 1344.35 | 1351.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 11:15:00 | 1327.91 | 1344.35 | 1351.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 1340.20 | 1335.40 | 1343.54 | SL hit (close>ema200) qty=0.50 sl=1335.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 1340.20 | 1335.40 | 1343.54 | SL hit (close>ema200) qty=0.50 sl=1335.40 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 11:15:00 | 1356.00 | 1339.09 | 1336.83 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1316.00 | 1335.09 | 1336.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 14:15:00 | 1311.50 | 1322.80 | 1329.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 1308.00 | 1295.29 | 1304.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 13:15:00 | 1308.00 | 1295.29 | 1304.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 1308.00 | 1295.29 | 1304.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 1308.00 | 1295.29 | 1304.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 1311.40 | 1298.51 | 1305.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:00:00 | 1311.40 | 1298.51 | 1305.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 1315.80 | 1304.43 | 1306.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 11:00:00 | 1315.80 | 1304.43 | 1306.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 1292.70 | 1302.08 | 1305.09 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 1319.00 | 1305.88 | 1305.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 09:15:00 | 1320.60 | 1308.83 | 1306.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 10:15:00 | 1340.90 | 1344.12 | 1333.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-08 10:45:00 | 1340.30 | 1344.12 | 1333.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 1339.50 | 1343.20 | 1334.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:30:00 | 1340.20 | 1343.20 | 1334.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 1333.50 | 1341.66 | 1335.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 1333.50 | 1341.66 | 1335.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 1330.60 | 1339.45 | 1335.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 1339.00 | 1339.45 | 1335.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 1346.00 | 1340.76 | 1336.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:45:00 | 1356.80 | 1345.56 | 1341.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 1355.00 | 1345.56 | 1341.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 13:45:00 | 1355.80 | 1349.03 | 1344.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 14:30:00 | 1354.10 | 1351.06 | 1345.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 1349.40 | 1356.13 | 1352.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 1340.00 | 1350.01 | 1350.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 1340.00 | 1350.01 | 1350.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 1340.00 | 1350.01 | 1350.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 1340.00 | 1350.01 | 1350.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 1340.00 | 1350.01 | 1350.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 13:15:00 | 1335.80 | 1347.17 | 1349.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1301.70 | 1293.71 | 1301.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 10:15:00 | 1301.70 | 1293.71 | 1301.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1301.70 | 1293.71 | 1301.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 1301.70 | 1293.71 | 1301.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1295.50 | 1294.07 | 1300.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 1287.00 | 1295.93 | 1299.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1282.00 | 1258.72 | 1256.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1282.00 | 1258.72 | 1256.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 1287.90 | 1264.56 | 1258.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 11:15:00 | 1295.20 | 1295.99 | 1282.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:00:00 | 1295.20 | 1295.99 | 1282.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 1292.90 | 1305.46 | 1298.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 1292.90 | 1305.46 | 1298.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 1296.80 | 1303.73 | 1298.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 1288.80 | 1303.73 | 1298.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 09:15:00 | 1276.00 | 1298.18 | 1296.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 10:00:00 | 1276.00 | 1298.18 | 1296.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 1271.30 | 1292.81 | 1294.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 11:15:00 | 1264.80 | 1287.20 | 1291.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 1254.50 | 1250.95 | 1260.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 1254.50 | 1250.95 | 1260.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1254.50 | 1250.95 | 1260.19 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 10:15:00 | 1266.60 | 1261.79 | 1261.51 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 1261.60 | 1266.71 | 1267.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 13:15:00 | 1260.20 | 1265.41 | 1266.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 1265.20 | 1263.06 | 1264.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 14:15:00 | 1265.20 | 1263.06 | 1264.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1265.20 | 1263.06 | 1264.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 1265.20 | 1263.06 | 1264.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1267.00 | 1263.85 | 1264.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1266.30 | 1263.85 | 1264.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1265.00 | 1264.08 | 1264.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:15:00 | 1264.20 | 1264.08 | 1264.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 10:45:00 | 1264.60 | 1264.18 | 1264.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 1279.00 | 1267.09 | 1265.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-18 13:15:00 | 1279.00 | 1267.09 | 1265.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 13:15:00 | 1279.00 | 1267.09 | 1265.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 14:15:00 | 1283.70 | 1270.42 | 1267.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 1273.50 | 1274.81 | 1270.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 11:00:00 | 1273.50 | 1274.81 | 1270.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 1268.00 | 1273.45 | 1270.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 1268.00 | 1273.45 | 1270.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 1268.80 | 1272.52 | 1270.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 1268.80 | 1272.52 | 1270.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 1268.50 | 1271.71 | 1270.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 14:00:00 | 1268.50 | 1271.71 | 1270.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 1264.00 | 1270.17 | 1269.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 1264.00 | 1270.17 | 1269.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 1279.10 | 1273.73 | 1271.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:00:00 | 1284.20 | 1274.42 | 1272.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 11:45:00 | 1283.80 | 1277.73 | 1274.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 13:15:00 | 1280.90 | 1278.18 | 1274.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 1282.30 | 1279.19 | 1276.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 1275.50 | 1278.45 | 1276.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:00:00 | 1275.50 | 1278.45 | 1276.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 1274.90 | 1277.74 | 1276.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 1274.90 | 1277.74 | 1276.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1271.00 | 1276.39 | 1275.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 1271.00 | 1276.39 | 1275.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1273.70 | 1275.85 | 1275.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 1271.50 | 1275.85 | 1275.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1270.00 | 1274.68 | 1274.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1270.00 | 1274.68 | 1274.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1270.00 | 1274.68 | 1274.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-24 13:15:00 | 1270.00 | 1274.68 | 1274.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 13:15:00 | 1270.00 | 1274.68 | 1274.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 1263.60 | 1272.47 | 1273.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 10:15:00 | 1231.90 | 1231.03 | 1238.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 10:45:00 | 1231.70 | 1231.03 | 1238.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 1224.00 | 1218.45 | 1224.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 13:15:00 | 1212.00 | 1217.77 | 1222.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 09:30:00 | 1211.90 | 1217.76 | 1220.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-06 10:15:00 | 1213.50 | 1217.76 | 1220.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 1243.00 | 1217.13 | 1215.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 1243.00 | 1217.13 | 1215.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-07 12:15:00 | 1243.00 | 1217.13 | 1215.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 12:15:00 | 1243.00 | 1217.13 | 1215.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 10:15:00 | 1243.80 | 1237.86 | 1232.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 1238.30 | 1239.08 | 1234.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 13:45:00 | 1237.40 | 1239.08 | 1234.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 1233.10 | 1237.40 | 1234.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:45:00 | 1239.60 | 1238.98 | 1235.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 1294.00 | 1315.11 | 1317.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 1294.00 | 1315.11 | 1317.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 12:15:00 | 1288.60 | 1307.66 | 1313.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 1312.70 | 1296.26 | 1304.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 1312.70 | 1296.26 | 1304.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1312.70 | 1296.26 | 1304.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 1312.70 | 1296.26 | 1304.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1313.30 | 1299.67 | 1304.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:30:00 | 1312.90 | 1299.67 | 1304.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 1316.40 | 1307.82 | 1307.64 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 1305.80 | 1307.29 | 1307.48 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 13:15:00 | 1310.20 | 1308.05 | 1307.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 14:15:00 | 1315.90 | 1309.62 | 1308.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 1305.00 | 1309.75 | 1308.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 1305.00 | 1309.75 | 1308.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1305.00 | 1309.75 | 1308.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 1305.00 | 1309.75 | 1308.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1310.00 | 1309.80 | 1308.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 1309.70 | 1309.80 | 1308.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1315.10 | 1310.86 | 1309.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:45:00 | 1327.20 | 1318.52 | 1313.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 1303.10 | 1321.06 | 1322.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 13:15:00 | 1303.10 | 1321.06 | 1322.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 15:15:00 | 1302.00 | 1314.49 | 1318.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 11:15:00 | 1276.80 | 1275.71 | 1289.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 11:30:00 | 1277.60 | 1275.71 | 1289.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1279.60 | 1277.33 | 1287.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:45:00 | 1282.80 | 1277.33 | 1287.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1290.20 | 1280.72 | 1286.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 1286.90 | 1280.72 | 1286.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1291.40 | 1282.86 | 1287.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 1291.40 | 1282.86 | 1287.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 1305.90 | 1290.05 | 1289.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 1311.90 | 1294.42 | 1291.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 1307.10 | 1322.44 | 1312.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 1307.10 | 1322.44 | 1312.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1307.10 | 1322.44 | 1312.33 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 1289.90 | 1304.51 | 1306.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 1287.60 | 1296.35 | 1300.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 1298.00 | 1293.20 | 1296.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 1298.00 | 1293.20 | 1296.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 1298.00 | 1293.20 | 1296.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 1298.00 | 1293.20 | 1296.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 1306.60 | 1295.88 | 1297.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 1314.40 | 1295.88 | 1297.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 1320.90 | 1300.89 | 1299.69 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 13:15:00 | 1301.60 | 1306.71 | 1306.97 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 1313.80 | 1307.78 | 1307.24 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 1295.30 | 1306.30 | 1307.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 10:15:00 | 1289.70 | 1296.86 | 1301.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 12:15:00 | 1297.40 | 1296.25 | 1300.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 13:00:00 | 1297.40 | 1296.25 | 1300.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1302.60 | 1297.52 | 1300.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 1302.60 | 1297.52 | 1300.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1303.60 | 1298.74 | 1300.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:30:00 | 1301.10 | 1298.74 | 1300.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1300.00 | 1298.99 | 1300.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1294.60 | 1298.99 | 1300.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 10:30:00 | 1298.30 | 1299.12 | 1300.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 1297.90 | 1299.47 | 1300.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1313.00 | 1299.97 | 1300.05 | SL hit (close>static) qty=1.00 sl=1307.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1313.00 | 1299.97 | 1300.05 | SL hit (close>static) qty=1.00 sl=1307.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1313.00 | 1299.97 | 1300.05 | SL hit (close>static) qty=1.00 sl=1307.10 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 1314.00 | 1302.77 | 1301.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 13:15:00 | 1316.10 | 1308.11 | 1304.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 14:15:00 | 1313.00 | 1313.84 | 1310.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 14:45:00 | 1314.00 | 1313.84 | 1310.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 1320.00 | 1314.66 | 1311.05 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 1306.50 | 1311.31 | 1311.79 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 1318.00 | 1312.49 | 1312.08 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 1306.80 | 1311.85 | 1312.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 1302.20 | 1309.92 | 1311.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 1310.90 | 1309.73 | 1310.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 12:15:00 | 1310.90 | 1309.73 | 1310.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1310.90 | 1309.73 | 1310.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:45:00 | 1312.00 | 1309.73 | 1310.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1309.70 | 1309.72 | 1310.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:30:00 | 1308.90 | 1309.72 | 1310.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1287.00 | 1302.95 | 1307.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 12:00:00 | 1283.00 | 1294.30 | 1299.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:30:00 | 1279.60 | 1291.71 | 1297.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 1282.80 | 1288.79 | 1294.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1283.00 | 1267.35 | 1267.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1283.00 | 1267.35 | 1267.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1283.00 | 1267.35 | 1267.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 09:15:00 | 1283.00 | 1267.35 | 1267.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1291.60 | 1272.20 | 1269.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 15:15:00 | 1280.20 | 1280.56 | 1275.34 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 09:15:00 | 1292.70 | 1280.56 | 1275.34 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 10:15:00 | 1285.80 | 1280.91 | 1275.97 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 11:45:00 | 1287.00 | 1281.78 | 1277.23 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-12 12:15:00 | 1287.20 | 1281.78 | 1277.23 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1283.70 | 1286.40 | 1281.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 1283.70 | 1286.40 | 1281.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1291.90 | 1289.69 | 1285.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 1280.80 | 1287.59 | 1285.50 | SL hit (close<ema400) qty=1.00 sl=1285.50 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 1280.80 | 1287.59 | 1285.50 | SL hit (close<ema400) qty=1.00 sl=1285.50 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 1280.80 | 1287.59 | 1285.50 | SL hit (close<ema400) qty=1.00 sl=1285.50 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-12-16 11:15:00 | 1280.80 | 1287.59 | 1285.50 | SL hit (close<ema400) qty=1.00 sl=1285.50 alert=retest1 |

### Cycle 48 — SELL (started 2025-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 14:15:00 | 1275.00 | 1284.06 | 1285.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 1258.70 | 1277.38 | 1281.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 1258.80 | 1258.29 | 1266.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 1258.80 | 1258.29 | 1266.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1276.00 | 1262.52 | 1267.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 1276.00 | 1262.52 | 1267.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 1279.90 | 1266.00 | 1268.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 1279.40 | 1266.00 | 1268.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 1288.20 | 1272.52 | 1271.06 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 1277.80 | 1283.95 | 1284.02 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 10:15:00 | 1289.30 | 1285.06 | 1284.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 11:15:00 | 1294.80 | 1287.01 | 1285.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 09:15:00 | 1284.40 | 1293.15 | 1289.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 1284.40 | 1293.15 | 1289.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1284.40 | 1293.15 | 1289.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 1284.40 | 1293.15 | 1289.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 1280.00 | 1290.52 | 1289.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:45:00 | 1280.20 | 1290.52 | 1289.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-12-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 12:15:00 | 1277.20 | 1286.08 | 1287.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 1275.40 | 1283.14 | 1285.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1283.00 | 1274.79 | 1279.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 1283.00 | 1274.79 | 1279.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1283.00 | 1274.79 | 1279.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:15:00 | 1289.10 | 1274.79 | 1279.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1290.00 | 1277.83 | 1280.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:30:00 | 1291.80 | 1277.83 | 1280.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 1308.20 | 1286.81 | 1283.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 1315.60 | 1302.09 | 1293.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 1395.10 | 1395.75 | 1374.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 09:30:00 | 1394.70 | 1395.75 | 1374.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 1385.70 | 1396.84 | 1389.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:30:00 | 1383.50 | 1396.84 | 1389.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 1393.00 | 1396.07 | 1389.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 1397.10 | 1396.50 | 1390.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 10:15:00 | 1398.40 | 1395.04 | 1390.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 1380.70 | 1392.17 | 1389.98 | SL hit (close<static) qty=1.00 sl=1381.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 10:15:00 | 1380.70 | 1392.17 | 1389.98 | SL hit (close<static) qty=1.00 sl=1381.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1368.00 | 1384.43 | 1386.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1365.70 | 1380.68 | 1384.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 1343.60 | 1337.37 | 1351.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 1343.60 | 1337.37 | 1351.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1351.50 | 1342.75 | 1350.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1351.60 | 1342.75 | 1350.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1347.00 | 1343.60 | 1350.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1349.00 | 1343.60 | 1350.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1344.60 | 1343.80 | 1349.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 1335.20 | 1342.74 | 1348.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:15:00 | 1333.80 | 1341.04 | 1346.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 1336.60 | 1342.36 | 1346.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 1336.20 | 1338.72 | 1343.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 1344.90 | 1339.96 | 1343.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:00:00 | 1344.90 | 1339.96 | 1343.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 1349.70 | 1341.90 | 1344.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:00:00 | 1349.70 | 1341.90 | 1344.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1348.30 | 1343.18 | 1344.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:00:00 | 1348.30 | 1343.18 | 1344.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 1344.70 | 1343.49 | 1344.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:45:00 | 1347.60 | 1343.49 | 1344.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 1348.00 | 1344.39 | 1345.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 1353.10 | 1344.39 | 1345.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1367.30 | 1348.97 | 1347.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1367.30 | 1348.97 | 1347.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1367.30 | 1348.97 | 1347.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 1367.30 | 1348.97 | 1347.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 1367.30 | 1348.97 | 1347.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 1369.00 | 1352.98 | 1349.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 1356.00 | 1357.31 | 1352.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 1356.00 | 1357.31 | 1352.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 14:15:00 | 1359.20 | 1357.69 | 1353.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 14:30:00 | 1356.30 | 1357.69 | 1353.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 1357.70 | 1357.69 | 1353.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 10:00:00 | 1364.80 | 1359.11 | 1354.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 13:15:00 | 1340.00 | 1352.69 | 1352.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 13:15:00 | 1340.00 | 1352.69 | 1352.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 14:15:00 | 1333.90 | 1348.94 | 1351.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1310.80 | 1309.82 | 1321.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-22 11:30:00 | 1304.20 | 1308.56 | 1318.94 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 12:15:00 | 1307.60 | 1308.37 | 1317.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:30:00 | 1316.90 | 1308.37 | 1317.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 1316.50 | 1310.00 | 1317.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 14:00:00 | 1316.50 | 1310.00 | 1317.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 1322.00 | 1312.40 | 1318.16 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 1322.00 | 1312.40 | 1318.16 | SL hit (close>ema400) qty=1.00 sl=1318.16 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 1322.00 | 1312.40 | 1318.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 1320.00 | 1313.92 | 1318.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 1311.70 | 1313.92 | 1318.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 1327.40 | 1305.22 | 1303.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 1327.40 | 1305.22 | 1303.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 1343.70 | 1329.93 | 1319.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 1357.60 | 1361.67 | 1347.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:00:00 | 1357.60 | 1361.67 | 1347.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1341.00 | 1370.53 | 1359.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 1341.00 | 1370.53 | 1359.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 1335.70 | 1363.56 | 1357.38 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 14:15:00 | 1297.80 | 1343.82 | 1349.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 15:15:00 | 1295.40 | 1334.14 | 1344.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 1330.00 | 1325.75 | 1335.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 13:15:00 | 1330.00 | 1325.75 | 1335.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1330.00 | 1325.75 | 1335.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:45:00 | 1335.20 | 1325.75 | 1335.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 1328.30 | 1326.26 | 1334.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:45:00 | 1326.10 | 1326.26 | 1334.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1343.80 | 1330.37 | 1335.17 | EMA400 retest candle locked (from downside) |

### Cycle 59 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 1352.80 | 1340.83 | 1339.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 1358.50 | 1344.37 | 1340.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 1403.10 | 1406.99 | 1391.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 1403.10 | 1406.99 | 1391.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 1400.80 | 1455.01 | 1440.93 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 1396.40 | 1426.54 | 1430.02 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 10:15:00 | 1447.40 | 1431.14 | 1430.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 10:15:00 | 1459.70 | 1448.12 | 1440.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 13:15:00 | 1459.50 | 1461.23 | 1454.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 1459.50 | 1461.23 | 1454.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 1505.00 | 1512.11 | 1500.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 14:00:00 | 1505.00 | 1512.11 | 1500.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1493.60 | 1508.41 | 1499.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 1493.60 | 1508.41 | 1499.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1491.00 | 1504.93 | 1498.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1501.70 | 1504.93 | 1498.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 1533.00 | 1535.33 | 1526.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:45:00 | 1548.00 | 1537.73 | 1528.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 11:30:00 | 1551.00 | 1541.74 | 1531.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 15:15:00 | 1544.00 | 1540.30 | 1533.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-02 11:30:00 | 1550.50 | 1560.98 | 1560.79 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 1530.30 | 1554.85 | 1558.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 1530.30 | 1554.85 | 1558.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 1530.30 | 1554.85 | 1558.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 1530.30 | 1554.85 | 1558.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 1530.30 | 1554.85 | 1558.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 1489.00 | 1541.20 | 1550.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1500.80 | 1494.67 | 1516.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:00:00 | 1500.80 | 1494.67 | 1516.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 1502.60 | 1491.14 | 1506.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 1509.50 | 1491.14 | 1506.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 1502.70 | 1493.46 | 1505.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 1492.40 | 1493.46 | 1505.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-09 09:15:00 | 1343.16 | 1475.20 | 1489.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 10:15:00 | 1497.30 | 1459.23 | 1456.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 11:15:00 | 1513.20 | 1470.02 | 1462.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 1475.70 | 1487.92 | 1475.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 1475.70 | 1487.92 | 1475.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 1475.70 | 1487.92 | 1475.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 1475.70 | 1487.92 | 1475.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 1479.80 | 1486.29 | 1476.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-13 11:15:00 | 1492.70 | 1486.29 | 1476.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 1466.10 | 1482.69 | 1476.33 | SL hit (close<static) qty=1.00 sl=1469.10 alert=retest2 |

### Cycle 64 — SELL (started 2026-03-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-16 09:15:00 | 1457.20 | 1470.38 | 1471.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 1429.50 | 1462.20 | 1468.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1479.20 | 1454.09 | 1459.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 1479.20 | 1454.09 | 1459.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1479.20 | 1454.09 | 1459.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 1479.20 | 1454.09 | 1459.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1456.90 | 1454.65 | 1459.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 1445.80 | 1454.65 | 1459.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 10:00:00 | 1449.30 | 1446.25 | 1452.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 11:45:00 | 1453.00 | 1448.73 | 1452.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 14:30:00 | 1451.30 | 1453.58 | 1453.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 15:15:00 | 1456.90 | 1454.25 | 1454.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 15:15:00 | 1456.90 | 1454.25 | 1454.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 15:15:00 | 1456.90 | 1454.25 | 1454.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 15:15:00 | 1456.90 | 1454.25 | 1454.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 15:15:00 | 1456.90 | 1454.25 | 1454.24 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 1436.50 | 1450.70 | 1452.63 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 09:15:00 | 1481.20 | 1455.16 | 1453.42 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 1449.30 | 1452.88 | 1453.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 1400.70 | 1442.19 | 1448.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 1406.70 | 1378.36 | 1392.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 1406.70 | 1378.36 | 1392.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 1406.70 | 1378.36 | 1392.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 1406.70 | 1378.36 | 1392.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 1416.00 | 1385.89 | 1394.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 1416.00 | 1385.89 | 1394.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 1384.90 | 1387.76 | 1392.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 14:15:00 | 1372.00 | 1386.05 | 1391.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 14:15:00 | 1303.40 | 1330.91 | 1354.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 09:15:00 | 1333.80 | 1327.67 | 1349.06 | SL hit (close>ema200) qty=0.50 sl=1327.67 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 10:15:00 | 1353.20 | 1337.57 | 1337.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1370.90 | 1347.83 | 1342.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1440.90 | 1444.52 | 1425.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:00:00 | 1440.90 | 1444.52 | 1425.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1490.20 | 1468.94 | 1451.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 1494.50 | 1468.94 | 1451.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 1495.80 | 1475.67 | 1463.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 09:15:00 | 1643.95 | 1624.27 | 1599.26 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-04-22 09:15:00 | 1645.38 | 1624.27 | 1599.26 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 1718.00 | 1742.05 | 1742.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 1705.90 | 1734.82 | 1739.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 10:15:00 | 1719.50 | 1719.28 | 1729.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 10:15:00 | 1719.50 | 1719.28 | 1729.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 1719.50 | 1719.28 | 1729.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 1719.50 | 1719.28 | 1729.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 1760.60 | 1727.54 | 1731.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:00:00 | 1760.60 | 1727.54 | 1731.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 1740.10 | 1730.05 | 1732.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:00:00 | 1732.00 | 1730.44 | 1732.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 15:00:00 | 1733.40 | 1731.03 | 1732.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1754.20 | 1736.81 | 1735.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1754.20 | 1736.81 | 1735.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 1754.20 | 1736.81 | 1735.08 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 11:15:00 | 1721.20 | 1731.78 | 1732.98 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 1734.90 | 1729.33 | 1729.07 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 15:15:00 | 1720.00 | 1727.46 | 1728.25 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1736.50 | 1729.27 | 1729.00 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 11:15:00 | 1725.00 | 1729.16 | 1729.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 12:15:00 | 1718.00 | 1726.93 | 1728.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 09:15:00 | 1729.60 | 1722.98 | 1725.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 1729.60 | 1722.98 | 1725.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1729.60 | 1722.98 | 1725.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:15:00 | 1729.30 | 1722.98 | 1725.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1732.00 | 1724.78 | 1726.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:30:00 | 1730.90 | 1724.78 | 1726.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 12:15:00 | 1731.00 | 1727.07 | 1727.07 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 14:15:00 | 1725.40 | 1726.96 | 1727.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 15:15:00 | 1717.50 | 1725.07 | 1726.17 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-16 12:45:00 | 1434.50 | 2025-05-16 13:15:00 | 1441.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-05-23 14:45:00 | 1392.50 | 2025-05-26 09:15:00 | 1429.50 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2025-05-30 10:15:00 | 1382.00 | 2025-06-03 09:15:00 | 1416.80 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-06-02 10:00:00 | 1381.30 | 2025-06-03 09:15:00 | 1416.80 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-06-02 11:00:00 | 1381.50 | 2025-06-03 09:15:00 | 1416.80 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-06-11 15:15:00 | 1434.00 | 2025-06-12 10:15:00 | 1421.70 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-06-12 09:45:00 | 1434.40 | 2025-06-12 10:15:00 | 1421.70 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-07-02 10:30:00 | 1448.00 | 2025-07-02 12:15:00 | 1463.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-07-07 13:00:00 | 1435.90 | 2025-07-18 10:15:00 | 1364.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-07 14:45:00 | 1436.10 | 2025-07-18 10:15:00 | 1364.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-08 10:15:00 | 1438.10 | 2025-07-18 10:15:00 | 1366.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-10 10:45:00 | 1436.00 | 2025-07-18 10:15:00 | 1364.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-07 13:00:00 | 1435.90 | 2025-07-21 12:15:00 | 1371.40 | STOP_HIT | 0.50 | 4.49% |
| SELL | retest2 | 2025-07-07 14:45:00 | 1436.10 | 2025-07-21 12:15:00 | 1371.40 | STOP_HIT | 0.50 | 4.51% |
| SELL | retest2 | 2025-07-08 10:15:00 | 1438.10 | 2025-07-21 12:15:00 | 1371.40 | STOP_HIT | 0.50 | 4.64% |
| SELL | retest2 | 2025-07-10 10:45:00 | 1436.00 | 2025-07-21 12:15:00 | 1371.40 | STOP_HIT | 0.50 | 4.50% |
| SELL | retest2 | 2025-07-11 10:15:00 | 1411.60 | 2025-07-22 13:15:00 | 1341.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-11 14:30:00 | 1410.30 | 2025-07-22 13:15:00 | 1342.07 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-07-11 15:15:00 | 1399.00 | 2025-07-22 14:15:00 | 1339.78 | PARTIAL | 0.50 | 4.23% |
| SELL | retest2 | 2025-07-11 10:15:00 | 1411.60 | 2025-07-23 11:15:00 | 1356.90 | STOP_HIT | 0.50 | 3.88% |
| SELL | retest2 | 2025-07-11 14:30:00 | 1410.30 | 2025-07-23 11:15:00 | 1356.90 | STOP_HIT | 0.50 | 3.79% |
| SELL | retest2 | 2025-07-11 15:15:00 | 1399.00 | 2025-07-23 11:15:00 | 1356.90 | STOP_HIT | 0.50 | 3.01% |
| SELL | retest2 | 2025-07-14 10:15:00 | 1412.70 | 2025-07-25 11:15:00 | 1329.05 | PARTIAL | 0.50 | 5.92% |
| SELL | retest2 | 2025-07-16 11:15:00 | 1397.80 | 2025-07-25 11:15:00 | 1327.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-14 10:15:00 | 1412.70 | 2025-07-28 09:15:00 | 1340.20 | STOP_HIT | 0.50 | 5.13% |
| SELL | retest2 | 2025-07-16 11:15:00 | 1397.80 | 2025-07-28 09:15:00 | 1340.20 | STOP_HIT | 0.50 | 4.12% |
| BUY | retest2 | 2025-08-12 09:45:00 | 1356.80 | 2025-08-14 12:15:00 | 1340.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-08-12 10:15:00 | 1355.00 | 2025-08-14 12:15:00 | 1340.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-08-12 13:45:00 | 1355.80 | 2025-08-14 12:15:00 | 1340.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-12 14:30:00 | 1354.10 | 2025-08-14 12:15:00 | 1340.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-08-21 15:15:00 | 1287.00 | 2025-09-02 09:15:00 | 1282.00 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2025-09-18 10:15:00 | 1264.20 | 2025-09-18 13:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2025-09-18 10:45:00 | 1264.60 | 2025-09-18 13:15:00 | 1279.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-09-23 10:00:00 | 1284.20 | 2025-09-24 13:15:00 | 1270.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-09-23 11:45:00 | 1283.80 | 2025-09-24 13:15:00 | 1270.00 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-09-23 13:15:00 | 1280.90 | 2025-09-24 13:15:00 | 1270.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-09-24 09:15:00 | 1282.30 | 2025-09-24 13:15:00 | 1270.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-03 13:15:00 | 1212.00 | 2025-10-07 12:15:00 | 1243.00 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-10-06 09:30:00 | 1211.90 | 2025-10-07 12:15:00 | 1243.00 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-10-06 10:15:00 | 1213.50 | 2025-10-07 12:15:00 | 1243.00 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-10-13 10:45:00 | 1239.60 | 2025-10-28 10:15:00 | 1294.00 | STOP_HIT | 1.00 | 4.39% |
| BUY | retest2 | 2025-11-03 09:45:00 | 1327.20 | 2025-11-04 13:15:00 | 1303.10 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1294.60 | 2025-11-26 09:15:00 | 1313.00 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-11-25 10:30:00 | 1298.30 | 2025-11-26 09:15:00 | 1313.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-25 12:15:00 | 1297.90 | 2025-11-26 09:15:00 | 1313.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-05 12:00:00 | 1283.00 | 2025-12-11 09:15:00 | 1283.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-12-05 14:30:00 | 1279.60 | 2025-12-11 09:15:00 | 1283.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-12-08 09:45:00 | 1282.80 | 2025-12-11 09:15:00 | 1283.00 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest1 | 2025-12-12 09:15:00 | 1292.70 | 2025-12-16 11:15:00 | 1280.80 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest1 | 2025-12-12 10:15:00 | 1285.80 | 2025-12-16 11:15:00 | 1280.80 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest1 | 2025-12-12 11:45:00 | 1287.00 | 2025-12-16 11:15:00 | 1280.80 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-12-12 12:15:00 | 1287.20 | 2025-12-16 11:15:00 | 1280.80 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-01-07 14:30:00 | 1397.10 | 2026-01-08 10:15:00 | 1380.70 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-01-08 10:15:00 | 1398.40 | 2026-01-08 10:15:00 | 1380.70 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-13 12:00:00 | 1335.20 | 2026-01-16 09:15:00 | 1367.30 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-01-13 14:15:00 | 1333.80 | 2026-01-16 09:15:00 | 1367.30 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-01-14 09:15:00 | 1336.60 | 2026-01-16 09:15:00 | 1367.30 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2026-01-14 11:00:00 | 1336.20 | 2026-01-16 09:15:00 | 1367.30 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-01-19 10:00:00 | 1364.80 | 2026-01-19 13:15:00 | 1340.00 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest1 | 2026-01-22 11:30:00 | 1304.20 | 2026-01-22 14:15:00 | 1322.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-01-23 09:15:00 | 1311.70 | 2026-01-28 09:15:00 | 1327.40 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-02-24 09:45:00 | 1548.00 | 2026-03-02 12:15:00 | 1530.30 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-02-24 11:30:00 | 1551.00 | 2026-03-02 12:15:00 | 1530.30 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-02-24 15:15:00 | 1544.00 | 2026-03-02 12:15:00 | 1530.30 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-03-02 11:30:00 | 1550.50 | 2026-03-02 12:15:00 | 1530.30 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2026-03-06 09:15:00 | 1492.40 | 2026-03-09 09:15:00 | 1343.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-13 11:15:00 | 1492.70 | 2026-03-13 12:15:00 | 1466.10 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-03-17 11:15:00 | 1445.80 | 2026-03-18 15:15:00 | 1456.90 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2026-03-18 10:00:00 | 1449.30 | 2026-03-18 15:15:00 | 1456.90 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2026-03-18 11:45:00 | 1453.00 | 2026-03-18 15:15:00 | 1456.90 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2026-03-18 14:30:00 | 1451.30 | 2026-03-18 15:15:00 | 1456.90 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-03-27 14:15:00 | 1372.00 | 2026-03-30 14:15:00 | 1303.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-27 14:15:00 | 1372.00 | 2026-04-01 09:15:00 | 1333.80 | STOP_HIT | 0.50 | 2.78% |
| BUY | retest2 | 2026-04-13 10:15:00 | 1494.50 | 2026-04-22 09:15:00 | 1643.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 1495.80 | 2026-04-22 09:15:00 | 1645.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-30 14:00:00 | 1732.00 | 2026-05-04 09:15:00 | 1754.20 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-04-30 15:00:00 | 1733.40 | 2026-05-04 09:15:00 | 1754.20 | STOP_HIT | 1.00 | -1.20% |
