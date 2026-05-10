# Wockhardt Ltd. (WOCKPHARMA)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1611.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 77 |
| ALERT1 | 50 |
| ALERT2 | 48 |
| ALERT2_SKIP | 20 |
| ALERT3 | 134 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 51 |
| PARTIAL | 7 |
| TARGET_HIT | 5 |
| STOP_HIT | 52 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 62 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 42
- **Target hits / Stop hits / Partials:** 5 / 50 / 7
- **Avg / median % per leg:** 0.63% / -0.66%
- **Sum % (uncompounded):** 39.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 6 | 27.3% | 1 | 21 | 0 | -0.05% | -1.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 6 | 27.3% | 1 | 21 | 0 | -0.05% | -1.0% |
| SELL (all) | 40 | 14 | 35.0% | 4 | 29 | 7 | 1.00% | 40.2% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.70% | -6.8% |
| SELL @ 3rd Alert (retest2) | 36 | 14 | 38.9% | 4 | 25 | 7 | 1.31% | 47.0% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.70% | -6.8% |
| retest2 (combined) | 58 | 20 | 34.5% | 5 | 46 | 7 | 0.79% | 46.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1266.30 | 1253.06 | 1252.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 1312.90 | 1271.81 | 1262.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 1278.10 | 1284.97 | 1272.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:30:00 | 1278.70 | 1284.97 | 1272.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 1275.00 | 1282.18 | 1273.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 09:15:00 | 1280.50 | 1282.18 | 1273.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 10:15:00 | 1267.20 | 1278.64 | 1273.44 | SL hit (close<static) qty=1.00 sl=1270.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 14:15:00 | 1255.20 | 1269.09 | 1270.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 09:15:00 | 1247.50 | 1262.89 | 1267.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 11:15:00 | 1275.20 | 1263.05 | 1266.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 11:15:00 | 1275.20 | 1263.05 | 1266.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1275.20 | 1263.05 | 1266.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:45:00 | 1282.00 | 1263.05 | 1266.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 1273.00 | 1265.04 | 1266.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:15:00 | 1278.20 | 1265.04 | 1266.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 1290.30 | 1270.09 | 1269.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 11:15:00 | 1307.30 | 1286.21 | 1277.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 1343.70 | 1344.77 | 1320.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 1343.70 | 1344.77 | 1320.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 1339.80 | 1350.69 | 1336.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 1339.80 | 1350.69 | 1336.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1325.90 | 1345.73 | 1335.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 1317.90 | 1345.73 | 1335.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 15:15:00 | 1324.90 | 1341.56 | 1334.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:15:00 | 1343.30 | 1341.56 | 1334.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 12:15:00 | 1327.50 | 1337.39 | 1334.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 1320.50 | 1333.90 | 1333.89 | SL hit (close<static) qty=1.00 sl=1320.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 1320.50 | 1333.90 | 1333.89 | SL hit (close<static) qty=1.00 sl=1320.70 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 10:15:00 | 1320.90 | 1331.30 | 1332.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 12:15:00 | 1312.00 | 1325.79 | 1329.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 11:15:00 | 1339.30 | 1324.88 | 1326.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 11:15:00 | 1339.30 | 1324.88 | 1326.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 1339.30 | 1324.88 | 1326.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 1339.30 | 1324.88 | 1326.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 1333.60 | 1326.62 | 1327.56 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 13:15:00 | 1340.40 | 1329.38 | 1328.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 15:15:00 | 1343.00 | 1333.80 | 1330.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 10:15:00 | 1335.00 | 1335.19 | 1332.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 11:00:00 | 1335.00 | 1335.19 | 1332.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 11:15:00 | 1330.90 | 1334.34 | 1332.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:30:00 | 1330.00 | 1334.34 | 1332.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 1329.20 | 1333.31 | 1331.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 12:30:00 | 1325.70 | 1333.31 | 1331.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 1325.00 | 1331.65 | 1331.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:45:00 | 1325.00 | 1331.65 | 1331.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 1332.30 | 1332.52 | 1331.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 1335.50 | 1332.52 | 1331.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 1332.40 | 1332.50 | 1331.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 1321.50 | 1332.50 | 1331.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 1333.20 | 1332.64 | 1331.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 1327.90 | 1332.64 | 1331.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 1336.70 | 1333.45 | 1332.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 1332.90 | 1333.45 | 1332.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 1342.90 | 1350.68 | 1344.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:00:00 | 1342.90 | 1350.68 | 1344.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1336.20 | 1347.78 | 1343.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 1336.20 | 1347.78 | 1343.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 1335.00 | 1345.23 | 1342.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:30:00 | 1337.80 | 1344.88 | 1342.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 10:00:00 | 1343.50 | 1344.88 | 1342.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 1326.00 | 1341.11 | 1341.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-29 10:15:00 | 1326.00 | 1341.11 | 1341.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 10:15:00 | 1326.00 | 1341.11 | 1341.36 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 1344.80 | 1341.68 | 1341.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 1413.00 | 1356.47 | 1348.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 1463.80 | 1464.91 | 1436.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 09:45:00 | 1454.80 | 1464.91 | 1436.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 1513.80 | 1523.41 | 1509.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:45:00 | 1510.20 | 1523.41 | 1509.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 1503.10 | 1519.35 | 1509.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 13:45:00 | 1501.00 | 1519.35 | 1509.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 1503.70 | 1516.22 | 1508.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 14:45:00 | 1501.80 | 1516.22 | 1508.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 1506.00 | 1514.18 | 1508.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 1489.20 | 1514.18 | 1508.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 1494.20 | 1510.18 | 1507.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 09:30:00 | 1534.10 | 1511.75 | 1508.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-11 11:15:00 | 1687.51 | 1595.11 | 1554.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 15:15:00 | 1720.00 | 1755.40 | 1755.75 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 1853.00 | 1774.92 | 1764.59 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 1740.00 | 1756.41 | 1758.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 14:15:00 | 1716.40 | 1748.41 | 1754.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 13:15:00 | 1732.00 | 1702.42 | 1715.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 13:15:00 | 1732.00 | 1702.42 | 1715.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 1732.00 | 1702.42 | 1715.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 1732.00 | 1702.42 | 1715.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1731.60 | 1708.26 | 1716.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 1731.60 | 1708.26 | 1716.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 1725.10 | 1711.62 | 1717.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:30:00 | 1722.30 | 1716.66 | 1719.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 1733.90 | 1722.23 | 1721.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 1733.90 | 1722.23 | 1721.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 12:15:00 | 1754.70 | 1728.72 | 1724.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1731.10 | 1749.19 | 1741.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 14:15:00 | 1731.10 | 1749.19 | 1741.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 1731.10 | 1749.19 | 1741.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 1731.10 | 1749.19 | 1741.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 1750.00 | 1749.35 | 1742.40 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 13:15:00 | 1718.90 | 1736.54 | 1738.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 09:15:00 | 1685.00 | 1721.03 | 1730.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 10:15:00 | 1654.10 | 1650.18 | 1668.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 10:45:00 | 1656.50 | 1650.18 | 1668.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 1701.50 | 1660.44 | 1671.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:45:00 | 1701.40 | 1660.44 | 1671.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 1705.30 | 1669.41 | 1674.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:45:00 | 1722.70 | 1669.41 | 1674.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 1728.00 | 1681.13 | 1679.48 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 1664.00 | 1680.25 | 1681.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 1653.70 | 1673.11 | 1677.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 10:15:00 | 1681.00 | 1674.68 | 1677.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 10:15:00 | 1681.00 | 1674.68 | 1677.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1681.00 | 1674.68 | 1677.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 1681.00 | 1674.68 | 1677.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 1676.00 | 1674.95 | 1677.25 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 12:15:00 | 1695.60 | 1679.08 | 1678.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 13:15:00 | 1704.00 | 1684.06 | 1681.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 1726.90 | 1730.41 | 1714.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 11:15:00 | 1726.90 | 1730.41 | 1714.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1726.90 | 1730.41 | 1714.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 1717.30 | 1730.41 | 1714.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1720.00 | 1729.93 | 1719.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 1742.40 | 1729.93 | 1719.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 1738.10 | 1731.56 | 1721.44 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 15:15:00 | 1697.00 | 1716.30 | 1718.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 1681.50 | 1709.34 | 1714.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 1686.20 | 1677.23 | 1691.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 1686.20 | 1677.23 | 1691.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1686.20 | 1677.23 | 1691.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 1694.40 | 1677.23 | 1691.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 1692.80 | 1680.35 | 1692.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:00:00 | 1692.80 | 1680.35 | 1692.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 1717.30 | 1687.74 | 1694.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 1717.30 | 1687.74 | 1694.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 1721.50 | 1694.49 | 1696.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 13:00:00 | 1721.50 | 1694.49 | 1696.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 13:15:00 | 1720.70 | 1699.73 | 1698.96 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 14:15:00 | 1695.00 | 1700.55 | 1701.07 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 1788.00 | 1716.99 | 1708.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 11:15:00 | 1807.50 | 1769.81 | 1746.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 1805.60 | 1818.06 | 1797.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:45:00 | 1810.00 | 1818.06 | 1797.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1780.50 | 1812.46 | 1810.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 1780.50 | 1812.46 | 1810.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 1782.10 | 1806.39 | 1807.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 09:15:00 | 1768.20 | 1783.08 | 1793.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 10:15:00 | 1733.50 | 1724.01 | 1742.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 10:15:00 | 1733.50 | 1724.01 | 1742.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1733.50 | 1724.01 | 1742.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:45:00 | 1736.70 | 1724.01 | 1742.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1726.00 | 1720.78 | 1731.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 1732.80 | 1720.78 | 1731.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 1710.00 | 1718.62 | 1729.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 11:30:00 | 1721.70 | 1718.62 | 1729.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1738.00 | 1697.26 | 1705.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 1762.40 | 1697.26 | 1705.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1713.40 | 1700.49 | 1705.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:30:00 | 1710.10 | 1701.83 | 1705.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 12:30:00 | 1707.50 | 1697.19 | 1699.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 1624.59 | 1638.33 | 1658.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 15:15:00 | 1622.12 | 1638.33 | 1658.78 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-05 15:15:00 | 1539.09 | 1558.58 | 1585.89 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2025-08-05 15:15:00 | 1536.75 | 1558.58 | 1585.89 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 13:15:00 | 1509.60 | 1492.26 | 1491.95 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 10:15:00 | 1500.00 | 1509.45 | 1510.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 11:15:00 | 1495.00 | 1506.56 | 1508.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 1496.20 | 1490.42 | 1495.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 1496.20 | 1490.42 | 1495.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1496.20 | 1490.42 | 1495.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1496.60 | 1490.42 | 1495.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 1503.70 | 1493.07 | 1496.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:45:00 | 1508.70 | 1493.07 | 1496.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1491.00 | 1492.66 | 1495.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:15:00 | 1488.90 | 1492.66 | 1495.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 12:45:00 | 1490.40 | 1491.67 | 1495.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 13:15:00 | 1415.88 | 1427.24 | 1436.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-01 14:15:00 | 1414.45 | 1424.39 | 1434.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 1429.30 | 1423.99 | 1430.92 | SL hit (close>ema200) qty=0.50 sl=1423.99 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-02 11:15:00 | 1429.30 | 1423.99 | 1430.92 | SL hit (close>ema200) qty=0.50 sl=1423.99 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1466.10 | 1432.34 | 1431.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 11:15:00 | 1479.40 | 1441.75 | 1435.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1475.80 | 1478.16 | 1464.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 1475.80 | 1478.16 | 1464.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 1461.00 | 1474.15 | 1467.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 11:45:00 | 1461.50 | 1474.15 | 1467.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 1466.50 | 1472.62 | 1466.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 1464.80 | 1472.62 | 1466.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 1481.40 | 1474.38 | 1468.30 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 1456.60 | 1465.82 | 1466.75 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 09:15:00 | 1517.50 | 1475.19 | 1470.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 13:15:00 | 1544.50 | 1518.18 | 1500.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 10:15:00 | 1525.80 | 1532.16 | 1514.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 10:45:00 | 1537.60 | 1532.16 | 1514.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1535.40 | 1529.11 | 1520.02 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 13:15:00 | 1511.10 | 1519.06 | 1519.92 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 1567.90 | 1527.35 | 1523.31 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 1517.00 | 1538.02 | 1538.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 11:15:00 | 1514.70 | 1524.69 | 1530.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 1475.30 | 1474.02 | 1486.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 13:00:00 | 1475.30 | 1474.02 | 1486.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 1485.20 | 1476.25 | 1486.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 1495.30 | 1476.25 | 1486.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 1486.00 | 1478.20 | 1486.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:45:00 | 1494.00 | 1478.20 | 1486.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1487.30 | 1480.02 | 1486.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 1487.80 | 1480.02 | 1486.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 1498.10 | 1483.64 | 1487.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 1498.10 | 1483.64 | 1487.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1487.00 | 1484.31 | 1487.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 13:15:00 | 1484.80 | 1484.97 | 1487.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 1410.56 | 1462.90 | 1475.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-09-26 14:15:00 | 1336.32 | 1400.36 | 1437.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 29 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 1538.20 | 1452.35 | 1447.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 14:15:00 | 1585.00 | 1478.88 | 1459.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-30 15:15:00 | 1507.80 | 1514.01 | 1493.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 1474.80 | 1514.01 | 1493.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 1493.60 | 1509.92 | 1493.59 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 10:15:00 | 1466.70 | 1486.41 | 1488.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 13:15:00 | 1465.50 | 1479.60 | 1484.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1410.40 | 1404.90 | 1419.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 10:15:00 | 1419.10 | 1404.90 | 1419.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 1401.40 | 1404.20 | 1417.58 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 1434.00 | 1415.96 | 1415.54 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 1402.90 | 1415.44 | 1415.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 1388.90 | 1410.13 | 1413.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 1378.80 | 1377.51 | 1387.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 1378.80 | 1377.51 | 1387.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1378.80 | 1377.51 | 1387.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 1370.00 | 1377.21 | 1386.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 14:15:00 | 1371.00 | 1348.47 | 1349.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-21 14:15:00 | 1371.00 | 1352.98 | 1351.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-21 14:15:00 | 1371.00 | 1352.98 | 1351.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1371.00 | 1352.98 | 1351.77 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 1340.00 | 1353.60 | 1354.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 1335.00 | 1349.88 | 1352.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 10:15:00 | 1305.60 | 1305.54 | 1319.47 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-28 14:15:00 | 1298.40 | 1304.21 | 1315.33 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1308.80 | 1302.45 | 1310.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:45:00 | 1311.60 | 1302.45 | 1310.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 1317.00 | 1305.36 | 1311.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 1317.00 | 1305.36 | 1311.23 | SL hit (close>ema400) qty=1.00 sl=1311.23 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 1317.00 | 1305.36 | 1311.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 1307.50 | 1305.79 | 1310.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 15:00:00 | 1305.50 | 1306.72 | 1310.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:30:00 | 1299.00 | 1304.03 | 1308.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1351.00 | 1300.57 | 1300.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1351.00 | 1300.57 | 1300.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 09:15:00 | 1351.00 | 1300.57 | 1300.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 13:15:00 | 1394.60 | 1341.90 | 1321.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 1369.40 | 1371.36 | 1351.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 13:30:00 | 1372.00 | 1371.36 | 1351.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1350.30 | 1365.16 | 1353.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 1350.30 | 1365.16 | 1353.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 1347.10 | 1361.55 | 1352.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:30:00 | 1353.60 | 1361.20 | 1353.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 12:30:00 | 1355.60 | 1360.68 | 1353.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 15:00:00 | 1357.80 | 1359.60 | 1354.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 1326.50 | 1351.44 | 1351.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 1326.50 | 1351.44 | 1351.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 1326.50 | 1351.44 | 1351.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 1326.50 | 1351.44 | 1351.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 10:15:00 | 1319.30 | 1345.01 | 1348.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 1337.00 | 1336.86 | 1342.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:30:00 | 1340.00 | 1336.86 | 1342.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1327.10 | 1334.61 | 1340.78 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1355.10 | 1342.35 | 1340.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 1364.40 | 1350.72 | 1345.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 12:15:00 | 1352.90 | 1355.53 | 1350.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 12:15:00 | 1352.90 | 1355.53 | 1350.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1352.90 | 1355.53 | 1350.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:45:00 | 1351.80 | 1355.53 | 1350.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 1354.10 | 1355.24 | 1350.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 13:30:00 | 1356.00 | 1355.24 | 1350.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1350.00 | 1354.19 | 1350.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1350.00 | 1354.19 | 1350.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1344.10 | 1352.18 | 1350.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1346.60 | 1352.18 | 1350.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1347.00 | 1349.37 | 1349.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:15:00 | 1350.60 | 1349.37 | 1349.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:00:00 | 1350.90 | 1349.67 | 1349.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 1345.30 | 1348.80 | 1349.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 1345.30 | 1348.80 | 1349.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 12:15:00 | 1345.30 | 1348.80 | 1349.01 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 1376.20 | 1353.79 | 1351.09 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 12:15:00 | 1346.60 | 1355.12 | 1356.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 1332.60 | 1342.79 | 1348.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 1292.80 | 1291.88 | 1306.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 1292.80 | 1291.88 | 1306.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1309.80 | 1295.47 | 1306.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 1282.50 | 1295.47 | 1306.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1310.80 | 1283.11 | 1285.40 | SL hit (close>static) qty=1.00 sl=1309.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:45:00 | 1285.00 | 1284.99 | 1285.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 11:15:00 | 1375.60 | 1271.47 | 1270.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 11:15:00 | 1375.60 | 1271.47 | 1270.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 12:15:00 | 1460.50 | 1309.27 | 1287.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 1438.00 | 1469.20 | 1418.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-03 10:00:00 | 1438.00 | 1469.20 | 1418.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1419.60 | 1449.03 | 1421.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:00:00 | 1419.60 | 1449.03 | 1421.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1411.80 | 1441.58 | 1420.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:45:00 | 1412.80 | 1441.58 | 1420.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 1412.90 | 1431.21 | 1419.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 09:15:00 | 1403.50 | 1431.21 | 1419.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 10:15:00 | 1362.10 | 1409.03 | 1410.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 11:15:00 | 1344.90 | 1396.21 | 1404.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 09:15:00 | 1373.50 | 1361.92 | 1381.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1373.50 | 1361.92 | 1381.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1373.50 | 1361.92 | 1381.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 1383.80 | 1361.92 | 1381.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1365.20 | 1362.58 | 1379.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 15:15:00 | 1348.60 | 1361.93 | 1374.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:30:00 | 1351.70 | 1338.22 | 1340.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1351.70 | 1340.34 | 1339.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1351.70 | 1340.34 | 1339.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 1351.70 | 1340.34 | 1339.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 13:15:00 | 1357.30 | 1346.04 | 1342.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 14:15:00 | 1359.40 | 1361.97 | 1354.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 14:30:00 | 1359.50 | 1361.97 | 1354.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1360.10 | 1361.59 | 1355.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 1340.40 | 1361.59 | 1355.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 1347.10 | 1358.69 | 1354.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:45:00 | 1341.40 | 1358.69 | 1354.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1372.70 | 1361.50 | 1356.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 1375.60 | 1364.32 | 1358.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 13:30:00 | 1376.10 | 1367.27 | 1360.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 14:30:00 | 1375.90 | 1369.37 | 1362.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 1430.50 | 1439.58 | 1439.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 1430.50 | 1439.58 | 1439.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 1430.50 | 1439.58 | 1439.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 14:15:00 | 1430.50 | 1439.58 | 1439.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 15:15:00 | 1424.00 | 1436.46 | 1438.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 1398.30 | 1394.47 | 1410.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:00:00 | 1398.30 | 1394.47 | 1410.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 1392.30 | 1394.04 | 1408.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:15:00 | 1381.20 | 1394.04 | 1408.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 1440.20 | 1401.04 | 1409.22 | SL hit (close>static) qty=1.00 sl=1414.90 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1445.00 | 1416.42 | 1415.20 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 12:15:00 | 1408.70 | 1419.67 | 1419.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 14:15:00 | 1404.50 | 1414.75 | 1417.57 | Break + close below crossover candle low |

### Cycle 47 — BUY (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 10:15:00 | 1477.00 | 1425.34 | 1421.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 12:15:00 | 1488.70 | 1458.47 | 1443.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 1477.60 | 1493.92 | 1477.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 10:15:00 | 1477.60 | 1493.92 | 1477.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 1477.60 | 1493.92 | 1477.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 1477.60 | 1493.92 | 1477.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 1476.80 | 1490.49 | 1477.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:30:00 | 1477.90 | 1490.49 | 1477.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 1471.90 | 1486.78 | 1477.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:00:00 | 1471.90 | 1486.78 | 1477.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 1466.80 | 1482.78 | 1476.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:30:00 | 1464.80 | 1482.78 | 1476.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 1441.00 | 1467.45 | 1470.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 1429.70 | 1454.71 | 1463.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 1402.50 | 1401.28 | 1425.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 1393.70 | 1401.28 | 1425.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1401.60 | 1391.10 | 1401.93 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 1431.20 | 1403.77 | 1403.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 10:15:00 | 1434.20 | 1409.85 | 1406.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 15:15:00 | 1423.50 | 1425.34 | 1417.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 09:15:00 | 1405.00 | 1425.34 | 1417.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 1412.80 | 1422.83 | 1416.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 1397.20 | 1422.83 | 1416.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 1432.00 | 1424.66 | 1418.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 1414.80 | 1424.66 | 1418.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 1422.40 | 1424.21 | 1418.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 11:45:00 | 1415.00 | 1424.21 | 1418.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 1416.80 | 1423.56 | 1419.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 15:00:00 | 1416.80 | 1423.56 | 1419.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 1417.60 | 1422.37 | 1419.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 1408.20 | 1422.37 | 1419.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1404.40 | 1418.78 | 1418.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 1404.40 | 1418.78 | 1418.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 1405.10 | 1416.04 | 1416.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 1381.20 | 1409.07 | 1413.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1382.00 | 1362.32 | 1376.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 1382.00 | 1362.32 | 1376.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1382.00 | 1362.32 | 1376.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 1378.90 | 1362.32 | 1376.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1371.00 | 1364.06 | 1376.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 1367.30 | 1364.06 | 1376.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 10:30:00 | 1362.00 | 1357.08 | 1361.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:45:00 | 1366.10 | 1353.58 | 1354.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 14:15:00 | 1370.30 | 1355.90 | 1355.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 14:15:00 | 1370.30 | 1355.90 | 1355.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 14:15:00 | 1370.30 | 1355.90 | 1355.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 14:15:00 | 1370.30 | 1355.90 | 1355.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 1379.90 | 1362.48 | 1358.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1376.10 | 1376.36 | 1369.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 1376.10 | 1376.36 | 1369.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1376.10 | 1376.36 | 1369.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:30:00 | 1367.60 | 1376.36 | 1369.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1377.30 | 1376.55 | 1369.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:45:00 | 1370.00 | 1376.55 | 1369.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 1371.20 | 1375.48 | 1369.95 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 1343.00 | 1364.14 | 1366.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-02 09:15:00 | 1319.70 | 1355.25 | 1361.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 1321.50 | 1320.51 | 1337.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-03 09:15:00 | 1379.00 | 1320.51 | 1337.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1356.50 | 1327.71 | 1339.11 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 1356.90 | 1345.99 | 1344.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 1379.90 | 1353.95 | 1348.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1384.00 | 1388.11 | 1370.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 1384.00 | 1388.11 | 1370.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 1369.60 | 1384.41 | 1370.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 1365.00 | 1384.41 | 1370.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 1397.10 | 1386.95 | 1373.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 1458.60 | 1390.79 | 1384.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 09:15:00 | 1414.20 | 1415.25 | 1404.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 14:30:00 | 1421.20 | 1407.21 | 1403.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:00:00 | 1408.00 | 1410.34 | 1406.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 1404.30 | 1409.13 | 1405.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 11:00:00 | 1404.30 | 1409.13 | 1405.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1409.90 | 1409.29 | 1406.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:15:00 | 1406.40 | 1409.29 | 1406.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 12:15:00 | 1405.00 | 1408.43 | 1406.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 12:30:00 | 1401.80 | 1408.43 | 1406.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 13:15:00 | 1410.10 | 1408.76 | 1406.56 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1387.00 | 1402.32 | 1403.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1387.00 | 1402.32 | 1403.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1387.00 | 1402.32 | 1403.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 1387.00 | 1402.32 | 1403.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 1387.00 | 1402.32 | 1403.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 1378.10 | 1397.48 | 1401.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 10:15:00 | 1393.00 | 1386.69 | 1392.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 10:15:00 | 1393.00 | 1386.69 | 1392.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 1393.00 | 1386.69 | 1392.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:00:00 | 1393.00 | 1386.69 | 1392.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 1407.40 | 1390.83 | 1393.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 1407.40 | 1390.83 | 1393.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 1413.50 | 1395.37 | 1395.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 1416.10 | 1395.37 | 1395.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2026-02-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 13:15:00 | 1409.20 | 1398.13 | 1396.97 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 1391.40 | 1395.95 | 1396.12 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 12:15:00 | 1397.40 | 1396.16 | 1396.11 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 13:15:00 | 1395.00 | 1395.93 | 1396.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 14:15:00 | 1389.00 | 1394.54 | 1395.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 1399.10 | 1394.81 | 1395.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 1399.10 | 1394.81 | 1395.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 1399.10 | 1394.81 | 1395.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 1399.10 | 1394.81 | 1395.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 1402.90 | 1396.43 | 1396.00 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 1393.50 | 1395.73 | 1395.82 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 14:15:00 | 1398.30 | 1396.25 | 1396.04 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 15:15:00 | 1393.50 | 1395.70 | 1395.81 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 09:15:00 | 1408.80 | 1398.32 | 1396.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 12:15:00 | 1422.00 | 1406.98 | 1401.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 1410.90 | 1417.38 | 1409.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 11:00:00 | 1410.90 | 1417.38 | 1409.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1398.60 | 1413.62 | 1408.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1398.60 | 1413.62 | 1408.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1405.10 | 1411.92 | 1408.55 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 1395.30 | 1406.40 | 1406.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 15:15:00 | 1385.00 | 1402.12 | 1404.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 15:15:00 | 1370.00 | 1368.91 | 1379.56 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1353.90 | 1368.91 | 1379.56 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 12:00:00 | 1360.10 | 1363.67 | 1374.22 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-24 15:15:00 | 1360.00 | 1361.69 | 1370.48 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1367.50 | 1362.58 | 1369.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:30:00 | 1365.30 | 1362.58 | 1369.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1367.00 | 1363.46 | 1369.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:45:00 | 1363.30 | 1363.46 | 1369.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 11:15:00 | 1363.10 | 1363.39 | 1368.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 11:30:00 | 1368.50 | 1363.39 | 1368.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 1360.00 | 1362.71 | 1367.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:30:00 | 1356.00 | 1361.99 | 1367.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 1382.30 | 1366.05 | 1368.40 | SL hit (close>ema400) qty=1.00 sl=1368.40 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 1382.30 | 1366.05 | 1368.40 | SL hit (close>ema400) qty=1.00 sl=1368.40 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 1382.30 | 1366.05 | 1368.40 | SL hit (close>ema400) qty=1.00 sl=1368.40 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-02-25 14:15:00 | 1382.30 | 1366.05 | 1368.40 | SL hit (close>static) qty=1.00 sl=1368.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 1391.60 | 1373.60 | 1371.58 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 15:15:00 | 1360.80 | 1370.10 | 1370.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1344.10 | 1360.83 | 1365.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1313.30 | 1298.52 | 1315.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 1313.30 | 1298.52 | 1315.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1313.30 | 1298.52 | 1315.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:00:00 | 1313.30 | 1298.52 | 1315.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1310.20 | 1300.86 | 1315.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 1298.40 | 1303.97 | 1313.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 1322.10 | 1308.34 | 1312.99 | SL hit (close>static) qty=1.00 sl=1315.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:45:00 | 1303.30 | 1306.91 | 1311.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 1299.90 | 1306.39 | 1310.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-09 09:15:00 | 1264.70 | 1305.89 | 1309.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1323.20 | 1281.71 | 1289.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 1323.20 | 1281.71 | 1289.37 | SL hit (close>static) qty=1.00 sl=1315.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 1323.20 | 1281.71 | 1289.37 | SL hit (close>static) qty=1.00 sl=1315.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 1323.20 | 1281.71 | 1289.37 | SL hit (close>static) qty=1.00 sl=1315.50 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-03-10 10:00:00 | 1323.20 | 1281.71 | 1289.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 1318.90 | 1289.14 | 1292.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:15:00 | 1313.40 | 1289.14 | 1292.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 11:45:00 | 1313.00 | 1293.88 | 1293.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 12:15:00 | 1322.10 | 1299.52 | 1296.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 12:15:00 | 1322.10 | 1299.52 | 1296.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1322.10 | 1299.52 | 1296.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 1327.80 | 1308.45 | 1301.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 13:15:00 | 1319.90 | 1321.00 | 1312.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 14:15:00 | 1318.10 | 1321.00 | 1312.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1308.40 | 1318.48 | 1311.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1308.40 | 1318.48 | 1311.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1304.00 | 1315.58 | 1311.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1289.00 | 1315.58 | 1311.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 12:15:00 | 1307.50 | 1308.98 | 1308.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:00:00 | 1307.50 | 1308.98 | 1308.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 13:15:00 | 1305.60 | 1308.30 | 1308.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 1295.90 | 1305.82 | 1307.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 1204.70 | 1197.55 | 1213.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 1204.70 | 1197.55 | 1213.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1204.70 | 1197.55 | 1213.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:30:00 | 1214.20 | 1197.55 | 1213.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 1223.00 | 1202.64 | 1214.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 1223.00 | 1202.64 | 1214.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 1214.80 | 1205.07 | 1214.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:45:00 | 1210.50 | 1206.52 | 1213.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 1195.00 | 1210.90 | 1214.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1149.97 | 1174.22 | 1188.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 1135.25 | 1174.22 | 1188.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-23 12:15:00 | 1089.45 | 1139.29 | 1167.64 | Target hit (10%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 1134.00 | 1125.18 | 1150.77 | SL hit (close>ema200) qty=0.50 sl=1125.18 alert=retest2 |

### Cycle 69 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1225.40 | 1153.65 | 1153.06 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 1164.00 | 1183.62 | 1184.36 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 1242.40 | 1195.38 | 1189.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 1272.90 | 1216.68 | 1200.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 1245.40 | 1247.49 | 1224.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 11:15:00 | 1229.80 | 1243.57 | 1226.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 1229.80 | 1243.57 | 1226.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:30:00 | 1223.30 | 1243.57 | 1226.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 1247.20 | 1244.30 | 1228.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 12:30:00 | 1230.00 | 1244.30 | 1228.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1242.90 | 1253.66 | 1239.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:45:00 | 1241.30 | 1253.66 | 1239.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 1307.00 | 1276.30 | 1258.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:15:00 | 1317.00 | 1276.30 | 1258.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 1342.10 | 1294.39 | 1277.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 1319.30 | 1348.52 | 1345.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 1331.90 | 1342.33 | 1343.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 1331.90 | 1342.33 | 1343.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 1331.90 | 1342.33 | 1343.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 1331.90 | 1342.33 | 1343.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 13:15:00 | 1321.80 | 1335.93 | 1339.84 | Break + close below crossover candle low |

### Cycle 73 — BUY (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 14:15:00 | 1369.20 | 1342.59 | 1342.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 10:15:00 | 1386.00 | 1365.57 | 1358.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 1406.90 | 1416.85 | 1403.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 15:00:00 | 1406.90 | 1416.85 | 1403.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 1402.20 | 1413.92 | 1403.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1423.00 | 1413.92 | 1403.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 1411.40 | 1433.43 | 1433.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 1411.40 | 1433.43 | 1433.47 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 1444.20 | 1428.96 | 1428.29 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 1413.40 | 1428.86 | 1428.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 11:15:00 | 1405.90 | 1415.90 | 1421.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 12:15:00 | 1405.60 | 1399.87 | 1407.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 13:00:00 | 1405.60 | 1399.87 | 1407.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 1402.00 | 1400.30 | 1407.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 14:15:00 | 1398.40 | 1400.30 | 1407.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 15:00:00 | 1397.10 | 1399.66 | 1406.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1432.80 | 1405.62 | 1407.77 | SL hit (close>static) qty=1.00 sl=1412.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 1432.80 | 1405.62 | 1407.77 | SL hit (close>static) qty=1.00 sl=1412.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 1431.60 | 1410.82 | 1409.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 1588.00 | 1452.81 | 1430.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 1653.00 | 1673.52 | 1603.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:30:00 | 1652.00 | 1673.52 | 1603.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1624.90 | 1644.87 | 1620.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:30:00 | 1610.10 | 1644.87 | 1620.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 1615.10 | 1638.92 | 1620.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 1615.10 | 1638.92 | 1620.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 1611.70 | 1633.47 | 1619.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 1611.70 | 1633.47 | 1619.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 1608.50 | 1628.48 | 1618.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:45:00 | 1609.50 | 1628.48 | 1618.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 09:15:00 | 1280.50 | 2025-05-14 10:15:00 | 1267.20 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-05-21 09:15:00 | 1343.30 | 2025-05-22 09:15:00 | 1320.50 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-05-21 12:15:00 | 1327.50 | 2025-05-22 09:15:00 | 1320.50 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-05-29 09:30:00 | 1337.80 | 2025-05-29 10:15:00 | 1326.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-05-29 10:00:00 | 1343.50 | 2025-05-29 10:15:00 | 1326.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-06-10 09:30:00 | 1534.10 | 2025-06-11 11:15:00 | 1687.51 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-23 09:30:00 | 1722.30 | 2025-06-23 11:15:00 | 1733.90 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-07-28 11:30:00 | 1710.10 | 2025-08-01 15:15:00 | 1624.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-29 12:30:00 | 1707.50 | 2025-08-01 15:15:00 | 1622.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 11:30:00 | 1710.10 | 2025-08-05 15:15:00 | 1539.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-29 12:30:00 | 1707.50 | 2025-08-05 15:15:00 | 1536.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-21 12:15:00 | 1488.90 | 2025-09-01 13:15:00 | 1415.88 | PARTIAL | 0.50 | 4.90% |
| SELL | retest2 | 2025-08-21 12:45:00 | 1490.40 | 2025-09-01 14:15:00 | 1414.45 | PARTIAL | 0.50 | 5.10% |
| SELL | retest2 | 2025-08-21 12:15:00 | 1488.90 | 2025-09-02 11:15:00 | 1429.30 | STOP_HIT | 0.50 | 4.00% |
| SELL | retest2 | 2025-08-21 12:45:00 | 1490.40 | 2025-09-02 11:15:00 | 1429.30 | STOP_HIT | 0.50 | 4.10% |
| SELL | retest2 | 2025-09-25 13:15:00 | 1484.80 | 2025-09-26 09:15:00 | 1410.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 13:15:00 | 1484.80 | 2025-09-26 14:15:00 | 1336.32 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-10-15 12:15:00 | 1370.00 | 2025-10-21 14:15:00 | 1371.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-10-21 14:15:00 | 1371.00 | 2025-10-21 14:15:00 | 1371.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest1 | 2025-10-28 14:15:00 | 1298.40 | 2025-10-29 11:15:00 | 1317.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-10-29 15:00:00 | 1305.50 | 2025-11-03 09:15:00 | 1351.00 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2025-10-30 09:30:00 | 1299.00 | 2025-11-03 09:15:00 | 1351.00 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-11-06 11:30:00 | 1353.60 | 2025-11-07 09:15:00 | 1326.50 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-11-06 12:30:00 | 1355.60 | 2025-11-07 09:15:00 | 1326.50 | STOP_HIT | 1.00 | -2.15% |
| BUY | retest2 | 2025-11-06 15:00:00 | 1357.80 | 2025-11-07 09:15:00 | 1326.50 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-11-14 11:15:00 | 1350.60 | 2025-11-14 12:15:00 | 1345.30 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-11-14 12:00:00 | 1350.90 | 2025-11-14 12:15:00 | 1345.30 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-11-25 09:15:00 | 1282.50 | 2025-11-27 09:15:00 | 1310.80 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-11-27 11:45:00 | 1285.00 | 2025-12-01 11:15:00 | 1375.60 | STOP_HIT | 1.00 | -7.05% |
| SELL | retest2 | 2025-12-05 15:15:00 | 1348.60 | 2025-12-15 09:15:00 | 1351.70 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-12-10 10:30:00 | 1351.70 | 2025-12-15 09:15:00 | 1351.70 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-12-18 12:00:00 | 1375.60 | 2025-12-29 14:15:00 | 1430.50 | STOP_HIT | 1.00 | 3.99% |
| BUY | retest2 | 2025-12-18 13:30:00 | 1376.10 | 2025-12-29 14:15:00 | 1430.50 | STOP_HIT | 1.00 | 3.95% |
| BUY | retest2 | 2025-12-18 14:30:00 | 1375.90 | 2025-12-29 14:15:00 | 1430.50 | STOP_HIT | 1.00 | 3.97% |
| SELL | retest2 | 2025-12-31 11:15:00 | 1381.20 | 2025-12-31 12:15:00 | 1440.20 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2026-01-22 11:15:00 | 1367.30 | 2026-01-29 14:15:00 | 1370.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2026-01-27 10:30:00 | 1362.00 | 2026-01-29 14:15:00 | 1370.30 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-01-29 09:45:00 | 1366.10 | 2026-01-29 14:15:00 | 1370.30 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-02-09 09:15:00 | 1458.60 | 2026-02-12 09:15:00 | 1387.00 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2026-02-10 09:15:00 | 1414.20 | 2026-02-12 09:15:00 | 1387.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-02-10 14:30:00 | 1421.20 | 2026-02-12 09:15:00 | 1387.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-02-11 10:00:00 | 1408.00 | 2026-02-12 09:15:00 | 1387.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest1 | 2026-02-24 09:15:00 | 1353.90 | 2026-02-25 14:15:00 | 1382.30 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest1 | 2026-02-24 12:00:00 | 1360.10 | 2026-02-25 14:15:00 | 1382.30 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest1 | 2026-02-24 15:15:00 | 1360.00 | 2026-02-25 14:15:00 | 1382.30 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2026-02-25 13:30:00 | 1356.00 | 2026-02-25 14:15:00 | 1382.30 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-03-05 13:45:00 | 1298.40 | 2026-03-06 09:15:00 | 1322.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-03-06 10:45:00 | 1303.30 | 2026-03-10 09:15:00 | 1323.20 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2026-03-06 14:45:00 | 1299.90 | 2026-03-10 09:15:00 | 1323.20 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2026-03-09 09:15:00 | 1264.70 | 2026-03-10 09:15:00 | 1323.20 | STOP_HIT | 1.00 | -4.63% |
| SELL | retest2 | 2026-03-10 11:15:00 | 1313.40 | 2026-03-10 12:15:00 | 1322.10 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-03-10 11:45:00 | 1313.00 | 2026-03-10 12:15:00 | 1322.10 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-03-18 12:45:00 | 1210.50 | 2026-03-23 09:15:00 | 1149.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1195.00 | 2026-03-23 09:15:00 | 1135.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 12:45:00 | 1210.50 | 2026-03-23 12:15:00 | 1089.45 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 1195.00 | 2026-03-24 09:15:00 | 1134.00 | STOP_HIT | 0.50 | 5.10% |
| BUY | retest2 | 2026-04-07 10:15:00 | 1317.00 | 2026-04-13 11:15:00 | 1331.90 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2026-04-08 09:15:00 | 1342.10 | 2026-04-13 11:15:00 | 1331.90 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-04-13 09:45:00 | 1319.30 | 2026-04-13 11:15:00 | 1331.90 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2026-04-22 09:15:00 | 1423.00 | 2026-04-24 11:15:00 | 1411.40 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-04-30 14:15:00 | 1398.40 | 2026-05-04 09:15:00 | 1432.80 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-04-30 15:00:00 | 1397.10 | 2026-05-04 09:15:00 | 1432.80 | STOP_HIT | 1.00 | -2.56% |
