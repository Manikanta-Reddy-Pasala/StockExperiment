# Torrent Power Ltd. (TORNTPOWER)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1717.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 7 |
| TARGET_HIT | 10 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 25
- **Target hits / Stop hits / Partials:** 10 / 25 / 7
- **Avg / median % per leg:** 1.99% / -1.25%
- **Sum % (uncompounded):** 83.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 10 | 3 | 30.0% | 3 | 7 | 0 | 1.71% | 17.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 10 | 3 | 30.0% | 3 | 7 | 0 | 1.71% | 17.1% |
| SELL (all) | 32 | 14 | 43.8% | 7 | 18 | 7 | 2.08% | 66.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 14 | 43.8% | 7 | 18 | 7 | 2.08% | 66.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 17 | 40.5% | 10 | 25 | 7 | 1.99% | 83.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 15:15:00 | 1588.00 | 1777.23 | 1778.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 1568.15 | 1762.29 | 1770.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 14:15:00 | 1675.90 | 1670.84 | 1715.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 15:00:00 | 1675.90 | 1670.84 | 1715.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 1715.00 | 1671.28 | 1715.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 09:15:00 | 1663.40 | 1671.28 | 1715.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 14:30:00 | 1666.90 | 1671.19 | 1712.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 09:30:00 | 1674.05 | 1671.06 | 1711.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-11 15:00:00 | 1667.75 | 1663.84 | 1703.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 1685.50 | 1663.96 | 1701.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 15:15:00 | 1680.00 | 1663.96 | 1701.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 11:15:00 | 1596.00 | 1659.01 | 1694.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 14:15:00 | 1590.35 | 1657.34 | 1693.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 1580.23 | 1656.29 | 1692.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 1583.56 | 1656.29 | 1692.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-19 09:15:00 | 1584.36 | 1656.29 | 1692.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-12-20 14:15:00 | 1497.06 | 1649.52 | 1686.95 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 10:15:00 | 1533.70 | 1421.03 | 1420.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-04 12:15:00 | 1537.10 | 1423.11 | 1421.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 1508.70 | 1518.51 | 1482.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 12:00:00 | 1508.70 | 1518.51 | 1482.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 1485.00 | 1517.53 | 1482.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:15:00 | 1462.50 | 1517.53 | 1482.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1468.90 | 1517.05 | 1482.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 1466.80 | 1517.05 | 1482.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1485.00 | 1516.73 | 1482.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:30:00 | 1465.50 | 1516.73 | 1482.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 1471.90 | 1516.28 | 1482.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:45:00 | 1473.70 | 1516.28 | 1482.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 1470.20 | 1515.83 | 1482.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:30:00 | 1474.90 | 1515.83 | 1482.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 1476.70 | 1515.14 | 1482.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:00:00 | 1476.70 | 1515.14 | 1482.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 1479.00 | 1514.78 | 1482.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 1500.40 | 1514.78 | 1482.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 11:15:00 | 1474.30 | 1513.81 | 1482.56 | SL hit (close<static) qty=1.00 sl=1474.70 alert=retest2 |

### Cycle 3 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 1396.00 | 1464.20 | 1464.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 09:15:00 | 1388.10 | 1451.19 | 1457.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 13:15:00 | 1441.50 | 1431.25 | 1444.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 13:15:00 | 1441.50 | 1431.25 | 1444.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1441.50 | 1431.25 | 1444.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:45:00 | 1442.90 | 1431.25 | 1444.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 1450.70 | 1431.44 | 1444.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 14:30:00 | 1453.90 | 1431.44 | 1444.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 1452.70 | 1431.65 | 1444.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 1463.00 | 1431.65 | 1444.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1433.40 | 1432.61 | 1444.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 10:30:00 | 1425.40 | 1432.50 | 1443.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 1426.90 | 1429.04 | 1441.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 11:15:00 | 1450.00 | 1419.72 | 1433.90 | SL hit (close>static) qty=1.00 sl=1445.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 1320.00 | 1300.28 | 1300.25 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 1259.10 | 1300.52 | 1300.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 1252.00 | 1300.04 | 1300.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 13:15:00 | 1293.00 | 1292.59 | 1296.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 14:00:00 | 1293.00 | 1292.59 | 1296.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1291.90 | 1292.32 | 1295.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:45:00 | 1295.00 | 1292.32 | 1295.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 1295.60 | 1292.10 | 1295.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 1295.60 | 1292.10 | 1295.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1281.40 | 1291.99 | 1295.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1264.60 | 1291.37 | 1295.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 12:15:00 | 1304.20 | 1287.81 | 1292.44 | SL hit (close>static) qty=1.00 sl=1303.00 alert=retest2 |

### Cycle 6 — BUY (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 15:15:00 | 1400.60 | 1296.22 | 1295.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 11:15:00 | 1402.70 | 1299.32 | 1297.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 15:15:00 | 1320.10 | 1320.53 | 1309.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-12 09:15:00 | 1309.90 | 1320.53 | 1309.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1329.10 | 1320.62 | 1309.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-12 09:45:00 | 1315.80 | 1320.62 | 1309.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 1313.50 | 1328.63 | 1315.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 1313.80 | 1328.63 | 1315.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 1314.30 | 1328.49 | 1315.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 15:00:00 | 1322.00 | 1326.25 | 1315.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 11:15:00 | 1297.60 | 1325.52 | 1315.40 | SL hit (close<static) qty=1.00 sl=1304.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-04 09:15:00 | 1663.40 | 2024-12-18 11:15:00 | 1596.00 | PARTIAL | 0.50 | 4.05% |
| SELL | retest2 | 2024-12-05 14:30:00 | 1666.90 | 2024-12-18 14:15:00 | 1590.35 | PARTIAL | 0.50 | 4.59% |
| SELL | retest2 | 2024-12-06 09:30:00 | 1674.05 | 2024-12-19 09:15:00 | 1580.23 | PARTIAL | 0.50 | 5.60% |
| SELL | retest2 | 2024-12-11 15:00:00 | 1667.75 | 2024-12-19 09:15:00 | 1583.56 | PARTIAL | 0.50 | 5.05% |
| SELL | retest2 | 2024-12-12 15:15:00 | 1680.00 | 2024-12-19 09:15:00 | 1584.36 | PARTIAL | 0.50 | 5.69% |
| SELL | retest2 | 2024-12-04 09:15:00 | 1663.40 | 2024-12-20 14:15:00 | 1497.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-05 14:30:00 | 1666.90 | 2024-12-20 14:15:00 | 1500.21 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-06 09:30:00 | 1674.05 | 2024-12-20 14:15:00 | 1506.64 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-11 15:00:00 | 1667.75 | 2024-12-20 14:15:00 | 1500.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-12-12 15:15:00 | 1680.00 | 2024-12-20 14:15:00 | 1512.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-05-06 09:15:00 | 1500.40 | 2025-05-06 11:15:00 | 1474.30 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-05-07 10:30:00 | 1484.40 | 2025-05-08 12:15:00 | 1463.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-05-08 11:30:00 | 1481.50 | 2025-05-08 12:15:00 | 1463.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-06-12 10:30:00 | 1425.40 | 2025-06-24 11:15:00 | 1450.00 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-06-17 10:30:00 | 1426.90 | 2025-06-24 11:15:00 | 1450.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-07-08 11:00:00 | 1425.00 | 2025-07-21 09:15:00 | 1357.55 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2025-07-08 12:30:00 | 1429.00 | 2025-07-22 10:15:00 | 1353.75 | PARTIAL | 0.50 | 5.27% |
| SELL | retest2 | 2025-07-08 11:00:00 | 1425.00 | 2025-08-01 14:15:00 | 1286.10 | TARGET_HIT | 0.50 | 9.75% |
| SELL | retest2 | 2025-07-08 12:30:00 | 1429.00 | 2025-08-04 09:15:00 | 1282.50 | TARGET_HIT | 0.50 | 10.25% |
| SELL | retest2 | 2025-10-28 13:30:00 | 1277.00 | 2025-10-29 10:15:00 | 1312.70 | STOP_HIT | 1.00 | -2.80% |
| SELL | retest2 | 2025-10-29 09:30:00 | 1283.90 | 2025-10-29 10:15:00 | 1312.70 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-11-06 09:45:00 | 1278.10 | 2025-11-10 12:15:00 | 1305.90 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-11-07 13:45:00 | 1282.80 | 2025-11-10 12:15:00 | 1305.90 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-11-12 11:15:00 | 1288.50 | 2025-11-17 09:15:00 | 1320.90 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-11-12 13:15:00 | 1290.30 | 2025-11-17 09:15:00 | 1320.90 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-11-12 14:00:00 | 1289.90 | 2025-11-17 09:15:00 | 1320.90 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2025-11-13 09:30:00 | 1290.10 | 2025-11-17 09:15:00 | 1320.90 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-11-21 14:45:00 | 1287.70 | 2025-11-24 13:15:00 | 1302.60 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-11-24 10:30:00 | 1290.70 | 2025-11-24 13:15:00 | 1302.60 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-24 12:15:00 | 1288.10 | 2025-11-24 13:15:00 | 1302.60 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-11-25 15:15:00 | 1290.00 | 2025-11-26 09:15:00 | 1313.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1264.60 | 2025-12-26 12:15:00 | 1304.20 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-12-30 09:15:00 | 1273.20 | 2025-12-31 12:15:00 | 1308.20 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-12-30 11:45:00 | 1273.20 | 2025-12-31 12:15:00 | 1308.20 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2025-12-30 12:15:00 | 1272.90 | 2025-12-31 12:15:00 | 1308.20 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2026-01-22 15:00:00 | 1322.00 | 2026-01-23 11:15:00 | 1297.60 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2026-01-28 09:15:00 | 1334.90 | 2026-02-01 14:15:00 | 1297.80 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2026-02-01 13:45:00 | 1322.50 | 2026-02-01 14:15:00 | 1297.80 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-02-01 14:30:00 | 1322.00 | 2026-02-01 15:15:00 | 1295.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-02-02 09:15:00 | 1313.00 | 2026-02-09 15:15:00 | 1444.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 11:00:00 | 1306.40 | 2026-04-08 09:15:00 | 1437.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 12:15:00 | 1303.80 | 2026-04-08 09:15:00 | 1434.18 | TARGET_HIT | 1.00 | 10.00% |
