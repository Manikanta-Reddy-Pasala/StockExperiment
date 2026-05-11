# Central Depository Services (India) Ltd. (CDSL)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1261.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 8 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 0 / 8
- **Target hits / Stop hits / Partials:** 0 / 8 / 0
- **Avg / median % per leg:** -1.88% / -1.46%
- **Sum % (uncompounded):** -15.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.68% | -3.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.68% | -3.4% |
| SELL (all) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.95% | -11.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.95% | -11.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.88% | -15.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 1456.00 | 1288.07 | 1287.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 11:15:00 | 1472.30 | 1299.46 | 1293.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 14:15:00 | 1691.50 | 1700.87 | 1600.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 15:00:00 | 1691.50 | 1700.87 | 1600.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 1638.00 | 1700.85 | 1629.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:30:00 | 1637.00 | 1700.85 | 1629.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 1630.00 | 1700.15 | 1629.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 1630.00 | 1700.15 | 1629.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 1634.80 | 1699.50 | 1629.08 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1512.00 | 1590.57 | 1590.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1506.70 | 1589.73 | 1590.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 10:15:00 | 1551.60 | 1551.09 | 1567.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 11:00:00 | 1551.60 | 1551.09 | 1567.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1562.30 | 1551.48 | 1566.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 1562.40 | 1551.48 | 1566.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 1568.00 | 1550.42 | 1563.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 1568.00 | 1550.42 | 1563.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1570.90 | 1550.63 | 1563.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:30:00 | 1574.70 | 1550.63 | 1563.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1569.20 | 1551.14 | 1563.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 1569.00 | 1551.14 | 1563.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1569.60 | 1551.32 | 1563.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1579.50 | 1551.32 | 1563.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 1567.70 | 1556.96 | 1565.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 1570.50 | 1556.96 | 1565.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 1572.70 | 1557.23 | 1565.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:00:00 | 1572.70 | 1557.23 | 1565.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1573.00 | 1527.98 | 1546.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 1588.40 | 1527.98 | 1546.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1577.70 | 1528.48 | 1546.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:45:00 | 1583.50 | 1528.48 | 1546.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 1543.30 | 1531.13 | 1547.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:45:00 | 1547.70 | 1531.13 | 1547.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1535.20 | 1531.31 | 1547.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 1532.60 | 1531.31 | 1547.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1557.40 | 1531.68 | 1547.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1557.40 | 1531.68 | 1547.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1559.80 | 1531.96 | 1547.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 1552.20 | 1532.16 | 1547.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1574.80 | 1533.08 | 1547.54 | SL hit (close>static) qty=1.00 sl=1564.80 alert=retest2 |

### Cycle 3 — BUY (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 13:15:00 | 1613.20 | 1559.64 | 1559.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 10:15:00 | 1622.00 | 1570.48 | 1565.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 1581.10 | 1581.94 | 1572.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 10:15:00 | 1581.10 | 1581.94 | 1572.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1581.10 | 1581.94 | 1572.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 1581.10 | 1581.94 | 1572.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1546.00 | 1583.17 | 1573.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 1546.00 | 1583.17 | 1573.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1557.20 | 1582.92 | 1573.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 13:30:00 | 1566.80 | 1576.30 | 1570.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-03 10:45:00 | 1559.90 | 1602.25 | 1590.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 1537.10 | 1598.39 | 1589.08 | SL hit (close<static) qty=1.00 sl=1542.20 alert=retest2 |

### Cycle 4 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 1491.80 | 1580.64 | 1580.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 14:15:00 | 1482.50 | 1577.86 | 1579.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-10 09:15:00 | 1399.70 | 1379.94 | 1434.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-10 09:30:00 | 1411.60 | 1379.94 | 1434.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 10:15:00 | 1284.50 | 1223.91 | 1284.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 11:00:00 | 1284.50 | 1223.91 | 1284.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 11:15:00 | 1287.20 | 1224.54 | 1284.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 12:00:00 | 1287.20 | 1224.54 | 1284.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 1282.80 | 1225.12 | 1284.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:00:00 | 1277.90 | 1225.65 | 1284.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 15:15:00 | 1291.00 | 1226.90 | 1284.65 | SL hit (close>static) qty=1.00 sl=1289.20 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-09 14:00:00 | 1552.20 | 2025-10-10 09:15:00 | 1574.80 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-11-07 13:30:00 | 1566.80 | 2025-12-04 11:15:00 | 1537.10 | STOP_HIT | 1.00 | -1.90% |
| BUY | retest2 | 2025-12-03 10:45:00 | 1559.90 | 2025-12-04 11:15:00 | 1537.10 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-04-08 14:00:00 | 1277.90 | 2026-04-08 15:15:00 | 1291.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1269.10 | 2026-04-09 11:15:00 | 1290.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-04-09 13:30:00 | 1277.10 | 2026-04-10 09:15:00 | 1296.50 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2026-04-09 14:00:00 | 1277.80 | 2026-04-10 09:15:00 | 1296.50 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-04-13 09:15:00 | 1274.90 | 2026-04-15 09:15:00 | 1333.70 | STOP_HIT | 1.00 | -4.61% |
