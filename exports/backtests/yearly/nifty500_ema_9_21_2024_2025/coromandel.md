# Coromandel International Ltd. (COROMANDEL)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 1928.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 172 |
| ALERT1 | 115 |
| ALERT2 | 111 |
| ALERT2_SKIP | 65 |
| ALERT3 | 318 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 134 |
| PARTIAL | 3 |
| TARGET_HIT | 4 |
| STOP_HIT | 131 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 138 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 105
- **Target hits / Stop hits / Partials:** 4 / 131 / 3
- **Avg / median % per leg:** -0.52% / -1.08%
- **Sum % (uncompounded):** -71.98%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 18 | 27.7% | 4 | 61 | 0 | -0.10% | -6.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 65 | 18 | 27.7% | 4 | 61 | 0 | -0.10% | -6.7% |
| SELL (all) | 73 | 15 | 20.5% | 0 | 70 | 3 | -0.89% | -65.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.13% | -3.1% |
| SELL @ 3rd Alert (retest2) | 72 | 15 | 20.8% | 0 | 69 | 3 | -0.86% | -62.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.13% | -3.1% |
| retest2 (combined) | 137 | 33 | 24.1% | 4 | 130 | 3 | -0.50% | -68.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 1201.50 | 1193.03 | 1192.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 15:15:00 | 1209.00 | 1196.22 | 1193.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 11:15:00 | 1251.00 | 1252.83 | 1244.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-18 12:00:00 | 1251.00 | 1252.83 | 1244.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 1252.00 | 1252.66 | 1245.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 1246.70 | 1252.66 | 1245.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 1232.80 | 1248.69 | 1244.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:00:00 | 1232.80 | 1248.69 | 1244.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1235.00 | 1245.95 | 1243.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 10:45:00 | 1235.40 | 1245.95 | 1243.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 12:15:00 | 1235.00 | 1241.33 | 1241.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 1228.80 | 1237.73 | 1239.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 1239.30 | 1235.09 | 1237.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-22 14:15:00 | 1239.30 | 1235.09 | 1237.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 1239.30 | 1235.09 | 1237.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 15:00:00 | 1239.30 | 1235.09 | 1237.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 1242.00 | 1236.47 | 1237.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 1239.25 | 1236.47 | 1237.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 1234.75 | 1236.13 | 1237.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 10:30:00 | 1227.60 | 1235.46 | 1237.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 11:15:00 | 1229.00 | 1235.46 | 1237.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-23 13:00:00 | 1230.00 | 1233.93 | 1236.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 10:15:00 | 1264.95 | 1239.58 | 1237.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 1264.95 | 1239.58 | 1237.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 10:15:00 | 1270.70 | 1253.84 | 1248.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 1270.00 | 1280.40 | 1272.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 1270.00 | 1280.40 | 1272.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1270.00 | 1280.40 | 1272.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 1270.00 | 1280.40 | 1272.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 1280.85 | 1280.49 | 1273.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 12:30:00 | 1287.85 | 1282.38 | 1275.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 14:30:00 | 1296.80 | 1285.21 | 1277.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-04 12:30:00 | 1300.00 | 1322.31 | 1315.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-10 10:15:00 | 1416.63 | 1397.83 | 1382.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 15:15:00 | 1545.50 | 1569.33 | 1572.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 1530.45 | 1561.55 | 1568.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 1537.50 | 1531.59 | 1540.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 1537.50 | 1531.59 | 1540.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 1537.50 | 1531.59 | 1540.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:00:00 | 1537.50 | 1531.59 | 1540.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 1535.10 | 1532.29 | 1540.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:30:00 | 1537.55 | 1532.29 | 1540.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 1538.20 | 1533.47 | 1540.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 12:15:00 | 1531.65 | 1533.47 | 1540.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 13:15:00 | 1562.50 | 1541.78 | 1542.81 | SL hit (close>static) qty=1.00 sl=1553.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 1559.55 | 1545.33 | 1544.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 09:15:00 | 1568.00 | 1554.51 | 1550.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 1588.20 | 1592.35 | 1581.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-02 09:45:00 | 1589.80 | 1592.35 | 1581.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 1575.85 | 1589.05 | 1580.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:45:00 | 1574.15 | 1589.05 | 1580.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 1573.85 | 1586.01 | 1580.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 11:30:00 | 1570.00 | 1586.01 | 1580.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-07-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 15:15:00 | 1567.75 | 1575.32 | 1576.16 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 09:15:00 | 1584.25 | 1577.11 | 1576.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 11:15:00 | 1595.15 | 1581.10 | 1579.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 12:15:00 | 1579.55 | 1580.79 | 1579.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 12:15:00 | 1579.55 | 1580.79 | 1579.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 12:15:00 | 1579.55 | 1580.79 | 1579.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:00:00 | 1579.55 | 1580.79 | 1579.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 13:15:00 | 1579.20 | 1580.47 | 1579.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 13:45:00 | 1578.95 | 1580.47 | 1579.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 1578.85 | 1580.15 | 1579.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 15:00:00 | 1578.85 | 1580.15 | 1579.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 1578.00 | 1579.72 | 1579.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 1582.80 | 1579.72 | 1579.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 1583.50 | 1580.47 | 1579.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 10:45:00 | 1588.45 | 1580.74 | 1579.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 11:15:00 | 1589.50 | 1580.74 | 1579.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:00:00 | 1591.55 | 1586.20 | 1582.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 10:15:00 | 1587.00 | 1602.32 | 1602.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 1587.00 | 1602.32 | 1602.57 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 12:15:00 | 1609.25 | 1602.96 | 1602.77 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 13:15:00 | 1600.90 | 1602.55 | 1602.60 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 14:15:00 | 1614.05 | 1604.85 | 1603.64 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 11:15:00 | 1606.55 | 1607.87 | 1607.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 13:15:00 | 1593.20 | 1603.10 | 1605.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 09:15:00 | 1592.75 | 1589.05 | 1595.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 1592.75 | 1589.05 | 1595.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1592.75 | 1589.05 | 1595.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 10:00:00 | 1569.70 | 1587.26 | 1591.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 11:45:00 | 1576.25 | 1583.64 | 1588.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 1600.55 | 1588.88 | 1590.50 | SL hit (close>static) qty=1.00 sl=1599.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 1629.35 | 1598.74 | 1594.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 10:15:00 | 1632.10 | 1605.42 | 1598.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 1585.70 | 1607.87 | 1603.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 1585.70 | 1607.87 | 1603.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 1585.70 | 1607.87 | 1603.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 10:00:00 | 1585.70 | 1607.87 | 1603.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 1625.70 | 1611.43 | 1605.70 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 1573.50 | 1600.65 | 1601.60 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 1619.95 | 1603.30 | 1601.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 13:15:00 | 1629.70 | 1608.58 | 1603.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1604.25 | 1613.61 | 1607.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 1604.25 | 1613.61 | 1607.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 1604.25 | 1613.61 | 1607.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:45:00 | 1599.95 | 1613.61 | 1607.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 1606.45 | 1612.18 | 1607.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:00:00 | 1606.45 | 1612.18 | 1607.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 1611.90 | 1612.12 | 1608.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:45:00 | 1601.45 | 1612.12 | 1608.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 1640.75 | 1617.85 | 1611.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 1657.50 | 1626.10 | 1621.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 11:00:00 | 1652.60 | 1636.02 | 1627.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 12:15:00 | 1640.60 | 1655.57 | 1657.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 1640.60 | 1655.57 | 1657.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 1632.75 | 1651.01 | 1654.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 10:15:00 | 1646.45 | 1643.86 | 1649.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-02 11:00:00 | 1646.45 | 1643.86 | 1649.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 1642.85 | 1643.65 | 1648.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:30:00 | 1647.65 | 1643.65 | 1648.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 1649.05 | 1626.23 | 1632.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 1649.05 | 1626.23 | 1632.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 1647.80 | 1630.54 | 1633.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:45:00 | 1625.95 | 1634.41 | 1635.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 13:15:00 | 1632.50 | 1634.41 | 1635.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 12:30:00 | 1630.70 | 1624.92 | 1627.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 13:15:00 | 1631.90 | 1624.92 | 1627.62 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 1643.65 | 1628.66 | 1629.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:00:00 | 1643.65 | 1628.66 | 1629.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 1629.35 | 1628.80 | 1629.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:30:00 | 1644.45 | 1628.80 | 1629.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-07 15:15:00 | 1642.65 | 1631.57 | 1630.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 1642.65 | 1631.57 | 1630.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 1657.60 | 1636.78 | 1632.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 12:15:00 | 1629.00 | 1640.17 | 1635.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 12:15:00 | 1629.00 | 1640.17 | 1635.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 1629.00 | 1640.17 | 1635.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 13:00:00 | 1629.00 | 1640.17 | 1635.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 1642.75 | 1640.69 | 1636.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 1660.95 | 1640.22 | 1637.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 12:15:00 | 1689.30 | 1703.80 | 1703.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 12:15:00 | 1689.30 | 1703.80 | 1703.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 13:15:00 | 1685.40 | 1700.12 | 1702.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 1700.00 | 1696.61 | 1699.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 1700.00 | 1696.61 | 1699.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1700.00 | 1696.61 | 1699.73 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 1747.50 | 1707.33 | 1704.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 1760.00 | 1722.51 | 1711.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 09:15:00 | 1734.40 | 1734.71 | 1720.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 10:00:00 | 1734.40 | 1734.71 | 1720.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 1724.80 | 1732.73 | 1721.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 11:00:00 | 1724.80 | 1732.73 | 1721.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 11:15:00 | 1751.90 | 1736.56 | 1724.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 12:15:00 | 1754.90 | 1736.56 | 1724.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 13:30:00 | 1754.90 | 1741.30 | 1728.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 1719.45 | 1734.35 | 1729.25 | SL hit (close<static) qty=1.00 sl=1722.75 alert=retest2 |

### Cycle 20 — SELL (started 2024-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 15:15:00 | 1750.15 | 1758.12 | 1758.96 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 10:15:00 | 1761.00 | 1757.58 | 1757.48 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 11:15:00 | 1746.10 | 1755.29 | 1756.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 1742.70 | 1749.71 | 1753.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 14:15:00 | 1725.40 | 1725.18 | 1734.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 15:00:00 | 1725.40 | 1725.18 | 1734.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 15:15:00 | 1734.30 | 1727.00 | 1734.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:15:00 | 1754.90 | 1727.00 | 1734.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1763.00 | 1734.20 | 1736.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 1763.00 | 1734.20 | 1736.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-08-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 10:15:00 | 1766.15 | 1740.59 | 1739.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 1769.00 | 1756.42 | 1749.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 10:15:00 | 1754.70 | 1756.08 | 1749.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 11:00:00 | 1754.70 | 1756.08 | 1749.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 1752.40 | 1755.34 | 1749.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:30:00 | 1749.10 | 1755.34 | 1749.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 1747.25 | 1753.72 | 1749.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:45:00 | 1746.20 | 1753.72 | 1749.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 1746.15 | 1752.21 | 1749.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:45:00 | 1745.70 | 1752.21 | 1749.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 1743.50 | 1748.99 | 1748.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 1755.75 | 1748.99 | 1748.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 14:15:00 | 1730.45 | 1746.78 | 1748.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 14:15:00 | 1730.45 | 1746.78 | 1748.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-04 10:15:00 | 1725.95 | 1738.17 | 1743.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-05 09:15:00 | 1733.95 | 1730.57 | 1736.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 1733.95 | 1730.57 | 1736.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 1733.95 | 1730.57 | 1736.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:45:00 | 1728.55 | 1730.57 | 1736.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 11:15:00 | 1733.00 | 1730.88 | 1735.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 12:00:00 | 1733.00 | 1730.88 | 1735.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 1706.75 | 1694.66 | 1703.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 1706.75 | 1694.66 | 1703.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 1711.95 | 1698.12 | 1704.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 1719.65 | 1698.12 | 1704.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 10:15:00 | 1715.90 | 1705.29 | 1706.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 11:15:00 | 1709.70 | 1705.29 | 1706.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 15:15:00 | 1710.00 | 1707.33 | 1707.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 15:15:00 | 1710.00 | 1707.33 | 1707.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 1715.90 | 1709.04 | 1708.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 12:15:00 | 1707.80 | 1709.90 | 1708.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-11 12:15:00 | 1707.80 | 1709.90 | 1708.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 12:15:00 | 1707.80 | 1709.90 | 1708.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 12:45:00 | 1708.70 | 1709.90 | 1708.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 13:15:00 | 1704.45 | 1708.81 | 1708.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 13:45:00 | 1707.95 | 1708.81 | 1708.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2024-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 14:15:00 | 1695.70 | 1706.18 | 1707.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 12:15:00 | 1691.40 | 1699.35 | 1703.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-13 09:15:00 | 1699.85 | 1696.22 | 1700.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-13 09:15:00 | 1699.85 | 1696.22 | 1700.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1699.85 | 1696.22 | 1700.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:45:00 | 1704.40 | 1696.22 | 1700.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 1693.35 | 1695.64 | 1699.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 11:15:00 | 1688.05 | 1695.64 | 1699.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 13:15:00 | 1689.85 | 1694.55 | 1698.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-13 15:15:00 | 1689.70 | 1693.95 | 1697.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-16 12:00:00 | 1689.05 | 1691.57 | 1694.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 1700.95 | 1693.44 | 1695.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 13:00:00 | 1700.95 | 1693.44 | 1695.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 1700.35 | 1694.82 | 1695.95 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-16 15:15:00 | 1711.95 | 1699.03 | 1697.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 15:15:00 | 1711.95 | 1699.03 | 1697.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 09:15:00 | 1713.00 | 1701.82 | 1699.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 14:15:00 | 1713.45 | 1715.18 | 1708.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 14:15:00 | 1713.45 | 1715.18 | 1708.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 1713.45 | 1715.18 | 1708.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:45:00 | 1708.00 | 1715.18 | 1708.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1710.00 | 1714.60 | 1709.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:00:00 | 1710.00 | 1714.60 | 1709.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1702.70 | 1712.22 | 1708.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:00:00 | 1702.70 | 1712.22 | 1708.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 1707.35 | 1711.24 | 1708.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 12:30:00 | 1713.80 | 1710.95 | 1708.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 14:15:00 | 1710.85 | 1709.56 | 1708.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 10:15:00 | 1685.00 | 1707.45 | 1708.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 1685.00 | 1707.45 | 1708.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 1681.25 | 1702.21 | 1705.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-23 11:15:00 | 1670.00 | 1663.57 | 1673.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-23 12:00:00 | 1670.00 | 1663.57 | 1673.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 1675.00 | 1665.86 | 1673.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 13:00:00 | 1675.00 | 1665.86 | 1673.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 13:15:00 | 1675.00 | 1667.69 | 1674.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-23 14:00:00 | 1675.00 | 1667.69 | 1674.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 1670.25 | 1668.20 | 1673.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 11:45:00 | 1663.30 | 1667.22 | 1671.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 15:00:00 | 1663.45 | 1666.68 | 1670.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 1655.50 | 1666.13 | 1669.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 1679.55 | 1640.13 | 1644.77 | SL hit (close>static) qty=1.00 sl=1676.85 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 11:15:00 | 1668.40 | 1650.56 | 1649.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 14:15:00 | 1677.00 | 1657.52 | 1654.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 09:15:00 | 1710.80 | 1710.85 | 1690.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 11:15:00 | 1683.55 | 1704.93 | 1691.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 1683.55 | 1704.93 | 1691.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 1683.55 | 1704.93 | 1691.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 1667.25 | 1697.40 | 1688.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 1667.25 | 1697.40 | 1688.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 15:15:00 | 1666.50 | 1680.99 | 1682.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 10:15:00 | 1659.65 | 1674.67 | 1679.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 1603.65 | 1583.67 | 1598.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 1603.65 | 1583.67 | 1598.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 1603.65 | 1583.67 | 1598.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 1603.65 | 1583.67 | 1598.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 1613.80 | 1589.70 | 1600.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:30:00 | 1614.30 | 1589.70 | 1600.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 14:15:00 | 1604.65 | 1602.72 | 1604.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 14:30:00 | 1609.10 | 1602.72 | 1604.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 15:15:00 | 1609.45 | 1604.07 | 1604.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:15:00 | 1631.05 | 1604.07 | 1604.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 09:15:00 | 1635.50 | 1610.35 | 1607.49 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 1608.25 | 1624.53 | 1625.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 10:15:00 | 1602.45 | 1615.64 | 1620.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 1608.50 | 1601.64 | 1609.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 10:15:00 | 1608.50 | 1601.64 | 1609.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 1608.50 | 1601.64 | 1609.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:45:00 | 1605.25 | 1601.64 | 1609.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1615.05 | 1604.33 | 1609.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 1615.05 | 1604.33 | 1609.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 1611.20 | 1605.70 | 1609.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 1605.00 | 1612.01 | 1612.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 09:15:00 | 1618.55 | 1613.32 | 1612.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 09:15:00 | 1618.55 | 1613.32 | 1612.61 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 10:15:00 | 1601.10 | 1610.87 | 1611.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 11:15:00 | 1593.25 | 1607.35 | 1609.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 14:15:00 | 1605.65 | 1602.76 | 1606.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 14:15:00 | 1605.65 | 1602.76 | 1606.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 1605.65 | 1602.76 | 1606.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:45:00 | 1605.15 | 1602.76 | 1606.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 1600.50 | 1602.31 | 1606.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 1610.25 | 1602.31 | 1606.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 1598.45 | 1601.54 | 1605.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:45:00 | 1605.15 | 1601.54 | 1605.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1588.45 | 1583.94 | 1592.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:30:00 | 1582.50 | 1583.94 | 1592.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 1601.35 | 1587.42 | 1593.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:30:00 | 1603.80 | 1587.42 | 1593.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 1607.60 | 1591.46 | 1594.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 1607.60 | 1591.46 | 1594.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 14:15:00 | 1604.95 | 1596.84 | 1596.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-24 09:15:00 | 1631.00 | 1605.78 | 1600.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 09:15:00 | 1614.25 | 1624.99 | 1615.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 1614.25 | 1624.99 | 1615.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1614.25 | 1624.99 | 1615.28 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 13:15:00 | 1586.25 | 1607.77 | 1609.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 1581.80 | 1598.55 | 1603.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 1607.30 | 1597.10 | 1600.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 14:15:00 | 1607.30 | 1597.10 | 1600.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 1607.30 | 1597.10 | 1600.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 15:00:00 | 1607.30 | 1597.10 | 1600.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 1617.05 | 1601.09 | 1602.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 1620.00 | 1601.09 | 1602.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 1618.95 | 1604.66 | 1603.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 1631.70 | 1613.17 | 1607.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1653.55 | 1663.23 | 1646.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-04 10:00:00 | 1653.55 | 1663.23 | 1646.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1646.60 | 1659.90 | 1646.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 1646.60 | 1659.90 | 1646.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 1664.00 | 1660.72 | 1648.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 14:15:00 | 1672.75 | 1652.42 | 1648.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 1673.95 | 1654.93 | 1650.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 12:15:00 | 1733.10 | 1739.33 | 1739.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 12:15:00 | 1733.10 | 1739.33 | 1739.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 1729.70 | 1736.11 | 1738.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 11:15:00 | 1707.45 | 1699.35 | 1710.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 12:00:00 | 1707.45 | 1699.35 | 1710.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 1717.10 | 1702.90 | 1711.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 1717.10 | 1702.90 | 1711.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 1713.00 | 1704.92 | 1711.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 15:15:00 | 1708.10 | 1707.81 | 1712.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 10:15:00 | 1727.80 | 1713.16 | 1713.71 | SL hit (close>static) qty=1.00 sl=1719.45 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 11:15:00 | 1738.40 | 1718.21 | 1715.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 12:15:00 | 1740.35 | 1722.63 | 1718.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-18 14:15:00 | 1721.70 | 1724.10 | 1719.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 14:15:00 | 1721.70 | 1724.10 | 1719.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 14:15:00 | 1721.70 | 1724.10 | 1719.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:30:00 | 1721.00 | 1724.10 | 1719.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 15:15:00 | 1721.35 | 1723.55 | 1719.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-19 09:15:00 | 1734.80 | 1723.55 | 1719.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 13:15:00 | 1764.60 | 1782.84 | 1784.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-11-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 13:15:00 | 1764.60 | 1782.84 | 1784.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 09:15:00 | 1755.00 | 1772.63 | 1779.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 1768.15 | 1762.21 | 1769.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 1768.15 | 1762.21 | 1769.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 1768.15 | 1762.21 | 1769.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:45:00 | 1772.90 | 1762.21 | 1769.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 1754.85 | 1760.74 | 1767.83 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 11:15:00 | 1785.15 | 1768.16 | 1767.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 13:15:00 | 1794.10 | 1775.40 | 1770.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 11:15:00 | 1785.45 | 1785.80 | 1778.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-02 11:45:00 | 1783.20 | 1785.80 | 1778.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 12:15:00 | 1773.25 | 1783.29 | 1777.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:00:00 | 1773.25 | 1783.29 | 1777.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 1777.75 | 1782.18 | 1777.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 13:45:00 | 1774.50 | 1782.18 | 1777.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 1775.60 | 1780.86 | 1777.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 14:45:00 | 1774.30 | 1780.86 | 1777.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 1772.40 | 1779.17 | 1777.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 1774.25 | 1779.17 | 1777.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 1768.00 | 1776.94 | 1776.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 1768.00 | 1776.94 | 1776.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 10:15:00 | 1770.00 | 1775.55 | 1775.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 11:15:00 | 1761.35 | 1772.71 | 1774.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 12:15:00 | 1774.35 | 1773.04 | 1774.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 12:15:00 | 1774.35 | 1773.04 | 1774.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 1774.35 | 1773.04 | 1774.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:30:00 | 1776.15 | 1773.04 | 1774.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 13:15:00 | 1777.50 | 1773.93 | 1774.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 13:30:00 | 1777.55 | 1773.93 | 1774.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 14:15:00 | 1776.10 | 1774.36 | 1774.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 15:15:00 | 1775.60 | 1774.36 | 1774.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 1775.60 | 1774.61 | 1774.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:15:00 | 1793.35 | 1774.61 | 1774.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 09:15:00 | 1797.30 | 1779.15 | 1776.99 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 13:15:00 | 1762.90 | 1775.70 | 1776.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 14:15:00 | 1758.90 | 1772.34 | 1774.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 12:15:00 | 1763.70 | 1762.92 | 1768.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 13:00:00 | 1763.70 | 1762.92 | 1768.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 13:15:00 | 1768.00 | 1763.93 | 1768.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 13:30:00 | 1767.30 | 1763.93 | 1768.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 14:15:00 | 1768.95 | 1764.94 | 1768.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-05 14:30:00 | 1768.85 | 1764.94 | 1768.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 1768.00 | 1765.55 | 1768.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 1783.15 | 1765.55 | 1768.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1767.75 | 1765.99 | 1768.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 10:45:00 | 1765.50 | 1765.71 | 1767.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 14:45:00 | 1766.30 | 1765.75 | 1767.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 13:45:00 | 1762.30 | 1758.61 | 1761.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-09 15:15:00 | 1762.05 | 1760.54 | 1762.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 1762.05 | 1760.84 | 1762.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 09:15:00 | 1777.10 | 1760.84 | 1762.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-10 09:15:00 | 1776.20 | 1763.91 | 1763.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 1776.20 | 1763.91 | 1763.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 10:15:00 | 1794.70 | 1770.07 | 1766.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 15:15:00 | 1776.20 | 1777.54 | 1772.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 15:15:00 | 1776.20 | 1777.54 | 1772.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 15:15:00 | 1776.20 | 1777.54 | 1772.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 09:30:00 | 1789.70 | 1777.49 | 1772.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 14:15:00 | 1768.60 | 1776.09 | 1774.04 | SL hit (close<static) qty=1.00 sl=1770.90 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 1750.80 | 1774.76 | 1775.26 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 10:15:00 | 1814.30 | 1778.42 | 1774.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 13:15:00 | 1820.20 | 1797.03 | 1785.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 1807.75 | 1819.45 | 1803.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 12:15:00 | 1807.75 | 1819.45 | 1803.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 1807.75 | 1819.45 | 1803.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:45:00 | 1799.90 | 1819.45 | 1803.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 15:15:00 | 1805.00 | 1814.92 | 1805.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 1866.15 | 1814.92 | 1805.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 11:45:00 | 1821.00 | 1819.56 | 1810.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 12:45:00 | 1830.50 | 1821.94 | 1812.08 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 11:15:00 | 1827.60 | 1847.14 | 1849.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 11:15:00 | 1827.60 | 1847.14 | 1849.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 12:15:00 | 1818.65 | 1841.45 | 1846.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 14:15:00 | 1845.90 | 1840.26 | 1844.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 14:15:00 | 1845.90 | 1840.26 | 1844.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 1845.90 | 1840.26 | 1844.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 1845.90 | 1840.26 | 1844.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 1848.00 | 1841.81 | 1845.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 1859.00 | 1841.81 | 1845.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1855.00 | 1844.44 | 1845.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:30:00 | 1856.90 | 1844.44 | 1845.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1852.25 | 1846.01 | 1846.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:45:00 | 1845.05 | 1844.90 | 1845.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 1841.40 | 1845.08 | 1845.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:15:00 | 1831.20 | 1846.06 | 1846.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 13:30:00 | 1842.65 | 1842.07 | 1843.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 14:15:00 | 1871.10 | 1847.87 | 1846.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2024-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 14:15:00 | 1871.10 | 1847.87 | 1846.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-31 12:15:00 | 1875.45 | 1856.99 | 1851.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-02 09:15:00 | 1902.90 | 1904.92 | 1886.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-02 09:45:00 | 1904.00 | 1904.92 | 1886.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 1947.80 | 1953.36 | 1938.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 1936.65 | 1953.36 | 1938.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 13:15:00 | 1940.20 | 1949.10 | 1938.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 13:45:00 | 1938.65 | 1949.10 | 1938.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 1934.90 | 1946.26 | 1938.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 14:45:00 | 1944.10 | 1946.26 | 1938.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 1941.95 | 1945.40 | 1938.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 09:15:00 | 1956.45 | 1945.40 | 1938.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 12:30:00 | 1945.85 | 1947.33 | 1942.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 13:15:00 | 1943.70 | 1947.33 | 1942.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 09:15:00 | 1925.80 | 1944.25 | 1942.68 | SL hit (close<static) qty=1.00 sl=1932.70 alert=retest2 |

### Cycle 50 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 1922.50 | 1939.90 | 1940.85 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 1972.50 | 1942.20 | 1940.17 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 10:15:00 | 1933.85 | 1944.08 | 1944.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 1860.35 | 1916.70 | 1930.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 14:15:00 | 1832.35 | 1830.52 | 1857.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 15:00:00 | 1832.35 | 1830.52 | 1857.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 1874.25 | 1838.80 | 1846.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:00:00 | 1874.25 | 1838.80 | 1846.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 10:15:00 | 1878.35 | 1846.71 | 1848.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 10:45:00 | 1898.50 | 1846.71 | 1848.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1886.90 | 1854.75 | 1852.39 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 1829.05 | 1863.01 | 1863.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 1809.00 | 1843.66 | 1851.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1858.50 | 1804.84 | 1821.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1858.50 | 1804.84 | 1821.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1858.50 | 1804.84 | 1821.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 1858.50 | 1804.84 | 1821.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1851.45 | 1814.16 | 1824.05 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 1849.95 | 1832.41 | 1830.93 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 1813.80 | 1832.56 | 1833.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1750.75 | 1812.91 | 1824.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 1739.80 | 1724.74 | 1747.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 10:00:00 | 1739.80 | 1724.74 | 1747.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 1749.00 | 1729.60 | 1747.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:00:00 | 1749.00 | 1729.60 | 1747.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 1771.20 | 1737.92 | 1749.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 1771.80 | 1737.92 | 1749.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 1780.10 | 1746.35 | 1752.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 13:00:00 | 1780.10 | 1746.35 | 1752.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 1796.75 | 1763.02 | 1759.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 1837.20 | 1781.20 | 1768.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 1813.10 | 1814.44 | 1795.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 1803.70 | 1811.26 | 1804.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1803.70 | 1811.26 | 1804.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1803.45 | 1811.26 | 1804.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1803.60 | 1809.73 | 1804.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 1810.00 | 1809.73 | 1804.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1810.15 | 1809.81 | 1805.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 1826.75 | 1804.50 | 1803.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 1816.70 | 1806.40 | 1804.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 11:00:00 | 1818.00 | 1808.72 | 1805.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 11:30:00 | 1817.00 | 1809.77 | 1806.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 1810.80 | 1810.71 | 1807.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:00:00 | 1810.80 | 1810.71 | 1807.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 1811.40 | 1810.85 | 1807.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:30:00 | 1809.00 | 1810.85 | 1807.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 1812.05 | 1811.09 | 1808.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 1834.45 | 1811.09 | 1808.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 14:15:00 | 1858.65 | 1862.92 | 1863.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 14:15:00 | 1858.65 | 1862.92 | 1863.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1839.45 | 1857.37 | 1860.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1838.70 | 1822.15 | 1832.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 1838.70 | 1822.15 | 1832.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1838.70 | 1822.15 | 1832.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:45:00 | 1844.80 | 1822.15 | 1832.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1873.55 | 1832.43 | 1836.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:00:00 | 1873.55 | 1832.43 | 1836.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 12:15:00 | 1880.60 | 1842.06 | 1840.56 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 1799.80 | 1839.94 | 1842.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 1769.90 | 1825.93 | 1836.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 13:15:00 | 1701.75 | 1696.27 | 1727.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 13:45:00 | 1703.45 | 1696.27 | 1727.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 1733.25 | 1707.32 | 1725.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 1733.25 | 1707.32 | 1725.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 1727.90 | 1711.43 | 1725.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 1741.60 | 1711.43 | 1725.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 1726.30 | 1714.41 | 1725.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:30:00 | 1727.70 | 1714.41 | 1725.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 1710.40 | 1713.61 | 1724.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 15:15:00 | 1708.95 | 1715.32 | 1723.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 09:15:00 | 1737.10 | 1718.66 | 1723.28 | SL hit (close>static) qty=1.00 sl=1731.40 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 14:15:00 | 1742.40 | 1728.30 | 1726.53 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 09:15:00 | 1698.15 | 1723.05 | 1724.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 10:15:00 | 1688.00 | 1716.04 | 1721.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-21 14:15:00 | 1711.40 | 1704.72 | 1713.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-21 15:00:00 | 1711.40 | 1704.72 | 1713.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 1711.35 | 1706.04 | 1712.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-24 10:15:00 | 1695.85 | 1704.92 | 1711.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 11:15:00 | 1611.06 | 1655.79 | 1679.74 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-25 13:15:00 | 1663.95 | 1654.66 | 1674.90 | SL hit (close>ema200) qty=0.50 sl=1654.66 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 1660.00 | 1653.22 | 1652.57 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 10:15:00 | 1644.95 | 1651.00 | 1651.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-04 13:15:00 | 1641.25 | 1648.26 | 1650.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-05 09:15:00 | 1695.30 | 1645.54 | 1647.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 1695.30 | 1645.54 | 1647.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 1695.30 | 1645.54 | 1647.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 1688.35 | 1645.54 | 1647.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1730.30 | 1662.49 | 1654.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 1761.45 | 1682.28 | 1664.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 12:15:00 | 1749.30 | 1754.71 | 1737.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 12:45:00 | 1747.40 | 1754.71 | 1737.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1750.55 | 1752.69 | 1739.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 1737.60 | 1752.69 | 1739.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 1737.00 | 1749.55 | 1739.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 1714.00 | 1749.55 | 1739.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1706.65 | 1740.97 | 1736.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 1708.00 | 1740.97 | 1736.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 1716.35 | 1736.05 | 1734.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:45:00 | 1706.70 | 1736.05 | 1734.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 1715.85 | 1732.01 | 1732.91 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 09:15:00 | 1753.25 | 1733.53 | 1732.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 1796.60 | 1759.90 | 1748.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 12:15:00 | 1958.00 | 1958.32 | 1916.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 13:00:00 | 1958.00 | 1958.32 | 1916.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 1958.10 | 1972.19 | 1959.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:30:00 | 1954.50 | 1972.19 | 1959.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 1964.05 | 1970.57 | 1959.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:15:00 | 1962.80 | 1970.57 | 1959.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 1967.50 | 1969.95 | 1960.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 1996.35 | 1973.55 | 1963.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 14:45:00 | 1993.00 | 1988.25 | 1977.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:30:00 | 1985.55 | 1983.98 | 1977.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:30:00 | 1986.35 | 1980.21 | 1977.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 1979.00 | 1979.97 | 1977.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 15:15:00 | 1980.00 | 1979.97 | 1977.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 15:15:00 | 1980.00 | 1979.98 | 1977.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:15:00 | 2000.40 | 1979.98 | 1977.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 15:15:00 | 1993.15 | 1987.95 | 1984.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:30:00 | 1984.50 | 1989.72 | 1986.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 12:00:00 | 1986.90 | 1988.04 | 1985.88 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 12:15:00 | 1958.80 | 1982.19 | 1983.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 12:15:00 | 1958.80 | 1982.19 | 1983.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 13:15:00 | 1949.30 | 1975.61 | 1980.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 15:15:00 | 1976.00 | 1975.57 | 1979.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 15:15:00 | 1976.00 | 1975.57 | 1979.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1976.00 | 1975.57 | 1979.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1992.15 | 1975.57 | 1979.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1999.00 | 1980.25 | 1981.25 | EMA400 retest candle locked (from downside) |

### Cycle 69 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 1993.45 | 1982.89 | 1982.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-01 14:15:00 | 2058.25 | 2008.43 | 1996.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 2030.00 | 2035.06 | 2020.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 09:45:00 | 2029.05 | 2035.06 | 2020.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 2025.30 | 2031.94 | 2022.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:45:00 | 2024.95 | 2031.94 | 2022.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 13:15:00 | 2038.85 | 2033.32 | 2023.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 13:30:00 | 2027.90 | 2033.32 | 2023.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 2057.15 | 2054.84 | 2037.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:15:00 | 2027.65 | 2054.84 | 2037.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 2049.80 | 2053.83 | 2038.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:30:00 | 2055.75 | 2053.83 | 2038.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 11:15:00 | 2046.65 | 2052.40 | 2039.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:00:00 | 2046.65 | 2052.40 | 2039.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 12:15:00 | 2039.10 | 2049.74 | 2039.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 12:45:00 | 2023.80 | 2049.74 | 2039.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 2028.75 | 2045.54 | 2038.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:00:00 | 2028.75 | 2045.54 | 2038.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 2013.20 | 2039.07 | 2035.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 15:15:00 | 2020.00 | 2039.07 | 2035.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 2020.00 | 2035.26 | 2034.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:15:00 | 1953.50 | 2035.26 | 2034.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 1929.95 | 2014.20 | 2024.98 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 2067.10 | 2011.37 | 2008.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 2093.55 | 2027.81 | 2016.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-09 09:15:00 | 2023.90 | 2041.53 | 2027.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-09 09:15:00 | 2023.90 | 2041.53 | 2027.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 09:15:00 | 2023.90 | 2041.53 | 2027.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 09:45:00 | 2020.90 | 2041.53 | 2027.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 10:15:00 | 2014.55 | 2036.13 | 2026.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-09 11:00:00 | 2014.55 | 2036.13 | 2026.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-09 11:15:00 | 2037.15 | 2036.33 | 2027.53 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-09 14:15:00 | 2003.65 | 2021.69 | 2022.53 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-04-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 09:15:00 | 2128.00 | 2043.48 | 2032.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 14:15:00 | 2173.80 | 2110.45 | 2082.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 2129.60 | 2135.41 | 2102.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 10:30:00 | 2119.30 | 2135.41 | 2102.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 2124.10 | 2129.70 | 2108.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 2109.90 | 2129.70 | 2108.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 2151.60 | 2133.38 | 2115.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 10:30:00 | 2169.60 | 2139.73 | 2119.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:15:00 | 2196.50 | 2127.11 | 2120.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 15:15:00 | 2210.00 | 2233.32 | 2235.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 2210.00 | 2233.32 | 2235.73 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 09:15:00 | 2326.90 | 2252.03 | 2244.01 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 12:15:00 | 2217.70 | 2243.56 | 2245.56 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 2296.30 | 2246.24 | 2245.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 2325.80 | 2281.25 | 2266.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 13:15:00 | 2316.20 | 2320.27 | 2301.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 13:15:00 | 2316.20 | 2320.27 | 2301.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 2316.20 | 2320.27 | 2301.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 13:45:00 | 2306.30 | 2320.27 | 2301.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 2284.00 | 2313.02 | 2300.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 2284.00 | 2313.02 | 2300.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 2253.60 | 2301.13 | 2295.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 2253.90 | 2301.13 | 2295.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 2312.90 | 2303.45 | 2297.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:15:00 | 2321.40 | 2303.45 | 2297.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:45:00 | 2319.30 | 2306.66 | 2299.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-14 12:15:00 | 2340.30 | 2382.51 | 2386.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 12:15:00 | 2340.30 | 2382.51 | 2386.08 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 15:15:00 | 2401.00 | 2387.09 | 2387.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 09:15:00 | 2499.10 | 2409.49 | 2397.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 09:15:00 | 2450.00 | 2450.37 | 2428.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 10:00:00 | 2450.00 | 2450.37 | 2428.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 2436.50 | 2447.60 | 2429.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 2417.00 | 2447.60 | 2429.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 2434.80 | 2445.04 | 2429.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 15:00:00 | 2449.00 | 2444.35 | 2433.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:00:00 | 2449.20 | 2446.38 | 2436.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:45:00 | 2449.40 | 2445.87 | 2436.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 14:45:00 | 2458.00 | 2446.74 | 2439.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 2455.00 | 2452.42 | 2443.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 2442.80 | 2452.42 | 2443.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 2443.20 | 2450.58 | 2443.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 2443.20 | 2450.58 | 2443.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-20 11:15:00 | 2370.80 | 2434.62 | 2436.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 2370.80 | 2434.62 | 2436.76 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 2458.00 | 2416.77 | 2415.50 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 13:15:00 | 2408.70 | 2414.39 | 2414.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 14:15:00 | 2389.80 | 2409.47 | 2412.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 2436.10 | 2410.64 | 2412.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 2436.10 | 2410.64 | 2412.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 2436.10 | 2410.64 | 2412.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 2436.10 | 2410.64 | 2412.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 2397.70 | 2408.05 | 2410.92 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 13:15:00 | 2414.70 | 2409.38 | 2409.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 2443.60 | 2416.23 | 2412.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 2401.00 | 2433.69 | 2425.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 14:15:00 | 2401.00 | 2433.69 | 2425.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 2401.00 | 2433.69 | 2425.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:00:00 | 2401.00 | 2433.69 | 2425.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 2400.00 | 2426.96 | 2423.12 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-05-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 09:15:00 | 2358.00 | 2413.16 | 2417.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 11:15:00 | 2352.70 | 2393.16 | 2406.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 2357.70 | 2353.46 | 2379.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 2357.70 | 2353.46 | 2379.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 2357.70 | 2353.46 | 2379.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:00:00 | 2357.70 | 2353.46 | 2379.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 15:15:00 | 2279.00 | 2282.39 | 2308.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:15:00 | 2342.30 | 2282.39 | 2308.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 2353.40 | 2296.59 | 2312.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:45:00 | 2349.60 | 2296.59 | 2312.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 2349.80 | 2307.23 | 2315.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:45:00 | 2352.70 | 2307.23 | 2315.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 14:15:00 | 2314.10 | 2315.60 | 2318.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-03 14:30:00 | 2309.20 | 2315.60 | 2318.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 15:15:00 | 2327.90 | 2318.06 | 2318.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:15:00 | 2317.90 | 2318.06 | 2318.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 2294.10 | 2313.27 | 2316.69 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 15:15:00 | 2323.00 | 2316.25 | 2316.16 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 09:15:00 | 2310.50 | 2315.10 | 2315.64 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 09:15:00 | 2345.00 | 2310.82 | 2310.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 11:15:00 | 2358.60 | 2326.13 | 2318.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 2341.70 | 2361.48 | 2341.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 2341.70 | 2361.48 | 2341.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2341.70 | 2361.48 | 2341.43 | EMA400 retest candle locked (from upside) |

### Cycle 88 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 2320.00 | 2345.40 | 2348.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 2295.70 | 2335.46 | 2343.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 2298.30 | 2296.86 | 2313.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 2278.20 | 2296.86 | 2313.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 2332.00 | 2288.72 | 2297.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:45:00 | 2353.10 | 2288.72 | 2297.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 2337.20 | 2298.42 | 2301.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:30:00 | 2338.10 | 2298.42 | 2301.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 2309.90 | 2301.63 | 2302.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 2309.90 | 2301.63 | 2302.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 2303.40 | 2301.98 | 2302.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 2311.10 | 2301.98 | 2302.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 2314.10 | 2304.40 | 2303.41 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 2294.60 | 2302.44 | 2302.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 2284.70 | 2298.89 | 2300.98 | Break + close below crossover candle low |

### Cycle 91 — BUY (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 09:15:00 | 2341.00 | 2303.28 | 2301.52 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 2287.20 | 2300.41 | 2301.95 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 2311.20 | 2302.57 | 2302.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 14:15:00 | 2395.80 | 2335.83 | 2322.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 12:15:00 | 2358.50 | 2361.39 | 2343.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-25 13:00:00 | 2358.50 | 2361.39 | 2343.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 2348.50 | 2358.81 | 2343.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:00:00 | 2348.50 | 2358.81 | 2343.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 2361.40 | 2359.33 | 2345.13 | EMA400 retest candle locked (from upside) |

### Cycle 94 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 2308.00 | 2336.17 | 2337.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 12:15:00 | 2299.60 | 2328.86 | 2334.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 13:15:00 | 2303.90 | 2277.19 | 2298.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 13:15:00 | 2303.90 | 2277.19 | 2298.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 2303.90 | 2277.19 | 2298.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 14:00:00 | 2303.90 | 2277.19 | 2298.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 14:15:00 | 2416.40 | 2305.03 | 2308.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 15:00:00 | 2416.40 | 2305.03 | 2308.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-06-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 15:15:00 | 2395.70 | 2323.16 | 2316.82 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 2325.00 | 2363.43 | 2364.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 2299.90 | 2344.74 | 2355.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 2309.20 | 2297.01 | 2320.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 2309.20 | 2297.01 | 2320.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2309.20 | 2297.01 | 2320.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:30:00 | 2291.30 | 2296.65 | 2316.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 2270.00 | 2248.11 | 2247.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 2270.00 | 2248.11 | 2247.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 13:15:00 | 2286.60 | 2258.26 | 2252.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 12:15:00 | 2366.00 | 2371.24 | 2348.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 12:45:00 | 2365.10 | 2371.24 | 2348.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 2341.70 | 2363.38 | 2348.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:45:00 | 2335.00 | 2363.38 | 2348.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 2350.00 | 2360.70 | 2348.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 2397.30 | 2360.70 | 2348.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 14:15:00 | 2341.70 | 2358.09 | 2358.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-07-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 14:15:00 | 2341.70 | 2358.09 | 2358.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 2314.00 | 2347.98 | 2353.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 2348.70 | 2327.45 | 2336.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 10:15:00 | 2348.70 | 2327.45 | 2336.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 2348.70 | 2327.45 | 2336.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:00:00 | 2348.70 | 2327.45 | 2336.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 2359.00 | 2333.76 | 2338.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 2358.20 | 2333.76 | 2338.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 2357.70 | 2342.47 | 2341.74 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-07-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 15:15:00 | 2335.30 | 2340.42 | 2340.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 2330.70 | 2338.48 | 2339.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 11:15:00 | 2351.30 | 2339.44 | 2340.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 11:15:00 | 2351.30 | 2339.44 | 2340.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 2351.30 | 2339.44 | 2340.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 2351.30 | 2339.44 | 2340.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 2346.20 | 2340.80 | 2340.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 14:15:00 | 2352.20 | 2343.83 | 2342.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 2336.20 | 2343.90 | 2342.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 2336.20 | 2343.90 | 2342.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2336.20 | 2343.90 | 2342.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 2336.20 | 2343.90 | 2342.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 10:15:00 | 2325.00 | 2340.12 | 2340.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 11:15:00 | 2315.00 | 2335.10 | 2338.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 2334.20 | 2332.39 | 2336.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 15:00:00 | 2334.20 | 2332.39 | 2336.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 2338.80 | 2332.49 | 2335.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 11:00:00 | 2325.40 | 2331.07 | 2334.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 2331.00 | 2328.24 | 2332.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 2419.60 | 2346.51 | 2340.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 2419.60 | 2346.51 | 2340.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 09:15:00 | 2460.00 | 2425.84 | 2409.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 2643.60 | 2646.96 | 2598.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 09:30:00 | 2646.40 | 2646.96 | 2598.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 2607.20 | 2637.37 | 2609.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 2607.20 | 2637.37 | 2609.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 2583.40 | 2626.58 | 2607.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 2583.40 | 2626.58 | 2607.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 2590.00 | 2619.26 | 2605.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:15:00 | 2628.20 | 2619.26 | 2605.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 12:15:00 | 2573.30 | 2596.68 | 2598.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 12:15:00 | 2573.30 | 2596.68 | 2598.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 13:15:00 | 2554.40 | 2588.22 | 2594.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 10:15:00 | 2604.90 | 2576.83 | 2585.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 10:15:00 | 2604.90 | 2576.83 | 2585.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 2604.90 | 2576.83 | 2585.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 2607.40 | 2576.83 | 2585.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 2587.00 | 2578.86 | 2585.18 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 14:15:00 | 2619.40 | 2594.21 | 2591.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 15:15:00 | 2624.10 | 2600.18 | 2594.21 | Break + close above crossover candle high |

### Cycle 106 — SELL (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 09:15:00 | 2544.60 | 2589.07 | 2589.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 2526.70 | 2567.19 | 2579.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 2436.20 | 2433.99 | 2463.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 10:15:00 | 2455.00 | 2433.99 | 2463.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 2435.60 | 2434.31 | 2461.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:30:00 | 2431.60 | 2433.47 | 2458.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 09:30:00 | 2430.70 | 2432.52 | 2448.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 09:15:00 | 2310.02 | 2383.63 | 2415.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-13 09:15:00 | 2309.16 | 2383.63 | 2415.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 2312.60 | 2297.09 | 2345.76 | SL hit (close>ema200) qty=0.50 sl=2297.09 alert=retest2 |

### Cycle 107 — BUY (started 2025-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 10:15:00 | 2400.80 | 2359.31 | 2355.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 2459.00 | 2408.55 | 2393.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 2443.10 | 2444.29 | 2423.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 2443.10 | 2444.29 | 2423.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 2429.50 | 2441.33 | 2424.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 14:45:00 | 2428.40 | 2441.33 | 2424.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 2453.50 | 2441.15 | 2426.86 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-08-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 10:15:00 | 2406.90 | 2421.32 | 2422.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 2388.00 | 2407.05 | 2414.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 2373.50 | 2367.00 | 2386.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 15:00:00 | 2373.50 | 2367.00 | 2386.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 14:15:00 | 2304.90 | 2295.86 | 2318.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 15:00:00 | 2304.90 | 2295.86 | 2318.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 2319.90 | 2300.66 | 2318.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:15:00 | 2301.40 | 2300.66 | 2318.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2308.00 | 2302.13 | 2317.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:15:00 | 2330.70 | 2302.13 | 2317.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 2281.10 | 2297.93 | 2314.09 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 2335.90 | 2321.82 | 2319.92 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 11:15:00 | 2288.60 | 2315.18 | 2317.07 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 2389.60 | 2327.37 | 2321.09 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 2276.10 | 2326.05 | 2328.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 2253.60 | 2287.08 | 2302.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 2217.00 | 2216.39 | 2245.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 10:45:00 | 2217.00 | 2216.39 | 2245.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 2227.10 | 2218.72 | 2234.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:15:00 | 2249.70 | 2218.72 | 2234.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 2244.90 | 2223.96 | 2235.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 2223.80 | 2222.16 | 2233.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:45:00 | 2222.50 | 2221.96 | 2228.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 2246.20 | 2232.80 | 2231.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 2246.20 | 2232.80 | 2231.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 10:15:00 | 2266.90 | 2239.62 | 2234.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 13:15:00 | 2245.90 | 2247.95 | 2240.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:00:00 | 2245.90 | 2247.95 | 2240.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 2229.60 | 2244.28 | 2239.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 15:00:00 | 2229.60 | 2244.28 | 2239.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 2230.00 | 2241.43 | 2238.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 2233.40 | 2241.43 | 2238.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-09-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 10:15:00 | 2204.40 | 2231.10 | 2234.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 11:15:00 | 2193.20 | 2223.52 | 2230.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 10:15:00 | 2227.30 | 2220.24 | 2225.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 10:15:00 | 2227.30 | 2220.24 | 2225.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 2227.30 | 2220.24 | 2225.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:45:00 | 2231.00 | 2220.24 | 2225.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 11:15:00 | 2243.60 | 2224.92 | 2226.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 12:00:00 | 2243.60 | 2224.92 | 2226.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 2251.00 | 2230.13 | 2229.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 13:15:00 | 2257.60 | 2235.63 | 2231.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 10:15:00 | 2293.40 | 2293.57 | 2272.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 11:00:00 | 2293.40 | 2293.57 | 2272.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2306.60 | 2296.62 | 2279.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:30:00 | 2312.20 | 2297.89 | 2281.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:15:00 | 2313.50 | 2297.89 | 2281.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 2272.50 | 2295.31 | 2283.08 | SL hit (close<static) qty=1.00 sl=2276.60 alert=retest2 |

### Cycle 116 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 2221.50 | 2271.14 | 2273.64 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 10:15:00 | 2288.50 | 2269.87 | 2269.03 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 2259.10 | 2268.64 | 2268.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 2247.20 | 2262.78 | 2266.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 2241.40 | 2240.66 | 2251.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 09:15:00 | 2241.40 | 2240.66 | 2251.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 2241.40 | 2240.66 | 2251.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:15:00 | 2258.50 | 2240.66 | 2251.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 2268.20 | 2246.17 | 2252.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:45:00 | 2269.90 | 2246.17 | 2252.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 2278.00 | 2252.54 | 2254.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:45:00 | 2276.50 | 2252.54 | 2254.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 2264.90 | 2256.28 | 2256.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 13:45:00 | 2265.20 | 2256.28 | 2256.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 2238.50 | 2243.53 | 2249.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:30:00 | 2250.10 | 2243.53 | 2249.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 2183.60 | 2231.43 | 2242.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 12:30:00 | 2161.70 | 2196.55 | 2219.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 15:15:00 | 2279.00 | 2216.71 | 2223.21 | SL hit (close>static) qty=1.00 sl=2243.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 12:15:00 | 2279.40 | 2231.01 | 2227.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 2283.70 | 2249.96 | 2240.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 2314.60 | 2316.06 | 2293.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 12:15:00 | 2295.80 | 2313.57 | 2298.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 2295.80 | 2313.57 | 2298.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:00:00 | 2295.80 | 2313.57 | 2298.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 2296.10 | 2310.08 | 2298.22 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 2258.50 | 2286.79 | 2290.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 2233.50 | 2268.88 | 2280.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-13 11:15:00 | 2228.20 | 2225.41 | 2237.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-13 11:30:00 | 2234.80 | 2225.41 | 2237.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 2230.00 | 2226.73 | 2236.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 2230.00 | 2226.73 | 2236.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 2223.80 | 2226.14 | 2234.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 2230.00 | 2226.14 | 2234.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 2197.80 | 2187.76 | 2202.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 2195.90 | 2187.76 | 2202.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 2198.70 | 2191.36 | 2201.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:30:00 | 2204.90 | 2191.36 | 2201.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 2193.60 | 2191.81 | 2200.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 2199.50 | 2191.81 | 2200.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 2192.60 | 2191.97 | 2199.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:15:00 | 2175.10 | 2190.49 | 2198.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:15:00 | 2172.70 | 2178.94 | 2187.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 12:45:00 | 2164.80 | 2168.97 | 2180.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-21 14:30:00 | 2175.10 | 2173.29 | 2173.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 2175.00 | 2173.63 | 2173.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 2175.00 | 2173.63 | 2173.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 12:15:00 | 2197.00 | 2179.50 | 2176.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 13:15:00 | 2175.70 | 2178.74 | 2176.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 13:15:00 | 2175.70 | 2178.74 | 2176.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 2175.70 | 2178.74 | 2176.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:45:00 | 2175.10 | 2178.74 | 2176.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2164.60 | 2175.91 | 2175.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 2164.60 | 2175.91 | 2175.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 2163.20 | 2173.37 | 2174.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 2146.40 | 2166.48 | 2170.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 15:15:00 | 2175.10 | 2159.74 | 2164.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 15:15:00 | 2175.10 | 2159.74 | 2164.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 2175.10 | 2159.74 | 2164.80 | EMA400 retest candle locked (from downside) |

### Cycle 123 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 2229.00 | 2173.59 | 2170.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 2245.00 | 2215.20 | 2194.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 12:15:00 | 2238.20 | 2240.23 | 2217.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-28 13:00:00 | 2238.20 | 2240.23 | 2217.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 2240.00 | 2250.12 | 2238.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 2240.00 | 2250.12 | 2238.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 2237.80 | 2247.65 | 2238.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:30:00 | 2235.00 | 2247.65 | 2238.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 2234.80 | 2245.08 | 2238.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 12:00:00 | 2234.80 | 2245.08 | 2238.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 2239.90 | 2244.05 | 2238.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 2239.90 | 2244.05 | 2238.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 124 — SELL (started 2025-10-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 13:15:00 | 2177.60 | 2230.76 | 2233.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 10:15:00 | 2115.00 | 2179.57 | 2205.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 2133.40 | 2128.89 | 2152.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 15:00:00 | 2133.40 | 2128.89 | 2152.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 2160.00 | 2135.11 | 2153.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 10:15:00 | 2129.20 | 2135.55 | 2151.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 11:15:00 | 2168.00 | 2144.75 | 2153.46 | SL hit (close>static) qty=1.00 sl=2165.20 alert=retest2 |

### Cycle 125 — BUY (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 15:15:00 | 2163.70 | 2158.04 | 2157.98 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 2148.90 | 2156.22 | 2157.15 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 2162.10 | 2158.05 | 2157.86 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 14:15:00 | 2145.90 | 2156.24 | 2157.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 2118.20 | 2146.83 | 2152.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 2142.60 | 2141.09 | 2148.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 12:15:00 | 2142.60 | 2141.09 | 2148.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 12:15:00 | 2142.60 | 2141.09 | 2148.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:00:00 | 2142.60 | 2141.09 | 2148.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 2148.70 | 2142.18 | 2147.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 2148.70 | 2142.18 | 2147.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 2149.00 | 2143.55 | 2147.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 2130.30 | 2143.55 | 2147.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 2131.60 | 2141.16 | 2146.04 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 2170.20 | 2151.20 | 2149.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 11:15:00 | 2178.40 | 2163.59 | 2156.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 2158.00 | 2168.89 | 2162.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 2158.00 | 2168.89 | 2162.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 2158.00 | 2168.89 | 2162.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:00:00 | 2158.00 | 2168.89 | 2162.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 2172.00 | 2169.51 | 2163.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 11:15:00 | 2175.80 | 2169.51 | 2163.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-28 10:15:00 | 2393.38 | 2356.91 | 2327.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 2341.00 | 2353.37 | 2354.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 2302.60 | 2340.62 | 2348.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 14:15:00 | 2311.00 | 2291.64 | 2304.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 14:15:00 | 2311.00 | 2291.64 | 2304.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 2311.00 | 2291.64 | 2304.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 2311.00 | 2291.64 | 2304.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 2285.80 | 2290.47 | 2302.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 2284.90 | 2293.34 | 2302.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 2312.90 | 2303.05 | 2304.82 | SL hit (close>static) qty=1.00 sl=2312.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 14:15:00 | 2318.80 | 2306.20 | 2306.09 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 2278.40 | 2301.25 | 2303.90 | EMA200 below EMA400 |

### Cycle 133 — BUY (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 12:15:00 | 2325.70 | 2307.51 | 2306.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-09 14:15:00 | 2368.80 | 2321.03 | 2312.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 09:15:00 | 2325.00 | 2330.46 | 2318.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 10:00:00 | 2325.00 | 2330.46 | 2318.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 2317.90 | 2326.87 | 2319.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:45:00 | 2316.10 | 2326.87 | 2319.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 12:15:00 | 2309.70 | 2323.43 | 2318.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 12:45:00 | 2308.10 | 2323.43 | 2318.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 2280.20 | 2311.44 | 2313.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 2248.10 | 2295.34 | 2305.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 2290.90 | 2271.75 | 2285.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 2290.90 | 2271.75 | 2285.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 2290.90 | 2271.75 | 2285.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 2290.90 | 2271.75 | 2285.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 2309.10 | 2279.22 | 2287.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 2309.10 | 2279.22 | 2287.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 2307.60 | 2284.90 | 2289.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:15:00 | 2316.80 | 2284.90 | 2289.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 2322.10 | 2292.34 | 2292.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 13:15:00 | 2331.90 | 2300.25 | 2295.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 2297.40 | 2305.66 | 2300.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 2297.40 | 2305.66 | 2300.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2297.40 | 2305.66 | 2300.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 2293.00 | 2305.66 | 2300.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 2298.10 | 2304.15 | 2299.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:45:00 | 2290.30 | 2304.15 | 2299.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 2298.40 | 2303.00 | 2299.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:45:00 | 2294.30 | 2303.00 | 2299.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 2317.30 | 2305.86 | 2301.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 13:15:00 | 2323.80 | 2305.86 | 2301.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 15:15:00 | 2321.30 | 2310.90 | 2304.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:30:00 | 2321.10 | 2317.11 | 2309.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 14:00:00 | 2320.70 | 2318.61 | 2311.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 2307.10 | 2316.31 | 2311.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:00:00 | 2307.10 | 2316.31 | 2311.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 2302.30 | 2313.51 | 2310.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 2284.80 | 2313.51 | 2310.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-17 09:15:00 | 2283.50 | 2307.50 | 2308.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — SELL (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 09:15:00 | 2283.50 | 2307.50 | 2308.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 10:15:00 | 2273.50 | 2300.70 | 2305.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 14:15:00 | 2251.50 | 2249.54 | 2266.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 15:00:00 | 2251.50 | 2249.54 | 2266.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 2259.30 | 2253.77 | 2264.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 2247.90 | 2253.77 | 2264.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 14:00:00 | 2245.70 | 2253.21 | 2261.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 2343.50 | 2271.27 | 2268.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 2343.50 | 2271.27 | 2268.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 15:15:00 | 2425.00 | 2302.01 | 2283.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 2403.10 | 2404.26 | 2379.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 15:00:00 | 2403.10 | 2404.26 | 2379.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 2381.90 | 2399.02 | 2383.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 11:00:00 | 2381.90 | 2399.02 | 2383.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 2383.00 | 2395.82 | 2383.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 2383.00 | 2395.82 | 2383.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 2380.50 | 2392.76 | 2383.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 2380.50 | 2392.76 | 2383.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 2382.50 | 2390.70 | 2383.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:15:00 | 2362.60 | 2390.70 | 2383.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 2350.10 | 2382.58 | 2380.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 2371.40 | 2382.58 | 2380.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 2368.00 | 2379.67 | 2379.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 09:15:00 | 2398.60 | 2379.67 | 2379.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 2372.50 | 2378.15 | 2378.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 2372.50 | 2378.15 | 2378.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 2362.70 | 2375.06 | 2377.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 2311.20 | 2287.82 | 2320.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 2311.20 | 2287.82 | 2320.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2311.20 | 2287.82 | 2320.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:30:00 | 2315.10 | 2287.82 | 2320.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 2337.40 | 2297.73 | 2321.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 2337.40 | 2297.73 | 2321.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 2304.90 | 2299.17 | 2320.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 12:45:00 | 2294.60 | 2297.55 | 2317.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 2262.00 | 2291.84 | 2311.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 14:15:00 | 2298.70 | 2285.39 | 2284.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 14:15:00 | 2298.70 | 2285.39 | 2284.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 14:15:00 | 2310.10 | 2294.63 | 2289.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 2307.30 | 2312.27 | 2304.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 2307.30 | 2312.27 | 2304.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 2307.30 | 2312.27 | 2304.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:30:00 | 2314.90 | 2312.27 | 2304.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 2307.10 | 2311.24 | 2304.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:00:00 | 2307.10 | 2311.24 | 2304.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 2288.30 | 2306.65 | 2302.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 11:45:00 | 2296.50 | 2306.65 | 2302.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 12:15:00 | 2278.50 | 2301.02 | 2300.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 13:15:00 | 2275.00 | 2301.02 | 2300.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 13:15:00 | 2273.40 | 2295.50 | 2298.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 14:15:00 | 2267.20 | 2289.84 | 2295.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 10:15:00 | 2298.50 | 2286.62 | 2292.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 10:15:00 | 2298.50 | 2286.62 | 2292.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 2298.50 | 2286.62 | 2292.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 2298.50 | 2286.62 | 2292.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 2312.70 | 2291.84 | 2293.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:45:00 | 2312.10 | 2291.84 | 2293.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-09 12:15:00 | 2310.00 | 2295.47 | 2295.43 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-01-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 09:15:00 | 2265.00 | 2295.96 | 2296.47 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-01-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 10:15:00 | 2318.20 | 2295.54 | 2293.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 11:15:00 | 2326.80 | 2301.79 | 2296.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 2321.30 | 2327.96 | 2313.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 10:00:00 | 2321.30 | 2327.96 | 2313.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 2323.80 | 2326.38 | 2315.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:30:00 | 2318.00 | 2326.38 | 2315.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 2320.00 | 2324.63 | 2316.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:30:00 | 2320.40 | 2324.63 | 2316.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 2320.70 | 2323.84 | 2316.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 2320.70 | 2323.84 | 2316.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2322.70 | 2323.00 | 2317.53 | EMA400 retest candle locked (from upside) |

### Cycle 144 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 2298.10 | 2315.13 | 2316.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 2275.50 | 2298.53 | 2307.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 13:15:00 | 2303.60 | 2284.60 | 2293.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 13:15:00 | 2303.60 | 2284.60 | 2293.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 2303.60 | 2284.60 | 2293.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 15:00:00 | 2269.90 | 2281.66 | 2291.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:30:00 | 2274.40 | 2262.65 | 2270.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:45:00 | 2267.10 | 2263.05 | 2268.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 14:00:00 | 2276.00 | 2265.64 | 2269.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2282.00 | 2268.91 | 2270.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 2282.00 | 2268.91 | 2270.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 2284.90 | 2272.11 | 2272.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 2284.90 | 2272.11 | 2272.00 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 09:15:00 | 2256.40 | 2268.97 | 2270.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 2252.10 | 2264.39 | 2267.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 2266.90 | 2259.86 | 2264.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 2266.90 | 2259.86 | 2264.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 2266.90 | 2259.86 | 2264.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 13:45:00 | 2231.50 | 2256.86 | 2261.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 11:15:00 | 2230.50 | 2251.42 | 2257.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 12:15:00 | 2270.90 | 2256.99 | 2255.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 2270.90 | 2256.99 | 2255.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 2287.30 | 2263.05 | 2258.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 14:15:00 | 2261.50 | 2262.74 | 2259.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 14:15:00 | 2261.50 | 2262.74 | 2259.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 2261.50 | 2262.74 | 2259.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 15:00:00 | 2261.50 | 2262.74 | 2259.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 2216.00 | 2253.75 | 2255.61 | EMA200 below EMA400 |

### Cycle 149 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 2310.10 | 2260.16 | 2256.26 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 2237.60 | 2257.50 | 2257.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 2222.20 | 2250.44 | 2254.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 15:15:00 | 2235.00 | 2215.63 | 2229.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 15:15:00 | 2235.00 | 2215.63 | 2229.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 2235.00 | 2215.63 | 2229.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 2227.70 | 2215.63 | 2229.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 2217.30 | 2215.97 | 2228.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 10:15:00 | 2211.30 | 2215.97 | 2228.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 12:15:00 | 2259.70 | 2234.83 | 2234.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 2259.70 | 2234.83 | 2234.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 2269.40 | 2241.74 | 2237.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 09:15:00 | 2247.50 | 2255.19 | 2246.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 09:15:00 | 2247.50 | 2255.19 | 2246.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 2247.50 | 2255.19 | 2246.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 09:30:00 | 2248.70 | 2255.19 | 2246.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 2244.50 | 2253.05 | 2245.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:45:00 | 2237.30 | 2253.05 | 2245.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 2252.40 | 2252.92 | 2246.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 2253.60 | 2252.92 | 2246.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 2270.40 | 2256.45 | 2249.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 2260.90 | 2256.45 | 2249.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 2259.70 | 2267.80 | 2259.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:45:00 | 2259.40 | 2267.80 | 2259.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 2257.50 | 2265.74 | 2259.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 2257.50 | 2265.74 | 2259.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 2251.40 | 2262.87 | 2258.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 15:00:00 | 2251.40 | 2262.87 | 2258.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 2252.80 | 2260.86 | 2257.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 2245.90 | 2257.53 | 2256.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 2268.70 | 2259.76 | 2257.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:30:00 | 2248.40 | 2259.76 | 2257.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 2259.60 | 2262.12 | 2259.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:45:00 | 2252.10 | 2262.12 | 2259.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 2272.00 | 2264.10 | 2260.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 2279.20 | 2261.68 | 2259.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 11:15:00 | 2274.70 | 2261.71 | 2260.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 15:15:00 | 2287.00 | 2275.53 | 2268.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 13:15:00 | 2260.00 | 2266.32 | 2266.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 2260.00 | 2266.32 | 2266.61 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 15:15:00 | 2275.00 | 2268.34 | 2267.49 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 2255.60 | 2265.79 | 2266.41 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 12:15:00 | 2271.60 | 2267.69 | 2267.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 13:15:00 | 2278.20 | 2269.79 | 2268.16 | Break + close above crossover candle high |

### Cycle 156 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 2241.10 | 2265.06 | 2266.52 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 12:15:00 | 2292.60 | 2269.00 | 2266.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 15:15:00 | 2296.00 | 2278.02 | 2273.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 11:15:00 | 2358.00 | 2358.68 | 2330.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 12:00:00 | 2358.00 | 2358.68 | 2330.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 2322.50 | 2349.90 | 2331.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:00:00 | 2322.50 | 2349.90 | 2331.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2359.10 | 2351.74 | 2333.99 | EMA400 retest candle locked (from upside) |

### Cycle 158 — SELL (started 2026-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 13:15:00 | 2309.20 | 2324.92 | 2326.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 10:15:00 | 2290.50 | 2313.49 | 2320.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 13:15:00 | 2292.00 | 2285.60 | 2297.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-23 14:00:00 | 2292.00 | 2285.60 | 2297.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 2290.00 | 2286.48 | 2296.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 14:45:00 | 2283.90 | 2286.48 | 2296.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 2290.70 | 2287.32 | 2296.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 2273.40 | 2287.32 | 2296.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 11:45:00 | 2282.50 | 2285.19 | 2292.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 12:15:00 | 2298.00 | 2285.95 | 2288.33 | SL hit (close>static) qty=1.00 sl=2296.60 alert=retest2 |

### Cycle 159 — BUY (started 2026-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 14:15:00 | 2309.10 | 2290.67 | 2290.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 14:15:00 | 2327.70 | 2309.30 | 2301.04 | Break + close above crossover candle high |

### Cycle 160 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 2229.30 | 2297.09 | 2297.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 2216.10 | 2259.76 | 2276.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 12:15:00 | 2044.90 | 2040.21 | 2079.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 13:00:00 | 2044.90 | 2040.21 | 2079.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 2008.10 | 2004.41 | 2029.13 | EMA400 retest candle locked (from downside) |

### Cycle 161 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 2096.90 | 2049.17 | 2044.40 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 2029.50 | 2045.45 | 2045.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 15:15:00 | 2014.70 | 2034.72 | 2040.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 11:15:00 | 2025.00 | 2024.80 | 2033.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:00:00 | 2025.00 | 2024.80 | 2033.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 1997.50 | 2013.00 | 2023.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 10:30:00 | 1987.70 | 2007.60 | 2020.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 11:00:00 | 1986.00 | 2007.60 | 2020.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:00:00 | 1969.60 | 1999.77 | 2010.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 10:15:00 | 2037.30 | 1994.10 | 1997.30 | SL hit (close>static) qty=1.00 sl=2037.00 alert=retest2 |

### Cycle 163 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 2037.50 | 2002.78 | 2000.96 | EMA200 above EMA400 |

### Cycle 164 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1983.90 | 2011.95 | 2013.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1976.70 | 1995.87 | 2004.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 1866.90 | 1864.03 | 1905.38 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-24 10:30:00 | 1852.40 | 1862.16 | 1900.77 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1910.30 | 1874.17 | 1899.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 1910.30 | 1874.17 | 1899.67 | SL hit (close>ema400) qty=1.00 sl=1899.67 alert=retest1 |

### Cycle 165 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 1994.20 | 1924.74 | 1917.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 2014.20 | 1953.87 | 1933.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 1971.70 | 1985.95 | 1960.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 1971.70 | 1985.95 | 1960.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 1950.90 | 1978.94 | 1959.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 1950.90 | 1978.94 | 1959.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 1954.30 | 1974.01 | 1958.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:15:00 | 1960.00 | 1974.01 | 1958.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:45:00 | 1958.50 | 1970.01 | 1958.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 13:15:00 | 1959.80 | 1970.01 | 1958.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 14:15:00 | 1927.50 | 1962.43 | 1956.95 | SL hit (close<static) qty=1.00 sl=1939.00 alert=retest2 |

### Cycle 166 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 1887.60 | 1942.21 | 1948.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-02 09:15:00 | 1837.90 | 1910.59 | 1923.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 13:15:00 | 1890.00 | 1873.55 | 1888.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 13:15:00 | 1890.00 | 1873.55 | 1888.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 1890.00 | 1873.55 | 1888.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:00:00 | 1890.00 | 1873.55 | 1888.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 1900.00 | 1878.84 | 1889.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:45:00 | 1898.60 | 1878.84 | 1889.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 1884.90 | 1880.05 | 1889.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 1864.60 | 1880.30 | 1888.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 11:15:00 | 1947.50 | 1896.70 | 1894.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — BUY (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 11:15:00 | 1947.50 | 1896.70 | 1894.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 12:15:00 | 1982.30 | 1913.82 | 1902.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 2130.00 | 2144.82 | 2117.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 2130.00 | 2144.82 | 2117.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2130.00 | 2144.82 | 2117.22 | EMA400 retest candle locked (from upside) |

### Cycle 168 — SELL (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 12:15:00 | 2100.80 | 2111.21 | 2111.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 14:15:00 | 2087.40 | 2103.02 | 2107.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 10:15:00 | 2062.40 | 2053.98 | 2067.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 10:15:00 | 2062.40 | 2053.98 | 2067.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 10:15:00 | 2062.40 | 2053.98 | 2067.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 10:45:00 | 2064.00 | 2053.98 | 2067.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 11:15:00 | 2070.40 | 2057.26 | 2067.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 11:45:00 | 2070.60 | 2057.26 | 2067.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 2073.00 | 2060.41 | 2068.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 13:00:00 | 2073.00 | 2060.41 | 2068.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 169 — BUY (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 10:15:00 | 2080.90 | 2071.93 | 2071.37 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 2059.80 | 2069.50 | 2070.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 14:15:00 | 2056.50 | 2067.57 | 2069.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 15:15:00 | 2042.00 | 2041.30 | 2051.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-23 09:15:00 | 2028.10 | 2041.30 | 2051.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2048.80 | 2042.80 | 2051.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:45:00 | 2055.90 | 2042.80 | 2051.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 2026.20 | 2039.48 | 2049.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 11:15:00 | 2024.80 | 2039.48 | 2049.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:45:00 | 2024.80 | 2035.12 | 2042.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 2022.80 | 2023.08 | 2025.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:30:00 | 2024.70 | 2024.25 | 2025.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 2030.00 | 2025.40 | 2025.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:00:00 | 2006.90 | 2021.70 | 2024.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:00:00 | 2013.10 | 1993.16 | 1999.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 15:15:00 | 2012.90 | 1999.24 | 1998.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 2012.90 | 1999.24 | 1998.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 2015.60 | 2004.01 | 2001.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 1997.50 | 2006.61 | 2003.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 1997.50 | 2006.61 | 2003.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1997.50 | 2006.61 | 2003.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:45:00 | 1998.00 | 2006.61 | 2003.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 2015.40 | 2008.37 | 2004.44 | EMA400 retest candle locked (from upside) |

### Cycle 172 — SELL (started 2026-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 13:15:00 | 1961.20 | 2000.59 | 2002.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 1922.00 | 1974.90 | 1989.06 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-23 10:30:00 | 1227.60 | 2024-05-24 10:15:00 | 1264.95 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2024-05-23 11:15:00 | 1229.00 | 2024-05-24 10:15:00 | 1264.95 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2024-05-23 13:00:00 | 1230.00 | 2024-05-24 10:15:00 | 1264.95 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2024-05-30 12:30:00 | 1287.85 | 2024-06-10 10:15:00 | 1416.63 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-30 14:30:00 | 1296.80 | 2024-06-10 11:15:00 | 1426.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 12:30:00 | 1300.00 | 2024-06-10 11:15:00 | 1430.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-26 12:15:00 | 1531.65 | 2024-06-26 13:15:00 | 1562.50 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-07-05 10:45:00 | 1588.45 | 2024-07-10 10:15:00 | 1587.00 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2024-07-05 11:15:00 | 1589.50 | 2024-07-10 10:15:00 | 1587.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-07-05 14:00:00 | 1591.55 | 2024-07-10 10:15:00 | 1587.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2024-07-19 10:00:00 | 1569.70 | 2024-07-19 13:15:00 | 1600.55 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2024-07-19 11:45:00 | 1576.25 | 2024-07-19 13:15:00 | 1600.55 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-07-29 09:15:00 | 1657.50 | 2024-08-01 12:15:00 | 1640.60 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-07-29 11:00:00 | 1652.60 | 2024-08-01 12:15:00 | 1640.60 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-08-06 12:45:00 | 1625.95 | 2024-08-07 15:15:00 | 1642.65 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-08-06 13:15:00 | 1632.50 | 2024-08-07 15:15:00 | 1642.65 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-08-07 12:30:00 | 1630.70 | 2024-08-07 15:15:00 | 1642.65 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2024-08-07 13:15:00 | 1631.90 | 2024-08-07 15:15:00 | 1642.65 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1660.95 | 2024-08-14 12:15:00 | 1689.30 | STOP_HIT | 1.00 | 1.71% |
| BUY | retest2 | 2024-08-19 12:15:00 | 1754.90 | 2024-08-20 10:15:00 | 1719.45 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-08-19 13:30:00 | 1754.90 | 2024-08-20 10:15:00 | 1719.45 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-08-20 14:45:00 | 1754.20 | 2024-08-23 15:15:00 | 1750.15 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-08-21 09:45:00 | 1756.80 | 2024-08-23 15:15:00 | 1750.15 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-09-03 09:15:00 | 1755.75 | 2024-09-03 14:15:00 | 1730.45 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-09-10 11:15:00 | 1709.70 | 2024-09-10 15:15:00 | 1710.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2024-09-13 11:15:00 | 1688.05 | 2024-09-16 15:15:00 | 1711.95 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2024-09-13 13:15:00 | 1689.85 | 2024-09-16 15:15:00 | 1711.95 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-09-13 15:15:00 | 1689.70 | 2024-09-16 15:15:00 | 1711.95 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-09-16 12:00:00 | 1689.05 | 2024-09-16 15:15:00 | 1711.95 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-09-18 12:30:00 | 1713.80 | 2024-09-19 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-09-18 14:15:00 | 1710.85 | 2024-09-19 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-09-24 11:45:00 | 1663.30 | 2024-09-27 09:15:00 | 1679.55 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-09-24 15:00:00 | 1663.45 | 2024-09-27 09:15:00 | 1679.55 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-09-25 09:15:00 | 1655.50 | 2024-09-27 09:15:00 | 1679.55 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-10-21 09:15:00 | 1605.00 | 2024-10-21 09:15:00 | 1618.55 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-11-05 14:15:00 | 1672.75 | 2024-11-12 12:15:00 | 1733.10 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2024-11-05 15:15:00 | 1673.95 | 2024-11-12 12:15:00 | 1733.10 | STOP_HIT | 1.00 | 3.53% |
| SELL | retest2 | 2024-11-14 15:15:00 | 1708.10 | 2024-11-18 10:15:00 | 1727.80 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-11-19 09:15:00 | 1734.80 | 2024-11-26 13:15:00 | 1764.60 | STOP_HIT | 1.00 | 1.72% |
| SELL | retest2 | 2024-12-06 10:45:00 | 1765.50 | 2024-12-10 09:15:00 | 1776.20 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-12-06 14:45:00 | 1766.30 | 2024-12-10 09:15:00 | 1776.20 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-12-09 13:45:00 | 1762.30 | 2024-12-10 09:15:00 | 1776.20 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-12-09 15:15:00 | 1762.05 | 2024-12-10 09:15:00 | 1776.20 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-12-11 09:30:00 | 1789.70 | 2024-12-11 14:15:00 | 1768.60 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-12-12 11:15:00 | 1787.90 | 2024-12-13 09:15:00 | 1750.80 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-12-12 12:30:00 | 1787.05 | 2024-12-13 09:15:00 | 1750.80 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-12-18 09:15:00 | 1866.15 | 2024-12-26 11:15:00 | 1827.60 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2024-12-18 11:45:00 | 1821.00 | 2024-12-26 11:15:00 | 1827.60 | STOP_HIT | 1.00 | 0.36% |
| BUY | retest2 | 2024-12-18 12:45:00 | 1830.50 | 2024-12-26 11:15:00 | 1827.60 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2024-12-27 11:45:00 | 1845.05 | 2024-12-30 14:15:00 | 1871.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-12-27 14:45:00 | 1841.40 | 2024-12-30 14:15:00 | 1871.10 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2024-12-30 09:15:00 | 1831.20 | 2024-12-30 14:15:00 | 1871.10 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-12-30 13:30:00 | 1842.65 | 2024-12-30 14:15:00 | 1871.10 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-01-07 09:15:00 | 1956.45 | 2025-01-08 09:15:00 | 1925.80 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-01-07 12:30:00 | 1945.85 | 2025-01-08 09:15:00 | 1925.80 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-01-07 13:15:00 | 1943.70 | 2025-01-08 09:15:00 | 1925.80 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-02-03 09:15:00 | 1826.75 | 2025-02-07 14:15:00 | 1858.65 | STOP_HIT | 1.00 | 1.75% |
| BUY | retest2 | 2025-02-03 10:15:00 | 1816.70 | 2025-02-07 14:15:00 | 1858.65 | STOP_HIT | 1.00 | 2.31% |
| BUY | retest2 | 2025-02-03 11:00:00 | 1818.00 | 2025-02-07 14:15:00 | 1858.65 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2025-02-03 11:30:00 | 1817.00 | 2025-02-07 14:15:00 | 1858.65 | STOP_HIT | 1.00 | 2.29% |
| BUY | retest2 | 2025-02-04 09:15:00 | 1834.45 | 2025-02-07 14:15:00 | 1858.65 | STOP_HIT | 1.00 | 1.32% |
| SELL | retest2 | 2025-02-19 15:15:00 | 1708.95 | 2025-02-20 09:15:00 | 1737.10 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-02-24 10:15:00 | 1695.85 | 2025-02-25 11:15:00 | 1611.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-24 10:15:00 | 1695.85 | 2025-02-25 13:15:00 | 1663.95 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2025-02-27 09:30:00 | 1676.50 | 2025-03-03 15:15:00 | 1660.00 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest2 | 2025-03-24 09:15:00 | 1996.35 | 2025-03-27 12:15:00 | 1958.80 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-03-24 14:45:00 | 1993.00 | 2025-03-27 12:15:00 | 1958.80 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-03-25 09:30:00 | 1985.55 | 2025-03-27 12:15:00 | 1958.80 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-03-25 13:30:00 | 1986.35 | 2025-03-27 12:15:00 | 1958.80 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-03-26 09:15:00 | 2000.40 | 2025-03-27 12:15:00 | 1958.80 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-03-26 15:15:00 | 1993.15 | 2025-03-27 12:15:00 | 1958.80 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-03-27 09:30:00 | 1984.50 | 2025-03-27 12:15:00 | 1958.80 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-03-27 12:00:00 | 1986.90 | 2025-03-27 12:15:00 | 1958.80 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-04-17 10:30:00 | 2169.60 | 2025-04-30 15:15:00 | 2210.00 | STOP_HIT | 1.00 | 1.86% |
| BUY | retest2 | 2025-04-21 09:15:00 | 2196.50 | 2025-04-30 15:15:00 | 2210.00 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2025-05-09 11:15:00 | 2321.40 | 2025-05-14 12:15:00 | 2340.30 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-05-09 11:45:00 | 2319.30 | 2025-05-14 12:15:00 | 2340.30 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2025-05-16 15:00:00 | 2449.00 | 2025-05-20 11:15:00 | 2370.80 | STOP_HIT | 1.00 | -3.19% |
| BUY | retest2 | 2025-05-19 10:00:00 | 2449.20 | 2025-05-20 11:15:00 | 2370.80 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2025-05-19 10:45:00 | 2449.40 | 2025-05-20 11:15:00 | 2370.80 | STOP_HIT | 1.00 | -3.21% |
| BUY | retest2 | 2025-05-19 14:45:00 | 2458.00 | 2025-05-20 11:15:00 | 2370.80 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-07-03 11:30:00 | 2291.30 | 2025-07-09 11:15:00 | 2270.00 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-07-16 09:15:00 | 2397.30 | 2025-07-17 14:15:00 | 2341.70 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-07-24 11:00:00 | 2325.40 | 2025-07-24 13:15:00 | 2419.60 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2025-07-24 12:45:00 | 2331.00 | 2025-07-24 13:15:00 | 2419.60 | STOP_HIT | 1.00 | -3.80% |
| BUY | retest2 | 2025-08-04 09:15:00 | 2628.20 | 2025-08-04 12:15:00 | 2573.30 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-08-11 11:30:00 | 2431.60 | 2025-08-13 09:15:00 | 2310.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-12 09:30:00 | 2430.70 | 2025-08-13 09:15:00 | 2309.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-11 11:30:00 | 2431.60 | 2025-08-14 09:15:00 | 2312.60 | STOP_HIT | 0.50 | 4.89% |
| SELL | retest2 | 2025-08-12 09:30:00 | 2430.70 | 2025-08-14 09:15:00 | 2312.60 | STOP_HIT | 0.50 | 4.86% |
| SELL | retest2 | 2025-09-10 11:30:00 | 2223.80 | 2025-09-12 09:15:00 | 2246.20 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-09-11 10:45:00 | 2222.50 | 2025-09-12 09:15:00 | 2246.20 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-09-18 14:30:00 | 2312.20 | 2025-09-19 09:15:00 | 2272.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-09-18 15:15:00 | 2313.50 | 2025-09-19 09:15:00 | 2272.50 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-09-29 12:30:00 | 2161.70 | 2025-09-29 15:15:00 | 2279.00 | STOP_HIT | 1.00 | -5.43% |
| SELL | retest2 | 2025-10-16 11:15:00 | 2175.10 | 2025-10-23 09:15:00 | 2175.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-10-17 10:15:00 | 2172.70 | 2025-10-23 09:15:00 | 2175.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-10-17 12:45:00 | 2164.80 | 2025-10-23 09:15:00 | 2175.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-10-21 14:30:00 | 2175.10 | 2025-10-23 09:15:00 | 2175.00 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-11-04 10:15:00 | 2129.20 | 2025-11-04 11:15:00 | 2168.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-12 11:15:00 | 2175.80 | 2025-11-28 10:15:00 | 2393.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-12-08 09:45:00 | 2284.90 | 2025-12-08 13:15:00 | 2312.90 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-15 13:15:00 | 2323.80 | 2025-12-17 09:15:00 | 2283.50 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-12-15 15:15:00 | 2321.30 | 2025-12-17 09:15:00 | 2283.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-12-16 11:30:00 | 2321.10 | 2025-12-17 09:15:00 | 2283.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-12-16 14:00:00 | 2320.70 | 2025-12-17 09:15:00 | 2283.50 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-12-19 11:15:00 | 2247.90 | 2025-12-19 14:15:00 | 2343.50 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2025-12-19 14:00:00 | 2245.70 | 2025-12-19 14:15:00 | 2343.50 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2025-12-29 09:15:00 | 2398.60 | 2025-12-29 13:15:00 | 2372.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-31 12:45:00 | 2294.60 | 2026-01-05 14:15:00 | 2298.70 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2025-12-31 15:00:00 | 2262.00 | 2026-01-05 14:15:00 | 2298.70 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-01-20 15:00:00 | 2269.90 | 2026-01-22 15:15:00 | 2284.90 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2026-01-22 10:30:00 | 2274.40 | 2026-01-22 15:15:00 | 2284.90 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-01-22 12:45:00 | 2267.10 | 2026-01-22 15:15:00 | 2284.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-01-22 14:00:00 | 2276.00 | 2026-01-22 15:15:00 | 2284.90 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2026-01-27 13:45:00 | 2231.50 | 2026-01-29 12:15:00 | 2270.90 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-01-28 11:15:00 | 2230.50 | 2026-01-29 12:15:00 | 2270.90 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2026-02-03 10:15:00 | 2211.30 | 2026-02-03 12:15:00 | 2259.70 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2026-02-09 09:15:00 | 2279.20 | 2026-02-10 13:15:00 | 2260.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2026-02-09 11:15:00 | 2274.70 | 2026-02-10 13:15:00 | 2260.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-02-09 15:15:00 | 2287.00 | 2026-02-10 13:15:00 | 2260.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-02-24 09:15:00 | 2273.40 | 2026-02-25 12:15:00 | 2298.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-02-24 11:45:00 | 2282.50 | 2026-02-25 12:15:00 | 2298.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-02-25 13:30:00 | 2282.10 | 2026-02-25 14:15:00 | 2309.10 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-03-13 10:30:00 | 1987.70 | 2026-03-17 10:15:00 | 2037.30 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-03-13 11:00:00 | 1986.00 | 2026-03-17 10:15:00 | 2037.30 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-03-16 10:00:00 | 1969.60 | 2026-03-17 10:15:00 | 2037.30 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest1 | 2026-03-24 10:30:00 | 1852.40 | 2026-03-24 12:15:00 | 1910.30 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2026-03-27 12:15:00 | 1960.00 | 2026-03-27 14:15:00 | 1927.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2026-03-27 12:45:00 | 1958.50 | 2026-03-27 14:15:00 | 1927.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-03-27 13:15:00 | 1959.80 | 2026-03-27 14:15:00 | 1927.50 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-04-07 09:30:00 | 1864.60 | 2026-04-07 11:15:00 | 1947.50 | STOP_HIT | 1.00 | -4.45% |
| SELL | retest2 | 2026-04-23 11:15:00 | 2024.80 | 2026-05-05 15:15:00 | 2012.90 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2026-04-24 09:45:00 | 2024.80 | 2026-05-05 15:15:00 | 2012.90 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2026-04-27 14:30:00 | 2022.80 | 2026-05-05 15:15:00 | 2012.90 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2026-04-28 09:30:00 | 2024.70 | 2026-05-05 15:15:00 | 2012.90 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2026-04-28 12:00:00 | 2006.90 | 2026-05-05 15:15:00 | 2012.90 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2026-05-04 10:00:00 | 2013.10 | 2026-05-05 15:15:00 | 2012.90 | STOP_HIT | 1.00 | 0.01% |
