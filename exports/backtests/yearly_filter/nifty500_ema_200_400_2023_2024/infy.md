# Infosys Ltd. (INFY)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1179.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 10 |
| ALERT1 | 9 |
| ALERT2 | 10 |
| ALERT2_SKIP | 2 |
| ALERT3 | 55 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 50 |
| PARTIAL | 3 |
| TARGET_HIT | 7 |
| STOP_HIT | 47 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 57 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 46
- **Target hits / Stop hits / Partials:** 7 / 47 / 3
- **Avg / median % per leg:** -0.01% / -0.89%
- **Sum % (uncompounded):** -0.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 5 | 22.7% | 5 | 17 | 0 | 0.52% | 11.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 5 | 22.7% | 5 | 17 | 0 | 0.52% | 11.3% |
| SELL (all) | 35 | 6 | 17.1% | 2 | 30 | 3 | -0.35% | -12.1% |
| SELL @ 2nd Alert (retest1) | 6 | 4 | 66.7% | 2 | 2 | 2 | 2.55% | 15.3% |
| SELL @ 3rd Alert (retest2) | 29 | 2 | 6.9% | 0 | 28 | 1 | -0.95% | -27.4% |
| retest1 (combined) | 6 | 4 | 66.7% | 2 | 2 | 2 | 2.55% | 15.3% |
| retest2 (combined) | 51 | 7 | 13.7% | 5 | 45 | 1 | -0.31% | -16.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 11:15:00 | 1451.85 | 1326.28 | 1326.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 13:15:00 | 1472.50 | 1336.08 | 1331.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 09:15:00 | 1342.45 | 1354.48 | 1341.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 09:15:00 | 1342.45 | 1354.48 | 1341.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 09:15:00 | 1342.45 | 1354.48 | 1341.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 09:15:00 | 1359.30 | 1350.73 | 1340.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-27 15:00:00 | 1353.70 | 1351.06 | 1341.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-31 11:15:00 | 1352.25 | 1350.40 | 1341.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 14:15:00 | 1350.45 | 1351.79 | 1342.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-12 09:15:00 | 1487.48 | 1422.84 | 1394.53 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 15:15:00 | 1354.00 | 1423.18 | 1423.25 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-11-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-23 15:15:00 | 1450.75 | 1419.11 | 1419.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 09:15:00 | 1459.00 | 1421.87 | 1420.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-13 10:15:00 | 1439.35 | 1445.18 | 1434.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-13 11:00:00 | 1439.35 | 1445.18 | 1434.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 11:15:00 | 1435.85 | 1445.09 | 1434.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-13 11:30:00 | 1437.20 | 1445.09 | 1434.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 12:15:00 | 1440.00 | 1445.04 | 1434.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 13:30:00 | 1440.85 | 1445.02 | 1434.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-15 15:15:00 | 1584.93 | 1456.29 | 1441.00 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-03-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 15:15:00 | 1487.15 | 1600.15 | 1600.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-03 14:15:00 | 1481.55 | 1574.85 | 1586.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 11:15:00 | 1453.65 | 1452.00 | 1488.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 11:30:00 | 1455.40 | 1452.00 | 1488.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 14:15:00 | 1471.20 | 1445.18 | 1473.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 15:00:00 | 1471.20 | 1445.18 | 1473.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 1473.00 | 1445.46 | 1473.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 1496.45 | 1445.46 | 1473.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 1521.80 | 1446.22 | 1473.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 10:00:00 | 1521.80 | 1446.22 | 1473.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 1564.90 | 1489.39 | 1489.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 12:15:00 | 1567.95 | 1490.17 | 1489.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 11:15:00 | 1733.60 | 1734.29 | 1651.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 11:30:00 | 1721.25 | 1734.29 | 1651.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1881.25 | 1918.25 | 1870.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 14:15:00 | 1890.35 | 1851.02 | 1849.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 13:15:00 | 1888.05 | 1853.52 | 1851.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 13:45:00 | 1887.30 | 1853.87 | 1851.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 12:15:00 | 1861.00 | 1864.42 | 1857.27 | SL hit (close<static) qty=1.00 sl=1869.25 alert=retest2 |

### Cycle 6 — SELL (started 2025-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 15:15:00 | 1823.00 | 1894.80 | 1895.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 13:15:00 | 1817.65 | 1873.62 | 1882.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 1519.10 | 1516.92 | 1605.62 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-02 12:15:00 | 1497.70 | 1516.66 | 1604.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-09 09:45:00 | 1497.20 | 1514.27 | 1589.94 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 12:15:00 | 1607.40 | 1516.35 | 1587.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-12 12:15:00 | 1607.40 | 1516.35 | 1587.29 | SL hit (close>ema400) qty=1.00 sl=1587.29 alert=retest1 |

### Cycle 7 — BUY (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 11:15:00 | 1605.50 | 1591.48 | 1591.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 09:15:00 | 1641.90 | 1592.54 | 1592.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 10:15:00 | 1594.20 | 1605.23 | 1599.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 10:15:00 | 1594.20 | 1605.23 | 1599.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 1594.20 | 1605.23 | 1599.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 1594.20 | 1605.23 | 1599.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1590.80 | 1605.08 | 1599.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:30:00 | 1591.10 | 1605.08 | 1599.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 1593.80 | 1604.77 | 1599.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:30:00 | 1594.50 | 1604.77 | 1599.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 1590.70 | 1601.84 | 1597.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 1590.70 | 1601.84 | 1597.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 1587.10 | 1601.70 | 1597.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:45:00 | 1586.20 | 1601.70 | 1597.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 1591.30 | 1601.04 | 1597.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 10:30:00 | 1591.90 | 1601.04 | 1597.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1596.00 | 1601.24 | 1597.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 1596.00 | 1601.24 | 1597.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1595.00 | 1601.18 | 1597.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:00:00 | 1595.00 | 1601.18 | 1597.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1588.50 | 1601.01 | 1597.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 13:00:00 | 1588.50 | 1601.01 | 1597.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1588.20 | 1600.26 | 1597.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:45:00 | 1580.20 | 1600.26 | 1597.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 1560.80 | 1594.94 | 1595.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 1552.30 | 1594.52 | 1594.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 14:15:00 | 1496.40 | 1495.73 | 1531.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:45:00 | 1497.30 | 1495.73 | 1531.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1535.40 | 1496.20 | 1529.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 1532.40 | 1496.20 | 1529.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1534.90 | 1496.58 | 1529.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:00:00 | 1534.90 | 1496.58 | 1529.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 1533.30 | 1498.07 | 1529.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 1532.20 | 1498.07 | 1529.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 1531.70 | 1498.72 | 1529.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:15:00 | 1534.20 | 1498.72 | 1529.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 1524.60 | 1498.98 | 1529.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:30:00 | 1533.20 | 1498.98 | 1529.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 1527.80 | 1499.75 | 1529.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 1527.80 | 1499.75 | 1529.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1525.80 | 1500.00 | 1529.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 1513.20 | 1500.00 | 1529.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-05 10:15:00 | 1437.54 | 1494.33 | 1520.45 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-09 10:15:00 | 1496.70 | 1488.08 | 1515.43 | SL hit (close>ema200) qty=0.50 sl=1488.08 alert=retest2 |

### Cycle 9 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 1540.30 | 1500.97 | 1500.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 09:15:00 | 1571.80 | 1506.56 | 1503.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1591.10 | 1606.85 | 1574.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 10:00:00 | 1591.10 | 1606.85 | 1574.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1616.70 | 1634.65 | 1606.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 1605.90 | 1634.65 | 1606.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1550.60 | 1635.99 | 1609.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:00:00 | 1550.60 | 1635.99 | 1609.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 1405.10 | 1587.88 | 1588.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 11:15:00 | 1399.50 | 1586.00 | 1587.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 09:15:00 | 1313.10 | 1310.98 | 1382.90 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-10 09:30:00 | 1288.70 | 1314.93 | 1375.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:15:00 | 1278.60 | 1311.56 | 1360.61 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1224.26 | 1303.24 | 1352.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 09:15:00 | 1214.67 | 1303.24 | 1352.77 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2026-04-24 13:15:00 | 1159.83 | 1297.91 | 1349.10 | Target hit (10%) qty=0.50 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-06-30 11:45:00 | 1324.80 | 2023-06-30 12:15:00 | 1331.10 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-06-30 12:15:00 | 1325.05 | 2023-06-30 12:15:00 | 1331.10 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2023-07-10 09:30:00 | 1324.20 | 2023-07-10 11:15:00 | 1334.10 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2023-07-27 09:15:00 | 1359.30 | 2023-09-12 09:15:00 | 1487.48 | TARGET_HIT | 1.00 | 9.43% |
| BUY | retest2 | 2023-07-27 15:00:00 | 1353.70 | 2023-09-12 09:15:00 | 1485.50 | TARGET_HIT | 1.00 | 9.74% |
| BUY | retest2 | 2023-07-31 11:15:00 | 1352.25 | 2023-09-12 11:15:00 | 1495.23 | TARGET_HIT | 1.00 | 10.57% |
| BUY | retest2 | 2023-08-02 14:15:00 | 1350.45 | 2023-09-12 11:15:00 | 1489.07 | TARGET_HIT | 1.00 | 10.26% |
| BUY | retest2 | 2023-09-29 11:30:00 | 1432.30 | 2023-10-13 09:15:00 | 1423.05 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-10-03 11:15:00 | 1426.15 | 2023-10-23 10:15:00 | 1419.70 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2023-10-04 10:45:00 | 1425.45 | 2023-10-23 10:15:00 | 1419.70 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2023-10-04 13:15:00 | 1426.50 | 2023-10-23 10:15:00 | 1419.70 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-10-04 14:30:00 | 1438.95 | 2023-10-23 11:15:00 | 1418.35 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2023-10-16 09:15:00 | 1436.35 | 2023-10-23 11:15:00 | 1418.35 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2023-10-17 09:15:00 | 1441.75 | 2023-10-23 11:15:00 | 1418.35 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2023-10-19 11:15:00 | 1435.30 | 2023-10-23 11:15:00 | 1418.35 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-12-13 13:30:00 | 1440.85 | 2023-12-15 15:15:00 | 1584.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-22 14:15:00 | 1890.35 | 2024-11-28 12:15:00 | 1861.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-11-25 13:15:00 | 1888.05 | 2024-11-28 12:15:00 | 1861.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2024-11-25 13:45:00 | 1887.30 | 2024-11-28 12:15:00 | 1861.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-12-03 10:15:00 | 1890.00 | 2024-12-31 09:15:00 | 1845.90 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2024-12-27 09:30:00 | 1915.90 | 2025-01-17 09:15:00 | 1821.70 | STOP_HIT | 1.00 | -4.92% |
| BUY | retest2 | 2024-12-27 13:15:00 | 1915.45 | 2025-01-17 09:15:00 | 1821.70 | STOP_HIT | 1.00 | -4.89% |
| BUY | retest2 | 2025-01-02 09:45:00 | 1915.30 | 2025-01-17 09:15:00 | 1821.70 | STOP_HIT | 1.00 | -4.89% |
| BUY | retest2 | 2025-01-02 10:15:00 | 1918.80 | 2025-01-17 09:15:00 | 1821.70 | STOP_HIT | 1.00 | -5.06% |
| BUY | retest2 | 2025-01-08 13:00:00 | 1912.00 | 2025-01-17 09:15:00 | 1821.70 | STOP_HIT | 1.00 | -4.72% |
| SELL | retest1 | 2025-05-02 12:15:00 | 1497.70 | 2025-05-12 12:15:00 | 1607.40 | STOP_HIT | 1.00 | -7.32% |
| SELL | retest1 | 2025-05-09 09:45:00 | 1497.20 | 2025-05-12 12:15:00 | 1607.40 | STOP_HIT | 1.00 | -7.36% |
| SELL | retest2 | 2025-05-14 12:45:00 | 1582.50 | 2025-05-15 12:15:00 | 1598.60 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-05-15 09:15:00 | 1584.50 | 2025-05-15 12:15:00 | 1598.60 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-05-15 11:00:00 | 1575.00 | 2025-05-15 12:15:00 | 1598.60 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-05-16 11:15:00 | 1581.80 | 2025-05-29 09:15:00 | 1592.40 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-05-16 14:15:00 | 1586.80 | 2025-05-29 09:15:00 | 1592.40 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-05-19 09:15:00 | 1571.10 | 2025-06-10 10:15:00 | 1593.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-05-29 10:30:00 | 1585.10 | 2025-06-10 10:15:00 | 1593.80 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-06-23 09:30:00 | 1584.40 | 2025-06-24 09:15:00 | 1598.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-06-24 14:45:00 | 1581.50 | 2025-06-25 09:15:00 | 1596.80 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-06-24 15:15:00 | 1579.60 | 2025-06-25 09:15:00 | 1596.80 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-08-28 09:15:00 | 1513.20 | 2025-09-05 10:15:00 | 1437.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-28 09:15:00 | 1513.20 | 2025-09-09 10:15:00 | 1496.70 | STOP_HIT | 0.50 | 1.09% |
| SELL | retest2 | 2025-09-10 09:45:00 | 1522.80 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-09-11 09:15:00 | 1516.80 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-09-12 09:30:00 | 1524.00 | 2025-09-18 09:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-10-13 11:00:00 | 1488.40 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2025-10-13 13:15:00 | 1486.50 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-10-14 12:30:00 | 1488.30 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -3.00% |
| SELL | retest2 | 2025-10-14 15:15:00 | 1488.00 | 2025-10-23 09:15:00 | 1532.90 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-10-28 12:00:00 | 1491.50 | 2025-10-28 14:15:00 | 1501.80 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-10-28 14:15:00 | 1492.30 | 2025-10-28 14:15:00 | 1501.80 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-10-30 09:45:00 | 1491.20 | 2025-11-10 09:15:00 | 1503.60 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-10-30 13:15:00 | 1492.20 | 2025-11-10 09:15:00 | 1503.60 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-11-17 09:15:00 | 1500.40 | 2025-11-17 14:15:00 | 1508.30 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-11-17 13:00:00 | 1502.30 | 2025-11-17 14:15:00 | 1508.30 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-11-18 09:15:00 | 1498.30 | 2025-11-19 09:15:00 | 1527.90 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest1 | 2026-04-10 09:30:00 | 1288.70 | 2026-04-24 09:15:00 | 1224.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-22 09:15:00 | 1278.60 | 2026-04-24 09:15:00 | 1214.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-10 09:30:00 | 1288.70 | 2026-04-24 13:15:00 | 1159.83 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2026-04-22 09:15:00 | 1278.60 | 2026-04-28 12:15:00 | 1150.74 | TARGET_HIT | 0.50 | 10.00% |
