# Glenmark Pharmaceuticals Ltd. (GLENMARK)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 2361.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 70 |
| ALERT1 | 50 |
| ALERT2 | 49 |
| ALERT2_SKIP | 20 |
| ALERT3 | 127 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 72 |
| PARTIAL | 9 |
| TARGET_HIT | 0 |
| STOP_HIT | 72 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 81 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 48
- **Target hits / Stop hits / Partials:** 0 / 72 / 9
- **Avg / median % per leg:** 0.54% / -0.71%
- **Sum % (uncompounded):** 43.79%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 30 | 8 | 26.7% | 0 | 30 | 0 | -0.32% | -9.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 30 | 8 | 26.7% | 0 | 30 | 0 | -0.32% | -9.7% |
| SELL (all) | 51 | 25 | 49.0% | 0 | 42 | 9 | 1.05% | 53.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 51 | 25 | 49.0% | 0 | 42 | 9 | 1.05% | 53.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 81 | 33 | 40.7% | 0 | 72 | 9 | 0.54% | 43.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-13 11:15:00 | 1420.10 | 1402.14 | 1401.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 09:15:00 | 1433.10 | 1415.01 | 1408.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 11:15:00 | 1439.30 | 1440.07 | 1428.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:00:00 | 1439.30 | 1440.07 | 1428.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 1436.50 | 1440.37 | 1434.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:30:00 | 1432.50 | 1440.37 | 1434.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 1450.80 | 1442.46 | 1435.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 12:15:00 | 1454.00 | 1442.46 | 1435.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 09:45:00 | 1464.80 | 1450.33 | 1442.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:45:00 | 1455.10 | 1450.52 | 1443.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 12:15:00 | 1457.90 | 1450.52 | 1443.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 13:15:00 | 1441.60 | 1448.52 | 1444.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:00:00 | 1441.60 | 1448.52 | 1444.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1432.10 | 1445.24 | 1443.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-19 14:15:00 | 1432.10 | 1445.24 | 1443.07 | SL hit (close<static) qty=1.00 sl=1434.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 1442.00 | 1443.52 | 1443.58 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 1453.00 | 1445.42 | 1444.43 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 11:15:00 | 1433.00 | 1443.59 | 1443.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 1425.60 | 1437.12 | 1440.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 1432.50 | 1432.05 | 1436.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 14:45:00 | 1434.10 | 1432.05 | 1436.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 1427.50 | 1430.89 | 1434.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 10:45:00 | 1413.20 | 1427.49 | 1433.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 12:30:00 | 1413.20 | 1422.48 | 1429.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 09:45:00 | 1413.00 | 1418.38 | 1425.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 1410.80 | 1394.13 | 1393.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 1410.80 | 1394.13 | 1393.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 1431.20 | 1401.55 | 1396.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 1503.20 | 1504.32 | 1481.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:30:00 | 1504.70 | 1504.32 | 1481.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1621.10 | 1615.57 | 1600.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 12:00:00 | 1626.40 | 1618.45 | 1604.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 11:15:00 | 1657.90 | 1662.86 | 1662.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 11:15:00 | 1657.90 | 1662.86 | 1662.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 09:15:00 | 1645.60 | 1657.95 | 1660.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 1643.40 | 1642.02 | 1648.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:45:00 | 1640.10 | 1642.02 | 1648.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 1636.60 | 1637.89 | 1644.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 1644.10 | 1637.89 | 1644.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1661.00 | 1641.96 | 1645.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:15:00 | 1674.50 | 1641.96 | 1645.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 1688.00 | 1651.17 | 1648.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 1704.00 | 1661.73 | 1653.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 1685.10 | 1687.41 | 1676.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 1685.10 | 1687.41 | 1676.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 1714.00 | 1692.50 | 1680.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 10:00:00 | 1739.00 | 1712.55 | 1697.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 11:15:00 | 1729.70 | 1715.76 | 1700.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 12:45:00 | 1730.00 | 1721.74 | 1706.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 1732.10 | 1725.86 | 1712.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 1717.80 | 1723.61 | 1714.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:00:00 | 1717.80 | 1723.61 | 1714.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 1715.10 | 1721.91 | 1714.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 1715.10 | 1721.91 | 1714.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 1716.60 | 1720.85 | 1714.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 15:15:00 | 1720.00 | 1718.86 | 1714.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 12:15:00 | 1806.60 | 1822.68 | 1824.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 12:15:00 | 1806.60 | 1822.68 | 1824.17 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 09:15:00 | 1859.40 | 1828.46 | 1826.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 10:15:00 | 1875.20 | 1837.81 | 1830.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 09:15:00 | 2178.00 | 2184.42 | 2107.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 10:00:00 | 2178.00 | 2184.42 | 2107.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 2195.50 | 2213.40 | 2188.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:45:00 | 2198.40 | 2213.40 | 2188.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 11:15:00 | 2200.50 | 2208.17 | 2190.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 11:30:00 | 2185.20 | 2208.17 | 2190.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 2205.50 | 2217.36 | 2205.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:45:00 | 2204.70 | 2217.36 | 2205.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 2199.00 | 2213.68 | 2204.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 2205.00 | 2213.68 | 2204.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 2199.40 | 2210.83 | 2204.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 2199.40 | 2210.83 | 2204.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 2226.40 | 2213.94 | 2206.40 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-07-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 14:15:00 | 2185.40 | 2205.83 | 2206.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 2165.70 | 2195.19 | 2201.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 12:15:00 | 2152.10 | 2143.27 | 2155.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 13:15:00 | 2153.30 | 2143.27 | 2155.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 2153.30 | 2145.27 | 2155.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 13:45:00 | 2150.00 | 2145.27 | 2155.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 2144.20 | 2145.06 | 2154.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:45:00 | 2140.00 | 2147.58 | 2150.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 10:15:00 | 2154.90 | 2149.05 | 2151.06 | SL hit (close>static) qty=1.00 sl=2154.70 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 11:15:00 | 2172.20 | 2153.68 | 2152.98 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 2148.90 | 2152.70 | 2152.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-29 10:15:00 | 2134.40 | 2145.65 | 2149.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 12:15:00 | 2147.30 | 2144.45 | 2148.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 12:15:00 | 2147.30 | 2144.45 | 2148.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2147.30 | 2144.45 | 2148.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 12:45:00 | 2146.30 | 2144.45 | 2148.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2142.50 | 2144.06 | 2147.50 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 2165.70 | 2150.71 | 2150.07 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 2143.50 | 2150.75 | 2151.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 2116.90 | 2140.77 | 2145.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 2088.10 | 2083.09 | 2101.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 12:45:00 | 2093.20 | 2083.09 | 2101.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 2033.80 | 2029.31 | 2045.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 2037.00 | 2029.31 | 2045.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 2045.00 | 2032.81 | 2042.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:00:00 | 2045.00 | 2032.81 | 2042.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2064.30 | 2039.11 | 2044.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2064.30 | 2039.11 | 2044.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2060.10 | 2043.31 | 2045.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 2042.60 | 2043.31 | 2045.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 2049.60 | 2038.61 | 2038.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 2049.60 | 2038.61 | 2038.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 12:15:00 | 2068.90 | 2051.39 | 2045.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 15:15:00 | 2053.40 | 2054.94 | 2048.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 09:15:00 | 2042.70 | 2054.94 | 2048.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 2059.30 | 2055.81 | 2049.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:30:00 | 2044.10 | 2055.81 | 2049.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2059.60 | 2056.57 | 2050.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 2051.00 | 2056.57 | 2050.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 2053.00 | 2055.85 | 2051.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 2053.00 | 2055.85 | 2051.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 2041.80 | 2053.04 | 2050.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:45:00 | 2044.40 | 2053.04 | 2050.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 14:15:00 | 2024.40 | 2047.32 | 2047.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 09:15:00 | 2022.20 | 2035.38 | 2039.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 10:15:00 | 1947.90 | 1934.29 | 1951.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-21 10:45:00 | 1945.50 | 1934.29 | 1951.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1957.00 | 1938.83 | 1952.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:45:00 | 1958.30 | 1938.83 | 1952.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1948.90 | 1940.84 | 1951.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 13:15:00 | 1940.60 | 1940.84 | 1951.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 14:30:00 | 1940.70 | 1942.30 | 1950.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:45:00 | 1941.80 | 1944.38 | 1949.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:00:00 | 1936.40 | 1942.79 | 1948.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1953.00 | 1942.59 | 1945.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 1953.00 | 1942.59 | 1945.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1956.40 | 1945.35 | 1946.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 1956.40 | 1945.35 | 1946.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-25 12:15:00 | 1960.90 | 1949.62 | 1948.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 12:15:00 | 1960.90 | 1949.62 | 1948.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 12:15:00 | 1969.80 | 1956.83 | 1953.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 1951.30 | 1958.04 | 1954.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 1951.30 | 1958.04 | 1954.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 1951.30 | 1958.04 | 1954.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 1951.30 | 1958.04 | 1954.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 1950.00 | 1956.43 | 1953.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-28 09:15:00 | 1936.90 | 1956.43 | 1953.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 10:15:00 | 1950.00 | 1952.34 | 1952.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 1934.00 | 1946.35 | 1949.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 1942.00 | 1941.57 | 1946.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 1942.00 | 1941.57 | 1946.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 1926.60 | 1930.18 | 1937.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 1933.90 | 1930.18 | 1937.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1930.90 | 1925.74 | 1930.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 14:45:00 | 1914.50 | 1924.94 | 1928.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 09:15:00 | 1965.90 | 1932.34 | 1931.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 09:15:00 | 1965.90 | 1932.34 | 1931.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 10:15:00 | 1991.50 | 1944.17 | 1936.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 14:15:00 | 1989.40 | 1990.72 | 1975.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-04 15:00:00 | 1989.40 | 1990.72 | 1975.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 2106.50 | 2120.61 | 2100.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 2090.00 | 2120.61 | 2100.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 2112.50 | 2118.99 | 2101.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:30:00 | 2107.80 | 2118.99 | 2101.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 2127.00 | 2136.24 | 2124.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 2108.80 | 2136.24 | 2124.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 2105.60 | 2130.11 | 2123.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:45:00 | 2097.00 | 2130.11 | 2123.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 2107.50 | 2125.59 | 2121.65 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 12:15:00 | 2096.60 | 2117.65 | 2118.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 09:15:00 | 2076.30 | 2101.71 | 2110.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 2074.10 | 2058.21 | 2072.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 2074.10 | 2058.21 | 2072.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 2074.10 | 2058.21 | 2072.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 2074.10 | 2058.21 | 2072.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 2069.70 | 2060.50 | 2072.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 2078.50 | 2060.50 | 2072.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 2071.90 | 2062.78 | 2072.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:30:00 | 2070.40 | 2062.78 | 2072.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 2078.20 | 2065.87 | 2072.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:30:00 | 2074.20 | 2065.87 | 2072.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 2109.90 | 2074.67 | 2076.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:00:00 | 2109.90 | 2074.67 | 2076.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 14:15:00 | 2108.30 | 2081.40 | 2079.19 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 2071.10 | 2083.96 | 2084.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 11:15:00 | 2038.80 | 2074.93 | 2080.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 11:15:00 | 2035.90 | 2035.82 | 2053.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:00:00 | 2035.90 | 2035.82 | 2053.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 2045.10 | 2027.71 | 2037.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:00:00 | 2045.10 | 2027.71 | 2037.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 2025.70 | 2027.31 | 2036.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 2017.30 | 2025.41 | 2035.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1983.00 | 2032.82 | 2035.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 1983.90 | 1965.63 | 1964.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 1983.90 | 1965.63 | 1964.81 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 11:15:00 | 1964.00 | 1965.57 | 1965.60 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 15:15:00 | 1971.80 | 1966.54 | 1965.89 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 1950.40 | 1963.31 | 1964.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1932.30 | 1955.63 | 1960.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 1949.90 | 1941.66 | 1949.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 1949.90 | 1941.66 | 1949.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 1949.90 | 1941.66 | 1949.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:45:00 | 1934.70 | 1940.39 | 1946.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 15:00:00 | 1935.10 | 1939.33 | 1945.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 1929.40 | 1938.40 | 1943.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 11:15:00 | 1964.00 | 1943.38 | 1945.10 | SL hit (close>static) qty=1.00 sl=1961.80 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 1839.00 | 1828.57 | 1827.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 1868.00 | 1842.08 | 1834.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 15:15:00 | 1889.10 | 1892.77 | 1882.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 09:15:00 | 1867.10 | 1892.77 | 1882.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1872.00 | 1888.61 | 1881.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:15:00 | 1863.50 | 1888.61 | 1881.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1857.90 | 1882.47 | 1879.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 1857.90 | 1882.47 | 1879.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 12:15:00 | 1867.50 | 1876.68 | 1877.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 1859.30 | 1873.21 | 1875.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 09:15:00 | 1826.60 | 1825.30 | 1842.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 10:00:00 | 1826.60 | 1825.30 | 1842.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 1837.60 | 1821.16 | 1830.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 1844.00 | 1821.16 | 1830.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1827.50 | 1822.43 | 1830.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 1830.80 | 1822.43 | 1830.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1835.20 | 1824.98 | 1830.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:30:00 | 1848.50 | 1824.98 | 1830.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 1834.80 | 1826.94 | 1831.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 13:30:00 | 1832.90 | 1828.28 | 1831.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 15:00:00 | 1829.60 | 1828.54 | 1831.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 1830.80 | 1830.45 | 1831.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 10:15:00 | 1819.60 | 1831.44 | 1832.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1822.20 | 1829.59 | 1831.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:00:00 | 1817.90 | 1825.78 | 1828.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:00:00 | 1816.70 | 1823.97 | 1827.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1838.10 | 1825.20 | 1827.65 | SL hit (close>static) qty=1.00 sl=1836.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 1851.70 | 1830.50 | 1829.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 1875.40 | 1848.61 | 1839.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 1879.90 | 1881.18 | 1862.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 1879.90 | 1881.18 | 1862.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1892.00 | 1884.58 | 1867.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:00:00 | 1910.00 | 1889.66 | 1871.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 1923.20 | 1892.11 | 1873.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 10:00:00 | 1917.00 | 1902.18 | 1886.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 15:15:00 | 1867.30 | 1882.40 | 1882.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 1867.30 | 1882.40 | 1882.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 1836.90 | 1873.30 | 1878.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 1853.00 | 1848.36 | 1859.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 1853.00 | 1848.36 | 1859.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 1853.00 | 1848.36 | 1859.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:30:00 | 1850.90 | 1848.36 | 1859.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 1853.80 | 1848.24 | 1856.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:15:00 | 1849.00 | 1848.24 | 1856.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 14:00:00 | 1849.10 | 1848.41 | 1856.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 09:30:00 | 1847.00 | 1846.04 | 1852.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:45:00 | 1847.50 | 1850.95 | 1853.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 13:15:00 | 1859.50 | 1852.66 | 1854.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 13:45:00 | 1859.00 | 1852.66 | 1854.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-20 14:15:00 | 1870.30 | 1856.19 | 1855.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 1870.30 | 1856.19 | 1855.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 15:15:00 | 1884.90 | 1861.93 | 1858.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 14:15:00 | 1842.10 | 1861.93 | 1860.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 14:15:00 | 1842.10 | 1861.93 | 1860.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1842.10 | 1861.93 | 1860.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 1842.10 | 1861.93 | 1860.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 1843.20 | 1858.19 | 1859.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 13:15:00 | 1837.60 | 1852.13 | 1855.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 1884.70 | 1855.04 | 1855.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 1884.70 | 1855.04 | 1855.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 1884.70 | 1855.04 | 1855.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 1884.70 | 1855.04 | 1855.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 10:15:00 | 1875.00 | 1859.03 | 1857.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 09:15:00 | 1897.70 | 1877.24 | 1868.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1942.70 | 1944.53 | 1929.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:45:00 | 1942.50 | 1944.53 | 1929.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1940.00 | 1943.90 | 1934.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:30:00 | 1935.80 | 1943.90 | 1934.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 1938.40 | 1942.80 | 1935.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 1937.00 | 1942.80 | 1935.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1945.00 | 1943.24 | 1936.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:45:00 | 1926.20 | 1943.24 | 1936.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1965.40 | 1967.95 | 1958.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:30:00 | 1955.00 | 1967.95 | 1958.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 1967.90 | 1968.06 | 1960.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 14:45:00 | 1963.80 | 1968.06 | 1960.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1970.80 | 1968.12 | 1961.79 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 1962.20 | 1969.00 | 1969.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 11:15:00 | 1946.10 | 1964.42 | 1966.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1943.00 | 1936.09 | 1947.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 1943.00 | 1936.09 | 1947.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1936.50 | 1936.18 | 1946.63 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 15:15:00 | 1951.10 | 1947.10 | 1946.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1961.00 | 1951.11 | 1948.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 12:15:00 | 1947.00 | 1951.95 | 1949.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 12:15:00 | 1947.00 | 1951.95 | 1949.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 1947.00 | 1951.95 | 1949.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 13:00:00 | 1947.00 | 1951.95 | 1949.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 1951.80 | 1951.92 | 1949.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 1963.50 | 1954.65 | 1951.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 10:45:00 | 1959.90 | 1962.88 | 1959.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 10:15:00 | 1946.00 | 1961.69 | 1963.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 1946.00 | 1961.69 | 1963.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 12:15:00 | 1941.00 | 1956.23 | 1960.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 1952.00 | 1949.39 | 1955.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 10:15:00 | 1952.00 | 1949.39 | 1955.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1952.00 | 1949.39 | 1955.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 1955.40 | 1949.39 | 1955.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1954.10 | 1950.65 | 1954.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:00:00 | 1954.10 | 1950.65 | 1954.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 1949.90 | 1950.50 | 1954.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:30:00 | 1947.80 | 1951.68 | 1954.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 1958.00 | 1952.95 | 1954.85 | SL hit (close>static) qty=1.00 sl=1955.70 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 1977.80 | 1957.92 | 1956.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 2012.10 | 1980.52 | 1969.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 12:15:00 | 2046.00 | 2047.04 | 2019.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 13:00:00 | 2046.00 | 2047.04 | 2019.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 2025.40 | 2042.77 | 2026.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 2023.20 | 2042.77 | 2026.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 2019.30 | 2038.08 | 2025.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 2019.30 | 2038.08 | 2025.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 2021.80 | 2034.82 | 2025.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:15:00 | 2026.60 | 2034.82 | 2025.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 2007.00 | 2022.14 | 2022.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 2007.00 | 2022.14 | 2022.19 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 2037.00 | 2016.12 | 2015.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 11:15:00 | 2042.80 | 2028.67 | 2022.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 2021.50 | 2029.95 | 2025.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 2021.50 | 2029.95 | 2025.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 2021.50 | 2029.95 | 2025.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 2024.30 | 2029.95 | 2025.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 2027.00 | 2029.36 | 2025.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 10:15:00 | 2036.20 | 2027.21 | 2025.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 09:15:00 | 2060.40 | 2086.83 | 2087.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2026-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 09:15:00 | 2060.40 | 2086.83 | 2087.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 2038.60 | 2068.57 | 2078.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 2016.90 | 2012.42 | 2034.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 2019.00 | 2012.42 | 2034.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1961.90 | 2003.13 | 2026.56 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 2021.50 | 2015.77 | 2015.40 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 12:15:00 | 2008.00 | 2014.25 | 2014.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 1996.80 | 2010.76 | 2013.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 1952.90 | 1946.43 | 1964.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 1952.90 | 1946.43 | 1964.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1985.50 | 1950.55 | 1960.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 1985.50 | 1950.55 | 1960.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1978.10 | 1956.06 | 1962.00 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 1978.20 | 1967.19 | 1966.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 1992.20 | 1972.19 | 1968.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 11:15:00 | 1982.50 | 1987.94 | 1978.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 12:00:00 | 1982.50 | 1987.94 | 1978.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 13:15:00 | 1975.20 | 1985.72 | 1979.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 14:00:00 | 1975.20 | 1985.72 | 1979.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 1966.80 | 1981.94 | 1978.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 1966.80 | 1981.94 | 1978.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 1968.70 | 1979.29 | 1977.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 1987.10 | 1979.29 | 1977.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 11:15:00 | 1982.40 | 1994.80 | 1995.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1982.40 | 1994.80 | 1995.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 1971.30 | 1990.10 | 1993.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 1914.10 | 1906.73 | 1938.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 1914.10 | 1906.73 | 1938.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 1922.20 | 1909.83 | 1936.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 1975.50 | 1909.83 | 1936.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1961.60 | 1920.18 | 1938.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 13:30:00 | 1954.60 | 1946.47 | 1947.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 1953.40 | 1947.85 | 1947.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 1953.40 | 1947.85 | 1947.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 1987.40 | 1957.31 | 1952.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 12:15:00 | 1955.70 | 1958.41 | 1954.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 12:15:00 | 1955.70 | 1958.41 | 1954.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 1955.70 | 1958.41 | 1954.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 12:30:00 | 1955.30 | 1958.41 | 1954.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 1965.00 | 1959.73 | 1955.08 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 09:15:00 | 1929.50 | 1952.98 | 1953.14 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 1974.30 | 1956.42 | 1953.98 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 1930.20 | 1949.30 | 1951.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 11:15:00 | 1918.80 | 1943.20 | 1948.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 1966.20 | 1942.23 | 1944.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 1966.20 | 1942.23 | 1944.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1966.20 | 1942.23 | 1944.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 1966.20 | 1942.23 | 1944.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 10:15:00 | 1976.50 | 1949.09 | 1947.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 1986.90 | 1970.06 | 1964.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 2010.30 | 2020.64 | 2005.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-13 10:00:00 | 2010.30 | 2020.64 | 2005.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 2038.30 | 2024.17 | 2008.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 11:45:00 | 2056.80 | 2029.16 | 2026.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 12:30:00 | 2046.60 | 2032.47 | 2028.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 15:15:00 | 2006.70 | 2025.09 | 2026.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 2006.70 | 2025.09 | 2026.17 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 11:15:00 | 2046.70 | 2028.28 | 2027.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 2054.90 | 2042.63 | 2037.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 2120.00 | 2132.00 | 2113.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 2120.00 | 2132.00 | 2113.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 2120.00 | 2132.00 | 2113.90 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 2052.00 | 2106.45 | 2108.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 10:15:00 | 2041.50 | 2093.46 | 2102.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 2066.50 | 2060.49 | 2078.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 2066.50 | 2060.49 | 2078.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 2066.50 | 2060.49 | 2078.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 2087.20 | 2060.49 | 2078.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 2082.40 | 2064.87 | 2078.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 2082.40 | 2064.87 | 2078.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 2081.20 | 2068.14 | 2078.92 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 2101.10 | 2085.31 | 2084.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 2130.90 | 2094.43 | 2088.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 2066.00 | 2104.64 | 2098.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 2066.00 | 2104.64 | 2098.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 2066.00 | 2104.64 | 2098.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:15:00 | 2057.50 | 2104.64 | 2098.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 2072.30 | 2098.18 | 2096.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:30:00 | 2061.20 | 2098.18 | 2096.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 2077.30 | 2094.00 | 2094.79 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-03-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 13:15:00 | 2105.80 | 2095.91 | 2095.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 14:15:00 | 2120.20 | 2100.77 | 2097.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 2225.00 | 2248.63 | 2211.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:30:00 | 2225.00 | 2248.63 | 2211.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 2186.60 | 2238.33 | 2224.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 2186.60 | 2238.33 | 2224.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 2179.00 | 2226.46 | 2220.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:30:00 | 2168.00 | 2226.46 | 2220.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 2180.50 | 2209.85 | 2213.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 14:15:00 | 2171.00 | 2198.67 | 2207.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 2177.10 | 2174.24 | 2188.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:30:00 | 2181.60 | 2174.24 | 2188.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 2209.50 | 2167.04 | 2173.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 2209.50 | 2167.04 | 2173.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 2211.70 | 2175.97 | 2177.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 2211.70 | 2175.97 | 2177.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 2232.10 | 2187.20 | 2182.39 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 2151.30 | 2182.54 | 2182.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 2129.30 | 2171.89 | 2177.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 2140.80 | 2132.50 | 2151.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 2140.80 | 2132.50 | 2151.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 2140.80 | 2132.50 | 2151.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:45:00 | 2144.90 | 2132.50 | 2151.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 2165.80 | 2139.16 | 2152.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 2165.80 | 2139.16 | 2152.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 2170.70 | 2145.47 | 2154.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:45:00 | 2167.90 | 2145.47 | 2154.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 14:15:00 | 2180.40 | 2163.10 | 2161.32 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 2122.50 | 2157.69 | 2159.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 2115.10 | 2149.17 | 2155.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 2112.60 | 2111.79 | 2130.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 2112.60 | 2111.79 | 2130.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 2112.60 | 2111.79 | 2130.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 2101.00 | 2111.79 | 2130.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:00:00 | 2096.20 | 2101.42 | 2120.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 09:15:00 | 2146.30 | 2107.50 | 2116.83 | SL hit (close>static) qty=1.00 sl=2137.10 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 11:15:00 | 2164.50 | 2126.31 | 2124.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 13:15:00 | 2168.80 | 2139.64 | 2130.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 2152.20 | 2161.99 | 2151.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 2152.20 | 2161.99 | 2151.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 2152.20 | 2161.99 | 2151.99 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 2122.00 | 2146.50 | 2148.45 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 2196.80 | 2156.56 | 2152.85 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-01 13:15:00 | 2109.80 | 2143.14 | 2147.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 14:15:00 | 2100.00 | 2134.51 | 2143.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 2080.70 | 2077.69 | 2104.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-02 14:00:00 | 2080.70 | 2077.69 | 2104.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 2090.50 | 2081.62 | 2096.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:45:00 | 2091.10 | 2081.62 | 2096.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 2100.00 | 2085.29 | 2096.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:15:00 | 2108.90 | 2085.29 | 2096.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 2100.00 | 2088.24 | 2097.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:30:00 | 2100.00 | 2088.24 | 2097.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 2103.30 | 2091.25 | 2097.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:45:00 | 2108.70 | 2091.25 | 2097.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 10:15:00 | 2122.00 | 2102.27 | 2101.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 2139.40 | 2119.89 | 2111.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 2161.30 | 2163.98 | 2147.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 13:45:00 | 2158.40 | 2163.98 | 2147.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 2174.70 | 2168.79 | 2155.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:45:00 | 2162.00 | 2168.79 | 2155.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 12:15:00 | 2167.90 | 2168.50 | 2157.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 13:30:00 | 2176.40 | 2168.80 | 2158.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 2145.20 | 2162.36 | 2157.97 | SL hit (close<static) qty=1.00 sl=2155.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 2231.40 | 2236.09 | 2236.41 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 2241.50 | 2237.17 | 2236.88 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 11:15:00 | 2227.80 | 2235.33 | 2236.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 12:15:00 | 2227.30 | 2233.73 | 2235.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 14:15:00 | 2233.00 | 2232.57 | 2234.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 14:15:00 | 2233.00 | 2232.57 | 2234.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 2233.00 | 2232.57 | 2234.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 15:15:00 | 2235.00 | 2232.57 | 2234.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 2235.00 | 2233.06 | 2234.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 2218.50 | 2233.06 | 2234.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 14:15:00 | 2240.90 | 2228.79 | 2230.73 | SL hit (close>static) qty=1.00 sl=2237.90 alert=retest2 |

### Cycle 69 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 2352.00 | 2254.48 | 2242.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 2396.70 | 2329.13 | 2309.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 2397.70 | 2410.22 | 2388.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 11:00:00 | 2397.70 | 2410.22 | 2388.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 2403.60 | 2406.17 | 2390.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 13:45:00 | 2410.00 | 2407.00 | 2392.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 2425.20 | 2405.37 | 2394.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 2409.20 | 2405.83 | 2396.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 09:15:00 | 2388.00 | 2396.25 | 2395.25 | SL hit (close<static) qty=1.00 sl=2389.30 alert=retest2 |

### Cycle 70 — SELL (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 15:15:00 | 2379.00 | 2399.77 | 2402.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 09:15:00 | 2349.70 | 2373.31 | 2385.37 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 12:15:00 | 1454.00 | 2025-05-19 14:15:00 | 1432.10 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2025-05-19 09:45:00 | 1464.80 | 2025-05-19 14:15:00 | 1432.10 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2025-05-19 11:45:00 | 1455.10 | 2025-05-19 14:15:00 | 1432.10 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-05-19 12:15:00 | 1457.90 | 2025-05-19 14:15:00 | 1432.10 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-05-23 10:45:00 | 1413.20 | 2025-05-29 15:15:00 | 1410.80 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-05-23 12:30:00 | 1413.20 | 2025-05-29 15:15:00 | 1410.80 | STOP_HIT | 1.00 | 0.17% |
| SELL | retest2 | 2025-05-26 09:45:00 | 1413.00 | 2025-05-29 15:15:00 | 1410.80 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-06-11 12:00:00 | 1626.40 | 2025-06-18 11:15:00 | 1657.90 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2025-06-26 10:00:00 | 1739.00 | 2025-07-09 12:15:00 | 1806.60 | STOP_HIT | 1.00 | 3.89% |
| BUY | retest2 | 2025-06-26 11:15:00 | 1729.70 | 2025-07-09 12:15:00 | 1806.60 | STOP_HIT | 1.00 | 4.45% |
| BUY | retest2 | 2025-06-26 12:45:00 | 1730.00 | 2025-07-09 12:15:00 | 1806.60 | STOP_HIT | 1.00 | 4.43% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1732.10 | 2025-07-09 12:15:00 | 1806.60 | STOP_HIT | 1.00 | 4.30% |
| BUY | retest2 | 2025-06-27 15:15:00 | 1720.00 | 2025-07-09 12:15:00 | 1806.60 | STOP_HIT | 1.00 | 5.03% |
| SELL | retest2 | 2025-07-28 09:45:00 | 2140.00 | 2025-07-28 10:15:00 | 2154.90 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-08-08 09:15:00 | 2042.60 | 2025-08-11 14:15:00 | 2049.60 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-08-21 13:15:00 | 1940.60 | 2025-08-25 12:15:00 | 1960.90 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-08-21 14:30:00 | 1940.70 | 2025-08-25 12:15:00 | 1960.90 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-08-22 10:45:00 | 1941.80 | 2025-08-25 12:15:00 | 1960.90 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-08-22 12:00:00 | 1936.40 | 2025-08-25 12:15:00 | 1960.90 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-09-02 14:45:00 | 1914.50 | 2025-09-03 09:15:00 | 1965.90 | STOP_HIT | 1.00 | -2.68% |
| SELL | retest2 | 2025-09-24 14:45:00 | 2017.30 | 2025-10-03 12:15:00 | 1983.90 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1983.00 | 2025-10-03 12:15:00 | 1983.90 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2025-10-09 13:45:00 | 1934.70 | 2025-10-10 11:15:00 | 1964.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-10-09 15:00:00 | 1935.10 | 2025-10-10 11:15:00 | 1964.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2025-10-10 10:15:00 | 1929.40 | 2025-10-10 11:15:00 | 1964.00 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1930.00 | 2025-10-24 11:15:00 | 1833.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 10:15:00 | 1907.70 | 2025-10-24 14:15:00 | 1812.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-13 11:00:00 | 1904.10 | 2025-10-24 14:15:00 | 1812.03 | PARTIAL | 0.50 | 4.84% |
| SELL | retest2 | 2025-10-13 12:30:00 | 1903.50 | 2025-10-27 09:15:00 | 1808.89 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2025-10-14 09:15:00 | 1907.40 | 2025-10-27 09:15:00 | 1808.32 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2025-10-14 10:45:00 | 1890.00 | 2025-10-27 09:15:00 | 1804.81 | PARTIAL | 0.50 | 4.51% |
| SELL | retest2 | 2025-10-14 14:30:00 | 1899.80 | 2025-10-27 09:15:00 | 1804.43 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2025-10-14 15:00:00 | 1899.40 | 2025-10-27 09:15:00 | 1803.29 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-10-15 13:15:00 | 1898.20 | 2025-10-28 13:15:00 | 1795.50 | PARTIAL | 0.50 | 5.41% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1930.00 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 5.53% |
| SELL | retest2 | 2025-10-13 10:15:00 | 1907.70 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2025-10-13 11:00:00 | 1904.10 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.25% |
| SELL | retest2 | 2025-10-13 12:30:00 | 1903.50 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.22% |
| SELL | retest2 | 2025-10-14 09:15:00 | 1907.40 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.41% |
| SELL | retest2 | 2025-10-14 10:45:00 | 1890.00 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 3.53% |
| SELL | retest2 | 2025-10-14 14:30:00 | 1899.80 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.03% |
| SELL | retest2 | 2025-10-14 15:00:00 | 1899.40 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 4.01% |
| SELL | retest2 | 2025-10-15 13:15:00 | 1898.20 | 2025-10-28 15:15:00 | 1823.20 | STOP_HIT | 0.50 | 3.95% |
| SELL | retest2 | 2025-10-23 13:45:00 | 1851.40 | 2025-10-29 12:15:00 | 1839.00 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-10-23 15:00:00 | 1847.00 | 2025-10-29 12:15:00 | 1839.00 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-11-10 13:30:00 | 1832.90 | 2025-11-12 09:15:00 | 1838.10 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2025-11-10 15:00:00 | 1829.60 | 2025-11-12 09:15:00 | 1838.10 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-11-11 09:15:00 | 1830.80 | 2025-11-12 10:15:00 | 1851.70 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-11 10:15:00 | 1819.60 | 2025-11-12 10:15:00 | 1851.70 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-11-11 14:00:00 | 1817.90 | 2025-11-12 10:15:00 | 1851.70 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-11-11 15:00:00 | 1816.70 | 2025-11-12 10:15:00 | 1851.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-11-12 09:45:00 | 1815.10 | 2025-11-12 10:15:00 | 1851.70 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-11-14 11:00:00 | 1910.00 | 2025-11-17 15:15:00 | 1867.30 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-11-14 12:15:00 | 1923.20 | 2025-11-17 15:15:00 | 1867.30 | STOP_HIT | 1.00 | -2.91% |
| BUY | retest2 | 2025-11-17 10:00:00 | 1917.00 | 2025-11-17 15:15:00 | 1867.30 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-11-19 13:15:00 | 1849.00 | 2025-11-20 14:15:00 | 1870.30 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-19 14:00:00 | 1849.10 | 2025-11-20 14:15:00 | 1870.30 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-20 09:30:00 | 1847.00 | 2025-11-20 14:15:00 | 1870.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-11-20 12:45:00 | 1847.50 | 2025-11-20 14:15:00 | 1870.30 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-12-12 09:45:00 | 1963.50 | 2025-12-17 10:15:00 | 1946.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-12-15 10:45:00 | 1959.90 | 2025-12-17 10:15:00 | 1946.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-12-18 14:30:00 | 1947.80 | 2025-12-18 15:15:00 | 1958.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-12-24 12:15:00 | 2026.60 | 2025-12-26 09:15:00 | 2007.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2026-01-02 10:15:00 | 2036.20 | 2026-01-09 09:15:00 | 2060.40 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2026-01-27 09:15:00 | 1987.10 | 2026-02-01 11:15:00 | 1982.40 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2026-02-03 13:30:00 | 1954.60 | 2026-02-03 14:15:00 | 1953.40 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2026-02-19 11:45:00 | 2056.80 | 2026-02-19 15:15:00 | 2006.70 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2026-02-19 12:30:00 | 2046.60 | 2026-02-19 15:15:00 | 2006.70 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-03-24 10:15:00 | 2101.00 | 2026-03-25 09:15:00 | 2146.30 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2026-03-24 13:00:00 | 2096.20 | 2026-03-25 09:15:00 | 2146.30 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2026-04-10 13:30:00 | 2176.40 | 2026-04-13 09:15:00 | 2145.20 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2026-04-13 11:45:00 | 2176.50 | 2026-04-20 15:15:00 | 2231.40 | STOP_HIT | 1.00 | 2.52% |
| SELL | retest2 | 2026-04-22 09:15:00 | 2218.50 | 2026-04-22 14:15:00 | 2240.90 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2026-04-30 13:45:00 | 2410.00 | 2026-05-05 09:15:00 | 2388.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-05-04 09:15:00 | 2425.20 | 2026-05-05 09:15:00 | 2388.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-05-04 11:15:00 | 2409.20 | 2026-05-05 09:15:00 | 2388.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-05-05 09:30:00 | 2411.40 | 2026-05-06 14:15:00 | 2362.20 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-05-05 11:15:00 | 2409.20 | 2026-05-06 14:15:00 | 2362.20 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-05-05 13:00:00 | 2416.00 | 2026-05-06 14:15:00 | 2362.20 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2026-05-06 09:45:00 | 2417.20 | 2026-05-06 14:15:00 | 2362.20 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-05-06 10:45:00 | 2413.80 | 2026-05-06 14:15:00 | 2362.20 | STOP_HIT | 1.00 | -2.14% |
