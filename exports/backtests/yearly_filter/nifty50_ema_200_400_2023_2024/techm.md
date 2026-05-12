# TECHM (TECHM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1460.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 1 |
| ALERT3 | 42 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 24 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 20
- **Target hits / Stop hits / Partials:** 3 / 21 / 3
- **Avg / median % per leg:** 0.51% / -1.25%
- **Sum % (uncompounded):** 13.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| SELL (all) | 25 | 5 | 20.0% | 1 | 21 | 3 | -0.25% | -6.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 25 | 5 | 20.0% | 1 | 21 | 3 | -0.25% | -6.3% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 7 | 25.9% | 3 | 21 | 3 | 0.51% | 13.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 14:15:00 | 1116.80 | 1203.93 | 1204.30 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-04 09:15:00 | 1225.65 | 1188.96 | 1188.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 11:15:00 | 1256.40 | 1205.08 | 1198.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-03 14:15:00 | 1243.70 | 1248.12 | 1227.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-03 15:00:00 | 1243.70 | 1248.12 | 1227.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 12:15:00 | 1227.60 | 1247.69 | 1229.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 13:00:00 | 1227.60 | 1247.69 | 1229.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 13:15:00 | 1231.00 | 1247.52 | 1229.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 13:30:00 | 1226.50 | 1247.52 | 1229.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 1228.50 | 1247.33 | 1229.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 15:00:00 | 1228.50 | 1247.33 | 1229.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 15:15:00 | 1223.00 | 1247.09 | 1229.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 1246.45 | 1247.09 | 1229.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-15 09:15:00 | 1371.10 | 1251.54 | 1233.71 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-26 12:15:00 | 1258.75 | 1284.16 | 1284.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 13:15:00 | 1251.00 | 1281.91 | 1283.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 13:15:00 | 1274.00 | 1272.77 | 1277.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-04 14:00:00 | 1274.00 | 1272.77 | 1277.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 14:15:00 | 1281.55 | 1272.85 | 1277.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 14:30:00 | 1287.20 | 1272.85 | 1277.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 15:15:00 | 1278.15 | 1272.91 | 1277.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-05 09:15:00 | 1265.50 | 1272.91 | 1277.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-16 11:15:00 | 1202.22 | 1264.03 | 1272.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-26 09:15:00 | 1312.30 | 1239.07 | 1256.51 | SL hit (close>ema200) qty=0.50 sl=1239.07 alert=retest2 |

### Cycle 4 — BUY (started 2024-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 09:15:00 | 1307.00 | 1265.68 | 1265.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 12:15:00 | 1321.45 | 1267.20 | 1266.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 14:15:00 | 1287.10 | 1287.18 | 1277.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-29 15:00:00 | 1287.10 | 1287.18 | 1277.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1276.00 | 1287.05 | 1277.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:15:00 | 1272.85 | 1287.05 | 1277.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 1279.35 | 1286.98 | 1277.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:30:00 | 1281.45 | 1286.98 | 1277.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 1262.15 | 1286.67 | 1277.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:00:00 | 1262.15 | 1286.67 | 1277.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 13:15:00 | 1245.00 | 1286.26 | 1277.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 14:00:00 | 1245.00 | 1286.26 | 1277.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 13:15:00 | 1271.30 | 1276.42 | 1273.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-05 13:45:00 | 1273.00 | 1276.42 | 1273.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1264.30 | 1276.30 | 1273.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 1264.30 | 1276.30 | 1273.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 1268.00 | 1276.22 | 1273.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 09:15:00 | 1293.00 | 1276.22 | 1273.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-21 09:15:00 | 1422.30 | 1322.88 | 1301.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 14:15:00 | 1651.60 | 1690.58 | 1690.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 1633.20 | 1685.34 | 1687.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 12:15:00 | 1682.95 | 1678.60 | 1683.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 12:15:00 | 1682.95 | 1678.60 | 1683.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 1682.95 | 1678.60 | 1683.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 12:30:00 | 1683.75 | 1678.60 | 1683.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 13:15:00 | 1682.10 | 1678.63 | 1683.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 13:30:00 | 1688.60 | 1678.63 | 1683.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 1693.90 | 1678.79 | 1683.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 1693.90 | 1678.79 | 1683.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 1690.10 | 1678.90 | 1683.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 1682.10 | 1678.90 | 1683.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 14:15:00 | 1699.00 | 1679.34 | 1684.03 | SL hit (close>static) qty=1.00 sl=1695.60 alert=retest2 |

### Cycle 6 — BUY (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 14:15:00 | 1575.60 | 1503.54 | 1503.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 1590.00 | 1505.15 | 1504.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 1536.90 | 1540.70 | 1525.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-03 10:00:00 | 1536.90 | 1540.70 | 1525.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1599.70 | 1639.85 | 1604.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 1599.70 | 1639.85 | 1604.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1596.40 | 1639.42 | 1604.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:45:00 | 1594.20 | 1639.42 | 1604.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 1598.30 | 1634.30 | 1603.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 1583.80 | 1634.30 | 1603.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1585.80 | 1633.82 | 1603.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 1585.50 | 1633.82 | 1603.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 1594.80 | 1625.45 | 1601.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 1594.80 | 1625.45 | 1601.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1583.20 | 1624.49 | 1601.93 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 10:15:00 | 1460.30 | 1585.43 | 1585.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 13:15:00 | 1447.50 | 1581.51 | 1583.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 1525.00 | 1520.51 | 1544.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-14 11:00:00 | 1525.00 | 1520.51 | 1544.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1527.90 | 1514.93 | 1537.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 1532.00 | 1514.93 | 1537.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1528.50 | 1505.38 | 1524.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:00:00 | 1528.50 | 1505.38 | 1524.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1529.90 | 1505.63 | 1524.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:30:00 | 1535.00 | 1505.63 | 1524.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 1528.80 | 1506.06 | 1524.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 13:00:00 | 1528.80 | 1506.06 | 1524.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 13:15:00 | 1528.80 | 1506.29 | 1524.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 14:00:00 | 1528.80 | 1506.29 | 1524.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1526.50 | 1506.69 | 1524.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:15:00 | 1514.70 | 1506.69 | 1524.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1521.60 | 1507.44 | 1524.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 10:00:00 | 1510.80 | 1508.47 | 1524.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 1533.00 | 1509.83 | 1524.15 | SL hit (close>static) qty=1.00 sl=1530.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 11:15:00 | 1546.10 | 1472.93 | 1472.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 09:15:00 | 1574.70 | 1476.65 | 1474.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 14:15:00 | 1578.30 | 1579.91 | 1546.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 14:30:00 | 1579.90 | 1579.91 | 1546.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 09:15:00 | 1628.90 | 1665.17 | 1613.68 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-23 13:15:00 | 1441.90 | 1589.59 | 1589.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 14:15:00 | 1439.60 | 1588.09 | 1589.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1422.70 | 1416.16 | 1473.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 1425.20 | 1416.16 | 1473.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 1459.40 | 1413.81 | 1461.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:30:00 | 1455.80 | 1413.81 | 1461.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 1451.40 | 1414.19 | 1461.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 1437.50 | 1422.32 | 1461.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 1440.80 | 1424.27 | 1461.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1462.10 | 1425.95 | 1459.41 | SL hit (close>static) qty=1.00 sl=1462.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-01-09 09:15:00 | 1246.45 | 2024-01-15 09:15:00 | 1371.10 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-05 09:15:00 | 1265.50 | 2024-04-16 11:15:00 | 1202.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-05 09:15:00 | 1265.50 | 2024-04-26 09:15:00 | 1312.30 | STOP_HIT | 0.50 | -3.70% |
| SELL | retest2 | 2024-04-30 10:30:00 | 1276.10 | 2024-05-07 13:15:00 | 1287.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2024-05-09 09:15:00 | 1275.90 | 2024-05-14 13:15:00 | 1280.55 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2024-05-09 12:30:00 | 1274.40 | 2024-05-14 13:15:00 | 1280.55 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2024-05-10 11:15:00 | 1262.10 | 2024-05-14 13:15:00 | 1280.55 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-05-10 14:00:00 | 1261.40 | 2024-05-14 13:15:00 | 1280.55 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-05-10 15:15:00 | 1262.45 | 2024-05-16 09:15:00 | 1293.05 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-05-13 15:15:00 | 1260.80 | 2024-05-16 09:15:00 | 1293.05 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2024-06-06 09:15:00 | 1293.00 | 2024-06-21 09:15:00 | 1422.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 1682.10 | 2025-02-10 14:15:00 | 1699.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-02-11 10:00:00 | 1689.40 | 2025-02-18 13:15:00 | 1698.70 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-02-11 11:30:00 | 1689.30 | 2025-02-18 13:15:00 | 1698.70 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-02-19 09:15:00 | 1676.05 | 2025-02-19 10:15:00 | 1697.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-02-20 12:00:00 | 1670.80 | 2025-02-27 09:15:00 | 1587.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-20 12:00:00 | 1670.80 | 2025-02-28 10:15:00 | 1503.72 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-15 10:00:00 | 1510.80 | 2025-09-16 14:15:00 | 1533.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1493.70 | 2025-09-26 11:15:00 | 1419.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 09:15:00 | 1493.70 | 2025-10-09 12:15:00 | 1473.00 | STOP_HIT | 0.50 | 1.39% |
| SELL | retest2 | 2025-11-27 11:45:00 | 1510.20 | 2025-12-02 14:15:00 | 1535.70 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-11-27 12:15:00 | 1512.20 | 2025-12-02 14:15:00 | 1535.70 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-09 09:30:00 | 1437.50 | 2026-04-15 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2026-04-10 09:15:00 | 1440.80 | 2026-04-15 09:15:00 | 1462.10 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-04-22 10:15:00 | 1437.50 | 2026-04-22 13:15:00 | 1471.80 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-04-22 14:15:00 | 1446.90 | 2026-04-22 15:15:00 | 1465.00 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-04-23 15:15:00 | 1416.90 | 2026-04-30 13:15:00 | 1480.00 | STOP_HIT | 1.00 | -4.45% |
