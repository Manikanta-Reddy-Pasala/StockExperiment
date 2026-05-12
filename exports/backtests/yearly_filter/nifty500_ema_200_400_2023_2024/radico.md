# Radico Khaitan Ltd (RADICO)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 3481.90
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 3 |
| ALERT3 | 55 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 39 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 38 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 35
- **Target hits / Stop hits / Partials:** 1 / 38 / 3
- **Avg / median % per leg:** -0.67% / -1.33%
- **Sum % (uncompounded):** -28.03%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 1 | 4.0% | 1 | 24 | 0 | -1.10% | -27.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 1 | 4.0% | 1 | 24 | 0 | -1.10% | -27.4% |
| SELL (all) | 17 | 6 | 35.3% | 0 | 14 | 3 | -0.03% | -0.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 6 | 35.3% | 0 | 14 | 3 | -0.03% | -0.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 7 | 16.7% | 1 | 38 | 3 | -0.67% | -28.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 12:15:00 | 1212.30 | 1277.65 | 1277.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 13:15:00 | 1206.85 | 1276.95 | 1277.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-09 09:15:00 | 1239.25 | 1225.93 | 1245.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-09 09:45:00 | 1241.80 | 1225.93 | 1245.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 09:15:00 | 1250.65 | 1225.67 | 1243.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 10:00:00 | 1250.65 | 1225.67 | 1243.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 10:15:00 | 1246.80 | 1225.88 | 1243.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-11 11:15:00 | 1239.00 | 1225.88 | 1243.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-12 11:15:00 | 1235.75 | 1226.29 | 1243.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 14:15:00 | 1258.90 | 1227.21 | 1243.44 | SL hit (close>static) qty=1.00 sl=1256.60 alert=retest2 |

### Cycle 2 — BUY (started 2023-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-07 13:15:00 | 1351.25 | 1247.97 | 1247.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-07 14:15:00 | 1364.00 | 1249.12 | 1248.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-17 12:15:00 | 1614.10 | 1620.95 | 1542.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-17 12:45:00 | 1614.95 | 1620.95 | 1542.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 1631.20 | 1681.88 | 1627.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 09:30:00 | 1639.55 | 1681.88 | 1627.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 1624.45 | 1681.31 | 1627.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:00:00 | 1624.45 | 1681.31 | 1627.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 1621.15 | 1680.71 | 1627.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 11:30:00 | 1621.75 | 1680.71 | 1627.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 1620.00 | 1678.95 | 1627.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 1620.00 | 1678.95 | 1627.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 1621.60 | 1678.38 | 1627.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 09:15:00 | 1625.00 | 1678.38 | 1627.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 10:15:00 | 1625.00 | 1677.84 | 1627.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 11:45:00 | 1625.75 | 1676.80 | 1627.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 14:00:00 | 1625.60 | 1675.79 | 1627.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 09:15:00 | 1613.10 | 1674.18 | 1626.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-26 09:15:00 | 1613.10 | 1674.18 | 1626.93 | SL hit (close<static) qty=1.00 sl=1614.15 alert=retest2 |

### Cycle 3 — SELL (started 2024-03-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-15 10:15:00 | 1551.75 | 1601.27 | 1601.27 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 12:15:00 | 1667.80 | 1601.30 | 1601.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-26 10:15:00 | 1672.85 | 1606.38 | 1603.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 10:15:00 | 1712.30 | 1716.67 | 1680.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-07 10:30:00 | 1720.85 | 1716.67 | 1680.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 1680.75 | 1716.11 | 1680.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:45:00 | 1678.45 | 1716.11 | 1680.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 13:15:00 | 1692.30 | 1715.87 | 1680.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-07 14:30:00 | 1701.75 | 1715.77 | 1680.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-08 09:45:00 | 1697.45 | 1715.48 | 1680.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-08 13:15:00 | 1666.75 | 1714.08 | 1680.93 | SL hit (close<static) qty=1.00 sl=1674.95 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 14:15:00 | 1578.35 | 1665.02 | 1665.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 1537.45 | 1658.75 | 1662.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 10:15:00 | 1708.05 | 1653.54 | 1659.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 10:15:00 | 1708.05 | 1653.54 | 1659.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 10:15:00 | 1708.05 | 1653.54 | 1659.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 11:00:00 | 1708.05 | 1653.54 | 1659.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 11:15:00 | 1707.55 | 1654.08 | 1659.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 12:00:00 | 1707.55 | 1654.08 | 1659.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2024-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-10 12:15:00 | 1701.95 | 1664.72 | 1664.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 14:15:00 | 1714.90 | 1665.60 | 1665.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 13:15:00 | 1726.35 | 1738.10 | 1709.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-03 14:00:00 | 1726.35 | 1738.10 | 1709.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 11:15:00 | 1715.30 | 1736.62 | 1710.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:45:00 | 1713.10 | 1736.62 | 1710.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 12:15:00 | 1712.20 | 1736.38 | 1710.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 12:45:00 | 1710.45 | 1736.38 | 1710.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 13:15:00 | 1711.60 | 1736.13 | 1710.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:00:00 | 1711.60 | 1736.13 | 1710.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 14:15:00 | 1717.50 | 1735.95 | 1710.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 14:30:00 | 1708.00 | 1735.95 | 1710.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 1690.05 | 1735.32 | 1710.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:45:00 | 1692.40 | 1735.32 | 1710.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 1679.15 | 1734.76 | 1710.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 1679.15 | 1734.76 | 1710.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 1700.80 | 1711.04 | 1702.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:30:00 | 1702.95 | 1711.04 | 1702.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 1705.05 | 1710.99 | 1702.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:30:00 | 1701.50 | 1710.99 | 1702.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 1702.05 | 1710.90 | 1702.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 1713.80 | 1710.90 | 1702.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 1711.65 | 1710.90 | 1702.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 1704.65 | 1710.90 | 1702.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 1700.90 | 1710.80 | 1702.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 1700.90 | 1710.80 | 1702.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 1718.40 | 1710.88 | 1702.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 09:30:00 | 1741.05 | 1706.21 | 1700.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-24 14:45:00 | 1725.40 | 1707.03 | 1701.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 09:30:00 | 1726.35 | 1707.26 | 1701.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:00:00 | 1732.10 | 1707.56 | 1701.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1702.55 | 1715.58 | 1707.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 1690.00 | 1715.58 | 1707.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1682.05 | 1715.25 | 1707.12 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 1682.05 | 1715.25 | 1707.12 | SL hit (close<static) qty=1.00 sl=1697.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 10:15:00 | 1679.55 | 1700.82 | 1700.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1666.00 | 1700.03 | 1700.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-19 12:15:00 | 1695.20 | 1692.87 | 1696.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 12:15:00 | 1695.20 | 1692.87 | 1696.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 12:15:00 | 1695.20 | 1692.87 | 1696.62 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-22 15:15:00 | 1763.00 | 1700.19 | 1699.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-23 09:15:00 | 1781.15 | 1701.00 | 1700.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 2017.30 | 2020.81 | 1924.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-04 09:30:00 | 2006.75 | 2020.81 | 1924.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 2391.55 | 2469.36 | 2374.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:45:00 | 2383.80 | 2469.36 | 2374.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 2328.80 | 2463.37 | 2374.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 2328.80 | 2463.37 | 2374.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 2327.60 | 2462.02 | 2374.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 13:00:00 | 2342.60 | 2375.33 | 2343.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 13:45:00 | 2351.95 | 2375.17 | 2343.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 09:15:00 | 2317.20 | 2374.00 | 2343.24 | SL hit (close<static) qty=1.00 sl=2318.65 alert=retest2 |

### Cycle 9 — SELL (started 2025-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 11:15:00 | 2075.80 | 2317.17 | 2317.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 2048.15 | 2250.32 | 2277.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 14:15:00 | 2200.20 | 2185.88 | 2238.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 15:00:00 | 2200.20 | 2185.88 | 2238.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 2210.65 | 2133.07 | 2193.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:45:00 | 2210.00 | 2133.07 | 2193.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 2196.00 | 2133.70 | 2193.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-10 10:15:00 | 2178.80 | 2139.23 | 2194.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 09:45:00 | 2179.00 | 2142.70 | 2194.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 12:45:00 | 2178.65 | 2143.79 | 2193.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 14:00:00 | 2179.85 | 2144.15 | 2193.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2160.00 | 2145.03 | 2193.45 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 2230.30 | 2150.28 | 2192.87 | SL hit (close>static) qty=1.00 sl=2212.85 alert=retest2 |

### Cycle 10 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 2357.00 | 2221.74 | 2221.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 14:15:00 | 2426.00 | 2227.30 | 2224.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 2233.45 | 2260.11 | 2242.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 2233.45 | 2260.11 | 2242.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 2233.45 | 2260.11 | 2242.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 2317.05 | 2259.00 | 2242.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-06 09:15:00 | 2548.76 | 2387.82 | 2329.04 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2026-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 11:15:00 | 2933.90 | 3139.49 | 3140.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 14:15:00 | 2903.30 | 3133.01 | 3136.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-06 14:15:00 | 2772.70 | 2747.51 | 2853.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-06 14:45:00 | 2756.10 | 2747.51 | 2853.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 2847.80 | 2750.18 | 2845.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 2847.80 | 2750.18 | 2845.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 2844.40 | 2751.12 | 2845.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:15:00 | 2870.90 | 2751.12 | 2845.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 2877.20 | 2752.37 | 2845.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 2877.20 | 2752.37 | 2845.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 2881.80 | 2753.66 | 2845.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 09:15:00 | 2815.70 | 2753.66 | 2845.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 2858.40 | 2758.54 | 2845.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 14:15:00 | 2715.48 | 2766.70 | 2841.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 2766.60 | 2766.43 | 2840.52 | SL hit (close>ema200) qty=0.50 sl=2766.43 alert=retest2 |

### Cycle 12 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 3201.90 | 2821.63 | 2821.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 14:15:00 | 3255.00 | 2833.39 | 2827.33 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-10-11 11:15:00 | 1239.00 | 2023-10-12 14:15:00 | 1258.90 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2023-10-12 11:15:00 | 1235.75 | 2023-10-12 14:15:00 | 1258.90 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2023-10-17 14:45:00 | 1235.85 | 2023-10-18 11:15:00 | 1258.65 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2023-10-20 14:45:00 | 1239.10 | 2023-10-23 15:15:00 | 1177.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 14:45:00 | 1239.10 | 2023-10-27 13:15:00 | 1232.60 | STOP_HIT | 0.50 | 0.52% |
| SELL | retest2 | 2023-10-31 15:00:00 | 1221.65 | 2023-11-02 12:15:00 | 1254.00 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2023-11-01 15:00:00 | 1223.75 | 2023-11-02 12:15:00 | 1254.00 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2024-02-23 09:15:00 | 1625.00 | 2024-02-26 09:15:00 | 1613.10 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-02-23 10:15:00 | 1625.00 | 2024-02-26 09:15:00 | 1613.10 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2024-02-23 11:45:00 | 1625.75 | 2024-02-26 09:15:00 | 1613.10 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-02-23 14:00:00 | 1625.60 | 2024-02-26 09:15:00 | 1613.10 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2024-02-27 12:15:00 | 1617.20 | 2024-02-27 13:15:00 | 1601.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-02-27 15:00:00 | 1619.95 | 2024-02-28 14:15:00 | 1603.85 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-02-28 11:30:00 | 1622.75 | 2024-02-28 14:15:00 | 1603.85 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-03-02 09:45:00 | 1616.90 | 2024-03-04 09:15:00 | 1586.75 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2024-05-07 14:30:00 | 1701.75 | 2024-05-08 13:15:00 | 1666.75 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2024-05-08 09:45:00 | 1697.45 | 2024-05-08 13:15:00 | 1666.75 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-05-16 11:00:00 | 1697.85 | 2024-05-24 09:15:00 | 1674.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-05-22 09:45:00 | 1696.90 | 2024-05-24 09:15:00 | 1674.00 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-07-24 09:30:00 | 1741.05 | 2024-08-02 09:15:00 | 1682.05 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2024-07-24 14:45:00 | 1725.40 | 2024-08-02 09:15:00 | 1682.05 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2024-07-25 09:30:00 | 1726.35 | 2024-08-02 09:15:00 | 1682.05 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2024-07-25 12:00:00 | 1732.10 | 2024-08-02 09:15:00 | 1682.05 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2024-08-02 15:00:00 | 1709.40 | 2024-08-05 10:15:00 | 1656.85 | STOP_HIT | 1.00 | -3.07% |
| BUY | retest2 | 2024-08-06 10:00:00 | 1693.85 | 2024-08-06 13:15:00 | 1677.90 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-08-06 10:45:00 | 1700.55 | 2024-08-06 13:15:00 | 1677.90 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-08-06 11:30:00 | 1693.95 | 2024-08-06 13:15:00 | 1677.90 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-08-09 14:45:00 | 1701.55 | 2024-08-12 09:15:00 | 1665.95 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-08-13 10:00:00 | 1687.50 | 2024-08-13 10:15:00 | 1679.55 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-01-20 13:00:00 | 2342.60 | 2025-01-21 09:15:00 | 2317.20 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-01-20 13:45:00 | 2351.95 | 2025-01-21 09:15:00 | 2317.20 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-03-10 10:15:00 | 2178.80 | 2025-03-17 09:15:00 | 2230.30 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-03-11 09:45:00 | 2179.00 | 2025-03-17 09:15:00 | 2230.30 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2025-03-11 12:45:00 | 2178.65 | 2025-03-17 09:15:00 | 2230.30 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-03-11 14:00:00 | 2179.85 | 2025-03-17 09:15:00 | 2230.30 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-04-08 09:15:00 | 2317.05 | 2025-05-06 09:15:00 | 2548.76 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-12 09:15:00 | 2815.70 | 2026-03-17 14:15:00 | 2715.48 | PARTIAL | 0.50 | 3.56% |
| SELL | retest2 | 2026-03-12 09:15:00 | 2815.70 | 2026-03-18 09:15:00 | 2766.60 | STOP_HIT | 0.50 | 1.74% |
| SELL | retest2 | 2026-03-12 15:00:00 | 2858.40 | 2026-03-19 09:15:00 | 2674.91 | PARTIAL | 0.50 | 6.42% |
| SELL | retest2 | 2026-03-12 15:00:00 | 2858.40 | 2026-03-25 13:15:00 | 2734.80 | STOP_HIT | 0.50 | 4.32% |
| SELL | retest2 | 2026-04-15 10:45:00 | 2865.70 | 2026-04-15 12:15:00 | 2900.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2026-04-15 11:45:00 | 2867.90 | 2026-04-15 12:15:00 | 2900.00 | STOP_HIT | 1.00 | -1.12% |
