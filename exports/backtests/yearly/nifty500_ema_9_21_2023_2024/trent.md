# Trent Ltd. (TRENT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4249.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 204 |
| ALERT1 | 138 |
| ALERT2 | 137 |
| ALERT2_SKIP | 65 |
| ALERT3 | 364 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 150 |
| PARTIAL | 13 |
| TARGET_HIT | 7 |
| STOP_HIT | 150 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 169 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 113
- **Target hits / Stop hits / Partials:** 7 / 149 / 13
- **Avg / median % per leg:** 0.37% / -0.71%
- **Sum % (uncompounded):** 63.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 95 | 24 | 25.3% | 7 | 88 | 0 | 0.06% | 5.8% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 3 | 0 | -2.08% | -6.2% |
| BUY @ 3rd Alert (retest2) | 92 | 22 | 23.9% | 7 | 85 | 0 | 0.13% | 12.1% |
| SELL (all) | 74 | 32 | 43.2% | 0 | 61 | 13 | 0.77% | 57.3% |
| SELL @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.89% | 7.6% |
| SELL @ 3rd Alert (retest2) | 70 | 30 | 42.9% | 0 | 58 | 12 | 0.71% | 49.7% |
| retest1 (combined) | 7 | 4 | 57.1% | 0 | 6 | 1 | 0.19% | 1.3% |
| retest2 (combined) | 162 | 52 | 32.1% | 7 | 143 | 12 | 0.38% | 61.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-18 15:15:00 | 1489.00 | 1497.86 | 1498.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-19 09:15:00 | 1476.60 | 1493.61 | 1496.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-19 12:15:00 | 1493.00 | 1491.24 | 1494.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-19 13:00:00 | 1493.00 | 1491.24 | 1494.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 14:15:00 | 1493.65 | 1491.59 | 1494.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 11:15:00 | 1483.00 | 1491.34 | 1493.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-22 12:15:00 | 1484.45 | 1491.25 | 1493.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-22 14:15:00 | 1499.20 | 1491.76 | 1492.89 | SL hit (close>static) qty=1.00 sl=1498.30 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-23 09:15:00 | 1503.80 | 1495.48 | 1494.46 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 13:15:00 | 1493.40 | 1496.16 | 1496.21 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 14:15:00 | 1501.00 | 1497.13 | 1496.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 15:15:00 | 1503.00 | 1498.30 | 1497.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-31 14:15:00 | 1554.35 | 1564.62 | 1552.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 14:15:00 | 1554.35 | 1564.62 | 1552.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 14:15:00 | 1554.35 | 1564.62 | 1552.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-31 15:00:00 | 1554.35 | 1564.62 | 1552.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 1553.85 | 1562.47 | 1552.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 10:30:00 | 1566.00 | 1562.11 | 1554.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 12:45:00 | 1567.85 | 1564.19 | 1556.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-01 15:15:00 | 1569.00 | 1564.16 | 1557.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-12 10:15:00 | 1589.05 | 1597.48 | 1597.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-12 10:15:00 | 1589.05 | 1597.48 | 1597.98 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2023-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 09:15:00 | 1663.15 | 1609.94 | 1602.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 14:15:00 | 1684.00 | 1651.23 | 1628.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-15 09:15:00 | 1691.00 | 1694.85 | 1670.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-15 09:30:00 | 1691.95 | 1694.85 | 1670.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 09:15:00 | 1700.00 | 1692.00 | 1680.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 12:00:00 | 1717.00 | 1698.50 | 1685.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 13:00:00 | 1714.90 | 1701.78 | 1688.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-16 15:00:00 | 1713.50 | 1706.23 | 1692.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 09:45:00 | 1716.90 | 1710.57 | 1697.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 1702.00 | 1709.11 | 1700.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 14:00:00 | 1702.00 | 1709.11 | 1700.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 14:15:00 | 1709.35 | 1709.16 | 1701.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-19 15:15:00 | 1711.60 | 1709.16 | 1701.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 13:15:00 | 1713.90 | 1709.81 | 1705.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-20 14:15:00 | 1711.20 | 1709.97 | 1705.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 10:15:00 | 1713.00 | 1709.09 | 1708.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 1710.35 | 1709.34 | 1708.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-06-22 12:15:00 | 1702.80 | 1707.25 | 1707.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-22 12:15:00 | 1702.80 | 1707.25 | 1707.76 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-22 14:15:00 | 1712.40 | 1708.12 | 1708.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-23 09:15:00 | 1720.85 | 1711.63 | 1709.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-23 11:15:00 | 1704.90 | 1710.51 | 1709.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-23 11:15:00 | 1704.90 | 1710.51 | 1709.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 11:15:00 | 1704.90 | 1710.51 | 1709.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 12:00:00 | 1704.90 | 1710.51 | 1709.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 12:15:00 | 1711.05 | 1710.62 | 1709.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-23 13:45:00 | 1716.95 | 1711.93 | 1710.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-04 12:15:00 | 1746.90 | 1757.42 | 1758.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2023-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 12:15:00 | 1746.90 | 1757.42 | 1758.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 13:15:00 | 1746.05 | 1755.15 | 1757.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-05 14:15:00 | 1731.00 | 1726.76 | 1738.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-05 15:00:00 | 1731.00 | 1726.76 | 1738.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 09:15:00 | 1733.75 | 1728.07 | 1736.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 09:45:00 | 1736.05 | 1728.07 | 1736.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 10:15:00 | 1736.90 | 1729.83 | 1736.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 10:45:00 | 1737.95 | 1729.83 | 1736.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 11:15:00 | 1733.20 | 1730.51 | 1736.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 11:30:00 | 1736.00 | 1730.51 | 1736.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 13:15:00 | 1736.70 | 1731.70 | 1735.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 14:00:00 | 1736.70 | 1731.70 | 1735.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-06 14:15:00 | 1757.05 | 1736.77 | 1737.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-06 15:00:00 | 1757.05 | 1736.77 | 1737.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2023-07-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 15:15:00 | 1750.00 | 1739.42 | 1739.00 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 10:15:00 | 1729.95 | 1737.99 | 1738.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 12:15:00 | 1707.90 | 1730.62 | 1734.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-11 09:15:00 | 1694.00 | 1689.68 | 1704.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-11 09:15:00 | 1694.00 | 1689.68 | 1704.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 1694.00 | 1689.68 | 1704.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 09:45:00 | 1697.50 | 1689.68 | 1704.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 13:15:00 | 1694.35 | 1693.55 | 1702.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 13:45:00 | 1695.10 | 1693.55 | 1702.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 1688.85 | 1692.22 | 1699.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-12 11:00:00 | 1675.70 | 1688.92 | 1697.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 10:30:00 | 1677.35 | 1684.10 | 1690.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 13:45:00 | 1677.60 | 1680.58 | 1687.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-14 10:15:00 | 1674.95 | 1676.74 | 1683.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 1691.15 | 1674.68 | 1679.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 15:00:00 | 1691.15 | 1674.68 | 1679.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 1690.00 | 1677.75 | 1680.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:15:00 | 1685.90 | 1677.75 | 1680.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-07-17 09:15:00 | 1711.60 | 1684.52 | 1683.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2023-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 09:15:00 | 1711.60 | 1684.52 | 1683.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 11:15:00 | 1726.05 | 1718.25 | 1710.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 13:15:00 | 1715.35 | 1719.06 | 1712.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-21 14:00:00 | 1715.35 | 1719.06 | 1712.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 14:15:00 | 1715.00 | 1718.25 | 1712.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 14:45:00 | 1709.90 | 1718.25 | 1712.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 15:15:00 | 1716.00 | 1717.80 | 1713.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:15:00 | 1711.70 | 1717.80 | 1713.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 09:15:00 | 1719.05 | 1718.05 | 1713.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 09:30:00 | 1716.55 | 1718.05 | 1713.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 11:15:00 | 1717.35 | 1720.14 | 1715.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 12:00:00 | 1717.35 | 1720.14 | 1715.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 12:15:00 | 1719.90 | 1720.09 | 1715.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 14:15:00 | 1725.75 | 1719.69 | 1716.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-24 14:15:00 | 1706.95 | 1717.15 | 1715.31 | SL hit (close<static) qty=1.00 sl=1715.80 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-25 09:15:00 | 1704.25 | 1712.61 | 1713.44 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 12:15:00 | 1710.00 | 1709.29 | 1709.24 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 13:15:00 | 1707.45 | 1708.92 | 1709.08 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 15:15:00 | 1710.00 | 1709.19 | 1709.18 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-28 10:15:00 | 1708.30 | 1709.01 | 1709.10 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 11:15:00 | 1712.80 | 1709.77 | 1709.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-28 12:15:00 | 1714.05 | 1710.63 | 1709.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 09:15:00 | 1736.80 | 1744.44 | 1733.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 10:00:00 | 1736.80 | 1744.44 | 1733.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 1720.45 | 1739.65 | 1732.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 11:00:00 | 1720.45 | 1739.65 | 1732.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 11:15:00 | 1710.00 | 1733.72 | 1730.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 11:45:00 | 1707.30 | 1733.72 | 1730.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2023-08-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-01 13:15:00 | 1716.05 | 1726.94 | 1727.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-01 14:15:00 | 1709.45 | 1723.44 | 1725.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-04 09:15:00 | 1704.95 | 1692.79 | 1699.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-04 09:15:00 | 1704.95 | 1692.79 | 1699.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 09:15:00 | 1704.95 | 1692.79 | 1699.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 09:30:00 | 1706.00 | 1692.79 | 1699.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 10:15:00 | 1700.00 | 1694.24 | 1699.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 11:15:00 | 1711.65 | 1694.24 | 1699.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 11:15:00 | 1715.00 | 1698.39 | 1701.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-04 12:00:00 | 1715.00 | 1698.39 | 1701.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-04 12:15:00 | 1703.50 | 1699.41 | 1701.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 13:30:00 | 1699.85 | 1699.86 | 1701.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-07 09:30:00 | 1701.45 | 1698.42 | 1700.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-08 14:15:00 | 1707.00 | 1694.17 | 1694.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2023-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 14:15:00 | 1707.00 | 1694.17 | 1694.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-08 15:15:00 | 1718.00 | 1698.94 | 1696.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-18 14:15:00 | 1977.00 | 1987.17 | 1967.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-18 15:00:00 | 1977.00 | 1987.17 | 1967.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 15:15:00 | 1965.00 | 1982.74 | 1966.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-21 09:15:00 | 1972.00 | 1982.74 | 1966.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 09:15:00 | 1973.00 | 1980.79 | 1967.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 10:15:00 | 1982.50 | 1980.79 | 1967.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-25 12:15:00 | 2009.55 | 2031.33 | 2032.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2023-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 12:15:00 | 2009.55 | 2031.33 | 2032.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-28 09:15:00 | 2003.05 | 2018.74 | 2025.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 2015.95 | 2007.48 | 2014.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-29 09:15:00 | 2015.95 | 2007.48 | 2014.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 09:15:00 | 2015.95 | 2007.48 | 2014.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 09:45:00 | 2017.20 | 2007.48 | 2014.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 2022.05 | 2010.39 | 2015.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 10:30:00 | 2022.05 | 2010.39 | 2015.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 11:15:00 | 2023.40 | 2012.99 | 2016.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 11:45:00 | 2030.50 | 2012.99 | 2016.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 14:15:00 | 2050.00 | 2023.50 | 2020.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-30 09:15:00 | 2060.65 | 2035.17 | 2026.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-31 09:15:00 | 2049.40 | 2055.33 | 2043.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-31 10:00:00 | 2049.40 | 2055.33 | 2043.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 14:15:00 | 2052.05 | 2054.28 | 2047.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 15:00:00 | 2052.05 | 2054.28 | 2047.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 15:15:00 | 2053.30 | 2054.08 | 2048.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 09:15:00 | 2082.40 | 2054.08 | 2048.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-01 14:15:00 | 2057.40 | 2053.54 | 2050.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-04 11:15:00 | 2035.55 | 2049.86 | 2050.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2023-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-04 11:15:00 | 2035.55 | 2049.86 | 2050.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-04 13:15:00 | 2034.45 | 2045.29 | 2048.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-05 09:15:00 | 2048.50 | 2041.65 | 2045.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-05 09:15:00 | 2048.50 | 2041.65 | 2045.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 09:15:00 | 2048.50 | 2041.65 | 2045.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-05 09:45:00 | 2048.50 | 2041.65 | 2045.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-05 10:15:00 | 2050.80 | 2043.48 | 2045.84 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 12:15:00 | 2065.00 | 2050.24 | 2048.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 14:15:00 | 2084.10 | 2070.49 | 2064.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 11:15:00 | 2079.10 | 2080.71 | 2071.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-08 11:30:00 | 2080.00 | 2080.71 | 2071.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 12:15:00 | 2077.30 | 2080.03 | 2072.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 13:00:00 | 2077.30 | 2080.03 | 2072.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 14:15:00 | 2088.35 | 2080.74 | 2073.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-08 14:30:00 | 2071.50 | 2080.74 | 2073.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 11:15:00 | 2090.25 | 2087.17 | 2079.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 11:45:00 | 2092.30 | 2087.17 | 2079.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-11 14:15:00 | 2088.25 | 2087.63 | 2081.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-11 15:00:00 | 2088.25 | 2087.63 | 2081.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 2049.30 | 2080.04 | 2079.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 2054.40 | 2080.04 | 2079.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2023-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 10:15:00 | 2068.60 | 2077.75 | 2078.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 11:15:00 | 2036.60 | 2069.52 | 2074.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 2064.25 | 2055.34 | 2062.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 11:15:00 | 2064.25 | 2055.34 | 2062.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 2064.25 | 2055.34 | 2062.92 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 15:15:00 | 2070.00 | 2062.05 | 2061.15 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 09:15:00 | 2053.00 | 2060.24 | 2060.41 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-18 10:15:00 | 2064.20 | 2061.03 | 2060.75 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 11:15:00 | 2056.05 | 2060.04 | 2060.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 12:15:00 | 2049.95 | 2058.02 | 2059.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-20 10:15:00 | 2060.05 | 2055.00 | 2057.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-20 10:15:00 | 2060.05 | 2055.00 | 2057.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-20 10:15:00 | 2060.05 | 2055.00 | 2057.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-20 10:30:00 | 2058.15 | 2055.00 | 2057.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-20 11:15:00 | 2075.60 | 2059.12 | 2058.84 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 11:15:00 | 2051.95 | 2060.28 | 2060.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-21 14:15:00 | 2041.50 | 2052.53 | 2056.69 | Break + close below crossover candle low |

### Cycle 32 — BUY (started 2023-09-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-21 15:15:00 | 2099.90 | 2062.00 | 2060.62 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2023-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-22 13:15:00 | 2057.40 | 2059.76 | 2059.89 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 14:15:00 | 2065.10 | 2060.82 | 2060.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-22 15:15:00 | 2070.00 | 2062.66 | 2061.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-25 09:15:00 | 2051.20 | 2060.37 | 2060.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-25 09:15:00 | 2051.20 | 2060.37 | 2060.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 2051.20 | 2060.37 | 2060.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-25 10:00:00 | 2051.20 | 2060.37 | 2060.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2023-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 10:15:00 | 2055.00 | 2059.29 | 2059.84 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 11:15:00 | 2079.15 | 2063.27 | 2061.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 12:15:00 | 2106.25 | 2071.86 | 2065.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 15:15:00 | 2143.00 | 2148.43 | 2122.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-27 09:15:00 | 2143.70 | 2148.43 | 2122.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 2122.00 | 2143.15 | 2122.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:00:00 | 2122.00 | 2143.15 | 2122.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 2095.50 | 2133.62 | 2119.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-27 11:00:00 | 2095.50 | 2133.62 | 2119.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 2096.00 | 2126.09 | 2117.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 13:00:00 | 2102.80 | 2121.44 | 2116.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 13:30:00 | 2107.95 | 2119.69 | 2116.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-28 10:15:00 | 2103.10 | 2114.18 | 2114.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-09-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 10:15:00 | 2103.10 | 2114.18 | 2114.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-28 12:15:00 | 2081.00 | 2105.29 | 2110.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-29 12:15:00 | 2076.90 | 2076.57 | 2089.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-29 12:30:00 | 2083.85 | 2076.57 | 2089.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 13:15:00 | 2086.95 | 2078.65 | 2089.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-29 14:00:00 | 2086.95 | 2078.65 | 2089.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 2067.85 | 2077.45 | 2086.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-03 15:15:00 | 2055.00 | 2066.99 | 2077.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-06 10:00:00 | 2054.95 | 2044.23 | 2047.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-06 11:15:00 | 2061.00 | 2049.98 | 2049.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2023-10-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-06 11:15:00 | 2061.00 | 2049.98 | 2049.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 13:15:00 | 2069.00 | 2055.15 | 2052.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 2047.80 | 2058.92 | 2055.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 2047.80 | 2058.92 | 2055.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 2047.80 | 2058.92 | 2055.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 11:15:00 | 2058.95 | 2058.15 | 2055.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-09 14:45:00 | 2086.25 | 2059.36 | 2056.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 15:15:00 | 2050.00 | 2057.55 | 2057.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-10 15:15:00 | 2050.00 | 2057.55 | 2057.81 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 2067.65 | 2059.57 | 2058.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 11:15:00 | 2075.90 | 2062.61 | 2060.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-12 09:15:00 | 2073.90 | 2077.77 | 2069.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-12 10:00:00 | 2073.90 | 2077.77 | 2069.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 11:15:00 | 2078.60 | 2077.70 | 2070.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 13:30:00 | 2085.40 | 2078.60 | 2072.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-13 09:15:00 | 2089.60 | 2078.94 | 2073.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-17 14:15:00 | 2080.55 | 2089.65 | 2089.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 14:15:00 | 2080.55 | 2089.65 | 2089.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 09:15:00 | 2057.00 | 2081.89 | 2086.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 11:15:00 | 2056.90 | 2051.36 | 2063.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-19 12:00:00 | 2056.90 | 2051.36 | 2063.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 13:15:00 | 2068.25 | 2054.53 | 2062.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 14:00:00 | 2068.25 | 2054.53 | 2062.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 2061.35 | 2055.89 | 2062.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-19 15:15:00 | 2059.50 | 2055.89 | 2062.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 1956.52 | 1988.23 | 2005.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-26 11:15:00 | 1990.00 | 1987.89 | 2002.64 | SL hit (close>ema200) qty=0.50 sl=1987.89 alert=retest2 |

### Cycle 42 — BUY (started 2023-10-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-26 15:15:00 | 2035.00 | 2014.05 | 2011.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-27 09:15:00 | 2125.65 | 2036.37 | 2022.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-30 09:15:00 | 2079.30 | 2097.58 | 2068.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-30 10:00:00 | 2079.30 | 2097.58 | 2068.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 2211.25 | 2221.83 | 2206.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-07 13:45:00 | 2255.95 | 2248.52 | 2222.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-11-08 09:15:00 | 2481.55 | 2346.37 | 2278.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-08 10:15:00 | 2824.70 | 2848.74 | 2848.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-08 12:15:00 | 2810.95 | 2836.98 | 2843.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 14:15:00 | 2847.50 | 2835.37 | 2841.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-08 14:15:00 | 2847.50 | 2835.37 | 2841.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 14:15:00 | 2847.50 | 2835.37 | 2841.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 15:00:00 | 2847.50 | 2835.37 | 2841.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 15:15:00 | 2844.45 | 2837.19 | 2841.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-11 09:15:00 | 2871.00 | 2837.19 | 2841.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — BUY (started 2023-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 09:15:00 | 2877.50 | 2845.25 | 2844.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 09:15:00 | 2924.95 | 2887.28 | 2875.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-15 14:15:00 | 2988.30 | 2990.89 | 2966.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-15 15:00:00 | 2988.30 | 2990.89 | 2966.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 09:15:00 | 2975.10 | 2988.20 | 2969.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 10:15:00 | 2962.20 | 2988.20 | 2969.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 10:15:00 | 2989.25 | 2988.41 | 2971.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 12:30:00 | 3005.00 | 2982.02 | 2976.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 14:00:00 | 2992.75 | 2984.17 | 2977.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 11:45:00 | 2998.00 | 2991.35 | 2984.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 14:15:00 | 2949.00 | 2976.18 | 2978.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 2949.00 | 2976.18 | 2978.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-21 12:15:00 | 2938.65 | 2955.29 | 2966.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 2953.95 | 2951.83 | 2962.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 15:00:00 | 2953.95 | 2951.83 | 2962.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 2969.90 | 2955.44 | 2963.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 2964.70 | 2955.44 | 2963.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 2965.95 | 2957.55 | 2963.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 12:15:00 | 2951.30 | 2961.34 | 2964.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 09:15:00 | 2977.40 | 2964.20 | 2964.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2023-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 09:15:00 | 2977.40 | 2964.20 | 2964.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 09:15:00 | 2992.50 | 2974.92 | 2970.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 3019.45 | 3021.14 | 3007.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 3019.45 | 3021.14 | 3007.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 3019.45 | 3021.14 | 3007.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 09:30:00 | 3005.15 | 3021.14 | 3007.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 10:15:00 | 3023.90 | 3036.55 | 3025.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 10:45:00 | 3020.40 | 3036.55 | 3025.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 11:15:00 | 3026.40 | 3034.52 | 3025.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 11:45:00 | 3016.30 | 3034.52 | 3025.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 12:15:00 | 3021.55 | 3031.93 | 3024.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 12:30:00 | 3020.55 | 3031.93 | 3024.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 13:15:00 | 3009.15 | 3027.37 | 3023.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 13:45:00 | 3005.85 | 3027.37 | 3023.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-01 14:15:00 | 2999.10 | 3021.72 | 3021.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-01 15:00:00 | 2999.10 | 3021.72 | 3021.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2024-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-01 15:15:00 | 3005.00 | 3018.37 | 3019.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 09:15:00 | 2968.65 | 3008.43 | 3015.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-02 13:15:00 | 2997.35 | 2993.37 | 3004.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-02 14:00:00 | 2997.35 | 2993.37 | 3004.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 14:15:00 | 2992.30 | 2993.16 | 3003.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-02 14:30:00 | 2994.90 | 2993.16 | 3003.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 15:15:00 | 2996.00 | 2993.73 | 3002.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:15:00 | 3014.00 | 2993.73 | 3002.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 09:15:00 | 2997.90 | 2994.56 | 3002.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 09:30:00 | 2995.40 | 2994.56 | 3002.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-03 10:15:00 | 3031.70 | 3001.99 | 3004.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-03 11:00:00 | 3031.70 | 3001.99 | 3004.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2024-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-03 11:15:00 | 3040.75 | 3009.74 | 3008.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-03 12:15:00 | 3071.80 | 3022.15 | 3013.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-04 13:15:00 | 3072.60 | 3074.28 | 3051.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-04 13:45:00 | 3071.40 | 3074.28 | 3051.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 10:15:00 | 3058.45 | 3071.47 | 3057.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-05 10:45:00 | 3065.00 | 3071.47 | 3057.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 11:15:00 | 3068.10 | 3070.80 | 3058.52 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 14:15:00 | 3042.10 | 3056.55 | 3057.35 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 3125.00 | 3068.94 | 3062.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 09:15:00 | 3171.00 | 3118.88 | 3095.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 14:15:00 | 3172.95 | 3179.80 | 3154.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-11 15:00:00 | 3172.95 | 3179.80 | 3154.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 15:15:00 | 3200.00 | 3202.21 | 3190.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:15:00 | 3184.70 | 3202.21 | 3190.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 3191.10 | 3199.99 | 3190.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:30:00 | 3183.00 | 3199.99 | 3190.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 3173.80 | 3194.75 | 3188.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 10:45:00 | 3175.00 | 3194.75 | 3188.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 11:15:00 | 3157.25 | 3187.25 | 3186.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 12:00:00 | 3157.25 | 3187.25 | 3186.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 12:15:00 | 3150.50 | 3179.90 | 3182.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-18 09:15:00 | 3118.90 | 3150.05 | 3162.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 10:15:00 | 3164.20 | 3152.88 | 3162.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 10:15:00 | 3164.20 | 3152.88 | 3162.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 3164.20 | 3152.88 | 3162.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 11:00:00 | 3164.20 | 3152.88 | 3162.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 3156.90 | 3153.69 | 3161.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-18 12:30:00 | 3141.50 | 3155.71 | 3162.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-18 14:15:00 | 3145.45 | 3156.17 | 3161.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-19 09:15:00 | 3167.15 | 3157.78 | 3160.98 | SL hit (close>static) qty=1.00 sl=3164.20 alert=retest2 |

### Cycle 52 — BUY (started 2024-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-19 12:15:00 | 3171.20 | 3164.05 | 3163.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 14:15:00 | 3215.95 | 3175.32 | 3168.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 14:15:00 | 3191.95 | 3205.51 | 3190.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-20 14:15:00 | 3191.95 | 3205.51 | 3190.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 3191.95 | 3205.51 | 3190.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 15:00:00 | 3191.95 | 3205.51 | 3190.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 3187.00 | 3201.81 | 3190.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 3211.85 | 3201.81 | 3190.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 3193.95 | 3200.24 | 3190.87 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2024-01-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 15:15:00 | 3160.00 | 3183.70 | 3186.75 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-01-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-24 12:15:00 | 3208.05 | 3188.81 | 3188.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-24 13:15:00 | 3237.35 | 3198.51 | 3192.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-25 10:15:00 | 3189.65 | 3204.60 | 3198.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 10:15:00 | 3189.65 | 3204.60 | 3198.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 3189.65 | 3204.60 | 3198.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:00:00 | 3189.65 | 3204.60 | 3198.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 3187.20 | 3201.12 | 3197.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 12:00:00 | 3187.20 | 3201.12 | 3197.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 12:15:00 | 3199.00 | 3200.70 | 3197.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 13:45:00 | 3205.70 | 3201.29 | 3198.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 14:15:00 | 3208.85 | 3201.29 | 3198.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-30 12:15:00 | 3141.60 | 3208.87 | 3215.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-30 12:15:00 | 3141.60 | 3208.87 | 3215.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-30 13:15:00 | 3095.80 | 3186.25 | 3205.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 14:15:00 | 3094.35 | 3091.04 | 3132.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-31 15:00:00 | 3094.35 | 3091.04 | 3132.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 3106.00 | 3093.63 | 3126.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-01 09:30:00 | 3120.95 | 3093.63 | 3126.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 3158.15 | 3105.38 | 3115.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 09:30:00 | 3154.55 | 3105.38 | 3115.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 10:15:00 | 3155.00 | 3115.31 | 3118.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-02 10:45:00 | 3162.75 | 3115.31 | 3118.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-02-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-02 11:15:00 | 3175.50 | 3127.35 | 3124.13 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 11:15:00 | 3057.45 | 3113.74 | 3120.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 12:15:00 | 3015.00 | 3093.99 | 3110.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 10:15:00 | 3103.70 | 3068.29 | 3088.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-06 10:15:00 | 3103.70 | 3068.29 | 3088.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 10:15:00 | 3103.70 | 3068.29 | 3088.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 11:00:00 | 3103.70 | 3068.29 | 3088.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 11:15:00 | 3030.00 | 3060.63 | 3083.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-07 09:15:00 | 3010.50 | 3045.79 | 3067.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-07 12:15:00 | 3228.10 | 3058.71 | 3064.25 | SL hit (close>static) qty=1.00 sl=3108.00 alert=retest2 |

### Cycle 58 — BUY (started 2024-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 13:15:00 | 3490.25 | 3145.02 | 3102.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-07 14:15:00 | 3620.35 | 3240.08 | 3150.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-09 10:15:00 | 3693.65 | 3731.37 | 3550.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-09 11:00:00 | 3693.65 | 3731.37 | 3550.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 09:15:00 | 4023.65 | 4031.40 | 4012.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 10:15:00 | 4003.05 | 4031.40 | 4012.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 10:15:00 | 3976.35 | 4020.39 | 4009.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 11:00:00 | 3976.35 | 4020.39 | 4009.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-21 11:15:00 | 3947.85 | 4005.88 | 4003.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-21 11:30:00 | 3958.80 | 4005.88 | 4003.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2024-02-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 12:15:00 | 3931.80 | 3991.07 | 3997.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 3903.05 | 3973.46 | 3988.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 14:15:00 | 3959.20 | 3897.67 | 3930.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-22 14:15:00 | 3959.20 | 3897.67 | 3930.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 3959.20 | 3897.67 | 3930.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 3959.20 | 3897.67 | 3930.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 15:15:00 | 3937.90 | 3905.72 | 3930.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 09:15:00 | 3922.80 | 3905.72 | 3930.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-27 09:15:00 | 3973.95 | 3882.61 | 3891.30 | SL hit (close>static) qty=1.00 sl=3966.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 10:15:00 | 3959.75 | 3898.04 | 3897.52 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-29 09:15:00 | 3843.85 | 3903.72 | 3907.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-29 10:15:00 | 3785.95 | 3880.17 | 3896.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 14:15:00 | 3880.50 | 3859.62 | 3879.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-29 14:15:00 | 3880.50 | 3859.62 | 3879.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 14:15:00 | 3880.50 | 3859.62 | 3879.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 15:00:00 | 3880.50 | 3859.62 | 3879.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 15:15:00 | 3888.00 | 3865.30 | 3880.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-01 09:15:00 | 3912.45 | 3865.30 | 3880.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-01 09:15:00 | 3886.85 | 3869.61 | 3880.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 10:45:00 | 3871.00 | 3870.40 | 3880.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-01 12:00:00 | 3865.65 | 3869.45 | 3878.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 14:15:00 | 3896.30 | 3884.62 | 3884.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2024-03-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-01 14:15:00 | 3896.30 | 3884.62 | 3884.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-02 09:15:00 | 3916.00 | 3892.33 | 3888.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 09:15:00 | 3887.35 | 3895.38 | 3890.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 09:15:00 | 3887.35 | 3895.38 | 3890.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 09:15:00 | 3887.35 | 3895.38 | 3890.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 09:30:00 | 3855.05 | 3895.38 | 3890.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 10:15:00 | 3906.20 | 3897.54 | 3892.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 12:00:00 | 3923.00 | 3902.64 | 3895.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-04 15:00:00 | 3921.70 | 3908.95 | 3900.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 09:45:00 | 3926.75 | 3910.55 | 3902.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-05 12:15:00 | 3861.70 | 3893.94 | 3896.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-05 12:15:00 | 3861.70 | 3893.94 | 3896.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 09:15:00 | 3798.35 | 3863.74 | 3880.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-06 13:15:00 | 3833.15 | 3829.05 | 3855.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-06 13:45:00 | 3833.35 | 3829.05 | 3855.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 14:15:00 | 3875.45 | 3838.33 | 3857.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-06 15:00:00 | 3875.45 | 3838.33 | 3857.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 3874.00 | 3845.46 | 3858.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 3872.50 | 3845.46 | 3858.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-03-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-07 10:15:00 | 3959.80 | 3880.90 | 3873.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 09:15:00 | 4155.00 | 3974.26 | 3926.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 11:15:00 | 3940.00 | 4004.04 | 3950.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 11:15:00 | 3940.00 | 4004.04 | 3950.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 11:15:00 | 3940.00 | 4004.04 | 3950.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 12:00:00 | 3940.00 | 4004.04 | 3950.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 12:15:00 | 4036.00 | 4010.43 | 3958.10 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2024-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 11:15:00 | 3917.55 | 3972.23 | 3974.98 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2024-03-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 10:15:00 | 4031.50 | 3972.24 | 3970.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 11:15:00 | 4065.00 | 3990.79 | 3979.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 15:15:00 | 4050.00 | 4050.25 | 4029.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-18 09:15:00 | 4067.65 | 4050.25 | 4029.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 4011.55 | 4042.51 | 4027.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 10:00:00 | 4011.55 | 4042.51 | 4027.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 4033.65 | 4040.74 | 4028.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 12:15:00 | 4063.30 | 4042.31 | 4030.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 13:00:00 | 4060.70 | 4045.99 | 4032.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-19 10:15:00 | 3983.25 | 4037.23 | 4035.01 | SL hit (close<static) qty=1.00 sl=4007.30 alert=retest2 |

### Cycle 67 — SELL (started 2024-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 11:15:00 | 3980.00 | 4025.78 | 4030.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-19 14:15:00 | 3946.10 | 4000.25 | 4016.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 3950.90 | 3949.13 | 3981.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-20 12:45:00 | 3946.00 | 3949.13 | 3981.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 14:15:00 | 3963.40 | 3953.66 | 3977.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 15:00:00 | 3963.40 | 3953.66 | 3977.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 3978.95 | 3959.23 | 3976.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 12:15:00 | 3962.00 | 3969.66 | 3978.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 15:15:00 | 3998.00 | 3984.69 | 3983.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — BUY (started 2024-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 15:15:00 | 3998.00 | 3984.69 | 3983.33 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-03-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-22 09:15:00 | 3952.85 | 3978.32 | 3980.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-22 10:15:00 | 3928.05 | 3968.27 | 3975.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-22 15:15:00 | 3959.10 | 3955.12 | 3965.18 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-03-26 09:15:00 | 3902.00 | 3955.12 | 3965.18 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 3895.35 | 3895.99 | 3921.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-27 09:30:00 | 3916.55 | 3895.99 | 3921.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 3907.00 | 3887.99 | 3903.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-03-28 09:15:00 | 3907.00 | 3887.99 | 3903.61 | SL hit (close>ema400) qty=1.00 sl=3903.61 alert=retest1 |

### Cycle 70 — BUY (started 2024-03-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 12:15:00 | 3948.60 | 3917.97 | 3915.02 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-01 12:15:00 | 3882.70 | 3915.06 | 3917.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-01 13:15:00 | 3870.85 | 3906.21 | 3913.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 11:15:00 | 3897.85 | 3892.17 | 3898.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 11:15:00 | 3897.85 | 3892.17 | 3898.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 11:15:00 | 3897.85 | 3892.17 | 3898.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 12:00:00 | 3897.85 | 3892.17 | 3898.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 12:15:00 | 3895.10 | 3892.76 | 3898.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 13:15:00 | 3902.95 | 3892.76 | 3898.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 3894.25 | 3893.05 | 3897.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 13:30:00 | 3898.95 | 3893.05 | 3897.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 3929.95 | 3900.43 | 3900.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 15:00:00 | 3929.95 | 3900.43 | 3900.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2024-04-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-03 15:15:00 | 3920.00 | 3904.35 | 3902.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-04 09:15:00 | 3968.45 | 3917.17 | 3908.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 11:15:00 | 3956.90 | 3982.80 | 3956.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 11:15:00 | 3956.90 | 3982.80 | 3956.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 11:15:00 | 3956.90 | 3982.80 | 3956.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 11:45:00 | 3951.00 | 3982.80 | 3956.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 12:15:00 | 3939.95 | 3974.23 | 3955.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 13:00:00 | 3939.95 | 3974.23 | 3955.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 13:15:00 | 3934.05 | 3966.20 | 3953.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 14:00:00 | 3934.05 | 3966.20 | 3953.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 11:15:00 | 3961.15 | 3960.46 | 3954.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 11:30:00 | 3958.40 | 3960.46 | 3954.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 12:15:00 | 3959.85 | 3960.34 | 3954.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 13:15:00 | 3946.15 | 3960.34 | 3954.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 13:15:00 | 3950.00 | 3958.27 | 3954.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 13:30:00 | 3941.80 | 3958.27 | 3954.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-08 14:15:00 | 3929.85 | 3952.59 | 3952.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-08 15:00:00 | 3929.85 | 3952.59 | 3952.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2024-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 15:15:00 | 3930.00 | 3948.07 | 3950.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 12:15:00 | 3918.95 | 3936.29 | 3943.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 09:15:00 | 3927.60 | 3919.59 | 3931.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 09:15:00 | 3927.60 | 3919.59 | 3931.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 3927.60 | 3919.59 | 3931.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:30:00 | 3936.20 | 3919.59 | 3931.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 10:15:00 | 3958.65 | 3927.40 | 3934.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:45:00 | 3949.00 | 3927.40 | 3934.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 3951.80 | 3932.28 | 3935.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 12:15:00 | 3957.55 | 3932.28 | 3935.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-04-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 12:15:00 | 4089.90 | 3963.81 | 3949.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-12 10:15:00 | 4099.50 | 4029.20 | 3989.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-15 09:15:00 | 4057.95 | 4058.47 | 4025.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-15 09:15:00 | 4057.95 | 4058.47 | 4025.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-15 09:15:00 | 4057.95 | 4058.47 | 4025.07 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2024-04-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 15:15:00 | 3970.00 | 4010.88 | 4013.93 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-04-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-18 09:15:00 | 4117.00 | 4016.19 | 4011.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 11:15:00 | 4136.90 | 4055.06 | 4030.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-18 14:15:00 | 4073.30 | 4081.47 | 4051.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-18 15:00:00 | 4073.30 | 4081.47 | 4051.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-19 09:15:00 | 4016.20 | 4074.26 | 4053.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 13:00:00 | 4105.00 | 4061.08 | 4050.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-19 13:45:00 | 4088.65 | 4070.99 | 4056.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-29 14:15:00 | 4497.52 | 4317.97 | 4295.67 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 11:15:00 | 4448.55 | 4502.85 | 4503.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 12:15:00 | 4409.50 | 4470.15 | 4484.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 4455.40 | 4447.51 | 4467.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 09:15:00 | 4455.40 | 4447.51 | 4467.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 4455.40 | 4447.51 | 4467.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:45:00 | 4450.20 | 4447.51 | 4467.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 4424.65 | 4444.46 | 4462.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:30:00 | 4408.05 | 4440.51 | 4454.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 10:15:00 | 4494.05 | 4457.00 | 4454.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 10:15:00 | 4494.05 | 4457.00 | 4454.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 11:15:00 | 4520.70 | 4469.74 | 4460.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 15:15:00 | 4522.00 | 4525.03 | 4506.56 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-05-16 09:15:00 | 4605.45 | 4525.03 | 4506.56 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 4651.65 | 4662.05 | 4636.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:45:00 | 4636.60 | 4662.05 | 4636.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 4609.95 | 4651.63 | 4634.31 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-22 10:15:00 | 4609.95 | 4651.63 | 4634.31 | SL hit (close<ema400) qty=1.00 sl=4634.31 alert=retest1 |

### Cycle 79 — SELL (started 2024-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 13:15:00 | 4607.00 | 4622.01 | 4622.99 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 14:15:00 | 4644.30 | 4626.46 | 4624.93 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 15:15:00 | 4602.00 | 4621.57 | 4622.84 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 4645.25 | 4626.31 | 4624.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 13:15:00 | 4691.70 | 4652.81 | 4638.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 4672.85 | 4701.75 | 4681.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 4672.85 | 4701.75 | 4681.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 4672.85 | 4701.75 | 4681.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 4672.85 | 4701.75 | 4681.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 4657.85 | 4692.97 | 4678.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:00:00 | 4657.85 | 4692.97 | 4678.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 4659.60 | 4686.30 | 4677.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 11:30:00 | 4656.30 | 4686.30 | 4677.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 4689.95 | 4688.48 | 4680.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 15:00:00 | 4689.95 | 4688.48 | 4680.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 4688.00 | 4688.38 | 4681.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 4704.90 | 4688.38 | 4681.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 11:15:00 | 4624.05 | 4674.81 | 4676.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 4624.05 | 4674.81 | 4676.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 13:15:00 | 4615.45 | 4655.97 | 4667.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 4686.00 | 4650.70 | 4661.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 4686.00 | 4650.70 | 4661.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 4686.00 | 4650.70 | 4661.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 4686.00 | 4650.70 | 4661.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 4673.10 | 4655.18 | 4662.26 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 12:15:00 | 4702.50 | 4667.60 | 4666.86 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 4620.00 | 4659.44 | 4663.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 09:15:00 | 4549.05 | 4634.66 | 4649.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 4660.95 | 4608.14 | 4628.36 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 13:15:00 | 4660.95 | 4608.14 | 4628.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 13:15:00 | 4660.95 | 4608.14 | 4628.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:00:00 | 4660.95 | 4608.14 | 4628.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 4523.05 | 4591.13 | 4618.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 14:45:00 | 4691.35 | 4591.13 | 4618.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 4673.55 | 4596.23 | 4615.60 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 4682.00 | 4632.16 | 4628.73 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 4515.00 | 4623.07 | 4628.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 4350.00 | 4568.45 | 4603.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 4552.35 | 4551.62 | 4588.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 13:15:00 | 4552.35 | 4551.62 | 4588.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 13:15:00 | 4552.35 | 4551.62 | 4588.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 13:45:00 | 4598.15 | 4551.62 | 4588.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 4588.50 | 4554.74 | 4583.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 09:15:00 | 4601.60 | 4554.74 | 4583.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 4661.00 | 4575.99 | 4590.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 10:15:00 | 4739.50 | 4575.99 | 4590.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 4859.00 | 4632.59 | 4614.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 11:15:00 | 4882.10 | 4682.49 | 4639.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 12:15:00 | 4943.40 | 4951.23 | 4919.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 12:45:00 | 4944.25 | 4951.23 | 4919.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 4897.00 | 4936.19 | 4917.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 4897.00 | 4936.19 | 4917.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 4930.00 | 4934.95 | 4918.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:30:00 | 4962.50 | 4941.55 | 4923.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 12:15:00 | 5223.40 | 5290.41 | 5291.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — SELL (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 12:15:00 | 5223.40 | 5290.41 | 5291.54 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 10:15:00 | 5326.45 | 5289.39 | 5288.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 14:15:00 | 5392.85 | 5324.34 | 5306.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 10:15:00 | 5355.30 | 5356.12 | 5328.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 10:45:00 | 5366.25 | 5356.12 | 5328.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 5317.45 | 5346.31 | 5330.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:30:00 | 5330.35 | 5346.31 | 5330.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 5347.50 | 5346.55 | 5332.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 11:00:00 | 5358.90 | 5340.39 | 5332.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 09:45:00 | 5362.10 | 5355.38 | 5345.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 12:15:00 | 5316.75 | 5340.77 | 5340.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2024-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 12:15:00 | 5316.75 | 5340.77 | 5340.90 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 09:15:00 | 5412.85 | 5349.61 | 5343.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 10:15:00 | 5481.40 | 5375.97 | 5356.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 09:15:00 | 5420.95 | 5434.90 | 5400.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 5420.95 | 5434.90 | 5400.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 5420.95 | 5434.90 | 5400.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:30:00 | 5424.80 | 5434.90 | 5400.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 5429.95 | 5433.91 | 5403.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:30:00 | 5414.80 | 5433.91 | 5403.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 12:15:00 | 5487.50 | 5510.45 | 5490.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:00:00 | 5487.50 | 5510.45 | 5490.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 5481.25 | 5504.61 | 5490.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:15:00 | 5483.00 | 5504.61 | 5490.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 5496.90 | 5503.07 | 5490.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:45:00 | 5466.00 | 5503.07 | 5490.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 5504.00 | 5503.25 | 5491.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 5555.85 | 5503.25 | 5491.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 13:15:00 | 5560.25 | 5576.98 | 5578.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 13:15:00 | 5560.25 | 5576.98 | 5578.31 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 14:15:00 | 5590.85 | 5579.75 | 5579.45 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 5534.65 | 5573.06 | 5576.60 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 10:15:00 | 5618.15 | 5575.53 | 5571.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 14:15:00 | 5639.55 | 5603.49 | 5587.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 5668.25 | 5678.76 | 5657.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 12:00:00 | 5668.25 | 5678.76 | 5657.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 5652.45 | 5673.50 | 5657.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:45:00 | 5649.95 | 5673.50 | 5657.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 5641.45 | 5667.09 | 5655.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:45:00 | 5640.80 | 5667.09 | 5655.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 5642.00 | 5658.87 | 5653.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 5516.70 | 5658.87 | 5653.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 09:15:00 | 5500.05 | 5627.11 | 5639.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 5338.20 | 5569.33 | 5612.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 14:15:00 | 5202.25 | 5192.49 | 5270.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 15:00:00 | 5202.25 | 5192.49 | 5270.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 5181.55 | 5197.11 | 5259.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 5047.85 | 5207.40 | 5253.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 5123.30 | 5212.64 | 5248.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 5346.50 | 5253.05 | 5258.92 | SL hit (close>static) qty=1.00 sl=5282.85 alert=retest2 |

### Cycle 98 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 5317.45 | 5265.93 | 5264.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 11:15:00 | 5361.00 | 5284.94 | 5273.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 14:15:00 | 5282.95 | 5292.76 | 5280.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 14:15:00 | 5282.95 | 5292.76 | 5280.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 14:15:00 | 5282.95 | 5292.76 | 5280.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-24 15:00:00 | 5282.95 | 5292.76 | 5280.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 15:15:00 | 5286.00 | 5291.41 | 5281.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:15:00 | 5303.00 | 5291.41 | 5281.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 5250.35 | 5283.20 | 5278.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:45:00 | 5323.35 | 5289.91 | 5282.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 12:15:00 | 5310.95 | 5289.91 | 5282.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 15:15:00 | 5325.00 | 5292.36 | 5285.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-07-31 14:15:00 | 5855.69 | 5740.48 | 5630.43 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2024-08-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 14:15:00 | 5535.00 | 5658.89 | 5675.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 5372.95 | 5582.68 | 5636.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 5435.95 | 5409.27 | 5498.59 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-08-06 12:45:00 | 5328.00 | 5389.11 | 5466.77 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 5367.15 | 5322.54 | 5396.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:00:00 | 5367.15 | 5322.54 | 5396.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 12:15:00 | 5388.00 | 5337.75 | 5390.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:00:00 | 5388.00 | 5337.75 | 5390.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 5435.35 | 5357.27 | 5394.71 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 5435.35 | 5357.27 | 5394.71 | SL hit (close>ema400) qty=1.00 sl=5394.71 alert=retest1 |

### Cycle 100 — BUY (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 09:15:00 | 5623.40 | 5445.17 | 5428.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 11:15:00 | 5673.05 | 5522.40 | 5468.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 09:15:00 | 6634.95 | 6638.74 | 6550.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:00:00 | 6634.95 | 6638.74 | 6550.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 6922.45 | 6961.65 | 6924.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 15:00:00 | 6922.45 | 6961.65 | 6924.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 6930.00 | 6955.32 | 6925.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 6910.00 | 6955.32 | 6925.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 6895.75 | 6943.41 | 6922.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 6895.75 | 6943.41 | 6922.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 6899.95 | 6934.71 | 6920.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 6897.95 | 6934.71 | 6920.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-08-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 12:15:00 | 6834.85 | 6904.62 | 6908.51 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 09:15:00 | 7063.70 | 6915.88 | 6909.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 11:15:00 | 7154.90 | 6991.53 | 6946.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 11:15:00 | 7134.70 | 7143.96 | 7066.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 11:30:00 | 7121.90 | 7143.96 | 7066.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 7175.00 | 7188.22 | 7146.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 09:15:00 | 7254.80 | 7188.22 | 7146.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 13:15:00 | 7198.95 | 7169.48 | 7149.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 7109.75 | 7150.06 | 7147.33 | SL hit (close<static) qty=1.00 sl=7130.05 alert=retest2 |

### Cycle 103 — SELL (started 2024-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 11:15:00 | 7038.00 | 7127.65 | 7137.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 12:15:00 | 6997.10 | 7101.54 | 7124.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 14:15:00 | 7144.95 | 7069.04 | 7086.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 14:15:00 | 7144.95 | 7069.04 | 7086.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 7144.95 | 7069.04 | 7086.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:00:00 | 7144.95 | 7069.04 | 7086.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 7116.00 | 7078.43 | 7089.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-05 09:15:00 | 7109.00 | 7078.43 | 7089.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-05 09:15:00 | 7200.00 | 7102.75 | 7099.60 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2024-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 14:15:00 | 7073.15 | 7116.03 | 7117.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 13:15:00 | 7056.65 | 7094.44 | 7105.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 7134.75 | 7102.50 | 7108.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 14:15:00 | 7134.75 | 7102.50 | 7108.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 7134.75 | 7102.50 | 7108.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 7134.75 | 7102.50 | 7108.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 7124.00 | 7106.80 | 7109.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 7177.00 | 7106.80 | 7109.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 09:15:00 | 7147.65 | 7114.97 | 7113.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 7242.00 | 7171.85 | 7151.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 10:15:00 | 7171.25 | 7171.73 | 7153.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 11:00:00 | 7171.25 | 7171.73 | 7153.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 7165.90 | 7170.56 | 7154.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:30:00 | 7161.65 | 7170.56 | 7154.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 7272.00 | 7350.29 | 7315.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:45:00 | 7275.00 | 7350.29 | 7315.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 7309.90 | 7342.21 | 7315.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 7356.05 | 7320.23 | 7311.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 7249.45 | 7306.07 | 7305.60 | SL hit (close<static) qty=1.00 sl=7250.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 7245.00 | 7293.86 | 7300.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 12:15:00 | 7203.75 | 7271.13 | 7288.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 7335.00 | 7276.66 | 7287.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 7335.00 | 7276.66 | 7287.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 7335.00 | 7276.66 | 7287.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 7335.00 | 7276.66 | 7287.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 7316.20 | 7284.57 | 7290.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 7290.55 | 7284.57 | 7290.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 7370.00 | 7301.65 | 7297.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 11:15:00 | 7436.20 | 7340.30 | 7316.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 13:15:00 | 7590.00 | 7603.75 | 7535.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 14:00:00 | 7590.00 | 7603.75 | 7535.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 7619.70 | 7618.62 | 7573.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 13:15:00 | 7635.00 | 7618.62 | 7573.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 14:30:00 | 7633.90 | 7620.40 | 7582.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:00:00 | 7626.45 | 7620.40 | 7582.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-26 09:15:00 | 7879.25 | 7619.32 | 7585.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 7835.80 | 7820.10 | 7759.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 7835.80 | 7820.10 | 7759.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 7762.30 | 7804.04 | 7762.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 09:30:00 | 7779.90 | 7804.04 | 7762.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 7671.95 | 7777.62 | 7754.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:00:00 | 7671.95 | 7777.62 | 7754.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-30 11:15:00 | 7580.00 | 7738.10 | 7738.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 11:15:00 | 7580.00 | 7738.10 | 7738.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 13:15:00 | 7525.15 | 7675.20 | 7708.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 14:15:00 | 7612.80 | 7602.20 | 7642.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 15:00:00 | 7612.80 | 7602.20 | 7642.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 7556.60 | 7598.33 | 7633.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 13:15:00 | 7494.65 | 7574.90 | 7613.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 14:15:00 | 7505.65 | 7562.32 | 7604.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-08 09:30:00 | 7468.20 | 7455.22 | 7465.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-08 10:15:00 | 7798.00 | 7523.77 | 7495.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 10:15:00 | 7798.00 | 7523.77 | 7495.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 11:15:00 | 7908.00 | 7600.62 | 7533.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 8129.60 | 8151.13 | 7972.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 09:45:00 | 8150.00 | 8151.13 | 7972.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 8117.40 | 8087.90 | 8020.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 12:30:00 | 8187.05 | 8116.66 | 8051.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 15:15:00 | 8125.00 | 8140.16 | 8141.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2024-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 15:15:00 | 8125.00 | 8140.16 | 8141.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 8001.00 | 8112.33 | 8128.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 12:15:00 | 7815.35 | 7800.21 | 7911.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 13:00:00 | 7815.35 | 7800.21 | 7911.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 7768.70 | 7754.82 | 7803.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 11:45:00 | 7650.00 | 7720.81 | 7779.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 7267.50 | 7410.28 | 7485.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-25 14:15:00 | 7360.05 | 7339.10 | 7420.82 | SL hit (close>ema200) qty=0.50 sl=7339.10 alert=retest2 |

### Cycle 112 — BUY (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 12:15:00 | 6666.00 | 6565.87 | 6559.84 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 6508.00 | 6548.70 | 6553.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 6469.65 | 6532.89 | 6545.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 10:15:00 | 6604.45 | 6547.20 | 6550.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 10:15:00 | 6604.45 | 6547.20 | 6550.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 6604.45 | 6547.20 | 6550.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 11:00:00 | 6604.45 | 6547.20 | 6550.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 6542.70 | 6546.30 | 6550.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 13:15:00 | 6520.00 | 6543.84 | 6548.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-13 14:00:00 | 6529.30 | 6540.93 | 6546.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 09:45:00 | 6498.00 | 6385.35 | 6427.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 13:00:00 | 6529.30 | 6443.22 | 6445.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 13:15:00 | 6497.70 | 6454.12 | 6450.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 13:15:00 | 6497.70 | 6454.12 | 6450.64 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 6411.05 | 6445.51 | 6447.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-19 15:15:00 | 6396.80 | 6435.76 | 6442.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 09:15:00 | 6479.60 | 6444.53 | 6445.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 6479.60 | 6444.53 | 6445.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 6479.60 | 6444.53 | 6445.85 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 10:15:00 | 6459.00 | 6447.43 | 6447.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 6510.05 | 6461.58 | 6454.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 6727.00 | 6728.73 | 6654.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 10:00:00 | 6727.00 | 6728.73 | 6654.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 14:15:00 | 6657.90 | 6702.78 | 6669.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 15:00:00 | 6657.90 | 6702.78 | 6669.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 6654.00 | 6693.03 | 6667.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 6673.25 | 6693.03 | 6667.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 14:15:00 | 6731.80 | 6793.71 | 6764.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 15:00:00 | 6731.80 | 6793.71 | 6764.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 15:15:00 | 6775.00 | 6789.97 | 6765.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:15:00 | 6820.00 | 6789.97 | 6765.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 09:45:00 | 6790.25 | 6784.56 | 6765.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:15:00 | 6786.00 | 6784.56 | 6765.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:15:00 | 6796.75 | 6777.32 | 6766.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 6789.95 | 6779.84 | 6768.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 14:30:00 | 6804.95 | 6781.51 | 6770.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 6837.90 | 6796.23 | 6779.01 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 12:15:00 | 6814.75 | 6800.08 | 6783.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 12:45:00 | 6812.00 | 6796.27 | 6783.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 13:15:00 | 6797.25 | 6796.47 | 6784.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 14:45:00 | 6816.00 | 6798.54 | 6786.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 09:15:00 | 6706.10 | 6780.45 | 6780.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 09:15:00 | 6706.10 | 6780.45 | 6780.73 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 14:15:00 | 6852.00 | 6772.26 | 6772.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-05 13:15:00 | 6891.15 | 6826.51 | 6807.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 6867.95 | 6941.55 | 6903.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 09:15:00 | 6867.95 | 6941.55 | 6903.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 6867.95 | 6941.55 | 6903.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:00:00 | 6867.95 | 6941.55 | 6903.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 6901.90 | 6933.62 | 6903.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:30:00 | 6923.15 | 6934.78 | 6906.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:00:00 | 6917.95 | 6934.32 | 6911.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:45:00 | 6952.05 | 6938.46 | 6915.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 11:15:00 | 6843.00 | 6900.59 | 6903.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 11:15:00 | 6843.00 | 6900.59 | 6903.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 12:15:00 | 6820.85 | 6884.64 | 6896.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 6880.90 | 6875.84 | 6889.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 14:15:00 | 6880.90 | 6875.84 | 6889.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 6880.90 | 6875.84 | 6889.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 6880.90 | 6875.84 | 6889.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 6875.70 | 6875.68 | 6887.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:30:00 | 6859.20 | 6875.68 | 6887.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 6881.15 | 6876.77 | 6886.63 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 13:15:00 | 6926.70 | 6894.79 | 6892.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 14:15:00 | 7085.40 | 6932.91 | 6910.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 15:15:00 | 6990.50 | 7009.55 | 6974.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-13 09:15:00 | 6977.35 | 7009.55 | 6974.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 6920.00 | 6991.64 | 6969.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:00:00 | 6920.00 | 6991.64 | 6969.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 6898.60 | 6973.03 | 6963.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 6910.95 | 6973.03 | 6963.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 6990.90 | 6977.91 | 6967.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:45:00 | 6985.25 | 6977.91 | 6967.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 6991.75 | 6990.35 | 6977.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:30:00 | 6968.95 | 6990.35 | 6977.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 6980.70 | 6988.42 | 6977.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 6980.70 | 6988.42 | 6977.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 6971.05 | 6984.95 | 6977.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:30:00 | 6991.65 | 6984.95 | 6977.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 6981.05 | 6984.17 | 6977.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 12:30:00 | 6975.85 | 6984.17 | 6977.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 6980.90 | 6983.52 | 6977.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:30:00 | 6954.90 | 6983.52 | 6977.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 14:15:00 | 7000.00 | 6986.81 | 6979.93 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 6921.00 | 6967.62 | 6973.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 12:15:00 | 6911.15 | 6956.33 | 6967.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 14:15:00 | 6950.00 | 6948.40 | 6961.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-17 15:00:00 | 6950.00 | 6948.40 | 6961.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 6988.95 | 6951.17 | 6960.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:00:00 | 6988.95 | 6951.17 | 6960.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 6966.20 | 6954.17 | 6960.69 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 11:15:00 | 7015.00 | 6966.34 | 6965.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 12:15:00 | 7127.00 | 6998.47 | 6980.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 14:15:00 | 7092.35 | 7100.45 | 7058.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-19 15:00:00 | 7092.35 | 7100.45 | 7058.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 6960.50 | 7073.76 | 7062.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 6960.50 | 7073.76 | 7062.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 6886.10 | 7036.23 | 7046.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 6802.80 | 6989.54 | 7024.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 7008.80 | 6972.96 | 7005.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 10:15:00 | 7008.80 | 6972.96 | 7005.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 7008.80 | 6972.96 | 7005.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:00:00 | 7008.80 | 6972.96 | 7005.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 7027.05 | 6983.78 | 7007.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:45:00 | 7022.00 | 6983.78 | 7007.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 6965.30 | 6980.08 | 7003.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 15:00:00 | 6947.45 | 6975.83 | 6998.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 12:15:00 | 7045.00 | 7007.74 | 7005.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 12:15:00 | 7045.00 | 7007.74 | 7005.43 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2024-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 09:15:00 | 6973.30 | 6999.77 | 7002.58 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 13:15:00 | 7025.65 | 7006.01 | 7003.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 7069.35 | 7018.68 | 7009.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 7005.95 | 7097.18 | 7073.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 7005.95 | 7097.18 | 7073.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 7005.95 | 7097.18 | 7073.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 7005.95 | 7097.18 | 7073.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 7001.00 | 7077.95 | 7066.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 11:15:00 | 7039.35 | 7077.95 | 7066.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 11:45:00 | 7053.45 | 7073.24 | 7065.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 13:15:00 | 7027.35 | 7060.81 | 7061.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 7027.35 | 7060.81 | 7061.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 6932.05 | 7035.06 | 7049.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 7118.65 | 7013.99 | 7025.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 7118.65 | 7013.99 | 7025.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 7118.65 | 7013.99 | 7025.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 7118.65 | 7013.99 | 7025.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2024-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 14:15:00 | 7133.80 | 7037.96 | 7035.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 10:15:00 | 7146.15 | 7081.07 | 7057.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-01 11:15:00 | 7080.00 | 7080.86 | 7059.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 11:15:00 | 7080.00 | 7080.86 | 7059.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 7080.00 | 7080.86 | 7059.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 12:00:00 | 7080.00 | 7080.86 | 7059.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 7059.95 | 7077.79 | 7063.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 15:00:00 | 7059.95 | 7077.79 | 7063.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 7084.40 | 7079.11 | 7065.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 7106.85 | 7079.11 | 7065.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-02 09:15:00 | 7086.00 | 7080.49 | 7067.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-02 10:45:00 | 7134.00 | 7094.60 | 7075.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 12:15:00 | 7038.50 | 7195.38 | 7205.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 12:15:00 | 7038.50 | 7195.38 | 7205.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 6992.65 | 7154.84 | 7186.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 14:15:00 | 6582.35 | 6578.05 | 6664.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 15:00:00 | 6582.35 | 6578.05 | 6664.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 6384.50 | 6251.58 | 6316.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 6384.50 | 6251.58 | 6316.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 6463.75 | 6294.01 | 6329.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:45:00 | 6485.00 | 6294.01 | 6329.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 6274.95 | 6252.06 | 6285.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:00:00 | 6274.95 | 6252.06 | 6285.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 6209.55 | 6243.56 | 6278.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:30:00 | 6240.70 | 6243.56 | 6278.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 5757.85 | 5703.34 | 5775.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 13:45:00 | 5783.75 | 5703.34 | 5775.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 5471.20 | 5437.69 | 5497.20 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 5568.00 | 5520.09 | 5517.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 15:15:00 | 5611.00 | 5538.27 | 5525.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 5508.95 | 5556.91 | 5542.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 5508.95 | 5556.91 | 5542.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 5508.95 | 5556.91 | 5542.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 5508.95 | 5556.91 | 5542.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 5506.05 | 5546.73 | 5539.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 5476.20 | 5546.73 | 5539.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 5740.00 | 5583.99 | 5557.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:15:00 | 5752.30 | 5583.99 | 5557.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 14:45:00 | 5751.40 | 5701.02 | 5634.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 09:30:00 | 5753.00 | 5720.45 | 5655.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 10:45:00 | 5814.65 | 5732.85 | 5667.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 5806.50 | 6043.51 | 5967.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:00:00 | 5806.50 | 6043.51 | 5967.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 5768.50 | 5988.51 | 5949.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-04 10:30:00 | 5787.35 | 5988.51 | 5949.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-02-04 12:15:00 | 5681.20 | 5885.53 | 5906.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — SELL (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 12:15:00 | 5681.20 | 5885.53 | 5906.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 5597.20 | 5769.43 | 5822.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 5228.60 | 5214.45 | 5310.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 15:00:00 | 5228.60 | 5214.45 | 5310.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 13:15:00 | 5230.60 | 5233.59 | 5279.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 13:30:00 | 5265.70 | 5233.59 | 5279.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 5301.50 | 5248.41 | 5274.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:00:00 | 5301.50 | 5248.41 | 5274.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 5365.55 | 5271.84 | 5282.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-13 11:00:00 | 5365.55 | 5271.84 | 5282.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 13:15:00 | 5278.30 | 5277.48 | 5283.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 15:00:00 | 5252.75 | 5272.54 | 5280.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-18 10:15:00 | 4990.11 | 5068.42 | 5123.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 5063.00 | 5017.61 | 5065.92 | SL hit (close>ema200) qty=0.50 sl=5017.61 alert=retest2 |

### Cycle 132 — BUY (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 14:15:00 | 5096.20 | 5053.24 | 5052.04 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-02-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 13:15:00 | 5037.05 | 5052.29 | 5053.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 4982.15 | 5035.77 | 5045.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 12:15:00 | 5045.00 | 5019.89 | 5033.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 12:15:00 | 5045.00 | 5019.89 | 5033.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 5045.00 | 5019.89 | 5033.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 5045.00 | 5019.89 | 5033.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 5030.00 | 5021.91 | 5033.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:30:00 | 5022.95 | 5021.91 | 5033.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 5067.65 | 5031.06 | 5036.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 5067.65 | 5031.06 | 5036.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 5050.65 | 5034.98 | 5037.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:15:00 | 5103.25 | 5034.98 | 5037.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 4838.65 | 4825.09 | 4865.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 4826.50 | 4825.09 | 4865.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 4799.25 | 4819.92 | 4859.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 10:30:00 | 4785.05 | 4819.49 | 4855.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 12:15:00 | 4975.35 | 4850.94 | 4863.88 | SL hit (close>static) qty=1.00 sl=4901.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 13:15:00 | 4964.60 | 4873.67 | 4873.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 5053.00 | 4941.68 | 4907.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 13:15:00 | 5106.75 | 5107.66 | 5029.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 13:30:00 | 5099.80 | 5107.66 | 5029.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 5029.80 | 5092.53 | 5042.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:30:00 | 5048.90 | 5092.53 | 5042.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 5038.60 | 5081.75 | 5041.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:15:00 | 4981.55 | 5081.75 | 5041.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 5071.55 | 5056.29 | 5039.98 | EMA400 retest candle locked (from upside) |

### Cycle 135 — SELL (started 2025-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 13:15:00 | 5006.35 | 5032.14 | 5034.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 09:15:00 | 4899.00 | 4994.88 | 5016.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 4899.95 | 4870.40 | 4925.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 11:00:00 | 4899.95 | 4870.40 | 4925.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 4922.00 | 4880.72 | 4924.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 12:00:00 | 4922.00 | 4880.72 | 4924.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 12:15:00 | 4965.80 | 4897.73 | 4928.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:00:00 | 4965.80 | 4897.73 | 4928.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 5002.65 | 4918.72 | 4935.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 13:45:00 | 5012.45 | 4918.72 | 4935.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 4998.00 | 4946.94 | 4945.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 09:15:00 | 5027.35 | 4963.02 | 4953.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 11:15:00 | 5024.25 | 5035.86 | 5005.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 11:45:00 | 5019.70 | 5035.86 | 5005.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 5035.00 | 5035.69 | 5008.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 13:30:00 | 5040.50 | 5035.75 | 5010.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 5048.00 | 5035.75 | 5010.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 5085.00 | 5030.07 | 5012.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 13:15:00 | 5170.50 | 5196.91 | 5200.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-03-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-21 13:15:00 | 5170.50 | 5196.91 | 5200.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-21 14:15:00 | 5149.90 | 5187.51 | 5195.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-25 10:15:00 | 5104.60 | 5093.92 | 5127.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:30:00 | 5094.15 | 5093.92 | 5127.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 5202.30 | 5115.60 | 5134.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:00:00 | 5202.30 | 5115.60 | 5134.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 5203.90 | 5133.26 | 5140.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:45:00 | 5231.90 | 5133.26 | 5140.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 13:15:00 | 5226.95 | 5152.00 | 5148.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-26 10:15:00 | 5260.25 | 5185.45 | 5166.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 5331.35 | 5401.92 | 5347.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 12:15:00 | 5331.35 | 5401.92 | 5347.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 5331.35 | 5401.92 | 5347.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:00:00 | 5331.35 | 5401.92 | 5347.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 5336.15 | 5388.77 | 5346.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 14:45:00 | 5359.65 | 5378.23 | 5345.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 15:15:00 | 5306.05 | 5363.80 | 5342.26 | SL hit (close<static) qty=1.00 sl=5308.00 alert=retest2 |

### Cycle 139 — SELL (started 2025-04-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 15:15:00 | 5567.90 | 5593.99 | 5597.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 4543.60 | 5383.92 | 5501.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-11 09:15:00 | 4749.00 | 4679.78 | 4795.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 10:15:00 | 4806.25 | 4705.07 | 4796.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 10:15:00 | 4806.25 | 4705.07 | 4796.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:00:00 | 4806.25 | 4705.07 | 4796.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 4817.00 | 4727.46 | 4798.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-11 11:30:00 | 4807.90 | 4727.46 | 4798.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 12:15:00 | 4804.85 | 4742.94 | 4799.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 13:15:00 | 4790.50 | 4742.94 | 4799.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 4931.00 | 4794.05 | 4805.98 | SL hit (close>static) qty=1.00 sl=4830.00 alert=retest2 |

### Cycle 140 — BUY (started 2025-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 11:15:00 | 4880.50 | 4826.61 | 4819.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-16 09:15:00 | 4992.00 | 4884.87 | 4852.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 4968.50 | 4978.41 | 4925.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 10:00:00 | 4968.50 | 4978.41 | 4925.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 5235.00 | 5301.14 | 5242.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 5235.00 | 5301.14 | 5242.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 5260.00 | 5292.92 | 5243.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 5225.00 | 5292.92 | 5243.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 5311.00 | 5309.24 | 5276.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:30:00 | 5258.50 | 5309.24 | 5276.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 5193.00 | 5299.37 | 5289.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 5193.00 | 5299.37 | 5289.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 5136.00 | 5266.70 | 5275.79 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 15:15:00 | 5510.00 | 5273.78 | 5245.96 | EMA200 above EMA400 |

### Cycle 143 — SELL (started 2025-04-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 12:15:00 | 5173.00 | 5230.84 | 5232.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 15:15:00 | 5135.00 | 5194.65 | 5213.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 5238.00 | 5203.32 | 5215.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 5238.00 | 5203.32 | 5215.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 5238.00 | 5203.32 | 5215.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 12:15:00 | 5132.00 | 5191.54 | 5208.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 14:15:00 | 5152.00 | 5187.37 | 5203.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 15:15:00 | 5132.50 | 5180.39 | 5198.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 5313.50 | 5199.35 | 5203.58 | SL hit (close>static) qty=1.00 sl=5268.50 alert=retest2 |

### Cycle 144 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 5364.50 | 5232.38 | 5218.21 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 5187.50 | 5252.81 | 5261.01 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 5296.50 | 5260.74 | 5256.94 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 5183.00 | 5242.04 | 5249.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 09:15:00 | 5134.00 | 5220.43 | 5239.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 5328.00 | 5182.60 | 5200.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 5328.00 | 5182.60 | 5200.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 5328.00 | 5182.60 | 5200.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:30:00 | 5328.00 | 5182.60 | 5200.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 5362.00 | 5218.48 | 5214.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 5381.50 | 5251.09 | 5229.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 5333.00 | 5347.89 | 5297.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:30:00 | 5342.00 | 5347.89 | 5297.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 5381.00 | 5351.51 | 5321.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 5413.00 | 5363.07 | 5343.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:00:00 | 5415.50 | 5373.56 | 5350.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:45:00 | 5420.50 | 5389.84 | 5359.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 5457.00 | 5505.82 | 5511.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 5457.00 | 5505.82 | 5511.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 5443.50 | 5493.35 | 5505.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 5485.00 | 5479.31 | 5494.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 5485.00 | 5479.31 | 5494.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 5485.00 | 5479.31 | 5494.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 5511.50 | 5479.31 | 5494.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 5420.00 | 5467.45 | 5487.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 13:45:00 | 5405.00 | 5451.39 | 5476.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 5350.00 | 5449.73 | 5471.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:30:00 | 5418.00 | 5382.23 | 5414.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 5458.50 | 5432.57 | 5429.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 5458.50 | 5432.57 | 5429.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 5495.00 | 5445.06 | 5435.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 10:15:00 | 5548.00 | 5557.48 | 5524.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:30:00 | 5537.50 | 5557.48 | 5524.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 5601.00 | 5632.53 | 5615.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 5578.50 | 5632.53 | 5615.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 5619.00 | 5629.83 | 5615.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 5626.00 | 5629.83 | 5615.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 5625.00 | 5622.28 | 5616.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 5597.00 | 5610.57 | 5612.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 5597.00 | 5610.57 | 5612.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 5577.50 | 5601.45 | 5607.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 5649.00 | 5553.28 | 5569.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 5649.00 | 5553.28 | 5569.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 5649.00 | 5553.28 | 5569.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 5649.00 | 5553.28 | 5569.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 5631.50 | 5568.92 | 5575.48 | EMA400 retest candle locked (from downside) |

### Cycle 152 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 5679.00 | 5590.94 | 5584.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 5703.50 | 5613.45 | 5595.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 5876.00 | 5876.88 | 5806.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 5876.00 | 5876.88 | 5806.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 5816.50 | 5848.93 | 5814.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 5816.50 | 5848.93 | 5814.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 5812.50 | 5841.64 | 5814.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 5825.50 | 5841.64 | 5814.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 5794.50 | 5832.21 | 5812.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 5794.50 | 5832.21 | 5812.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 5805.00 | 5826.77 | 5811.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 5818.50 | 5826.77 | 5811.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 5796.00 | 5820.62 | 5810.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 5796.00 | 5820.62 | 5810.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 5789.00 | 5814.29 | 5808.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:45:00 | 5791.00 | 5814.29 | 5808.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 5776.00 | 5799.63 | 5802.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 5725.50 | 5777.15 | 5790.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 5610.00 | 5597.21 | 5655.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 5640.50 | 5597.21 | 5655.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 5610.00 | 5599.77 | 5651.40 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 5731.50 | 5671.20 | 5663.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 09:15:00 | 5781.00 | 5722.61 | 5704.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 6063.50 | 6082.73 | 5989.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:00:00 | 6063.50 | 6082.73 | 5989.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 6085.50 | 6101.58 | 6058.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 6096.00 | 6101.58 | 6058.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 6091.00 | 6099.46 | 6061.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 6073.50 | 6099.46 | 6061.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 6070.50 | 6091.84 | 6064.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 6016.00 | 6091.84 | 6064.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 6074.50 | 6094.89 | 6076.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 6074.50 | 6094.89 | 6076.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 6063.00 | 6088.51 | 6075.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:15:00 | 6054.00 | 6088.51 | 6075.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 6051.00 | 6081.01 | 6073.44 | EMA400 retest candle locked (from upside) |

### Cycle 155 — SELL (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 14:15:00 | 6036.50 | 6066.75 | 6067.94 | EMA200 below EMA400 |

### Cycle 156 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 6221.50 | 6079.56 | 6068.91 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 5758.00 | 6099.62 | 6128.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 5573.00 | 5928.68 | 6041.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 5481.50 | 5448.63 | 5544.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:45:00 | 5492.00 | 5448.63 | 5544.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 5361.00 | 5339.31 | 5358.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 5361.00 | 5339.31 | 5358.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 5357.50 | 5342.95 | 5358.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:15:00 | 5368.00 | 5342.95 | 5358.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 5388.00 | 5351.96 | 5360.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 5388.00 | 5351.96 | 5360.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 5396.00 | 5360.77 | 5364.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 5450.00 | 5360.77 | 5364.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 5412.50 | 5371.11 | 5368.56 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 5369.50 | 5393.18 | 5395.60 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 10:15:00 | 5411.50 | 5390.89 | 5390.44 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 5380.00 | 5389.13 | 5389.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 5363.50 | 5382.54 | 5386.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 5369.50 | 5363.41 | 5371.07 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:15:00 | 5282.00 | 5363.41 | 5371.07 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 14:15:00 | 5017.90 | 5055.68 | 5119.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 5034.00 | 5016.68 | 5055.15 | SL hit (close>ema200) qty=0.50 sl=5016.68 alert=retest1 |

### Cycle 162 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 5103.00 | 5047.98 | 5046.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 10:15:00 | 5108.00 | 5059.98 | 5052.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 5208.00 | 5224.15 | 5178.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 10:00:00 | 5208.00 | 5224.15 | 5178.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 5398.00 | 5359.50 | 5310.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 13:30:00 | 5469.50 | 5398.41 | 5359.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 14:15:00 | 5374.50 | 5394.17 | 5394.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 5374.50 | 5394.17 | 5394.88 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 5563.50 | 5422.89 | 5407.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 5590.00 | 5456.31 | 5424.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 5505.00 | 5521.59 | 5470.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 5505.00 | 5521.59 | 5470.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 5503.00 | 5512.34 | 5478.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 5473.50 | 5512.34 | 5478.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 5479.50 | 5500.19 | 5481.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:45:00 | 5480.50 | 5500.19 | 5481.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 5503.50 | 5500.85 | 5483.23 | EMA400 retest candle locked (from upside) |

### Cycle 165 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 5440.00 | 5474.52 | 5475.43 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 5480.00 | 5472.12 | 5471.96 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 5456.00 | 5471.00 | 5471.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 5443.00 | 5465.40 | 5469.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 5471.00 | 5466.52 | 5469.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 5471.00 | 5466.52 | 5469.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 5471.00 | 5466.52 | 5469.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 5471.00 | 5466.52 | 5469.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 5454.00 | 5464.01 | 5468.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:45:00 | 5445.50 | 5457.57 | 5464.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:30:00 | 5444.50 | 5436.00 | 5444.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 5395.00 | 5331.92 | 5331.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 5395.00 | 5331.92 | 5331.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 5427.50 | 5362.57 | 5346.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 5397.50 | 5423.09 | 5395.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 5397.50 | 5423.09 | 5395.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 5397.50 | 5423.09 | 5395.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 5397.50 | 5423.09 | 5395.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 5413.50 | 5421.17 | 5397.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:45:00 | 5446.50 | 5412.41 | 5400.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 13:15:00 | 5395.00 | 5487.00 | 5494.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 5395.00 | 5487.00 | 5494.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 5320.50 | 5453.70 | 5478.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 5188.00 | 5180.65 | 5222.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:30:00 | 5190.00 | 5180.65 | 5222.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 5175.00 | 5145.47 | 5153.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:30:00 | 5199.50 | 5145.47 | 5153.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 5222.00 | 5169.82 | 5163.67 | EMA200 above EMA400 |

### Cycle 171 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 5144.00 | 5173.28 | 5173.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 5119.50 | 5154.51 | 5164.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 5120.00 | 5111.84 | 5133.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 5120.00 | 5111.84 | 5133.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 5120.00 | 5111.84 | 5133.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 5120.00 | 5111.84 | 5133.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 4762.70 | 4712.71 | 4732.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 4767.20 | 4712.71 | 4732.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 4829.80 | 4736.12 | 4741.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 4829.80 | 4736.12 | 4741.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 4830.00 | 4754.90 | 4749.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 4843.80 | 4772.68 | 4757.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 4795.80 | 4804.01 | 4782.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 11:00:00 | 4795.80 | 4804.01 | 4782.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 4799.30 | 4803.07 | 4783.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:30:00 | 4782.20 | 4803.07 | 4783.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 4775.40 | 4797.53 | 4783.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:00:00 | 4775.40 | 4797.53 | 4783.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 4775.00 | 4793.03 | 4782.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:15:00 | 4789.50 | 4793.03 | 4782.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 15:15:00 | 4705.00 | 4793.06 | 4793.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 15:15:00 | 4705.00 | 4793.06 | 4793.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 09:15:00 | 4675.80 | 4769.60 | 4782.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 4712.90 | 4707.99 | 4737.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 4712.90 | 4707.99 | 4737.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 4712.90 | 4707.99 | 4737.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 4725.90 | 4707.99 | 4737.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 4663.10 | 4648.50 | 4669.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 4697.00 | 4648.50 | 4669.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 4680.90 | 4654.98 | 4670.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 4689.00 | 4654.98 | 4670.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 4708.80 | 4665.75 | 4673.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 4708.80 | 4665.75 | 4673.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 4711.70 | 4681.38 | 4679.84 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 4646.50 | 4677.53 | 4679.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 4613.00 | 4649.01 | 4661.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 4664.30 | 4640.03 | 4652.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 4664.30 | 4640.03 | 4652.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 4664.30 | 4640.03 | 4652.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 4658.70 | 4640.03 | 4652.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 4658.40 | 4643.70 | 4652.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 4679.00 | 4643.70 | 4652.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 4618.50 | 4638.66 | 4649.68 | EMA400 retest candle locked (from downside) |

### Cycle 176 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 4739.00 | 4667.21 | 4659.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 4764.00 | 4686.57 | 4669.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 15:15:00 | 4795.00 | 4800.67 | 4763.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:15:00 | 4805.00 | 4800.67 | 4763.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 4779.00 | 4797.48 | 4783.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 4779.00 | 4797.48 | 4783.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 4792.00 | 4804.26 | 4793.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 4792.00 | 4804.26 | 4793.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 4776.40 | 4798.69 | 4791.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 4776.40 | 4798.69 | 4791.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 4787.00 | 4796.35 | 4791.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 4795.10 | 4796.35 | 4791.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:00:00 | 4791.10 | 4790.35 | 4789.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 4789.70 | 4790.20 | 4789.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 15:15:00 | 4782.20 | 4788.60 | 4789.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 4782.20 | 4788.60 | 4789.00 | EMA200 below EMA400 |

### Cycle 178 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 4805.50 | 4791.98 | 4790.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 4825.00 | 4803.12 | 4796.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 12:15:00 | 4802.00 | 4802.89 | 4796.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-27 13:00:00 | 4802.00 | 4802.89 | 4796.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 4802.40 | 4802.79 | 4797.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:30:00 | 4813.30 | 4799.19 | 4796.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 4773.50 | 4794.06 | 4794.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 4773.50 | 4794.06 | 4794.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 4751.90 | 4785.62 | 4790.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 4756.90 | 4752.22 | 4766.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 12:00:00 | 4756.90 | 4752.22 | 4766.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 4763.80 | 4754.54 | 4766.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:00:00 | 4763.80 | 4754.54 | 4766.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 4776.00 | 4758.83 | 4767.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 4776.00 | 4758.83 | 4767.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 4778.20 | 4762.70 | 4768.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 4781.70 | 4762.70 | 4768.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 4710.60 | 4700.89 | 4715.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 4710.60 | 4700.89 | 4715.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 4714.00 | 4703.51 | 4715.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 4695.40 | 4703.51 | 4715.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 4671.30 | 4697.07 | 4711.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 4665.00 | 4691.34 | 4707.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:00:00 | 4665.10 | 4682.68 | 4700.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:00:00 | 4665.00 | 4669.31 | 4687.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 4656.30 | 4671.63 | 4680.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 4671.20 | 4671.55 | 4679.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 13:30:00 | 4637.00 | 4659.60 | 4671.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 14:30:00 | 4635.40 | 4652.92 | 4667.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4431.75 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4431.85 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4431.75 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4423.48 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4405.15 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4403.63 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 4332.00 | 4328.33 | 4395.94 | SL hit (close>ema200) qty=0.50 sl=4328.33 alert=retest2 |

### Cycle 180 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 4400.00 | 4377.74 | 4376.31 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 4369.00 | 4384.97 | 4385.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 4362.30 | 4380.44 | 4383.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 4370.10 | 4369.91 | 4377.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 13:00:00 | 4370.10 | 4369.91 | 4377.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 4354.50 | 4361.06 | 4370.09 | EMA400 retest candle locked (from downside) |

### Cycle 182 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 4406.00 | 4376.35 | 4374.79 | EMA200 above EMA400 |

### Cycle 183 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 4359.30 | 4375.63 | 4376.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 4335.10 | 4365.66 | 4371.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 4347.30 | 4292.46 | 4312.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 4347.30 | 4292.46 | 4312.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 4347.30 | 4292.46 | 4312.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 4347.30 | 4292.46 | 4312.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 4321.90 | 4298.35 | 4313.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:45:00 | 4307.10 | 4306.75 | 4314.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 4091.75 | 4146.94 | 4172.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 4095.40 | 4091.37 | 4116.77 | SL hit (close>ema200) qty=0.50 sl=4091.37 alert=retest2 |

### Cycle 184 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 4082.90 | 4063.36 | 4062.89 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 4051.50 | 4061.57 | 4062.19 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 4074.30 | 4064.12 | 4063.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 12:15:00 | 4087.50 | 4068.50 | 4065.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 4083.70 | 4102.25 | 4092.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 4083.70 | 4102.25 | 4092.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 4083.70 | 4102.25 | 4092.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 4075.10 | 4102.25 | 4092.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 4074.00 | 4096.60 | 4090.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 4070.70 | 4096.60 | 4090.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 4053.40 | 4082.90 | 4085.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 4044.40 | 4075.20 | 4081.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 15:15:00 | 4034.00 | 4031.62 | 4048.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:15:00 | 4062.00 | 4031.62 | 4048.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 4037.00 | 4032.69 | 4047.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 4026.50 | 4032.75 | 4046.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 4063.90 | 4052.49 | 4052.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 188 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 4063.90 | 4052.49 | 4052.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 4128.10 | 4067.61 | 4059.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 4183.00 | 4186.49 | 4152.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:15:00 | 4222.00 | 4186.49 | 4152.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 4246.20 | 4271.13 | 4251.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 4246.20 | 4271.13 | 4251.52 | SL hit (close<ema400) qty=1.00 sl=4251.52 alert=retest1 |

### Cycle 189 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 4231.50 | 4241.12 | 4242.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 4199.00 | 4232.70 | 4238.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 4246.30 | 4224.75 | 4230.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 4246.30 | 4224.75 | 4230.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 4246.30 | 4224.75 | 4230.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:15:00 | 4253.00 | 4224.75 | 4230.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 4270.00 | 4233.80 | 4234.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 4270.00 | 4233.80 | 4234.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 190 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 4283.80 | 4243.80 | 4238.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 4304.30 | 4269.32 | 4255.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 4378.40 | 4378.53 | 4336.82 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 11:00:00 | 4423.50 | 4387.52 | 4344.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 4117.50 | 4355.98 | 4353.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 4117.50 | 4355.98 | 4353.72 | SL hit (close<ema400) qty=1.00 sl=4353.72 alert=retest1 |

### Cycle 191 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 4100.00 | 4304.78 | 4330.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 4057.70 | 4194.41 | 4268.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 4015.00 | 3994.31 | 4027.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 4015.00 | 3994.31 | 4027.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 4015.00 | 3994.31 | 4027.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:00:00 | 4015.00 | 3994.31 | 4027.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 4008.90 | 3997.23 | 4026.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 3977.40 | 4012.52 | 4023.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 3778.53 | 3844.19 | 3882.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 3821.10 | 3790.92 | 3828.68 | SL hit (close>ema200) qty=0.50 sl=3790.92 alert=retest2 |

### Cycle 192 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 3841.70 | 3796.69 | 3791.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 3851.80 | 3807.71 | 3797.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 3775.90 | 3823.35 | 3811.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 3775.90 | 3823.35 | 3811.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 3775.90 | 3823.35 | 3811.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 3775.90 | 3823.35 | 3811.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 3775.60 | 3813.80 | 3807.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 3766.10 | 3813.80 | 3807.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 3779.00 | 3801.35 | 3802.96 | EMA200 below EMA400 |

### Cycle 194 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 3819.40 | 3804.96 | 3804.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 3824.30 | 3808.83 | 3806.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 3801.60 | 3809.97 | 3807.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 3801.60 | 3809.97 | 3807.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 3801.60 | 3809.97 | 3807.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:15:00 | 3792.60 | 3809.97 | 3807.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 195 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 3780.00 | 3803.98 | 3804.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 3763.50 | 3791.89 | 3798.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 11:15:00 | 3783.50 | 3782.32 | 3790.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 3783.50 | 3782.32 | 3790.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 3783.50 | 3782.32 | 3790.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 3724.90 | 3782.32 | 3790.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 3805.60 | 3742.23 | 3744.94 | SL hit (close>static) qty=1.00 sl=3804.90 alert=retest2 |

### Cycle 196 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 3806.00 | 3754.98 | 3750.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 3822.70 | 3784.07 | 3766.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 4077.00 | 4078.94 | 3994.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 4077.00 | 4078.94 | 3994.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 4138.30 | 4166.32 | 4139.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 4138.30 | 4166.32 | 4139.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 4163.00 | 4165.66 | 4142.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:30:00 | 4136.00 | 4165.66 | 4142.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 4214.50 | 4249.17 | 4216.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 4198.00 | 4249.17 | 4216.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 4226.50 | 4244.63 | 4217.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:30:00 | 4262.30 | 4253.71 | 4224.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 14:30:00 | 4240.00 | 4243.11 | 4238.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 4218.50 | 4233.69 | 4234.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 09:15:00 | 4218.50 | 4233.69 | 4234.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 10:15:00 | 4184.00 | 4223.75 | 4230.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 4187.80 | 4173.15 | 4188.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 4187.80 | 4173.15 | 4188.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 4187.80 | 4173.15 | 4188.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 4187.80 | 4173.15 | 4188.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 4181.90 | 4174.90 | 4188.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 4165.00 | 4174.90 | 4188.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 13:15:00 | 3956.75 | 4002.85 | 4044.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 3888.80 | 3880.89 | 3914.46 | SL hit (close>ema200) qty=0.50 sl=3880.89 alert=retest2 |

### Cycle 198 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 3593.30 | 3556.65 | 3554.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 3649.80 | 3575.28 | 3562.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 3645.00 | 3646.70 | 3618.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 14:30:00 | 3638.00 | 3646.70 | 3618.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3538.30 | 3622.99 | 3612.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 3538.30 | 3622.99 | 3612.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 3520.00 | 3602.39 | 3603.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 3514.70 | 3574.87 | 3590.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 3559.50 | 3538.89 | 3565.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:00:00 | 3559.50 | 3538.89 | 3565.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 3573.00 | 3545.71 | 3565.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 3573.00 | 3545.71 | 3565.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 3587.60 | 3554.09 | 3567.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 3589.70 | 3554.09 | 3567.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 3566.70 | 3558.52 | 3567.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 3566.70 | 3558.52 | 3567.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 3563.00 | 3559.41 | 3567.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:15:00 | 3572.00 | 3559.41 | 3567.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 3572.00 | 3561.93 | 3567.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 3482.90 | 3561.93 | 3567.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 12:15:00 | 3524.70 | 3460.72 | 3454.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 200 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 3524.70 | 3460.72 | 3454.32 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 3407.60 | 3450.62 | 3455.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 3342.40 | 3409.54 | 3432.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3511.20 | 3372.48 | 3393.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3511.20 | 3372.48 | 3393.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3511.20 | 3372.48 | 3393.53 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 3489.00 | 3411.23 | 3408.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 3497.10 | 3428.40 | 3416.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 3414.80 | 3456.12 | 3436.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 3414.80 | 3456.12 | 3436.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3414.80 | 3456.12 | 3436.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 3414.90 | 3456.12 | 3436.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 3419.20 | 3448.74 | 3434.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 3435.80 | 3448.74 | 3434.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-06 09:15:00 | 3779.38 | 3554.24 | 3493.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 203 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 4244.00 | 4315.67 | 4317.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 4221.10 | 4246.86 | 4258.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 14:15:00 | 4157.80 | 4139.97 | 4171.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 15:00:00 | 4157.80 | 4139.97 | 4171.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 4144.90 | 4129.33 | 4146.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 4205.30 | 4129.33 | 4146.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 4266.40 | 4156.74 | 4157.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 4266.40 | 4156.74 | 4157.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 204 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 4250.00 | 4175.39 | 4166.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 4296.10 | 4220.96 | 4191.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 4291.10 | 4291.43 | 4252.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 4291.10 | 4291.43 | 4252.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 4253.50 | 4285.78 | 4262.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 4259.20 | 4285.78 | 4262.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 4230.60 | 4274.75 | 4259.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 4230.60 | 4274.75 | 4259.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 4249.10 | 4257.94 | 4254.56 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-22 11:15:00 | 1483.00 | 2023-05-22 14:15:00 | 1499.20 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2023-05-22 12:15:00 | 1484.45 | 2023-05-22 14:15:00 | 1499.20 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2023-06-01 10:30:00 | 1566.00 | 2023-06-12 10:15:00 | 1589.05 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2023-06-01 12:45:00 | 1567.85 | 2023-06-12 10:15:00 | 1589.05 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2023-06-01 15:15:00 | 1569.00 | 2023-06-12 10:15:00 | 1589.05 | STOP_HIT | 1.00 | 1.28% |
| BUY | retest2 | 2023-06-16 12:00:00 | 1717.00 | 2023-06-22 12:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-06-16 13:00:00 | 1714.90 | 2023-06-22 12:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2023-06-16 15:00:00 | 1713.50 | 2023-06-22 12:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-06-19 09:45:00 | 1716.90 | 2023-06-22 12:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2023-06-19 15:15:00 | 1711.60 | 2023-06-22 12:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2023-06-20 13:15:00 | 1713.90 | 2023-06-22 12:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2023-06-20 14:15:00 | 1711.20 | 2023-06-22 12:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-06-22 10:15:00 | 1713.00 | 2023-06-22 12:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-06-23 13:45:00 | 1716.95 | 2023-07-04 12:15:00 | 1746.90 | STOP_HIT | 1.00 | 1.74% |
| SELL | retest2 | 2023-07-12 11:00:00 | 1675.70 | 2023-07-17 09:15:00 | 1711.60 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2023-07-13 10:30:00 | 1677.35 | 2023-07-17 09:15:00 | 1711.60 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2023-07-13 13:45:00 | 1677.60 | 2023-07-17 09:15:00 | 1711.60 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2023-07-14 10:15:00 | 1674.95 | 2023-07-17 09:15:00 | 1711.60 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2023-07-24 14:15:00 | 1725.75 | 2023-07-24 14:15:00 | 1706.95 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2023-08-04 13:30:00 | 1699.85 | 2023-08-08 14:15:00 | 1707.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2023-08-07 09:30:00 | 1701.45 | 2023-08-08 14:15:00 | 1707.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2023-08-21 10:15:00 | 1982.50 | 2023-08-25 12:15:00 | 2009.55 | STOP_HIT | 1.00 | 1.36% |
| BUY | retest2 | 2023-09-01 09:15:00 | 2082.40 | 2023-09-04 11:15:00 | 2035.55 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2023-09-01 14:15:00 | 2057.40 | 2023-09-04 11:15:00 | 2035.55 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2023-09-27 13:00:00 | 2102.80 | 2023-09-28 10:15:00 | 2103.10 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2023-09-27 13:30:00 | 2107.95 | 2023-09-28 10:15:00 | 2103.10 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2023-10-03 15:15:00 | 2055.00 | 2023-10-06 11:15:00 | 2061.00 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2023-10-06 10:00:00 | 2054.95 | 2023-10-06 11:15:00 | 2061.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2023-10-09 11:15:00 | 2058.95 | 2023-10-10 15:15:00 | 2050.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-10-09 14:45:00 | 2086.25 | 2023-10-10 15:15:00 | 2050.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2023-10-12 13:30:00 | 2085.40 | 2023-10-17 14:15:00 | 2080.55 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2023-10-13 09:15:00 | 2089.60 | 2023-10-17 14:15:00 | 2080.55 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2023-10-19 15:15:00 | 2059.50 | 2023-10-26 09:15:00 | 1956.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-19 15:15:00 | 2059.50 | 2023-10-26 11:15:00 | 1990.00 | STOP_HIT | 0.50 | 3.37% |
| BUY | retest2 | 2023-11-07 13:45:00 | 2255.95 | 2023-11-08 09:15:00 | 2481.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-19 12:30:00 | 3005.00 | 2023-12-20 14:15:00 | 2949.00 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2023-12-19 14:00:00 | 2992.75 | 2023-12-20 14:15:00 | 2949.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-12-20 11:45:00 | 2998.00 | 2023-12-20 14:15:00 | 2949.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2023-12-22 12:15:00 | 2951.30 | 2023-12-26 09:15:00 | 2977.40 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-01-18 12:30:00 | 3141.50 | 2024-01-19 09:15:00 | 3167.15 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2024-01-18 14:15:00 | 3145.45 | 2024-01-19 09:15:00 | 3167.15 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-01-19 12:15:00 | 3152.00 | 2024-01-19 12:15:00 | 3171.20 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-01-25 13:45:00 | 3205.70 | 2024-01-30 12:15:00 | 3141.60 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-01-25 14:15:00 | 3208.85 | 2024-01-30 12:15:00 | 3141.60 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-02-07 09:15:00 | 3010.50 | 2024-02-07 12:15:00 | 3228.10 | STOP_HIT | 1.00 | -7.23% |
| SELL | retest2 | 2024-02-23 09:15:00 | 3922.80 | 2024-02-27 09:15:00 | 3973.95 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-03-01 10:45:00 | 3871.00 | 2024-03-01 14:15:00 | 3896.30 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-03-01 12:00:00 | 3865.65 | 2024-03-01 14:15:00 | 3896.30 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-03-04 12:00:00 | 3923.00 | 2024-03-05 12:15:00 | 3861.70 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-03-04 15:00:00 | 3921.70 | 2024-03-05 12:15:00 | 3861.70 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2024-03-05 09:45:00 | 3926.75 | 2024-03-05 12:15:00 | 3861.70 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-03-18 12:15:00 | 4063.30 | 2024-03-19 10:15:00 | 3983.25 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2024-03-18 13:00:00 | 4060.70 | 2024-03-19 10:15:00 | 3983.25 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-03-21 12:15:00 | 3962.00 | 2024-03-21 15:15:00 | 3998.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest1 | 2024-03-26 09:15:00 | 3902.00 | 2024-03-28 09:15:00 | 3907.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-04-19 13:00:00 | 4105.00 | 2024-04-29 14:15:00 | 4497.52 | TARGET_HIT | 1.00 | 9.56% |
| BUY | retest2 | 2024-04-19 13:45:00 | 4088.65 | 2024-04-30 09:15:00 | 4515.50 | TARGET_HIT | 1.00 | 10.44% |
| SELL | retest2 | 2024-05-13 09:30:00 | 4408.05 | 2024-05-14 10:15:00 | 4494.05 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest1 | 2024-05-16 09:15:00 | 4605.45 | 2024-05-22 10:15:00 | 4609.95 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2024-05-28 09:15:00 | 4704.90 | 2024-05-28 11:15:00 | 4624.05 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-06-12 09:30:00 | 4962.50 | 2024-06-21 12:15:00 | 5223.40 | STOP_HIT | 1.00 | 5.26% |
| BUY | retest2 | 2024-06-26 11:00:00 | 5358.90 | 2024-06-27 12:15:00 | 5316.75 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-06-27 09:45:00 | 5362.10 | 2024-06-27 12:15:00 | 5316.75 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-07-04 09:15:00 | 5555.85 | 2024-07-09 13:15:00 | 5560.25 | STOP_HIT | 1.00 | 0.08% |
| SELL | retest2 | 2024-07-23 12:15:00 | 5047.85 | 2024-07-24 09:15:00 | 5346.50 | STOP_HIT | 1.00 | -5.92% |
| SELL | retest2 | 2024-07-23 13:30:00 | 5123.30 | 2024-07-24 09:15:00 | 5346.50 | STOP_HIT | 1.00 | -4.36% |
| BUY | retest2 | 2024-07-25 11:45:00 | 5323.35 | 2024-07-31 14:15:00 | 5855.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-25 12:15:00 | 5310.95 | 2024-07-31 14:15:00 | 5842.05 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-07-25 15:15:00 | 5325.00 | 2024-07-31 14:15:00 | 5857.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-08-06 12:45:00 | 5328.00 | 2024-08-07 13:15:00 | 5435.35 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2024-09-02 09:15:00 | 7254.80 | 2024-09-03 10:15:00 | 7109.75 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2024-09-02 13:15:00 | 7198.95 | 2024-09-03 10:15:00 | 7109.75 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-09-19 09:15:00 | 7356.05 | 2024-09-19 09:15:00 | 7249.45 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-09-25 13:15:00 | 7635.00 | 2024-09-30 11:15:00 | 7580.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-09-25 14:30:00 | 7633.90 | 2024-09-30 11:15:00 | 7580.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2024-09-25 15:00:00 | 7626.45 | 2024-09-30 11:15:00 | 7580.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-09-26 09:15:00 | 7879.25 | 2024-09-30 11:15:00 | 7580.00 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2024-10-03 13:15:00 | 7494.65 | 2024-10-08 10:15:00 | 7798.00 | STOP_HIT | 1.00 | -4.05% |
| SELL | retest2 | 2024-10-03 14:15:00 | 7505.65 | 2024-10-08 10:15:00 | 7798.00 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2024-10-08 09:30:00 | 7468.20 | 2024-10-08 10:15:00 | 7798.00 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2024-10-11 12:30:00 | 8187.05 | 2024-10-15 15:15:00 | 8125.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-10-21 11:45:00 | 7650.00 | 2024-10-25 10:15:00 | 7267.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 11:45:00 | 7650.00 | 2024-10-25 14:15:00 | 7360.05 | STOP_HIT | 0.50 | 3.79% |
| SELL | retest2 | 2024-11-13 13:15:00 | 6520.00 | 2024-11-19 13:15:00 | 6497.70 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2024-11-13 14:00:00 | 6529.30 | 2024-11-19 13:15:00 | 6497.70 | STOP_HIT | 1.00 | 0.48% |
| SELL | retest2 | 2024-11-19 09:45:00 | 6498.00 | 2024-11-19 13:15:00 | 6497.70 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2024-11-19 13:00:00 | 6529.30 | 2024-11-19 13:15:00 | 6497.70 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2024-11-29 09:15:00 | 6820.00 | 2024-12-03 09:15:00 | 6706.10 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-11-29 09:45:00 | 6790.25 | 2024-12-03 09:15:00 | 6706.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2024-11-29 10:15:00 | 6786.00 | 2024-12-03 09:15:00 | 6706.10 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-11-29 13:15:00 | 6796.75 | 2024-12-03 09:15:00 | 6706.10 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-11-29 14:30:00 | 6804.95 | 2024-12-03 09:15:00 | 6706.10 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-12-02 09:30:00 | 6837.90 | 2024-12-03 09:15:00 | 6706.10 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-12-02 12:15:00 | 6814.75 | 2024-12-03 09:15:00 | 6706.10 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-12-02 12:45:00 | 6812.00 | 2024-12-03 09:15:00 | 6706.10 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-12-02 14:45:00 | 6816.00 | 2024-12-03 09:15:00 | 6706.10 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-12-09 11:30:00 | 6923.15 | 2024-12-10 11:15:00 | 6843.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-12-09 14:00:00 | 6917.95 | 2024-12-10 11:15:00 | 6843.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-12-09 14:45:00 | 6952.05 | 2024-12-10 11:15:00 | 6843.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-12-23 15:00:00 | 6947.45 | 2024-12-24 12:15:00 | 7045.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-12-30 11:15:00 | 7039.35 | 2024-12-30 13:15:00 | 7027.35 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-12-30 11:45:00 | 7053.45 | 2024-12-30 13:15:00 | 7027.35 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-01-02 10:45:00 | 7134.00 | 2025-01-06 12:15:00 | 7038.50 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-01-31 10:15:00 | 5752.30 | 2025-02-04 12:15:00 | 5681.20 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-01-31 14:45:00 | 5751.40 | 2025-02-04 12:15:00 | 5681.20 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-02-01 09:30:00 | 5753.00 | 2025-02-04 12:15:00 | 5681.20 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-02-01 10:45:00 | 5814.65 | 2025-02-04 12:15:00 | 5681.20 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-02-13 15:00:00 | 5252.75 | 2025-02-18 10:15:00 | 4990.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:00:00 | 5252.75 | 2025-02-19 09:15:00 | 5063.00 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-03-03 10:30:00 | 4785.05 | 2025-03-03 12:15:00 | 4975.35 | STOP_HIT | 1.00 | -3.98% |
| BUY | retest2 | 2025-03-13 13:30:00 | 5040.50 | 2025-03-21 13:15:00 | 5170.50 | STOP_HIT | 1.00 | 2.58% |
| BUY | retest2 | 2025-03-13 14:15:00 | 5048.00 | 2025-03-21 13:15:00 | 5170.50 | STOP_HIT | 1.00 | 2.43% |
| BUY | retest2 | 2025-03-17 09:15:00 | 5085.00 | 2025-03-21 13:15:00 | 5170.50 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2025-03-28 14:45:00 | 5359.65 | 2025-03-28 15:15:00 | 5306.05 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-04-01 09:15:00 | 5509.15 | 2025-04-04 15:15:00 | 5567.90 | STOP_HIT | 1.00 | 1.07% |
| SELL | retest2 | 2025-04-11 13:15:00 | 4790.50 | 2025-04-15 09:15:00 | 4931.00 | STOP_HIT | 1.00 | -2.93% |
| SELL | retest2 | 2025-05-02 12:15:00 | 5132.00 | 2025-05-05 09:15:00 | 5313.50 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-05-02 14:15:00 | 5152.00 | 2025-05-05 09:15:00 | 5313.50 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-05-02 15:15:00 | 5132.50 | 2025-05-05 09:15:00 | 5313.50 | STOP_HIT | 1.00 | -3.53% |
| BUY | retest2 | 2025-05-15 10:30:00 | 5413.00 | 2025-05-20 13:15:00 | 5457.00 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-05-15 12:00:00 | 5415.50 | 2025-05-20 13:15:00 | 5457.00 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2025-05-15 12:45:00 | 5420.50 | 2025-05-20 13:15:00 | 5457.00 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-05-21 13:45:00 | 5405.00 | 2025-05-26 10:15:00 | 5458.50 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-05-22 09:15:00 | 5350.00 | 2025-05-26 10:15:00 | 5458.50 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-05-23 09:30:00 | 5418.00 | 2025-05-26 10:15:00 | 5458.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-02 11:15:00 | 5626.00 | 2025-06-03 12:15:00 | 5597.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-06-02 15:15:00 | 5625.00 | 2025-06-03 12:15:00 | 5597.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-07-24 09:15:00 | 5282.00 | 2025-07-28 14:15:00 | 5017.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-07-24 09:15:00 | 5282.00 | 2025-07-30 09:15:00 | 5034.00 | STOP_HIT | 0.50 | 4.70% |
| SELL | retest2 | 2025-07-31 09:15:00 | 5017.50 | 2025-07-31 12:15:00 | 5061.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-31 14:45:00 | 5020.00 | 2025-08-01 09:15:00 | 5103.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-01 09:15:00 | 5012.00 | 2025-08-01 09:15:00 | 5103.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-08-11 13:30:00 | 5469.50 | 2025-08-14 14:15:00 | 5374.50 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-08-22 12:45:00 | 5445.50 | 2025-09-01 10:15:00 | 5395.00 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-08-25 14:30:00 | 5444.50 | 2025-09-01 10:15:00 | 5395.00 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2025-09-03 14:45:00 | 5446.50 | 2025-09-08 13:15:00 | 5395.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-10-03 14:15:00 | 4789.50 | 2025-10-06 15:15:00 | 4705.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-10-24 09:15:00 | 4795.10 | 2025-10-24 15:15:00 | 4782.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-10-24 12:00:00 | 4791.10 | 2025-10-24 15:15:00 | 4782.20 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-10-24 15:00:00 | 4789.70 | 2025-10-24 15:15:00 | 4782.20 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-10-28 09:30:00 | 4813.30 | 2025-10-28 10:15:00 | 4773.50 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-11-04 11:15:00 | 4665.00 | 2025-11-10 09:15:00 | 4431.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 13:00:00 | 4665.10 | 2025-11-10 09:15:00 | 4431.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 10:00:00 | 4665.00 | 2025-11-10 09:15:00 | 4431.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-07 09:15:00 | 4656.30 | 2025-11-10 09:15:00 | 4423.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-07 13:30:00 | 4637.00 | 2025-11-10 09:15:00 | 4405.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-07 14:30:00 | 4635.40 | 2025-11-10 09:15:00 | 4403.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:15:00 | 4665.00 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 7.14% |
| SELL | retest2 | 2025-11-04 13:00:00 | 4665.10 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 7.14% |
| SELL | retest2 | 2025-11-06 10:00:00 | 4665.00 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 7.14% |
| SELL | retest2 | 2025-11-07 09:15:00 | 4656.30 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 6.96% |
| SELL | retest2 | 2025-11-07 13:30:00 | 4637.00 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 6.58% |
| SELL | retest2 | 2025-11-07 14:30:00 | 4635.40 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 6.55% |
| SELL | retest2 | 2025-11-26 12:45:00 | 4307.10 | 2025-12-08 13:15:00 | 4091.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 12:45:00 | 4307.10 | 2025-12-10 09:15:00 | 4095.40 | STOP_HIT | 0.50 | 4.92% |
| SELL | retest2 | 2025-12-19 11:15:00 | 4026.50 | 2025-12-19 15:15:00 | 4063.90 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest1 | 2025-12-24 09:15:00 | 4222.00 | 2025-12-29 11:15:00 | 4246.20 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-12-30 09:45:00 | 4250.70 | 2025-12-30 10:15:00 | 4231.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-30 10:15:00 | 4247.00 | 2025-12-30 10:15:00 | 4231.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-01-05 11:00:00 | 4423.50 | 2026-01-06 09:15:00 | 4117.50 | STOP_HIT | 1.00 | -6.92% |
| SELL | retest2 | 2026-01-13 12:00:00 | 3977.40 | 2026-01-21 09:15:00 | 3778.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 3977.40 | 2026-01-22 09:15:00 | 3821.10 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2026-02-01 12:15:00 | 3724.90 | 2026-02-03 10:15:00 | 3805.60 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-02-13 11:30:00 | 4262.30 | 2026-02-17 09:15:00 | 4218.50 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-02-16 14:30:00 | 4240.00 | 2026-02-17 09:15:00 | 4218.50 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-02-19 09:15:00 | 4165.00 | 2026-02-24 13:15:00 | 3956.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:15:00 | 4165.00 | 2026-02-27 11:15:00 | 3888.80 | STOP_HIT | 0.50 | 6.63% |
| SELL | retest2 | 2026-03-23 09:15:00 | 3482.90 | 2026-03-25 12:15:00 | 3524.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-04-02 11:15:00 | 3435.80 | 2026-04-06 09:15:00 | 3779.38 | TARGET_HIT | 1.00 | 10.00% |
