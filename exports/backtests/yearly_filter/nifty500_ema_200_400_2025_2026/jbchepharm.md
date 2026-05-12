# J.B. Chemicals & Pharmaceuticals Ltd. (JBCHEPHARM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 2155.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 0 |
| ALERT3 | 33 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 14 |
| PARTIAL | 0 |
| TARGET_HIT | 7 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 7
- **Target hits / Stop hits / Partials:** 7 / 7 / 0
- **Avg / median % per leg:** 4.50% / 9.23%
- **Sum % (uncompounded):** 62.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 14 | 7 | 50.0% | 7 | 7 | 0 | 4.50% | 63.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 14 | 7 | 50.0% | 7 | 7 | 0 | 4.50% | 63.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 7 | 50.0% | 7 | 7 | 0 | 4.50% | 63.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 13:15:00 | 1668.90 | 1628.96 | 1628.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 1690.50 | 1631.26 | 1630.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 1680.80 | 1681.34 | 1661.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 1680.80 | 1681.34 | 1661.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1685.00 | 1712.81 | 1682.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:30:00 | 1684.70 | 1712.81 | 1682.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 10:15:00 | 1680.80 | 1712.49 | 1682.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:00:00 | 1680.80 | 1712.49 | 1682.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 11:15:00 | 1684.10 | 1712.20 | 1682.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 11:30:00 | 1680.30 | 1712.20 | 1682.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 1684.30 | 1711.93 | 1682.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 12:30:00 | 1684.10 | 1711.93 | 1682.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 1679.20 | 1711.60 | 1682.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 1678.20 | 1711.60 | 1682.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 1678.30 | 1711.27 | 1682.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 1678.30 | 1711.27 | 1682.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 1677.00 | 1710.93 | 1682.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 1662.60 | 1710.93 | 1682.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1642.60 | 1709.57 | 1682.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 1642.60 | 1709.57 | 1682.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 1657.00 | 1666.81 | 1665.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 1651.90 | 1666.81 | 1665.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 12:15:00 | 1666.00 | 1666.79 | 1665.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 12:30:00 | 1665.90 | 1666.79 | 1665.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 1665.50 | 1666.78 | 1665.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:00:00 | 1665.50 | 1666.78 | 1665.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 1664.80 | 1666.76 | 1665.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 15:15:00 | 1660.00 | 1666.76 | 1665.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 1660.00 | 1666.69 | 1665.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 1657.20 | 1666.69 | 1665.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 1651.70 | 1666.43 | 1665.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:15:00 | 1646.00 | 1666.43 | 1665.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1668.20 | 1665.75 | 1665.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:30:00 | 1660.90 | 1665.75 | 1665.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 1671.40 | 1699.45 | 1685.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:45:00 | 1677.40 | 1699.45 | 1685.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 1666.10 | 1699.12 | 1685.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 1666.10 | 1699.12 | 1685.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 13:15:00 | 1683.00 | 1696.34 | 1684.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 13:30:00 | 1675.00 | 1696.34 | 1684.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 1689.20 | 1696.27 | 1684.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 1691.60 | 1696.12 | 1684.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 1674.10 | 1714.06 | 1704.05 | SL hit (close<static) qty=1.00 sl=1680.30 alert=retest2 |

### Cycle 2 — SELL (started 2025-10-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 12:15:00 | 1653.10 | 1698.92 | 1698.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 13:15:00 | 1650.00 | 1695.44 | 1697.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 10:15:00 | 1685.00 | 1684.72 | 1690.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:45:00 | 1684.80 | 1684.72 | 1690.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 1689.20 | 1684.76 | 1690.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 1689.20 | 1684.76 | 1690.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 1687.90 | 1684.79 | 1690.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 1696.40 | 1684.79 | 1690.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 1695.00 | 1684.90 | 1690.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 1695.00 | 1684.90 | 1690.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 1697.70 | 1685.02 | 1690.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:30:00 | 1698.70 | 1685.02 | 1690.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 1695.30 | 1685.12 | 1690.70 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 1803.20 | 1694.19 | 1694.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 15:15:00 | 1815.40 | 1700.73 | 1697.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1735.10 | 1743.68 | 1723.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:00:00 | 1735.10 | 1743.68 | 1723.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 1726.60 | 1743.05 | 1723.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 14:45:00 | 1721.70 | 1743.05 | 1723.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 1854.50 | 1853.66 | 1820.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 12:45:00 | 1873.00 | 1853.93 | 1821.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 13:30:00 | 1872.10 | 1854.19 | 1821.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 15:15:00 | 1871.60 | 1854.34 | 1821.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:45:00 | 1883.50 | 1854.91 | 1822.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-24 09:15:00 | 2060.30 | 1916.67 | 1872.62 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-08-08 10:15:00 | 1691.60 | 2025-09-15 10:15:00 | 1674.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-09-17 12:30:00 | 1694.00 | 2025-09-25 12:15:00 | 1680.20 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-17 14:30:00 | 1691.60 | 2025-09-25 12:15:00 | 1680.20 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-09-18 12:45:00 | 1690.10 | 2025-09-25 12:15:00 | 1680.20 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-09-29 15:00:00 | 1707.00 | 2025-09-30 09:15:00 | 1687.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-09-30 14:45:00 | 1706.50 | 2025-10-01 12:15:00 | 1686.40 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-10-01 09:15:00 | 1713.60 | 2025-10-01 12:15:00 | 1686.40 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-01-30 12:45:00 | 1873.00 | 2026-02-24 09:15:00 | 2060.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 13:30:00 | 1872.10 | 2026-02-24 09:15:00 | 2059.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-30 15:15:00 | 1871.60 | 2026-02-24 09:15:00 | 2058.76 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-01 10:45:00 | 1883.50 | 2026-02-24 10:15:00 | 2071.85 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 11:15:00 | 1955.50 | 2026-05-06 10:15:00 | 2136.09 | TARGET_HIT | 1.00 | 9.23% |
| BUY | retest2 | 2026-04-06 12:00:00 | 1941.90 | 2026-05-06 11:15:00 | 2141.26 | TARGET_HIT | 1.00 | 10.27% |
| BUY | retest2 | 2026-04-07 10:00:00 | 1946.60 | 2026-05-06 13:15:00 | 2151.05 | TARGET_HIT | 1.00 | 10.50% |
