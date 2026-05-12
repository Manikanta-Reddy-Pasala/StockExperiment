# Bharti Hexacom Ltd. (BHARTIHEXA)

## Backtest Summary

- **Window:** 2024-04-12 09:15:00 → 2026-05-11 15:15:00 (3584 bars)
- **Last close:** 1467.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 3 |
| ALERT3 | 38 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 37 |
| PARTIAL | 8 |
| TARGET_HIT | 5 |
| STOP_HIT | 36 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 13 / 36
- **Target hits / Stop hits / Partials:** 5 / 36 / 8
- **Avg / median % per leg:** -0.15% / -2.16%
- **Sum % (uncompounded):** -7.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 8 | 27.6% | 4 | 21 | 4 | 0.55% | 15.9% |
| BUY @ 2nd Alert (retest1) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| BUY @ 3rd Alert (retest2) | 21 | 0 | 0.0% | 0 | 21 | 0 | -2.10% | -44.1% |
| SELL (all) | 20 | 5 | 25.0% | 1 | 15 | 4 | -1.17% | -23.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 5 | 25.0% | 1 | 15 | 4 | -1.17% | -23.4% |
| retest1 (combined) | 8 | 8 | 100.0% | 4 | 0 | 4 | 7.50% | 60.0% |
| retest2 (combined) | 41 | 5 | 12.2% | 1 | 36 | 4 | -1.65% | -67.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 15:15:00 | 1270.00 | 1402.56 | 1402.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 1268.05 | 1389.95 | 1396.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-30 10:15:00 | 1386.10 | 1377.06 | 1388.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-30 11:00:00 | 1386.10 | 1377.06 | 1388.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 1361.85 | 1376.90 | 1388.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 14:45:00 | 1358.00 | 1376.69 | 1388.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-31 15:15:00 | 1358.00 | 1375.43 | 1387.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:45:00 | 1357.80 | 1374.87 | 1386.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-03 09:15:00 | 1356.10 | 1374.51 | 1386.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 1408.40 | 1373.72 | 1385.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-03 15:15:00 | 1408.40 | 1373.72 | 1385.52 | SL hit (close>static) qty=1.00 sl=1391.30 alert=retest2 |

### Cycle 2 — BUY (started 2025-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 10:15:00 | 1471.70 | 1362.21 | 1362.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 15:15:00 | 1476.00 | 1366.18 | 1364.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 11:15:00 | 1742.20 | 1751.87 | 1666.79 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 09:30:00 | 1770.40 | 1751.52 | 1668.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-18 14:45:00 | 1761.50 | 1751.87 | 1670.94 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:15:00 | 1768.00 | 1751.79 | 1671.30 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-20 09:15:00 | 1784.50 | 1751.70 | 1674.03 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 12:15:00 | 1849.58 | 1761.40 | 1685.75 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 15:15:00 | 1858.92 | 1764.05 | 1688.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-24 15:15:00 | 1856.40 | 1764.05 | 1688.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-25 09:15:00 | 1873.73 | 1765.22 | 1689.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-06-27 14:15:00 | 1947.44 | 1781.28 | 1704.46 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 3 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 1716.30 | 1766.45 | 1766.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 10:15:00 | 1699.50 | 1765.78 | 1766.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 10:15:00 | 1744.60 | 1708.98 | 1731.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 1744.60 | 1708.98 | 1731.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1744.60 | 1708.98 | 1731.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 10:45:00 | 1735.10 | 1708.98 | 1731.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 1726.60 | 1709.15 | 1731.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 12:15:00 | 1717.00 | 1709.15 | 1731.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 12:45:00 | 1723.90 | 1709.28 | 1731.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 14:15:00 | 1757.00 | 1709.97 | 1731.72 | SL hit (close>static) qty=1.00 sl=1749.90 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 14:15:00 | 1791.00 | 1744.91 | 1744.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1826.60 | 1746.10 | 1745.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1765.10 | 1791.71 | 1771.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 1765.10 | 1791.71 | 1771.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1765.10 | 1791.71 | 1771.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 1801.50 | 1787.73 | 1771.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 10:00:00 | 1803.20 | 1787.73 | 1771.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 11:45:00 | 1801.00 | 1788.44 | 1772.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 15:00:00 | 1800.70 | 1788.76 | 1772.56 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1767.20 | 1790.05 | 1773.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 1767.20 | 1790.05 | 1773.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1785.30 | 1790.00 | 1774.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 10:30:00 | 1788.30 | 1789.54 | 1774.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 1817.40 | 1789.47 | 1774.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 12:45:00 | 1796.00 | 1797.13 | 1780.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 1760.70 | 1796.55 | 1780.34 | SL hit (close<static) qty=1.00 sl=1765.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 1709.40 | 1769.86 | 1770.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 13:15:00 | 1697.80 | 1769.14 | 1769.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 12:15:00 | 1749.50 | 1746.64 | 1756.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 13:00:00 | 1749.50 | 1746.64 | 1756.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 1762.90 | 1746.88 | 1756.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 1762.90 | 1746.88 | 1756.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1758.50 | 1746.99 | 1756.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1738.00 | 1746.99 | 1756.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 1766.10 | 1746.90 | 1756.56 | SL hit (close>static) qty=1.00 sl=1765.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 1848.00 | 1764.40 | 1764.38 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2026-01-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-13 12:15:00 | 1680.00 | 1767.29 | 1767.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-14 09:15:00 | 1657.60 | 1763.65 | 1765.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 09:15:00 | 1701.40 | 1652.03 | 1693.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 1701.40 | 1652.03 | 1693.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1701.40 | 1652.03 | 1693.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1701.40 | 1652.03 | 1693.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1726.20 | 1652.76 | 1693.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 1726.20 | 1652.76 | 1693.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 1719.50 | 1655.89 | 1694.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 09:45:00 | 1719.90 | 1655.89 | 1694.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 1731.00 | 1656.63 | 1694.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-10 10:45:00 | 1706.70 | 1661.34 | 1695.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 1621.37 | 1669.04 | 1688.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-02 09:15:00 | 1536.03 | 1665.89 | 1685.90 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-01-30 14:45:00 | 1358.00 | 2025-02-03 15:15:00 | 1408.40 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-01-31 15:15:00 | 1358.00 | 2025-02-03 15:15:00 | 1408.40 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-02-01 11:45:00 | 1357.80 | 2025-02-03 15:15:00 | 1408.40 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-02-03 09:15:00 | 1356.10 | 2025-02-03 15:15:00 | 1408.40 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-02-05 12:45:00 | 1357.40 | 2025-02-07 12:15:00 | 1411.00 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1338.50 | 2025-02-18 11:15:00 | 1290.19 | PARTIAL | 0.50 | 3.61% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1338.50 | 2025-02-19 13:15:00 | 1372.45 | STOP_HIT | 0.50 | -2.54% |
| SELL | retest2 | 2025-02-11 15:15:00 | 1344.05 | 2025-02-24 09:15:00 | 1276.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 15:15:00 | 1358.10 | 2025-02-24 10:15:00 | 1271.58 | PARTIAL | 0.50 | 6.37% |
| SELL | retest2 | 2025-02-11 15:15:00 | 1344.05 | 2025-03-05 14:15:00 | 1378.65 | STOP_HIT | 0.50 | -2.57% |
| SELL | retest2 | 2025-02-13 15:15:00 | 1358.10 | 2025-03-05 14:15:00 | 1378.65 | STOP_HIT | 0.50 | -1.51% |
| SELL | retest2 | 2025-03-11 11:30:00 | 1328.95 | 2025-03-11 14:15:00 | 1456.05 | STOP_HIT | 1.00 | -9.56% |
| SELL | retest2 | 2025-03-18 09:15:00 | 1325.00 | 2025-03-24 10:15:00 | 1382.45 | STOP_HIT | 1.00 | -4.34% |
| SELL | retest2 | 2025-03-20 14:00:00 | 1329.00 | 2025-03-24 10:15:00 | 1382.45 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2025-03-20 14:30:00 | 1329.15 | 2025-03-24 10:15:00 | 1382.45 | STOP_HIT | 1.00 | -4.01% |
| BUY | retest1 | 2025-06-18 09:30:00 | 1770.40 | 2025-06-24 12:15:00 | 1849.58 | PARTIAL | 0.50 | 4.47% |
| BUY | retest1 | 2025-06-18 14:45:00 | 1761.50 | 2025-06-24 15:15:00 | 1858.92 | PARTIAL | 0.50 | 5.53% |
| BUY | retest1 | 2025-06-19 09:15:00 | 1768.00 | 2025-06-24 15:15:00 | 1856.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-20 09:15:00 | 1784.50 | 2025-06-25 09:15:00 | 1873.73 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-06-18 09:30:00 | 1770.40 | 2025-06-27 14:15:00 | 1947.44 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-06-18 14:45:00 | 1761.50 | 2025-06-27 14:15:00 | 1937.65 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-06-19 09:15:00 | 1768.00 | 2025-06-27 14:15:00 | 1944.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-06-20 09:15:00 | 1784.50 | 2025-06-27 14:15:00 | 1962.95 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-07-15 09:15:00 | 1795.30 | 2025-07-29 09:15:00 | 1741.90 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2025-07-15 14:00:00 | 1770.90 | 2025-07-29 09:15:00 | 1741.90 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-07-29 13:45:00 | 1776.60 | 2025-08-07 13:15:00 | 1740.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-08-08 14:30:00 | 1769.50 | 2025-08-11 09:15:00 | 1722.00 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2025-08-20 09:15:00 | 1769.90 | 2025-08-28 13:15:00 | 1762.90 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-08-20 13:30:00 | 1770.00 | 2025-08-28 13:15:00 | 1762.90 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-08-20 14:30:00 | 1771.90 | 2025-08-28 13:15:00 | 1762.90 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-08-28 14:15:00 | 1780.60 | 2025-08-29 09:15:00 | 1747.60 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-08-29 11:30:00 | 1779.00 | 2025-09-05 09:15:00 | 1757.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-09-01 13:15:00 | 1784.50 | 2025-09-05 10:15:00 | 1736.00 | STOP_HIT | 1.00 | -2.72% |
| BUY | retest2 | 2025-09-01 15:00:00 | 1776.00 | 2025-09-05 10:15:00 | 1736.00 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-09-02 09:15:00 | 1783.60 | 2025-09-05 10:15:00 | 1736.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-09-03 11:00:00 | 1791.10 | 2025-09-05 10:15:00 | 1736.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-10-07 12:15:00 | 1717.00 | 2025-10-07 14:15:00 | 1757.00 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-10-07 12:45:00 | 1723.90 | 2025-10-07 14:15:00 | 1757.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-11-11 09:30:00 | 1801.50 | 2025-11-21 14:15:00 | 1760.70 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-11-11 10:00:00 | 1803.20 | 2025-11-21 14:15:00 | 1760.70 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-11-12 11:45:00 | 1801.00 | 2025-11-21 14:15:00 | 1760.70 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-11-12 15:00:00 | 1800.70 | 2025-11-24 11:15:00 | 1749.70 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2025-11-17 10:30:00 | 1788.30 | 2025-11-24 11:15:00 | 1749.70 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-11-18 09:15:00 | 1817.40 | 2025-11-24 11:15:00 | 1749.70 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-11-21 12:45:00 | 1796.00 | 2025-11-24 11:15:00 | 1749.70 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2025-11-26 11:30:00 | 1788.10 | 2025-11-27 09:15:00 | 1762.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1738.00 | 2025-12-18 15:15:00 | 1766.10 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-02-10 10:45:00 | 1706.70 | 2026-02-27 10:15:00 | 1621.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-10 10:45:00 | 1706.70 | 2026-03-02 09:15:00 | 1536.03 | TARGET_HIT | 0.50 | 10.00% |
