# Concord Biotech Ltd. (CONCORDBIO)

## Backtest Summary

- **Window:** 2023-08-18 09:15:00 → 2026-05-11 15:15:00 (4701 bars)
- **Last close:** 1205.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 25 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 28 |
| PARTIAL | 3 |
| TARGET_HIT | 6 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 33 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 22
- **Target hits / Stop hits / Partials:** 6 / 24 / 3
- **Avg / median % per leg:** 0.83% / -1.76%
- **Sum % (uncompounded):** 27.55%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 6 | 26.1% | 5 | 18 | 0 | 0.80% | 18.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 6 | 26.1% | 5 | 18 | 0 | 0.80% | 18.3% |
| SELL (all) | 10 | 5 | 50.0% | 1 | 6 | 3 | 0.92% | 9.2% |
| SELL @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.35% | 17.4% |
| SELL @ 3rd Alert (retest2) | 6 | 2 | 33.3% | 0 | 5 | 1 | -1.36% | -8.2% |
| retest1 (combined) | 4 | 3 | 75.0% | 1 | 1 | 2 | 4.35% | 17.4% |
| retest2 (combined) | 29 | 8 | 27.6% | 5 | 23 | 1 | 0.35% | 10.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 13:15:00 | 1454.25 | 1495.04 | 1495.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 13:15:00 | 1437.10 | 1492.07 | 1493.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 1487.00 | 1484.54 | 1489.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 14:15:00 | 1487.00 | 1484.54 | 1489.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 1487.00 | 1484.54 | 1489.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 1487.00 | 1484.54 | 1489.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1480.00 | 1484.45 | 1489.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:15:00 | 1465.00 | 1484.35 | 1489.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:30:00 | 1457.50 | 1483.83 | 1489.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 15:15:00 | 1465.00 | 1483.83 | 1489.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 10:30:00 | 1460.85 | 1483.35 | 1488.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 13:15:00 | 1501.00 | 1473.04 | 1482.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 14:00:00 | 1501.00 | 1473.04 | 1482.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 1522.00 | 1473.53 | 1482.42 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-14 14:15:00 | 1522.00 | 1473.53 | 1482.42 | SL hit (close>static) qty=1.00 sl=1507.05 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 13:15:00 | 1570.10 | 1490.63 | 1490.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-20 15:15:00 | 1572.10 | 1492.23 | 1491.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 11:15:00 | 1498.95 | 1499.95 | 1495.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 11:15:00 | 1498.95 | 1499.95 | 1495.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 1498.95 | 1499.95 | 1495.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:00:00 | 1498.95 | 1499.95 | 1495.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 1499.60 | 1499.94 | 1495.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 13:45:00 | 1521.45 | 1500.06 | 1495.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 14:45:00 | 1515.65 | 1500.43 | 1495.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-26 15:15:00 | 1480.15 | 1502.23 | 1496.90 | SL hit (close<static) qty=1.00 sl=1494.30 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 12:15:00 | 1799.25 | 2093.58 | 2094.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-19 09:15:00 | 1716.25 | 2080.25 | 2087.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 1721.85 | 1720.89 | 1819.93 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-04 09:30:00 | 1682.65 | 1719.70 | 1815.93 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 1598.52 | 1714.55 | 1810.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 1726.40 | 1692.00 | 1785.32 | SL hit (close>ema200) qty=0.50 sl=1692.00 alert=retest1 |

### Cycle 4 — SELL (started 2025-04-22 09:30:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-22 09:30:00 | 1695.60 | 1700.61 | 1777.79 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-25 09:15:00 | 1610.82 | 1693.05 | 1766.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2025-04-30 14:15:00 | 1526.04 | 1663.28 | 1741.49 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 5 — BUY (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 15:15:00 | 2009.00 | 1684.35 | 1682.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 2052.50 | 1688.01 | 1684.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-20 14:15:00 | 1833.50 | 1852.35 | 1781.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-20 15:00:00 | 1833.50 | 1852.35 | 1781.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1825.00 | 1851.57 | 1781.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:45:00 | 1837.20 | 1851.43 | 1782.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 14:00:00 | 1836.70 | 1850.69 | 1782.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 10:45:00 | 1839.80 | 1845.28 | 1783.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 10:15:00 | 1836.10 | 1841.93 | 1785.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 1790.10 | 1837.68 | 1788.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 1788.00 | 1837.68 | 1788.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1770.00 | 1837.00 | 1788.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-02 09:15:00 | 1770.00 | 1837.00 | 1788.81 | SL hit (close<static) qty=1.00 sl=1773.60 alert=retest2 |

### Cycle 6 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 1611.40 | 1795.14 | 1795.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 1600.80 | 1789.62 | 1793.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 11:15:00 | 1763.00 | 1731.43 | 1758.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 11:15:00 | 1763.00 | 1731.43 | 1758.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 11:15:00 | 1763.00 | 1731.43 | 1758.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:00:00 | 1763.00 | 1731.43 | 1758.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 1741.40 | 1731.53 | 1758.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 1790.30 | 1731.53 | 1758.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1765.80 | 1732.43 | 1758.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 1765.80 | 1732.43 | 1758.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1764.40 | 1732.75 | 1758.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 1770.30 | 1732.75 | 1758.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1743.00 | 1734.68 | 1758.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 10:15:00 | 1734.90 | 1734.68 | 1758.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-08 10:15:00 | 1648.15 | 1711.45 | 1739.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-19 09:15:00 | 1679.30 | 1677.43 | 1712.13 | SL hit (close>ema200) qty=0.50 sl=1677.43 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-06 12:15:00 | 1465.00 | 2024-06-14 14:15:00 | 1522.00 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2024-06-06 14:30:00 | 1457.50 | 2024-06-14 14:15:00 | 1522.00 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2024-06-06 15:15:00 | 1465.00 | 2024-06-14 14:15:00 | 1522.00 | STOP_HIT | 1.00 | -3.89% |
| SELL | retest2 | 2024-06-07 10:30:00 | 1460.85 | 2024-06-14 14:15:00 | 1522.00 | STOP_HIT | 1.00 | -4.19% |
| BUY | retest2 | 2024-06-25 13:45:00 | 1521.45 | 2024-06-26 15:15:00 | 1480.15 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2024-06-25 14:45:00 | 1515.65 | 2024-06-26 15:15:00 | 1480.15 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-06-27 09:45:00 | 1518.35 | 2024-07-04 14:15:00 | 1670.18 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-27 12:00:00 | 1515.65 | 2024-07-04 14:15:00 | 1667.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-01 14:45:00 | 1608.50 | 2024-08-02 09:15:00 | 1577.50 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-08-08 09:15:00 | 1615.10 | 2024-08-08 12:15:00 | 1587.25 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-08-08 10:15:00 | 1606.40 | 2024-08-08 12:15:00 | 1587.25 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1604.75 | 2024-08-09 10:15:00 | 1589.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-08-16 11:45:00 | 1631.50 | 2024-08-27 13:15:00 | 1585.80 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2024-08-19 11:00:00 | 1626.30 | 2024-08-27 13:15:00 | 1585.80 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2024-08-19 15:00:00 | 1634.35 | 2024-08-27 13:15:00 | 1585.80 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2024-08-21 12:00:00 | 1631.25 | 2024-08-27 13:15:00 | 1585.80 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2024-08-23 15:00:00 | 1652.80 | 2024-09-05 09:15:00 | 1818.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-29 09:15:00 | 1758.00 | 2024-09-11 09:15:00 | 1933.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-17 11:15:00 | 1626.30 | 2025-02-18 12:15:00 | 1799.25 | STOP_HIT | 1.00 | 10.63% |
| SELL | retest1 | 2025-04-04 09:30:00 | 1682.65 | 2025-04-07 09:15:00 | 1598.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-04-04 09:30:00 | 1682.65 | 2025-04-15 09:15:00 | 1726.40 | STOP_HIT | 0.50 | -2.60% |
| SELL | retest1 | 2025-04-22 09:30:00 | 1695.60 | 2025-04-25 09:15:00 | 1610.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-04-22 09:30:00 | 1695.60 | 2025-04-30 14:15:00 | 1526.04 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-06-23 10:45:00 | 1837.20 | 2025-07-02 09:15:00 | 1770.00 | STOP_HIT | 1.00 | -3.66% |
| BUY | retest2 | 2025-06-23 14:00:00 | 1836.70 | 2025-07-02 09:15:00 | 1770.00 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2025-06-25 10:45:00 | 1839.80 | 2025-07-02 09:15:00 | 1770.00 | STOP_HIT | 1.00 | -3.79% |
| BUY | retest2 | 2025-06-27 10:15:00 | 1836.10 | 2025-07-02 09:15:00 | 1770.00 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-07-08 14:30:00 | 1810.30 | 2025-07-24 10:15:00 | 1979.78 | TARGET_HIT | 1.00 | 9.36% |
| BUY | retest2 | 2025-07-08 15:15:00 | 1800.00 | 2025-07-31 14:15:00 | 1770.10 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-10 11:30:00 | 1801.90 | 2025-07-31 14:15:00 | 1770.10 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-07-11 13:45:00 | 1799.80 | 2025-07-31 14:15:00 | 1770.10 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-25 10:15:00 | 1734.90 | 2025-09-08 10:15:00 | 1648.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 10:15:00 | 1734.90 | 2025-09-19 09:15:00 | 1679.30 | STOP_HIT | 0.50 | 3.20% |
