# SUNPHARMA (SUNPHARMA.NS)

## Backtest Summary

- **Source:** Yahoo chart API (1H bars)
- **Window:** 2023-06-06 09:15:00 → 2026-05-06 15:30:00 (4997 bars)
- **Last close:** 1850.20
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 9 |
| ALERT2_SKIP | 8 |
| ALERT3 | 12 |
| PENDING | 28 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 1 |
| ENTRY2 | 20 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 20
- **Target hits / Stop hits / Partials:** 0 / 21 / 0
- **Avg / median % per leg:** -1.60% / -1.60%
- **Sum % (uncompounded):** -33.64%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.29% | -9.2% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.71% | -3.7% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.81% | -5.4% |
| SELL (all) | 17 | 1 | 5.9% | 0 | 17 | 0 | -1.44% | -24.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 1 | 5.9% | 0 | 17 | 0 | -1.44% | -24.5% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -3.71% | -3.7% |
| retest2 (combined) | 20 | 1 | 5.0% | 0 | 20 | 0 | -1.50% | -29.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 11:15:00 | 1446.10 | 1510.41 | 1510.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 12:15:00 | 1443.05 | 1509.73 | 1510.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 12:15:00 | 1498.35 | 1496.88 | 1502.98 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 13:15:00 | 1508.60 | 1496.99 | 1503.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 13:15:00 | 1508.60 | 1496.99 | 1503.01 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-06-20 09:15:00 | 1480.05 | 1502.52 | 1504.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 1475.40 | 1502.25 | 1504.61 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2024-06-26 11:15:00 | 1509.85 | 1499.25 | 1502.61 | SL hit qty=1.00 sl=1509.85 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 1525.40 | 1505.54 | 1505.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 1535.65 | 1506.05 | 1505.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-23 09:15:00 | 1873.30 | 1880.50 | 1824.28 | EMA200 retest candle locked |
| Cross detected — sustain check pending | 2024-10-28 13:15:00 | 1896.95 | 1876.18 | 1828.56 | ENTRY1 cross detected — sustain check pending (15m) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 14:15:00 | 1901.30 | 1876.43 | 1828.92 | BUY ENTRY1 attempt 1/4 (retest1 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1837.70 | 1876.01 | 1830.80 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 1830.80 | 1876.01 | 1830.80 | SL hit qty=1.00 sl=1830.80 alert=retest1 |
| Cross detected — sustain check pending | 2024-10-30 12:15:00 | 1868.80 | 1875.52 | 1831.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-10-30 13:15:00 | 1851.75 | 1875.29 | 1831.33 | ENTRY2 sustain failed after 60m |

### Cycle 3 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 1730.00 | 1811.05 | 1811.43 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1850.20 | 1809.01 | 1808.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 1859.15 | 1810.34 | 1809.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 15:15:00 | 1831.95 | 1834.06 | 1823.66 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 1820.10 | 1833.92 | 1823.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1820.10 | 1833.92 | 1823.64 | EMA400 retest candle locked |

### Cycle 5 — SELL (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 10:15:00 | 1748.65 | 1815.13 | 1815.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1717.20 | 1806.11 | 1809.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 11:15:00 | 1670.65 | 1668.95 | 1714.47 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 1715.60 | 1670.50 | 1712.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1715.60 | 1670.50 | 1712.60 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-01 12:15:00 | 1686.45 | 1708.71 | 1722.38 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-01 13:15:00 | 1691.55 | 1708.54 | 1722.23 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-04 09:15:00 | 1676.10 | 1712.81 | 1723.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-04 10:15:00 | 1699.75 | 1712.68 | 1723.18 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 1649.50 | 1711.89 | 1722.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 10:15:00 | 1649.45 | 1711.27 | 1722.11 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-04-08 14:15:00 | 1687.20 | 1707.81 | 1719.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-04-08 15:15:00 | 1688.35 | 1707.61 | 1719.58 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-09 09:15:00 | 1671.90 | 1707.26 | 1719.34 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 10:15:00 | 1677.60 | 1706.96 | 1719.14 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1719.75 | 1704.68 | 1717.62 | SL hit qty=1.00 sl=1719.75 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 1719.75 | 1704.68 | 1717.62 | SL hit qty=1.00 sl=1719.75 alert=retest2 |
| Cross detected — sustain check pending | 2025-04-11 14:15:00 | 1687.85 | 1704.01 | 1716.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 15:15:00 | 1687.90 | 1703.85 | 1716.82 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 1719.75 | 1704.00 | 1716.82 | SL hit qty=1.00 sl=1719.75 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 1818.00 | 1726.01 | 1725.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 10:15:00 | 1829.00 | 1727.04 | 1726.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-09 09:15:00 | 1753.30 | 1762.34 | 1746.74 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 1753.30 | 1762.34 | 1746.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1753.30 | 1762.34 | 1746.74 | EMA400 retest candle locked |

### Cycle 7 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 1681.10 | 1736.05 | 1736.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 1670.60 | 1735.40 | 1735.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1719.10 | 1702.91 | 1715.74 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1719.10 | 1702.91 | 1715.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1719.10 | 1702.91 | 1715.74 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-12 13:15:00 | 1691.50 | 1702.91 | 1715.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 14:15:00 | 1687.80 | 1702.76 | 1715.35 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 1726.60 | 1679.05 | 1691.57 | SL hit qty=1.00 sl=1726.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-16 14:15:00 | 1700.90 | 1682.54 | 1692.76 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 15:15:00 | 1702.00 | 1682.73 | 1692.80 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-17 14:15:00 | 1703.10 | 1684.20 | 1693.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 15:15:00 | 1702.60 | 1684.39 | 1693.30 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-28 14:15:00 | 1700.50 | 1686.88 | 1692.75 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 15:15:00 | 1699.20 | 1687.01 | 1692.78 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 1705.90 | 1687.19 | 1692.85 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1726.60 | 1689.06 | 1693.60 | SL hit qty=1.00 sl=1726.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1726.60 | 1689.06 | 1693.60 | SL hit qty=1.00 sl=1726.60 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 1726.60 | 1689.06 | 1693.60 | SL hit qty=1.00 sl=1726.60 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-01 09:15:00 | 1638.70 | 1693.08 | 1695.40 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 10:15:00 | 1641.00 | 1692.57 | 1695.12 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| CROSSOVER_SKIP | 2025-10-20 12:15:00 | 1691.90 | 1640.29 | 1640.06 | HTF filter: close below htf_sma |
| Cross detected — sustain check pending | 2025-10-23 12:15:00 | 1695.00 | 1644.99 | 1642.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:15:00 | 1691.00 | 1645.45 | 1642.71 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-10-27 09:15:00 | 1691.90 | 1650.11 | 1645.23 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 10:15:00 | 1696.70 | 1650.57 | 1645.49 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 1714.90 | 1656.00 | 1648.64 | SL hit qty=1.00 sl=1714.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 1714.90 | 1656.00 | 1648.64 | SL hit qty=1.00 sl=1714.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 1714.90 | 1656.00 | 1648.64 | SL hit qty=1.00 sl=1714.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-30 09:15:00 | 1686.30 | 1659.19 | 1650.48 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:15:00 | 1695.60 | 1659.55 | 1650.70 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 1695.80 | 1659.91 | 1650.93 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-10-31 14:15:00 | 1689.10 | 1663.45 | 1653.18 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:15:00 | 1690.70 | 1663.72 | 1653.37 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 1698.40 | 1664.05 | 1653.58 | SL hit qty=1.00 sl=1698.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-04 10:15:00 | 1689.30 | 1666.72 | 1655.35 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 1688.00 | 1666.93 | 1655.52 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-11-04 13:15:00 | 1684.40 | 1667.36 | 1655.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 14:15:00 | 1687.60 | 1667.57 | 1656.01 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-11-04 15:15:00 | 1698.40 | 1667.79 | 1656.18 | SL hit qty=1.00 sl=1698.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 15:15:00 | 1698.40 | 1667.79 | 1656.18 | SL hit qty=1.00 sl=1698.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 1714.90 | 1667.83 | 1656.26 | SL hit qty=1.00 sl=1714.90 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-06 13:15:00 | 1686.00 | 1668.69 | 1656.92 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 14:15:00 | 1686.80 | 1668.87 | 1657.07 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 1686.00 | 1669.04 | 1657.21 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-11-07 09:15:00 | 1698.40 | 1669.26 | 1657.38 | SL hit qty=1.00 sl=1698.40 alert=retest2 |
| Cross detected — sustain check pending | 2025-11-07 12:15:00 | 1684.20 | 1669.82 | 1657.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-11-07 13:15:00 | 1694.30 | 1670.06 | 1658.02 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 1678.00 | 1744.01 | 1736.25 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 10:15:00 | 1677.70 | 1743.35 | 1735.96 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-20 15:15:00 | 1613.80 | 1728.49 | 1728.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 1613.80 | 1728.49 | 1728.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 13:15:00 | 1605.00 | 1699.82 | 1713.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1683.10 | 1679.90 | 1701.17 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1683.10 | 1679.90 | 1701.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1683.10 | 1679.90 | 1701.17 | EMA400 retest candle locked |

### Cycle 9 — BUY (started 2026-02-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 11:15:00 | 1777.80 | 1709.49 | 1709.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 15:15:00 | 1787.40 | 1712.26 | 1710.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 1759.70 | 1760.58 | 1740.50 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 14:15:00 | 1739.70 | 1760.37 | 1740.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1739.70 | 1760.37 | 1740.79 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-03-20 09:15:00 | 1761.20 | 1760.28 | 1740.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 10:15:00 | 1766.60 | 1760.34 | 1741.07 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-24 09:15:00 | 1761.80 | 1761.33 | 1742.80 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 1761.70 | 1761.33 | 1742.89 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-03-25 09:15:00 | 1775.60 | 1761.44 | 1743.49 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 10:15:00 | 1783.10 | 1761.65 | 1743.69 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2026-04-01 09:15:00 | 1777.80 | 1766.36 | 1747.90 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-01 10:15:00 | 1757.90 | 1766.28 | 1747.95 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-01 11:15:00 | 1761.50 | 1766.23 | 1748.02 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2026-04-01 12:15:00 | 1740.10 | 1765.97 | 1747.98 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 1738.30 | 1765.97 | 1747.98 | SL hit qty=1.00 sl=1738.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 1738.30 | 1765.97 | 1747.98 | SL hit qty=1.00 sl=1738.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-01 12:15:00 | 1738.30 | 1765.97 | 1747.98 | SL hit qty=1.00 sl=1738.30 alert=retest2 |

### Cycle 10 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 1653.60 | 1734.06 | 1734.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1640.00 | 1713.09 | 1722.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1737.50 | 1708.13 | 1719.77 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1737.50 | 1708.13 | 1719.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1737.50 | 1708.13 | 1719.77 | EMA400 retest candle locked |

### Cycle 11 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 1829.70 | 1729.44 | 1729.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1842.70 | 1735.85 | 1732.46 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-20 10:15:00 | 1475.40 | 2024-06-26 11:15:00 | 1509.85 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest1 | 2024-10-28 14:15:00 | 1901.30 | 2024-10-30 09:15:00 | 1830.80 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-04-07 10:15:00 | 1649.45 | 2025-04-11 09:15:00 | 1719.75 | STOP_HIT | 1.00 | -4.26% |
| SELL | retest2 | 2025-04-09 10:15:00 | 1677.60 | 2025-04-11 09:15:00 | 1719.75 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2025-04-11 15:15:00 | 1687.90 | 2025-04-15 09:15:00 | 1719.75 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-06-12 14:15:00 | 1687.80 | 2025-07-15 11:15:00 | 1726.60 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-07-16 15:15:00 | 1702.00 | 2025-07-30 09:15:00 | 1726.60 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-07-17 15:15:00 | 1702.60 | 2025-07-30 09:15:00 | 1726.60 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-07-28 15:15:00 | 1699.20 | 2025-07-30 09:15:00 | 1726.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-08-01 10:15:00 | 1641.00 | 2025-10-29 10:15:00 | 1714.90 | STOP_HIT | 1.00 | -4.50% |
| SELL | retest2 | 2025-10-23 13:15:00 | 1691.00 | 2025-10-29 10:15:00 | 1714.90 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-10-27 10:15:00 | 1696.70 | 2025-10-29 10:15:00 | 1714.90 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-30 10:15:00 | 1695.60 | 2025-11-03 09:15:00 | 1698.40 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest2 | 2025-10-31 15:15:00 | 1690.70 | 2025-11-04 15:15:00 | 1698.40 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-11-04 11:15:00 | 1688.00 | 2025-11-04 15:15:00 | 1698.40 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-11-04 14:15:00 | 1687.60 | 2025-11-06 09:15:00 | 1714.90 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-06 14:15:00 | 1686.80 | 2025-11-07 09:15:00 | 1698.40 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2026-01-16 10:15:00 | 1677.70 | 2026-01-20 15:15:00 | 1613.80 | STOP_HIT | 1.00 | 3.81% |
| BUY | retest2 | 2026-03-20 10:15:00 | 1766.60 | 2026-04-01 12:15:00 | 1738.30 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-03-24 10:15:00 | 1761.70 | 2026-04-01 12:15:00 | 1738.30 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-03-25 10:15:00 | 1783.10 | 2026-04-01 12:15:00 | 1738.30 | STOP_HIT | 1.00 | -2.51% |
