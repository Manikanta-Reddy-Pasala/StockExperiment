# SUNPHARMA (SUNPHARMA)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1832.40
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 5 |
| ALERT3 | 7 |
| PENDING | 36 |
| PENDING_CANCEL | 19 |
| ENTRY1 | 0 |
| ENTRY2 | 16 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 16 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 14
- **Target hits / Stop hits / Partials:** 0 / 16 / 0
- **Avg / median % per leg:** -1.98% / -2.36%
- **Sum % (uncompounded):** -31.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 16 | 2 | 12.5% | 0 | 16 | 0 | -1.98% | -31.6% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 2 | 12.5% | 0 | 16 | 0 | -1.98% | -31.6% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 16 | 2 | 12.5% | 0 | 16 | 0 | -1.98% | -31.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 09:15:00 | 1745.50 | 1811.76 | 1812.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 1729.80 | 1810.94 | 1811.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 1810.50 | 1805.63 | 1808.83 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 1810.50 | 1805.63 | 1808.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1810.50 | 1805.63 | 1808.83 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-03 09:15:00 | 1790.60 | 1805.44 | 1808.62 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-03 10:15:00 | 1796.15 | 1805.35 | 1808.56 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-05 10:15:00 | 1782.50 | 1804.23 | 1807.77 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-05 11:15:00 | 1804.75 | 1804.24 | 1807.76 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2024-12-13 09:15:00 | 1788.65 | 1805.54 | 1807.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2024-12-13 11:15:00 | 1800.85 | 1805.27 | 1807.69 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2024-12-17 11:15:00 | 1791.65 | 1805.21 | 1807.49 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 13:15:00 | 1788.10 | 1804.87 | 1807.30 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2024-12-19 13:15:00 | 1815.85 | 1804.70 | 1807.04 | SL hit (close>static) qty=1.00 sl=1812.55 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-10 11:15:00 | 1782.65 | 1832.47 | 1823.45 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 13:15:00 | 1779.35 | 1831.44 | 1823.02 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-01-16 10:15:00 | 1748.65 | 1815.25 | 1815.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 10:15:00 | 1748.65 | 1815.25 | 1815.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1717.30 | 1806.28 | 1810.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 10:15:00 | 1668.80 | 1668.44 | 1714.01 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-17 09:15:00 | 1715.60 | 1670.02 | 1711.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1715.60 | 1670.02 | 1711.95 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-04-01 12:15:00 | 1686.45 | 1708.37 | 1721.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-01 13:15:00 | 1691.35 | 1708.20 | 1721.71 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-04 09:15:00 | 1675.55 | 1712.50 | 1722.81 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-04 10:15:00 | 1699.75 | 1712.37 | 1722.69 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-07 09:15:00 | 1649.50 | 1711.56 | 1721.98 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-07 11:15:00 | 1655.00 | 1710.38 | 1721.29 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-04-08 14:15:00 | 1687.15 | 1707.60 | 1719.32 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-08 15:15:00 | 1691.00 | 1707.44 | 1719.18 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-04-09 09:15:00 | 1671.90 | 1707.08 | 1718.94 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 11:15:00 | 1673.15 | 1706.45 | 1718.51 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-04-11 14:15:00 | 1687.85 | 1703.86 | 1716.59 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-15 09:15:00 | 1718.20 | 1703.85 | 1716.45 | ENTRY2 sustain failed after 5460m |
| Cross detected — sustain check pending | 2025-04-16 09:15:00 | 1688.30 | 1703.76 | 1715.98 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-04-16 10:15:00 | 1693.60 | 1703.66 | 1715.86 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-04-17 11:15:00 | 1725.60 | 1703.41 | 1715.26 | SL hit (close>static) qty=1.00 sl=1720.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-17 11:15:00 | 1725.60 | 1703.41 | 1715.26 | SL hit (close>static) qty=1.00 sl=1720.15 alert=retest2 |
| Cross detected — sustain check pending | 2025-05-12 09:15:00 | 1686.40 | 1760.19 | 1746.01 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-12 11:15:00 | 1695.80 | 1758.82 | 1745.46 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-05-12 13:15:00 | 1682.80 | 1757.42 | 1744.89 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-05-12 15:15:00 | 1689.90 | 1756.02 | 1744.31 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-05-23 09:15:00 | 1670.90 | 1740.83 | 1738.44 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 1670.20 | 1739.43 | 1737.76 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-05-26 09:15:00 | 1688.30 | 1736.57 | 1736.35 | ENTRY2 cross detected — sustain check pending (75m) |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 1681.10 | 1736.02 | 1736.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 1681.10 | 1736.02 | 1736.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 1670.40 | 1735.37 | 1735.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1719.10 | 1702.86 | 1715.62 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1719.10 | 1702.86 | 1715.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1719.10 | 1702.86 | 1715.62 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-06-12 13:15:00 | 1691.50 | 1702.86 | 1715.37 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 15:15:00 | 1684.70 | 1702.53 | 1715.08 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-06-13 12:15:00 | 1690.50 | 1701.92 | 1714.53 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 14:15:00 | 1688.40 | 1701.62 | 1714.25 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-06-30 09:15:00 | 1692.00 | 1682.86 | 1699.21 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-06-30 10:15:00 | 1700.80 | 1683.04 | 1699.21 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-06-30 12:15:00 | 1684.80 | 1683.18 | 1699.12 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-30 14:15:00 | 1675.70 | 1683.14 | 1698.95 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-07-14 13:15:00 | 1688.80 | 1677.92 | 1691.28 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:15:00 | 1681.70 | 1677.99 | 1691.18 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 1701.20 | 1678.22 | 1691.23 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 1728.70 | 1680.40 | 1692.01 | SL hit (close>static) qty=1.00 sl=1727.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 1728.70 | 1680.40 | 1692.01 | SL hit (close>static) qty=1.00 sl=1727.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 1728.70 | 1680.40 | 1692.01 | SL hit (close>static) qty=1.00 sl=1727.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-15 14:15:00 | 1728.70 | 1680.40 | 1692.01 | SL hit (close>static) qty=1.00 sl=1727.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-22 09:15:00 | 1674.10 | 1685.50 | 1693.19 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:15:00 | 1677.80 | 1685.34 | 1693.03 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2025-07-29 10:15:00 | 1711.80 | 1687.45 | 1692.90 | SL hit (close>static) qty=1.00 sl=1710.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-01 09:15:00 | 1638.70 | 1693.02 | 1695.33 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:15:00 | 1636.00 | 1691.95 | 1694.76 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-10-20 15:15:00 | 1687.80 | 1641.67 | 1640.77 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-21 13:15:00 | 1691.40 | 1642.16 | 1641.02 | ENTRY2 sustain failed after 1320m |
| Cross detected — sustain check pending | 2025-10-23 14:15:00 | 1686.70 | 1645.81 | 1642.92 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-23 15:15:00 | 1695.00 | 1646.30 | 1643.17 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-10-28 10:15:00 | 1685.00 | 1653.36 | 1647.09 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:15:00 | 1683.80 | 1653.94 | 1647.44 | SELL ENTRY2 attempt 3/4 (retest2 break sustained 120m) |
| Cross detected — sustain check pending | 2025-10-28 14:15:00 | 1686.30 | 1654.60 | 1647.84 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-28 15:15:00 | 1692.80 | 1654.98 | 1648.06 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 1712.90 | 1655.99 | 1648.64 | SL hit (close>static) qty=1.00 sl=1710.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 1712.90 | 1655.99 | 1648.64 | SL hit (close>static) qty=1.00 sl=1710.80 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-30 09:15:00 | 1686.20 | 1659.12 | 1650.45 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-10-30 10:15:00 | 1695.60 | 1659.49 | 1650.67 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-11-04 13:15:00 | 1684.40 | 1667.32 | 1655.83 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-11-04 15:15:00 | 1702.70 | 1667.87 | 1656.22 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-11-06 09:15:00 | 1672.20 | 1667.91 | 1656.30 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-11-06 11:15:00 | 1690.30 | 1668.30 | 1656.61 | ENTRY2 sustain failed after 120m |
| Cross detected — sustain check pending | 2025-11-06 13:15:00 | 1686.00 | 1668.77 | 1656.96 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:15:00 | 1686.00 | 1669.11 | 1657.25 | SELL ENTRY2 attempt 4/4 (retest2 break sustained 120m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1690.90 | 1669.33 | 1657.42 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-11-07 12:15:00 | 1684.30 | 1669.88 | 1657.87 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2025-11-07 13:15:00 | 1694.30 | 1670.12 | 1658.05 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-11-11 14:15:00 | 1717.30 | 1674.10 | 1660.98 | SL hit (close>static) qty=1.00 sl=1710.80 alert=retest2 |
| Cross detected — sustain check pending | 2026-01-16 09:15:00 | 1678.00 | 1744.02 | 1736.25 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:15:00 | 1675.10 | 1742.68 | 1735.66 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-01-20 15:15:00 | 1621.30 | 1728.59 | 1728.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 1621.30 | 1728.59 | 1728.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 1587.80 | 1697.02 | 1711.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.24 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.24 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2026-04-02 09:15:00 | 1659.20 | 1763.41 | 1746.42 | ENTRY2 cross detected — sustain check pending (75m) |
| Sustain check cancelled (price retraced) | 2026-04-02 10:15:00 | 1680.00 | 1762.58 | 1746.09 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2026-04-06 09:15:00 | 1663.60 | 1758.13 | 1744.32 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:15:00 | 1670.50 | 1756.36 | 1743.56 | SELL ENTRY2 attempt 1/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-09 11:15:00 | 1726.00 | 1747.44 | 1740.12 | SL hit (close>static) qty=1.00 sl=1725.00 alert=retest2 |
| Cross detected — sustain check pending | 2026-04-10 09:15:00 | 1656.70 | 1745.30 | 1739.23 | ENTRY2 cross detected — sustain check pending (75m) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 11:15:00 | 1631.50 | 1743.13 | 1738.20 | SELL ENTRY2 attempt 2/4 (retest2 break sustained 120m) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1670.00 | 1733.20 | 1733.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 09:15:00 | 1670.00 | 1733.20 | 1733.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1640.00 | 1711.32 | 1721.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1737.50 | 1706.49 | 1718.31 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1737.50 | 1706.49 | 1718.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1737.50 | 1706.49 | 1718.31 | EMA400 retest candle locked |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-17 13:15:00 | 1788.10 | 2024-12-19 13:15:00 | 1815.85 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-01-10 13:15:00 | 1779.35 | 2025-01-16 10:15:00 | 1748.65 | STOP_HIT | 1.00 | 1.73% |
| SELL | retest2 | 2025-04-07 11:15:00 | 1655.00 | 2025-04-17 11:15:00 | 1725.60 | STOP_HIT | 1.00 | -4.27% |
| SELL | retest2 | 2025-04-09 11:15:00 | 1673.15 | 2025-04-17 11:15:00 | 1725.60 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2025-05-23 11:15:00 | 1670.20 | 2025-05-26 10:15:00 | 1681.10 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-06-12 15:15:00 | 1684.70 | 2025-07-15 14:15:00 | 1728.70 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2025-06-13 14:15:00 | 1688.40 | 2025-07-15 14:15:00 | 1728.70 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2025-06-30 14:15:00 | 1675.70 | 2025-07-15 14:15:00 | 1728.70 | STOP_HIT | 1.00 | -3.16% |
| SELL | retest2 | 2025-07-14 15:15:00 | 1681.70 | 2025-07-15 14:15:00 | 1728.70 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-07-22 11:15:00 | 1677.80 | 2025-07-29 10:15:00 | 1711.80 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-08-01 11:15:00 | 1636.00 | 2025-10-29 10:15:00 | 1712.90 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2025-10-28 12:15:00 | 1683.80 | 2025-10-29 10:15:00 | 1712.90 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-11-06 15:15:00 | 1686.00 | 2025-11-11 14:15:00 | 1717.30 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2026-01-16 11:15:00 | 1675.10 | 2026-01-20 15:15:00 | 1621.30 | STOP_HIT | 1.00 | 3.21% |
| SELL | retest2 | 2026-04-06 11:15:00 | 1670.50 | 2026-04-09 11:15:00 | 1726.00 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2026-04-10 11:15:00 | 1631.50 | 2026-04-15 09:15:00 | 1670.00 | STOP_HIT | 1.00 | -2.36% |
