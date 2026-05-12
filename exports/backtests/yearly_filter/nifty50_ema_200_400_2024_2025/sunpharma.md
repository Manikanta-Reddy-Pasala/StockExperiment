# SUNPHARMA (SUNPHARMA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1845.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 12 |
| ALERT2 | 11 |
| ALERT2_SKIP | 5 |
| ALERT3 | 51 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 54 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 56 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 53
- **Target hits / Stop hits / Partials:** 0 / 56 / 0
- **Avg / median % per leg:** -1.77% / -1.21%
- **Sum % (uncompounded):** -98.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 0 | 0.0% | 0 | 20 | 0 | -2.04% | -40.9% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.12% | -10.2% |
| BUY @ 3rd Alert (retest2) | 18 | 0 | 0.0% | 0 | 18 | 0 | -1.70% | -30.6% |
| SELL (all) | 36 | 3 | 8.3% | 0 | 36 | 0 | -1.61% | -58.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 36 | 3 | 8.3% | 0 | 36 | 0 | -1.61% | -58.1% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -5.12% | -10.2% |
| retest2 (combined) | 54 | 3 | 5.6% | 0 | 54 | 0 | -1.64% | -88.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 1451.20 | 1511.05 | 1511.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 11:15:00 | 1446.10 | 1510.41 | 1511.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-07 12:15:00 | 1498.35 | 1496.90 | 1503.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-07 13:00:00 | 1498.35 | 1496.90 | 1503.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 13:15:00 | 1508.35 | 1497.01 | 1503.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 14:00:00 | 1508.35 | 1497.01 | 1503.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 14:15:00 | 1509.40 | 1497.14 | 1503.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-07 15:15:00 | 1506.70 | 1497.14 | 1503.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 15:15:00 | 1506.70 | 1497.23 | 1503.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-10 09:15:00 | 1509.05 | 1497.23 | 1503.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 1499.55 | 1498.88 | 1504.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 09:30:00 | 1498.60 | 1499.53 | 1504.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 14:15:00 | 1510.50 | 1499.92 | 1504.19 | SL hit (close>static) qty=1.00 sl=1510.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 14:15:00 | 1535.65 | 1505.99 | 1505.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 1537.20 | 1506.54 | 1506.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-23 09:15:00 | 1873.30 | 1880.44 | 1824.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 12:30:00 | 1889.70 | 1875.87 | 1828.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-10-28 13:15:00 | 1893.10 | 1875.87 | 1828.14 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1837.80 | 1875.86 | 1830.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:30:00 | 1840.30 | 1875.86 | 1830.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 17:15:00 | 1860.40 | 1873.36 | 1832.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 1860.40 | 1873.36 | 1832.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1794.65 | 1872.44 | 1832.41 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 1794.65 | 1872.44 | 1832.41 | SL hit (close<ema400) qty=1.00 sl=1832.41 alert=retest1 |

### Cycle 3 — SELL (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 09:15:00 | 1745.50 | 1811.76 | 1811.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 10:15:00 | 1729.80 | 1810.94 | 1811.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 1810.50 | 1805.63 | 1808.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 1810.50 | 1805.63 | 1808.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1810.50 | 1805.63 | 1808.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 1809.35 | 1805.63 | 1808.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 1802.65 | 1805.60 | 1808.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 12:00:00 | 1800.30 | 1805.55 | 1808.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 09:15:00 | 1792.35 | 1805.59 | 1808.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 14:45:00 | 1800.15 | 1805.17 | 1808.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 09:45:00 | 1797.25 | 1804.97 | 1808.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1793.00 | 1804.45 | 1807.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 10:45:00 | 1781.30 | 1804.23 | 1807.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 13:15:00 | 1815.35 | 1804.38 | 1807.71 | SL hit (close>static) qty=1.00 sl=1814.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1850.20 | 1809.03 | 1808.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 1859.15 | 1810.35 | 1809.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-08 15:15:00 | 1832.80 | 1834.09 | 1823.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 09:15:00 | 1822.10 | 1834.09 | 1823.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1820.05 | 1833.95 | 1823.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 1820.05 | 1833.95 | 1823.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1821.10 | 1833.82 | 1823.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:45:00 | 1819.20 | 1833.82 | 1823.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 11:15:00 | 1826.35 | 1833.75 | 1823.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 12:15:00 | 1828.90 | 1833.75 | 1823.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 14:00:00 | 1830.65 | 1833.68 | 1823.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-09 15:15:00 | 1830.05 | 1833.62 | 1823.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-10 09:15:00 | 1811.60 | 1833.36 | 1823.77 | SL hit (close<static) qty=1.00 sl=1820.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 10:15:00 | 1748.65 | 1815.25 | 1815.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-28 09:15:00 | 1717.30 | 1806.28 | 1810.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 10:15:00 | 1668.80 | 1668.44 | 1714.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 11:00:00 | 1668.80 | 1668.44 | 1714.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 1715.60 | 1670.02 | 1711.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:30:00 | 1717.35 | 1670.02 | 1711.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 1709.65 | 1670.42 | 1711.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-17 12:00:00 | 1702.40 | 1670.73 | 1711.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 09:15:00 | 1721.10 | 1672.49 | 1711.76 | SL hit (close>static) qty=1.00 sl=1718.50 alert=retest2 |

### Cycle 6 — BUY (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-25 15:15:00 | 1781.80 | 1725.07 | 1724.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 09:15:00 | 1817.90 | 1726.00 | 1725.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 15:15:00 | 1755.00 | 1762.28 | 1746.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-09 09:15:00 | 1749.90 | 1762.28 | 1746.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 1753.30 | 1762.19 | 1746.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 1753.30 | 1762.19 | 1746.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 1740.80 | 1761.98 | 1746.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 11:00:00 | 1740.80 | 1761.98 | 1746.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 1740.10 | 1761.76 | 1746.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:15:00 | 1739.20 | 1761.76 | 1746.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 1734.50 | 1761.49 | 1746.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:45:00 | 1733.70 | 1761.49 | 1746.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 1748.00 | 1760.94 | 1746.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 1646.20 | 1760.94 | 1746.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1686.40 | 1760.19 | 1746.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 09:15:00 | 1724.00 | 1756.02 | 1744.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 1719.30 | 1747.22 | 1740.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 15:15:00 | 1714.00 | 1744.07 | 1739.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 13:00:00 | 1715.40 | 1742.19 | 1739.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 1681.10 | 1736.02 | 1736.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 1681.10 | 1736.02 | 1736.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 11:15:00 | 1670.40 | 1735.37 | 1735.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-12 09:15:00 | 1719.10 | 1702.86 | 1715.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 1719.10 | 1702.86 | 1715.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 1719.10 | 1702.86 | 1715.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:30:00 | 1715.40 | 1702.86 | 1715.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1707.30 | 1702.90 | 1715.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-12 13:15:00 | 1702.00 | 1702.97 | 1715.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 1726.80 | 1678.98 | 1691.49 | SL hit (close>static) qty=1.00 sl=1723.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 1691.90 | 1640.23 | 1640.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 09:15:00 | 1701.40 | 1643.22 | 1641.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 09:15:00 | 1754.20 | 1771.79 | 1737.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 1743.50 | 1771.27 | 1737.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1743.50 | 1771.27 | 1737.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:15:00 | 1754.40 | 1770.18 | 1737.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1757.80 | 1768.55 | 1737.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 10:15:00 | 1754.90 | 1768.37 | 1738.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 1735.00 | 1767.12 | 1739.58 | SL hit (close<static) qty=1.00 sl=1737.20 alert=retest2 |

### Cycle 9 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 1621.30 | 1728.59 | 1728.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 09:15:00 | 1587.80 | 1697.02 | 1711.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1683.00 | 1675.58 | 1698.24 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2026-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 10:15:00 | 1780.10 | 1707.48 | 1707.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 1804.10 | 1724.48 | 1716.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 10:15:00 | 1759.70 | 1760.04 | 1739.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 11:00:00 | 1759.70 | 1760.04 | 1739.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 1739.70 | 1759.85 | 1739.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 1739.70 | 1759.85 | 1739.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 1749.60 | 1759.75 | 1739.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1764.40 | 1759.75 | 1739.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 1775.50 | 1760.70 | 1741.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-25 09:15:00 | 1770.00 | 1760.71 | 1742.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 15:15:00 | 1762.00 | 1765.89 | 1746.83 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 1740.10 | 1765.60 | 1747.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:00:00 | 1740.10 | 1765.60 | 1747.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1721.50 | 1765.16 | 1747.03 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 1721.50 | 1765.16 | 1747.03 | SL hit (close<static) qty=1.00 sl=1738.90 alert=retest2 |

### Cycle 11 — SELL (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 09:15:00 | 1670.00 | 1733.20 | 1733.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 1640.00 | 1711.32 | 1721.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1737.50 | 1706.49 | 1718.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1737.50 | 1706.49 | 1718.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1737.50 | 1706.49 | 1718.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1737.50 | 1706.49 | 1718.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1722.30 | 1706.64 | 1718.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 1741.40 | 1706.64 | 1718.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 1757.80 | 1708.76 | 1719.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 11:00:00 | 1757.80 | 1708.76 | 1719.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 10:15:00 | 1829.70 | 1728.28 | 1727.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 10:15:00 | 1842.70 | 1734.74 | 1731.25 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-06-13 09:30:00 | 1498.60 | 2024-06-13 14:15:00 | 1510.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-06-20 09:15:00 | 1477.35 | 2024-06-26 11:15:00 | 1511.75 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-06-24 12:30:00 | 1498.00 | 2024-06-26 11:15:00 | 1511.75 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-06-25 11:15:00 | 1496.55 | 2024-06-26 11:15:00 | 1511.75 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-06-26 09:15:00 | 1498.00 | 2024-06-26 11:15:00 | 1511.75 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest1 | 2024-10-28 12:30:00 | 1889.70 | 2024-11-04 09:15:00 | 1794.65 | STOP_HIT | 1.00 | -5.03% |
| BUY | retest1 | 2024-10-28 13:15:00 | 1893.10 | 2024-11-04 09:15:00 | 1794.65 | STOP_HIT | 1.00 | -5.20% |
| SELL | retest2 | 2024-12-02 12:00:00 | 1800.30 | 2024-12-05 13:15:00 | 1815.35 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-12-03 09:15:00 | 1792.35 | 2024-12-05 13:15:00 | 1815.35 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-12-03 14:45:00 | 1800.15 | 2024-12-05 13:15:00 | 1815.35 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-12-04 09:45:00 | 1797.25 | 2024-12-05 13:15:00 | 1815.35 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-12-05 10:45:00 | 1781.30 | 2024-12-05 13:15:00 | 1815.35 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-12-13 09:45:00 | 1791.10 | 2024-12-13 13:15:00 | 1809.60 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-12-17 12:00:00 | 1791.65 | 2024-12-18 10:15:00 | 1808.95 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-12-17 13:00:00 | 1787.30 | 2024-12-18 10:15:00 | 1808.95 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-12-18 14:45:00 | 1799.45 | 2024-12-19 12:15:00 | 1810.55 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2024-12-18 15:15:00 | 1799.00 | 2024-12-19 12:15:00 | 1810.55 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2024-12-20 12:45:00 | 1798.45 | 2024-12-23 10:15:00 | 1820.00 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-12-23 09:30:00 | 1799.00 | 2024-12-23 10:15:00 | 1820.00 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-01-09 12:15:00 | 1828.90 | 2025-01-10 09:15:00 | 1811.60 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-01-09 14:00:00 | 1830.65 | 2025-01-10 09:15:00 | 1811.60 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-01-09 15:15:00 | 1830.05 | 2025-01-10 09:15:00 | 1811.60 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-03-17 12:00:00 | 1702.40 | 2025-03-18 09:15:00 | 1721.10 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-04-01 10:15:00 | 1704.30 | 2025-04-03 09:15:00 | 1780.25 | STOP_HIT | 1.00 | -4.46% |
| SELL | retest2 | 2025-04-01 10:45:00 | 1700.60 | 2025-04-03 09:15:00 | 1780.25 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2025-04-04 09:45:00 | 1694.20 | 2025-04-17 11:15:00 | 1725.60 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-04-11 10:30:00 | 1688.10 | 2025-04-17 13:15:00 | 1743.40 | STOP_HIT | 1.00 | -3.28% |
| SELL | retest2 | 2025-04-11 12:45:00 | 1691.75 | 2025-04-17 13:15:00 | 1743.40 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-04-11 13:45:00 | 1691.80 | 2025-04-17 13:15:00 | 1743.40 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2025-04-16 09:30:00 | 1688.40 | 2025-04-17 13:15:00 | 1743.40 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-05-13 09:15:00 | 1724.00 | 2025-05-26 10:15:00 | 1681.10 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2025-05-15 13:00:00 | 1719.30 | 2025-05-26 10:15:00 | 1681.10 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-05-20 15:15:00 | 1714.00 | 2025-05-26 10:15:00 | 1681.10 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-05-22 13:00:00 | 1715.40 | 2025-05-26 10:15:00 | 1681.10 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2025-06-12 13:15:00 | 1702.00 | 2025-07-15 11:15:00 | 1726.80 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-07-16 14:15:00 | 1704.70 | 2025-07-18 12:15:00 | 1695.10 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2025-07-16 15:00:00 | 1701.00 | 2025-07-21 09:15:00 | 1696.70 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2025-07-17 14:45:00 | 1704.40 | 2025-07-25 10:15:00 | 1697.40 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-07-18 11:30:00 | 1687.10 | 2025-07-25 10:15:00 | 1697.40 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-07-21 09:15:00 | 1681.90 | 2025-07-30 09:15:00 | 1726.40 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-07-22 09:15:00 | 1679.70 | 2025-07-30 09:15:00 | 1726.40 | STOP_HIT | 1.00 | -2.78% |
| SELL | retest2 | 2025-07-24 09:30:00 | 1687.60 | 2025-07-30 09:15:00 | 1726.40 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-08-01 09:15:00 | 1619.00 | 2025-10-20 09:15:00 | 1693.50 | STOP_HIT | 1.00 | -4.60% |
| SELL | retest2 | 2025-10-17 12:45:00 | 1682.00 | 2025-10-20 09:15:00 | 1693.50 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-10-17 14:45:00 | 1680.00 | 2025-10-20 09:15:00 | 1693.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-12-19 09:15:00 | 1754.40 | 2025-12-24 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-22 09:15:00 | 1757.80 | 2025-12-24 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-12-22 10:15:00 | 1754.90 | 2025-12-24 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-01-06 13:30:00 | 1754.30 | 2026-01-09 13:15:00 | 1728.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-01-09 15:15:00 | 1736.10 | 2026-01-12 09:15:00 | 1719.70 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-01-12 13:30:00 | 1734.30 | 2026-01-13 09:15:00 | 1719.10 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2026-01-13 14:45:00 | 1732.40 | 2026-01-14 09:15:00 | 1704.00 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1764.40 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2026-03-24 09:15:00 | 1775.50 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -3.04% |
| BUY | retest2 | 2026-03-25 09:15:00 | 1770.00 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2026-03-30 15:15:00 | 1762.00 | 2026-04-01 13:15:00 | 1721.50 | STOP_HIT | 1.00 | -2.30% |
