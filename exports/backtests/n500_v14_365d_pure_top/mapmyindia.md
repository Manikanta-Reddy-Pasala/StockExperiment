# C.E. Info Systems Ltd. (MAPMYINDIA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 957.00
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
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 23 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 20 |
| PARTIAL | 9 |
| TARGET_HIT | 8 |
| STOP_HIT | 12 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 11
- **Target hits / Stop hits / Partials:** 8 / 12 / 9
- **Avg / median % per leg:** 3.85% / 5.00%
- **Sum % (uncompounded):** 111.76%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.74% | -1.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.74% | -1.7% |
| SELL (all) | 28 | 18 | 64.3% | 8 | 11 | 9 | 4.05% | 113.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 18 | 64.3% | 8 | 11 | 9 | 4.05% | 113.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 18 | 62.1% | 8 | 12 | 9 | 3.85% | 111.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 1744.80 | 1822.33 | 1822.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 14:15:00 | 1736.90 | 1807.43 | 1814.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 14:15:00 | 1798.00 | 1793.76 | 1805.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 15:00:00 | 1798.00 | 1793.76 | 1805.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 1812.70 | 1793.42 | 1804.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 1812.70 | 1793.42 | 1804.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 15:15:00 | 1808.50 | 1793.57 | 1804.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:15:00 | 1801.00 | 1793.57 | 1804.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 1807.00 | 1793.38 | 1804.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 1807.00 | 1793.38 | 1804.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 1797.00 | 1793.42 | 1804.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 09:15:00 | 1785.70 | 1794.65 | 1804.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 11:30:00 | 1786.40 | 1794.36 | 1804.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 14:30:00 | 1787.60 | 1794.31 | 1803.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 1825.10 | 1794.62 | 1804.03 | SL hit (close>static) qty=1.00 sl=1809.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 1825.10 | 1794.62 | 1804.03 | SL hit (close>static) qty=1.00 sl=1809.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-18 15:15:00 | 1825.10 | 1794.62 | 1804.03 | SL hit (close>static) qty=1.00 sl=1809.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 14:15:00 | 1787.20 | 1805.86 | 1808.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1790.00 | 1794.41 | 1801.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 15:00:00 | 1767.30 | 1793.53 | 1801.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:30:00 | 1774.80 | 1792.16 | 1800.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 14:00:00 | 1774.80 | 1792.16 | 1800.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 1811.10 | 1792.35 | 1800.15 | SL hit (close>static) qty=1.00 sl=1809.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 1811.10 | 1792.35 | 1800.15 | SL hit (close>static) qty=1.00 sl=1804.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 1811.10 | 1792.35 | 1800.15 | SL hit (close>static) qty=1.00 sl=1804.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 1811.10 | 1792.35 | 1800.15 | SL hit (close>static) qty=1.00 sl=1804.50 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 1774.10 | 1792.27 | 1799.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 09:15:00 | 1685.39 | 1777.30 | 1789.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 1666.90 | 1659.90 | 1699.09 | SL hit (close>ema200) qty=0.50 sl=1659.90 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1687.90 | 1660.31 | 1698.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:45:00 | 1692.40 | 1660.31 | 1698.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 14:15:00 | 1686.10 | 1661.24 | 1698.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 14:45:00 | 1698.40 | 1661.24 | 1698.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 1697.00 | 1662.56 | 1697.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 1670.20 | 1663.77 | 1697.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 1706.00 | 1666.28 | 1697.02 | SL hit (close>static) qty=1.00 sl=1702.20 alert=retest2 |

### Cycle 2 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 1818.10 | 1722.19 | 1721.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-21 13:15:00 | 1854.00 | 1729.18 | 1725.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 15:15:00 | 1760.40 | 1763.45 | 1747.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-07 09:15:00 | 1753.20 | 1763.45 | 1747.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1752.10 | 1763.34 | 1747.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:30:00 | 1764.70 | 1763.31 | 1747.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 09:15:00 | 1734.00 | 1767.58 | 1750.34 | SL hit (close<static) qty=1.00 sl=1741.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 1700.20 | 1737.93 | 1738.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 14:15:00 | 1679.00 | 1735.89 | 1737.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1729.40 | 1712.98 | 1724.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 1729.40 | 1712.98 | 1724.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 1729.40 | 1712.98 | 1724.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 1688.90 | 1713.26 | 1723.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 14:00:00 | 1688.00 | 1711.98 | 1722.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 1692.20 | 1683.81 | 1699.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 10:15:00 | 1690.40 | 1683.92 | 1699.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1677.00 | 1684.01 | 1699.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 1681.50 | 1684.01 | 1699.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 1719.00 | 1682.84 | 1698.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 1719.00 | 1682.84 | 1698.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 1720.00 | 1683.21 | 1698.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 1721.20 | 1683.21 | 1698.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 1723.00 | 1683.94 | 1698.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 1714.60 | 1683.94 | 1698.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 1735.00 | 1684.45 | 1698.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:45:00 | 1734.50 | 1684.45 | 1698.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 1695.80 | 1686.50 | 1699.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:45:00 | 1696.00 | 1686.50 | 1699.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 1712.00 | 1686.86 | 1699.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 1712.50 | 1686.86 | 1699.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 1728.40 | 1687.28 | 1699.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 1728.40 | 1687.28 | 1699.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 1715.20 | 1687.55 | 1699.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 1711.80 | 1687.55 | 1699.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 13:15:00 | 1712.70 | 1688.46 | 1699.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 1740.00 | 1689.19 | 1700.15 | SL hit (close>static) qty=1.00 sl=1730.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 1740.00 | 1689.19 | 1700.15 | SL hit (close>static) qty=1.00 sl=1730.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1713.40 | 1689.45 | 1700.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 11:00:00 | 1712.30 | 1690.00 | 1700.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 1705.70 | 1690.31 | 1700.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:30:00 | 1699.90 | 1690.31 | 1700.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1704.80 | 1690.45 | 1700.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:15:00 | 1702.60 | 1690.45 | 1700.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1698.00 | 1690.53 | 1700.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 09:30:00 | 1690.80 | 1692.50 | 1700.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 1682.80 | 1692.83 | 1700.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 1627.73 | 1689.97 | 1698.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 1626.68 | 1689.97 | 1698.88 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 11:15:00 | 1607.59 | 1688.49 | 1698.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 11:15:00 | 1606.26 | 1688.49 | 1698.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 12:15:00 | 1604.45 | 1687.61 | 1697.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 12:15:00 | 1603.60 | 1687.61 | 1697.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 12:15:00 | 1605.88 | 1687.61 | 1697.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 12:15:00 | 1598.66 | 1687.61 | 1697.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-14 10:15:00 | 1542.06 | 1666.33 | 1685.55 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-14 11:15:00 | 1541.07 | 1665.04 | 1684.80 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-14 14:15:00 | 1522.98 | 1661.00 | 1682.48 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-14 14:15:00 | 1521.36 | 1661.00 | 1682.48 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-14 14:15:00 | 1521.72 | 1661.00 | 1682.48 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-16 10:15:00 | 1520.01 | 1656.90 | 1680.09 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-16 10:15:00 | 1519.20 | 1656.90 | 1680.09 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-16 10:15:00 | 1514.52 | 1656.90 | 1680.09 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-07-18 09:15:00 | 1785.70 | 2025-07-18 15:15:00 | 1825.10 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-07-18 11:30:00 | 1786.40 | 2025-07-18 15:15:00 | 1825.10 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-07-18 14:30:00 | 1787.60 | 2025-07-18 15:15:00 | 1825.10 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2025-08-01 14:15:00 | 1787.20 | 2025-08-11 14:15:00 | 1811.10 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-08-08 15:00:00 | 1767.30 | 2025-08-11 14:15:00 | 1811.10 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-08-11 13:30:00 | 1774.80 | 2025-08-11 14:15:00 | 1811.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-08-11 14:00:00 | 1774.80 | 2025-08-11 14:15:00 | 1811.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-08-13 15:00:00 | 1774.10 | 2025-08-26 09:15:00 | 1685.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-13 15:00:00 | 1774.10 | 2025-10-03 14:15:00 | 1666.90 | STOP_HIT | 0.50 | 6.04% |
| SELL | retest2 | 2025-10-08 15:15:00 | 1670.20 | 2025-10-10 09:15:00 | 1706.00 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-11-07 11:30:00 | 1764.70 | 2025-11-11 09:15:00 | 1734.00 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-12-04 09:15:00 | 1688.90 | 2026-01-01 14:15:00 | 1740.00 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-12-04 14:00:00 | 1688.00 | 2026-01-01 14:15:00 | 1740.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-12-26 09:15:00 | 1692.20 | 2026-01-09 09:15:00 | 1627.73 | PARTIAL | 0.50 | 3.81% |
| SELL | retest2 | 2025-12-26 10:15:00 | 1690.40 | 2026-01-09 09:15:00 | 1626.68 | PARTIAL | 0.50 | 3.77% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1711.80 | 2026-01-09 11:15:00 | 1607.59 | PARTIAL | 0.50 | 6.09% |
| SELL | retest2 | 2026-01-01 13:15:00 | 1712.70 | 2026-01-09 11:15:00 | 1606.26 | PARTIAL | 0.50 | 6.21% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1713.40 | 2026-01-09 12:15:00 | 1604.45 | PARTIAL | 0.50 | 6.36% |
| SELL | retest2 | 2026-01-02 11:00:00 | 1712.30 | 2026-01-09 12:15:00 | 1603.60 | PARTIAL | 0.50 | 6.35% |
| SELL | retest2 | 2026-01-07 09:30:00 | 1690.80 | 2026-01-09 12:15:00 | 1605.88 | PARTIAL | 0.50 | 5.02% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1682.80 | 2026-01-09 12:15:00 | 1598.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-26 09:15:00 | 1692.20 | 2026-01-14 10:15:00 | 1542.06 | TARGET_HIT | 0.50 | 8.87% |
| SELL | retest2 | 2025-12-26 10:15:00 | 1690.40 | 2026-01-14 11:15:00 | 1541.07 | TARGET_HIT | 0.50 | 8.83% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1711.80 | 2026-01-14 14:15:00 | 1522.98 | TARGET_HIT | 0.50 | 11.03% |
| SELL | retest2 | 2026-01-01 13:15:00 | 1712.70 | 2026-01-14 14:15:00 | 1521.36 | TARGET_HIT | 0.50 | 11.17% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1713.40 | 2026-01-14 14:15:00 | 1521.72 | TARGET_HIT | 0.50 | 11.19% |
| SELL | retest2 | 2026-01-02 11:00:00 | 1712.30 | 2026-01-16 10:15:00 | 1520.01 | TARGET_HIT | 0.50 | 11.23% |
| SELL | retest2 | 2026-01-07 09:30:00 | 1690.80 | 2026-01-16 10:15:00 | 1519.20 | TARGET_HIT | 0.50 | 10.15% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1682.80 | 2026-01-16 10:15:00 | 1514.52 | TARGET_HIT | 0.50 | 10.00% |
