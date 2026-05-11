# Coforge Ltd. (COFORGE)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 1365.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 6 |
| ALERT2_SKIP | 1 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / Stop hits / Partials:** 1 / 7 / 1
- **Avg / median % per leg:** -0.48% / -2.51%
- **Sum % (uncompounded):** -4.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.89% | -11.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.89% | -11.5% |
| SELL (all) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.45% | 7.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 1 | 3 | 1 | 1.45% | 7.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 2 | 22.2% | 1 | 7 | 1 | -0.48% | -4.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 1684.50 | 1530.34 | 1530.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 1696.10 | 1540.71 | 1535.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 1856.00 | 1856.70 | 1780.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:15:00 | 1858.60 | 1856.70 | 1780.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 1698.60 | 1855.79 | 1790.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:00:00 | 1698.60 | 1855.79 | 1790.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 1702.90 | 1854.27 | 1789.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 09:15:00 | 1723.00 | 1846.13 | 1787.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 1679.80 | 1843.05 | 1786.20 | SL hit (close<static) qty=1.00 sl=1683.70 alert=retest2 |

### Cycle 2 — SELL (started 2025-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 10:15:00 | 1637.60 | 1752.48 | 1752.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 11:15:00 | 1630.40 | 1751.26 | 1751.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 12:15:00 | 1719.80 | 1719.09 | 1734.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 13:00:00 | 1719.80 | 1719.09 | 1734.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 1728.90 | 1718.59 | 1733.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 1728.90 | 1718.59 | 1733.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 1743.00 | 1718.89 | 1733.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 1719.50 | 1725.71 | 1735.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 1750.00 | 1726.09 | 1735.35 | SL hit (close>static) qty=1.00 sl=1748.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 1810.70 | 1737.87 | 1737.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 13:15:00 | 1816.00 | 1740.10 | 1738.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 1732.60 | 1745.83 | 1741.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 1732.60 | 1745.83 | 1741.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1732.60 | 1745.83 | 1741.98 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 1636.10 | 1737.68 | 1738.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 1629.70 | 1736.61 | 1737.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 1686.00 | 1678.07 | 1703.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:45:00 | 1682.50 | 1678.07 | 1703.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1708.20 | 1678.45 | 1703.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 1726.40 | 1678.45 | 1703.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1695.40 | 1678.62 | 1703.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:45:00 | 1690.70 | 1688.01 | 1704.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 1750.00 | 1689.12 | 1705.01 | SL hit (close>static) qty=1.00 sl=1710.20 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 1832.10 | 1717.88 | 1717.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 1839.70 | 1765.84 | 1747.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 11:15:00 | 1847.10 | 1852.34 | 1807.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-10 11:45:00 | 1850.40 | 1852.34 | 1807.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1797.70 | 1851.57 | 1818.78 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 1661.90 | 1793.23 | 1793.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 14:15:00 | 1660.20 | 1790.61 | 1792.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 09:15:00 | 1739.20 | 1736.04 | 1758.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 10:15:00 | 1732.20 | 1736.04 | 1758.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1706.10 | 1700.14 | 1730.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 1607.10 | 1701.16 | 1730.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 12:15:00 | 1526.74 | 1656.88 | 1700.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-12 09:15:00 | 1446.39 | 1650.88 | 1697.00 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-25 09:15:00 | 1723.00 | 2025-07-25 10:15:00 | 1679.80 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-07-28 09:45:00 | 1719.00 | 2025-08-06 09:15:00 | 1667.40 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2025-07-28 11:15:00 | 1718.20 | 2025-08-06 09:15:00 | 1667.40 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-07-29 09:30:00 | 1720.40 | 2025-08-06 09:15:00 | 1667.40 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-08-28 09:15:00 | 1719.50 | 2025-08-28 13:15:00 | 1750.00 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-09-04 09:30:00 | 1711.90 | 2025-09-10 09:15:00 | 1754.20 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2025-10-14 11:45:00 | 1690.70 | 2025-10-15 09:15:00 | 1750.00 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2026-02-04 09:15:00 | 1607.10 | 2026-02-11 12:15:00 | 1526.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 09:15:00 | 1607.10 | 2026-02-12 09:15:00 | 1446.39 | TARGET_HIT | 0.50 | 10.00% |
