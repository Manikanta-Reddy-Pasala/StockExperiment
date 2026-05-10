# Coforge Ltd. (COFORGE)

## Backtest Summary

- **Window:** 2025-04-21 09:15:00 → 2026-05-08 15:15:00 (1822 bars)
- **Last close:** 1365.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 5 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 5 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 3 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 3
- **Target hits / Stop hits / Partials:** 2 / 3 / 1
- **Avg / median % per leg:** 2.80% / 5.00%
- **Sum % (uncompounded):** 16.81%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 1 | 33.3% | 1 | 2 | 0 | 1.77% | 5.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 1 | 33.3% | 1 | 2 | 0 | 1.77% | 5.3% |
| SELL (all) | 3 | 2 | 66.7% | 1 | 1 | 1 | 3.83% | 11.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 2 | 66.7% | 1 | 1 | 1 | 3.83% | 11.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 3 | 50.0% | 2 | 3 | 1 | 2.80% | 16.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 1696.50 | 1719.30 | 1719.37 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 1759.80 | 1719.77 | 1719.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 1771.50 | 1720.28 | 1719.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 09:15:00 | 1732.60 | 1745.73 | 1734.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 1732.60 | 1745.73 | 1734.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 1732.60 | 1745.73 | 1734.11 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1594.00 | 1723.68 | 1724.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1571.40 | 1722.16 | 1723.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 14:15:00 | 1686.00 | 1678.02 | 1697.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:45:00 | 1682.50 | 1678.02 | 1697.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1708.20 | 1678.40 | 1697.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 1726.40 | 1678.40 | 1697.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1695.40 | 1678.57 | 1697.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 11:45:00 | 1690.70 | 1687.98 | 1700.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 09:15:00 | 1750.00 | 1689.09 | 1700.52 | SL hit (close>static) qty=1.00 sl=1710.20 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 11:15:00 | 1768.50 | 1709.88 | 1709.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1838.90 | 1713.12 | 1711.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 09:15:00 | 1736.30 | 1745.64 | 1730.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 09:15:00 | 1736.30 | 1745.64 | 1730.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 1736.30 | 1745.64 | 1730.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:15:00 | 1763.10 | 1745.20 | 1731.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-04 09:15:00 | 1939.41 | 1825.20 | 1786.04 | Target hit (10%) qty=1.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:45:00 | 1757.50 | 1846.54 | 1816.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 12:00:00 | 1752.70 | 1844.75 | 1815.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 1714.00 | 1839.54 | 1813.97 | SL hit (close<static) qty=1.00 sl=1715.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 1714.00 | 1839.54 | 1813.97 | SL hit (close<static) qty=1.00 sl=1715.00 alert=retest2 |

### Cycle 5 — SELL (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 13:15:00 | 1662.00 | 1791.92 | 1792.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 15:15:00 | 1657.10 | 1789.28 | 1790.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 09:15:00 | 1739.20 | 1736.04 | 1758.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 10:15:00 | 1732.20 | 1736.04 | 1758.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 1706.10 | 1700.14 | 1730.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-04 09:15:00 | 1607.10 | 1701.16 | 1729.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-11 12:15:00 | 1526.74 | 1656.88 | 1700.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-12 09:15:00 | 1446.39 | 1650.88 | 1696.73 | Target hit (10%) qty=0.50 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-14 11:45:00 | 1690.70 | 2025-10-15 09:15:00 | 1750.00 | STOP_HIT | 1.00 | -3.51% |
| BUY | retest2 | 2025-11-10 10:15:00 | 1763.10 | 2025-12-04 09:15:00 | 1939.41 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-24 09:45:00 | 1757.50 | 2025-12-26 09:15:00 | 1714.00 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2025-12-24 12:00:00 | 1752.70 | 2025-12-26 09:15:00 | 1714.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2026-02-04 09:15:00 | 1607.10 | 2026-02-11 12:15:00 | 1526.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-04 09:15:00 | 1607.10 | 2026-02-12 09:15:00 | 1446.39 | TARGET_HIT | 0.50 | 10.00% |
