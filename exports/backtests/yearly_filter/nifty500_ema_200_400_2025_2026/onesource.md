# Onesource Specialty Pharma Ltd. (ONESOURCE)

## Backtest Summary

- **Window:** 2025-01-24 09:15:00 → 2026-05-11 15:15:00 (2221 bars)
- **Last close:** 1802.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 2 |
| ALERT1 | 2 |
| ALERT2 | 1 |
| ALERT2_SKIP | 1 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 15 |
| TARGET_HIT | 9 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 31 / 11
- **Target hits / Stop hits / Partials:** 9 / 18 / 15
- **Avg / median % per leg:** 3.04% / 4.54%
- **Sum % (uncompounded):** 127.86%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 42 | 31 | 73.8% | 9 | 18 | 15 | 3.04% | 127.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 42 | 31 | 73.8% | 9 | 18 | 15 | 3.04% | 127.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 31 | 73.8% | 9 | 18 | 15 | 3.04% | 127.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 1829.10 | 1888.29 | 1888.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 1812.30 | 1865.81 | 1875.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 1839.50 | 1830.27 | 1853.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 14:15:00 | 1839.50 | 1830.27 | 1853.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1839.50 | 1830.27 | 1853.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 1839.50 | 1830.27 | 1853.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 1889.90 | 1825.22 | 1848.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 14:15:00 | 1837.60 | 1828.92 | 1848.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 13:15:00 | 1837.90 | 1834.42 | 1849.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 09:30:00 | 1835.50 | 1834.89 | 1849.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 12:15:00 | 1835.00 | 1835.06 | 1849.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1840.00 | 1834.98 | 1849.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 1840.90 | 1834.98 | 1849.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 1840.80 | 1835.06 | 1849.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 1840.80 | 1835.06 | 1849.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 1855.00 | 1835.26 | 1849.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 09:15:00 | 1836.20 | 1836.14 | 1849.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:00:00 | 1828.00 | 1836.05 | 1848.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 15:00:00 | 1826.60 | 1835.44 | 1848.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 14:15:00 | 1857.90 | 1834.53 | 1847.44 | SL hit (close>static) qty=1.00 sl=1855.00 alert=retest2 |

### Cycle 2 — BUY (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 10:15:00 | 1571.90 | 1439.24 | 1439.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 11:15:00 | 1592.00 | 1440.76 | 1439.97 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-10-08 14:15:00 | 1837.60 | 2025-10-20 14:15:00 | 1857.90 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-10-13 13:15:00 | 1837.90 | 2025-10-20 14:15:00 | 1857.90 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-10-14 09:30:00 | 1835.50 | 2025-10-20 14:15:00 | 1857.90 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-10-14 12:15:00 | 1835.00 | 2025-10-29 15:15:00 | 1860.70 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-10-17 09:15:00 | 1836.20 | 2025-11-07 09:15:00 | 1745.72 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2025-10-17 10:00:00 | 1828.00 | 2025-11-07 09:15:00 | 1746.01 | PARTIAL | 0.50 | 4.49% |
| SELL | retest2 | 2025-10-17 15:00:00 | 1826.60 | 2025-11-07 09:15:00 | 1743.72 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2025-10-23 15:00:00 | 1836.80 | 2025-11-07 09:15:00 | 1743.25 | PARTIAL | 0.50 | 5.09% |
| SELL | retest2 | 2025-10-17 09:15:00 | 1836.20 | 2025-11-17 09:15:00 | 1807.30 | STOP_HIT | 0.50 | 1.57% |
| SELL | retest2 | 2025-10-17 10:00:00 | 1828.00 | 2025-11-17 09:15:00 | 1807.30 | STOP_HIT | 0.50 | 1.13% |
| SELL | retest2 | 2025-10-17 15:00:00 | 1826.60 | 2025-11-17 09:15:00 | 1807.30 | STOP_HIT | 0.50 | 1.06% |
| SELL | retest2 | 2025-10-23 15:00:00 | 1836.80 | 2025-11-17 09:15:00 | 1807.30 | STOP_HIT | 0.50 | 1.61% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1738.00 | 2025-11-20 13:15:00 | 1685.87 | PARTIAL | 0.50 | 3.00% |
| SELL | retest2 | 2025-11-07 15:00:00 | 1774.60 | 2025-11-20 13:15:00 | 1686.25 | PARTIAL | 0.50 | 4.98% |
| SELL | retest2 | 2025-11-10 13:15:00 | 1775.00 | 2025-11-20 13:15:00 | 1685.11 | PARTIAL | 0.50 | 5.06% |
| SELL | retest2 | 2025-11-10 14:30:00 | 1773.80 | 2025-11-21 09:15:00 | 1662.50 | PARTIAL | 0.50 | 6.27% |
| SELL | retest2 | 2025-11-17 15:15:00 | 1750.00 | 2025-11-21 11:15:00 | 1651.10 | PARTIAL | 0.50 | 5.65% |
| SELL | retest2 | 2025-11-07 09:15:00 | 1738.00 | 2025-11-21 12:15:00 | 1597.14 | TARGET_HIT | 0.50 | 8.10% |
| SELL | retest2 | 2025-11-07 15:00:00 | 1774.60 | 2025-11-21 12:15:00 | 1597.50 | TARGET_HIT | 0.50 | 9.98% |
| SELL | retest2 | 2025-11-10 13:15:00 | 1775.00 | 2025-11-21 12:15:00 | 1596.42 | TARGET_HIT | 0.50 | 10.06% |
| SELL | retest2 | 2025-11-10 14:30:00 | 1773.80 | 2025-11-28 10:15:00 | 1575.00 | TARGET_HIT | 0.50 | 11.21% |
| SELL | retest2 | 2025-11-17 15:15:00 | 1750.00 | 2025-11-28 13:15:00 | 1564.20 | TARGET_HIT | 0.50 | 10.62% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1788.90 | 2026-01-12 10:15:00 | 1702.21 | PARTIAL | 0.50 | 4.85% |
| SELL | retest2 | 2026-01-01 10:15:00 | 1780.30 | 2026-01-12 11:15:00 | 1699.45 | PARTIAL | 0.50 | 4.54% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1788.90 | 2026-01-12 14:15:00 | 1737.90 | STOP_HIT | 0.50 | 2.85% |
| SELL | retest2 | 2026-01-01 10:15:00 | 1780.30 | 2026-01-12 14:15:00 | 1737.90 | STOP_HIT | 0.50 | 2.38% |
| SELL | retest2 | 2026-01-01 15:00:00 | 1791.80 | 2026-01-13 14:15:00 | 1779.40 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2026-01-13 09:30:00 | 1707.60 | 2026-01-13 14:15:00 | 1779.40 | STOP_HIT | 1.00 | -4.20% |
| SELL | retest2 | 2026-01-13 10:45:00 | 1713.70 | 2026-01-20 09:15:00 | 1691.28 | PARTIAL | 0.50 | 1.31% |
| SELL | retest2 | 2026-01-19 11:00:00 | 1706.30 | 2026-01-21 09:15:00 | 1620.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-19 14:45:00 | 1707.80 | 2026-01-21 09:15:00 | 1622.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 10:45:00 | 1713.70 | 2026-01-21 10:15:00 | 1602.27 | TARGET_HIT | 0.50 | 6.50% |
| SELL | retest2 | 2026-01-19 11:00:00 | 1706.30 | 2026-01-22 14:15:00 | 1537.02 | TARGET_HIT | 0.50 | 9.92% |
| SELL | retest2 | 2026-01-19 14:45:00 | 1707.80 | 2026-01-22 15:15:00 | 1535.67 | TARGET_HIT | 0.50 | 10.08% |
| SELL | retest2 | 2026-03-13 12:30:00 | 1408.10 | 2026-03-13 14:15:00 | 1494.40 | STOP_HIT | 1.00 | -6.13% |
| SELL | retest2 | 2026-03-16 10:30:00 | 1424.60 | 2026-03-16 12:15:00 | 1449.10 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-03-16 11:00:00 | 1423.00 | 2026-03-16 12:15:00 | 1449.10 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1403.40 | 2026-03-24 09:15:00 | 1333.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1403.40 | 2026-03-25 09:15:00 | 1263.06 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-08 11:45:00 | 1345.00 | 2026-04-10 14:15:00 | 1442.00 | STOP_HIT | 1.00 | -7.21% |
| SELL | retest2 | 2026-04-08 12:15:00 | 1348.00 | 2026-04-10 14:15:00 | 1442.00 | STOP_HIT | 1.00 | -6.97% |
| SELL | retest2 | 2026-04-10 12:45:00 | 1351.10 | 2026-04-10 14:15:00 | 1442.00 | STOP_HIT | 1.00 | -6.73% |
