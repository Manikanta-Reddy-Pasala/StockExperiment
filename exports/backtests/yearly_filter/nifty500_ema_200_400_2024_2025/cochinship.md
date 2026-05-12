# Cochin Shipyard Ltd. (COCHINSHIP)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1769.40
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 9 |
| PARTIAL | 6 |
| TARGET_HIT | 7 |
| STOP_HIT | 4 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 17 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 16 / 1
- **Target hits / Stop hits / Partials:** 7 / 4 / 6
- **Avg / median % per leg:** 6.12% / 5.00%
- **Sum % (uncompounded):** 104.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 5 | 4 | 80.0% | 4 | 1 | 0 | 7.71% | 38.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 5 | 4 | 80.0% | 4 | 1 | 0 | 7.71% | 38.6% |
| SELL (all) | 12 | 12 | 100.0% | 3 | 3 | 6 | 5.46% | 65.5% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.78% | 11.1% |
| SELL @ 3rd Alert (retest2) | 8 | 8 | 100.0% | 3 | 1 | 4 | 6.80% | 54.4% |
| retest1 (combined) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.78% | 11.1% |
| retest2 (combined) | 13 | 12 | 92.3% | 7 | 2 | 4 | 7.15% | 93.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 1926.25 | 2149.37 | 2149.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 13:15:00 | 1916.50 | 2147.05 | 2148.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 09:15:00 | 1504.05 | 1454.77 | 1593.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:30:00 | 1504.05 | 1454.77 | 1593.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 1656.15 | 1474.52 | 1590.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:30:00 | 1656.15 | 1474.52 | 1590.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 1660.00 | 1558.70 | 1609.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 15:00:00 | 1653.10 | 1562.54 | 1610.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-18 12:15:00 | 1570.44 | 1573.01 | 1610.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-18 15:15:00 | 1580.00 | 1573.05 | 1610.06 | SL hit (close>ema200) qty=0.50 sl=1573.05 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-22 14:15:00 | 1489.70 | 1405.09 | 1404.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 1498.60 | 1411.64 | 1408.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 10:15:00 | 1413.60 | 1415.93 | 1410.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 10:15:00 | 1413.60 | 1415.93 | 1410.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 1413.60 | 1415.93 | 1410.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 1413.60 | 1415.93 | 1410.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 1423.00 | 1416.00 | 1410.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:45:00 | 1408.60 | 1416.00 | 1410.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 15:15:00 | 1415.00 | 1416.17 | 1410.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:15:00 | 1425.00 | 1416.17 | 1410.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 1500.40 | 1417.01 | 1411.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 09:15:00 | 1541.30 | 1422.16 | 1413.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-30 09:15:00 | 1695.43 | 1437.10 | 1421.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 12:15:00 | 1734.00 | 1883.43 | 1884.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 15:15:00 | 1731.00 | 1879.00 | 1881.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-02 12:15:00 | 1748.50 | 1735.68 | 1788.47 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-03 13:15:00 | 1729.20 | 1736.41 | 1787.03 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-04 09:15:00 | 1718.00 | 1736.45 | 1786.29 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 09:15:00 | 1642.74 | 1722.20 | 1773.63 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-09 11:15:00 | 1632.10 | 1720.47 | 1772.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-12 11:15:00 | 1713.90 | 1708.58 | 1760.78 | SL hit (close>ema200) qty=0.50 sl=1708.58 alert=retest1 |

### Cycle 4 — BUY (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 10:15:00 | 1891.80 | 1795.48 | 1795.26 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 1768.10 | 1800.02 | 1800.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 1719.00 | 1798.93 | 1799.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 09:15:00 | 1796.70 | 1784.71 | 1791.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 1796.70 | 1784.71 | 1791.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 1796.70 | 1784.71 | 1791.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:45:00 | 1796.80 | 1784.71 | 1791.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 1791.80 | 1784.78 | 1791.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 11:15:00 | 1788.50 | 1784.78 | 1791.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 1786.00 | 1784.96 | 1791.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 09:15:00 | 1741.60 | 1785.11 | 1791.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:15:00 | 1699.07 | 1784.54 | 1791.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:15:00 | 1696.70 | 1784.54 | 1791.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 09:15:00 | 1654.52 | 1784.54 | 1791.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-12-08 14:15:00 | 1609.65 | 1701.61 | 1736.93 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-04-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 15:15:00 | 1678.80 | 1481.89 | 1481.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 1686.30 | 1483.92 | 1482.43 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-12-12 15:00:00 | 1653.10 | 2024-12-18 12:15:00 | 1570.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-12 15:00:00 | 1653.10 | 2024-12-18 15:15:00 | 1580.00 | STOP_HIT | 0.50 | 4.42% |
| BUY | retest2 | 2025-04-29 09:15:00 | 1541.30 | 2025-04-30 09:15:00 | 1695.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-06 10:45:00 | 1519.00 | 2025-05-14 09:15:00 | 1670.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 09:15:00 | 1517.70 | 2025-05-14 09:15:00 | 1669.47 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-05-12 09:45:00 | 1519.00 | 2025-05-14 09:15:00 | 1670.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-16 11:45:00 | 1937.80 | 2025-07-17 11:15:00 | 1909.80 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest1 | 2025-09-03 13:15:00 | 1729.20 | 2025-09-09 09:15:00 | 1642.74 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-04 09:15:00 | 1718.00 | 2025-09-09 11:15:00 | 1632.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-03 13:15:00 | 1729.20 | 2025-09-12 11:15:00 | 1713.90 | STOP_HIT | 0.50 | 0.88% |
| SELL | retest1 | 2025-09-04 09:15:00 | 1718.00 | 2025-09-12 11:15:00 | 1713.90 | STOP_HIT | 0.50 | 0.24% |
| SELL | retest2 | 2025-11-12 11:15:00 | 1788.50 | 2025-11-13 09:15:00 | 1699.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 14:15:00 | 1786.00 | 2025-11-13 09:15:00 | 1696.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-13 09:15:00 | 1741.60 | 2025-11-13 09:15:00 | 1654.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-12 11:15:00 | 1788.50 | 2025-12-08 14:15:00 | 1609.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-12 14:15:00 | 1786.00 | 2025-12-08 14:15:00 | 1607.40 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-13 09:15:00 | 1741.60 | 2025-12-09 09:15:00 | 1567.44 | TARGET_HIT | 0.50 | 10.00% |
