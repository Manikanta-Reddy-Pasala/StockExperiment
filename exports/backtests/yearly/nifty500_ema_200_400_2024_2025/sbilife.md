# SBI Life Insurance Company Ltd. (SBILIFE)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1871.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 2 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 29 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 15
- **Target hits / Stop hits / Partials:** 4 / 25 / 7
- **Avg / median % per leg:** 1.68% / 0.31%
- **Sum % (uncompounded):** 60.47%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 8 | 50.0% | 4 | 12 | 0 | 1.74% | 27.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 8 | 50.0% | 4 | 12 | 0 | 1.74% | 27.8% |
| SELL (all) | 20 | 13 | 65.0% | 0 | 13 | 7 | 1.63% | 32.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 20 | 13 | 65.0% | 0 | 13 | 7 | 1.63% | 32.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 21 | 58.3% | 4 | 25 | 7 | 1.68% | 60.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 12:15:00 | 1504.35 | 1447.42 | 1447.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 1511.45 | 1455.57 | 1451.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 11:15:00 | 1836.05 | 1838.92 | 1768.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 11:45:00 | 1836.95 | 1838.92 | 1768.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1760.40 | 1830.94 | 1773.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 1760.40 | 1830.94 | 1773.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 1760.50 | 1830.24 | 1773.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 1765.30 | 1830.24 | 1773.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 1715.00 | 1782.02 | 1760.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:45:00 | 1711.20 | 1782.02 | 1760.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-28 13:15:00 | 1604.50 | 1743.47 | 1743.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 10:15:00 | 1593.40 | 1716.01 | 1729.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 11:15:00 | 1448.95 | 1446.36 | 1514.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 12:00:00 | 1448.95 | 1446.36 | 1514.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 14:15:00 | 1499.70 | 1455.18 | 1503.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 14:45:00 | 1499.40 | 1455.18 | 1503.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 1472.90 | 1455.72 | 1503.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 14:30:00 | 1463.25 | 1468.65 | 1503.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 11:30:00 | 1466.15 | 1459.88 | 1491.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 14:00:00 | 1465.80 | 1459.99 | 1491.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 14:30:00 | 1467.55 | 1460.13 | 1490.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1489.30 | 1461.93 | 1490.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:45:00 | 1488.35 | 1461.93 | 1490.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1466.65 | 1461.98 | 1490.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:45:00 | 1480.65 | 1461.98 | 1490.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1480.45 | 1462.16 | 1490.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1500.65 | 1462.16 | 1490.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 12:15:00 | 1390.09 | 1461.49 | 1489.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 12:15:00 | 1392.84 | 1461.49 | 1489.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 12:15:00 | 1392.51 | 1461.49 | 1489.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-01 12:15:00 | 1394.17 | 1461.49 | 1489.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-01 13:15:00 | 1463.00 | 1461.51 | 1489.72 | SL hit (close>ema200) qty=0.50 sl=1461.51 alert=retest2 |

### Cycle 3 — BUY (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 09:15:00 | 1538.50 | 1473.16 | 1472.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 13:15:00 | 1551.00 | 1475.89 | 1474.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1464.95 | 1496.14 | 1485.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 1464.95 | 1496.14 | 1485.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1464.95 | 1496.14 | 1485.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 14:45:00 | 1492.15 | 1492.65 | 1484.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 10:00:00 | 1488.50 | 1492.62 | 1484.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-09 12:00:00 | 1490.00 | 1492.52 | 1484.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-11 09:15:00 | 1509.00 | 1492.23 | 1484.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-22 09:15:00 | 1637.35 | 1517.82 | 1499.60 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 1786.70 | 1809.18 | 1809.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 1778.00 | 1808.87 | 1809.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 1808.20 | 1805.73 | 1807.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 13:15:00 | 1808.20 | 1805.73 | 1807.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 1808.20 | 1805.73 | 1807.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 1808.20 | 1805.73 | 1807.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 1810.70 | 1805.78 | 1807.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 1810.70 | 1805.78 | 1807.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 1813.00 | 1805.86 | 1807.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 1822.20 | 1805.86 | 1807.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 1828.00 | 1806.08 | 1807.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 1828.50 | 1806.08 | 1807.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 1808.10 | 1806.97 | 1808.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 1802.20 | 1806.97 | 1808.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 1821.20 | 1807.11 | 1808.07 | SL hit (close>static) qty=1.00 sl=1812.10 alert=retest2 |

### Cycle 5 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 1850.00 | 1809.08 | 1809.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 1898.00 | 1821.18 | 1815.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 14:15:00 | 1963.70 | 1964.61 | 1916.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 1963.70 | 1964.61 | 1916.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2023.80 | 2046.62 | 2010.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 09:45:00 | 2030.00 | 2043.21 | 2009.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 11:30:00 | 2032.00 | 2042.90 | 2010.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 12:00:00 | 2029.80 | 2042.90 | 2010.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 10:15:00 | 1999.90 | 2042.69 | 2012.05 | SL hit (close<static) qty=1.00 sl=2009.30 alert=retest2 |

### Cycle 6 — SELL (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-10 15:15:00 | 1963.00 | 2013.11 | 2013.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 09:15:00 | 1940.10 | 2012.38 | 2012.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 10:15:00 | 1914.90 | 1898.14 | 1942.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-08 11:00:00 | 1914.90 | 1898.14 | 1942.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 1942.30 | 1899.91 | 1940.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 10:00:00 | 1942.30 | 1899.91 | 1940.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 1946.00 | 1900.37 | 1940.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 1946.00 | 1900.37 | 1940.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 11:15:00 | 1942.90 | 1900.80 | 1940.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 1933.00 | 1901.10 | 1940.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 1960.00 | 1903.55 | 1939.71 | SL hit (close>static) qty=1.00 sl=1950.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-29 09:15:00 | 1425.85 | 2024-06-04 09:15:00 | 1354.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 09:15:00 | 1425.85 | 2024-06-06 10:15:00 | 1430.00 | STOP_HIT | 0.50 | -0.29% |
| SELL | retest2 | 2024-06-12 12:00:00 | 1448.85 | 2024-06-14 10:15:00 | 1455.30 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2024-06-13 10:45:00 | 1447.05 | 2024-06-14 10:15:00 | 1455.30 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-06-13 11:45:00 | 1448.80 | 2024-06-14 10:15:00 | 1455.30 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-01-21 14:30:00 | 1463.25 | 2025-02-01 12:15:00 | 1390.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-30 11:30:00 | 1466.15 | 2025-02-01 12:15:00 | 1392.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-30 14:00:00 | 1465.80 | 2025-02-01 12:15:00 | 1392.51 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-30 14:30:00 | 1467.55 | 2025-02-01 12:15:00 | 1394.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 14:30:00 | 1463.25 | 2025-02-01 13:15:00 | 1463.00 | STOP_HIT | 0.50 | 0.02% |
| SELL | retest2 | 2025-01-30 11:30:00 | 1466.15 | 2025-02-01 13:15:00 | 1463.00 | STOP_HIT | 0.50 | 0.21% |
| SELL | retest2 | 2025-01-30 14:00:00 | 1465.80 | 2025-02-01 13:15:00 | 1463.00 | STOP_HIT | 0.50 | 0.19% |
| SELL | retest2 | 2025-01-30 14:30:00 | 1467.55 | 2025-02-01 13:15:00 | 1463.00 | STOP_HIT | 0.50 | 0.31% |
| SELL | retest2 | 2025-02-28 09:15:00 | 1454.35 | 2025-03-05 09:15:00 | 1381.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 1454.35 | 2025-03-17 15:15:00 | 1439.50 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2025-03-18 14:00:00 | 1452.75 | 2025-03-19 13:15:00 | 1485.50 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-04-08 14:45:00 | 1492.15 | 2025-04-22 09:15:00 | 1637.35 | TARGET_HIT | 1.00 | 9.73% |
| BUY | retest2 | 2025-04-09 10:00:00 | 1488.50 | 2025-04-22 10:15:00 | 1639.00 | TARGET_HIT | 1.00 | 10.11% |
| BUY | retest2 | 2025-04-09 12:00:00 | 1490.00 | 2025-04-25 09:15:00 | 1641.37 | TARGET_HIT | 1.00 | 10.16% |
| BUY | retest2 | 2025-04-11 09:15:00 | 1509.00 | 2025-04-25 09:15:00 | 1659.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-18 13:00:00 | 1778.00 | 2025-10-07 15:15:00 | 1786.70 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-10-01 11:15:00 | 1775.60 | 2025-10-07 15:15:00 | 1786.70 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2025-10-03 11:30:00 | 1775.50 | 2025-10-07 15:15:00 | 1786.70 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2025-10-03 13:30:00 | 1773.90 | 2025-10-07 15:15:00 | 1786.70 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1802.20 | 2025-10-13 09:15:00 | 1821.20 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-01-27 09:45:00 | 2030.00 | 2026-01-29 10:15:00 | 1999.90 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-01-27 11:30:00 | 2032.00 | 2026-01-29 10:15:00 | 1999.90 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2026-01-27 12:00:00 | 2029.80 | 2026-01-29 10:15:00 | 1999.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-02-03 09:15:00 | 2047.70 | 2026-02-03 14:15:00 | 2000.50 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2026-02-10 12:45:00 | 2012.00 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-02-12 10:30:00 | 2010.00 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2026-02-12 12:45:00 | 2011.20 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-03-02 10:45:00 | 2011.80 | 2026-03-04 09:15:00 | 1972.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2026-04-10 12:30:00 | 1933.00 | 2026-04-15 09:15:00 | 1960.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-04-21 09:15:00 | 1920.00 | 2026-04-23 11:15:00 | 1824.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-21 09:15:00 | 1920.00 | 2026-05-07 09:15:00 | 1874.70 | STOP_HIT | 0.50 | 2.36% |
