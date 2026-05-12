# Cholamandalam Financial Holdings Ltd. (CHOLAHLDNG)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1785.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 7 |
| ALERT2 | 7 |
| ALERT2_SKIP | 6 |
| ALERT3 | 35 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 11 |
| TARGET_HIT | 10 |
| STOP_HIT | 25 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 20
- **Target hits / Stop hits / Partials:** 10 / 25 / 11
- **Avg / median % per leg:** 2.65% / 2.18%
- **Sum % (uncompounded):** 121.91%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 4 | 30.8% | 4 | 9 | 0 | 1.55% | 20.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 4 | 30.8% | 4 | 9 | 0 | 1.55% | 20.2% |
| SELL (all) | 33 | 22 | 66.7% | 6 | 16 | 11 | 3.08% | 101.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 22 | 66.7% | 6 | 16 | 11 | 3.08% | 101.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 46 | 26 | 56.5% | 10 | 25 | 11 | 2.65% | 121.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-18 09:15:00 | 1540.10 | 1761.55 | 1762.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-18 10:15:00 | 1511.00 | 1759.05 | 1761.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 09:15:00 | 1541.05 | 1501.94 | 1576.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-03 09:45:00 | 1549.95 | 1501.94 | 1576.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 11:15:00 | 1566.60 | 1503.09 | 1576.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 11:30:00 | 1554.75 | 1503.09 | 1576.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1585.15 | 1511.85 | 1572.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 1585.15 | 1511.85 | 1572.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1589.20 | 1512.62 | 1572.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 13:00:00 | 1569.75 | 1513.92 | 1572.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:15:00 | 1539.10 | 1515.88 | 1572.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 14:15:00 | 1491.26 | 1515.68 | 1570.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 14:15:00 | 1462.14 | 1515.68 | 1570.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 10:15:00 | 1522.30 | 1512.43 | 1566.30 | SL hit (close>ema200) qty=0.50 sl=1512.43 alert=retest2 |

### Cycle 2 — BUY (started 2025-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 11:15:00 | 1600.20 | 1526.49 | 1526.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-10 10:15:00 | 1636.65 | 1536.32 | 1531.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 09:15:00 | 1553.55 | 1651.12 | 1604.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 09:15:00 | 1553.55 | 1651.12 | 1604.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 09:15:00 | 1553.55 | 1651.12 | 1604.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 11:15:00 | 1583.60 | 1650.25 | 1604.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 12:15:00 | 1582.80 | 1649.46 | 1604.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:15:00 | 1632.70 | 1646.46 | 1603.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-08 13:15:00 | 1741.96 | 1648.36 | 1605.81 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 15:15:00 | 1847.90 | 1943.69 | 1943.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 09:15:00 | 1826.20 | 1942.52 | 1943.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 1939.80 | 1928.72 | 1935.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 1939.80 | 1928.72 | 1935.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 1939.80 | 1928.72 | 1935.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:00:00 | 1939.80 | 1928.72 | 1935.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 1954.50 | 1928.97 | 1936.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 15:00:00 | 1927.40 | 1929.52 | 1936.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 1922.60 | 1929.64 | 1936.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 09:15:00 | 1831.03 | 1925.09 | 1933.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 09:15:00 | 1826.47 | 1925.09 | 1933.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-29 09:15:00 | 1734.66 | 1892.08 | 1914.23 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-10-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 15:15:00 | 1942.00 | 1885.99 | 1885.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 09:15:00 | 1962.30 | 1891.80 | 1888.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 13:15:00 | 1900.00 | 1913.26 | 1900.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 13:15:00 | 1900.00 | 1913.26 | 1900.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 1900.00 | 1913.26 | 1900.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 14:00:00 | 1900.00 | 1913.26 | 1900.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 1890.80 | 1913.03 | 1900.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 1890.80 | 1913.03 | 1900.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 1882.00 | 1912.72 | 1900.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 1870.00 | 1912.72 | 1900.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 1901.20 | 1912.57 | 1900.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:45:00 | 1900.00 | 1912.57 | 1900.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 1900.80 | 1912.45 | 1900.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 13:15:00 | 1908.90 | 1912.34 | 1900.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 1884.90 | 1924.51 | 1909.82 | SL hit (close<static) qty=1.00 sl=1899.90 alert=retest2 |

### Cycle 5 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 1803.10 | 1898.20 | 1898.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 13:15:00 | 1793.00 | 1896.27 | 1897.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 11:15:00 | 1907.40 | 1889.41 | 1893.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 1907.40 | 1889.41 | 1893.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 1907.40 | 1889.41 | 1893.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 11:45:00 | 1897.60 | 1889.41 | 1893.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 1921.80 | 1889.73 | 1893.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:00:00 | 1921.80 | 1889.73 | 1893.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 1901.40 | 1889.99 | 1893.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 15:15:00 | 1885.00 | 1889.99 | 1893.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:30:00 | 1887.70 | 1887.60 | 1892.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:00:00 | 1885.20 | 1887.73 | 1892.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 1910.80 | 1887.93 | 1892.39 | SL hit (close>static) qty=1.00 sl=1904.60 alert=retest2 |

### Cycle 6 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 1932.00 | 1896.29 | 1896.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 11:15:00 | 1946.80 | 1896.80 | 1896.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1894.30 | 1898.48 | 1897.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 09:15:00 | 1894.30 | 1898.48 | 1897.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 1894.30 | 1898.48 | 1897.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 10:00:00 | 1894.30 | 1898.48 | 1897.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 1911.10 | 1898.61 | 1897.32 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 1830.10 | 1895.88 | 1896.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 1820.50 | 1895.13 | 1895.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 11:15:00 | 1901.50 | 1876.74 | 1885.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 11:15:00 | 1901.50 | 1876.74 | 1885.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1901.50 | 1876.74 | 1885.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:00:00 | 1901.50 | 1876.74 | 1885.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1894.40 | 1876.92 | 1885.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 1894.40 | 1876.92 | 1885.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 11:15:00 | 1884.00 | 1876.73 | 1885.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:45:00 | 1884.00 | 1876.73 | 1885.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 12:15:00 | 1881.50 | 1876.78 | 1885.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 12:30:00 | 1880.70 | 1876.78 | 1885.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 1898.90 | 1877.00 | 1885.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 1898.90 | 1877.00 | 1885.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 1900.10 | 1877.23 | 1885.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:15:00 | 1891.20 | 1877.23 | 1885.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 1877.80 | 1877.48 | 1885.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 11:30:00 | 1876.00 | 1877.45 | 1885.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 14:15:00 | 1877.00 | 1867.46 | 1878.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1852.60 | 1867.68 | 1878.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 09:15:00 | 1900.80 | 1868.13 | 1878.74 | SL hit (close>static) qty=1.00 sl=1891.40 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 11:30:00 | 1115.80 | 2024-05-31 14:15:00 | 1084.85 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2024-05-16 12:30:00 | 1113.70 | 2024-05-31 14:15:00 | 1084.85 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2024-05-16 13:15:00 | 1114.90 | 2024-05-31 14:15:00 | 1084.85 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-05-17 13:45:00 | 1114.25 | 2024-05-31 14:15:00 | 1084.85 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-05-31 10:30:00 | 1094.60 | 2024-05-31 15:15:00 | 1078.05 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest2 | 2024-05-31 12:15:00 | 1095.00 | 2024-05-31 15:15:00 | 1078.05 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2024-06-03 09:15:00 | 1106.05 | 2024-06-04 11:15:00 | 1076.20 | STOP_HIT | 1.00 | -2.70% |
| BUY | retest2 | 2024-06-03 10:45:00 | 1099.55 | 2024-06-04 11:15:00 | 1076.20 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2024-06-06 09:15:00 | 1173.95 | 2024-06-14 09:15:00 | 1291.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-09 13:00:00 | 1569.75 | 2025-01-10 14:15:00 | 1491.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-10 09:15:00 | 1539.10 | 2025-01-10 14:15:00 | 1462.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 13:00:00 | 1569.75 | 2025-01-14 10:15:00 | 1522.30 | STOP_HIT | 0.50 | 3.02% |
| SELL | retest2 | 2025-01-10 09:15:00 | 1539.10 | 2025-01-14 10:15:00 | 1522.30 | STOP_HIT | 0.50 | 1.09% |
| SELL | retest2 | 2025-02-25 10:15:00 | 1564.90 | 2025-02-25 12:15:00 | 1598.65 | STOP_HIT | 1.00 | -2.16% |
| SELL | retest2 | 2025-02-25 14:00:00 | 1564.00 | 2025-02-27 09:15:00 | 1646.10 | STOP_HIT | 1.00 | -5.25% |
| BUY | retest2 | 2025-04-07 11:15:00 | 1583.60 | 2025-04-08 13:15:00 | 1741.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-07 12:15:00 | 1582.80 | 2025-04-08 13:15:00 | 1741.08 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-08 09:15:00 | 1632.70 | 2025-04-11 14:15:00 | 1795.97 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-08-18 15:00:00 | 1927.40 | 2025-08-20 09:15:00 | 1831.03 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-19 09:15:00 | 1922.60 | 2025-08-20 09:15:00 | 1826.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-18 15:00:00 | 1927.40 | 2025-08-29 09:15:00 | 1734.66 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-08-19 09:15:00 | 1922.60 | 2025-08-29 09:15:00 | 1730.34 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-09-16 10:45:00 | 1936.50 | 2025-09-17 09:15:00 | 1987.50 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-09-16 12:30:00 | 1935.80 | 2025-09-17 09:15:00 | 1987.50 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1917.10 | 2025-10-01 10:15:00 | 1826.28 | PARTIAL | 0.50 | 4.74% |
| SELL | retest2 | 2025-09-30 13:45:00 | 1922.40 | 2025-10-01 10:15:00 | 1825.71 | PARTIAL | 0.50 | 5.03% |
| SELL | retest2 | 2025-09-30 09:15:00 | 1917.10 | 2025-10-01 13:15:00 | 1883.40 | STOP_HIT | 0.50 | 1.76% |
| SELL | retest2 | 2025-09-30 13:45:00 | 1922.40 | 2025-10-01 13:15:00 | 1883.40 | STOP_HIT | 0.50 | 2.03% |
| SELL | retest2 | 2025-09-30 14:15:00 | 1921.80 | 2025-10-03 14:15:00 | 1821.24 | PARTIAL | 0.50 | 5.23% |
| SELL | retest2 | 2025-09-30 14:15:00 | 1921.80 | 2025-10-06 12:15:00 | 1880.00 | STOP_HIT | 0.50 | 2.18% |
| BUY | retest2 | 2025-11-10 13:15:00 | 1908.90 | 2025-11-19 09:15:00 | 1884.90 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-12-01 15:15:00 | 1885.00 | 2025-12-05 10:15:00 | 1910.80 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-12-03 14:30:00 | 1887.70 | 2025-12-05 10:15:00 | 1910.80 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-12-04 12:00:00 | 1885.20 | 2025-12-05 10:15:00 | 1910.80 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-12-08 13:45:00 | 1881.50 | 2025-12-09 11:15:00 | 1909.30 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-12-26 11:30:00 | 1876.00 | 2026-01-05 09:15:00 | 1900.80 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-01-01 14:15:00 | 1877.00 | 2026-01-05 09:15:00 | 1900.80 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2026-01-02 09:15:00 | 1852.60 | 2026-01-05 09:15:00 | 1900.80 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2026-01-07 11:00:00 | 1876.50 | 2026-01-16 09:15:00 | 1782.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 13:15:00 | 1878.40 | 2026-01-16 09:15:00 | 1784.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 13:45:00 | 1879.80 | 2026-01-16 09:15:00 | 1785.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1878.10 | 2026-01-16 09:15:00 | 1784.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:00:00 | 1876.50 | 2026-01-20 12:15:00 | 1688.85 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-07 13:15:00 | 1878.40 | 2026-01-20 12:15:00 | 1690.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-07 13:45:00 | 1879.80 | 2026-01-20 12:15:00 | 1691.82 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 1878.10 | 2026-01-20 12:15:00 | 1690.29 | TARGET_HIT | 0.50 | 10.00% |
