# Glenmark Pharmaceuticals Ltd. (GLENMARK)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 2361.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 4 |
| ALERT2 | 4 |
| ALERT2_SKIP | 2 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 44 |
| PARTIAL | 2 |
| TARGET_HIT | 9 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 35
- **Target hits / Stop hits / Partials:** 9 / 35 / 2
- **Avg / median % per leg:** 0.69% / -1.33%
- **Sum % (uncompounded):** 31.52%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 25 | 8 | 32.0% | 8 | 17 | 0 | 1.91% | 47.7% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 25 | 8 | 32.0% | 8 | 17 | 0 | 1.91% | 47.7% |
| SELL (all) | 21 | 3 | 14.3% | 1 | 18 | 2 | -0.77% | -16.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 3 | 14.3% | 1 | 18 | 2 | -0.77% | -16.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 46 | 11 | 23.9% | 9 | 35 | 2 | 0.69% | 31.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-11-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 14:15:00 | 1476.55 | 1626.97 | 1626.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1465.95 | 1565.37 | 1588.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 1561.40 | 1561.25 | 1585.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 13:00:00 | 1561.40 | 1561.25 | 1585.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 1574.40 | 1552.18 | 1574.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:45:00 | 1567.80 | 1552.18 | 1574.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 1588.00 | 1552.53 | 1574.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:00:00 | 1588.00 | 1552.53 | 1574.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 11:15:00 | 1590.00 | 1552.91 | 1574.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 11:30:00 | 1589.75 | 1552.91 | 1574.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1546.00 | 1491.18 | 1523.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:00:00 | 1546.00 | 1491.18 | 1523.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 1530.40 | 1491.57 | 1523.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:45:00 | 1542.25 | 1491.57 | 1523.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 1531.10 | 1494.08 | 1524.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 11:00:00 | 1531.10 | 1494.08 | 1524.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 1528.65 | 1494.42 | 1524.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 14:00:00 | 1515.85 | 1494.95 | 1524.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 12:15:00 | 1440.06 | 1493.38 | 1522.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-14 09:15:00 | 1364.26 | 1480.71 | 1513.34 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 2 — BUY (started 2025-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 14:15:00 | 1520.00 | 1431.16 | 1431.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 1558.10 | 1433.28 | 1432.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 09:15:00 | 2022.20 | 2024.63 | 1900.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 1918.90 | 2010.03 | 1905.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 1918.90 | 2010.03 | 1905.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:30:00 | 1902.40 | 2010.03 | 1905.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 15:15:00 | 1919.70 | 1988.19 | 1912.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 1931.20 | 1988.19 | 1912.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 11:15:00 | 1905.60 | 1986.12 | 1912.82 | SL hit (close<static) qty=1.00 sl=1910.70 alert=retest2 |

### Cycle 3 — SELL (started 2025-10-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 15:15:00 | 1855.00 | 1955.98 | 1956.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 14:15:00 | 1847.00 | 1948.52 | 1952.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 1909.20 | 1884.52 | 1911.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 1909.20 | 1884.52 | 1911.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1909.20 | 1884.52 | 1911.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 1909.20 | 1884.52 | 1911.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 1913.00 | 1884.80 | 1911.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:30:00 | 1913.50 | 1884.80 | 1911.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 1906.00 | 1885.01 | 1911.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 12:30:00 | 1910.40 | 1885.01 | 1911.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 1910.00 | 1885.41 | 1911.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 11:00:00 | 1910.00 | 1885.41 | 1911.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 11:15:00 | 1901.90 | 1885.58 | 1910.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:15:00 | 1923.20 | 1885.58 | 1910.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 1925.00 | 1885.97 | 1911.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 12:45:00 | 1929.50 | 1885.97 | 1911.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 1910.00 | 1886.21 | 1911.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 14:45:00 | 1903.40 | 1886.28 | 1910.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:30:00 | 1900.60 | 1886.77 | 1910.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 1947.70 | 1879.23 | 1900.43 | SL hit (close>static) qty=1.00 sl=1925.60 alert=retest2 |

### Cycle 4 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 1949.40 | 1915.78 | 1915.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 10:15:00 | 1961.00 | 1917.33 | 1916.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 14:15:00 | 2005.00 | 2006.95 | 1973.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-09 15:00:00 | 2005.00 | 2006.95 | 1973.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 1981.70 | 2006.66 | 1973.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 14:45:00 | 2019.00 | 2006.40 | 1974.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 09:15:00 | 1961.90 | 2006.07 | 1974.23 | SL hit (close<static) qty=1.00 sl=1967.60 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-02-10 14:00:00 | 1515.85 | 2025-02-11 12:15:00 | 1440.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 14:00:00 | 1515.85 | 2025-02-14 09:15:00 | 1364.26 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-01 10:45:00 | 1515.00 | 2025-04-03 10:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-04-02 11:15:00 | 1519.55 | 2025-04-03 10:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-04-02 13:15:00 | 1520.70 | 2025-04-03 10:15:00 | 1544.00 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1415.20 | 2025-04-09 14:15:00 | 1344.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1415.20 | 2025-04-24 09:15:00 | 1434.00 | STOP_HIT | 0.50 | -1.33% |
| SELL | retest2 | 2025-04-24 10:30:00 | 1424.00 | 2025-05-14 11:15:00 | 1437.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-04-24 13:00:00 | 1420.20 | 2025-05-14 11:15:00 | 1437.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-05-08 09:45:00 | 1422.20 | 2025-05-30 10:15:00 | 1440.20 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-05-08 13:30:00 | 1426.70 | 2025-05-30 10:15:00 | 1440.20 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-05-08 14:45:00 | 1416.90 | 2025-05-30 10:15:00 | 1440.20 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-05-22 09:15:00 | 1426.70 | 2025-05-30 10:15:00 | 1440.20 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-05-22 12:45:00 | 1425.90 | 2025-05-30 10:15:00 | 1440.20 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-05-23 09:15:00 | 1417.00 | 2025-05-30 10:15:00 | 1440.20 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-05-23 10:15:00 | 1418.60 | 2025-06-02 12:15:00 | 1481.40 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2025-05-23 12:00:00 | 1416.40 | 2025-06-02 12:15:00 | 1481.40 | STOP_HIT | 1.00 | -4.59% |
| SELL | retest2 | 2025-05-23 14:30:00 | 1418.20 | 2025-06-02 12:15:00 | 1481.40 | STOP_HIT | 1.00 | -4.46% |
| BUY | retest2 | 2025-09-01 09:15:00 | 1931.20 | 2025-09-01 11:15:00 | 1905.60 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-09-01 14:30:00 | 1926.00 | 2025-09-09 13:15:00 | 2118.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 15:00:00 | 1927.20 | 2025-09-10 09:15:00 | 2119.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-02 09:30:00 | 1928.40 | 2025-09-10 09:15:00 | 2121.24 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-14 14:45:00 | 1903.40 | 2025-11-27 09:15:00 | 1947.70 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2025-11-17 10:30:00 | 1900.60 | 2025-11-27 09:15:00 | 1947.70 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-12-09 09:15:00 | 1895.70 | 2025-12-09 10:15:00 | 1932.50 | STOP_HIT | 1.00 | -1.94% |
| BUY | retest2 | 2026-01-12 14:45:00 | 2019.00 | 2026-01-13 09:15:00 | 1961.90 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-01-14 09:15:00 | 2027.30 | 2026-01-20 09:15:00 | 1967.40 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2026-01-14 10:15:00 | 2021.80 | 2026-01-20 09:15:00 | 1967.40 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2026-01-16 10:00:00 | 2024.80 | 2026-01-20 09:15:00 | 1967.40 | STOP_HIT | 1.00 | -2.83% |
| BUY | retest2 | 2026-01-20 11:30:00 | 1980.70 | 2026-01-20 13:15:00 | 1941.30 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2026-01-22 09:45:00 | 1992.70 | 2026-01-23 14:15:00 | 1966.80 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2026-01-22 11:00:00 | 1978.10 | 2026-01-23 14:15:00 | 1966.80 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-01-22 11:45:00 | 1977.20 | 2026-01-27 10:15:00 | 1973.10 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2026-01-22 15:15:00 | 2004.00 | 2026-02-01 09:15:00 | 1975.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2026-01-23 12:30:00 | 1993.20 | 2026-02-01 09:15:00 | 1975.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2026-01-27 09:30:00 | 2004.70 | 2026-02-01 09:15:00 | 1975.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-01-27 14:45:00 | 1996.60 | 2026-02-01 09:15:00 | 1975.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-01-29 11:00:00 | 2005.90 | 2026-02-01 12:15:00 | 1971.30 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-29 12:30:00 | 2009.60 | 2026-02-01 14:15:00 | 1947.40 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2026-01-30 11:15:00 | 2006.60 | 2026-02-01 14:15:00 | 1947.40 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2026-01-30 12:00:00 | 2006.50 | 2026-02-01 14:15:00 | 1947.40 | STOP_HIT | 1.00 | -2.95% |
| BUY | retest2 | 2026-02-11 13:30:00 | 2011.30 | 2026-03-10 12:15:00 | 2212.43 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-11 14:00:00 | 2012.60 | 2026-03-10 12:15:00 | 2213.86 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-12 09:45:00 | 2016.90 | 2026-03-10 12:15:00 | 2218.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-13 10:15:00 | 2018.80 | 2026-03-10 12:15:00 | 2220.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 11:30:00 | 2048.90 | 2026-04-15 14:15:00 | 2253.79 | TARGET_HIT | 1.00 | 10.00% |
