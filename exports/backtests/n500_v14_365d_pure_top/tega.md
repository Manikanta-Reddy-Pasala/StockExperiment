# Tega Industries Ltd. (TEGA)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
- **Last close:** 1659.00
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
| ALERT2 | 2 |
| ALERT2_SKIP | 0 |
| ALERT3 | 46 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 11 |
| TARGET_HIT | 3 |
| STOP_HIT | 31 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 45 (incl. partial bookings)
- **Trades open at end:** 6
- **Winners / losers:** 17 / 28
- **Target hits / Stop hits / Partials:** 3 / 31 / 11
- **Avg / median % per leg:** 1.08% / -0.83%
- **Sum % (uncompounded):** 48.62%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 28 | 1 | 3.6% | 1 | 27 | 0 | -0.95% | -26.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 28 | 1 | 3.6% | 1 | 27 | 0 | -0.95% | -26.5% |
| SELL (all) | 17 | 16 | 94.1% | 2 | 4 | 11 | 4.42% | 75.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 17 | 16 | 94.1% | 2 | 4 | 11 | 4.42% | 75.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 45 | 17 | 37.8% | 3 | 31 | 11 | 1.08% | 48.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-28 11:15:00 | 1622.00 | 1427.11 | 1426.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 11:15:00 | 1633.40 | 1440.20 | 1433.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 11:15:00 | 1512.10 | 1519.08 | 1483.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 11:45:00 | 1514.00 | 1519.08 | 1483.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1484.80 | 1518.22 | 1484.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:45:00 | 1485.60 | 1518.22 | 1484.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 1499.10 | 1518.03 | 1484.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:45:00 | 1500.50 | 1517.61 | 1484.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1502.60 | 1517.25 | 1484.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 1480.10 | 1516.42 | 1484.81 | SL hit (close<static) qty=1.00 sl=1484.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 1480.10 | 1516.42 | 1484.81 | SL hit (close<static) qty=1.00 sl=1484.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 1505.00 | 1512.79 | 1484.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 15:15:00 | 1478.00 | 1511.37 | 1484.75 | SL hit (close<static) qty=1.00 sl=1484.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:00:00 | 1501.10 | 1511.12 | 1484.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1482.20 | 1510.36 | 1485.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 1482.20 | 1510.36 | 1485.28 | SL hit (close<static) qty=1.00 sl=1484.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-06-23 09:45:00 | 1480.50 | 1510.36 | 1485.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1476.90 | 1510.03 | 1485.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 1488.50 | 1508.38 | 1485.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 11:45:00 | 1486.20 | 1507.83 | 1485.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 1498.00 | 1506.65 | 1484.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 1471.80 | 1505.54 | 1484.81 | SL hit (close<static) qty=1.00 sl=1472.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 1471.80 | 1505.54 | 1484.81 | SL hit (close<static) qty=1.00 sl=1472.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 12:15:00 | 1471.80 | 1505.54 | 1484.81 | SL hit (close<static) qty=1.00 sl=1472.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 09:15:00 | 1488.20 | 1502.56 | 1484.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-04 09:15:00 | 1637.02 | 1521.83 | 1497.54 | Target hit (10%) qty=1.00 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 1882.00 | 1959.24 | 1886.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 1882.00 | 1959.24 | 1886.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 1879.90 | 1958.45 | 1886.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:30:00 | 1872.40 | 1958.45 | 1886.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1886.00 | 1957.00 | 1886.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 1886.00 | 1957.00 | 1886.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1885.00 | 1956.28 | 1886.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1882.60 | 1956.28 | 1886.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1900.20 | 1955.73 | 1886.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 1896.00 | 1955.73 | 1886.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 1878.00 | 1954.38 | 1886.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 11:45:00 | 1876.00 | 1954.38 | 1886.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1880.00 | 1953.64 | 1886.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:30:00 | 1884.70 | 1953.64 | 1886.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1886.00 | 1952.96 | 1886.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:45:00 | 1883.00 | 1952.96 | 1886.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 1890.00 | 1952.34 | 1886.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:30:00 | 1883.50 | 1952.34 | 1886.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 1891.30 | 1945.26 | 1902.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 1891.30 | 1945.26 | 1902.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 1889.30 | 1944.70 | 1902.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:30:00 | 1898.00 | 1944.70 | 1902.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 1889.00 | 1944.14 | 1902.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 13:00:00 | 1889.00 | 1944.14 | 1902.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 1875.90 | 1941.59 | 1902.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 1875.90 | 1941.59 | 1902.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 1902.00 | 1937.51 | 1901.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 12:15:00 | 1910.00 | 1937.18 | 1901.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 1880.00 | 1935.14 | 1902.74 | SL hit (close<static) qty=1.00 sl=1891.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 1905.40 | 1931.72 | 1902.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 11:15:00 | 1904.60 | 1931.43 | 1902.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1908.20 | 1929.76 | 1901.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 1904.00 | 1929.18 | 1901.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:45:00 | 1898.00 | 1929.18 | 1901.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 1900.10 | 1928.67 | 1901.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:45:00 | 1900.00 | 1928.67 | 1901.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 1904.90 | 1928.44 | 1902.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:45:00 | 1898.00 | 1928.44 | 1902.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 1917.00 | 1928.32 | 1902.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 14:30:00 | 1901.90 | 1928.32 | 1902.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1898.00 | 1927.79 | 1902.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:30:00 | 1895.50 | 1927.79 | 1902.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 1915.50 | 1927.67 | 1902.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 09:30:00 | 1931.20 | 1926.11 | 1902.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 13:00:00 | 1926.30 | 1928.45 | 1906.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 14:15:00 | 1891.20 | 1928.10 | 1906.86 | SL hit (close<static) qty=1.00 sl=1891.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 14:15:00 | 1891.20 | 1928.10 | 1906.86 | SL hit (close<static) qty=1.00 sl=1891.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 14:15:00 | 1891.20 | 1928.10 | 1906.86 | SL hit (close<static) qty=1.00 sl=1891.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 14:15:00 | 1891.20 | 1928.10 | 1906.86 | SL hit (close<static) qty=1.00 sl=1896.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 14:15:00 | 1891.20 | 1928.10 | 1906.86 | SL hit (close<static) qty=1.00 sl=1896.40 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 09:30:00 | 1930.50 | 1926.14 | 1906.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 13:15:00 | 1930.00 | 1927.31 | 1908.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 1911.90 | 1933.24 | 1914.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 1911.90 | 1933.24 | 1914.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 1935.80 | 1933.27 | 1914.46 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 1888.00 | 1929.33 | 1914.61 | SL hit (close<static) qty=1.00 sl=1896.40 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-21 14:15:00 | 1888.00 | 1929.33 | 1914.61 | SL hit (close<static) qty=1.00 sl=1896.40 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:00:00 | 1945.20 | 1921.71 | 1912.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:00:00 | 1937.40 | 1921.87 | 1912.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 15:00:00 | 1941.70 | 1922.71 | 1913.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 11:30:00 | 1937.00 | 1923.25 | 1913.62 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 1923.80 | 1924.97 | 1915.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:30:00 | 1914.00 | 1924.97 | 1915.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 1905.10 | 1924.77 | 1915.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1905.10 | 1924.77 | 1915.43 | SL hit (close<static) qty=1.00 sl=1911.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1905.10 | 1924.77 | 1915.43 | SL hit (close<static) qty=1.00 sl=1911.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1905.10 | 1924.77 | 1915.43 | SL hit (close<static) qty=1.00 sl=1911.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 12:15:00 | 1905.10 | 1924.77 | 1915.43 | SL hit (close<static) qty=1.00 sl=1911.90 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-04 13:00:00 | 1905.10 | 1924.77 | 1915.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1905.70 | 1924.58 | 1915.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:30:00 | 1907.80 | 1924.58 | 1915.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1906.10 | 1922.66 | 1914.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 15:00:00 | 1906.10 | 1922.66 | 1914.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 1917.10 | 1912.97 | 1910.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 15:00:00 | 1923.70 | 1913.08 | 1910.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 1908.90 | 1913.07 | 1910.80 | SL hit (close<static) qty=1.00 sl=1910.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 10:30:00 | 1926.00 | 1913.11 | 1910.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:00:00 | 1922.30 | 1913.11 | 1910.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 11:45:00 | 1923.00 | 1913.15 | 1910.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 1914.80 | 1913.22 | 1911.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 15:15:00 | 1918.30 | 1913.22 | 1911.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 1918.30 | 1913.27 | 1911.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 1914.20 | 1913.27 | 1911.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 1911.50 | 1913.25 | 1911.04 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 1910.00 | 1913.23 | 1911.05 | SL hit (close<static) qty=1.00 sl=1910.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 1910.00 | 1913.23 | 1911.05 | SL hit (close<static) qty=1.00 sl=1910.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 11:15:00 | 1910.00 | 1913.23 | 1911.05 | SL hit (close<static) qty=1.00 sl=1910.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 13:00:00 | 1923.60 | 1912.67 | 1910.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 09:30:00 | 1923.90 | 1912.79 | 1910.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 15:00:00 | 1925.00 | 1930.37 | 1921.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 1905.00 | 1934.22 | 1924.58 | SL hit (close<static) qty=1.00 sl=1905.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 1905.00 | 1934.22 | 1924.58 | SL hit (close<static) qty=1.00 sl=1905.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 1905.00 | 1934.22 | 1924.58 | SL hit (close<static) qty=1.00 sl=1905.30 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 1919.90 | 1932.72 | 1924.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 1893.20 | 1932.33 | 1923.95 | SL hit (close<static) qty=1.00 sl=1905.30 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 15:15:00 | 1920.00 | 1930.30 | 1923.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 09:15:00 | 1926.10 | 1930.30 | 1923.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 1924.80 | 1930.25 | 1923.32 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 1880.30 | 1917.33 | 1917.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 15:15:00 | 1863.00 | 1912.87 | 1915.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-04 14:15:00 | 1835.60 | 1808.11 | 1851.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-04 15:00:00 | 1835.60 | 1808.11 | 1851.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 1832.90 | 1806.97 | 1845.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 09:30:00 | 1846.40 | 1806.97 | 1845.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 1776.00 | 1745.99 | 1798.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 1819.70 | 1745.99 | 1798.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 1827.70 | 1746.81 | 1798.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 1827.70 | 1746.81 | 1798.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 10:15:00 | 1813.20 | 1747.47 | 1798.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:45:00 | 1802.70 | 1748.60 | 1798.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 1851.40 | 1753.22 | 1799.19 | SL hit (close>static) qty=1.00 sl=1829.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 1783.20 | 1762.10 | 1801.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 12:15:00 | 1694.04 | 1760.24 | 1798.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-06 09:15:00 | 1763.90 | 1755.83 | 1793.95 | SL hit (close>ema200) qty=0.50 sl=1755.83 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 12:45:00 | 1802.00 | 1751.70 | 1786.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 15:15:00 | 1800.00 | 1752.62 | 1786.26 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 1788.10 | 1753.63 | 1786.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 1788.10 | 1753.63 | 1786.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 1794.10 | 1754.03 | 1786.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:30:00 | 1791.80 | 1754.03 | 1786.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 12:15:00 | 1763.70 | 1754.13 | 1786.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 13:15:00 | 1760.30 | 1754.13 | 1786.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 1748.10 | 1754.39 | 1785.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 15:15:00 | 1711.90 | 1753.11 | 1784.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-16 15:15:00 | 1710.00 | 1753.11 | 1784.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:15:00 | 1672.28 | 1747.00 | 1778.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 09:15:00 | 1660.69 | 1747.00 | 1778.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-03-19 15:15:00 | 1621.80 | 1741.84 | 1775.17 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-03-19 15:15:00 | 1620.00 | 1741.84 | 1775.17 | Target hit (10%) qty=0.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 1739.00 | 1738.08 | 1772.11 | SL hit (close>ema200) qty=0.50 sl=1738.08 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 15:15:00 | 1739.00 | 1738.08 | 1772.11 | SL hit (close>ema200) qty=0.50 sl=1738.08 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 09:30:00 | 1762.00 | 1712.16 | 1739.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 10:30:00 | 1756.00 | 1712.56 | 1739.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 14:15:00 | 1739.90 | 1713.81 | 1739.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 14:45:00 | 1742.10 | 1713.81 | 1739.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 1738.30 | 1714.06 | 1739.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 09:15:00 | 1726.20 | 1714.06 | 1739.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-20 11:45:00 | 1727.10 | 1714.45 | 1739.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 12:30:00 | 1732.90 | 1715.31 | 1739.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 1730.40 | 1716.35 | 1738.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1727.80 | 1714.64 | 1735.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1727.80 | 1714.64 | 1735.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 11:15:00 | 1673.90 | 1713.69 | 1734.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 15:15:00 | 1668.20 | 1712.18 | 1733.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-29 14:15:00 | 1646.26 | 1709.40 | 1731.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 09:15:00 | 1643.88 | 1708.22 | 1730.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 1639.89 | 1707.58 | 1730.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-30 10:15:00 | 1640.74 | 1707.58 | 1730.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-16 13:45:00 | 1500.50 | 2025-06-17 12:15:00 | 1480.10 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-06-17 09:15:00 | 1502.60 | 2025-06-17 12:15:00 | 1480.10 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2025-06-19 09:30:00 | 1505.00 | 2025-06-19 15:15:00 | 1478.00 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2025-06-20 11:00:00 | 1501.10 | 2025-06-23 09:15:00 | 1482.20 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-06-24 09:15:00 | 1488.50 | 2025-06-25 12:15:00 | 1471.80 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-24 11:45:00 | 1486.20 | 2025-06-25 12:15:00 | 1471.80 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1498.00 | 2025-06-25 12:15:00 | 1471.80 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1488.20 | 2025-07-04 09:15:00 | 1637.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-20 12:15:00 | 1910.00 | 2025-10-24 09:15:00 | 1880.00 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2025-10-27 09:45:00 | 1905.40 | 2025-11-06 14:15:00 | 1891.20 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-10-27 11:15:00 | 1904.60 | 2025-11-06 14:15:00 | 1891.20 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1908.20 | 2025-11-06 14:15:00 | 1891.20 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-31 09:30:00 | 1931.20 | 2025-11-06 14:15:00 | 1891.20 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-11-06 13:00:00 | 1926.30 | 2025-11-06 14:15:00 | 1891.20 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-10 09:30:00 | 1930.50 | 2025-11-21 14:15:00 | 1888.00 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-11-11 13:15:00 | 1930.00 | 2025-11-21 14:15:00 | 1888.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2025-11-28 10:00:00 | 1945.20 | 2025-12-04 12:15:00 | 1905.10 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-11-28 11:00:00 | 1937.40 | 2025-12-04 12:15:00 | 1905.10 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-11-28 15:00:00 | 1941.70 | 2025-12-04 12:15:00 | 1905.10 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-12-01 11:30:00 | 1937.00 | 2025-12-04 12:15:00 | 1905.10 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-12-12 15:00:00 | 1923.70 | 2025-12-15 09:15:00 | 1908.90 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-12-16 10:30:00 | 1926.00 | 2025-12-17 11:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-12-16 11:00:00 | 1922.30 | 2025-12-17 11:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-12-16 11:45:00 | 1923.00 | 2025-12-17 11:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-18 13:00:00 | 1923.60 | 2026-01-05 12:15:00 | 1905.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-12-19 09:30:00 | 1923.90 | 2026-01-05 12:15:00 | 1905.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-12-29 15:00:00 | 1925.00 | 2026-01-05 12:15:00 | 1905.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2026-01-06 12:15:00 | 1919.90 | 2026-01-06 12:15:00 | 1893.20 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-02-25 12:45:00 | 1802.70 | 2026-02-26 11:15:00 | 1851.40 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1783.20 | 2026-03-04 12:15:00 | 1694.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 1783.20 | 2026-03-06 09:15:00 | 1763.90 | STOP_HIT | 0.50 | 1.08% |
| SELL | retest2 | 2026-03-12 12:45:00 | 1802.00 | 2026-03-16 15:15:00 | 1711.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 15:15:00 | 1800.00 | 2026-03-16 15:15:00 | 1710.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 13:15:00 | 1760.30 | 2026-03-19 09:15:00 | 1672.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 09:15:00 | 1748.10 | 2026-03-19 09:15:00 | 1660.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-12 12:45:00 | 1802.00 | 2026-03-19 15:15:00 | 1621.80 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-12 15:15:00 | 1800.00 | 2026-03-19 15:15:00 | 1620.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-03-13 13:15:00 | 1760.30 | 2026-03-20 15:15:00 | 1739.00 | STOP_HIT | 0.50 | 1.21% |
| SELL | retest2 | 2026-03-16 09:15:00 | 1748.10 | 2026-03-20 15:15:00 | 1739.00 | STOP_HIT | 0.50 | 0.52% |
| SELL | retest2 | 2026-04-17 09:30:00 | 1762.00 | 2026-04-28 11:15:00 | 1673.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-17 10:30:00 | 1756.00 | 2026-04-28 15:15:00 | 1668.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-20 09:15:00 | 1726.20 | 2026-04-29 14:15:00 | 1646.26 | PARTIAL | 0.50 | 4.63% |
| SELL | retest2 | 2026-04-20 11:45:00 | 1727.10 | 2026-04-30 09:15:00 | 1643.88 | PARTIAL | 0.50 | 4.82% |
| SELL | retest2 | 2026-04-21 12:30:00 | 1732.90 | 2026-04-30 10:15:00 | 1639.89 | PARTIAL | 0.50 | 5.37% |
| SELL | retest2 | 2026-04-23 10:15:00 | 1730.40 | 2026-04-30 10:15:00 | 1640.74 | PARTIAL | 0.50 | 5.18% |
