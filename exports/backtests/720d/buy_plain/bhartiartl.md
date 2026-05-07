# BHARTIARTL (BHARTIARTL)

## Backtest Summary

- **Source:** Fyers history API (1H bars)
- **Window:** 2024-05-18 09:15:00 → 2026-05-07 15:15:00 (3402 bars)
- **Last close:** 1829.60
- **Strategy:** EMA 200/400 1H crossover (v2 BTC rules)
- **Target:** entry × (1 ± 30%)
- **Partial:** 50% qty booked @ 15%, trail SL → entry
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 3 attempts each at retest1 and retest2

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 2 |
| ALERT2 | 2 |
| ALERT2_SKIP | 2 |
| ALERT3 | 7 |
| PENDING | 30 |
| PENDING_CANCEL | 7 |
| ENTRY1 | 0 |
| ENTRY2 | 23 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 23 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 24 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 20
- **Target hits / Stop hits / Partials:** 0 / 23 / 1
- **Avg / median % per leg:** -0.62% / -1.93%
- **Sum % (uncompounded):** -14.94%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 4 | 16.7% | 0 | 23 | 1 | -0.62% | -14.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 24 | 4 | 16.7% | 0 | 23 | 1 | -0.62% | -14.9% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 24 | 4 | 16.7% | 0 | 23 | 1 | -0.62% | -14.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-12-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 12:15:00 | 1671.50 | 1600.00 | 1599.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 14:15:00 | 1682.05 | 1601.55 | 1600.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-18 10:15:00 | 1603.55 | 1607.40 | 1603.86 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 10:15:00 | 1603.55 | 1607.40 | 1603.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 1603.55 | 1607.40 | 1603.86 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2024-12-27 09:15:00 | 1618.30 | 1603.30 | 1602.28 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2024-12-27 10:15:00 | 1615.00 | 1603.42 | 1602.35 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-01-10 12:15:00 | 1617.25 | 1600.57 | 1600.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 13:15:00 | 1616.60 | 1600.73 | 1601.01 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-13 11:15:00 | 1600.40 | 1601.07 | 1601.18 | SL hit (close<static) qty=1.00 sl=1602.85 alert=retest2 |
| Cross detected — sustain check pending | 2025-01-14 09:15:00 | 1617.00 | 1601.05 | 1601.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-14 10:15:00 | 1618.50 | 1601.23 | 1601.26 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-01-14 11:15:00 | 1622.30 | 1601.44 | 1601.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 11:15:00 | 1622.30 | 1601.44 | 1601.36 | EMA200 above EMA400 |

### Cycle 3 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 1610.15 | 1601.32 | 1601.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 1621.60 | 1601.75 | 1601.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 10:15:00 | 1611.50 | 1615.26 | 1609.12 | EMA200 retest candle locked |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 10:15:00 | 1611.50 | 1615.26 | 1609.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 1611.50 | 1615.26 | 1609.12 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-01-30 14:15:00 | 1641.95 | 1615.31 | 1609.84 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 15:15:00 | 1642.50 | 1615.59 | 1610.01 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-01 13:15:00 | 1636.50 | 1616.61 | 1610.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-01 14:15:00 | 1620.60 | 1616.65 | 1610.90 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-03 09:15:00 | 1636.55 | 1616.93 | 1611.10 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:15:00 | 1657.95 | 1617.33 | 1611.33 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-04 10:15:00 | 1637.40 | 1619.40 | 1612.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 11:15:00 | 1642.65 | 1619.63 | 1612.74 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-06 11:15:00 | 1640.60 | 1623.90 | 1615.42 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-06 12:15:00 | 1632.00 | 1623.98 | 1615.50 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-02-07 09:15:00 | 1700.00 | 1624.62 | 1615.99 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:15:00 | 1696.15 | 1625.33 | 1616.39 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 1633.50 | 1653.54 | 1635.37 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-02-21 09:15:00 | 1651.80 | 1652.52 | 1635.47 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-21 10:15:00 | 1642.00 | 1652.42 | 1635.50 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-02-24 11:15:00 | 1604.05 | 1650.64 | 1635.26 | SL hit (close<static) qty=1.00 sl=1608.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-24 11:15:00 | 1604.05 | 1650.64 | 1635.26 | SL hit (close<static) qty=1.00 sl=1608.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-24 11:15:00 | 1604.05 | 1650.64 | 1635.26 | SL hit (close<static) qty=1.00 sl=1608.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-24 11:15:00 | 1604.05 | 1650.64 | 1635.26 | SL hit (close<static) qty=1.00 sl=1608.50 alert=retest2 |
| Cross detected — sustain check pending | 2025-02-27 10:15:00 | 1648.85 | 1648.00 | 1634.85 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-27 11:15:00 | 1650.40 | 1648.02 | 1634.93 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-02-27 14:15:00 | 1650.30 | 1648.09 | 1635.16 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-02-27 15:15:00 | 1646.00 | 1648.07 | 1635.21 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-02-28 09:15:00 | 1615.80 | 1647.75 | 1635.11 | SL hit (close<static) qty=1.00 sl=1629.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-10 09:15:00 | 1650.65 | 1633.62 | 1629.50 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-10 10:15:00 | 1646.00 | 1633.75 | 1629.58 | ENTRY2 sustain failed after 60m |
| Cross detected — sustain check pending | 2025-03-11 11:15:00 | 1653.25 | 1634.17 | 1629.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 12:15:00 | 1661.55 | 1634.45 | 1630.12 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-13 09:15:00 | 1652.85 | 1636.35 | 1631.33 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 10:15:00 | 1648.65 | 1636.47 | 1631.42 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-03-18 09:15:00 | 1650.65 | 1636.86 | 1631.94 | ENTRY2 cross detected — sustain check pending (15m) |
| Sustain check cancelled (price retraced) | 2025-03-18 10:15:00 | 1639.95 | 1636.89 | 1631.98 | ENTRY2 sustain failed after 60m |
| Stop hit — per-position SL triggered | 2025-03-18 13:15:00 | 1627.35 | 1636.70 | 1631.95 | SL hit (close<static) qty=1.00 sl=1629.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-18 13:15:00 | 1627.35 | 1636.70 | 1631.95 | SL hit (close<static) qty=1.00 sl=1629.00 alert=retest2 |
| Cross detected — sustain check pending | 2025-03-20 09:15:00 | 1664.85 | 1636.69 | 1632.17 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 10:15:00 | 1671.40 | 1637.03 | 1632.37 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Partial book — 50% qty @ 15%, trail SL → entry | 2025-06-20 10:15:00 | 1922.11 | 1850.30 | 1817.61 | Partial book 0.50 @ 15%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-11 09:15:00 | 1939.90 | 1948.67 | 1891.57 | SL hit (close<ema200) qty=0.50 sl=1948.67 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 1896.00 | 1942.50 | 1897.35 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-07-23 09:15:00 | 1916.50 | 1935.13 | 1897.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:15:00 | 1924.50 | 1935.02 | 1898.10 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-07-28 14:15:00 | 1890.40 | 1933.84 | 1901.87 | SL hit (close<static) qty=1.00 sl=1892.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-07-29 11:15:00 | 1921.80 | 1932.77 | 1901.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 12:15:00 | 1916.80 | 1932.62 | 1902.04 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-30 09:15:00 | 1935.00 | 1932.22 | 1902.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 10:15:00 | 1930.90 | 1932.21 | 1902.58 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-07-31 12:15:00 | 1919.10 | 1931.33 | 1903.44 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:15:00 | 1920.90 | 1931.23 | 1903.53 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1912.70 | 1930.70 | 1903.68 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 1883.90 | 1929.23 | 1903.60 | SL hit (close<static) qty=1.00 sl=1892.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 1883.90 | 1929.23 | 1903.60 | SL hit (close<static) qty=1.00 sl=1892.30 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 14:15:00 | 1883.90 | 1929.23 | 1903.60 | SL hit (close<static) qty=1.00 sl=1892.30 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-05 09:15:00 | 1926.90 | 1927.31 | 1903.74 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 10:15:00 | 1933.70 | 1927.37 | 1903.89 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-07 14:15:00 | 1920.10 | 1927.12 | 1905.79 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:15:00 | 1920.10 | 1927.05 | 1905.87 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 1867.50 | 1926.46 | 1905.67 | SL hit (close<static) qty=1.00 sl=1903.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 1867.50 | 1926.46 | 1905.67 | SL hit (close<static) qty=1.00 sl=1903.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-08-19 09:15:00 | 1926.70 | 1907.59 | 1899.04 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 10:15:00 | 1920.30 | 1907.72 | 1899.15 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-08-20 09:15:00 | 1938.70 | 1908.25 | 1899.66 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:15:00 | 1933.40 | 1908.50 | 1899.83 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 1914.20 | 1913.30 | 1903.50 | EMA400 retest candle locked |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1897.90 | 1912.98 | 1903.67 | SL hit (close<static) qty=1.00 sl=1903.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 1897.90 | 1912.98 | 1903.67 | SL hit (close<static) qty=1.00 sl=1903.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-09-16 11:15:00 | 1934.00 | 1902.87 | 1900.57 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 1930.60 | 1903.15 | 1900.72 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-18 10:15:00 | 1932.80 | 1907.23 | 1902.96 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 11:15:00 | 1936.90 | 1907.53 | 1903.13 | BUY ENTRY2 attempt 2/4 (retest2 break sustained 60m) |
| Cross detected — sustain check pending | 2025-09-24 12:15:00 | 1931.20 | 1916.56 | 1908.59 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 13:15:00 | 1935.60 | 1916.75 | 1908.73 | BUY ENTRY2 attempt 3/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 1896.50 | 1918.01 | 1910.35 | SL hit (close<static) qty=1.00 sl=1900.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 1896.50 | 1918.01 | 1910.35 | SL hit (close<static) qty=1.00 sl=1900.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 09:15:00 | 1896.50 | 1918.01 | 1910.35 | SL hit (close<static) qty=1.00 sl=1900.10 alert=retest2 |
| Cross detected — sustain check pending | 2025-10-07 09:15:00 | 1933.60 | 1910.54 | 1907.30 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:15:00 | 1928.30 | 1910.72 | 1907.41 | BUY ENTRY2 attempt 4/4 (retest2 break sustained 60m) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 2041.20 | 2085.16 | 2045.91 | EMA400 retest candle locked |
| Cross detected — sustain check pending | 2025-12-12 09:15:00 | 2063.90 | 2083.33 | 2046.14 | ENTRY2 cross detected — sustain check pending (15m) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-12 10:15:00 | 2068.00 | 2083.18 | 2046.25 | BUY ENTRY2 attempt 1/4 (retest2 break sustained 60m) |
| Stop hit — per-position SL triggered | 2026-01-09 13:15:00 | 2027.20 | 2093.91 | 2072.34 | SL hit (close<static) qty=1.00 sl=2038.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 10:15:00 | 1888.80 | 1996.58 | 2016.04 | SL hit (close<static) qty=1.00 sl=1900.10 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-01-10 13:15:00 | 1616.60 | 2025-01-13 11:15:00 | 1600.40 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-01-14 10:15:00 | 1618.50 | 2025-01-14 11:15:00 | 1622.30 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-01-30 15:15:00 | 1642.50 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-02-03 10:15:00 | 1657.95 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -3.25% |
| BUY | retest2 | 2025-02-04 11:15:00 | 1642.65 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-02-07 10:15:00 | 1696.15 | 2025-02-24 11:15:00 | 1604.05 | STOP_HIT | 1.00 | -5.43% |
| BUY | retest2 | 2025-02-27 11:15:00 | 1650.40 | 2025-02-28 09:15:00 | 1615.80 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2025-03-11 12:15:00 | 1661.55 | 2025-03-18 13:15:00 | 1627.35 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-03-13 10:15:00 | 1648.65 | 2025-03-18 13:15:00 | 1627.35 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-03-20 10:15:00 | 1671.40 | 2025-06-20 10:15:00 | 1922.11 | PARTIAL | 0.50 | 15.00% |
| BUY | retest2 | 2025-03-20 10:15:00 | 1671.40 | 2025-07-11 09:15:00 | 1939.90 | STOP_HIT | 0.50 | 16.06% |
| BUY | retest2 | 2025-07-23 10:15:00 | 1924.50 | 2025-07-28 14:15:00 | 1890.40 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-07-29 12:15:00 | 1916.80 | 2025-08-01 14:15:00 | 1883.90 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-07-30 10:15:00 | 1930.90 | 2025-08-01 14:15:00 | 1883.90 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2025-07-31 13:15:00 | 1920.90 | 2025-08-01 14:15:00 | 1883.90 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-08-05 10:15:00 | 1933.70 | 2025-08-08 09:15:00 | 1867.50 | STOP_HIT | 1.00 | -3.42% |
| BUY | retest2 | 2025-08-07 15:15:00 | 1920.10 | 2025-08-08 09:15:00 | 1867.50 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-08-19 10:15:00 | 1920.30 | 2025-08-28 09:15:00 | 1897.90 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-08-20 10:15:00 | 1933.40 | 2025-08-28 09:15:00 | 1897.90 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-09-16 12:15:00 | 1930.60 | 2025-09-30 09:15:00 | 1896.50 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-09-18 11:15:00 | 1936.90 | 2025-09-30 09:15:00 | 1896.50 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-09-24 13:15:00 | 1935.60 | 2025-09-30 09:15:00 | 1896.50 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-10-07 10:15:00 | 1928.30 | 2026-01-09 13:15:00 | 2027.20 | STOP_HIT | 1.00 | 5.13% |
| BUY | retest2 | 2025-12-12 10:15:00 | 2068.00 | 2026-02-27 10:15:00 | 1888.80 | STOP_HIT | 1.00 | -8.67% |
