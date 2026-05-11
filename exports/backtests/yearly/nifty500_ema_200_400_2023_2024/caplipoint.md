# Caplin Point Laboratories Ltd. (CAPLIPOINT)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 1854.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 8 |
| ALERT2 | 8 |
| ALERT2_SKIP | 2 |
| ALERT3 | 45 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 44 |
| PARTIAL | 7 |
| TARGET_HIT | 9 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 51 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 30
- **Target hits / Stop hits / Partials:** 9 / 35 / 7
- **Avg / median % per leg:** 0.92% / -1.30%
- **Sum % (uncompounded):** 47.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 23 | 6 | 26.1% | 6 | 17 | 0 | 0.76% | 17.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 23 | 6 | 26.1% | 6 | 17 | 0 | 0.76% | 17.4% |
| SELL (all) | 28 | 15 | 53.6% | 3 | 18 | 7 | 1.06% | 29.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 15 | 53.6% | 3 | 18 | 7 | 1.06% | 29.7% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 51 | 21 | 41.2% | 9 | 35 | 7 | 0.92% | 47.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 11:15:00 | 1265.25 | 1384.68 | 1385.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-21 12:15:00 | 1263.50 | 1383.48 | 1384.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-01 09:15:00 | 1371.10 | 1353.69 | 1368.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-01 09:15:00 | 1371.10 | 1353.69 | 1368.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 09:15:00 | 1371.10 | 1353.69 | 1368.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 10:15:00 | 1382.00 | 1353.69 | 1368.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 10:15:00 | 1403.90 | 1354.19 | 1368.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 11:00:00 | 1403.90 | 1354.19 | 1368.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 11:15:00 | 1388.95 | 1354.54 | 1368.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 11:30:00 | 1395.40 | 1354.54 | 1368.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 13:15:00 | 1365.35 | 1354.77 | 1368.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 13:30:00 | 1369.00 | 1354.77 | 1368.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 14:15:00 | 1374.45 | 1354.96 | 1368.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-01 15:00:00 | 1374.45 | 1354.96 | 1368.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-01 15:15:00 | 1373.00 | 1355.14 | 1368.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-02 09:15:00 | 1345.40 | 1355.14 | 1368.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 09:15:00 | 1317.95 | 1354.77 | 1368.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 10:15:00 | 1304.45 | 1352.22 | 1366.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-03 11:30:00 | 1306.75 | 1351.39 | 1365.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 12:00:00 | 1306.40 | 1348.77 | 1364.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-04 12:45:00 | 1306.35 | 1348.35 | 1363.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 1339.00 | 1328.32 | 1346.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-24 13:30:00 | 1328.65 | 1329.09 | 1346.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-25 09:30:00 | 1326.80 | 1329.32 | 1346.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-25 10:15:00 | 1356.95 | 1329.60 | 1346.16 | SL hit (close>static) qty=1.00 sl=1353.95 alert=retest2 |

### Cycle 2 — BUY (started 2024-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 11:15:00 | 1469.90 | 1334.77 | 1334.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 14:15:00 | 1484.00 | 1374.76 | 1357.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 13:15:00 | 1507.80 | 1514.72 | 1464.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 14:00:00 | 1507.80 | 1514.72 | 1464.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1474.00 | 1513.58 | 1464.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:45:00 | 1463.65 | 1513.58 | 1464.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1869.55 | 1905.95 | 1835.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:45:00 | 1868.65 | 1905.95 | 1835.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 1824.95 | 1903.15 | 1835.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 15:00:00 | 1824.95 | 1903.15 | 1835.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 1835.00 | 1902.47 | 1835.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-24 09:15:00 | 1846.15 | 1902.47 | 1835.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-24 09:45:00 | 1840.45 | 1901.80 | 1835.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-24 10:15:00 | 1836.85 | 1901.80 | 1835.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-24 12:00:00 | 1838.55 | 1900.47 | 1835.45 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 1840.90 | 1899.88 | 1835.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:15:00 | 1830.25 | 1899.88 | 1835.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 1846.15 | 1899.34 | 1835.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-24 14:45:00 | 1858.00 | 1899.04 | 1835.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 09:15:00 | 1813.00 | 1897.75 | 1835.69 | SL hit (close<static) qty=1.00 sl=1821.05 alert=retest2 |

### Cycle 3 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 2024.05 | 2198.38 | 2199.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 1956.75 | 2153.13 | 2173.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 10:15:00 | 1956.55 | 1956.52 | 2030.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-20 11:00:00 | 1956.55 | 1956.52 | 2030.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2014.75 | 1958.23 | 2027.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2014.75 | 1958.23 | 2027.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2020.05 | 1958.85 | 2027.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:45:00 | 2064.00 | 1959.82 | 2027.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 2075.00 | 1960.97 | 2027.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-24 11:00:00 | 2075.00 | 1960.97 | 2027.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 2050.95 | 1969.25 | 2029.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 13:15:00 | 2043.00 | 1969.25 | 2029.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 14:15:00 | 2044.00 | 1970.03 | 2029.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 15:15:00 | 2042.00 | 1970.93 | 2029.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 09:30:00 | 2013.80 | 1972.09 | 2029.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 09:15:00 | 1940.85 | 1973.41 | 2028.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 09:15:00 | 1941.80 | 1973.41 | 2028.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-27 09:15:00 | 1939.90 | 1973.41 | 2028.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-27 10:15:00 | 1982.90 | 1973.50 | 2028.00 | SL hit (close>ema200) qty=0.50 sl=1973.50 alert=retest2 |

### Cycle 4 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 2271.90 | 1978.94 | 1977.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 14:15:00 | 2296.40 | 1993.77 | 1985.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 09:15:00 | 2055.70 | 2081.69 | 2041.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 10:00:00 | 2055.70 | 2081.69 | 2041.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 2025.10 | 2079.78 | 2041.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 2025.10 | 2079.78 | 2041.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 2022.00 | 2079.20 | 2041.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 2022.00 | 2079.20 | 2041.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 2099.70 | 2085.91 | 2052.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 2102.00 | 2086.00 | 2052.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 2043.90 | 2083.42 | 2053.35 | SL hit (close<static) qty=1.00 sl=2045.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 1930.00 | 2053.01 | 2053.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 09:15:00 | 1900.90 | 2047.97 | 2050.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 2069.00 | 2044.02 | 2048.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 2069.00 | 2044.02 | 2048.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 2069.00 | 2044.02 | 2048.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 2069.00 | 2044.02 | 2048.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 2021.00 | 2043.79 | 2048.45 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 2139.50 | 2053.21 | 2052.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 12:15:00 | 2143.10 | 2058.18 | 2055.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 2086.70 | 2098.61 | 2079.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-26 15:15:00 | 2079.50 | 2098.61 | 2079.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 2079.50 | 2098.42 | 2079.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:30:00 | 2109.20 | 2098.63 | 2079.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-01 09:30:00 | 2113.70 | 2101.90 | 2082.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-17 10:15:00 | 2320.12 | 2161.36 | 2122.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 2038.50 | 2120.31 | 2120.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 2009.50 | 2111.51 | 2116.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 13:15:00 | 1952.60 | 1949.76 | 1996.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 14:00:00 | 1952.60 | 1949.76 | 1996.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 1977.90 | 1941.96 | 1975.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 09:45:00 | 1981.70 | 1941.96 | 1975.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1967.40 | 1942.21 | 1975.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 12:30:00 | 1964.50 | 1944.75 | 1974.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 09:15:00 | 1981.90 | 1945.78 | 1974.82 | SL hit (close>static) qty=1.00 sl=1979.20 alert=retest2 |

### Cycle 8 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 1860.70 | 1725.35 | 1724.68 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-11-03 13:00:00 | 1042.40 | 2023-11-10 10:15:00 | 1146.64 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-11-03 14:30:00 | 1047.35 | 2023-11-10 10:15:00 | 1152.09 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-04-03 10:15:00 | 1304.45 | 2024-04-25 10:15:00 | 1356.95 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2024-04-03 11:30:00 | 1306.75 | 2024-04-25 10:15:00 | 1356.95 | STOP_HIT | 1.00 | -3.84% |
| SELL | retest2 | 2024-04-04 12:00:00 | 1306.40 | 2024-05-10 09:15:00 | 1260.65 | PARTIAL | 0.50 | 3.50% |
| SELL | retest2 | 2024-04-04 12:00:00 | 1306.40 | 2024-05-14 10:15:00 | 1323.00 | STOP_HIT | 0.50 | -1.27% |
| SELL | retest2 | 2024-04-04 12:45:00 | 1306.35 | 2024-05-16 11:15:00 | 1357.00 | STOP_HIT | 1.00 | -3.88% |
| SELL | retest2 | 2024-04-24 13:30:00 | 1328.65 | 2024-05-16 13:15:00 | 1384.00 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2024-04-25 09:30:00 | 1326.80 | 2024-05-16 13:15:00 | 1384.00 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest2 | 2024-05-03 09:30:00 | 1327.00 | 2024-05-16 13:15:00 | 1384.00 | STOP_HIT | 1.00 | -4.30% |
| SELL | retest2 | 2024-05-14 14:15:00 | 1328.05 | 2024-05-16 13:15:00 | 1384.00 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2024-05-17 12:30:00 | 1356.90 | 2024-05-21 09:15:00 | 1289.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-17 13:00:00 | 1353.35 | 2024-05-21 09:15:00 | 1285.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-17 12:30:00 | 1356.90 | 2024-06-06 09:15:00 | 1307.90 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2024-05-17 13:00:00 | 1353.35 | 2024-06-06 09:15:00 | 1307.90 | STOP_HIT | 0.50 | 3.36% |
| SELL | retest2 | 2024-06-07 13:00:00 | 1356.10 | 2024-06-10 09:15:00 | 1456.80 | STOP_HIT | 1.00 | -7.43% |
| BUY | retest2 | 2024-10-24 09:15:00 | 1846.15 | 2024-10-25 09:15:00 | 1813.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-10-24 09:45:00 | 1840.45 | 2024-10-25 09:15:00 | 1813.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-10-24 10:15:00 | 1836.85 | 2024-10-25 09:15:00 | 1813.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-10-24 12:00:00 | 1838.55 | 2024-10-25 09:15:00 | 1813.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-10-24 14:45:00 | 1858.00 | 2024-10-25 09:15:00 | 1813.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-10-29 12:30:00 | 1869.70 | 2024-11-01 17:15:00 | 2056.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-01-28 12:00:00 | 1864.95 | 2025-01-31 09:15:00 | 2051.45 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-25 13:15:00 | 2043.00 | 2025-03-27 09:15:00 | 1940.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 14:15:00 | 2044.00 | 2025-03-27 09:15:00 | 1941.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 15:15:00 | 2042.00 | 2025-03-27 09:15:00 | 1939.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-25 13:15:00 | 2043.00 | 2025-03-27 10:15:00 | 1982.90 | STOP_HIT | 0.50 | 2.94% |
| SELL | retest2 | 2025-03-25 14:15:00 | 2044.00 | 2025-03-27 10:15:00 | 1982.90 | STOP_HIT | 0.50 | 2.99% |
| SELL | retest2 | 2025-03-25 15:15:00 | 2042.00 | 2025-03-27 10:15:00 | 1982.90 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2025-03-26 09:30:00 | 2013.80 | 2025-04-03 14:15:00 | 2058.80 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2025-03-28 13:45:00 | 1992.60 | 2025-04-03 14:15:00 | 2058.80 | STOP_HIT | 1.00 | -3.32% |
| SELL | retest2 | 2025-03-28 15:15:00 | 1990.00 | 2025-04-07 09:15:00 | 1812.42 | TARGET_HIT | 1.00 | 8.92% |
| SELL | retest2 | 2025-04-04 09:45:00 | 1971.70 | 2025-04-07 09:15:00 | 1774.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-13 13:15:00 | 1993.90 | 2025-05-19 09:15:00 | 2086.20 | STOP_HIT | 1.00 | -4.63% |
| BUY | retest2 | 2025-06-18 11:30:00 | 2102.00 | 2025-06-20 09:15:00 | 2043.90 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-06-25 09:30:00 | 2101.70 | 2025-07-01 14:15:00 | 1999.90 | STOP_HIT | 1.00 | -4.84% |
| BUY | retest2 | 2025-06-25 11:00:00 | 2105.90 | 2025-07-01 14:15:00 | 1999.90 | STOP_HIT | 1.00 | -5.03% |
| BUY | retest2 | 2025-06-25 11:30:00 | 2101.60 | 2025-07-01 14:15:00 | 1999.90 | STOP_HIT | 1.00 | -4.84% |
| BUY | retest2 | 2025-07-16 10:00:00 | 2095.20 | 2025-07-25 14:15:00 | 2038.00 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-07-17 09:15:00 | 2098.00 | 2025-07-25 14:15:00 | 2038.00 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-07-18 11:00:00 | 2095.50 | 2025-07-25 14:15:00 | 2038.00 | STOP_HIT | 1.00 | -2.74% |
| BUY | retest2 | 2025-07-21 10:15:00 | 2103.70 | 2025-07-25 14:15:00 | 2038.00 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-07-28 09:30:00 | 2065.40 | 2025-07-28 14:15:00 | 2046.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-07-28 14:30:00 | 2064.70 | 2025-07-29 10:15:00 | 2041.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-07-29 12:45:00 | 2071.20 | 2025-08-01 09:15:00 | 2039.60 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-08-28 09:30:00 | 2109.20 | 2025-09-17 10:15:00 | 2320.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-01 09:30:00 | 2113.70 | 2025-09-17 10:15:00 | 2325.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-26 09:45:00 | 2092.90 | 2025-09-26 13:15:00 | 2057.20 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-12-24 12:30:00 | 1964.50 | 2025-12-26 09:15:00 | 1981.90 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-12-26 14:00:00 | 1947.80 | 2025-12-30 09:15:00 | 1850.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-26 14:00:00 | 1947.80 | 2025-12-30 12:15:00 | 1753.02 | TARGET_HIT | 0.50 | 10.00% |
