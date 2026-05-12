# Bombay Burmah Trading Corporation Ltd. (BBTC)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 1563.30
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 9 |
| ALERT1 | 9 |
| ALERT2 | 10 |
| ALERT2_SKIP | 4 |
| ALERT3 | 57 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 75 |
| PARTIAL | 17 |
| TARGET_HIT | 18 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 97 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 37 / 60
- **Target hits / Stop hits / Partials:** 18 / 62 / 17
- **Avg / median % per leg:** 1.59% / -0.91%
- **Sum % (uncompounded):** 154.30%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 34 | 4 | 11.8% | 4 | 30 | 0 | -0.37% | -12.6% |
| BUY @ 2nd Alert (retest1) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.84% | -9.2% |
| BUY @ 3rd Alert (retest2) | 29 | 4 | 13.8% | 4 | 25 | 0 | -0.12% | -3.4% |
| SELL (all) | 63 | 33 | 52.4% | 14 | 32 | 17 | 2.65% | 166.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 63 | 33 | 52.4% | 14 | 32 | 17 | 2.65% | 166.9% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -1.84% | -9.2% |
| retest2 (combined) | 92 | 37 | 40.2% | 18 | 57 | 17 | 1.78% | 163.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-18 09:15:00 | 946.65 | 942.15 | 923.40 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-19 09:15:00 | 944.90 | 942.40 | 924.17 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-19 11:00:00 | 945.15 | 942.41 | 924.36 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-05-19 13:00:00 | 944.75 | 942.49 | 924.58 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 15:15:00 | 928.90 | 944.30 | 929.81 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2023-05-30 15:15:00 | 928.90 | 944.30 | 929.81 | SL hit (close<ema400) qty=1.00 sl=929.81 alert=retest1 |

### Cycle 2 — SELL (started 2023-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 12:15:00 | 983.75 | 1030.76 | 1030.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 13:15:00 | 982.00 | 1030.27 | 1030.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-04 12:15:00 | 1022.70 | 1020.15 | 1025.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-04 13:00:00 | 1022.70 | 1020.15 | 1025.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 1064.70 | 1020.59 | 1025.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:00:00 | 1064.70 | 1020.59 | 1025.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 1056.25 | 1020.95 | 1025.54 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2023-09-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 14:15:00 | 1094.95 | 1030.29 | 1030.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-06 15:15:00 | 1101.00 | 1031.00 | 1030.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-13 14:15:00 | 1189.90 | 1192.15 | 1142.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-13 15:00:00 | 1189.90 | 1192.15 | 1142.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 1343.30 | 1395.00 | 1340.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 15:00:00 | 1343.30 | 1395.00 | 1340.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 15:15:00 | 1339.00 | 1394.45 | 1340.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 09:15:00 | 1352.00 | 1394.45 | 1340.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 10:00:00 | 1351.15 | 1394.01 | 1340.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 10:30:00 | 1345.15 | 1387.00 | 1344.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 11:45:00 | 1344.55 | 1386.63 | 1344.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 15:15:00 | 1347.00 | 1385.16 | 1344.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:15:00 | 1350.00 | 1385.16 | 1344.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 11:15:00 | 1339.75 | 1383.99 | 1344.27 | SL hit (close<static) qty=1.00 sl=1342.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-04-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-18 15:15:00 | 1528.00 | 1610.76 | 1610.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-19 10:15:00 | 1498.00 | 1608.64 | 1609.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-26 11:15:00 | 1599.00 | 1597.43 | 1603.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-26 11:15:00 | 1599.00 | 1597.43 | 1603.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 11:15:00 | 1599.00 | 1597.43 | 1603.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-26 11:30:00 | 1605.90 | 1597.43 | 1603.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-06 10:15:00 | 1585.00 | 1587.39 | 1597.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-06 11:00:00 | 1585.00 | 1587.39 | 1597.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 1587.15 | 1582.88 | 1594.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:15:00 | 1554.15 | 1582.41 | 1593.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 10:45:00 | 1550.00 | 1570.41 | 1586.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-15 15:00:00 | 1550.95 | 1568.41 | 1584.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 13:15:00 | 1563.10 | 1565.30 | 1580.88 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 1571.15 | 1565.12 | 1580.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:30:00 | 1574.95 | 1565.12 | 1580.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 1476.44 | 1551.36 | 1569.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 1473.40 | 1551.36 | 1569.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-30 14:15:00 | 1484.94 | 1551.36 | 1569.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:15:00 | 1472.50 | 1549.99 | 1568.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-06-04 11:15:00 | 1398.74 | 1538.35 | 1561.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 09:15:00 | 1637.80 | 1573.07 | 1572.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 1697.80 | 1577.82 | 1575.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-05 10:15:00 | 2127.50 | 2130.91 | 1971.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-05 11:00:00 | 2127.50 | 2130.91 | 1971.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 2524.35 | 2653.25 | 2484.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 11:45:00 | 2490.10 | 2653.25 | 2484.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 15:15:00 | 2485.00 | 2647.07 | 2484.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-08 09:15:00 | 2522.00 | 2647.07 | 2484.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-10-17 10:15:00 | 2774.20 | 2651.55 | 2523.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-12-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 13:15:00 | 2416.00 | 2571.67 | 2571.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 14:15:00 | 2405.75 | 2570.01 | 2570.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 2228.15 | 2139.11 | 2265.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 2228.15 | 2139.11 | 2265.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 2310.05 | 2140.81 | 2265.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 12:15:00 | 2215.30 | 2143.66 | 2264.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 2220.00 | 2146.20 | 2264.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 2104.53 | 2147.29 | 2259.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 2109.00 | 2147.29 | 2259.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-01 13:15:00 | 2167.75 | 2133.42 | 2232.00 | SL hit (close>ema200) qty=0.50 sl=2133.42 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 2004.80 | 1894.81 | 1894.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 09:15:00 | 2019.00 | 1898.97 | 1896.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-16 09:15:00 | 1927.90 | 1962.72 | 1937.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 1927.90 | 1962.72 | 1937.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 1927.90 | 1962.72 | 1937.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 10:00:00 | 1927.90 | 1962.72 | 1937.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 1938.70 | 1962.48 | 1937.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 13:15:00 | 1944.50 | 1961.95 | 1937.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 13:00:00 | 1941.30 | 1961.97 | 1938.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 13:30:00 | 1944.00 | 1961.79 | 1938.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 12:00:00 | 1942.10 | 1961.38 | 1938.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 1930.00 | 1961.06 | 1938.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:45:00 | 1926.90 | 1961.06 | 1938.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 1936.10 | 1960.82 | 1938.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 14:45:00 | 1943.70 | 1960.58 | 1938.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:30:00 | 1943.30 | 1960.00 | 1938.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 1912.50 | 1959.53 | 1938.17 | SL hit (close<static) qty=1.00 sl=1927.10 alert=retest2 |

### Cycle 8 — SELL (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 15:15:00 | 1862.60 | 1945.44 | 1945.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 1857.00 | 1942.99 | 1944.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-04 09:15:00 | 1888.10 | 1856.77 | 1888.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 1888.10 | 1856.77 | 1888.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1888.10 | 1856.77 | 1888.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 1888.10 | 1856.77 | 1888.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 1873.00 | 1856.93 | 1888.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:45:00 | 1854.70 | 1858.06 | 1887.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 1851.10 | 1857.60 | 1885.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 14:15:00 | 1860.00 | 1857.58 | 1885.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 15:15:00 | 1855.00 | 1857.64 | 1885.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1893.00 | 1857.97 | 1884.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 1900.10 | 1857.97 | 1884.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1882.00 | 1858.21 | 1884.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:30:00 | 1880.40 | 1858.43 | 1884.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:15:00 | 1880.90 | 1859.65 | 1884.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:45:00 | 1877.90 | 1859.85 | 1884.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 1881.90 | 1860.95 | 1884.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1897.70 | 1861.32 | 1884.86 | SL hit (close>static) qty=1.00 sl=1893.90 alert=retest2 |

### Cycle 9 — BUY (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 14:15:00 | 2045.50 | 1886.96 | 1886.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 15:15:00 | 2047.90 | 1888.56 | 1887.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 15:15:00 | 1945.40 | 1947.11 | 1922.97 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-06 09:15:00 | 1961.50 | 1947.11 | 1922.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 1917.50 | 1946.60 | 1923.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-06 11:15:00 | 1917.50 | 1946.60 | 1923.07 | SL hit (close<ema400) qty=1.00 sl=1923.07 alert=retest1 |

### Cycle 10 — SELL (started 2025-11-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-26 14:15:00 | 1839.80 | 1910.54 | 1910.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 10:15:00 | 1836.70 | 1904.08 | 1907.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 12:15:00 | 1914.30 | 1890.55 | 1899.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 12:15:00 | 1914.30 | 1890.55 | 1899.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 1914.30 | 1890.55 | 1899.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:15:00 | 1923.40 | 1890.55 | 1899.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 1903.80 | 1890.68 | 1899.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 1886.90 | 1890.68 | 1899.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:00:00 | 1871.30 | 1890.49 | 1899.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 15:15:00 | 1885.50 | 1879.58 | 1892.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 1887.90 | 1879.99 | 1892.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 13:15:00 | 1891.80 | 1880.10 | 1892.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 13:30:00 | 1896.40 | 1880.10 | 1892.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1864.80 | 1879.95 | 1892.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 15:15:00 | 1854.00 | 1879.95 | 1892.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 1861.20 | 1879.62 | 1892.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-15 09:15:00 | 1853.30 | 1879.35 | 1891.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 09:15:00 | 1860.90 | 1878.78 | 1890.54 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1918.10 | 1879.17 | 1890.68 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-16 09:15:00 | 1918.10 | 1879.17 | 1890.68 | SL hit (close>static) qty=1.00 sl=1894.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2023-05-18 09:15:00 | 946.65 | 2023-05-30 15:15:00 | 928.90 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest1 | 2023-05-19 09:15:00 | 944.90 | 2023-05-30 15:15:00 | 928.90 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest1 | 2023-05-19 11:00:00 | 945.15 | 2023-05-30 15:15:00 | 928.90 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2023-05-19 13:00:00 | 944.75 | 2023-05-30 15:15:00 | 928.90 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2023-05-31 13:45:00 | 939.00 | 2023-06-13 10:15:00 | 1032.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-06-06 11:00:00 | 936.00 | 2023-06-13 10:15:00 | 1029.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-13 09:15:00 | 1352.00 | 2023-12-20 11:15:00 | 1339.75 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-12-13 10:00:00 | 1351.15 | 2023-12-20 12:15:00 | 1333.35 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2023-12-19 10:30:00 | 1345.15 | 2023-12-20 12:15:00 | 1333.35 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2023-12-19 11:45:00 | 1344.55 | 2023-12-20 12:15:00 | 1333.35 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2023-12-20 09:15:00 | 1350.00 | 2023-12-20 12:15:00 | 1333.35 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2023-12-22 09:45:00 | 1350.00 | 2023-12-22 10:15:00 | 1337.55 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2023-12-22 10:45:00 | 1350.10 | 2023-12-22 11:15:00 | 1329.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2023-12-26 10:30:00 | 1365.90 | 2023-12-27 09:15:00 | 1502.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-04-04 13:30:00 | 1661.25 | 2024-04-05 12:15:00 | 1625.05 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-05-09 10:15:00 | 1554.15 | 2024-05-30 14:15:00 | 1476.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-14 10:45:00 | 1550.00 | 2024-05-30 14:15:00 | 1473.40 | PARTIAL | 0.50 | 4.94% |
| SELL | retest2 | 2024-05-15 15:00:00 | 1550.95 | 2024-05-30 14:15:00 | 1484.94 | PARTIAL | 0.50 | 4.26% |
| SELL | retest2 | 2024-05-21 13:15:00 | 1563.10 | 2024-05-31 09:15:00 | 1472.50 | PARTIAL | 0.50 | 5.80% |
| SELL | retest2 | 2024-05-09 10:15:00 | 1554.15 | 2024-06-04 11:15:00 | 1398.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-14 10:45:00 | 1550.00 | 2024-06-04 11:15:00 | 1395.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-15 15:00:00 | 1550.95 | 2024-06-04 11:15:00 | 1395.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-05-21 13:15:00 | 1563.10 | 2024-06-04 11:15:00 | 1406.79 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-06-06 14:15:00 | 1531.65 | 2024-06-07 10:15:00 | 1608.25 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2024-10-08 09:15:00 | 2522.00 | 2024-10-17 10:15:00 | 2774.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-18 09:45:00 | 2515.00 | 2024-11-29 12:15:00 | 2434.50 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2024-11-25 09:15:00 | 2541.40 | 2024-11-29 12:15:00 | 2434.50 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2025-01-23 12:15:00 | 2215.30 | 2025-01-27 09:15:00 | 2104.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:15:00 | 2220.00 | 2025-01-27 09:15:00 | 2109.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 12:15:00 | 2215.30 | 2025-02-01 13:15:00 | 2167.75 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2025-01-23 15:15:00 | 2220.00 | 2025-02-01 13:15:00 | 2167.75 | STOP_HIT | 0.50 | 2.35% |
| BUY | retest2 | 2025-06-16 13:15:00 | 1944.50 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-06-17 13:00:00 | 1941.30 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-06-17 13:30:00 | 1944.00 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-06-18 12:00:00 | 1942.10 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-06-18 14:45:00 | 1943.70 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-06-19 09:30:00 | 1943.30 | 2025-06-19 10:15:00 | 1912.50 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1942.20 | 2025-07-09 15:15:00 | 1940.10 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-06-27 10:00:00 | 1956.20 | 2025-07-09 15:15:00 | 1940.10 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-07-08 14:45:00 | 1960.10 | 2025-07-09 15:15:00 | 1940.10 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-07-09 10:15:00 | 1961.50 | 2025-07-11 15:15:00 | 1940.00 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-09 14:00:00 | 1962.90 | 2025-07-22 15:15:00 | 1936.10 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-10 09:15:00 | 1963.40 | 2025-07-28 10:15:00 | 1935.80 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-07-15 13:30:00 | 1988.60 | 2025-07-28 10:15:00 | 1935.80 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2025-07-23 09:45:00 | 2002.00 | 2025-07-28 11:15:00 | 1923.00 | STOP_HIT | 1.00 | -3.95% |
| BUY | retest2 | 2025-07-24 09:15:00 | 2009.30 | 2025-07-28 11:15:00 | 1923.00 | STOP_HIT | 1.00 | -4.30% |
| SELL | retest2 | 2025-09-05 10:45:00 | 1854.70 | 2025-09-12 09:15:00 | 1897.70 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-09-09 09:15:00 | 1851.10 | 2025-09-12 09:15:00 | 1897.70 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-09-09 14:15:00 | 1860.00 | 2025-09-12 09:15:00 | 1897.70 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-09-09 15:15:00 | 1855.00 | 2025-09-12 09:15:00 | 1897.70 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2025-09-10 11:30:00 | 1880.40 | 2025-09-15 10:15:00 | 1911.10 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-09-11 10:15:00 | 1880.90 | 2025-09-15 10:15:00 | 1911.10 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2025-09-11 10:45:00 | 1877.90 | 2025-09-15 10:15:00 | 1911.10 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-09-12 09:15:00 | 1881.90 | 2025-09-15 10:15:00 | 1911.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-09-24 09:15:00 | 1875.00 | 2025-09-24 13:15:00 | 1890.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-09-24 12:00:00 | 1874.50 | 2025-09-24 13:15:00 | 1890.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-24 12:45:00 | 1876.10 | 2025-09-24 13:15:00 | 1890.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-24 14:30:00 | 1874.40 | 2025-09-26 14:15:00 | 1780.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:30:00 | 1874.40 | 2025-10-06 09:15:00 | 1887.80 | STOP_HIT | 0.50 | -0.71% |
| SELL | retest2 | 2025-10-08 09:15:00 | 1864.70 | 2025-10-10 15:15:00 | 1889.00 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2025-10-08 10:00:00 | 1869.50 | 2025-10-10 15:15:00 | 1889.00 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-10-08 11:00:00 | 1864.20 | 2025-10-10 15:15:00 | 1889.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-10-13 09:15:00 | 1863.00 | 2025-10-13 14:15:00 | 1886.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-10-13 10:30:00 | 1851.10 | 2025-10-13 15:15:00 | 1895.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest1 | 2025-11-06 09:15:00 | 1961.50 | 2025-11-06 11:15:00 | 1917.50 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-12-03 14:15:00 | 1886.90 | 2025-12-16 09:15:00 | 1918.10 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-12-03 15:00:00 | 1871.30 | 2025-12-16 09:15:00 | 1918.10 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-12-09 15:15:00 | 1885.50 | 2025-12-16 09:15:00 | 1918.10 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1887.90 | 2025-12-16 09:15:00 | 1918.10 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-12-10 15:15:00 | 1854.00 | 2025-12-22 09:15:00 | 1924.80 | STOP_HIT | 1.00 | -3.82% |
| SELL | retest2 | 2025-12-11 09:30:00 | 1861.20 | 2025-12-22 09:15:00 | 1924.80 | STOP_HIT | 1.00 | -3.42% |
| SELL | retest2 | 2025-12-15 09:15:00 | 1853.30 | 2025-12-22 09:15:00 | 1924.80 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-12-16 09:15:00 | 1860.90 | 2026-01-07 09:15:00 | 1887.50 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-12-16 11:15:00 | 1896.00 | 2026-01-07 10:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-18 15:15:00 | 1890.00 | 2026-01-07 10:15:00 | 1910.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-12-19 11:00:00 | 1896.30 | 2026-01-07 10:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-23 14:45:00 | 1893.40 | 2026-01-07 10:15:00 | 1910.00 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-12-24 11:00:00 | 1882.90 | 2026-01-12 09:15:00 | 1792.56 | PARTIAL | 0.50 | 4.80% |
| SELL | retest2 | 2025-12-30 15:00:00 | 1873.30 | 2026-01-12 09:15:00 | 1791.22 | PARTIAL | 0.50 | 4.38% |
| SELL | retest2 | 2025-12-31 09:30:00 | 1882.10 | 2026-01-12 09:15:00 | 1793.51 | PARTIAL | 0.50 | 4.71% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1881.50 | 2026-01-12 09:15:00 | 1798.73 | PARTIAL | 0.50 | 4.40% |
| SELL | retest2 | 2026-01-01 11:00:00 | 1864.60 | 2026-01-12 09:15:00 | 1788.76 | PARTIAL | 0.50 | 4.07% |
| SELL | retest2 | 2026-01-01 15:15:00 | 1867.90 | 2026-01-12 09:15:00 | 1787.99 | PARTIAL | 0.50 | 4.28% |
| SELL | retest2 | 2026-01-05 11:45:00 | 1870.90 | 2026-01-12 09:15:00 | 1787.42 | PARTIAL | 0.50 | 4.46% |
| SELL | retest2 | 2026-01-05 12:30:00 | 1872.00 | 2026-01-12 11:15:00 | 1777.73 | PARTIAL | 0.50 | 5.04% |
| SELL | retest2 | 2026-01-06 11:15:00 | 1867.90 | 2026-01-12 11:15:00 | 1779.63 | PARTIAL | 0.50 | 4.73% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1863.40 | 2026-01-20 11:15:00 | 1770.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-24 11:00:00 | 1882.90 | 2026-01-20 14:15:00 | 1698.21 | TARGET_HIT | 0.50 | 9.81% |
| SELL | retest2 | 2025-12-30 15:00:00 | 1873.30 | 2026-01-20 14:15:00 | 1684.17 | TARGET_HIT | 0.50 | 10.10% |
| SELL | retest2 | 2025-12-31 09:30:00 | 1882.10 | 2026-01-20 14:15:00 | 1696.95 | TARGET_HIT | 0.50 | 9.84% |
| SELL | retest2 | 2026-01-01 09:15:00 | 1881.50 | 2026-01-20 14:15:00 | 1699.11 | TARGET_HIT | 0.50 | 9.69% |
| SELL | retest2 | 2026-01-01 11:00:00 | 1864.60 | 2026-01-20 14:15:00 | 1704.06 | TARGET_HIT | 0.50 | 8.61% |
| SELL | retest2 | 2026-01-01 15:15:00 | 1867.90 | 2026-01-20 14:15:00 | 1694.61 | TARGET_HIT | 0.50 | 9.28% |
| SELL | retest2 | 2026-01-05 11:45:00 | 1870.90 | 2026-01-20 14:15:00 | 1685.97 | TARGET_HIT | 0.50 | 9.88% |
| SELL | retest2 | 2026-01-05 12:30:00 | 1872.00 | 2026-01-20 14:15:00 | 1693.89 | TARGET_HIT | 0.50 | 9.51% |
| SELL | retest2 | 2026-01-06 11:15:00 | 1867.90 | 2026-01-20 14:15:00 | 1693.35 | TARGET_HIT | 0.50 | 9.34% |
| SELL | retest2 | 2026-01-08 11:15:00 | 1863.40 | 2026-02-01 12:15:00 | 1677.06 | TARGET_HIT | 0.50 | 10.00% |
