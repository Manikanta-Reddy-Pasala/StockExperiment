# Adani Green Energy Ltd. (ADANIGREEN)

## Backtest Summary

- **Window:** 2022-04-07 13:15:00 → 2026-05-08 15:15:00 (7050 bars)
- **Last close:** 1350.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 15 |
| ALERT1 | 11 |
| ALERT2 | 10 |
| ALERT2_SKIP | 4 |
| ALERT3 | 70 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 81 |
| PARTIAL | 12 |
| TARGET_HIT | 10 |
| STOP_HIT | 71 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 93 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 24 / 69
- **Target hits / Stop hits / Partials:** 10 / 71 / 12
- **Avg / median % per leg:** -0.26% / -1.05%
- **Sum % (uncompounded):** -23.74%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 3 | 6.1% | 3 | 46 | 0 | -1.82% | -89.2% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 49 | 3 | 6.1% | 3 | 46 | 0 | -1.82% | -89.2% |
| SELL (all) | 44 | 21 | 47.7% | 7 | 25 | 12 | 1.49% | 65.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 44 | 21 | 47.7% | 7 | 25 | 12 | 1.49% | 65.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 93 | 24 | 25.8% | 10 | 71 | 12 | -0.26% | -23.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 10:15:00 | 995.85 | 963.68 | 963.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 09:15:00 | 1034.00 | 965.80 | 964.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 13:15:00 | 1018.00 | 1018.49 | 995.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-04 14:00:00 | 1018.00 | 1018.49 | 995.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 09:15:00 | 971.20 | 1017.91 | 995.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 09:15:00 | 1029.75 | 994.51 | 988.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 11:45:00 | 1019.00 | 995.17 | 988.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 12:15:00 | 1020.00 | 995.17 | 988.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-21 15:15:00 | 1020.00 | 995.93 | 989.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 992.90 | 997.86 | 990.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:00:00 | 992.90 | 997.86 | 990.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 14:15:00 | 973.05 | 997.61 | 990.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 15:00:00 | 973.05 | 997.61 | 990.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 15:15:00 | 977.00 | 997.41 | 990.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-24 09:15:00 | 992.75 | 997.41 | 990.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 10:30:00 | 980.45 | 996.44 | 990.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 11:15:00 | 978.50 | 996.44 | 990.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-25 13:30:00 | 977.15 | 995.85 | 990.18 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 972.25 | 995.21 | 989.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:30:00 | 970.10 | 995.21 | 989.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-08-29 10:15:00 | 969.05 | 993.91 | 989.49 | SL hit (close<static) qty=1.00 sl=970.80 alert=retest2 |

### Cycle 2 — SELL (started 2023-09-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-01 12:15:00 | 944.05 | 985.44 | 985.55 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-11 14:15:00 | 1012.95 | 985.29 | 985.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-11 15:15:00 | 1020.00 | 985.63 | 985.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-13 09:15:00 | 972.85 | 986.53 | 985.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 09:15:00 | 972.85 | 986.53 | 985.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 09:15:00 | 972.85 | 986.53 | 985.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-13 10:00:00 | 972.85 | 986.53 | 985.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 10:15:00 | 985.00 | 986.52 | 985.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 09:15:00 | 992.60 | 986.40 | 985.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 10:15:00 | 989.70 | 986.40 | 985.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-14 10:45:00 | 988.95 | 986.41 | 985.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-15 09:15:00 | 988.15 | 986.25 | 985.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 11:15:00 | 990.50 | 997.58 | 992.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-29 12:00:00 | 990.50 | 997.58 | 992.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-29 12:15:00 | 991.50 | 997.52 | 992.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 14:00:00 | 992.45 | 997.47 | 992.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-29 14:15:00 | 987.25 | 997.36 | 992.37 | SL hit (close<static) qty=1.00 sl=988.25 alert=retest2 |

### Cycle 4 — SELL (started 2023-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 14:15:00 | 939.50 | 987.96 | 988.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-09 15:15:00 | 937.00 | 987.45 | 987.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-07 09:15:00 | 938.05 | 931.05 | 951.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-07 09:15:00 | 938.05 | 931.05 | 951.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 938.05 | 931.05 | 951.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-07 12:15:00 | 925.95 | 931.04 | 951.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-20 09:15:00 | 924.00 | 934.61 | 948.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 13:00:00 | 926.35 | 933.56 | 947.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-21 13:45:00 | 926.35 | 933.49 | 947.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 940.45 | 931.28 | 944.88 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-28 09:15:00 | 1006.50 | 932.23 | 945.09 | SL hit (close>static) qty=1.00 sl=953.25 alert=retest2 |

### Cycle 5 — BUY (started 2023-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 14:15:00 | 1025.00 | 956.84 | 956.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 1111.35 | 959.01 | 957.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-13 09:15:00 | 1801.20 | 1867.64 | 1736.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-13 09:15:00 | 1801.20 | 1867.64 | 1736.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 1801.20 | 1867.64 | 1736.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 10:00:00 | 1801.20 | 1867.64 | 1736.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 11:15:00 | 1731.70 | 1865.33 | 1737.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 11:45:00 | 1731.35 | 1865.33 | 1737.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 12:15:00 | 1777.10 | 1864.45 | 1737.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-14 09:15:00 | 1804.35 | 1860.50 | 1737.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 09:15:00 | 1800.95 | 1859.56 | 1806.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 12:15:00 | 1796.95 | 1857.53 | 1806.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-22 15:15:00 | 1796.70 | 1855.50 | 1806.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 14:15:00 | 1809.70 | 1850.27 | 1806.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-24 14:45:00 | 1801.75 | 1850.27 | 1806.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 15:15:00 | 1810.00 | 1849.87 | 1806.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:15:00 | 1809.00 | 1849.87 | 1806.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 1805.05 | 1849.42 | 1806.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 09:45:00 | 1800.00 | 1849.42 | 1806.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 1800.90 | 1848.94 | 1806.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 11:00:00 | 1800.90 | 1848.94 | 1806.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 11:15:00 | 1795.70 | 1848.41 | 1806.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 12:00:00 | 1795.70 | 1848.41 | 1806.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 09:15:00 | 1800.75 | 1846.24 | 1806.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-26 10:00:00 | 1800.75 | 1846.24 | 1806.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-26 10:15:00 | 1800.50 | 1845.78 | 1806.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 14:00:00 | 1804.90 | 1844.51 | 1806.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-26 15:00:00 | 1807.50 | 1844.14 | 1806.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 12:30:00 | 1805.05 | 1842.13 | 1806.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-29 15:00:00 | 1808.70 | 1841.40 | 1806.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-29 15:15:00 | 1806.10 | 1841.05 | 1806.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-30 09:30:00 | 1828.20 | 1840.82 | 1806.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-30 14:15:00 | 1796.15 | 1839.29 | 1806.47 | SL hit (close<static) qty=1.00 sl=1797.10 alert=retest2 |

### Cycle 6 — SELL (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 09:15:00 | 1781.20 | 1816.84 | 1816.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-03 10:15:00 | 1771.00 | 1813.80 | 1815.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 1788.00 | 1774.66 | 1791.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-23 11:00:00 | 1788.00 | 1774.66 | 1791.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 1770.10 | 1774.62 | 1791.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:45:00 | 1765.00 | 1774.62 | 1791.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 1821.15 | 1768.55 | 1786.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 15:00:00 | 1821.15 | 1768.55 | 1786.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 1848.00 | 1769.34 | 1786.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 1841.05 | 1769.34 | 1786.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 1828.00 | 1773.68 | 1788.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 09:45:00 | 1827.00 | 1773.68 | 1788.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 1794.25 | 1799.76 | 1800.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 11:45:00 | 1788.60 | 1799.72 | 1800.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 12:30:00 | 1782.50 | 1799.53 | 1800.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 14:00:00 | 1788.95 | 1799.42 | 1799.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 10:15:00 | 1786.75 | 1798.81 | 1799.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 14:15:00 | 1775.10 | 1794.28 | 1797.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:45:00 | 1783.95 | 1794.28 | 1797.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 1788.00 | 1794.09 | 1797.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 10:15:00 | 1780.30 | 1794.09 | 1797.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:00:00 | 1781.05 | 1793.96 | 1797.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 11:45:00 | 1780.00 | 1793.83 | 1796.93 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 13:00:00 | 1780.00 | 1793.69 | 1796.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1699.17 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1693.38 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1699.50 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1697.41 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1691.28 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1692.00 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1691.00 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-12 09:15:00 | 1691.00 | 1792.52 | 1796.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 1786.70 | 1791.42 | 1795.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:30:00 | 1800.75 | 1791.42 | 1795.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 1798.65 | 1791.49 | 1795.58 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-08-12 14:15:00 | 1798.65 | 1791.49 | 1795.58 | SL hit (close>ema200) qty=0.50 sl=1791.49 alert=retest2 |

### Cycle 7 — BUY (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 13:15:00 | 1873.50 | 1799.57 | 1799.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 14:15:00 | 1901.20 | 1800.59 | 1799.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 10:15:00 | 1834.25 | 1836.68 | 1820.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 11:00:00 | 1834.25 | 1836.68 | 1820.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 1819.10 | 1836.50 | 1820.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 1819.10 | 1836.50 | 1820.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 1821.00 | 1836.35 | 1820.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 15:00:00 | 1829.85 | 1836.08 | 1820.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 10:00:00 | 1831.00 | 1836.00 | 1820.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 11:00:00 | 1834.10 | 1835.98 | 1820.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 11:30:00 | 1828.10 | 1835.93 | 1820.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1872.85 | 1854.97 | 1834.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 1851.00 | 1854.97 | 1834.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 1841.00 | 1855.96 | 1836.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 10:00:00 | 1841.00 | 1855.96 | 1836.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 1812.50 | 1855.49 | 1836.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-11 14:15:00 | 1812.50 | 1855.49 | 1836.28 | SL hit (close<static) qty=1.00 sl=1817.65 alert=retest2 |

### Cycle 8 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 1746.60 | 1857.41 | 1857.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 1742.55 | 1852.12 | 1854.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 888.45 | 882.61 | 978.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 09:30:00 | 884.60 | 882.61 | 978.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 966.80 | 889.74 | 966.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 10:45:00 | 982.55 | 889.74 | 966.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 964.70 | 890.48 | 966.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 13:45:00 | 950.40 | 891.83 | 966.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-24 11:00:00 | 951.00 | 894.33 | 966.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 09:15:00 | 948.20 | 897.34 | 966.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 10:00:00 | 950.30 | 897.87 | 966.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 958.50 | 903.61 | 963.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 958.50 | 903.61 | 963.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 954.95 | 904.13 | 962.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 954.90 | 904.13 | 962.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 966.40 | 904.75 | 962.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 966.40 | 904.75 | 962.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 962.70 | 905.32 | 962.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 11:15:00 | 961.50 | 905.32 | 962.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 12:15:00 | 957.45 | 905.90 | 962.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 09:15:00 | 913.42 | 909.21 | 961.39 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-02 09:15:00 | 916.35 | 909.21 | 961.39 | SL hit (close>static) qty=0.50 sl=909.21 alert=retest2 |

### Cycle 9 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 1002.25 | 953.13 | 952.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 10:15:00 | 1017.90 | 953.77 | 953.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 12:15:00 | 989.00 | 995.99 | 979.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 13:00:00 | 989.00 | 995.99 | 979.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 977.50 | 995.67 | 979.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:45:00 | 974.60 | 995.67 | 979.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 989.60 | 995.61 | 979.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:00:00 | 996.90 | 995.53 | 979.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 15:15:00 | 977.50 | 994.64 | 979.95 | SL hit (close<static) qty=1.00 sl=977.90 alert=retest2 |

### Cycle 10 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 910.05 | 991.45 | 991.65 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 13:15:00 | 1148.65 | 975.00 | 974.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 14:15:00 | 1158.00 | 976.82 | 975.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 11:15:00 | 1027.80 | 1030.03 | 1010.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 12:00:00 | 1027.80 | 1030.03 | 1010.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1020.80 | 1034.45 | 1017.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 1019.70 | 1034.45 | 1017.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 1017.20 | 1034.02 | 1017.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 1017.20 | 1034.02 | 1017.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 1017.90 | 1033.86 | 1017.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:45:00 | 1015.70 | 1033.86 | 1017.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 15:15:00 | 1017.00 | 1033.69 | 1017.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 1020.50 | 1033.69 | 1017.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 1015.40 | 1033.51 | 1017.10 | SL hit (close<static) qty=1.00 sl=1015.80 alert=retest2 |

### Cycle 12 — SELL (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 10:15:00 | 1000.30 | 1034.12 | 1034.12 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 1050.20 | 1034.11 | 1034.09 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 1024.00 | 1034.03 | 1034.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1021.60 | 1033.91 | 1034.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1033.20 | 1024.53 | 1028.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:45:00 | 1034.70 | 1024.53 | 1028.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 1027.40 | 1024.56 | 1028.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:15:00 | 1022.70 | 1025.96 | 1028.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-09 09:15:00 | 971.57 | 1022.88 | 1027.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-19 10:15:00 | 920.43 | 997.37 | 1012.49 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 15 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 1095.35 | 932.66 | 932.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 1105.80 | 937.57 | 934.57 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-08-21 09:15:00 | 1029.75 | 2023-08-29 10:15:00 | 969.05 | STOP_HIT | 1.00 | -5.89% |
| BUY | retest2 | 2023-08-21 11:45:00 | 1019.00 | 2023-08-29 10:15:00 | 969.05 | STOP_HIT | 1.00 | -4.90% |
| BUY | retest2 | 2023-08-21 12:15:00 | 1020.00 | 2023-08-29 10:15:00 | 969.05 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2023-08-21 15:15:00 | 1020.00 | 2023-08-29 10:15:00 | 969.05 | STOP_HIT | 1.00 | -5.00% |
| BUY | retest2 | 2023-08-24 09:15:00 | 992.75 | 2023-09-01 12:15:00 | 944.05 | STOP_HIT | 1.00 | -4.91% |
| BUY | retest2 | 2023-08-25 10:30:00 | 980.45 | 2023-09-01 12:15:00 | 944.05 | STOP_HIT | 1.00 | -3.71% |
| BUY | retest2 | 2023-08-25 11:15:00 | 978.50 | 2023-09-01 12:15:00 | 944.05 | STOP_HIT | 1.00 | -3.52% |
| BUY | retest2 | 2023-08-25 13:30:00 | 977.15 | 2023-09-01 12:15:00 | 944.05 | STOP_HIT | 1.00 | -3.39% |
| BUY | retest2 | 2023-09-14 09:15:00 | 992.60 | 2023-09-29 14:15:00 | 987.25 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-09-14 10:15:00 | 989.70 | 2023-10-05 14:15:00 | 961.35 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2023-09-14 10:45:00 | 988.95 | 2023-10-05 14:15:00 | 961.35 | STOP_HIT | 1.00 | -2.79% |
| BUY | retest2 | 2023-09-15 09:15:00 | 988.15 | 2023-10-05 14:15:00 | 961.35 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2023-09-29 14:00:00 | 992.45 | 2023-10-05 14:15:00 | 961.35 | STOP_HIT | 1.00 | -3.13% |
| SELL | retest2 | 2023-11-07 12:15:00 | 925.95 | 2023-11-28 09:15:00 | 1006.50 | STOP_HIT | 1.00 | -8.70% |
| SELL | retest2 | 2023-11-20 09:15:00 | 924.00 | 2023-11-28 09:15:00 | 1006.50 | STOP_HIT | 1.00 | -8.93% |
| SELL | retest2 | 2023-11-21 13:00:00 | 926.35 | 2023-11-28 09:15:00 | 1006.50 | STOP_HIT | 1.00 | -8.65% |
| SELL | retest2 | 2023-11-21 13:45:00 | 926.35 | 2023-11-28 09:15:00 | 1006.50 | STOP_HIT | 1.00 | -8.65% |
| BUY | retest2 | 2024-03-14 09:15:00 | 1804.35 | 2024-04-30 14:15:00 | 1796.15 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-04-22 09:15:00 | 1800.95 | 2024-04-30 14:15:00 | 1796.15 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2024-04-22 12:15:00 | 1796.95 | 2024-04-30 14:15:00 | 1796.15 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-04-22 15:15:00 | 1796.70 | 2024-04-30 14:15:00 | 1796.15 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2024-04-26 14:00:00 | 1804.90 | 2024-04-30 14:15:00 | 1796.15 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-04-26 15:00:00 | 1807.50 | 2024-05-09 14:15:00 | 1708.95 | STOP_HIT | 1.00 | -5.45% |
| BUY | retest2 | 2024-04-29 12:30:00 | 1805.05 | 2024-05-09 14:15:00 | 1708.95 | STOP_HIT | 1.00 | -5.32% |
| BUY | retest2 | 2024-04-29 15:00:00 | 1808.70 | 2024-05-09 14:15:00 | 1708.95 | STOP_HIT | 1.00 | -5.52% |
| BUY | retest2 | 2024-04-30 09:30:00 | 1828.20 | 2024-05-09 14:15:00 | 1708.95 | STOP_HIT | 1.00 | -6.52% |
| BUY | retest2 | 2024-05-15 09:45:00 | 1818.50 | 2024-06-03 09:15:00 | 2000.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-04 09:30:00 | 1882.15 | 2024-06-04 10:15:00 | 1687.65 | STOP_HIT | 1.00 | -10.33% |
| BUY | retest2 | 2024-06-05 13:30:00 | 1816.85 | 2024-06-13 14:15:00 | 1798.45 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-06-06 09:15:00 | 1888.55 | 2024-07-02 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -5.68% |
| BUY | retest2 | 2024-06-13 09:15:00 | 1833.85 | 2024-07-02 09:15:00 | 1781.20 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2024-08-05 11:45:00 | 1788.60 | 2024-08-12 09:15:00 | 1699.17 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-05 12:30:00 | 1782.50 | 2024-08-12 09:15:00 | 1693.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-05 14:00:00 | 1788.95 | 2024-08-12 09:15:00 | 1699.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-06 10:15:00 | 1786.75 | 2024-08-12 09:15:00 | 1697.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 10:15:00 | 1780.30 | 2024-08-12 09:15:00 | 1691.28 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 11:00:00 | 1781.05 | 2024-08-12 09:15:00 | 1692.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 11:45:00 | 1780.00 | 2024-08-12 09:15:00 | 1691.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-09 13:00:00 | 1780.00 | 2024-08-12 09:15:00 | 1691.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-05 11:45:00 | 1788.60 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -0.56% |
| SELL | retest2 | 2024-08-05 12:30:00 | 1782.50 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -0.91% |
| SELL | retest2 | 2024-08-05 14:00:00 | 1788.95 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -0.54% |
| SELL | retest2 | 2024-08-06 10:15:00 | 1786.75 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -0.67% |
| SELL | retest2 | 2024-08-09 10:15:00 | 1780.30 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -1.03% |
| SELL | retest2 | 2024-08-09 11:00:00 | 1781.05 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -0.99% |
| SELL | retest2 | 2024-08-09 11:45:00 | 1780.00 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -1.05% |
| SELL | retest2 | 2024-08-09 13:00:00 | 1780.00 | 2024-08-12 14:15:00 | 1798.65 | STOP_HIT | 0.50 | -1.05% |
| SELL | retest2 | 2024-08-13 15:15:00 | 1810.00 | 2024-08-19 09:15:00 | 1871.05 | STOP_HIT | 1.00 | -3.37% |
| SELL | retest2 | 2024-08-14 09:45:00 | 1808.95 | 2024-08-19 09:15:00 | 1871.05 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-08-16 09:30:00 | 1810.45 | 2024-08-19 09:15:00 | 1871.05 | STOP_HIT | 1.00 | -3.35% |
| BUY | retest2 | 2024-08-29 15:00:00 | 1829.85 | 2024-09-11 14:15:00 | 1812.50 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-08-30 10:00:00 | 1831.00 | 2024-09-11 14:15:00 | 1812.50 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2024-08-30 11:00:00 | 1834.10 | 2024-09-11 14:15:00 | 1812.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-08-30 11:30:00 | 1828.10 | 2024-09-11 14:15:00 | 1812.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-09-12 09:15:00 | 1828.90 | 2024-09-12 14:15:00 | 1811.05 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-09-16 09:15:00 | 1860.35 | 2024-09-24 09:15:00 | 2046.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-21 13:45:00 | 950.40 | 2025-04-02 09:15:00 | 913.42 | PARTIAL | 0.50 | 3.89% |
| SELL | retest2 | 2025-03-21 13:45:00 | 950.40 | 2025-04-02 09:15:00 | 916.35 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2025-03-24 11:00:00 | 951.00 | 2025-04-07 09:15:00 | 855.36 | TARGET_HIT | 1.00 | 10.06% |
| SELL | retest2 | 2025-03-25 09:15:00 | 948.20 | 2025-04-07 09:15:00 | 855.90 | TARGET_HIT | 1.00 | 9.73% |
| SELL | retest2 | 2025-03-25 10:00:00 | 950.30 | 2025-04-07 09:15:00 | 853.38 | TARGET_HIT | 1.00 | 10.20% |
| SELL | retest2 | 2025-03-28 11:15:00 | 961.50 | 2025-04-07 09:15:00 | 855.27 | TARGET_HIT | 1.00 | 11.05% |
| SELL | retest2 | 2025-03-28 12:15:00 | 957.45 | 2025-04-07 09:15:00 | 861.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-03 12:30:00 | 959.50 | 2025-04-07 09:15:00 | 863.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-17 10:45:00 | 956.40 | 2025-04-23 14:15:00 | 953.15 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-04-23 10:00:00 | 930.65 | 2025-04-25 10:15:00 | 908.58 | PARTIAL | 0.50 | 2.37% |
| SELL | retest2 | 2025-04-23 10:00:00 | 930.65 | 2025-04-28 09:15:00 | 932.00 | STOP_HIT | 0.50 | -0.15% |
| SELL | retest2 | 2025-04-25 09:45:00 | 925.00 | 2025-05-05 10:15:00 | 957.25 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2025-04-28 12:45:00 | 938.10 | 2025-05-05 10:15:00 | 957.25 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-04-29 09:30:00 | 937.75 | 2025-05-05 10:15:00 | 957.25 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-05-06 11:15:00 | 934.45 | 2025-05-08 14:15:00 | 887.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 11:15:00 | 934.45 | 2025-05-12 09:15:00 | 934.95 | STOP_HIT | 0.50 | -0.05% |
| SELL | retest2 | 2025-05-12 10:00:00 | 934.95 | 2025-05-13 09:15:00 | 956.80 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2025-05-12 11:15:00 | 936.15 | 2025-05-13 09:15:00 | 956.80 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-06-16 14:00:00 | 996.90 | 2025-06-17 15:15:00 | 977.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-06-24 11:30:00 | 998.40 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-06-27 09:15:00 | 1003.90 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2025-07-04 14:45:00 | 997.30 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -1.98% |
| BUY | retest2 | 2025-07-08 13:15:00 | 987.40 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-07-08 14:00:00 | 988.50 | 2025-07-25 14:15:00 | 977.60 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-07-28 09:15:00 | 989.50 | 2025-08-01 14:15:00 | 971.70 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-08-01 09:15:00 | 994.80 | 2025-08-01 14:15:00 | 971.70 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-10-28 09:15:00 | 1020.50 | 2025-10-28 09:15:00 | 1015.40 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2025-10-29 09:15:00 | 1025.90 | 2025-10-29 11:15:00 | 1128.49 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-24 10:00:00 | 1026.00 | 2025-11-24 13:15:00 | 1014.70 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-11-24 12:00:00 | 1020.00 | 2025-11-24 13:15:00 | 1014.70 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-11-27 13:30:00 | 1033.20 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-11-27 15:15:00 | 1033.00 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-11-28 09:30:00 | 1035.20 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-12-01 12:45:00 | 1036.80 | 2025-12-02 13:15:00 | 1021.70 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-01-06 13:15:00 | 1022.70 | 2026-01-09 09:15:00 | 971.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:15:00 | 1022.70 | 2026-01-19 10:15:00 | 920.43 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-08 11:30:00 | 1019.80 | 2026-04-08 15:15:00 | 1035.00 | STOP_HIT | 1.00 | -1.49% |
