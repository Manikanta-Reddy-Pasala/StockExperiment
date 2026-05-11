# Sun Pharmaceutical Industries Ltd. (SUNPHARMA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 1845.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 203 |
| ALERT1 | 145 |
| ALERT2 | 145 |
| ALERT2_SKIP | 85 |
| ALERT3 | 426 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 211 |
| PARTIAL | 11 |
| TARGET_HIT | 6 |
| STOP_HIT | 205 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 222 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 79 / 143
- **Target hits / Stop hits / Partials:** 6 / 205 / 11
- **Avg / median % per leg:** 0.66% / -0.44%
- **Sum % (uncompounded):** 145.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 125 | 52 | 41.6% | 6 | 119 | 0 | 1.09% | 136.8% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 4 | 0 | 0.03% | 0.1% |
| BUY @ 3rd Alert (retest2) | 121 | 49 | 40.5% | 6 | 115 | 0 | 1.13% | 136.7% |
| SELL (all) | 97 | 27 | 27.8% | 0 | 86 | 11 | 0.09% | 9.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 97 | 27 | 27.8% | 0 | 86 | 11 | 0.09% | 9.0% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 4 | 0 | 0.03% | 0.1% |
| retest2 (combined) | 218 | 76 | 34.9% | 6 | 201 | 11 | 0.67% | 145.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 12:15:00 | 938.80 | 934.38 | 934.09 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 15:15:00 | 930.15 | 934.60 | 935.00 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2023-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-24 10:15:00 | 941.95 | 936.12 | 935.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-24 11:15:00 | 951.00 | 939.09 | 937.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-25 10:15:00 | 945.50 | 945.89 | 942.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-25 10:45:00 | 944.95 | 945.89 | 942.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-26 09:15:00 | 945.25 | 945.07 | 943.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 10:15:00 | 947.20 | 945.07 | 943.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 14:15:00 | 948.00 | 945.19 | 943.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-26 14:45:00 | 959.80 | 951.23 | 946.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 12:15:00 | 995.05 | 1004.68 | 1005.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2023-06-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 12:15:00 | 995.05 | 1004.68 | 1005.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-08 13:15:00 | 991.50 | 1002.04 | 1004.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 15:15:00 | 988.00 | 986.64 | 990.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-13 09:15:00 | 988.50 | 986.64 | 990.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 09:15:00 | 993.10 | 987.93 | 990.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:00:00 | 993.10 | 987.93 | 990.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 10:15:00 | 992.05 | 988.75 | 990.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 10:30:00 | 989.75 | 988.75 | 990.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 11:15:00 | 993.85 | 989.77 | 990.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 12:00:00 | 993.85 | 989.77 | 990.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 13:15:00 | 991.20 | 990.00 | 990.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 14:00:00 | 991.20 | 990.00 | 990.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 14:15:00 | 988.30 | 989.66 | 990.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-13 14:45:00 | 991.65 | 989.66 | 990.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 09:15:00 | 993.45 | 990.15 | 990.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-14 10:00:00 | 993.45 | 990.15 | 990.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 10:15:00 | 989.10 | 989.94 | 990.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 13:15:00 | 987.80 | 990.26 | 990.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-14 14:15:00 | 988.80 | 990.09 | 990.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 10:15:00 | 988.40 | 989.99 | 990.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 11:30:00 | 988.45 | 989.68 | 990.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 12:15:00 | 990.35 | 989.81 | 990.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-15 12:45:00 | 990.80 | 989.81 | 990.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-15 13:15:00 | 989.00 | 989.65 | 989.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-15 15:15:00 | 987.05 | 989.41 | 989.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-16 14:15:00 | 992.55 | 990.31 | 990.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2023-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 14:15:00 | 992.55 | 990.31 | 990.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-16 15:15:00 | 994.50 | 991.15 | 990.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-20 09:15:00 | 994.20 | 998.42 | 995.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-20 09:15:00 | 994.20 | 998.42 | 995.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 09:15:00 | 994.20 | 998.42 | 995.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:00:00 | 994.20 | 998.42 | 995.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 992.00 | 997.13 | 995.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:30:00 | 990.00 | 997.13 | 995.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 13:15:00 | 993.65 | 995.41 | 994.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 13:30:00 | 993.20 | 995.41 | 994.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2023-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-20 15:15:00 | 992.60 | 994.52 | 994.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-21 09:15:00 | 986.05 | 992.82 | 993.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 13:15:00 | 992.85 | 992.30 | 993.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 13:15:00 | 992.85 | 992.30 | 993.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 13:15:00 | 992.85 | 992.30 | 993.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-21 14:45:00 | 990.15 | 992.25 | 993.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 09:30:00 | 991.00 | 992.41 | 993.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 10:15:00 | 990.15 | 992.41 | 993.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 11:00:00 | 991.05 | 992.14 | 992.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 11:15:00 | 992.00 | 992.11 | 992.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 11:30:00 | 992.75 | 992.11 | 992.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 12:15:00 | 991.25 | 991.94 | 992.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 12:30:00 | 992.60 | 991.94 | 992.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 13:15:00 | 992.55 | 992.06 | 992.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 13:30:00 | 992.90 | 992.06 | 992.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 14:15:00 | 990.20 | 991.69 | 992.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 14:30:00 | 993.00 | 991.69 | 992.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 978.05 | 988.91 | 991.01 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-06-26 09:15:00 | 995.35 | 990.52 | 990.61 | SL hit (close>static) qty=1.00 sl=993.40 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 10:15:00 | 995.00 | 991.42 | 991.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 11:15:00 | 1002.00 | 993.53 | 992.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 09:15:00 | 990.15 | 994.45 | 993.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 990.15 | 994.45 | 993.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 990.15 | 994.45 | 993.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:45:00 | 991.35 | 994.45 | 993.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 988.35 | 993.23 | 992.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:00:00 | 988.35 | 993.23 | 992.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-28 09:15:00 | 1001.10 | 998.76 | 996.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-28 11:00:00 | 1008.70 | 1000.75 | 997.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-28 12:15:00 | 1006.15 | 1001.47 | 997.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 12:15:00 | 1036.50 | 1043.77 | 1043.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 1036.50 | 1043.77 | 1043.99 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 13:15:00 | 1045.90 | 1042.90 | 1042.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-10 15:15:00 | 1049.00 | 1044.25 | 1043.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-13 13:15:00 | 1072.65 | 1075.78 | 1070.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-13 14:00:00 | 1072.65 | 1075.78 | 1070.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 1070.05 | 1074.57 | 1070.98 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2023-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 12:15:00 | 1067.90 | 1071.88 | 1072.25 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2023-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-19 10:15:00 | 1074.30 | 1072.32 | 1072.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-19 15:15:00 | 1085.50 | 1077.25 | 1074.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 11:15:00 | 1093.20 | 1094.18 | 1087.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-21 12:00:00 | 1093.20 | 1094.18 | 1087.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 10:15:00 | 1100.50 | 1100.94 | 1097.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 10:45:00 | 1099.45 | 1100.94 | 1097.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 11:15:00 | 1099.90 | 1100.73 | 1097.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-25 11:45:00 | 1099.15 | 1100.73 | 1097.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 12:15:00 | 1099.40 | 1100.47 | 1097.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-26 09:30:00 | 1106.20 | 1100.96 | 1098.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 13:15:00 | 1128.80 | 1135.65 | 1136.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 13:15:00 | 1128.80 | 1135.65 | 1136.50 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-08-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-03 09:15:00 | 1166.40 | 1142.34 | 1139.35 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2023-08-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-04 11:15:00 | 1128.50 | 1139.47 | 1140.92 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 1159.00 | 1141.42 | 1140.87 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-08-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 12:15:00 | 1145.00 | 1147.57 | 1147.76 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 14:15:00 | 1152.40 | 1148.71 | 1148.26 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-11 09:15:00 | 1137.70 | 1147.48 | 1148.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-11 10:15:00 | 1131.80 | 1144.35 | 1146.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-16 10:15:00 | 1135.40 | 1134.65 | 1137.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-16 10:15:00 | 1135.40 | 1134.65 | 1137.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 10:15:00 | 1135.40 | 1134.65 | 1137.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 10:30:00 | 1137.80 | 1134.65 | 1137.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 11:15:00 | 1140.85 | 1135.89 | 1137.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 11:30:00 | 1140.30 | 1135.89 | 1137.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-16 12:15:00 | 1141.65 | 1137.04 | 1138.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-16 12:45:00 | 1142.75 | 1137.04 | 1138.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 15:15:00 | 1145.00 | 1139.83 | 1139.27 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 10:15:00 | 1135.30 | 1140.00 | 1140.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 15:15:00 | 1129.05 | 1136.53 | 1138.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 10:15:00 | 1141.65 | 1137.23 | 1138.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-21 10:15:00 | 1141.65 | 1137.23 | 1138.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 10:15:00 | 1141.65 | 1137.23 | 1138.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 10:45:00 | 1141.90 | 1137.23 | 1138.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 11:15:00 | 1140.70 | 1137.93 | 1138.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 11:45:00 | 1143.00 | 1137.93 | 1138.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 1137.30 | 1137.19 | 1137.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 15:00:00 | 1137.30 | 1137.19 | 1137.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 1136.10 | 1136.97 | 1137.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 1137.00 | 1136.97 | 1137.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-22 09:15:00 | 1131.85 | 1135.95 | 1137.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-23 09:15:00 | 1120.35 | 1136.11 | 1136.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-05 10:15:00 | 1116.60 | 1111.11 | 1110.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2023-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 10:15:00 | 1116.60 | 1111.11 | 1110.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 13:15:00 | 1134.55 | 1117.81 | 1113.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-07 09:15:00 | 1133.35 | 1136.40 | 1129.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-07 10:00:00 | 1133.35 | 1136.40 | 1129.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 13:15:00 | 1129.65 | 1133.65 | 1129.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 13:45:00 | 1130.40 | 1133.65 | 1129.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 14:15:00 | 1132.05 | 1133.33 | 1130.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-08 12:15:00 | 1135.80 | 1132.91 | 1130.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:15:00 | 1136.65 | 1132.03 | 1131.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 15:00:00 | 1136.90 | 1132.89 | 1131.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-21 11:15:00 | 1144.85 | 1148.23 | 1148.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2023-09-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-21 11:15:00 | 1144.85 | 1148.23 | 1148.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-22 09:15:00 | 1131.90 | 1144.34 | 1146.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-26 12:15:00 | 1125.40 | 1125.18 | 1130.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-26 13:00:00 | 1125.40 | 1125.18 | 1130.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 1125.25 | 1125.69 | 1129.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-26 14:30:00 | 1128.55 | 1125.69 | 1129.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 09:15:00 | 1132.75 | 1127.33 | 1129.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 09:30:00 | 1134.20 | 1127.33 | 1129.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 10:15:00 | 1133.70 | 1128.61 | 1130.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 10:45:00 | 1131.70 | 1128.61 | 1130.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-27 11:15:00 | 1134.00 | 1129.69 | 1130.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-27 12:00:00 | 1134.00 | 1129.69 | 1130.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 12:15:00 | 1140.75 | 1131.90 | 1131.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 1147.45 | 1137.77 | 1134.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 12:15:00 | 1138.00 | 1139.76 | 1136.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-28 13:00:00 | 1138.00 | 1139.76 | 1136.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 1136.70 | 1139.15 | 1136.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 13:45:00 | 1135.90 | 1139.15 | 1136.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 14:15:00 | 1130.30 | 1137.38 | 1135.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 15:00:00 | 1130.30 | 1137.38 | 1135.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 15:15:00 | 1135.85 | 1137.07 | 1135.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-29 09:30:00 | 1139.25 | 1137.94 | 1136.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-04 09:15:00 | 1127.50 | 1142.17 | 1144.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 1127.50 | 1142.17 | 1144.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 10:15:00 | 1122.60 | 1138.26 | 1142.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 15:15:00 | 1124.60 | 1121.27 | 1127.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-06 09:15:00 | 1128.10 | 1121.27 | 1127.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 09:15:00 | 1126.55 | 1122.33 | 1126.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 11:00:00 | 1122.90 | 1125.88 | 1127.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 11:45:00 | 1122.85 | 1125.50 | 1126.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 12:30:00 | 1123.00 | 1125.14 | 1126.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 13:15:00 | 1123.20 | 1125.14 | 1126.49 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 13:15:00 | 1124.50 | 1125.01 | 1126.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-09 13:30:00 | 1124.10 | 1125.01 | 1126.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 14:15:00 | 1126.80 | 1125.37 | 1126.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-09 15:00:00 | 1126.80 | 1125.37 | 1126.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 15:15:00 | 1120.10 | 1124.32 | 1125.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-11 09:15:00 | 1130.30 | 1126.17 | 1125.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2023-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-11 09:15:00 | 1130.30 | 1126.17 | 1125.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 10:15:00 | 1132.95 | 1127.52 | 1126.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 1125.05 | 1128.25 | 1127.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 13:15:00 | 1125.05 | 1128.25 | 1127.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 1125.05 | 1128.25 | 1127.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 13:45:00 | 1124.30 | 1128.25 | 1127.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 1128.35 | 1128.27 | 1127.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 14:30:00 | 1125.60 | 1128.27 | 1127.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 15:15:00 | 1126.50 | 1127.92 | 1127.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:15:00 | 1130.35 | 1127.92 | 1127.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 09:15:00 | 1124.55 | 1127.24 | 1127.01 | SL hit (close<static) qty=1.00 sl=1125.00 alert=retest2 |

### Cycle 26 — SELL (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 13:15:00 | 1125.70 | 1126.75 | 1126.84 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-10-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-12 14:15:00 | 1131.50 | 1127.70 | 1127.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-13 11:15:00 | 1140.00 | 1131.79 | 1129.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-16 11:15:00 | 1139.90 | 1140.41 | 1135.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-16 12:00:00 | 1139.90 | 1140.41 | 1135.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 14:15:00 | 1135.25 | 1138.40 | 1136.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-16 14:30:00 | 1135.80 | 1138.40 | 1136.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 15:15:00 | 1135.00 | 1137.72 | 1135.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 09:15:00 | 1137.45 | 1137.72 | 1135.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-17 14:30:00 | 1138.25 | 1138.92 | 1137.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-18 09:15:00 | 1141.85 | 1138.15 | 1137.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-19 15:15:00 | 1136.60 | 1143.39 | 1143.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-10-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-19 15:15:00 | 1136.60 | 1143.39 | 1143.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 11:15:00 | 1130.05 | 1139.02 | 1141.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-27 09:15:00 | 1109.60 | 1108.49 | 1115.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-27 09:15:00 | 1109.60 | 1108.49 | 1115.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 1109.60 | 1108.49 | 1115.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 09:30:00 | 1109.00 | 1108.49 | 1115.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 1117.70 | 1111.01 | 1115.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 12:00:00 | 1117.70 | 1111.01 | 1115.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 12:15:00 | 1114.90 | 1111.79 | 1115.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 13:00:00 | 1114.90 | 1111.79 | 1115.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 13:15:00 | 1118.50 | 1113.13 | 1116.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 14:00:00 | 1118.50 | 1113.13 | 1116.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 14:15:00 | 1112.25 | 1112.95 | 1115.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 14:30:00 | 1116.40 | 1112.95 | 1115.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 15:15:00 | 1117.00 | 1113.76 | 1115.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-30 09:15:00 | 1109.50 | 1113.76 | 1115.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-30 09:15:00 | 1108.05 | 1112.62 | 1115.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-31 09:45:00 | 1099.45 | 1109.62 | 1112.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-01 15:15:00 | 1117.00 | 1104.38 | 1102.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2023-11-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-01 15:15:00 | 1117.00 | 1104.38 | 1102.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 12:15:00 | 1135.75 | 1119.82 | 1111.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-08 15:15:00 | 1173.25 | 1173.80 | 1164.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-09 09:15:00 | 1169.55 | 1173.80 | 1164.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-10 09:15:00 | 1173.90 | 1176.65 | 1171.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-10 09:30:00 | 1172.35 | 1176.65 | 1171.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 11:15:00 | 1176.85 | 1179.21 | 1176.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-13 11:45:00 | 1177.35 | 1179.21 | 1176.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 12:15:00 | 1176.90 | 1178.75 | 1176.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-13 15:15:00 | 1178.10 | 1178.18 | 1176.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 09:15:00 | 1174.00 | 1177.33 | 1176.45 | SL hit (close<static) qty=1.00 sl=1174.10 alert=retest2 |

### Cycle 30 — SELL (started 2023-11-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 12:15:00 | 1194.25 | 1199.36 | 1199.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-28 09:15:00 | 1190.90 | 1196.23 | 1198.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-29 09:15:00 | 1196.95 | 1193.43 | 1195.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-29 09:15:00 | 1196.95 | 1193.43 | 1195.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 1196.95 | 1193.43 | 1195.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 10:00:00 | 1196.95 | 1193.43 | 1195.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 1197.80 | 1194.31 | 1195.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-29 11:15:00 | 1192.50 | 1194.31 | 1195.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-29 15:15:00 | 1206.25 | 1198.04 | 1197.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2023-11-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 15:15:00 | 1206.25 | 1198.04 | 1197.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 10:15:00 | 1212.35 | 1202.28 | 1199.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 11:15:00 | 1225.05 | 1226.71 | 1220.27 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2023-12-04 13:15:00 | 1228.80 | 1226.37 | 1220.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 1232.20 | 1228.89 | 1223.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 10:30:00 | 1234.70 | 1229.52 | 1224.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 13:00:00 | 1234.35 | 1231.16 | 1226.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-07 09:15:00 | 1235.05 | 1238.60 | 1235.13 | SL hit (close<ema400) qty=1.00 sl=1235.13 alert=retest1 |

### Cycle 32 — SELL (started 2023-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 10:15:00 | 1232.65 | 1235.63 | 1235.95 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2023-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 14:15:00 | 1240.00 | 1236.51 | 1236.17 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2023-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 10:15:00 | 1230.05 | 1235.16 | 1235.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 11:15:00 | 1225.05 | 1233.14 | 1234.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 13:15:00 | 1226.40 | 1221.72 | 1225.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-13 13:15:00 | 1226.40 | 1221.72 | 1225.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 13:15:00 | 1226.40 | 1221.72 | 1225.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 14:00:00 | 1226.40 | 1221.72 | 1225.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 14:15:00 | 1233.40 | 1224.06 | 1226.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-13 15:00:00 | 1233.40 | 1224.06 | 1226.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 1232.15 | 1225.68 | 1227.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:15:00 | 1229.00 | 1225.68 | 1227.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 10:15:00 | 1232.75 | 1227.26 | 1227.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 11:00:00 | 1232.75 | 1227.26 | 1227.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2023-12-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 11:15:00 | 1233.80 | 1228.56 | 1228.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 12:15:00 | 1234.75 | 1229.80 | 1228.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 14:15:00 | 1229.05 | 1230.39 | 1229.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-14 14:15:00 | 1229.05 | 1230.39 | 1229.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 14:15:00 | 1229.05 | 1230.39 | 1229.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-14 15:00:00 | 1229.05 | 1230.39 | 1229.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 15:15:00 | 1234.00 | 1231.12 | 1229.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-15 09:15:00 | 1238.25 | 1231.12 | 1229.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 11:15:00 | 1238.65 | 1243.20 | 1243.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2023-12-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 11:15:00 | 1238.65 | 1243.20 | 1243.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 14:15:00 | 1232.35 | 1239.61 | 1241.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 12:15:00 | 1231.95 | 1231.62 | 1236.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 12:45:00 | 1233.35 | 1231.62 | 1236.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 1237.85 | 1233.33 | 1236.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 09:15:00 | 1242.85 | 1233.33 | 1236.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 1247.25 | 1236.12 | 1237.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 1247.25 | 1236.12 | 1237.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2023-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 10:15:00 | 1247.35 | 1238.36 | 1237.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-27 12:15:00 | 1250.10 | 1246.98 | 1244.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 12:15:00 | 1251.90 | 1253.26 | 1249.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 12:15:00 | 1251.90 | 1253.26 | 1249.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 12:15:00 | 1251.90 | 1253.26 | 1249.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-28 13:00:00 | 1251.90 | 1253.26 | 1249.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 12:15:00 | 1254.00 | 1257.20 | 1254.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 13:00:00 | 1254.00 | 1257.20 | 1254.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 13:15:00 | 1254.30 | 1256.62 | 1254.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-29 14:00:00 | 1254.30 | 1256.62 | 1254.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 14:15:00 | 1259.55 | 1257.21 | 1254.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 12:00:00 | 1260.80 | 1257.71 | 1255.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 14:15:00 | 1261.10 | 1257.88 | 1256.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 09:15:00 | 1277.05 | 1258.71 | 1256.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-16 11:15:00 | 1312.80 | 1323.51 | 1323.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 11:15:00 | 1312.80 | 1323.51 | 1323.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-17 09:15:00 | 1300.20 | 1313.96 | 1318.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-18 09:15:00 | 1308.75 | 1304.22 | 1309.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-18 09:15:00 | 1308.75 | 1304.22 | 1309.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 09:15:00 | 1308.75 | 1304.22 | 1309.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 10:00:00 | 1308.75 | 1304.22 | 1309.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 10:15:00 | 1315.00 | 1306.38 | 1310.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 10:30:00 | 1315.80 | 1306.38 | 1310.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-18 11:15:00 | 1321.15 | 1309.33 | 1311.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-18 12:00:00 | 1321.15 | 1309.33 | 1311.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-01-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 13:15:00 | 1333.40 | 1316.35 | 1314.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-18 14:15:00 | 1336.40 | 1320.36 | 1316.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 10:15:00 | 1330.60 | 1332.98 | 1327.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 10:45:00 | 1332.35 | 1332.98 | 1327.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 13:15:00 | 1328.95 | 1332.12 | 1328.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 13:45:00 | 1329.45 | 1332.12 | 1328.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 1326.70 | 1331.04 | 1328.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 15:00:00 | 1326.70 | 1331.04 | 1328.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 1327.85 | 1330.40 | 1328.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 1354.00 | 1330.40 | 1328.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-23 09:15:00 | 1378.00 | 1339.92 | 1332.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 10:15:00 | 1383.70 | 1339.92 | 1332.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 09:45:00 | 1386.00 | 1370.00 | 1354.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 11:00:00 | 1381.75 | 1372.35 | 1357.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-24 13:45:00 | 1384.35 | 1376.47 | 1362.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 1366.35 | 1376.81 | 1366.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:00:00 | 1366.35 | 1376.81 | 1366.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 1350.20 | 1371.49 | 1365.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-25 11:00:00 | 1350.20 | 1371.49 | 1365.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 11:15:00 | 1360.30 | 1369.25 | 1364.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 13:30:00 | 1364.40 | 1365.46 | 1363.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 14:45:00 | 1365.80 | 1365.05 | 1363.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-08 09:15:00 | 1500.84 | 1491.61 | 1475.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-02-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-14 12:15:00 | 1510.75 | 1523.95 | 1525.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-15 12:15:00 | 1507.35 | 1516.06 | 1520.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-16 10:15:00 | 1514.80 | 1512.31 | 1516.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-16 10:15:00 | 1514.80 | 1512.31 | 1516.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-16 10:15:00 | 1514.80 | 1512.31 | 1516.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 13:30:00 | 1509.00 | 1512.30 | 1515.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 14:45:00 | 1508.65 | 1511.73 | 1514.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-16 15:15:00 | 1507.55 | 1511.73 | 1514.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-19 09:45:00 | 1508.75 | 1510.93 | 1513.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 10:15:00 | 1515.00 | 1511.74 | 1514.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-19 11:00:00 | 1515.00 | 1511.74 | 1514.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 11:15:00 | 1524.25 | 1514.24 | 1514.98 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-02-19 11:15:00 | 1524.25 | 1514.24 | 1514.98 | SL hit (close>static) qty=1.00 sl=1519.25 alert=retest2 |

### Cycle 41 — BUY (started 2024-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 12:15:00 | 1523.80 | 1516.15 | 1515.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 13:15:00 | 1527.05 | 1518.33 | 1516.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 09:15:00 | 1516.95 | 1521.17 | 1518.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 09:15:00 | 1516.95 | 1521.17 | 1518.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 09:15:00 | 1516.95 | 1521.17 | 1518.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 10:00:00 | 1516.95 | 1521.17 | 1518.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 1519.25 | 1520.78 | 1518.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:15:00 | 1515.80 | 1520.78 | 1518.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 11:15:00 | 1521.60 | 1520.95 | 1519.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 11:30:00 | 1517.90 | 1520.95 | 1519.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 11:15:00 | 1546.55 | 1541.27 | 1535.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-22 13:45:00 | 1550.50 | 1543.80 | 1537.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-01 13:15:00 | 1551.30 | 1565.87 | 1567.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-03-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-01 13:15:00 | 1551.30 | 1565.87 | 1567.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-05 09:15:00 | 1546.60 | 1554.89 | 1558.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-05 11:15:00 | 1561.10 | 1555.07 | 1558.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-05 11:15:00 | 1561.10 | 1555.07 | 1558.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 11:15:00 | 1561.10 | 1555.07 | 1558.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 12:00:00 | 1561.10 | 1555.07 | 1558.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 12:15:00 | 1570.00 | 1558.05 | 1559.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-05 13:00:00 | 1570.00 | 1558.05 | 1559.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-05 13:15:00 | 1570.05 | 1560.45 | 1560.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-05 14:15:00 | 1574.50 | 1563.26 | 1561.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-11 14:15:00 | 1592.00 | 1608.57 | 1600.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-11 14:15:00 | 1592.00 | 1608.57 | 1600.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 14:15:00 | 1592.00 | 1608.57 | 1600.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-11 15:00:00 | 1592.00 | 1608.57 | 1600.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 15:15:00 | 1592.15 | 1605.28 | 1600.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-12 09:15:00 | 1597.95 | 1605.28 | 1600.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-12 11:00:00 | 1593.40 | 1601.17 | 1599.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 13:15:00 | 1589.90 | 1596.51 | 1597.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 13:15:00 | 1589.90 | 1596.51 | 1597.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-12 14:15:00 | 1583.45 | 1593.89 | 1596.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 11:15:00 | 1563.35 | 1559.81 | 1571.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-14 12:00:00 | 1563.35 | 1559.81 | 1571.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 14:15:00 | 1568.20 | 1561.81 | 1569.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 15:00:00 | 1568.20 | 1561.81 | 1569.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 15:15:00 | 1569.00 | 1563.24 | 1569.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 09:30:00 | 1564.20 | 1561.52 | 1567.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-18 14:00:00 | 1566.00 | 1557.10 | 1559.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 15:15:00 | 1569.35 | 1561.88 | 1561.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 15:15:00 | 1569.35 | 1561.88 | 1561.12 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-03-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 14:15:00 | 1549.70 | 1560.95 | 1561.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 1533.00 | 1553.27 | 1557.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 1556.50 | 1547.66 | 1553.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 12:15:00 | 1556.50 | 1547.66 | 1553.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 1556.50 | 1547.66 | 1553.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:00:00 | 1556.50 | 1547.66 | 1553.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 1554.55 | 1549.04 | 1553.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-20 15:00:00 | 1549.00 | 1549.03 | 1553.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-20 15:15:00 | 1556.70 | 1550.57 | 1553.38 | SL hit (close>static) qty=1.00 sl=1556.65 alert=retest2 |

### Cycle 47 — BUY (started 2024-03-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 12:15:00 | 1564.75 | 1556.05 | 1555.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 13:15:00 | 1568.70 | 1558.58 | 1556.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 10:15:00 | 1597.95 | 1598.13 | 1584.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 10:45:00 | 1597.25 | 1598.13 | 1584.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-27 09:15:00 | 1600.50 | 1601.07 | 1592.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 11:15:00 | 1607.00 | 1601.69 | 1593.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 15:15:00 | 1608.45 | 1606.90 | 1598.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-03 10:00:00 | 1613.95 | 1620.83 | 1619.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 09:15:00 | 1598.75 | 1616.60 | 1618.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 09:15:00 | 1598.75 | 1616.60 | 1618.23 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 1625.15 | 1618.44 | 1617.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-05 11:15:00 | 1627.35 | 1620.22 | 1618.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-05 12:15:00 | 1617.35 | 1619.65 | 1618.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-05 12:15:00 | 1617.35 | 1619.65 | 1618.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 12:15:00 | 1617.35 | 1619.65 | 1618.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 12:45:00 | 1617.10 | 1619.65 | 1618.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-05 13:15:00 | 1616.65 | 1619.05 | 1618.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-05 13:30:00 | 1616.50 | 1619.05 | 1618.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-04-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 14:15:00 | 1608.30 | 1616.90 | 1617.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-08 11:15:00 | 1605.35 | 1612.76 | 1615.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-09 13:15:00 | 1604.35 | 1602.57 | 1607.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 13:15:00 | 1604.35 | 1602.57 | 1607.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 1604.35 | 1602.57 | 1607.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-09 13:45:00 | 1607.50 | 1602.57 | 1607.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 1603.00 | 1602.65 | 1606.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-10 09:15:00 | 1593.70 | 1602.28 | 1606.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-12 09:15:00 | 1576.50 | 1599.23 | 1601.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-04-15 09:15:00 | 1514.01 | 1550.54 | 1570.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-04-15 14:15:00 | 1540.55 | 1539.74 | 1556.57 | SL hit (close>ema200) qty=0.50 sl=1539.74 alert=retest2 |

### Cycle 51 — BUY (started 2024-04-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 14:15:00 | 1540.80 | 1529.32 | 1528.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 15:15:00 | 1547.00 | 1532.86 | 1530.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 10:15:00 | 1514.05 | 1530.74 | 1530.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-23 10:15:00 | 1514.05 | 1530.74 | 1530.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 1514.05 | 1530.74 | 1530.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 11:00:00 | 1514.05 | 1530.74 | 1530.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2024-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 11:15:00 | 1508.25 | 1526.24 | 1528.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-23 12:15:00 | 1498.00 | 1520.59 | 1525.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 09:15:00 | 1513.95 | 1497.07 | 1504.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 1513.95 | 1497.07 | 1504.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 1513.95 | 1497.07 | 1504.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:00:00 | 1513.95 | 1497.07 | 1504.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 1506.30 | 1498.91 | 1504.37 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 14:15:00 | 1522.40 | 1508.25 | 1507.28 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 14:15:00 | 1506.00 | 1507.69 | 1507.74 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2024-04-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-26 15:15:00 | 1516.00 | 1509.35 | 1508.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-29 09:15:00 | 1522.90 | 1512.06 | 1509.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-30 09:15:00 | 1518.70 | 1519.38 | 1515.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-30 09:15:00 | 1518.70 | 1519.38 | 1515.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 09:15:00 | 1518.70 | 1519.38 | 1515.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:00:00 | 1518.70 | 1519.38 | 1515.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 10:15:00 | 1518.40 | 1519.19 | 1515.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 10:30:00 | 1518.15 | 1519.19 | 1515.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 11:15:00 | 1513.00 | 1517.95 | 1515.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 11:45:00 | 1513.00 | 1517.95 | 1515.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 12:15:00 | 1513.50 | 1517.06 | 1515.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 13:00:00 | 1513.50 | 1517.06 | 1515.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-30 13:15:00 | 1509.60 | 1515.57 | 1514.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-30 13:45:00 | 1509.60 | 1515.57 | 1514.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 1502.85 | 1513.02 | 1513.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-30 15:15:00 | 1498.60 | 1510.14 | 1512.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 09:15:00 | 1524.40 | 1512.99 | 1513.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 09:15:00 | 1524.40 | 1512.99 | 1513.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 09:15:00 | 1524.40 | 1512.99 | 1513.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 09:45:00 | 1521.00 | 1512.99 | 1513.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2024-05-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 10:15:00 | 1521.00 | 1514.59 | 1514.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 11:15:00 | 1526.40 | 1516.95 | 1515.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-02 14:15:00 | 1518.30 | 1519.16 | 1516.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 14:15:00 | 1518.30 | 1519.16 | 1516.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 14:15:00 | 1518.30 | 1519.16 | 1516.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 09:45:00 | 1533.85 | 1520.21 | 1517.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-03 10:30:00 | 1530.15 | 1521.07 | 1518.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-03 11:15:00 | 1512.75 | 1519.41 | 1517.86 | SL hit (close<static) qty=1.00 sl=1515.90 alert=retest2 |

### Cycle 58 — SELL (started 2024-05-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 14:15:00 | 1512.00 | 1515.96 | 1516.50 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 11:15:00 | 1532.10 | 1519.44 | 1517.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-07 09:15:00 | 1542.35 | 1527.91 | 1522.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 10:15:00 | 1523.20 | 1526.96 | 1522.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 10:15:00 | 1523.20 | 1526.96 | 1522.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 1523.20 | 1526.96 | 1522.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 11:00:00 | 1523.20 | 1526.96 | 1522.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 11:15:00 | 1515.75 | 1524.72 | 1522.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 12:00:00 | 1515.75 | 1524.72 | 1522.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 12:15:00 | 1512.40 | 1522.26 | 1521.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 13:00:00 | 1512.40 | 1522.26 | 1521.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2024-05-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 14:15:00 | 1515.25 | 1519.88 | 1520.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-09 09:15:00 | 1506.15 | 1516.95 | 1518.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 1513.90 | 1506.25 | 1511.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-10 09:15:00 | 1513.90 | 1506.25 | 1511.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 09:15:00 | 1513.90 | 1506.25 | 1511.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 09:45:00 | 1525.85 | 1506.25 | 1511.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 1512.85 | 1507.57 | 1511.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:30:00 | 1514.55 | 1507.57 | 1511.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 1503.40 | 1506.74 | 1510.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 14:30:00 | 1499.50 | 1506.17 | 1509.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 12:15:00 | 1518.50 | 1509.92 | 1509.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 1518.50 | 1509.92 | 1509.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 1521.40 | 1512.22 | 1510.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 1528.80 | 1533.77 | 1525.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 10:00:00 | 1528.80 | 1533.77 | 1525.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 1528.10 | 1532.63 | 1526.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 1524.50 | 1532.63 | 1526.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 1522.05 | 1530.52 | 1525.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 11:45:00 | 1526.00 | 1530.52 | 1525.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 12:15:00 | 1524.85 | 1529.38 | 1525.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 12:00:00 | 1528.90 | 1526.32 | 1525.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-16 12:15:00 | 1514.00 | 1523.86 | 1524.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2024-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 12:15:00 | 1514.00 | 1523.86 | 1524.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 1510.35 | 1521.16 | 1523.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 1534.90 | 1523.91 | 1524.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 1534.90 | 1523.91 | 1524.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 1534.90 | 1523.91 | 1524.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 15:00:00 | 1534.90 | 1523.91 | 1524.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 1530.05 | 1525.13 | 1524.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 12:15:00 | 1543.90 | 1533.35 | 1530.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 15:15:00 | 1533.00 | 1536.35 | 1533.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 15:15:00 | 1533.00 | 1536.35 | 1533.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 1533.00 | 1536.35 | 1533.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 1525.80 | 1536.35 | 1533.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 1520.85 | 1533.25 | 1531.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:00:00 | 1520.85 | 1533.25 | 1531.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 1519.30 | 1530.46 | 1530.82 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 15:15:00 | 1563.00 | 1533.23 | 1531.17 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 09:15:00 | 1480.05 | 1522.59 | 1526.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 1465.45 | 1478.97 | 1489.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 13:15:00 | 1475.70 | 1471.01 | 1479.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-28 14:00:00 | 1475.70 | 1471.01 | 1479.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 1473.40 | 1470.72 | 1476.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:15:00 | 1478.80 | 1470.72 | 1476.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 1480.00 | 1472.58 | 1477.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:45:00 | 1481.55 | 1472.58 | 1477.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 1489.80 | 1476.02 | 1478.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:45:00 | 1487.85 | 1476.02 | 1478.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 1490.50 | 1478.92 | 1479.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:30:00 | 1493.50 | 1478.92 | 1479.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 1462.60 | 1474.44 | 1477.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 14:30:00 | 1450.40 | 1464.38 | 1470.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 11:00:00 | 1451.20 | 1461.15 | 1467.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 11:00:00 | 1449.25 | 1459.12 | 1461.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 12:00:00 | 1414.35 | 1450.16 | 1456.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 1440.90 | 1448.31 | 1455.48 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 1377.88 | 1448.31 | 1455.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 12:15:00 | 1378.64 | 1448.31 | 1455.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| ALERT3_SIDEWAYS | 2024-06-04 12:45:00 | 1450.50 | 1448.31 | 1455.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 15:15:00 | 1453.80 | 1445.32 | 1452.04 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-06-04 15:15:00 | 1453.80 | 1445.32 | 1452.04 | SL hit (close>ema200) qty=0.50 sl=1445.32 alert=retest2 |

### Cycle 67 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 1481.90 | 1459.78 | 1457.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 1489.45 | 1471.29 | 1463.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 1472.65 | 1473.75 | 1466.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 1472.65 | 1473.75 | 1466.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 1472.65 | 1473.75 | 1466.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:45:00 | 1470.80 | 1473.75 | 1466.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 1469.85 | 1472.97 | 1466.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:30:00 | 1471.55 | 1472.97 | 1466.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 11:15:00 | 1460.95 | 1470.57 | 1466.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 12:00:00 | 1460.95 | 1470.57 | 1466.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 12:15:00 | 1458.95 | 1468.24 | 1465.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 13:00:00 | 1458.95 | 1468.24 | 1465.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 13:15:00 | 1463.50 | 1467.30 | 1465.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-06 14:45:00 | 1472.00 | 1467.98 | 1465.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 14:15:00 | 1503.35 | 1512.06 | 1512.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 14:15:00 | 1503.35 | 1512.06 | 1512.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 15:15:00 | 1502.95 | 1510.24 | 1511.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 09:15:00 | 1491.80 | 1477.93 | 1484.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 09:15:00 | 1491.80 | 1477.93 | 1484.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 09:15:00 | 1491.80 | 1477.93 | 1484.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:45:00 | 1490.50 | 1477.93 | 1484.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 10:15:00 | 1498.40 | 1482.03 | 1486.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 10:30:00 | 1497.65 | 1482.03 | 1486.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2024-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 13:15:00 | 1494.60 | 1489.28 | 1488.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 10:15:00 | 1500.30 | 1494.14 | 1491.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 13:15:00 | 1510.20 | 1513.22 | 1508.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 13:30:00 | 1509.55 | 1513.22 | 1508.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 1517.55 | 1514.09 | 1508.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 14:30:00 | 1507.90 | 1514.09 | 1508.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 1519.80 | 1521.99 | 1516.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 1517.20 | 1521.99 | 1516.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 1507.30 | 1519.05 | 1516.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:00:00 | 1507.30 | 1519.05 | 1516.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 1507.25 | 1516.69 | 1515.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:45:00 | 1507.10 | 1516.69 | 1515.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 12:15:00 | 1510.45 | 1514.37 | 1514.40 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 1518.45 | 1515.19 | 1514.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 14:15:00 | 1520.65 | 1516.28 | 1515.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 15:15:00 | 1513.50 | 1515.72 | 1515.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 15:15:00 | 1513.50 | 1515.72 | 1515.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 15:15:00 | 1513.50 | 1515.72 | 1515.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:15:00 | 1515.75 | 1515.72 | 1515.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 1521.95 | 1516.97 | 1515.76 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 10:45:00 | 1524.80 | 1518.55 | 1516.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 13:00:00 | 1525.00 | 1520.44 | 1517.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:45:00 | 1527.65 | 1522.97 | 1520.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 11:30:00 | 1527.20 | 1523.98 | 1520.97 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 1552.70 | 1560.83 | 1555.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:00:00 | 1552.70 | 1560.83 | 1555.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 1556.05 | 1559.87 | 1555.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:30:00 | 1554.60 | 1559.87 | 1555.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 1554.25 | 1558.75 | 1555.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:30:00 | 1551.25 | 1557.71 | 1555.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 1558.85 | 1557.94 | 1555.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 11:15:00 | 1564.50 | 1557.94 | 1555.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 15:15:00 | 1576.00 | 1582.01 | 1582.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2024-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 15:15:00 | 1576.00 | 1582.01 | 1582.53 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2024-07-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 09:15:00 | 1589.05 | 1583.42 | 1583.12 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 1575.10 | 1581.76 | 1582.39 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 12:15:00 | 1592.95 | 1584.01 | 1583.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 13:15:00 | 1594.60 | 1586.13 | 1584.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-19 09:15:00 | 1579.40 | 1587.23 | 1585.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-19 09:15:00 | 1579.40 | 1587.23 | 1585.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 1579.40 | 1587.23 | 1585.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:45:00 | 1575.40 | 1587.23 | 1585.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 1584.00 | 1586.58 | 1585.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:30:00 | 1578.55 | 1586.58 | 1585.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 1581.70 | 1585.61 | 1585.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:45:00 | 1581.75 | 1585.61 | 1585.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 1577.50 | 1583.98 | 1584.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 1566.45 | 1580.48 | 1582.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 1579.45 | 1576.69 | 1580.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 1579.45 | 1576.69 | 1580.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 1579.45 | 1576.69 | 1580.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 1579.45 | 1576.69 | 1580.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 1580.15 | 1577.38 | 1580.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 1583.85 | 1577.38 | 1580.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 1587.20 | 1579.35 | 1580.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 1587.20 | 1579.35 | 1580.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 1579.25 | 1579.33 | 1580.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 1581.85 | 1579.33 | 1580.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 1585.70 | 1580.60 | 1581.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:00:00 | 1585.70 | 1580.60 | 1581.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 14:15:00 | 1587.75 | 1582.03 | 1581.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 10:15:00 | 1595.25 | 1586.33 | 1583.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 1609.65 | 1612.95 | 1603.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-25 10:15:00 | 1616.45 | 1612.95 | 1603.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 1614.70 | 1613.30 | 1604.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 11:30:00 | 1622.50 | 1617.14 | 1607.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-06 13:15:00 | 1711.20 | 1722.57 | 1722.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2024-08-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 13:15:00 | 1711.20 | 1722.57 | 1722.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 14:15:00 | 1706.80 | 1719.41 | 1721.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 1726.80 | 1719.07 | 1720.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 1726.80 | 1719.07 | 1720.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1726.80 | 1719.07 | 1720.87 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 1727.40 | 1722.80 | 1722.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 09:15:00 | 1741.40 | 1731.44 | 1727.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 13:15:00 | 1735.10 | 1735.39 | 1730.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 13:30:00 | 1740.60 | 1735.39 | 1730.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 1732.60 | 1735.34 | 1731.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 1743.75 | 1735.34 | 1731.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 10:30:00 | 1739.95 | 1735.94 | 1732.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 09:30:00 | 1739.95 | 1736.78 | 1734.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 10:30:00 | 1740.35 | 1738.60 | 1735.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 13:15:00 | 1739.20 | 1738.89 | 1736.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 13:30:00 | 1736.75 | 1738.89 | 1736.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 1732.50 | 1737.61 | 1736.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 15:00:00 | 1732.50 | 1737.61 | 1736.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 1733.80 | 1736.85 | 1735.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:15:00 | 1744.40 | 1736.85 | 1735.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 1743.50 | 1738.18 | 1736.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:00:00 | 1746.20 | 1739.78 | 1737.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 11:30:00 | 1746.95 | 1741.32 | 1738.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 1746.45 | 1740.09 | 1738.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 1746.20 | 1740.01 | 1739.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 1738.15 | 1739.64 | 1739.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:00:00 | 1738.15 | 1739.64 | 1739.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-08-16 10:15:00 | 1735.00 | 1738.71 | 1738.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2024-08-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 10:15:00 | 1735.00 | 1738.71 | 1738.73 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 1738.90 | 1738.75 | 1738.75 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 12:15:00 | 1732.50 | 1737.50 | 1738.18 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 1742.00 | 1739.02 | 1738.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 10:15:00 | 1746.10 | 1741.38 | 1739.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 15:15:00 | 1745.00 | 1745.16 | 1742.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:15:00 | 1761.25 | 1745.16 | 1742.65 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 13:15:00 | 1759.90 | 1761.00 | 1758.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 13:30:00 | 1760.00 | 1761.00 | 1758.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 1750.00 | 1758.80 | 1757.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-22 14:15:00 | 1750.00 | 1758.80 | 1757.43 | SL hit (close<ema400) qty=1.00 sl=1757.43 alert=retest1 |

### Cycle 84 — SELL (started 2024-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 09:15:00 | 1821.85 | 1825.36 | 1825.36 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2024-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 10:15:00 | 1834.80 | 1825.86 | 1824.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 11:15:00 | 1839.00 | 1828.49 | 1826.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 14:15:00 | 1846.20 | 1846.29 | 1839.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 15:00:00 | 1846.20 | 1846.29 | 1839.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 1850.85 | 1847.94 | 1841.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 10:15:00 | 1856.20 | 1847.94 | 1841.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 11:30:00 | 1855.40 | 1849.32 | 1843.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:00:00 | 1855.50 | 1850.43 | 1844.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:30:00 | 1858.95 | 1853.20 | 1846.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 1853.05 | 1856.38 | 1851.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:00:00 | 1853.05 | 1856.38 | 1851.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 1859.10 | 1856.38 | 1852.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 11:15:00 | 1864.40 | 1857.70 | 1853.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 13:15:00 | 1864.55 | 1860.35 | 1855.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 10:15:00 | 1865.90 | 1861.99 | 1857.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 11:45:00 | 1864.25 | 1862.64 | 1858.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1855.25 | 1862.70 | 1860.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:00:00 | 1855.25 | 1862.70 | 1860.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1858.30 | 1861.82 | 1860.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:15:00 | 1857.65 | 1861.82 | 1860.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 1860.05 | 1861.46 | 1860.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 1847.20 | 1858.61 | 1859.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 1847.20 | 1858.61 | 1859.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 1836.40 | 1854.17 | 1857.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 1851.15 | 1848.15 | 1853.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 10:00:00 | 1851.15 | 1848.15 | 1853.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 1849.80 | 1848.48 | 1852.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:45:00 | 1854.20 | 1848.48 | 1852.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 1846.75 | 1848.14 | 1852.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 12:15:00 | 1846.10 | 1848.14 | 1852.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 14:45:00 | 1845.85 | 1845.94 | 1850.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:30:00 | 1845.35 | 1845.56 | 1849.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 10:00:00 | 1844.35 | 1845.56 | 1849.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 1853.35 | 1847.12 | 1849.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 11:00:00 | 1853.35 | 1847.12 | 1849.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 11:15:00 | 1852.00 | 1848.09 | 1849.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-09-20 12:15:00 | 1859.80 | 1850.44 | 1850.70 | SL hit (close>static) qty=1.00 sl=1853.90 alert=retest2 |

### Cycle 87 — BUY (started 2024-09-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 13:15:00 | 1854.55 | 1851.26 | 1851.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 1865.20 | 1854.05 | 1852.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 11:15:00 | 1858.45 | 1858.74 | 1855.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 12:00:00 | 1858.45 | 1858.74 | 1855.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1870.00 | 1864.01 | 1859.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 13:45:00 | 1870.85 | 1866.27 | 1862.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-24 15:15:00 | 1875.00 | 1866.38 | 1862.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 15:15:00 | 1874.20 | 1865.38 | 1863.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 14:15:00 | 1911.35 | 1917.59 | 1917.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 14:15:00 | 1911.35 | 1917.59 | 1917.64 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-04 10:15:00 | 1923.30 | 1917.49 | 1917.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 11:15:00 | 1945.05 | 1923.00 | 1919.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 13:15:00 | 1918.20 | 1922.82 | 1920.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 13:15:00 | 1918.20 | 1922.82 | 1920.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 13:15:00 | 1918.20 | 1922.82 | 1920.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:00:00 | 1918.20 | 1922.82 | 1920.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 1910.15 | 1920.29 | 1919.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 14:45:00 | 1910.10 | 1920.29 | 1919.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2024-10-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 15:15:00 | 1907.95 | 1917.82 | 1918.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 1904.25 | 1915.11 | 1917.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 14:15:00 | 1905.95 | 1904.77 | 1910.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 15:00:00 | 1905.95 | 1904.77 | 1910.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1910.45 | 1905.30 | 1909.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 09:45:00 | 1912.75 | 1905.30 | 1909.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 1916.95 | 1907.63 | 1910.24 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 1919.50 | 1912.19 | 1911.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 1926.75 | 1916.46 | 1914.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 14:15:00 | 1929.50 | 1929.81 | 1922.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-09 15:00:00 | 1929.50 | 1929.81 | 1922.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 1925.85 | 1929.80 | 1923.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:45:00 | 1922.70 | 1929.80 | 1923.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 1917.25 | 1927.29 | 1923.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 1917.25 | 1927.29 | 1923.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 1918.10 | 1925.45 | 1922.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:30:00 | 1917.70 | 1925.45 | 1922.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2024-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 13:15:00 | 1894.85 | 1916.91 | 1919.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 1888.00 | 1911.13 | 1916.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 11:15:00 | 1906.00 | 1903.94 | 1910.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-11 12:00:00 | 1906.00 | 1903.94 | 1910.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 1912.10 | 1904.48 | 1906.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 13:30:00 | 1911.85 | 1904.48 | 1906.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 1909.65 | 1905.52 | 1907.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 10:00:00 | 1903.25 | 1905.45 | 1906.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 10:30:00 | 1900.55 | 1902.46 | 1904.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-17 09:30:00 | 1900.40 | 1899.03 | 1901.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 11:15:00 | 1911.15 | 1899.39 | 1898.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 11:15:00 | 1911.15 | 1899.39 | 1898.51 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 14:15:00 | 1897.65 | 1900.74 | 1900.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-21 15:15:00 | 1891.00 | 1898.79 | 1899.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 09:15:00 | 1863.00 | 1852.16 | 1861.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-25 09:15:00 | 1863.00 | 1852.16 | 1861.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 1863.00 | 1852.16 | 1861.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:30:00 | 1849.05 | 1859.26 | 1861.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 1870.15 | 1861.44 | 1862.64 | SL hit (close>static) qty=1.00 sl=1867.30 alert=retest2 |

### Cycle 95 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 1886.60 | 1866.47 | 1864.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-28 13:15:00 | 1896.95 | 1875.17 | 1869.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 09:15:00 | 1878.55 | 1883.48 | 1875.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 09:15:00 | 1878.55 | 1883.48 | 1875.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 1878.55 | 1883.48 | 1875.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:30:00 | 1875.95 | 1883.48 | 1875.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 1870.30 | 1880.85 | 1874.75 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 09:15:00 | 1837.80 | 1866.36 | 1869.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 1794.65 | 1845.33 | 1854.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 1820.20 | 1814.99 | 1829.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 09:15:00 | 1820.20 | 1814.99 | 1829.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 1820.20 | 1814.99 | 1829.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 12:45:00 | 1800.25 | 1812.10 | 1824.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 13:30:00 | 1802.45 | 1810.51 | 1822.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 1801.50 | 1809.40 | 1821.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 13:15:00 | 1834.75 | 1822.22 | 1823.08 | SL hit (close>static) qty=1.00 sl=1833.00 alert=retest2 |

### Cycle 97 — BUY (started 2024-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 15:15:00 | 1829.40 | 1824.50 | 1824.02 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2024-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 09:15:00 | 1800.60 | 1819.72 | 1821.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-07 11:15:00 | 1788.00 | 1809.72 | 1816.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 09:15:00 | 1803.80 | 1798.92 | 1807.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-08 09:15:00 | 1803.80 | 1798.92 | 1807.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 1803.80 | 1798.92 | 1807.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:45:00 | 1801.65 | 1798.92 | 1807.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 10:15:00 | 1801.15 | 1799.37 | 1806.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 12:15:00 | 1796.45 | 1804.89 | 1806.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:00:00 | 1794.70 | 1802.25 | 1804.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 1819.45 | 1804.30 | 1805.21 | SL hit (close>static) qty=1.00 sl=1807.35 alert=retest2 |

### Cycle 99 — BUY (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 10:15:00 | 1822.50 | 1807.94 | 1806.78 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 15:15:00 | 1797.00 | 1806.37 | 1806.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 09:15:00 | 1775.65 | 1800.22 | 1804.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 10:15:00 | 1759.95 | 1755.54 | 1765.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 11:00:00 | 1759.95 | 1755.54 | 1765.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 1777.00 | 1759.83 | 1766.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:00:00 | 1777.00 | 1759.83 | 1766.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 12:15:00 | 1790.30 | 1765.93 | 1768.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:45:00 | 1790.00 | 1765.93 | 1768.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 14:15:00 | 1775.30 | 1771.62 | 1771.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 15:15:00 | 1810.90 | 1786.70 | 1780.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 12:15:00 | 1791.20 | 1794.48 | 1786.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 12:45:00 | 1790.85 | 1794.48 | 1786.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 13:15:00 | 1786.15 | 1792.81 | 1786.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 14:00:00 | 1786.15 | 1792.81 | 1786.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 1796.70 | 1793.59 | 1787.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-25 15:15:00 | 1810.00 | 1793.59 | 1787.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 09:15:00 | 1783.25 | 1794.15 | 1789.02 | SL hit (close<static) qty=1.00 sl=1785.35 alert=retest2 |

### Cycle 102 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 11:15:00 | 1766.30 | 1785.11 | 1785.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-26 12:15:00 | 1759.10 | 1779.90 | 1783.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 15:15:00 | 1741.90 | 1739.53 | 1750.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 09:15:00 | 1765.05 | 1739.53 | 1750.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 1763.10 | 1744.24 | 1751.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 09:45:00 | 1765.10 | 1744.24 | 1751.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 1784.80 | 1752.35 | 1754.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 1784.80 | 1752.35 | 1754.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 11:15:00 | 1788.70 | 1759.62 | 1757.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 12:15:00 | 1794.95 | 1766.69 | 1760.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 09:15:00 | 1790.60 | 1797.74 | 1786.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 10:00:00 | 1790.60 | 1797.74 | 1786.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 1793.65 | 1797.91 | 1792.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 14:00:00 | 1815.35 | 1801.96 | 1797.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 1818.35 | 1805.24 | 1799.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 09:15:00 | 1817.50 | 1806.14 | 1800.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 12:00:00 | 1814.95 | 1809.28 | 1803.33 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 1805.05 | 1809.46 | 1804.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:30:00 | 1804.90 | 1809.46 | 1804.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 1803.35 | 1808.24 | 1804.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:15:00 | 1812.40 | 1808.24 | 1804.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 10:15:00 | 1800.00 | 1806.18 | 1804.47 | SL hit (close<static) qty=1.00 sl=1803.20 alert=retest2 |

### Cycle 104 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 1797.05 | 1802.98 | 1803.52 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2024-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 14:15:00 | 1811.20 | 1804.63 | 1804.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 1814.45 | 1807.29 | 1805.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 15:15:00 | 1807.00 | 1813.25 | 1810.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 15:15:00 | 1807.00 | 1813.25 | 1810.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 1807.00 | 1813.25 | 1810.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:30:00 | 1819.15 | 1813.57 | 1810.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 11:15:00 | 1803.60 | 1810.63 | 1809.80 | SL hit (close<static) qty=1.00 sl=1805.80 alert=retest2 |

### Cycle 106 — SELL (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 12:15:00 | 1800.15 | 1808.54 | 1808.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 1788.65 | 1803.88 | 1806.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 11:15:00 | 1800.85 | 1799.96 | 1804.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 11:15:00 | 1800.85 | 1799.96 | 1804.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 1800.85 | 1799.96 | 1804.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 12:00:00 | 1800.85 | 1799.96 | 1804.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 1803.25 | 1800.61 | 1804.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 1803.25 | 1800.61 | 1804.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 1809.60 | 1802.41 | 1804.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:30:00 | 1811.70 | 1802.41 | 1804.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 1814.75 | 1804.88 | 1805.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 15:00:00 | 1814.75 | 1804.88 | 1805.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2024-12-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 15:15:00 | 1811.75 | 1806.25 | 1806.09 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 09:15:00 | 1799.65 | 1804.93 | 1805.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 10:15:00 | 1795.10 | 1802.97 | 1804.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 1808.55 | 1802.66 | 1804.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 12:15:00 | 1808.55 | 1802.66 | 1804.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 1808.55 | 1802.66 | 1804.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 1808.55 | 1802.66 | 1804.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 13:15:00 | 1814.60 | 1805.05 | 1805.01 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 10:15:00 | 1799.40 | 1805.26 | 1805.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 11:15:00 | 1791.65 | 1802.54 | 1804.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 09:15:00 | 1808.25 | 1797.79 | 1800.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 1808.25 | 1797.79 | 1800.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 1808.25 | 1797.79 | 1800.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:45:00 | 1812.15 | 1797.79 | 1800.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 1808.95 | 1800.02 | 1801.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:45:00 | 1808.70 | 1800.02 | 1801.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2024-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 12:15:00 | 1804.60 | 1802.20 | 1801.98 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2024-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 15:15:00 | 1799.00 | 1801.60 | 1801.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 09:15:00 | 1796.65 | 1800.61 | 1801.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 10:15:00 | 1802.80 | 1801.05 | 1801.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 10:15:00 | 1802.80 | 1801.05 | 1801.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 1802.80 | 1801.05 | 1801.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 10:45:00 | 1803.90 | 1801.05 | 1801.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2024-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 11:15:00 | 1806.60 | 1802.16 | 1801.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 13:15:00 | 1815.85 | 1806.24 | 1803.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 1811.25 | 1811.74 | 1807.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 1811.25 | 1811.74 | 1807.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 1811.25 | 1811.74 | 1807.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 09:45:00 | 1811.45 | 1811.74 | 1807.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 1808.35 | 1811.50 | 1808.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 12:00:00 | 1808.35 | 1811.50 | 1808.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 1797.30 | 1808.66 | 1807.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 13:00:00 | 1797.30 | 1808.66 | 1807.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 1807.00 | 1808.33 | 1807.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 14:15:00 | 1809.15 | 1808.33 | 1807.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 15:15:00 | 1808.50 | 1807.86 | 1806.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 10:15:00 | 1809.65 | 1807.93 | 1807.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 13:15:00 | 1874.65 | 1878.84 | 1879.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 13:15:00 | 1874.65 | 1878.84 | 1879.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 09:15:00 | 1862.35 | 1874.64 | 1877.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 15:15:00 | 1849.20 | 1846.71 | 1855.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:15:00 | 1853.60 | 1846.71 | 1855.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 1859.25 | 1849.22 | 1855.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 1855.40 | 1849.22 | 1855.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 1859.95 | 1851.36 | 1855.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:15:00 | 1861.00 | 1851.36 | 1855.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 1855.85 | 1852.26 | 1855.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 13:00:00 | 1854.30 | 1852.67 | 1855.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 09:15:00 | 1761.58 | 1790.10 | 1809.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-14 12:15:00 | 1762.65 | 1758.22 | 1775.09 | SL hit (close>ema200) qty=0.50 sl=1758.22 alert=retest2 |

### Cycle 115 — BUY (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 09:15:00 | 1781.10 | 1764.04 | 1763.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 11:15:00 | 1792.50 | 1780.36 | 1777.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 13:15:00 | 1780.40 | 1781.72 | 1778.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 13:15:00 | 1780.40 | 1781.72 | 1778.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 1780.40 | 1781.72 | 1778.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 14:15:00 | 1781.50 | 1781.72 | 1778.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 14:15:00 | 1762.25 | 1777.83 | 1776.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 15:00:00 | 1762.25 | 1777.83 | 1776.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 15:15:00 | 1764.40 | 1775.14 | 1775.71 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-22 09:15:00 | 1783.10 | 1776.73 | 1776.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-22 12:15:00 | 1796.35 | 1783.73 | 1779.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 09:15:00 | 1824.00 | 1825.11 | 1809.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-24 09:15:00 | 1824.00 | 1825.11 | 1809.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1824.00 | 1825.11 | 1809.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:30:00 | 1823.20 | 1825.11 | 1809.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 14:15:00 | 1818.30 | 1823.82 | 1815.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-24 14:45:00 | 1815.00 | 1823.82 | 1815.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 1801.30 | 1818.53 | 1814.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:00:00 | 1801.30 | 1818.53 | 1814.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 1799.00 | 1814.62 | 1812.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:30:00 | 1795.00 | 1814.62 | 1812.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 11:15:00 | 1791.00 | 1809.90 | 1810.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 12:15:00 | 1784.05 | 1804.73 | 1808.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 11:15:00 | 1730.40 | 1725.88 | 1749.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-29 12:00:00 | 1730.40 | 1725.88 | 1749.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 1743.55 | 1734.67 | 1740.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 15:00:00 | 1743.55 | 1734.67 | 1740.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 1737.15 | 1735.17 | 1740.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:30:00 | 1762.10 | 1739.27 | 1741.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 1746.40 | 1740.70 | 1742.31 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-01-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 11:15:00 | 1760.35 | 1744.63 | 1743.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 1766.25 | 1753.27 | 1748.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 1753.00 | 1754.75 | 1750.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 1753.00 | 1754.75 | 1750.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1753.00 | 1754.75 | 1750.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1753.00 | 1754.75 | 1750.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1756.70 | 1755.14 | 1751.29 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 1740.00 | 1748.68 | 1748.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 10:15:00 | 1722.40 | 1743.43 | 1746.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-03 12:15:00 | 1741.65 | 1740.43 | 1744.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 12:15:00 | 1741.65 | 1740.43 | 1744.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 12:15:00 | 1741.65 | 1740.43 | 1744.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-03 13:00:00 | 1741.65 | 1740.43 | 1744.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1745.95 | 1741.07 | 1743.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 1753.55 | 1741.07 | 1743.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 10:15:00 | 1760.95 | 1745.04 | 1745.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 12:15:00 | 1763.80 | 1751.37 | 1748.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 09:15:00 | 1748.15 | 1755.84 | 1751.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 1748.15 | 1755.84 | 1751.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 1748.15 | 1755.84 | 1751.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 1748.15 | 1755.84 | 1751.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 1743.60 | 1753.39 | 1751.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:15:00 | 1740.00 | 1753.39 | 1751.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 1744.55 | 1751.62 | 1750.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 12:45:00 | 1747.85 | 1750.95 | 1750.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 09:15:00 | 1743.20 | 1750.17 | 1750.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 1743.20 | 1750.17 | 1750.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 10:15:00 | 1737.65 | 1747.67 | 1749.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 13:15:00 | 1747.00 | 1746.97 | 1748.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 13:15:00 | 1747.00 | 1746.97 | 1748.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 1747.00 | 1746.97 | 1748.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 1748.10 | 1746.97 | 1748.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 1739.75 | 1745.52 | 1747.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 14:45:00 | 1746.00 | 1745.52 | 1747.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1733.65 | 1743.04 | 1746.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:30:00 | 1727.20 | 1741.66 | 1743.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 1722.30 | 1735.21 | 1739.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:30:00 | 1723.70 | 1732.32 | 1737.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 12:15:00 | 1748.25 | 1719.46 | 1717.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 1748.25 | 1719.46 | 1717.97 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-02-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 11:15:00 | 1700.25 | 1721.03 | 1721.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 12:15:00 | 1689.45 | 1714.72 | 1718.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 09:15:00 | 1718.00 | 1709.47 | 1714.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-17 09:15:00 | 1718.00 | 1709.47 | 1714.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 1718.00 | 1709.47 | 1714.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:15:00 | 1723.25 | 1709.47 | 1714.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 10:15:00 | 1727.05 | 1712.99 | 1715.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 10:45:00 | 1731.45 | 1712.99 | 1715.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 12:15:00 | 1715.00 | 1713.56 | 1715.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 12:45:00 | 1716.55 | 1713.56 | 1715.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 1710.00 | 1712.85 | 1714.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 1717.85 | 1712.85 | 1714.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1716.00 | 1713.48 | 1714.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1716.00 | 1713.48 | 1714.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1711.15 | 1713.01 | 1714.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:15:00 | 1727.55 | 1713.01 | 1714.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 1720.00 | 1714.41 | 1714.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:30:00 | 1724.00 | 1714.41 | 1714.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 1708.05 | 1713.14 | 1714.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:15:00 | 1705.15 | 1713.14 | 1714.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 13:00:00 | 1705.75 | 1709.78 | 1712.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 13:45:00 | 1704.25 | 1708.61 | 1711.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 09:15:00 | 1619.89 | 1640.20 | 1651.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 09:15:00 | 1620.46 | 1640.20 | 1651.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-25 09:15:00 | 1619.04 | 1640.20 | 1651.62 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-27 11:15:00 | 1625.90 | 1619.73 | 1631.06 | SL hit (close>ema200) qty=0.50 sl=1619.73 alert=retest2 |

### Cycle 125 — BUY (started 2025-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 11:15:00 | 1599.45 | 1584.66 | 1584.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 12:15:00 | 1613.05 | 1590.33 | 1586.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 15:15:00 | 1607.00 | 1609.08 | 1602.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 1609.55 | 1609.17 | 1603.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 1609.55 | 1609.17 | 1603.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:45:00 | 1609.75 | 1609.17 | 1603.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 1611.30 | 1610.07 | 1605.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 1608.65 | 1610.07 | 1605.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 1652.25 | 1618.49 | 1610.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:00:00 | 1659.25 | 1626.64 | 1614.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 11:45:00 | 1657.75 | 1633.50 | 1619.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 13:45:00 | 1656.35 | 1641.50 | 1625.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 14:30:00 | 1656.10 | 1644.27 | 1628.14 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 1785.40 | 1784.19 | 1773.00 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-03-26 09:15:00 | 1761.20 | 1768.11 | 1768.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 09:15:00 | 1761.20 | 1768.11 | 1768.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 10:15:00 | 1749.00 | 1764.29 | 1767.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 14:15:00 | 1739.65 | 1728.57 | 1736.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 14:15:00 | 1739.65 | 1728.57 | 1736.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 1739.65 | 1728.57 | 1736.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:00:00 | 1739.65 | 1728.57 | 1736.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 1733.05 | 1729.47 | 1736.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:15:00 | 1722.10 | 1729.47 | 1736.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 09:45:00 | 1722.80 | 1727.10 | 1734.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 1780.25 | 1724.67 | 1720.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 1780.25 | 1724.67 | 1720.78 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 1709.95 | 1730.06 | 1732.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 1649.50 | 1703.46 | 1717.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 1677.10 | 1674.62 | 1693.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 1685.95 | 1674.62 | 1693.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 1695.50 | 1681.32 | 1692.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:00:00 | 1695.50 | 1681.32 | 1692.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 1703.40 | 1685.73 | 1693.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 1705.15 | 1685.73 | 1693.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 1692.95 | 1687.18 | 1693.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 13:30:00 | 1695.90 | 1687.18 | 1693.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 1687.15 | 1687.17 | 1692.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 1663.50 | 1687.94 | 1692.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:45:00 | 1683.55 | 1676.25 | 1681.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-11 13:15:00 | 1690.90 | 1684.10 | 1683.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — BUY (started 2025-04-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 13:15:00 | 1690.90 | 1684.10 | 1683.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1718.20 | 1692.01 | 1687.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 13:15:00 | 1698.90 | 1699.38 | 1693.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 14:00:00 | 1698.90 | 1699.38 | 1693.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 09:15:00 | 1688.30 | 1698.60 | 1694.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 09:45:00 | 1705.90 | 1695.80 | 1694.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 11:15:00 | 1819.00 | 1826.93 | 1827.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 1819.00 | 1826.93 | 1827.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 13:15:00 | 1815.10 | 1823.18 | 1825.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-13 09:15:00 | 1705.60 | 1701.75 | 1725.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 09:15:00 | 1705.60 | 1701.75 | 1725.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 09:15:00 | 1705.60 | 1701.75 | 1725.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 10:15:00 | 1693.40 | 1701.75 | 1725.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-13 11:00:00 | 1693.90 | 1700.18 | 1722.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-14 11:00:00 | 1695.00 | 1700.11 | 1711.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 09:30:00 | 1689.70 | 1702.51 | 1708.34 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 1696.80 | 1701.37 | 1707.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:30:00 | 1705.70 | 1701.37 | 1707.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 1704.10 | 1701.92 | 1707.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 1704.10 | 1701.92 | 1707.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 1719.30 | 1705.39 | 1708.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:00:00 | 1719.30 | 1705.39 | 1708.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-15 13:15:00 | 1737.70 | 1711.86 | 1710.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 131 — BUY (started 2025-05-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 13:15:00 | 1737.70 | 1711.86 | 1710.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 14:15:00 | 1740.00 | 1717.48 | 1713.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 1735.10 | 1736.77 | 1730.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 1735.10 | 1736.77 | 1730.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 1729.10 | 1735.23 | 1729.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 1729.10 | 1735.23 | 1729.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 1729.00 | 1733.99 | 1729.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 1740.80 | 1733.99 | 1729.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 1722.60 | 1731.43 | 1729.99 | SL hit (close<static) qty=1.00 sl=1726.60 alert=retest2 |

### Cycle 132 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 1710.40 | 1727.22 | 1728.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 1708.40 | 1723.46 | 1726.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 1738.80 | 1725.01 | 1726.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 1738.80 | 1725.01 | 1726.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1738.80 | 1725.01 | 1726.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 09:30:00 | 1745.20 | 1725.01 | 1726.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 10:15:00 | 1740.70 | 1728.15 | 1727.80 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 1712.50 | 1727.67 | 1728.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 09:15:00 | 1670.90 | 1711.65 | 1719.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 15:15:00 | 1689.00 | 1687.16 | 1701.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-26 09:15:00 | 1688.20 | 1687.16 | 1701.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1688.30 | 1687.39 | 1699.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 11:00:00 | 1681.10 | 1686.13 | 1698.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 11:45:00 | 1681.30 | 1679.57 | 1686.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 12:15:00 | 1679.60 | 1679.57 | 1686.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 15:15:00 | 1681.50 | 1680.31 | 1685.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 1681.50 | 1680.55 | 1685.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 1683.10 | 1680.55 | 1685.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 1681.40 | 1680.72 | 1684.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 14:15:00 | 1667.60 | 1677.28 | 1681.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-28 14:45:00 | 1666.70 | 1674.83 | 1680.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-29 11:15:00 | 1687.00 | 1678.23 | 1680.01 | SL hit (close>static) qty=1.00 sl=1686.90 alert=retest2 |

### Cycle 135 — BUY (started 2025-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 13:15:00 | 1688.20 | 1682.17 | 1681.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 14:15:00 | 1699.50 | 1685.64 | 1683.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 10:15:00 | 1684.80 | 1688.37 | 1685.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 10:15:00 | 1684.80 | 1688.37 | 1685.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 1684.80 | 1688.37 | 1685.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 1684.80 | 1688.37 | 1685.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 1685.00 | 1687.70 | 1685.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 1684.10 | 1687.70 | 1685.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 1685.20 | 1687.20 | 1685.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 12:30:00 | 1685.70 | 1687.20 | 1685.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 1686.70 | 1687.10 | 1685.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:30:00 | 1682.60 | 1687.10 | 1685.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 1676.10 | 1684.90 | 1684.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 1676.10 | 1684.90 | 1684.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 1674.90 | 1682.90 | 1683.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 1661.50 | 1672.19 | 1676.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 1670.60 | 1670.09 | 1674.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:00:00 | 1670.60 | 1670.09 | 1674.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 13:15:00 | 1665.60 | 1662.94 | 1667.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:00:00 | 1665.60 | 1662.94 | 1667.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 1662.70 | 1662.89 | 1667.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 15:00:00 | 1662.70 | 1662.89 | 1667.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 1670.70 | 1664.63 | 1667.49 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 1687.30 | 1671.00 | 1670.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 13:15:00 | 1692.00 | 1683.99 | 1679.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 1685.50 | 1688.05 | 1683.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 10:15:00 | 1685.50 | 1688.05 | 1683.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 1685.50 | 1688.05 | 1683.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 1684.00 | 1688.05 | 1683.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1682.40 | 1686.92 | 1683.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:30:00 | 1682.00 | 1686.92 | 1683.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 1681.40 | 1685.82 | 1683.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 1680.70 | 1685.82 | 1683.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1690.10 | 1686.67 | 1683.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 1721.80 | 1688.57 | 1686.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 09:15:00 | 1679.00 | 1691.12 | 1690.97 | SL hit (close<static) qty=1.00 sl=1680.80 alert=retest2 |

### Cycle 138 — SELL (started 2025-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 10:15:00 | 1683.30 | 1689.56 | 1690.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-16 09:15:00 | 1675.40 | 1686.27 | 1688.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 1686.50 | 1683.60 | 1686.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 13:15:00 | 1686.50 | 1683.60 | 1686.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 1686.50 | 1683.60 | 1686.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:00:00 | 1686.50 | 1683.60 | 1686.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 1686.40 | 1684.16 | 1686.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 1685.80 | 1684.16 | 1686.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 1679.10 | 1683.14 | 1685.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 1665.50 | 1683.14 | 1685.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 13:15:00 | 1660.90 | 1655.17 | 1655.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 1660.90 | 1655.17 | 1655.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 1665.30 | 1657.19 | 1656.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 1652.20 | 1657.16 | 1656.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 1652.20 | 1657.16 | 1656.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 1652.20 | 1657.16 | 1656.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 1652.20 | 1657.16 | 1656.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1654.90 | 1656.70 | 1656.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 12:15:00 | 1660.40 | 1656.72 | 1656.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:15:00 | 1662.20 | 1656.82 | 1656.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 11:15:00 | 1653.40 | 1664.93 | 1665.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 1653.40 | 1664.93 | 1665.65 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2025-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 14:15:00 | 1669.30 | 1665.95 | 1665.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 09:15:00 | 1674.20 | 1667.67 | 1666.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 12:15:00 | 1684.80 | 1687.21 | 1679.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 13:00:00 | 1684.80 | 1687.21 | 1679.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 1686.90 | 1687.15 | 1680.28 | EMA400 retest candle locked (from upside) |

### Cycle 142 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 1665.10 | 1676.16 | 1676.77 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 12:15:00 | 1680.70 | 1673.92 | 1673.79 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 13:15:00 | 1671.20 | 1673.38 | 1673.55 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 14:15:00 | 1677.60 | 1674.22 | 1673.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 1689.60 | 1678.22 | 1675.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 12:15:00 | 1680.00 | 1680.38 | 1677.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 13:00:00 | 1680.00 | 1680.38 | 1677.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 1678.90 | 1680.69 | 1678.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 1678.80 | 1680.69 | 1678.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 1676.40 | 1679.84 | 1678.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:15:00 | 1675.40 | 1679.84 | 1678.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 1678.20 | 1679.51 | 1678.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 1672.90 | 1679.51 | 1678.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 1682.70 | 1680.15 | 1678.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 1680.70 | 1680.15 | 1678.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 11:15:00 | 1679.90 | 1680.10 | 1678.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 11:45:00 | 1678.60 | 1680.10 | 1678.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 1680.00 | 1680.08 | 1678.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:45:00 | 1679.50 | 1680.08 | 1678.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 13:15:00 | 1676.50 | 1679.36 | 1678.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 13:45:00 | 1676.00 | 1679.36 | 1678.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 1675.90 | 1678.67 | 1678.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 1675.90 | 1678.67 | 1678.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 1675.90 | 1678.12 | 1678.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 1670.90 | 1678.12 | 1678.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 09:15:00 | 1667.00 | 1675.89 | 1677.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 09:15:00 | 1661.30 | 1673.63 | 1675.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 13:15:00 | 1668.00 | 1667.79 | 1671.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:00:00 | 1668.00 | 1667.79 | 1671.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 1673.10 | 1668.86 | 1671.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 1673.10 | 1668.86 | 1671.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 1670.80 | 1669.24 | 1671.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 1674.80 | 1669.24 | 1671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 1668.20 | 1669.04 | 1671.29 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 11:15:00 | 1673.70 | 1667.27 | 1667.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 13:15:00 | 1674.60 | 1669.35 | 1668.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 14:15:00 | 1680.80 | 1681.91 | 1676.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 1680.80 | 1681.91 | 1676.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1701.00 | 1709.05 | 1702.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 1701.00 | 1709.05 | 1702.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1701.90 | 1707.62 | 1702.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 1714.20 | 1707.62 | 1702.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 1705.50 | 1707.19 | 1702.95 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 1688.50 | 1700.79 | 1702.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 09:15:00 | 1674.10 | 1690.52 | 1694.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 1685.50 | 1680.82 | 1685.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 11:15:00 | 1685.50 | 1680.82 | 1685.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 1685.50 | 1680.82 | 1685.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 12:00:00 | 1685.50 | 1680.82 | 1685.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 1686.00 | 1681.86 | 1685.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 13:15:00 | 1681.60 | 1681.86 | 1685.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 1692.50 | 1687.04 | 1686.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 1692.50 | 1687.04 | 1686.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 10:15:00 | 1697.40 | 1691.26 | 1688.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 15:15:00 | 1699.00 | 1700.12 | 1696.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:15:00 | 1712.70 | 1700.12 | 1696.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 1716.40 | 1725.29 | 1717.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:15:00 | 1734.70 | 1726.43 | 1718.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 12:00:00 | 1732.90 | 1727.73 | 1720.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:00:00 | 1735.00 | 1729.34 | 1722.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 1702.80 | 1724.03 | 1720.37 | SL hit (close<static) qty=1.00 sl=1713.40 alert=retest2 |

### Cycle 150 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 1638.70 | 1703.12 | 1711.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 13:15:00 | 1604.20 | 1620.31 | 1633.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1599.80 | 1592.40 | 1608.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 1599.80 | 1592.40 | 1608.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 1603.90 | 1592.88 | 1598.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 1603.90 | 1592.88 | 1598.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 1599.90 | 1594.28 | 1598.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:15:00 | 1596.40 | 1594.28 | 1598.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 1611.60 | 1600.98 | 1600.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 1611.60 | 1600.98 | 1600.46 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 11:15:00 | 1624.30 | 1629.38 | 1629.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 09:15:00 | 1618.30 | 1627.04 | 1628.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 13:15:00 | 1626.20 | 1626.13 | 1627.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 13:15:00 | 1626.20 | 1626.13 | 1627.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 13:15:00 | 1626.20 | 1626.13 | 1627.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 14:00:00 | 1626.20 | 1626.13 | 1627.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1632.80 | 1627.46 | 1627.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 1632.80 | 1627.46 | 1627.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — BUY (started 2025-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 15:15:00 | 1635.00 | 1628.97 | 1628.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 1645.00 | 1633.05 | 1630.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 14:15:00 | 1642.70 | 1643.61 | 1639.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 15:00:00 | 1642.70 | 1643.61 | 1639.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 1639.50 | 1642.79 | 1639.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 1635.30 | 1642.79 | 1639.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1641.60 | 1642.55 | 1639.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 1641.60 | 1642.55 | 1639.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1647.00 | 1643.44 | 1640.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 14:15:00 | 1649.60 | 1645.34 | 1642.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1619.80 | 1643.71 | 1642.52 | SL hit (close<static) qty=1.00 sl=1639.90 alert=retest2 |

### Cycle 154 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 1619.10 | 1638.79 | 1640.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 12:15:00 | 1610.10 | 1629.50 | 1635.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 12:15:00 | 1590.70 | 1587.48 | 1598.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 13:00:00 | 1590.70 | 1587.48 | 1598.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 13:15:00 | 1595.20 | 1589.03 | 1598.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 09:15:00 | 1590.10 | 1590.96 | 1597.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 10:00:00 | 1588.30 | 1590.43 | 1596.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 15:15:00 | 1580.00 | 1574.63 | 1574.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 1580.00 | 1574.63 | 1574.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 09:15:00 | 1586.90 | 1577.09 | 1575.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 11:15:00 | 1588.20 | 1590.05 | 1584.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:00:00 | 1588.20 | 1590.05 | 1584.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 1580.20 | 1588.79 | 1585.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 1580.20 | 1588.79 | 1585.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 1580.00 | 1587.03 | 1585.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:15:00 | 1589.30 | 1587.03 | 1585.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 1582.30 | 1585.33 | 1584.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:45:00 | 1582.50 | 1585.33 | 1584.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — SELL (started 2025-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 12:15:00 | 1582.50 | 1584.01 | 1584.07 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2025-09-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 13:15:00 | 1584.60 | 1584.12 | 1584.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 1591.30 | 1585.56 | 1584.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 10:15:00 | 1587.00 | 1589.08 | 1586.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 10:15:00 | 1587.00 | 1589.08 | 1586.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1587.00 | 1589.08 | 1586.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 1587.00 | 1589.08 | 1586.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1593.00 | 1589.86 | 1587.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 15:15:00 | 1596.00 | 1590.99 | 1588.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 1630.50 | 1640.42 | 1640.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 1630.50 | 1640.42 | 1640.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 1627.50 | 1632.98 | 1636.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 1639.90 | 1632.08 | 1634.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 10:15:00 | 1639.90 | 1632.08 | 1634.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1639.90 | 1632.08 | 1634.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 11:00:00 | 1639.90 | 1632.08 | 1634.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 11:15:00 | 1640.30 | 1633.72 | 1634.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 12:00:00 | 1640.30 | 1633.72 | 1634.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 1609.40 | 1596.42 | 1608.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 1614.80 | 1596.42 | 1608.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1611.60 | 1599.46 | 1608.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 1613.00 | 1599.46 | 1608.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 1603.00 | 1599.04 | 1602.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:30:00 | 1602.10 | 1599.04 | 1602.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 1594.70 | 1598.18 | 1602.15 | EMA400 retest candle locked (from downside) |

### Cycle 159 — BUY (started 2025-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 09:15:00 | 1630.70 | 1604.09 | 1603.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 13:15:00 | 1657.00 | 1640.26 | 1629.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 1645.20 | 1651.64 | 1644.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 1645.20 | 1651.64 | 1644.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 1645.20 | 1651.64 | 1644.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:30:00 | 1642.80 | 1651.64 | 1644.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1645.10 | 1650.34 | 1644.62 | EMA400 retest candle locked (from upside) |

### Cycle 160 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 1630.90 | 1641.59 | 1641.89 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-10-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 12:15:00 | 1653.30 | 1643.28 | 1642.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 1658.80 | 1647.96 | 1644.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 1663.50 | 1665.03 | 1657.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 09:45:00 | 1663.40 | 1665.03 | 1657.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 1663.30 | 1663.74 | 1658.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:15:00 | 1665.60 | 1663.74 | 1658.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:45:00 | 1665.10 | 1663.85 | 1658.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 13:15:00 | 1653.40 | 1659.63 | 1659.60 | SL hit (close<static) qty=1.00 sl=1655.80 alert=retest2 |

### Cycle 162 — SELL (started 2025-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 14:15:00 | 1654.70 | 1658.64 | 1659.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 09:15:00 | 1645.70 | 1653.81 | 1656.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 13:15:00 | 1655.20 | 1652.01 | 1654.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 13:15:00 | 1655.20 | 1652.01 | 1654.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 13:15:00 | 1655.20 | 1652.01 | 1654.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:00:00 | 1655.20 | 1652.01 | 1654.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 14:15:00 | 1659.60 | 1653.53 | 1654.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 14:30:00 | 1661.40 | 1653.53 | 1654.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 15:15:00 | 1656.40 | 1654.10 | 1655.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:15:00 | 1664.50 | 1654.10 | 1655.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2025-10-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 09:15:00 | 1670.80 | 1657.44 | 1656.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 1674.30 | 1660.82 | 1658.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 12:15:00 | 1695.00 | 1695.49 | 1686.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 13:00:00 | 1695.00 | 1695.49 | 1686.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 1686.70 | 1693.01 | 1687.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 1686.70 | 1693.01 | 1687.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 1695.00 | 1693.41 | 1687.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 1689.60 | 1693.41 | 1687.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 1692.40 | 1693.21 | 1688.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:30:00 | 1695.00 | 1693.21 | 1688.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 1690.20 | 1692.61 | 1688.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 1695.30 | 1693.61 | 1689.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 13:45:00 | 1695.70 | 1694.31 | 1690.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 1699.30 | 1695.31 | 1691.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 10:15:00 | 1693.60 | 1694.80 | 1691.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 1696.50 | 1695.14 | 1692.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:30:00 | 1700.20 | 1694.61 | 1693.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 1685.00 | 1692.69 | 1692.33 | SL hit (close<static) qty=1.00 sl=1687.00 alert=retest2 |

### Cycle 164 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 1681.00 | 1690.35 | 1691.30 | EMA200 below EMA400 |

### Cycle 165 — BUY (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 09:15:00 | 1699.50 | 1691.30 | 1691.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 10:15:00 | 1712.90 | 1695.62 | 1693.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1686.20 | 1703.71 | 1699.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1686.20 | 1703.71 | 1699.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1686.20 | 1703.71 | 1699.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 1684.50 | 1703.71 | 1699.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1695.60 | 1702.09 | 1699.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 11:15:00 | 1696.10 | 1702.09 | 1699.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:00:00 | 1695.80 | 1700.83 | 1699.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 12:30:00 | 1697.00 | 1700.63 | 1699.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 10:30:00 | 1696.00 | 1699.52 | 1699.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 11:15:00 | 1695.30 | 1698.68 | 1698.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 166 — SELL (started 2025-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 11:15:00 | 1695.30 | 1698.68 | 1698.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 1689.00 | 1695.66 | 1697.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 1696.70 | 1695.12 | 1696.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 1696.70 | 1695.12 | 1696.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1696.70 | 1695.12 | 1696.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 1696.70 | 1695.12 | 1696.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 1695.90 | 1695.28 | 1696.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:45:00 | 1697.50 | 1695.28 | 1696.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 1699.70 | 1696.16 | 1696.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 1699.70 | 1696.16 | 1696.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 1695.50 | 1696.03 | 1696.82 | EMA400 retest candle locked (from downside) |

### Cycle 167 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 1705.00 | 1698.71 | 1697.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 15:15:00 | 1706.60 | 1700.29 | 1698.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 09:15:00 | 1696.80 | 1699.59 | 1698.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 09:15:00 | 1696.80 | 1699.59 | 1698.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1696.80 | 1699.59 | 1698.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 1696.80 | 1699.59 | 1698.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 1689.30 | 1697.53 | 1697.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 1684.40 | 1693.07 | 1695.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 15:15:00 | 1702.70 | 1694.10 | 1695.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 15:15:00 | 1702.70 | 1694.10 | 1695.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1702.70 | 1694.10 | 1695.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1704.60 | 1694.10 | 1695.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1672.20 | 1689.72 | 1693.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 1668.30 | 1689.72 | 1693.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 1696.40 | 1691.50 | 1691.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 169 — BUY (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 09:15:00 | 1696.40 | 1691.50 | 1691.35 | EMA200 above EMA400 |

### Cycle 170 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 1690.00 | 1691.20 | 1691.23 | EMA200 below EMA400 |

### Cycle 171 — BUY (started 2025-11-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 11:15:00 | 1698.30 | 1692.62 | 1691.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 12:15:00 | 1700.60 | 1694.22 | 1692.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 15:15:00 | 1692.80 | 1694.84 | 1693.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 15:15:00 | 1692.80 | 1694.84 | 1693.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 1692.80 | 1694.84 | 1693.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 1699.20 | 1694.84 | 1693.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 11:00:00 | 1698.40 | 1696.41 | 1694.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 1797.10 | 1810.15 | 1810.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1797.10 | 1810.15 | 1810.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 10:15:00 | 1796.10 | 1807.34 | 1809.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 1801.60 | 1800.53 | 1805.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 15:00:00 | 1801.60 | 1800.53 | 1805.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 1794.30 | 1799.99 | 1804.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:45:00 | 1785.30 | 1797.05 | 1802.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 11:15:00 | 1806.00 | 1803.27 | 1803.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2025-12-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 11:15:00 | 1806.00 | 1803.27 | 1803.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 13:15:00 | 1816.40 | 1806.49 | 1804.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 1809.40 | 1809.50 | 1806.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 1809.40 | 1809.50 | 1806.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 1809.40 | 1809.50 | 1806.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 1806.50 | 1809.50 | 1806.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 1809.00 | 1809.40 | 1806.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 1807.30 | 1809.40 | 1806.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1809.00 | 1809.32 | 1807.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 1809.00 | 1809.32 | 1807.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1808.30 | 1809.11 | 1807.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 1808.30 | 1809.11 | 1807.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1807.40 | 1808.77 | 1807.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1807.40 | 1808.77 | 1807.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1806.00 | 1808.22 | 1807.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 1801.80 | 1808.22 | 1807.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 1805.00 | 1807.57 | 1806.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 1792.70 | 1807.57 | 1806.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1798.00 | 1805.66 | 1806.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 1785.70 | 1796.23 | 1800.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 1795.10 | 1795.03 | 1798.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 12:15:00 | 1795.10 | 1795.03 | 1798.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1795.10 | 1795.03 | 1798.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:30:00 | 1795.00 | 1795.03 | 1798.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 14:15:00 | 1785.20 | 1783.52 | 1788.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 14:45:00 | 1790.20 | 1783.52 | 1788.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 1788.50 | 1784.52 | 1788.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 1786.70 | 1784.11 | 1787.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 1795.60 | 1786.41 | 1788.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 1795.60 | 1786.41 | 1788.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 1798.20 | 1788.77 | 1789.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 1798.20 | 1788.77 | 1789.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 175 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 1798.00 | 1790.61 | 1790.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 1806.40 | 1795.47 | 1792.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 10:15:00 | 1796.90 | 1798.32 | 1794.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 10:15:00 | 1796.90 | 1798.32 | 1794.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 1796.90 | 1798.32 | 1794.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 1794.60 | 1798.32 | 1794.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 1793.20 | 1798.96 | 1796.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:45:00 | 1792.60 | 1798.96 | 1796.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 1794.20 | 1798.01 | 1796.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 1785.00 | 1798.01 | 1796.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 1783.50 | 1795.11 | 1795.14 | EMA200 below EMA400 |

### Cycle 177 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 1797.10 | 1795.02 | 1794.89 | EMA200 above EMA400 |

### Cycle 178 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 1780.90 | 1792.20 | 1793.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 1777.60 | 1784.95 | 1788.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 12:15:00 | 1788.10 | 1784.05 | 1787.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 12:15:00 | 1788.10 | 1784.05 | 1787.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 1788.10 | 1784.05 | 1787.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:00:00 | 1788.10 | 1784.05 | 1787.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 1790.90 | 1785.42 | 1787.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 1790.90 | 1785.42 | 1787.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1749.90 | 1748.01 | 1756.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:30:00 | 1754.00 | 1748.01 | 1756.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 1755.70 | 1749.55 | 1756.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:00:00 | 1755.70 | 1749.55 | 1756.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 11:15:00 | 1758.00 | 1751.24 | 1756.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 11:30:00 | 1758.30 | 1751.24 | 1756.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 12:15:00 | 1763.80 | 1753.75 | 1757.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 13:00:00 | 1763.80 | 1753.75 | 1757.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 179 — BUY (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 15:15:00 | 1771.50 | 1761.32 | 1760.03 | EMA200 above EMA400 |

### Cycle 180 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 1753.40 | 1759.91 | 1760.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 09:15:00 | 1741.60 | 1756.25 | 1758.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 1725.00 | 1722.09 | 1730.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 12:00:00 | 1725.00 | 1722.09 | 1730.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 1721.00 | 1717.45 | 1721.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:45:00 | 1719.80 | 1717.45 | 1721.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1720.00 | 1717.96 | 1721.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1724.50 | 1717.96 | 1721.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1721.10 | 1718.59 | 1721.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:30:00 | 1713.40 | 1717.97 | 1720.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 1725.80 | 1720.83 | 1720.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 1725.80 | 1720.83 | 1720.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 1731.80 | 1723.03 | 1721.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 1734.00 | 1735.42 | 1729.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 1734.00 | 1735.42 | 1729.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 1730.60 | 1734.45 | 1730.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 15:00:00 | 1730.60 | 1734.45 | 1730.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 1725.70 | 1732.70 | 1729.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:15:00 | 1730.40 | 1732.70 | 1729.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1737.30 | 1733.62 | 1730.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:15:00 | 1744.60 | 1735.06 | 1731.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 1743.80 | 1736.39 | 1732.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 1749.20 | 1760.38 | 1760.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 1744.70 | 1757.24 | 1758.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — SELL (started 2026-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 11:15:00 | 1744.70 | 1757.24 | 1758.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 1738.60 | 1753.51 | 1756.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 1736.10 | 1731.10 | 1740.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 1736.10 | 1731.10 | 1740.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 1738.50 | 1732.58 | 1740.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 1743.40 | 1732.58 | 1740.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 1736.00 | 1733.27 | 1740.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 1728.00 | 1733.27 | 1740.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 1719.10 | 1730.43 | 1738.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 1714.60 | 1726.26 | 1734.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 1715.50 | 1723.23 | 1731.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 1712.60 | 1720.82 | 1728.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1628.87 | 1676.50 | 1692.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1629.72 | 1676.50 | 1692.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 09:15:00 | 1626.97 | 1676.50 | 1692.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-19 11:15:00 | 1680.10 | 1675.89 | 1689.32 | SL hit (close>ema200) qty=0.50 sl=1675.89 alert=retest2 |

### Cycle 183 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 1644.30 | 1634.90 | 1634.71 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 1631.50 | 1634.22 | 1634.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 1627.40 | 1631.84 | 1633.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 1640.10 | 1633.49 | 1633.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 1640.10 | 1633.49 | 1633.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 1640.10 | 1633.49 | 1633.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 1640.10 | 1633.49 | 1633.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 185 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 1639.00 | 1634.59 | 1634.30 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 1619.10 | 1631.49 | 1632.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 11:15:00 | 1615.00 | 1626.05 | 1630.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 12:15:00 | 1596.60 | 1593.79 | 1602.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 13:00:00 | 1596.60 | 1593.79 | 1602.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 1603.60 | 1595.75 | 1603.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 1603.60 | 1595.75 | 1603.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 1595.90 | 1595.78 | 1602.37 | EMA400 retest candle locked (from downside) |

### Cycle 187 — BUY (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 11:15:00 | 1618.40 | 1606.37 | 1605.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 1630.00 | 1616.45 | 1611.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 1693.80 | 1700.61 | 1682.20 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 12:30:00 | 1703.20 | 1700.15 | 1686.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 14:30:00 | 1704.80 | 1700.90 | 1689.21 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1693.40 | 1699.75 | 1690.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 1690.70 | 1699.75 | 1690.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 1693.90 | 1697.21 | 1691.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 12:45:00 | 1693.60 | 1697.21 | 1691.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 1707.60 | 1708.18 | 1703.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 13:30:00 | 1707.20 | 1708.18 | 1703.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 1707.60 | 1708.06 | 1704.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 1704.40 | 1708.06 | 1704.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 1706.00 | 1707.65 | 1704.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 1711.90 | 1707.65 | 1704.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:15:00 | 1710.70 | 1707.78 | 1704.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 1710.40 | 1708.84 | 1705.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 13:15:00 | 1711.50 | 1708.84 | 1705.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 1706.00 | 1709.00 | 1706.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-11 15:15:00 | 1706.00 | 1709.00 | 1706.84 | SL hit (close<ema400) qty=1.00 sl=1706.84 alert=retest1 |

### Cycle 188 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 1698.00 | 1706.92 | 1708.05 | EMA200 below EMA400 |

### Cycle 189 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 1715.80 | 1706.75 | 1705.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 1717.60 | 1708.92 | 1706.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 1720.10 | 1721.79 | 1717.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 11:00:00 | 1720.10 | 1721.79 | 1717.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 1713.00 | 1720.15 | 1718.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 1713.00 | 1720.15 | 1718.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 1713.00 | 1718.72 | 1717.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 1709.60 | 1718.72 | 1717.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 1720.40 | 1718.64 | 1717.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 13:30:00 | 1722.20 | 1720.45 | 1718.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 1723.70 | 1722.67 | 1720.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 15:15:00 | 1742.90 | 1758.08 | 1759.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 190 — SELL (started 2026-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 15:15:00 | 1742.90 | 1758.08 | 1759.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1728.20 | 1752.10 | 1756.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 13:15:00 | 1748.40 | 1747.90 | 1752.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 14:00:00 | 1748.40 | 1747.90 | 1752.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 14:15:00 | 1752.90 | 1748.90 | 1752.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 14:45:00 | 1752.20 | 1748.90 | 1752.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 15:15:00 | 1753.20 | 1749.76 | 1752.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 1739.50 | 1749.76 | 1752.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 1779.30 | 1753.70 | 1752.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 191 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 1779.30 | 1753.70 | 1752.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 1804.10 | 1783.51 | 1771.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 1776.50 | 1789.72 | 1780.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 1776.50 | 1789.72 | 1780.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 1776.50 | 1789.72 | 1780.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-09 12:45:00 | 1799.30 | 1792.65 | 1783.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 1801.20 | 1817.22 | 1817.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 1801.20 | 1817.22 | 1817.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 1795.00 | 1807.49 | 1812.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 1792.50 | 1786.09 | 1796.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 1792.50 | 1786.09 | 1796.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1792.50 | 1786.09 | 1796.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 1795.90 | 1786.09 | 1796.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1791.00 | 1787.07 | 1795.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 1796.90 | 1787.07 | 1795.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 1803.50 | 1789.87 | 1795.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 1803.50 | 1789.87 | 1795.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 1805.40 | 1792.97 | 1796.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 1805.40 | 1792.97 | 1796.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 1795.50 | 1793.87 | 1796.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:45:00 | 1792.10 | 1794.01 | 1795.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 13:15:00 | 1791.40 | 1794.01 | 1795.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 1782.40 | 1767.10 | 1766.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 1782.40 | 1767.10 | 1766.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 1791.30 | 1771.94 | 1769.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 12:15:00 | 1794.70 | 1797.06 | 1786.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 13:00:00 | 1794.70 | 1797.06 | 1786.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 1789.80 | 1795.84 | 1788.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 1789.80 | 1795.84 | 1788.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1779.70 | 1792.41 | 1787.79 | EMA400 retest candle locked (from upside) |

### Cycle 194 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 1774.40 | 1783.51 | 1784.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 1753.40 | 1777.49 | 1781.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1777.80 | 1775.07 | 1779.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1777.80 | 1775.07 | 1779.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1777.80 | 1775.07 | 1779.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 1784.00 | 1775.07 | 1779.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1758.30 | 1771.72 | 1777.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 12:45:00 | 1753.00 | 1763.74 | 1773.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 1665.35 | 1728.82 | 1752.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 1689.20 | 1685.37 | 1708.70 | SL hit (close>ema200) qty=0.50 sl=1685.37 alert=retest2 |

### Cycle 195 — BUY (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 15:15:00 | 1721.00 | 1710.44 | 1709.96 | EMA200 above EMA400 |

### Cycle 196 — SELL (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-08 09:15:00 | 1694.10 | 1707.17 | 1708.52 | EMA200 below EMA400 |

### Cycle 197 — BUY (started 2026-04-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 14:15:00 | 1714.10 | 1709.07 | 1708.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 1725.00 | 1712.89 | 1710.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 13:15:00 | 1712.70 | 1715.79 | 1712.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 13:15:00 | 1712.70 | 1715.79 | 1712.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 1712.70 | 1715.79 | 1712.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:45:00 | 1713.60 | 1715.79 | 1712.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 1716.50 | 1715.93 | 1713.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 1713.20 | 1715.93 | 1713.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — SELL (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 09:15:00 | 1656.70 | 1703.94 | 1708.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-10 10:15:00 | 1640.00 | 1691.15 | 1702.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 11:15:00 | 1659.70 | 1658.94 | 1674.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-13 12:00:00 | 1659.70 | 1658.94 | 1674.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 1670.00 | 1659.34 | 1668.84 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 13:15:00 | 1689.10 | 1673.68 | 1673.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 1697.40 | 1678.43 | 1675.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 11:15:00 | 1675.90 | 1684.20 | 1679.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 11:15:00 | 1675.90 | 1684.20 | 1679.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 1675.90 | 1684.20 | 1679.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:45:00 | 1680.40 | 1684.20 | 1679.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 1679.30 | 1683.22 | 1679.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:30:00 | 1671.30 | 1683.22 | 1679.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 1683.00 | 1683.18 | 1680.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 1687.50 | 1684.76 | 1681.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:45:00 | 1689.30 | 1688.21 | 1683.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 11:15:00 | 1676.80 | 1684.77 | 1682.58 | SL hit (close<static) qty=1.00 sl=1677.00 alert=retest2 |

### Cycle 200 — SELL (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 13:15:00 | 1675.90 | 1681.11 | 1681.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 11:15:00 | 1670.20 | 1675.55 | 1678.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 1673.90 | 1671.62 | 1674.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 1673.90 | 1671.62 | 1674.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 1673.90 | 1671.62 | 1674.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 1674.30 | 1671.62 | 1674.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 1674.80 | 1671.34 | 1673.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:00:00 | 1674.80 | 1671.34 | 1673.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 1670.30 | 1671.13 | 1673.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:45:00 | 1675.30 | 1671.13 | 1673.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 1664.90 | 1669.89 | 1672.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:15:00 | 1656.20 | 1667.50 | 1671.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 12:15:00 | 1659.00 | 1664.82 | 1669.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 1695.60 | 1672.45 | 1671.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — BUY (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 09:15:00 | 1695.60 | 1672.45 | 1671.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 12:15:00 | 1702.10 | 1683.90 | 1677.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 14:15:00 | 1677.70 | 1684.71 | 1678.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 14:15:00 | 1677.70 | 1684.71 | 1678.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 14:15:00 | 1677.70 | 1684.71 | 1678.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 15:00:00 | 1677.70 | 1684.71 | 1678.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 15:15:00 | 1678.50 | 1683.47 | 1678.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-24 09:15:00 | 1654.00 | 1683.47 | 1678.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 1640.00 | 1674.77 | 1675.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1630.60 | 1665.94 | 1671.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 1737.50 | 1656.12 | 1660.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 1737.50 | 1656.12 | 1660.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1737.50 | 1656.12 | 1660.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 1737.50 | 1656.12 | 1660.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 1722.30 | 1669.36 | 1665.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 14:15:00 | 1780.10 | 1758.29 | 1738.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 1804.00 | 1812.98 | 1794.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-05 10:00:00 | 1804.00 | 1812.98 | 1794.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 1829.70 | 1838.86 | 1827.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:00:00 | 1829.70 | 1838.86 | 1827.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 1845.30 | 1838.75 | 1832.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 10:30:00 | 1850.40 | 1841.50 | 1834.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 11:45:00 | 1850.10 | 1843.32 | 1835.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 12:30:00 | 1850.10 | 1844.65 | 1836.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 13:15:00 | 1851.10 | 1844.65 | 1836.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-18 10:15:00 | 929.00 | 2023-05-22 12:15:00 | 938.80 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2023-05-18 12:30:00 | 925.30 | 2023-05-22 12:15:00 | 938.80 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2023-05-18 15:00:00 | 928.90 | 2023-05-22 12:15:00 | 938.80 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2023-05-19 09:45:00 | 927.85 | 2023-05-22 12:15:00 | 938.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2023-05-26 10:15:00 | 947.20 | 2023-06-08 12:15:00 | 995.05 | STOP_HIT | 1.00 | 5.05% |
| BUY | retest2 | 2023-05-26 14:15:00 | 948.00 | 2023-06-08 12:15:00 | 995.05 | STOP_HIT | 1.00 | 4.96% |
| BUY | retest2 | 2023-05-26 14:45:00 | 959.80 | 2023-06-08 12:15:00 | 995.05 | STOP_HIT | 1.00 | 3.67% |
| SELL | retest2 | 2023-06-14 13:15:00 | 987.80 | 2023-06-16 14:15:00 | 992.55 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2023-06-14 14:15:00 | 988.80 | 2023-06-16 14:15:00 | 992.55 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2023-06-15 10:15:00 | 988.40 | 2023-06-16 14:15:00 | 992.55 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2023-06-15 11:30:00 | 988.45 | 2023-06-16 14:15:00 | 992.55 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2023-06-15 15:15:00 | 987.05 | 2023-06-16 14:15:00 | 992.55 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-06-21 14:45:00 | 990.15 | 2023-06-26 09:15:00 | 995.35 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2023-06-22 09:30:00 | 991.00 | 2023-06-26 09:15:00 | 995.35 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2023-06-22 10:15:00 | 990.15 | 2023-06-26 09:15:00 | 995.35 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2023-06-22 11:00:00 | 991.05 | 2023-06-26 09:15:00 | 995.35 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-06-28 11:00:00 | 1008.70 | 2023-07-07 12:15:00 | 1036.50 | STOP_HIT | 1.00 | 2.76% |
| BUY | retest2 | 2023-06-28 12:15:00 | 1006.15 | 2023-07-07 12:15:00 | 1036.50 | STOP_HIT | 1.00 | 3.02% |
| BUY | retest2 | 2023-07-26 09:30:00 | 1106.20 | 2023-08-02 13:15:00 | 1128.80 | STOP_HIT | 1.00 | 2.04% |
| SELL | retest2 | 2023-08-23 09:15:00 | 1120.35 | 2023-09-05 10:15:00 | 1116.60 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2023-09-08 12:15:00 | 1135.80 | 2023-09-21 11:15:00 | 1144.85 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2023-09-11 09:15:00 | 1136.65 | 2023-09-21 11:15:00 | 1144.85 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2023-09-11 15:00:00 | 1136.90 | 2023-09-21 11:15:00 | 1144.85 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2023-09-29 09:30:00 | 1139.25 | 2023-10-04 09:15:00 | 1127.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-10-09 11:00:00 | 1122.90 | 2023-10-11 09:15:00 | 1130.30 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2023-10-09 11:45:00 | 1122.85 | 2023-10-11 09:15:00 | 1130.30 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2023-10-09 12:30:00 | 1123.00 | 2023-10-11 09:15:00 | 1130.30 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2023-10-09 13:15:00 | 1123.20 | 2023-10-11 09:15:00 | 1130.30 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2023-10-12 09:15:00 | 1130.35 | 2023-10-12 09:15:00 | 1124.55 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2023-10-17 09:15:00 | 1137.45 | 2023-10-19 15:15:00 | 1136.60 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2023-10-17 14:30:00 | 1138.25 | 2023-10-19 15:15:00 | 1136.60 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2023-10-18 09:15:00 | 1141.85 | 2023-10-19 15:15:00 | 1136.60 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2023-10-31 09:45:00 | 1099.45 | 2023-11-01 15:15:00 | 1117.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2023-11-13 15:15:00 | 1178.10 | 2023-11-15 09:15:00 | 1174.00 | STOP_HIT | 1.00 | -0.35% |
| BUY | retest2 | 2023-11-15 13:30:00 | 1179.00 | 2023-11-24 12:15:00 | 1194.25 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2023-11-15 15:00:00 | 1181.35 | 2023-11-24 12:15:00 | 1194.25 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2023-11-29 11:15:00 | 1192.50 | 2023-11-29 15:15:00 | 1206.25 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest1 | 2023-12-04 13:15:00 | 1228.80 | 2023-12-07 09:15:00 | 1235.05 | STOP_HIT | 1.00 | 0.51% |
| BUY | retest2 | 2023-12-05 10:30:00 | 1234.70 | 2023-12-11 10:15:00 | 1232.65 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2023-12-05 13:00:00 | 1234.35 | 2023-12-11 10:15:00 | 1232.65 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2023-12-07 14:00:00 | 1235.05 | 2023-12-11 10:15:00 | 1232.65 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2023-12-08 14:30:00 | 1234.05 | 2023-12-11 10:15:00 | 1232.65 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2023-12-15 09:15:00 | 1238.25 | 2023-12-20 11:15:00 | 1238.65 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2024-01-01 12:00:00 | 1260.80 | 2024-01-16 11:15:00 | 1312.80 | STOP_HIT | 1.00 | 4.12% |
| BUY | retest2 | 2024-01-01 14:15:00 | 1261.10 | 2024-01-16 11:15:00 | 1312.80 | STOP_HIT | 1.00 | 4.10% |
| BUY | retest2 | 2024-01-02 09:15:00 | 1277.05 | 2024-01-16 11:15:00 | 1312.80 | STOP_HIT | 1.00 | 2.80% |
| BUY | retest2 | 2024-01-23 10:15:00 | 1383.70 | 2024-02-08 09:15:00 | 1500.84 | TARGET_HIT | 1.00 | 8.47% |
| BUY | retest2 | 2024-01-24 09:45:00 | 1386.00 | 2024-02-08 09:15:00 | 1502.38 | TARGET_HIT | 1.00 | 8.40% |
| BUY | retest2 | 2024-01-24 11:00:00 | 1381.75 | 2024-02-09 10:15:00 | 1519.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-01-24 13:45:00 | 1384.35 | 2024-02-09 11:15:00 | 1522.07 | TARGET_HIT | 1.00 | 9.95% |
| BUY | retest2 | 2024-01-25 13:30:00 | 1364.40 | 2024-02-09 11:15:00 | 1524.60 | TARGET_HIT | 1.00 | 11.74% |
| BUY | retest2 | 2024-01-25 14:45:00 | 1365.80 | 2024-02-09 11:15:00 | 1522.79 | TARGET_HIT | 1.00 | 11.49% |
| SELL | retest2 | 2024-02-16 13:30:00 | 1509.00 | 2024-02-19 11:15:00 | 1524.25 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-02-16 14:45:00 | 1508.65 | 2024-02-19 11:15:00 | 1524.25 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-02-16 15:15:00 | 1507.55 | 2024-02-19 11:15:00 | 1524.25 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2024-02-19 09:45:00 | 1508.75 | 2024-02-19 11:15:00 | 1524.25 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-02-22 13:45:00 | 1550.50 | 2024-03-01 13:15:00 | 1551.30 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2024-03-12 09:15:00 | 1597.95 | 2024-03-12 13:15:00 | 1589.90 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-03-12 11:00:00 | 1593.40 | 2024-03-12 13:15:00 | 1589.90 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-03-15 09:30:00 | 1564.20 | 2024-03-18 15:15:00 | 1569.35 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2024-03-18 14:00:00 | 1566.00 | 2024-03-18 15:15:00 | 1569.35 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2024-03-20 15:00:00 | 1549.00 | 2024-03-20 15:15:00 | 1556.70 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-03-21 09:30:00 | 1548.15 | 2024-03-21 10:15:00 | 1563.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2024-03-27 11:15:00 | 1607.00 | 2024-04-04 09:15:00 | 1598.75 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-03-27 15:15:00 | 1608.45 | 2024-04-04 09:15:00 | 1598.75 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-04-03 10:00:00 | 1613.95 | 2024-04-04 09:15:00 | 1598.75 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2024-04-10 09:15:00 | 1593.70 | 2024-04-15 09:15:00 | 1514.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-04-10 09:15:00 | 1593.70 | 2024-04-15 14:15:00 | 1540.55 | STOP_HIT | 0.50 | 3.34% |
| SELL | retest2 | 2024-04-12 09:15:00 | 1576.50 | 2024-04-22 14:15:00 | 1540.80 | STOP_HIT | 1.00 | 2.26% |
| BUY | retest2 | 2024-05-03 09:45:00 | 1533.85 | 2024-05-03 11:15:00 | 1512.75 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-05-03 10:30:00 | 1530.15 | 2024-05-03 11:15:00 | 1512.75 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-05-10 14:30:00 | 1499.50 | 2024-05-13 12:15:00 | 1518.50 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-05-16 12:00:00 | 1528.90 | 2024-05-16 12:15:00 | 1514.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-05-30 14:30:00 | 1450.40 | 2024-06-04 12:15:00 | 1377.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-31 11:00:00 | 1451.20 | 2024-06-04 12:15:00 | 1378.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-30 14:30:00 | 1450.40 | 2024-06-04 15:15:00 | 1453.80 | STOP_HIT | 0.50 | -0.23% |
| SELL | retest2 | 2024-05-31 11:00:00 | 1451.20 | 2024-06-04 15:15:00 | 1453.80 | STOP_HIT | 0.50 | -0.18% |
| SELL | retest2 | 2024-06-04 11:00:00 | 1449.25 | 2024-06-05 11:15:00 | 1481.90 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2024-06-04 12:00:00 | 1414.35 | 2024-06-05 11:15:00 | 1481.90 | STOP_HIT | 1.00 | -4.78% |
| BUY | retest2 | 2024-06-06 14:45:00 | 1472.00 | 2024-06-19 14:15:00 | 1503.35 | STOP_HIT | 1.00 | 2.13% |
| BUY | retest2 | 2024-07-02 10:45:00 | 1524.80 | 2024-07-16 15:15:00 | 1576.00 | STOP_HIT | 1.00 | 3.36% |
| BUY | retest2 | 2024-07-02 13:00:00 | 1525.00 | 2024-07-16 15:15:00 | 1576.00 | STOP_HIT | 1.00 | 3.34% |
| BUY | retest2 | 2024-07-03 09:45:00 | 1527.65 | 2024-07-16 15:15:00 | 1576.00 | STOP_HIT | 1.00 | 3.16% |
| BUY | retest2 | 2024-07-03 11:30:00 | 1527.20 | 2024-07-16 15:15:00 | 1576.00 | STOP_HIT | 1.00 | 3.20% |
| BUY | retest2 | 2024-07-09 11:15:00 | 1564.50 | 2024-07-16 15:15:00 | 1576.00 | STOP_HIT | 1.00 | 0.74% |
| BUY | retest2 | 2024-07-25 11:30:00 | 1622.50 | 2024-08-06 13:15:00 | 1711.20 | STOP_HIT | 1.00 | 5.47% |
| BUY | retest2 | 2024-08-09 09:15:00 | 1743.75 | 2024-08-16 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2024-08-09 10:30:00 | 1739.95 | 2024-08-16 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-08-12 09:30:00 | 1739.95 | 2024-08-16 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-08-12 10:30:00 | 1740.35 | 2024-08-16 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2024-08-13 11:00:00 | 1746.20 | 2024-08-16 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-08-13 11:30:00 | 1746.95 | 2024-08-16 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2024-08-14 09:15:00 | 1746.45 | 2024-08-16 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-08-16 09:15:00 | 1746.20 | 2024-08-16 10:15:00 | 1735.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest1 | 2024-08-20 09:15:00 | 1761.25 | 2024-08-22 14:15:00 | 1750.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2024-08-23 09:15:00 | 1761.35 | 2024-09-09 09:15:00 | 1821.85 | STOP_HIT | 1.00 | 3.43% |
| BUY | retest2 | 2024-09-12 10:15:00 | 1856.20 | 2024-09-18 12:15:00 | 1847.20 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-09-12 11:30:00 | 1855.40 | 2024-09-18 12:15:00 | 1847.20 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-09-12 14:00:00 | 1855.50 | 2024-09-18 12:15:00 | 1847.20 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-09-12 14:30:00 | 1858.95 | 2024-09-18 12:15:00 | 1847.20 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2024-09-16 11:15:00 | 1864.40 | 2024-09-18 12:15:00 | 1847.20 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-09-16 13:15:00 | 1864.55 | 2024-09-18 12:15:00 | 1847.20 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-09-17 10:15:00 | 1865.90 | 2024-09-18 12:15:00 | 1847.20 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2024-09-17 11:45:00 | 1864.25 | 2024-09-18 12:15:00 | 1847.20 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2024-09-19 12:15:00 | 1846.10 | 2024-09-20 12:15:00 | 1859.80 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-09-19 14:45:00 | 1845.85 | 2024-09-20 12:15:00 | 1859.80 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-09-20 09:30:00 | 1845.35 | 2024-09-20 12:15:00 | 1859.80 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-09-20 10:00:00 | 1844.35 | 2024-09-20 12:15:00 | 1859.80 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-09-24 13:45:00 | 1870.85 | 2024-10-03 14:15:00 | 1911.35 | STOP_HIT | 1.00 | 2.16% |
| BUY | retest2 | 2024-09-24 15:15:00 | 1875.00 | 2024-10-03 14:15:00 | 1911.35 | STOP_HIT | 1.00 | 1.94% |
| BUY | retest2 | 2024-09-25 15:15:00 | 1874.20 | 2024-10-03 14:15:00 | 1911.35 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2024-10-15 10:00:00 | 1903.25 | 2024-10-18 11:15:00 | 1911.15 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2024-10-16 10:30:00 | 1900.55 | 2024-10-18 11:15:00 | 1911.15 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2024-10-17 09:30:00 | 1900.40 | 2024-10-18 11:15:00 | 1911.15 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2024-10-28 09:30:00 | 1849.05 | 2024-10-28 10:15:00 | 1870.15 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-11-05 12:45:00 | 1800.25 | 2024-11-06 13:15:00 | 1834.75 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-11-05 13:30:00 | 1802.45 | 2024-11-06 13:15:00 | 1834.75 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-11-05 15:15:00 | 1801.50 | 2024-11-06 13:15:00 | 1834.75 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2024-11-11 12:15:00 | 1796.45 | 2024-11-12 09:15:00 | 1819.45 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2024-11-11 15:00:00 | 1794.70 | 2024-11-12 09:15:00 | 1819.45 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2024-11-25 15:15:00 | 1810.00 | 2024-11-26 09:15:00 | 1783.25 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-12-05 14:00:00 | 1815.35 | 2024-12-09 10:15:00 | 1800.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-12-05 15:00:00 | 1818.35 | 2024-12-10 10:15:00 | 1803.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-12-06 09:15:00 | 1817.50 | 2024-12-10 12:15:00 | 1797.05 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-12-06 12:00:00 | 1814.95 | 2024-12-10 12:15:00 | 1797.05 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2024-12-09 09:15:00 | 1812.40 | 2024-12-10 12:15:00 | 1797.05 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-12-10 09:15:00 | 1809.00 | 2024-12-10 12:15:00 | 1797.05 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-12-10 11:15:00 | 1808.00 | 2024-12-10 12:15:00 | 1797.05 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-12-12 09:30:00 | 1819.15 | 2024-12-12 11:15:00 | 1803.60 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-12-20 14:15:00 | 1809.15 | 2025-01-02 13:15:00 | 1874.65 | STOP_HIT | 1.00 | 3.62% |
| BUY | retest2 | 2024-12-20 15:15:00 | 1808.50 | 2025-01-02 13:15:00 | 1874.65 | STOP_HIT | 1.00 | 3.66% |
| BUY | retest2 | 2024-12-23 10:15:00 | 1809.65 | 2025-01-02 13:15:00 | 1874.65 | STOP_HIT | 1.00 | 3.59% |
| SELL | retest2 | 2025-01-07 13:00:00 | 1854.30 | 2025-01-13 09:15:00 | 1761.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 13:00:00 | 1854.30 | 2025-01-14 12:15:00 | 1762.65 | STOP_HIT | 0.50 | 4.94% |
| BUY | retest2 | 2025-02-05 12:45:00 | 1747.85 | 2025-02-06 09:15:00 | 1743.20 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-02-10 09:30:00 | 1727.20 | 2025-02-13 12:15:00 | 1748.25 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-02-11 09:15:00 | 1722.30 | 2025-02-13 12:15:00 | 1748.25 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-02-11 10:30:00 | 1723.70 | 2025-02-13 12:15:00 | 1748.25 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-02-18 11:15:00 | 1705.15 | 2025-02-25 09:15:00 | 1619.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-18 13:00:00 | 1705.75 | 2025-02-25 09:15:00 | 1620.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-18 13:45:00 | 1704.25 | 2025-02-25 09:15:00 | 1619.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-18 11:15:00 | 1705.15 | 2025-02-27 11:15:00 | 1625.90 | STOP_HIT | 0.50 | 4.65% |
| SELL | retest2 | 2025-02-18 13:00:00 | 1705.75 | 2025-02-27 11:15:00 | 1625.90 | STOP_HIT | 0.50 | 4.68% |
| SELL | retest2 | 2025-02-18 13:45:00 | 1704.25 | 2025-02-27 11:15:00 | 1625.90 | STOP_HIT | 0.50 | 4.60% |
| BUY | retest2 | 2025-03-11 11:00:00 | 1659.25 | 2025-03-26 09:15:00 | 1761.20 | STOP_HIT | 1.00 | 6.14% |
| BUY | retest2 | 2025-03-11 11:45:00 | 1657.75 | 2025-03-26 09:15:00 | 1761.20 | STOP_HIT | 1.00 | 6.24% |
| BUY | retest2 | 2025-03-11 13:45:00 | 1656.35 | 2025-03-26 09:15:00 | 1761.20 | STOP_HIT | 1.00 | 6.33% |
| BUY | retest2 | 2025-03-11 14:30:00 | 1656.10 | 2025-03-26 09:15:00 | 1761.20 | STOP_HIT | 1.00 | 6.35% |
| SELL | retest2 | 2025-04-01 09:15:00 | 1722.10 | 2025-04-03 09:15:00 | 1780.25 | STOP_HIT | 1.00 | -3.38% |
| SELL | retest2 | 2025-04-01 09:45:00 | 1722.80 | 2025-04-03 09:15:00 | 1780.25 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2025-04-09 09:15:00 | 1663.50 | 2025-04-11 13:15:00 | 1690.90 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-04-11 10:45:00 | 1683.55 | 2025-04-11 13:15:00 | 1690.90 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-04-17 09:45:00 | 1705.90 | 2025-05-06 11:15:00 | 1819.00 | STOP_HIT | 1.00 | 6.63% |
| SELL | retest2 | 2025-05-13 10:15:00 | 1693.40 | 2025-05-15 13:15:00 | 1737.70 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2025-05-13 11:00:00 | 1693.90 | 2025-05-15 13:15:00 | 1737.70 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-05-14 11:00:00 | 1695.00 | 2025-05-15 13:15:00 | 1737.70 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-05-15 09:30:00 | 1689.70 | 2025-05-15 13:15:00 | 1737.70 | STOP_HIT | 1.00 | -2.84% |
| BUY | retest2 | 2025-05-20 09:15:00 | 1740.80 | 2025-05-20 12:15:00 | 1722.60 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-05-26 11:00:00 | 1681.10 | 2025-05-29 11:15:00 | 1687.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2025-05-27 11:45:00 | 1681.30 | 2025-05-29 11:15:00 | 1687.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-05-27 12:15:00 | 1679.60 | 2025-05-29 13:15:00 | 1688.20 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-05-27 15:15:00 | 1681.50 | 2025-05-29 13:15:00 | 1688.20 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-05-28 14:15:00 | 1667.60 | 2025-05-29 13:15:00 | 1688.20 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-05-28 14:45:00 | 1666.70 | 2025-05-29 13:15:00 | 1688.20 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-06-12 09:15:00 | 1721.80 | 2025-06-13 09:15:00 | 1679.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2025-06-17 09:15:00 | 1665.50 | 2025-06-20 13:15:00 | 1660.90 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-06-23 12:15:00 | 1660.40 | 2025-06-26 11:15:00 | 1653.40 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-06-23 13:15:00 | 1662.20 | 2025-06-26 11:15:00 | 1653.40 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-07-23 13:15:00 | 1681.60 | 2025-07-24 13:15:00 | 1692.50 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-07-31 11:15:00 | 1734.70 | 2025-07-31 14:15:00 | 1702.80 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-31 12:00:00 | 1732.90 | 2025-07-31 14:15:00 | 1702.80 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-07-31 14:00:00 | 1735.00 | 2025-07-31 14:15:00 | 1702.80 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-08-11 12:15:00 | 1596.40 | 2025-08-11 15:15:00 | 1611.60 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-08-25 14:15:00 | 1649.60 | 2025-08-26 09:15:00 | 1619.80 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-01 09:15:00 | 1590.10 | 2025-09-04 15:15:00 | 1580.00 | STOP_HIT | 1.00 | 0.64% |
| SELL | retest2 | 2025-09-01 10:00:00 | 1588.30 | 2025-09-04 15:15:00 | 1580.00 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-09-10 15:15:00 | 1596.00 | 2025-09-23 10:15:00 | 1630.50 | STOP_HIT | 1.00 | 2.16% |
| BUY | retest2 | 2025-10-13 12:15:00 | 1665.60 | 2025-10-14 13:15:00 | 1653.40 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-10-13 12:45:00 | 1665.10 | 2025-10-14 13:15:00 | 1653.40 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-10-24 11:30:00 | 1695.30 | 2025-10-28 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-10-24 13:45:00 | 1695.70 | 2025-10-28 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-24 15:00:00 | 1699.30 | 2025-10-28 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-10-27 10:15:00 | 1693.60 | 2025-10-28 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-28 09:30:00 | 1700.20 | 2025-10-28 10:15:00 | 1685.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-30 11:15:00 | 1696.10 | 2025-10-31 11:15:00 | 1695.30 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-10-30 12:00:00 | 1695.80 | 2025-10-31 11:15:00 | 1695.30 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-10-30 12:30:00 | 1697.00 | 2025-10-31 11:15:00 | 1695.30 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-10-31 10:30:00 | 1696.00 | 2025-10-31 11:15:00 | 1695.30 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-11-06 10:15:00 | 1668.30 | 2025-11-10 09:15:00 | 1696.40 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-11-11 09:15:00 | 1699.20 | 2025-12-02 09:15:00 | 1797.10 | STOP_HIT | 1.00 | 5.76% |
| BUY | retest2 | 2025-11-11 11:00:00 | 1698.40 | 2025-12-02 09:15:00 | 1797.10 | STOP_HIT | 1.00 | 5.81% |
| SELL | retest2 | 2025-12-03 10:45:00 | 1785.30 | 2025-12-04 11:15:00 | 1806.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-01-01 09:30:00 | 1713.40 | 2026-01-02 09:15:00 | 1725.80 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2026-01-06 11:15:00 | 1744.60 | 2026-01-09 11:15:00 | 1744.70 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2026-01-06 12:15:00 | 1743.80 | 2026-01-09 11:15:00 | 1744.70 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2026-01-09 10:45:00 | 1749.20 | 2026-01-09 11:15:00 | 1744.70 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-01-13 11:45:00 | 1714.60 | 2026-01-19 09:15:00 | 1628.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1715.50 | 2026-01-19 09:15:00 | 1629.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1712.60 | 2026-01-19 09:15:00 | 1626.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:45:00 | 1714.60 | 2026-01-19 11:15:00 | 1680.10 | STOP_HIT | 0.50 | 2.01% |
| SELL | retest2 | 2026-01-13 14:00:00 | 1715.50 | 2026-01-19 11:15:00 | 1680.10 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2026-01-14 09:30:00 | 1712.60 | 2026-01-19 11:15:00 | 1680.10 | STOP_HIT | 0.50 | 1.90% |
| BUY | retest1 | 2026-02-05 12:30:00 | 1703.20 | 2026-02-11 15:15:00 | 1706.00 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest1 | 2026-02-05 14:30:00 | 1704.80 | 2026-02-11 15:15:00 | 1706.00 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2026-02-11 09:15:00 | 1711.90 | 2026-02-13 13:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-02-11 11:15:00 | 1710.70 | 2026-02-13 13:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-02-11 12:45:00 | 1710.40 | 2026-02-13 13:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2026-02-11 13:15:00 | 1711.50 | 2026-02-13 13:15:00 | 1702.80 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-02-12 13:15:00 | 1715.70 | 2026-02-13 14:15:00 | 1698.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-02-12 15:00:00 | 1716.20 | 2026-02-13 14:15:00 | 1698.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2026-02-20 13:30:00 | 1722.20 | 2026-02-27 15:15:00 | 1742.90 | STOP_HIT | 1.00 | 1.20% |
| BUY | retest2 | 2026-02-23 10:30:00 | 1723.70 | 2026-02-27 15:15:00 | 1742.90 | STOP_HIT | 1.00 | 1.11% |
| SELL | retest2 | 2026-03-04 09:15:00 | 1739.50 | 2026-03-05 09:15:00 | 1779.30 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-03-09 12:45:00 | 1799.30 | 2026-03-13 12:15:00 | 1801.20 | STOP_HIT | 1.00 | 0.11% |
| SELL | retest2 | 2026-03-18 12:45:00 | 1792.10 | 2026-03-25 10:15:00 | 1782.40 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2026-03-18 13:15:00 | 1791.40 | 2026-03-25 10:15:00 | 1782.40 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2026-04-01 12:45:00 | 1753.00 | 2026-04-02 09:15:00 | 1665.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 12:45:00 | 1753.00 | 2026-04-06 12:15:00 | 1689.20 | STOP_HIT | 0.50 | 3.64% |
| BUY | retest2 | 2026-04-16 14:30:00 | 1687.50 | 2026-04-17 11:15:00 | 1676.80 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-04-17 09:45:00 | 1689.30 | 2026-04-17 11:15:00 | 1676.80 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-04-22 10:15:00 | 1656.20 | 2026-04-23 09:15:00 | 1695.60 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2026-04-22 12:15:00 | 1659.00 | 2026-04-23 09:15:00 | 1695.60 | STOP_HIT | 1.00 | -2.21% |
