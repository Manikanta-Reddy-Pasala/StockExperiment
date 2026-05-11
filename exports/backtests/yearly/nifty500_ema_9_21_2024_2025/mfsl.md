# Max Financial Services Ltd. (MFSL)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1695.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 159 |
| ALERT1 | 101 |
| ALERT2 | 99 |
| ALERT2_SKIP | 52 |
| ALERT3 | 279 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 132 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 135 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 135 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 42 / 93
- **Target hits / Stop hits / Partials:** 2 / 132 / 1
- **Avg / median % per leg:** -0.19% / -0.74%
- **Sum % (uncompounded):** -25.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 66 | 22 | 33.3% | 2 | 64 | 0 | 0.52% | 34.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.77% | -2.8% |
| BUY @ 3rd Alert (retest2) | 65 | 22 | 33.8% | 2 | 63 | 0 | 0.57% | 36.8% |
| SELL (all) | 69 | 20 | 29.0% | 0 | 68 | 1 | -0.86% | -59.1% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.47% | 0.5% |
| SELL @ 3rd Alert (retest2) | 68 | 19 | 27.9% | 0 | 67 | 1 | -0.88% | -59.6% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -1.15% | -2.3% |
| retest2 (combined) | 133 | 41 | 30.8% | 2 | 130 | 1 | -0.17% | -22.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 15:15:00 | 977.00 | 970.30 | 969.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 985.00 | 973.24 | 970.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 1014.90 | 1016.24 | 1007.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 10:15:00 | 1006.85 | 1014.36 | 1007.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 1006.85 | 1014.36 | 1007.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:00:00 | 1006.85 | 1014.36 | 1007.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 1002.50 | 1011.99 | 1006.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:00:00 | 1002.50 | 1011.99 | 1006.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 1000.25 | 1009.64 | 1006.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:30:00 | 1001.90 | 1009.64 | 1006.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 1005.05 | 1007.12 | 1005.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 989.85 | 1007.12 | 1005.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 09:15:00 | 974.50 | 1000.60 | 1002.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 12:15:00 | 969.35 | 986.91 | 995.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 15:15:00 | 967.50 | 964.96 | 976.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 09:30:00 | 969.90 | 963.71 | 974.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 958.75 | 962.06 | 968.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 11:00:00 | 957.45 | 961.14 | 967.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 14:30:00 | 956.60 | 959.33 | 964.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-27 15:15:00 | 956.00 | 959.33 | 964.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 09:15:00 | 987.50 | 964.43 | 965.77 | SL hit (close>static) qty=1.00 sl=972.45 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 10:15:00 | 990.60 | 969.67 | 968.02 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 942.35 | 968.64 | 971.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 10:15:00 | 930.05 | 960.92 | 967.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 13:15:00 | 931.25 | 928.48 | 940.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 14:00:00 | 931.25 | 928.48 | 940.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 928.00 | 924.65 | 935.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 898.95 | 934.22 | 936.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 14:00:00 | 908.05 | 911.27 | 922.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 12:30:00 | 924.10 | 919.24 | 921.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-05 13:30:00 | 924.65 | 921.01 | 922.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-06 09:15:00 | 929.75 | 924.15 | 923.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 929.75 | 924.15 | 923.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 935.70 | 926.46 | 924.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 14:15:00 | 927.20 | 927.40 | 925.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 14:30:00 | 928.30 | 927.40 | 925.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 15:15:00 | 930.05 | 927.93 | 926.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 09:15:00 | 933.00 | 927.93 | 926.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 940.35 | 930.41 | 927.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 10:45:00 | 950.75 | 939.62 | 934.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 980.70 | 988.47 | 989.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 980.70 | 988.47 | 989.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 11:15:00 | 975.80 | 983.32 | 986.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 12:15:00 | 978.00 | 977.83 | 981.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 12:15:00 | 978.00 | 977.83 | 981.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 978.00 | 977.83 | 981.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 13:45:00 | 973.80 | 977.12 | 980.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 975.75 | 976.76 | 979.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 12:00:00 | 973.35 | 976.07 | 978.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 10:30:00 | 976.10 | 973.98 | 976.11 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 974.85 | 974.16 | 975.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:30:00 | 971.00 | 973.93 | 975.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 970.80 | 973.78 | 975.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 09:15:00 | 980.30 | 974.34 | 975.29 | SL hit (close>static) qty=1.00 sl=977.90 alert=retest2 |

### Cycle 7 — BUY (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 11:15:00 | 987.10 | 978.08 | 976.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 12:15:00 | 993.40 | 981.15 | 978.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 09:15:00 | 971.00 | 984.82 | 981.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 971.00 | 984.82 | 981.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 971.00 | 984.82 | 981.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 10:00:00 | 971.00 | 984.82 | 981.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 10:15:00 | 982.85 | 984.43 | 981.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 985.80 | 981.07 | 980.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:15:00 | 985.75 | 981.78 | 981.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 11:15:00 | 986.10 | 982.07 | 981.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 14:15:00 | 992.80 | 996.92 | 996.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 14:15:00 | 992.80 | 996.92 | 996.92 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 09:15:00 | 1000.30 | 996.85 | 996.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 10:15:00 | 1003.55 | 998.19 | 997.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 1006.90 | 1007.80 | 1003.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 10:00:00 | 1006.90 | 1007.80 | 1003.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 1004.00 | 1007.04 | 1003.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 998.50 | 1007.04 | 1003.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 1011.45 | 1007.92 | 1004.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 12:30:00 | 1028.15 | 1013.04 | 1006.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:30:00 | 1014.90 | 1022.38 | 1022.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 10:15:00 | 1022.80 | 1029.51 | 1030.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 1022.80 | 1029.51 | 1030.11 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 11:15:00 | 1026.25 | 1022.95 | 1022.83 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 1017.20 | 1021.80 | 1022.32 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 1030.15 | 1024.00 | 1023.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 15:15:00 | 1041.00 | 1027.40 | 1024.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 1109.55 | 1110.58 | 1093.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-26 15:00:00 | 1109.55 | 1110.58 | 1093.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 1097.70 | 1104.76 | 1098.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 15:00:00 | 1097.70 | 1104.76 | 1098.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 1097.50 | 1103.31 | 1098.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:15:00 | 1094.40 | 1103.31 | 1098.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 1086.30 | 1099.91 | 1097.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 10:00:00 | 1086.30 | 1099.91 | 1097.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 1092.40 | 1098.40 | 1096.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:30:00 | 1096.40 | 1097.96 | 1096.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 13:00:00 | 1096.00 | 1097.57 | 1096.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 13:30:00 | 1097.50 | 1097.35 | 1096.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 14:15:00 | 1087.00 | 1095.28 | 1095.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 14:15:00 | 1087.00 | 1095.28 | 1095.89 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 11:15:00 | 1103.55 | 1095.99 | 1095.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 12:15:00 | 1107.00 | 1098.20 | 1096.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 14:15:00 | 1103.75 | 1108.12 | 1104.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 14:15:00 | 1103.75 | 1108.12 | 1104.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 1103.75 | 1108.12 | 1104.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 1100.90 | 1108.12 | 1104.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 1099.95 | 1106.49 | 1104.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 1091.10 | 1106.49 | 1104.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1090.70 | 1103.33 | 1103.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:45:00 | 1085.60 | 1103.33 | 1103.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-08-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 10:15:00 | 1099.50 | 1102.56 | 1102.71 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-02 13:15:00 | 1106.80 | 1103.27 | 1102.98 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 1083.40 | 1099.73 | 1101.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 15:15:00 | 1056.95 | 1070.54 | 1079.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 13:15:00 | 1073.95 | 1067.59 | 1074.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 14:00:00 | 1073.95 | 1067.59 | 1074.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 1084.80 | 1071.03 | 1075.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 1084.80 | 1071.03 | 1075.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 1087.85 | 1074.40 | 1076.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:15:00 | 1076.05 | 1074.40 | 1076.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 10:15:00 | 1091.25 | 1077.86 | 1077.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 10:15:00 | 1091.25 | 1077.86 | 1077.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-08 12:15:00 | 1102.10 | 1084.68 | 1080.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 1097.65 | 1104.40 | 1095.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 13:00:00 | 1097.65 | 1104.40 | 1095.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 1100.00 | 1103.52 | 1095.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:30:00 | 1095.60 | 1103.52 | 1095.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 1092.50 | 1101.32 | 1095.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:45:00 | 1093.25 | 1101.32 | 1095.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 1096.60 | 1100.37 | 1095.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 1090.75 | 1100.37 | 1095.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1076.40 | 1095.58 | 1093.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 10:00:00 | 1076.40 | 1095.58 | 1093.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-08-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 10:15:00 | 1075.45 | 1091.55 | 1092.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 10:15:00 | 1069.95 | 1079.28 | 1084.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 15:15:00 | 1038.15 | 1021.38 | 1035.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 15:15:00 | 1038.15 | 1021.38 | 1035.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 1038.15 | 1021.38 | 1035.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 09:15:00 | 1006.80 | 1021.38 | 1035.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:15:00 | 1008.70 | 1019.11 | 1033.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-20 11:30:00 | 1009.50 | 1006.48 | 1015.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-21 10:15:00 | 1029.55 | 1018.81 | 1018.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 1029.55 | 1018.81 | 1018.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 11:15:00 | 1037.45 | 1022.54 | 1020.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 09:15:00 | 1051.45 | 1053.05 | 1043.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 1051.45 | 1053.05 | 1043.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 1078.60 | 1080.72 | 1073.01 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 09:15:00 | 1061.10 | 1070.32 | 1070.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 1040.10 | 1064.28 | 1068.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 1054.00 | 1048.57 | 1056.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 1054.00 | 1048.57 | 1056.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 1054.00 | 1048.57 | 1056.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:00:00 | 1054.00 | 1048.57 | 1056.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 1072.30 | 1053.32 | 1058.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 11:00:00 | 1072.30 | 1053.32 | 1058.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 11:15:00 | 1064.95 | 1055.64 | 1058.91 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-08-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 13:15:00 | 1077.30 | 1062.46 | 1061.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 1100.85 | 1073.55 | 1067.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 09:15:00 | 1124.10 | 1125.73 | 1114.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-05 09:15:00 | 1124.10 | 1125.73 | 1114.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 09:15:00 | 1124.10 | 1125.73 | 1114.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-11 11:30:00 | 1150.75 | 1140.01 | 1133.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 1150.55 | 1143.79 | 1137.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 11:45:00 | 1151.85 | 1146.34 | 1140.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 12:15:00 | 1151.80 | 1146.34 | 1140.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 1145.65 | 1148.21 | 1143.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 09:45:00 | 1144.45 | 1148.21 | 1143.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 1146.85 | 1147.43 | 1144.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 11:30:00 | 1145.00 | 1147.43 | 1144.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 1142.90 | 1146.53 | 1144.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:45:00 | 1142.50 | 1146.53 | 1144.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 1140.60 | 1145.34 | 1143.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 13:45:00 | 1140.80 | 1145.34 | 1143.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 1141.25 | 1144.52 | 1143.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 15:15:00 | 1141.00 | 1144.52 | 1143.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 15:15:00 | 1141.00 | 1143.82 | 1143.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:15:00 | 1139.55 | 1143.82 | 1143.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-09-16 09:15:00 | 1138.15 | 1142.68 | 1142.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 09:15:00 | 1138.15 | 1142.68 | 1142.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 10:15:00 | 1125.90 | 1139.33 | 1141.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 14:15:00 | 1136.00 | 1134.89 | 1138.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-16 14:45:00 | 1136.45 | 1134.89 | 1138.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 1135.85 | 1134.31 | 1137.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:15:00 | 1141.40 | 1134.31 | 1137.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 1142.65 | 1135.97 | 1137.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:45:00 | 1143.00 | 1135.97 | 1137.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 11:15:00 | 1140.00 | 1136.78 | 1137.92 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-09-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-17 13:15:00 | 1148.75 | 1140.68 | 1139.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-17 14:15:00 | 1151.00 | 1142.74 | 1140.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 12:15:00 | 1145.30 | 1147.46 | 1144.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-18 13:00:00 | 1145.30 | 1147.46 | 1144.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 1145.15 | 1146.99 | 1144.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 1145.90 | 1146.99 | 1144.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 1140.85 | 1145.77 | 1143.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 14:45:00 | 1140.95 | 1145.77 | 1143.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 1140.05 | 1144.62 | 1143.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 1160.45 | 1144.62 | 1143.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-01 14:15:00 | 1185.00 | 1187.62 | 1187.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 14:15:00 | 1185.00 | 1187.62 | 1187.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 15:15:00 | 1182.00 | 1186.49 | 1187.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-03 09:15:00 | 1190.50 | 1187.29 | 1187.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 09:15:00 | 1190.50 | 1187.29 | 1187.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 1190.50 | 1187.29 | 1187.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:30:00 | 1197.10 | 1187.29 | 1187.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 1186.65 | 1187.17 | 1187.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 11:30:00 | 1183.30 | 1186.66 | 1187.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 12:00:00 | 1184.65 | 1186.66 | 1187.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 13:00:00 | 1183.80 | 1186.09 | 1186.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 11:30:00 | 1181.50 | 1181.10 | 1183.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 12:15:00 | 1175.60 | 1180.00 | 1182.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-04 13:15:00 | 1166.70 | 1180.00 | 1182.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-07 09:45:00 | 1173.70 | 1172.14 | 1177.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-09 09:15:00 | 1192.10 | 1161.24 | 1162.65 | SL hit (close>static) qty=1.00 sl=1184.30 alert=retest2 |

### Cycle 27 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 1181.00 | 1165.20 | 1164.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 14:15:00 | 1200.80 | 1187.38 | 1184.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 1185.65 | 1188.12 | 1185.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 09:15:00 | 1185.65 | 1188.12 | 1185.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 1185.65 | 1188.12 | 1185.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:00:00 | 1185.65 | 1188.12 | 1185.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 1186.30 | 1187.75 | 1185.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:15:00 | 1181.70 | 1187.75 | 1185.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 1189.45 | 1188.09 | 1185.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:30:00 | 1182.75 | 1188.09 | 1185.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 1189.70 | 1188.41 | 1186.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:00:00 | 1189.70 | 1188.41 | 1186.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 1183.30 | 1188.96 | 1186.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:45:00 | 1182.40 | 1188.96 | 1186.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 1189.00 | 1188.97 | 1187.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:15:00 | 1198.05 | 1188.97 | 1187.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 1194.50 | 1190.07 | 1187.74 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 1180.25 | 1187.68 | 1188.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 13:15:00 | 1174.75 | 1184.48 | 1186.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 1185.00 | 1181.58 | 1184.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 1185.00 | 1181.58 | 1184.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1185.00 | 1181.58 | 1184.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:00:00 | 1185.00 | 1181.58 | 1184.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 1185.05 | 1182.28 | 1184.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:45:00 | 1183.95 | 1182.28 | 1184.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 1191.95 | 1184.21 | 1185.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 1191.95 | 1184.21 | 1185.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 12:15:00 | 1193.40 | 1186.05 | 1185.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 13:15:00 | 1200.00 | 1188.84 | 1187.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-21 11:15:00 | 1188.05 | 1192.18 | 1189.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 11:15:00 | 1188.05 | 1192.18 | 1189.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 1188.05 | 1192.18 | 1189.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:00:00 | 1188.05 | 1192.18 | 1189.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 1184.30 | 1190.60 | 1189.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:45:00 | 1181.95 | 1190.60 | 1189.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 1186.70 | 1189.82 | 1189.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 14:30:00 | 1193.75 | 1191.02 | 1189.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 15:00:00 | 1195.80 | 1191.02 | 1189.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 12:15:00 | 1179.55 | 1189.38 | 1189.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 12:15:00 | 1179.55 | 1189.38 | 1189.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 1171.40 | 1184.98 | 1187.56 | Break + close below crossover candle low |

### Cycle 31 — BUY (started 2024-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 09:15:00 | 1258.60 | 1197.95 | 1192.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 10:15:00 | 1270.00 | 1212.36 | 1199.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-25 11:15:00 | 1274.25 | 1283.94 | 1265.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-25 11:45:00 | 1272.00 | 1283.94 | 1265.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 13:15:00 | 1266.40 | 1278.39 | 1266.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-25 14:00:00 | 1266.40 | 1278.39 | 1266.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 14:15:00 | 1274.65 | 1277.64 | 1267.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 09:45:00 | 1276.20 | 1275.38 | 1267.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-28 11:00:00 | 1276.30 | 1275.56 | 1268.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 11:15:00 | 1263.65 | 1273.18 | 1268.21 | SL hit (close<static) qty=1.00 sl=1265.50 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 1251.65 | 1264.73 | 1265.84 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2024-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 11:15:00 | 1277.00 | 1267.18 | 1266.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 12:15:00 | 1288.90 | 1271.53 | 1268.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-29 15:15:00 | 1271.85 | 1273.25 | 1270.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-29 15:15:00 | 1271.85 | 1273.25 | 1270.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 15:15:00 | 1271.85 | 1273.25 | 1270.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 09:15:00 | 1264.90 | 1273.25 | 1270.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 1267.00 | 1272.00 | 1270.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 10:15:00 | 1263.40 | 1272.00 | 1270.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 1260.25 | 1269.65 | 1269.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 1260.25 | 1269.65 | 1269.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 11:15:00 | 1259.50 | 1267.62 | 1268.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 12:15:00 | 1250.95 | 1264.29 | 1266.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 11:15:00 | 1262.40 | 1258.93 | 1262.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 11:15:00 | 1262.40 | 1258.93 | 1262.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 1262.40 | 1258.93 | 1262.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:45:00 | 1266.85 | 1258.93 | 1262.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 1265.50 | 1260.25 | 1262.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 12:30:00 | 1263.55 | 1260.25 | 1262.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 13:15:00 | 1272.30 | 1262.66 | 1263.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:45:00 | 1270.85 | 1262.66 | 1263.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 14:15:00 | 1283.60 | 1266.85 | 1265.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 1289.35 | 1274.37 | 1269.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 1257.60 | 1272.08 | 1269.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 1257.60 | 1272.08 | 1269.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 1257.60 | 1272.08 | 1269.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 1257.60 | 1272.08 | 1269.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 1249.95 | 1267.65 | 1267.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 1249.95 | 1267.65 | 1267.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 1250.00 | 1264.12 | 1265.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 11:15:00 | 1230.00 | 1249.90 | 1257.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 1249.40 | 1247.08 | 1254.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 14:00:00 | 1249.40 | 1247.08 | 1254.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 1250.30 | 1247.72 | 1253.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 15:15:00 | 1244.05 | 1247.72 | 1253.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-06 09:30:00 | 1247.20 | 1244.21 | 1251.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-14 13:15:00 | 1231.95 | 1219.51 | 1218.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 13:15:00 | 1231.95 | 1219.51 | 1218.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 12:15:00 | 1235.20 | 1227.79 | 1223.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 10:15:00 | 1226.75 | 1231.49 | 1227.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 10:15:00 | 1226.75 | 1231.49 | 1227.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 1226.75 | 1231.49 | 1227.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 1226.75 | 1231.49 | 1227.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 1201.75 | 1225.54 | 1225.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-19 12:00:00 | 1201.75 | 1225.54 | 1225.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 12:15:00 | 1187.20 | 1217.87 | 1221.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 1168.70 | 1196.01 | 1208.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 1185.55 | 1172.57 | 1181.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 1185.55 | 1172.57 | 1181.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 1185.55 | 1172.57 | 1181.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:30:00 | 1188.90 | 1172.57 | 1181.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 1187.65 | 1175.58 | 1181.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 11:00:00 | 1187.65 | 1175.58 | 1181.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 1166.85 | 1170.51 | 1176.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 10:45:00 | 1167.65 | 1170.51 | 1176.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 12:15:00 | 1177.10 | 1172.22 | 1175.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:00:00 | 1177.10 | 1172.22 | 1175.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 13:15:00 | 1180.05 | 1173.79 | 1176.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-26 13:45:00 | 1181.30 | 1173.79 | 1176.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 15:15:00 | 1181.50 | 1176.26 | 1177.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:15:00 | 1193.70 | 1176.26 | 1177.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 1197.10 | 1180.43 | 1178.86 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 1137.55 | 1178.61 | 1181.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 13:15:00 | 1103.00 | 1163.49 | 1173.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 1129.60 | 1119.49 | 1131.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 10:00:00 | 1129.60 | 1119.49 | 1131.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 1123.30 | 1120.25 | 1130.70 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-04 09:15:00 | 1164.55 | 1139.18 | 1136.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 10:15:00 | 1170.00 | 1145.34 | 1139.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 09:15:00 | 1156.05 | 1160.58 | 1151.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-05 09:15:00 | 1156.05 | 1160.58 | 1151.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1156.05 | 1160.58 | 1151.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 1156.05 | 1160.58 | 1151.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 1154.70 | 1159.40 | 1151.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:45:00 | 1152.75 | 1159.40 | 1151.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 1170.90 | 1167.55 | 1159.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:30:00 | 1169.10 | 1167.55 | 1159.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 11:15:00 | 1166.10 | 1167.45 | 1160.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 11:45:00 | 1157.80 | 1167.45 | 1160.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 12:15:00 | 1169.00 | 1167.76 | 1161.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 12:45:00 | 1167.50 | 1167.76 | 1161.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 1160.45 | 1166.32 | 1162.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 1160.45 | 1166.32 | 1162.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 1167.70 | 1166.60 | 1162.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 1160.00 | 1166.60 | 1162.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 1161.95 | 1165.67 | 1162.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:30:00 | 1157.15 | 1165.67 | 1162.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 1173.00 | 1167.13 | 1163.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 13:15:00 | 1175.80 | 1168.44 | 1164.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 09:15:00 | 1158.00 | 1169.37 | 1170.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 09:15:00 | 1158.00 | 1169.37 | 1170.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 10:15:00 | 1151.60 | 1165.82 | 1168.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 1126.60 | 1122.38 | 1132.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 1126.60 | 1122.38 | 1132.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 1160.40 | 1130.57 | 1134.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 1160.40 | 1130.57 | 1134.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 1146.95 | 1138.58 | 1137.89 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 09:15:00 | 1129.00 | 1140.82 | 1142.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 15:15:00 | 1103.00 | 1116.83 | 1125.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1107.55 | 1105.36 | 1113.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 12:15:00 | 1111.65 | 1106.62 | 1112.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 12:15:00 | 1111.65 | 1106.62 | 1112.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 12:30:00 | 1113.55 | 1106.62 | 1112.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 13:15:00 | 1109.25 | 1107.14 | 1111.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:00:00 | 1101.25 | 1105.97 | 1110.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 12:00:00 | 1103.50 | 1105.59 | 1109.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 1115.85 | 1107.63 | 1109.11 | SL hit (close>static) qty=1.00 sl=1112.85 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 15:15:00 | 1115.05 | 1109.80 | 1109.16 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 1093.80 | 1106.60 | 1107.76 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 1116.20 | 1109.49 | 1108.74 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 09:15:00 | 1099.70 | 1108.28 | 1108.47 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 12:15:00 | 1111.15 | 1107.92 | 1107.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 1120.50 | 1110.44 | 1108.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 11:15:00 | 1118.65 | 1118.93 | 1114.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 12:00:00 | 1118.65 | 1118.93 | 1114.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 1113.70 | 1118.19 | 1114.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 1113.70 | 1118.19 | 1114.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 1111.40 | 1116.83 | 1114.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 1111.40 | 1116.83 | 1114.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 1110.10 | 1115.48 | 1114.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 1096.75 | 1115.48 | 1114.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 1084.80 | 1109.35 | 1111.39 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 14:15:00 | 1111.65 | 1103.72 | 1103.57 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 12:15:00 | 1091.90 | 1102.43 | 1103.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 14:15:00 | 1088.75 | 1098.01 | 1101.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 1105.90 | 1098.79 | 1100.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 1105.90 | 1098.79 | 1100.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 1105.90 | 1098.79 | 1100.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 1105.90 | 1098.79 | 1100.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 10:15:00 | 1095.95 | 1098.22 | 1100.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 12:45:00 | 1093.15 | 1097.80 | 1099.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 09:15:00 | 1038.49 | 1053.56 | 1064.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-15 15:15:00 | 1041.00 | 1038.82 | 1050.69 | SL hit (close>ema200) qty=0.50 sl=1038.82 alert=retest2 |

### Cycle 53 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 1080.35 | 1061.79 | 1059.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 1089.10 | 1069.47 | 1064.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 1067.50 | 1079.10 | 1073.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 1067.50 | 1079.10 | 1073.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 1067.50 | 1079.10 | 1073.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:30:00 | 1067.50 | 1079.10 | 1073.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 1062.00 | 1075.68 | 1072.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:45:00 | 1064.50 | 1075.68 | 1072.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 13:15:00 | 1072.90 | 1072.49 | 1071.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 13:30:00 | 1071.25 | 1072.49 | 1071.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 1077.45 | 1073.48 | 1071.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:30:00 | 1078.00 | 1073.48 | 1071.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 1075.00 | 1073.78 | 1072.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:15:00 | 1075.00 | 1073.78 | 1072.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 1068.10 | 1072.65 | 1071.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 1068.10 | 1072.65 | 1071.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 1063.45 | 1070.81 | 1071.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 13:15:00 | 1058.55 | 1067.04 | 1069.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1044.65 | 1040.26 | 1049.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 1044.65 | 1040.26 | 1049.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 1044.65 | 1040.26 | 1049.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:15:00 | 1050.85 | 1040.26 | 1049.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 1056.45 | 1043.50 | 1050.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:45:00 | 1057.40 | 1043.50 | 1050.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 1055.45 | 1045.89 | 1050.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 1048.80 | 1051.22 | 1052.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 11:15:00 | 1063.45 | 1054.42 | 1053.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 11:15:00 | 1063.45 | 1054.42 | 1053.59 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 13:15:00 | 1047.35 | 1052.52 | 1052.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 1038.45 | 1049.71 | 1051.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 15:15:00 | 1045.95 | 1042.48 | 1045.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 15:15:00 | 1045.95 | 1042.48 | 1045.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 15:15:00 | 1045.95 | 1042.48 | 1045.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 09:15:00 | 1042.30 | 1042.48 | 1045.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 1040.65 | 1042.11 | 1045.13 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 12:15:00 | 1055.75 | 1047.42 | 1047.01 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-28 15:15:00 | 1046.15 | 1046.81 | 1046.85 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 1057.65 | 1048.98 | 1047.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 10:15:00 | 1066.05 | 1052.40 | 1049.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 1069.30 | 1069.61 | 1062.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:45:00 | 1068.40 | 1069.61 | 1062.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1072.00 | 1110.99 | 1097.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:45:00 | 1078.35 | 1110.99 | 1097.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1113.80 | 1111.55 | 1099.25 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 1081.15 | 1099.30 | 1099.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 15:15:00 | 1078.45 | 1092.21 | 1096.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 1111.00 | 1095.97 | 1097.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 1111.00 | 1095.97 | 1097.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 1111.00 | 1095.97 | 1097.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 09:45:00 | 1099.85 | 1095.97 | 1097.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 1099.45 | 1096.67 | 1097.56 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-02-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 13:15:00 | 1100.55 | 1098.62 | 1098.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 09:15:00 | 1134.05 | 1105.51 | 1101.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 13:15:00 | 1116.10 | 1117.43 | 1109.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 14:00:00 | 1116.10 | 1117.43 | 1109.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 1103.65 | 1114.50 | 1110.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:30:00 | 1099.00 | 1114.50 | 1110.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 1113.70 | 1114.34 | 1110.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 11:45:00 | 1118.20 | 1114.00 | 1110.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 1116.55 | 1110.10 | 1109.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:15:00 | 1116.15 | 1111.54 | 1110.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:45:00 | 1117.00 | 1111.12 | 1110.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 1100.00 | 1108.89 | 1109.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 1100.00 | 1108.89 | 1109.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 11:15:00 | 1094.80 | 1106.08 | 1108.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 10:15:00 | 1076.55 | 1063.58 | 1077.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 10:15:00 | 1076.55 | 1063.58 | 1077.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 1076.55 | 1063.58 | 1077.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:00:00 | 1076.55 | 1063.58 | 1077.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 11:15:00 | 1093.80 | 1069.62 | 1078.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 11:45:00 | 1090.00 | 1069.62 | 1078.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 12:15:00 | 1097.50 | 1075.20 | 1080.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 12:30:00 | 1091.85 | 1075.20 | 1080.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 09:15:00 | 1097.20 | 1085.37 | 1084.24 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 15:15:00 | 1079.85 | 1084.92 | 1085.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 1069.45 | 1080.29 | 1082.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 11:15:00 | 1060.95 | 1059.14 | 1068.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 12:00:00 | 1060.95 | 1059.14 | 1068.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 1066.25 | 1061.23 | 1067.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:30:00 | 1071.75 | 1061.23 | 1067.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 1068.70 | 1062.72 | 1067.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 1068.70 | 1062.72 | 1067.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 1064.55 | 1063.09 | 1067.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 1048.65 | 1063.09 | 1067.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 12:30:00 | 1059.05 | 1059.62 | 1064.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 15:15:00 | 1057.00 | 1059.95 | 1063.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 10:45:00 | 1057.55 | 1061.60 | 1063.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 1063.55 | 1061.99 | 1063.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 12:30:00 | 1055.95 | 1061.56 | 1063.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 13:30:00 | 1056.85 | 1059.92 | 1062.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 10:00:00 | 1056.60 | 1050.09 | 1053.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-21 12:15:00 | 1056.45 | 1053.13 | 1054.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 12:15:00 | 1056.40 | 1053.78 | 1054.41 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-02-21 14:15:00 | 1056.55 | 1054.95 | 1054.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 14:15:00 | 1056.55 | 1054.95 | 1054.87 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-02-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 15:15:00 | 1050.00 | 1053.96 | 1054.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 1033.25 | 1049.82 | 1052.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 12:15:00 | 1019.00 | 1018.53 | 1026.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 13:00:00 | 1019.00 | 1018.53 | 1026.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 13:15:00 | 1028.70 | 1020.57 | 1026.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 13:45:00 | 1030.95 | 1020.57 | 1026.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 14:15:00 | 1028.00 | 1022.05 | 1027.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-27 14:45:00 | 1028.40 | 1022.05 | 1027.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 1025.00 | 1022.64 | 1026.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 1013.20 | 1022.64 | 1026.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 10:15:00 | 1014.60 | 1002.17 | 1001.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1014.60 | 1002.17 | 1001.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 13:15:00 | 1015.90 | 1008.35 | 1004.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 13:15:00 | 1037.20 | 1037.64 | 1030.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 14:00:00 | 1037.20 | 1037.64 | 1030.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 1058.50 | 1069.88 | 1059.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 1058.50 | 1069.88 | 1059.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 1064.20 | 1068.74 | 1059.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:30:00 | 1059.20 | 1068.74 | 1059.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 15:15:00 | 1065.00 | 1066.03 | 1060.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 09:15:00 | 1062.70 | 1066.03 | 1060.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 1059.65 | 1064.76 | 1060.37 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 1051.00 | 1057.15 | 1057.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 09:15:00 | 1044.70 | 1054.66 | 1056.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 15:15:00 | 1046.10 | 1044.57 | 1049.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 09:15:00 | 1066.60 | 1044.57 | 1049.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 1051.60 | 1045.97 | 1049.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 1060.50 | 1045.97 | 1049.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 1058.65 | 1048.51 | 1050.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 10:45:00 | 1061.25 | 1048.51 | 1050.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 1063.80 | 1051.57 | 1051.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:00:00 | 1063.80 | 1051.57 | 1051.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 1071.95 | 1055.64 | 1053.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 1074.25 | 1061.37 | 1056.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1143.90 | 1151.21 | 1137.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 09:45:00 | 1146.70 | 1151.21 | 1137.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 1146.50 | 1150.27 | 1138.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 10:30:00 | 1144.65 | 1150.27 | 1138.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 1143.40 | 1148.05 | 1139.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 12:30:00 | 1143.20 | 1148.05 | 1139.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 1141.00 | 1146.64 | 1139.63 | EMA400 retest candle locked (from upside) |

### Cycle 70 — SELL (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 10:15:00 | 1118.00 | 1134.04 | 1135.41 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 11:15:00 | 1146.45 | 1133.52 | 1132.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 1149.60 | 1139.51 | 1135.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 14:15:00 | 1146.30 | 1147.02 | 1142.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 14:30:00 | 1154.45 | 1147.02 | 1142.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 1140.95 | 1145.75 | 1142.60 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 09:15:00 | 1131.60 | 1140.37 | 1141.08 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 1146.25 | 1141.81 | 1141.58 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 10:15:00 | 1140.05 | 1141.69 | 1141.73 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-04-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 12:15:00 | 1145.50 | 1142.29 | 1141.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 15:15:00 | 1149.55 | 1145.31 | 1143.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 1142.95 | 1144.84 | 1143.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 1142.95 | 1144.84 | 1143.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 1142.95 | 1144.84 | 1143.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:30:00 | 1156.80 | 1148.67 | 1145.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 1127.15 | 1152.66 | 1150.18 | SL hit (close<static) qty=1.00 sl=1135.90 alert=retest2 |

### Cycle 76 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 1119.85 | 1146.10 | 1147.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 11:15:00 | 1108.70 | 1138.62 | 1143.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 1130.75 | 1130.22 | 1136.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 1130.75 | 1130.22 | 1136.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 1130.75 | 1130.22 | 1136.86 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 1150.45 | 1138.24 | 1137.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 13:15:00 | 1155.20 | 1141.63 | 1139.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 1159.55 | 1161.51 | 1152.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 14:00:00 | 1159.55 | 1161.51 | 1152.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 1225.90 | 1233.71 | 1227.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 1225.90 | 1233.71 | 1227.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 1231.60 | 1233.29 | 1227.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 1226.20 | 1233.29 | 1227.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 1236.00 | 1236.73 | 1232.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 09:15:00 | 1269.20 | 1239.77 | 1235.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 13:15:00 | 1284.30 | 1292.89 | 1293.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 13:15:00 | 1284.30 | 1292.89 | 1293.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 14:15:00 | 1279.40 | 1290.19 | 1292.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 1279.50 | 1272.96 | 1279.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 10:15:00 | 1279.50 | 1272.96 | 1279.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 1279.50 | 1272.96 | 1279.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 1279.50 | 1272.96 | 1279.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1281.50 | 1274.67 | 1279.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:15:00 | 1288.70 | 1274.67 | 1279.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1289.80 | 1277.70 | 1280.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 1289.80 | 1277.70 | 1280.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1286.90 | 1280.63 | 1281.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 1286.90 | 1280.63 | 1281.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 1285.80 | 1282.38 | 1282.05 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 1273.20 | 1280.63 | 1281.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 1261.10 | 1275.49 | 1278.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 14:15:00 | 1267.10 | 1265.93 | 1271.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 15:00:00 | 1267.10 | 1265.93 | 1271.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 1292.10 | 1271.97 | 1273.28 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 1292.30 | 1276.04 | 1275.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 15:15:00 | 1299.90 | 1288.45 | 1284.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 13:15:00 | 1355.50 | 1357.96 | 1343.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:30:00 | 1357.30 | 1357.96 | 1343.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 1380.80 | 1380.67 | 1372.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 1375.60 | 1380.67 | 1372.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 1385.10 | 1382.31 | 1374.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 14:15:00 | 1391.70 | 1384.16 | 1377.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-02 12:15:00 | 1530.87 | 1516.06 | 1504.02 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 15:15:00 | 1503.00 | 1510.81 | 1511.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 09:15:00 | 1495.90 | 1507.82 | 1510.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 11:15:00 | 1509.30 | 1507.67 | 1509.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 11:15:00 | 1509.30 | 1507.67 | 1509.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 1509.30 | 1507.67 | 1509.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 1509.30 | 1507.67 | 1509.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 1504.30 | 1506.99 | 1509.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-05 13:30:00 | 1495.50 | 1505.34 | 1508.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 10:15:00 | 1514.80 | 1507.16 | 1508.06 | SL hit (close>static) qty=1.00 sl=1512.80 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 15:15:00 | 1511.50 | 1508.68 | 1508.52 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-06-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 09:15:00 | 1499.60 | 1506.86 | 1507.71 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-06-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 13:15:00 | 1514.20 | 1508.62 | 1508.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 14:15:00 | 1520.10 | 1510.92 | 1509.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 11:15:00 | 1513.50 | 1515.70 | 1512.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 11:15:00 | 1513.50 | 1515.70 | 1512.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1513.50 | 1515.70 | 1512.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 1513.90 | 1515.70 | 1512.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 1516.20 | 1515.80 | 1513.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:30:00 | 1516.10 | 1515.80 | 1513.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 1530.70 | 1531.05 | 1525.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 10:45:00 | 1526.20 | 1531.05 | 1525.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 1520.10 | 1528.86 | 1525.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 1520.10 | 1528.86 | 1525.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 1520.20 | 1527.13 | 1524.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:45:00 | 1519.60 | 1527.13 | 1524.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 1513.90 | 1522.96 | 1523.07 | EMA200 below EMA400 |

### Cycle 87 — BUY (started 2025-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-13 15:15:00 | 1531.20 | 1523.75 | 1522.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 09:15:00 | 1551.00 | 1529.20 | 1525.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 14:15:00 | 1589.70 | 1592.16 | 1578.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 15:00:00 | 1589.70 | 1592.16 | 1578.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 1575.90 | 1588.25 | 1581.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 1575.90 | 1588.25 | 1581.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 1573.20 | 1585.24 | 1581.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 1569.40 | 1585.24 | 1581.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 1576.50 | 1580.70 | 1579.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:45:00 | 1592.20 | 1583.06 | 1580.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 14:00:00 | 1591.90 | 1586.32 | 1583.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 14:30:00 | 1591.00 | 1587.97 | 1584.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 12:15:00 | 1598.70 | 1593.82 | 1592.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 1621.40 | 1625.47 | 1618.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 1614.30 | 1625.47 | 1618.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 15:15:00 | 1634.00 | 1631.62 | 1625.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 09:15:00 | 1619.80 | 1631.62 | 1625.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 1634.90 | 1632.28 | 1626.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:15:00 | 1651.70 | 1637.85 | 1631.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 11:00:00 | 1650.90 | 1643.04 | 1635.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 15:00:00 | 1653.60 | 1650.56 | 1641.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:30:00 | 1649.70 | 1651.72 | 1643.96 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 1653.00 | 1651.97 | 1644.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:30:00 | 1645.90 | 1651.97 | 1644.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 1642.90 | 1649.96 | 1645.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:00:00 | 1642.90 | 1649.96 | 1645.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 1643.10 | 1648.59 | 1645.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:30:00 | 1638.90 | 1648.59 | 1645.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 1631.80 | 1645.17 | 1644.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 1631.80 | 1645.17 | 1644.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 1633.40 | 1642.81 | 1643.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 10:15:00 | 1633.40 | 1642.81 | 1643.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 11:15:00 | 1626.00 | 1639.45 | 1641.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 09:15:00 | 1568.60 | 1565.51 | 1573.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 09:15:00 | 1568.60 | 1565.51 | 1573.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1568.60 | 1565.51 | 1573.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 1572.10 | 1565.51 | 1573.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 1573.60 | 1567.13 | 1573.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 1573.60 | 1567.13 | 1573.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 1576.40 | 1568.98 | 1573.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 1576.40 | 1568.98 | 1573.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 1576.70 | 1570.52 | 1573.72 | EMA400 retest candle locked (from downside) |

### Cycle 89 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 1584.30 | 1576.24 | 1575.95 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 11:15:00 | 1564.50 | 1575.09 | 1575.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 1562.00 | 1572.47 | 1574.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 09:15:00 | 1579.10 | 1570.59 | 1572.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 1579.10 | 1570.59 | 1572.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 1579.10 | 1570.59 | 1572.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 1579.10 | 1570.59 | 1572.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 1585.50 | 1573.57 | 1573.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:45:00 | 1588.90 | 1573.57 | 1573.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 1577.80 | 1574.42 | 1574.09 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 10:15:00 | 1570.50 | 1573.97 | 1574.33 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 1585.00 | 1575.52 | 1574.79 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 12:15:00 | 1569.80 | 1574.26 | 1574.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 09:15:00 | 1542.20 | 1566.97 | 1571.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 1539.30 | 1538.84 | 1550.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 1544.50 | 1540.66 | 1545.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 1544.50 | 1540.66 | 1545.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:00:00 | 1544.50 | 1540.66 | 1545.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 1545.70 | 1541.67 | 1545.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 1542.60 | 1541.67 | 1545.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 1552.90 | 1543.92 | 1546.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 12:00:00 | 1552.90 | 1543.92 | 1546.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 12:15:00 | 1550.00 | 1545.13 | 1546.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 13:45:00 | 1547.60 | 1545.05 | 1546.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 1556.60 | 1548.20 | 1547.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 10:15:00 | 1556.60 | 1548.20 | 1547.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 12:15:00 | 1564.80 | 1552.77 | 1549.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 1556.30 | 1557.64 | 1553.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 1556.30 | 1557.64 | 1553.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1556.30 | 1557.64 | 1553.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 1556.30 | 1557.64 | 1553.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 1551.40 | 1556.39 | 1553.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 1550.60 | 1556.39 | 1553.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 11:15:00 | 1547.70 | 1554.65 | 1552.87 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 14:15:00 | 1550.00 | 1551.82 | 1551.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 1536.20 | 1548.57 | 1550.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 15:15:00 | 1531.60 | 1528.62 | 1533.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 09:15:00 | 1532.30 | 1528.62 | 1533.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1552.20 | 1533.34 | 1535.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 1552.20 | 1533.34 | 1535.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 1546.70 | 1536.01 | 1536.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 1540.10 | 1536.83 | 1536.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 1537.90 | 1537.04 | 1536.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-07-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 12:15:00 | 1537.90 | 1537.04 | 1536.97 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-07-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 13:15:00 | 1530.50 | 1535.73 | 1536.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 15:15:00 | 1524.00 | 1532.34 | 1534.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-30 14:15:00 | 1509.50 | 1500.73 | 1510.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 14:15:00 | 1509.50 | 1500.73 | 1510.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 14:15:00 | 1509.50 | 1500.73 | 1510.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 15:00:00 | 1509.50 | 1500.73 | 1510.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 1504.60 | 1501.50 | 1509.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:45:00 | 1498.30 | 1499.20 | 1508.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 1501.70 | 1498.73 | 1505.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 15:00:00 | 1502.30 | 1500.22 | 1505.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 15:15:00 | 1493.00 | 1482.34 | 1481.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 15:15:00 | 1493.00 | 1482.34 | 1481.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 12:15:00 | 1495.80 | 1488.47 | 1484.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-11 09:15:00 | 1533.40 | 1533.91 | 1517.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-11 09:45:00 | 1535.40 | 1533.91 | 1517.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1616.50 | 1629.69 | 1614.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 1609.70 | 1629.69 | 1614.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 1620.90 | 1627.93 | 1614.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 1615.50 | 1627.93 | 1614.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 1635.30 | 1632.46 | 1622.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 12:45:00 | 1643.90 | 1637.68 | 1627.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 14:30:00 | 1644.40 | 1639.31 | 1630.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 15:15:00 | 1645.40 | 1639.31 | 1630.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 1644.60 | 1652.89 | 1644.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 1646.10 | 1651.53 | 1644.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 1648.40 | 1651.53 | 1644.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 1650.70 | 1651.37 | 1645.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 11:15:00 | 1654.90 | 1645.97 | 1644.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 09:15:00 | 1636.10 | 1650.03 | 1648.62 | SL hit (close<static) qty=1.00 sl=1644.50 alert=retest2 |

### Cycle 100 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 1633.10 | 1645.12 | 1646.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 13:15:00 | 1629.10 | 1640.94 | 1644.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 10:15:00 | 1644.20 | 1636.42 | 1640.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 10:15:00 | 1644.20 | 1636.42 | 1640.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 1644.20 | 1636.42 | 1640.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 1644.20 | 1636.42 | 1640.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 1626.10 | 1634.36 | 1639.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:00:00 | 1623.90 | 1632.27 | 1637.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 1620.80 | 1618.55 | 1619.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 09:15:00 | 1640.00 | 1622.84 | 1621.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 09:15:00 | 1640.00 | 1622.84 | 1621.51 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 1611.50 | 1622.64 | 1622.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 10:15:00 | 1603.50 | 1618.81 | 1621.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 1615.30 | 1611.93 | 1616.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 1615.30 | 1611.93 | 1616.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 1615.30 | 1611.93 | 1616.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 1615.30 | 1611.93 | 1616.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 1620.00 | 1613.55 | 1616.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 1590.60 | 1613.55 | 1616.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 1603.10 | 1611.46 | 1615.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 11:45:00 | 1569.30 | 1599.04 | 1609.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 12:45:00 | 1570.00 | 1593.39 | 1605.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 10:30:00 | 1566.80 | 1581.66 | 1586.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 15:15:00 | 1596.30 | 1587.51 | 1587.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 1596.30 | 1587.51 | 1587.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 1609.70 | 1591.95 | 1589.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 1600.90 | 1602.76 | 1597.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 1600.90 | 1602.76 | 1597.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 1600.90 | 1602.76 | 1597.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 1605.50 | 1602.76 | 1597.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 1603.00 | 1602.81 | 1597.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:15:00 | 1604.70 | 1602.81 | 1597.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 11:45:00 | 1607.00 | 1602.73 | 1598.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:15:00 | 1604.90 | 1602.73 | 1598.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:45:00 | 1604.40 | 1603.10 | 1598.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1592.90 | 1602.22 | 1599.98 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 1592.90 | 1602.22 | 1599.98 | SL hit (close<static) qty=1.00 sl=1596.20 alert=retest2 |

### Cycle 104 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 1594.10 | 1598.58 | 1598.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 13:15:00 | 1588.30 | 1596.52 | 1597.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 15:15:00 | 1581.60 | 1581.33 | 1586.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 15:15:00 | 1581.60 | 1581.33 | 1586.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 1581.60 | 1581.33 | 1586.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 1577.80 | 1581.33 | 1586.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1572.80 | 1579.63 | 1585.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 1566.60 | 1577.22 | 1583.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:45:00 | 1563.70 | 1574.78 | 1582.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:15:00 | 1567.00 | 1573.92 | 1581.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:45:00 | 1566.70 | 1572.28 | 1579.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 1558.00 | 1552.95 | 1558.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 1557.20 | 1552.95 | 1558.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 1570.50 | 1556.46 | 1559.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:00:00 | 1570.50 | 1556.46 | 1559.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 1568.20 | 1558.80 | 1560.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 12:30:00 | 1557.80 | 1559.55 | 1560.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 14:00:00 | 1563.00 | 1560.24 | 1560.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 1564.20 | 1561.47 | 1561.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 1564.20 | 1561.47 | 1561.11 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 1555.70 | 1560.35 | 1560.67 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 10:15:00 | 1565.50 | 1561.38 | 1560.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 11:15:00 | 1575.80 | 1564.27 | 1562.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 10:15:00 | 1577.90 | 1578.29 | 1571.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 11:00:00 | 1577.90 | 1578.29 | 1571.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 1577.80 | 1581.67 | 1577.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 1577.80 | 1581.67 | 1577.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 1566.00 | 1578.54 | 1576.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 15:00:00 | 1566.00 | 1578.54 | 1576.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 1563.50 | 1575.53 | 1575.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 1560.30 | 1570.37 | 1573.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 1553.80 | 1550.72 | 1557.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 13:15:00 | 1553.80 | 1550.72 | 1557.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 1553.80 | 1550.72 | 1557.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 13:45:00 | 1553.30 | 1550.72 | 1557.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 1558.00 | 1552.18 | 1557.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 1558.00 | 1552.18 | 1557.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1564.00 | 1554.54 | 1558.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 1583.10 | 1554.54 | 1558.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 1574.80 | 1558.60 | 1559.84 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 1580.70 | 1564.65 | 1562.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 10:15:00 | 1585.40 | 1575.25 | 1569.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 13:15:00 | 1603.20 | 1612.78 | 1599.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 13:45:00 | 1604.70 | 1612.78 | 1599.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 1604.50 | 1609.88 | 1600.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:15:00 | 1601.50 | 1609.88 | 1600.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 1624.30 | 1612.77 | 1602.30 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 1585.80 | 1599.76 | 1601.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 1583.40 | 1592.34 | 1597.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 1581.50 | 1576.92 | 1584.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 1581.50 | 1576.92 | 1584.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 1587.00 | 1578.94 | 1584.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 1587.00 | 1578.94 | 1584.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 1592.00 | 1581.55 | 1585.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 1592.00 | 1581.55 | 1585.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 15:15:00 | 1596.00 | 1589.32 | 1588.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 09:15:00 | 1598.40 | 1591.13 | 1589.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 13:15:00 | 1581.90 | 1590.46 | 1589.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 13:15:00 | 1581.90 | 1590.46 | 1589.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 1581.90 | 1590.46 | 1589.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 14:00:00 | 1581.90 | 1590.46 | 1589.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 14:15:00 | 1580.20 | 1588.41 | 1588.88 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 12:15:00 | 1596.00 | 1590.34 | 1589.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 13:15:00 | 1607.30 | 1593.73 | 1591.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 1604.90 | 1608.40 | 1602.50 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:15:00 | 1630.30 | 1608.40 | 1602.50 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 1606.80 | 1611.99 | 1606.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:30:00 | 1602.80 | 1611.99 | 1606.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 1610.00 | 1611.59 | 1606.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 1585.10 | 1605.58 | 1605.28 | SL hit (close<ema400) qty=1.00 sl=1605.28 alert=retest1 |

### Cycle 114 — SELL (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 10:15:00 | 1585.20 | 1601.50 | 1603.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-16 11:15:00 | 1569.10 | 1595.02 | 1600.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 13:15:00 | 1558.10 | 1557.45 | 1572.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-17 14:00:00 | 1558.10 | 1557.45 | 1572.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 1520.00 | 1522.30 | 1531.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 11:45:00 | 1507.20 | 1517.71 | 1527.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 1530.40 | 1524.81 | 1524.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 1530.40 | 1524.81 | 1524.28 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 1519.80 | 1523.81 | 1523.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 12:15:00 | 1513.00 | 1521.65 | 1522.88 | Break + close below crossover candle low |

### Cycle 117 — BUY (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 09:15:00 | 1536.40 | 1523.15 | 1522.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 1560.30 | 1530.58 | 1526.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 1545.70 | 1552.64 | 1545.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 1545.70 | 1552.64 | 1545.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 1545.70 | 1552.64 | 1545.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 1545.70 | 1552.64 | 1545.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 1545.80 | 1551.27 | 1545.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 1547.30 | 1551.27 | 1545.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1557.00 | 1552.42 | 1546.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 12:45:00 | 1566.00 | 1554.77 | 1548.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 1576.40 | 1557.36 | 1551.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-12 14:15:00 | 1722.60 | 1684.68 | 1654.06 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 1679.50 | 1683.52 | 1684.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 1674.40 | 1681.70 | 1683.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1689.30 | 1673.09 | 1676.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1689.30 | 1673.09 | 1676.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1689.30 | 1673.09 | 1676.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 1687.90 | 1673.09 | 1676.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 1688.10 | 1676.09 | 1677.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:30:00 | 1696.50 | 1676.09 | 1677.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 1688.20 | 1678.52 | 1678.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 1696.10 | 1684.01 | 1681.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 1683.20 | 1687.44 | 1683.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 1683.20 | 1687.44 | 1683.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 1683.20 | 1687.44 | 1683.66 | EMA400 retest candle locked (from upside) |

### Cycle 120 — SELL (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 13:15:00 | 1673.90 | 1681.28 | 1681.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 1663.30 | 1677.68 | 1679.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 13:15:00 | 1663.80 | 1660.35 | 1668.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 13:15:00 | 1663.80 | 1660.35 | 1668.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 1663.80 | 1660.35 | 1668.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:15:00 | 1673.20 | 1660.35 | 1668.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 1682.00 | 1664.68 | 1669.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 1682.00 | 1664.68 | 1669.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 1675.50 | 1666.85 | 1670.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 1683.00 | 1666.85 | 1670.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 1679.10 | 1671.39 | 1671.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 1677.50 | 1671.39 | 1671.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 1679.70 | 1673.05 | 1672.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 12:15:00 | 1688.20 | 1676.08 | 1673.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 1721.50 | 1726.67 | 1716.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-28 10:00:00 | 1721.50 | 1726.67 | 1716.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 10:15:00 | 1709.30 | 1723.19 | 1715.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:00:00 | 1709.30 | 1723.19 | 1715.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 1705.20 | 1719.60 | 1715.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:45:00 | 1699.30 | 1719.60 | 1715.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 1698.40 | 1712.06 | 1712.18 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 1717.00 | 1710.28 | 1710.06 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 1671.80 | 1702.58 | 1706.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 1667.40 | 1679.78 | 1689.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 1676.90 | 1673.35 | 1683.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 1676.90 | 1673.35 | 1683.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1677.80 | 1674.24 | 1682.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 1679.50 | 1674.24 | 1682.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 13:15:00 | 1678.80 | 1675.82 | 1681.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 13:30:00 | 1679.00 | 1675.82 | 1681.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 1687.50 | 1678.15 | 1681.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:45:00 | 1686.50 | 1678.15 | 1681.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 1685.00 | 1679.52 | 1682.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:15:00 | 1666.70 | 1679.52 | 1682.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 12:15:00 | 1692.50 | 1682.89 | 1682.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2025-12-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 12:15:00 | 1692.50 | 1682.89 | 1682.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 09:15:00 | 1703.80 | 1690.08 | 1686.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 1683.00 | 1690.43 | 1687.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 12:15:00 | 1683.00 | 1690.43 | 1687.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 1683.00 | 1690.43 | 1687.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:45:00 | 1681.60 | 1690.43 | 1687.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 1693.10 | 1690.96 | 1688.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 14:45:00 | 1694.90 | 1691.91 | 1688.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 09:15:00 | 1674.60 | 1689.41 | 1688.33 | SL hit (close<static) qty=1.00 sl=1680.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-12-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 11:15:00 | 1683.20 | 1703.02 | 1704.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 1675.50 | 1690.29 | 1696.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 14:15:00 | 1665.30 | 1664.33 | 1676.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 15:00:00 | 1665.30 | 1664.33 | 1676.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 1672.70 | 1666.35 | 1674.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 10:45:00 | 1673.00 | 1666.35 | 1674.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 1686.00 | 1670.28 | 1675.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 1688.00 | 1670.28 | 1675.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 1682.30 | 1672.68 | 1676.05 | EMA400 retest candle locked (from downside) |

### Cycle 127 — BUY (started 2025-12-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 15:15:00 | 1685.20 | 1678.10 | 1677.90 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 09:15:00 | 1670.10 | 1676.50 | 1677.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 10:15:00 | 1661.00 | 1673.40 | 1675.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 1676.30 | 1672.59 | 1674.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 13:15:00 | 1676.30 | 1672.59 | 1674.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 1676.30 | 1672.59 | 1674.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:45:00 | 1679.10 | 1672.59 | 1674.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 14:15:00 | 1690.80 | 1676.23 | 1676.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 15:15:00 | 1697.80 | 1687.79 | 1683.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 1683.60 | 1687.85 | 1683.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 10:15:00 | 1683.60 | 1687.85 | 1683.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 10:15:00 | 1683.60 | 1687.85 | 1683.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:00:00 | 1683.60 | 1687.85 | 1683.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 1683.70 | 1687.02 | 1683.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 1677.70 | 1687.02 | 1683.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 1693.40 | 1688.30 | 1684.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 12:30:00 | 1687.90 | 1688.30 | 1684.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 1686.00 | 1690.61 | 1686.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 1704.60 | 1690.61 | 1686.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:15:00 | 1701.60 | 1692.55 | 1688.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:15:00 | 1701.90 | 1695.88 | 1690.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 13:15:00 | 1681.80 | 1693.06 | 1690.12 | SL hit (close<static) qty=1.00 sl=1686.00 alert=retest2 |

### Cycle 130 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 1674.40 | 1686.86 | 1687.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 1666.10 | 1682.71 | 1685.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 10:15:00 | 1675.00 | 1673.62 | 1678.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 11:00:00 | 1675.00 | 1673.62 | 1678.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 1670.20 | 1673.16 | 1677.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:30:00 | 1674.20 | 1673.16 | 1677.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 1642.80 | 1641.71 | 1653.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 1642.40 | 1641.71 | 1653.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1648.40 | 1643.05 | 1653.38 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1673.00 | 1660.27 | 1658.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 1685.50 | 1668.91 | 1663.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 12:15:00 | 1664.60 | 1674.93 | 1671.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 12:15:00 | 1664.60 | 1674.93 | 1671.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 1664.60 | 1674.93 | 1671.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 1664.60 | 1674.93 | 1671.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 1666.90 | 1673.33 | 1670.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:45:00 | 1662.20 | 1673.33 | 1670.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 15:15:00 | 1677.00 | 1673.62 | 1671.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 1695.00 | 1673.62 | 1671.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 1692.90 | 1713.66 | 1715.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 1692.90 | 1713.66 | 1715.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 13:15:00 | 1681.30 | 1691.94 | 1701.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 10:15:00 | 1648.10 | 1647.52 | 1660.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 10:45:00 | 1658.30 | 1647.52 | 1660.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1647.40 | 1647.85 | 1655.20 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 12:15:00 | 1665.20 | 1655.86 | 1655.47 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 15:15:00 | 1650.30 | 1654.49 | 1654.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 10:15:00 | 1639.90 | 1650.42 | 1652.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 1620.40 | 1617.15 | 1629.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 1620.40 | 1617.15 | 1629.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 1641.70 | 1622.36 | 1629.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 1641.70 | 1622.36 | 1629.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 1616.40 | 1621.17 | 1628.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 1615.30 | 1621.17 | 1628.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 1616.30 | 1623.20 | 1626.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 1611.90 | 1621.17 | 1624.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 13:15:00 | 1621.90 | 1608.28 | 1607.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — BUY (started 2026-01-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 13:15:00 | 1621.90 | 1608.28 | 1607.19 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 1599.50 | 1608.20 | 1608.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 14:15:00 | 1584.20 | 1601.18 | 1605.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 1597.00 | 1596.97 | 1602.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 10:00:00 | 1597.00 | 1596.97 | 1602.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 1596.90 | 1596.96 | 1601.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:45:00 | 1608.90 | 1596.96 | 1601.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 1618.80 | 1601.33 | 1603.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:00:00 | 1618.80 | 1601.33 | 1603.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 1613.60 | 1603.78 | 1604.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:30:00 | 1615.90 | 1603.78 | 1604.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 1619.00 | 1606.82 | 1605.63 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 1596.80 | 1604.10 | 1604.82 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 1631.10 | 1609.50 | 1607.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 09:15:00 | 1636.10 | 1619.75 | 1613.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 14:15:00 | 1699.60 | 1700.91 | 1686.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 15:00:00 | 1699.60 | 1700.91 | 1686.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1697.60 | 1700.15 | 1688.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 1722.70 | 1700.12 | 1693.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:00:00 | 1721.00 | 1704.29 | 1695.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:30:00 | 1726.70 | 1709.31 | 1698.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:00:00 | 1727.00 | 1714.21 | 1702.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 11:15:00 | 1750.00 | 1743.32 | 1733.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 11:15:00 | 1780.50 | 1744.31 | 1737.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:15:00 | 1768.20 | 1755.02 | 1743.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 1841.00 | 1852.96 | 1854.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-02-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 14:15:00 | 1841.00 | 1852.96 | 1854.14 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-02-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 09:15:00 | 1862.70 | 1853.02 | 1852.29 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-02-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 13:15:00 | 1848.70 | 1851.58 | 1851.90 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 1857.40 | 1852.74 | 1852.40 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1840.90 | 1850.67 | 1851.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 1838.20 | 1848.18 | 1850.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 1747.80 | 1741.63 | 1763.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 15:00:00 | 1747.80 | 1741.63 | 1763.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 1705.00 | 1695.00 | 1711.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:15:00 | 1726.30 | 1695.00 | 1711.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 1720.00 | 1700.00 | 1711.85 | EMA400 retest candle locked (from downside) |

### Cycle 145 — BUY (started 2026-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 14:15:00 | 1732.80 | 1718.14 | 1717.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 10:15:00 | 1742.80 | 1728.45 | 1722.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 1729.20 | 1729.73 | 1724.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 13:00:00 | 1729.20 | 1729.73 | 1724.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1731.10 | 1730.00 | 1724.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 1739.40 | 1730.00 | 1724.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1722.40 | 1728.48 | 1724.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1722.40 | 1728.48 | 1724.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1722.10 | 1727.20 | 1724.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1685.60 | 1727.20 | 1724.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1686.00 | 1718.96 | 1720.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 1674.50 | 1696.89 | 1706.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1645.60 | 1636.98 | 1657.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:30:00 | 1652.70 | 1636.98 | 1657.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 1654.40 | 1640.63 | 1651.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 13:00:00 | 1654.40 | 1640.63 | 1651.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 1642.90 | 1641.08 | 1650.32 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 1663.00 | 1655.81 | 1655.10 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1643.40 | 1654.82 | 1656.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1627.90 | 1643.25 | 1649.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 11:15:00 | 1639.40 | 1638.99 | 1645.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 11:15:00 | 1639.40 | 1638.99 | 1645.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 1639.40 | 1638.99 | 1645.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 1641.00 | 1638.99 | 1645.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 1635.00 | 1637.80 | 1642.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 1591.70 | 1637.80 | 1642.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 12:15:00 | 1610.20 | 1599.81 | 1599.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 1610.20 | 1599.81 | 1599.09 | EMA200 above EMA400 |

### Cycle 150 — SELL (started 2026-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 14:15:00 | 1585.50 | 1596.79 | 1597.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-25 15:15:00 | 1581.10 | 1593.65 | 1596.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1526.50 | 1509.97 | 1534.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1526.50 | 1509.97 | 1534.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1526.50 | 1509.97 | 1534.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 1537.00 | 1509.97 | 1534.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1498.90 | 1507.76 | 1531.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:45:00 | 1487.40 | 1505.79 | 1528.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 12:30:00 | 1495.00 | 1503.13 | 1525.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:15:00 | 1469.20 | 1484.21 | 1485.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 1574.20 | 1502.09 | 1492.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 1574.20 | 1502.09 | 1492.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 1615.20 | 1524.71 | 1503.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 1631.30 | 1638.24 | 1611.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 1631.30 | 1638.24 | 1611.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 1631.30 | 1638.24 | 1611.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:45:00 | 1642.40 | 1637.61 | 1615.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 1643.40 | 1637.61 | 1615.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 12:15:00 | 1668.00 | 1677.67 | 1678.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2026-04-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 12:15:00 | 1668.00 | 1677.67 | 1678.58 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-04-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 14:15:00 | 1690.20 | 1680.16 | 1679.55 | EMA200 above EMA400 |

### Cycle 154 — SELL (started 2026-04-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-21 10:15:00 | 1662.00 | 1675.82 | 1677.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-21 11:15:00 | 1657.00 | 1672.06 | 1675.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 15:15:00 | 1639.50 | 1634.57 | 1648.10 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-23 09:15:00 | 1609.30 | 1634.57 | 1648.10 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 1589.30 | 1587.27 | 1602.03 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-27 14:15:00 | 1601.70 | 1594.51 | 1600.38 | SL hit (close>ema400) qty=1.00 sl=1600.38 alert=retest1 |

### Cycle 155 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 1613.70 | 1604.97 | 1603.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 1626.00 | 1610.94 | 1606.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 13:15:00 | 1607.10 | 1612.72 | 1609.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 13:15:00 | 1607.10 | 1612.72 | 1609.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 1607.10 | 1612.72 | 1609.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 1607.10 | 1612.72 | 1609.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 1608.40 | 1611.85 | 1609.26 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 1575.30 | 1604.55 | 1606.39 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 1608.70 | 1601.37 | 1600.40 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 1588.10 | 1598.72 | 1599.29 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 1635.80 | 1603.34 | 1600.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 1652.40 | 1629.23 | 1615.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 13:15:00 | 1690.80 | 1691.67 | 1670.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 14:00:00 | 1690.80 | 1691.67 | 1670.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-27 11:00:00 | 957.45 | 2024-05-28 09:15:00 | 987.50 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2024-05-27 14:30:00 | 956.60 | 2024-05-28 09:15:00 | 987.50 | STOP_HIT | 1.00 | -3.23% |
| SELL | retest2 | 2024-05-27 15:15:00 | 956.00 | 2024-05-28 09:15:00 | 987.50 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2024-06-04 09:15:00 | 898.95 | 2024-06-06 09:15:00 | 929.75 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-06-04 14:00:00 | 908.05 | 2024-06-06 09:15:00 | 929.75 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-06-05 12:30:00 | 924.10 | 2024-06-06 09:15:00 | 929.75 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2024-06-05 13:30:00 | 924.65 | 2024-06-06 09:15:00 | 929.75 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest2 | 2024-06-10 10:45:00 | 950.75 | 2024-06-24 09:15:00 | 980.70 | STOP_HIT | 1.00 | 3.15% |
| SELL | retest2 | 2024-06-26 13:45:00 | 973.80 | 2024-07-01 09:15:00 | 980.30 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2024-06-27 10:30:00 | 975.75 | 2024-07-01 09:15:00 | 980.30 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2024-06-27 12:00:00 | 973.35 | 2024-07-01 11:15:00 | 987.10 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-06-28 10:30:00 | 976.10 | 2024-07-01 11:15:00 | 987.10 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-06-28 12:30:00 | 971.00 | 2024-07-01 11:15:00 | 987.10 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-06-28 14:30:00 | 970.80 | 2024-07-01 11:15:00 | 987.10 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2024-07-03 09:15:00 | 985.80 | 2024-07-08 14:15:00 | 992.80 | STOP_HIT | 1.00 | 0.71% |
| BUY | retest2 | 2024-07-03 10:15:00 | 985.75 | 2024-07-08 14:15:00 | 992.80 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2024-07-03 11:15:00 | 986.10 | 2024-07-08 14:15:00 | 992.80 | STOP_HIT | 1.00 | 0.68% |
| BUY | retest2 | 2024-07-10 12:30:00 | 1028.15 | 2024-07-19 10:15:00 | 1022.80 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-07-15 09:30:00 | 1014.90 | 2024-07-19 10:15:00 | 1022.80 | STOP_HIT | 1.00 | 0.78% |
| BUY | retest2 | 2024-07-30 11:30:00 | 1096.40 | 2024-07-30 14:15:00 | 1087.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2024-07-30 13:00:00 | 1096.00 | 2024-07-30 14:15:00 | 1087.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-07-30 13:30:00 | 1097.50 | 2024-07-30 14:15:00 | 1087.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2024-08-08 09:15:00 | 1076.05 | 2024-08-08 10:15:00 | 1091.25 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-08-19 09:15:00 | 1006.80 | 2024-08-21 10:15:00 | 1029.55 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2024-08-19 10:15:00 | 1008.70 | 2024-08-21 10:15:00 | 1029.55 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2024-08-20 11:30:00 | 1009.50 | 2024-08-21 10:15:00 | 1029.55 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-09-11 11:30:00 | 1150.75 | 2024-09-16 09:15:00 | 1138.15 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-09-12 09:15:00 | 1150.55 | 2024-09-16 09:15:00 | 1138.15 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-09-12 11:45:00 | 1151.85 | 2024-09-16 09:15:00 | 1138.15 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-09-12 12:15:00 | 1151.80 | 2024-09-16 09:15:00 | 1138.15 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-09-19 09:15:00 | 1160.45 | 2024-10-01 14:15:00 | 1185.00 | STOP_HIT | 1.00 | 2.12% |
| SELL | retest2 | 2024-10-03 11:30:00 | 1183.30 | 2024-10-09 09:15:00 | 1192.10 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-10-03 12:00:00 | 1184.65 | 2024-10-09 09:15:00 | 1192.10 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2024-10-03 13:00:00 | 1183.80 | 2024-10-09 10:15:00 | 1181.00 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2024-10-04 11:30:00 | 1181.50 | 2024-10-09 10:15:00 | 1181.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2024-10-04 13:15:00 | 1166.70 | 2024-10-09 10:15:00 | 1181.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-10-07 09:45:00 | 1173.70 | 2024-10-09 10:15:00 | 1181.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2024-10-21 14:30:00 | 1193.75 | 2024-10-22 12:15:00 | 1179.55 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-10-21 15:00:00 | 1195.80 | 2024-10-22 12:15:00 | 1179.55 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-10-28 09:45:00 | 1276.20 | 2024-10-28 11:15:00 | 1263.65 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-10-28 11:00:00 | 1276.30 | 2024-10-28 11:15:00 | 1263.65 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-11-05 15:15:00 | 1244.05 | 2024-11-14 13:15:00 | 1231.95 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2024-11-06 09:30:00 | 1247.20 | 2024-11-14 13:15:00 | 1231.95 | STOP_HIT | 1.00 | 1.22% |
| BUY | retest2 | 2024-12-09 13:15:00 | 1175.80 | 2024-12-11 09:15:00 | 1158.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-12-24 15:00:00 | 1101.25 | 2024-12-26 14:15:00 | 1115.85 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-12-26 12:00:00 | 1103.50 | 2024-12-26 14:15:00 | 1115.85 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-12-26 14:30:00 | 1104.90 | 2024-12-26 15:15:00 | 1113.05 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-12-27 15:15:00 | 1102.10 | 2024-12-30 14:15:00 | 1115.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-01-09 12:45:00 | 1093.15 | 2025-01-15 09:15:00 | 1038.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 12:45:00 | 1093.15 | 2025-01-15 15:15:00 | 1041.00 | STOP_HIT | 0.50 | 4.77% |
| SELL | retest2 | 2025-01-16 10:30:00 | 1088.10 | 2025-01-16 11:15:00 | 1080.35 | STOP_HIT | 1.00 | 0.71% |
| SELL | retest2 | 2025-01-24 09:45:00 | 1048.80 | 2025-01-24 11:15:00 | 1063.45 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-02-06 11:45:00 | 1118.20 | 2025-02-10 10:15:00 | 1100.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-02-07 11:15:00 | 1116.55 | 2025-02-10 10:15:00 | 1100.00 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-02-07 13:15:00 | 1116.15 | 2025-02-10 10:15:00 | 1100.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-02-10 09:45:00 | 1117.00 | 2025-02-10 10:15:00 | 1100.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-02-18 09:15:00 | 1048.65 | 2025-02-21 14:15:00 | 1056.55 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-02-18 12:30:00 | 1059.05 | 2025-02-21 14:15:00 | 1056.55 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-02-18 15:15:00 | 1057.00 | 2025-02-21 14:15:00 | 1056.55 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-02-19 10:45:00 | 1057.55 | 2025-02-21 14:15:00 | 1056.55 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2025-02-19 12:30:00 | 1055.95 | 2025-02-21 14:15:00 | 1056.55 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-02-19 13:30:00 | 1056.85 | 2025-02-21 14:15:00 | 1056.55 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-02-21 10:00:00 | 1056.60 | 2025-02-21 14:15:00 | 1056.55 | STOP_HIT | 1.00 | 0.00% |
| SELL | retest2 | 2025-02-21 12:15:00 | 1056.45 | 2025-02-21 14:15:00 | 1056.55 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-02-28 09:15:00 | 1013.20 | 2025-03-05 10:15:00 | 1014.60 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2025-04-04 10:30:00 | 1156.80 | 2025-04-07 09:15:00 | 1127.15 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2025-04-25 09:15:00 | 1269.20 | 2025-05-05 13:15:00 | 1284.30 | STOP_HIT | 1.00 | 1.19% |
| BUY | retest2 | 2025-05-21 14:15:00 | 1391.70 | 2025-06-02 12:15:00 | 1530.87 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-05 13:30:00 | 1495.50 | 2025-06-06 10:15:00 | 1514.80 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2025-06-20 10:45:00 | 1592.20 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2025-06-20 14:00:00 | 1591.90 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | 2.61% |
| BUY | retest2 | 2025-06-20 14:30:00 | 1591.00 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2025-06-24 12:15:00 | 1598.70 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest2 | 2025-07-01 09:15:00 | 1651.70 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-07-01 11:00:00 | 1650.90 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-01 15:00:00 | 1653.60 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-07-02 09:30:00 | 1649.70 | 2025-07-03 10:15:00 | 1633.40 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-07-21 13:45:00 | 1547.60 | 2025-07-22 10:15:00 | 1556.60 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-07-28 12:00:00 | 1540.10 | 2025-07-28 12:15:00 | 1537.90 | STOP_HIT | 1.00 | 0.14% |
| SELL | retest2 | 2025-07-31 09:45:00 | 1498.30 | 2025-08-06 15:15:00 | 1493.00 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-07-31 13:00:00 | 1501.70 | 2025-08-06 15:15:00 | 1493.00 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-07-31 15:00:00 | 1502.30 | 2025-08-06 15:15:00 | 1493.00 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2025-08-20 12:45:00 | 1643.90 | 2025-08-26 09:15:00 | 1636.10 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-08-20 14:30:00 | 1644.40 | 2025-08-26 11:15:00 | 1633.10 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-08-20 15:15:00 | 1645.40 | 2025-08-26 11:15:00 | 1633.10 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-08-22 09:30:00 | 1644.60 | 2025-08-26 11:15:00 | 1633.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-08-25 11:15:00 | 1654.90 | 2025-08-26 11:15:00 | 1633.10 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-08-28 13:00:00 | 1623.90 | 2025-09-02 09:15:00 | 1640.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-02 09:15:00 | 1620.80 | 2025-09-02 09:15:00 | 1640.00 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2025-09-04 11:45:00 | 1569.30 | 2025-09-09 15:15:00 | 1596.30 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-09-04 12:45:00 | 1570.00 | 2025-09-09 15:15:00 | 1596.30 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-09 10:30:00 | 1566.80 | 2025-09-09 15:15:00 | 1596.30 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-09-11 11:15:00 | 1604.70 | 2025-09-12 09:15:00 | 1592.90 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-09-11 11:45:00 | 1607.00 | 2025-09-12 09:15:00 | 1592.90 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-11 12:15:00 | 1604.90 | 2025-09-12 09:15:00 | 1592.90 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-09-11 12:45:00 | 1604.40 | 2025-09-12 09:15:00 | 1592.90 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-09-16 11:15:00 | 1566.60 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | 0.15% |
| SELL | retest2 | 2025-09-16 11:45:00 | 1563.70 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-09-16 13:15:00 | 1567.00 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-09-16 13:45:00 | 1566.70 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | 0.16% |
| SELL | retest2 | 2025-09-19 12:30:00 | 1557.80 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-09-19 14:00:00 | 1563.00 | 2025-09-22 09:15:00 | 1564.20 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest1 | 2025-10-15 09:15:00 | 1630.30 | 2025-10-16 09:15:00 | 1585.10 | STOP_HIT | 1.00 | -2.77% |
| SELL | retest2 | 2025-10-27 11:45:00 | 1507.20 | 2025-10-29 10:15:00 | 1530.40 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-11-03 12:45:00 | 1566.00 | 2025-11-12 14:15:00 | 1722.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-04 09:15:00 | 1576.40 | 2025-11-18 15:15:00 | 1679.50 | STOP_HIT | 1.00 | 6.54% |
| SELL | retest2 | 2025-12-05 09:15:00 | 1666.70 | 2025-12-05 12:15:00 | 1692.50 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-12-08 14:45:00 | 1694.90 | 2025-12-09 09:15:00 | 1674.60 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-12-09 12:15:00 | 1695.00 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-12-09 13:00:00 | 1694.90 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-12-10 09:15:00 | 1694.70 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-10 11:30:00 | 1707.50 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-12-10 13:15:00 | 1710.70 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-12-11 11:15:00 | 1706.20 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-11 12:00:00 | 1709.50 | 2025-12-15 11:15:00 | 1683.20 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-12-24 09:15:00 | 1704.60 | 2025-12-24 13:15:00 | 1681.80 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-12-24 10:15:00 | 1701.60 | 2025-12-24 13:15:00 | 1681.80 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-24 13:15:00 | 1701.90 | 2025-12-24 13:15:00 | 1681.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2026-01-05 09:15:00 | 1695.00 | 2026-01-08 11:15:00 | 1692.90 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2026-01-22 11:15:00 | 1615.30 | 2026-01-28 13:15:00 | 1621.90 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2026-01-23 09:15:00 | 1616.30 | 2026-01-28 13:15:00 | 1621.90 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-01-23 13:15:00 | 1611.90 | 2026-01-28 13:15:00 | 1621.90 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-02-09 09:15:00 | 1722.70 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 6.87% |
| BUY | retest2 | 2026-02-09 10:00:00 | 1721.00 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 6.97% |
| BUY | retest2 | 2026-02-09 10:30:00 | 1726.70 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 6.62% |
| BUY | retest2 | 2026-02-09 13:00:00 | 1727.00 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 6.60% |
| BUY | retest2 | 2026-02-12 11:15:00 | 1780.50 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 3.40% |
| BUY | retest2 | 2026-02-12 13:15:00 | 1768.20 | 2026-02-24 14:15:00 | 1841.00 | STOP_HIT | 1.00 | 4.12% |
| SELL | retest2 | 2026-03-23 09:15:00 | 1591.70 | 2026-03-25 12:15:00 | 1610.20 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-01 11:45:00 | 1487.40 | 2026-04-08 09:15:00 | 1574.20 | STOP_HIT | 1.00 | -5.84% |
| SELL | retest2 | 2026-04-01 12:30:00 | 1495.00 | 2026-04-08 09:15:00 | 1574.20 | STOP_HIT | 1.00 | -5.30% |
| SELL | retest2 | 2026-04-07 09:15:00 | 1469.20 | 2026-04-08 09:15:00 | 1574.20 | STOP_HIT | 1.00 | -7.15% |
| BUY | retest2 | 2026-04-13 11:45:00 | 1642.40 | 2026-04-20 12:15:00 | 1668.00 | STOP_HIT | 1.00 | 1.56% |
| BUY | retest2 | 2026-04-13 12:15:00 | 1643.40 | 2026-04-20 12:15:00 | 1668.00 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest1 | 2026-04-23 09:15:00 | 1609.30 | 2026-04-27 14:15:00 | 1601.70 | STOP_HIT | 1.00 | 0.47% |
