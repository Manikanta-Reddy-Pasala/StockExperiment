# PVR INOX Ltd. (PVRINOX)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 1075.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 83 |
| ALERT1 | 52 |
| ALERT2 | 51 |
| ALERT2_SKIP | 33 |
| ALERT3 | 121 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 59 |
| PARTIAL | 5 |
| TARGET_HIT | 1 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 15 / 46
- **Target hits / Stop hits / Partials:** 1 / 55 / 5
- **Avg / median % per leg:** -0.12% / -1.02%
- **Sum % (uncompounded):** -7.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 26 | 6 | 23.1% | 1 | 24 | 1 | -0.39% | -10.0% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.13% | 10.3% |
| BUY @ 3rd Alert (retest2) | 24 | 4 | 16.7% | 1 | 23 | 0 | -0.85% | -20.3% |
| SELL (all) | 35 | 9 | 25.7% | 0 | 31 | 4 | 0.08% | 2.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 9 | 25.7% | 0 | 31 | 4 | 0.08% | 2.7% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.13% | 10.3% |
| retest2 (combined) | 59 | 13 | 22.0% | 1 | 54 | 4 | -0.30% | -17.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 963.60 | 938.26 | 935.72 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 937.00 | 943.77 | 944.40 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 949.95 | 945.01 | 944.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 13:15:00 | 959.65 | 947.94 | 946.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 951.75 | 954.04 | 949.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 951.75 | 954.04 | 949.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 951.75 | 954.04 | 949.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:30:00 | 986.60 | 959.83 | 953.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:15:00 | 969.55 | 992.22 | 990.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 10:15:00 | 969.85 | 987.75 | 988.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-20 10:15:00 | 969.85 | 987.75 | 988.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 969.85 | 987.75 | 988.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 958.05 | 981.81 | 985.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 15:15:00 | 972.40 | 972.34 | 979.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 09:15:00 | 958.50 | 972.34 | 979.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 955.00 | 960.42 | 968.40 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 977.95 | 967.36 | 967.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 983.00 | 976.20 | 972.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 09:15:00 | 988.00 | 989.77 | 984.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 09:15:00 | 988.00 | 989.77 | 984.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 09:15:00 | 988.00 | 989.77 | 984.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 10:45:00 | 1008.85 | 993.46 | 986.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:15:00 | 1003.85 | 1004.25 | 999.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 10:45:00 | 1001.60 | 1002.91 | 1000.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:45:00 | 1001.55 | 1001.57 | 1000.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 13:15:00 | 993.45 | 999.95 | 999.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 14:00:00 | 993.45 | 999.95 | 999.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 985.05 | 996.97 | 998.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 985.05 | 996.97 | 998.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 985.05 | 996.97 | 998.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 985.05 | 996.97 | 998.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 985.05 | 996.97 | 998.13 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 1037.20 | 1005.20 | 1001.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 14:15:00 | 1042.20 | 1015.23 | 1009.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 10:15:00 | 1051.35 | 1052.14 | 1038.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-05 10:45:00 | 1053.65 | 1052.14 | 1038.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 1037.40 | 1046.36 | 1039.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:45:00 | 1039.60 | 1046.36 | 1039.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 1039.85 | 1045.06 | 1039.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:30:00 | 1036.60 | 1045.06 | 1039.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 1040.15 | 1044.08 | 1039.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 1031.00 | 1044.08 | 1039.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 1027.95 | 1040.85 | 1038.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:30:00 | 1028.75 | 1040.85 | 1038.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 1027.70 | 1038.22 | 1037.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 1027.70 | 1038.22 | 1037.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 11:15:00 | 1026.50 | 1035.88 | 1036.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 1020.25 | 1029.97 | 1033.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 1012.65 | 1012.19 | 1017.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 1012.65 | 1012.19 | 1017.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1012.65 | 1012.19 | 1017.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 10:45:00 | 1008.00 | 1011.53 | 1016.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-06-13 09:15:00 | 957.60 | 983.62 | 995.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 958.25 | 957.55 | 969.22 | SL hit (close>ema200) qty=0.50 sl=957.55 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 14:15:00 | 962.80 | 959.30 | 959.20 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 952.00 | 957.84 | 958.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 14:15:00 | 939.20 | 951.58 | 955.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 11:15:00 | 947.20 | 947.06 | 951.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 11:45:00 | 947.25 | 947.06 | 951.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 948.90 | 947.31 | 950.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:00:00 | 948.90 | 947.31 | 950.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 952.70 | 948.39 | 950.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 14:45:00 | 949.70 | 948.39 | 950.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 951.95 | 949.10 | 951.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 955.70 | 949.10 | 951.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 956.40 | 950.56 | 951.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:15:00 | 963.90 | 950.56 | 951.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 972.00 | 954.85 | 953.37 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 959.00 | 965.05 | 965.59 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 15:15:00 | 969.00 | 966.23 | 965.98 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 09:15:00 | 957.00 | 964.38 | 965.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 956.50 | 960.37 | 962.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 14:15:00 | 960.90 | 959.81 | 961.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 960.90 | 959.81 | 961.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 960.90 | 959.81 | 961.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 960.90 | 959.81 | 961.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 959.00 | 959.65 | 961.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:15:00 | 957.00 | 959.65 | 961.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 961.30 | 959.98 | 961.19 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 969.65 | 963.28 | 962.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 13:15:00 | 974.95 | 966.57 | 964.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-04 11:15:00 | 971.80 | 972.63 | 968.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-04 11:45:00 | 971.60 | 972.63 | 968.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 964.55 | 972.31 | 970.16 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 964.30 | 968.60 | 968.81 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 13:15:00 | 980.55 | 970.99 | 969.88 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 965.55 | 969.00 | 969.43 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 993.80 | 973.87 | 971.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 1002.95 | 979.69 | 974.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 996.40 | 1005.59 | 997.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 09:15:00 | 996.40 | 1005.59 | 997.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 996.40 | 1005.59 | 997.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:15:00 | 990.05 | 1005.59 | 997.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 988.95 | 1002.26 | 996.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 988.95 | 1002.26 | 996.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 988.35 | 999.48 | 996.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:15:00 | 988.85 | 999.48 | 996.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-07-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 15:15:00 | 986.00 | 993.46 | 993.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 09:15:00 | 971.90 | 989.15 | 991.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 997.30 | 984.02 | 986.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 997.30 | 984.02 | 986.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 997.30 | 984.02 | 986.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 997.30 | 984.02 | 986.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 990.60 | 985.34 | 986.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 11:15:00 | 987.80 | 985.34 | 986.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 1005.60 | 989.39 | 988.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 1005.60 | 989.39 | 988.55 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 10:15:00 | 978.55 | 987.23 | 988.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 975.85 | 978.08 | 981.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 12:15:00 | 987.75 | 974.08 | 977.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 12:15:00 | 987.75 | 974.08 | 977.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 987.75 | 974.08 | 977.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 13:00:00 | 987.75 | 974.08 | 977.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 13:15:00 | 1008.65 | 980.99 | 980.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-18 14:15:00 | 1017.35 | 988.27 | 983.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 1019.35 | 1022.70 | 1010.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 10:45:00 | 1018.20 | 1022.70 | 1010.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 1011.00 | 1020.36 | 1010.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:00:00 | 1011.00 | 1020.36 | 1010.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 1019.50 | 1020.19 | 1011.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 12:30:00 | 1013.00 | 1020.19 | 1011.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 1013.30 | 1018.81 | 1011.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 14:00:00 | 1013.30 | 1018.81 | 1011.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 1014.60 | 1017.97 | 1011.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 12:15:00 | 1026.80 | 1015.78 | 1012.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 1024.00 | 1018.12 | 1014.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 1008.65 | 1015.07 | 1014.22 | SL hit (close<static) qty=1.00 sl=1009.15 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-24 11:15:00 | 1008.65 | 1015.07 | 1014.22 | SL hit (close<static) qty=1.00 sl=1009.15 alert=retest2 |

### Cycle 24 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 1004.55 | 1012.96 | 1013.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 997.85 | 1008.09 | 1010.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 14:15:00 | 988.90 | 983.34 | 989.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 14:15:00 | 988.90 | 983.34 | 989.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 988.90 | 983.34 | 989.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:00:00 | 988.90 | 983.34 | 989.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 985.45 | 983.76 | 988.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 1002.25 | 983.76 | 988.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 999.15 | 986.84 | 989.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 14:30:00 | 986.90 | 989.93 | 990.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:15:00 | 986.50 | 989.93 | 990.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 988.05 | 989.01 | 989.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 11:15:00 | 995.00 | 991.32 | 990.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 11:15:00 | 995.00 | 991.32 | 990.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-31 11:15:00 | 995.00 | 991.32 | 990.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 11:15:00 | 995.00 | 991.32 | 990.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 13:15:00 | 1001.95 | 994.79 | 992.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 1000.40 | 1006.60 | 1001.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 14:15:00 | 1000.40 | 1006.60 | 1001.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 1000.40 | 1006.60 | 1001.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 1000.40 | 1006.60 | 1001.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 994.00 | 1004.08 | 1000.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 10:45:00 | 1002.50 | 1003.55 | 1000.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-08 09:15:00 | 1102.75 | 1060.44 | 1045.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 1073.20 | 1077.46 | 1077.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 09:15:00 | 1066.55 | 1075.53 | 1076.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 1099.65 | 1070.17 | 1071.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 1099.65 | 1070.17 | 1071.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 1099.65 | 1070.17 | 1071.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:00:00 | 1099.65 | 1070.17 | 1071.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 1094.95 | 1075.13 | 1073.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 09:15:00 | 1121.60 | 1094.25 | 1088.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 1121.90 | 1126.48 | 1117.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-25 09:30:00 | 1120.60 | 1126.48 | 1117.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 11:15:00 | 1118.60 | 1123.99 | 1117.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:15:00 | 1115.00 | 1123.99 | 1117.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 1108.15 | 1120.83 | 1116.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 1108.15 | 1120.83 | 1116.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 1111.70 | 1119.00 | 1116.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 09:30:00 | 1116.45 | 1116.74 | 1115.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 1122.65 | 1125.96 | 1121.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 1105.75 | 1121.05 | 1121.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 1105.75 | 1121.05 | 1121.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 1105.75 | 1121.05 | 1121.71 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 1119.20 | 1118.61 | 1118.54 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 1116.60 | 1118.13 | 1118.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 13:15:00 | 1113.20 | 1117.14 | 1117.86 | Break + close below crossover candle low |

### Cycle 31 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 1135.00 | 1119.17 | 1118.56 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1113.70 | 1121.95 | 1122.02 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 12:15:00 | 1126.90 | 1122.94 | 1122.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 1132.50 | 1124.85 | 1123.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 09:15:00 | 1116.60 | 1128.80 | 1126.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 1116.60 | 1128.80 | 1126.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 1116.60 | 1128.80 | 1126.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 1116.60 | 1128.80 | 1126.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 1120.90 | 1127.22 | 1125.58 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 12:15:00 | 1120.00 | 1124.24 | 1124.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 1116.40 | 1122.20 | 1123.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 1134.90 | 1122.74 | 1123.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 1134.90 | 1122.74 | 1123.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 1134.90 | 1122.74 | 1123.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 1133.10 | 1122.74 | 1123.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 1141.50 | 1126.49 | 1124.99 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 1118.20 | 1125.63 | 1126.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 13:15:00 | 1111.40 | 1122.79 | 1125.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 15:15:00 | 1125.00 | 1122.77 | 1124.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 15:15:00 | 1125.00 | 1122.77 | 1124.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 15:15:00 | 1125.00 | 1122.77 | 1124.68 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 1140.90 | 1126.39 | 1126.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 1144.50 | 1130.01 | 1127.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 10:15:00 | 1131.80 | 1133.68 | 1131.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 10:15:00 | 1131.80 | 1133.68 | 1131.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 1131.80 | 1133.68 | 1131.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 10:45:00 | 1130.70 | 1133.68 | 1131.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 1122.20 | 1131.38 | 1130.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 12:00:00 | 1122.20 | 1131.38 | 1130.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 12:15:00 | 1122.70 | 1129.65 | 1129.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 13:15:00 | 1121.00 | 1127.92 | 1128.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 1133.50 | 1121.28 | 1123.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 1133.50 | 1121.28 | 1123.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 1133.50 | 1121.28 | 1123.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 1133.50 | 1121.28 | 1123.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 1130.00 | 1123.03 | 1124.09 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 1139.70 | 1126.36 | 1125.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 1146.50 | 1134.98 | 1130.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 11:15:00 | 1133.00 | 1137.36 | 1133.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 11:15:00 | 1133.00 | 1137.36 | 1133.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 1133.00 | 1137.36 | 1133.12 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 12:15:00 | 1126.40 | 1131.84 | 1132.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 1121.50 | 1128.75 | 1130.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 10:15:00 | 1114.00 | 1110.85 | 1115.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 10:30:00 | 1114.90 | 1110.85 | 1115.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 1110.30 | 1110.74 | 1115.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:45:00 | 1104.80 | 1110.63 | 1113.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 12:30:00 | 1106.10 | 1110.45 | 1113.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 1088.60 | 1111.14 | 1112.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 12:15:00 | 1104.20 | 1111.35 | 1111.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 1110.10 | 1107.76 | 1109.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 1110.10 | 1107.76 | 1109.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 1100.00 | 1106.20 | 1108.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 1110.60 | 1106.94 | 1109.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 1107.00 | 1106.95 | 1108.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:30:00 | 1103.00 | 1106.06 | 1108.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 15:00:00 | 1100.00 | 1105.05 | 1107.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 1111.20 | 1093.05 | 1091.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 1111.20 | 1093.05 | 1091.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 1111.20 | 1093.05 | 1091.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 1111.20 | 1093.05 | 1091.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 1111.20 | 1093.05 | 1091.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 1111.20 | 1093.05 | 1091.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 1111.20 | 1093.05 | 1091.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 1125.60 | 1099.56 | 1094.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 1108.80 | 1109.10 | 1103.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 1108.70 | 1109.10 | 1103.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1102.30 | 1107.74 | 1103.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:30:00 | 1102.10 | 1107.74 | 1103.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 1102.60 | 1106.71 | 1103.70 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 14:15:00 | 1095.80 | 1101.97 | 1102.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 15:15:00 | 1090.00 | 1099.58 | 1101.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 10:15:00 | 1104.80 | 1099.83 | 1100.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 10:15:00 | 1104.80 | 1099.83 | 1100.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 1104.80 | 1099.83 | 1100.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 1104.80 | 1099.83 | 1100.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 1094.90 | 1098.84 | 1100.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 13:45:00 | 1094.20 | 1097.16 | 1099.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 15:00:00 | 1093.50 | 1096.43 | 1098.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1105.80 | 1097.27 | 1098.69 | SL hit (close>static) qty=1.00 sl=1105.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-09 09:15:00 | 1105.80 | 1097.27 | 1098.69 | SL hit (close>static) qty=1.00 sl=1105.10 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:00:00 | 1094.40 | 1097.31 | 1098.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 11:45:00 | 1094.60 | 1094.68 | 1096.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 1092.60 | 1094.26 | 1095.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 13:30:00 | 1091.90 | 1093.79 | 1095.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 14:00:00 | 1091.90 | 1093.79 | 1095.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 1098.80 | 1093.48 | 1094.83 | SL hit (close>static) qty=1.00 sl=1098.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 1098.80 | 1093.48 | 1094.83 | SL hit (close>static) qty=1.00 sl=1098.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 1100.00 | 1095.67 | 1095.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 1100.00 | 1095.67 | 1095.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 1100.00 | 1095.67 | 1095.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 12:15:00 | 1108.40 | 1100.30 | 1098.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 14:15:00 | 1102.00 | 1102.08 | 1099.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:45:00 | 1100.00 | 1102.08 | 1099.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 1108.50 | 1103.36 | 1100.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:00:00 | 1110.20 | 1104.73 | 1101.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:30:00 | 1110.50 | 1105.18 | 1101.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 1099.10 | 1103.56 | 1101.89 | SL hit (close<static) qty=1.00 sl=1099.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 1099.10 | 1103.56 | 1101.89 | SL hit (close<static) qty=1.00 sl=1099.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-16 09:15:00 | 1111.80 | 1103.50 | 1102.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 14:15:00 | 1096.80 | 1102.73 | 1102.66 | SL hit (close<static) qty=1.00 sl=1099.90 alert=retest2 |

### Cycle 44 — SELL (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 15:15:00 | 1095.10 | 1101.20 | 1101.97 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 1107.60 | 1102.43 | 1102.32 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 1083.00 | 1099.69 | 1101.17 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 10:15:00 | 1116.60 | 1101.56 | 1101.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 11:15:00 | 1123.60 | 1105.97 | 1103.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-24 09:15:00 | 1151.80 | 1153.78 | 1142.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-24 09:30:00 | 1159.00 | 1153.78 | 1142.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 1145.10 | 1150.89 | 1143.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 1145.30 | 1150.89 | 1143.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1212.30 | 1228.34 | 1216.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:45:00 | 1214.10 | 1228.34 | 1216.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 1214.70 | 1225.61 | 1216.35 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-11-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 10:15:00 | 1196.50 | 1212.20 | 1213.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 11:15:00 | 1190.10 | 1207.78 | 1210.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 14:15:00 | 1206.20 | 1205.62 | 1209.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 14:15:00 | 1206.20 | 1205.62 | 1209.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 1206.20 | 1205.62 | 1209.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:30:00 | 1205.60 | 1205.62 | 1209.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 1205.00 | 1205.50 | 1208.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 1190.20 | 1201.28 | 1206.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 1130.69 | 1146.91 | 1164.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-07 11:15:00 | 1148.60 | 1146.29 | 1161.15 | SL hit (close>ema200) qty=0.50 sl=1146.29 alert=retest2 |

### Cycle 49 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 1119.90 | 1107.38 | 1105.79 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 1103.00 | 1108.62 | 1108.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 14:15:00 | 1099.20 | 1106.47 | 1107.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 1067.70 | 1058.26 | 1069.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 10:00:00 | 1067.70 | 1058.26 | 1069.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 1068.50 | 1060.31 | 1069.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:30:00 | 1067.00 | 1060.31 | 1069.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 11:15:00 | 1073.10 | 1062.87 | 1069.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 12:00:00 | 1073.10 | 1062.87 | 1069.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 1070.30 | 1064.35 | 1069.96 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 1081.50 | 1073.15 | 1072.97 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 1069.20 | 1072.36 | 1072.63 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 1078.60 | 1071.69 | 1071.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 13:15:00 | 1096.00 | 1077.66 | 1074.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-04 09:15:00 | 1096.60 | 1118.00 | 1107.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 09:15:00 | 1096.60 | 1118.00 | 1107.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 09:15:00 | 1096.60 | 1118.00 | 1107.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:00:00 | 1096.60 | 1118.00 | 1107.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 1100.10 | 1114.42 | 1107.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 12:45:00 | 1103.80 | 1109.61 | 1105.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 14:15:00 | 1103.30 | 1108.21 | 1105.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 15:15:00 | 1090.10 | 1103.59 | 1103.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 15:15:00 | 1090.10 | 1103.59 | 1103.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-04 15:15:00 | 1090.10 | 1103.59 | 1103.95 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 10:15:00 | 1108.90 | 1099.96 | 1099.95 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 1090.70 | 1098.11 | 1099.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1088.00 | 1095.60 | 1097.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 1091.50 | 1086.23 | 1091.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 1091.50 | 1086.23 | 1091.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1091.50 | 1086.23 | 1091.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 1091.50 | 1086.23 | 1091.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1078.50 | 1084.68 | 1090.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 1072.00 | 1080.74 | 1086.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 10:15:00 | 1119.40 | 1073.03 | 1068.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 1119.40 | 1073.03 | 1068.50 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 1070.00 | 1075.39 | 1076.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 1069.00 | 1074.12 | 1075.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 15:15:00 | 1075.00 | 1073.17 | 1074.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 15:15:00 | 1075.00 | 1073.17 | 1074.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 1075.00 | 1073.17 | 1074.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 1057.10 | 1073.17 | 1074.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 1056.90 | 1055.44 | 1058.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:15:00 | 1004.24 | 1018.16 | 1028.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-26 11:15:00 | 1004.06 | 1018.16 | 1028.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 1018.80 | 1007.17 | 1013.70 | SL hit (close>ema200) qty=0.50 sl=1007.17 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 1018.80 | 1007.17 | 1013.70 | SL hit (close>ema200) qty=0.50 sl=1007.17 alert=retest2 |

### Cycle 59 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 1012.60 | 1007.28 | 1007.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 09:15:00 | 1019.50 | 1010.96 | 1008.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 15:15:00 | 1017.00 | 1018.41 | 1014.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 15:15:00 | 1017.00 | 1018.41 | 1014.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 15:15:00 | 1017.00 | 1018.41 | 1014.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-02 09:15:00 | 1033.20 | 1018.41 | 1014.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 12:15:00 | 1020.80 | 1031.95 | 1033.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 1020.80 | 1031.95 | 1033.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 1019.70 | 1029.50 | 1031.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 999.30 | 987.63 | 998.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 999.30 | 987.63 | 998.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 999.30 | 987.63 | 998.94 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 13:15:00 | 1020.00 | 1005.72 | 1004.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 1047.80 | 1014.13 | 1008.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 12:15:00 | 1019.00 | 1024.29 | 1017.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-14 12:45:00 | 1019.40 | 1024.29 | 1017.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 13:15:00 | 1021.30 | 1023.69 | 1017.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 13:30:00 | 1016.20 | 1023.69 | 1017.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 14:15:00 | 1021.40 | 1023.24 | 1017.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 14:30:00 | 1019.10 | 1023.24 | 1017.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 1017.00 | 1021.99 | 1017.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 1013.90 | 1021.99 | 1017.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 1014.00 | 1020.39 | 1017.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:45:00 | 1009.10 | 1020.39 | 1017.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 11:15:00 | 1024.10 | 1019.49 | 1017.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 15:00:00 | 1029.60 | 1021.47 | 1018.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 1009.70 | 1018.96 | 1018.18 | SL hit (close<static) qty=1.00 sl=1010.10 alert=retest2 |

### Cycle 62 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 1009.20 | 1017.01 | 1017.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 996.90 | 1012.99 | 1015.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 974.40 | 957.18 | 969.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 974.40 | 957.18 | 969.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 974.40 | 957.18 | 969.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 973.80 | 957.18 | 969.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 963.30 | 958.41 | 968.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:15:00 | 954.30 | 959.03 | 968.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 945.30 | 940.93 | 940.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 945.30 | 940.93 | 940.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 952.80 | 943.30 | 941.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 12:15:00 | 967.50 | 968.53 | 959.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:30:00 | 969.95 | 968.53 | 959.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 991.05 | 973.04 | 962.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 978.60 | 973.04 | 962.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 953.30 | 969.09 | 961.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 953.30 | 969.09 | 961.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 959.00 | 967.07 | 961.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 960.70 | 967.07 | 961.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 959.00 | 965.93 | 962.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:45:00 | 960.50 | 965.93 | 962.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 964.00 | 965.54 | 962.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 15:00:00 | 967.70 | 965.97 | 963.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 09:15:00 | 980.50 | 965.78 | 963.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 984.90 | 992.68 | 993.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-06 10:15:00 | 984.90 | 992.68 | 993.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 10:15:00 | 984.90 | 992.68 | 993.26 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 999.85 | 994.21 | 993.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 1007.50 | 996.87 | 995.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 11:15:00 | 1072.45 | 1088.60 | 1072.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 11:15:00 | 1072.45 | 1088.60 | 1072.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 1072.45 | 1088.60 | 1072.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 1072.45 | 1088.60 | 1072.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 1079.40 | 1086.76 | 1072.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:30:00 | 1071.50 | 1086.76 | 1072.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 1075.50 | 1084.51 | 1073.18 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 1057.80 | 1069.53 | 1069.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 09:15:00 | 1039.80 | 1057.35 | 1063.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 10:15:00 | 1038.40 | 1037.69 | 1047.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:45:00 | 1039.15 | 1037.69 | 1047.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 1049.00 | 1039.95 | 1047.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 1049.00 | 1039.95 | 1047.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 1038.90 | 1039.74 | 1046.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 09:30:00 | 1031.60 | 1039.23 | 1044.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-18 15:15:00 | 1051.65 | 1045.34 | 1045.38 | SL hit (close>static) qty=1.00 sl=1050.95 alert=retest2 |

### Cycle 67 — BUY (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 09:15:00 | 1046.80 | 1045.63 | 1045.51 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 1037.50 | 1044.01 | 1044.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 1031.70 | 1041.55 | 1043.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 1029.75 | 1028.15 | 1034.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 10:45:00 | 1030.75 | 1028.15 | 1034.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 1032.00 | 1030.11 | 1033.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 1032.00 | 1030.11 | 1033.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 1031.50 | 1030.39 | 1033.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:45:00 | 1033.40 | 1030.39 | 1033.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 1039.60 | 1032.23 | 1034.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 1039.40 | 1032.23 | 1034.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 1025.75 | 1030.94 | 1033.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 10:30:00 | 1023.20 | 1029.80 | 1032.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:15:00 | 1021.65 | 1029.80 | 1032.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:45:00 | 1021.75 | 1026.70 | 1030.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 1017.65 | 1026.87 | 1029.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 1012.85 | 1024.07 | 1028.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1037.50 | 1029.30 | 1028.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1037.50 | 1029.30 | 1028.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1037.50 | 1029.30 | 1028.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 10:15:00 | 1037.50 | 1029.30 | 1028.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 1037.50 | 1029.30 | 1028.29 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 1023.60 | 1029.63 | 1029.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 1022.00 | 1025.99 | 1027.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 10:15:00 | 1025.90 | 1025.02 | 1026.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 10:15:00 | 1025.90 | 1025.02 | 1026.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 1025.90 | 1025.02 | 1026.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-02 11:00:00 | 1025.90 | 1025.02 | 1026.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 11:15:00 | 1015.30 | 1023.08 | 1025.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 12:30:00 | 1011.90 | 1020.10 | 1024.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 09:15:00 | 999.60 | 1018.02 | 1021.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 1033.20 | 1016.97 | 1016.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 11:15:00 | 1033.20 | 1016.97 | 1016.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 11:15:00 | 1033.20 | 1016.97 | 1016.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 12:15:00 | 1040.00 | 1021.58 | 1018.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 09:15:00 | 1024.90 | 1024.98 | 1021.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-06 10:00:00 | 1024.90 | 1024.98 | 1021.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 1023.20 | 1024.62 | 1021.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 1022.80 | 1024.62 | 1021.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1022.90 | 1024.02 | 1021.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:45:00 | 1023.10 | 1024.02 | 1021.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1022.60 | 1023.73 | 1021.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 14:00:00 | 1022.60 | 1023.73 | 1021.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1023.70 | 1023.73 | 1021.96 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 991.10 | 1017.04 | 1019.22 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 1036.00 | 1018.15 | 1017.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 10:15:00 | 1040.00 | 1022.52 | 1019.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 1039.40 | 1039.59 | 1031.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 10:15:00 | 1039.40 | 1039.59 | 1031.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 1039.40 | 1039.59 | 1031.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:45:00 | 1035.30 | 1039.59 | 1031.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1035.00 | 1037.41 | 1032.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 1034.60 | 1037.41 | 1032.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1026.00 | 1035.13 | 1031.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1026.00 | 1035.13 | 1031.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 15:15:00 | 1022.00 | 1032.50 | 1030.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:15:00 | 1003.70 | 1032.50 | 1030.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 1005.70 | 1027.14 | 1028.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 14:15:00 | 995.00 | 1012.62 | 1020.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 978.10 | 972.67 | 985.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 973.90 | 972.67 | 985.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 988.30 | 976.66 | 985.11 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 13:15:00 | 1002.00 | 989.32 | 989.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 1008.00 | 993.06 | 990.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 11:15:00 | 1012.20 | 1016.46 | 1008.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 11:15:00 | 1012.20 | 1016.46 | 1008.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 11:15:00 | 1012.20 | 1016.46 | 1008.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:00:00 | 1012.20 | 1016.46 | 1008.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 995.50 | 1012.26 | 1007.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:00:00 | 995.50 | 1012.26 | 1007.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 1001.00 | 1010.01 | 1006.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 1027.70 | 1005.92 | 1005.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 11:30:00 | 1009.00 | 1006.77 | 1006.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 12:45:00 | 1008.40 | 1007.16 | 1006.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 993.50 | 1004.43 | 1005.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 993.50 | 1004.43 | 1005.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 13:15:00 | 993.50 | 1004.43 | 1005.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 993.50 | 1004.43 | 1005.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 985.80 | 1000.70 | 1003.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-25 09:15:00 | 962.00 | 951.54 | 962.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 09:15:00 | 962.00 | 951.54 | 962.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 962.00 | 951.54 | 962.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-27 09:15:00 | 938.90 | 955.11 | 959.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 13:15:00 | 946.35 | 938.40 | 938.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 946.35 | 938.40 | 938.35 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 923.90 | 936.94 | 937.86 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 945.05 | 938.83 | 938.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 09:15:00 | 953.85 | 942.54 | 940.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 945.00 | 946.62 | 943.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 15:15:00 | 945.00 | 946.62 | 943.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 945.00 | 946.62 | 943.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 965.00 | 946.62 | 943.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 13:15:00 | 942.90 | 952.82 | 952.09 | SL hit (close<static) qty=1.00 sl=943.20 alert=retest2 |

### Cycle 80 — SELL (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 14:15:00 | 942.90 | 950.84 | 951.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 933.85 | 940.56 | 944.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 952.90 | 942.14 | 944.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 952.90 | 942.14 | 944.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 952.90 | 942.14 | 944.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 09:45:00 | 960.00 | 942.14 | 944.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 951.40 | 943.99 | 945.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-15 11:15:00 | 945.25 | 943.99 | 945.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 10:15:00 | 951.25 | 946.33 | 945.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 10:15:00 | 951.25 | 946.33 | 945.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 953.95 | 949.27 | 947.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 10:15:00 | 945.60 | 948.54 | 947.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 10:15:00 | 945.60 | 948.54 | 947.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 945.60 | 948.54 | 947.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 11:00:00 | 945.60 | 948.54 | 947.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 939.45 | 946.72 | 946.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 12:00:00 | 939.45 | 946.72 | 946.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2026-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 12:15:00 | 939.80 | 945.34 | 945.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-17 14:15:00 | 936.35 | 942.66 | 944.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-20 12:15:00 | 939.80 | 938.19 | 941.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 12:15:00 | 939.80 | 938.19 | 941.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 12:15:00 | 939.80 | 938.19 | 941.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 12:45:00 | 941.60 | 938.19 | 941.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 968.20 | 944.08 | 942.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 10:15:00 | 976.15 | 967.45 | 958.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-24 15:15:00 | 999.00 | 1003.18 | 995.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-27 09:15:00 | 1014.55 | 1003.18 | 995.04 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-28 10:15:00 | 1065.28 | 1030.90 | 1015.93 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 1068.00 | 1075.37 | 1057.67 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-30 09:15:00 | 1068.00 | 1075.37 | 1057.67 | SL hit (close<ema200) qty=0.50 sl=1075.37 alert=retest1 |
| ALERT3_SIDEWAYS | 2026-04-30 09:30:00 | 1058.15 | 1075.37 | 1057.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 1064.00 | 1071.57 | 1065.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:00:00 | 1064.00 | 1071.57 | 1065.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 1061.70 | 1069.60 | 1065.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:45:00 | 1061.00 | 1069.60 | 1065.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 13:15:00 | 1058.70 | 1067.42 | 1064.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 13:30:00 | 1058.20 | 1067.42 | 1064.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 15:15:00 | 1058.50 | 1064.27 | 1063.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 1069.70 | 1064.27 | 1063.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 12:45:00 | 1065.00 | 1064.90 | 1064.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 11:45:00 | 1063.10 | 1066.02 | 1065.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:45:00 | 1064.50 | 1065.34 | 1064.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 1074.50 | 1068.54 | 1066.59 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 12:30:00 | 986.60 | 2025-05-20 10:15:00 | 969.85 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-05-20 10:15:00 | 969.55 | 2025-05-20 10:15:00 | 969.85 | STOP_HIT | 1.00 | 0.03% |
| BUY | retest2 | 2025-05-28 10:45:00 | 1008.85 | 2025-05-30 14:15:00 | 985.05 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2025-05-29 15:15:00 | 1003.85 | 2025-05-30 14:15:00 | 985.05 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-05-30 10:45:00 | 1001.60 | 2025-05-30 14:15:00 | 985.05 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-30 12:45:00 | 1001.55 | 2025-05-30 14:15:00 | 985.05 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-06-11 10:45:00 | 1008.00 | 2025-06-13 09:15:00 | 957.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-06-11 10:45:00 | 1008.00 | 2025-06-16 13:15:00 | 958.25 | STOP_HIT | 0.50 | 4.94% |
| SELL | retest2 | 2025-07-15 11:15:00 | 987.80 | 2025-07-15 11:15:00 | 1005.60 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2025-07-23 12:15:00 | 1026.80 | 2025-07-24 11:15:00 | 1008.65 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-07-24 09:15:00 | 1024.00 | 2025-07-24 11:15:00 | 1008.65 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-07-30 14:30:00 | 986.90 | 2025-07-31 11:15:00 | 995.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-07-30 15:15:00 | 986.50 | 2025-07-31 11:15:00 | 995.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-31 10:00:00 | 988.05 | 2025-07-31 11:15:00 | 995.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-08-04 10:45:00 | 1002.50 | 2025-08-08 09:15:00 | 1102.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-26 09:30:00 | 1116.45 | 2025-08-29 09:15:00 | 1105.75 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-08-28 09:15:00 | 1122.65 | 2025-08-29 09:15:00 | 1105.75 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2025-09-24 11:45:00 | 1104.80 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-09-24 12:30:00 | 1106.10 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-09-26 09:15:00 | 1088.60 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2025-09-26 12:15:00 | 1104.20 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-09-29 11:30:00 | 1103.00 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-09-29 15:00:00 | 1100.00 | 2025-10-03 13:15:00 | 1111.20 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-08 13:45:00 | 1094.20 | 2025-10-09 09:15:00 | 1105.80 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-10-08 15:00:00 | 1093.50 | 2025-10-09 09:15:00 | 1105.80 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-10-09 13:00:00 | 1094.40 | 2025-10-13 09:15:00 | 1098.80 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-10-10 11:45:00 | 1094.60 | 2025-10-13 09:15:00 | 1098.80 | STOP_HIT | 1.00 | -0.38% |
| SELL | retest2 | 2025-10-10 13:30:00 | 1091.90 | 2025-10-13 11:15:00 | 1100.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-10-10 14:00:00 | 1091.90 | 2025-10-13 11:15:00 | 1100.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-10-15 10:00:00 | 1110.20 | 2025-10-15 13:15:00 | 1099.10 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-10-15 10:30:00 | 1110.50 | 2025-10-15 13:15:00 | 1099.10 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-10-16 09:15:00 | 1111.80 | 2025-10-16 14:15:00 | 1096.80 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-11-04 09:30:00 | 1190.20 | 2025-11-07 09:15:00 | 1130.69 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 09:30:00 | 1190.20 | 2025-11-07 11:15:00 | 1148.60 | STOP_HIT | 0.50 | 3.50% |
| BUY | retest2 | 2025-12-04 12:45:00 | 1103.80 | 2025-12-04 15:15:00 | 1090.10 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-12-04 14:15:00 | 1103.30 | 2025-12-04 15:15:00 | 1090.10 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-12-10 10:45:00 | 1072.00 | 2025-12-15 10:15:00 | 1119.40 | STOP_HIT | 1.00 | -4.42% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1057.10 | 2025-12-26 11:15:00 | 1004.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-22 09:15:00 | 1056.90 | 2025-12-26 11:15:00 | 1004.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-18 09:15:00 | 1057.10 | 2025-12-29 14:15:00 | 1018.80 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2025-12-22 09:15:00 | 1056.90 | 2025-12-29 14:15:00 | 1018.80 | STOP_HIT | 0.50 | 3.60% |
| BUY | retest2 | 2026-01-02 09:15:00 | 1033.20 | 2026-01-08 12:15:00 | 1020.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-01-16 15:00:00 | 1029.60 | 2026-01-19 09:15:00 | 1009.70 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-22 12:15:00 | 954.30 | 2026-01-29 15:15:00 | 945.30 | STOP_HIT | 1.00 | 0.94% |
| BUY | retest2 | 2026-02-02 15:00:00 | 967.70 | 2026-02-06 10:15:00 | 984.90 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2026-02-03 09:15:00 | 980.50 | 2026-02-06 10:15:00 | 984.90 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2026-02-18 09:30:00 | 1031.60 | 2026-02-18 15:15:00 | 1051.65 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-02-23 10:30:00 | 1023.20 | 2026-02-25 10:15:00 | 1037.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2026-02-23 11:15:00 | 1021.65 | 2026-02-25 10:15:00 | 1037.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-02-23 13:45:00 | 1021.75 | 2026-02-25 10:15:00 | 1037.50 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2026-02-24 09:15:00 | 1017.65 | 2026-02-25 10:15:00 | 1037.50 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2026-03-02 12:30:00 | 1011.90 | 2026-03-05 11:15:00 | 1033.20 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2026-03-04 09:15:00 | 999.60 | 2026-03-05 11:15:00 | 1033.20 | STOP_HIT | 1.00 | -3.36% |
| BUY | retest2 | 2026-03-20 09:15:00 | 1027.70 | 2026-03-20 13:15:00 | 993.50 | STOP_HIT | 1.00 | -3.33% |
| BUY | retest2 | 2026-03-20 11:30:00 | 1009.00 | 2026-03-20 13:15:00 | 993.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-20 12:45:00 | 1008.40 | 2026-03-20 13:15:00 | 993.50 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-03-27 09:15:00 | 938.90 | 2026-04-01 13:15:00 | 946.35 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-04-08 09:15:00 | 965.00 | 2026-04-09 13:15:00 | 942.90 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-04-15 11:15:00 | 945.25 | 2026-04-16 10:15:00 | 951.25 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest1 | 2026-04-27 09:15:00 | 1014.55 | 2026-04-28 10:15:00 | 1065.28 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-27 09:15:00 | 1014.55 | 2026-04-30 09:15:00 | 1068.00 | STOP_HIT | 0.50 | 5.27% |
