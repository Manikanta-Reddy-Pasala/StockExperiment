# Nuvama Wealth Management Ltd. (NUVAMA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 1631.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 173 |
| ALERT1 | 103 |
| ALERT2 | 102 |
| ALERT2_SKIP | 40 |
| ALERT3 | 244 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 107 |
| PARTIAL | 13 |
| TARGET_HIT | 8 |
| STOP_HIT | 107 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 128 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 81
- **Target hits / Stop hits / Partials:** 8 / 107 / 13
- **Avg / median % per leg:** 0.24% / -1.06%
- **Sum % (uncompounded):** 30.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 47 | 19 | 40.4% | 6 | 39 | 2 | 0.60% | 28.1% |
| BUY @ 2nd Alert (retest1) | 5 | 4 | 80.0% | 1 | 2 | 2 | 4.05% | 20.2% |
| BUY @ 3rd Alert (retest2) | 42 | 15 | 35.7% | 5 | 37 | 0 | 0.19% | 7.9% |
| SELL (all) | 81 | 28 | 34.6% | 2 | 68 | 11 | 0.03% | 2.5% |
| SELL @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -1.26% | -6.3% |
| SELL @ 3rd Alert (retest2) | 76 | 27 | 35.5% | 2 | 63 | 11 | 0.12% | 8.8% |
| retest1 (combined) | 10 | 5 | 50.0% | 1 | 7 | 2 | 1.40% | 14.0% |
| retest2 (combined) | 118 | 42 | 35.6% | 7 | 100 | 11 | 0.14% | 16.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 972.62 | 962.42 | 961.57 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 951.60 | 963.79 | 963.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 09:15:00 | 944.51 | 954.65 | 958.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 10:15:00 | 955.65 | 954.85 | 958.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 10:15:00 | 955.65 | 954.85 | 958.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 955.65 | 954.85 | 958.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:45:00 | 956.77 | 954.85 | 958.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 11:15:00 | 965.98 | 957.07 | 959.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 12:00:00 | 965.98 | 957.07 | 959.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 12:15:00 | 969.20 | 959.50 | 960.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-27 13:00:00 | 969.20 | 959.50 | 960.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 13:15:00 | 972.47 | 962.09 | 961.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 09:15:00 | 1000.00 | 972.73 | 966.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-30 09:15:00 | 972.70 | 1003.64 | 997.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-30 09:15:00 | 972.70 | 1003.64 | 997.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 972.70 | 1003.64 | 997.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 10:00:00 | 972.70 | 1003.64 | 997.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 10:15:00 | 966.00 | 996.11 | 994.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-30 11:00:00 | 966.00 | 996.11 | 994.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 965.07 | 989.90 | 991.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 963.19 | 981.36 | 987.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 973.48 | 966.90 | 973.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 973.48 | 966.90 | 973.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 973.48 | 966.90 | 973.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:30:00 | 967.63 | 967.56 | 972.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 13:15:00 | 1008.59 | 980.60 | 977.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 1008.59 | 980.60 | 977.78 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 931.17 | 975.66 | 977.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 870.00 | 954.53 | 967.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 11:15:00 | 928.03 | 919.80 | 939.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:00:00 | 928.03 | 919.80 | 939.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 930.54 | 923.27 | 936.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:45:00 | 937.68 | 923.27 | 936.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 935.21 | 925.66 | 936.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 968.16 | 925.66 | 936.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 956.01 | 931.73 | 937.96 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 11:15:00 | 975.86 | 947.52 | 944.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 12:15:00 | 979.39 | 953.89 | 947.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 1011.45 | 1013.61 | 994.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 11:45:00 | 1010.74 | 1013.61 | 994.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 1007.00 | 1012.00 | 1000.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:15:00 | 1020.20 | 1012.37 | 1002.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 09:15:00 | 1022.38 | 1015.12 | 1007.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 14:30:00 | 1022.93 | 1017.62 | 1012.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 12:15:00 | 1004.02 | 1009.52 | 1010.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 12:15:00 | 1004.02 | 1009.52 | 1010.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-13 13:15:00 | 1001.09 | 1007.83 | 1009.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 14:15:00 | 1018.98 | 1010.06 | 1010.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 14:15:00 | 1018.98 | 1010.06 | 1010.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 14:15:00 | 1018.98 | 1010.06 | 1010.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-13 15:00:00 | 1018.98 | 1010.06 | 1010.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-06-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 15:15:00 | 1019.50 | 1011.95 | 1011.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 09:15:00 | 1024.93 | 1016.99 | 1014.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 13:15:00 | 1038.00 | 1048.10 | 1037.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-20 13:15:00 | 1038.00 | 1048.10 | 1037.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 13:15:00 | 1038.00 | 1048.10 | 1037.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 14:00:00 | 1038.00 | 1048.10 | 1037.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 1052.94 | 1049.07 | 1038.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 15:15:00 | 1057.60 | 1049.07 | 1038.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 1009.20 | 1042.46 | 1037.82 | SL hit (close<static) qty=1.00 sl=1038.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 11:15:00 | 1009.60 | 1030.69 | 1032.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 09:15:00 | 1003.47 | 1018.47 | 1025.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 11:15:00 | 1006.98 | 1001.17 | 1009.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-25 11:45:00 | 1005.84 | 1001.17 | 1009.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 995.00 | 987.27 | 994.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:00:00 | 995.00 | 987.27 | 994.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 992.25 | 988.26 | 994.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:30:00 | 998.00 | 988.26 | 994.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 1000.00 | 987.78 | 991.08 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 11:15:00 | 1002.03 | 993.71 | 993.39 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-06-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 15:15:00 | 986.40 | 992.44 | 993.05 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 1024.98 | 998.95 | 995.95 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 09:15:00 | 990.02 | 1004.20 | 1005.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-04 13:15:00 | 979.41 | 992.92 | 999.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 11:15:00 | 948.37 | 944.67 | 955.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 11:45:00 | 949.29 | 944.67 | 955.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 954.90 | 946.72 | 955.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:00:00 | 954.90 | 946.72 | 955.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 973.77 | 952.13 | 956.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 973.77 | 952.13 | 956.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 981.27 | 957.96 | 959.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:30:00 | 975.21 | 957.96 | 959.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 15:15:00 | 972.00 | 960.76 | 960.28 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 954.43 | 959.86 | 959.97 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 11:15:00 | 964.00 | 960.69 | 960.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 12:15:00 | 975.28 | 963.61 | 961.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 09:15:00 | 957.31 | 965.54 | 963.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 957.31 | 965.54 | 963.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 957.31 | 965.54 | 963.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:00:00 | 957.31 | 965.54 | 963.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 955.32 | 963.50 | 962.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 10:30:00 | 957.84 | 963.50 | 962.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2024-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 11:15:00 | 957.21 | 962.24 | 962.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-12 09:15:00 | 947.24 | 956.24 | 959.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 10:15:00 | 956.67 | 956.33 | 958.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 10:15:00 | 956.67 | 956.33 | 958.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 956.67 | 956.33 | 958.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:30:00 | 959.67 | 956.33 | 958.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 962.00 | 957.46 | 959.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:00:00 | 962.00 | 957.46 | 959.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 959.26 | 957.82 | 959.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 09:15:00 | 947.99 | 958.69 | 959.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 966.39 | 960.35 | 959.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 966.39 | 960.35 | 959.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 11:15:00 | 979.46 | 965.19 | 961.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 15:15:00 | 980.00 | 981.42 | 975.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 09:15:00 | 972.58 | 981.42 | 975.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 961.81 | 977.50 | 974.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 961.81 | 977.50 | 974.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 953.56 | 972.71 | 972.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:45:00 | 953.91 | 972.71 | 972.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 11:15:00 | 952.99 | 968.77 | 970.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 939.78 | 948.17 | 954.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 14:15:00 | 977.47 | 953.47 | 955.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 14:15:00 | 977.47 | 953.47 | 955.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 977.47 | 953.47 | 955.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:45:00 | 974.80 | 953.47 | 955.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-07-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 15:15:00 | 978.95 | 958.56 | 958.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 994.32 | 965.72 | 961.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 09:15:00 | 1314.64 | 1315.31 | 1283.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 09:15:00 | 1314.64 | 1315.31 | 1283.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 1314.64 | 1315.31 | 1283.26 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 1204.21 | 1271.06 | 1277.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 1166.43 | 1250.13 | 1267.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 09:15:00 | 1190.80 | 1186.08 | 1210.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 1190.80 | 1186.08 | 1210.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 1190.80 | 1186.08 | 1210.78 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 1237.04 | 1214.81 | 1212.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 1262.80 | 1231.63 | 1222.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 14:15:00 | 1262.75 | 1263.21 | 1244.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 15:00:00 | 1262.75 | 1263.21 | 1244.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 1262.29 | 1262.89 | 1247.57 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 12:15:00 | 1234.38 | 1243.06 | 1243.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 1232.25 | 1240.90 | 1242.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 10:15:00 | 1231.85 | 1223.09 | 1228.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 10:15:00 | 1231.85 | 1223.09 | 1228.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 1231.85 | 1223.09 | 1228.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:45:00 | 1234.01 | 1223.09 | 1228.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 1231.69 | 1224.81 | 1229.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:30:00 | 1235.48 | 1224.81 | 1229.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 14:15:00 | 1233.02 | 1227.48 | 1229.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 15:00:00 | 1233.02 | 1227.48 | 1229.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 15:15:00 | 1224.00 | 1226.79 | 1228.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-19 09:15:00 | 1243.60 | 1226.79 | 1228.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 1269.41 | 1235.31 | 1232.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 13:15:00 | 1298.71 | 1254.22 | 1242.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 14:15:00 | 1302.95 | 1312.26 | 1286.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 15:00:00 | 1302.95 | 1312.26 | 1286.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 1264.81 | 1300.81 | 1285.40 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-21 13:15:00 | 1253.77 | 1274.80 | 1276.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-21 15:15:00 | 1245.00 | 1265.16 | 1271.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 09:15:00 | 1252.87 | 1249.77 | 1258.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-23 10:00:00 | 1252.87 | 1249.77 | 1258.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 10:15:00 | 1248.99 | 1249.61 | 1257.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 10:45:00 | 1244.97 | 1255.14 | 1257.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 11:15:00 | 1247.03 | 1255.14 | 1257.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 13:15:00 | 1265.60 | 1256.19 | 1257.23 | SL hit (close>static) qty=1.00 sl=1258.73 alert=retest2 |

### Cycle 27 — BUY (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-26 15:15:00 | 1263.52 | 1258.76 | 1258.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-27 09:15:00 | 1293.79 | 1265.77 | 1261.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-28 11:15:00 | 1304.00 | 1312.93 | 1295.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-28 11:45:00 | 1309.38 | 1312.93 | 1295.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 12:15:00 | 1292.98 | 1308.94 | 1295.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 12:30:00 | 1290.99 | 1308.94 | 1295.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 1290.00 | 1305.15 | 1294.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 09:15:00 | 1295.98 | 1300.24 | 1294.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 11:00:00 | 1300.86 | 1301.70 | 1295.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-29 12:15:00 | 1298.60 | 1299.38 | 1295.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 15:15:00 | 1284.90 | 1293.09 | 1293.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 15:15:00 | 1284.90 | 1293.09 | 1293.42 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-08-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 09:15:00 | 1298.42 | 1294.15 | 1293.87 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 11:15:00 | 1284.16 | 1291.93 | 1292.89 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 09:15:00 | 1333.99 | 1297.81 | 1294.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 14:15:00 | 1349.59 | 1326.13 | 1313.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 11:15:00 | 1333.09 | 1339.02 | 1324.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 12:15:00 | 1323.19 | 1335.86 | 1324.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 1323.19 | 1335.86 | 1324.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:00:00 | 1323.19 | 1335.86 | 1324.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 1337.56 | 1336.20 | 1325.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 09:30:00 | 1345.09 | 1338.28 | 1329.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 1312.88 | 1325.45 | 1326.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 09:15:00 | 1312.88 | 1325.45 | 1326.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 13:15:00 | 1304.09 | 1318.03 | 1322.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 15:15:00 | 1318.00 | 1316.25 | 1320.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 1297.17 | 1312.43 | 1318.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 1297.17 | 1312.43 | 1318.41 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 15:15:00 | 1347.23 | 1323.77 | 1320.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 11:15:00 | 1373.65 | 1344.71 | 1335.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 14:15:00 | 1368.66 | 1372.01 | 1358.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 14:45:00 | 1370.48 | 1372.01 | 1358.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 10:15:00 | 1362.97 | 1381.13 | 1374.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 11:00:00 | 1362.97 | 1381.13 | 1374.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 11:15:00 | 1363.03 | 1377.51 | 1373.23 | EMA400 retest candle locked (from upside) |

### Cycle 34 — SELL (started 2024-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-16 14:15:00 | 1336.74 | 1364.98 | 1368.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-16 15:15:00 | 1326.40 | 1357.27 | 1364.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 09:15:00 | 1348.22 | 1339.29 | 1348.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 1348.22 | 1339.29 | 1348.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 1348.22 | 1339.29 | 1348.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:00:00 | 1348.22 | 1339.29 | 1348.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 1349.47 | 1341.33 | 1348.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 10:45:00 | 1354.35 | 1341.33 | 1348.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 1343.20 | 1341.70 | 1348.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:45:00 | 1350.79 | 1341.70 | 1348.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 1320.00 | 1332.17 | 1340.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 10:15:00 | 1309.61 | 1332.17 | 1340.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 12:45:00 | 1309.69 | 1321.56 | 1332.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 1304.00 | 1318.43 | 1329.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-20 12:15:00 | 1363.42 | 1336.65 | 1334.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 1363.42 | 1336.65 | 1334.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 13:15:00 | 1395.00 | 1348.32 | 1339.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 15:15:00 | 1373.20 | 1374.15 | 1362.46 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-09-24 09:15:00 | 1389.67 | 1374.15 | 1362.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 1339.24 | 1367.17 | 1360.35 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-24 09:15:00 | 1339.24 | 1367.17 | 1360.35 | SL hit (close<ema400) qty=1.00 sl=1360.35 alert=retest1 |

### Cycle 36 — SELL (started 2024-09-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 13:15:00 | 1343.14 | 1355.87 | 1356.60 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-09-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 14:15:00 | 1362.12 | 1357.12 | 1357.10 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 11:15:00 | 1345.00 | 1356.61 | 1357.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 1337.70 | 1349.19 | 1352.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 1350.21 | 1340.48 | 1346.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 14:15:00 | 1350.21 | 1340.48 | 1346.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 1350.21 | 1340.48 | 1346.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 1350.21 | 1340.48 | 1346.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 1365.00 | 1345.38 | 1347.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 1387.24 | 1345.38 | 1347.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 1389.50 | 1354.21 | 1351.57 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 15:15:00 | 1344.00 | 1359.05 | 1360.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 09:15:00 | 1317.84 | 1350.81 | 1356.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 09:15:00 | 1219.31 | 1207.15 | 1240.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 09:45:00 | 1223.60 | 1207.15 | 1240.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 1220.00 | 1206.62 | 1223.00 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2024-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 14:15:00 | 1265.24 | 1236.49 | 1233.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 1275.98 | 1248.48 | 1239.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 14:15:00 | 1276.22 | 1278.84 | 1266.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 15:00:00 | 1276.22 | 1278.84 | 1266.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 1400.61 | 1423.09 | 1409.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 12:45:00 | 1432.56 | 1422.71 | 1412.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-18 14:15:00 | 1430.00 | 1423.45 | 1414.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-21 09:15:00 | 1432.08 | 1424.24 | 1416.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-21 15:15:00 | 1389.20 | 1409.22 | 1411.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 15:15:00 | 1389.20 | 1409.22 | 1411.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 1341.02 | 1397.63 | 1406.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 1392.05 | 1357.81 | 1376.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 09:15:00 | 1392.05 | 1357.81 | 1376.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 1392.05 | 1357.81 | 1376.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:00:00 | 1392.05 | 1357.81 | 1376.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 1373.17 | 1360.88 | 1376.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 10:15:00 | 1331.88 | 1370.21 | 1374.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 09:15:00 | 1321.01 | 1336.63 | 1351.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 12:45:00 | 1332.66 | 1339.92 | 1348.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 10:15:00 | 1335.21 | 1323.13 | 1329.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 10:15:00 | 1350.03 | 1328.51 | 1331.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-30 11:00:00 | 1350.03 | 1328.51 | 1331.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-10-30 12:15:00 | 1351.12 | 1336.05 | 1334.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-10-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 12:15:00 | 1351.12 | 1336.05 | 1334.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 12:15:00 | 1369.00 | 1349.43 | 1342.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 15:15:00 | 1396.80 | 1400.11 | 1384.00 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-11-05 09:15:00 | 1412.43 | 1400.11 | 1384.00 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 1393.20 | 1399.62 | 1387.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:45:00 | 1386.05 | 1399.62 | 1387.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 12:15:00 | 1399.40 | 1399.58 | 1388.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:15:00 | 1408.64 | 1399.58 | 1388.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-06 13:15:00 | 1483.05 | 1443.20 | 1419.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-07 15:15:00 | 1467.20 | 1472.77 | 1452.39 | SL hit (close<ema200) qty=0.50 sl=1472.77 alert=retest1 |

### Cycle 44 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 1421.00 | 1443.74 | 1444.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 1409.20 | 1436.83 | 1440.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 1331.77 | 1300.14 | 1325.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 1331.77 | 1300.14 | 1325.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 1331.77 | 1300.14 | 1325.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 1331.77 | 1300.14 | 1325.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 1331.00 | 1306.31 | 1326.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:15:00 | 1342.84 | 1306.31 | 1326.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 11:15:00 | 1345.00 | 1314.05 | 1328.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 11:30:00 | 1343.48 | 1314.05 | 1328.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 1331.09 | 1333.27 | 1333.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 11:15:00 | 1330.00 | 1333.27 | 1333.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 1333.65 | 1333.35 | 1333.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 1333.65 | 1333.35 | 1333.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 1335.40 | 1333.76 | 1333.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 1335.76 | 1333.76 | 1333.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 1329.03 | 1332.81 | 1333.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 15:15:00 | 1319.00 | 1331.25 | 1332.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-19 10:15:00 | 1337.06 | 1331.46 | 1332.30 | SL hit (close>static) qty=1.00 sl=1336.36 alert=retest2 |

### Cycle 45 — BUY (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 14:15:00 | 1320.15 | 1310.83 | 1310.22 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 15:15:00 | 1306.60 | 1310.31 | 1310.65 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 09:15:00 | 1335.20 | 1315.29 | 1312.88 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 11:15:00 | 1305.94 | 1316.53 | 1317.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 13:15:00 | 1298.92 | 1312.55 | 1315.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 14:15:00 | 1315.00 | 1313.04 | 1315.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 14:15:00 | 1315.00 | 1313.04 | 1315.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 1315.00 | 1313.04 | 1315.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 15:00:00 | 1315.00 | 1313.04 | 1315.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 1320.60 | 1314.55 | 1316.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 1335.80 | 1314.55 | 1316.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2024-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 09:15:00 | 1332.00 | 1318.04 | 1317.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 12:15:00 | 1350.65 | 1338.12 | 1330.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 15:15:00 | 1340.00 | 1340.46 | 1333.78 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:15:00 | 1363.72 | 1340.46 | 1333.78 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 1359.43 | 1351.75 | 1344.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 10:30:00 | 1373.35 | 1356.85 | 1347.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 13:45:00 | 1369.73 | 1365.75 | 1354.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 14:45:00 | 1369.00 | 1366.22 | 1355.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:15:00 | 1370.00 | 1366.22 | 1355.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 15:15:00 | 1392.00 | 1395.89 | 1386.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 09:15:00 | 1400.77 | 1395.89 | 1386.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-10 09:15:00 | 1431.91 | 1406.33 | 1392.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Target hit | 2024-12-11 09:15:00 | 1500.09 | 1466.42 | 1434.89 | Target hit (10%) qty=0.50 alert=retest1 |

### Cycle 50 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 1374.05 | 1434.32 | 1437.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-23 09:15:00 | 1364.50 | 1372.03 | 1375.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 09:15:00 | 1374.40 | 1367.54 | 1370.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-24 09:15:00 | 1374.40 | 1367.54 | 1370.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 1374.40 | 1367.54 | 1370.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 10:00:00 | 1374.40 | 1367.54 | 1370.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 10:15:00 | 1371.00 | 1368.23 | 1370.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 11:30:00 | 1368.46 | 1368.33 | 1370.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 12:00:00 | 1368.73 | 1368.33 | 1370.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 13:00:00 | 1367.14 | 1368.09 | 1370.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:15:00 | 1365.00 | 1368.60 | 1370.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 15:15:00 | 1365.00 | 1367.88 | 1369.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 09:15:00 | 1374.00 | 1367.88 | 1369.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 1364.84 | 1367.27 | 1369.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 10:15:00 | 1362.04 | 1367.27 | 1369.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 1374.40 | 1370.23 | 1369.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 1374.40 | 1370.23 | 1369.89 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-12-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 13:15:00 | 1368.80 | 1369.71 | 1369.77 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 1370.80 | 1369.75 | 1369.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 14:15:00 | 1376.00 | 1371.39 | 1370.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 1370.89 | 1371.87 | 1370.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 1370.89 | 1371.87 | 1370.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 1370.89 | 1371.87 | 1370.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 1379.94 | 1373.69 | 1372.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 1370.65 | 1404.12 | 1408.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 1370.65 | 1404.12 | 1408.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 1364.24 | 1396.14 | 1404.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 10:15:00 | 1375.68 | 1375.36 | 1387.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 11:00:00 | 1375.68 | 1375.36 | 1387.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 1368.77 | 1374.68 | 1382.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:00:00 | 1348.99 | 1369.54 | 1379.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-13 12:15:00 | 1281.54 | 1303.90 | 1322.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 14:15:00 | 1214.09 | 1275.28 | 1305.66 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 1227.28 | 1225.87 | 1225.75 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 1213.70 | 1223.44 | 1224.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 11:15:00 | 1201.76 | 1219.10 | 1222.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 1133.17 | 1127.17 | 1156.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 1133.17 | 1127.17 | 1156.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 1126.69 | 1131.42 | 1145.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:45:00 | 1118.85 | 1130.90 | 1141.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 09:15:00 | 1062.91 | 1109.83 | 1128.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-27 15:15:00 | 1006.96 | 1052.37 | 1087.15 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 1096.91 | 1069.47 | 1067.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 14:15:00 | 1118.03 | 1093.21 | 1084.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 1093.79 | 1096.94 | 1088.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 1093.79 | 1096.94 | 1088.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 1093.79 | 1096.94 | 1088.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:00:00 | 1093.79 | 1096.94 | 1088.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 1082.45 | 1094.04 | 1087.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 1082.45 | 1094.04 | 1087.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 1083.87 | 1092.01 | 1087.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 1074.84 | 1092.01 | 1087.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 1074.61 | 1088.53 | 1086.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 1074.61 | 1088.53 | 1086.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 1090.66 | 1088.95 | 1086.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 14:15:00 | 1092.76 | 1088.95 | 1086.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 1091.57 | 1089.48 | 1086.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 1021.19 | 1075.58 | 1081.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 09:15:00 | 1021.19 | 1075.58 | 1081.03 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 1117.44 | 1071.21 | 1068.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 12:15:00 | 1156.96 | 1088.36 | 1076.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 1169.33 | 1172.66 | 1143.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 10:45:00 | 1170.00 | 1172.66 | 1143.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 1152.45 | 1170.11 | 1155.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 09:45:00 | 1144.36 | 1170.11 | 1155.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 1151.41 | 1166.37 | 1154.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 1158.93 | 1166.37 | 1154.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 15:15:00 | 1130.00 | 1149.98 | 1150.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 15:15:00 | 1130.00 | 1149.98 | 1150.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 09:15:00 | 1107.65 | 1141.51 | 1146.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 1132.00 | 1126.47 | 1135.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 15:00:00 | 1132.00 | 1126.47 | 1135.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 1107.12 | 1090.90 | 1100.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-13 13:30:00 | 1088.57 | 1095.44 | 1100.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 1086.36 | 1095.35 | 1099.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:00:00 | 1080.55 | 1092.39 | 1097.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1034.14 | 1057.52 | 1074.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1032.04 | 1057.52 | 1074.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 09:15:00 | 1026.52 | 1057.52 | 1074.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-17 11:15:00 | 1062.72 | 1057.61 | 1071.38 | SL hit (close>ema200) qty=0.50 sl=1057.61 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 1081.86 | 1062.87 | 1062.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 14:15:00 | 1102.75 | 1075.44 | 1068.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 1107.20 | 1109.95 | 1094.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 1107.20 | 1109.95 | 1094.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 1104.69 | 1116.05 | 1106.30 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 13:15:00 | 1083.80 | 1100.24 | 1101.26 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 11:15:00 | 1105.24 | 1100.63 | 1100.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-25 14:15:00 | 1121.96 | 1106.49 | 1103.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 15:15:00 | 1106.00 | 1106.39 | 1103.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-27 09:15:00 | 1125.47 | 1106.39 | 1103.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 64 — SELL (started 2025-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 10:15:00 | 1070.21 | 1100.90 | 1101.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 13:15:00 | 1063.80 | 1085.03 | 1093.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 14:15:00 | 1085.89 | 1085.20 | 1092.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-27 15:00:00 | 1085.89 | 1085.20 | 1092.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 1092.54 | 1062.94 | 1074.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 15:00:00 | 1092.54 | 1062.94 | 1074.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 15:15:00 | 1080.46 | 1066.44 | 1074.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:15:00 | 1044.00 | 1066.44 | 1074.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 10:15:00 | 1038.60 | 1032.33 | 1046.64 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 1071.80 | 1047.34 | 1047.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 1092.92 | 1069.38 | 1059.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 15:15:00 | 1088.59 | 1089.43 | 1076.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 09:15:00 | 1096.85 | 1089.43 | 1076.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 1095.80 | 1095.29 | 1086.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-10 09:15:00 | 1109.00 | 1095.29 | 1086.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 1039.80 | 1094.42 | 1094.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 1039.80 | 1094.42 | 1094.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 1033.78 | 1066.47 | 1080.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 12:15:00 | 1045.59 | 1043.92 | 1059.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 12:30:00 | 1043.63 | 1043.92 | 1059.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 1037.88 | 1043.93 | 1056.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 1024.43 | 1042.42 | 1054.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 14:15:00 | 1067.23 | 1044.54 | 1042.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 14:15:00 | 1067.23 | 1044.54 | 1042.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 1098.89 | 1056.89 | 1048.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 1238.83 | 1245.83 | 1223.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-25 14:15:00 | 1238.60 | 1240.06 | 1228.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 14:15:00 | 1238.60 | 1240.06 | 1228.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 14:45:00 | 1225.25 | 1240.06 | 1228.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 1224.14 | 1238.00 | 1233.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 1223.55 | 1238.00 | 1233.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 1229.98 | 1236.40 | 1232.78 | EMA400 retest candle locked (from upside) |

### Cycle 68 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 1216.77 | 1230.33 | 1230.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 11:15:00 | 1204.00 | 1222.58 | 1226.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 1223.98 | 1216.74 | 1222.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 1223.98 | 1216.74 | 1222.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 1223.98 | 1216.74 | 1222.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 1223.98 | 1216.74 | 1222.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 1220.20 | 1217.43 | 1222.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 1228.89 | 1217.43 | 1222.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 1231.56 | 1220.26 | 1223.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:00:00 | 1231.56 | 1220.26 | 1223.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 1227.94 | 1221.80 | 1223.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:30:00 | 1232.44 | 1221.80 | 1223.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 1215.49 | 1221.05 | 1222.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:30:00 | 1215.07 | 1220.77 | 1222.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:15:00 | 1212.20 | 1220.00 | 1222.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:00:00 | 1211.32 | 1217.01 | 1220.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 14:15:00 | 1154.32 | 1180.69 | 1194.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 14:15:00 | 1151.59 | 1180.69 | 1194.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-02 14:15:00 | 1150.75 | 1180.69 | 1194.91 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-03 12:15:00 | 1181.86 | 1174.34 | 1185.36 | SL hit (close>ema200) qty=0.50 sl=1174.34 alert=retest2 |

### Cycle 69 — BUY (started 2025-04-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 11:15:00 | 1112.82 | 1093.43 | 1092.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 1135.30 | 1105.26 | 1098.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 14:15:00 | 1226.10 | 1229.55 | 1221.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 15:00:00 | 1226.10 | 1229.55 | 1221.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 1236.00 | 1230.43 | 1223.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 10:45:00 | 1250.50 | 1234.26 | 1225.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 12:30:00 | 1253.50 | 1240.42 | 1230.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 13:30:00 | 1249.20 | 1251.19 | 1243.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-28 14:45:00 | 1254.60 | 1250.61 | 1243.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 1247.50 | 1249.14 | 1244.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:30:00 | 1252.80 | 1249.14 | 1244.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 10:15:00 | 1250.80 | 1249.47 | 1244.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 11:15:00 | 1254.30 | 1249.47 | 1244.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-30 09:15:00 | 1231.40 | 1243.09 | 1243.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 09:15:00 | 1231.40 | 1243.09 | 1243.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-05 09:15:00 | 1174.10 | 1208.28 | 1219.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-06 09:15:00 | 1190.30 | 1185.39 | 1199.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-06 09:45:00 | 1180.00 | 1185.39 | 1199.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 1183.10 | 1170.37 | 1180.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 1182.00 | 1170.37 | 1180.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 1184.60 | 1173.22 | 1181.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:30:00 | 1186.10 | 1173.22 | 1181.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 1192.20 | 1178.92 | 1182.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:45:00 | 1189.80 | 1178.92 | 1182.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 1182.20 | 1179.57 | 1182.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 1195.80 | 1179.57 | 1182.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 1198.00 | 1183.26 | 1183.78 | EMA400 retest candle locked (from downside) |

### Cycle 71 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 1210.00 | 1188.61 | 1186.16 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 1172.00 | 1186.86 | 1187.04 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-05-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 12:15:00 | 1193.20 | 1187.68 | 1187.24 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 14:15:00 | 1183.90 | 1187.02 | 1187.02 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 15:15:00 | 1190.00 | 1187.62 | 1187.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 1214.00 | 1192.89 | 1189.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 10:15:00 | 1335.20 | 1337.57 | 1315.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 11:00:00 | 1335.20 | 1337.57 | 1315.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 1340.00 | 1339.86 | 1326.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:15:00 | 1348.90 | 1339.86 | 1326.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:45:00 | 1354.40 | 1343.63 | 1329.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 13:15:00 | 1347.50 | 1368.77 | 1365.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 15:15:00 | 1360.00 | 1363.41 | 1363.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 1360.00 | 1363.41 | 1363.45 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 1365.10 | 1363.75 | 1363.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 1370.00 | 1365.63 | 1364.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 13:15:00 | 1360.80 | 1365.15 | 1364.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 13:15:00 | 1360.80 | 1365.15 | 1364.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 13:15:00 | 1360.80 | 1365.15 | 1364.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:00:00 | 1360.80 | 1365.15 | 1364.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 1363.00 | 1364.72 | 1364.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 14:45:00 | 1361.30 | 1364.72 | 1364.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 1370.20 | 1365.82 | 1364.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 1375.00 | 1365.82 | 1364.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 1377.20 | 1368.10 | 1366.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 10:00:00 | 1390.80 | 1378.25 | 1372.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 12:15:00 | 1389.70 | 1382.55 | 1375.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 1392.40 | 1385.83 | 1379.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:30:00 | 1389.50 | 1388.23 | 1382.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 1387.40 | 1388.71 | 1384.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 14:45:00 | 1389.50 | 1388.71 | 1384.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 1384.00 | 1387.77 | 1384.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 1417.00 | 1387.77 | 1384.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 13:30:00 | 1393.20 | 1392.71 | 1388.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 13:15:00 | 1442.20 | 1451.77 | 1451.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 1442.20 | 1451.77 | 1451.91 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 1469.90 | 1454.08 | 1452.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 14:15:00 | 1474.00 | 1464.58 | 1458.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 1500.00 | 1511.61 | 1495.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 1500.00 | 1511.61 | 1495.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 1497.00 | 1508.69 | 1495.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 1493.40 | 1508.69 | 1495.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 1500.20 | 1506.99 | 1495.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:30:00 | 1494.10 | 1506.99 | 1495.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 1474.30 | 1502.46 | 1497.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 1474.30 | 1502.46 | 1497.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 1491.20 | 1500.21 | 1496.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 1475.20 | 1500.21 | 1496.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 1478.80 | 1492.38 | 1493.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 1462.80 | 1480.04 | 1486.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 1418.00 | 1417.60 | 1434.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 14:00:00 | 1418.00 | 1417.60 | 1434.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 1452.10 | 1422.72 | 1432.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 1452.10 | 1422.72 | 1432.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 1446.10 | 1427.40 | 1433.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:15:00 | 1441.00 | 1431.62 | 1435.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 12:15:00 | 1460.30 | 1437.35 | 1437.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 1460.30 | 1437.35 | 1437.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 13:15:00 | 1475.40 | 1444.96 | 1440.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 12:15:00 | 1460.00 | 1461.99 | 1453.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 12:45:00 | 1459.80 | 1461.99 | 1453.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 1450.90 | 1459.74 | 1453.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 1450.90 | 1459.74 | 1453.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 1457.50 | 1459.29 | 1454.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:15:00 | 1458.80 | 1459.29 | 1454.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 1452.30 | 1457.89 | 1453.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 1452.30 | 1457.89 | 1453.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 1431.60 | 1452.64 | 1451.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:00:00 | 1431.60 | 1452.64 | 1451.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 11:15:00 | 1424.00 | 1446.91 | 1449.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 12:15:00 | 1403.40 | 1438.21 | 1445.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 1398.20 | 1394.97 | 1411.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 09:45:00 | 1390.60 | 1394.97 | 1411.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 1408.10 | 1397.60 | 1410.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 1408.00 | 1397.60 | 1410.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 1406.00 | 1399.00 | 1408.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:45:00 | 1401.60 | 1399.00 | 1408.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 1412.00 | 1401.60 | 1408.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 1412.00 | 1401.60 | 1408.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 1410.80 | 1403.44 | 1408.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:15:00 | 1410.80 | 1403.44 | 1408.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 1414.40 | 1405.63 | 1409.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 09:45:00 | 1430.30 | 1405.63 | 1409.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 1419.20 | 1408.35 | 1410.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:45:00 | 1417.50 | 1408.35 | 1410.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 1435.70 | 1413.82 | 1412.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 1468.00 | 1433.11 | 1423.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 15:15:00 | 1640.00 | 1645.17 | 1611.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 09:15:00 | 1644.90 | 1645.17 | 1611.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 1620.00 | 1638.82 | 1614.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 1617.10 | 1638.82 | 1614.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 1671.50 | 1649.86 | 1630.37 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 09:15:00 | 1601.70 | 1626.39 | 1626.98 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 1664.00 | 1633.08 | 1629.87 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 1561.80 | 1618.94 | 1624.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 1478.20 | 1579.96 | 1605.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 1509.50 | 1509.31 | 1554.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 09:45:00 | 1505.00 | 1509.31 | 1554.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1499.80 | 1498.87 | 1525.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 1489.50 | 1497.00 | 1522.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 13:15:00 | 1496.90 | 1495.22 | 1517.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 14:15:00 | 1497.40 | 1495.87 | 1515.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 15:00:00 | 1491.80 | 1495.06 | 1513.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 1486.00 | 1477.91 | 1492.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:30:00 | 1464.40 | 1473.24 | 1482.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 1491.30 | 1482.31 | 1482.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 1491.30 | 1482.31 | 1482.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 1509.10 | 1490.68 | 1486.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 1509.20 | 1510.80 | 1500.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 13:45:00 | 1511.40 | 1510.80 | 1500.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 1499.00 | 1508.44 | 1499.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 14:30:00 | 1502.70 | 1508.44 | 1499.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 1502.40 | 1507.23 | 1500.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 1546.60 | 1507.23 | 1500.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 12:15:00 | 1516.00 | 1531.71 | 1532.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 12:15:00 | 1516.00 | 1531.71 | 1532.48 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 09:15:00 | 1551.40 | 1533.31 | 1532.61 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 1513.30 | 1542.45 | 1543.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 1483.80 | 1530.72 | 1538.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 15:15:00 | 1447.80 | 1447.72 | 1466.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-30 09:15:00 | 1451.00 | 1447.72 | 1466.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 1480.60 | 1455.60 | 1464.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:00:00 | 1480.60 | 1455.60 | 1464.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 1481.60 | 1460.80 | 1465.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 1481.40 | 1460.80 | 1465.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 15:15:00 | 1484.00 | 1468.77 | 1468.74 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 09:15:00 | 1451.50 | 1465.32 | 1467.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 12:15:00 | 1446.30 | 1459.37 | 1463.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 13:15:00 | 1465.60 | 1460.62 | 1463.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 13:15:00 | 1465.60 | 1460.62 | 1463.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 1465.60 | 1460.62 | 1463.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 14:00:00 | 1465.60 | 1460.62 | 1463.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 14:15:00 | 1444.20 | 1457.33 | 1462.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 1408.00 | 1454.87 | 1460.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 1391.40 | 1376.03 | 1375.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 1391.40 | 1376.03 | 1375.33 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 14:15:00 | 1370.10 | 1374.50 | 1374.74 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 10:15:00 | 1380.70 | 1375.87 | 1375.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 13:15:00 | 1386.00 | 1378.87 | 1376.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 10:15:00 | 1382.60 | 1389.40 | 1383.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 10:15:00 | 1382.60 | 1389.40 | 1383.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 1382.60 | 1389.40 | 1383.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 1376.40 | 1389.40 | 1383.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 1374.30 | 1386.38 | 1382.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 12:00:00 | 1374.30 | 1386.38 | 1382.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 12:15:00 | 1370.40 | 1383.18 | 1381.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:00:00 | 1370.40 | 1383.18 | 1381.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 1395.00 | 1386.22 | 1383.01 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 1377.00 | 1381.63 | 1381.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-19 09:15:00 | 1346.40 | 1372.08 | 1376.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 13:15:00 | 1354.20 | 1352.54 | 1360.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:00:00 | 1354.20 | 1352.54 | 1360.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 1370.00 | 1356.03 | 1361.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:15:00 | 1375.40 | 1356.03 | 1361.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 1375.40 | 1359.91 | 1362.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 09:15:00 | 1407.80 | 1359.91 | 1362.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 1413.40 | 1370.61 | 1367.37 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 1376.90 | 1384.73 | 1385.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 09:15:00 | 1362.60 | 1376.66 | 1381.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 12:15:00 | 1295.00 | 1292.52 | 1310.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 13:00:00 | 1295.00 | 1292.52 | 1310.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 1315.20 | 1295.89 | 1300.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 1317.90 | 1295.89 | 1300.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 1306.90 | 1298.09 | 1300.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 12:45:00 | 1299.00 | 1299.42 | 1300.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 15:15:00 | 1307.10 | 1302.60 | 1302.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 1307.10 | 1302.60 | 1302.12 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 1289.80 | 1301.12 | 1301.99 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 1318.40 | 1302.83 | 1302.51 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 1284.70 | 1302.24 | 1302.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 09:15:00 | 1279.60 | 1290.67 | 1295.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 12:15:00 | 1286.70 | 1286.57 | 1292.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 13:00:00 | 1286.70 | 1286.57 | 1292.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 1293.10 | 1283.91 | 1288.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 1290.80 | 1283.91 | 1288.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 1293.00 | 1285.73 | 1289.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 11:00:00 | 1293.00 | 1285.73 | 1289.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 1284.40 | 1285.46 | 1288.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:45:00 | 1275.00 | 1281.21 | 1285.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 1268.40 | 1281.17 | 1285.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 1301.30 | 1272.40 | 1272.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 12:15:00 | 1301.30 | 1272.40 | 1272.13 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 13:15:00 | 1269.90 | 1277.11 | 1277.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-17 09:15:00 | 1264.00 | 1273.94 | 1275.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-17 14:15:00 | 1273.80 | 1270.04 | 1272.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-17 14:15:00 | 1273.80 | 1270.04 | 1272.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 1273.80 | 1270.04 | 1272.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 1271.30 | 1270.04 | 1272.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 1276.90 | 1271.41 | 1273.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 1277.80 | 1271.41 | 1273.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 1282.90 | 1273.71 | 1274.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 1282.90 | 1273.71 | 1274.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 1288.00 | 1276.57 | 1275.41 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 1263.60 | 1277.83 | 1277.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 1254.90 | 1273.25 | 1275.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 14:15:00 | 1240.00 | 1239.85 | 1247.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 15:00:00 | 1240.00 | 1239.85 | 1247.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 1232.00 | 1218.99 | 1229.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 10:00:00 | 1232.00 | 1218.99 | 1229.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 1238.20 | 1222.83 | 1229.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 11:30:00 | 1224.90 | 1222.87 | 1229.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 13:15:00 | 1227.90 | 1224.89 | 1229.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 11:15:00 | 1242.00 | 1232.12 | 1231.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-09-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 11:15:00 | 1242.00 | 1232.12 | 1231.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-29 13:15:00 | 1247.50 | 1237.05 | 1233.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 12:15:00 | 1444.60 | 1444.71 | 1431.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-13 13:00:00 | 1444.60 | 1444.71 | 1431.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 1438.80 | 1442.77 | 1432.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:45:00 | 1432.00 | 1442.77 | 1432.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 1432.00 | 1440.27 | 1435.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:30:00 | 1432.20 | 1440.27 | 1435.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 1426.60 | 1437.53 | 1434.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:00:00 | 1426.60 | 1437.53 | 1434.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 1428.60 | 1435.75 | 1434.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 1428.60 | 1435.75 | 1434.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 1438.00 | 1435.10 | 1434.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 1434.80 | 1435.10 | 1434.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 1445.20 | 1437.12 | 1435.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 1439.90 | 1437.12 | 1435.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 1445.20 | 1443.31 | 1440.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 1441.00 | 1443.31 | 1440.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 1422.40 | 1440.04 | 1439.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 1422.40 | 1440.04 | 1439.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — SELL (started 2025-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 10:15:00 | 1427.60 | 1437.55 | 1438.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 12:15:00 | 1414.00 | 1430.48 | 1435.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 13:15:00 | 1410.60 | 1406.15 | 1416.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 14:00:00 | 1410.60 | 1406.15 | 1416.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 1413.00 | 1406.59 | 1415.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 1421.70 | 1413.89 | 1417.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 14:15:00 | 1450.00 | 1421.11 | 1420.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 10:15:00 | 1467.70 | 1445.88 | 1439.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 1444.60 | 1475.47 | 1468.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 1444.60 | 1475.47 | 1468.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 1444.60 | 1475.47 | 1468.03 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 12:15:00 | 1451.70 | 1461.35 | 1462.57 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 1480.00 | 1464.66 | 1463.70 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 1456.60 | 1463.05 | 1463.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 11:15:00 | 1442.60 | 1457.24 | 1460.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 1472.00 | 1453.58 | 1456.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 1472.00 | 1453.58 | 1456.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 1472.00 | 1453.58 | 1456.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:30:00 | 1472.40 | 1453.58 | 1456.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 1433.50 | 1449.56 | 1454.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:30:00 | 1467.50 | 1449.56 | 1454.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 1455.90 | 1431.14 | 1435.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:00:00 | 1455.90 | 1431.14 | 1435.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 1462.40 | 1437.39 | 1437.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 1462.40 | 1437.39 | 1437.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 11:15:00 | 1451.20 | 1440.15 | 1438.87 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 15:15:00 | 1434.00 | 1438.02 | 1438.37 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 09:15:00 | 1468.30 | 1444.08 | 1441.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 09:15:00 | 1486.90 | 1463.09 | 1456.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 1463.00 | 1484.54 | 1474.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 1463.00 | 1484.54 | 1474.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 1463.00 | 1484.54 | 1474.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:00:00 | 1463.00 | 1484.54 | 1474.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 1452.50 | 1478.13 | 1472.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 1452.50 | 1478.13 | 1472.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1469.60 | 1472.32 | 1470.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 1474.00 | 1471.36 | 1470.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 1465.00 | 1470.09 | 1470.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 1465.00 | 1470.09 | 1470.20 | EMA200 below EMA400 |

### Cycle 117 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 1471.70 | 1470.26 | 1470.21 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 13:15:00 | 1467.10 | 1469.63 | 1469.92 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-11-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 14:15:00 | 1472.10 | 1470.12 | 1470.12 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 1466.10 | 1469.32 | 1469.76 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 1484.10 | 1472.27 | 1471.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 11:15:00 | 1500.00 | 1479.34 | 1474.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 13:15:00 | 1479.50 | 1481.98 | 1476.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 14:00:00 | 1479.50 | 1481.98 | 1476.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 1474.50 | 1480.48 | 1476.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 1474.50 | 1480.48 | 1476.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 1476.00 | 1479.58 | 1476.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 1474.90 | 1479.58 | 1476.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 1462.80 | 1476.23 | 1475.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:00:00 | 1462.80 | 1476.23 | 1475.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 1465.40 | 1474.06 | 1474.36 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 1474.40 | 1472.16 | 1471.94 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1460.30 | 1469.79 | 1470.88 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 1470.10 | 1467.31 | 1467.28 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 1464.60 | 1466.77 | 1467.04 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 11:15:00 | 1470.40 | 1467.50 | 1467.34 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 1465.60 | 1467.12 | 1467.18 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 1469.60 | 1467.61 | 1467.40 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 1464.20 | 1466.88 | 1467.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 1454.70 | 1464.44 | 1465.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 1460.00 | 1459.74 | 1462.76 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-21 14:30:00 | 1457.10 | 1458.04 | 1461.71 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 1451.30 | 1413.06 | 1423.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 1451.30 | 1413.06 | 1423.91 | SL hit (close>ema400) qty=1.00 sl=1423.91 alert=retest1 |

### Cycle 131 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 1460.90 | 1431.26 | 1430.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 12:15:00 | 1470.20 | 1457.41 | 1447.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 14:15:00 | 1493.50 | 1494.98 | 1483.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 14:45:00 | 1487.70 | 1494.98 | 1483.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 1471.10 | 1489.63 | 1483.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 10:00:00 | 1471.10 | 1489.63 | 1483.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 1472.60 | 1486.22 | 1482.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:15:00 | 1472.60 | 1486.22 | 1482.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 1464.40 | 1478.82 | 1479.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 1462.10 | 1475.47 | 1477.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 11:15:00 | 1420.00 | 1417.59 | 1429.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-05 11:30:00 | 1425.20 | 1417.59 | 1429.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 13:15:00 | 1441.40 | 1423.20 | 1429.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:00:00 | 1441.40 | 1423.20 | 1429.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 1433.30 | 1425.22 | 1430.00 | EMA400 retest candle locked (from downside) |

### Cycle 133 — BUY (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 10:15:00 | 1448.00 | 1434.37 | 1433.35 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2025-12-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 15:15:00 | 1425.80 | 1432.57 | 1433.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 1409.70 | 1427.99 | 1431.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 1423.20 | 1419.56 | 1425.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:30:00 | 1419.00 | 1419.56 | 1425.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 09:15:00 | 1421.60 | 1419.17 | 1423.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 1404.00 | 1416.44 | 1422.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 12:30:00 | 1408.90 | 1414.93 | 1420.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 14:15:00 | 1401.40 | 1413.78 | 1419.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 09:15:00 | 1434.70 | 1415.17 | 1418.35 | SL hit (close>static) qty=1.00 sl=1424.00 alert=retest2 |

### Cycle 135 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 1442.40 | 1420.62 | 1420.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 1456.50 | 1427.80 | 1423.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 1453.00 | 1456.57 | 1446.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:45:00 | 1454.00 | 1456.57 | 1446.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 1451.80 | 1454.69 | 1450.29 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2025-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 14:15:00 | 1436.20 | 1448.05 | 1448.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 1435.80 | 1445.60 | 1447.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 10:15:00 | 1448.10 | 1445.89 | 1447.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 10:15:00 | 1448.10 | 1445.89 | 1447.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 1448.10 | 1445.89 | 1447.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 1448.10 | 1445.89 | 1447.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 1450.00 | 1446.71 | 1447.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 1440.40 | 1446.71 | 1447.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 1462.60 | 1444.57 | 1445.45 | SL hit (close>static) qty=1.00 sl=1450.90 alert=retest2 |

### Cycle 137 — BUY (started 2025-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 10:15:00 | 1462.10 | 1448.07 | 1446.96 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 10:15:00 | 1426.30 | 1445.61 | 1447.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 11:15:00 | 1418.40 | 1440.17 | 1444.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 1434.60 | 1432.39 | 1439.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 15:00:00 | 1434.60 | 1432.39 | 1439.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 1419.80 | 1430.11 | 1437.30 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-12-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 12:15:00 | 1452.60 | 1436.38 | 1435.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 14:15:00 | 1456.10 | 1442.85 | 1438.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 10:15:00 | 1498.00 | 1502.18 | 1481.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 1498.00 | 1502.18 | 1481.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 1495.50 | 1497.57 | 1485.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 1490.00 | 1497.57 | 1485.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 1439.00 | 1484.17 | 1481.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 1439.00 | 1484.17 | 1481.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2025-12-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 11:15:00 | 1465.00 | 1478.07 | 1478.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 14:15:00 | 1455.50 | 1470.54 | 1474.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 15:15:00 | 1472.50 | 1470.93 | 1474.69 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 09:15:00 | 1450.00 | 1470.93 | 1474.69 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 10:30:00 | 1454.00 | 1463.58 | 1470.42 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 11:15:00 | 1455.00 | 1463.58 | 1470.42 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-30 14:15:00 | 1452.00 | 1457.40 | 1465.51 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 1477.00 | 1459.94 | 1464.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 09:15:00 | 1477.00 | 1459.94 | 1464.51 | SL hit (close>ema400) qty=1.00 sl=1464.51 alert=retest1 |

### Cycle 141 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 1486.50 | 1469.26 | 1468.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 1492.50 | 1473.91 | 1470.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 1456.30 | 1473.63 | 1471.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 09:15:00 | 1456.30 | 1473.63 | 1471.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 1456.30 | 1473.63 | 1471.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-01 10:00:00 | 1456.30 | 1473.63 | 1471.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 1456.00 | 1470.10 | 1470.41 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-01-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 09:15:00 | 1496.80 | 1472.36 | 1469.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 10:15:00 | 1498.70 | 1477.63 | 1471.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 15:15:00 | 1486.10 | 1487.71 | 1479.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 15:15:00 | 1486.10 | 1487.71 | 1479.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 1486.10 | 1487.71 | 1479.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 1500.60 | 1487.71 | 1479.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 11:15:00 | 1478.80 | 1485.87 | 1481.10 | SL hit (close<static) qty=1.00 sl=1479.70 alert=retest2 |

### Cycle 144 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 1477.00 | 1479.47 | 1479.72 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 15:15:00 | 1487.70 | 1480.18 | 1479.88 | EMA200 above EMA400 |

### Cycle 146 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 1467.10 | 1477.56 | 1478.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 1444.90 | 1471.03 | 1475.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 11:15:00 | 1459.90 | 1459.52 | 1465.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 11:30:00 | 1461.00 | 1459.52 | 1465.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 1421.00 | 1416.30 | 1425.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 1439.40 | 1416.30 | 1425.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 1443.90 | 1421.82 | 1427.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 1447.30 | 1421.82 | 1427.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 1448.10 | 1427.08 | 1428.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 1447.20 | 1427.08 | 1428.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 1447.80 | 1433.48 | 1431.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 1490.70 | 1446.09 | 1438.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 12:15:00 | 1466.50 | 1470.52 | 1460.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-19 13:00:00 | 1466.50 | 1470.52 | 1460.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 1465.00 | 1469.41 | 1460.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:00:00 | 1465.00 | 1469.41 | 1460.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 1460.20 | 1467.57 | 1460.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 14:30:00 | 1463.30 | 1467.57 | 1460.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 1461.90 | 1466.44 | 1460.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:15:00 | 1432.20 | 1466.44 | 1460.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 1442.50 | 1461.65 | 1459.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 1423.40 | 1461.65 | 1459.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 1433.70 | 1456.06 | 1456.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 1429.90 | 1450.83 | 1454.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 1398.60 | 1391.03 | 1412.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:30:00 | 1396.50 | 1391.03 | 1412.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 1296.30 | 1287.52 | 1319.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 1314.30 | 1287.52 | 1319.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 1331.80 | 1303.23 | 1315.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 1331.80 | 1303.23 | 1315.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 1319.00 | 1306.39 | 1315.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:45:00 | 1307.80 | 1309.51 | 1316.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:30:00 | 1299.50 | 1309.19 | 1315.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 12:15:00 | 1304.50 | 1310.15 | 1315.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-29 15:15:00 | 1333.00 | 1318.20 | 1317.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 1333.00 | 1318.20 | 1317.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 1351.30 | 1324.82 | 1320.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 1316.00 | 1334.17 | 1329.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 1316.00 | 1334.17 | 1329.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 1316.00 | 1334.17 | 1329.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:45:00 | 1312.40 | 1334.17 | 1329.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 10:15:00 | 1332.80 | 1333.89 | 1329.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:30:00 | 1340.40 | 1330.77 | 1328.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 1306.50 | 1325.92 | 1326.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 1306.50 | 1325.92 | 1326.58 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 1363.50 | 1304.03 | 1299.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 1369.00 | 1317.02 | 1306.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 11:15:00 | 1369.70 | 1373.70 | 1356.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 12:00:00 | 1369.70 | 1373.70 | 1356.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 1342.90 | 1369.04 | 1361.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 1342.90 | 1369.04 | 1361.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 1331.70 | 1361.57 | 1358.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:45:00 | 1329.00 | 1361.57 | 1358.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 1330.50 | 1351.15 | 1353.92 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 1390.60 | 1359.69 | 1356.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 1396.80 | 1367.11 | 1360.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 09:15:00 | 1372.10 | 1379.23 | 1371.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 09:45:00 | 1375.90 | 1379.23 | 1371.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 1389.70 | 1381.33 | 1372.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 11:30:00 | 1396.10 | 1385.02 | 1375.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 09:15:00 | 1367.70 | 1381.46 | 1377.65 | SL hit (close<static) qty=1.00 sl=1368.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 1361.30 | 1374.50 | 1374.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 1359.30 | 1371.46 | 1373.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 14:15:00 | 1287.10 | 1286.03 | 1302.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 15:00:00 | 1287.10 | 1286.03 | 1302.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 1280.30 | 1281.23 | 1292.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 13:30:00 | 1290.60 | 1281.23 | 1292.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 1296.90 | 1285.71 | 1292.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 1315.60 | 1285.71 | 1292.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 1302.90 | 1289.14 | 1293.71 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 12:15:00 | 1312.50 | 1298.96 | 1297.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 1313.00 | 1301.77 | 1298.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 1297.00 | 1305.60 | 1302.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 1297.00 | 1305.60 | 1302.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1297.00 | 1305.60 | 1302.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 11:45:00 | 1293.10 | 1305.60 | 1302.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1308.80 | 1306.24 | 1303.20 | EMA400 retest candle locked (from upside) |

### Cycle 156 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 1294.40 | 1300.93 | 1301.50 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 10:15:00 | 1308.10 | 1302.37 | 1302.10 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 14:15:00 | 1285.30 | 1300.25 | 1301.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 1281.80 | 1296.56 | 1299.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 09:15:00 | 1277.40 | 1273.95 | 1283.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 10:00:00 | 1277.40 | 1273.95 | 1283.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 11:15:00 | 1268.70 | 1273.77 | 1281.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 12:15:00 | 1266.20 | 1273.77 | 1281.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 10:30:00 | 1266.70 | 1272.26 | 1277.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 12:30:00 | 1266.60 | 1270.77 | 1275.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-25 13:45:00 | 1267.30 | 1270.42 | 1275.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 1274.90 | 1271.32 | 1275.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:30:00 | 1271.10 | 1271.32 | 1275.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 1270.60 | 1271.17 | 1274.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 1279.90 | 1271.17 | 1274.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 1281.50 | 1273.24 | 1275.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:30:00 | 1274.20 | 1274.69 | 1275.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:15:00 | 1272.20 | 1274.69 | 1275.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 15:15:00 | 1280.00 | 1276.16 | 1276.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 1280.00 | 1276.16 | 1276.00 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 1241.30 | 1269.19 | 1272.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 1226.00 | 1247.76 | 1258.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 1217.50 | 1203.41 | 1216.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 1217.50 | 1203.41 | 1216.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 1217.50 | 1203.41 | 1216.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:45:00 | 1222.40 | 1203.41 | 1216.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 1220.30 | 1206.79 | 1216.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 11:00:00 | 1220.30 | 1206.79 | 1216.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 1243.00 | 1225.12 | 1222.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 1249.60 | 1230.02 | 1225.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 12:15:00 | 1227.30 | 1231.73 | 1227.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 12:15:00 | 1227.30 | 1231.73 | 1227.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 1227.30 | 1231.73 | 1227.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:00:00 | 1227.30 | 1231.73 | 1227.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 13:15:00 | 1229.70 | 1231.32 | 1227.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 13:30:00 | 1229.00 | 1231.32 | 1227.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 1224.60 | 1229.98 | 1227.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 1224.60 | 1229.98 | 1227.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 1230.50 | 1230.08 | 1227.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 1180.00 | 1230.08 | 1227.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 162 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 1187.10 | 1221.49 | 1223.98 | EMA200 below EMA400 |

### Cycle 163 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 1227.10 | 1217.60 | 1216.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 13:15:00 | 1236.60 | 1221.40 | 1218.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 1222.10 | 1229.09 | 1224.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 12:15:00 | 1222.10 | 1229.09 | 1224.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 1222.10 | 1229.09 | 1224.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 1222.10 | 1229.09 | 1224.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 1220.00 | 1227.27 | 1224.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 1220.00 | 1227.27 | 1224.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 1211.90 | 1224.20 | 1223.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 1211.90 | 1224.20 | 1223.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 1208.60 | 1221.08 | 1221.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 1177.50 | 1212.36 | 1217.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 1160.00 | 1154.98 | 1169.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 1164.10 | 1154.98 | 1169.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 1159.90 | 1156.12 | 1167.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:15:00 | 1160.00 | 1156.12 | 1167.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 1156.50 | 1156.20 | 1166.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 1150.30 | 1156.40 | 1165.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 1173.00 | 1160.43 | 1164.17 | SL hit (close>static) qty=1.00 sl=1170.50 alert=retest2 |

### Cycle 165 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 1186.20 | 1168.08 | 1167.15 | EMA200 above EMA400 |

### Cycle 166 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1154.90 | 1169.05 | 1169.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 1148.10 | 1159.94 | 1164.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 1130.00 | 1118.89 | 1130.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 1130.00 | 1118.89 | 1130.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 1130.00 | 1118.89 | 1130.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 1130.00 | 1118.89 | 1130.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 1151.00 | 1125.31 | 1132.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:45:00 | 1151.10 | 1125.31 | 1132.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 1152.00 | 1130.65 | 1133.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:00:00 | 1152.00 | 1130.65 | 1133.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 167 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 1148.50 | 1137.35 | 1136.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 1161.30 | 1142.14 | 1138.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 14:15:00 | 1195.50 | 1196.18 | 1178.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 14:45:00 | 1192.00 | 1196.18 | 1178.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 1162.60 | 1190.88 | 1179.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 1162.60 | 1190.88 | 1179.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 1158.60 | 1184.42 | 1177.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 11:00:00 | 1158.60 | 1184.42 | 1177.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 1149.00 | 1173.11 | 1173.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 15:15:00 | 1140.00 | 1162.66 | 1168.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 11:15:00 | 1162.40 | 1159.66 | 1165.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 12:00:00 | 1162.40 | 1159.66 | 1165.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 1148.10 | 1157.74 | 1163.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 14:15:00 | 1146.40 | 1157.74 | 1163.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:15:00 | 1146.20 | 1139.82 | 1148.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 09:15:00 | 1167.00 | 1145.73 | 1149.44 | SL hit (close>static) qty=1.00 sl=1164.20 alert=retest2 |

### Cycle 169 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1166.30 | 1151.90 | 1151.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1176.80 | 1156.88 | 1154.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 13:15:00 | 1164.60 | 1168.70 | 1163.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 14:00:00 | 1164.60 | 1168.70 | 1163.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 1176.80 | 1170.32 | 1164.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 15:15:00 | 1179.00 | 1170.32 | 1164.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 09:15:00 | 1296.90 | 1263.10 | 1226.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 1366.10 | 1376.40 | 1377.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 1357.10 | 1370.76 | 1373.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 10:15:00 | 1365.30 | 1358.17 | 1364.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 10:15:00 | 1365.30 | 1358.17 | 1364.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 1365.30 | 1358.17 | 1364.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 1365.30 | 1358.17 | 1364.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 1368.80 | 1360.30 | 1364.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:00:00 | 1368.80 | 1360.30 | 1364.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 1357.00 | 1359.64 | 1363.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:15:00 | 1355.50 | 1359.64 | 1363.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 13:15:00 | 1377.10 | 1365.33 | 1364.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — BUY (started 2026-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 13:15:00 | 1377.10 | 1365.33 | 1364.58 | EMA200 above EMA400 |

### Cycle 172 — SELL (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 12:15:00 | 1355.50 | 1364.54 | 1365.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 13:15:00 | 1349.10 | 1361.46 | 1363.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 1334.40 | 1330.42 | 1342.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 1334.50 | 1330.42 | 1342.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 1333.20 | 1330.98 | 1341.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 1323.80 | 1330.76 | 1340.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:15:00 | 1325.40 | 1330.76 | 1340.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 1323.80 | 1329.37 | 1338.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 10:45:00 | 1326.90 | 1333.11 | 1336.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 11:15:00 | 1345.70 | 1335.63 | 1337.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 11:30:00 | 1353.20 | 1335.63 | 1337.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-05 12:15:00 | 1351.10 | 1338.73 | 1338.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 1351.10 | 1338.73 | 1338.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 1381.90 | 1349.67 | 1343.71 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 10:15:00 | 976.21 | 2024-05-13 11:15:00 | 1044.65 | STOP_HIT | 1.00 | -7.01% |
| SELL | retest2 | 2024-05-15 14:45:00 | 979.23 | 2024-05-21 13:15:00 | 930.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-17 09:30:00 | 971.00 | 2024-05-21 13:15:00 | 922.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-21 09:15:00 | 982.73 | 2024-05-21 13:15:00 | 933.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-15 14:45:00 | 979.23 | 2024-05-22 09:15:00 | 950.07 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2024-05-17 09:30:00 | 971.00 | 2024-05-22 09:15:00 | 950.07 | STOP_HIT | 0.50 | 2.16% |
| SELL | retest2 | 2024-05-21 09:15:00 | 982.73 | 2024-05-22 09:15:00 | 950.07 | STOP_HIT | 0.50 | 3.32% |
| SELL | retest2 | 2024-06-03 10:30:00 | 967.63 | 2024-06-03 13:15:00 | 1008.59 | STOP_HIT | 1.00 | -4.23% |
| BUY | retest2 | 2024-06-11 12:15:00 | 1020.20 | 2024-06-13 12:15:00 | 1004.02 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2024-06-12 09:15:00 | 1022.38 | 2024-06-13 12:15:00 | 1004.02 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-06-12 14:30:00 | 1022.93 | 2024-06-13 12:15:00 | 1004.02 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-06-20 15:15:00 | 1057.60 | 2024-06-21 09:15:00 | 1009.20 | STOP_HIT | 1.00 | -4.58% |
| SELL | retest2 | 2024-07-15 09:15:00 | 947.99 | 2024-07-16 09:15:00 | 966.39 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2024-08-26 10:45:00 | 1244.97 | 2024-08-26 13:15:00 | 1265.60 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-08-26 11:15:00 | 1247.03 | 2024-08-26 13:15:00 | 1265.60 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-08-29 09:15:00 | 1295.98 | 2024-08-29 15:15:00 | 1284.90 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-08-29 11:00:00 | 1300.86 | 2024-08-29 15:15:00 | 1284.90 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2024-08-29 12:15:00 | 1298.60 | 2024-08-29 15:15:00 | 1284.90 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2024-09-05 09:30:00 | 1345.09 | 2024-09-06 09:15:00 | 1312.88 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-09-19 10:15:00 | 1309.61 | 2024-09-20 12:15:00 | 1363.42 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2024-09-19 12:45:00 | 1309.69 | 2024-09-20 12:15:00 | 1363.42 | STOP_HIT | 1.00 | -4.10% |
| SELL | retest2 | 2024-09-19 15:15:00 | 1304.00 | 2024-09-20 12:15:00 | 1363.42 | STOP_HIT | 1.00 | -4.56% |
| BUY | retest1 | 2024-09-24 09:15:00 | 1389.67 | 2024-09-24 09:15:00 | 1339.24 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2024-10-18 12:45:00 | 1432.56 | 2024-10-21 15:15:00 | 1389.20 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2024-10-18 14:15:00 | 1430.00 | 2024-10-21 15:15:00 | 1389.20 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-10-21 09:15:00 | 1432.08 | 2024-10-21 15:15:00 | 1389.20 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2024-10-25 10:15:00 | 1331.88 | 2024-10-30 12:15:00 | 1351.12 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-10-28 09:15:00 | 1321.01 | 2024-10-30 12:15:00 | 1351.12 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-10-28 12:45:00 | 1332.66 | 2024-10-30 12:15:00 | 1351.12 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-10-30 10:15:00 | 1335.21 | 2024-10-30 12:15:00 | 1351.12 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest1 | 2024-11-05 09:15:00 | 1412.43 | 2024-11-06 13:15:00 | 1483.05 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-11-05 09:15:00 | 1412.43 | 2024-11-07 15:15:00 | 1467.20 | STOP_HIT | 0.50 | 3.88% |
| BUY | retest2 | 2024-11-05 13:15:00 | 1408.64 | 2024-11-08 13:15:00 | 1421.00 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2024-11-18 15:15:00 | 1319.00 | 2024-11-19 10:15:00 | 1337.06 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2024-11-19 15:00:00 | 1320.60 | 2024-11-25 14:15:00 | 1320.15 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2024-11-25 09:45:00 | 1313.15 | 2024-11-25 14:15:00 | 1320.15 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest1 | 2024-12-04 09:15:00 | 1363.72 | 2024-12-10 09:15:00 | 1431.91 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-12-04 09:15:00 | 1363.72 | 2024-12-11 09:15:00 | 1500.09 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-12-05 10:30:00 | 1373.35 | 2024-12-11 09:15:00 | 1510.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-05 13:45:00 | 1369.73 | 2024-12-11 09:15:00 | 1506.70 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-05 14:45:00 | 1369.00 | 2024-12-11 09:15:00 | 1505.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-05 15:15:00 | 1370.00 | 2024-12-11 09:15:00 | 1507.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-10 09:15:00 | 1400.77 | 2024-12-12 09:15:00 | 1377.45 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-12-24 11:30:00 | 1368.46 | 2024-12-27 09:15:00 | 1374.40 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-12-24 12:00:00 | 1368.73 | 2024-12-27 09:15:00 | 1374.40 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-12-24 13:00:00 | 1367.14 | 2024-12-27 09:15:00 | 1374.40 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2024-12-24 15:15:00 | 1365.00 | 2024-12-27 09:15:00 | 1374.40 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-12-26 10:15:00 | 1362.04 | 2024-12-27 09:15:00 | 1374.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-12-31 12:30:00 | 1379.94 | 2025-01-06 10:15:00 | 1370.65 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-01-08 11:00:00 | 1348.99 | 2025-01-13 12:15:00 | 1281.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:00:00 | 1348.99 | 2025-01-13 14:15:00 | 1214.09 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 13:45:00 | 1118.85 | 2025-01-27 09:15:00 | 1062.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:45:00 | 1118.85 | 2025-01-27 15:15:00 | 1006.96 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-01 14:15:00 | 1092.76 | 2025-02-03 09:15:00 | 1021.19 | STOP_HIT | 1.00 | -6.55% |
| BUY | retest2 | 2025-02-01 15:00:00 | 1091.57 | 2025-02-03 09:15:00 | 1021.19 | STOP_HIT | 1.00 | -6.45% |
| BUY | retest2 | 2025-02-07 11:15:00 | 1158.93 | 2025-02-07 15:15:00 | 1130.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2025-02-13 13:30:00 | 1088.57 | 2025-02-17 09:15:00 | 1034.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 1086.36 | 2025-02-17 09:15:00 | 1032.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:00:00 | 1080.55 | 2025-02-17 09:15:00 | 1026.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-13 13:30:00 | 1088.57 | 2025-02-17 11:15:00 | 1062.72 | STOP_HIT | 0.50 | 2.37% |
| SELL | retest2 | 2025-02-14 09:15:00 | 1086.36 | 2025-02-17 11:15:00 | 1062.72 | STOP_HIT | 0.50 | 2.18% |
| SELL | retest2 | 2025-02-14 10:00:00 | 1080.55 | 2025-02-17 11:15:00 | 1062.72 | STOP_HIT | 0.50 | 1.65% |
| BUY | retest2 | 2025-03-10 09:15:00 | 1109.00 | 2025-03-11 09:15:00 | 1039.80 | STOP_HIT | 1.00 | -6.24% |
| SELL | retest2 | 2025-03-13 09:15:00 | 1024.43 | 2025-03-17 14:15:00 | 1067.23 | STOP_HIT | 1.00 | -4.18% |
| SELL | retest2 | 2025-03-28 13:30:00 | 1215.07 | 2025-04-02 14:15:00 | 1154.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 15:15:00 | 1212.20 | 2025-04-02 14:15:00 | 1151.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-01 10:00:00 | 1211.32 | 2025-04-02 14:15:00 | 1150.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-03-28 13:30:00 | 1215.07 | 2025-04-03 12:15:00 | 1181.86 | STOP_HIT | 0.50 | 2.73% |
| SELL | retest2 | 2025-03-28 15:15:00 | 1212.20 | 2025-04-03 12:15:00 | 1181.86 | STOP_HIT | 0.50 | 2.50% |
| SELL | retest2 | 2025-04-01 10:00:00 | 1211.32 | 2025-04-03 12:15:00 | 1181.86 | STOP_HIT | 0.50 | 2.43% |
| BUY | retest2 | 2025-04-25 10:45:00 | 1250.50 | 2025-04-30 09:15:00 | 1231.40 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-04-25 12:30:00 | 1253.50 | 2025-04-30 09:15:00 | 1231.40 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-04-28 13:30:00 | 1249.20 | 2025-04-30 09:15:00 | 1231.40 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-04-28 14:45:00 | 1254.60 | 2025-04-30 09:15:00 | 1231.40 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-04-29 11:15:00 | 1254.30 | 2025-04-30 09:15:00 | 1231.40 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2025-05-19 10:15:00 | 1348.90 | 2025-05-22 15:15:00 | 1360.00 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-05-19 10:45:00 | 1354.40 | 2025-05-22 15:15:00 | 1360.00 | STOP_HIT | 1.00 | 0.41% |
| BUY | retest2 | 2025-05-22 13:15:00 | 1347.50 | 2025-05-22 15:15:00 | 1360.00 | STOP_HIT | 1.00 | 0.93% |
| BUY | retest2 | 2025-05-27 10:00:00 | 1390.80 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 3.70% |
| BUY | retest2 | 2025-05-27 12:15:00 | 1389.70 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 3.78% |
| BUY | retest2 | 2025-05-28 09:15:00 | 1392.40 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 3.58% |
| BUY | retest2 | 2025-05-28 11:30:00 | 1389.50 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 3.79% |
| BUY | retest2 | 2025-05-29 09:15:00 | 1417.00 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 1.78% |
| BUY | retest2 | 2025-05-29 13:30:00 | 1393.20 | 2025-06-05 13:15:00 | 1442.20 | STOP_HIT | 1.00 | 3.52% |
| SELL | retest2 | 2025-06-17 12:15:00 | 1441.00 | 2025-06-17 12:15:00 | 1460.30 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-07-08 11:00:00 | 1489.50 | 2025-07-15 10:15:00 | 1491.30 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-07-08 13:15:00 | 1496.90 | 2025-07-15 10:15:00 | 1491.30 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2025-07-08 14:15:00 | 1497.40 | 2025-07-15 10:15:00 | 1491.30 | STOP_HIT | 1.00 | 0.41% |
| SELL | retest2 | 2025-07-08 15:00:00 | 1491.80 | 2025-07-15 10:15:00 | 1491.30 | STOP_HIT | 1.00 | 0.03% |
| SELL | retest2 | 2025-07-11 13:30:00 | 1464.40 | 2025-07-15 10:15:00 | 1491.30 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2025-07-17 09:15:00 | 1546.60 | 2025-07-21 12:15:00 | 1516.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-08-01 09:15:00 | 1408.00 | 2025-08-12 12:15:00 | 1391.40 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2025-09-03 12:45:00 | 1299.00 | 2025-09-03 15:15:00 | 1307.10 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-10 14:45:00 | 1275.00 | 2025-09-12 12:15:00 | 1301.30 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-09-11 09:15:00 | 1268.40 | 2025-09-12 12:15:00 | 1301.30 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2025-09-26 11:30:00 | 1224.90 | 2025-09-29 11:15:00 | 1242.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-26 13:15:00 | 1227.90 | 2025-09-29 11:15:00 | 1242.00 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2025-11-12 09:15:00 | 1474.00 | 2025-11-12 09:15:00 | 1465.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest1 | 2025-11-21 14:30:00 | 1457.10 | 2025-11-26 09:15:00 | 1451.30 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-12-10 10:45:00 | 1404.00 | 2025-12-11 09:15:00 | 1434.70 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2025-12-10 12:30:00 | 1408.90 | 2025-12-11 09:15:00 | 1434.70 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-10 14:15:00 | 1401.40 | 2025-12-11 09:15:00 | 1434.70 | STOP_HIT | 1.00 | -2.38% |
| SELL | retest2 | 2025-12-17 12:15:00 | 1440.40 | 2025-12-18 09:15:00 | 1462.60 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest1 | 2025-12-30 09:15:00 | 1450.00 | 2025-12-31 09:15:00 | 1477.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest1 | 2025-12-30 10:30:00 | 1454.00 | 2025-12-31 09:15:00 | 1477.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest1 | 2025-12-30 11:15:00 | 1455.00 | 2025-12-31 09:15:00 | 1477.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest1 | 2025-12-30 14:15:00 | 1452.00 | 2025-12-31 09:15:00 | 1477.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-01-06 09:15:00 | 1500.60 | 2026-01-06 11:15:00 | 1478.80 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-29 09:45:00 | 1307.80 | 2026-01-29 15:15:00 | 1333.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-29 10:30:00 | 1299.50 | 2026-01-29 15:15:00 | 1333.00 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2026-01-29 12:15:00 | 1304.50 | 2026-01-29 15:15:00 | 1333.00 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2026-02-01 11:30:00 | 1340.40 | 2026-02-01 12:15:00 | 1306.50 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2026-02-10 11:30:00 | 1396.10 | 2026-02-11 09:15:00 | 1367.70 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2026-02-24 12:15:00 | 1266.20 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-02-25 10:30:00 | 1266.70 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-02-25 12:30:00 | 1266.60 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-02-25 13:45:00 | 1267.30 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-02-26 11:30:00 | 1274.20 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2026-02-26 12:15:00 | 1272.20 | 2026-02-26 15:15:00 | 1280.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-03-17 12:15:00 | 1150.30 | 2026-03-18 09:15:00 | 1173.00 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-04-01 14:15:00 | 1146.40 | 2026-04-06 09:15:00 | 1167.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-04-02 14:15:00 | 1146.20 | 2026-04-06 09:15:00 | 1167.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2026-04-07 15:15:00 | 1179.00 | 2026-04-09 09:15:00 | 1296.90 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-27 13:15:00 | 1355.50 | 2026-04-28 13:15:00 | 1377.10 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-05-04 10:45:00 | 1323.80 | 2026-05-05 12:15:00 | 1351.10 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-05-04 11:15:00 | 1325.40 | 2026-05-05 12:15:00 | 1351.10 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2026-05-04 12:00:00 | 1323.80 | 2026-05-05 12:15:00 | 1351.10 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2026-05-05 10:45:00 | 1326.90 | 2026-05-05 12:15:00 | 1351.10 | STOP_HIT | 1.00 | -1.82% |
