# Vijaya Diagnostic Centre Ltd. (VIJAYA)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 1275.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 85 |
| ALERT1 | 54 |
| ALERT2 | 54 |
| ALERT2_SKIP | 32 |
| ALERT3 | 128 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 66 |
| PARTIAL | 6 |
| TARGET_HIT | 2 |
| STOP_HIT | 66 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 74 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 55
- **Target hits / Stop hits / Partials:** 2 / 66 / 6
- **Avg / median % per leg:** -0.01% / -0.86%
- **Sum % (uncompounded):** -0.90%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 5 | 16.1% | 2 | 29 | 0 | -0.05% | -1.7% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.84% | -1.7% |
| BUY @ 3rd Alert (retest2) | 29 | 5 | 17.2% | 2 | 27 | 0 | -0.00% | -0.0% |
| SELL (all) | 43 | 14 | 32.6% | 0 | 37 | 6 | 0.02% | 0.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 43 | 14 | 32.6% | 0 | 37 | 6 | 0.02% | 0.8% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -0.84% | -1.7% |
| retest2 (combined) | 72 | 19 | 26.4% | 2 | 64 | 6 | 0.01% | 0.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 1013.70 | 1002.77 | 1001.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 1017.10 | 1007.45 | 1004.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-12 14:15:00 | 987.10 | 1003.38 | 1002.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 14:15:00 | 987.10 | 1003.38 | 1002.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 14:15:00 | 987.10 | 1003.38 | 1002.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-12 15:00:00 | 987.10 | 1003.38 | 1002.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-12 15:15:00 | 990.00 | 1000.70 | 1001.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-13 12:15:00 | 978.40 | 991.30 | 996.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 11:15:00 | 905.50 | 903.11 | 923.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-16 12:00:00 | 905.50 | 903.11 | 923.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 924.30 | 908.71 | 918.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:15:00 | 935.00 | 908.71 | 918.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 938.60 | 914.69 | 920.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:00:00 | 938.60 | 914.69 | 920.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 922.20 | 921.83 | 922.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 14:30:00 | 928.20 | 921.83 | 922.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 921.00 | 921.66 | 922.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 922.40 | 921.66 | 922.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 925.20 | 922.37 | 922.65 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 931.00 | 923.52 | 922.90 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 10:15:00 | 919.00 | 927.50 | 927.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 11:15:00 | 916.20 | 925.24 | 926.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 938.90 | 922.40 | 924.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 938.90 | 922.40 | 924.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 938.90 | 922.40 | 924.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 939.00 | 922.40 | 924.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 933.40 | 924.60 | 925.00 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-05-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 11:15:00 | 936.40 | 926.96 | 926.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 10:15:00 | 950.50 | 935.23 | 930.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 11:15:00 | 948.10 | 948.19 | 941.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 11:30:00 | 947.00 | 948.19 | 941.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 941.90 | 946.38 | 943.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 940.10 | 946.38 | 943.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 951.00 | 947.31 | 944.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 949.20 | 947.31 | 944.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 945.90 | 949.26 | 946.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-30 10:15:00 | 940.70 | 949.26 | 946.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 946.70 | 948.75 | 946.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 12:30:00 | 953.00 | 948.36 | 946.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-30 15:00:00 | 958.80 | 950.31 | 948.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 12:15:00 | 945.00 | 947.16 | 947.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 12:15:00 | 945.00 | 947.16 | 947.21 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 13:15:00 | 956.00 | 948.93 | 948.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 14:15:00 | 961.00 | 951.34 | 949.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 956.70 | 959.14 | 955.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 956.70 | 959.14 | 955.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 956.70 | 959.14 | 955.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 955.20 | 959.14 | 955.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 969.10 | 973.12 | 968.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:45:00 | 964.55 | 973.12 | 968.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 965.00 | 971.50 | 968.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 966.20 | 971.50 | 968.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 963.20 | 969.84 | 967.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:30:00 | 964.35 | 969.84 | 967.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 971.85 | 972.64 | 970.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:45:00 | 973.05 | 972.64 | 970.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 974.95 | 973.13 | 970.80 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 15:15:00 | 965.00 | 968.65 | 969.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 949.40 | 961.34 | 965.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 15:15:00 | 959.00 | 954.43 | 957.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 15:15:00 | 959.00 | 954.43 | 957.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 15:15:00 | 959.00 | 954.43 | 957.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:15:00 | 951.30 | 954.43 | 957.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 956.35 | 954.81 | 957.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 14:45:00 | 948.60 | 952.57 | 954.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 947.40 | 952.25 | 954.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 10:30:00 | 948.05 | 950.66 | 953.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 13:15:00 | 961.00 | 954.29 | 954.56 | SL hit (close>static) qty=1.00 sl=960.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 966.15 | 956.66 | 955.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 15:15:00 | 970.45 | 959.42 | 956.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 953.55 | 958.24 | 956.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 953.55 | 958.24 | 956.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 953.55 | 958.24 | 956.65 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 946.05 | 954.53 | 955.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 12:15:00 | 944.10 | 952.45 | 954.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 950.40 | 950.22 | 952.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 12:15:00 | 958.45 | 951.68 | 952.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 958.45 | 951.68 | 952.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:00:00 | 958.45 | 951.68 | 952.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 957.60 | 952.86 | 952.97 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 956.35 | 953.56 | 953.28 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 948.00 | 952.68 | 952.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 943.95 | 950.93 | 952.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 11:15:00 | 937.05 | 935.82 | 940.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 12:00:00 | 937.05 | 935.82 | 940.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 931.15 | 933.11 | 937.18 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 945.70 | 937.85 | 937.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 950.50 | 940.38 | 938.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-30 11:15:00 | 988.15 | 989.14 | 980.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 988.15 | 989.14 | 980.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 976.95 | 985.27 | 982.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:45:00 | 976.00 | 985.27 | 982.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 982.60 | 984.73 | 982.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-02 09:15:00 | 985.70 | 982.36 | 981.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 10:15:00 | 974.75 | 980.68 | 981.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 974.75 | 980.68 | 981.07 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 1001.30 | 984.89 | 982.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 11:15:00 | 1021.45 | 1004.32 | 996.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 1009.45 | 1013.07 | 1005.17 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 12:45:00 | 1014.05 | 1013.27 | 1006.63 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-07 14:00:00 | 1015.55 | 1013.73 | 1007.44 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 1012.00 | 1013.14 | 1008.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 1010.75 | 1013.14 | 1008.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 1006.30 | 1011.78 | 1008.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-08 09:15:00 | 1006.30 | 1011.78 | 1008.08 | SL hit (close<ema400) qty=1.00 sl=1008.08 alert=retest1 |

### Cycle 16 — SELL (started 2025-07-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 13:15:00 | 1000.55 | 1005.37 | 1005.81 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 1036.55 | 1011.12 | 1008.26 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 1009.95 | 1014.43 | 1014.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 1002.70 | 1012.08 | 1013.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 1016.70 | 1012.51 | 1013.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-11 11:15:00 | 1016.70 | 1012.51 | 1013.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 1016.70 | 1012.51 | 1013.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 12:00:00 | 1016.70 | 1012.51 | 1013.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 1018.05 | 1013.62 | 1013.92 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 13:15:00 | 1022.40 | 1015.38 | 1014.69 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 10:15:00 | 1003.50 | 1012.39 | 1013.52 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 13:15:00 | 1022.00 | 1014.58 | 1014.27 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-15 11:15:00 | 1010.10 | 1014.10 | 1014.42 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 1018.00 | 1014.79 | 1014.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 1025.90 | 1017.52 | 1015.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 13:15:00 | 1034.30 | 1042.72 | 1037.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 13:15:00 | 1034.30 | 1042.72 | 1037.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 1034.30 | 1042.72 | 1037.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 1034.30 | 1042.72 | 1037.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 1031.00 | 1040.38 | 1036.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:00:00 | 1031.00 | 1040.38 | 1036.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 1016.30 | 1033.28 | 1033.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 10:15:00 | 1010.00 | 1028.62 | 1031.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 1018.10 | 1018.08 | 1024.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 1018.10 | 1018.08 | 1024.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 1018.10 | 1018.08 | 1024.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 1022.15 | 1018.08 | 1024.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 1016.65 | 1014.88 | 1019.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 1009.50 | 1014.88 | 1019.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 1016.50 | 1015.20 | 1019.18 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 09:15:00 | 1060.50 | 1024.77 | 1021.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 10:15:00 | 1082.10 | 1036.24 | 1027.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 09:15:00 | 1076.00 | 1080.98 | 1057.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-25 09:45:00 | 1076.25 | 1080.98 | 1057.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 1062.25 | 1073.26 | 1061.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 1062.25 | 1073.26 | 1061.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 1070.00 | 1072.61 | 1061.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:30:00 | 1063.00 | 1072.61 | 1061.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 1084.10 | 1075.29 | 1065.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 14:00:00 | 1116.75 | 1077.31 | 1068.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 1103.65 | 1098.77 | 1097.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 1076.55 | 1093.09 | 1095.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 12:15:00 | 1076.55 | 1093.09 | 1095.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 1070.85 | 1083.64 | 1089.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-01 09:15:00 | 1086.60 | 1084.23 | 1089.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 1086.60 | 1084.23 | 1089.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 1086.60 | 1084.23 | 1089.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-01 09:30:00 | 1081.50 | 1084.23 | 1089.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 1086.60 | 1084.71 | 1089.27 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-08-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 12:15:00 | 1095.70 | 1089.76 | 1089.26 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 1083.50 | 1088.72 | 1089.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 14:15:00 | 1072.10 | 1081.67 | 1085.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 1057.90 | 1056.05 | 1063.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 1057.90 | 1056.05 | 1063.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 1060.00 | 1056.84 | 1063.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 1053.60 | 1056.84 | 1063.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 1053.80 | 1056.23 | 1062.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:45:00 | 1042.00 | 1052.97 | 1058.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 15:15:00 | 1047.20 | 1049.42 | 1054.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 15:15:00 | 1058.00 | 1055.43 | 1055.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-08-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 15:15:00 | 1058.00 | 1055.43 | 1055.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 1058.10 | 1055.96 | 1055.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 13:15:00 | 1055.80 | 1056.79 | 1056.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 13:15:00 | 1055.80 | 1056.79 | 1056.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 1055.80 | 1056.79 | 1056.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 1055.80 | 1056.79 | 1056.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 1057.60 | 1056.95 | 1056.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 1068.10 | 1056.95 | 1056.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 1055.10 | 1058.37 | 1057.08 | SL hit (close<static) qty=1.00 sl=1055.80 alert=retest2 |

### Cycle 30 — SELL (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 09:15:00 | 1050.50 | 1055.98 | 1056.39 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 09:15:00 | 1061.00 | 1054.93 | 1054.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 10:15:00 | 1065.30 | 1057.00 | 1055.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 11:15:00 | 1057.00 | 1057.00 | 1055.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 11:15:00 | 1057.00 | 1057.00 | 1055.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 11:15:00 | 1057.00 | 1057.00 | 1055.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 12:00:00 | 1057.00 | 1057.00 | 1055.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 12:15:00 | 1042.40 | 1054.08 | 1054.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 09:15:00 | 1035.90 | 1044.93 | 1049.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 12:15:00 | 1027.00 | 1023.76 | 1032.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 12:45:00 | 1025.70 | 1023.76 | 1032.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 1024.30 | 1022.52 | 1030.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 14:45:00 | 1025.30 | 1022.52 | 1030.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 1034.20 | 1024.50 | 1029.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:00:00 | 1034.20 | 1024.50 | 1029.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 1020.70 | 1023.74 | 1029.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 11:30:00 | 1019.10 | 1022.79 | 1028.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 1013.00 | 1022.18 | 1026.14 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 1027.90 | 1007.75 | 1006.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 1027.90 | 1007.75 | 1006.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 10:15:00 | 1033.50 | 1012.90 | 1009.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 1021.10 | 1027.94 | 1019.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 1021.10 | 1027.94 | 1019.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 1021.10 | 1027.94 | 1019.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 1049.40 | 1026.14 | 1022.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:45:00 | 1046.70 | 1033.77 | 1025.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 1083.20 | 1092.71 | 1093.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-09-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 14:15:00 | 1083.20 | 1092.71 | 1093.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 09:15:00 | 1062.70 | 1084.84 | 1089.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 14:15:00 | 1053.00 | 1052.42 | 1062.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 15:00:00 | 1053.00 | 1052.42 | 1062.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 1050.10 | 1052.05 | 1060.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 11:00:00 | 1039.30 | 1046.53 | 1048.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:00:00 | 1039.80 | 1045.08 | 1047.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 13:45:00 | 1039.90 | 1043.77 | 1046.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 14:15:00 | 1062.20 | 1047.76 | 1047.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 14:15:00 | 1062.20 | 1047.76 | 1047.04 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 11:15:00 | 1045.00 | 1046.83 | 1046.86 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 14:15:00 | 1051.60 | 1047.31 | 1047.03 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-09-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 09:15:00 | 1042.40 | 1046.05 | 1046.49 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 15:15:00 | 1051.20 | 1045.73 | 1045.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 09:15:00 | 1062.30 | 1049.04 | 1047.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 11:15:00 | 1048.10 | 1049.81 | 1047.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 11:15:00 | 1048.10 | 1049.81 | 1047.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 1048.10 | 1049.81 | 1047.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:00:00 | 1048.10 | 1049.81 | 1047.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 1049.40 | 1049.72 | 1048.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:30:00 | 1048.50 | 1049.72 | 1048.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 1051.40 | 1050.52 | 1048.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:00:00 | 1047.70 | 1049.96 | 1048.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 10:15:00 | 1047.20 | 1049.41 | 1048.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 10:45:00 | 1046.00 | 1049.41 | 1048.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 1039.50 | 1047.43 | 1047.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 09:15:00 | 1020.40 | 1038.49 | 1043.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 15:15:00 | 1025.50 | 1018.54 | 1024.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 15:15:00 | 1025.50 | 1018.54 | 1024.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 1025.50 | 1018.54 | 1024.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:45:00 | 1009.90 | 1014.44 | 1022.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 10:30:00 | 1009.05 | 1009.04 | 1009.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 11:30:00 | 1011.40 | 1009.29 | 1009.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 13:00:00 | 1011.30 | 1009.69 | 1010.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 13:15:00 | 1013.50 | 1010.45 | 1010.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 1013.50 | 1010.45 | 1010.37 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 1007.60 | 1010.15 | 1010.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 11:15:00 | 1002.95 | 1008.47 | 1009.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 09:15:00 | 1005.80 | 1003.95 | 1006.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 1005.80 | 1003.95 | 1006.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 1005.80 | 1003.95 | 1006.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 11:45:00 | 997.00 | 1001.89 | 1005.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 995.75 | 1000.94 | 1003.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 11:15:00 | 990.00 | 980.02 | 979.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 11:15:00 | 990.00 | 980.02 | 979.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 13:15:00 | 995.25 | 988.02 | 984.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 990.30 | 992.15 | 987.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:00:00 | 990.30 | 992.15 | 987.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 988.00 | 991.32 | 987.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 988.00 | 991.32 | 987.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 987.90 | 990.64 | 987.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:45:00 | 986.50 | 990.64 | 987.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 980.90 | 988.69 | 987.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 980.90 | 988.69 | 987.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 986.50 | 988.25 | 987.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 14:30:00 | 988.70 | 988.65 | 987.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 989.00 | 988.65 | 987.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 989.00 | 992.41 | 992.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 987.85 | 991.50 | 991.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-10-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 09:15:00 | 987.85 | 991.50 | 991.71 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 998.65 | 991.75 | 991.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 11:15:00 | 1013.55 | 1002.51 | 999.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 1001.25 | 1005.84 | 1002.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 1001.25 | 1005.84 | 1002.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 1001.25 | 1005.84 | 1002.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 1000.05 | 1005.84 | 1002.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 1005.15 | 1005.70 | 1002.76 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 14:15:00 | 994.35 | 1000.89 | 1001.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 09:15:00 | 983.30 | 996.73 | 999.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 990.85 | 990.72 | 995.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 14:00:00 | 990.85 | 990.72 | 995.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 1005.00 | 993.08 | 995.01 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 1006.00 | 997.11 | 996.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-03 12:15:00 | 1012.85 | 1000.26 | 998.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 15:15:00 | 1017.00 | 1023.89 | 1015.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-04 15:15:00 | 1017.00 | 1023.89 | 1015.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 1017.00 | 1023.89 | 1015.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 1001.05 | 1023.89 | 1015.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 1013.00 | 1021.71 | 1015.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:30:00 | 1024.55 | 1020.43 | 1017.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 1006.30 | 1016.17 | 1017.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 10:15:00 | 1006.30 | 1016.17 | 1017.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 11:15:00 | 1000.70 | 1013.08 | 1015.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 14:15:00 | 1021.50 | 1010.17 | 1013.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 14:15:00 | 1021.50 | 1010.17 | 1013.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 14:15:00 | 1021.50 | 1010.17 | 1013.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 15:00:00 | 1021.50 | 1010.17 | 1013.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 15:15:00 | 1020.00 | 1012.14 | 1013.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:15:00 | 1006.00 | 1012.14 | 1013.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 1013.80 | 1010.48 | 1012.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:00:00 | 1013.80 | 1010.48 | 1012.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 1018.65 | 1012.12 | 1012.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:00:00 | 1018.65 | 1012.12 | 1012.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 1018.90 | 1013.47 | 1013.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 14:45:00 | 1019.30 | 1013.47 | 1013.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 1015.30 | 1013.84 | 1013.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 10:15:00 | 1024.20 | 1017.15 | 1015.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 1013.45 | 1019.68 | 1017.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 1013.45 | 1019.68 | 1017.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 1013.45 | 1019.68 | 1017.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 1013.45 | 1019.68 | 1017.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 1043.00 | 1024.34 | 1020.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:45:00 | 1058.25 | 1038.62 | 1030.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:30:00 | 1060.45 | 1048.09 | 1037.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 1016.00 | 1038.57 | 1040.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 1016.00 | 1038.57 | 1040.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 12:15:00 | 1009.05 | 1021.98 | 1028.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 1016.10 | 1013.99 | 1021.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 09:15:00 | 1016.10 | 1013.99 | 1021.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 1016.10 | 1013.99 | 1021.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 12:15:00 | 1003.00 | 1012.32 | 1019.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 982.10 | 1009.90 | 1013.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 1034.00 | 1004.63 | 1007.82 | SL hit (close>static) qty=1.00 sl=1028.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 1012.50 | 1000.60 | 1000.38 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 1000.05 | 1003.60 | 1003.93 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 1012.55 | 1005.47 | 1004.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 10:15:00 | 1036.75 | 1011.73 | 1007.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 10:15:00 | 1037.10 | 1047.37 | 1032.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 11:00:00 | 1037.10 | 1047.37 | 1032.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 1021.50 | 1042.20 | 1031.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 1020.45 | 1042.20 | 1031.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 1017.95 | 1037.35 | 1030.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 1017.95 | 1037.35 | 1030.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-12-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 09:15:00 | 1014.60 | 1025.20 | 1025.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 1000.40 | 1015.37 | 1020.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 1015.75 | 1015.45 | 1020.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 15:00:00 | 1015.75 | 1015.45 | 1020.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 1026.50 | 1015.90 | 1018.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 1026.50 | 1015.90 | 1018.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 1020.00 | 1016.72 | 1018.85 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 1031.10 | 1020.17 | 1020.08 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 1013.75 | 1019.40 | 1019.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-11 09:15:00 | 1001.75 | 1014.47 | 1017.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 13:15:00 | 988.00 | 984.02 | 991.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-15 13:45:00 | 987.45 | 984.02 | 991.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 977.05 | 979.51 | 984.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 961.85 | 979.53 | 982.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 14:15:00 | 986.80 | 978.06 | 979.40 | SL hit (close>static) qty=1.00 sl=986.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 989.00 | 981.02 | 980.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 998.00 | 985.85 | 983.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 10:15:00 | 990.00 | 990.20 | 986.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 10:15:00 | 990.00 | 990.20 | 986.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 10:15:00 | 990.00 | 990.20 | 986.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:45:00 | 986.95 | 990.20 | 986.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 1005.00 | 996.96 | 991.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:30:00 | 1007.80 | 1001.69 | 994.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 09:15:00 | 1039.00 | 1045.46 | 1045.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 09:15:00 | 1039.00 | 1045.46 | 1045.80 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2026-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 10:15:00 | 1049.70 | 1046.31 | 1046.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 1054.50 | 1047.94 | 1046.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 1043.70 | 1053.96 | 1051.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 1043.70 | 1053.96 | 1051.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 1043.70 | 1053.96 | 1051.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 1043.70 | 1053.96 | 1051.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 1054.50 | 1054.07 | 1051.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:15:00 | 1057.20 | 1054.07 | 1051.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:45:00 | 1060.60 | 1054.63 | 1052.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 12:45:00 | 1056.30 | 1055.07 | 1052.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:30:00 | 1056.60 | 1055.05 | 1052.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 1050.70 | 1054.38 | 1053.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 12:15:00 | 1050.00 | 1052.26 | 1052.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 12:15:00 | 1050.00 | 1052.26 | 1052.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 1049.10 | 1051.63 | 1051.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 991.30 | 985.13 | 995.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:00:00 | 991.30 | 985.13 | 995.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 998.50 | 987.80 | 995.89 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 15:15:00 | 1005.00 | 998.53 | 998.23 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 990.40 | 996.90 | 997.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 09:15:00 | 954.50 | 974.87 | 982.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 15:15:00 | 975.00 | 965.72 | 972.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 15:15:00 | 975.00 | 965.72 | 972.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 975.00 | 965.72 | 972.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:15:00 | 950.00 | 965.72 | 972.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 951.90 | 962.96 | 970.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:15:00 | 946.10 | 962.96 | 970.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:45:00 | 946.30 | 960.38 | 968.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 14:15:00 | 985.90 | 968.25 | 970.07 | SL hit (close>static) qty=1.00 sl=975.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 09:15:00 | 983.30 | 973.12 | 972.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 991.80 | 982.09 | 977.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 992.40 | 993.82 | 987.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 15:00:00 | 992.40 | 993.82 | 987.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 971.30 | 990.29 | 986.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 971.30 | 990.29 | 986.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 972.80 | 986.79 | 985.54 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 975.00 | 984.43 | 984.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 14:15:00 | 968.80 | 978.26 | 981.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-29 15:15:00 | 940.00 | 939.59 | 951.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-30 09:15:00 | 939.10 | 939.59 | 951.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 943.80 | 940.44 | 951.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:45:00 | 950.00 | 940.44 | 951.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 945.80 | 941.51 | 950.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:30:00 | 948.30 | 941.51 | 950.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 946.80 | 942.57 | 950.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:30:00 | 951.00 | 942.57 | 950.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 948.80 | 943.81 | 950.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 948.80 | 943.81 | 950.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 960.10 | 947.07 | 951.10 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-02-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 10:15:00 | 959.55 | 953.03 | 953.00 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 11:15:00 | 952.10 | 952.84 | 952.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 12:15:00 | 948.15 | 951.90 | 952.49 | Break + close below crossover candle low |

### Cycle 67 — BUY (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 13:15:00 | 964.25 | 954.37 | 953.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 14:15:00 | 981.10 | 959.72 | 956.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 949.05 | 959.23 | 956.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 10:15:00 | 949.05 | 959.23 | 956.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 949.05 | 959.23 | 956.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 949.05 | 959.23 | 956.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 945.30 | 956.45 | 955.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 945.30 | 956.45 | 955.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 948.00 | 954.76 | 955.15 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 967.55 | 957.51 | 956.35 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 10:15:00 | 949.95 | 955.41 | 955.67 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 964.40 | 956.60 | 956.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 13:15:00 | 966.30 | 958.54 | 957.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 962.00 | 962.13 | 959.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 11:00:00 | 962.00 | 962.13 | 959.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 961.45 | 961.99 | 959.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:45:00 | 958.30 | 961.99 | 959.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 12:15:00 | 960.35 | 961.66 | 959.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:00:00 | 960.35 | 961.66 | 959.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 961.65 | 961.66 | 959.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:45:00 | 958.10 | 961.66 | 959.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 980.10 | 965.35 | 961.80 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 13:15:00 | 963.55 | 965.05 | 965.16 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 980.55 | 967.76 | 966.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 990.00 | 972.21 | 968.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 11:15:00 | 1011.20 | 1013.29 | 1001.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 11:30:00 | 1010.15 | 1013.29 | 1001.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 1003.65 | 1011.46 | 1004.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 1001.75 | 1011.46 | 1004.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 1004.35 | 1010.03 | 1004.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 10:30:00 | 1003.15 | 1010.03 | 1004.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 1003.60 | 1008.75 | 1004.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:30:00 | 1006.55 | 1008.75 | 1004.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 1000.35 | 1007.07 | 1004.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:30:00 | 1000.60 | 1007.07 | 1004.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 1000.10 | 1005.67 | 1003.96 | EMA400 retest candle locked (from upside) |

### Cycle 74 — SELL (started 2026-02-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 15:15:00 | 991.85 | 1001.37 | 1002.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 987.70 | 998.64 | 1000.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 09:15:00 | 1003.15 | 994.91 | 997.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-16 09:15:00 | 1003.15 | 994.91 | 997.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 1003.15 | 994.91 | 997.26 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2026-02-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 11:15:00 | 1013.90 | 999.96 | 999.22 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2026-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 12:15:00 | 1001.65 | 1009.23 | 1009.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 13:15:00 | 1000.15 | 1007.42 | 1008.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 1016.55 | 1009.24 | 1009.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 1016.55 | 1009.24 | 1009.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 1016.55 | 1009.24 | 1009.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 1016.55 | 1009.24 | 1009.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 1015.95 | 1010.59 | 1009.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 1019.05 | 1012.28 | 1010.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 11:15:00 | 1011.15 | 1013.19 | 1011.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 11:15:00 | 1011.15 | 1013.19 | 1011.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 1011.15 | 1013.19 | 1011.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 1011.15 | 1013.19 | 1011.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 1005.20 | 1011.59 | 1010.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:45:00 | 1000.15 | 1011.59 | 1010.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 1010.00 | 1011.27 | 1010.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-19 15:15:00 | 1011.30 | 1010.82 | 1010.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 11:00:00 | 1010.80 | 1010.67 | 1010.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 12:00:00 | 1011.45 | 1010.82 | 1010.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 12:15:00 | 1005.00 | 1009.66 | 1010.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 1005.00 | 1009.66 | 1010.17 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-02-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 09:15:00 | 1014.85 | 1010.17 | 1010.14 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 1000.75 | 1010.80 | 1010.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 10:15:00 | 990.55 | 1006.75 | 1009.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 10:15:00 | 993.75 | 993.12 | 999.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:45:00 | 992.65 | 993.12 | 999.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 996.05 | 991.33 | 995.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 12:15:00 | 989.00 | 991.17 | 994.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 13:30:00 | 986.90 | 990.27 | 993.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 972.20 | 987.99 | 988.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:15:00 | 939.55 | 956.49 | 968.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-04 13:15:00 | 937.55 | 956.49 | 968.41 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-05 12:15:00 | 951.70 | 948.87 | 958.42 | SL hit (close>ema200) qty=0.50 sl=948.87 alert=retest2 |

### Cycle 81 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 957.00 | 949.46 | 948.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 13:15:00 | 963.00 | 953.93 | 951.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 15:15:00 | 955.00 | 956.33 | 952.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-12 09:15:00 | 945.40 | 956.33 | 952.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 944.30 | 953.92 | 952.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 943.90 | 953.92 | 952.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 952.00 | 953.54 | 952.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:45:00 | 954.40 | 953.39 | 952.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:15:00 | 953.20 | 953.39 | 952.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 15:00:00 | 954.50 | 952.65 | 952.02 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 15:15:00 | 945.00 | 951.12 | 951.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2026-03-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 15:15:00 | 945.00 | 951.12 | 951.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 942.30 | 949.35 | 950.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 934.10 | 927.05 | 933.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 934.10 | 927.05 | 933.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 934.10 | 927.05 | 933.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 934.10 | 927.05 | 933.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 927.00 | 927.04 | 932.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 933.00 | 927.04 | 932.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 944.20 | 930.47 | 933.83 | EMA400 retest candle locked (from downside) |

### Cycle 83 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 942.20 | 936.62 | 936.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 947.60 | 939.88 | 938.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 938.40 | 942.19 | 940.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 14:15:00 | 938.40 | 942.19 | 940.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 938.40 | 942.19 | 940.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 14:45:00 | 939.20 | 942.19 | 940.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 938.00 | 941.36 | 939.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 919.80 | 941.36 | 939.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 920.90 | 937.26 | 938.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 912.60 | 918.23 | 924.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 919.50 | 918.48 | 923.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 15:00:00 | 919.50 | 918.48 | 923.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 907.50 | 892.46 | 897.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 904.60 | 892.46 | 897.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 898.00 | 893.57 | 897.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 11:30:00 | 894.20 | 894.23 | 897.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:15:00 | 896.00 | 894.23 | 897.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 14:15:00 | 895.10 | 895.29 | 897.39 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 849.49 | 872.39 | 882.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 851.20 | 872.39 | 882.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-30 09:15:00 | 850.35 | 872.39 | 882.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-30 14:15:00 | 863.00 | 861.12 | 871.79 | SL hit (close>ema200) qty=0.50 sl=861.12 alert=retest2 |

### Cycle 85 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 885.00 | 876.45 | 875.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 895.85 | 884.55 | 880.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 09:15:00 | 883.65 | 885.56 | 881.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-06 09:15:00 | 883.65 | 885.56 | 881.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 883.65 | 885.56 | 881.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:45:00 | 881.50 | 885.56 | 881.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 881.85 | 884.82 | 881.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 881.85 | 884.82 | 881.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 884.70 | 884.80 | 882.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:15:00 | 885.65 | 884.80 | 882.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:45:00 | 886.25 | 885.18 | 882.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 11:15:00 | 974.22 | 962.71 | 942.50 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-30 12:30:00 | 953.00 | 2025-06-02 12:15:00 | 945.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-05-30 15:00:00 | 958.80 | 2025-06-02 12:15:00 | 945.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-06-13 14:45:00 | 948.60 | 2025-06-16 13:15:00 | 961.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-16 09:15:00 | 947.40 | 2025-06-16 13:15:00 | 961.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-06-16 10:30:00 | 948.05 | 2025-06-16 13:15:00 | 961.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2025-07-02 09:15:00 | 985.70 | 2025-07-02 10:15:00 | 974.75 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest1 | 2025-07-07 12:45:00 | 1014.05 | 2025-07-08 09:15:00 | 1006.30 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest1 | 2025-07-07 14:00:00 | 1015.55 | 2025-07-08 09:15:00 | 1006.30 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-07-28 14:00:00 | 1116.75 | 2025-07-31 12:15:00 | 1076.55 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2025-07-31 09:15:00 | 1103.65 | 2025-07-31 12:15:00 | 1076.55 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2025-08-11 09:45:00 | 1042.00 | 2025-08-12 15:15:00 | 1058.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2025-08-11 15:15:00 | 1047.20 | 2025-08-12 15:15:00 | 1058.00 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-08-13 15:15:00 | 1068.10 | 2025-08-14 09:15:00 | 1055.10 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-08-14 11:45:00 | 1060.00 | 2025-08-14 12:15:00 | 1055.70 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-08-14 15:15:00 | 1060.00 | 2025-08-18 09:15:00 | 1050.50 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-08-25 11:30:00 | 1019.10 | 2025-09-01 09:15:00 | 1027.90 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-08-26 09:15:00 | 1013.00 | 2025-09-01 09:15:00 | 1027.90 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-09-03 09:15:00 | 1049.40 | 2025-09-09 14:15:00 | 1083.20 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2025-09-03 09:45:00 | 1046.70 | 2025-09-09 14:15:00 | 1083.20 | STOP_HIT | 1.00 | 3.49% |
| SELL | retest2 | 2025-09-18 11:00:00 | 1039.30 | 2025-09-19 14:15:00 | 1062.20 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2025-09-18 13:00:00 | 1039.80 | 2025-09-19 14:15:00 | 1062.20 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-09-18 13:45:00 | 1039.90 | 2025-09-19 14:15:00 | 1062.20 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-09-30 09:45:00 | 1009.90 | 2025-10-03 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-03 10:30:00 | 1009.05 | 2025-10-03 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-03 11:30:00 | 1011.40 | 2025-10-03 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-10-03 13:00:00 | 1011.30 | 2025-10-03 13:15:00 | 1013.50 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-10-07 11:45:00 | 997.00 | 2025-10-15 11:15:00 | 990.00 | STOP_HIT | 1.00 | 0.70% |
| SELL | retest2 | 2025-10-08 10:15:00 | 995.75 | 2025-10-15 11:15:00 | 990.00 | STOP_HIT | 1.00 | 0.58% |
| BUY | retest2 | 2025-10-17 14:30:00 | 988.70 | 2025-10-24 09:15:00 | 987.85 | STOP_HIT | 1.00 | -0.09% |
| BUY | retest2 | 2025-10-20 10:45:00 | 989.00 | 2025-10-24 09:15:00 | 987.85 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-10-24 09:15:00 | 989.00 | 2025-10-24 09:15:00 | 987.85 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2025-11-07 10:30:00 | 1024.55 | 2025-11-10 10:15:00 | 1006.30 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-11-14 09:45:00 | 1058.25 | 2025-11-18 09:15:00 | 1016.00 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2025-11-14 12:30:00 | 1060.45 | 2025-11-18 09:15:00 | 1016.00 | STOP_HIT | 1.00 | -4.19% |
| SELL | retest2 | 2025-11-20 12:15:00 | 1003.00 | 2025-11-24 14:15:00 | 1034.00 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-11-24 09:15:00 | 982.10 | 2025-11-24 14:15:00 | 1034.00 | STOP_HIT | 1.00 | -5.28% |
| SELL | retest2 | 2025-11-26 09:15:00 | 996.45 | 2025-11-27 14:15:00 | 1011.20 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-11-26 14:15:00 | 1003.65 | 2025-11-27 14:15:00 | 1011.20 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-11-27 10:45:00 | 1000.05 | 2025-11-27 14:15:00 | 1011.20 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-11-27 11:15:00 | 999.80 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-11-27 14:00:00 | 998.95 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-11-28 09:15:00 | 999.45 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-11-28 10:15:00 | 989.05 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-11-28 13:30:00 | 994.40 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-12-01 09:15:00 | 993.55 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-12-01 10:45:00 | 994.30 | 2025-12-01 13:15:00 | 1012.50 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-18 09:15:00 | 961.85 | 2025-12-18 14:15:00 | 986.80 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2025-12-23 09:30:00 | 1007.80 | 2026-01-02 09:15:00 | 1039.00 | STOP_HIT | 1.00 | 3.10% |
| BUY | retest2 | 2026-01-05 11:15:00 | 1057.20 | 2026-01-06 12:15:00 | 1050.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-01-05 11:45:00 | 1060.60 | 2026-01-06 12:15:00 | 1050.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-01-05 12:45:00 | 1056.30 | 2026-01-06 12:15:00 | 1050.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-01-05 13:30:00 | 1056.60 | 2026-01-06 12:15:00 | 1050.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2026-01-21 10:15:00 | 946.10 | 2026-01-21 14:15:00 | 985.90 | STOP_HIT | 1.00 | -4.21% |
| SELL | retest2 | 2026-01-21 10:45:00 | 946.30 | 2026-01-21 14:15:00 | 985.90 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-02-19 15:15:00 | 1011.30 | 2026-02-20 12:15:00 | 1005.00 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-02-20 11:00:00 | 1010.80 | 2026-02-20 12:15:00 | 1005.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-02-20 12:00:00 | 1011.45 | 2026-02-20 12:15:00 | 1005.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2026-02-26 12:15:00 | 989.00 | 2026-03-04 13:15:00 | 939.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 13:30:00 | 986.90 | 2026-03-04 13:15:00 | 937.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 12:15:00 | 989.00 | 2026-03-05 12:15:00 | 951.70 | STOP_HIT | 0.50 | 3.77% |
| SELL | retest2 | 2026-02-26 13:30:00 | 986.90 | 2026-03-05 12:15:00 | 951.70 | STOP_HIT | 0.50 | 3.57% |
| SELL | retest2 | 2026-03-02 09:15:00 | 972.20 | 2026-03-09 09:15:00 | 923.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-02 09:15:00 | 972.20 | 2026-03-09 14:15:00 | 945.70 | STOP_HIT | 0.50 | 2.73% |
| BUY | retest2 | 2026-03-12 11:45:00 | 954.40 | 2026-03-12 15:15:00 | 945.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2026-03-12 12:15:00 | 953.20 | 2026-03-12 15:15:00 | 945.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-03-12 15:00:00 | 954.50 | 2026-03-12 15:15:00 | 945.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-03-25 11:30:00 | 894.20 | 2026-03-30 09:15:00 | 849.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 12:15:00 | 896.00 | 2026-03-30 09:15:00 | 851.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 14:15:00 | 895.10 | 2026-03-30 09:15:00 | 850.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-25 11:30:00 | 894.20 | 2026-03-30 14:15:00 | 863.00 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2026-03-25 12:15:00 | 896.00 | 2026-03-30 14:15:00 | 863.00 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2026-03-25 14:15:00 | 895.10 | 2026-03-30 14:15:00 | 863.00 | STOP_HIT | 0.50 | 3.59% |
| BUY | retest2 | 2026-04-06 12:15:00 | 885.65 | 2026-04-09 11:15:00 | 974.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-06 12:45:00 | 886.25 | 2026-04-09 11:15:00 | 974.88 | TARGET_HIT | 1.00 | 10.00% |
